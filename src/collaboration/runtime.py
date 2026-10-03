"""Model calls behind a collaboration: the local agent and the API doctor / reconstructor."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class CollaborationRuntime:
    """Model calls used by a rerun; tests pass fakes, real runs use build_runtime()."""

    doctor: Callable[[str, str], str]  # (system, user) -> clinician text
    agent: Callable[[str, str, int], str]  # (system, user, max_tokens) -> agent text
    reconstruct: Callable[[dict, list], dict]  # (case, dialogue) -> parsed reconstruction
    doctor_system: str
    agent_system: str
    judgment_system: str
    max_tokens: int
    close: Optional[Callable[[], None]] = None  # releases a loaded local model (GPU memory)
    continuation_guideline: str = ""  # appended to the agent prompt in turns after the continued one


def run_with_retries(call: Callable[[], dict], *, format_attempts: int = 1,
                     on_attempt: Optional[Callable[[int, dict], None]] = None) -> dict:
    """The retry policy of every continuation / response call (labels, candidates, RQ2
        matched continuation, rollout provider, rubric reruns).
    """
    for attempt in range(1, format_attempts + 1):
        result = call()
        if on_attempt:
            on_attempt(attempt, result)
        if result["status"] == "completed":
            break
    return result


class BatchedAgent:
    """Thread-safe agent generation that batches concurrent calls on one GPU model."""

    def __init__(self, model, tokenizer, max_batch: int):
        import queue
        import threading

        self.model, self.tokenizer, self.max_batch = model, tokenizer, max_batch
        self.requests = queue.Queue()
        self.deferred = []  # requests taken from the queue whose settings did not match the batch
        threading.Thread(target=self._serve, daemon=True).start()

    def __call__(self, system: str, user: str, max_tokens: int, *, do_sample: bool = False,
                 temperature: float | None = None, top_p: float | None = None) -> str:
        from concurrent.futures import Future

        prompt = self.tokenizer.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        settings = {"max_new_tokens": max_tokens, "do_sample": do_sample}
        if do_sample and temperature is not None:
            settings["temperature"] = temperature
        if do_sample and top_p is not None:
            settings["top_p"] = top_p
        future = Future()
        self.requests.put((tuple(sorted(settings.items())), prompt, future))
        return future.result()

    def close(self) -> None:
        self.requests.put(None)

    def _next_batch(self):
        import queue

        first = self.deferred.pop(0) if self.deferred else self.requests.get()
        if first is None:  # close()
            return None
        batch, rest = [first], []
        for item in self.deferred:
            (batch if item[0] == first[0] and len(batch) < self.max_batch else rest).append(item)
        while len(batch) < self.max_batch:
            try:
                item = self.requests.get_nowait()
            except queue.Empty:
                break
            if item is None:  # close(): finish this batch first
                self.requests.put(None)
                break
            (batch if item[0] == first[0] else rest).append(item)
        self.deferred = rest
        return batch

    def _serve(self) -> None:
        # Never let this thread die: callers block on their futures without a timeout.
        while True:
            batch = []
            try:
                batch = self._next_batch()
                if batch is None:
                    self.model = None
                    return
                texts = self._generate(dict(batch[0][0]), [prompt for _, prompt, _ in batch])
            except Exception as error:  # noqa: BLE001 - delivered to every waiting caller
                for _, _, future in batch:
                    future.set_exception(error)
                continue
            for (_, _, future), text in zip(batch, texts):
                future.set_result(text)

    def _generate(self, settings: dict, prompts: list[str]) -> list[str]:
        import torch

        try:
            self.tokenizer.padding_side = "left"
            inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.model.device)
            with torch.inference_mode():
                output = self.model.generate(**inputs, pad_token_id=self.tokenizer.eos_token_id, **settings)
        except torch.cuda.OutOfMemoryError:
            if len(prompts) == 1:
                raise
            torch.cuda.empty_cache()
            half = len(prompts) // 2
            return self._generate(settings, prompts[:half]) + self._generate(settings, prompts[half:])
        new_tokens = output[:, inputs["input_ids"].shape[-1]:]
        return [self.tokenizer.decode(ids, skip_special_tokens=True).strip() for ids in new_tokens]


def load_local_agent(agent_model: str, batch_size: int = 1) -> tuple[Callable[..., str], Callable[[], None]]:
    """(generate(system, user, max_tokens, **sampling) -> text, close()) for a local causal LM.
    batch_size > 1: generate is thread-safe and batches concurrent calls (BatchedAgent)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from .generator import agent_call

    from src.common.config import local_model_path

    agent_model = local_model_path(agent_model)
    tokenizer = AutoTokenizer.from_pretrained(agent_model, trust_remote_code=True)
    loaded = {"model": AutoModelForCausalLM.from_pretrained(
        agent_model, dtype=torch.float16, device_map="auto", trust_remote_code=True)}
    loaded["model"].eval()
    if batch_size > 1:
        generate = BatchedAgent(loaded["model"], tokenizer, batch_size)
    else:
        def generate(system, user, max_tokens, **sampling):
            return agent_call(loaded["model"], tokenizer, system, user, max_tokens, **sampling)

    def close() -> None:
        if batch_size > 1:
            generate.close()
        loaded.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return generate, close


def build_runtime(*, agent_model: str, doctor_model: str, reconstructor_model: str,
                  max_tokens: int, root, agent_decoding: Optional[dict] = None,
                  doctor_decoding: Optional[dict] = None, agent_batch_size: int = 1,
                  doctor_prompt: Optional[str] = None, continuation_guideline: Optional[str] = None,
                  reconstruction_max_tokens: int = 2048,
                  ) -> CollaborationRuntime:
    """Real runtime: local agent + OpenRouter doctor/reconstructor, as in generation."""
    import os

    from .generator import doctor_call, reconstruct_call

    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")
    agent_decoding = dict(agent_decoding or {})
    doctor_decoding = dict(doctor_decoding or {})
    doctor_tokens = doctor_decoding.get("max_tokens", max_tokens)
    doctor_temperature = doctor_decoding.get("temperature", 0.4)
    agent_tokens = agent_decoding.get("max_new_tokens", max_tokens)
    sampling = {k: agent_decoding[k] for k in ("do_sample", "temperature", "top_p") if k in agent_decoding}
    generate, close = load_local_agent(agent_model, agent_batch_size)

    return CollaborationRuntime(
        doctor=lambda system, user: doctor_call(key, doctor_model, system, user, doctor_tokens, doctor_temperature),
        # Decoding settings apply to agent responses; the turn-local judgment call
        # (a different token budget) stays greedy as in generation.
        agent=lambda system, user, tokens: (
            generate(system, user, agent_tokens, **sampling) if tokens == max_tokens
            else generate(system, user, tokens)),
        reconstruct=lambda case, dialogue: reconstruct_call(
            key, reconstructor_model, case, dialogue, reconstruction_max_tokens
        ),
        doctor_system=(root / (doctor_prompt or "prompts/user_simulator.txt")).read_text(encoding="utf-8"),
        agent_system=(root / "prompts/agent.txt").read_text(encoding="utf-8"),
        judgment_system=(root / "prompts/local_epistemic_judgment.txt").read_text(encoding="utf-8"),
        max_tokens=max_tokens,
        close=close,
        continuation_guideline=continuation_guideline or "",
    )
