"""Models and cases of an AgentClinic collaboration run, built from a SERL run config."""

from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from src.common.config import local_model_path, IntegrationUnavailable, read_config
from src.common.io import read_jsonl
from src.data.schema import digest
from src.judges.llm import openrouter_chat

from .outcome import DiagnosisJudge, frozen_equivalence
from .pipeline import Appender
from .protocol import ClinicRuntime, doctor_opening
from .selection import split_eligible_cases


@dataclass
class Clinic:
    cases: dict[str, dict]      # every eligible case with its split
    runtime: ClinicRuntime
    judge: DiagnosisJudge
    generation: Callable        # context manager: the Agent model generates inside it
    source: dict                # the label run's config
    close: Callable = lambda: None  # release the Agent model before another one is loaded


def load_policy(policy: dict, adapter: Optional[Path], *, trainable: bool, device: str = "cuda:0"):
    """(model, tokenizer): the Agent model, with the adapter when given or a new LoRA when trainable."""
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not device.startswith("cuda:") or not torch.cuda.is_available():
        raise IntegrationUnavailable(f"the Agent model requires an explicit available GPU, got {device}")
    torch.cuda.set_device(torch.device(device))
    source = local_model_path(policy.get("checkpoint") or policy["model_id"])
    tokenizer = AutoTokenizer.from_pretrained(source)
    if policy.get("device_map") == "auto":  # a model larger than one GPU, for generation only: spread over the visible GPUs
        if trainable:
            raise IntegrationUnavailable("policy.device_map auto is for generation only, not for training")
        model = AutoModelForCausalLM.from_pretrained(source, torch_dtype=getattr(torch, policy["torch_dtype"]), device_map="auto")
        if adapter:
            model = PeftModel.from_pretrained(model, str(adapter), is_trainable=False)
        model.eval()
        return model, tokenizer
    model = AutoModelForCausalLM.from_pretrained(source, torch_dtype=getattr(torch, policy["torch_dtype"]))
    if adapter:
        model = PeftModel.from_pretrained(model, str(adapter), is_trainable=trainable)
    elif trainable:
        lora = policy["lora"]
        model = get_peft_model(model, LoraConfig(
            task_type="CAUSAL_LM", r=lora["r"], lora_alpha=lora["alpha"], lora_dropout=lora["dropout"],
            target_modules=lora["target_modules"], use_rslora=lora.get("use_rslora", False)))
    model.to(device)
    if not trainable:
        model.eval()
    return model, tokenizer


def policy_agent(model, tokenizer, decoding: dict, batch_size: int):
    """(sampled agent, greedy agent, generation context, close). Concurrent calls are batched on the
    model; the context puts the model in eval mode while replies are generated."""
    from src.collaboration.runtime import BatchedAgent

    settings = dict(decoding)
    tokens = settings.pop("max_new_tokens")
    generate = BatchedAgent(model, tokenizer, batch_size)

    @contextmanager
    def generation():
        training = model.training
        model.eval()
        try:
            yield
        finally:
            model.train(training)

    def close() -> None:
        import torch

        generate.close()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return (lambda system, user: generate(system, user, tokens, **settings),
            lambda system, user: generate(system, user, tokens), generation, close)


def build_judge(config: dict, root: Path, source: dict, judge_chat=None) -> DiagnosisJudge:
    judge_chat = judge_chat or openrouter_chat(source["models"]["judge"]["model_id"],
                                               root / config["output_dir"] / "cache_diagnosis_judge.jsonl")
    return DiagnosisJudge(judge_chat, (root / source["prompts"]["diagnosis_equivalence"]).read_text(encoding="utf-8"),
                          params=source.get("diagnosis_judge_params"), frozen=frozen_equivalence(source, root),
                          policy=source.get("diagnosis_judge_policy"))


def build_clinic(config: dict, root: Path, model, tokenizer, *, doctor=None, judge_chat=None) -> Clinic:
    """doctor and judge_chat default to the OpenRouter models of the label run's config."""
    source = read_config(root / config["environment"]["config"])
    models, decoding, prompts = source["models"], source["decoding"], source["prompts"]
    if doctor is None:
        from src.collaboration.generator import doctor_call

        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise IntegrationUnavailable("OPENROUTER_API_KEY is not set")
        doctor = lambda system, user: doctor_call(key, models["doctor"]["model_id"], system, user,  # noqa: E731
                                                  decoding["doctor"]["max_tokens"], decoding["doctor"]["temperature"])
    agent, greedy, generation, close = policy_agent(model, tokenizer, decoding["agent"], config["rollout"]["agent_batch_size"])
    runtime = ClinicRuntime(doctor=doctor, agent=agent, agent_greedy=greedy,
                            doctor_system=(root / prompts["doctor"]).read_text(encoding="utf-8"),
                            agent_system=(root / prompts["consultant"]).read_text(encoding="utf-8"),
                            state_system=(root / config["environment"]["state_prompt"]).read_text(encoding="utf-8"))
    return Clinic(split_eligible_cases(source, root, allowed_splits=("train", "dev")), runtime, build_judge(config, root, source, judge_chat),
                  generation, source, close)


def protocol_identity(config: dict, root: Path) -> str:
    """Hash of what defines the collaboration environment of a run: the label run's models, decoding and prompts, the
    state-elicitation prompt and the label protocol hash.
    """
    source = read_config(root / config["environment"]["config"])
    prompts = {name: digest((root / path).read_text(encoding="utf-8")) for name, path in sorted(source["prompts"].items())}
    protocol = {}
    if source.get("protocol"):  # the main protocol: its length, stopping rule and the two instructions that define it
        from .protocol import NEUTRAL_AGENT_TASK, NEXT_STEP

        protocol = {"protocol": {k: source["protocol"][k] for k in ("max_turns", "early_stopping")},
                    "next_step": digest(NEXT_STEP), "agent_task": digest(NEUTRAL_AGENT_TASK)}
    return digest({"models": source["models"], "decoding": source["decoding"], "prompts": prompts, **protocol,
                   "doctor_view": source.get("doctor_view"),
                   "state_prompt": digest((root / config["environment"]["state_prompt"]).read_text(encoding="utf-8")),
                   "label_protocol": config["reward"]["effectiveness"].get("expected_continuation_protocol_hash")})[:16]


def check_identity(rows: list[dict], identity: str, where) -> None:
    other = sorted({str(r.get("protocol_identity")) for r in rows} - {identity})
    if other:
        raise IntegrationUnavailable(f"{where} holds trajectories of another protocol ({other[0]}, expected {identity}); "
                                     "use a new output directory instead of mixing them")


def frozen_histories(config: dict, root: Path, cases: dict[str, dict]) -> list[dict]:
    """The label run's D1 histories (TRAIN and DEV cases)."""
    histories = [row for row in read_jsonl(root / config["environment"]["histories"]) if row["case_id"] in cases]
    if any(cases[row["case_id"]]["split"] == "test" for row in histories):
        raise ValueError("the frozen histories contain TEST cases")
    return histories


def opening_histories(clinic: Clinic, split: str, path: Path, workers: int) -> list[dict]:
    """D1 histories of a split that has none (evaluation), written once to path so that every system
    answers the same Doctor message."""
    from concurrent.futures import ThreadPoolExecutor

    log = Appender(path, key=lambda r: r["case_id"])

    def run(case):
        history = doctor_opening(case, clinic.runtime)
        log.add({"history_id": f"{case['case_id']}:A1", "case_id": case["case_id"], "history": history,
                 "doctor_provisional_diagnosis": history[0]["provisional_diagnosis"]})

    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(run, [c for c in clinic.cases.values() if c["split"] == split and c["case_id"] not in log.keys]))
    return sorted(log.rows, key=lambda r: r["case_id"])
