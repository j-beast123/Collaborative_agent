import threading
import urllib.error
from concurrent.futures import ThreadPoolExecutor

import pytest

torch = pytest.importorskip("torch")

from src.collaboration.runtime import BatchedAgent  # noqa: E402


class FakeTokenizer:
    padding_side = "right"
    eos_token_id = 0

    def __init__(self):
        self.batch_prompts = []

    def apply_chat_template(self, messages, **kwargs):
        return messages[1]["content"]

    def __call__(self, prompts, **kwargs):
        assert self.padding_side == "left"
        self.batch_prompts = list(prompts)

        class Batch(dict):
            def to(self, device):
                return self

        return Batch(input_ids=torch.zeros(len(prompts), 1, dtype=torch.long))

    def decode(self, ids, **kwargs):
        return f" reply:{self.batch_prompts[int(ids[0]) - 100]} "


class FakeModel:
    device = "cpu"

    def __init__(self, tokenizer, fail_on=None):
        self.tokenizer, self.fail_on, self.batches = tokenizer, fail_on, []
        self.lock = threading.Lock()

    def generate(self, input_ids, pad_token_id, **settings):
        prompts = list(self.tokenizer.batch_prompts)
        self.batches.append((prompts, settings))
        if self.fail_on and self.fail_on in prompts:
            raise RuntimeError("boom")
        n = input_ids.shape[0]
        return torch.cat([input_ids, torch.arange(100, 100 + n).unsqueeze(1)], dim=1)


def test_batched_agent_returns_each_callers_own_reply_and_groups_settings():
    tokenizer = FakeTokenizer()
    model = FakeModel(tokenizer)
    agent = BatchedAgent(model, tokenizer, max_batch=4)
    jobs = [(f"p{i}", 512 if i % 2 else 384, i % 2 == 1) for i in range(12)]
    with ThreadPoolExecutor(12) as pool:
        replies = list(pool.map(lambda j: agent("sys", j[0], j[1], do_sample=j[2], temperature=0.7), jobs))
    assert replies == [f"reply:p{i}" for i in range(12)]
    for prompts, settings in model.batches:
        assert 1 <= len(prompts) <= 4
        expected = {"max_new_tokens": 512, "do_sample": True, "temperature": 0.7} if settings["do_sample"] \
            else {"max_new_tokens": 384, "do_sample": False}
        assert settings == expected  # greedy calls never get sampling settings
        assert all((int(p[1:]) % 2 == 1) == settings["do_sample"] for p in prompts)


def test_batched_agent_delivers_errors_and_keeps_serving():
    tokenizer = FakeTokenizer()
    agent = BatchedAgent(FakeModel(tokenizer, fail_on="bad"), tokenizer, max_batch=1)
    with pytest.raises(RuntimeError, match="boom"):
        agent("sys", "bad", 10)
    assert agent("sys", "good", 10) == "reply:good"


def test_doctor_call_retries_network_errors_and_empty_bodies(monkeypatch):
    import io
    import json

    import src.collaboration.generator as generator

    monkeypatch.setattr("time.sleep", lambda s: None)
    replies = iter([urllib.error.URLError("connection reset"), TimeoutError(), {"error": {"message": "provider down"}},
                    {"choices": [{"message": {"content": None}}]}, {"choices": [{"message": {"content": " ok "}}]}])

    def urlopen(request, timeout):
        reply = next(replies)
        if isinstance(reply, Exception):
            raise reply
        return io.BytesIO(json.dumps(reply).encode())

    monkeypatch.setattr(generator.urllib.request, "urlopen", urlopen)
    assert generator.doctor_call("key", "model", "system", "user", 10) == "ok"

    monkeypatch.setattr(generator.urllib.request, "urlopen",
                        lambda request, timeout: (_ for _ in ()).throw(urllib.error.URLError("down")))
    with pytest.raises(urllib.error.URLError):  # persistent failures still surface
        generator.doctor_call("key", "model", "system", "user", 10)
