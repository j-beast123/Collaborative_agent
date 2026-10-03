"""Outcome: is the Doctor's final diagnosis the Correct_Diagnosis?"""

from __future__ import annotations

import json
import re
from typing import Optional

import fcntl
import threading
from pathlib import Path

from src.judges.llm import CachedChat, ChatRequest, parse_json_object

PROMPT_VERSION = "diagnosis-equivalence-v3"


def normalize(diagnosis: str) -> str:
    text = re.sub(r"\([^)]*\)", " ", (diagnosis or "").lower())  # drop parenthetical abbreviations
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", text).split())


class FrozenEquivalence:
    """{(normalize(reference), normalize(predicted)): equivalent} in a JSONL file shared by all processes."""

    def __init__(self, path) -> None:
        self.path, self.lock = Path(path), threading.Lock()
        self.decisions: dict[tuple[str, str], bool] = {}
        self.read = 0  # bytes of the file already loaded
        self.pair_locks: dict[tuple[str, str], threading.Lock] = {}
        self.load()

    def load(self) -> None:
        if not self.path.is_file() or self.path.stat().st_size == self.read:
            return
        with self.path.open("rb") as handle:
            handle.seek(self.read)
            data = handle.read()
        self.read += len(data)
        for line in data.decode("utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                self.decisions.setdefault((row["reference"], row["predicted"]), bool(row["equivalent"]))  # the first decision stays

    def get(self, reference: str, predicted: str) -> Optional[bool]:
        return self.decisions.get((normalize(reference), normalize(predicted)))

    def decide(self, reference: str, predicted: str, judge) -> dict:
        """The judge is called outside the file lock (a call can take seconds, and several runs share the file); the lock is
        held only to read and to append. If another process froze the pair meanwhile, its decision is the one returned."""
        key = (normalize(reference), normalize(predicted))
        frozen = {"method": "frozen", "reason": "frozen decision of this normalized diagnosis pair"}
        with self.lock:
            self.load()
            if key in self.decisions:
                return {"equivalent": self.decisions[key], **frozen}
            pair_lock = self.pair_locks.setdefault(key, threading.Lock())
        with pair_lock:  # one call per pair in this process
            with self.lock:
                self.load()
                if key in self.decisions:
                    return {"equivalent": self.decisions[key], **frozen}
            verdict = judge()
            if not isinstance(verdict.get("equivalent"), bool):
                return verdict
            with self.lock:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a+", encoding="utf-8") as handle:
                    fcntl.flock(handle, fcntl.LOCK_EX)
                    try:
                        self.load()
                        if key in self.decisions:  # decided by another process while the judge was called here
                            return {"equivalent": self.decisions[key], **frozen}
                        line = json.dumps({"reference": key[0], "predicted": key[1], "equivalent": verdict["equivalent"],
                                           "source": "judge", "reason": verdict.get("reason"),
                                           **({"votes": verdict["votes"], "policy": verdict["policy"]} if verdict.get("votes") else {}),
                                           "raw_reference": reference, "raw_predicted": predicted}, ensure_ascii=False) + "\n"
                        handle.write(line)
                        handle.flush()
                        self.decisions[key] = verdict["equivalent"]
                        self.read += len(line.encode("utf-8"))
                        return verdict
                    finally:
                        fcntl.flock(handle, fcntl.LOCK_UN)


MAJORITY_POLICY = "majority-v1"


class DiagnosisJudge:
    """policy None: one call decides a pair."""

    def __init__(self, chat: CachedChat, system_prompt: str, params: Optional[dict] = None,
                 frozen: Optional[FrozenEquivalence] = None, policy: Optional[str] = None) -> None:
        if policy not in (None, MAJORITY_POLICY):
            raise ValueError(f"unknown outcome-judge policy {policy!r}")
        self.chat, self.system_prompt, self.frozen, self.policy = chat, system_prompt, frozen, policy
        self.params = params or {"temperature": 0, "max_tokens": 120}

    def equivalent(self, reference: str, predicted: str) -> dict:
        """{"equivalent": bool or None, "method": exact | judge | empty | judge_failed, "reason"}."""
        if not (predicted or "").strip():
            return {"equivalent": False, "method": "empty", "reason": "no predicted diagnosis"}
        if normalize(reference) == normalize(predicted):
            return {"equivalent": True, "method": "exact", "reason": "normalized strings are equal"}
        if self.frozen is not None:
            return self.frozen.decide(reference, predicted, lambda: self.judge(reference, predicted))
        return self.judge(reference, predicted)

    def judge(self, reference: str, predicted: str) -> dict:
        """The decision of a pair: one call, or the votes of the majority policy."""
        if self.policy != MAJORITY_POLICY:
            return self.vote(reference, predicted)
        votes = [self.vote(reference, predicted, cached=False) for _ in range(2)]
        if votes[0]["equivalent"] != votes[1]["equivalent"] or not isinstance(votes[0]["equivalent"], bool):
            votes += [self.vote(reference, predicted, cached=False) for _ in range(3)]
        valid = [v["equivalent"] for v in votes if isinstance(v["equivalent"], bool)]
        if len(valid) < len(votes):  # a failed call: no decision is frozen from an incomplete vote
            return {"equivalent": None, "method": "judge_failed", "reason": "a vote of the majority policy failed"}
        decision = sum(valid) * 2 > len(valid)
        return {"equivalent": decision, "method": "judge", "votes": valid, "policy": MAJORITY_POLICY,
                "reason": next(v["reason"] for v in votes if v["equivalent"] is decision)}

    def vote(self, reference: str, predicted: str, cached: bool = True) -> dict:
        """One call of the equivalence judge on the two strings as they are."""
        user = json.dumps({"reference_diagnosis": reference, "predicted_diagnosis": predicted}, ensure_ascii=False)
        request = ChatRequest(self.system_prompt, user, PROMPT_VERSION, self.params)
        result = self.chat.complete(request) if cached else self.chat.backend.complete(request)
        parsed, _ = parse_json_object(result.text) if result.status == "ok" else (None, "call_failed")
        if not parsed or not isinstance(parsed.get("equivalent"), bool):
            return {"equivalent": None, "method": "judge_failed", "reason": result.error or "unparsable judge output"}
        return {"equivalent": parsed["equivalent"], "method": "judge", "reason": parsed.get("reason")}


def frozen_equivalence(source: dict, root) -> Optional[FrozenEquivalence]:
    """The frozen decisions an environment config names (diagnosis_equivalence_frozen: a path), else None."""
    path = (source or {}).get("diagnosis_equivalence_frozen")
    return FrozenEquivalence(Path(root) / path) if path else None
