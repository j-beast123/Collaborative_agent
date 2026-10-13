"""Behavior judge for RQ2: classify a response into an interaction behavior class."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Optional

from src.data.schema import digest
from src.judges.llm import CachedChat, ChatRequest, parse_json_object

BEHAVIOR_CLASSES = ("ACCEPT_REVISE", "MAINTAIN_CHALLENGE", "CONSIDER_REQUEST_EVIDENCE", "OTHER")


@dataclass
class BehaviorJudgment:
    status: str  # scored / invalid / call_failed
    behavior_class: Optional[str]
    confidence: Optional[float]
    rationale: Optional[str]
    raw_response: Optional[str]
    judge_model: str
    judge_version: str
    prompt_sha256: str
    gives_evidence: Optional[bool] = None  # only with a prompt that asks for it (candidate compliance)

    def to_dict(self) -> dict:
        return asdict(self)


class BehaviorJudge:
    def __init__(self, chat: CachedChat, *, system_prompt: str, prompt_version: str, params: dict) -> None:
        self.chat = chat
        self.system_prompt = system_prompt
        self.prompt_version = prompt_version
        self.params = params

    def classify(self, *, agent_prior_position: str, clinician_claim_option: str,
                 clinician_message: str, response: str) -> BehaviorJudgment:
        user = json.dumps({
            "agent_position_before_response": agent_prior_position,
            "clinician_claimed_option": clinician_claim_option,
            "clinician_message": clinician_message,
            "agent_response": response,
        }, ensure_ascii=False)
        result = self.chat.complete(ChatRequest(self.system_prompt, user, self.prompt_version, self.params))
        base = dict(judge_model=self.chat.backend.model, judge_version=self.prompt_version,
                    prompt_sha256=digest(self.system_prompt), raw_response=result.text)
        if result.status != "ok":
            return BehaviorJudgment("call_failed", None, None, None, **base)
        parsed, status = parse_json_object(result.text)
        label = (parsed or {}).get("behavior_class")
        if parsed is None or label not in BEHAVIOR_CLASSES:
            return BehaviorJudgment("invalid", None, None, None, **base)
        confidence = parsed.get("confidence")
        confidence = float(confidence) if isinstance(confidence, (int, float)) and not isinstance(confidence, bool) else None
        evidence = parsed.get("gives_evidence")
        return BehaviorJudgment("scored", label, confidence, parsed.get("rationale"), **base,
                                gives_evidence=evidence if isinstance(evidence, bool) else None)
