"""ContinuationSpec: the one definition of how a history is continued to a final answer."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from src.common.config import IntegrationUnavailable
from src.data.schema import digest

SPEC_VERSION = "2"
SEED_MODES = ("paired_by_history", "independent")
# Only protocols implemented by src/collaboration/generator.py and rerun.py are accepted.
TURN_ORDERS = (("doctor", "agent"),)
STOPPING_RULES = ("generator_stop_reason", "fixed_collaboration_turns")  # the latter: src/agentclinic/protocol.py
FINAL_DECISION_ROLES = ("doctor", "agent")
SUCCESS_CRITERIA = ("final_option_matches_ground_truth", "final_diagnosis_equivalent_to_ground_truth")
PROTOCOL_VARIANT_FIELDS = ("doctor_prompt", "continuation_guideline")
DEFAULT_DOCTOR_PROMPT = "prompts/user_simulator.txt"


@dataclass(frozen=True)
class ModelRef:
    model_id: Optional[str] = None
    revision: Optional[str] = None
    checkpoint: Optional[str] = None  # local path used to load a local model, if any


@dataclass(frozen=True)
class SeedPolicy:
    # paired_by_history: rollout i uses the same seed for BEFORE and AFTER of a history and for every candidate
    # response of it, where the backend supports seeding (local sampling).
    mode: Optional[str] = None
    base_seed: Optional[int] = None


@dataclass(frozen=True)
class ContinuationSpec:
    agent_model: ModelRef = field(default_factory=ModelRef)        # continuation agent policy
    doctor_model: ModelRef = field(default_factory=ModelRef)       # collaborator simulator
    reconstructor_model: ModelRef = field(default_factory=ModelRef)  # belief reconstruction (turn >= 2)
    turn_order: Optional[tuple] = None
    max_rounds: Optional[int] = None          # total rounds; remaining = max_rounds - current round + 1
    stopping_rule: Optional[str] = None
    final_decision_role: Optional[str] = None
    decoding: Optional[dict] = None           # {"agent": {...}, "doctor": {...}}
    success_criterion: Optional[str] = None
    k: Optional[int] = None
    seed_policy: SeedPolicy = field(default_factory=SeedPolicy)
    prompt_versions: Optional[dict] = None
    doctor_prompt: Optional[str] = None           # doctor system prompt file; None = prompts/user_simulator.txt
    continuation_guideline: Optional[str] = None  # appended to the agent prompt in turns after the labelled one
    spec_version: str = SPEC_VERSION

    def __post_init__(self):
        if self.k is not None and (type(self.k) is not int or self.k < 1):
            raise ValueError("continuation k must be a positive integer")
        if self.max_rounds is not None and (type(self.max_rounds) is not int or self.max_rounds < 1):
            raise ValueError("max_rounds must be a positive integer")
        if self.turn_order is not None and tuple(self.turn_order) not in TURN_ORDERS:
            raise ValueError(f"turn_order must be one of {TURN_ORDERS}")
        if self.stopping_rule is not None and self.stopping_rule not in STOPPING_RULES:
            raise ValueError(f"stopping_rule must be one of {STOPPING_RULES}")
        if self.final_decision_role is not None and self.final_decision_role not in FINAL_DECISION_ROLES:
            raise ValueError(f"final_decision_role must be one of {FINAL_DECISION_ROLES}")
        if self.success_criterion is not None and self.success_criterion not in SUCCESS_CRITERIA:
            raise ValueError(f"success_criterion must be one of {SUCCESS_CRITERIA}")
        if self.decoding is not None and (not isinstance(self.decoding, dict) or
                                          set(self.decoding) - {"agent", "doctor"}):
            raise ValueError("decoding must be {'agent': {...}, 'doctor': {...}}")
        if self.seed_policy.mode is not None and self.seed_policy.mode not in SEED_MODES:
            raise ValueError(f"seed_policy.mode must be one of {SEED_MODES}")
        if self.seed_policy.base_seed is not None and type(self.seed_policy.base_seed) is not int:
            raise ValueError("seed_policy.base_seed must be an integer")

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ContinuationSpec":
        raw = {k: v for k, v in dict(raw).items() if not k.endswith("_status") and k != "notes"}
        unknown = set(raw) - {f for f in cls.__dataclass_fields__}
        if unknown:
            raise ValueError(f"Unknown continuation fields: {sorted(unknown)}")
        if raw.get("turn_order") is not None:
            raw["turn_order"] = tuple(raw["turn_order"])
        for name in ("agent_model", "doctor_model", "reconstructor_model"):
            raw[name] = ModelRef(**(raw.get(name) or {}))
        raw["seed_policy"] = SeedPolicy(**(raw.get("seed_policy") or {}))
        return cls(**raw)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for name in PROTOCOL_VARIANT_FIELDS:  # unset: the generation protocol, with its original hash
            if data[name] is None:
                del data[name]
        if data["turn_order"] is not None:
            data["turn_order"] = list(data["turn_order"])
        return data

    def protocol_hash(self) -> str:
        return digest(self.to_dict())

    def missing_fields(self) -> list[str]:
        missing = []
        for role in ("agent_model", "doctor_model", "reconstructor_model"):
            ref = getattr(self, role)
            if not ref.model_id or not ref.revision:
                missing.append(f"{role}.model_id/revision")
        for name in ("turn_order", "max_rounds", "stopping_rule", "final_decision_role", "decoding",
                     "success_criterion", "k", "prompt_versions"):
            if getattr(self, name) in (None, {}, ()):
                missing.append(name)
        if self.seed_policy.mode is None or self.seed_policy.base_seed is None:
            missing.append("seed_policy.mode/base_seed")
        return missing

    def require_resolved(self) -> None:
        missing = self.missing_fields()
        if missing:
            raise IntegrationUnavailable("Continuation protocol unresolved (TODO): " + ", ".join(missing))

    def rollout_seeds(self, history_id: str, state_id: str) -> list[int]:
        if self.k is None or self.seed_policy.mode is None or self.seed_policy.base_seed is None:
            raise IntegrationUnavailable("Rollout seeds need k and a seed policy")
        key = history_id if self.seed_policy.mode == "paired_by_history" else state_id
        return [int(digest([self.seed_policy.base_seed, key, index])[:8], 16) for index in range(self.k)]
