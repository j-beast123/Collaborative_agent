"""Frozen typed records and their JSON persistence."""
from __future__ import annotations
import dataclasses, hashlib, json, os, types, typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


def sha256_json(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


@dataclass(frozen=True)
class GeneratorInput:
    """Everything the Rubric Generator may see: visible case fields, the pre-action history and the structured state
    elicited before any candidate reply (Doctor-side context / Agent-side state). Never gold, replies or outcomes."""
    case_id: str
    scenario: str
    question: str
    options: tuple[tuple[str, str], ...]
    history: tuple[tuple[str, str, int, str], ...]      # (speaker, stage, round, text)
    turn: int
    state_status: str
    doctor_side_context: Optional[dict]
    agent_side_state: Optional[dict]
    sha256: str

    def payload(self) -> dict:
        return {k: v for k, v in dataclasses.asdict(self).items() if k != 'sha256'}

    def verify(self) -> None:
        if sha256_json(self.payload()) != self.sha256: raise RuntimeError(f'GeneratorInput {self.case_id} hash mismatch')

    def case_view(self) -> dict:
        return {'case_id': self.case_id, 'scenario': self.scenario, 'question': self.question, 'options': dict(self.options)}

    def prefix(self) -> list[dict]:
        return [{'speaker': s, 'stage': st, 'round': r, 'text': t} for s, st, r, t in self.history]


@dataclass(frozen=True)
class EphemeralRubricRecord:
    rubric_id: str
    rubric_hash: Optional[str]
    prompt_id: str
    prompt_sha256: str
    state_key: str
    generator_input_sha256: str
    rubric: Optional[dict]
    gates: Optional[dict]
    valid: bool
    failure: Optional[str]                                # None | malformed_after_max_attempts | gate_failed
    attempts: tuple[dict, ...]
    created_step: int


@dataclass(frozen=True)
class ContinuationRecord:
    key: str
    base_model: str
    policy_sha256: str
    protocol_sha256: str
    case_id: str
    state_key: str
    response_sha256: str                                  # or 'NATURAL'
    seed: int
    final_option: Optional[str]
    success: Optional[float]
    termination_reason: Optional[str]
    natural_response: Optional[str]
    dialogue: list


@dataclass(frozen=True)
class BranchRecord:
    response_id: str
    path_id: str
    parent_path_id: str
    state_key: str
    turn: int
    seed: int
    prompt_hash: str
    response: str
    response_sha256: str
    completion_ids: tuple[int, ...]
    old_policy_logprobs: tuple[float, ...]
    continuation_keys: tuple[str, ...]
    outcomes: tuple[Optional[float], ...]
    R_O: Optional[float]
    R_A: Optional[float]
    R_A_hard: Optional[float]
    R_A_probabilities: Optional[dict]
    R_E: Optional[float]
    reward: Optional[float]
    behavior_logprobs: tuple[float, ...] = ()          # teacher-forced log pi_old


@dataclass(frozen=True)
class StateEvidence:
    step: int
    state_key: str
    case_id: str
    turn: int
    rollout_seed: int
    generator_input: GeneratorInput
    rubric: EphemeralRubricRecord
    branches: tuple[BranchRecord, ...]
    prompt_id: str
    prompt_sha256: str
    policy_sha256: str
    informative: bool
    Q_cov: Optional[float]                                # None iff not informative
    V_h: Optional[float]
    baseline_keys: tuple[str, ...]
    diagnostics: dict
    update: dict


@dataclass(frozen=True)
class PromptVersion:
    prompt_id: str
    parent_prompt_id: Optional[str]
    sha256: str
    event: int
    depth: int
    rank: Optional[int]
    status: str                                           # accepted | rejected | retained | pruned
    source: str                                           # config | refiner
    reflection: Optional[str]
    direction: Optional[str]
    search_state_keys: tuple[str, ...]
    acceptance_state_keys: tuple[str, ...]
    metrics: dict
    acceptance: Optional[dict]
    rejection_reason: Optional[str]


@dataclass(frozen=True)
class RunState:
    run_id: str
    config_sha256: str
    run_seed: int
    step: int
    optimizer_steps: int
    checkpoint: str
    checkpoint_sha256: str
    adapter_sha256: str
    prompt_id: str
    prompt_sha256: str
    evidence_cursor: int
    lineage_cursor: int
    last_event_step: int
    finished: bool = False
    history: tuple[dict, ...] = field(default_factory=tuple)   # accepted prompt chain


def to_dict(record) -> dict:
    return dataclasses.asdict(record)


def from_dict(cls, data: dict):
    hints = typing.get_type_hints(cls)
    return cls(**{f.name: _coerce(hints[f.name], data[f.name]) for f in dataclasses.fields(cls) if f.name in data})


def _coerce(tp, value):
    if value is None: return None
    origin, args = typing.get_origin(tp), typing.get_args(tp)
    if origin in (typing.Union, types.UnionType):
        return _coerce(next(a for a in args if a is not type(None)), value)
    if dataclasses.is_dataclass(tp): return from_dict(tp, value)
    if origin is tuple:
        if len(args) == 2 and args[1] is Ellipsis: return tuple(_coerce(args[0], v) for v in value)
        return tuple(_coerce(a, v) for a, v in zip(args, value))
    return value


def atomic_write_json(path: Path, obj) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w') as f:
        json.dump(obj, f, ensure_ascii=False, default=str); f.flush(); os.fsync(f.fileno())
    tmp.replace(path)


def read_json(path: Path):
    return json.loads(Path(path).read_text())


def dir_sha256(path: Path) -> str:
    """Hash of every file (relative path + bytes) under a checkpoint directory."""
    h = hashlib.sha256()
    for p in sorted(q for q in Path(path).rglob('*') if q.is_file()):
        h.update(str(p.relative_to(path)).encode()); h.update(p.read_bytes())
    return h.hexdigest()
