"""Case-tree records. A dialogue entry is (speaker, stage, round, text)."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

Turn = tuple[str, str, int, str]


def to_turns(dialogue: list[dict]) -> tuple[Turn, ...]:
    return tuple((d['speaker'], d.get('stage') or '', int(d.get('round') or 0), d['text']) for d in dialogue)


def to_dialogue(turns) -> list[dict]:
    return [{'speaker': s, 'stage': st, 'round': r, 'text': t} for s, st, r, t in turns]


@dataclass(frozen=True)
class AgentNode:
    node_id: str                          # '<case_id>/<i1>.<i2>...' : child index per Agent decision point
    parent_id: Optional[str]
    case_id: str
    depth: int                            # Agent turn (1-based)
    prefix: tuple[Turn, ...]              # dialogue before this reply; ends with the Doctor message it answers
    response: str
    trace: dict                           # prompt/completion ids, generation and behavior-policy log-probs
    policy_sha256: str
    seed: int
    stop_reason: Optional[str]            # set on terminal nodes
    terminal: bool
    next_doctor: Optional[str]            # Doctor reply to this branch (non-terminal nodes)


@dataclass(frozen=True)
class Leaf:
    leaf_id: str                          # the terminal node id
    case_id: str
    path: tuple[str, ...]                 # node ids root -> leaf
    dialogue: tuple[Turn, ...]            # full trajectory including the Doctor FINAL
    final_text: str
    final_option: str
    outcome: Optional[float]              # 1.0 success, 0.0 failure, None if the FINAL has no parseable option
    stop_reason: str


@dataclass(frozen=True)
class Tree:
    case_id: str
    policy_sha256: str
    max_agent_turns: int
    branching: int
    nodes: tuple[AgentNode, ...]
    leaves: tuple[Leaf, ...]


@dataclass(frozen=True)
class DoctorFeedback:
    leaf_id: str
    outcome: float
    text: str
    prompt_sha256: str


@dataclass(frozen=True)
class TrajectoryRubricInput:
    """Everything the post-hoc Rubric Generator sees. No gold answer field: the current contract is no-gold."""
    leaf_id: str
    case_id: str
    scenario: str
    question: str
    options: tuple[tuple[str, str], ...]
    dialogue: tuple[Turn, ...]            # full trajectory, last entry is the Doctor FINAL
    outcome: str                          # 'success' | 'failure'
    feedback: str
    sha256: str
    agent_node_ids: tuple[str, ...] = ()   # Agent replies in dialogue order; copied from Leaf.path
    agent_pre_response_states: tuple[dict, ...] = ()


@dataclass(frozen=True)
class TrajectoryRubric:
    """The ONE rubric r_tau the Rubric Generator writes for one completed trajectory (one generator call)."""
    rubric_id: str
    leaf_id: str
    prompt_sha256: str
    raw: Optional[str]
    rubric: Optional[dict]
    valid: bool                           # parsed and structurally complete (0/1/2 levels)
    problems: tuple[str, ...]
    gate: Optional[dict]                  # post-hoc gate result (None when no gate_fn is given)
    selected_agent_turn_id: Optional[str] = None
    # Canonical AgentNode.node_id of the ONE source reply selected for intervention.
    # Parser resolves analysis.selected_agent_turn_id to a source leaf.path node.
    # None supports legacy/invalid records; new rubrics must have a valid selection
    # before verification. Selection is NOT the set of applicability matches.


@dataclass(frozen=True)
class InterventionRun:
    rubric_id: str
    leaf_id: str
    source_outcome: float
    level: str                            # '0' on success sources, '2' on failure sources
    node_id: str                          # intervened Agent turn
    turn: int
    intervention_text: str
    seed: int
    final_option: str
    outcome: Optional[float]
    continuation_policy_sha256: str       # policy that wrote the intervention reply and ran the continuation
    intervention_seed: Optional[int] = None  # separate seed for fresh intervention reply
    status: str = 'complete'              # complete | incomplete
    error: Optional[str] = None
    final_text: str = ''


@dataclass(frozen=True)
class RubricEffectiveness:
    rubric_id: str
    leaf_id: str
    level: str
    target_node_ids: tuple[str, ...]      # one generator-selected Agent node; not all applicability matches
    runs: tuple[InterventionRun, ...]
    per_node_R_O: dict                    # node_id -> share of the K reruns reaching the target outcome
    R_O: Optional[float]                  # desired outcome count / 3; None for no_match or incomplete
    status: str                           # single_target | no_match | incomplete


@dataclass(frozen=True)
class SelectionDecision:
    rubric_id: str
    leaf_id: str
    retained: bool
    policy: str
    reason: str


@dataclass(frozen=True)
class ResponseRubricScore:
    response_node_id: str
    rubric_id: str
    source_leaf_id: str
    match: bool                           # applicability; False = No Match ('NA'), no score
    score: Optional[float]                # expected level / 2 when matched
    hard: Optional[float]
    probabilities: Optional[dict]
