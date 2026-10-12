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
    next_doctor_status: Optional[dict] = None    # protocol v2: hidden status of next_doctor (audit / halting only; never in a dialogue)
    prefix_doctor_status: Optional[dict] = None  # protocol v2: hidden status of the Doctor message this reply answers
    prefix_partner_state: Optional[dict] = None  # private hypothesis/confidence before this reply
    next_partner_state: Optional[dict] = None


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
    rollout_budget: str = 'full_tree'


@dataclass(frozen=True)
class DoctorFeedback:
    leaf_id: str
    outcome: float
    text: str
    prompt_sha256: str
    target_reply_id: Optional[int] = None  # behavioral locus (feedback v3/v4): the ONE Agent reply the feedback's behavior is about
    locus: Optional[dict] = None          # audit: {'status', 'raw', 'leakage'}; None before v3
    valid: bool = True                    # v4: False when the answer-leakage guard still fails after its single repair
    kind: str = 'physician_after_decision' # v4: 'behavior_observer_v4' (dialogue only; no question, options, FINAL or outcome)


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
    feedback_kind: str = 'physician_after_decision'   # how the feedback was produced (part of the sha only when not the default)
    target_reply_id: Optional[int] = None  # v2.1: the feedback's behavioral locus, shown to the Rubric Generator (in the sha only when set)


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
    source_check: Optional[dict] = None   # v2.1 source-self-applicability at the source locus (status, stage1, repair); None before v2.1
    intervention_instruction: Optional[str] = None  # separate execution instruction; never part of the rubric shown to scorers



@dataclass(frozen=True)
class RubricAssignment:
    """Same-reason reuse: the ONE rubric a decided trajectory maps to within its case and outcome stratum."""
    leaf_id: str
    case_id: str
    stratum: str                          # 'success' | 'failure' (separate banks)
    order: int                            # position in the deterministic processing order of its stratum
    canonical_rubric_id: str
    assignment: str                       # CREATED_NEW | REUSED_EXISTING | CREATE_REJECTED (generated but invalid / gate-failed: no active rubric)
    decision: Optional[dict]              # the MATCH-vs-NEW answer (decision, rubric_id, checks, rationale); None when the bank was empty
    note: str                             # why a reuse was not accepted, why a generated rubric was rejected, or ''
    feedback_sha256: str                  # the trajectory's Doctor feedback (as in its rubric input)
    intervention_spec: Optional[dict] = None  # member-specific target/instruction; canonical rubric stays unchanged


@dataclass(frozen=True)
class CanonicalSupport:
    """KEEP / effectiveness of one rubric from its members' own verifications: KEEP iff max_j s_j >= min_R_O and
    R_O(R) = max_j s_j (its single effectiveness value; member count never multiplies it). Other fields: audit only."""
    rubric_id: str
    stratum: str
    member_count: int
    member_s: dict                        # leaf_id -> s_j (None when the canonical rubric matches no reply of that trajectory)
    supported_member_count: int
    support_ratio: float
    mean_s_all: Optional[float]           # over members with an s_j (None excluded)
    mean_s_supported: Optional[float]
    min_s: Optional[float]
    max_s: Optional[float]
    retained: bool
    weight: Optional[float]               # R_O(R) = max_j s_j for a kept rubric (its R_A weight), else None

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
    agent_seed: Optional[int] = None
    rerun_index: Optional[int] = None
    intervention_spec: Optional[dict] = None


@dataclass(frozen=True)
class RubricEffectiveness:
    rubric_id: str
    leaf_id: str
    level: str
    target_node_ids: tuple[str, ...]      # source-trajectory responses the rubric matches (intervention targets)
    runs: tuple[InterventionRun, ...]
    per_node_R_O: dict                    # node_id -> share of the K reruns reaching the target outcome
    R_O: Optional[float]                  # s(tau): the trajectory rubric's verification score (mean over targets); None for no_match
    status: str                           # single_target | no_match | multi_target


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
    scoring_status: Optional[str] = None  # scored | not_applicable | group_unscorable
    scoring_evidence: Optional[dict] = None
    individual_assessment: Optional[dict] = None  # original verdict before a same-state rubric mask

    def __post_init__(self):
        # Old persisted records did not carry a status; keep their original
        # applicability/score semantics rather than manufacturing an assessment.
        if self.scoring_status is None:
            object.__setattr__(self, 'scoring_status', 'scored' if self.match else 'not_applicable')
