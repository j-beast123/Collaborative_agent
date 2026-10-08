"""Training-time rubric verification, protocol bounded_witness_v1: does a consolidated rubric have a causal witness?

Same-reason consolidation maps every decided trajectory to one rubric of its outcome stratum (rubric_bank.build_banks).
Verification here asks one question per gate-passed rubric R: is there, among a few DIFFERENT interaction contexts that
share R's behavioral reason, one where acting on R changes the downstream outcome? It does not estimate every member's
s_j (prompt-evolution J still does that, through rubric_bank.verify_members, on its own path).

- Support order (tree topology only, never rerun outcomes): R's source trajectory (the member whose rubric was generated)
  first; then, greedily, the member whose path shares the shortest prefix with every member already chosen (largest
  minimum branch distance); ties by leaf id.
- Target: the member's own behavioral locus (feedback v3 target_reply_id = Agent reply n = depth-n node of its path).
  A member without a locus is skipped. The rubric must be Stage-1 Applicable at that reply (the scorer's applicability
  question only; no level call); otherwise the member is skipped and does not count as a support.
- Causal test per valid support: the normal Score-2 (failure source) / Score-0 (success source) intervention reply and
  K reruns, existing seeds and cache keys. Reruns 0 and 1 first; the support is decided as soon as two reruns agree
  (PASS: both reach the target outcome; FAIL: both miss it); rerun 2 runs only after a split. PASS iff >= 2 of K = 3.
- KEEP at the first passing support (verification of R stops there); after max_valid_supports valid supports without a
  witness, or when no candidate is left: NOT_VALIDATED (no witness found under bounded verification; not 'causally
  false'). Only KEEP rubrics enter response scoring; each enters once, with equal weight (agent_reward.r_a_weighting)."""
from __future__ import annotations
import dataclasses
from typing import Callable, Optional
from src.online_rl.tree.cache import CallCache
from src.online_rl.tree.effectiveness import intervention_level, run_interventions
from src.online_rl.tree.posthoc import case_view
from src.online_rl.tree.records import AgentNode, Leaf, RubricAssignment, TrajectoryRubric, to_dialogue
from src.online_rl.tree.scoring import VISIBLE
from src.online_rl.integrated.records import sha256_json

PROTOCOL = 'bounded_witness_v1'


@dataclasses.dataclass(frozen=True)
class WitnessVerification:
    rubric_id: str
    stratum: str
    support_count: int                    # members mapped to the rubric
    support_trajectory_ids: tuple[str, ...]
    candidate_order: tuple[str, ...]      # topology order in which members were considered
    tested: tuple[dict, ...]              # per considered member: leaf, target_reply_id, node, stage1, reruns, outcomes, result
    num_valid_supports: int               # considered members with a locus whose target was Stage-1 Applicable
    num_reruns_executed: int
    validated_by_trajectory_id: Optional[str]
    decision: str                         # KEEP | NOT_VALIDATED
    stop_reason: str                      # witness_found | max_valid_supports | candidates_exhausted


def shared_prefix(a: Leaf, b: Leaf) -> int:
    n = 0
    for x, y in zip(a.path, b.path):
        if x != y: break
        n += 1
    return n


def support_order(members: list[Leaf], source_leaf_id: str) -> list[Leaf]:
    """Source first, then greedy maximum branch diversity: the member minimising its largest shared path prefix with the
    members already chosen; ties by leaf id. Depends only on tree paths."""
    rest = sorted(members, key=lambda l: l.leaf_id)
    src = next((l for l in rest if l.leaf_id == source_leaf_id), rest[0]); chosen = [src]; rest.remove(src)
    while rest:
        nxt = min(rest, key=lambda l: (max(shared_prefix(l, c) for c in chosen), l.leaf_id))
        chosen.append(nxt); rest.remove(nxt)
    return chosen


def stage1_decision(case: dict, node: AgentNode, rubric: dict, applicability: Callable, cache: CallCache) -> dict:
    """Stage-1 applicability of one rubric (dict with rubric_id) at one node; cached per (node, policy, rubric id, scorer)."""
    visible = [{k: rubric[k] for k in VISIBLE}]
    ident = {'node': node.node_id, 'policy': node.policy_sha256, 'rubric': rubric['rubric_id'],
             'model': getattr(applicability, 'model', None), 'protocol': getattr(applicability, 'protocol', None)}
    if rubric.get('source_repair'): ident['content'] = sha256_json(visible)          # v2.1 repaired condition: same id, its own Stage-1 key
    r = cache.call('rubric_applicability', ident,
                   lambda: applicability(case_view(case), to_dialogue(node.prefix), visible))
    return r[rubric['rubric_id']]


def stage1_applicable(case: dict, node: AgentNode, rub: TrajectoryRubric, applicability: Callable, cache: CallCache) -> bool:
    return bool(stage1_decision(case, node, rub.rubric, applicability, cache)['applicable'])


def source_stage1_for(case: dict, nodes: dict, leaf: Leaf, locus: Optional[int], applicability: Callable, cache: CallCache) -> Callable:
    """v2.1: rubric dict -> {'applicable', 'stage1', 'node_id'} at the leaf's own behavioral locus, or None without a locus."""
    if locus is None or not 1 <= locus <= len(leaf.path): return lambda rubric: None
    node = nodes[leaf.path[locus - 1]]
    def check(rubric):
        r = stage1_decision(case, node, rubric, applicability, cache)
        return {'applicable': bool(r['applicable']), 'stage1': r.get('stage1'), 'node_id': node.node_id}
    return check


def witness_test(case: dict, leaf: Leaf, node: AgentNode, rub: TrajectoryRubric, *, reruns: int, run_seed: int, policy_sha256: str,
                 writer: Callable, continuation: Callable, cache: CallCache) -> tuple[bool, list]:
    """K reruns at one target with the 2-of-K short-circuit (K = 3: indices 0 and 1, index 2 only after a split)."""
    if reruns != 3: raise ValueError('bounded_witness_v1 is defined for K = 3')
    target = 1.0 if intervention_level(leaf.outcome) == '2' else 0.0
    kw = dict(reruns=reruns, run_seed=run_seed, policy_sha256=policy_sha256, writer=writer, continuation=continuation, cache=cache)
    runs = run_interventions(case, leaf, node, rub, indices=[0, 1], workers=2, **kw)
    hits = [(r.outcome or 0.0) == target for r in runs]
    if hits[0] != hits[1]:
        runs += run_interventions(case, leaf, node, rub, indices=[2], **kw); hits.append((runs[-1].outcome or 0.0) == target)
    return sum(hits) >= 2, runs


def verify_witness(case: dict, nodes: dict[str, AgentNode], leaves: dict[str, Leaf], canon: dict[str, TrajectoryRubric],
                   assignments: list[RubricAssignment], loci: dict[str, Optional[int]], *, applicability: Callable, writer: Callable,
                   continuation: Callable, reruns: int, run_seed: int, policy_sha256: str, cache: CallCache, max_valid_supports: int = 4,
                   workers: int = 1) -> dict[str, WitnessVerification]:
    """One WitnessVerification per gate-passed rubric. Rubrics are independent; workers > 1 (execution only) verifies them
    concurrently, results keyed by rubric id."""
    if not isinstance(max_valid_supports, int) or max_valid_supports < 1: raise ValueError('max_valid_supports must be a positive integer')

    def one(cid):
        rub = canon[cid]; ms = [a for a in assignments if a.canonical_rubric_id == cid]
        source = next((a.leaf_id for a in ms if a.assignment == 'CREATED_NEW'), ms[0].leaf_id)
        order = support_order([leaves[a.leaf_id] for a in ms], source)
        tested, valid, n_reruns, by = [], 0, 0, None
        for lf in order:
            if valid >= max_valid_supports: break
            n = loci.get(lf.leaf_id)
            row = {'leaf_id': lf.leaf_id, 'target_reply_id': n, 'node_id': None, 'stage1_applicable': None, 'reruns': 0, 'outcomes': [], 'result': None}
            if n is None or not 1 <= n <= len(lf.path): tested.append({**row, 'result': 'no_locus'}); continue
            node = nodes[lf.path[n - 1]]; row['node_id'] = node.node_id
            ok = stage1_applicable(case, node, rub, applicability, cache); row['stage1_applicable'] = ok
            if not ok: tested.append({**row, 'result': 'no_match'}); continue
            valid += 1
            passed, runs = witness_test(case, lf, node, rub, reruns=reruns, run_seed=run_seed, policy_sha256=policy_sha256,
                                        writer=writer, continuation=continuation, cache=cache)
            n_reruns += len(runs)
            tested.append({**row, 'reruns': len(runs), 'outcomes': [r.outcome for r in runs], 'level': runs[0].level,
                           'positive': sum((r.outcome or 0.0) == (1.0 if runs[0].level == '2' else 0.0) for r in runs),
                           'result': 'PASS' if passed else 'FAIL'})
            if passed: by = lf.leaf_id; break
        stop = 'witness_found' if by else ('max_valid_supports' if valid >= max_valid_supports else 'candidates_exhausted')
        return cid, WitnessVerification(cid, ms[0].stratum, len(ms), tuple(a.leaf_id for a in ms), tuple(l.leaf_id for l in order),
                                        tuple(tested), valid, n_reruns, by, 'KEEP' if by else 'NOT_VALIDATED', stop)
    ids = [cid for cid, r in canon.items() if r.valid]
    if workers > 1 and len(ids) > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(min(workers, len(ids))) as pool: out = dict(pool.map(one, ids))
    else: out = dict(one(cid) for cid in ids)
    return {cid: out[cid] for cid in ids}


def witness_telemetry(verifications: dict[str, WitnessVerification]) -> dict:
    v = list(verifications.values())
    return {'protocol': PROTOCOL, 'rubrics': len(v), 'keep': sum(x.decision == 'KEEP' for x in v),
            'not_validated': sum(x.decision == 'NOT_VALIDATED' for x in v),
            'kept_success': sum(x.decision == 'KEEP' and x.stratum == 'success' for x in v),
            'kept_failure': sum(x.decision == 'KEEP' and x.stratum == 'failure' for x in v),
            'valid_supports': sum(x.num_valid_supports for x in v), 'reruns': sum(x.num_reruns_executed for x in v),
            'stage1_calls': sum(1 for x in v for t in x.tested if t['stage1_applicable'] is not None),
            'no_locus': sum(1 for x in v for t in x.tested if t['result'] == 'no_locus'),
            'no_match': sum(1 for x in v for t in x.tested if t['result'] == 'no_match')}
