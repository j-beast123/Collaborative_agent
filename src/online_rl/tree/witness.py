"""Training-time rubric verification, protocol bounded_witness_v1: does a consolidated rubric have a causal witness?

Same-reason consolidation maps every decided trajectory to one rubric of its outcome stratum (rubric_bank.build_banks).
Verification here asks one question per gate-passed rubric R: is there, among a few DIFFERENT interaction contexts that
share R's behavioral reason, one where acting on R changes the downstream outcome? It does not estimate every member's
s_j (prompt-evolution J still does that, through rubric_bank.verify_members, on its own path).

- Support order (tree topology only, never rerun outcomes): R's source trajectory (the member whose rubric was generated)
  first; then, greedily, the member whose path shares the shortest prefix with every member already chosen (largest
  minimum branch distance); ties by leaf id.
- Target: with joint instruction generation, the member assignment's own selected Agent reply and frozen guide;
  legacy rubrics without instructions use the feedback behavioral locus. Missing joint specifications are errors.
  A legacy member without a locus is skipped. The rubric must be Stage-1 Applicable at that reply (the scorer's applicability
  question only; no level call); otherwise the member is skipped and does not count as a support.
- Causal test per valid support: the normal Score-2 (failure source) / Score-0 (success source) intervention reply and
  K reruns, existing seeds and cache keys. Reruns 0 and 1 first; the support is decided as soon as two reruns agree
  (PASS: two parsed outcomes reach the target; FAIL: two parsed outcomes miss it). Otherwise rerun 2 runs. An unparsed
  outcome is neither a hit nor a miss; without two real hits or misses after K = 3 the support is INCOMPLETE.
- KEEP at the first passing support (verification of R stops there); after max_valid_supports valid supports without a
  witness, or when no candidate is left: NOT_VALIDATED (no witness found under bounded verification; not 'causally
  false'). A rubric without a PASS and with any INCOMPLETE support remains INCOMPLETE. Only KEEP rubrics enter response
  scoring; each enters once, with equal weight (agent_reward.r_a_weighting)."""
from __future__ import annotations
import dataclasses
import hashlib
import random
from typing import Callable, Optional
from src.online_rl.tree.cache import CallCache
from src.online_rl.tree.effectiveness import intervention_level, reaches_target, run_interventions
from src.online_rl.tree.member_intervention import resolve_member_intervention
from src.online_rl.tree.posthoc import case_view
from src.online_rl.tree.records import AgentNode, Leaf, RubricAssignment, TrajectoryRubric, to_dialogue
from src.online_rl.tree.scoring import VISIBLE
from src.online_rl.integrated.records import sha256_json
from src.online_rl.tree.agent_state import visible_rubric

PROTOCOL = 'bounded_witness_v1'
ALL_MEMBERS_PROTOCOL = 'all_members_witness_v1'
RANDOM10_PROTOCOL = 'random10_witness_v1'
RANDOM5_PROTOCOL = 'random5_witness_v1'
RANDOM10_SAMPLING = 'uniform_random_without_replacement_v1'
RANDOM5_SAMPLING = 'uniform_random5_without_replacement_v1'
FIXED_SOURCE_PROTOCOL = 'fixed_source_linear_majority_v1'
PROTOCOLS = (PROTOCOL, ALL_MEMBERS_PROTOCOL, RANDOM10_PROTOCOL, RANDOM5_PROTOCOL, FIXED_SOURCE_PROTOCOL)


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
    decision: str                         # KEEP | NOT_VALIDATED | INCOMPLETE
    stop_reason: str                      # witness_found | max_valid_supports | candidates_exhausted
    protocol: str = PROTOCOL


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


def random_support_sample(members: list[Leaf], *, run_seed: int, case_id: str, rubric_id: str, limit: int = 10,
                          protocol: str = RANDOM10_PROTOCOL) -> list[Leaf]:
    """Uniform, reproducible sample without replacement; independent of outcomes and leaf ordering."""
    if limit < 1: raise ValueError('random witness sample limit must be positive')
    pool = sorted(members, key=lambda l: l.leaf_id)
    material = f'{run_seed}|{case_id}|{rubric_id}|{protocol}'.encode()
    seed = int.from_bytes(hashlib.sha256(material).digest()[:8], 'big')
    return random.Random(seed).sample(pool, min(limit, len(pool)))


def stage1_decision(case: dict, node: AgentNode, rubric: dict, applicability: Callable, cache: CallCache) -> dict:
    """Stage-1 applicability, cached by the complete visible state and rubric content."""
    if getattr(applicability, 'applicability_contract', None) == 'compact_condition_v7_1':
        from src.online_rl.tree.applicability import state_applicability
        return state_applicability(case, node, rubric, applicability, cache)
    visible = [visible_rubric(rubric)]
    ident = {'node': node.node_id, 'policy': node.policy_sha256, 'rubric': rubric['rubric_id'],
             'content': sha256_json(visible),
             'state_sha256': sha256_json([case_view(case), to_dialogue(node.prefix)]),
             'model': getattr(applicability, 'model', None), 'protocol': getattr(applicability, 'protocol', None)}
    r = cache.call('rubric_applicability', ident,
                   lambda: applicability(case_view(case), to_dialogue(node.prefix), visible))
    return r[rubric['rubric_id']]


def stage1_applicable(case: dict, node: AgentNode, rub: TrajectoryRubric, applicability: Callable, cache: CallCache) -> bool:
    return bool(stage1_decision(case, node, rub.rubric, applicability, cache)['applicable'])


def source_stage1_for(case: dict, nodes: dict, leaf: Leaf, locus: Optional[int], applicability: Callable, cache: CallCache) -> Callable:
    """v2.1: rubric dict -> {'applicable', 'stage1', 'node_id'} at the leaf's own behavioral locus, or None without a locus."""
    def check(rubric):
        provenance = rubric.get('source_provenance') or {}
        selected = (provenance.get('analysis') or {}).get('selected_agent_turn_id') if provenance.get('selected_turn_policy') == 'generator_v1' else locus
        if isinstance(selected, bool) or not isinstance(selected, int) or not 1 <= selected <= len(leaf.path): return None
        node = nodes[leaf.path[selected - 1]]
        r = stage1_decision(case, node, rubric, applicability, cache)
        return {'applicable': bool(r['applicable']), 'stage1': r.get('stage1'), 'node_id': node.node_id, 'target_reply_id': selected}
    return check


def witness_test(case: dict, leaf: Leaf, node: AgentNode, rub: TrajectoryRubric, *, reruns: int, run_seed: int, policy_sha256: str,
                 writer: Callable, continuation: Callable, cache: CallCache, member_spec: Optional[dict] = None) -> tuple[Optional[bool], list]:
    """Two real target hits pass, two real misses fail; otherwise all three slots run and the result is incomplete."""
    if reruns != 3: raise ValueError('bounded_witness_v1 is defined for K = 3')
    level = intervention_level(leaf.outcome)
    kw = dict(reruns=reruns, run_seed=run_seed, policy_sha256=policy_sha256, writer=writer, continuation=continuation, cache=cache,
              member_spec=member_spec)
    runs = run_interventions(case, leaf, node, rub, indices=[0, 1], workers=2, **kw)
    hits = [reaches_target(level, r.outcome) if r.outcome in (0.0, 1.0) else None for r in runs]
    if hits.count(True) < 2 and hits.count(False) < 2:
        runs += run_interventions(case, leaf, node, rub, indices=[2], **kw)
        hits.append(reaches_target(level, runs[-1].outcome) if runs[-1].outcome in (0.0, 1.0) else None)
    return True if hits.count(True) >= 2 else False if hits.count(False) >= 2 else None, runs


def verify_witness(case: dict, nodes: dict[str, AgentNode], leaves: dict[str, Leaf], canon: dict[str, TrajectoryRubric],
                   assignments: list[RubricAssignment], loci: dict[str, Optional[int]], *, applicability: Callable, writer: Callable,
                   continuation: Callable, reruns: int, run_seed: int, policy_sha256: str, cache: CallCache, max_valid_supports: Optional[int] = 4,
                   workers: int = 1, support_sampling: str = 'topology') -> dict[str, WitnessVerification]:
    """One WitnessVerification per gate-passed rubric. Rubrics are independent; workers > 1 (execution only) verifies them
    concurrently, results keyed by rubric id."""
    if max_valid_supports is not None and (not isinstance(max_valid_supports, int) or max_valid_supports < 1):
        raise ValueError('max_valid_supports must be a positive integer or None (all members)')
    if support_sampling not in ('topology', RANDOM10_SAMPLING, RANDOM5_SAMPLING):
        raise ValueError(f'unknown witness support_sampling {support_sampling!r}')
    if support_sampling == RANDOM10_SAMPLING and max_valid_supports != 10:
        raise ValueError('random10_witness_v1 requires max_valid_supports = 10')
    if support_sampling == RANDOM5_SAMPLING and max_valid_supports != 5:
        raise ValueError('random5_witness_v1 requires max_valid_supports = 5')
    protocol = (RANDOM10_PROTOCOL if support_sampling == RANDOM10_SAMPLING else
                RANDOM5_PROTOCOL if support_sampling == RANDOM5_SAMPLING else
                ALL_MEMBERS_PROTOCOL if max_valid_supports is None else PROTOCOL)
    # Validate every active assignment before sampling or early stopping. Otherwise an
    # old/missing member guide could silently survive when an earlier member passes.
    specs = {}
    for assignment in assignments:
        rubric = canon[assignment.canonical_rubric_id]
        if not rubric.valid or assignment.assignment == 'CREATE_REJECTED': continue
        key = (assignment.canonical_rubric_id, assignment.leaf_id)
        if key in specs: raise ValueError('duplicate rubric member assignment')
        spec = resolve_member_intervention(rubric, leaves[assignment.leaf_id], member_spec=assignment.intervention_spec)
        if spec is not None:
            leaf = leaves[assignment.leaf_id]
            resolve_member_intervention(rubric, leaf, member_spec=spec, node=nodes[leaf.path[spec['selected_agent_turn_id'] - 1]])
        specs[key] = spec

    def one(cid):
        rub = canon[cid]; ms = [a for a in assignments if a.canonical_rubric_id == cid]
        source = next((a.leaf_id for a in ms if a.assignment == 'CREATED_NEW'), ms[0].leaf_id)
        members = [leaves[a.leaf_id] for a in ms]
        if support_sampling == RANDOM10_SAMPLING:
            order = random_support_sample(members, run_seed=run_seed, case_id=case['case_id'], rubric_id=cid,
                                          limit=10, protocol=RANDOM10_PROTOCOL)
        elif support_sampling == RANDOM5_SAMPLING:
            order = random_support_sample(members, run_seed=run_seed, case_id=case['case_id'], rubric_id=cid,
                                          limit=5, protocol=RANDOM5_PROTOCOL)
        else:
            order = support_order(members, source)
        tested, valid, n_reruns, by = [], 0, 0, None
        for lf in order:
            if max_valid_supports is not None and valid >= max_valid_supports: break
            member_spec = specs[(cid, lf.leaf_id)]
            n = member_spec['selected_agent_turn_id'] if member_spec is not None else loci.get(lf.leaf_id)
            row = {'leaf_id': lf.leaf_id, 'target_reply_id': n, 'node_id': None, 'stage1_applicable': None, 'reruns': 0, 'outcomes': [], 'result': None,
                   'intervention_spec': member_spec}
            if n is None or not 1 <= n <= len(lf.path): tested.append({**row, 'result': 'no_locus'}); continue
            node = nodes[lf.path[n - 1]]; row['node_id'] = node.node_id
            ok = stage1_applicable(case, node, rub, applicability, cache); row['stage1_applicable'] = ok
            if not ok: tested.append({**row, 'result': 'no_match'}); continue
            valid += 1
            passed, runs = witness_test(case, lf, node, rub, reruns=reruns, run_seed=run_seed, policy_sha256=policy_sha256,
                                        writer=writer, continuation=continuation, cache=cache, member_spec=member_spec)
            n_reruns += len(runs)
            parsed = [r for r in runs if r.outcome in (0.0, 1.0)]
            positives = sum(reaches_target(r.level, r.outcome) for r in parsed)
            errors = [{'rerun_index': i, 'status': 'unparsed_outcome', 'outcome': r.outcome,
                       'final_option': r.final_option} for i, r in enumerate(runs) if r.outcome not in (0.0, 1.0)]
            tested.append({**row, 'reruns': len(runs), 'outcomes': [r.outcome for r in runs], 'level': runs[0].level,
                           'positive': positives, 'target_hits': positives, 'target_misses': len(parsed) - positives,
                           'valid_outcomes': len(parsed), 'error_count': len(errors), 'errors': errors,
                           'planned_denominator': reruns, 'executed_denominator': len(runs), 'valid_denominator': len(parsed),
                           'intervention_reply': runs[0].intervention_text,
                           'rerun_records': [dataclasses.asdict(r) for r in runs],
                           'unparsed': sum(r.outcome is None for r in runs),
                           'result': 'PASS' if passed is True else 'FAIL' if passed is False else 'INCOMPLETE'})
            if passed is True: by = lf.leaf_id; break
        stop = 'witness_found' if by else ('max_valid_supports' if max_valid_supports is not None and valid >= max_valid_supports else 'candidates_exhausted')
        decision = 'KEEP' if by else 'INCOMPLETE' if any(t['result'] == 'INCOMPLETE' for t in tested) else 'NOT_VALIDATED'
        return cid, WitnessVerification(cid, ms[0].stratum, len(ms), tuple(a.leaf_id for a in ms), tuple(l.leaf_id for l in order),
                                        tuple(tested), valid, n_reruns, by, decision, stop, protocol)
    ids = [cid for cid, r in canon.items() if r.valid]
    if workers > 1 and len(ids) > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(min(workers, len(ids))) as pool: out = dict(pool.map(one, ids))
    else: out = dict(one(cid) for cid in ids)
    return {cid: out[cid] for cid in ids}


def witness_telemetry(verifications: dict[str, WitnessVerification]) -> dict:
    v = list(verifications.values())
    protocols={x.protocol for x in v}
    if len(protocols)>1:raise ValueError('Mixed witness protocols')
    return {'protocol': next(iter(protocols), PROTOCOL), 'rubrics': len(v), 'keep': sum(x.decision == 'KEEP' for x in v),
            'not_validated': sum(x.decision == 'NOT_VALIDATED' for x in v),
            'incomplete': sum(x.decision == 'INCOMPLETE' for x in v),
            'incomplete_supports': sum(t['result'] == 'INCOMPLETE' for x in v for t in x.tested),
            'kept_success': sum(x.decision == 'KEEP' and x.stratum == 'success' for x in v),
            'kept_failure': sum(x.decision == 'KEEP' and x.stratum == 'failure' for x in v),
            'valid_supports': sum(x.num_valid_supports for x in v), 'reruns': sum(x.num_reruns_executed for x in v),
            'unparsed_reruns': sum(t.get('unparsed', 0) for x in v for t in x.tested),
            'valid_reruns': sum(t.get('valid_outcomes', 0) for x in v for t in x.tested),
            'rerun_errors': sum(t.get('error_count', 0) for x in v for t in x.tested),
            'stage1_calls': sum(1 for x in v for t in x.tested if t['stage1_applicable'] is not None),
            'no_locus': sum(1 for x in v for t in x.tested if t['result'] == 'no_locus'),
            'no_match': sum(1 for x in v for t in x.tested if t['result'] == 'no_match')}
