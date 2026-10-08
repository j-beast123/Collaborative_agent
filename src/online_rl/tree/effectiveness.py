"""Verification of a trajectory's one rubric. Failure trajectory: Score-2 behavior, s(tau) = successful reruns / K.
Success trajectory: Score-0 behavior, s(tau) = failed reruns / K. The behavior is written into each source response the
rubric matches, followed by K continuations (K = reruns per intervention, not a rubric count). s(tau) is stored as R_O
and only keeps/removes the rubric, weights it inside R_A and scores Rubric-Generator prompts."""
from __future__ import annotations
import inspect
from typing import Callable, Optional
from src.online_rl.integrated.continuation import derive_seed
from src.online_rl.tree.cache import CallCache
from src.online_rl.tree.decisions import Strategies
from src.online_rl.tree.records import AgentNode, InterventionRun, Leaf, TrajectoryRubric, RubricEffectiveness, SelectionDecision, to_dialogue
from src.online_rl.tree.scoring import score_nodes


def intervention_level(source_outcome: float) -> str:
    if source_outcome not in (0.0, 1.0): raise ValueError(f'source outcome must be 0 or 1, got {source_outcome!r}')
    return '0' if source_outcome == 1.0 else '2'


def target_rate(level: str, outcomes: list) -> float:
    """Share of reruns reaching the target outcome (success for Score-2, failure for Score-0); an unparseable FINAL is not correct."""
    target = 1.0 if level == '2' else 0.0
    return sum((o or 0.0) == target for o in outcomes) / len(outcomes)


def rerun_turn_cache(cache: CallCache, case: dict, rerun: dict, index: int) -> Callable:
    """Inner cache of one rerun: every Doctor call / Agent turn of the continuation is stored when it succeeds, keyed by
    the complete rerun identity (case, rubric, target node, level, policy, intervention reply, rerun seed and index) plus
    the turn's own identity (model, system prompt, decoding, stage, input sha256). A resumed rerun reuses the turns
    already observed and continues from the first missing one; the outer 'rerun' record is unchanged. A failed call
    raises before anything is stored."""
    owner = {**rerun, 'case_id': case['case_id'], 'rerun_index': index}
    return lambda kind, identity, fn: cache.call(f'rerun_{kind}_turn', {'rerun': owner, 'turn': identity}, fn)


def run_interventions(case: dict, leaf: Leaf, node: AgentNode, rub: TrajectoryRubric, *, reruns: int, run_seed: int, policy_sha256: str,
                      writer: Callable, continuation: Callable, cache: CallCache, workers: int = 1,
                      indices: Optional[list[int]] = None) -> list[InterventionRun]:
    """writer(case, prefix, turn, rubric, level, seed) -> reply; continuation(case, prefix, turn, reply, seed) -> {'final_option', 'outcome'}.
    indices (default: all K) runs only those rerun indices; each index keeps its own seed and cache key, so a subset is
    exactly the corresponding part of the full K reruns."""
    if not isinstance(reruns, int) or reruns < 1: raise ValueError('reruns_per_intervention (K) must be a positive integer')
    if node.node_id not in leaf.path: raise ValueError(f'{node.node_id} is not an Agent turn of {leaf.leaf_id}')
    level = intervention_level(leaf.outcome); prefix = to_dialogue(node.prefix); key = node.node_id   # seeds shared by every rubric on this target
    if getattr(node, 'prefix_doctor_status', None) is not None and prefix:                  # protocol v2: status of the last Doctor message
        prefix[-1] = {**prefix[-1], 'status': node.prefix_doctor_status}
    ident = {'rubric': rub.rubric_id, 'node': node.node_id, 'level': level, 'policy': policy_sha256}
    text = cache.call('behavior_text', ident, lambda: writer(case, prefix, node.depth, rub.rubric, level, derive_seed(run_seed, key, 'guided', 0)))
    turns = 'turn_cache' in inspect.signature(continuation).parameters

    def one(i):                                                     # rerun i: its own seed, cache key and per-turn cache
        seed = derive_seed(run_seed, key, 'rerun', i); rerun = {**ident, 'text': text, 'seed': seed}
        tc = rerun_turn_cache(cache, case, rerun, i) if turns else None
        r = cache.call('rerun', rerun, lambda: continuation(case, prefix, node.depth, text, seed, **({'turn_cache': tc} if tc else {})))
        return InterventionRun(rub.rubric_id, leaf.leaf_id, leaf.outcome, level, node.node_id, node.depth, text, seed,
                               r.get('final_option') or '', r.get('outcome'), policy_sha256)
    idx = list(range(reruns)) if indices is None else list(indices)
    if any(not 0 <= i < reruns for i in idx): raise ValueError(f'rerun indices {idx} outside 0..{reruns - 1}')
    if workers > 1 and len(idx) > 1:                                # independent reruns; the policy serialises its own generation
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(min(workers, len(idx))) as ex: return list(ex.map(one, idx))
    return [one(i) for i in idx]


def evaluate_rubric(case: dict, nodes: dict, leaf: Leaf, rub: TrajectoryRubric, *, scorer: Callable, writer: Callable,
                    continuation: Callable, reruns: int, run_seed: int, policy_sha256: str, strategies: Strategies,
                    cache: CallCache, workers: int = 1) -> tuple[list, Optional[RubricEffectiveness]]:
    """Match the trajectory's rubric on its own source trajectory, intervene at every matched response, K reruns each,
    aggregate per target into s(tau). An invalid rubric (structure or gate) is not verified: ([], None)."""
    if rub.leaf_id != leaf.leaf_id: raise ValueError(f'rubric {rub.rubric_id} was written for {rub.leaf_id}, not {leaf.leaf_id}')
    if not rub.valid: return [], None
    source_scores = score_nodes(case, [nodes[n] for n in leaf.path], [rub], scorer, cache); level = intervention_level(leaf.outcome)
    targets = [s.response_node_id for s in source_scores if s.match]
    runs = {t: run_interventions(case, leaf, nodes[t], rub, reruns=reruns, run_seed=run_seed, policy_sha256=policy_sha256, writer=writer,
                                 continuation=continuation, cache=cache, workers=workers) for t in targets}
    per_node = {t: target_rate(level, [r.outcome for r in rs]) for t, rs in runs.items()}
    if not targets: status, s = 'no_match', None
    elif len(targets) == 1: status, s = 'single_target', per_node[targets[0]]
    else: status, s = 'multi_target', strategies.need('multi_target')(per_node)
    return source_scores, RubricEffectiveness(rub.rubric_id, leaf.leaf_id, level, tuple(targets), tuple(r for t in targets for r in runs[t]), per_node, s, status)


def select(effs: list[RubricEffectiveness], policy: dict) -> list[SelectionDecision]:
    """KEEP / REMOVE of each trajectory's rubric by its own verification: {'mode': 'threshold', 'min_R_O': x} keeps it iff
    s(tau) >= x. A rubric without s(tau) (No Match) is removed."""
    if policy.get('mode') != 'threshold': raise ValueError(f"selection.mode must be 'threshold', got {policy.get('mode')!r}")
    t = policy['min_R_O']
    if not isinstance(t, (int, float)): raise ValueError(f'selection.min_R_O must be a number, got {t!r}')
    return [SelectionDecision(e.rubric_id, e.leaf_id, e.R_O is not None and e.R_O >= t, 'threshold', f'{e.status}: s={e.R_O} vs {t}') for e in effs]
