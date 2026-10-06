"""Verify one trajectory rubric at its generator-selected Agent reply only.

Each of exactly three reruns regenerates the Agent reply with the same Score-0
(success source) or Score-2 (failure source) prompt intervention, then continues
until Doctor FINAL. R_O is the desired outcome count / 3; incomplete reruns
produce no effectiveness score and cannot pass selection.
"""
from __future__ import annotations
from copy import deepcopy
import math
from typing import Callable, Optional
from src.online_rl.integrated.continuation import derive_seed
from src.online_rl.integrated.records import sha256_json
from src.online_rl.tree.cache import CallCache
from src.online_rl.tree.decisions import Strategies
from src.online_rl.tree.records import AgentNode, InterventionRun, Leaf, TrajectoryRubric, RubricEffectiveness, SelectionDecision, to_dialogue
from src.online_rl.tree.scoring import score_nodes

INTERVENTION_PROTOCOL = 'selected_agent_turn_fresh_reply_v2'
NUM_RERUNS = 3


def intervention_level(source_outcome: float) -> str:
    if source_outcome not in (0.0, 1.0):
        raise ValueError(f'source outcome must be 0 or 1, got {source_outcome!r}')
    return '0' if source_outcome == 1.0 else '2'


def target_rate(level: str, outcomes: list) -> Optional[float]:
    """No denominator shrinking or None-to-failure conversion."""
    if level not in ('0', '2'):
        raise ValueError(f'Invalid intervention level: {level!r}')
    if len(outcomes) != NUM_RERUNS or any(o is None or o not in (0.0, 1.0) for o in outcomes):
        return None
    target = 1.0 if level == '2' else 0.0
    return sum(o == target for o in outcomes) / NUM_RERUNS


def run_interventions(case: dict, leaf: Leaf, node: AgentNode, rub: TrajectoryRubric, *, reruns: int, run_seed: int, policy_sha256: str,
                      writer: Callable, continuation: Callable, cache: CallCache,
                      rerun_policy_mode: str = 'source') -> list[InterventionRun]:
    """writer(case, prefix, turn, rubric, level, seed) -> newly sampled Agent reply.

    writer MUST insert the behavior instruction into the Agent policy's prompt;
    it must not independently rewrite a response using another model. This file
    cannot enforce the callback's implementation. continuation receives that
    fresh reply and returns {'final_option', 'outcome'}; an optional status/error
    marks incomplete execution. Later-turn instruction scope remains the
    callback's explicit protocol decision, not changed here.
    """
    if type(reruns) is not int or reruns != NUM_RERUNS:
        raise ValueError('This verification protocol requires exactly 3 reruns')
    if rub.leaf_id != leaf.leaf_id or node.node_id != rub.selected_agent_turn_id or node.node_id not in leaf.path:
        raise ValueError('Intervention must use the selected Agent node of the source trajectory')
    if node.case_id != leaf.case_id or case['case_id'] != leaf.case_id:
        raise ValueError('Case, leaf and selected node must belong to the same case')
    if rerun_policy_mode not in ('source', 'current'):
        raise ValueError('rerun_policy_mode must be source or current')
    if rerun_policy_mode == 'source' and node.policy_sha256 != policy_sha256:
        raise ValueError('Rerun policy must equal the fixed source generation policy')
    level = intervention_level(leaf.outcome)
    prefix = to_dialogue(node.prefix)
    key = f'{rub.rubric_id}|{leaf.leaf_id}|{node.node_id}'
    ident = {
        'protocol': INTERVENTION_PROTOCOL, 'rubric': rub.rubric_id,
        'rubric_sha256': sha256_json(rub.rubric), 'leaf': leaf.leaf_id,
        'node': node.node_id, 'prefix_sha256': sha256_json(prefix),
        'case_sha256': sha256_json(case), 'level': level, 'policy': policy_sha256,
        'source_policy': node.policy_sha256, 'policy_mode': rerun_policy_mode,
    }
    from src.collaboration.generator import DoctorProtocolError
    runs = []
    for i in range(reruns):
        intervention_seed = derive_seed(run_seed, key, 'guided', i)
        seed = derive_seed(run_seed, key, 'rerun', i)
        text, final_option, outcome, error, final_text = '', '', None, None, ''
        try:
            text = cache.call('behavior_text', {**ident, 'index': i, 'seed': intervention_seed},
                              lambda intervention_seed=intervention_seed: writer(
                                  deepcopy(case), deepcopy(prefix), node.depth, deepcopy(rub.rubric), level, intervention_seed))
            if not isinstance(text, str) or not text.strip():
                error = 'writer returned no usable Agent reply'
                text = ''
            else:
                result = cache.call('rerun', {**ident, 'index': i, 'text': text, 'seed': seed,
                                             'intervention_seed': intervention_seed},
                                    lambda text=text, seed=seed: continuation(
                                        deepcopy(case), deepcopy(prefix), node.depth, text, seed))
                if isinstance(result, dict) and result.get('status') == 'incomplete':
                    # Preserve immutable original evidence; retry only the Doctor
                    # FINAL for incomplete cached continuations, with the same
                    # fresh Agent text and requested seed. Successful calls stay.
                    result = cache.call('rerun_final_recovery',
                        {**ident, 'index': i, 'text': text, 'seed': seed,
                         'intervention_seed': intervention_seed, 'recovery_protocol': 'bounded_final_repair_v1'},
                        lambda text=text, seed=seed: continuation(
                            deepcopy(case), deepcopy(prefix), node.depth, text, seed))
                if not isinstance(result, dict):
                    error = 'continuation returned no result object'
                else:
                    final_text = result.get('final_text') or ''
                    final_option = result.get('final_option') or ''
                    reported_status = result.get('status')
                    if result.get('error') or reported_status in ('error', 'incomplete', 'timeout'):
                        error = str(result.get('error') or reported_status)
                    elif not final_option or result.get('outcome') not in (0.0, 1.0):
                        error = 'missing parseable FINAL or valid binary outcome'
                    else:
                        outcome = float(result['outcome'])
        except (OSError, TimeoutError, DoctorProtocolError) as exc:
            # Network/IO errors are incomplete executions, never task failures.
            # Do not swallow programming exceptions; callers can fix/resume them.
            error = f'{type(exc).__name__}: {exc}'
        runs.append(InterventionRun(
            rub.rubric_id, leaf.leaf_id, leaf.outcome, level, node.node_id,
            node.depth, text, seed, final_option, outcome, policy_sha256,
            intervention_seed=intervention_seed,
            status='complete' if outcome is not None else 'incomplete', error=error, final_text=final_text,
        ))
    return runs


def evaluate_rubric(case: dict, nodes: dict, leaf: Leaf, rub: TrajectoryRubric, *, scorer: Callable, writer: Callable,
                    continuation: Callable, reruns: int, run_seed: int, policy_sha256: str, strategies: Strategies,
                    cache: CallCache, rerun_policy_mode: str = 'source') -> tuple[list, Optional[RubricEffectiveness]]:
    """Verify only the selected node. Matching other nodes belongs to later training.
    The source behavior score is diagnostic only and is not an acceptance reward.
    strategies remains in the signature for compatibility; multi_target is unused.
    """
    if type(reruns) is not int or reruns != NUM_RERUNS:
        raise ValueError('This verification protocol requires exactly 3 reruns')
    if rub.leaf_id != leaf.leaf_id:
        raise ValueError(f'rubric {rub.rubric_id} was written for {rub.leaf_id}, not {leaf.leaf_id}')
    if not rub.valid:
        return [], None
    target = rub.selected_agent_turn_id
    if not isinstance(target, str) or target not in leaf.path or target not in nodes:
        raise ValueError('Valid rubric must select a real Agent node on its source trajectory')
    node = nodes[target]
    if node.node_id != target:
        raise ValueError('Node dictionary key disagrees with node ID')
    level = intervention_level(leaf.outcome)
    source_scores = score_nodes(case, [node], [rub], scorer, cache)
    matched = any(s.response_node_id == target and s.rubric_id == rub.rubric_id and s.match for s in source_scores)
    if not matched:
        return source_scores, RubricEffectiveness(rub.rubric_id, leaf.leaf_id, level, (target,), (), {}, None, 'no_match')
    runs = run_interventions(case, leaf, node, rub, reruns=reruns, run_seed=run_seed,
                             policy_sha256=policy_sha256, writer=writer, continuation=continuation, cache=cache,
                             rerun_policy_mode=rerun_policy_mode)
    score = target_rate(level, [r.outcome for r in runs])
    status = 'single_target' if score is not None else 'incomplete'
    return source_scores, RubricEffectiveness(
        rub.rubric_id, leaf.leaf_id, level, (target,), tuple(runs), {target: score}, score, status)


def select(effs: list[RubricEffectiveness], policy: dict) -> list[SelectionDecision]:
    """Exactly three valid reruns and at least two desired outcomes are required.
    Retain the legacy threshold config shape, but reject alternate thresholds.
    """
    if policy.get('mode') != 'threshold':
        raise ValueError(f"selection.mode must be 'threshold', got {policy.get('mode')!r}")
    threshold = policy.get('min_R_O')
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not math.isclose(threshold, 2 / 3, rel_tol=0, abs_tol=1e-8):
        raise ValueError('selection.min_R_O must be 2/3 for the agreed 2-of-3 protocol')
    decisions = []
    for e in effs:
        expected_level = intervention_level(e.runs[0].source_outcome) if e.runs else None
        complete = (
            e.status == 'single_target' and len(e.target_node_ids) == 1 and len(e.runs) == NUM_RERUNS
            and e.level == expected_level
            and all(r.node_id == e.target_node_ids[0] and r.rubric_id == e.rubric_id
                    and r.leaf_id == e.leaf_id and r.level == e.level
                    and r.source_outcome == e.runs[0].source_outcome and r.status == 'complete'
                    for r in e.runs)
        )
        actual = target_rate(e.level, [r.outcome for r in e.runs]) if complete else None
        retained = actual is not None and e.R_O is not None and math.isclose(e.R_O, actual) and actual >= 2 / 3
        reason = f'{e.status}: desired outcomes={round(actual * 3) if actual is not None else "incomplete"}/3; require 2/3'
        decisions.append(SelectionDecision(e.rubric_id, e.leaf_id, retained, 'threshold', reason))
    return decisions
