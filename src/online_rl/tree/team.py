"""Strategies of the case tree.
- multi-target s(tau): mean of the per-target verification scores.
- R_A: s(tau)-weighted mean of the matched kept rubric scores; None if none matches.
- credit: outcome component = mean of the node's descendant leaf outcomes; reward = w_outcome * outcome + w_rubric * R_A
  (a missing R_A is omitted when an outcome reward is active; with rubric-only reward it excludes the response);
  same-state siblings form one GRPO group.
- prompt unit value: the trajectory's s(tau) (prompt search aggregates it into J = S_success * S_failure)."""
from __future__ import annotations
import math
from typing import Optional
from src.online_rl.integrated.episode import assert_single_prompt
from src.online_rl.integrated.metrics import reward as weighted_reward
from src.online_rl.tree.decisions import Strategies
from src.online_rl.tree.records import ResponseRubricScore, Tree
from src.online_rl.tree.reward import descendant_outcomes

RUBRIC_ONLY_NA_POLICY = 'exclude_unassessed_active_components_v2'


def mean_target_R_O(per_node: dict[str, float]) -> float:
    return sum(per_node.values()) / len(per_node)


def effectiveness_weighted_r_a(scores: list[ResponseRubricScore], weights: dict[str, float]) -> Optional[float]:
    used = []
    for s in scores:
        status = getattr(s, 'scoring_status', None) or ('scored' if s.match else 'not_applicable')
        if status == 'not_retrieved':
            if s.match is not False or any(value is not None for value in (s.score, s.hard, s.probabilities)):
                raise RuntimeError(f'Invalid not_retrieved assessment: {s.response_node_id}/{s.rubric_id}')
            continue
        if status in ('group_unscorable', 'rubric_unscorable'):
            continue
        if status == 'applicability_unresolved' and not s.match and s.score is None:   # abstained pair: excluded from R_A
            continue
        if status == 'not_applicable' and not s.match:
            continue
        if status != 'scored' or not s.match:
            raise RuntimeError(f'Invalid rubric assessment status: {s.response_node_id}/{s.rubric_id}: {status}')
        if s.score is None or not math.isfinite(s.score) or not 0 <= s.score <= 1:
            raise RuntimeError(f'Missing or invalid scored rubric value: {s.response_node_id}/{s.rubric_id}')
        weight = weights.get(s.rubric_id, 0)
        if weight > 0:
            used.append((s.score, weight))
    return sum(w * x for x, w in used) / sum(w for _, w in used) if used else None


def compose_reward(outcome: Optional[float], ra: Optional[float], w_outcome: float, w_rubric: float) -> Optional[float]:
    """Single reward formula; None excludes the response from GRPO."""
    return weighted_reward(outcome, ra, w_outcome, w_rubric, rubric_valid=ra is not None)


def node_credit(w_outcome: float, w_rubric: float):
    def credit(tree: Tree, leaf_outcomes: dict[str, float], node_r_a: dict[str, Optional[float]]) -> dict:
        desc = descendant_outcomes(tree); nodes, groups = {}, {}
        for n in tree.nodes:
            outs = list(desc[n.node_id].values())
            outcome = sum(outs) / len(outs) if outs else None; ra = node_r_a.get(n.node_id)
            nodes[n.node_id] = {'outcome_component': outcome, 'descendant_leaves': len(outs), 'R_A': ra,
                                'reward': compose_reward(outcome, ra, w_outcome, w_rubric)}
            groups.setdefault(n.parent_id or 'root', []).append(n.node_id)
        return {'nodes': nodes, 'groups': groups, 'reward_weights': {'w_outcome': w_outcome, 'w_rubric': w_rubric}}
    return credit


def sibling_grpo_update(updater):
    """One optimizer step per case: every set of same-state siblings is one same-prompt GRPO group."""
    def update(tree: Tree, targets: dict) -> dict:
        by_id = {n.node_id: n for n in tree.nodes}; groups = []
        for key, ids in sorted(targets['groups'].items()):
            recs = [{'trajectory_id': nid, 'case_id': f'{tree.case_id}|{key}', 'reward': targets['nodes'][nid]['reward'],
                     'rollout_policy_sha256': by_id[nid].policy_sha256, 'agent_traces': [by_id[nid].trace]} for nid in ids]
            assert_single_prompt(recs); groups.append(recs)
        return updater.update_groups(groups)
    return update


def trajectory_score(unit_result: dict) -> float:
    """s(tau) of the trajectory's one rubric; 0 when the rubric is invalid or matches no reply."""
    e = unit_result['effectiveness']
    return e.R_O if e is not None and e.R_O is not None else 0.0


def team_strategies(updater, w_outcome: float, w_rubric: float) -> Strategies:
    return Strategies(multi_target=mean_target_R_O, r_a_aggregator=effectiveness_weighted_r_a, credit_assignment=node_credit(w_outcome, w_rubric),
                      agent_update=sibling_grpo_update(updater), prompt_unit_value=trajectory_score)
