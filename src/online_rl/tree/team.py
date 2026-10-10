"""Strategies of the case tree.
- multi-target s(tau): mean of the per-target verification scores.
- R_A: s(tau)-weighted mean of the matched kept rubric scores; None if none matches.
- credit: outcome component = mean of the node's descendant leaf outcomes; reward = w_outcome * outcome + w_rubric * R_A
  (a missing R_A is omitted when an outcome reward is active; with rubric-only reward it excludes the response);
  same-state siblings form one GRPO group.
- prompt unit value: the trajectory's s(tau) (prompt search aggregates it into J = S_success * S_failure)."""
from __future__ import annotations
from typing import Optional
from src.online_rl.integrated.episode import assert_single_prompt
from src.online_rl.tree.decisions import Strategies
from src.online_rl.tree.records import ResponseRubricScore, Tree
from src.online_rl.tree.reward import descendant_outcomes

RUBRIC_ONLY_NA_POLICY = 'exclude_unassessed_v1'


def mean_target_R_O(per_node: dict[str, float]) -> float:
    return sum(per_node.values()) / len(per_node)


def effectiveness_weighted_r_a(scores: list[ResponseRubricScore], weights: dict[str, float]) -> Optional[float]:
    used = [(s.score, weights[s.rubric_id]) for s in scores if s.match and weights.get(s.rubric_id, 0) > 0]
    return sum(w * x for x, w in used) / sum(w for _, w in used) if used else None


def compose_reward(outcome: Optional[float], ra: Optional[float], w_outcome: float, w_rubric: float) -> Optional[float]:
    """Single reward formula; None excludes the response from GRPO."""
    unassessed_rubric_only = w_outcome == 0.0 and w_rubric > 0.0 and ra is None
    if outcome is None or unassessed_rubric_only: return None
    return w_outcome * outcome + (w_rubric * ra if ra is not None else 0.0)


CREDIT_MODES = ('shared', 'decision_only')


def node_credit(w_outcome: float, w_rubric: float, credit_mode: str = 'shared'):
    """shared (the frozen rule): one reward per action, applied to both of its segments.
    decision_only (diagnostic E2): reward = R_D = 1[the action's private option is the gold option] (0 when no option was written),
    and only the private segment (segment 0) enters the loss; no outcome or rubric term, no visible-reply loss."""
    if credit_mode not in CREDIT_MODES: raise ValueError(f'agent_reward.credit must be one of {CREDIT_MODES}')

    def credit(tree: Tree, leaf_outcomes: dict[str, float], node_r_a: dict[str, Optional[float]]) -> dict:
        desc = descendant_outcomes(tree); nodes, groups = {}, {}
        for n in tree.nodes:
            outs = list(desc[n.node_id].values())
            outcome = sum(outs) / len(outs) if outs else None; ra = node_r_a.get(n.node_id)
            nodes[n.node_id] = {'outcome_component': outcome, 'descendant_leaves': len(outs), 'R_A': ra,
                                'reward': compose_reward(outcome, ra, w_outcome, w_rubric)}
            trace = getattr(n, 'trace', None) or {}
            if 'private_outcome' in trace: nodes[n.node_id]['R_D'] = float(trace['private_outcome'] or 0.0)
            if credit_mode == 'decision_only':
                if 'private_outcome' not in trace: raise RuntimeError(f'decision_only credit needs a precommit action at {n.node_id}')
                nodes[n.node_id]['reward'] = nodes[n.node_id]['R_D']
            groups.setdefault(n.parent_id or 'root', []).append(n.node_id)
        out = {'nodes': nodes, 'groups': groups, 'reward_weights': {'w_outcome': w_outcome, 'w_rubric': w_rubric}}
        if credit_mode == 'decision_only': out.update(credit='decision_only', loss_segments=[0])
        return out
    return credit


def sibling_grpo_update(updater):
    """One optimizer step per case: every set of same-state siblings is one same-prompt GRPO group."""
    def update(tree: Tree, targets: dict) -> dict:
        by_id = {n.node_id: n for n in tree.nodes}; groups = []
        for key, ids in sorted(targets['groups'].items()):
            recs = [{'trajectory_id': nid, 'case_id': f'{tree.case_id}|{key}', 'reward': targets['nodes'][nid]['reward'], 'loss_segments': targets.get('loss_segments'),
                     'rollout_policy_sha256': by_id[nid].policy_sha256, 'agent_traces': [by_id[nid].trace]} for nid in ids]
            assert_single_prompt(recs); groups.append(recs)
        return updater.update_groups(groups)
    return update


def trajectory_score(unit_result: dict) -> float:
    """s(tau) of the trajectory's one rubric; 0 when the rubric is invalid or matches no reply."""
    e = unit_result['effectiveness']
    return e.R_O if e is not None and e.R_O is not None else 0.0


def team_strategies(updater, w_outcome: float, w_rubric: float, credit_mode: str = 'shared') -> Strategies:
    return Strategies(multi_target=mean_target_R_O, r_a_aggregator=effectiveness_weighted_r_a, credit_assignment=node_credit(w_outcome, w_rubric, credit_mode),
                      agent_update=sibling_grpo_update(updater), prompt_unit_value=trajectory_score)
