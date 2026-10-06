"""Strategies of the case tree.
- multi-target s(tau): mean of the per-target verification scores.
- R_A: s(tau)-weighted mean of the matched kept rubric scores; None if none matches.
- credit: outcome component = mean of the node's descendant leaf outcomes; reward = w_outcome * outcome + w_rubric * R_A
  (a missing R_A is omitted, not renormalised); same-state siblings form one GRPO group.
- prompt unit value: completed trajectory R_O; invalid/no-match zero, technical incomplete None.
  Prompt objective: product of source-stratified mean R_O values."""
from __future__ import annotations
from typing import Optional
from src.online_rl.integrated.episode import assert_single_prompt
from src.online_rl.tree.decisions import Strategies
from src.online_rl.tree.records import ResponseRubricScore, Tree
from src.online_rl.tree.reward import descendant_outcomes


def mean_target_R_O(per_node: dict[str, float]) -> float:
    return sum(per_node.values()) / len(per_node)


def effectiveness_weighted_r_a(scores: list[ResponseRubricScore], weights: dict[str, float]) -> Optional[float]:
    used = [(s.score, weights[s.rubric_id]) for s in scores if s.match and weights.get(s.rubric_id, 0) > 0]
    return sum(w * x for x, w in used) / sum(w for _, w in used) if used else None


def node_credit(w_outcome: float, w_rubric: float):
    def credit(tree: Tree, leaf_outcomes: dict[str, float], node_r_a: dict[str, Optional[float]]) -> dict:
        desc = descendant_outcomes(tree); nodes, groups = {}, {}
        for n in tree.nodes:
            outs = list(desc[n.node_id].values())
            outcome = sum(outs) / len(outs) if outs else None; ra = node_r_a.get(n.node_id)
            reward = None if outcome is None else w_outcome * outcome + (w_rubric * ra if ra is not None else 0.0)
            nodes[n.node_id] = {'outcome_component': outcome, 'descendant_leaves': len(outs), 'R_A': ra, 'reward': reward}
            groups.setdefault(n.parent_id or 'root', []).append(n.node_id)
        return {'nodes': nodes, 'groups': groups}
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


def trajectory_score(unit_result: dict) -> Optional[float]:
    """Invalid/no-match rubrics count zero; technical incompletes defer comparison."""
    e = unit_result['effectiveness']
    if e is not None and (e.status == 'incomplete' or any(r.status != 'complete' for r in getattr(e, 'runs', ()))):
        return None
    return e.R_O if e is not None and e.R_O is not None else 0.0


def team_strategies(updater, w_outcome: float, w_rubric: float) -> Strategies:
    return Strategies(multi_target=mean_target_R_O, r_a_aggregator=effectiveness_weighted_r_a, credit_assignment=node_credit(w_outcome, w_rubric),
                      agent_update=sibling_grpo_update(updater), prompt_unit_value=trajectory_score)
