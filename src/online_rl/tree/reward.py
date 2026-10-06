"""Agent reward inputs: a response-level rubric component (R_A) and a final-outcome component. Rerun effectiveness
enters only as the weights of the rubric scores inside R_A, never as its own reward term."""
from __future__ import annotations
from typing import Optional
from src.online_rl.tree.decisions import Strategies
from src.online_rl.tree.records import ResponseRubricScore, Tree
from src.online_rl.tree.scoring import scores_by_response


def leaf_outcomes(tree: Tree) -> dict[str, float]:
    return {l.leaf_id: l.outcome for l in tree.leaves if l.outcome is not None}


def descendant_outcomes(tree: Tree) -> dict[str, dict[str, float]]:
    """response_node_id -> {leaf_id: final outcome} of every decided leaf below it (node/sibling-level relation)."""
    out = {n.node_id: {} for n in tree.nodes}
    for l in tree.leaves:
        if l.outcome is None: continue
        for nid in l.path: out[nid][l.leaf_id] = l.outcome
    return out


def response_r_a(scores: list[ResponseRubricScore], rubric_weights: dict[str, float], strategies: Strategies) -> dict[str, Optional[float]]:
    """Raw per-(response, rubric) records, No Match included, go to the aggregation with each retained rubric's weight."""
    agg = strategies.need('r_a_aggregator')
    return {nid: agg(ss, rubric_weights) for nid, ss in scores_by_response(scores).items()}


def agent_targets(tree: Tree, scores: list[ResponseRubricScore], rubric_weights: dict[str, float], strategies: Strategies) -> dict:
    """rubric_weights only weight rubric scores inside R_A; there is no separate rerun-effectiveness reward term."""
    return strategies.need('credit_assignment')(tree, leaf_outcomes(tree), response_r_a(scores, rubric_weights, strategies))
