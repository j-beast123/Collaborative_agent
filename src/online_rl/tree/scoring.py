"""Response-level rubric scoring with the two-stage scorer (applicability on the pre-reply dialogue, then the 0/1/2 level;
non-applicable = No Match, kept as a record without a score). Scores stay raw, one record per (response, rubric)."""
from __future__ import annotations
from typing import Callable
from src.online_rl.integrated.metrics import hard_value, r_a
from src.online_rl.tree.cache import CallCache
from src.online_rl.tree.posthoc import case_view
from src.online_rl.tree.records import AgentNode, ResponseRubricScore, TrajectoryRubric, to_dialogue

VISIBLE = ('rubric_id', 'name', 'applicability_condition', 'required_behavior', 'scores')


def score_nodes(case: dict, nodes: list[AgentNode], kept: list[TrajectoryRubric], scorer: Callable, cache: CallCache) -> list[ResponseRubricScore]:
    """scorer(case_view, prefix, rubrics, reply) -> {rubric_id: {'applicable': bool, 'probabilities': dict|None, 'hard': int|None}};
    all rubrics of one response go in one call."""
    if not kept: return []
    rubrics = [{k: c.rubric[k] for k in VISIBLE} for c in kept]; src = {c.rubric_id: c.leaf_id for c in kept}; out = []
    for n in nodes:
        res = cache.call('rubric_score', {'node': n.node_id, 'policy': n.policy_sha256, 'rubrics': sorted(src)},
                         lambda n=n: scorer(case_view(case), to_dialogue(n.prefix), rubrics, n.response))
        for rid in sorted(src):
            r = res[rid]
            if r['applicable']:
                out.append(ResponseRubricScore(n.node_id, rid, src[rid], True, r_a(r['probabilities']), hard_value(r['hard']), r['probabilities']))
            else:
                out.append(ResponseRubricScore(n.node_id, rid, src[rid], False, None, None, None))
    return out


def scores_by_response(scores: list[ResponseRubricScore]) -> dict[str, list[ResponseRubricScore]]:
    out: dict[str, list[ResponseRubricScore]] = {}
    for s in scores: out.setdefault(s.response_node_id, []).append(s)
    return out
