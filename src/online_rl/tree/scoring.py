"""Response-level rubric scoring with the two-stage scorer (applicability on the pre-reply dialogue, then the 0/1/2 level;
non-applicable = No Match, kept as a record without a score). Scores stay raw, one record per (response, rubric)."""
from __future__ import annotations
from typing import Callable
from src.online_rl.integrated.metrics import hard_value, r_a
from src.online_rl.tree.cache import CallCache
from src.online_rl.tree.posthoc import case_view
from src.online_rl.tree.records import AgentNode, ResponseRubricScore, TrajectoryRubric, to_dialogue
from src.online_rl.tree.agent_state import visible_rubric

VISIBLE = ('rubric_id', 'name', 'applicability_condition', 'required_behavior', 'scores')


def score_nodes(case: dict, nodes: list[AgentNode], kept: list[TrajectoryRubric], scorer: Callable, cache: CallCache,
                workers: int = 1) -> list[ResponseRubricScore]:
    """scorer(case_view, prefix, rubrics, reply) -> {rubric_id: {'applicable': bool, 'probabilities': dict|None, 'hard': int|None}};
    all rubrics of one response go in one call. workers > 1 (execution only): the per-response calls are independent and run
    concurrently; results are collected in node order."""
    if not kept: return []
    rubrics = [visible_rubric(c.rubric) for c in kept]; src = {c.rubric_id: c.leaf_id for c in kept}; out = []

    def one(n):
        return cache.call('rubric_score', {'node': n.node_id, 'policy': n.policy_sha256, 'rubrics': sorted(src), 'model': getattr(scorer, 'model', None),
                                           'protocol': getattr(scorer, 'protocol', None)},
                          lambda: scorer(case_view(case), to_dialogue(n.prefix), rubrics, n.response))
    if workers > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(workers) as pool: results = list(pool.map(one, nodes))
    else: results = [one(n) for n in nodes]
    for n, res in zip(nodes, results):
        for rid in sorted(src):
            r = res[rid]
            if r['applicable']:
                score = r_a(r['probabilities']) if r['probabilities'] else hard_value(r['hard'])     # soft (legacy Jev) or level / 2 (hard)
                out.append(ResponseRubricScore(n.node_id, rid, src[rid], True, score, hard_value(r['hard']), r['probabilities']))
            else:
                out.append(ResponseRubricScore(n.node_id, rid, src[rid], False, None, None, None))
    return out


def scores_by_response(scores: list[ResponseRubricScore]) -> dict[str, list[ResponseRubricScore]]:
    out: dict[str, list[ResponseRubricScore]] = {}
    for s in scores: out.setdefault(s.response_node_id, []).append(s)
    return out
