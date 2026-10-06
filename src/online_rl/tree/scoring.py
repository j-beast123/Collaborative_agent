"""Response-level scoring of canonical nested rubrics through a legacy judge view.

Stored rubrics keep user_side_context / agent_side_state / behavior. Only the
scorer input is projected to name/applicability_condition/required_behavior/scores.
No Match is recorded without a score, not converted into score zero.
"""
from __future__ import annotations
from typing import Callable
from src.online_rl.integrated.metrics import hard_value, r_a
from src.online_rl.integrated.records import sha256_json
from src.online_rl.tree.cache import CallCache
from src.online_rl.tree.posthoc import case_view
from src.online_rl.tree.records import AgentNode, ResponseRubricScore, TrajectoryRubric, to_dialogue

SCORING_PROTOCOL = 'state_shared_matching_review_before_behavior_v3'


def score_nodes(case: dict, nodes: list[AgentNode], kept: list[TrajectoryRubric], scorer: Callable, cache: CallCache) -> list[ResponseRubricScore]:
    """scorer(view, prefix, visible_rubrics, reply) -> per-rubric match and level.
    Split-capable scorers match/review each distinct pre-response state once,
    batching its rubric set before scoring any response. State includes recorded
    Agent belief; sibling topology alone never establishes state equivalence.
    """
    if not kept:
        return []
    # Local import avoids importing GPU/API libraries at module import time.
    from src.online_rl.tree.backends import visible_behavior_rubric
    view = case_view(case)
    rubrics, src = [], {}
    for c in kept:
        if not c.valid or not isinstance(c.rubric, dict):
            raise ValueError(f'{c.rubric_id}: only structurally valid rubrics can be scored')
        if c.rubric_id in src:
            raise ValueError(f'Duplicate rubric_id in one scoring call: {c.rubric_id}')
        source_case = c.rubric.get('source_case_id')
        if source_case is not None and source_case != view['case_id']:
            raise ValueError(f'{c.rubric_id}: rubric belongs to a different case')
        # Do not send analysis, FINAL/outcome, feedback, or selected-turn metadata.
        rubrics.append(visible_behavior_rubric({**c.rubric, 'rubric_id': c.rubric_id}))
        src[c.rubric_id] = c.leaf_id
    rubrics.sort(key=lambda r: r['rubric_id'])
    rubric_hash = sha256_json(rubrics)
    out = []
    prepared = []
    state_matches = {}
    split = callable(getattr(scorer, 'match_state', None)) and callable(getattr(scorer, 'score_behavior', None))
    # Complete state-level matching and review before any response-level scoring.
    for n in nodes:
        if n.case_id != view['case_id']:
            raise ValueError(f'{n.node_id}: node belongs to a different case')
        prefix = to_dialogue(n.prefix)
        node_view = dict(view)
        packet = n.trace.get('pre_response_belief')
        if packet:
            node_view['agent_pre_response_state'] = packet['state']
        state_identity = {
            'judge': getattr(scorer, 'judge_identity', None), 'protocol': SCORING_PROTOCOL,
            'policy': n.policy_sha256, 'case_sha256': sha256_json(node_view),
            'prefix_sha256': sha256_json(prefix), 'rubrics_sha256': rubric_hash,
        }
        state_key = sha256_json(state_identity)
        if split and state_key not in state_matches:
            matches = cache.call('state_applicability', state_identity,
                                 lambda v=node_view, p=prefix: scorer.match_state(v, p, rubrics))
            if callable(getattr(scorer, 'review_state', None)):
                review_identity = {**state_identity, 'reviewer': getattr(scorer, 'review_identity', None)}
                reviews = cache.call('state_applicability_review', review_identity,
                                     lambda v=node_view, p=prefix: scorer.review_state(v, p, rubrics))
                if set(reviews) != set(src) or any(type(r.get('applicable')) is not bool for r in reviews.values()):
                    raise ValueError('Invalid state applicability review')
                # Independent review decides final applicability; NA never becomes 0.
                matches = {rid: {'value': 'Y' if r['applicable'] else 'N',
                                 'probabilities': {'Y': float(r['applicable']), 'N': float(not r['applicable'])}}
                           for rid, r in reviews.items()}
            state_matches[state_key] = matches
        prepared.append((n, prefix, node_view, state_key))
    for n, prefix, node_view, state_key in prepared:
        identity = {
            'judge': getattr(scorer, 'judge_identity', None), 'protocol': SCORING_PROTOCOL, 'node': n.node_id, 'policy': n.policy_sha256,
            'case_sha256': sha256_json(node_view), 'prefix_sha256': sha256_json(prefix),
            'response_sha256': sha256_json(n.response), 'rubrics_sha256': rubric_hash,
            'matching_sha256': sha256_json(state_matches[state_key]) if split else None,
            'reviewer': getattr(scorer, 'review_identity', None),
        }
        result = cache.call('rubric_score', identity,
                            lambda n=n, prefix=prefix, node_view=node_view, state_key=state_key:
                                scorer.score_behavior(node_view, prefix, rubrics, n.response, state_matches[state_key])
                                if split else scorer(node_view, prefix, rubrics, n.response))
        if not isinstance(result, dict):
            raise ValueError(f'{n.node_id}: scorer must return a per-rubric object')
        for rid in sorted(src):
            if rid not in result or not isinstance(result[rid], dict):
                raise ValueError(f'{n.node_id}: missing scorer result for {rid}')
            record = result[rid]
            if type(record.get('applicable')) is not bool:
                raise ValueError(f'{n.node_id}/{rid}: applicability must be boolean')
            if not record['applicable']:
                out.append(ResponseRubricScore(n.node_id, rid, src[rid], False, None, None, None))
                continue
            probabilities, hard = record.get('probabilities'), record.get('hard')
            if not isinstance(probabilities, dict):
                raise ValueError(f'{n.node_id}/{rid}: matched rubric requires probabilities; do not silently assign zero')
            out.append(ResponseRubricScore(n.node_id, rid, src[rid], True,
                                           r_a(probabilities), hard_value(hard) if hard is not None else None, probabilities))
    return out


def scores_by_response(scores: list[ResponseRubricScore]) -> dict[str, list[ResponseRubricScore]]:
    out: dict[str, list[ResponseRubricScore]] = {}
    for score in scores:
        out.setdefault(score.response_node_id, []).append(score)
    return out
