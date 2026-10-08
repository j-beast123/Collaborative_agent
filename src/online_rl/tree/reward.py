"""Agent reward inputs: a response-level rubric component (R_A) and a final-outcome component. Rerun effectiveness
enters only as the weights of the rubric scores inside R_A, never as its own reward term."""
from __future__ import annotations
import re
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


def _spread(xs: list) -> bool:
    v = [x for x in xs if x is not None]
    return len(v) >= 2 and max(v) - min(v) > 1e-9


def sibling_group_diagnostics(tree: Tree, targets: dict, update: Optional[dict] = None) -> Optional[dict]:
    """Diagnostic only (never used for training): per same-state sibling group, the outcome component, R_A and final reward
    of each sibling and whether each varies within the group; summary counts overall and by depth. Zero-variance groups are
    reported, not hidden."""
    if not isinstance(targets, dict) or 'groups' not in targets or 'nodes' not in targets: return None   # a credit rule without sibling groups
    depth = {n.node_id: n.depth for n in tree.nodes}; groups = []
    for key, ids in sorted(targets['groups'].items()):
        t = [targets['nodes'][i] for i in ids]
        groups.append({'group': key, 'depth': depth[ids[0]], 'size': len(ids), 'outcome': [x['outcome_component'] for x in t], 'R_A': [x['R_A'] for x in t],
                       'reward': [x['reward'] for x in t], 'outcome_varies': _spread([x['outcome_component'] for x in t]),
                       'R_A_varies': _spread([x['R_A'] for x in t]), 'R_A_present': sum(x['R_A'] is not None for x in t),
                       'R_A_nonzero': any(x['R_A'] for x in t),
                       'reward_varies': _spread([x['reward'] for x in t])})
    def summary(gs):
        n = len(gs)
        return {'groups': n, 'outcome_varies': sum(g['outcome_varies'] for g in gs), 'R_A_varies': sum(g['R_A_varies'] for g in gs),
                'R_A_nonzero': sum(g['R_A_nonzero'] for g in gs),
                'reward_varies': sum(g['reward_varies'] for g in gs), 'reward_varies_fraction': (sum(g['reward_varies'] for g in gs) / n) if n else None,
                'reward_varies_only_through_R_A': sum(g['reward_varies'] and not g['outcome_varies'] for g in gs)}
    return {'summary': {**summary(groups), 'groups_updated': (update or {}).get('groups_used'), 'optimizer_steps': (update or {}).get('optimizer_steps', 0)},
            'by_depth': {d: summary([g for g in groups if g['depth'] == d]) for d in sorted({g['depth'] for g in groups})}, 'groups': groups}


_AGREE = re.compile(r"\b(i agree|i concur|you're right|you are right|you're correct|you are correct|exactly|absolutely right)\b", re.I)
_CHALLENGE = re.compile(r"\b(however|but i|i'd push back|i would push back|i'm not sure|i am not sure|i disagree|less likely|rather than|on the other hand)\b", re.I)


def interaction_diagnostics(tree: Tree, targets: Optional[dict]) -> dict:
    """Diagnostic only (never used for training): Agent reply behavior (heuristic agreement / challenge / alternative /
    evidence-request flags), sibling option diversity, Doctor option changes (hidden status, protocol v2), turn and stop
    depths, leaves and reward statistics of one case tree."""
    from src.collaboration.generator import option
    nodes = list(tree.nodes); kids: dict = {}
    for n in nodes: kids.setdefault(n.parent_id or 'root', []).append(n)
    def doctor_option(n):
        st = getattr(n, 'prefix_doctor_status', None) or {}
        return st.get('current_option') if st.get('current_option') not in (None, 'none') else (option(n.prefix[-1][3]) or None)
    flags = {'agreement': 0, 'challenge': 0, 'names_other_option_than_doctor': 0, 'question_or_evidence_request': 0}
    for n in nodes:
        r = n.response or ''; a = option(r)
        flags['agreement'] += bool(_AGREE.search(r[:200])); flags['challenge'] += bool(_CHALLENGE.search(r))
        flags['names_other_option_than_doctor'] += bool(a and doctor_option(n) and a != doctor_option(n)); flags['question_or_evidence_request'] += '?' in r
    diversity = [len({option(n.response) for n in g if option(n.response)}) for g in kids.values()]
    changes = sum(1 for n in nodes if getattr(n, 'next_doctor_status', None) and (n.next_doctor_status.get('status') or {}).get('current_option') not in (None, 'none')
                  and doctor_option(n) and n.next_doctor_status['status']['current_option'] != doctor_option(n))
    rewards = [v['reward'] for v in (targets or {}).get('nodes', {}).values() if v.get('reward') is not None]
    depth = [len(l.path) for l in tree.leaves]
    return {'agent_replies': len(nodes), 'agent_reply_flags': flags,
            'sibling_option_diversity': {'mean_distinct_options': (sum(diversity) / len(diversity)) if diversity else None,
                                         'groups_with_more_than_one_option': sum(d > 1 for d in diversity), 'groups': len(diversity)},
            'doctor_option_changes': changes, 'leaves': len(tree.leaves), 'mean_agent_turns': (sum(depth) / len(depth)) if depth else None,
            'stop_depths': {d: depth.count(d) for d in sorted(set(depth))},
            'stop_reasons': {k: sum(l.stop_reason == k for l in tree.leaves) for k in sorted({l.stop_reason for l in tree.leaves})},
            'leaf_accuracy': (sum(l.outcome == 1.0 for l in tree.leaves) / len(tree.leaves)) if tree.leaves else None,
            'reward': {'n': len(rewards), 'mean': (sum(rewards) / len(rewards)) if rewards else None,
                       'min': min(rewards) if rewards else None, 'max': max(rewards) if rewards else None}}

