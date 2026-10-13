"""Case-local consolidation selects existing immutable rubric text only.

No outcome reward or retention is inferred by consolidation. Verification budget
and aggregation must be configured separately before replacing source evaluation.
"""
from pathlib import Path
from src.online_rl.integrated.records import sha256_json

PROMPT = Path(__file__).resolve().parents[3] / 'prompts/sherpa/rubric_consolidator_v1.txt'
PROTOCOL = 'case_local_existing_representative_v1'


def consolidate(case_id, candidates, generate, cache, *, model_identity=None):
    """generate(system, payload) -> parsed JSON. No source feedback/GT/outcome input.

    Reject invalid/foreign/duplicate candidates before a model call. Canonical
    selected turns remain on the originals; grouping never invents an intervention
    target or transfers the representative's source outcome to its members.
    """
    from src.online_rl.tree.posthoc import rubric_problems
    by_id = {}
    for rubric in candidates:
        if not rubric.valid or not isinstance(rubric.rubric, dict):
            raise ValueError('Consolidation requires valid candidate rubrics')
        if rubric.rubric_id in by_id:
            raise ValueError('Duplicate consolidation candidate ID')
        if rubric.rubric.get('source_case_id', case_id) != case_id:
            raise ValueError('Cannot consolidate across cases')
        if rubric_problems(rubric.rubric):
            raise ValueError('Invalid consolidation candidate schema')
        by_id[rubric.rubric_id] = rubric
    if not by_id:
        return {'protocol': PROTOCOL, 'case_id': case_id, 'groups': [], 'representatives': [], 'originals': []}
    visible = [{'rubric_id': rid, 'candidate_rubric': r.rubric.get('candidate_rubric', r.rubric)}
               for rid, r in sorted(by_id.items())]
    system = PROMPT.read_text()
    identity = {'protocol': PROTOCOL, 'case_id': case_id, 'prompt': sha256_json(system),
                'candidates': sha256_json(visible), 'model': model_identity}
    if len(visible) == 1:
        raw = {'groups': [{'representative_id': visible[0]['rubric_id'],
                          'member_ids': [visible[0]['rubric_id']], 'reason': 'Only one candidate'}]}
    else:
        raw = cache.call('rubric_consolidation', identity, lambda: generate(system, {'case_id': case_id, 'candidates': visible}))
    if not isinstance(raw, dict) or not isinstance(raw.get('groups'), list):
        raise ValueError('Consolidation must return a groups array')
    seen, groups = set(), []
    for group in raw['groups']:
        if not isinstance(group, dict):
            raise ValueError('Invalid consolidation group')
        ids, representative = group.get('member_ids'), group.get('representative_id')
        if not isinstance(ids, list) or not ids or any(not isinstance(i, str) for i in ids):
            raise ValueError('Group must contain existing member IDs')
        if len(set(ids)) != len(ids) or seen.intersection(ids) or any(i not in by_id for i in ids):
            raise ValueError('Unknown, repeated, or overlapping consolidation members')
        if representative not in ids or not isinstance(group.get('reason'), str) or not group['reason'].strip():
            raise ValueError('Representative must be a group member with a reason')
        seen.update(ids)
        groups.append({'representative_id': representative, 'member_ids': list(ids), 'reason': group['reason'],
                       'source_targets': [{'rubric_id': i, 'leaf_id': by_id[i].leaf_id,
                                           'selected_agent_turn_id': by_id[i].selected_agent_turn_id} for i in ids]})
    if seen != set(by_id):
        raise ValueError('Consolidation omitted candidates')
    return {'protocol': PROTOCOL, 'case_id': case_id, 'groups': groups,
            'representatives': [by_id[g['representative_id']] for g in groups],
            'originals': list(by_id.values())}


def serializable_consolidation(result):
    from src.online_rl.integrated.records import to_dict
    return {**result, 'representatives': [to_dict(r) for r in result['representatives']],
            'originals': [to_dict(r) for r in result['originals']]}
