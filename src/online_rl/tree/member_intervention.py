"""Frozen per-trajectory instructions for a shared canonical rubric.

Matching a behavioral criterion does not establish that another trajectory has
the same intervention location or supports the creator's context-specific guide.
"""
from __future__ import annotations

from src.online_rl.integrated.records import sha256_json
from src.online_rl.tree.records import to_dialogue

VERSION = 'trajectory_member_instruction_v1'
MEMBER_INTERVENTION_VERSION = VERSION
EXACT_SOURCE_CASE_VERSION = 'trajectory_member_instruction_exact_source_case_v1'
FIELDS = {'version', 'leaf_id', 'case_id', 'canonical_rubric_id', 'rubric_sha256',
          'selected_agent_turn_id', 'intervention_instruction', 'requested_level',
          'prefix_sha256', 'origin'}
EXACT_PROOF_FIELDS = {'candidate_rubric_id', 'candidate_sha256', 'content_sha256',
                      'selected_agent_turn_id', 'intervention_instruction_sha256'}


def visible_prefix(prefix):
    """Canonical visible transcript representation, never hidden Doctor status."""
    return [{'speaker': row['speaker'], 'stage': row.get('stage') or '',
             'round': int(row.get('round') or 0), 'text': row['text']} for row in prefix]


def _prefix_at(dialogue, turn):
    if isinstance(turn, bool) or not isinstance(turn, int) or turn < 1:
        raise ValueError('Member selected_agent_turn_id must be a positive integer')
    transcript = to_dialogue(dialogue)
    positions = [i for i, row in enumerate(transcript) if row['speaker'] == 'agent']
    if turn > len(positions):
        raise ValueError('Member selected_agent_turn_id is absent from its trajectory')
    return visible_prefix(transcript[:positions[turn - 1]])


def _level(outcome):
    if outcome in ('success', 1.0): return '0'
    if outcome in ('failure', 0.0): return '2'
    raise ValueError('Member instruction requires a decided source outcome')


def _make(rub, *, leaf_id, case_id, dialogue, outcome, selected_agent_turn_id,
          intervention_instruction, origin, version=VERSION, generated_candidate_proof=None):
    if not getattr(rub, 'intervention_instruction', None):
        raise ValueError('A member instruction requires a joint canonical rubric')
    if not isinstance(intervention_instruction, str) or not intervention_instruction.strip():
        raise ValueError('Member intervention_instruction must be a nonblank string')
    if version == VERSION:
        allowed_origins = ('generated', 'matched')
    elif version == EXACT_SOURCE_CASE_VERSION:
        allowed_origins = ('generated', 'generated_exact_duplicate')
    else:
        raise ValueError('Unknown member instruction version')
    if origin not in allowed_origins:
        raise ValueError('Unknown member instruction origin')
    if origin == 'generated_exact_duplicate':
        from src.online_rl.tree.cumulative_bank import content_hash
        proof = generated_candidate_proof
        if not isinstance(proof, dict) or set(proof) != EXACT_PROOF_FIELDS:
            raise ValueError('Exact duplicate requires a complete generated candidate proof')
        if (proof['candidate_rubric_id'] == rub.rubric_id or
                not isinstance(proof['candidate_rubric_id'], str) or not proof['candidate_rubric_id'] or
                not isinstance(proof['candidate_sha256'], str) or len(proof['candidate_sha256']) != 64 or
                proof['content_sha256'] != content_hash(rub.rubric) or
                proof['selected_agent_turn_id'] != selected_agent_turn_id or
                proof['intervention_instruction_sha256'] != sha256_json(intervention_instruction)):
            raise ValueError('Exact duplicate proof disagrees with canonical criterion or member guide')
    elif generated_candidate_proof is not None:
        raise ValueError('Only exact generated duplicates may carry candidate proof')
    provenance = (rub.rubric or {}).get('source_provenance') or {}
    source_leaf = provenance.get('source_trajectory_id')
    if source_leaf is not None and source_leaf != rub.leaf_id:
        raise ValueError('Canonical rubric leaf_id disagrees with its immutable source trajectory provenance')
    if origin == 'generated' and source_leaf is not None and source_leaf != leaf_id:
        raise ValueError('Generated member instruction cannot be reassigned from its source trajectory')
    if provenance.get('source_case_id') is not None and provenance['source_case_id'] != case_id:
        raise ValueError('Member instruction cannot reuse a rubric from a different case')
    if provenance.get('source_outcome') is not None and _level(provenance['source_outcome']) != _level(outcome):
        raise ValueError('Member instruction cannot change its canonical success/failure direction')
    if origin == 'generated' and rub.leaf_id != leaf_id:
        raise ValueError('Generated member instruction must belong to the canonical source')
    if origin == 'generated':
        source_turn = (((rub.rubric or {}).get('source_provenance') or {}).get('analysis') or {}).get('selected_agent_turn_id')
        if selected_agent_turn_id != source_turn or intervention_instruction != rub.intervention_instruction:
            raise ValueError('Generated member instruction must preserve its jointly generated source guide and turn')
    prefix = _prefix_at(dialogue, selected_agent_turn_id)
    spec = {'version': version, 'leaf_id': leaf_id, 'case_id': case_id,
            'canonical_rubric_id': rub.rubric_id, 'rubric_sha256': sha256_json(rub.rubric),
            'selected_agent_turn_id': selected_agent_turn_id,
            'intervention_instruction': intervention_instruction, 'requested_level': _level(outcome),
            'prefix_sha256': sha256_json(prefix), 'origin': origin}
    if generated_candidate_proof is not None:
        spec['generated_candidate_proof'] = dict(generated_candidate_proof)
    return spec


def make_member_intervention(rub, tin, *, selected_agent_turn_id, intervention_instruction, origin,
                             version=VERSION, generated_candidate_proof=None):
    return _make(rub, leaf_id=tin.leaf_id, case_id=tin.case_id, dialogue=tin.dialogue,
                 outcome=tin.outcome, selected_agent_turn_id=selected_agent_turn_id,
                 intervention_instruction=intervention_instruction, origin=origin, version=version,
                 generated_candidate_proof=generated_candidate_proof)


def source_member_intervention(rub, leaf):
    """Standalone source verification only; never a fallback for reused members."""
    if rub.leaf_id != leaf.leaf_id:
        raise ValueError('Cannot reuse the canonical source instruction for another member')
    analysis = ((rub.rubric or {}).get('source_provenance') or {}).get('analysis') or {}
    return _make(rub, leaf_id=leaf.leaf_id, case_id=leaf.case_id, dialogue=leaf.dialogue,
                 outcome=leaf.outcome, selected_agent_turn_id=analysis.get('selected_agent_turn_id'),
                 intervention_instruction=rub.intervention_instruction, origin='generated')


def resolve_member_intervention(rub, leaf, member_spec=None, node=None):
    if not getattr(rub, 'intervention_instruction', None):
        if member_spec is not None: raise ValueError('Legacy rubric cannot carry a joint member instruction')
        return None
    if not isinstance(member_spec, dict) or set(member_spec) != (FIELDS | ({'generated_candidate_proof'} if member_spec.get('origin') == 'generated_exact_duplicate' else set())):
        raise ValueError('Joint verification requires a complete member intervention_spec; legacy assignments cannot be reused')
    expected = _make(rub, leaf_id=leaf.leaf_id, case_id=leaf.case_id, dialogue=leaf.dialogue,
                     outcome=leaf.outcome, selected_agent_turn_id=member_spec['selected_agent_turn_id'],
                     intervention_instruction=member_spec['intervention_instruction'], origin=member_spec['origin'],
                     version=member_spec['version'], generated_candidate_proof=member_spec.get('generated_candidate_proof'))
    if member_spec != expected:
        raise ValueError('Member intervention identity differs from its trajectory, rubric, outcome, or prefix')
    turn = member_spec['selected_agent_turn_id']
    if turn > len(leaf.path):
        raise ValueError('Member selected turn is absent from its Agent node path')
    if node is not None:
        if node.case_id != leaf.case_id or node.node_id != leaf.path[turn - 1] or node.depth != turn:
            raise ValueError('Member intervention target node does not match its selected turn')
        if sha256_json(visible_prefix(to_dialogue(node.prefix))) != member_spec['prefix_sha256']:
            raise ValueError('Member intervention target prefix differs from its frozen transcript')
    return dict(member_spec)
