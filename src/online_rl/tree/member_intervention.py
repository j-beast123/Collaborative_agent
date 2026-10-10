"""Frozen per-trajectory instructions for a shared canonical rubric.

Matching a behavioral criterion does not establish that another trajectory has
the same intervention location or supports the creator's context-specific guide.
"""
from __future__ import annotations

from src.online_rl.integrated.records import sha256_json
from src.online_rl.tree.records import to_dialogue

VERSION = 'trajectory_member_instruction_v1'
MEMBER_INTERVENTION_VERSION = VERSION
FIELDS = {'version', 'leaf_id', 'case_id', 'canonical_rubric_id', 'rubric_sha256',
          'selected_agent_turn_id', 'intervention_instruction', 'requested_level',
          'prefix_sha256', 'origin'}


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
          intervention_instruction, origin):
    if not getattr(rub, 'intervention_instruction', None):
        raise ValueError('A member instruction requires a joint canonical rubric')
    if not isinstance(intervention_instruction, str) or not intervention_instruction.strip():
        raise ValueError('Member intervention_instruction must be a nonblank string')
    if origin not in ('generated', 'matched'):
        raise ValueError('Unknown member instruction origin')
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
    return {'version': VERSION, 'leaf_id': leaf_id, 'case_id': case_id,
            'canonical_rubric_id': rub.rubric_id, 'rubric_sha256': sha256_json(rub.rubric),
            'selected_agent_turn_id': selected_agent_turn_id,
            'intervention_instruction': intervention_instruction, 'requested_level': _level(outcome),
            'prefix_sha256': sha256_json(prefix), 'origin': origin}


def make_member_intervention(rub, tin, *, selected_agent_turn_id, intervention_instruction, origin):
    return _make(rub, leaf_id=tin.leaf_id, case_id=tin.case_id, dialogue=tin.dialogue,
                 outcome=tin.outcome, selected_agent_turn_id=selected_agent_turn_id,
                 intervention_instruction=intervention_instruction, origin=origin)


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
    if not isinstance(member_spec, dict) or set(member_spec) != FIELDS:
        raise ValueError('Joint verification requires a complete member intervention_spec; legacy assignments cannot be reused')
    expected = _make(rub, leaf_id=leaf.leaf_id, case_id=leaf.case_id, dialogue=leaf.dialogue,
                     outcome=leaf.outcome, selected_agent_turn_id=member_spec['selected_agent_turn_id'],
                     intervention_instruction=member_spec['intervention_instruction'], origin=member_spec['origin'])
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
