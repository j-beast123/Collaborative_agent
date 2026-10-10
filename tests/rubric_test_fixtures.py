"""Synthetic rubric examples shared by the active training regression tests."""
from src.online_rl.tree.records import TrajectoryRubricInput


def _tin(outcome='success'):
    return TrajectoryRubricInput('case-7/1.2', 'case-7', '', 'question', (),
        (('doctor', 'DISCUSS', 1, 'Please compare these two considerations.'),
         ('agent', 'DISCUSS', 1, 'I will compare the stated considerations.'),
         ('doctor', 'FINAL', 2, 'A')),
        outcome, 'The reply addressed the comparison.', 'sha', 'behavior_observer_v5', 1)


def _output(tin):
    behavior = 'Connects the requested comparison to the stated decision criterion.'
    context = 'The Doctor requests a comparison before the Agent response.'
    state = 'No additional recorded Agent state is required.'
    scores = {str(i): {'user_side_context': context, 'agent_side_state': state,
                       'behavior': f'{behavior} level {i}'} for i in range(3)}
    # Keep distinct but make level 2 the high-level behavior used by the legacy trainer field.
    scores['2']['behavior'] = behavior
    scores['1']['behavior'] = 'Makes the comparison but does not connect it to the criterion.'
    scores['0']['behavior'] = 'Does not address the requested comparison.'
    return {'rubric_id': 'candidate-1', 'source_case_id': tin.case_id, 'source_trajectory_id': tin.leaf_id,
        'source_outcome': tin.outcome,
        'analysis': {'surface_request': 'compare', 'decision_shaping_contribution': None,
            'user_contribution': 'The Doctor requests a comparison.', 'agent_side_state': state,
            'surface_handling': 'The Agent compares.', 'reasoning_handling': None, 'agent_handling': 'The Agent makes the comparison.',
            'interaction_need': 'Address the requested comparison.', 'selected_behavior': behavior,
            'selection_rationale': 'This reply directly responds to the request.',
            'source_behavior_support': 'The reply makes the requested comparison.',
            'behavior_to_preserve_or_correct': behavior, 'source_score': 2 if tin.outcome == 'success' else 0,
            'score_0_basis': 'counterfactual_removal' if tin.outcome == 'success' else 'observed_problem',
            'score_construction_check': {'score_0_basis': 'Remove only the comparison.',
                'score_1_component': 'Makes a comparison.', 'score_1_remaining_gap': 'Does not connect it to the criterion.',
                'score_2_basis': 'The source connects it to the criterion.',
                'conclusion_independence': 'Either conclusion can be supported.'},
            'selected_agent_turn_id': 1,
            'feedback_check': {'claim': None, 'assessment': 'not_provided', 'dialogue_basis': None},
            'evidence_strength': 'strong', 'limitations': [],
            'evidence_spans': [{'turn_id': 1, 'speaker': 'doctor', 'quote': 'Please compare these two considerations.'},
                               {'turn_id': 1, 'speaker': 'agent', 'quote': 'I will compare the stated considerations.'}]},
        'candidate_rubric': {'name': 'Requested comparison handling',
            'applicability': {'user_side_context': context, 'agent_side_state': state}, 'scores': scores}}
