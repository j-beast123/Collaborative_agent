from src.online_rl.sherpa.prompt_feedback import discrimination_feedback


def test_feedback_reports_wrong_pair_and_item_discrimination_without_final():
    dialogue = [{'speaker': 'agent', 'text': 'Check the assumption.', 'stage': 'A1'},
                {'speaker': 'doctor', 'text': 'Final answer: A', 'stage': 'FINAL'}]
    trajectories = [{'trajectory_id': key, 'outcome': outcome, 'dialogue': dialogue}
                    for key, outcome in [('good', 1), ('bad', 0), ('good2', 1)]]
    assessment = {'valid': True, 'generation': {'rubrics': [{'name': 'assumption'}, {'name': 'objection'}]},
                  'scores': [{'trajectory_id': 'good', 'reward': 0, 'scores': [0, 0]},
                             {'trajectory_id': 'bad', 'reward': 1, 'scores': [1, 1]},
                             {'trajectory_id': 'good2', 'reward': 0.5, 'scores': [1, 0]}]}
    feedback = discrimination_feedback(trajectories, assessment)
    pair = feedback['misranked_or_tied_pairs'][0]
    assert pair['success']['trajectory_id'] == 'good'
    assert pair['failure']['trajectory_id'] == 'bad'
    assert pair['failure_minus_success_reward'] == 1
    assert len(pair['success']['dialogue']) == 1
    rates = feedback['rubric_outcome_rates']
    assert rates[0]['success_satisfaction_rate'] == 0.5
    assert rates[0]['failure_satisfaction_rate'] == 1
    assert rates[1]['success_satisfaction_rate'] == 0


def test_ties_are_feedback_and_invalid_generation_has_no_invented_pairs():
    trajectories = [{'trajectory_id': key, 'outcome': outcome, 'dialogue': []}
                    for key, outcome in [('good', 1), ('bad', 0)]]
    result = {'valid': True, 'generation': {'rubrics': [{'name': 'item'}]},
              'scores': [{'trajectory_id': key, 'reward': 1, 'scores': [1]} for key in ['good', 'bad']]}
    assert discrimination_feedback(trajectories, result)['misranked_or_tied_pairs'][0]['failure_minus_success_reward'] == 0
    invalid = discrimination_feedback(trajectories, {'valid': False, 'generation': {'rubrics': []}, 'scores': []})
    assert invalid == {'misranked_or_tied_pairs': [], 'rubric_outcome_rates': [], 'generation_valid': False}
