"""Observable scoring mistakes for improving case-specific rubric generation."""


def discrimination_feedback(trajectories, assessment, *, max_pairs=2):
    scores = {row['trajectory_id']: row for row in assessment.get('scores', [])}
    rubrics = assessment['generation']['rubrics']
    groups = {outcome: [row for row in trajectories
                       if row['outcome'] == outcome and row['trajectory_id'] in scores]
              for outcome in (0, 1)}
    def view(row):
        score = scores[row['trajectory_id']]
        return {'trajectory_id': row['trajectory_id'], 'outcome': row['outcome'],
                'reward': score['reward'], 'rubric_scores': score['scores'],
                'dialogue': [{key: turn.get(key) for key in ('speaker', 'round', 'text')}
                             for turn in row['dialogue'] if turn.get('stage') != 'FINAL']}
    pairs = [(scores[failed['trajectory_id']]['reward'] - scores[success['trajectory_id']]['reward'],
              success, failed)
             for success in groups[1] for failed in groups[0]
             if scores[success['trajectory_id']]['reward'] <= scores[failed['trajectory_id']]['reward']]
    pairs.sort(key=lambda pair: (-pair[0], pair[1]['trajectory_id'], pair[2]['trajectory_id']))
    selected, seen = [], set()
    for gap, success, failed in pairs:
        ids = {success['trajectory_id'], failed['trajectory_id']}
        if seen & ids: continue
        selected.append({'failure_minus_success_reward': gap,
                         'success': view(success), 'failure': view(failed)})
        seen.update(ids)
        if len(selected) == max_pairs: break
    rates = []
    for index, rubric in enumerate(rubrics):
        row = {'rubric_index': index, 'name': rubric['name']}
        for outcome, label in ((1, 'success'), (0, 'failure')):
            values = [scores[t['trajectory_id']]['scores'][index] for t in groups[outcome]]
            row[label + '_satisfaction_rate'] = sum(values) / len(values) if values else None
        rates.append(row)
    return {'misranked_or_tied_pairs': selected, 'rubric_outcome_rates': rates,
            'generation_valid': assessment['valid']}
