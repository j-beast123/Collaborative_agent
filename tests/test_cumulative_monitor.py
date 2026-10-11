import json

from scripts.online_rl import watch_five_gpu as watch


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def prepared_record():
    return {'tree': {'nodes': [{'node_id': 'n'}]}, 'cumulative_reuse': [{'leaf_id': 'L1'}, {'leaf_id': 'L2'}],
            'rubric_bank': {'verification_pending': True, 'witness': [], 'telemetry': {'created_new': 2}},
            'timings': {'rollout_wall_seconds': 12}}


def test_bank_monitor_tracks_prepare_verify_score_and_final_outcomes(tmp_path, monkeypatch):
    monkeypatch.setattr(watch, 'gpu_sample', lambda: ({}, None))
    cfg = {'five_gpu': {'rollout_gpus': [0, 1, 2, 3, 4], 'execution_mode': 'shared_ready_tasks_v1'},
           'rubric': {'cumulative_bank': {'protocol': 'cumulative_keep_bank_v1'}},
           'rollout': {'branching': 1, 'rounds': 1}, 'schedule': {'cases': 120}}
    batch = tmp_path / 'batches/batch_000008'
    write(batch / 'manifest.json', {'assignments': [{'worker': i, 'index': 35+i} for i in range(5)],
                                    'case_ids': [f'c{i}' for i in range(5)]})
    write(tmp_path / 'run_state.json', {'case_cursor': 35, 'batch_cursor': 7,
                                       'rubric_bank_sha256': 'old', 'rubric_bank_version': 1})
    write(tmp_path / 'rubric_bank/snapshots/old.json', {'entries': [{}, {}, {}], 'version': 1})
    write(batch / 'bank_input.json', {'rubric_bank_sha256': 'old'})
    for i in range(5):
        directory = batch / f'case_{35+i:06d}'
        write(directory / 'bank_prepare.json', prepared_record())
    monitor = watch.Monitor(tmp_path, cfg)
    sample = monitor.snapshot()
    assert sample['pipeline']['bank_phase_complete'] == {'prepare': 5, 'verify': 0, 'score': 0}
    assert sample['pipeline']['phase_complete'] == {'P0': 0, 'P1': 0, 'union': 0}
    assert sample['cumulative_bank']['committed_size'] == 3
    assert sample['cumulative_bank']['frozen_size'] is None
    assert sample['cumulative_bank']['new_keep'] is None
    assert sample['cumulative_bank']['reused_trajectories'] == 10
    assert all(row['phase'] == 'bank rerun verification / wait' for row in sample['gpus'])
    for i in range(5):
        verified = prepared_record()
        verified['rubric_bank'].update(verification_pending=False, witness=[{'decision': 'KEEP'}])
        write(batch / f'case_{35+i:06d}/bank_verify.json', verified)
    write(batch / 'bank_frozen.json', {'entries': [{}, {}, {}, {}], 'version': 2})
    sample = monitor.snapshot()
    assert sample['pipeline']['bank_phase_complete'] == {'prepare': 5, 'verify': 5, 'score': 0}
    assert sample['cumulative_bank']['new_keep'] == 5
    assert sample['cumulative_bank']['added_unique_rubrics'] == 1
    assert all(row['phase'] == 'bank scoring / wait' for row in sample['gpus'])
    scored = prepared_record()
    scored.update(response_scores=[{'response_node_id': 'n', 'rubric_id': 'R', 'match': False,
                                    'scoring_status': 'not_retrieved', 'score': None}],
                  agent_targets={'reward_weights': {'w_outcome': .5}, 'groups': {'g': ['n']},
                                 'nodes': {'n': {'reward': .5, 'outcome_component': 1.0}}})
    write(batch / 'case_000035/bank_score.json', scored)
    sample = monitor.snapshot()
    assert sample['pipeline']['bank_phase_complete']['score'] == 1
    assert sample['pipeline']['score_distribution']['union']['counts']['NR'] == 1
    assert sample['pipeline']['score_distribution']['union']['counts']['NA'] == 0
    assert sample['pipeline']['outcome_signal']['targets'] == 1
    assert sample['pipeline']['outcome_signal']['mean_contribution'] == .5
    assert sample['gpus'][0]['phase'] == 'waiting at batch barrier'
    for render in (watch.render_compact_dashboard, watch.render_dashboard, watch.render):
        text = render(sample)
        assert 'prepare 5/5' in text and 'verify 5/5' in text and 'score 1/5' in text
        assert 'Bank stored 3 (v1)' in text and 'scoring 4 (top8/state, v2)' in text
        assert 'new +1 unique' in text and 'NA stays unscored' in text
        assert 'new KEEP 5' in text and 'reused 10 trajectories' in text
        assert 'P1' not in text
    compact = watch.render_compact_dashboard(sample)
    assert all(f'GPU{i}' in compact for i in range(5))
    assert 'NR:1' in compact


def test_full_bank_monitor_uses_frozen_size_and_unique_growth_after_commit(tmp_path, monkeypatch):
    monkeypatch.setattr(watch, 'gpu_sample', lambda: ({}, None))
    cfg = {'five_gpu': {'rollout_gpus': list(range(5)), 'execution_mode': 'shared_ready_tasks_v1'},
           'rubric': {'cumulative_bank': {'protocol': 'cumulative_keep_bank_v1', 'scoring_scope': 'all',
                                         'retrieval_top_k': 8}},
           'rollout': {'branching': 1, 'rounds': 1}, 'schedule': {'cases': 120}}
    batch = tmp_path / 'batches/batch_000002'
    write(batch / 'manifest.json', {'assignments': [], 'case_ids': []})
    write(tmp_path / 'run_state.json', {'case_cursor': 10, 'batch_cursor': 2,
                                       'rubric_bank_sha256': 'after', 'rubric_bank_version': 2})
    write(tmp_path / 'rubric_bank/snapshots/before.json', {'entries': [{}] * 8, 'version': 1})
    write(tmp_path / 'rubric_bank/snapshots/after.json', {'entries': [{}] * 9, 'version': 2})
    write(batch / 'bank_input.json', {'rubric_bank_sha256': 'before', 'scoring_scope': 'all'})
    write(batch / 'bank_frozen.json', {'entries': [{}] * 9, 'version': 2})
    write(batch / 'bank_metrics.json', {'added_unique_rubrics': 1, 'reused_trajectories': 17,
                                       'scoring_scope': 'all'})
    sample = watch.Monitor(tmp_path, cfg).snapshot()
    bank = sample['cumulative_bank']
    assert bank['committed_size'] == bank['frozen_size'] == 9
    assert bank['added_unique_rubrics'] == 1  # Compare to batch input, not newly committed bank.
    assert bank['reused_trajectories'] == 17
    assert bank['scoring_scope'] == 'all' and bank['scoring_top_k'] is None
    for render in (watch.render_compact_dashboard, watch.render_dashboard, watch.render):
        text = render(sample)
        assert 'Bank stored 9 (v2) | scoring 9 (all, v2) | new +1 unique | reused 17 trajectories' in text
        assert 'NA stays unscored' in text
    text = watch.render_compact_dashboard(sample)
    assert all(f'GPU{i}' in text for i in range(5))


def test_new_all_scope_config_does_not_relabel_existing_legacy_batch(tmp_path):
    write(tmp_path / 'batch/bank_input.json', {'rubric_bank_sha256': 'old'})
    cfg = {'rubric': {'cumulative_bank': {'scoring_scope': 'all', 'retrieval_top_k': 8}}}
    bank = watch.cumulative_bank_status(tmp_path, {}, tmp_path / 'batch', {}, cfg)
    assert bank['scoring_scope'] == 'top_k' and bank['scoring_top_k'] == 8
    assert '(top8/state)' in watch.render_bank_status({'cumulative_bank': bank})


def test_config_before_manifest_resolves_extends_and_shows_empty_start(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(watch, 'gpu_sample', lambda: ({}, None))
    base = tmp_path / 'base.json'
    config = tmp_path / 'fresh.json'
    run = tmp_path / 'unstarted_run'
    write(base, {'five_gpu': {'rollout_gpus': list(range(5)), 'execution_mode': 'shared_ready_tasks_v1'},
                 'rubric': {'cumulative_bank': {'protocol': 'cumulative_keep_bank_v1', 'retrieval_top_k': 8}},
                 'rollout': {'branching': 4, 'rounds': 2}, 'schedule': {'cases': 120}})
    write(config, {'extends': str(base), 'output_dir': str(run),
                   'rubric': {'cumulative_bank': {'scoring_scope': 'all'}}})
    monkeypatch.setattr(watch.sys, 'argv', ['watch_five_gpu.py', '--config', str(config), '--once', '--compact'])
    watch.main()
    text = capsys.readouterr().out
    assert 'no batch' in text
    assert 'Bank stored 0 (not started) | scoring pending (all) | new pending unique | reused 0 trajectories' in text
    assert all(f'GPU{i}' in text for i in range(5))
    assert 'v0' not in text and 'scoring 0' not in text
    assert not run.exists()  # Monitoring cannot initialize the bank or the run.


def test_missing_committed_snapshot_never_becomes_empty_bank(tmp_path):
    cfg = {'rubric': {'cumulative_bank': {'scoring_scope': 'all'}}}
    bank = watch.cumulative_bank_status(tmp_path, {'rubric_bank_sha256': 'missing', 'rubric_bank_version': 2},
                                        None, {}, cfg)
    assert bank['committed_size'] is None and not bank['empty_start_pending']
    assert 'Bank stored - (v2)' in watch.render_bank_status({'cumulative_bank': bank})
    cfg['rubric']['cumulative_bank']['initial_snapshot_file'] = 'seed.json'
    seeded = watch.cumulative_bank_status(tmp_path, {}, None, {}, cfg)
    assert seeded['committed_size'] is None and not seeded['empty_start_pending']


def test_idle_eval_monitor_uses_explicit_priority_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(watch, 'pid_alive', lambda pid: pid == 123)
    eval_dir = tmp_path / 'eval'
    other = tmp_path / 'shared_priority'
    write(tmp_path / 'idle_eval_runtime.json', {'output_dir': str(eval_dir), 'priority_dir': str(other), 'checkpoint_cases': 20})
    write(eval_dir / 'status.json', {'pid': 123, 'stage': 'running'})
    write(eval_dir / 'evaluation/summary.json', {'evaluated': 39, 'accuracy': .7})
    write(other / 'priority.json', {'gpus': {'4': {'calls': {'a': {'owner': {'pid': 123}}},
                                                 'residents': {'b': {'owner': {'pid': 123}}}}}})
    status = watch.idle_evaluation_status(tmp_path)
    assert status['active_gpus'] == ['4'] and status['resident_gpus'] == ['4']
    assert status['controller_alive'] and status['evaluated'] == 39
    # Older runtime pointers retain their original default directory behavior.
    write(tmp_path / 'idle_eval_runtime.json', {'output_dir': str(eval_dir), 'checkpoint_cases': 20})
    write(tmp_path / 'idle_eval_priority/priority.json', {'gpus': {'0': {'calls': {'a': {'owner': {'pid': 123}}}}}})
    assert watch.idle_evaluation_status(tmp_path)['active_gpus'] == ['0']


def test_idle_eval_monitor_follows_continuation_link(tmp_path, monkeypatch):
    monkeypatch.setattr(watch, 'pid_alive', lambda pid: pid in (123, 456))
    run = tmp_path / 'training'
    evaluation = tmp_path / 'base_c0'
    write(run / 'idle_eval_link.json', {'output': str(evaluation), 'checkpoint_cases': 0})
    write(evaluation / 'status.json', {
        'pid': 123, 'stage': 'running', 'evaluated': 102,
        'worker_pids': {'2': 456}})
    write(evaluation / 'evaluation/summary.json', {
        'dataset_cases': 622, 'evaluated': 101, 'correct': 74,
        'accuracy': 74 / 101, 'status': 'pending'})
    status = watch.idle_evaluation_status(run)
    assert status['checkpoint_cases'] == 0
    assert status['evaluated'] == 102
    assert status['active_gpus'] == ['2']
    assert status['controller_alive']
    assert 'Idle eval C0 · 102/622' in watch.render_idle_evaluation(status)


def test_fresh_training_monitor_keeps_historical_c20_progress_separate_and_shows_wait(tmp_path, monkeypatch):
    monkeypatch.setattr(watch, 'gpu_sample', lambda: ({}, None))
    monkeypatch.setattr(watch, 'pid_alive', lambda pid: pid == 123)
    run = tmp_path / 'fresh_training'
    evaluation = tmp_path / 'historical_c20'
    cfg = {'five_gpu': {'rollout_gpus': list(range(5)), 'execution_mode': 'shared_ready_tasks_v1'},
           'rubric': {'cumulative_bank': {'protocol': 'cumulative_keep_bank_v1', 'scoring_scope': 'all'}},
           'rollout': {'branching': 4, 'rounds': 2}, 'schedule': {'cases': 120}}
    write(run / 'run_state.json', {'case_cursor': 0, 'batch_cursor': 0,
                                   'rubric_bank_sha256': 'empty', 'rubric_bank_version': 0})
    write(run / 'rubric_bank/snapshots/empty.json', {'version': 0, 'entries': []})
    write(run / 'batches/batch_000001/manifest.json', {
        'assignments': [{'worker': i, 'index': i} for i in range(5)],
        'case_ids': [f'new_case_{i}' for i in range(5)]})
    write(run / 'idle_eval_runtime.json', {'output_dir': str(evaluation), 'checkpoint_cases': 20,
                                          'priority_dir': 'new_training_priority'})
    write(evaluation / 'evaluation/summary.json', {'dataset_cases': 622, 'evaluated': 439,
                                                   'correct': 305, 'accuracy': 305 / 439,
                                                   'status': 'pending'})
    write(evaluation / 'status.json', {'pid': 123, 'stage': 'waiting_training_priority_hooks'})
    # A different/default priority file must not leak its activity into the
    # explicitly linked evaluation of a historical checkpoint.
    write(run / 'idle_eval_priority/priority.json', {
        'gpus': {'4': {'calls': {'old': {'owner': {'pid': 123}}}}}})
    write(run / 'new_training_priority/priority.json', {'gpus': {}})
    monitor = watch.Monitor(run, cfg)
    snapshot = monitor.snapshot()
    text = watch.render_compact_dashboard(snapshot)
    assert 'Cases 0/120' in text
    assert ('Idle eval C20 · 439/622 · accuracy 69.5% · '
            'waiting_training_priority_hooks · active GPUs -') in text
    assert '439/120' not in text and 'Cases 439/' not in text
    assert all(f'GPU{i}' in text for i in range(5))
    assert snapshot['idle_evaluation']['controller_alive']
    assert snapshot['idle_evaluation']['active_gpus'] == []

    write(evaluation / 'status.json', {'pid': 123, 'stage': 'running'})
    write(run / 'new_training_priority/priority.json', {
        'gpus': {str(gpu): {'calls': {'call': {'owner': {'pid': 123}}}} for gpu in (0, 2)}})
    running = watch.render_compact_dashboard(monitor.snapshot())
    assert 'Cases 0/120' in running
    assert 'Idle eval C20 · 439/622 · accuracy 69.5% · running · active GPUs 0,2' in running
    write(evaluation / 'status.json', {'pid': 999, 'stage': 'running'})
    stopped = watch.render_compact_dashboard(monitor.snapshot())
    assert 'Idle eval C20 · 439/622 · accuracy 69.5% · stopped' in stopped


def test_early_reflection_does_not_advance_bank_generation_before_tree_complete(tmp_path, monkeypatch):
    monkeypatch.setattr(watch, 'gpu_sample', lambda: ({}, None))
    batch = tmp_path / 'batches/batch_000001'
    case = batch / 'case_000000'
    write(batch / 'manifest.json', {'assignments': [{'worker': 0, 'index': 0}], 'case_ids': ['c1']})
    case.mkdir()
    cfg = {'five_gpu': {'rollout_gpus': [0]}, 'rubric': {'cumulative_bank': {'protocol': 'cumulative_keep_bank_v1'}},
           'rollout': {'branching': 2, 'rounds': 1}, 'schedule': {'cases': 120}}
    cache = case / 'calls.jsonl'
    def append(kind, identity=None):
        with cache.open('a') as stream:
            stream.write(json.dumps({'kind': kind, 'identity': identity or {}}) + '\n')
    append('agent', {'path': '0'})
    append('agent', {'path': '1'})
    append('final', {'path': '0'})
    append('feedback', {'leaf': 'c1/0'})
    monitor = watch.Monitor(tmp_path, cfg)
    row = monitor.snapshot()['gpus'][0]
    assert row['leaves']['remaining_min'] == 1
    assert row['phase'] == 'bank rollout / Doctor reflection'
    append('final', {'path': '1'})
    row = monitor.snapshot()['gpus'][0]
    assert row['leaves']['tree_complete']
    assert row['phase'] == 'bank Doctor reflection / wait'
    append('cumulative_bank_match')
    assert monitor.snapshot()['gpus'][0]['phase'] == 'bank reuse matching'
    append('rubric_generation')
    assert monitor.snapshot()['gpus'][0]['phase'] == 'bank candidate generation / reuse'
    # Saved protocol boundaries always outrank inferred completed-cache stages.
    write(case / 'bank_prepare.json', prepared_record())
    assert monitor.snapshot()['gpus'][0]['phase'] == 'bank rerun verification / wait'
