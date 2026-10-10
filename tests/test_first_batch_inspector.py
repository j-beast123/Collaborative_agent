import json

from scripts.online_rl.inspect_first_batch import inspect
from src.online_rl.integrated.records import sha256_json
from src.online_rl.tree.cumulative_bank import empty_snapshot


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_empty_keep_batch_is_reviewable_without_inventing_scores(tmp_path):
    initial = empty_snapshot()
    base = {'protocol': initial['protocol'], 'version': 1,
            'parent_sha256': initial['sha256'], 'entries': []}
    bank = {**base, 'sha256': sha256_json(base)}
    write(tmp_path / 'rubric_bank' / 'snapshots' / f"{bank['sha256']}.json", bank)
    write(tmp_path / 'run_state.json', {'case_cursor': 5, 'batch_cursor': 1,
                                      'rubric_bank_sha256': bank['sha256'], 'checkpoint': 'checkpoint'})
    write(tmp_path / 'run_manifest.json', {'protocol_sha256': 'protocol', 'config': {
        'policy': {'init_adapter': None},
        'rubric': {'assignment_protocol': 'exact_generated_source_case_v1',
                   'cumulative_bank': {'scoring_scope': 'all'}}}})
    for i in range(5):
        write(tmp_path / 'cases' / f'case_{i:06d}.json', {
            'case_cursor': i, 'case_id': f'case{i}', 'policy_sha256': 'policy',
            'prompt_sha256': 'prompt', 'counts': {},
            'tree': {'nodes': [], 'leaves': []},
            'rubric_bank': {'assignments': [], 'witness': [], 'telemetry': {}},
            'selection': [], 'trajectory_rubrics': [], 'response_scores': [],
            'cumulative_bank_input': initial, 'cumulative_reuse': [],
            'cumulative_bank_scoring': {'snapshot_sha256': bank['sha256'], 'scoring_scope': 'all'},
            'group_diagnostics': {}, 'agent_targets': {}, 'batch_update': {}})
    report = inspect(tmp_path)
    assert report['frozen_bank_unique_rubrics'] == 0
    assert len(report['cases']) == 5
    assert all(c['score_distributions'] == {} for c in report['cases'])
