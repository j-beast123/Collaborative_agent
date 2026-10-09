"""Static checks for the 300-case production configs, runtime profile and launcher (no GPU, no API, no model)."""
import json
import subprocess
from pathlib import Path
import pytest
from src.online_rl.tree import experiment as X

ROOT = X.ROOT
PROD = 'configs/experiments/reward_prompt_matrix/production'
ARMS = {'r03_o07': (0.3, 0.7), 'r07_o03': (0.7, 0.3)}            # (w_rubric, w_outcome)


def cfg(arm): return X.load_config(f'{PROD}/{arm}_evolving_300.json')


def profile(): return json.loads((ROOT / PROD / 'runtime_profile.json').read_text())


@pytest.mark.parametrize('arm', list(ARMS))
def test_production_arm_protocol_and_reward_orientation(arm):
    c = cfg(arm); w_r, w_o = ARMS[arm]
    assert (c['agent_reward']['w_rubric'], c['agent_reward']['w_outcome']) == (w_r, w_o)
    assert c['schedule']['cases'] == 300 and c['five_gpu']['batch_cases'] == 5
    assert (c['rollout']['branching'], c['rollout']['rounds']) == (4, 4)
    assert c['experiment']['prompt_mode'] == 'evolving' and c['prompt_update']['enabled'] and c['rubric']['enabled']
    assert c['feedback']['model'] == 'anthropic/claude-4.6-sonnet-20260217'
    assert c['rubric']['prompt_sha256'].startswith('f8d6e394') and c['schedule']['seed'] == 0
    assert c['output_dir'] == f"outputs/reward_prompt_matrix_300/{arm}__evolving__sonnet46__seed0"


def test_arms_differ_only_in_reward_weights():
    a, b = cfg('r03_o07'), cfg('r07_o03')
    names = {'name', 'condition', 'output_dir'}
    assert {k for k in set(a) | set(b) if a.get(k) != b.get(k)} <= names | {'agent_reward', 'experiment'}
    assert {k: v for k, v in a['agent_reward'].items() if k not in ('w_outcome', 'w_rubric')} == \
           {k: v for k, v in b['agent_reward'].items() if k not in ('w_outcome', 'w_rubric')}
    diff = {k for k in set(a['experiment']) | set(b['experiment']) if a['experiment'].get(k) != b['experiment'].get(k)}
    assert diff == {'reward_weight_rubric', 'reward_weight_outcome', 'run_id'}


def test_runtime_profile_is_decided_and_shared():
    p = profile()
    assert p['status'] == 'decided' and isinstance(p['api_workers'], int) and isinstance(p['parallel_strata'], bool)
    assert {cfg(arm)['five_gpu']['api_workers'] for arm in ARMS} == {p['api_workers']}
    assert p['checkpoint_evaluation'] == 'async' and not any(cfg(arm)['five_gpu']['checkpoint_evaluation_enabled'] for arm in ARMS)


def test_production_runs_are_isolated_from_pilot_and_each_other(tmp_path):
    r03, r07 = cfg('r03_o07'), cfg('r07_o03')
    pilot = X.load_config('configs/experiments/reward_prompt_matrix/reward_sweep/r03_o07_evolving.json')
    assert len({r03['output_dir'], r07['output_dir'], pilot['output_dir']}) == 3
    assert len({X.fingerprint(r03), X.fingerprint(r07), X.fingerprint(pilot)}) == 3
    out = tmp_path / 'r07'; ck = out / 'checkpoints/checkpoint_000000'
    (ck).mkdir(parents=True); (ck / 'COMPLETE.json').write_text(json.dumps({'experiment_fingerprint': X.fingerprint(r03)}))
    with pytest.raises(RuntimeError, match='Run state'):                       # R07 refuses R03 state
        X.check_resume(r07, out, {'experiment_fingerprint': X.fingerprint(r03), 'theta0_policy_sha256': 't', 'checkpoint': str(ck)}, 't')


def test_doctor_port_overlay_changes_runtime_only(tmp_path):
    base = f'{PROD}/r07_o03_evolving_300.json'
    overlay = tmp_path / 'ports.json'
    overlay.write_text(json.dumps({'extends': base, 'five_gpu': {'doctor_urls': [f'http://127.0.0.1:{p}' for p in range(12000, 12005)]}}))
    a, b = cfg('r07_o03'), X.load_config(str(overlay))
    assert X.fingerprint(a) == X.fingerprint(b) and a['five_gpu']['doctor_urls'] != b['five_gpu']['doctor_urls']
    assert a['output_dir'] == b['output_dir']


def test_standby_files_have_no_machine_paths_and_launcher_parses():
    files = [*(ROOT / PROD).glob('*.json'), ROOT / 'scripts/online_rl/launch_production.sh', ROOT / 'scripts/online_rl/preflight_destination.py']
    assert all('/home/' not in f.read_text() for f in files)
    assert subprocess.run(['bash', '-n', str(ROOT / 'scripts/online_rl/launch_production.sh')]).returncode == 0
