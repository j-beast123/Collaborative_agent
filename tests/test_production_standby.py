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


P0_SHA = 'f8d6e39470d7f8a93ce8224cea9e5aa2ed2fa07ced9d982cc5e16d0e495d9bad'
R05 = {'evolving': 'r05_o05_evolving_300', 'p0_fixed': 'r05_o05_p0_fixed_300'}


def r05(mode): return X.load_config(f'{PROD}/{R05[mode]}.json')


def flat(d, p=''):
    return {k2: v2 for k, v in d.items() for k2, v2 in (flat(v, f'{p}{k}.').items() if isinstance(v, dict) else [(p + k, v)])}


@pytest.mark.parametrize('mode', list(R05))
def test_r05_production_arms_share_the_production_protocol(mode):
    c = r05(mode)
    assert (c['agent_reward']['w_rubric'], c['agent_reward']['w_outcome']) == (0.5, 0.5)
    assert c['schedule']['cases'] == 300 and c['five_gpu']['batch_cases'] == 5 and c['schedule']['seed'] == 0
    assert c['five_gpu']['api_workers'] == profile()['api_workers'] and not c['five_gpu']['checkpoint_evaluation_enabled']
    assert c['experiment']['prompt_mode'] == mode and c['rubric']['enabled'] and c['rubric']['prompt_sha256'] == P0_SHA
    assert c['prompt_update']['enabled'] == (mode == 'evolving')
    tag = {'evolving': 'evolving', 'p0_fixed': 'p0fixed'}[mode]
    assert c['output_dir'] == f'outputs/reward_prompt_matrix_300/r05_o05__{tag}__sonnet46__seed0'


def test_r05_research_difference_is_prompt_mode_only():
    evo, p0 = flat(r05('evolving')), flat(r05('p0_fixed'))
    bookkeeping = {'name', 'condition', 'output_dir', 'experiment.run_id', 'experiment.family'}
    derived = {'prompt_update.enabled'}                        # set from prompt_mode by experiment.apply_prompt_mode
    assert {k for k in evo.keys() | p0.keys() if evo.get(k) != p0.get(k)} - bookkeeping - derived == {'experiment.prompt_mode'}
    a, b = r05('evolving'), r05('p0_fixed')
    a['experiment']['prompt_mode'], a['prompt_update']['enabled'] = 'p0_fixed', False
    assert X.fingerprint(a) == X.fingerprint(b) != X.fingerprint(r05('evolving'))


def test_r05_evolving_matches_r03_production_except_reward_weights():
    r, o = flat(r05('evolving')), flat(cfg('r03_o07'))
    ignore = {'name', 'condition', 'output_dir', 'experiment.run_id', 'experiment.family',
              'agent_reward.w_rubric', 'agent_reward.w_outcome', 'experiment.reward_weight_rubric', 'experiment.reward_weight_outcome'}
    assert {k for k in r.keys() | o.keys() if r.get(k) != o.get(k)} - ignore == set()


def test_r05_launcher_names_and_preflight_mode_rules():
    sh = (ROOT / 'scripts/online_rl/launch_production.sh').read_text()
    assert 'r05_o05_evolving|r05_o05_p0_fixed) NAME=${ARM}_300' in sh
    from scripts.online_rl import preflight_destination as P
    assert P.EXPECTED['p0_sha256'] == P0_SHA and set(P.EXPECTED['prompt_mode']) == {'evolving', 'p0_fixed'}


SERVERS = [{'url': f'http://127.0.0.1:{p}', 'gpu': str(g), 'parallel': 1} for p, g in ((11535, 5), (11536, 6), (11537, 7))]
PACKED = [SERVERS[i]['url'] for i in (0, 0, 1, 1, 2)]


def test_packed_doctor_servers_are_declared_and_unbatched():
    from scripts.online_rl.run_five_gpu import doctor_endpoint_problems
    assert doctor_endpoint_problems({'doctor_urls': PACKED, 'doctor_servers': SERVERS}) == []
    assert doctor_endpoint_problems({'doctor_urls': [f'http://127.0.0.1:{p}' for p in range(5)]}) == []
    assert doctor_endpoint_problems({'doctor_urls': PACKED})                                  # shared URL without declared slots
    batched = [{**SERVERS[0], 'parallel': 2}, *SERVERS[1:]]
    assert any('parallel must be 1' in e for e in doctor_endpoint_problems({'doctor_urls': PACKED, 'doctor_servers': batched}))
    stray = PACKED[:4] + ['http://127.0.0.1:12000']
    assert any('not a declared' in e for e in doctor_endpoint_problems({'doctor_urls': stray, 'doctor_servers': SERVERS}))


def test_server_plan_starts_each_declared_server_once():
    from scripts.online_rl.start_five_gpu_doctors import server_plan
    plan = server_plan({'five_gpu': {'doctor_servers': SERVERS, 'doctor_urls': PACKED}})
    assert plan == [(s['gpu'], s['gpu'], s['url'], 1) for s in SERVERS]


def test_runtime_overlay_with_packed_doctors_keeps_the_research_fingerprint(tmp_path):
    from scripts.online_rl.run_five_gpu import preflight
    base = f'{PROD}/r07_o03_evolving_300.json'
    overlay = tmp_path / 'runtime.json'
    overlay.write_text(json.dumps({'extends': base, 'five_gpu': {'doctor_urls': PACKED, 'doctor_servers': SERVERS}}))
    a, b = cfg('r07_o03'), X.load_config(str(overlay))
    assert X.fingerprint(a) == X.fingerprint(b) and a['output_dir'] == b['output_dir']
    assert {k: v for k, v in a['five_gpu'].items() if k != 'doctor_urls'} == {k: v for k, v in b['five_gpu'].items() if k not in ('doctor_urls', 'doctor_servers')}
    assert not [e for e in preflight(b) if 'Doctor' in e]
