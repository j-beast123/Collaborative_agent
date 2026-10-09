"""Reward mixtures, Rubric-Generator prompt modes, run identity, resume guards and GPU remapping on the five-GPU batch
runner (CPU only; no API, no GPU)."""
import copy
import dataclasses
import itertools
import json
import threading
from concurrent.futures import Future
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
import pytest
from src.online_rl.integrated.records import atomic_write_json
from src.online_rl.tree import experiment as X
from src.online_rl.tree.devices import physical_gpus
from src.online_rl.tree.team import compose_reward, node_credit
from tests.test_five_gpu import tiny_policy

ROOT = X.ROOT
MATRIX = 'configs/experiments/reward_prompt_matrix'
REFERENCE = 'configs/online_rl/tree_v21_five_gpu_shared5_random10_promptv7.json'
SWEEP = {'r0_o1': (0.0, 1.0), 'r03_o07': (0.3, 0.7), 'r05_o05': (0.5, 0.5), 'r07_o03': (0.7, 0.3), 'r1_o0': (1.0, 0.0)}
SONNET46 = 'anthropic/claude-4.6-sonnet-20260217'
CASES = 10                                   # two five-case batches
TOL = X.REWARD_TOLERANCE
# (w_rubric, w_outcome) -> rewards of (outcome 1, R_A .5), (outcome 0, R_A 1), (outcome 1, R_A None)
EXPECTED = {(0.0, 1.0): (1.0, 0.0, 1.0), (0.3, 0.7): (0.85, 0.3, 0.7), (0.5, 0.5): (0.75, 0.5, 0.5),
            (0.7, 0.3): (0.65, 0.7, 0.3), (1.0, 0.0): (0.5, 1.0, None)}
POINTS = ((1.0, 0.5), (0.0, 1.0), (1.0, None))


REWARD_RUNS = {'r0_o1': 'reward_sweep/r0_o1_outcome_only.json', **{t: f'reward_sweep/{t}_evolving.json' for t in SWEEP if t != 'r0_o1'}}
EVOLVING = REWARD_RUNS['r05_o05']                                   # also the prompt-ablation EVOLVING arm
P0_FIXED, PFINAL = 'prompt_ablation/r05_o05_p0_fixed.json', 'prompt_ablation/r05_o05_pfinal_fixed.json'
RUBRIC_TAGS = [t for t in SWEEP if t != 'r0_o1']


def sweep(tag): return REWARD_RUNS[tag]


def cfg_of(name): return X.load_config(f'{MATRIX}/{name}')


def reference_reward(outcome, ra, w_outcome, w_rubric):
    """Oracle: the node_credit reward expression before compose_reward was extracted (b647f33), verbatim."""
    unassessed_rubric_only = w_outcome == 0.0 and w_rubric > 0.0 and ra is None
    return None if outcome is None or unassessed_rubric_only else w_outcome * outcome + (w_rubric * ra if ra is not None else 0.0)


def tiny_tree(prefix=''):
    nodes = [SimpleNamespace(node_id=f'{prefix}{i}', parent_id=None) for i in range(3)]
    leaves = [SimpleNamespace(leaf_id=f'{prefix}l{i}', outcome=o, path=(f'{prefix}{i}',)) for i, (o, _) in enumerate(POINTS)]
    return SimpleNamespace(nodes=nodes, leaves=leaves), {f'{prefix}{i}': ra for i, (_, ra) in enumerate(POINTS)}


# A. reward formula
@pytest.mark.parametrize('w_r,w_o', list(EXPECTED))
def test_reward_is_the_configured_weighted_sum(w_r, w_o):
    got = [compose_reward(o, ra, w_o, w_r) for o, ra in POINTS]
    for g, e in zip(got, EXPECTED[(w_r, w_o)]):
        assert (g is None) if e is None else g == pytest.approx(e, rel=0, abs=TOL)
    tree, ra = tiny_tree()
    targets = node_credit(w_o, w_r)(tree, None, ra)
    assert [targets['nodes'][str(i)]['reward'] for i in range(3)] == got
    assert targets['reward_weights'] == {'w_outcome': w_o, 'w_rubric': w_r}
    cfg = {'experiment': {}, 'agent_reward': {'w_outcome': w_o, 'w_rubric': w_r}, 'prompt_update': {'enabled': True},
           'rubric': {'enabled': w_r > 0}}
    if w_r == 0: targets['nodes']['0']['R_A'] = targets['nodes']['1']['R_A'] = None   # not computed when rubric weight is 0
    rows = X.check_batch(cfg, [{'case_cursor': 0, 'case_id': 'c', 'agent_targets': targets}])
    assert [r['final_reward'] for r in rows] == [targets['nodes'][str(i)]['reward'] for i in range(3)]
    assert {(r['rubric_weight'], r['outcome_weight']) for r in rows} == {(w_r, w_o)}
    assert {r['R_rubric_status'] for r in rows} == ({X.NOT_COMPUTED} if w_r == 0 else {X.SCORED, X.NA})


# C. mixed reward example
@pytest.mark.parametrize('w_r,w_o', [(0.3, 0.7), (0.5, 0.5), (0.7, 0.3)])
def test_mixed_reward_example(w_r, w_o):
    assert compose_reward(0.8, 0.4, w_o, w_r) == pytest.approx(w_o * 0.8 + w_r * 0.4, rel=0, abs=TOL)


# D / NA semantics
def test_na_semantics_equal_the_reference_formula():
    values = (None, 0.0, 0.25, 0.5, 0.8, 1.0)
    for (w_r, w_o), o, ra in itertools.product(EXPECTED, values, values):
        assert compose_reward(o, ra, w_o, w_r) == reference_reward(o, ra, w_o, w_r)
    for ra in (0.0, 0.5, 1.0):
        assert {compose_reward(o, ra, 0.0, 1.0) for o in (0.0, 0.5, 1.0)} == {ra}      # R1_O0: outcome has no effect
    assert compose_reward(1.0, None, 0.0, 1.0) is None                                # exclude_unassessed_v1
    assert compose_reward(1.0, None, 0.5, 0.5) == 0.5                                 # mixed: NA adds no rubric term
    for o in (0.0, 0.5, 1.0):
        assert {compose_reward(o, ra, 1.0, 0.0) for ra in (None, 0.0, 1.0)} == {o}    # R0_O1: rubric has no effect


# B. R0_O1 runs no rubric pipeline
def test_r0_o1_makes_no_rubric_feedback_matcher_witness_or_prompt_calls(tmp_path):
    from tests.test_tree_v1 import Log, mock_env, cfg as tree_cfg, search_backend, CASES as TREE_CASES, P0
    from src.online_rl.tree.team import team_strategies
    from src.online_rl.tree.trainer import TreeTrainer
    exp = cfg_of(sweep('r0_o1'))
    assert not exp['rubric']['enabled'] and not exp['prompt_update']['enabled']
    c = tree_cfg(cases=2)
    c['rubric']['enabled'] = exp['rubric']['enabled']; c['prompt_update']['enabled'] = exp['prompt_update']['enabled']
    c['agent_reward'] = {k: exp['agent_reward'][k] for k in ('w_outcome', 'w_rubric')}
    log, w = Log(), {'v': 0}
    env = dataclasses.replace(mock_env(log, w), assign_fn=lambda *a, **k: log.add('matcher'), gate_fn=lambda *a, **k: log.add('gate'))
    st = dataclasses.replace(team_strategies(None, exp['agent_reward']['w_outcome'], exp['agent_reward']['w_rubric']),
                             agent_update=lambda tree, targets: {'optimizer_step': False})
    tr = TreeTrainer(c, out_dir=tmp_path, cases=TREE_CASES, env=env, strategies=st, search_backend=search_backend(log), initial_prompt=P0,
                     policy_sha=lambda: 'p', protocol_sha='proto', feedback_prompt_sha256='fb')
    tr.run()
    assert log.counts.get('agent') and log.counts.get('final')
    rubric_side = ('feedback', 'rubric', 'matcher', 'gate', 'score', 'write', 'rerun', 'reflect', 'refine')
    assert {k: log.counts.get(k, 0) for k in rubric_side} == dict.fromkeys(rubric_side, 0)
    records = [json.loads(p.read_text()) for p in sorted((tmp_path / 'cases').glob('case_*.json'))]
    assert len(records) == 2 and not (tmp_path / 'prompt_events').exists()
    for r in records:
        assert all(n['R_A'] is None and n['reward'] == n['outcome_component'] for n in r['agent_targets']['nodes'].values())
    assert {r['prompt_sha256'] for r in records} == {tr.state.prompt_sha256}
    rows = X.check_batch({**exp, 'rubric': {**exp['rubric'], 'prompt_sha256': tr.state.prompt_sha256}}, records)
    assert rows and {r['R_rubric_status'] for r in rows} == {X.NOT_COMPUTED} and all(r['final_reward'] == r['R_outcome'] for r in rows)


def test_matrix_files_are_current_and_runs_are_unique():
    from scripts.online_rl.experiment_matrix import main
    main(['--check'])
    m = json.loads((ROOT / MATRIX / 'manifest.json').read_text())
    assert [r['config'] for r in m['families']['reward_sweep']] == [f'{MATRIX}/{REWARD_RUNS[t]}' for t in SWEEP]
    assert [(r['config'], r['priority'], r['status']) for r in m['families']['prompt_ablation']] == [
        (f'{MATRIX}/{P0_FIXED}', 2, 'ready'), (f'{MATRIX}/{EVOLVING}', 2, 'ready'), (f'{MATRIX}/{PFINAL}', 3, 'blocked_pfinal_missing')]
    assert m['run_count_by_priority'] == {'1': 5, '2': 1} and len(m['runs']) == 6 and len(list([f for fam in ('reward_sweep', 'prompt_ablation') for f in (ROOT / MATRIX / fam).glob('*.json')])) == 7
    runs = m['runs']; evo = runs['r05_o05__evolving__sonnet46__seed0']
    assert evo['families'] == ['prompt_ablation', 'reward_sweep'] and evo['priority'] == 1      # reused, not duplicated
    assert m['anchor_reward_rubric_outcome'] == [0.5, 0.5] and m['pfinal_source_run_id'] == evo['run_id']
    dirs = [r['output_dir'] for r in runs.values()]
    assert len(set(dirs)) == len(dirs) and all(Path(d).name == rid for rid, d in zip(runs, dirs))
    sweep_runs = {(r['reward_weight_rubric'], r['reward_weight_outcome']): r for r in runs.values() if 'reward_sweep' in r['families']}
    assert set(sweep_runs) == set(EXPECTED)
    oo = sweep_runs[(0.0, 1.0)]
    assert (oo['arm_label'], oo['effective_prompt_mode'], oo['prompt_evolution_applicable'], oo['prompt_evolution'], oo['rubric_pipeline']) == \
        ('Outcome-only', 'N/A_OUTCOME_ONLY', False, False, False) and oo['run_id'] == 'r0_o1__outcomeonly__sonnet46__seed0'
    for k, r in sweep_runs.items():
        if k != (0.0, 1.0): assert (r['prompt_mode'], r['prompt_evolution'], r['rubric_pipeline']) == ('evolving', True, True), k
    assert sweep_runs[(1.0, 0.0)]['arm_label'] == 'Rubric-reward-only + evolving rubric prompt'
    assert all((r['reward_weight_rubric'], r['reward_weight_outcome']) == (0.5, 0.5) for r in runs.values() if 'prompt_ablation' in r['families'])
    assert all(r['feedback_model'] == SONNET46 and len(r['feedback_prompt_sha256']) == 64 for r in runs.values())


def test_prompt_ablation_arms_differ_only_in_prompt_mode():
    p0, evo = cfg_of(P0_FIXED), cfg_of(EVOLVING)
    assert X.fingerprint(p0) != X.fingerprint(evo)
    strip = lambda c: {k: v for k, v in c.items() if k not in ('name', 'condition', 'output_dir', 'experiment', 'prompt_update')}
    assert strip(p0) == strip(evo) and p0['agent_reward'] == evo['agent_reward'] and p0['rubric'] == evo['rubric']
    assert {k: v for k, v in p0['prompt_update'].items() if k != 'enabled'} == {k: v for k, v in evo['prompt_update'].items() if k != 'enabled'}
    assert (p0['prompt_update']['enabled'], evo['prompt_update']['enabled']) == (False, True)
    diff = {k for k in set(p0['experiment']) | set(evo['experiment']) if p0['experiment'].get(k) != evo['experiment'].get(k)}
    assert diff == {'prompt_mode', 'run_id'}


def test_matrix_never_reads_evaluation_results():
    src = (ROOT / 'scripts/online_rl/experiment_matrix.py').read_text() + (ROOT / 'src/online_rl/tree/experiment.py').read_text()
    assert not any(w in src for w in ('evaluation/', 'summary.json', 'accuracy', 'casimedicos'))


def test_configs_differ_from_reference_protocol_only_in_reward_prompt_mode_feedback_model_and_runtime_path():
    ref = json.loads((ROOT / REFERENCE).read_text())
    allowed = {'name', 'condition', 'output_dir', 'experiment', 'agent_reward', 'feedback', 'prompt_update', 'rubric', 'five_gpu'}
    for f in sorted([f for fam in ('reward_sweep', 'prompt_ablation') for f in (ROOT / MATRIX / fam).glob('*.json')]):
        if f.name == Path(PFINAL).name: continue
        cfg = X.load_config(f.relative_to(ROOT)); e = cfg['experiment']
        assert {k for k in set(ref) | set(cfg) if ref.get(k) != cfg.get(k)} <= allowed, f
        for sec, keys in (('feedback', {'model'}), ('prompt_update', {'enabled'}), ('agent_reward', {'w_outcome', 'w_rubric'}),
                          ('rubric', {'enabled'}), ('five_gpu', {'ollama_binary'})):
            assert {k: v for k, v in cfg[sec].items() if k not in keys} == {k: v for k, v in ref[sec].items() if k not in keys}, (f, sec)
        assert cfg['feedback']['model'] == SONNET46 and e['feedback_prompt_sha256'] == ref['feedback']['prompt_sha256']
        assert (cfg['agent_reward']['w_rubric'], cfg['agent_reward']['w_outcome']) == (e['reward_weight_rubric'], e['reward_weight_outcome'])
        assert cfg['rubric']['enabled'] == (e['reward_weight_rubric'] > 0)
        assert cfg['prompt_update']['enabled'] == (e['prompt_mode'] == 'evolving')
        assert '/home/' not in cfg['five_gpu']['ollama_binary']


def test_every_ready_config_passes_runner_preflight(monkeypatch):
    from conftest import require_local
    from scripts.online_rl.run_five_gpu import preflight
    ref = json.loads((ROOT / REFERENCE).read_text())
    require_local(ref['data']['train'], ref['five_gpu']['eval_file'], ref['five_gpu']['train_order_file'], why='preflight hashes local data')
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)
    for tag in SWEEP: assert preflight(cfg_of(sweep(tag))) == [], tag
    assert preflight(cfg_of(P0_FIXED)) == []
    with pytest.raises(ValueError, match='pfinal_artifact is required'): cfg_of(PFINAL)    # F: blocked without Pfinal


@pytest.mark.parametrize('override,match', [({'reward_weight_rubric': 0.4}, 'sum to 1'),
    ({'reward_weight_rubric': 0.25, 'reward_weight_outcome': 0.75}, 'multiple'), ({'prompt_mode': 'no_prompt'}, 'prompt_mode'),
    ({'prompt_mode': 'pfinal_fixed'}, 'pfinal_artifact'), ({'pfinal_artifact': 'x.json'}, 'pfinal_artifact'),
    ({'prompt_mode': 'evolving', 'reward_weight_rubric': 0.0, 'reward_weight_outcome': 1.0}, 'needs a rubric reward'),
    ({'output_dir': 'outputs/other'}, 'output_dir'), ({'feedback_model': 'anthropic/claude-sonnet-4.6-guess'}, 'unregistered'),
    ({'train_order_sha256': '0' * 64}, 'train_order'), ({'seed': True}, 'seed'), ({'surprise': 1}, 'unknown'),
    ({'reward_weight_rubric': float('nan')}, 'numbers in'), ({'reward_weight_rubric': -0.5, 'reward_weight_outcome': 1.5}, 'numbers in'),
    ({'reward_weight_outcome': float('inf')}, 'numbers in')])
def test_invalid_experiment_fails_fast(override, match):
    raw = X.load_raw(f'{MATRIX}/{P0_FIXED}')
    with pytest.raises(ValueError, match=match): X.resolve(X.deep_merge(raw, {'experiment': override}))


def test_fingerprint_separates_every_research_axis():
    names = [sweep(t) for t in SWEEP] + [P0_FIXED]
    assert len({X.fingerprint(cfg_of(n)) for n in names}) == 6
    base = cfg_of(P0_FIXED)
    for path, value in ((('feedback', 'model'), 'x'), (('feedback', 'prompt_sha256'), 'f'), (('five_gpu', 'train_order_sha256'), 'y'),
                        (('data', 'train_sha256'), 'z'), (('policy', 'lora', 'init_seed'), 1), (('policy', 'init_adapter'), 'a'),
                        (('rubric', 'prompt_sha256'), 'p'), (('schedule', 'seed'), 1)):
        c = copy.deepcopy(base); d = c
        for k in path[:-1]: d = d[k]
        d[path[-1]] = value
        assert X.fingerprint(c) != X.fingerprint(base), path
    for path, value in ((('five_gpu', 'ollama_binary'), '/opt/ollama'), (('five_gpu', 'doctor_urls'), ['u'] * 5)):
        c = copy.deepcopy(base); c[path[0]][path[1]] = value
        assert X.fingerprint(c) == X.fingerprint(base), path                          # runtime-only


def test_checkpoint_from_another_run_or_condition_is_refused(tmp_path):
    cfg = cfg_of(P0_FIXED); fp = X.fingerprint(cfg); other = X.fingerprint(cfg_of(sweep('r07_o03')))
    out = tmp_path / 'a'; ck = out / 'checkpoints/checkpoint_000000'
    atomic_write_json(ck / 'COMPLETE.json', {'experiment_fingerprint': fp})
    good = {'experiment_fingerprint': fp, 'theta0_policy_sha256': 't0', 'checkpoint': str(ck)}
    X.check_resume(cfg, out, good, 't0')
    with pytest.raises(RuntimeError, match='theta0'): X.check_resume(cfg, out, good, 't1')
    for state in ({**good, 'experiment_fingerprint': other}, {'checkpoint': str(ck)}):
        with pytest.raises(RuntimeError, match='Run state'): X.check_resume(cfg, out, state, 't0')
    foreign = tmp_path / 'b/checkpoints/checkpoint_000000'
    atomic_write_json(foreign / 'COMPLETE.json', {'experiment_fingerprint': fp})
    with pytest.raises(RuntimeError, match='outside'): X.check_resume(cfg, out, {**good, 'checkpoint': str(foreign)}, 't0')
    atomic_write_json(ck / 'COMPLETE.json', {'experiment_fingerprint': other})
    with pytest.raises(RuntimeError, match='Checkpoint belongs'): X.check_resume(cfg, out, good, 't0')
    X.check_resume({k: v for k, v in cfg.items() if k != 'experiment'}, out, {'checkpoint': 'unused'}, 'any')


def prepare(tmp_path, cfg):
    """Shrink a resolved config to a CPU fixture: ten cases, local data, no checkpoint evaluation."""
    cfg = copy.deepcopy(cfg); tmp_path.mkdir(parents=True, exist_ok=True)
    cfg['output_dir'] = str(tmp_path / 'run'); cfg['schedule']['cases'] = CASES
    cfg['five_gpu'].pop('train_order_file'); cfg['five_gpu']['checkpoint_evaluation_enabled'] = False
    train = tmp_path / 'train.jsonl'
    train.write_text(''.join(json.dumps({'case_id': f'c{i}', 'question': f'q{i}', 'options': {'A': 'a', 'B': 'b'},
                                         'correct_option': 'A'}) + '\n' for i in range(CASES)))
    cfg['data']['train'] = str(train)
    return cfg


def run_harness(tmp_path, monkeypatch, cfg, tie=False, mutate=None):
    """The production coordinator with CPU LoRA weights and immediate shared-pool futures (as tests/test_five_gpu.py)."""
    import torch
    import scripts.online_rl.run_five_gpu as runner
    import scripts.online_rl.run_tree as tree_runner
    import src.online_rl.runtime as runtime
    import src.online_rl.integrated.grpo as grpo
    from src.online_rl.runtime import GRPOUpdater
    from src.online_rl.tree import batch_prompt as BP, five_gpu_worker as worker, gpu_pool
    def fresh_policy(*a, **k):
        # Each launch loads theta0, as SharedPolicy does.
        p = tiny_policy(); p.doctor = SimpleNamespace(verify=lambda: {'problems': []}, http=lambda *a: {}); return p
    for target, name, value in ((runtime, 'SharedPolicy', fresh_policy), (tree_runner, 'load_key', lambda: 'dummy'),
                                (torch.cuda, 'is_available', lambda: True), (torch.cuda, 'device_count', lambda: 5),
                                (torch.cuda, 'get_rng_state_all', lambda: []), (torch.cuda, 'set_rng_state_all', lambda s: None)):
        monkeypatch.setattr(target, name, value)
    path_open = Path.open
    log = SimpleNamespace(updates=[], phases=[], prompts=[], proposals=0, locks=[])

    def redirect(p, *a, **k):
        if str(p).startswith('/tmp/sherpa_tree_physical_gpu'):
            log.locks.append(str(p)); p = tmp_path / Path(p).name
        return path_open(p, *a, **k)
    monkeypatch.setattr(Path, 'open', redirect)

    class Updater(GRPOUpdater):
        def update_groups(self, groups):
            log.updates.append(len(groups)); self.steps += 1
            for p in self.policy.model.parameters():
                if p.requires_grad: p.data.add_(0.01)
            return {'optimizer_step': True, 'steps': self.steps}
    monkeypatch.setattr(grpo, 'make_updater', Updater)
    w = cfg['agent_reward']; originals = {}

    def record(job, sha, prompt_sha):
        cid = job['case']['case_id']; tree, ra = tiny_tree(f'{cid}/')
        if not cfg['rubric']['enabled']: ra = dict.fromkeys(ra)
        nodes = [{'node_id': n.node_id, 'parent_id': None, 'case_id': cid, 'depth': 1, 'prefix': [], 'response': 'x',
                  'trace': {'prompt_input_ids': [1]}, 'policy_sha256': sha, 'seed': i, 'stop_reason': None, 'terminal': True,
                  'next_doctor': None} for i, n in enumerate(tree.nodes)]
        return {'case_cursor': job['index'], 'case_id': cid, 'policy_sha256': sha, 'prompt_sha256': prompt_sha,
                'tree': {'case_id': cid, 'policy_sha256': sha, 'max_agent_turns': 3, 'branching': 3, 'nodes': nodes, 'leaves': []},
                'agent_targets': node_credit(w['w_outcome'], w['w_rubric'])(tree, None, ra)}

    class Executor:
        def __init__(self, **kwargs): pass

        def submit(self, fn, *args):
            future = Future()
            if fn.__name__ == 'prepare': future.set_result(args[1]); return future
            jobs, _, sha, *extra = args
            phase, prompt = (extra[0], extra[1]) if extra else ('fixed', None)
            log.phases.append(phase); out = []
            for j in jobs:
                psha = cfg['rubric']['prompt_sha256'] if phase == 'fixed' else prompt['sha256']
                log.prompts.append((phase, psha)); r = record(j, sha, psha)
                if mutate: r = mutate(r)
                if phase in ('P0', 'P1'):
                    keys = {r['case_id'] + '/success': 1., r['case_id'] + '/failure': 0.}
                    r['prompt_evaluation'] = {'per_trajectory': {k: .2 if phase == 'P0' or tie else .8 for k in keys},
                                              'source_outcome': keys, 'rerun_policy_sha256': sha}
                    if phase == 'P0': originals[r['case_id']] = r
                out.append(originals[r['case_id']] if phase == 'union' else r)
            future.set_result(out); return future

        def shutdown(self, **kwargs): pass

    class Actor:
        @contextmanager
        def changing_policy(self): yield

        def close(self): pass

    def propose(rows, text, store, *args):
        log.proposals += 1; cand = text + '\nRefine once.'
        return {'text': cand, 'sha256': store.put(cand), 'proposal': {}, 'valid_candidate': True}
    pool = tmp_path / 'pool'
    for target, name, value in ((worker, 'initialize', lambda *a, existing_policy=None: setattr(existing_policy, 'pool_actor', Actor())),
                                (runner, 'ProcessPoolExecutor', Executor), (runner, 'ThreadPoolExecutor', Executor),
                                (runner.multiprocessing, 'get_context', lambda *a: SimpleNamespace(Manager=lambda: SimpleNamespace(shutdown=lambda: None))),
                                (gpu_pool, 'create_pool', lambda *a: {'leases': {}, 'directory': str(pool), 'lock': threading.RLock()}),
                                (BP, 'propose_candidate', propose)):
        monkeypatch.setattr(target, name, value)
    return runner, log


def read(path): return json.loads(Path(path).read_text())


def case_records(run): return [read(p) for p in sorted((run / 'cases').glob('case_*.json'))]


def assert_fixed_run(run, cfg, log, prompt_sha):
    assert len(case_records(run)) == CASES and {r['prompt_sha256'] for r in case_records(run)} == {prompt_sha}
    assert set(log.phases) == {'fixed'} and log.proposals == 0 and not (run / 'prompt_events').exists()
    assert log.updates == [5, 5]                                                       # one update per five-case batch
    w = cfg['agent_reward']
    for b in (1, 2):
        rows = read(run / f'reward_components/batch_{b:06d}.json')
        assert len(rows) == 15 and {(r['rubric_weight'], r['outcome_weight']) for r in rows} == {(w['w_rubric'], w['w_outcome'])}
        assert all(r['final_reward'] == compose_reward(r['R_outcome'], r['R_rubric'], w['w_outcome'], w['w_rubric']) for r in rows)
    state = read(run / 'run_state.json')
    assert state['experiment_fingerprint'] == X.fingerprint(cfg) == read(Path(state['checkpoint']) / 'COMPLETE.json')['experiment_fingerprint']


# E. P0_FIXED arm and the outcome-only endpoint (no prompt evolution)
@pytest.mark.parametrize('name', [P0_FIXED, REWARD_RUNS['r0_o1']])
def test_fixed_prompt_runs_keep_p0_hash_across_batches(tmp_path, monkeypatch, name):
    cfg = prepare(tmp_path, cfg_of(name)); runner, log = run_harness(tmp_path, monkeypatch, cfg)
    runner.run(cfg)
    assert cfg['rubric']['prompt_sha256'] == json.loads((ROOT / REFERENCE).read_text())['rubric']['prompt_sha256']
    assert_fixed_run(tmp_path / 'run', cfg, log, cfg['rubric']['prompt_sha256'])
    rows = read(tmp_path / 'run/reward_components/batch_000001.json')
    if name == P0_FIXED: assert {r['R_rubric_status'] for r in rows} == {X.SCORED, X.NA}
    else: assert {r['R_rubric_status'] for r in rows} == {X.NOT_COMPUTED} and all(r['final_reward'] == r['R_outcome'] for r in rows)


# G. EVOLVING at every rubric-weighted ratio
@pytest.mark.parametrize('tag,tie', [(t, False) for t in RUBRIC_TAGS] + [('r05_o05', True)])
def test_evolving_uses_existing_batch_prompt_update_and_strict_adoption(tmp_path, monkeypatch, tag, tie):
    cfg = prepare(tmp_path, cfg_of(sweep(tag))); runner, log = run_harness(tmp_path, monkeypatch, cfg, tie=tie)
    runner.run(cfg)
    run = tmp_path / 'run'; p0 = cfg['rubric']['prompt_sha256']; state = read(run / 'run_state.json')
    events = [read(p) for p in sorted((run / 'prompt_events').glob('batch_*.json'))]
    assert len(events) == 2 and log.updates == [5, 5] and log.proposals == 2
    assert all(e['mode'] == 'five_case_p0_p1_v1' and e['candidate_seed']['seed_mode'] == 'first_completed_P0_v1' for e in events)
    assert all(e['training_pool'] == 'union_both_kept_deduplicated_v1' for e in events)
    batch_parents = [s for phase, s in log.prompts if phase == 'P0']
    assert batch_parents[:5] == [p0] * 5 and batch_parents[5:] == [events[0]['next_prompt']['sha256']] * 5
    for e in events:
        d = e['decision']; assert d['accepted'] == (d['candidate']['J'] > d['parent']['J'])
        assert d['parent']['J'] == d['parent']['S_success'] * d['parent']['S_failure']
    w = cfg['agent_reward']; rows = read(run / 'reward_components/batch_000002.json')
    assert {(r['rubric_weight'], r['outcome_weight']) for r in rows} == {(w['w_rubric'], w['w_outcome'])}
    assert all(r['final_reward'] == compose_reward(r['R_outcome'], r['R_rubric'], w['w_outcome'], w['w_rubric']) for r in rows)
    if tie:
        assert all(e['decision']['reason'] == 'tie_keep_parent' for e in events) and state['prompt_sha256'] == p0
    else:
        assert all(e['decision']['reason'] == 'higher_product' for e in events)
        assert events[1]['parent_prompt']['sha256'] == events[0]['next_prompt']['sha256'] != p0
        assert state['prompt_sha256'] == events[1]['next_prompt']['sha256']


def evolve_and_export(tmp_path, monkeypatch, name=EVOLVING, cfg=None):
    from scripts.online_rl.export_pfinal import export
    evo = prepare(tmp_path / 'evo', cfg or cfg_of(name))
    runner, _ = run_harness(tmp_path / 'evo', monkeypatch, evo); runner.run(evo)
    return export(tmp_path / 'evo/run', tmp_path / 'pfinal'), read(tmp_path / 'evo/run/run_state.json')


def pfinal_config(path):
    return X.resolve(X.deep_merge(X.load_raw(f'{MATRIX}/{PFINAL}'), {'experiment': {'pfinal_artifact': str(path)}}))


# F. PFINAL_FIXED
def test_pfinal_is_exported_from_a_completed_evolving_run_and_fixed_from_case_0(tmp_path, monkeypatch):
    path, state = evolve_and_export(tmp_path, monkeypatch); meta = X.load_pfinal(str(path))
    assert meta['prompt_sha256'] == state['prompt_sha256'] != meta['p0_sha256']
    assert meta['source_completed_cases'] == CASES and meta['source_batch'] == 2 and meta['source_checkpoint'] == state['checkpoint']
    assert meta['parent_prompt_sha256'] == meta['adoption_history'][-1]['parent_sha256'] and len(meta['adoption_history']) == 2
    assert meta['source_feedback_model'] == SONNET46 and meta['source_reward_weights'] == {'w_outcome': 0.5, 'w_rubric': 0.5}
    resolved = pfinal_config(path)
    assert resolved['experiment']['run_id'] == f"r05_o05__pfinal_{meta['prompt_sha256'][:8]}__sonnet46__seed0"
    assert resolved['experiment']['pfinal']['source_run_id'] == meta['source_run_id'] and not resolved['prompt_update']['enabled']
    cfg = prepare(tmp_path / 'fixed', resolved); runner, log = run_harness(tmp_path / 'fixed', monkeypatch, cfg)
    runner.run(cfg)
    assert case_records(tmp_path / 'fixed/run')[0]['case_cursor'] == 0
    assert_fixed_run(tmp_path / 'fixed/run', cfg, log, meta['prompt_sha256'])


@pytest.mark.parametrize('field,value,match', [('source_feedback_model', 'anthropic/claude-4.5-haiku-20251001', 'source condition differs'),
    ('source_reward_weights', {'w_outcome': 0.0, 'w_rubric': 1.0}, 'source condition differs'),
    ('source_train_order_sha256', 'x', 'source condition differs'), ('p0_sha256', 'x', 'does not start at P0|source condition'),
    ('source_prompt_mode', 'p0_fixed', 'must be an evolving run'),
    ('source_protocol_sha256', None, 'missing Pfinal provenance'), ('adoption_history', [], 'missing Pfinal provenance')])
def test_pfinal_with_missing_or_foreign_provenance_is_blocked(tmp_path, monkeypatch, field, value, match):
    path, _ = evolve_and_export(tmp_path, monkeypatch)
    meta = read(path); meta[field] = value; atomic_write_json(path, meta)
    with pytest.raises(ValueError, match=match): pfinal_config(path)


def test_pfinal_from_another_reward_ratio_or_feedback_model_is_refused(tmp_path, monkeypatch):
    path, _ = evolve_and_export(tmp_path, monkeypatch, sweep('r03_o07'))
    with pytest.raises(ValueError, match='source condition differs.*source_reward_weights'): pfinal_config(path)
    haiku = X.resolve(X.deep_merge(X.load_raw(f'{MATRIX}/{EVOLVING}'),
                                   {'experiment': {'feedback_model': 'anthropic/claude-4.5-haiku-20251001', 'output_dir': None}}))
    path, _ = evolve_and_export(tmp_path / 'haiku', monkeypatch, cfg=haiku)
    with pytest.raises(ValueError, match='source condition differs.*source_feedback_model'): pfinal_config(path)


def test_pfinal_theta0_and_protocol_must_match_the_launching_run(tmp_path, monkeypatch):
    path, state = evolve_and_export(tmp_path, monkeypatch)
    cfg = pfinal_config(path); p = cfg['experiment']['pfinal']
    assert p['source_protocol_sha256'] == state['protocol_sha256'] and p['source_prompt_mode'] == 'evolving'
    X.check_pfinal_source(cfg, p['source_theta0_policy_sha256'], p['source_protocol_sha256'])
    with pytest.raises(RuntimeError, match='theta0'): X.check_pfinal_source(cfg, 'other', p['source_protocol_sha256'])
    with pytest.raises(RuntimeError, match='protocol'): X.check_pfinal_source(cfg, p['source_theta0_policy_sha256'], 'other')


def test_pfinal_export_refuses_incomplete_or_unadopted_runs(tmp_path, monkeypatch):
    from scripts.online_rl.export_pfinal import export
    cfg = prepare(tmp_path, cfg_of(EVOLVING))
    runner, _ = run_harness(tmp_path, monkeypatch, cfg, tie=True); runner.run(cfg, max_batches=1)
    with pytest.raises(ValueError, match='incomplete'): export(tmp_path / 'run', tmp_path / 'pfinal')
    runner.run(cfg)
    with pytest.raises(ValueError, match='ever adopted'): export(tmp_path / 'run', tmp_path / 'pfinal')


# H. resume
def test_resume_restores_prompt_state_and_refuses_other_conditions(tmp_path, monkeypatch):
    cfg = prepare(tmp_path, cfg_of(EVOLVING)); runner, log = run_harness(tmp_path, monkeypatch, cfg)
    run = tmp_path / 'run'
    runner.run(cfg, max_batches=1)
    assert read(run / 'run_state.json')['prompt_sha256'] != cfg['rubric']['prompt_sha256']
    real = runner.atomic_write_json

    def crash(path, value):
        if Path(path).name == 'run_state.json' and value.get('case_cursor') == CASES: raise RuntimeError('simulated crash')
        return real(path, value)
    monkeypatch.setattr(runner, 'atomic_write_json', crash)
    with pytest.raises(RuntimeError, match='simulated crash'): runner.run(cfg)
    monkeypatch.setattr(runner, 'atomic_write_json', real)
    runner.run(cfg)
    state = read(run / 'run_state.json'); last = read(run / 'prompt_events/batch_000002.json')
    assert log.updates == [5, 5] and state['case_cursor'] == CASES and state['prompt_sha256'] == last['next_prompt']['sha256']
    assert read(Path(state['checkpoint']) / 'COMPLETE.json')['next_prompt'] == last['next_prompt']
    others = {'reward ratio': cfg_of(sweep('r07_o03')), 'prompt mode': cfg_of(P0_FIXED),
              'feedback model': X.resolve(X.deep_merge(X.load_raw(f'{MATRIX}/{EVOLVING}'),
                                                       {'experiment': {'feedback_model': 'anthropic/claude-4.5-haiku-20251001', 'output_dir': None}}))}
    for axis, other in others.items():
        with pytest.raises(RuntimeError, match='changed|another experiment'): runner.run(prepare(tmp_path, other))
    assert log.updates == [5, 5]


# I. GPU remap
def test_physical_gpu_mapping_is_runtime_only(tmp_path, monkeypatch):
    assert physical_gpus({}) == ['0', '1', '2', '3', '4'] and physical_gpus({'CUDA_VISIBLE_DEVICES': '3,4,5,6,7'}) == ['3', '4', '5', '6', '7']
    for bad in ('0,1', '3,3,4,5,6', '0,1,2,3,4,5'):
        with pytest.raises(ValueError, match='distinct physical'): physical_gpus({'CUDA_VISIBLE_DEVICES': bad})
    cfg = prepare(tmp_path, cfg_of(P0_FIXED)); runner, log = run_harness(tmp_path, monkeypatch, cfg)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0,1,2,3,4'); runner.run(cfg, max_batches=1)
    first = read(tmp_path / 'run/run_state.json')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '3,4,5,6,7'); runner.run(cfg)
    state = read(tmp_path / 'run/run_state.json')
    assert state['case_cursor'] == CASES and log.updates == [5, 5]
    assert (state['config_sha256'], state['experiment_fingerprint']) == (first['config_sha256'], first['experiment_fingerprint'])
    rt = [json.loads(l) for l in (tmp_path / 'run/runtime_provenance.jsonl').read_text().splitlines()]
    assert [r['logical_to_physical_gpu'] for r in rt] == [{str(i): str(i) for i in range(5)}, {str(i): str(i + 3) for i in range(5)}]
    assert log.locks == ['/tmp/sherpa_tree_physical_gpu4.lock', '/tmp/sherpa_tree_physical_gpu7.lock']   # updater = logical GPU4


def test_worker_pins_its_logical_gpu_to_the_mapped_physical_device(tmp_path, monkeypatch):
    import src.online_rl.runtime as runtime
    import src.online_rl.tree.five_gpu_worker as worker
    cfg = prepare(tmp_path, cfg_of(P0_FIXED))
    monkeypatch.setattr(runtime, 'SharedPolicy', lambda *a, **k: SimpleNamespace(doctor=SimpleNamespace(turn=lambda *a: None)))
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '3,4,5,6,7')
    worker.initialize(cfg, 2, 'dummy')
    import os
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '5' and worker._CFG['five_gpu']['rollout_gpus'] == [0, 1, 2, 3, 4]


@pytest.mark.parametrize('mutate,match', [
    (lambda r: {**r, 'prompt_sha256': 'f' * 64}, 'fixed prompt mode'),
    (lambda r: {**r, 'agent_targets': {**r['agent_targets'], 'reward_weights': {'w_outcome': .7, 'w_rubric': .3}}}, 'reward weights'),
    (lambda r: {**r, 'agent_targets': {**r['agent_targets'], 'nodes': {k: {**v, 'reward': v['reward'] + 1e-9}
                                                                       for k, v in r['agent_targets']['nodes'].items()}}}, 'configured mixture')])
def test_batch_guard_stops_before_the_update(tmp_path, monkeypatch, mutate, match):
    cfg = prepare(tmp_path, cfg_of(P0_FIXED)); runner, log = run_harness(tmp_path, monkeypatch, cfg, mutate=mutate)
    with pytest.raises(RuntimeError, match=match): runner.run(cfg)
    assert log.updates == []


def test_zero_rubric_weight_guard_rejects_rubric_artifacts(tmp_path, monkeypatch):
    cfg = prepare(tmp_path, cfg_of(sweep('r0_o1')))
    runner, log = run_harness(tmp_path, monkeypatch, cfg, mutate=lambda r: {**r, 'feedback': [{'text': 'x'}]})
    with pytest.raises(RuntimeError, match='zero rubric weight'): runner.run(cfg)
    assert log.updates == []


def test_real_union_record_passes_the_batch_guard_after_json_round_trip(tmp_path):
    """The EVOLVING training record comes from batch_prompt.score_union, not from TreeTrainer.step."""
    from tests.test_batch_prompt import assessments
    import test_witness_trainer as WT
    from src.online_rl.tree import batch_prompt as B
    from src.online_rl.tree.team import team_strategies
    tr, p0, p1, _ = assessments(tmp_path, eager_score=True)
    tr.st = team_strategies(None, 0.5, 0.5)
    union = json.loads(json.dumps({**B.score_union(tr, p0, p1, WT.P0R), 'case_cursor': 0}))
    cfg = {'experiment': {}, 'agent_reward': {'w_outcome': 0.5, 'w_rubric': 0.5}, 'prompt_update': {'enabled': True},
           'rubric': {'enabled': True}}
    rows = X.check_batch(cfg, [union])
    assert rows and union['agent_targets']['reward_weights'] == {'w_outcome': 0.5, 'w_rubric': 0.5}
    assert any(r['R_rubric_status'] == X.SCORED for r in rows)
    for r in rows:
        expected = None if r['R_outcome'] is None else 0.5 * r['R_outcome'] + (0.5 * r['R_rubric'] if r['R_rubric'] is not None else 0.0)
        assert r['final_reward'] == expected


def test_resume_with_a_different_theta0_is_refused(tmp_path, monkeypatch):
    import src.online_rl.runtime as runtime
    cfg = prepare(tmp_path, cfg_of(P0_FIXED)); runner, log = run_harness(tmp_path, monkeypatch, cfg)
    runner.run(cfg, max_batches=1)
    def other_theta0(*a, **k):
        p = tiny_policy(); p.doctor = SimpleNamespace(verify=lambda: {'problems': []}, http=lambda *a: {})
        for t in p.model.parameters():
            if t.requires_grad: t.data.add_(0.5)
        return p
    monkeypatch.setattr(runtime, 'SharedPolicy', other_theta0)
    with pytest.raises(RuntimeError, match='theta0'): runner.run(cfg)
    assert log.updates == [5]


def test_forged_pfinal_or_wrong_source_run_is_blocked(tmp_path, monkeypatch):
    path, _ = evolve_and_export(tmp_path, monkeypatch)
    meta = read(path); text = Path(meta['prompt_text_path']).read_text() + '\nForged edit.\n'
    forged = tmp_path / 'forged.txt'; forged.write_text(text)
    sha = __import__('hashlib').sha256(text.encode()).hexdigest()
    meta.update(prompt_text_path=str(forged), prompt_sha256=sha)
    meta['adoption_history'][-1]['candidate_sha256'] = sha
    atomic_write_json(path, meta)
    with pytest.raises(ValueError, match='differs from its source run'): pfinal_config(path)
    path, _ = evolve_and_export(tmp_path / 'b', monkeypatch)
    meta = read(path); meta['source_run_id'] = 'r05_o05__evolving__sonnet46__seed1'; atomic_write_json(path, meta)
    with pytest.raises(ValueError, match='matching evolving run'): pfinal_config(path)
