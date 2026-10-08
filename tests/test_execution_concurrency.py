"""Execution-only concurrency (Agent replicas / independent API calls) never changes logical results: concurrent tree
sampling, response scoring, member verification, feedback and evaluation episodes give exactly the serial records
(random delays make completion order differ from submission order). Fakes only."""
import json, random, threading, time
from pathlib import Path
from src.online_rl.tree.cache import CallCache
from src.online_rl.tree.rollout import build_tree
from src.online_rl.integrated.records import to_dict
from src.collaboration.generator import final_answer_option, stop_reason
from test_doctor_v2 import CASE, HALT, ST, FakeOllama, doctor


def jitter(): time.sleep(random.random() * 0.003)


class SeedAgent:
    """Reply depends only on (turn, seed); random delay; records concurrency."""
    def __init__(self): self.active, self.peak, self.lock = 0, 0, threading.Lock()

    def sample(self, case, prefix, turn, seed):
        with self.lock: self.active += 1; self.peak = max(self.peak, self.active)
        jitter()
        with self.lock: self.active -= 1
        return {'completion_text': f'Turn {turn}: I think {"ABCD"[seed % 4]} is best supported ({seed % 97}).', 'seed': seed}


def tree(agent_workers, cache=None):
    fake = FakeOllama(status_for=lambda r, u, s: ST(open_=(s % 3 == 0)))
    a = SeedAgent()
    t = build_tree(CASE, max_agent_turns=4, branching=3, run_seed=5, policy_sha256='p', protocol_sha256='proto', agent=a, doctor=doctor(fake, HALT),
                   stop_rule=stop_reason, final_option=final_answer_option, outcome_fn=lambda c, o: float(o == 'B') if o else None,
                   cache=cache or CallCache(None), behavior_logprobs=lambda trs: [[0.0] * 3 for _ in trs], workers=4, agent_workers=agent_workers)
    return t, a


def test_concurrent_tree_sampling_equals_the_serial_tree(tmp_path):
    random.seed(0); serial, a1 = tree(1, CallCache(tmp_path / 's.jsonl'))
    par, a4 = tree(4, CallCache(tmp_path / 'p.jsonl'))
    assert to_dict(serial) == to_dict(par) and len(serial.nodes) > 9
    assert a1.peak == 1 and a4.peak > 1                                             # actually concurrent
    key = lambda p: {json.loads(l)['key']: json.loads(l)['value'] for l in Path(p).read_text().splitlines()}
    assert key(tmp_path / 's.jsonl') == key(tmp_path / 'p.jsonl')                    # same logical calls, keys and values


def test_concurrent_response_scoring_keeps_node_order():
    from src.online_rl.tree.scoring import score_nodes
    from src.online_rl.tree.records import TrajectoryRubric
    t, _ = tree(1)
    rub = {'rubric_id': 'R1', 'name': 'r', 'applicability_condition': 'c', 'required_behavior': 'b', 'scores': {'0': 'x', '1': 'y', '2': 'z'}}
    kept = [TrajectoryRubric(**{**{f: None for f in TrajectoryRubric.__dataclass_fields__}, 'rubric_id': 'R1', 'leaf_id': 'L', 'rubric': rub, 'valid': True})]

    def scorer(view, prefix, rubrics, reply):
        jitter(); h = sum(map(ord, reply)) % 3
        return {'R1': {'applicable': h != 0, 'probabilities': None, 'hard': h}}
    s1 = score_nodes(CASE, list(t.nodes), kept, scorer, CallCache(None)); s8 = score_nodes(CASE, list(t.nodes), kept, scorer, CallCache(None), workers=8)
    assert [to_dict(x) for x in s1] == [to_dict(x) for x in s8] and len(s1) == len(t.nodes)


def test_concurrent_trainer_case_equals_the_serial_case(tmp_path):
    """Rubric-only-style case (feedback, bank, member verification, scoring, reruns) with every execution knob on."""
    from test_tree_v1 import P0, CASES, GroupUpdater, Log, cfg, mock_env, search_backend
    from src.online_rl.tree import team as T
    from src.online_rl.tree.conditions import arm_config
    from src.online_rl.tree.trainer import TreeTrainer
    recs = []
    for name, ex in (('serial', None), ('parallel', {'agent_workers': 4, 'api_workers': 8, 'member_workers': 3})):
        random.seed(1); w = {'v': 0}; calls = []
        c = arm_config({**cfg(cases=2, every=2, min_traj=4, rounds=3, branching=2)}, 'rubric_only'); c.pop('evaluation', None)
        tr = TreeTrainer(c, out_dir=tmp_path / name, cases=CASES, env=mock_env(Log(), w), strategies=T.team_strategies(GroupUpdater(w, calls), 0.0, 1.0),
                         search_backend=search_backend(Log()), initial_prompt=P0, policy_sha=lambda: f'policy-{w["v"]}', protocol_sha='proto',
                         feedback_prompt_sha256='fb', updater=GroupUpdater(w, calls), execution=ex)
        tr.run(); recs.append([{k: v for k, v in json.loads(p.read_text()).items() if k not in ('seconds', 'timings')} for p in sorted((tmp_path / name / 'cases').glob('*.json'))])
    assert recs[0] == recs[1] and len(recs[0]) == 2 and recs[0][0]['feedback']


def test_concurrent_evaluation_equals_serial(tmp_path):
    from src.online_rl.tree.evaluation import MilestoneEvaluator
    cases = [{**CASE, 'case_id': f'v{i}'} for i in range(12)]

    def play(case, seed): jitter(); return {'final_option': 'ABCD'[seed % 4], 'termination_reason': 'doctor_halt', 'num_discussion_rounds': seed % 5}
    prov = {'checkpoint_sha256': 'ck', 'policy_sha256': 'ck', 'prompt_id': None, 'prompt_sha256': None, 'config_sha256': None}
    out = []
    for w in (1, 6):
        play.workers = w
        r = MilestoneEvaluator(play, lambda c, o: float(o == 'B'), cases, 'val50', 'm', 'proto', 0)(milestone=0, checkpoint=Path('x'), out_dir=tmp_path / f'w{w}', provenance=prov)
        out.append((r['accuracy'], r['per_case']))
    assert out[0] == out[1] and [x['case_id'] for x in out[1][1]] == [c['case_id'] for c in cases]


def test_runtime_concurrency_uses_configured_member_workers(tmp_path, monkeypatch):
    from scripts.online_rl.run_tree import runtime_execution
    monkeypatch.delenv('SHERPA_MEMBER_WORKERS', raising=False)
    monkeypatch.delenv('SHERPA_API_WORKERS', raising=False)
    cfg = {'output_dir': str(tmp_path), 'execution': {'member_workers': 2},
           'five_gpu': {'api_workers': 3}, 'rollout': {'agent_sampling_protocol': 'per_row_rng_batch_v1'},}
    policy = type('Policy', (), {'n_replicas': 1})()
    ex = runtime_execution(cfg, policy)
    assert ex['member_workers'] == 2
    assert ex['witness_workers'] == 1
    policy.pool_execution = {'doctor_workers': 5}
    ex = runtime_execution(cfg, policy)
    assert ex['witness_workers'] == 5
    assert ex['api_workers'] == 3
