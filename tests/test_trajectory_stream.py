"""No model/API calls: queue timing, bank serialization, recovery and path integrity."""
from dataclasses import dataclass
from pathlib import Path
from threading import Event, RLock
from types import SimpleNamespace as NS
import time
import pytest

from src.online_rl.tree import trajectory_stream as S
from src.online_rl.tree.records import AgentNode, Leaf, DoctorFeedback

@dataclass(frozen=True)
class State:
    case_cursor: int = 0
    policy_sha256: str = 'policy'
    prompt_sha256: str = 'prompt'
    run_seed: int = 0

@dataclass(frozen=True)
class Tin:
    leaf_id: str


def setup(monkeypatch, tmp_path, process):
    monkeypatch.setattr(S.CB, 'validate_snapshot', lambda s: s)
    monkeypatch.setattr(S, 'freeze_trajectory_input', lambda c, l, f, **kw: Tin(l.leaf_id))
    monkeypatch.setattr(S.OCB, 'process_case', process)
    snapshot = {'entries': [], 'sha256': 'empty'}
    root = S.initialize(tmp_path/'global', snapshot, {'policy': 'policy'})
    lock = RLock()
    def make(cursor=0):
        out = tmp_path/f'case_{cursor}'; out.mkdir(exist_ok=True)
        trainer = NS(out=out, cumulative_bank={'discovery_trajectory_cap': None},
                     source_consistency=False, cfg={'rollout': {'rounds': 4, 'branching': 4}})
        return S.StreamingCase(trainer, {'case_id': f'c{cursor}'}, State(), 'p', snapshot,
                              {'root': root, 'lock': lock, 'case_cursor': cursor})
    return make, root, snapshot


def item(i, cursor=0):
    tid = f'c{cursor}/{i}'
    node = AgentNode(tid, None, f'c{cursor}', 1, (), 'response', {}, 'policy', i, 'stop', True, None)
    leaf = Leaf(tid, f'c{cursor}', (tid,), (), 'A', 'A', 1.0, 'stop')
    return leaf, DoctorFeedback(tid, 1.0, 'feedback', 'p'), (node,)


def built(kw, mutate=False):
    before = kw['snapshot']; ids = [l.leaf_id for l in kw['ruled']]
    after = {'entries': before['entries'] + ids, 'sha256': before['sha256']+'|'+ids[0]} if mutate else before
    tail = before['sha256']; decisions = []
    for i, tid in enumerate(ids):
        next_sha = after['sha256'] if i == 0 else tail
        decisions.append({'case_id': kw['case']['case_id'], 'leaf_id': tid,
                          'action': 'GENERATED_KEEP' if mutate and i == 0 else 'GENERATED_REMOVE',
                          'input_bank_sha256': tail, 'output_bank_sha256': next_sha})
        tail = next_sha
    report = {'trajectory_ids': ids, 'chunk_index': kw['chunk_offset'], 'generator_api_requests': 1,
              'generator_correction_requests': 0, 'peer_matching_calls': 0,
              'bank_before_sha256': before['sha256'], 'bank_after_sha256': after['sha256']}
    return {'working': after, 'canon': {}, 'assignments': [], 'witness': [], 'witness_objects': {},
            'selection': [], 'cumulative_reuse': [], 'decisions': decisions,
            'summary': {'chunks': [report], 'uncovered_ids': []},
            'order_next': {'success': kw['order_offsets']['success'] + len(ids), 'failure': 0}}


def test_starts_at_eight_before_rollout_end_and_flushes_tail(monkeypatch, tmp_path):
    started = Event(); release = Event(); calls = []
    def process(trainer, **kw):
        calls.append(kw)
        if len(calls) == 1:
            started.set(); assert release.wait(3)
        return built(kw)
    make, root, before = setup(monkeypatch, tmp_path, process)
    with make() as stream:
        for i in range(7): stream.push(*item(i))
        assert not started.wait(.05)
        stream.push(*item(7)); assert started.wait(3)
        assert not stream.progress['rollout_done']
        for i in range(8, 11): stream.push(*item(i))
        release.set(); result = stream.finish()
    assert [len(c['ruled']) for c in calls] == [8, 3]
    assert [c['chunk_offset'] for c in calls] == [0, 1]
    assert calls[1]['order_offsets']['success'] == 8
    assert result['summary']['eligible'] == 11
    assert S.audit(root, before)[0] == before
    for call in calls:
        assert {n.node_id for n in call['tree'].nodes} == {l.leaf_id for l in call['ruled']}


def test_resume_changed_completion_order_does_not_repeat_commit(monkeypatch, tmp_path):
    calls = []
    def process(trainer, **kw): calls.append(kw); return built(kw, mutate=True)
    make, root, before = setup(monkeypatch, tmp_path, process)
    with make() as stream:
        for i in range(11): stream.push(*item(i))
        first = stream.finish()
    with make() as stream:
        for i in reversed(range(11)): stream.push(*item(i))
        second = stream.finish()
    assert len(calls) == 2
    assert first['summary'] == second['summary']
    after, decisions = S.audit(root, before)
    assert len(after['entries']) == len(decisions) == 11
    assert len(S.read(Path(root)/'state.json')['transactions']) == 2


def test_two_cases_use_latest_bank_and_serialize_transactions(monkeypatch, tmp_path):
    calls = []; entered = Event(); release = Event()
    def process(trainer, **kw):
        calls.append(kw)
        if len(calls) == 1: entered.set(); assert release.wait(3)
        return built(kw, mutate=True)
    make, root, before = setup(monkeypatch, tmp_path, process)
    with make(0) as a, make(1) as b:
        for i in range(8): a.push(*item(i, 0))
        assert entered.wait(3)
        for i in range(8): b.push(*item(i, 1))
        assert len(calls) == 1
        release.set(); a.finish(); b.finish()
    assert calls[1]['snapshot']['sha256'] != before['sha256']
    assert len(S.audit(root, before)[1]) == 16


def test_receipt_written_before_state_commit_is_recovered(monkeypatch, tmp_path):
    calls = []
    def process(trainer, **kw): calls.append(kw); return built(kw, mutate=True)
    make, root, before = setup(monkeypatch, tmp_path, process)
    with make() as stream:
        for i in range(8): stream.push(*item(i))
        stream.finish()
    state = S.read(Path(root)/'state.json'); state.update(snapshot=before, transactions=[])
    S.atomic_write_json(Path(root)/'state.json', state)
    with make() as stream:
        for i in range(8): stream.push(*item(i))
        stream.finish()
    assert len(calls) == 1
    assert len(S.audit(root, before)[1]) == 8


def test_failed_chunk_propagates_without_commit(monkeypatch, tmp_path):
    def process(trainer, **kw): raise RuntimeError('judge failed')
    make, root, before = setup(monkeypatch, tmp_path, process)
    with pytest.raises(RuntimeError, match='judge failed'):
        with make() as stream:
            for i in range(8): stream.push(*item(i))
            stream.finish()
    assert S.read(Path(root)/'state.json')['transactions'] == []


def test_invalid_feedback_ignored_and_empty_case_is_valid(monkeypatch, tmp_path):
    def process(*args, **kw): raise AssertionError('No eligible trajectories')
    make, root, before = setup(monkeypatch, tmp_path, process)
    with make() as stream:
        leaf, fb, nodes = item(0)
        stream.push(leaf, DoctorFeedback(fb.leaf_id, 1.0, '', 'p', valid=False), nodes)
        result = stream.finish()
    assert result['summary']['eligible'] == 0 and result['working'] == before


def test_newly_committed_keep_rubric_is_scored_asynchronously(monkeypatch, tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from types import SimpleNamespace
    from src.online_rl.tree.records import Tree
    make, _root, _before = setup(monkeypatch, tmp_path, lambda **kw: built(kw))
    calls=[]
    monkeypatch.setattr(S.CB, 'scoring_rubrics', lambda _snapshot: [SimpleNamespace(rubric_id='new-rubric')])
    from src.online_rl.tree import scoring
    monkeypatch.setattr(scoring, 'score_nodes', lambda case, nodes, rubrics, scorer, cache, workers=1:
                        calls.append((case, nodes, rubrics, scorer, cache, workers)) or ['prefetched'])
    with make() as stream:
        stream.early_scoring=True
        stream.committed_score_executor=ThreadPoolExecutor(1)
        stream.trainer.env=SimpleNamespace(scorer='scorer')
        stream.trainer.cache='cache'
        leaf, _feedback, nodes=item(0)
        tree=Tree('c0','policy',4,4,tuple(nodes),(leaf,))
        stream.score_committed_delta(tree, {'entries':[]}, {'entries':[{'bank_id':'new-rubric'}]})
        assert stream.committed_score_futures[0][1].result()==['prefetched']
    assert len(calls)==1 and calls[0][2][0].rubric_id=='new-rubric'


def test_stream_config_identity_is_separate():
    from src.online_rl.tree.experiment import load_config
    from src.online_rl.tree.cumulative_run import config_identity
    root = Path(__file__).resolve().parents[1]
    cfg = load_config('configs/online_trajectory_stream.json')
    assert config_identity(cfg, root)['trajectory_streaming'] == S.VERSION
    assert cfg['agent_reward']['w_rubric'] == cfg['agent_reward']['w_outcome'] == .5
    cfg = load_config('configs/online_rubric_only_trajectory_stream.json')
    assert cfg['agent_reward']['w_rubric'] == 1 and cfg['agent_reward']['w_outcome'] == 0


def test_callback_has_exact_path_nodes_before_other_terminal_finishes():
    from concurrent.futures import ThreadPoolExecutor
    from src.online_rl.tree.rollout import build_tree
    from src.online_rl.tree.cache import CallCache
    from src.online_rl.integrated.records import to_dict
    observed = []; ready = Event(); release = Event(); slow = Event()
    class Doctor:
        status_protocol = True
        min_agent_turns = 2
        def turn(self, case, dialogue, round, rounds, seed):
            return {'visible': 'discussion', 'status': {'current_option': 'A'}, 'status_error': None, 'seed': seed}
        def halt(self, **kw): return 'doctor_halt' if kw['agent_turns'] >= 2 else None
        def final(self, case, dialogue, seed):
            if dialogue[-1]['text'] == 'second':
                slow.set(); assert release.wait(3)
            return 'A'
    class Agent:
        count = 0
        def sample(self, case, prefix, turn, seed):
            self.count += 1
            return {'completion_text': 'second' if self.count == 4 else 'first'}
    def callback(leaf, nodes): observed.append((leaf, nodes)); ready.set()
    with ThreadPoolExecutor(1) as executor:
        future = executor.submit(build_tree, {'case_id': 'c', 'options': {'A': 'a', 'B': 'b'}},
            max_agent_turns=2, branching=2, run_seed=0, policy_sha256='policy', protocol_sha256='protocol',
            agent=Agent(), doctor=Doctor(), stop_rule=lambda *args: None,
            final_option=lambda text: text, outcome_fn=lambda case, opt: 1., cache=CallCache(None),
            workers=2, on_leaf_nodes=callback)
        assert ready.wait(3) and slow.wait(3) and not future.done()
        release.set(); tree = future.result()
    nodes = {n.node_id: n for n in tree.nodes}; leaves = {l.leaf_id: l for l in tree.leaves}
    assert len(observed) == len(tree.leaves) == 4
    for leaf, path in observed:
        assert to_dict(leaf) == to_dict(leaves[leaf.leaf_id])
        assert [to_dict(n) for n in path] == [to_dict(nodes[i]) for i in leaf.path]


def test_receipt_round_trip_restores_typed_records():
    from src.online_rl.tree.records import TrajectoryRubric, RubricAssignment, SelectionDecision
    from src.online_rl.tree.witness import WitnessVerification
    rub = TrajectoryRubric('r', 'c/0', 'p', '{}', {'name': 'r'}, True, (), {'passed': True})
    assignment = RubricAssignment('c/0', 'c', 'success', 0, 'r', 'CREATED_NEW', None, '', 'f')
    choice = SelectionDecision('r', 'c/0', True, 'witness', 'hit')
    witness = WitnessVerification('r', 'success', 1, ('c/0',), ('c/0',), (), 1, 2, 'c/0', 'KEEP', 'witness_found')
    original = {'canon': {'r': rub}, 'assignments': [assignment], 'selection': [choice],
                'witness': [S.to_dict(witness)], 'witness_objects': {'r': witness}, 'working': {'sha256': 's'}}
    import json
    assert S.unpack(json.loads(json.dumps(S.pack(original)))) == original


def test_round_freezes_bank_after_streams_then_scores_every_case(monkeypatch, tmp_path):
    from src.online_rl.tree import cumulative_run as C
    from src.online_rl.tree.experiment import load_config
    cfg = load_config('configs/online_trajectory_stream.json')
    before = S.CB.empty_snapshot()
    monkeypatch.setattr(C, 'committed_snapshot', lambda *args: before)
    monkeypatch.setattr(C, 'publish_snapshot', lambda *args: None)
    phases = []
    def collect(phase, prompt):
        phases.append(phase)
        assert prompt['snapshot'] == before
        if phase == 'bank_stream':
            assert Path(prompt['stream_root']).is_dir()
            return [{'case_id': f'c{i}', 'policy_sha256': 'policy', 'cumulative_bank_input': before,
                     'offline_chunk_bank': {'selected_ids': []}, 'cumulative_reuse': []} for i in range(5)]
        assert phase == 'bank_score'
        assert (tmp_path/'batch'/'bank_frozen.json').exists()
        return [{'case_id': f'c{i}', 'policy_sha256': 'policy', 'cumulative_reuse': [],
                 'cumulative_bank_scoring': {'snapshot_sha256': before['sha256'], 'scoring_scope': 'all'}} for i in range(5)]
    rows, after = C.round_records(cfg, Path(__file__).resolve().parents[1], tmp_path,
        {'policy_sha256': 'policy', 'case_cursor': 0}, 1, tmp_path/'batch', collect)
    assert phases == ['bank_stream', 'bank_score'] and len(rows) == 5 and after == before
