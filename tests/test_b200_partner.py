"""No network, downloaded models or CUDA: real orchestration with fake model calls."""
import copy
from dataclasses import replace
import itertools
import json
from types import SimpleNamespace
import pytest

from src.online_rl.tree import partner_verification as V
from src.online_rl.tree.cache import CallCache
from src.online_rl.tree.records import AgentNode, Leaf, TrajectoryRubric, RubricAssignment, to_turns
from src.online_rl.tree.member_intervention import source_member_intervention
from src.online_rl.tree.b200_runner import load_config, preflight
from src.online_rl.runtime import SharedPolicy
from test_agent_precommit import fake_policy

CASE = {'case_id': 'b200_case', 'question': 'Which option?', 'scenario': '',
        'options': {'A': 'alpha', 'B': 'beta'}, 'ground_truth': {'correct_option': 'B'}}


def fixture(anchor=2, outcome=0):
    dialogue, nodes, path = [], {}, []
    for t in range(1, 5):
        dialogue.append({'speaker': 'doctor', 'stage': f'D{t}', 'round': t, 'text': f'Consider evidence {t}.'})
        nid = f'b200_case/{t}'
        nodes[nid] = AgentNode(nid, path[-1] if path else None, CASE['case_id'], t, to_turns(dialogue),
            f'Original reply {t}.', {}, 'policy', 0, None, False, None,
            prefix_doctor_status={'current_option': 'A', 'x': t})
        path.append(nid)
        dialogue.append({'speaker': 'agent', 'stage': f'A{t}', 'round': t, 'text': f'Original reply {t}.'})
    dialogue.append({'speaker': 'doctor', 'stage': 'FINAL', 'round': 0, 'text': 'Final answer: option A'})
    leaf = Leaf(path[-1], CASE['case_id'], tuple(path), to_turns(dialogue), 'Final answer: option A', 'A', outcome, 'max_agent_turns')
    rubric = TrajectoryRubric('R', leaf.leaf_id, 'prompt', '{}',
        {'rubric_id': 'R', 'source_provenance': {'analysis': {'selected_agent_turn_id': anchor}}},
        True, (), {'passed': True}, {'status': 'ok'}, 'Compare the relevant alternatives.')
    spec = source_member_intervention(rubric, leaf)
    assignment = RubricAssignment(leaf.leaf_id, CASE['case_id'], 'success' if outcome else 'failure', 0,
        'R', 'CREATED_NEW', None, '', 'feedback', spec)
    return nodes, leaf, rubric, assignment


class Doctor:
    min_agent_turns = 2
    def __init__(self, halt_at=None, final='Final answer: option B'):
        self.halt_at, self.answer, self.calls = halt_at, final, []
    def turn(self, case, dialogue, turn, rounds, seed):
        assert not any('guidance' in d['text'].lower() for d in dialogue)
        self.calls.append(('turn', turn, copy.deepcopy(dialogue), seed))
        return {'visible': f'Further evidence {turn}.', 'status': {'current_option': 'A', 'x': turn}, 'seed': seed, 'status_error': None}
    def halt(self, **kwargs):
        return 'doctor_halt' if self.halt_at and kwargs['agent_turns'] >= self.halt_at else None
    def final(self, case, dialogue, seed):
        self.calls.append(('final', 0, copy.deepcopy(dialogue), seed))
        return self.answer


def backend(doctor=None):
    p = fake_policy()
    p.args.rounds = 4
    p.doctor = doctor or Doctor()
    p.agent_turn = lambda *a, **k: SharedPolicy.agent_turn(p, *a, **k)
    return V.LinearInterventions(p, load_config())


@pytest.mark.parametrize('anchor', [1, 2, 3, 4])
def test_linear_continuation_keeps_prefix_total_turns_and_normal_protocol(anchor):
    nodes, leaf, rubric, assignment = fixture(anchor)
    b = backend(); node = nodes[leaf.path[anchor - 1]]
    plan = V.make_plan(CASE, node, leaf, rubric, assignment.intervention_spec, 0, 'policy', b.cfg)
    result = b.trial(CASE, node, plan, 0, CallCache(None))
    assert result['agent_turns_total'] == 4
    assert result['generated_agent_turns'] == 5 - anchor
    assert result['dialogue'][:len(node.prefix)] == V.to_dialogue(node.prefix)
    assert result['termination_reason'] == 'max_agent_turns'
    assert len(b.policy.calls) == 2 * (5 - anchor)
    for index, row in enumerate(result['agent_traces']):
        tr = row['trace']
        assert [s['role'] for s in tr['segments']] == ['private_assessment', 'visible_reply']
        assert 'Compare the relevant alternatives.' not in tr['segments'][0]['prompt_text']
        assert ('Compare the relevant alternatives.' in tr['segments'][1]['prompt_text']) == (index == 0)
    assert 'guidance' not in b.policy.agent_system.lower()
    b.policy.agent_turn(CASE, V.to_dialogue(node.prefix), anchor, 77)
    assert 'Verification guidance' not in b.policy.calls[-1][0]
    assert all('Original reply 4' not in d['text'] for d in result['dialogue'])


def test_doctor_halt_ends_before_turn_limit():
    nodes, leaf, rubric, a = fixture(1)
    b = backend(Doctor(halt_at=2)); node = nodes[leaf.path[0]]
    plan = V.make_plan(CASE, node, leaf, rubric, a.intervention_spec, 0, 'policy', b.cfg)
    result = b.trial(CASE, node, plan, 0, CallCache(None))
    assert result['agent_turns_total'] == 2 and result['termination_reason'] == 'doctor_halt'
    assert [c[1] for c in b.policy.doctor.calls if c[0] == 'turn'] == [2, 3]


def test_min_turns_prevents_early_agent_stop(monkeypatch):
    import src.collaboration.generator as G
    monkeypatch.setattr(G, 'stop_reason', lambda *a: 'agent_final_answer')
    nodes, leaf, rubric, a = fixture(1); b = backend(); node = nodes[leaf.path[0]]
    plan = V.make_plan(CASE, node, leaf, rubric, a.intervention_spec, 0, 'policy', b.cfg)
    result = b.trial(CASE, node, plan, 0, CallCache(None))
    assert result['agent_turns_total'] == 2 and result['termination_reason'] == 'agent_final_answer'


class Outcomes:
    def __init__(self, values):
        self.cfg = load_config(); self.values = values; self.calls = []
    def trial(self, case, node, plan, index, cache):
        self.calls.append((plan['source_leaf_id'], index, plan['trials'][index]['seed']))
        if isinstance(self.values[index], Exception): raise self.values[index]
        return {'trial_index': index, 'outcome': self.values[index], 'seeds': plan['trials'][index]}


def verify(b, cache=None, source=0, members=1):
    nodes, leaf, rubric, a = fixture(outcome=source)
    leaves = {leaf.leaf_id: leaf}; assignments = [a]
    for i in range(1, members):
        other = replace(leaf, leaf_id=f'other-{i}')
        leaves[other.leaf_id] = other
        assignments.append(replace(a, leaf_id=other.leaf_id, assignment='REUSED_EXISTING'))
    return V.verify_fixed_source(CASE, nodes, leaves, {'R': rubric}, assignments,
        backend=b, run_seed=0, policy_sha256='policy', cache=cache or CallCache(None))['R']


@pytest.mark.parametrize('source', [0, 1])
@pytest.mark.parametrize('flips', list(itertools.product([False, True], repeat=3)))
def test_all_majorities_match_full_three_without_imputing_third(source, flips):
    b = Outcomes([1 - source if flip else source for flip in flips])
    result = verify(b, source=source, members=5)
    assert result.decision == ('KEEP' if sum(flips) >= 2 else 'DROP')
    expected = 2 if flips[0] == flips[1] else 3
    assert len(b.calls) == result.num_reruns_executed == expected
    assert len(result.tested) == 1 and len(result.tested[0]['rerun_records']) == expected
    assert {c[0] for c in b.calls} == {'b200_case/4'}
    assert len({c[2] for c in b.calls}) == expected
    assert result.tested[0]['plan']['level'] == ('0' if source else '2')


@pytest.mark.parametrize('values,decision', [([None, None, None], 'INCOMPLETE'), ([None, 1, 1], 'KEEP'), ([None, 0, 0], 'DROP')])
def test_unparsed_final_never_means_wrong_answer(values, decision):
    b = Outcomes(values); result = verify(b)
    assert result.decision == decision and len(b.calls) == 3
    assert result.tested[0]['error_count'] == values.count(None)


def test_cache_resume_and_rubric_reuse_keep_source_seed_and_completed_trials(tmp_path):
    path = tmp_path / 'calls.jsonl'
    b = Outcomes([1, 0, RuntimeError('transient')])
    with pytest.raises(RuntimeError): verify(b, CallCache(path), members=4)
    saved = CallCache(path)
    assert len([r for r in saved.store.values() if r['kind'] == 'fixed_verification_trial']) == 2
    plans = [r['value'] for r in saved.store.values() if r['kind'] == 'fixed_verification_plan']
    assert len(plans) == 1 and len(plans[0]['trials']) == 3
    b2 = Outcomes([RuntimeError('must cache'), RuntimeError('must cache'), 1])
    result = verify(b2, saved, members=4)
    assert result.decision == 'KEEP' and [r[1] for r in b2.calls] == [2]
    assert result.tested[0]['plan'] == plans[0]
    assert verify(b2, CallCache(path), members=4).decision == 'KEEP'
    assert len(b2.calls) == 1


def test_per_turn_resume_skips_completed_guided_action(tmp_path):
    nodes, leaf, rubric, a = fixture(2); b = backend(); node = nodes[leaf.path[1]]
    plan = V.make_plan(CASE, node, leaf, rubric, a.intervention_spec, 0, 'policy', b.cfg)
    original = b.policy.doctor.turn
    b.policy.doctor.turn = lambda *a: (_ for _ in ()).throw(RuntimeError('transport'))
    with pytest.raises(RuntimeError): b.trial(CASE, node, plan, 0, CallCache(tmp_path / 'calls.jsonl'))
    assert len(b.policy.calls) == 2
    b.policy.doctor.turn = original
    b.trial(CASE, node, plan, 0, CallCache(tmp_path / 'calls.jsonl'))
    assert len(b.policy.calls) == 6  # A2 cached, only A3/A4 add two segments each.


def test_cpu_preflight_and_preserved_common_values(monkeypatch):
    monkeypatch.setattr('src.online_rl.tree.b200_runner.physical_devices', lambda: ['0', '1'])
    cfg = load_config()
    # Data files are private and absent in a clean clone; only data absence is permitted.
    assert all(e.startswith('Missing file:') for e in preflight(cfg, resources=False))
    from src.online_rl.tree.b200_runner import ROOT
    old = json.loads((ROOT / 'configs/experiments/train120_max2_fixed_p0.json').read_text())
    for key in ('optimizer', 'advantage_normalization', 'policy'):
        assert cfg[key] == old[key]
    for key in ('temperature', 'top_p', 'top_k', 'agent_protocol', 'agent_sampling_protocol', 'halting'):
        assert cfg['rollout'][key] == old['rollout'][key]
    assert cfg['data'] == old['data']
    assert cfg['rubric']['generator_model'] == cfg['rubric']['consolidate']['model'] == 'openai/gpt-6-luna'
    assert cfg['rubric']['generation']['reasoning_effort'] == cfg['rubric']['consolidate']['reasoning_effort'] == 'none'
    expected_generation = {**old['rubric']['generation'], 'reasoning_effort': 'none', 'strict_provider_parameters': True, 'temperature': None}
    assert cfg['rubric']['generation'] == expected_generation
    for k in ('gate_model', 'scorer_model', 'scorer_protocol'):
        assert cfg['judges'][k] == old['judges'][k]
    assert cfg['prompt_update']['enabled'] is False
