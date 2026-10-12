"""Behavioral pipeline checks with deterministic models; no GPU or external API."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.online_rl.doctor import OllamaDoctor, split_private_state
from src.online_rl.integrated.continuation import derive_seed, protocol_sha256
from src.online_rl.integrated.records import sha256_json, to_dict
from src.online_rl.tree.cache import CallCache
from src.online_rl.tree.experiment import check_batch, load_config
from src.online_rl.tree.five_gpu import update_batch
from src.online_rl.tree.group_backends import make_callbacks
from src.online_rl.tree.group_learning import (GroupEnv, GroupTrainer, config_problems, learn_case,
    reference_groups, reward_request, reward_values, validate_candidate)
from src.online_rl.tree.rollout import build_tree

ROOT = Path(__file__).resolve().parents[1]
CASE = {'case_id': 'case', 'question': 'Which option?', 'options': {'A': 'alpha', 'B': 'beta'},
        'ground_truth': {'correct_option': 'A'}}
ST = {'current_option': 'A', 'open_issue': True, 'collaborator_objection': False,
      'more_info_could_change_decision': False}
PRIVATE = {'current_option': 'B', 'hypothesis': 'PRIVATE_BELIEF', 'confidence': 0.7}


class FakeAgent:
    def __init__(self):
        self.calls = []
        self.indices = {}
        for depth in range(1, 5):
            for fork in range(depth):
                for child in range(4):
                    path = '.'.join(['0'] * fork + [str(child)] + ['0'] * (depth - fork - 1))
                    self.indices[derive_seed(11, 'case/' + path, 'agent', 0)] = child if fork == depth - 1 else 0

    def sample(self, case, prefix, turn, seed):
        assert all('PRIVATE_BELIEF' not in row['text'] for row in prefix)
        self.calls.append((turn, seed, copy.deepcopy(prefix)))
        behavior = 'HARMFUL' if self.indices.get(seed, 0) in (1, 3) else 'HELPFUL'
        return {'completion_text': behavior + f' argument {seed}', 'completion_ids': [seed % 100 + 1],
                'prompt_input_ids': [turn, len(prefix)], 'state_prompt_hash': sha256_json(prefix)}


class FakeDoctor:
    status_protocol = True
    min_agent_turns = 2

    def __init__(self, help_results=(0, 0, 1), harm_results=(1, 1, 0), all_same=False):
        self.help_results, self.harm_results, self.all_same = help_results, harm_results, all_same
        self.continuations = {}
        self.finals = 0

    def turn(self, case, dialogue, round, rounds, seed):
        assert all('PRIVATE_BELIEF' not in row['text'] for row in dialogue)
        return {'visible': f'partner turn {round}', 'status': ST, 'status_error': None,
                'seed': seed, 'private_state': PRIVATE}

    def halt(self, **kwargs):
        return None

    def final(self, case, dialogue, seed):
        self.finals += 1
        injected = next((row for row in dialogue if row['text'].startswith('CONTRAST_')), None)
        if injected:
            key = (injected['text'], injected['round'])
            index = self.continuations.get(key, 0)
            self.continuations[key] = index + 1
            outcomes = self.help_results if injected['text'] == 'CONTRAST_TO_FAILURE' else self.harm_results
            return 'A' if outcomes[index % 3] else 'B'
        return 'B' if not self.all_same and any('HARMFUL' in row['text'] for row in dialogue) else 'A'


def envelope(parsed):
    return {'valid': True, 'parsed': parsed, 'raw': json.dumps(parsed)}


def fixtures(*, help_results=(0, 0, 1), harm_results=(1, 1, 0), all_same=False, no_candidates=False, flat=False):
    cfg = load_config('configs/group_online.json')
    cfg['schedule']['seed'] = 11
    # Existing fixtures exercise the supported historical two-candidate mode.
    cfg['group_learning']['max_rubrics_per_group'] = 2
    cfg['group_learning']['pipeline'] = None
    cfg['group_learning']['candidate_direction'] = None
    cfg['group_learning']['generation_format_retries'] = 0
    cfg['group_learning']['reward_mode'] = None
    calls = {'reflection': [], 'generation': [], 'judge': [], 'write': []}
    agent, doctor = FakeAgent(), FakeDoctor(help_results, harm_results, all_same)
    def reflect(case, group):
        calls['reflection'].append(copy.deepcopy(group))
        assert group['partner_state'] == PRIVATE
        return envelope({'helpful': [{'response_id': 'S0', 'quote': 'HELPFUL', 'experience': 'I could compare evidence.'}],
                         'harmful': [] if all_same else [{'response_id': 'S1', 'quote': 'HARMFUL', 'experience': 'I lost the key distinction.'}]})
    def generate(case, group, reflection):
        calls['generation'].append(group['group_id'])
        candidates = []
        for kind, source, quote, contrast in [('helpful', 'S0', 'HELPFUL', 'CONTRAST_TO_FAILURE'),
                                              ('harmful', 'S1', 'HARMFUL', 'CONTRAST_TO_SUCCESS')]:
            if kind == 'harmful' and all_same: continue
            candidates.append({'name': kind + ' evidence comparison', 'kind': kind, 'source_response_id': source,
                'state': {'partner': 'Uncertain between alternatives.', 'assistant': 'Has shared case and discussion.'},
                'criteria': {'0': 'Omits the decisive comparison.', '1': 'Gives a partial comparison.', '2': 'Compares decisive evidence.'},
                'b': 'observed ' + kind, 'b_star': contrast, 'evidence_quote': quote})
        return envelope({'candidates': [] if no_candidates else candidates})
    def judge(request):
        calls['judge'].append(copy.deepcopy(request))
        return envelope({'scores': [{'response_id': row['response_id'],
            'rubric_scores': [1 if flat else 0 if row['text'].startswith('HARMFUL') else 2] * len(request['rubrics'])}
            for row in request['responses']]})
    def write(case, group, candidate, seed):
        calls['write'].append((group['anchor'], candidate['kind'], seed))
        return candidate['b_star']
    env = GroupEnv(agent, doctor, lambda *args: None, lambda text: text,
                   lambda case, option: float(option == 'A') if option in ('A', 'B') else None,
                   reflect, generate, judge, write, behavior_logprobs=lambda traces: [[-0.1] for trace in traces])
    cache = CallCache(None)
    tree = build_tree(CASE, max_agent_turns=4, branching=4, run_seed=11, policy_sha256='policy', protocol_sha256='protocol',
                      agent=agent, doctor=doctor, stop_rule=env.stop_rule, final_option=env.final_option, outcome_fn=env.outcome_fn,
                      cache=cache, budget='reference_spine', behavior_logprobs=env.behavior_logprobs)
    return cfg, env, cache, tree, calls


def learned(**kwargs):
    cfg, env, cache, tree, calls = fixtures(**kwargs)
    result = learn_case(CASE, tree, env=env, cfg=cfg, cache=cache, run_seed=11, protocol_sha='protocol')
    return cfg, env, cache, tree, calls, result


def test_full_pipeline_costs_exact_threshold_and_anchor_rewards():
    cfg, env, cache, tree, calls, result = learned()
    assert len(tree.leaves) == 13
    assert len(calls['reflection']) == len(calls['generation']) == len(calls['judge']) == 4
    assert len(calls['write']) == 8
    assert cache.calls['group_verification_final'] == 24
    assert len(result['rubric_book']) == 8
    assert all(rubric['s'] == 2 / 3 for rubric in result['rubric_book'])
    groups = reference_groups(tree)
    for group, report in zip(groups, result['groups']):
        assert report['status'] == 'scored'
        assert len(report['kept_rubric_ids']) == 2
        assert all(item['group_id'] == group['group_id'] for item in report['candidates'])
        for item in report['candidates']:
            assert len(item['reruns']) == 3
            for rerun in item['reruns']:
                assert rerun['dialogue'][:len(group['prefix'])] == group['prefix']
                assert len([row for row in rerun['dialogue'] if row['text'].startswith('CONTRAST_')]) == 1
    targets = result['agent_targets']
    assert len(targets['groups']) == 4 and len(targets['nodes']) == 16
    assert all(row['outcome_component'] is None and row['reward'] in (0.0, 1.0) for row in targets['nodes'].values())
    metrics = result['group_metrics']
    assert metrics['rubric_group_fraction'] == 1.0 and metrics['zero_advantage_fraction'] == 0.0
    assert metrics['verification_pass_rate'] == 1.0 and metrics['verification_s'] == [2 / 3] * 8
    assert len(metrics['response_lengths']) == 16
    assert metrics['partner_initial_correct'] == 0.0 and metrics['rescue'] is True
    assert metrics['break'] is False
    record = {'case_id': CASE['case_id'], 'case_cursor': 0, 'policy_sha256': 'policy',
              'tree': to_dict(tree), 'prompt_sha256': cfg['rubric']['prompt_sha256'], **result}
    assert len(check_batch(cfg, [record])) == 16


def test_reference_member_uses_reference_outcome_not_descendant_average():
    cfg, env, cache, tree, calls = fixtures()
    group = reference_groups(tree)[0]
    reference_node = tree.leaves[0].path[0]
    below = [leaf.outcome for leaf in tree.leaves if reference_node in leaf.path]
    assert 0.0 in below and 1.0 in below
    assert group['members'][0]['outcome'] == 1.0
    assert group['members'][0]['leaf_id'] == tree.leaves[0].leaf_id


def test_equal_outcomes_still_reflect_and_generate_and_equal_rewards_have_zero_advantage():
    cfg, env, cache, tree, calls, result = learned(all_same=True, flat=True)
    assert {leaf.outcome for leaf in tree.leaves} == {1.0}
    assert len(calls['reflection']) == len(calls['generation']) == 4
    assert len(result['rubric_book']) == 4
    assert result['group_metrics']['zero_advantage_fraction'] == 1.0
    assert all(report['advantages'] == [0.0] * 4 for report in result['groups'])


def test_one_of_three_fails_verification_and_group_without_rubric_is_skipped():
    cfg, env, cache, tree, calls, result = learned(help_results=(0, 1, 1), harm_results=(1, 0, 0))
    assert len(calls['write']) == 8 and cache.calls['group_verification_final'] == 24
    assert result['rubric_book'] == [] and result['agent_targets']['groups'] == {}
    assert calls['judge'] == []
    assert result['group_metrics']['verification_s'] == [1 / 3] * 8
    assert result['group_metrics']['verification_pass_rate'] == 0.0


def test_no_candidates_still_calls_generator_once_per_group():
    cfg, env, cache, tree, calls, result = learned(no_candidates=True)
    assert len(calls['reflection']) == len(calls['generation']) == 4
    assert calls['judge'] == calls['write'] == []
    assert result['group_metrics']['groups_with_rubric'] == 0


def test_reward_is_blind_shuffled_joint_and_normalized():
    cfg, env, cache, tree, calls, result = learned()
    groups = reference_groups(tree)
    assert any([row['text'] for row in request['responses']] != [member['reply'] for member in group['members']]
               for group, request in zip(groups, calls['judge']))
    for group, request in zip(groups, calls['judge']):
        assert {row['text'] for row in request['responses']} == {member['reply'] for member in group['members']}
        assert set(request) == {'case', 'shared_prefix', 'rubrics', 'responses'}
        for rubric in request['rubrics']:
            assert set(rubric) == {'rubric_id', 'name', 'state', 'criteria'}
        serialized = json.dumps(request)
        assert 'source_response_id' not in serialized and 'outcome' not in serialized
        assert 'b_star' not in serialized and 'ground_truth' not in serialized and 'PRIVATE_BELIEF' not in serialized
    mapping = {f'A{i}': str(i) for i in range(4)}
    raw = {'scores': [{'response_id': f'A{i}', 'rubric_scores': [2, 1]} for i in range(4)]}
    assert {row['reward'] for row in reward_values(raw, mapping, 2).values()} == {0.75}
    raw['scores'][0]['rubric_scores'][0] = True
    with pytest.raises(ValueError, match='integer'): reward_values(raw, mapping, 2)


def test_replay_never_repeats_reflection_generation_injection_judge_or_continuation():
    cfg, env, cache, tree, calls, result = learned()
    before = copy.deepcopy(calls)
    finals = env.doctor.finals
    again = learn_case(CASE, tree, env=env, cfg=cfg, cache=cache, run_seed=11, protocol_sha='protocol')
    assert again == result and calls == before and env.doctor.finals == finals


def test_batch_update_only_receives_anchor_reply_tokens_and_empty_batch_is_safe():
    cfg, env, cache, tree, calls, result = learned()
    record = {'tree': to_dict(tree), 'policy_sha256': 'policy', **result}
    class Updater:
        steps = 0
        def update_groups(self, groups):
            self.groups = groups
            return {'optimizer_step': True}
    updater = Updater()
    assert update_batch(updater, [record, record], 'policy', lambda: 'policy')['optimizer_step']
    assert len(updater.groups) == 8
    assert all(len(group) == 4 for group in updater.groups)
    ids = set(result['agent_targets']['nodes'])
    for group in updater.groups:
        for row in group:
            assert row['trajectory_id'] in ids
            assert len(row['agent_traces']) == 1 and len(row['agent_traces'][0]['completion_ids']) == 1
    empty = {**record, 'agent_targets': {'nodes': {}, 'groups': {}}}
    assert update_batch(updater, [empty], 'policy', lambda: 'policy')['optimizer_step'] is False
    with pytest.raises(RuntimeError, match='Policy changed'):
        update_batch(updater, [record], 'policy', lambda: 'other')


def test_group_trainer_durable_resume_and_protocol_guards(tmp_path):
    cfg, env, cache, tree, calls = fixtures()
    cfg['output_dir'] = str(tmp_path)
    trainer = GroupTrainer(cfg, CASE, env, lambda: 'policy', 'protocol', {})
    result = trainer.step()
    assert (tmp_path / 'group_rollout.json').exists() and (tmp_path / 'group_metrics.json').exists()
    restored = GroupTrainer(cfg, CASE, env, lambda: 'policy', 'protocol', {})
    assert restored.state.case_cursor == 1
    restored.validate_member_interventions(result)
    before = copy.deepcopy(calls)
    finals = env.doctor.finals
    replay = restored.step()
    assert {key: value for key, value in replay.items() if key != 'timings'} == {
        key: value for key, value in result.items() if key != 'timings'}
    assert calls == before and env.doctor.finals == finals
    with pytest.raises(RuntimeError, match='protocol/config changed'):
        GroupTrainer(cfg, CASE, env, lambda: 'policy', 'other', {})


def test_partner_private_state_is_recorded_from_first_turn_but_never_visible():
    cfg = load_config('configs/group_online.json')
    ro = cfg['rollout']
    state = {**ST, 'open_issue': False}
    requests = []
    def http(url, payload, timeout):
        requests.append(payload)
        if payload.get('format'):
            opening = payload['format']['properties']['status']['type'] == 'null'
            content = json.dumps({'reply': 'Visible clinical reasoning.',
                                  'status': None if opening else state, 'private_state': PRIVATE})
        else:
            content = 'Visible clinical reasoning.\nSTATUS: ' + json.dumps(state) + '\nPRIVATE_STATE: ' + json.dumps(PRIVATE)
        return {'message': {'content': content}, 'prompt_eval_count': 100, 'eval_count': 80}
    doctor = OllamaDoctor(ro['doctor_backend'], ro['halting'], (ROOT / ro['doctor_system_prompt_file']).read_text(), ro['max_tokens_doctor'], http=http)
    opening = doctor.turn(CASE, [], 1, 4, 12)
    second = doctor.turn(CASE, [{'speaker': 'agent', 'text': 'Compare evidence.'}], 2, 4, 13)
    assert opening['private_state'] == second['private_state'] == PRIVATE
    assert opening['visible'] == second['visible'] == 'Visible clinical reasoning.'
    assert 'PRIVATE_BELIEF' not in opening['visible'] and 'STATUS' not in second['visible']
    assert doctor.halt(agent_turns=1, status=state, prev_status=state, last_agent='A', options=CASE['options']) is None
    with pytest.raises(ValueError, match='confidence'):
        split_private_state('PRIVATE_STATE: ' + json.dumps({**PRIVATE, 'confidence': True}), CASE['options'])


def test_private_choice_accepts_only_unambiguous_label_or_exact_option_text():
    options = {'A': 'Bone graft', 'B': 'Suture', 'C': 'Aificial tissue graft', 'D': 'Bronchoscope'}
    for reported in ('D', ' d ', 'Bronchoscope', ' bronchoscope '):
        raw = 'PRIVATE_STATE: ' + json.dumps({**PRIVATE, 'current_option': reported})
        assert split_private_state(raw, options)['current_option'] == 'D'
    for reported in ('Bronchoscope or Suture', 'option D', '', None):
        raw = 'PRIVATE_STATE: ' + json.dumps({**PRIVATE, 'current_option': reported})
        with pytest.raises(ValueError, match='current_option'):
            split_private_state(raw, options)
    ambiguous = {'A': 'Same answer', 'B': 'Same answer'}
    raw = 'PRIVATE_STATE: ' + json.dumps({**PRIVATE, 'current_option': 'Same answer'})
    with pytest.raises(ValueError, match='current_option'):
        split_private_state(raw, ambiguous)


def test_real_callback_request_boundaries_and_single_call_counts(monkeypatch):
    cfg = load_config('configs/group_online.json')
    _, env, cache, tree, calls = fixtures(all_same=True)
    group = reference_groups(tree)[0]
    requests = []
    doctor = SimpleNamespace(system='Partner role', model='partner', _chat=lambda user, seed, tokens, system, format_schema=None: (
        requests.append(('reflection', json.loads(user))) or json.dumps({'reflection': 'No meaningful differences.'})))
    policy = SimpleNamespace(doctor=doctor, agent_system='Agent role')
    def transport(key, model, system, user, settings, schema, name, role):
        requests.append((role, json.loads(user)))
        assert settings['max_tokens'] > 0
        candidate = {'name': 'Evidence comparison', 'source_response_id': group['members'][0]['response_id'],
            'state': {'partner': 'Uncertain between alternatives.', 'assistant': 'Has shared case and discussion.'},
            'criteria': {'observed': 'Compares decisive evidence.', 'middle': 'Gives partial comparison.',
                         'contrast': 'Omits the comparison.'},
            'b': 'Compare decisive evidence.', 'b_star': 'Omit the decisive comparison.', 'evidence_quote': 'HELPFUL'}
        return json.dumps({'candidates': [candidate]} if role == 'group_rubric_generator' else
                          {'ranking': [{'response_ids': [f'A{i}' for i in range(4)],
                                        'reason': 'No rubric-grounded differences.'}]}), {}
    monkeypatch.setattr('src.online_rl.tree.group_backends._generator_json', transport)
    reflect, generate, judge, write = make_callbacks(cfg, policy, 'dummy', 'dummy', ROOT)
    reflected = reflect(CASE, group)
    assert reflected['valid'] and generate(CASE, group, reflected['parsed'])['valid']
    assert 'gold_diagnosis' not in requests[0][1] and 'ground_truth' not in json.dumps(requests[0][1])
    assert requests[0][1]['partner_state'] == PRIVATE
    request = {'responses': [{'response_id': f'A{i}', 'text': 'reply'} for i in range(4)]}
    assert judge(request)['valid']
    assert len(requests) == 3


def test_group_config_costs_sampling_and_protocol_identity():
    cfg = load_config('configs/group_online.json')
    assert config_problems(cfg, ROOT) == []
    assert cfg['rollout']['halting']['min_agent_turns'] == 2
    assert cfg['rollout']['agent_protocol'] is None
    assert cfg['agent_reward']['w_outcome'] == 0 and cfg['agent_reward']['w_rubric'] == 1
    assert cfg['rubric']['cumulative_bank'] is None
    assert cfg['five_gpu']['batch_cases'] == 5
    assert cfg['optimizer'] == load_config('configs/online_openai20.json')['optimizer']
    before = protocol_sha256(cfg, ROOT)
    other_worker = copy.deepcopy(cfg)
    other_worker['rollout']['doctor_backend']['endpoint'] = 'http://127.0.0.1:12999'
    other_worker['rollout']['doctor_backend']['gpu'] = 'another runtime actor'
    assert protocol_sha256(other_worker, ROOT) == before
    changed = copy.deepcopy(cfg)
    changed['group_learning']['show_gold_to_reflection'] = True
    assert protocol_sha256(changed, ROOT) != before


def test_malformed_judge_excludes_the_whole_group_instead_of_partial_rewards():
    from dataclasses import replace
    cfg, env, cache, tree, calls = fixtures()
    broken = replace(env, judge=lambda request: envelope({'scores': [
        {'response_id': 'A0', 'rubric_scores': [2, 2]}]}))
    result = learn_case(CASE, tree, env=broken, cfg=cfg, cache=cache, run_seed=11, protocol_sha='protocol')
    assert result['agent_targets']['groups'] == {} and result['agent_targets']['nodes'] == {}
    assert len(result['rubric_book']) == 8
    assert all(report['status'] == 'reward_unresolved' for report in result['groups'])


def test_contrast_with_unparsed_outcome_is_unresolved_and_never_a_numeric_score():
    from dataclasses import replace
    cfg, env, cache, tree, calls = fixtures()
    unresolved = replace(env, outcome_fn=lambda case, option: None)
    result = learn_case(CASE, tree, env=unresolved, cfg=cfg, cache=cache, run_seed=11, protocol_sha='protocol')
    assert not result['rubric_book'] and not result['agent_targets']['groups']
    assert all(item['s'] is None and item['status'] == 'verification_unresolved'
               for report in result['groups'] for item in report['candidates'])


def test_group_monitor_reports_learning_metrics_and_excludes_reruns_from_rollout(tmp_path):
    import monitor
    cfg, env, cache, tree, calls, result = learned()
    directory = tmp_path / 'batches' / 'batch_000001' / 'case_000000'
    directory.mkdir(parents=True)
    (directory / 'group_rollout.json').write_text(json.dumps({'counts': {'leaves': 13}}))
    (directory / 'group_progress.json').write_text(json.dumps({'groups': result['groups']}))
    (directory / 'group_metrics.json').write_text(json.dumps(result['group_metrics']))
    (directory / 'calls.jsonl').write_text('\n'.join(json.dumps(row) for row in cache.store.values()) + '\n')
    state = monitor.snapshot(tmp_path, total=5, batch_size=5, group_mode=True)
    first = state['rollout_cases'][0]
    assert first['rollout_done'] and first['trajectories'] == 13 and first['reflections'] == 4
    text = monitor.render(state)
    assert 'SHERPA GROUP' in text and 'verification 통과' in text and 'zero advantage' in text
    assert 'rubric 있는 그룹 4/4' in text and 'Bank 대기' not in text


def test_case_doctor_rebinding_keeps_shared_openings_and_isolated_rpc_context(tmp_path, monkeypatch):
    from src.online_rl.tree.five_gpu import cache_openings
    from src.online_rl.tree.gpu_pool import case_policy_view, PoolClient
    cfg = load_config('configs/group_online.json')
    ro = cfg['rollout']
    doctor = OllamaDoctor(ro['doctor_backend'], ro['halting'], (ROOT / ro['doctor_system_prompt_file']).read_text(), ro['max_tokens_doctor'])
    seen = []
    raw = 'Visible.\nSTATUS: ' + json.dumps(ST) + '\nPRIVATE_STATE: ' + json.dumps(PRIVATE)
    def rpc(self, op, *args):
        seen.append((dict(self.context), op))
        return raw
    monkeypatch.setattr(PoolClient, 'call', rpc)
    parent = SimpleNamespace(spec={}, expected='policy', source_gpu=0, timeout=1, context={'case_id': 'parent'})
    policy = SimpleNamespace(model=object(), doctor=doctor, pool_client=parent)
    cache_openings(doctor, tmp_path / 'openings.jsonl', {'protocol': 'private_state'})
    first = case_policy_view(policy, {'case_id': 'first'})
    second = case_policy_view(policy, {'case_id': 'second'})
    opening = first.doctor.turn(CASE, [], 1, 4, 0)
    second.doctor.turn(CASE, [], 1, 4, 0)
    first.doctor.turn(CASE, [], 2, 4, 1)
    second.doctor.turn(CASE, [], 2, 4, 2)
    assert opening['private_state'] == PRIVATE
    assert [context['case_id'] for context, op in seen] == ['first', 'first', 'second']
    assert first.doctor._shared_opening_cache is second.doctor._shared_opening_cache
    assert parent.context == {'case_id': 'parent'}


def test_agent_final_answer_text_alone_does_not_end_rollout():
    from dataclasses import replace
    from src.collaboration.generator import stop_reason
    cfg, env, cache, tree, calls = fixtures()
    class FinalAgent(FakeAgent):
        def sample(self, case, prefix, turn, seed):
            result = super().sample(case, prefix, turn, seed)
            result['completion_text'] = 'Final answer: option A.'
            return result
    assert stop_reason(2, 'Consider the evidence.', 'Final answer: option A.') is None
    agent = FinalAgent()
    rebuilt = build_tree(CASE, max_agent_turns=4, branching=4, run_seed=11,
        policy_sha256='policy', protocol_sha256='without-agent-final', agent=agent, doctor=env.doctor,
        stop_rule=stop_reason, final_option=env.final_option, outcome_fn=env.outcome_fn,
        cache=CallCache(None), budget='reference_spine')
    assert len(rebuilt.leaves) == 13 and all(len(leaf.path) == 4 for leaf in rebuilt.leaves)


def test_three_verification_reruns_overlap_and_preserve_seed_order():
    from threading import Barrier, Lock
    cfg, env, cache, tree, calls = fixtures()
    assert cfg['execution']['rerun_workers'] == 3
    assert cfg['five_gpu']['batch_cases'] == 5 and cfg['optimizer']['learning_rate'] == 1e-5
    barrier, lock = Barrier(3), Lock()
    activity = {'active': 0, 'peak': 0}
    original = env.doctor.final
    def final(case, dialogue, seed):
        with lock:
            activity['active'] += 1
            activity['peak'] = max(activity['peak'], activity['active'])
        try:
            barrier.wait(timeout=5)
            with lock:
                return original(case, dialogue, seed)
        finally:
            with lock:
                activity['active'] -= 1
    env.doctor.final = final
    result = learn_case(CASE, tree, env=env, cfg=cfg, cache=cache, run_seed=11, protocol_sha='protocol')
    assert activity['peak'] == 3
    assert cache.calls['group_verification_final'] == 24
    for report in result['groups']:
        for candidate in report['candidates']:
            expected = [derive_seed(11, candidate['rubric_id'], 'contrast_continuation', index) for index in range(3)]
            assert [rerun['seed'] for rerun in candidate['reruns']] == expected
            assert candidate['s'] == 2 / 3 and candidate['retained']
