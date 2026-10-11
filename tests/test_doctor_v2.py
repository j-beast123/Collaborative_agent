"""Doctor protocol v2 (src/online_rl/doctor.py): local Ollama Doctor, hidden status, adaptive halting, seeds, cache and
protocol identity. A scripted fake Ollama server only; no network."""
import json, re, urllib.error
from pathlib import Path
import pytest
from src.online_rl import doctor as D
from src.online_rl.tree.cache import CallCache
from src.online_rl.tree.rollout import build_tree
from src.collaboration.generator import final_answer_option, final_decision_prompt, stop_reason

ROOT = Path(__file__).resolve().parents[1]
CASE = {'case_id': 'c1', 'question': 'Which nerve supplies the deltoid?', 'scenario': '', 'options': {k: f'nerve {k}' for k in 'ABCD'},
        'ground_truth': {'correct_option': 'B'}}
DIGEST = 'a' * 64
BACKEND = {'engine': 'ollama', 'endpoint': 'http://127.0.0.1:11435', 'model': 'gemma3:12b-it-q8_0', 'model_digest': DIGEST,
           'quantization': 'Q8_0', 'ollama_version': '0.35.1', 'source_model': 'google/gemma-3-12b-it',
           'options': {'temperature': 0.4, 'top_p': 1.0, 'top_k': 0, 'min_p': 0.0, 'repeat_penalty': 1.0, 'presence_penalty': 0.0,
                       'frequency_penalty': 0.0, 'num_ctx': 8192}}
HALT = {'enabled': True, 'min_agent_turns': 2}
ST = lambda opt='B', open_=False, obj=False, more=False: {'current_option': opt, 'open_issue': open_, 'collaborator_objection': obj,
                                                          'more_info_could_change_decision': more}


class FakeOllama:
    """/api/chat: visible reply + a STATUS line chosen by status_for(round, user, seed); FINAL -> 'Final answer: option B'."""
    def __init__(self, status_for=lambda r, user, seed: ST(), digest=DIGEST, version='0.35.1', vram_full=True, fail=0):
        self.status_for, self.digest, self.version, self.vram_full, self.fail = status_for, digest, version, vram_full, fail
        self.chats = []

    def __call__(self, url, payload, timeout):
        if url.endswith('/api/version'): return {'version': self.version}
        if url.endswith('/api/tags'): return {'models': [{'name': BACKEND['model'], 'digest': self.digest}]}
        if url.endswith('/api/ps'): return {'models': [{'name': BACKEND['model'], 'size': 100, 'size_vram': 100 if self.vram_full else 60}]}
        if self.fail: self.fail -= 1; raise urllib.error.URLError('connection refused')
        assert len(payload['messages']) == 1 and payload['messages'][0]['role'] == 'user'
        user = payload['messages'][0]['content'].split('\n\n', 1)[1]; seed = payload['options']['seed']; self.chats.append((user, payload))
        if 'This is the final decision' in user: return {'message': {'content': 'Final answer: option B - nerve B'}, 'prompt_eval_count': 50, 'eval_count': 9}
        r = int(re.search(r'discussion turn (\d+) of', user).group(1))
        text = f'Doctor view at turn {r}: the axillary course matters most here.'
        st = self.status_for(r, user, seed)
        if st is not None and 'STATUS:' in user: text += '\nSTATUS: ' + (st if isinstance(st, str) else json.dumps(st))
        return {'message': {'content': text}, 'prompt_eval_count': 300, 'eval_count': 40}


def doctor(fake=None, halting=HALT, **kw):
    return D.OllamaDoctor({**BACKEND, **kw}, halting, 'SYSTEM', 512, http=fake or FakeOllama(), sleep=lambda s: None)


def test_short_final_keeps_complete_context_and_caps_only_final():
    fake=FakeOllama()
    doc=doctor(fake,final_response={'mode':'label_only_v1','max_tokens':32})
    discussion=[{'speaker':'doctor','stage':'D1','text':'Initial view'},
                {'speaker':'agent','stage':'A1','text':'New argument for option B'}]
    doc.turn(CASE,discussion,2,5,123)
    doc.final(CASE,discussion,456)
    doc.turn(CASE,discussion,3,5,789)
    assert [p['options']['num_predict'] for _,p in fake.chats]==[512,32,512]
    final_user=fake.chats[1][0]
    assert final_user==final_decision_prompt(CASE,discussion,label_only=True)
    assert 'New argument for option B' in final_user and CASE['question'] in final_user
    assert 'Output exactly one line' in final_user
    assert '<answer text>' not in final_user
    assert final_answer_option('Final answer: option B')=='B'
    assert doc.options['num_predict']==512
    assert doctor().protocol!=doc.protocol
    assert doctor(final_response={'mode':'label_only_v1','max_tokens':48}).protocol!=doc.protocol


@pytest.mark.parametrize('settings',[{'mode':'unknown','max_tokens':32},
                                   {'mode':'label_only_v1','max_tokens':8},
                                   {'mode':'label_only_v1','max_tokens':True}])
def test_invalid_short_final_settings_fail_before_calls(settings):
    with pytest.raises(ValueError,match='final_response'):doctor(final_response=settings)


# ---------- status parsing and the halt rule ----------
def test_status_is_cut_from_the_visible_reply_and_validated():
    vis, st, err = D.split_status('I favour B because of the course.\nSTATUS: ' + json.dumps(ST()), 'ABCD')
    assert vis == 'I favour B because of the course.' and st == ST() and err is None
    for bad in ('STATUS: {"current_option": "B"}', 'STATUS: not json', 'STATUS: ' + json.dumps({**ST(), 'open_issue': 'maybe'}),
                'STATUS: ' + json.dumps(ST(opt='E')), '**STATUS:** ' + json.dumps({**ST(), 'extra': 1})):
        vis, st, err = D.split_status('Reply text.\n' + bad, 'ABCD')
        assert st is None and err and vis == 'Reply text.' and 'STATUS' not in vis       # malformed: no status, still cut
    assert D.split_status('No status here.', 'ABCD')[1:] == (None, 'no status line')


def halt(agent_turns=2, status=ST(), prev=ST(), last='I agree that B fits best.'):
    return D.halt_reason(agent_turns=agent_turns, status=status, prev_status=prev, last_agent=last, options='ABCD', min_agent_turns=2)


def test_never_halt_before_two_agent_turns():
    assert halt(agent_turns=1) is None and halt(agent_turns=0) is None and halt(agent_turns=2) == 'doctor_halt'


def test_stable_settled_status_halts():
    assert halt() == 'doctor_halt' and halt(last='The axillary nerve course supports this.') == 'doctor_halt'


def test_a1_can_halt_only_when_fully_settled():
    settled = ST()
    assert D.halt_reason(agent_turns=1, status=settled, prev_status=None,
                         last_agent='I agree that B fits best.', options='ABCD',
                         min_agent_turns=1) == 'doctor_halt_a1_settled'
    assert D.halt_reason(agent_turns=1, status={**settled, 'open_issue': True},
                         prev_status=None, last_agent='I agree that B fits best.',
                         options='ABCD', min_agent_turns=1) is None
    assert D.halt_reason(agent_turns=1, status=settled, prev_status=None,
                         last_agent='I still favor option C.', options='ABCD',
                         min_agent_turns=1) is None


def test_a2_defaults_to_final_unless_every_extension_signal_is_active():
    base = dict(agent_turns=2, prev_status=ST(), last_agent='I still favor option C.',
                options='ABCD', min_agent_turns=1, default_final_after_agent_turns=2)
    for flag in D.FLAGS:
        status = {**ST(), **{key: True for key in D.FLAGS}, flag: False}
        assert D.halt_reason(status=status, **base) == 'doctor_halt_default_after_a2'
    all_active = {**ST(), **{key: True for key in D.FLAGS}}
    assert D.halt_reason(status=all_active, **base) is None


def test_a1_can_use_the_same_default_final_rule_as_a2():
    base = dict(agent_turns=1, prev_status=None, last_agent='I still favor option C.',
                options='ABCD', min_agent_turns=1, default_final_after_agent_turns=1)
    for flag in D.FLAGS:
        status = {**ST(), **{key: True for key in D.FLAGS}, flag: False}
        assert D.halt_reason(status=status, **base) == 'doctor_halt_default_after_a1'
    all_active = {**ST(), **{key: True for key in D.FLAGS}}
    assert D.halt_reason(status=all_active, **base) is None


@pytest.mark.parametrize('kw', [dict(prev=ST(opt='C')), dict(prev=None), dict(status=None), dict(status=ST(opt='none'), prev=ST(opt='none'))])
def test_unstable_or_missing_decision_continues(kw):
    assert halt(**kw) is None


def test_agent_advocating_another_option_continues():
    assert halt(last='I still think option C is better supported.') is None


@pytest.mark.parametrize('flag', ['open_issue', 'collaborator_objection', 'more_info_could_change_decision'])
def test_open_issue_objection_or_uncertainty_continues(flag):
    assert halt(status={**ST(), flag: True}) is None


@pytest.mark.parametrize('flag', ['open_issue','more_info_could_change_decision'])
def test_late_stable_choice_relaxes_only_issue_and_more_info_from_four(flag):
    doc=doctor(halting={**HALT,'stable_choice_no_objection_after_turns':4})
    kw={'status':{**ST(),flag:True},'prev_status':ST(),'last_agent':'I favor option B.','options':'ABCD'}
    assert doc.halt(agent_turns=2,**kw) is None
    assert doc.halt(agent_turns=3,**kw) is None
    assert doc.halt(agent_turns=4,**kw)=='doctor_halt_stable_choice'
    assert doc.halt(agent_turns=5,**kw)=='doctor_halt_stable_choice'
    assert doctor().halt(agent_turns=4,**kw) is None
    assert doc.protocol!=doctor().protocol


@pytest.mark.parametrize('change',[{'status':None},{'prev_status':None},
                                  {'prev_status':ST(opt='C')},
                                  {'status':ST(opt='none'),'prev_status':ST(opt='none')},
                                  {'status':{**ST(),'collaborator_objection':True}},
                                  {'last_agent':'I prefer option C.'}])
def test_late_halt_keeps_validity_stability_and_objection_guards(change):
    doc=doctor(halting={**HALT,'stable_choice_no_objection_after_turns':4})
    kw={'status':{**ST(),'open_issue':True},'prev_status':ST(),'last_agent':'I favor option B.','options':'ABCD'}
    assert doc.halt(agent_turns=4,**{**kw,**change}) is None


@pytest.mark.parametrize('late',[True,1,'4',4.0])
def test_late_halt_rejects_invalid_configuration(late):
    with pytest.raises(ValueError,match='stable_choice_no_objection_after_turns'):
        doctor(halting={**HALT,'stable_choice_no_objection_after_turns':late})


# ---------- one episode / continuation / tree ----------
class Args:
    rounds, max_completion_length, generation_temperature, top_p, top_k = 5, 50, 0.7, 0.9, 20


def episode(monkeypatch, fake, halting=HALT, agent_text='I agree the axillary nerve fits.'):
    from src.online_rl import environment as E
    monkeypatch.setattr(E, 'chat_prompt', lambda tok, system, user: user)
    monkeypatch.setattr(E, 'generate_with_trace', lambda *a, **k: {'completion_text': agent_text})
    return E.generate_trajectory(None, None, CASE, 'unused', 'AGENT', 'unused', Args, doctor=doctor(fake, halting), seed=11)


def test_episode_halts_after_two_agent_turns_and_finals_normally(monkeypatch):
    f = FakeOllama(); t = episode(monkeypatch, f)
    assert t['termination_reason'] == 'doctor_halt' and t['num_discussion_rounds'] == 2 and t['final_option'] == 'B'
    assert [d['stage'] for d in t['dialogue']] == ['D1', 'A1', 'D2', 'A2', 'D3', 'FINAL']
    final_user = f.chats[-1][0]
    assert final_user == final_decision_prompt(CASE, t['dialogue'][:-1]) and 'STATUS' not in final_user        # FINAL unchanged
    assert not any('STATUS' in d['text'] or 'open_issue' in d['text'] for d in t['dialogue'])
    assert all('STATUS' not in u.split('Conversation so far:')[1].split('Respond')[0] for u, _ in f.chats)       # never fed back


@pytest.mark.parametrize('reply', ['I agree the axillary nerve fits.', 'Final answer: option B'])
def test_three_turn_minimum_in_evaluation_including_final_answer_bypass(monkeypatch, reply):
    t = episode(monkeypatch, FakeOllama(), halting={'enabled': True, 'min_agent_turns': 3}, agent_text=reply)
    assert t['num_discussion_rounds'] == 3
    assert t['termination_reason'] == ('agent_final_answer' if 'Final answer' in reply else 'doctor_halt')


def test_halting_off_requests_the_same_status_but_runs_to_t_max(monkeypatch):
    on, off = FakeOllama(), FakeOllama()
    a, b = episode(monkeypatch, on), episode(monkeypatch, off, halting={'enabled': False, 'min_agent_turns': 2})
    assert a['termination_reason'] == 'doctor_halt' and b['termination_reason'] == 'max_agent_turns' and b['num_discussion_rounds'] == 5
    assert [c[0] for c in on.chats[:-1]] == [c[0] for c in off.chats[:len(on.chats) - 1]]                     # identical prefix (paired)
    assert [c[1]['options']['seed'] for c in on.chats[:-1]] == [c[1]['options']['seed'] for c in off.chats[:len(on.chats) - 1]]


def test_late_rule_applies_to_tree_evaluation_and_intervention_continuation(monkeypatch):
    from types import SimpleNamespace
    from src.online_rl import environment
    from src.prompt_opt.harness import Env
    settings={**HALT,'stable_choice_no_objection_after_turns':4}
    fake=lambda:FakeOllama(status_for=lambda r,u,s:ST(more=True))
    result=episode(monkeypatch,fake(),halting=settings)
    assert result['num_discussion_rounds']==4 and result['termination_reason']=='doctor_halt_stable_choice'
    t,_=tree(fake(),halting=settings,branching=1)
    assert [len(l.path) for l in t.leaves]==[4]
    assert t.leaves[0].stop_reason=='doctor_halt_stable_choice'
    policy=SimpleNamespace(torch=SimpleNamespace(manual_seed=lambda s:None),args=Args,
                           doctor=doctor(fake(),settings),agent_system='AGENT',ST=environment,
                           agent_user_prompt=lambda *a:'',agent=lambda *a:'I agree the axillary nerve fits.')
    rerun=Env._continue_v2(policy,CASE,[{'speaker':'doctor','stage':'D1','round':1,'text':'I favor option B.'}],
                           1,'I agree the axillary nerve fits.',11)
    assert rerun['agent_turns']==1 and rerun['termination_reason']=='intervention_agent_then_final'


def test_unsettled_doctor_runs_to_t_max(monkeypatch):
    t = episode(monkeypatch, FakeOllama(status_for=lambda r, u, s: ST(open_=True)))
    assert t['termination_reason'] == 'max_agent_turns' and t['num_discussion_rounds'] == 5
    t = episode(monkeypatch, FakeOllama(status_for=lambda r, u, s: 'garbled'))
    assert t['termination_reason'] == 'max_agent_turns'


def test_seeds_follow_the_call_identity(monkeypatch):
    f = FakeOllama(); episode(monkeypatch, f)
    seeds = [p['options']['seed'] for _, p in f.chats]
    assert seeds[:3] == [D.doctor_seed(11, 'c1', 'doctor', r) for r in (1, 2, 3)] and seeds[3] == D.doctor_seed(11, 'c1', 'final', 0)
    assert all(0 <= s < 2 ** 31 for s in seeds) and len(set(seeds)) == len(seeds)
    assert all(p['options']['temperature'] == 0.4 and p['options']['top_p'] == 1.0 and p['options']['top_k'] == 0 and
               p['options']['num_ctx'] == 8192 and p['options']['num_predict'] == 512 for _, p in f.chats)


class Agent:
    def __init__(self, text=lambda depth, seed: 'I agree that B is best supported.'): self.text = text

    def sample(self, case, prefix, turn, seed): return {'completion_text': self.text(turn, seed)}


def tree(fake=None, halting=HALT, cache=None, agent=None, branching=2, doc=None):
    doc = doc or doctor(fake, halting)
    return build_tree(CASE, max_agent_turns=5, branching=branching, run_seed=3, policy_sha256='p', protocol_sha256=doc.protocol,
                      agent=agent or Agent(), doctor=doc, stop_rule=stop_reason, final_option=final_answer_option,
                      outcome_fn=lambda c, o: float(o == 'B') if o else None, cache=cache or CallCache(None)), doc


def test_tree_halts_at_depth_two_and_keeps_status_out_of_every_dialogue():
    f = FakeOllama(); t, _ = tree(f)
    assert {l.stop_reason for l in t.leaves} == {'doctor_halt'} and {len(l.path) for l in t.leaves} == {2}
    assert len(t.nodes) == 2 + 4 and len(t.leaves) == 4                                          # no depth-3 branching below a halt
    text = json.dumps([[list(x) for x in n.prefix] + [n.response, n.next_doctor] for n in t.nodes] + [[list(x) for x in l.dialogue] for l in t.leaves])
    assert 'STATUS' not in text and 'open_issue' not in text and 'collaborator_objection' not in text
    assert all(l.dialogue[-2][0] == 'doctor' and l.dialogue[-1][1] == 'FINAL' for l in t.leaves)    # halting message, then FINAL
    halted = [n for n in t.nodes if n.stop_reason == 'doctor_halt']
    assert all(n.next_doctor_status['status'] == ST() and n.prefix_doctor_status == ST() for n in halted)


@pytest.mark.parametrize('reply', ['I agree the axillary nerve fits.', 'Final answer: option B'])
def test_training_tree_respects_three_turn_minimum_through_role_adapter(reply):
    from src.online_rl.tree.backends import RoleDoctor
    doc = RoleDoctor(doctor(FakeOllama(), {'enabled': True, 'min_agent_turns': 3}))
    t, _ = tree(doc=doc, agent=Agent(lambda d, s: reply), branching=1)
    assert [len(l.path) for l in t.leaves] == [3]


def test_intervention_continuation_is_one_agent_reply_then_final_even_with_three_turn_minimum():
    from types import SimpleNamespace
    from src.prompt_opt.harness import Env
    from src.online_rl import environment
    reply = 'Final answer: option B'
    policy = SimpleNamespace(torch=SimpleNamespace(manual_seed=lambda s: None), args=Args,
                             doctor=doctor(FakeOllama(), {'enabled': True, 'min_agent_turns': 3}),
                             agent_system='AGENT', ST=environment,
                             agent_user_prompt=lambda *a: '', agent=lambda *a: reply)
    result = Env._continue_v2(policy, CASE, [{'speaker': 'doctor', 'stage': 'D1', 'round': 1, 'text': 'I favor option B.'}],
                              1, reply, 11)
    assert result['agent_turns'] == 1 and result['termination_reason'] == 'intervention_agent_then_final'


def test_tree_never_halts_with_an_open_issue_or_a_dissenting_agent():
    t, _ = tree(FakeOllama(status_for=lambda r, u, s: ST(more=True)), branching=1)
    assert [l.stop_reason for l in t.leaves] == ['max_agent_turns']
    t, _ = tree(FakeOllama(), agent=Agent(lambda d, s: 'I would still choose option C here.'), branching=1)
    assert [l.stop_reason for l in t.leaves] == ['max_agent_turns']


def test_status_never_reaches_feedback_rubric_or_judge_inputs():
    from src.online_rl.tree import posthoc as PH
    from src.online_rl.tree.backends import stage2_questions
    from src.online_rl.tree.records import to_dialogue
    t, _ = tree(FakeOllama()); seen = []
    for l in t.leaves:
        fb = PH.doctor_feedback(CASE, l, lambda *a: seen.append(json.dumps(a, default=str)) or 'feedback', 'h', CallCache(None))
        seen.append(PH.trajectory_message(PH.freeze_trajectory_input(CASE, l, fb)))
    seen += [PH.labeled_dialogue(to_dialogue(n.prefix))[0] + n.response for n in t.nodes]                 # judge state = prefix + reply
    assert seen and not any('STATUS' in x or 'open_issue' in x or 'more_info_could_change_decision' in x for x in seen)
    assert callable(stage2_questions)


# ---------- cache and protocol identity ----------
def test_openrouter_cache_cannot_satisfy_the_local_doctor(tmp_path):
    from test_tree_v1 import Log, tree_for
    cache = CallCache(tmp_path / 'calls.jsonl'); tree_for(Log(), rounds=2, branching=1)                   # a legacy tree exists
    legacy = json.loads((ROOT / 'configs/online_rl/tree_v1.json').read_text())
    for key in ('doctor_backend', 'halting', 'doctor_opening'):
        legacy['rollout'].pop(key, None)
    from src.online_rl.integrated.continuation import protocol_sha256
    local = {**legacy, 'rollout': {**legacy['rollout'], 'doctor_backend': BACKEND, 'halting': HALT, 'doctor_model': 'ollama/x'}}
    assert protocol_sha256(legacy, ROOT) != protocol_sha256(local, ROOT)
    f = FakeOllama(); tree(f, cache=cache)
    n = len(f.chats); assert n > 0
    f2 = FakeOllama(); tree(f2, cache=cache); assert f2.chats == []                                        # same protocol: served from cache
    f3 = FakeOllama(); tree(f3, cache=cache, doc=D.OllamaDoctor({**BACKEND, 'model_digest': 'b' * 64}, HALT, 'SYSTEM', 512, http=f3))
    assert len(f3.chats) == n                                                                              # other digest: all called again


@pytest.mark.parametrize('change', [{'model_digest': 'b' * 64}, {'ollama_version': '0.36.0'}, {'quantization': 'Q4_K_M'},
                                    {'options': {**BACKEND['options'], 'num_ctx': 4096}}, {'options': {**BACKEND['options'], 'top_k': 64}}])
def test_any_backend_change_changes_the_protocol(change):
    assert doctor(**change).protocol != doctor().protocol and doctor(**change).model.startswith('ollama/gemma3:12b-it-q8_0@')


def test_identical_workers_share_the_protocol_and_endpoints_are_provenance(monkeypatch):
    a = doctor(); b = doctor(endpoint='http://127.0.0.1:9999')
    assert a.protocol == b.protocol and a.endpoints != b.endpoints                      # endpoint / GPU: deployment, not identity
    monkeypatch.setenv('SHERPA_DOCTOR_ENDPOINTS', 'http://127.0.0.1:1, http://127.0.0.1:2')
    c = doctor(); assert c.endpoints == ['http://127.0.0.1:1', 'http://127.0.0.1:2'] and c.protocol == a.protocol
    assert D.OllamaDoctor(BACKEND, HALT, 'OTHER', 512).protocol != a.protocol                # prompt still distinguishes


def test_halting_and_prompts_are_part_of_the_protocol():
    assert doctor(halting={'enabled': False, 'min_agent_turns': 2}).protocol != doctor().protocol
    assert D.OllamaDoctor(BACKEND, HALT, 'OTHER SYSTEM', 512).protocol != D.OllamaDoctor(BACKEND, HALT, 'SYSTEM', 512).protocol


def test_seed_is_part_of_each_doctor_cache_key():
    c = CallCache(None); f = FakeOllama(); tree(f, cache=c, branching=1)
    assert all('seed' in r['identity'] for r in c.store.values() if r['kind'] in ('doctor_turn', 'final'))


@pytest.mark.parametrize('short_final',[False,True])
def test_all_arms_share_the_doctor_protocol_and_halt_rule(short_final):
    from src.online_rl.tree.conditions import ARMS, arm_config
    from src.online_rl.integrated.continuation import protocol_sha256
    base = json.loads((ROOT / 'configs/online_rl/tree_v1.json').read_text())
    backend={**BACKEND}
    if short_final:backend['final_response']={'mode':'label_only_v1','max_tokens':32}
    base['rollout'] = {**base['rollout'], 'doctor_backend': backend, 'halting': HALT}
    system=(ROOT/base['rollout'].get('doctor_system_prompt_file','prompts/user_simulator.txt')).read_text()
    # Whole-run identity also includes arm-dependent rubric rerun semantics.
    # The shared Doctor identity is the experimental invariant being tested.
    assert len({D.make_doctor(arm_config(base,a)['rollout'],system).protocol for a in ARMS})==1
    if short_final:
        from copy import deepcopy
        legacy=deepcopy(base);legacy['rollout']['doctor_backend'].pop('final_response')
        for arm in ARMS:
            assert protocol_sha256(arm_config(base,arm),ROOT)!=protocol_sha256(arm_config(legacy,arm),ROOT)
    assert {json.dumps(arm_config(base, a)['rollout'], sort_keys=True) for a in ARMS} == {json.dumps(base['rollout'], sort_keys=True)}
    assert D.OllamaDoctor.halt is D.OllamaDoctor.halt and 'arm' not in D.halt_reason.__code__.co_varnames


# ---------- operational ----------
def test_endpoint_failure_is_explicit_and_nothing_is_cached():
    c = CallCache(None); f = FakeOllama(fail=50)
    with pytest.raises(D.DoctorUnavailable, match='unreachable'): tree(f, cache=c, branching=1)
    assert c.store == {}
    f = FakeOllama(fail=2); t, _ = tree(f, cache=c, branching=1); assert t.leaves                         # transient: retried, then works


def test_full_context_is_an_error_not_a_silent_truncation():
    class Full(FakeOllama):
        def __call__(self, url, payload, timeout):
            r = super().__call__(url, payload, timeout)
            return {**r, 'prompt_eval_count': 8100, 'eval_count': 100} if url.endswith('/api/chat') else r
    with pytest.raises(D.DoctorUnavailable, match='context full'): tree(Full(), branching=1)


def test_verify_checks_version_digest_and_gpu_residency():
    assert doctor().verify(require_loaded=True)['problems'] == []
    assert 'digest' in ' '.join(doctor(FakeOllama(digest='c' * 64)).verify()['problems'])
    assert 'version' in ' '.join(doctor(FakeOllama(version='0.1.0')).verify()['problems'])
    assert 'partly on CPU' in ' '.join(doctor(FakeOllama(vram_full=False)).verify(require_loaded=True)['problems'])


def test_parallel_doctor_calls_give_the_same_tree():
    from src.online_rl.integrated.records import to_dict
    st = lambda r, u, seed: ST(open_=seed % 3 == 0)                                  # some branches continue, some halt
    a, _ = tree(FakeOllama(status_for=st)); doc = doctor(FakeOllama(status_for=st))
    b = build_tree(CASE, max_agent_turns=5, branching=2, run_seed=3, policy_sha256='p', protocol_sha256=doc.protocol, agent=Agent(), doctor=doc,
                   stop_rule=stop_reason, final_option=final_answer_option, outcome_fn=lambda c, o: float(o == 'B') if o else None,
                   cache=CallCache(None), workers=4)
    assert to_dict(a) == to_dict(b) and len(a.leaves) > 4
