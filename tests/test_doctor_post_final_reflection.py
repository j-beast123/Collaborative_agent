import copy
import json
import queue
import shutil
import threading
from types import SimpleNamespace

import pytest

from src.online_rl.doctor import OllamaDoctor
from src.online_rl.integrated.records import sha256_json
from src.online_rl.tree.doctor_reflection import DoctorReflectionError, PROTOCOL, local_doctor_feedback, parse_reflection
from src.online_rl.tree.gpu_pool import attach, assert_idle, create_pool


CASE = {'case_id': 'case', 'scenario': 'Visible case.', 'question': 'Which option?',
        'options': {'A': 'First', 'B': 'Second'}, 'ground_truth': 'HIDDEN_GOLD', 'outcome': 'HIDDEN_OUTCOME'}
DIALOGUE = [{'speaker': 'doctor', 'stage': 'D1', 'text': 'Please compare the alternatives.', 'status': 'HIDDEN_STATUS'},
            {'speaker': 'agent', 'stage': 'A1', 'text': 'The first alternative fits the supplied facts.'},
            {'speaker': 'doctor', 'stage': 'FINAL', 'text': 'Final answer: option A'}]
GOOD = {'helpful': [{'agent_reply_id': 1, 'experience': 'I could examine the first alternative against the facts.'}],
        'unresolved': [{'agent_reply_id': 1, 'experience': 'I still needed a comparison with the other alternative.'}]}


class RecordingDoctor:
    model = 'ollama/gemma3:12b-it-q8_0@997a7c2c0975'
    protocol = 'pinned-doctor-protocol'
    system = 'You are a clinician. Use only visible information. Continue the discussion in five sentences.'

    def __init__(self, replies=None):
        self.replies = iter(replies or [json.dumps(GOOD)])
        self.calls = []

    def _chat(self, user, seed, max_tokens=None, system_override=None):
        self.calls.append({'user': user, 'seed': seed, 'max_tokens': max_tokens, 'system': system_override})
        return next(self.replies)


def test_reflection_is_same_local_doctor_after_final_with_visible_context_only(monkeypatch):
    monkeypatch.setattr('src.collaboration.generator.doctor_call', lambda *a, **k: pytest.fail('No observer API is allowed'))
    doctor = RecordingDoctor()
    feedback = local_doctor_feedback(doctor, 'Describe your experience.', 384, run_seed=9)
    value = feedback(CASE, DIALOGUE, 'HIDDEN_SUCCESS_LABEL')
    assert len(doctor.calls) == 1
    call = doctor.calls[0]
    assert 'Final answer: option A' in call['user'] and 'Agent reply 1' in call['user']
    assert CASE['question'] in call['user'] and 'Please compare' in call['user']
    assert not any(secret in json.dumps(call) for secret in ('HIDDEN_GOLD', 'HIDDEN_OUTCOME', 'HIDDEN_STATUS', 'HIDDEN_SUCCESS_LABEL'))
    assert call['system'].startswith(doctor.system)
    assert 'same Doctor' in call['system'] and 'replace the discussion task' in call['system']
    assert call['max_tokens'] == 384 and feedback.model == doctor.model
    assert value['kind'] == PROTOCOL and value['reflection'] == GOOD and value['target_reply_id'] == 1
    assert feedback.settings['doctor_protocol'] == doctor.protocol
    proof = value['evidence'][0]
    assert proof['messages'] == [{'role': 'user', 'content': f"{call['system']}\n\n{call['user']}"}]
    assert proof['messages_sha256'] == sha256_json(proof['messages'])


def test_outcome_does_not_affect_reflection_input_or_seed():
    doctor = RecordingDoctor([json.dumps(GOOD)] * 2)
    feedback = local_doctor_feedback(doctor, 'Reflect.', 384, run_seed=17)
    feedback(CASE, DIALOGUE, 'success')
    feedback(CASE, DIALOGUE, 'failure')
    assert doctor.calls[0] == doctor.calls[1]


def test_no_experience_is_valid_without_fabricated_locus():
    doctor = RecordingDoctor([json.dumps({'helpful': [], 'unresolved': []})])
    value = local_doctor_feedback(doctor, 'Reflect.', 384)(CASE, DIALOGUE, 'failure')
    assert value['valid'] is True and value['target_reply_id'] is None and value['status'] == 'ok'


def test_malformed_reference_gets_one_repair_with_distinct_seed_and_evidence():
    invalid = copy.deepcopy(GOOD)
    invalid['helpful'][0]['agent_reply_id'] = 2
    doctor = RecordingDoctor([json.dumps(invalid), json.dumps(GOOD)])
    value = local_doctor_feedback(doctor, 'Reflect.', 384)(CASE, DIALOGUE, 'failure')
    assert value['status'] == 'repaired' and len(value['raw']) == 2 and len(value['evidence']) == 2
    assert doctor.calls[0]['seed'] != doctor.calls[1]['seed']
    assert 'existing Agent reply' in doctor.calls[1]['user']
    assert 'validation_error' in value['evidence'][0]


@pytest.mark.parametrize('opener', ['```json', '```', '```JSON'])
def test_single_outer_json_fence_is_normalized_and_raw_preserved(opener):
    raw = opener + '\n' + json.dumps(GOOD) + '\n```'
    doctor = RecordingDoctor([raw])
    result = local_doctor_feedback(doctor, 'Reflect.', 384)(CASE, DIALOGUE, 'success')
    assert result['reflection'] == GOOD and result['raw'] == [raw]
    assert result['status'] == 'ok' and len(doctor.calls) == 1


@pytest.mark.parametrize('raw', [
    'Here is the JSON:\n```json\n' + json.dumps(GOOD) + '\n```',
    '```json\n' + json.dumps(GOOD) + '\n```\nMore commentary.',
    '```json\n' + json.dumps(GOOD) + '\n```\n```json\n{}\n```',
    '```json\n' + json.dumps(GOOD),
])
def test_fence_normalization_never_extracts_json_from_mixed_output(raw):
    with pytest.raises(ValueError, match='one JSON object'):
        parse_reflection(raw, 1)


def test_failed_repair_retains_exact_requests_and_raw_for_error_audit():
    doctor = RecordingDoctor(['invalid output', '```json\n{}\n```'])
    with pytest.raises(DoctorReflectionError) as error:
        local_doctor_feedback(doctor, 'Reflect.', 384)(CASE, DIALOGUE, 'failure')
    assert error.value.raw == ['invalid output', '```json\n{}\n```']
    assert len(error.value.evidence) == 2
    assert error.value.settings['model'] == doctor.model
    assert error.value.audit['evidence'] == error.value.evidence
    assert all(item['messages'] and item['validation_error'] for item in error.value.evidence)


@pytest.mark.parametrize('invalid', [
    {'helpful': 'none', 'unresolved': []},
    {'helpful': [{'agent_reply_id': True, 'experience': 'I reviewed it.'}], 'unresolved': []},
    {'helpful': [{'agent_reply_id': 1, 'experience': '  '}], 'unresolved': []},
    {'helpful': [{'agent_reply_id': 1, 'experience': 'I reviewed it.'}] * 2, 'unresolved': []},
    {'helpful': [], 'unresolved': [], 'score': 0},
])
def test_invalid_reflection_is_operational_error_not_zero_or_na(invalid):
    doctor = RecordingDoctor([json.dumps(invalid)] * 2)
    with pytest.raises(RuntimeError, match='failed local validation after one repair'):
        local_doctor_feedback(doctor, 'Reflect.', 384)(CASE, DIALOGUE, 'success')
    assert len(doctor.calls) == 2


@pytest.mark.parametrize('dialogue', [DIALOGUE[:-1], [], DIALOGUE + [DIALOGUE[0]], DIALOGUE + [DIALOGUE[-1]]])
def test_incomplete_or_reopened_trajectory_never_calls_doctor(dialogue):
    doctor = RecordingDoctor()
    with pytest.raises(ValueError, match='FINAL|final decision'):
        local_doctor_feedback(doctor, 'Reflect.', 384)(CASE, dialogue, 'success')
    assert doctor.calls == []


def test_reflection_identity_changes_for_runtime_model_prompt_tokens_and_seed():
    doctor = RecordingDoctor()
    original = local_doctor_feedback(doctor, 'Reflect.', 384, run_seed=1).protocol
    assert local_doctor_feedback(doctor, 'Reflect carefully.', 384, run_seed=1).protocol != original
    assert local_doctor_feedback(doctor, 'Reflect.', 512, run_seed=1).protocol != original
    assert local_doctor_feedback(doctor, 'Reflect.', 384, run_seed=2).protocol != original
    doctor.protocol = 'different-pinned-backend'
    assert local_doctor_feedback(doctor, 'Reflect.', 384, run_seed=1).protocol != original


def test_ollama_system_override_is_per_call_and_preserves_model_settings(monkeypatch):
    monkeypatch.delenv('SHERPA_DOCTOR_ENDPOINTS', raising=False)
    monkeypatch.delenv('API_USAGE_LOG', raising=False)
    seen = []
    def http(url, payload, timeout):
        seen.append(payload)
        return {'message': {'content': json.dumps(GOOD)}, 'prompt_eval_count': 5, 'eval_count': 8}
    backend = {'engine': 'ollama', 'endpoint': 'http://local', 'model': 'gemma3:12b-it-q8_0',
               'model_digest': 'a' * 64, 'quantization': 'Q8_0', 'ollama_version': '0.12.5',
               'options': {'num_ctx': 8192, 'temperature': 0.4}}
    doctor = OllamaDoctor(backend, None, 'Original clinical role.', 256, http=http)
    original_identity = copy.deepcopy(doctor.identity)
    doctor._chat('reflection', 1, 384, 'Original clinical role. Reflect now.')
    doctor._chat('normal turn', 2)
    assert seen[0]['messages'][0]['content'] == 'Original clinical role. Reflect now.\n\nreflection'
    assert seen[1]['messages'][0]['content'] == 'Original clinical role.\n\nnormal turn'
    assert [request['options']['num_predict'] for request in seen] == [384, 256]
    assert all(request['model'] == backend['model'] for request in seen)
    assert doctor.system == 'Original clinical role.' and doctor.identity == original_identity


def test_reflection_uses_shared_doctor_capacity_and_audits_actual_override(tmp_path):
    manager = SimpleNamespace(RLock=threading.RLock, dict=dict, Queue=queue.Queue)
    spec = create_pool(manager, [0], agent_capacity=4, doctor_capacity=1)
    doctor = RecordingDoctor([json.dumps(GOOD), json.dumps(GOOD)])
    policy = SimpleNamespace(doctor=doctor, _generate=lambda *a: None, gen_lock=threading.RLock(), model=object())
    actor = attach(policy, {'output_dir': str(tmp_path), 'optimizer': {'logprob_token_chunk_size': 32}}, spec, 0)
    actor.expected = policy.pool_client.expected = 'frozen-policy'
    try:
        result = local_doctor_feedback(doctor, 'Reflect.', 384)(CASE, DIALOGUE, 'success')
        assert result['reflection'] == GOOD and len(doctor.calls) == 1
        assert_idle(spec)
        row = json.loads((tmp_path / 'live/pool_gpu0.jsonl').read_text())
        assert row['op'] == 'doctor' and row['seed'] == result['evidence'][0]['seed']
        assert row['system_override_sha256'] == result['settings']['system_sha256']
        assert doctor._chat('next discussion', 51) == json.dumps(GOOD)
        assert doctor.calls[-1]['system'] is None
    finally:
        actor.close()
        shutil.rmtree(spec['directory'])


def test_multiple_experiences_repair_selects_existing_text_with_no_rewrite():
    multiple=copy.deepcopy(GOOD)
    multiple['helpful'].append({'agent_reply_id':1,'experience':'I compared a second consideration from the same reply.'})
    doctor=RecordingDoctor([json.dumps(multiple),json.dumps({'helpful_index':1,'unresolved_index':0})])
    fn=local_doctor_feedback(doctor,'Reflect.',512,run_seed=17)
    result=fn(CASE,DIALOGUE,'HIDDEN_OUTCOME_LABEL')
    assert result['status']=='repaired' and len(doctor.calls)==2
    assert result['reflection']['helpful']==[multiple['helpful'][1]]
    assert result['reflection']['unresolved']==multiple['unresolved']
    assert result['selection_repair']['choice']=={'helpful_index':1,'unresolved_index':0}
    request=doctor.calls[1]
    assert json.dumps(multiple,ensure_ascii=False) in request['user']
    assert 'indices in the supplied lists, NOT Agent reply IDs' in request['system']
    assert not any(secret in json.dumps(request) for secret in ('HIDDEN_GOLD','HIDDEN_STATUS','HIDDEN_OUTCOME_LABEL'))
    assert result['evidence'][1]['messages']==[{'role':'user','content':request['system']+'\n\n'+request['user']}]
    assert result['evidence'][1]['repair_mode']=='select_existing_experience_v1'


@pytest.mark.parametrize('index',[None,True,-1,2,'1'])
def test_invalid_selection_never_silently_truncates_or_zeros(index):
    multiple=copy.deepcopy(GOOD);multiple['helpful']*=2
    doctor=RecordingDoctor([json.dumps(multiple),json.dumps({'helpful_index':index,'unresolved_index':0})])
    with pytest.raises(DoctorReflectionError,match='helpful_index') as error:
        local_doctor_feedback(doctor,'Reflect.',512)(CASE,DIALOGUE,'success')
    assert len(doctor.calls)==2 and len(error.value.audit['raw'])==2


def test_empty_list_selection_and_seed_repeat_are_stable():
    multiple=copy.deepcopy(GOOD);multiple['helpful']*=2;multiple['unresolved']=[]
    replies=[json.dumps(multiple),json.dumps({'helpful_index':0,'unresolved_index':None})]
    a=RecordingDoctor(replies);b=RecordingDoctor(replies)
    first=local_doctor_feedback(a,'Reflect.',512,run_seed=42)(CASE,DIALOGUE,'success')
    second=local_doctor_feedback(b,'Reflect.',512,run_seed=42)(CASE,DIALOGUE,'failure')
    assert a.calls==b.calls and first==second
    assert first['reflection']['unresolved']==[]


def test_malformed_item_requires_full_format_repair_with_previous_raw():
    multiple=copy.deepcopy(GOOD);multiple['helpful']*=2;multiple['helpful'][1]={'agent_reply_id':999,'experience':'bad ID'}
    doctor=RecordingDoctor([json.dumps(multiple),json.dumps(GOOD)])
    result=local_doctor_feedback(doctor,'Reflect.',512)(CASE,DIALOGUE,'success')
    assert 'Previous output:' in doctor.calls[1]['user']
    assert result['reflection']==GOOD and 'selection_repair' not in result
    assert result['evidence'][1]['repair_mode']=='rewrite_format_v1'


def test_selection_success_cache_reuses_without_doctor_calls(tmp_path):
    from src.online_rl.tree.cache import CallCache
    multiple=copy.deepcopy(GOOD);multiple['helpful']*=2
    doctor=RecordingDoctor([json.dumps(multiple),json.dumps({'helpful_index':1,'unresolved_index':0})])
    fn=local_doctor_feedback(doctor,'Reflect.',512,run_seed=42)
    cache=CallCache(tmp_path/'calls.jsonl');identity={'protocol':fn.protocol,'settings':fn.settings,'case':'case'}
    result=cache.call('feedback',identity,lambda:fn(CASE,DIALOGUE,'success'))
    assert len(doctor.calls)==2 and fn.settings['format_repair_attempts']==1
    restored=CallCache(cache.path).call('feedback',identity,lambda:pytest.fail('Doctor called on resume'))
    assert restored==result
