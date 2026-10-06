import io
import json
import pytest
from src.online_rl.tree.ollama_doctor import OllamaDoctorCall
from src.online_rl.tree.backends import ApiDoctor, api_feedback


def transport(monkeypatch, *, done_reason='stop', digest='pinned'):
    seen=[]
    def urlopen(request, timeout):
        if isinstance(request, str):
            body={'models':[{'name':'gemma3:12b','digest':digest}]}
        else:
            seen.append(json.loads(request.data))
            body={'message':{'content':'Final answer: A'},'done':True,'done_reason':done_reason}
        return io.BytesIO(json.dumps(body).encode())
    monkeypatch.setattr('urllib.request.urlopen',urlopen)
    monkeypatch.delenv('API_USAGE_LOG',raising=False)
    return OllamaDoctorCall({'url':'http://local','model':'gemma3:12b','digest':'pinned'}), seen


def test_local_transport_does_not_send_credentials(monkeypatch):
    call, seen=transport(monkeypatch)
    assert call('SECRET','ollama/gemma3:12b','system','question',512)=='Final answer: A'
    assert 'SECRET' not in json.dumps(seen)
    assert seen[0]['messages']==[{'role':'user','content':'system\n\nquestion'}]
    assert seen[0]['options']['num_gpu']==-1


def test_truncation_is_technical_failure(monkeypatch):
    call, _=transport(monkeypatch,done_reason='length')
    with pytest.raises(ValueError,match='truncated'):
        call('','ollama/gemma3:12b','system','question',512)


def test_model_digest_must_match(monkeypatch):
    with pytest.raises(ValueError,match='digest'):
        transport(monkeypatch,digest='changed')


def test_source_final_and_feedback_share_local_transport(monkeypatch):
    call, seen=transport(monkeypatch)
    doctor=ApiDoctor('SECRET','ollama/gemma3:12b','system',512,call=call)
    case={'case_id':'case','question':'Question?','options':{'A':'one','B':'two'}}
    doctor.turn(case,[],1,5)
    doctor.final(case,[])
    api_feedback('SECRET','ollama/gemma3:12b','feedback',512,call=call)(case,[],1.0)
    assert len(seen)==3
    assert all(p['model']=='gemma3:12b' for p in seen)


def test_inline_final_is_parsed_but_conflicting_finals_are_rejected():
    from src.online_rl.tree.backends import parse_final_option
    options={'A':'one','B':'two','C':'three'}
    assert parse_final_option('I agree. Final answer: option C - three',options)=='C'
    assert parse_final_option('I favor A but option C is also possible.',options) is None
    assert parse_final_option('Final answer: A. Final answer: B.',options) is None
