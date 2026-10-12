from concurrent.futures import ThreadPoolExecutor,ProcessPoolExecutor
import multiprocessing
import threading
import time
import pytest
from src.common import api_slots
from src.online_rl.tree.experiment import load_config


@pytest.fixture(autouse=True)
def reset_slots():
    api_slots.install({})
    yield
    api_slots.install({})


def test_independent_limits_wait_and_release_after_errors():
    api_slots.install(api_slots.create({'gemini':2,'luna':3}))
    lock=threading.Lock();active={'gemini':0,'luna':0};peak=active.copy()
    def job(family):
        model='google/gemini-3.8-flash' if family=='gemini' else 'gpt-6-luna'
        with api_slots.lease(model):
            with lock:
                active[family]+=1;peak[family]=max(peak[family],active[family])
            time.sleep(.03)
            with lock:active[family]-=1
    with ThreadPoolExecutor(12) as pool:
        futures=[pool.submit(job,family) for family in ['gemini']*6+['luna']*6]
        for f in futures:f.result(timeout=5)
    assert peak=={'gemini':2,'luna':3}
    with pytest.raises(ValueError):
        with api_slots.lease('gpt-6-luna'):raise ValueError('failure')
    assert api_slots._slots['luna'].acquire(blocking=False)
    api_slots._slots['luna'].release()


def process_job(slots,active,peak,lock):
    api_slots.install(slots)
    with api_slots.lease('gpt-6-luna'):
        with lock:
            active.value+=1;peak.value=max(peak.value,active.value)
        time.sleep(.04)
        with lock:active.value-=1
    return True


def test_run_global_limit_is_shared_across_coordinator_processes():
    ctx=multiprocessing.get_context('spawn')
    with ctx.Manager() as manager:
        slots=api_slots.create({'luna':1},manager)
        active=manager.Value('i',0);peak=manager.Value('i',0);lock=manager.Lock()
        with ProcessPoolExecutor(2,mp_context=ctx) as pool:
            fs=[pool.submit(process_job,slots,active,peak,lock) for _ in range(4)]
            assert all(f.result(timeout=15) for f in fs)
        assert peak.value==1 and active.value==0


def test_direct_config_keeps_gemini_and_learning_settings():
    old=load_config('configs/online_doctor_v2.json');new=load_config('configs/online_openai20.json')
    old_rubric=__import__('copy').deepcopy(old['rubric'])
    old_rubric['cumulative_bank']['discovery_trajectory_cap']=None
    assert old_rubric==new['rubric'] and old['rollout']==new['rollout']
    assert old['agent_reward']==new['agent_reward'] and old['schedule']==new['schedule']
    assert new['five_gpu']['api_workers']==20 and new['execution']['chunk_workers']==8
    assert new['execution']['api_limits']=={'gemini':10,'luna':20}
    assert new['judges']['api_provider']=='openai_direct'
    assert all(new['judges'][key]=='gpt-6-luna' for key in ['gate_model','scorer_model','intervention_audit_model'])
    assert old['output_dir']!=new['output_dir']
    assert load_config('configs/online_rubric_only_openai20.json')['agent_reward']['w_outcome']==0


def test_openai_key_uses_environment_then_existing_env_file(tmp_path,monkeypatch):
    from scripts.online_rl.run_tree import load_key
    monkeypatch.setenv('OPENAI_API_KEY','dummy-environment-key')
    assert load_key({'api_provider':'openai_direct'})=='dummy-environment-key'
    monkeypatch.delenv('OPENAI_API_KEY')
    monkeypatch.delenv('SHERPA_OPENAI_ENV_FILE',raising=False)
    monkeypatch.setattr('scripts.online_rl.run_tree.Path.home',lambda:tmp_path)
    path=tmp_path/'private.env';path.write_text('OPENAI_API_KEY="dummy-file-key"\n')
    monkeypatch.setenv('SHERPA_ENV_FILE',str(path))
    assert load_key({'api_provider':'openai_direct'})=='dummy-file-key'
