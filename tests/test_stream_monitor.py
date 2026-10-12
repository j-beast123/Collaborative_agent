import json
from src.common.bank_monitor import BankCounters


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))


def test_live_queue_counts_use_actual_chunk_sizes_and_journal(tmp_path):
    case=tmp_path/'case_000003'
    save(case/'stream_plan.json',{'chunks':[['a']*8,['b']*8,['c']*3]})
    save(case/'stream_progress.json',{'ready_trajectories':22,'chunks_committed':0,'current_chunk':2})
    save(tmp_path/'trajectory_stream/state.json',{'transactions':['case_000003_chunk_0000']})
    before={str(p):p.stat().st_mtime_ns for p in tmp_path.rglob('*') if p.is_file()}
    flow=BankCounters().read(case,discovery_cap=None)['flow']
    assert flow['committed_trajectories']==8 and flow['committed_chunks']==1
    assert flow['processing_trajectories']==8 and flow['waiting_trajectories']==6
    assert before=={str(p):p.stat().st_mtime_ns for p in tmp_path.rglob('*') if p.is_file()}


def test_committed_tail_is_not_shown_as_processing_even_if_progress_lags(tmp_path):
    case=tmp_path/'case_000000'
    save(case/'stream_plan.json',{'chunks':[['a']*8,['b']*3]})
    save(case/'stream_progress.json',{'ready_trajectories':11,'current_chunk':2,'rollout_done':True})
    save(tmp_path/'trajectory_stream/state.json',{'transactions':['case_000000_chunk_0000','case_000000_chunk_0001']})
    flow=BankCounters().read(case)['flow']
    assert flow['committed_trajectories']==11 and flow['processing_trajectories']==0
    assert flow['waiting_trajectories']==0 and flow['current_chunk'] is None


def test_lock_wait_is_separate_from_active_bank_work(tmp_path):
    case=tmp_path/'case_000000'
    save(case/'stream_plan.json',{'chunks':[['a']*8,['b']*8]})
    save(case/'stream_progress.json',{'ready_trajectories':28,'current_chunk':2,'stage':'Bank lock 대기'})
    save(tmp_path/'trajectory_stream/state.json',{'transactions':['case_000000_chunk_0000']})
    flow=BankCounters().read(case)['flow']
    assert flow['committed_trajectories']==8 and flow['waiting_trajectories']==12
    assert flow['lock_wait_trajectories']==8 and flow['processing_trajectories']==0
