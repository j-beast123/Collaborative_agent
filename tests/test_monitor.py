import json
import os
from pathlib import Path
import pytest
import monitor
from src.common import progress


def save(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))


def test_existing_run_infers_phase_and_case_completion_without_writing(tmp_path):
    save(tmp_path/'run_state.json', {'case_cursor':5,'batch_cursor':1})
    save(tmp_path/'live/gpu0.json', {'pid':os.getpid(),'gpu':0,'phase':'bank_online',
                                   'active':{'bank_online.trajectory_match_generate_verify':1},
                                   'active_since':{'bank_online.trajectory_match_generate_verify':120},'updated_at':125})
    save(tmp_path/'batches/batch_000002/case_000005/bank_online.json', {})
    before={str(p):p.stat().st_mtime_ns for p in tmp_path.rglob('*') if p.is_file()}
    state=monitor.snapshot(tmp_path,now=150)
    assert state['batch']==2 and state['completed']==5
    assert state['stage']=='bank_online' and state['stage_seconds']==30
    assert state['files']['bank_online']==1 and not state['exact']
    assert before=={str(p):p.stat().st_mtime_ns for p in tmp_path.rglob('*') if p.is_file()}


def test_precise_stage_timer_resets_but_repeated_case_events_do_not(tmp_path,monkeypatch):
    clock=[100.]
    monkeypatch.setattr(progress.time,'time',lambda:clock[0])
    monkeypatch.setattr(progress,'_root',None)
    monkeypatch.setattr(progress,'_state',{})
    progress.configure(tmp_path)
    progress.emit({'stage':'bank_rollout','batch':1})
    clock[0]=130;progress.emit({'stage':'bank_online','batch':1,'case_cursor':0})
    clock[0]=140;progress.emit({'stage':'bank_online','batch':1,'case_cursor':1})
    save(tmp_path/'live/gpu0.json',{'pid':os.getpid(),'gpu':0})
    state=monitor.snapshot(tmp_path,now=150)
    assert state['exact'] and state['stage_seconds']==20 and state['batch_seconds']==50
    clock[0]=160;progress.emit({'stage':'batch_committed','batch':1,'completed_cases':5})
    assert monitor.read(tmp_path/'progress.json')['batch_seconds']==60
    clock[0]=170;progress.emit({'stage':'bank_rollout','batch':2})
    assert monitor.snapshot(tmp_path,now=175)['batch_seconds']==5
    events=[json.loads(line) for line in (tmp_path/'progress_events.jsonl').read_text().splitlines()]
    assert events[1]['previous_stage_seconds']==30


def test_corrupt_status_and_finished_training_are_safe(tmp_path):
    (tmp_path/'progress.json').write_text('{')
    save(tmp_path/'run_state.json',{'case_cursor':120,'batch_cursor':24})
    state=monitor.snapshot(tmp_path,now=150)
    assert state['stage']=='run_finished' and state['batch']==24
    assert '실행 완료' in monitor.render(state)
