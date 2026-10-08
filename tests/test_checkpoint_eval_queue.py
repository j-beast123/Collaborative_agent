import json
from pathlib import Path
import pytest
from src.online_rl.tree.eval_queue import enqueue,pending_jobs
from src.online_rl.tree.checkpoints import save_periodic_snapshot
from src.online_rl.integrated.records import atomic_write_json


def checkpoint(tmp_path,count):
    model=tmp_path/f'model{count}';model.mkdir();atomic_write_json(model/'COMPLETE.json',{'policy_version':f'v{count}'})
    state={'case_cursor':count,'policy_sha256':f'sha{count}','prompt_sha256':'prompt','checkpoint':str(model)}
    return save_periodic_snapshot(tmp_path,state,30,'prompt')


def test_queue_order_idempotence_and_completed_job_skip(tmp_path):
    paths=[]
    for count in [60,30]:
        p=checkpoint(tmp_path,count)
        paths.append(enqueue(tmp_path,p,{'device':'cuda:4'},{'agent':'A','doctor':'D'}))
        assert enqueue(tmp_path,p,{'device':'cuda:4'},{'agent':'A','doctor':'D'})==paths[-1]
    jobs=list(pending_jobs(tmp_path));assert [j['completed_cases'] for _,j in jobs]==[30,60]
    path,job=jobs[0];atomic_write_json(tmp_path/'evaluation/results'/path.stem/'summary.json',{'job_sha256':job['job_sha256']})
    assert [j['completed_cases'] for _,j in pending_jobs(tmp_path)]==[60]


def test_incomplete_checkpoint_and_job_mutation_rejected(tmp_path):
    c=checkpoint(tmp_path,30);(c/'model/COMPLETE.json').unlink()
    with pytest.raises(RuntimeError):enqueue(tmp_path,c,{}, {})
    atomic_write_json(c/'model/COMPLETE.json',{'policy_version':'v30'})
    p=enqueue(tmp_path,c,{},{});job=json.loads(p.read_text());job['policy_sha256']='different';atomic_write_json(p,job)
    with pytest.raises(RuntimeError):list(pending_jobs(tmp_path))


def test_job_keeps_checkpoint_and_settings_fixed(tmp_path):
    settings={'device':'cuda:4','data_file':'test.jsonl'};p=enqueue(tmp_path,checkpoint(tmp_path,30),settings,{'agent':'frozenA','doctor':'frozenD'})
    settings['data_file']='other.jsonl'
    job=json.loads(p.read_text());assert job['settings']['data_file']=='test.jsonl'
    assert job['policy_sha256']=='sha30'
