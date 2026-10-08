import json
from scripts.online_rl.watch_training import Journal


def test_incremental_monitor_counts_once_and_waits_for_complete_line(tmp_path):
    path=tmp_path/'calls.jsonl';record={'kind':'agent','identity':{'path':'0.1'}}
    text=json.dumps(record);path.write_text(text[:10]);journal=Journal();journal.update(path)
    assert not journal.counts and journal.offset==0
    with path.open('a') as f:f.write(text[10:]+'\n')
    journal.update(path);journal.update(path)
    assert journal.counts['agent']==1 and journal.depths[2]==1
    with path.open('a') as f:f.write(json.dumps({'kind':'rerun'})+'\n')
    journal.update(path)
    assert journal.counts['agent']==1 and journal.counts['rerun']==1


def test_terminal_prefix_prunes_remaining_subtree():
    from scripts.online_rl.watch_training import rollout_work
    j=Journal();j.depths[3]=1;j.final_paths.add('0.1.2')
    assert rollout_work(j,4)==(5,340)
    assert rollout_work(j,5)==(21,1364)
    j.depths[5]=100
    assert rollout_work(j,4)==(5,340)


def test_estimator_persists_samples_and_reports_stalled_case(tmp_path,capsys):
    from scripts.online_rl.watch_training import estimates
    active=tmp_path/'batch_000001';active.mkdir()
    j=Journal();j.depths[1]=1;j.counts['agent']=1
    cases=[(0,j,0,False)]
    estimates(tmp_path,active,cases,now=1000)
    estimates(tmp_path,active,cases,now=1120)
    text=capsys.readouterr().out
    assert '진행 없는 case' in text
    assert len(json.loads((tmp_path/'monitor_progress.json').read_text())['samples'])==2
