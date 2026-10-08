import json

from scripts.online_rl.watch_five_gpu import (JsonlTail, WorkerCache, leaf_progress, Monitor,
    render, render_compact_dashboard, summarize_phase, wilson_interval)
from src.common import live_gpu


def test_tail_partial_rows_and_mixed_training_log(tmp_path):
    path=tmp_path/'log'
    path.write_bytes(b'progress\n123\n{"stage":"BASE_eval"}\n{"stage":')
    tail=JsonlTail()
    assert list(tail.read(path))==[{'stage':'BASE_eval'}]
    assert list(tail.read(path))==[]
    with path.open('ab') as f:f.write(b'"agent_update"}\n')
    assert list(tail.read(path))==[{'stage':'agent_update'}]
    path.write_text('{"reset":true}\n')
    assert list(tail.read(path))==[{'reset':True}]


def test_activity_records_failure_and_clears_active(tmp_path,monkeypatch):
    monkeypatch.setattr(live_gpu,'_status',None)
    live_gpu.configure(tmp_path,2)
    try:
        with live_gpu.activity('generate'):
            row=json.loads((tmp_path/'live/gpu2.json').read_text())
            assert row['active']=={'generate':1}
            raise ValueError('private text')
    except ValueError:pass
    row=json.loads((tmp_path/'live/gpu2.json').read_text())
    assert row['active']=={}
    assert row['calls']['generate']==1
    assert row['last_error']=='ValueError'
    assert 'private text' not in (tmp_path/'live/gpu2.json').read_text()


def test_leaf_bounds_prune_early_terminals_and_use_actual_depth_width():
    # Two of four depth-2 branches finish; one continues to two depth-3 children.
    agents={'0','1','0.0','0.1','1.0','1.1','1.0.0','1.0.1'}
    progress=leaf_progress(agents,{'0.0','0.1'},2,4,2)
    assert progress['completed']==2
    assert progress['remaining_min']==4 and progress['remaining_max']==8
    assert progress['depth_target']==4  # two live depth-2 parents, not 2**3
    finished=leaf_progress(agents,{'0.0','0.1','1.1','1.0.0','1.0.1'},2,4,2)
    assert finished['tree_complete'] and finished['remaining_min']==finished['remaining_max']==0
    assert finished['completed']==5


def test_before_generation_min_turns_and_max_turns_bound_future_leaves():
    progress=leaf_progress(set(),set(),4,5,2)
    assert (progress['remaining_min'],progress['remaining_max'])==(16,1024)
    # At the turn limit, every unfinished planned sibling must become a leaf.
    at_limit=leaf_progress({'0','1','0.0','0.1','1.0','1.1'},{'0.0'},2,2,2)
    assert at_limit['remaining_min']==at_limit['remaining_max']==3


def test_cache_counts_unparsed_finals_but_excludes_rerun_paths(tmp_path):
    path=tmp_path/'calls.jsonl'
    rows=[{'kind':'agent','identity':{'path':'0'}},
          {'kind':'final','identity':{'path':'0'},'value':'unparseable'},
          {'kind':'rerun_agent_turn','identity':{'path':'1.0'}},
          {'kind':'rerun','identity':{'path':'1'}}]
    path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    cache=WorkerCache(path);cache.update(1);cache.update(2)
    assert cache.agent_paths=={'0'} and cache.final_paths=={'0'}
    assert cache.counts['agent']==cache.counts['final']==1


def test_monitor_uses_persisted_start_times_and_freezes_finished_case(tmp_path,monkeypatch):
    import scripts.online_rl.watch_five_gpu as watch
    monkeypatch.setattr(watch,'gpu_sample',lambda:({},None))
    monkeypatch.setattr(watch.time,'time',lambda:2000.)
    (tmp_path/'launch.json').write_text(json.dumps({'launched_at':1000}))
    batch=tmp_path/'batches/batch_000001';case=batch/'case_000000';case.mkdir(parents=True)
    manifest=batch/'manifest.json'
    manifest.write_text(json.dumps({'assignments':[{'worker':0,'index':0}],'case_ids':['c1']}))
    import os
    os.utime(manifest,(1100,1100))
    (case/'execution_runtime.jsonl').write_text(json.dumps({'time':'1970-01-01T00:20:00Z'})+'\n')
    rows=[{'kind':'agent','identity':{'path':str(j)}} for j in range(2)]
    rows += [{'kind':'final','identity':{'path':str(j)}} for j in range(2)]
    (case/'calls.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    union=case/'union.json';union.write_text('{}');os.utime(union,(1600,1600))
    cfg={'five_gpu':{'rollout_gpus':[0]},'rollout':{'branching':2,'rounds':3,'halting':{'min_agent_turns':1}},'schedule':{'cases':500}}
    monitor=Monitor(tmp_path,cfg);snapshot=monitor.snapshot()
    assert snapshot['run_elapsed_seconds']==1000 and snapshot['batch_elapsed_seconds']==900
    assert snapshot['gpus'][0]['case_elapsed_seconds']==400
    assert snapshot['leaves']=={'completed':2,'remaining_min':0,'remaining_max':0,'trees_complete':1,'trees_total':1}
    text=render(snapshot)
    assert 'Elapsed run 16m 40s' in text and '말단: 완료 2 | 남은 0' in text
    assert 'Agent 2/2' in text


def test_postprocessing_counts_requests_separately_from_nodes(tmp_path):
    rows=[{'kind':'rubric_score','identity':{'node':'n1','rubrics':['r1']}},
          {'kind':'rubric_score','identity':{'node':'n1','rubrics':['r2']}},
          {'kind':'rubric_score','identity':{'node':'n2','rubrics':['r1','r2']}},
          {'kind':'feedback','identity':{'leaf':'l1'}},
          {'kind':'feedback','identity':{'leaf':'l1'}},
          {'kind':'posthoc_gate','value':{'passed':True}},
          {'kind':'posthoc_gate','value':{'passed':False}},
          {'kind':'posthoc_gate','value':{}},
          {'kind':'rerun','value':{'outcome':0}},
          {'kind':'rerun','value':{'outcome':None}}]
    path=tmp_path/'calls.jsonl'
    path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    cache=WorkerCache(path);cache.update(1);cache.update(2)
    assert cache.counts['rubric_score']==3 and cache.scored_nodes=={'n1','n2'}
    assert cache.feedback_leaves=={'l1'}
    assert (cache.gate_pass,cache.gate_reject)==(1,1)
    assert cache.counts['rerun']==2 and cache.rerun_unparsed==1


def test_live_score_distribution_counts_response_rubric_pairs_and_na(tmp_path):
    rows=[
        {'kind':'rubric_score','identity':{'node':'n1'},'value':{
            'r0':{'applicable':True,'hard':0,'stage2':{'level':'0'}},
            'r1':{'applicable':True,'hard':1,'stage2':{'level':'1'}},
            'r2':{'applicable':True,'hard':2,'stage2':{'level':'2'}},
            'rna':{'applicable':False,'hard':None,'stage2':None}}},
        # A repeated cache record must not inflate the live histogram.
        {'kind':'rubric_score','identity':{'node':'n1'},'value':{
            'r0':{'applicable':True,'hard':0,'stage2':{'level':'0'}}}},
    ]
    path=tmp_path/'calls.jsonl';path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    cache=WorkerCache(path);cache.update(1)
    assert cache.live_score_distribution=={'0':1,'1':1,'2':1,'NA':1}
    assert len(cache.scored_pairs)==4


def test_dashboard_shows_rubric_keep_and_score_distribution_per_gpu_case():
    import scripts.online_rl.watch_five_gpu as watch
    snapshot={'time':'now','batch':'batch_1','run_dir':'run','api':{},'gpus':[{
        'gpu':0,'util':10,'memory_mb':32000,'total_mb':81920,'phase':'P0 scoring','progress':'A4',
        'pipeline':{'rubric_funnel':{
            'P0':{'rubrics_keep':2,'rubrics_validated':5},'P1':None},
            'live_scored_pairs':4,'live_score_distribution':{'0':1,'1':1,'2':1,'NA':1}}}],
        'pipeline':{'cases_total':1,'phase_complete':{'P0':0,'P1':0,'union':0},
            'one_step_reruns_done':0,'keep':{'P0':2,'P1':None},
            'score_distribution':{phase:{'cases':0,'counts':{k:0 for k in ('0','1','2','NA','other')}}
                                 for phase in ('P0','P1','union')}},'sample_window_seconds':60}
    text=watch.render_dashboard(snapshot)
    assert 'case stats · KEEP P0 2/5 / P1 pending · score n=4 0:1 1:1 2:1 NA:1' in text
    assert 'Per-GPU KEEP/scores follow the assigned case' in text


def test_compact_dashboard_uses_run_lock_instead_of_stale_handoff_pid():
    import scripts.online_rl.watch_five_gpu as watch
    text=watch.render_compact_dashboard({'time':'now','batch':'batch_1','run_dir':'run','api':{},
        'runner_active':True,'handoff':{'stage':'training_running','pid':123},'gpus':[],'pipeline':{},'leaves':{},
        'run_elapsed_seconds':3600})
    assert 'RUNNER ACTIVE · run lock held' in text
    assert 'RUNNER STOPPED' not in text


def test_runner_lock_detects_active_run(tmp_path):
    import fcntl
    import scripts.online_rl.watch_five_gpu as watch
    path=tmp_path/'runner.lock';path.touch()
    with path.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert watch.runner_lock_active(tmp_path) is True
    assert watch.runner_lock_active(tmp_path) is False


def test_one_step_rerun_progress_excludes_legacy_multiturn_cache(tmp_path):
    protocol='intervention_agent_once_doctor_final_once_v1'
    rows=[
        {'kind':'rerun','identity':{'continuation_protocol':protocol}},
        {'kind':'rerun','identity':{'rubric':'old'}},
        {'kind':'rerun_doctor_turn','identity':{'rerun':{'continuation_protocol':protocol,'case_id':'c1','rubric':'r1','node':'n1','level':'2'},'turn':{'stage':'FINAL'}}},
        {'kind':'rerun_doctor_turn','identity':{'rerun':{'continuation_protocol':protocol,'case_id':'c1','rubric':'r1','node':'n1','level':'2'},'turn':{'stage':'D3'}}},
        {'kind':'rerun_agent_turn','identity':{'rerun':{'continuation_protocol':'old'}}},
    ]
    path=tmp_path/'calls.jsonl';path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    cache=WorkerCache(path);cache.update(1)
    assert cache.one_step_reruns==1 and cache.one_step_final_calls==1
    assert len(cache.one_step_targets)==1 and cache.legacy_reruns==1


def test_live_witness_keep_appears_after_two_parsed_target_hits(tmp_path):
    protocol='intervention_agent_once_doctor_final_once_v1'
    rows=[
        {'kind':'rubric_applicability','identity':{'rubric':'r1','node':'n1'},
         'value':{'r1':{'applicable':True}}},
        {'kind':'rerun','identity':{'rubric':'r1','node':'n1','level':'2','seed':11,
                                    'continuation_protocol':protocol},'value':{'outcome':1.0}},
    ]
    path=tmp_path/'calls.jsonl';path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    cache=WorkerCache(path);cache.update(1)
    assert len(cache.live_witness_keep_ids())==0
    with path.open('a') as f:
        f.write(json.dumps({'kind':'rerun','identity':{'rubric':'r1','node':'n1','level':'2','seed':12,
                    'continuation_protocol':protocol},'value':{'outcome':1.0}})+'\n')
    cache.update(2)
    assert cache.live_witness_keep_ids()=={'r1'}
    # Unparsed outcomes cannot contribute as target hits.
    with path.open('a') as f:
        f.write(json.dumps({'kind':'rubric_applicability','identity':{'rubric':'r2','node':'n2'},
                    'value':{'r2':{'applicable':True}}})+'\n')
        for seed in (13,14):
            f.write(json.dumps({'kind':'rerun','identity':{'rubric':'r2','node':'n2','level':'0','seed':seed,
                        'continuation_protocol':protocol},'value':{'outcome':None}})+'\n')
    cache.update(3)
    assert cache.live_witness_keep_ids()=={'r1'}


def test_confirmed_scores_count_nodes_once_and_preserve_na():
    record={'tree':{'nodes':[{'node_id':n} for n in ('n1','n2','n3')]},
            'response_scores':[{'response_node_id':'n1','score':0},
                               {'response_node_id':'n1','score':1},
                               {'response_node_id':'n2','score':1},
                               {'response_node_id':'n3','score':None}],
            'agent_targets':{'nodes':{'n1':{'reward':0},'n2':{'reward':1},'n3':{'reward':None}}},
            'rubric_bank':{'witness':[{'decision':'KEEP'},{'decision':'NOT_VALIDATED'}]},
            'rubric_weights':{'r1':1},'training_rubrics':[{'rubric_id':'r1'}]}
    summary=summarize_phase(record)
    assert summary['total_nodes']==summary['scored_nodes']==3
    assert summary['valid_rewards']==2 and summary['na_rewards']==1
    assert summary['reward_sum']==1
    assert summary['keep']==summary['not_validated']==summary['training_rubrics']==1


def test_phase_rubric_funnel_reports_canonical_keep_denominator():
    summary=summarize_phase({'rubric_bank':{
        'witness':[{'decision':'KEEP'},{'decision':'NOT_VALIDATED'}],
        'telemetry':{'created_new':4,'rubrics':2,'keep':1,'not_validated':1}}})
    assert summary['rubrics_generated']==4
    assert summary['rubrics_validated']==2
    assert summary['rubrics_keep']==1
    assert summary['rubrics_not_validated']==1


def test_phase_summary_exposes_rollout_wall_time():
    summary=summarize_phase({'tree':{'nodes':[]},'response_scores':[],
                             'timings':{'rollout_wall_seconds':12.5},'seconds':30.0})
    assert summary['rollout_wall_seconds']==12.5 and summary['step_wall_seconds']==30.0


def test_phase_summary_counts_0_1_2_and_no_match_distribution():
    summary=summarize_phase({'tree':{'nodes':[]},'response_scores':[
        {'response_node_id':'n0','applicable':True,'hard':0.0,'score':0.0},
        {'response_node_id':'n1','applicable':True,'hard':0.5,'score':0.5},
        {'response_node_id':'n2','applicable':True,'hard':1.0,'score':1.0},
        {'response_node_id':'n3','applicable':False,'hard':None,'score':None}]})
    assert summary['score_distribution']=={'0':1,'1':1,'2':1,'NA':1,'other':0}
    assert summary['scored_pairs']==4


def test_partial_eval_accuracy_estimate_shows_wilson_interval():
    low,high=wilson_interval(39,50)
    assert round(low*100,1)==64.8 and round(high*100,1)==87.2
    dashboard=render_compact_dashboard({
        'time':'now','batch':'batch_000006','run_elapsed_seconds':60,
        'api':{},'gpus':[],
        'checkpoint_evaluation':{'checkpoint_cases':30,'evaluated':50,'total':622,
            'status':'pending','accuracy':.78,'correct':39,'accuracy_ci95':(low,high)}})
    assert '622 eval C30 · 50/622 · running · est 78.0%' in dashboard
    assert 'Wilson95 64.8–87.2%' in dashboard


def test_monitor_counts_per_case_eval_commits_ahead_of_stale_summary(tmp_path,monkeypatch):
    import scripts.online_rl.watch_five_gpu as watch
    monkeypatch.setattr(watch,'gpu_sample',lambda:({},None))
    job=tmp_path/'evaluation/checkpoint_000030'
    cases=job/'cases';cases.mkdir(parents=True)
    (job/'manifest.json').write_text(json.dumps({
        'completed_cases':30,'settings':{'eval_cases':622},'case_ids':['a','b','c']}))
    # Summary is behind case-level atomic commits, as it can be between flushes.
    (job/'summary.json').write_text(json.dumps({'evaluated':1,'correct':1,'accuracy':1.0,'status':'pending'}))
    for i,outcome in enumerate((1.0,0.0,None)):
        (cases/f'case_{i:06d}.json').write_text(json.dumps({'outcome':outcome}))
    cfg={'five_gpu':{'rollout_gpus':[]},'rollout':{'branching':2,'rounds':2},
         'schedule':{'cases':500}}
    result=Monitor(tmp_path,cfg).snapshot()['checkpoint_evaluation']
    assert result['evaluated']==3 and result['correct']==1 and result['unparsed']==1
    assert result['accuracy']==1/3
    assert result['status']=='pending'


def test_monitor_reports_api_latency_by_purpose(tmp_path,monkeypatch):
    import scripts.online_rl.watch_five_gpu as watch
    monkeypatch.setattr(watch,'gpu_sample',lambda:({},None))
    (tmp_path/'api_usage.jsonl').write_text('\n'.join(json.dumps(row) for row in (
        {'purpose':'rubric_scorer','latency_seconds':5,'request_seconds':2,'retry_wait_seconds':3,'cost_usd':.1},
        {'purpose':'rubric_generator','latency_seconds':4,'request_seconds':4,'retry_wait_seconds':0,'cost_usd':.2}))+'\n')
    cfg={'five_gpu':{'rollout_gpus':[]},'rollout':{'branching':2,'rounds':2},'schedule':{'cases':10}}
    snapshot=Monitor(tmp_path,cfg).snapshot();timing=snapshot['api']['timing']
    assert timing['timed_calls']==2 and timing['latency_seconds']==9
    assert timing['request_seconds']==6 and timing['retry_wait_seconds']==3
    assert snapshot['api']['by_purpose']['rubric_scorer']['calls']==1
    dashboard=watch.render_dashboard(snapshot)
    assert 'API $0.30 · 2 calls' in dashboard
    assert 'API time 0m 09s total · 4.5s avg' in dashboard
    assert len(dashboard.splitlines())<=7


def test_source_scoring_is_not_final_training_scoring(tmp_path,monkeypatch):
    import scripts.online_rl.watch_five_gpu as watch
    monkeypatch.setattr(watch,'gpu_sample',lambda:({},None))
    batch=tmp_path/'batches/batch_000001';case=batch/'case_000000';case.mkdir(parents=True)
    (batch/'manifest.json').write_text(json.dumps({'assignments':[{'worker':0,'index':0}],'case_ids':['c1']}))
    rows=[{'kind':'agent','identity':{'path':'0'}},
          {'kind':'final','identity':{'path':'0'}},
          {'kind':'rubric_score','identity':{'node':'n1','rubrics':['source-rubric']}}]
    (case/'calls.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    cfg={'five_gpu':{'rollout_gpus':[0]},'rollout':{'branching':1,'rounds':1},'schedule':{'cases':500}}
    monitor=Monitor(tmp_path,cfg)
    pending=monitor.snapshot()['pipeline']
    assert pending['scoring_requests']==pending['scoring_unique_nodes']==1
    assert pending['final_scored_nodes']==0 and pending['phase_complete']['union']==0
    assert pending['keep']=={'P0':None,'P1':None}
    (case/'P0.json').write_text(json.dumps({'rubric_bank':{'witness':[]}}))
    source_done=monitor.snapshot()['pipeline']
    assert source_done['keep']=={'P0':0,'P1':None}
    assert source_done['phase_complete']=={'P0':1,'P1':0,'union':0}
    union={'tree':{'nodes':[{'node_id':'n1'}]},
           'response_scores':[{'response_node_id':'n1','score':None}],
           'agent_targets':{'nodes':{'n1':{'reward':None}}}}
    (case/'union.json').write_text(json.dumps(union))
    snapshot=monitor.snapshot();confirmed=snapshot['pipeline']
    assert confirmed['final_scored_nodes']==confirmed['final_nodes_total']==1
    assert confirmed['final_valid_rewards']==0 and confirmed['final_na_rewards']==1
    assert confirmed['phase_complete']['union']==1
    assert monitor.snapshot()['pipeline']==confirmed
    assert '최종 학습채점 확정 1/1' in render(snapshot)


def test_doctor_progress_moves_while_leaf_count_stays_fixed(tmp_path,monkeypatch):
    import scripts.online_rl.watch_five_gpu as watch
    monkeypatch.setattr(watch,'gpu_sample',lambda:({},None))
    batch=tmp_path/'batches/batch_000001';case=batch/'case_000000';case.mkdir(parents=True)
    (batch/'manifest.json').write_text(json.dumps({'assignments':[{'worker':0,'index':0}],'case_ids':['c1']}))
    path=case/'calls.jsonl'
    rows=[{'kind':'agent','identity':{'path':str(i)}} for i in range(2)]
    rows += [{'kind':'doctor_turn','identity':{'path':''}},
             {'kind':'doctor_turn','identity':{'path':'0'}},
             {'kind':'rerun_doctor_turn','identity':{'path':'1'}}]
    path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    cfg={'five_gpu':{'rollout_gpus':[0]},'rollout':{'branching':2,'rounds':2},'schedule':{'cases':500}}
    monitor=Monitor(tmp_path,cfg);before=monitor.snapshot()
    assert before['gpus'][0]['depth_doctor_done']==1
    assert 'Doctor 완료 1 (최대 2) | FINAL 완료 0' in render(before)
    with path.open('a') as f:
        f.write(json.dumps({'kind':'doctor_turn','identity':{'path':'1'}})+'\n')
    after=monitor.snapshot()
    assert after['leaves']==before['leaves']
    assert after['gpus'][0]['depth_doctor_done']==2
    assert 'Doctor 완료 2 (최대 2) | FINAL 완료 0' in render(after)
    assert monitor.snapshot()['gpus'][0]['depth_doctor_done']==2
