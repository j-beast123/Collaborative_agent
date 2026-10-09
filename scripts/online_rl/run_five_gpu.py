"""Five-case shared ready-work pool; GPU4 updates after the complete batch barrier."""
import argparse
import copy
import fcntl
import json
import multiprocessing
import os
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.online_rl.integrated.records import atomic_write_json, sha256_json
from src.online_rl.tree.five_gpu import (EvaluationQueue, batch_slots, cache_openings,
    evaluation_adapter, load_checkpoint, save_checkpoint, training_order, update_batch, wait_with_evaluation)
from src.online_rl.tree import experiment
from src.online_rl.tree.devices import physical_gpus, record_runtime


def preflight(cfg):
    from scripts.online_rl.run_tree import blockers, PROMPT_HASHES, file_sha
    from src.online_rl.cases import load_cases, file_sha256
    errors = blockers(cfg)
    for section, file, digest in PROMPT_HASHES:
        if cfg[section][digest] != file_sha(cfg[section][file]):
            errors.append(f'Prompt hash differs: {section}.{file}')
    for section, path_key, hash_key in ((cfg['data'],'train','train_sha256'),
                                       (cfg['five_gpu'],'eval_file','eval_sha256')):
        path = ROOT/section[path_key]
        if not path.exists(): errors.append(f'Data missing: {path}')
        elif file_sha256(path) != section[hash_key]: errors.append(f'Data hash differs: {path}')
    fg = cfg['five_gpu']
    from src.online_rl.tree.gpu_pool import MODE as POOL_MODE
    shared = fg.get('execution_mode') == POOL_MODE
    if cfg['prompt_update'].get('candidate_seed') not in (None,'first_completed_P0_v1'):
        errors.append('Unknown common P1 seed mode')
    if cfg['prompt_update'].get('candidate_seed') and not shared:
        errors.append('First-completed P0 pipeline requires shared five-GPU execution')
    batch_cases = 5 if shared else 4
    if shared and fg.get('base_evaluation', {}).get('enabled'):
        errors.append('Shared five-GPU checkpoint evaluation cannot run BASE evaluation concurrently')
    if shared and (fg.get('agent_batch_size') != 4 or fg.get('doctor_concurrency') != 1 or
                   os.environ.get('SHERPA_AGENT_BATCH_SIZE','4') != '4' or os.environ.get('SHERPA_DOCTOR_CONCURRENCY','1') != '1'):
        errors.append('Shared mode requires the measured Agent4 / Doctor1 configuration')
    if fg['rollout_gpus'] != (list(range(5)) if shared else [0,1,2,3]) or fg['update_gpu'] != 4 or fg['lanes_per_gpu'] != 1 or fg.get('batch_cases') != batch_cases:
        errors.append(f'Need {batch_cases} case coordinators, one lane each, updater GPU4')
    if fg['checkpoint_every_cases'] != (30 if shared else 24) or len(fg['doctor_urls']) != 5 or len(set(fg['doctor_urls'])) != 5:
        errors.append('Need the declared snapshot interval and five distinct Doctor endpoints')
    if cfg.get('evaluation'):
        errors.append('Use the asynchronous evaluation queue in this runner')
    if cfg['prompt_update']['enabled']:
        from src.online_rl.tree.batch_prompt import MODE, FIVE_MODE
        pu=cfg['prompt_update']
        if pu.get('mode')!=(FIVE_MODE if shared else MODE) or pu.get('candidate_count')!=1 or pu.get('refinement_depth')!=1 or pu.get('every_n_cases')!=batch_cases:
            errors.append(f'Batch prompt evolution needs one P1 candidate, one refinement, every {batch_cases} cases')
        if cfg['acceptance'].get('rule')!='strictly_higher_product_v1':
            errors.append('Batch prompt acceptance must use the strictly higher directional product')
        if pu['reruns_per_intervention']!=cfg['intervention']['reruns_per_intervention']:
            errors.append('P0/P1 must use the same training rerun count')
        verification = cfg.get('verification', {})
        mode = verification.get('mode')
        if mode == 'all_members_witness_v1' and verification.get('max_valid_supports') is None:
            pass
        elif (mode == 'random10_witness_v1' and verification.get('max_valid_supports') == 10 and
              verification.get('support_sampling') == 'uniform_random_without_replacement_v1'):
            pass
        else:
            errors.append('Batch prompt evolution needs all-member witness or reproducible random-10 same-ID witness verification')
        expected_aggregation=('mean_sampled_witness_effect_per_direction_product_v1'
                              if mode=='random10_witness_v1' else 'mean_trajectory_effect_per_direction_product_v1')
        required={'aggregation':expected_aggregation,'tie':'keep_parent',
                  'missing_direction':'keep_parent','training_rubrics':'union_both_kept_deduplicated_v1'}
        if any(pu.get(k)!=v for k,v in required.items()):
            errors.append('Batch prompt aggregation, tie, missing-direction or union policy differs')
    if cfg['schedule']['cases']%batch_cases:
        errors.append(f'Training case count must be a multiple of {batch_cases}')
    opening=fg.get('opening_identity',{})
    if opening.get('system_prompt_sha256')!=cfg['rollout'].get('doctor_system_prompt_sha256'):
        errors.append('D1 identity must use the current Doctor prompt hash')
    if opening.get('options')!={**cfg['rollout']['doctor_backend']['options'],
                                'num_predict':cfg['rollout']['max_tokens_doctor']}:
        errors.append('D1 identity must use the current Doctor generation options')
    if cfg['rollout'].get('doctor_opening') or fg.get('opening_mode') != 'immutable_runtime_D1_v1':
        errors.append('This runner uses the declared immutable runtime D1 protocol')
    if cfg['rollout']['halting']['min_agent_turns'] not in (2,3):
        errors.append('Minimum Agent turns must be two or three')
    if (ROOT/cfg['data']['train']).exists() and (ROOT/fg['eval_file']).exists():
        train = load_cases(ROOT/cfg['data']['train']); test = load_cases(ROOT/fg['eval_file'])
        if len(test) != fg['eval_cases'] or len({c['case_id'] for c in test}) != len(test):
            errors.append('Evaluation size or unique case IDs differ')
        def question(c): return ' '.join(c['question'].split()).casefold()
        if {question(c) for c in train} & {question(c) for c in test}:
            errors.append('Training/evaluation question overlap')
        from src.online_rl.guards import assert_not_protected
        try: assert_not_protected(train, 'five-GPU training')
        except (ValueError, RuntimeError) as e: errors.append(str(e))
        try: training_order(cfg,train,ROOT)
        except (ValueError,OSError) as e: errors.append(str(e))
    try: physical_gpus()
    except ValueError as e: errors.append(str(e))
    return errors


def prompt_round(batch,jobs,state,collect,store,backend,cache,history,during_proposal=None,pipeline=None):
    """One common P1; optionally start from the first P0 while remaining P0 cases run."""
    from src.online_rl.tree.batch_prompt import choose, propose_candidate, MODE
    parent={'text':store.get(state['prompt_sha256']),'sha256':state['prompt_sha256'],'id':state['prompt_id']}
    print(json.dumps({'stage':'prompt_P0','batch':batch,'prompt_sha256':parent['sha256']}),flush=True)
    propose=lambda rows:propose_candidate(rows,parent['text'],store,backend,cache,[j['case'] for j in jobs],history)
    seed=None;records=None
    if pipeline:
        result=pipeline(parent,propose)
        p0,c1,candidate,seed=result[:4]
        if len(result)==5:records=result[4]
        p1={'text':candidate['text'],'sha256':candidate['sha256'],'id':candidate['id']}
    else:
        p0=collect('P0',parent)
        fn=lambda:propose(p0)
        candidate=during_proposal(fn) if during_proposal else fn()
        p1={'text':candidate['text'],'sha256':candidate['sha256'],'id':f'B{batch}-P1-{candidate["sha256"][:12]}'}
        print(json.dumps({'stage':'prompt_P1','batch':batch,'prompt_sha256':p1['sha256']}),flush=True)
        c1=collect('P1',p1)
    decision=choose(p0,c1)
    if not candidate['valid_candidate']:
        decision.update(accepted=False,winner='P0',reason='no_valid_distinct_candidate')
    winner=p1 if decision['accepted'] else parent
    if records is None:records=collect('union',{**p1,'parent_sha256':parent['sha256']})
    next_prompt={'id':winner['id'],'sha256':winner['sha256']}
    event={'mode':store.batch_mode if hasattr(store,'batch_mode') else MODE,'batch':batch,'case_ids':[j['case']['case_id'] for j in jobs],
           'rerun_policy_sha256':state['policy_sha256'],'parent_prompt':{k:parent[k] for k in ('id','sha256')},
           'candidate_prompt':{k:p1[k] for k in ('id','sha256')},'proposal':candidate['proposal'],
           'decision':decision,'next_prompt':next_prompt,'training_pool':'union_both_kept_deduplicated_v1'}
    if seed: event.update(candidate_seed=seed,selection_scope='all_batch_trajectories_v1')
    return records,event


def run(cfg, max_batches=None, eval_only=False):
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() < 5:
        raise RuntimeError('CUDA GPUs 0..4 must be visible')
    from scripts.online_rl.run_tree import load_key
    from src.online_rl.runtime import SharedPolicy
    from src.online_rl.integrated.grpo import make_updater
    from src.online_rl.integrated.continuation import adapter_sha256, protocol_sha256
    from src.online_rl.cases import load_cases
    from src.online_rl.environment import final_outcome
    from src.online_rl.tree.team import RUBRIC_ONLY_NA_POLICY
    from src.online_rl.tree.effectiveness import RERUN_CONTINUATION_PROTOCOL, RERUN_OUTCOME_POLICY
    from src.online_rl.tree import five_gpu_worker as worker
    fg = cfg['five_gpu']; out = (ROOT/cfg['output_dir']).resolve(); out.mkdir(parents=True,exist_ok=True)
    from src.online_rl.tree.gpu_pool import MODE as POOL_MODE, create_pool, assert_idle
    shared = fg.get('execution_mode') == POOL_MODE
    evaluation_enabled = fg.get('checkpoint_evaluation_enabled', True)
    if eval_only and not evaluation_enabled:
        raise RuntimeError('Evaluation is disabled in this training configuration')
    lock = (out/'runner.lock').open('a')
    try: fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError: raise RuntimeError('Another runner owns this output directory/GPU4')
    # Lock the updater's physical GPU.
    gpu_lock = Path(f"/tmp/sherpa_tree_physical_gpu{physical_gpus()[fg['update_gpu']]}.lock").open('a')
    try: fcntl.flock(gpu_lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close(); gpu_lock.close()
        raise RuntimeError('GPU4 is already owned by another five-GPU runner')
    # Keep both locks open throughout training and evaluation.
    os.environ['SHERPA_AGENT_REPLICAS'] = '1'
    os.environ.pop('SHERPA_DOCTOR_ENDPOINTS', None)
    os.environ['API_USAGE_LOG'] = str(out/'api_usage.jsonl')
    os.environ['SHERPA_API_WORKERS'] = str(fg.get('api_workers',3))
    record_runtime(out,cfg)
    key = load_key()
    local = copy.deepcopy(cfg)
    local['rollout']['doctor_backend']['endpoint'] = fg['doctor_urls'][4]
    torch.manual_seed(cfg['schedule']['seed'])
    from src.common.live_gpu import configure, activity, note
    configure(out,4)
    with activity('model_load'):
        policy = SharedPolicy(local,key,device='cuda:4')
    problems = policy.doctor.verify()['problems']
    if problems: raise RuntimeError('GPU4 Doctor identity: '+'; '.join(problems))
    if not shared:
        cache_openings(policy.doctor,out/'openings/eval.jsonl',fg['opening_identity'])
    updater = make_updater(policy,cfg)
    # Enabling checkpoint evaluation is a safe runtime-only resume change: the
    # trained policy, data, optimizer and rollout protocol remain identical.
    resume_cfg = copy.deepcopy(cfg)
    resume_cfg.setdefault('five_gpu', {})['checkpoint_evaluation_enabled'] = False
    config_sha = sha256_json(resume_cfg)
    evolving=cfg['prompt_update']['enabled']
    if evolving:
        from src.online_rl.integrated.prompt_search import PromptStore, sol_backend
        from src.online_rl.tree.cache import CallCache
        store=PromptStore(out/'prompts')
        store.batch_mode = cfg['prompt_update']['mode']
        initial_prompt=store.put((ROOT/cfg['rubric']['generator_prompt_file']).read_text())
        if initial_prompt!=cfg['rubric']['prompt_sha256']:raise RuntimeError('Initial generator prompt hash mismatch')
        pu=cfg['prompt_update']
        backend=sol_backend(key,(ROOT/pu['feedback_prompt_file']).read_text(),(ROOT/pu['refiner_prompt_file']).read_text(),
                            None,None,None,reflection_model=pu['reflection_model'],refiner_model=pu['refiner_model'])
        prompt_cache=CallCache(out/'prompt_calls.jsonl')
    protocol = sha256_json({'base':protocol_sha256(cfg,ROOT),'D1':fg['opening_identity'],
                            'mode':fg['opening_mode'],'na':RUBRIC_ONLY_NA_POLICY,
                            **({'execution_mode':POOL_MODE,'batch_cases':5} if shared else {})})
    path = out/'run_state.json'
    theta0 = adapter_sha256(policy.model)
    experiment.check_pfinal_source(cfg,theta0,protocol)
    if path.exists():
        state = json.loads(path.read_text())
        if state['config_sha256'] != config_sha or state['protocol_sha256'] != protocol:
            raise RuntimeError('Run configuration/protocol changed; use a new output directory')
        experiment.check_resume(cfg,out,state,theta0)
        load_checkpoint(updater,state['checkpoint'])
        if adapter_sha256(policy.model) != state['policy_sha256']: raise RuntimeError('Resume adapter mismatch')
        manifest_path = out/'run_manifest.json'
        if manifest_path.exists():
            run_manifest = json.loads(manifest_path.read_text())
            old_enabled = run_manifest.get('config', {}).get('five_gpu', {}).get('checkpoint_evaluation_enabled', False)
            if old_enabled != evaluation_enabled:
                run_manifest.setdefault('runtime_changes', []).append({
                    'checkpoint_evaluation_enabled': {'from': old_enabled, 'to': evaluation_enabled},
                    'note': 'Evaluation-only change; training config identity is normalized to disabled.'})
                run_manifest['config'] = cfg
                atomic_write_json(manifest_path, run_manifest)
    else:
        if eval_only: raise RuntimeError('--eval-only requires an existing run')
        if (out/'checkpoints').exists(): raise RuntimeError('Checkpoint artifacts without committed run state')
        initial = out/'checkpoints/checkpoint_000000'
        sha = adapter_sha256(policy.model)
        save_checkpoint(updater,initial,{'completed_cases':0,'policy_sha256':sha,'protocol_sha256':protocol,
                                         **experiment.receipt_fields(cfg)})
        state = {'case_cursor':0,'batch_cursor':0,'config_sha256':config_sha,'protocol_sha256':protocol,
                 'policy_sha256':sha,'checkpoint':str(initial),**experiment.state_fields(cfg,theta0)}
        if evolving:state.update(prompt_sha256=initial_prompt,prompt_id=cfg['rubric']['prompt_id'])
        atomic_write_json(out/'run_manifest.json',{'config':cfg,'protocol_sha256':protocol,
                            'rubric_only_na_policy':RUBRIC_ONLY_NA_POLICY,'rerun_outcome_policy':RERUN_OUTCOME_POLICY,
                            'rerun_continuation_protocol':RERUN_CONTINUATION_PROTOCOL})
        atomic_write_json(path,state)
    eval_cases = load_cases(ROOT/fg['eval_file']) if evaluation_enabled or fg.get('base_evaluation',{}).get('enabled') else []
    base_evaluation=None
    if fg.get('base_evaluation',{}).get('enabled'):
        from src.online_rl.tree.base_evaluation import BaseEvaluation
        base_evaluation=BaseEvaluation(out/'evaluation/BASE_raw_no_precommit',eval_cases,policy,cfg)
    queue = EvaluationQueue(out/'evaluation',eval_cases,{'seed':fg['eval_seed'],
        'data_sha256':fg['eval_sha256'],'protocol_sha256':protocol,'dataset':'CasiMedicos all_en_622'}) if evaluation_enabled else None
    for ckpt in sorted((out/'checkpoints').glob('checkpoint_*')):
        receipt = json.loads((ckpt/'COMPLETE.json').read_text())
        cursor = receipt['completed_cases']
        should_evaluate = cursor > 0 and (cursor % fg['checkpoint_every_cases'] == 0 or cursor == cfg['schedule']['cases'])
        if queue is not None and should_evaluate and cursor <= state['case_cursor']:
            queue.register(ckpt,receipt['completed_cases'],receipt['policy_sha256'])

    def play(manifest,case,seed):
        if shared:
            policy.pool_client.context = {'case_id':case['case_id'], 'phase':f"eval_{manifest['completed_cases']}"}
            episode = policy.generate(case,seed)
        else:
            with evaluation_adapter(policy,manifest['checkpoint']):
                if adapter_sha256(policy.model) != manifest['policy_sha256']:
                    raise RuntimeError('Evaluation checkpoint adapter mismatch')
                episode = policy.generate(case,seed)
        return {'dialogue':episode['dialogue'],'final_option':episode['final_option'],
                'num_discussion_rounds':episode['num_discussion_rounds'],
                'outcome':final_outcome(case,episode['final_option'])}

    def drain_checkpoint_evaluations():
        if queue is None:
            return False
        if shared:
            if len(executors) != 5 or parent_actor is None:
                raise RuntimeError('Shared checkpoint evaluation requires all five GPU actors')
            def load_eval_snapshot(manifest):
                assert_idle(pool_spec)
                prepared=[executors[w].submit(worker.prepare,manifest['checkpoint'],manifest['policy_sha256']) for w in range(5)]
                wait_with_evaluation(prepared,lambda:False)
                if adapter_sha256(policy.model) != manifest['policy_sha256']:
                    raise RuntimeError('Evaluation policy differs from its committed checkpoint')
            def progress(manifest, done, total):
                print(json.dumps({'stage':'checkpoint_eval','checkpoint_cases':manifest['completed_cases'],
                                  'evaluated':done,'total':total,'gpus':fg['rollout_gpus']},ensure_ascii=False),flush=True)
            try:
                return queue.run_all_parallel(play,workers=25,progress=progress,before_job=load_eval_snapshot)
            finally:
                # Restore the latest committed training weights and actor hashes,
                # even when catching up an older interrupted evaluation fails.
                prepared=[executors[w].submit(worker.prepare,state['checkpoint'],state['policy_sha256']) for w in range(5)]
                wait_with_evaluation(prepared,lambda:False)
                if adapter_sha256(policy.model) != state['policy_sha256']:
                    raise RuntimeError('Failed to restore the current training checkpoint after evaluation')
                assert_idle(pool_spec)
        return queue.run_all_parallel(play,workers=1)
    def evaluate_one():
        if base_evaluation is not None and base_evaluation.step():return True
        return queue.step(play) if queue is not None else False
    executors = []
    manager = pool_spec = parent_actor = None
    try:
        if not eval_only or (shared and evaluation_enabled):
            cases = load_cases(ROOT/cfg['data']['train']); byid = {c['case_id']:c for c in cases}
            order = training_order(cfg,cases,ROOT)
            if shared:
                manager = multiprocessing.get_context('spawn').Manager()
                pool_spec = create_pool(manager, list(range(5)), fg['agent_batch_size'], fg['doctor_concurrency'])
                worker.initialize(cfg, 4, key, pool_spec, existing_policy=policy)
                parent_actor = policy.pool_actor
            for w in range(4):
                executors.append(ProcessPoolExecutor(max_workers=1,mp_context=multiprocessing.get_context('spawn'),
                                 initializer=worker.initialize,initargs=(cfg,w,key,pool_spec)))
            if shared:
                executors.append(ThreadPoolExecutor(max_workers=1,thread_name_prefix='gpu4-case'))
        if not eval_only:
            if queue is not None:drain_checkpoint_evaluations()
            completed = 0
            while state['case_cursor'] < cfg['schedule']['cases'] and (max_batches is None or completed < max_batches):
                batch = state['batch_cursor']+1
                batchdir = out/'batches'/f'batch_{batch:06d}'
                slots = batch_slots(state['case_cursor'],cfg['schedule']['cases'],
                                    interval=fg['checkpoint_every_cases'],
                                    workers=len(fg['rollout_gpus']),lanes=fg['lanes_per_gpu'])
                jobs = [{**s,'case':byid[order[s['index']]],
                         'out':str(batchdir/f'case_{s["index"]:06d}')} for s in slots]
                manifest={'source_policy_sha256':state['policy_sha256'],
                          'assignments':slots,'case_ids':[j['case']['case_id'] for j in jobs]}
                manifest_path=batchdir/'manifest.json'
                if manifest_path.exists():
                    if json.loads(manifest_path.read_text())!=manifest:
                        raise RuntimeError('Saved pending batch assignment/policy mismatch')
                else:
                    atomic_write_json(manifest_path,manifest)
                print(json.dumps({'stage':'rollout','batch':batch,'cases':len(jobs),'gpus':fg['rollout_gpus']}),flush=True)
                def collect(phase=None,prompt=None):
                    if shared:
                        assert_idle(pool_spec)
                        prepared=[executors[w].submit(worker.prepare,state['checkpoint'],state['policy_sha256']) for w in range(5)]
                        wait_with_evaluation(prepared,lambda:False)
                    futures=[]
                    for w in range(len(executors)):
                        assigned=[j for j in jobs if j['worker']==w]
                        if not assigned:continue
                        args=(assigned,state['checkpoint'],state['policy_sha256'])
                        if phase is not None:args+= (phase,prompt)
                        futures.append(executors[w].submit(worker.run_many,*args))
                    results=wait_with_evaluation(futures,lambda:False) if shared else wait_with_evaluation(futures,evaluate_one)
                    if shared:assert_idle(pool_spec)
                    return sorted([r for result in results for r in result],key=lambda r:r['case_cursor'])
                next_prompt=None
                if evolving:
                    def during_proposal(fn):
                        from concurrent.futures import ThreadPoolExecutor
                        with ThreadPoolExecutor(max_workers=1) as pool:
                            return wait_with_evaluation([pool.submit(fn)],lambda:False if shared else evaluate_one())[0]
                    history=[]
                    for p in sorted((out/'prompt_events').glob('batch_*.json')):
                        if int(p.stem.split('_')[-1])>=batch:continue
                        past=json.loads(p.read_text()); d=past['decision']
                        history.append({'batch':past['batch'],'accepted':d['accepted'],'reason':d['reason'],
                                        'parent_J':d['parent']['J'],'candidate_J':d['candidate']['J'],
                                        'winner_sha256':past['next_prompt']['sha256']})
                    pipeline=None
                    if cfg['prompt_update'].get('candidate_seed')=='first_completed_P0_v1':
                        if not shared:raise RuntimeError('First-completed P0 pipeline requires the shared GPU pool')
                        def pipeline(parent,propose):
                            from src.online_rl.tree.pipelined_prompt import evaluate
                            assert_idle(pool_spec)
                            prepared=[executors[w].submit(worker.prepare,state['checkpoint'],state['policy_sha256']) for w in range(5)]
                            wait_with_evaluation(prepared,lambda:False)
                            def submit(phase,job,prompt):
                                return executors[job['worker']].submit(worker.run_many,[job],state['checkpoint'],state['policy_sha256'],phase,prompt)
                            # run_many returns a list; adapt without blocking the completion callback.
                            from concurrent.futures import Future
                            def submit_one(phase,job,prompt):
                                result=Future()
                                def done(f):
                                    try:result.set_result(f.result()[0])
                                    except BaseException as exc:result.set_exception(exc)
                                submit(phase,job,prompt).add_done_callback(done)
                                return result
                            return evaluate(jobs,parent,submit_one,propose,batchdir,state['policy_sha256'],eager_union=True)
                    records,event=prompt_round(batch,jobs,state,collect,store,backend,prompt_cache,history,during_proposal,pipeline)
                    atomic_write_json(out/'prompt_events'/f'batch_{batch:06d}.json',event)
                    next_prompt=event['next_prompt']
                else:
                    records=collect()
                if shared:
                    assert_idle(pool_spec)
                    with parent_actor.changing_policy():pass
                reward_rows = experiment.check_batch(cfg,records)
                if reward_rows:
                    atomic_write_json(out/'reward_components'/f'batch_{batch:06d}.json',reward_rows)
                digest = sha256_json(records); cursor = state['case_cursor']+len(records)
                milestone = cursor%fg['checkpoint_every_cases'] == 0 or cursor == cfg['schedule']['cases']
                ckpt = out/'checkpoints'/(f'checkpoint_{cursor:06d}' if milestone else f'recovery_{batch:06d}')
                if ckpt.exists():
                    proof = json.loads((ckpt/'COMPLETE.json').read_text())
                    experiment.check_receipt(cfg,proof,'Recovery checkpoint')
                    if proof['records_sha256'] != digest or proof['source_policy_sha256'] != state['policy_sha256']:
                        raise RuntimeError('Recovery receipt differs from replayed batch')
                    if evolving and proof.get('next_prompt')!=next_prompt:
                        raise RuntimeError('Recovery receipt prompt winner differs')
                    load_checkpoint(updater,ckpt); update = proof['update']
                else:
                    print(json.dumps({'stage':'agent_update','gpu':4,'batch':batch,'cases':len(records)}),flush=True)
                    # Release the idle evaluation Doctor's VRAM before backward/optimizer work.
                    policy.doctor.http(fg['doctor_urls'][4]+'/api/generate',
                        {'model':cfg['rollout']['doctor_backend']['model'],'keep_alive':0},60)
                    note(phase='agent_update',batch=batch)
                    with activity('agent_update'):
                        update = update_batch(updater,records,state['policy_sha256'],lambda:adapter_sha256(policy.model))
                    proof = {'records_sha256':digest,'source_policy_sha256':state['policy_sha256'],
                        'policy_sha256':adapter_sha256(policy.model),'completed_cases':cursor,
                        'protocol_sha256':protocol,'update':update,**experiment.receipt_fields(cfg)}
                    if evolving:proof['next_prompt']=next_prompt
                    save_checkpoint(updater,ckpt,proof)
                previous = Path(state['checkpoint'])
                state = {**state,'case_cursor':cursor,'batch_cursor':batch,
                         'checkpoint':str(ckpt),'policy_sha256':adapter_sha256(policy.model)}
                if evolving:state.update(prompt_sha256=next_prompt['sha256'],prompt_id=next_prompt['id'])
                for record in records:
                    record['batch_update'] = update
                    atomic_write_json(out/'cases'/f'case_{record["case_cursor"]:06d}.json',record)
                atomic_write_json(path,state)
                if milestone and queue is not None:
                    queue.register(ckpt,cursor,state['policy_sha256'])
                    drain_checkpoint_evaluations()
                if previous.name.startswith('recovery_'):
                    import shutil
                    shutil.rmtree(previous)
                completed += 1
                print(json.dumps({'stage':'batch_committed','completed_cases':cursor,'batch':batch}),flush=True)
        # A deliberately bounded diagnostic returns promptly; normal training drains every queued snapshot.
        if eval_only or max_batches is None:
            if shared and queue is not None:
                drain_checkpoint_evaluations()
            else:
                while evaluate_one(): pass
    finally:
        for executor in executors: executor.shutdown(wait=True,cancel_futures=True)
        if parent_actor is not None:parent_actor.close()
        if manager is not None:manager.shutdown()
        if pool_spec is not None:
            import shutil
            shutil.rmtree(pool_spec['directory'],ignore_errors=True)
        gpu_lock.close()
        lock.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',default='configs/online_rl/tree_v21_five_gpu_shared5.json')
    parser.add_argument('--launch',action='store_true')
    parser.add_argument('--eval-only',action='store_true')
    parser.add_argument('--max-batches',type=int)
    args = parser.parse_args()
    cfg = experiment.load_config(args.config)
    errors = preflight(cfg)
    print(json.dumps({'training_cases':cfg['schedule']['cases'],'rollout_gpus':cfg['five_gpu']['rollout_gpus'],'lanes_per_gpu':cfg['five_gpu']['lanes_per_gpu'],'batch_cases':cfg['five_gpu']['batch_cases'],'update_gpu':4,
                      'checkpoint_every_cases':cfg['five_gpu']['checkpoint_every_cases'],'evaluation_cases':cfg['five_gpu']['eval_cases'] if cfg['five_gpu'].get('checkpoint_evaluation_enabled',True) else 0,
                      'execution_mode':cfg['five_gpu'].get('execution_mode','fixed_case_v1'),
                      'output_dir':cfg['output_dir'],'experiment':cfg.get('experiment'),
                      'blockers':errors,'launch':args.launch},ensure_ascii=False,indent=2),flush=True)
    if errors: raise SystemExit(1)
    if args.max_batches is not None and args.max_batches < 1: parser.error('--max-batches must be positive')
    if args.launch: run(cfg,args.max_batches,args.eval_only)

if __name__ == '__main__': main()
