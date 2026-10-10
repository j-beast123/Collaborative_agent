"""Two Doctor GPUs + two Assistant GPUs; fixed five-case on-policy batches.

Reuses original tree/worker/GRPO/checkpoint/evaluator functions. Execution topology
is independent of batch_slots; no per-worker optimizer updates or async learning.
"""
import copy
import fcntl
import json
import multiprocessing
import os
from contextlib import nullcontext
from concurrent.futures import ProcessPoolExecutor,ThreadPoolExecutor
from pathlib import Path
from src.online_rl.integrated.records import atomic_write_json,sha256_json
from src.online_rl.tree.five_gpu import (EvaluationQueue,load_checkpoint,save_checkpoint,training_order,update_batch,wait_with_evaluation)
from src.online_rl.tree.gpu_pool import create_pool,assert_idle
from baselines.common.protocol import ROOT
from baselines.common import four_gpu_worker as worker

experiment=None  # injected per isolated runner instance


def batch_slots(cursor,total):
    if cursor%5 or total%5 or not 0<=cursor<=total:raise ValueError('Five-case aligned schedule required')
    return [{'index':i,'worker':(i-cursor)%2,'lane':0} for i in range(cursor,min(cursor+5,total))]


def preflight(cfg):
    # Reuse original checks on the scientific recipe; topology itself is checked separately.
    from scripts.online_rl import run_five_gpu as original
    old=copy.deepcopy(cfg)
    old['five_gpu'].update(rollout_gpus=list(range(5)),update_gpu=4,doctor_urls=[f'http://127.0.0.1:{11900+i}' for i in range(5)])
    import importlib.util
    spec=importlib.util.spec_from_file_location('comparison_recipe_check',ROOT/'scripts/online_rl/run_five_gpu.py')
    checks=importlib.util.module_from_spec(spec);spec.loader.exec_module(checks)
    checks.physical_gpus=lambda:list(map(str,range(5)))
    errors=checks.preflight(old)
    t=cfg['four_gpu'];fg=cfg['five_gpu']
    devices=t['doctor_physical_gpus']+t['assistant_physical_gpus']
    if len(devices)!=4 or len(set(devices))!=4:errors.append('Four distinct physical devices required')
    if len(t['doctor_physical_gpus'])!=2 or len(t['assistant_physical_gpus'])!=2:errors.append('Need Doctor2 + Assistant2')
    if fg['batch_cases']!=5 or fg['rollout_gpus']!=[0,1] or fg['update_gpu']!=1 or len(fg['doctor_urls'])!=2:errors.append('Four-GPU topology differs')
    if cfg['prompt_update']['enabled'] or cfg['rubric']['enabled']:errors.append('Original SHERPA pipelines must remain disabled')
    if t.get('eval_workers')!=10:errors.append('Use fixed 10-case evaluation concurrency on two actors')
    return errors


def require_assistant_cuda():
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count()!=2:raise RuntimeError('Expose exactly two Assistant GPUs to the Python runner')


def run(cfg,max_batches=None,eval_only=False):
    import torch
    from scripts.online_rl.run_tree import load_key
    from baselines.common.policy_loader import load_policy
    from src.online_rl.integrated.grpo import make_updater
    from src.online_rl.integrated.continuation import adapter_sha256,protocol_sha256
    from src.online_rl.cases import load_cases
    from src.online_rl.environment import final_outcome
    from src.common.live_gpu import configure,activity,note
    require_assistant_cuda()
    errors=preflight(cfg)
    if errors:raise RuntimeError('; '.join(errors))
    fg=cfg['five_gpu'];out=Path(cfg['output_dir']);out.mkdir(parents=True,exist_ok=True)
    evaluation_enabled=fg['checkpoint_evaluation_enabled']
    if eval_only and not evaluation_enabled:raise RuntimeError('Evaluation disabled')
    lock=(out/'runner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    executors=[];manager=pool=parent_actor=policy=None
    try:
        os.environ['SHERPA_AGENT_REPLICAS']='1';os.environ.pop('SHERPA_DOCTOR_ENDPOINTS',None)
        os.environ['API_USAGE_LOG']=str(out/'api_usage.jsonl');os.environ['SHERPA_API_WORKERS']=str(fg.get('api_workers',3))
        atomic_write_json(out/'execution_topology.json',{'topology':cfg['four_gpu'],'doctor_urls':fg['doctor_urls'],
            'assistant_cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'torch':torch.__version__,'cuda':torch.version.cuda})
        key=load_key();local=copy.deepcopy(cfg);local['rollout']['doctor_backend']['endpoint']=fg['doctor_urls'][1]
        torch.manual_seed(cfg['schedule']['seed']);configure(out,1)
        with activity('model_load'):policy=load_policy(local,key,device='cuda:1')
        updater=make_updater(policy,cfg);theta0=adapter_sha256(policy.model)
        protocol=sha256_json({'base':protocol_sha256(cfg,ROOT),'D1':fg['opening_identity'],
            'mode':fg['opening_mode'],'na':'exclude_unassessed_v1','execution_mode':'shared_ready_tasks_v1','batch_cases':5})
        experiment.check_pfinal_source(cfg,theta0,protocol)
        normalized=copy.deepcopy(cfg);normalized['five_gpu']['checkpoint_evaluation_enabled']=False
        config_sha=sha256_json(normalized);path=out/'run_state.json'
        if path.exists():
            state=json.loads(path.read_text())
            if state['config_sha256']!=config_sha or state['protocol_sha256']!=protocol:raise RuntimeError('Resume config/protocol differs')
            experiment.check_resume(cfg,out,state,theta0);load_checkpoint(updater,state['checkpoint'])
            if adapter_sha256(policy.model)!=state['policy_sha256']:raise RuntimeError('Resume adapter differs')
        else:
            if eval_only:raise RuntimeError('Evaluation requires completed training')
            if (out/'checkpoints').exists():raise RuntimeError('Checkpoint artifacts without run state')
            initial=out/'checkpoints/checkpoint_000000'
            save_checkpoint(updater,initial,{'completed_cases':0,'policy_sha256':theta0,'protocol_sha256':protocol,**experiment.receipt_fields(cfg)})
            state={'case_cursor':0,'batch_cursor':0,'config_sha256':config_sha,'protocol_sha256':protocol,
                'policy_sha256':theta0,'checkpoint':str(initial),**experiment.state_fields(cfg,theta0)}
            atomic_write_json(path,state)
        atomic_write_json(out/'run_manifest.json',{'config':cfg,'protocol_sha256':protocol,'execution':'Doctor2_Assistant2_five_case_barrier_v1'})
        cases=load_cases(ROOT/cfg['data']['train']);byid={c['case_id']:c for c in cases};order=training_order(cfg,cases,ROOT)
        ctx=multiprocessing.get_context('spawn');manager=ctx.Manager()
        pool=create_pool(manager,[0,1],fg['agent_batch_size'],fg['doctor_concurrency'])
        worker.initialize(cfg,1,key,pool,existing_policy=policy);parent_actor=policy.pool_actor
        executors=[ProcessPoolExecutor(max_workers=1,mp_context=ctx,initializer=worker.initialize,initargs=(cfg,0,key,pool)),
            ThreadPoolExecutor(max_workers=1,thread_name_prefix='assistant1-case')]
        def prepare(expected,checkpoint):
            assert_idle(pool)
            wait_with_evaluation([e.submit(worker.prepare,checkpoint,expected) for e in executors],lambda:False)
            if adapter_sha256(policy.model)!=expected:raise RuntimeError('Parent snapshot differs')
        if eval_only:
            if state['case_cursor']!=cfg['schedule']['cases']:raise RuntimeError('Final-only evaluation before completed training')
            tests=load_cases(ROOT/fg['eval_file'])
            queue=EvaluationQueue(out/'evaluation',tests,{'seed':fg['eval_seed'],'data_sha256':fg['eval_sha256'],
                'protocol_sha256':protocol,'dataset':'CasiMedicos all_en_622'})
            queue.register(state['checkpoint'],state['case_cursor'],state['policy_sha256'])
            def play(manifest,case,seed):
                episode=policy.generate(case,seed)
                return {'dialogue':episode['dialogue'],'final_option':episode['final_option'],
                    'num_discussion_rounds':episode['num_discussion_rounds'],'outcome':final_outcome(case,episode['final_option'])}
            queue.run_all_parallel(play,workers=cfg['four_gpu']['eval_workers'],
                before_job=lambda m:prepare(m['policy_sha256'],m['checkpoint']))
            assert_idle(pool);return
        if evaluation_enabled:raise RuntimeError('Training never evaluates test')
        completed=0
        while state['case_cursor']<cfg['schedule']['cases'] and (max_batches is None or completed<max_batches):
            batch=state['batch_cursor']+1;batchdir=out/'batches'/f'batch_{batch:06d}'
            slots=batch_slots(state['case_cursor'],cfg['schedule']['cases'])
            jobs=[{**s,'case':byid[order[s['index']]],'out':str(batchdir/f'case_{s["index"]:06d}')} for s in slots]
            manifest={'source_policy_sha256':state['policy_sha256'],'assignments':slots,'case_ids':[j['case']['case_id'] for j in jobs]}
            manifestpath=batchdir/'manifest.json'
            if manifestpath.exists() and json.loads(manifestpath.read_text())!=manifest:raise RuntimeError('Pending batch differs')
            atomic_write_json(manifestpath,manifest)
            prepare(state['policy_sha256'],state['checkpoint'])
            print(json.dumps({'stage':'rollout','batch':batch,'cases':5,'assistant_workers':2}),flush=True)
            futures=[e.submit(worker.run_many,[j for j in jobs if j['worker']==w],state['checkpoint'],state['policy_sha256']) for w,e in enumerate(executors)]
            collected=wait_with_evaluation(futures,lambda:False)
            records=sorted([r for rows in collected for r in rows],key=lambda r:r['case_cursor'])
            assert_idle(pool)
            # Both coordinators/actors are now idle; no current-policy calls may
            # overlap a temporary reference adapter or the optimizer update.
            with parent_actor.changing_policy():pass
            reference=getattr(experiment.engine,'reference',None)
            context=reference.at_barrier(executors[0],state['policy_sha256'],pool) if reference else nullcontext()
            with context:reward_rows=experiment.check_batch(cfg,records)
            assert_idle(pool)
            if reward_rows:atomic_write_json(out/'reward_components'/f'batch_{batch:06d}.json',reward_rows)
            digest=sha256_json(records);cursor=state['case_cursor']+len(records)
            milestone=cursor%fg['checkpoint_every_cases']==0 or cursor==cfg['schedule']['cases']
            ckpt=out/'checkpoints'/(f'checkpoint_{cursor:06d}' if milestone else f'recovery_{batch:06d}')
            if ckpt.exists():
                proof=json.loads((ckpt/'COMPLETE.json').read_text());experiment.check_receipt(cfg,proof,'Recovery checkpoint')
                if proof['records_sha256']!=digest or proof['source_policy_sha256']!=state['policy_sha256']:raise RuntimeError('Recovery batch differs')
                load_checkpoint(updater,ckpt);update=proof['update']
            else:
                print(json.dumps({'stage':'agent_update','batch':batch,'cases':5,'physical_gpu':cfg['four_gpu']['assistant_physical_gpus'][1]}),flush=True)
                note(phase='agent_update',batch=batch)
                with activity('agent_update'):update=update_batch(updater,records,state['policy_sha256'],lambda:adapter_sha256(policy.model))
                proof={'records_sha256':digest,'source_policy_sha256':state['policy_sha256'],'policy_sha256':adapter_sha256(policy.model),
                    'completed_cases':cursor,'protocol_sha256':protocol,'update':update,**experiment.receipt_fields(cfg)}
                save_checkpoint(updater,ckpt,proof)
            previous=Path(state['checkpoint'])
            state={**state,'case_cursor':cursor,'batch_cursor':batch,'checkpoint':str(ckpt),'policy_sha256':adapter_sha256(policy.model)}
            for record in records:
                record['batch_update']=update;atomic_write_json(out/'cases'/f'case_{record["case_cursor"]:06d}.json',record)
            atomic_write_json(path,state)
            if previous.name.startswith('recovery_'):
                import shutil
                shutil.rmtree(previous)
            completed+=1
            print(json.dumps({'stage':'batch_committed','completed_cases':cursor,'batch':batch}),flush=True)
    finally:
        for e in executors:e.shutdown(wait=True,cancel_futures=True)
        if parent_actor:parent_actor.close()
        if policy is not None and hasattr(policy,'_batcher'):policy._batcher.close()
        if manager:manager.shutdown()
        if pool:
            import shutil
            shutil.rmtree(pool['directory'],ignore_errors=True)
        lock.close()
