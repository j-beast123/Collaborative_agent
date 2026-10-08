"""Four rollout processes × three dialogue lanes, GPU4 GRPO batch barrier.
Dry run is the default. Checkpoints and immutable caches permit batch replay.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import copy
import json
import multiprocessing
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.online_rl.integrated.records import atomic_write_json,sha256_json
from src.online_rl.tree.batch import batch_slots,update_batch


def preflight(cfg):
    from scripts.online_rl.run_tree import blockers
    errors=blockers(cfg)
    # Output collisions against other configs are checked by the ordinary launcher.
    train=ROOT/cfg['data']['train']
    import hashlib
    if not train.exists():errors.append('Training data missing')
    elif not hashlib.sha256(train.read_bytes()).hexdigest().startswith(cfg['data']['train_sha256']):errors.append('Training data hash differs')
    if cfg['gpu']['rollout_devices']!=['cuda:0','cuda:1','cuda:2','cuda:3']:errors.append('Expected rollout GPUs 0..3')
    if cfg['gpu']['train_device']!='cuda:4':errors.append('Expected training GPU4')
    if cfg['batch']['cases_per_gpu']!=3:errors.append('Expected three dialogue lanes per GPU')
    if len(cfg['batch']['doctor_urls'])!=4:errors.append('Need four GPU-pinned local Doctor endpoints')
    if cfg['pilot'].get('backend')=='ollama':errors.append('Training requires traced HF Agent, not text-only GGUF')
    if cfg['intervention']['reruns_per_intervention']!=3:errors.append('Exactly three reruns required')
    if not isinstance(cfg['schedule']['cases'],int) or cfg['schedule']['cases']<1:errors.append('Positive number of cases required')
    if cfg['prompt_update']['every_n_cases']!=1:errors.append('Prompt optimization must be case-local')
    return errors


def run(cfg,max_batches=None):
    from scripts.online_rl.run_tree import load_key
    from src.online_rl.cases import load_cases
    from src.online_rl.guards import assert_not_protected
    from src.online_rl.runtime import build_train_policy
    from src.online_rl.integrated.grpo import make_updater
    from src.online_rl.integrated.continuation import adapter_sha256,protocol_sha256
    from src.online_rl.integrated.episode import schedule
    from src.online_rl.integrated.prompt_search import PromptStore
    from src.online_rl.tree import batch_worker as W
    import torch
    key=load_key();out=ROOT/cfg['output_dir'];out.mkdir(parents=True,exist_ok=True)
    os.environ['API_USAGE_LOG']=str(out/'api_usage.jsonl')
    statepath=out/'run_state.json';configsha=sha256_json(cfg);protocol=protocol_sha256(cfg,ROOT)
    cases=load_cases(ROOT/cfg['data']['train']);assert_not_protected(cases,'batch training cases');byid={c['case_id']:c for c in cases}
    if cfg['schedule']['cases']>len(byid):raise ValueError('This run samples cases without replacement')
    if not torch.cuda.is_available() or torch.cuda.device_count()<5:
        raise RuntimeError('A CUDA-enabled PyTorch environment with visible GPUs 0..4 is required')
    torch.manual_seed(cfg['schedule']['seed']);policy=build_train_policy(cfg,key);updater=make_updater(policy,cfg)
    store=PromptStore(out/'prompts')
    if statepath.exists():
        state=json.loads(statepath.read_text())
        if state['config_sha256']!=configsha or state['protocol_sha256']!=protocol:raise RuntimeError('Config or protocol changed; use a new output directory')
        updater.load(state['checkpoint'])
        if adapter_sha256(policy.model)!=state['policy_sha256']:raise RuntimeError('Checkpoint policy mismatch')
    else:
        if (out/'checkpoints').exists():raise RuntimeError('Checkpoints exist without committed run state')
        prompt=(ROOT/cfg['rubric']['generator_prompt_file']).read_text();sha=store.put(prompt)
        ckpt=updater.save(out/'checkpoints/recovery/batch_000000')
        state={'config_sha256':configsha,'protocol_sha256':protocol,'case_cursor':0,'batch_cursor':0,
               'policy_sha256':adapter_sha256(policy.model),'checkpoint':ckpt,'prompt_id':cfg['rubric']['prompt_id'],
               'prompt_sha256':sha,'history':[]}
        atomic_write_json(statepath,state);atomic_write_json(out/'run_manifest.json',{'config':cfg,'config_sha256':configsha,'protocol_sha256':protocol})
    if state['case_cursor'] and state['case_cursor']%cfg['checkpoint']['every_cases']==0:
        from src.online_rl.tree.checkpoints import save_periodic_snapshot
        save_periodic_snapshot(out,state,cfg['checkpoint']['every_cases'],store.get(state['prompt_sha256']))
    executors=[]
    try:
        for worker in range(4):
            executors.append(ProcessPoolExecutor(max_workers=1,mp_context=multiprocessing.get_context('spawn'),
                        initializer=W.initialize,initargs=(cfg,worker,key)))
        completed=0
        while state['case_cursor']<cfg['schedule']['cases'] and (max_batches is None or completed<max_batches):
            cursor=state['case_cursor'];batchid=state['batch_cursor']+1;sha=state['policy_sha256']
            # A short batch at each 30-case boundary preserves exact periodic snapshots.
            interval=cfg['checkpoint']['every_cases'];remaining=min(cfg['schedule']['cases']-cursor,interval-cursor%interval)
            slots=batch_slots(cursor,remaining);batchdir=out/'batches'/f'batch_{batchid:06d}';batchdir.mkdir(parents=True,exist_ok=True)
            jobs=[]
            for slot in slots:
                cid=schedule(sorted(byid),cfg['schedule']['seed'],slot.index,1)[0]
                jobs.append({'index':slot.index,'worker':slot.worker,'lane':slot.lane,'case':byid[cid],
                             'out':str(batchdir/'cases'/f'case_{slot.index:06d}')})
            atomic_write_json(batchdir/'manifest.json',{'policy_sha256':sha,'checkpoint':state['checkpoint'],
                            'case_indices':[j['index'] for j in jobs],'case_ids':[j['case']['case_id'] for j in jobs],
                            'assignments':[{'index':j['index'],'gpu':j['worker'],'lane':j['lane']} for j in jobs]})
            print(json.dumps({'stage':'rollout','batch':batchid,'cases':len(jobs),'policy_sha256':sha}),flush=True)
            futures=[executors[w].submit(W.rollout_many,[j for j in jobs if j['worker']==w],state['checkpoint'],sha)
                     for w in range(4) if any(j['worker']==w for j in jobs)]
            for future in futures:future.result()
            prompt_id=state['prompt_id'];prompt=store.get(state['prompt_sha256']);history=list(state['history']);records=[]
            for job in jobs:
                print(json.dumps({'stage':'case_prompt_optimization','batch':batchid,'case':job['index'],'prompt_id':prompt_id}),flush=True)
                record=executors[job['worker']].submit(W.evaluate,job,prompt_id,prompt,history,job['index']+1).result()
                record['batch_id']=batchid
                record['rollout_gpu']=job['worker']
                record['dialogue_lane']=job['lane']
                if record['policy_sha256']!=sha:raise RuntimeError('Source policy drifted inside batch')
                decision=record['prompt_event']['decision']
                if decision.get('status') in ('deferred','incomplete') and 'technical' in decision.get('reason',''):
                    raise RuntimeError('Technical prompt evaluation incomplete; batch left uncommitted for resume')
                records.append(record);prompt=record['next_prompt_text'];prompt_id=record['prompt_id'];store.put(prompt)
                history.append({'event':job['index']+1,'accepted':decision.get('accepted',False),'reason':decision.get('reason')})
            newcursor=cursor+len(jobs);checkpoint=out/'checkpoints/recovery'/f'batch_{batchid:06d}'
            digest=sha256_json(records)
            receipt=batchdir/'update.json'
            if checkpoint.exists():
                proof=json.loads((checkpoint/'batch_receipt.json').read_text())
                if proof['records_sha256']!=digest or proof['source_policy_sha256']!=sha:raise RuntimeError('Checkpoint does not match replayed batch')
                updater.load(str(checkpoint));update=proof['update']
            else:
                print(json.dumps({'stage':'agent_update','batch':batchid,'cases':len(records),'gpu':4}),flush=True)
                update=update_batch(updater,records,sha,lambda:adapter_sha256(policy.model))
                updater.completed_cases.update(r['case_id'] for r in records)
                temporary=checkpoint.with_name(checkpoint.name+'.pending')
                if temporary.exists():
                    import shutil
                    shutil.rmtree(temporary)
                updater.save(temporary)
                atomic_write_json(temporary/'batch_receipt.json',{'records_sha256':digest,'source_policy_sha256':sha,'update':update})
                temporary.rename(checkpoint)
            atomic_write_json(receipt,{'source_policy_sha256':sha,'new_policy_sha256':adapter_sha256(policy.model),'update':update,'cases':len(records)})
            state={**state,'case_cursor':newcursor,'batch_cursor':batchid,'policy_sha256':adapter_sha256(policy.model),
                   'checkpoint':str(checkpoint),'prompt_id':prompt_id,'prompt_sha256':store.put(prompt),'history':history}
            for record in records:
                atomic_write_json(out/'cases'/f'case_{record["case_cursor"]:06d}.json',record)
            atomic_write_json(statepath,state)
            if newcursor%interval==0:
                from src.online_rl.tree.checkpoints import save_periodic_snapshot
                save_periodic_snapshot(out,state,interval,prompt)
            print(json.dumps({'stage':'batch_committed','batch':batchid,'cases_completed':newcursor,'prompt_id':prompt_id,'update':update}),flush=True)
            completed+=1
    finally:
        for executor in executors:executor.shutdown(wait=True,cancel_futures=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/online_rl/tree_batch12_haiku_gemini_luna.json');p.add_argument('--launch',action='store_true');p.add_argument('--max-batches',type=int)
    args=p.parse_args();cfg=json.loads((ROOT/args.config).read_text());errors=preflight(cfg)
    print(json.dumps({'config':args.config,'rollout_gpus':[0,1,2,3],'cases_per_gpu':3,'max_batch_cases':12,'train_gpu':4,'blockers':errors},ensure_ascii=False,indent=2),flush=True)
    if errors:raise SystemExit(1)
    if args.launch:run(cfg,args.max_batches)

if __name__=='__main__':main()
