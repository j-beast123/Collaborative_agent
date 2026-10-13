"""Ten single-trajectory early-stop diagnostics on GPUs 1-4; no training."""
import os,json,sys,copy,traceback,argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import multiprocessing
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.online_rl.integrated.records import atomic_write_json,to_dict


def initialize(cfg,worker,key,checkpoint,sha):
    from src.online_rl.tree import batch_worker as W
    W.initialize(cfg,worker,key);W.load_checkpoint(checkpoint,sha)


def run_one(job):
    from src.online_rl.tree import batch_worker as W
    from src.online_rl.tree.rollout import build_tree
    from src.online_rl.environment import final_outcome
    from src.online_rl.tree.backends import parse_final_option
    from src.online_rl.integrated.continuation import protocol_sha256
    from src.collaboration.generator import stop_reason
    from src.online_rl.tree.batched_generation import Request
    try:
        cfg=W.CTX['cfg'];policy=W.CTX['policy'];case=job['case'];out=Path(job['out']);out.mkdir(parents=True,exist_ok=True)
        cache,doctor,*_=W.components(case,out)
        class Agent:
            def sample(self,case,prefix,turn,seed):
                packet=None
                if cfg.get('belief',{}).get('enabled'):
                    from src.online_rl.tree.belief import prepare
                    prompt,packet=prepare(policy,case,prefix,turn)
                else:prompt=policy.agent_user_prompt(case,prefix,turn)
                chat=policy.ST.chat_prompt(policy.tok,policy.agent_system,prompt)
                trace=policy.batcher.submit(Request(chat,seed,policy.args.max_completion_length))
                return {**trace,'seed':seed,'prompt_text':chat,'pre_response_belief':packet}
        tree=build_tree(case,max_agent_turns=5,branching=1,run_seed=cfg['schedule']['seed'],policy_sha256=policy.fixed_sha,
             protocol_sha256=protocol_sha256(cfg,ROOT),agent=Agent(),doctor=doctor,stop_rule=stop_reason,
             final_option=lambda t:parse_final_option(t,dict(case['options'])) or '',outcome_fn=final_outcome,cache=cache)
        leaf=tree.leaves[0];record={'index':job['index'],'case_id':case['case_id'],'status':'complete','gpu':job['gpu'],
                 'agent_turns':len(leaf.path),'stop_reason':leaf.stop_reason,'outcome':leaf.outcome,'tree':to_dict(tree)}
        atomic_write_json(out/'result.json',record)
        return {k:v for k,v in record.items() if k!='tree'}
    except Exception as e:
        record={'index':job['index'],'case_id':job['case']['case_id'],'status':'incomplete','gpu':job['gpu'],'error':str(e)[:1000]}
        atomic_write_json(Path(job['out'])/'error.json',{**record,'traceback':traceback.format_exc()});return record


def main(output_dir="outputs/pilots/early_stop10_seed20261007"):
    import collections
    from scripts.online_rl.run_tree import load_key
    from scripts.online_rl.start_batch_doctors import start
    from src.online_rl.cases import load_cases
    from src.collaboration.generator import EARLY_STOP_VERSION
    source=(ROOT/'outputs/latest_training').resolve();out=(ROOT/output_dir).resolve()
    cfg=copy.deepcopy(json.loads((source/'run_manifest.json').read_text())['config'])
    manifest=json.loads((source/'batches/batch_000001/manifest.json').read_text())
    cfg['output_dir']=str(out.relative_to(ROOT));cfg['rollout']['branching']=1
    cfg['gpu']['rollout_devices']=['cuda:1','cuda:2','cuda:3','cuda:4']
    cfg['batch']['doctor_urls']=[f'http://127.0.0.1:{p}' for p in range(11546,11550)]
    cfg['batch']['doctor_gpu_ids']=[1,2,3,4];cfg['batch']['doctor_parallelism']=1
    key=load_key();out.mkdir(parents=True,exist_ok=True);os.environ['API_USAGE_LOG']=str(out/'api_usage.jsonl')
    cases={c['case_id']:c for c in load_cases(ROOT/cfg['data']['train'])}
    jobs=[{'index':i,'case':cases[cid],'gpu':1+i%4,'out':str(out/'cases'/f'case_{i:03d}')} for i,cid in enumerate(manifest['case_ids'][:10])]
    atomic_write_json(out/'manifest.json',{'config':cfg,'source_run':str(source),'checkpoint':manifest['checkpoint'],'policy_sha256':manifest['policy_sha256'],'stop_version':EARLY_STOP_VERSION,'doctor_prompt_text':(ROOT/'prompts/user_simulator.txt').read_text(),'doctor_prompt_sha256':__import__('hashlib').sha256((ROOT/'prompts/user_simulator.txt').read_bytes()).hexdigest(),'case_ids':[j['case']['case_id'] for j in jobs],'branching':1,'total':10,'agent_updates':False})
    start(cfg);ctx=multiprocessing.get_context('spawn');pools=[];futures=[];rows=[]
    try:
        for worker in range(4):
            pool=ProcessPoolExecutor(max_workers=1,mp_context=ctx,initializer=initialize,initargs=(cfg,worker,key,manifest['checkpoint'],manifest['policy_sha256']));pools.append(pool)
            futures += [pool.submit(run_one,j) for j in jobs if j['gpu']==worker+1]
        for future in as_completed(futures):
            row=future.result();rows.append(row);completed=[r for r in rows if r['status']=='complete'];counts=collections.Counter(r['agent_turns'] for r in completed)
            summary={'finished':len(rows),'total':10,'completed':len(completed),'incomplete':len(rows)-len(completed),
                     'average_agent_turns':sum(r['agent_turns'] for r in completed)/len(completed) if completed else None,
                     'turn_counts':dict(counts),'stop_reasons':dict(collections.Counter(r['stop_reason'] for r in completed)),'cases':sorted(rows,key=lambda r:r['index'])}
            atomic_write_json(out/'progress.json',summary);print(json.dumps(summary),flush=True)
        atomic_write_json(out/'summary.json',summary)
    finally:
        for pool in pools:pool.shutdown()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',default='outputs/pilots/early_stop10_seed20261007');args=parser.parse_args()
    try:main(args.out)
    except Exception as e:
        atomic_write_json((ROOT/args.out).resolve()/'error.json',{'error':str(e),'traceback':traceback.format_exc()});raise
