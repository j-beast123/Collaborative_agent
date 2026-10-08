"""Restart an uncommitted first batch under random-10 using its saved rollout caches."""
import copy
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.diagnostics.after_batch_doctor_bench import info,same
from scripts.diagnostics.restart_shared_pool import descendants,frozen_batch
from src.online_rl.integrated.continuation import protocol_sha256
from src.online_rl.integrated.records import atomic_write_json,sha256_json
from src.online_rl.tree.team import RUBRIC_ONLY_NA_POLICY
from scripts.online_rl.watch_five_gpu import leaf_progress
from src.online_rl.tree.cache import CallCache

OLD_CONFIG='configs/online_rl/tree_v21_five_gpu_shared5.json'
BRIDGE_CONFIG='configs/online_rl/tree_v21_five_gpu_shared5_random10_reuse_active.json'
NEXT_CONFIG='configs/online_rl/tree_v21_five_gpu_shared5_halt4_visible_pipeline_random10.json'
STATUS=ROOT/'outputs/diagnostics/doctor_parallel_20261008/handoff_status.json'


def composite_protocol(cfg):
    return sha256_json({'base':protocol_sha256(cfg,ROOT),'D1':cfg['five_gpu']['opening_identity'],
        'mode':cfg['five_gpu']['opening_mode'],'na':RUBRIC_ONLY_NA_POLICY,
        'execution_mode':'shared_ready_tasks_v1','batch_cases':5})


def case_config(cfg,path,worker):
    out=copy.deepcopy(cfg);out['output_dir']=str(path.resolve());out['schedule']['cases']=1
    out['schedule'].pop('positions',None);out['schedule']['calibration_only_cursors']=[]
    out['prompt_update']['enabled']=False
    out['rollout']['doctor_backend']['endpoint']=cfg['five_gpu']['doctor_urls'][worker]
    out.setdefault('execution',{})['doctor_workers']=1
    out['execution']['agent_workers']=4
    return out


def validate(old,new,run,state,cases):
    if new['output_dir']!=old['output_dir']:raise RuntimeError('Bridge must resume into the same run directory')
    if state['config_sha256']!=sha256_json(old):raise RuntimeError('Source run config hash differs')
    if state['protocol_sha256']!=composite_protocol(old):raise RuntimeError('Source run protocol differs')
    if composite_protocol(new)!=state['protocol_sha256']:
        raise RuntimeError('Bridge does not preserve the frozen tree/cache protocol')
    if new.get('rollout_protocol_sha256_override')!=protocol_sha256(old,ROOT):
        raise RuntimeError('Bridge rollout-cache compatibility hash is not the source hash')
    if new['verification'].get('mode')!='random10_witness_v1' or new['verification'].get('max_valid_supports')!=10:
        raise RuntimeError('Bridge must use random-10 witness verification')
    if new['prompt_update'].get('candidate_seed')!='first_completed_P0_v1':
        raise RuntimeError('Bridge must seed P1 from first completed P0')
    if new['prompt_update'].get('aggregation')!='mean_sampled_witness_effect_per_direction_product_v1':
        raise RuntimeError('Bridge must score from sampled witness reruns')
    if new['five_gpu'].get('execution_mode')!='shared_ready_tasks_v1':
        raise RuntimeError('Bridge must use the existing shared five-GPU workers')
    if len(cases)!=5:raise RuntimeError('Expected one uncommitted five-case batch')
    for path in cases:
        if any((path/name).exists() for name in ('P1.json','union.json','result.json')):
            raise RuntimeError('Found later-phase or committed case output; refusing in-place restart')
        cache=CallCache(path/'calls.jsonl')
        rows=list(cache.store.values())
        agents={r.get('identity',{}).get('path') for r in rows if r.get('kind')=='agent'}
        finals={r.get('identity',{}).get('path') for r in rows if r.get('kind')=='final'}
        leaves=leaf_progress(agents,finals,new['rollout']['branching'],new['rollout']['rounds'],
            new['rollout'].get('halting',{}).get('min_agent_turns',2))
        if not leaves['tree_complete']:
            raise RuntimeError(f'Rollout cache incomplete for {path.name}; cannot promise no rerollout')
    if (run/'prompt_seed.json').exists() or (run/'prompt_candidate.json').exists():
        raise RuntimeError('P1 seed/candidate already exists; refusing to replace a frozen candidate')


def migrate_case(path,cfg,worker):
    # Recreate only the per-case state envelope. calls.jsonl and prompt store remain untouched.
    local=case_config(cfg,path,worker);config_sha=sha256_json(local);protocol=protocol_sha256(local,ROOT)
    state_path=path/'run_state.json';manifest_path=path/'run_manifest.json'
    if not state_path.exists() or not manifest_path.exists():raise RuntimeError(f'Missing case state in {path}')
    state=json.loads(state_path.read_text());manifest=json.loads(manifest_path.read_text())
    state['config_sha256']=config_sha
    manifest.update(config=local,config_sha256=config_sha,protocol_sha256=protocol,
                    rerun_continuation_protocol='intervention_agent_once_doctor_final_once_v1')
    atomic_write_json(state_path,state);atomic_write_json(manifest_path,manifest)
    return config_sha,protocol


def main():
    import argparse,fcntl
    p=argparse.ArgumentParser()
    p.add_argument('--pid',type=int,required=True)
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args()
    old=json.loads((ROOT/OLD_CONFIG).read_text());new=json.loads((ROOT/BRIDGE_CONFIG).read_text())
    run=(ROOT/old['output_dir']).resolve();state,cases=frozen_batch(old,run)
    validate(old,new,run,state,cases)
    active=info(a.pid)
    if b'scripts/online_rl/run_five_gpu.py' not in active['cmd'] or os.readlink(f'/proc/{a.pid}/cwd')!=str(ROOT):
        raise RuntimeError('PID is not the owned repository training coordinator')
    status=json.loads(STATUS.read_text())
    if status.get('pid')!=a.pid or status.get('config')!=OLD_CONFIG:
        raise RuntimeError('Handoff controller does not match the source coordinator')
    next_cfg=json.loads((ROOT/NEXT_CONFIG).read_text())
    if next_cfg['output_dir']==new['output_dir'] or (ROOT/next_cfg['output_dir']).exists():
        raise RuntimeError('Expected a fresh destination for the post-batch transition')
    helper=status.get('transition_controller_pid')
    helper_info=info(helper) if helper else None
    if helper_info and b'scripts/diagnostics/after_batch_halt4.py' not in helper_info['cmd']:
        raise RuntimeError('Recorded transition helper PID is not after_batch_halt4.py')
    print(json.dumps({'stage':'restart_preflight','pid':a.pid,'batch':1,'trees_reused':5,
        'p0_archived':sum((c/'P0.json').exists() for c in cases),'random10':True,
        'p0_p1_score_from_witness_reruns':True,'destination_after_batch':NEXT_CONFIG}),flush=True)
    if a.dry_run:return

    lock=(run/'pool_restart.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    env={}
    for entry in Path(f'/proc/{a.pid}/environ').read_bytes().split(b'\0'):
        if b'=' in entry:
            k,v=entry.split(b'=',1);env[os.fsdecode(k)]=os.fsdecode(v)
    if not env.get('OPENROUTER_API_KEY'):raise RuntimeError('API key is not present in coordinator environment')
    env.update(SHERPA_FIVE_GPU_CONFIG=BRIDGE_CONFIG,SHERPA_AGENT_BATCH_SIZE='4',SHERPA_DOCTOR_CONCURRENCY='1')
    old_info=active;owned=descendants(a.pid)
    doctor_pids={int(f.read_text()) for f in (run/'doctor_servers').glob('gpu*.pid')}
    if doctor_pids & {pid for pid,_ in owned}:raise RuntimeError('Doctor server is a coordinator descendant')

    # Stop the old watcher first so it cannot race the newly committed batch.
    if helper_info:
        os.kill(helper,signal.SIGTERM)
        deadline=time.monotonic()+10
        while same(helper,helper_info['start']) and time.monotonic()<deadline:time.sleep(.1)
        if same(helper,helper_info['start']):raise RuntimeError('Old transition helper did not stop')
    os.kill(a.pid,signal.SIGSTOP)
    stopped=False
    try:
        if json.loads((run/'run_state.json').read_text())!=state:raise RuntimeError('Committed state changed before stop')
        owned=descendants(a.pid)
        for pid,start in owned:
            if same(pid,start):os.kill(pid,signal.SIGTERM)
        os.kill(a.pid,signal.SIGTERM);os.kill(a.pid,signal.SIGCONT);stopped=True
        deadline=time.monotonic()+30
        while same(a.pid,old_info['start']) or any(same(pid,start) for pid,start in owned):
            if time.monotonic()>deadline:raise RuntimeError('Old coordinator workers did not exit')
            time.sleep(.2)

        audit={'source_config':OLD_CONFIG,'bridge_config':BRIDGE_CONFIG,'next_config':NEXT_CONFIG,
            'source_protocol_sha256':state['protocol_sha256'],'rollout_cache_protocol_sha256':new['rollout_protocol_sha256_override'],
            'completed_tree_caches_reused':5,'old_P0_archived':[],'effectiveness_source':'random10 witness reruns',
            'additional_full_member_reruns':0,'time':time.time()}
        assignments={a['index']:a for a in json.loads((run/'batches/batch_000001/manifest.json').read_text())['assignments']}
        for path in cases:
            oldp=path/'P0.json'
            if oldp.exists():
                archive=path/'P0_all_members_legacy.json'
                if archive.exists():raise RuntimeError(f'Archive already exists: {archive}')
                oldp.replace(archive);audit['old_P0_archived'].append(str(archive.relative_to(run)))
            index=int(path.name.rsplit('_',1)[-1]);worker=assignments[index]['worker']
            config_sha,case_protocol=migrate_case(path,new,worker)
        state_path=run/'run_state.json';root_state=json.loads(state_path.read_text())
        root_state['config_sha256']=sha256_json(new)
        if root_state['protocol_sha256']!=composite_protocol(new):raise RuntimeError('Migrated composite protocol is inconsistent')
        atomic_write_json(state_path,root_state)
        root_manifest=json.loads((run/'run_manifest.json').read_text())
        root_manifest.update(config=new,protocol_sha256=root_state['protocol_sha256'],restart_transition=audit)
        atomic_write_json(run/'run_manifest.json',root_manifest)
        atomic_write_json(run/'precommit_random10_restart.json',audit)

        status.update(stage='restarting_random10_reuse',pid=a.pid,config=BRIDGE_CONFIG,run_dir=str(run),
                      pending_config=NEXT_CONFIG,pending_candidate_seed='first_completed_P0_v1',
                      pending_rubric_schema=new['rubric'].get('schema'),pending_repeat_agent_reply_after_turns=4)
        atomic_write_json(STATUS,status)
        with (run/'train.log').open('a') as log:
            proc=subprocess.Popen(['bash','scripts/online_rl/launch_five_gpu.sh'],cwd=ROOT,env=env,
                stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
        status.update(stage='training_running',pid=proc.pid,config=BRIDGE_CONFIG,restarted_at=time.time(),
                      rollout_caches_reused=5,effectiveness_from_witness_sample=True)
        atomic_write_json(STATUS,status)
        helper_python=env.get('SHERPA_PYTHON','/home/kai/.venvs/gpu1_probe/bin/python')
        helper_cmd=[helper_python,'scripts/diagnostics/after_batch_halt4.py','--pid',str(proc.pid),
                    '--old-config',BRIDGE_CONFIG,'--new-config',NEXT_CONFIG,'--launch']
        with (run/'train.log').open('a') as log:
            watcher=subprocess.Popen(helper_cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,
                stdout=log,stderr=log,start_new_session=True)
        status['transition_controller_pid']=watcher.pid;atomic_write_json(STATUS,status)
        print(json.dumps({'stage':'restart_launched','pid':proc.pid,'watcher_pid':watcher.pid,
            'run_dir':str(run),'trees_reused':5}),flush=True)
        time.sleep(4)
        if proc.poll() is not None:raise RuntimeError('Restarted training exited; inspect train.log')
    except BaseException as exc:
        status.update(stage='random10_restart_failed',transition_error=type(exc).__name__)
        atomic_write_json(STATUS,status)
        raise


if __name__=='__main__':main()
