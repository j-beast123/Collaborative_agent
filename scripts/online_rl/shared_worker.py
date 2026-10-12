"""Second-server worker: two local GPU actors, four cases, no optimizer updates."""
import argparse
import copy
import fcntl
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import socket
import sys
import threading
import time
import uuid
from concurrent.futures import ProcessPoolExecutor

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    from src.online_rl.tree.experiment import load_config
    from src.online_rl.tree.devices import physical_gpus, gpu_count, record_runtime
    from src.online_rl.tree.gpu_pool import create_pool, assert_idle
    from src.online_rl.tree.shared_rollout import read
    from src.online_rl.integrated.records import atomic_write_json, sha256_json
    from src.online_rl.integrated.continuation import protocol_sha256
    from src.online_rl.tree import five_gpu_worker as worker
    from scripts.online_rl.run_tree import load_key
    from src.common.checkpoint_hub import file_digest
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/group_shared.json')
    args = parser.parse_args()
    cfg = load_config(args.config)
    out = (ROOT/cfg['output_dir']).resolve()
    home = out/'distributed/worker'
    home.mkdir(parents=True, exist_ok=True)
    owner = home/'owner'
    if owner.exists():
        prior = read(owner/'owner.json')
        if prior.get('host') == socket.gethostname() and not (Path('/proc')/str(prior.get('pid'))).exists():
            shutil.rmtree(owner)
        else:
            raise RuntimeError('Another worker owns this run; stop that worker first')
    owner.mkdir()
    atomic_write_json(owner/'owner.json', {'pid': os.getpid(), 'host': socket.gethostname()})
    heartbeat = {'pid': os.getpid(), 'host': socket.gethostname(), 'session': uuid.uuid4().hex,
                 'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                 'status': 'initializing', 'gpus': 2}
    stopped = threading.Event()
    def pulse():
        while not stopped.is_set():
            atomic_write_json(home/'heartbeat.json', {**heartbeat, 'timestamp': time.time()})
            stopped.wait(2)
    thread = threading.Thread(target=pulse, daemon=True); thread.start()
    locks, executors = [], []
    manager = None
    try:
        if gpu_count(cfg) != 2: raise ValueError('Shared worker requires two local GPUs')
        physical = physical_gpus(count=2)
        for device in physical:
            lock = Path(f'/tmp/sherpa_tree_physical_gpu{device}.lock').open('a')
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); locks.append(lock)
        local = copy.deepcopy(cfg)
        local['output_dir'] = str(home)
        local['checkpoint_upload'] = {'enabled': False}
        if protocol_sha256(local, ROOT) != protocol_sha256(cfg, ROOT):
            raise RuntimeError('Remote placement changed the research protocol')
        record_runtime(home, local)
        key = load_key(local)
        context = multiprocessing.get_context('spawn')
        manager = context.Manager()
        spec = create_pool(manager, [0,1], agent_capacity=local['five_gpu']['agent_batch_size'],
                           doctor_capacity=local['five_gpu']['doctor_concurrency'])
        from src.common.api_slots import create as create_api_slots
        spec['api_slots'] = create_api_slots(local.get('execution',{}).get('api_limits',{}), manager)
        for gpu in (0,1):
            executors.append(ProcessPoolExecutor(max_workers=1, mp_context=context,
                initializer=worker.initialize, initargs=(local,gpu,key,spec)))
        # Launch each model load immediately, while the coordinator initializes its server.
        ready=[executor.submit(worker.shared_ready) for executor in executors]
        from src.online_rl.cases import load_cases, file_sha256
        if file_sha256(ROOT/cfg['data']['train']) != cfg['data']['train_sha256']:
            raise RuntimeError('Remote training dataset hash differs')
        byid = {case['case_id']:case for case in load_cases(ROOT/cfg['data']['train'])}
        for future in ready:future.result()
        done = set()
        heartbeat['status'] = 'ready'
        print('Second server worker ready; waiting for four-case batch jobs.', flush=True)
        while True:
            pending = None
            for path in sorted((out/'distributed').glob('batch_*/request.json')):
                request = read(path)
                if not request or request['request_id'] in done: continue
                result = read(path.parent/'result.json')
                if result.get('ok') and result.get('request_id') == request['request_id']:
                    done.add(request['request_id']); continue
                pending = (path,request); break
            if pending is None:
                state = read(out/'run_state.json')
                if state.get('case_cursor',0) >= cfg['schedule']['cases']: break
                time.sleep(1); continue
            path, request = pending
            heartbeat.update(status='running', batch=request['batch'])
            try:
                if request['request_id'] != sha256_json({k:v for k,v in request.items() if k!='request_id'}):
                    raise RuntimeError('Shared request hash differs')
                if request['rollout_protocol_sha256'] != protocol_sha256(local, ROOT):
                    raise RuntimeError('Remote request protocol differs')
                receipt = read(Path(request['checkpoint'])/'COMPLETE.json')
                if receipt.get('policy_sha256') != request['source_policy_sha256']:
                    raise RuntimeError('Remote source checkpoint differs')
                assert_idle(spec)
                prepares = [executor.submit(worker.prepare, request['checkpoint'], request['source_policy_sha256'])
                            for executor in executors]
                for future in prepares: future.result()
                jobs = []
                for number, item in enumerate(request['jobs']):
                    if item['case_id'] not in byid: raise RuntimeError('Unknown remote case')
                    jobs.append({**item, 'worker':number%2, 'case':byid[item['case_id']]})
                futures = [executor.submit(worker.run_many, [j for j in jobs if j['worker']==gpu],
                            request['checkpoint'], request['source_policy_sha256'])
                           for gpu,executor in enumerate(executors)]
                records = [record for future in futures for record in future.result()]
                assert_idle(spec)
                files = []
                for record in records:
                    index = record['case_cursor']
                    job = next(j for j in jobs if j['index']==index)
                    record['rollout_server'] = socket.gethostname()
                    result_path = Path(job['out'])/'result.json'
                    atomic_write_json(result_path, record)
                    files.append({'index':index,'path':str(result_path),'sha256':file_digest(result_path)})
                atomic_write_json(path.parent/'result.json', {'ok':True,'request_id':request['request_id'],
                    'source_policy_sha256':request['source_policy_sha256'],
                    'rollout_protocol_sha256':request['rollout_protocol_sha256'],
                    'host':socket.gethostname(),'idle':True,'case_files':files})
                done.add(request['request_id']); heartbeat['status']='ready'
                print(f"Batch {request['batch']}: four remote cases finished; waiting for next policy.",flush=True)
            except Exception as exc:
                atomic_write_json(path.parent/'result.json', {'ok':False,'request_id':request['request_id'],
                                                          'error_type':type(exc).__name__})
                raise
    except BaseException as exc:
        heartbeat.update(status='error', error_type=type(exc).__name__)
        raise
    finally:
        stopped.set(); thread.join(timeout=5)
        atomic_write_json(home/'heartbeat.json', {**heartbeat,'status':heartbeat['status'] if heartbeat['status']=='error' else 'stopped','timestamp':time.time()})
        for executor in executors: executor.shutdown(wait=True, cancel_futures=True)
        if manager is not None: manager.shutdown()
        for lock in locks: lock.close()
        shutil.rmtree(owner, ignore_errors=True)


if __name__ == '__main__': main()
