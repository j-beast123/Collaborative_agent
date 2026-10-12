"""Cross-host rollout coordination using only atomic shared-folder files."""
import json
from pathlib import Path
import socket
import time

from src.online_rl.integrated.records import atomic_write_json, sha256_json


def read(path):
    try: return json.loads(Path(path).read_text())
    except (OSError, ValueError): return {}


def enabled(cfg):
    return bool((cfg.get('shared_rollout') or {}).get('enabled'))


def submit(cfg, out, jobs, state, batch, rollout_protocol):
    remote = [job for job in jobs if job.get('server') == 'remote']
    if not remote: return None
    folder = Path(out)/'distributed'/f'batch_{batch:06d}'
    request = {'config': cfg, 'batch': batch, 'source_policy_sha256': state['policy_sha256'],
               'checkpoint': state['checkpoint'], 'rollout_protocol_sha256': rollout_protocol,
               'jobs': [{key: job[key] for key in ('index', 'worker', 'lane', 'out')}
                        | {'case_id': job['case']['case_id']} for job in remote]}
    request['request_id'] = sha256_json(request)
    path = folder/'request.json'
    previous = read(path)
    if previous and previous['request_id'] != request['request_id']:
        raise RuntimeError('Shared-folder pending batch identity differs')
    result_path=folder/'result.json'
    previous_result=read(result_path)
    heartbeat=read(Path(out)/'distributed/worker/heartbeat.json')
    if previous_result and not previous_result.get('ok') and heartbeat.get('status') in ('ready','running'):
        result_path.rename(folder/f'failed_result_{time.time_ns()}.json')
    if not previous: atomic_write_json(path, request)
    return path


def wait_result(request_path, timeout=7200):
    from src.common.checkpoint_hub import file_digest
    request_path = Path(request_path)
    request = read(request_path)
    if request.get('request_id') != sha256_json({key:value for key,value in request.items() if key!='request_id'}):
        raise RuntimeError('Shared request file hash differs')
    folder = request_path.parent
    heartbeat_path = folder.parent/'worker/heartbeat.json'
    started = time.monotonic()
    while time.monotonic()-started < timeout:
        result = read(folder/'result.json')
        if result:
            if result.get('request_id') != request['request_id']:
                raise RuntimeError('Remote result belongs to a different batch')
            if not result.get('ok'):
                raise RuntimeError('Remote worker failed: '+result.get('error_type','unknown'))
            if (result.get('source_policy_sha256') != request['source_policy_sha256']
                    or result.get('rollout_protocol_sha256') != request['rollout_protocol_sha256']
                    or not result.get('idle')):
                raise RuntimeError('Remote worker did not finish the frozen-policy barrier')
            expected = {job['index'] for job in request['jobs']}
            rows = result.get('case_files', [])
            if {row['index'] for row in rows} != expected or len(rows) != len(expected):
                raise RuntimeError('Remote result case coverage differs')
            records = []
            for row in rows:
                path = Path(row['path'])
                job = next(job for job in request['jobs'] if job['index'] == row['index'])
                if path.resolve() != (Path(job['out'])/'result.json').resolve() or file_digest(path) != row['sha256']:
                    raise RuntimeError('Remote result file hash/path differs')
                record = read(path)
                if record.get('case_id') != job['case_id'] or record.get('policy_sha256') != request['source_policy_sha256']:
                    raise RuntimeError('Remote case/policy differs')
                records.append(record)
            return records
        heartbeat = read(heartbeat_path)
        if heartbeat:
            local_boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
            if heartbeat.get('boot_id') == local_boot:
                raise RuntimeError('The shared rollout worker must run on the other server')
            if heartbeat.get('status') == 'error':
                raise RuntimeError('Remote worker initialization failed: '+heartbeat.get('error_type','unknown'))
            if time.time()-heartbeat.get('timestamp',0) > 120:
                raise RuntimeError('Remote worker heartbeat expired; no update was applied')
        elif time.monotonic()-started > 300:
            raise RuntimeError('Start run_shared_worker.sh on the second server')
        time.sleep(1)
    raise TimeoutError('Shared-folder rollout timed out; resume from the saved batch')
