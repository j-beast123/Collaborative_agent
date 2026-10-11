"""Independent, resumable single-path checkpoint evaluation on an idle GPU.

This module deliberately imports no model library at import time. Each worker
sets CUDA visibility before constructing its own policy and local Doctor. Every
expensive call observes the cooperative training priority coordinator; model
residency is released only after the policy and Doctor have freed their memory.
"""
from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import select
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

from src.online_rl.integrated.records import atomic_write_json, sha256_json, sha256_text

ROOT = Path(__file__).resolve().parents[3]
MODULE = 'src.online_rl.tree.idle_eval_worker'
CONTRACT = 'idle_checkpoint_single_path_v1'
MIN_FREE_GIB = 40
EXPECTED_PAIR_GIB = 32


class WorkerRevoked(RuntimeError):
    """Finish the current native call, save its cache entry, then free this GPU."""


@contextmanager
def _locked(path, *, blocking=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        flags = fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB)
        fcntl.flock(stream, flags)
        try:
            yield stream
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def _read(path):
    return json.loads(Path(path).read_text())


def _file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_protocol(cfg):
    """Recompute the composite protocol persisted by run_five_gpu.run.

    This is CPU-only: identity helpers never construct policies or API clients.
    Comparing against the receipt detects prompt/protocol changes after capture.
    """
    from src.online_rl.integrated.continuation import protocol_sha256
    from src.online_rl.tree.team import RUBRIC_ONLY_NA_POLICY
    from src.online_rl.tree.gpu_pool import MODE as POOL_MODE
    from scripts.online_rl.run_tree import evidence_gate_identity, intervention_audit_identity
    from src.online_rl.tree.applicability import APPLICABILITY_CONTRACT
    from src.online_rl.tree.scoring import SCORE_RECORD_CONTRACT

    fg = cfg['five_gpu']
    parts = {'base': protocol_sha256(cfg, ROOT), 'D1': fg['opening_identity'],
             'mode': fg['opening_mode'], 'na': RUBRIC_ONLY_NA_POLICY,
             'execution_mode': POOL_MODE, 'batch_cases': 5}
    if cfg.get('rubric', {}).get('generation', {}).get('output_contract') == 'trajectory_analysis_candidate_v2_with_intervention':
        from src.online_rl.tree.member_intervention import MEMBER_INTERVENTION_VERSION
        parts['member_intervention_protocol'] = MEMBER_INTERVENTION_VERSION
    if cfg.get('feedback', {}).get('protocol') == 'physician_post_final_reflection_v1':
        parts['doctor_reflection'] = {key: cfg['feedback'][key]
                                      for key in ('protocol', 'model', 'prompt_sha256', 'max_tokens')}
        parts['fixed_rubric_prompt_sha256'] = cfg['rubric']['prompt_sha256']
    if cfg.get('rubric', {}).get('generation', {}).get('feedback_check_protocol') is not None:
        from src.online_rl.tree.backends import generation_settings
        parts['feedback_claim_audit'] = generation_settings(cfg['rubric'])
    gate = evidence_gate_identity(cfg, ROOT)
    if gate is not None:
        parts['rubric_gate'] = gate
    if cfg.get('rubric', {}).get('enabled', True):
        parts['applicability_contract'] = APPLICABILITY_CONTRACT
        parts['score_record_contract'] = SCORE_RECORD_CONTRACT
    audit = intervention_audit_identity(cfg, ROOT)
    if audit is not None:
        parts['intervention_instruction_audit'] = audit
    from src.online_rl.tree.cumulative_run import config_identity
    bank = config_identity(cfg, ROOT)
    if bank is not None:
        parts['cumulative_bank'] = bank
    parts['rubric_pipeline_mode'] = 'phase_barrier_reruns_then_union_score_v1'
    return sha256_json(parts)


def _inputs(config_path, snapshot_dir, existing_job=None):
    """Validate CPU-only inputs, including immutable snapshot payload hashes."""
    from src.online_rl.cases import normalize
    from src.online_rl.integrated.continuation import protocol_sha256
    from src.online_rl.tree.experiment import load_config
    from src.online_rl.tree.idle_eval_snapshot import _existing

    cfg = load_config(str(config_path))
    snapshot_dir = Path(snapshot_dir).resolve()
    proof = _read(snapshot_dir / 'SNAPSHOT.json')
    manifest = _read(snapshot_dir / 'run_manifest.json')
    _existing(snapshot_dir, Path(proof['source_run']), proof['completed_cases'], manifest)
    source_cfg = manifest['config']
    normalized = copy.deepcopy(cfg)
    normalized['five_gpu']['checkpoint_evaluation_enabled'] = False
    if sha256_json(normalized) != _read(snapshot_dir / 'run_state.json')['config_sha256']:
        raise RuntimeError('Evaluation config differs from the checkpoint config')
    # The receipt records the composite five-GPU training protocol. Compare
    # rollout fingerprints to one another, never to that composite hash.
    if protocol_sha256(cfg, ROOT) != protocol_sha256(source_cfg, ROOT):
        raise RuntimeError('Evaluation rollout protocol differs from the checkpoint config')
    for key in ('base_model', 'lora', 'init_adapter'):
        if cfg['policy'].get(key) != source_cfg['policy'].get(key):
            raise RuntimeError(f'Evaluation policy.{key} differs from checkpoint config')
    fg = cfg['five_gpu']
    if fg.get('execution_mode') != 'shared_ready_tasks_v1':
        raise ValueError('Idle evaluation requires cooperative shared GPU actors')
    if _checkpoint_protocol(cfg) != manifest['protocol_sha256']:
        # A partially completed immutable evaluation may outlive supervision
        # code changes. Resume only when its already-published job is pinned to
        # this exact snapshot and receipt; never use this path to create a job.
        if (not isinstance(existing_job, dict) or
                existing_job.get('contract') != CONTRACT or
                existing_job.get('snapshot_sha256') != sha256_json(proof) or
                existing_job.get('protocol_sha256') != manifest['protocol_sha256'] or
                existing_job.get('policy_sha256') != _read(snapshot_dir/'COMPLETE.json').get('policy_sha256')):
            raise RuntimeError('Evaluation prompt/protocol fingerprint differs from checkpoint receipt')
    for key in ('eval_sha256', 'eval_cases', 'eval_seed', 'opening_identity'):
        if fg[key] != source_cfg['five_gpu'][key]:
            raise RuntimeError(f'Evaluation five_gpu.{key} differs from checkpoint config')
    backend = cfg['rollout'].get('doctor_backend') or {}
    if (backend.get('engine') != 'ollama' or backend.get('ollama_version') != '0.12.5'
            or backend.get('quantization') != 'Q8_0'):
        raise ValueError('Idle evaluation requires the pinned local Ollama 0.12.5 Q8_0 Doctor')
    ro = cfg['rollout']
    for path_key, hash_key, default in (
            ('doctor_system_prompt_file', 'doctor_system_prompt_sha256', 'prompts/user_simulator.txt'),
            ('agent_system_prompt_file', 'agent_system_prompt_sha256', 'prompts/agent.txt')):
        if ro.get(hash_key) and _file_hash(ROOT / ro.get(path_key, default)) != ro[hash_key]:
            raise RuntimeError(f'Evaluation {path_key} hash differs')
    data_path = ROOT / fg['eval_file']
    if _file_hash(data_path) != fg['eval_sha256']:
        raise RuntimeError('Evaluation dataset file hash differs')
    original = [json.loads(line) for line in data_path.read_text().splitlines() if line.strip()]
    cases = [normalize(case) for case in original]
    if len(cases) != fg['eval_cases'] or len({c['case_id'] for c in cases}) != len(cases):
        raise RuntimeError('Evaluation dataset count or unique case IDs differ')
    receipt = _read(snapshot_dir / 'COMPLETE.json')
    return cfg, proof, receipt, cases, original


def ensure_job(config_path, snapshot_dir, eval_root):
    """Publish/validate a job manifest without importing torch or loading a GPU.

    ``eval_root`` is a dedicated directory for this one immutable checkpoint.
    This function is also the supervisor's preparation entry point.
    """
    existing_path = Path(eval_root)/'manifest.json'
    existing_job = _read(existing_path) if existing_path.exists() else None
    cfg, proof, receipt, cases, original = _inputs(config_path, snapshot_dir, existing_job)
    ro = copy.deepcopy(cfg['rollout'])
    ro['doctor_backend'].pop('endpoint', None)
    ro['doctor_backend'].pop('gpu', None)
    identity = {
        'contract': CONTRACT, 'completed_cases': receipt['completed_cases'],
        'snapshot_sha256': sha256_json(proof), 'policy_sha256': receipt['policy_sha256'],
        'protocol_sha256': receipt['protocol_sha256'], 'base_model': cfg['policy']['base_model'],
        'rollout': ro, 'episodes_per_case': 1, 'optimizer_steps': 0, 'rubrics': False,
        'agent_prompt_sha256': sha256_text((ROOT / ro.get('agent_system_prompt_file', 'prompts/agent.txt')).read_text()),
        'doctor_prompt_sha256': sha256_text((ROOT / ro.get('doctor_system_prompt_file', 'prompts/user_simulator.txt')).read_text()),
        'eval_sha256': cfg['five_gpu']['eval_sha256'], 'eval_seed': cfg['five_gpu']['eval_seed'],
        'opening_identity': cfg['five_gpu']['opening_identity'], 'dataset_cases': len(cases),
        'case_ids': [case['case_id'] for case in cases],
        'original_case_sha256': [sha256_json(case) for case in original],
    }
    identity['job_sha256'] = sha256_json(identity)
    root = Path(eval_root)
    root.mkdir(parents=True, exist_ok=True)
    with _locked(root / 'manifest.lock'):
        target = root / 'manifest.json'
        if target.exists():
            if _read(target) != identity:
                raise RuntimeError('Evaluation directory belongs to another immutable job')
        else:
            atomic_write_json(target, identity)
    summarize(root, identity)
    return identity


def _validate_result(row, manifest):
    index = row.get('index')
    if type(index) is not int or not 0 <= index < manifest['dataset_cases']:
        raise RuntimeError('Evaluation result index is invalid')
    if (row.get('job_sha256') != manifest['job_sha256']
            or row.get('policy_sha256') != manifest['policy_sha256']
            or row.get('case_id') != manifest['case_ids'][index]
            or sha256_json(row.get('original_case')) != manifest['original_case_sha256'][index]):
        raise RuntimeError('Evaluation result belongs to another job, policy, or original case')
    if (row.get('seed') != manifest['eval_seed'] + index
            or row.get('snapshot_sha256') != manifest['snapshot_sha256']):
        raise RuntimeError('Evaluation result seed or snapshot identity differs')
    turns = row.get('num_discussion_rounds')
    if (type(turns) is not int or not 0 <= turns <= manifest['rollout']['rounds']
            or row.get('max_discussion_rounds') != manifest['rollout']['rounds']):
        raise RuntimeError('Evaluation result discussion limit differs')
    from src.online_rl.cases import normalize
    from src.online_rl.environment import final_outcome
    if row.get('outcome') != final_outcome(normalize(row['original_case']), row.get('final_option')):
        raise RuntimeError('Evaluation result outcome differs from its original case and FINAL')


def summarize(eval_root, manifest=None):
    """Serialize summaries so a slower writer cannot overwrite newer progress."""
    root = Path(eval_root)
    manifest = manifest or _read(root / 'manifest.json')
    with _locked(root / 'summary.lock'):
        rows = []
        for path in sorted((root / 'cases').glob('case_*.json')):
            row = _read(path)
            _validate_result(row, manifest)
            if path.name != f'case_{row["index"]:06d}.json':
                raise RuntimeError('Evaluation result filename/index differ')
            rows.append(row)
        count = len(rows)
        summary = {
            'contract': CONTRACT, 'job_sha256': manifest['job_sha256'],
            'policy_sha256': manifest['policy_sha256'], 'completed_cases': manifest['completed_cases'],
            'dataset_cases': manifest['dataset_cases'], 'evaluated': count,
            'correct': sum(row['outcome'] == 1 for row in rows),
            'unparsed': sum(row['outcome'] is None for row in rows),
            'accuracy': sum(row['outcome'] == 1 for row in rows) / count if count else None,
            'mean_agent_turns': sum(row['num_discussion_rounds'] for row in rows) / count if count else None,
            'status': 'complete' if count == manifest['dataset_cases'] else 'pending',
        }
        atomic_write_json(root / 'summary.json', summary)
        return summary


@contextmanager
def claim_case(eval_root, cases, manifest):
    """Claim the first available missing case; flock releases on worker death."""
    root = Path(eval_root)
    for index, case in enumerate(cases):
        result = root / 'cases' / f'case_{index:06d}.json'
        if result.exists():
            _validate_result(_read(result), manifest)
            continue
        lock = root / 'claims' / f'case_{index:06d}.lock'
        lock.parent.mkdir(parents=True, exist_ok=True)
        handle = lock.open('a')
        try:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                continue
            if result.exists():
                _validate_result(_read(result), manifest)
                continue
            yield index, case, result
            return
        finally:
            handle.close()
    yield None


def _cache(path):
    """Recover only a torn final append; retain and validate all complete calls."""
    from src.online_rl.tree.cache import CallCache
    path = Path(path)
    if path.exists():
        raw = path.read_bytes()
        if raw and not raw.endswith(b'\n'):
            boundary = raw.rfind(b'\n') + 1
            try:
                json.loads(raw[boundary:])
            except (json.JSONDecodeError, UnicodeDecodeError):
                with path.open('r+b') as stream:
                    stream.truncate(boundary)
                    stream.flush()
                    os.fsync(stream.fileno())
            else:
                with path.open('ab') as stream:
                    stream.write(b'\n')
                    stream.flush()
                    os.fsync(stream.fileno())
    return CallCache(path)


def _cached(cache, kind, identity, fn):
    value = cache.call(kind, identity, fn)
    # CallCache appends before returning; make completed native calls survive
    # process interruption before committing the encompassing episode.
    if cache.path and cache.path.exists():
        with cache.path.open('rb') as stream:
            os.fsync(stream.fileno())
    return value


class EvaluationCalls:
    def __init__(self, coordinator, residency, gpu, *, sync=lambda: None, doctor=None, stop_path=None):
        self.coordinator, self.residency, self.gpu = coordinator, residency, gpu
        self.sync, self.doctor = sync, doctor
        self.stop_path = Path(stop_path) if stop_path is not None else None
        self.stopping = False
        self.active_call = None

    def check(self):
        if self.stopping or (self.stop_path is not None and self.stop_path.exists()) or self.residency.revoked():
            raise WorkerRevoked('Evaluation residency revoked for training')

    def run(self, fn):
        from src.online_rl.tree.idle_eval_priority import EvaluationDeferred
        while True:
            self.check()
            try:
                lease = self.coordinator.evaluation_call(self.gpu, self.residency)
                self.active_call = lease
                break
            except EvaluationDeferred:
                time.sleep(self.coordinator.poll_seconds)
        try:
            value = fn()
        except BaseException:
            self.sync()
            # A failed HTTP read does not prove Ollama stopped generating. Keep
            # training blocked until the owned guardian has stopped the server.
            if self.doctor is not None:
                self.doctor.close()
            lease.release()
            self.active_call = None
            raise
        else:
            # Do not acknowledge completion while CUDA kernels still run.
            # Failed synchronization intentionally leaves the lease quarantined
            # until this worker and its tracked Doctor guardian have exited.
            self.sync()
            if self.doctor is not None:
                self.doctor.register_descendants(self.residency)
            lease.release()
            self.active_call = None
            return value


def install_call_cache(policy, calls, cache, manifest, case, seed):
    """Cache Agent segments and every seeded native Doctor response separately."""
    original_generate, original_http = policy._generate, policy.doctor.http
    original_turn = policy.agent_turn
    original_trace = policy.ST.generate_with_trace
    actions = []
    common = {'job_sha256': manifest['job_sha256'], 'policy_sha256': manifest['policy_sha256'],
              'case_id': case['case_id'], 'episode_seed': seed}

    def generate(prompt, max_new_tokens, call_seed, segment='agent'):
        calls.check()
        identity = {**common, 'prompt': prompt, 'seed': call_seed,
                    'max_new_tokens': max_new_tokens, 'segment': segment,
                    'temperature': policy.args.generation_temperature, 'top_p': policy.args.top_p,
                    'top_k': policy.args.top_k,
                    'sampling_protocol': policy.cfg['rollout'].get('agent_sampling_protocol')}
        return _cached(cache, 'agent_segment', identity,
                       lambda: calls.run(lambda: original_generate(prompt, max_new_tokens, call_seed, segment)))

    def http(url, payload, timeout):
        if payload is None or not url.rstrip('/').endswith('/api/chat'):
            return original_http(url, payload, timeout)
        calls.check()
        identity = {**common, 'protocol': policy.doctor.protocol, 'request': payload}
        try:
            return _cached(cache, 'doctor_native_chat', identity,
                           lambda: calls.run(lambda: original_http(url, payload, timeout)))
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as error:
            # The server was stopped before releasing the call lease. Avoid
            # Doctor._chat's transport retry loop against that stopped server.
            raise RuntimeError('Dedicated evaluation Doctor call failed') from error

    def agent_turn(*args, **kwargs):
        value = original_turn(*args, **kwargs)
        actions.append(value)
        return value

    raw_index = 0

    def traced(model, tokenizer, prompt, max_new_tokens, temperature, top_p, top_k):
        nonlocal raw_index
        calls.check()
        raw_index += 1
        identity = {**common, 'prompt': prompt, 'call_index': raw_index,
                    'max_new_tokens': max_new_tokens, 'temperature': temperature,
                    'top_p': top_p, 'top_k': top_k}
        def sample():
            trace = original_trace(model, tokenizer, prompt, max_new_tokens, temperature, top_p, top_k)
            return {'trace': trace, 'cpu_rng': policy.torch.get_rng_state().tolist(),
                    'cuda_rng': policy.torch.cuda.get_rng_state(0).tolist()}
        value = _cached(cache, 'agent_raw_trace', identity, lambda: calls.run(sample))
        # Raw episodes use a continuous RNG stream. Restore its post-call state
        # even on a hit so resuming between raw Agent turns is deterministic.
        def restore_rng():
            policy.torch.set_rng_state(policy.torch.tensor(value['cpu_rng'], dtype=policy.torch.uint8))
            policy.torch.cuda.set_rng_state(policy.torch.tensor(value['cuda_rng'], dtype=policy.torch.uint8), 0)
        calls.run(restore_rng)
        return value['trace']

    policy._generate, policy.doctor.http, policy.agent_turn = generate, http, agent_turn
    if not policy.precommit:
        # Each worker owns its process; this avoids bypassing call priority in
        # environment_v2's legacy direct traced-generation path.
        policy.ST.generate_with_trace = traced

    def restore():
        policy._generate, policy.doctor.http, policy.agent_turn = original_generate, original_http, original_turn
        policy.ST.generate_with_trace = original_trace

    return actions, restore


def _process_live(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[0] not in ('Z', 'X')
    except FileNotFoundError:
        return False


def _descendants(pid):
    found, todo = set(), [pid]
    while todo:
        parent = todo.pop()
        for task in Path(f'/proc/{parent}/task').glob('*'):
            try:
                children = [int(p) for p in (task / 'children').read_text().split()]
            except FileNotFoundError:
                continue
            for child in children:
                if child not in found:
                    found.add(child)
                    todo.append(child)
    return found


def _group_pids(pgid):
    found = set()
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():
            continue
        try:
            fields = (path / 'stat').read_text().rsplit(')', 1)[1].split()
            if int(fields[2]) == pgid and fields[0] not in ('Z', 'X'):
                found.add(int(path.name))
        except FileNotFoundError:
            pass
    return found


def _stop_group(proc):
    """The guardian stays alive until all known Doctor descendants are gone."""
    children = _descendants(proc.pid) | _group_pids(proc.pid)
    for sig, duration in ((signal.SIGTERM, 5.0), (signal.SIGKILL, 5.0)):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            pass
        deadline = time.monotonic() + duration
        while time.monotonic() < deadline:
            proc.poll()
            children |= _group_pids(proc.pid)
            if proc.returncode is not None and not any(_process_live(pid) for pid in children):
                return
            time.sleep(0.05)
    # Never let an unkillable runner disappear from priority accounting.
    while proc.poll() is None or _group_pids(proc.pid) or any(_process_live(pid) for pid in children):
        time.sleep(0.5)


def _doctor_guardian(binary):
    """EOF on the worker-owned pipe also cleans up after SIGKILL of the worker."""
    stopped = False

    def stop(*_):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    if sys.stdin.buffer.readline() != b'GO\n' or stopped:
        return 0
    server = subprocess.Popen([binary, 'serve'], cwd=ROOT, start_new_session=True, close_fds=True)
    try:
        while not stopped and server.poll() is None:
            ready, _, _ = select.select([sys.stdin.buffer], [], [], 0.2)
            if ready and not os.read(sys.stdin.fileno(), 1024):
                break
    finally:
        _stop_group(server)
    return 0


class DedicatedDoctor:
    """Only this worker's private server/process group may be unloaded/stopped."""
    def __init__(self, cfg, root, gpu, executable, port):
        self.cfg, self.root, self.gpu = cfg, Path(root), gpu
        self.port, self.url = int(port), f'http://127.0.0.1:{int(port)}'
        if not 1024 <= self.port <= 65535:
            raise ValueError('Invalid dedicated Doctor port')
        if self.url in cfg['five_gpu']['doctor_urls']:
            raise ValueError('Evaluation cannot use a training Doctor endpoint')
        resolved = shutil.which(str(executable)) if '/' not in str(executable) else str(executable)
        if not resolved or not Path(resolved).is_file():
            raise FileNotFoundError(f'Pinned Ollama executable is missing: {executable}')
        self.binary = str(Path(resolved).resolve())
        self.process, self.known_pids = None, set()

    def start(self, residency, calls):
        # Refuse to adopt or signal a service already occupying the endpoint.
        with socket.socket() as probe:
            # A previous owned server can leave accepted sockets in TIME_WAIT.
            # Match the server's reuse policy, while still rejecting any live
            # listener (SO_REUSEPORT is deliberately not enabled).
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            probe.bind(('127.0.0.1', self.port))
        models = os.environ.get('SHERPA_OLLAMA_MODELS', self.cfg['five_gpu'].get(
            'ollama_models_dir', str(Path.home() / '.ollama/models')))
        if not Path(models).is_dir():
            raise FileNotFoundError(f'Existing Ollama model cache is missing: {models}')
        env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(self.gpu),
               'OLLAMA_HOST': f'127.0.0.1:{self.port}', 'OLLAMA_MODELS': str(models),
               'OLLAMA_MAX_LOADED_MODELS': '1', 'OLLAMA_NUM_PARALLEL': '1', 'OLLAMA_KEEP_ALIVE': '-1'}
        self.root.mkdir(parents=True, exist_ok=True)
        with (self.root / f'doctor_gpu{self.gpu}.log').open('a') as log:
            self.process = subprocess.Popen(
                [sys.executable, '-m', MODULE, '--doctor-guardian', self.binary], cwd=ROOT,
                env=env, stdin=subprocess.PIPE, stdout=log, stderr=log, start_new_session=True, close_fds=True)
        # The guardian cannot start Ollama until this registration is committed.
        residency.add_process(self.process.pid)
        self.process.stdin.write(b'GO\n')
        self.process.stdin.flush()
        deadline = time.monotonic() + 60
        from src.online_rl.doctor import http_json
        while time.monotonic() < deadline:
            calls.check()
            if self.process.poll() is not None:
                raise RuntimeError('Dedicated Doctor guardian exited during startup')
            try:
                version = http_json(self.url + '/api/version', None, 1)
                tags = http_json(self.url + '/api/tags', None, 1)
            except (urllib.error.URLError, TimeoutError, OSError):
                time.sleep(0.1)
                continue
            backend = self.cfg['rollout']['doctor_backend']
            model = next((row for row in tags.get('models', []) if row['name'] == backend['model']), None)
            if version.get('version') != backend['ollama_version'] or model is None or model.get('digest') != backend['model_digest']:
                raise RuntimeError('Dedicated Doctor version or cached model differs from pinned configuration')
            if (model.get('details') or {}).get('quantization_level') not in (None, backend['quantization']):
                raise RuntimeError('Dedicated Doctor quantization differs')
            self.register_descendants(residency)
            return
        raise RuntimeError('Dedicated Doctor did not become ready within 60 seconds')

    def register_descendants(self, residency):
        if self.process is None:
            return
        for pid in _descendants(self.process.pid) - self.known_pids:
            try:
                residency.add_process(pid)
            except RuntimeError:
                if _process_live(pid):
                    raise
            else:
                self.known_pids.add(pid)

    def close(self):
        if self.process is None:
            return
        # A native unload releases cached weights; the guardian subsequently
        # terminates its own process group even if unload times out.
        if self.process.poll() is None:
            try:
                from src.online_rl.doctor import http_json
                http_json(self.url + '/api/generate', {
                    'model': self.cfg['rollout']['doctor_backend']['model'], 'keep_alive': 0}, 3)
            except (OSError, ValueError):
                pass
        if self.process.stdin is not None and not self.process.stdin.closed:
            self.process.stdin.close()
        try:
            self.process.wait(timeout=20)
        except subprocess.TimeoutExpired as error:
            raise RuntimeError('Doctor guardian is still freeing GPU memory; residency remains held') from error
        if any(_process_live(pid) for pid in self.known_pids):
            raise RuntimeError('Doctor descendant remains alive; residency remains held')


def _free_policy(policy, torch):
    if policy is not None:
        batcher = getattr(policy, '_batcher', None)
        if batcher is not None:
            batcher.close()
            policy._batcher = None
        models = getattr(policy, 'models', [getattr(policy, 'model', None)])
        for model in models:
            if model is not None:
                model.to('cpu')
        policy.models = []
        policy.model = None
    gc.collect()
    if torch is not None and torch.cuda.is_initialized():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def release_worker_resources(policy, torch, doctor, coordinator, residency, gpu, *, active_call=None):
    """Ordinary shutdown also yields to training before copying/unloading.

    A revoked residency is the update's memory barrier, so cleanup must proceed
    without requesting a new call lease that an update will never admit.
    """
    from src.online_rl.tree.idle_eval_priority import EvaluationDeferred
    cleanup_call = active_call
    while cleanup_call is None and not residency.revoked():
        try:
            cleanup_call = coordinator.evaluation_call(gpu, residency)
            break
        except EvaluationDeferred:
            time.sleep(coordinator.poll_seconds)
    # Neither lease is acknowledged if synchronization/offload/server shutdown
    # fails. Dead-process pruning still waits for the tracked Doctor guardian.
    try:
        _free_policy(policy, torch)
    finally:
        if doctor is not None:
            doctor.close()
    if cleanup_call is not None:
        cleanup_call.release()
    residency.release()


def run_worker(config_path, snapshot_dir, eval_root, gpu, priority_dir, doctor_executable, doctor_port=None,
               *, require_training=True):
    """Run a separate Agent+Doctor pair; return after completion or revocation."""
    # This entry point must be invoked in a fresh spawned process.
    if 'torch' in sys.modules and sys.modules['torch'].cuda.is_initialized():
        raise RuntimeError('Idle evaluation worker must start before CUDA is initialized')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    os.environ['SHERPA_AGENT_REPLICAS'] = '1'
    os.environ['SHERPA_AGENT_BATCH_SIZE'] = '1'
    for key in ('SHERPA_DOCTOR_ENDPOINTS', 'SHERPA_AGENT_PROCESS_WORKERS', 'SHERPA_EXEC_LOG',
                'API_USAGE_ITERATION', 'SHERPA_ARM', 'OPENROUTER_API_KEY'):
        os.environ.pop(key, None)
    root = Path(eval_root).resolve()
    worker_root = root / 'workers' / f'gpu{gpu}'
    worker_root.mkdir(parents=True, exist_ok=True)
    os.environ['API_USAGE_LOG'] = str(worker_root / 'doctor_usage.jsonl')
    os.environ['API_USAGE_STAGE'] = 'idle_checkpoint_evaluation'
    manifest = ensure_job(config_path, snapshot_dir, root)
    existing_job = _read(Path(eval_root)/'manifest.json')
    cfg, proof, receipt, cases, original = _inputs(config_path, snapshot_dir, existing_job)
    if summarize(root, manifest)['status'] == 'complete':
        return {'status': 'complete', 'gpu': gpu, 'job_sha256': manifest['job_sha256']}
    from src.online_rl.tree.idle_eval_priority import EvaluationDeferred, PriorityCoordinator
    coordinator = PriorityCoordinator(priority_dir, require_training=require_training)
    try:
        residency = coordinator.evaluation_residency(gpu)
    except EvaluationDeferred:
        return {'status': 'deferred', 'gpu': gpu, 'job_sha256': manifest['job_sha256']}
    calls = EvaluationCalls(coordinator, residency, gpu, stop_path=root / 'STOP.json')
    policy, torch, doctor = None, None, None
    old_handlers = {}
    outcome = 'deferred'
    try:
        for sig in (signal.SIGTERM, signal.SIGINT):
            old_handlers[sig] = signal.signal(sig, lambda *_: setattr(calls, 'stopping', True))
        from src.common.live_gpu import configure, note
        configure(worker_root, gpu)
        note(phase='idle_eval_loading', job_sha256=manifest['job_sha256'],
             minimum_free_gpu_gib=MIN_FREE_GIB, expected_resident_pair_gib=EXPECTED_PAIR_GIB)

        def load():
            nonlocal torch, policy, doctor
            # Driver-reported free memory includes the independently resident
            # training pair; no training allocation may overlap this call lease.
            result = subprocess.run(['nvidia-smi', '-i', str(gpu), '--query-gpu=memory.free',
                                     '--format=csv,noheader,nounits'], check=True, capture_output=True, text=True)
            free_mib = int(result.stdout.strip())
            if free_mib < MIN_FREE_GIB * 1024:
                raise EvaluationDeferred(f'GPU {gpu}: {free_mib} MiB free, requires {MIN_FREE_GIB} GiB')
            doctor = DedicatedDoctor(cfg, worker_root, gpu, doctor_executable,
                                     11785 + int(gpu) if doctor_port is None else doctor_port)
            calls.doctor = doctor
            doctor.start(residency, calls)
            local_cfg = copy.deepcopy(cfg)
            local_cfg['policy']['init_adapter'] = str(Path(snapshot_dir).resolve() / 'adapter')
            local_cfg['rollout']['doctor_backend']['endpoint'] = doctor.url
            local_cfg['output_dir'] = str(worker_root)
            import torch as torch_module
            torch = torch_module
            calls.sync = lambda: torch.cuda.synchronize() if torch.cuda.is_initialized() else None
            from src.online_rl.runtime import SharedPolicy
            # Keep a partially initialized object reachable for exception cleanup.
            policy = SharedPolicy.__new__(SharedPolicy)
            SharedPolicy.__init__(policy, local_cfg, '', device='cuda:0')
            policy.doctor.timeout_s = min(policy.doctor.timeout_s, 60.0)
            from src.online_rl.integrated.continuation import adapter_sha256
            if adapter_sha256(policy.model) != receipt['policy_sha256']:
                raise RuntimeError('Loaded evaluation adapter differs from immutable C20 snapshot')
            if policy.doctor is None:
                raise RuntimeError('Evaluation requires local Doctor protocol v2')
            problems = policy.doctor.verify()['problems']
            if problems:
                raise RuntimeError('; '.join(problems))
            policy.model.eval()

        calls.run(load)
        from src.online_rl.environment import final_outcome
        while True:
            calls.check()
            with claim_case(root, cases, manifest) as claim:
                if claim is None:
                    if summarize(root, manifest)['status'] == 'complete':
                        outcome = 'complete'
                        break
                    time.sleep(coordinator.poll_seconds)
                    continue
                index, case, destination = claim
                started = time.monotonic()
                seed = cfg['five_gpu']['eval_seed'] + index
                note(phase='idle_eval_episode', case_index=index, case_id=case['case_id'])
                cache = _cache(root / 'calls' / f'case_{index:06d}.jsonl')
                actions, restore = install_call_cache(policy, calls, cache, manifest, case, seed)
                view = {key: case.get(key) for key in ('case_id', 'scenario', 'question', 'options')}
                opening_key = {'case': view, 'protocol': cfg['five_gpu']['opening_identity'],
                               'rounds': cfg['rollout']['rounds']}
                opening_sha = sha256_json(opening_key)
                opening_path = root.parent / 'openings' / f'{opening_sha}.jsonl'
                fixed_seed = int(opening_sha[:8], 16) % (2**31 - 1)

                def opening(case_, _episode_seed):
                    calls.check()
                    with _locked(opening_path.with_suffix('.lock')):
                        shared = _cache(opening_path)
                        return _cached(shared, 'shared_D1', opening_key,
                                       lambda: policy.doctor.turn(case_, [], 1, policy.args.rounds, fixed_seed))

                try:
                    episode = policy.generate(case, seed, opening=opening)
                finally:
                    restore()
                row = {'index': index, 'case_id': case['case_id'], 'original_case': original[index],
                       'job_sha256': manifest['job_sha256'], 'policy_sha256': receipt['policy_sha256'],
                       'snapshot_sha256': manifest['snapshot_sha256'], 'seed': seed,
                       'opening_identity_sha256': opening_sha, 'opening_seed': fixed_seed,
                       **episode, 'agent_actions': actions,
                       'outcome': final_outcome(case, episode['final_option']),
                       'seconds': time.monotonic() - started, 'evaluation_gpu': gpu}
                _validate_result(row, manifest)
                atomic_write_json(destination, row)
                summary = summarize(root, manifest)
                print(json.dumps({'stage': 'idle_eval_case_done', 'gpu': gpu, 'index': index,
                                  'case_id': case['case_id'], **summary}), flush=True)
                if summary['status'] == 'complete':
                    outcome = 'complete'
                    break
    except WorkerRevoked:
        outcome = 'revoked'
    except EvaluationDeferred:
        outcome = 'deferred'
    finally:
        # Do not use a residency context manager: cleanup errors must leave its
        # lease visible until this process and its tracked children really exit.
        release_worker_resources(policy, torch, doctor, coordinator, residency, gpu, active_call=calls.active_call)
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
    return {'status': outcome, 'gpu': gpu, 'job_sha256': manifest['job_sha256']}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--doctor-guardian', help=argparse.SUPPRESS)
    parser.add_argument('--config')
    parser.add_argument('--snapshot')
    parser.add_argument('--eval-root')
    parser.add_argument('--gpu', type=int)
    parser.add_argument('--priority-dir')
    parser.add_argument('--doctor-executable')
    parser.add_argument('--doctor-port', type=int)
    args = parser.parse_args(argv)
    if args.doctor_guardian:
        return _doctor_guardian(args.doctor_guardian)
    for name in ('config', 'snapshot', 'eval_root', 'gpu', 'priority_dir', 'doctor_executable'):
        if getattr(args, name) is None:
            parser.error(f'--{name.replace("_", "-")} is required')
    result = run_worker(args.config, args.snapshot, args.eval_root, args.gpu,
                        args.priority_dir, args.doctor_executable, args.doctor_port)
    print(json.dumps(result), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
