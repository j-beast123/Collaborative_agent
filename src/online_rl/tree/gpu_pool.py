"""Authenticated local RPC for ready work on five frozen GPU actors.

Case coordinators keep their trees/caches. Only ready model calls are leased to
an available actor. There is no retry after uncertain delivery; resume uses the
case CallCache. No adapter may change until every coordinator and lease finishes.
"""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import hashlib
import json
from multiprocessing.connection import Client, Listener
import os
from pathlib import Path
import tempfile
import threading
import time
import uuid

MODE = 'shared_ready_tasks_v1'


def create_pool(manager, gpus, agent_capacity=4, doctor_capacity=50):
    directory = tempfile.mkdtemp(prefix='sherpa_pool_')
    os.chmod(directory, 0o700)
    spec = {'addresses': {g: str(Path(directory)/f'gpu{g}.sock') for g in gpus},
            'directory': directory, 'authkey': os.urandom(32),
            'actor_workers': max(8, agent_capacity + doctor_capacity),
            'request_workers': {'agent_workers': len(gpus)*agent_capacity,
                                'doctor_workers': len(gpus)*doctor_capacity},
            'lock': manager.RLock(), 'leases': manager.dict(),
            'agent': manager.Queue(), 'doctor': manager.Queue()}
    for role, capacity in (('agent', agent_capacity), ('doctor', doctor_capacity)):
        for _ in range(capacity):
            for gpu in gpus:
                spec[role].put(gpu)
    return spec


def assert_idle(spec):
    with spec['lock']:
        if spec['leases']:
            raise RuntimeError('GPU work still in flight at the policy update barrier')


class PoolClient:
    def __init__(self, spec, expected, source_gpu, timeout=1800):
        self.spec, self.expected, self.source_gpu = spec, expected, source_gpu
        self.timeout = timeout
        self.context = {}

    def call(self, op, *args):
        from src.common.live_gpu import activity
        role = 'doctor' if op == 'doctor' else 'agent'
        task = uuid.uuid4().hex
        request = {'task_id': task, 'op': op, 'args': args,
                   'policy_sha256': self.expected, 'source_gpu': self.source_gpu,
                   'context': dict(self.context)}
        with activity('pool.' + role + '_wait'):
            gpu = self.spec[role].get(timeout=self.timeout)
        with self.spec['lock']:
            self.spec['leases'][task] = {'gpu': gpu, 'op': op}
        # On timeout/transport failure the actor may still be computing. Keep its
        # lease/capacity quarantined: the runner must fail, never update or retry.
        completed = False
        try:
            with Client(self.spec['addresses'][gpu], family='AF_UNIX', authkey=self.spec['authkey']) as conn:
                conn.send(request)
                if not conn.poll(self.timeout):
                    raise TimeoutError('GPU pool response timed out; resume this run')
                response = conn.recv()
            completed = True
            if not response['ok']:
                raise RuntimeError('GPU actor failed: ' + response['error'])
            return response['value']
        finally:
            if completed:
                with self.spec['lock']:
                    del self.spec['leases'][task]
                self.spec[role].put(gpu)


class Actor:
    def __init__(self, spec, gpu, handlers, audit_dir):
        self.spec, self.gpu, self.handlers = spec, gpu, handlers
        self.expected = None
        self.audit = Path(audit_dir)/f'pool_gpu{gpu}.jsonl'
        self.audit.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.active = 0
        self.listener = Listener(spec['addresses'][gpu], family='AF_UNIX', authkey=spec['authkey'])
        self.executor = ThreadPoolExecutor(max_workers=spec.get('actor_workers', 8), thread_name_prefix=f'gpu{gpu}-actor')
        self.thread = threading.Thread(target=self._listen, daemon=True)
        self.thread.start()

    @contextmanager
    def changing_policy(self):
        with self.lock:
            if self.active:
                raise RuntimeError('Cannot change an active GPU actor policy')
            self.expected = None
            yield

    def _listen(self):
        while True:
            try:
                conn = self.listener.accept()
            except (OSError, EOFError):
                return
            try:
                request = conn.recv()
            except (EOFError, OSError):
                conn.close()
                continue
            if request.get('op') == '_close':
                conn.close()
                return
            self.executor.submit(self._handle, conn, request)

    def _handle(self, conn, request):
        from src.common.live_gpu import activity, note
        start = time.monotonic()
        acquired = False
        result = {'ok': False, 'error': 'Unknown GPU operation'}
        try:
            with self.lock:
                if self.expected is None or request['policy_sha256'] != self.expected:
                    raise RuntimeError('Frozen GPU policy mismatch or actor unavailable')
                self.active += 1
                acquired = True
            note(pool_last_source_gpu=request['source_gpu'],
                 pool_last_case_id=request['context'].get('case_id'),
                 pool_last_operation=request['op'])
            from src.online_rl.tree.idle_eval_hooks import training_guard
            with training_guard(self.gpu, request['op']), activity('pool.execute.' + request['op']):
                value = self.handlers[request['op']](*request['args'])
            result = {'ok': True, 'value': value}
        except Exception as exc:
            # No backend exception text (which may contain credentials) in RPC logs.
            result = {'ok': False, 'error': type(exc).__name__}
        finally:
            if acquired:
                with self.lock:
                    self.active -= 1
            row = {k: request[k] for k in ('task_id', 'op', 'policy_sha256', 'source_gpu', 'context')}
            row.update(gpu=self.gpu, seconds=time.monotonic()-start, finished_at=time.time(),
                       ok=result['ok'], error=result.get('error'))
            if request['op'] in ('agent', 'doctor'):
                row['seed'] = request['args'][2 if request['op']=='agent' else 1]
                row['input_sha256'] = hashlib.sha256(request['args'][0].encode()).hexdigest()
                if request['op'] == 'doctor' and len(request['args']) > 3 and request['args'][3] is not None:
                    row['system_override_sha256'] = hashlib.sha256(request['args'][3].encode()).hexdigest()
                if request['op'] == 'doctor' and len(request['args']) > 4 and request['args'][4] is not None:
                    row['format_schema_sha256'] = hashlib.sha256(
                        json.dumps(request['args'][4], sort_keys=True).encode()).hexdigest()
            try:
                with self.lock:
                    with self.audit.open('a') as file:
                        file.write(json.dumps(row)+'\n')
            except OSError:
                result = {'ok':False, 'error':'AuditWriteError'}
            try:
                conn.send(result)
            finally:
                conn.close()

    def close(self):
        with Client(self.spec['addresses'][self.gpu], family='AF_UNIX', authkey=self.spec['authkey']) as conn:
            conn.send({'op': '_close'})
        self.thread.join(timeout=10)
        self.listener.close()
        self.executor.shutdown(wait=True)


def attach(policy, cfg, spec, gpu):
    """Capture native calls before replacing case-side calls with pool dispatch."""
    from src.online_rl.integrated.grpo import behavior_logprobs
    native_generate, native_chat = policy._generate, policy.doctor._chat

    def teacher(traces):
        with policy.gen_lock:
            return behavior_logprobs(policy.model, traces, cfg['optimizer']['logprob_token_chunk_size'])

    actor = Actor(spec, gpu, {'agent': native_generate, 'doctor': native_chat, 'teacher': teacher},
                  Path(cfg['output_dir'])/'live')
    client = PoolClient(spec, None, gpu)
    # _generate accepts keyword segment/max_new_tokens; normalize before RPC.
    def generate(prompt, max_new_tokens, seed, segment='agent'):
        return client.call('agent', prompt, max_new_tokens, seed, segment)
    policy._generate = generate
    def chat(user, seed, max_tokens=None, system_override=None, format_schema=None):
        args = ((user, seed, max_tokens, system_override, format_schema) if format_schema is not None else
                (user, seed, max_tokens, system_override) if system_override is not None else
                ((user, seed) if max_tokens is None else (user, seed, max_tokens)))
        return client.call('doctor', *args)
    policy.doctor._chat = chat
    policy.pool_behavior_logprobs = lambda traces: client.call('teacher', traces)
    policy.pool_actor, policy.pool_client = actor, client
    # Case submission windows span the whole pool; native per-GPU capacity stays
    # Agent4 / Doctor50. These runtime fields never alter config/cache identity.
    policy.pool_execution = dict(spec['request_workers'])
    return actor


def case_policy_view(policy, context):
    """Share frozen weights/actors, isolate each case's RPC context and Doctor proxy."""
    import copy
    view = copy.copy(policy)
    view.doctor = copy.copy(policy.doctor)
    parent = policy.pool_client
    client = PoolClient(parent.spec,parent.expected,parent.source_gpu,parent.timeout)
    client.context = dict(context)
    view.pool_client = client
    view._generate = lambda prompt,max_new_tokens,seed,segment='agent': client.call('agent',prompt,max_new_tokens,seed,segment)
    def chat(user,seed,max_tokens=None,system_override=None,format_schema=None):
        args=((user,seed,max_tokens,system_override,format_schema) if format_schema is not None else
              (user,seed,max_tokens,system_override) if system_override is not None else
              (user,seed) if max_tokens is None else (user,seed,max_tokens))
        return client.call('doctor',*args)
    view.doctor._chat = chat
    if hasattr(policy.doctor, '_shared_opening_cache'):
        from src.online_rl.tree.five_gpu import cache_openings
        # Rebind the native method to this case's Doctor before wrapping D1;
        # copying the old closure would retain another case's RPC client.
        view.doctor.turn = type(view.doctor).turn.__get__(view.doctor, type(view.doctor))
        cache_openings(view.doctor, policy.doctor._shared_opening_path,
                       policy.doctor._shared_opening_identity, cache=policy.doctor._shared_opening_cache)
    view.pool_behavior_logprobs = lambda traces: client.call('teacher',traces)
    return view
