"""Original-base inference through two managed, localhost-only vLLM servers.

The training policy remains Transformers/PEFT. No remote API or SDK retry is
used here. Existing frozen base evidence can be resumed across the engine change;
new traces explicitly identify the vLLM sampler (seeds are not cross-engine equal).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import threading
import time
from types import SimpleNamespace
import urllib.request

from src.online_rl.runtime import SharedPolicy


class VLLMPolicy:
    original_base_only = True
    collect_training_traces = False
    precommit = False
    agent_user_prompt = SharedPolicy.agent_user_prompt
    _H = SharedPolicy._H
    agent_turn = SharedPolicy.agent_turn
    sample_completions = SharedPolicy.sample_completions

    def __init__(self, cfg, root):
        from transformers import AutoTokenizer
        from src.online_rl import environment
        from src.online_rl.doctor import make_doctor
        self.cfg, self.root, self.ST = cfg, Path(root), environment
        ro = cfg['rollout']
        self.args = SimpleNamespace(max_completion_length=ro['max_completion_length'], rounds=ro['rounds'])
        self.agent_system = (self.root / ro.get('agent_system_prompt_file', 'prompts/agent.txt')).read_text()
        self.doctor = make_doctor(ro, (self.root / ro['doctor_system_prompt_file']).read_text())
        self.model_path = (self.root / cfg['policy']['local_model_path']).resolve()
        self.tok = AutoTokenizer.from_pretrained(self.model_path, local_files_only=True)
        generation = json.loads((self.model_path / 'generation_config.json').read_text())
        eos = generation.get('eos_token_id', self.tok.eos_token_id)
        self.stop_token_ids = eos if isinstance(eos, list) else [eos]
        spec = cfg['prompt_optimization']
        self.limit = threading.BoundedSemaphore(spec['assistant_max_inflight'])
        self.guard = threading.Lock()
        self.loads = [0] * spec['rollout_replicas']
        self.servers, self.urls, self.logs = [], [], []
        self.home = self.root / spec['output_dir']
        self.home.mkdir(parents=True, exist_ok=True)
        try:
            self._start(spec)
        except BaseException:
            self.close()
            raise

    def _start(self, spec):
        python = self.root / spec['vllm']['python']
        devices = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
        deadline = time.monotonic() + spec['vllm'].get('startup_timeout_seconds', 900)
        for index in range(spec['rollout_replicas']):
            with socket.socket() as sock:
                sock.bind(('127.0.0.1', 0))
                port = sock.getsockname()[1]
            command = server_command(python, self.model_path, port, spec)
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = devices[index]
            env['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
            # Local inference needs no cloud credentials.
            for key in ('OPENAI_API_KEY', 'OPENROUTER_API_KEY'):
                env.pop(key, None)
            log = (self.home / f'vllm_gpu{index}.log').open('a')
            self.logs.append(log)
            self.servers.append(subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                                  start_new_session=True))
            self.urls.append(f'http://127.0.0.1:{port}')
            print(f'vLLM GPU {devices[index]} starting; log: {log.name}', flush=True)
        for index, (proc, url) in enumerate(zip(self.servers, self.urls)):
            while True:
                if proc.poll() is not None:
                    raise RuntimeError(f'vLLM server exited ({proc.returncode}); inspect {self.logs[index].name}')
                try:
                    with urllib.request.urlopen(url + '/health', timeout=2) as response:
                        if response.status == 200:
                            break
                except OSError:
                    pass
                if time.monotonic() >= deadline:
                    raise TimeoutError(f'vLLM startup timed out; inspect {self.logs[index].name}')
                time.sleep(1)
        from src.online_rl.integrated.records import atomic_write_json
        atomic_write_json(self.home / 'inference_runtime.json', {
            'engine': 'vllm', 'sampler': 'vllm_seeded_v1', 'servers': self.urls,
            'pids': [p.pid for p in self.servers], 'devices': devices[:len(self.servers)],
            'settings': spec['vllm'], 'max_inflight': spec['assistant_max_inflight'],
            'resume_existing_frozen_evidence': True,
            'cross_engine_seed_equivalence': False})

    def _generate(self, prompt, max_new_tokens, seed):
        ro = self.cfg['rollout']
        # Encode with the existing tokenizer/template. Avoid server-side template
        # changes and request the actual generated token IDs, not re-tokenized text.
        payload = {'model': 'sherpa-base', 'prompt': self.tok.encode(prompt, add_special_tokens=False),
                   'max_tokens': max_new_tokens, 'temperature': ro['temperature'],
                   'top_p': ro['top_p'], 'top_k': ro['top_k'] or -1, 'seed': seed,
                   'repetition_penalty': 1.0, 'min_p': 0.0, 'return_token_ids': True,
                   'stop_token_ids': self.stop_token_ids}
        with self.limit:
            with self.guard:
                index = min(range(len(self.loads)), key=self.loads.__getitem__)
                self.loads[index] += 1
            try:
                from src.common.live_gpu import activity
                request = urllib.request.Request(self.urls[index] + '/v1/completions',
                    data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
                with activity('agent_generate'), urllib.request.urlopen(request, timeout=600) as response:
                    choice = json.load(response)['choices'][0]
                ids = choice.get('token_ids')
                if not isinstance(ids, list) or not all(type(token) is int for token in ids):
                    raise RuntimeError('vLLM did not return generated token IDs')
                return {'completion_text': choice['text'], 'completion_ids': ids,
                        'sampling_temperature': ro['temperature'], 'sampling_top_p': ro['top_p'],
                        'sampling_top_k': ro['top_k'], 'batch_sampling_protocol': 'vllm_seeded_v1'}
            finally:
                with self.guard:
                    self.loads[index] -= 1

    def close(self):
        # Kill whole process groups, including vLLM engine children, even when the
        # HTTP process has already failed. Never touch unrelated GPU processes.
        for proc in self.servers:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        for proc in self.servers:
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                pass
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()
        for log in self.logs:
            log.close()


def server_command(python, model_path, port, spec):
    v = spec['vllm']
    return [str(python), '-m', 'vllm.entrypoints.openai.api_server',
            '--model', str(model_path), '--served-model-name', 'sherpa-base',
            '--host', '127.0.0.1', '--port', str(port), '--tensor-parallel-size', '1',
            '--dtype', 'bfloat16', '--gpu-memory-utilization', str(v['gpu_memory_utilization']),
            '--max-model-len', str(v['max_model_len']), '--max-num-seqs', str(v['max_num_seqs']),
            '--generation-config', 'vllm', '--no-enable-log-requests']
