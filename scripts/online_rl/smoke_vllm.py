"""Local-only two-GPU vLLM smoke test. Never calls Doctor/generator/judge APIs."""
import fcntl
import json
import os
from pathlib import Path
import sys
import tempfile
from contextlib import ExitStack
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    from src.online_rl.vllm_policy import VLLMPolicy
    os.chdir(ROOT)
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0,1')
    os.environ.setdefault('OMP_NUM_THREADS', '4')
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    cfg = json.loads((ROOT / 'configs/sherpa.json').read_text())
    with ExitStack() as stack:
        for gpu in os.environ['CUDA_VISIBLE_DEVICES'].split(',')[:cfg['prompt_optimization']['rollout_replicas']]:
            lock = stack.enter_context(Path(f'/tmp/sherpa_tree_physical_gpu{gpu}.lock').open('a'))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        output = tempfile.mkdtemp(prefix='sherpa-vllm-smoke-')
        cfg['prompt_optimization']['output_dir'] = output
        print('Smoke logs: ' + output, flush=True)
        policy = VLLMPolicy(cfg, ROOT)
        stack.callback(policy.close)
        prompt = policy.ST.chat_prompt(policy.tok, 'You are a helpful assistant.', 'Count from one to five.')
        with ThreadPoolExecutor(128) as pool:
            rows = list(pool.map(lambda seed: policy._generate(prompt, 16, seed), range(128)))
        assert len(rows) == 128 and all(row['completion_ids'] for row in rows)
        policy.args.max_completion_length = 16
        case = {'case_id': 'synthetic', 'question': 'What is two plus two?', 'options': {'A': '4', 'B': '5'}}
        prefix = [{'speaker': 'doctor', 'text': 'I think the answer is A.'}]
        assert policy.sample_completions(case, prefix, 1, [1001])[0]['completion_ids']
        assert policy.sample_completions(case, prefix, 2, [1002])[0]['completion_ids']
        print(json.dumps({'requests': len(rows), 'tokens': sum(len(r['completion_ids']) for r in rows),
                          'sample': rows[0]['completion_text'], 'logs': output}), flush=True)


if __name__ == '__main__':
    main()
