"""Local-only assistant smoke: configured replicas, short traces, one GRPO update."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import fcntl
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    os.chdir(ROOT)
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0,1')
    os.environ['SHERPA_AGENT_BATCH_SIZE'] = '64'
    os.environ['SHERPA_AGENT_MAX_INFLIGHT'] = '128'
    os.environ['SHERPA_AGENT_PROCESS_WORKERS'] = '0'
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['OMP_NUM_THREADS'] = '4'
    from src.online_rl.tree.experiment import load_config
    from src.online_rl.runtime import SharedPolicy
    from src.online_rl.integrated.continuation import adapter_sha256
    from src.online_rl.integrated.grpo import make_updater
    cfg = load_config('configs/ours.json')
    replicas = cfg['training_execution']['rollout_replicas']
    os.environ['SHERPA_AGENT_REPLICAS'] = str(replicas)
    with ExitStack() as stack:
        for gpu in os.environ['CUDA_VISIBLE_DEVICES'].split(',')[:replicas]:
            lock = stack.enter_context(Path(f'/tmp/sherpa_tree_physical_gpu{gpu}.lock').open('a'))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        policy = SharedPolicy(cfg, '')
        def close():
            if hasattr(policy, '_batcher'): policy._batcher.close()
        stack.callback(close)
        before = adapter_sha256(policy.model)
        policy.sync_replicas(expected_sha256=before)
        prompt = policy.ST.chat_prompt(policy.tok, 'You are a helpful assistant.', 'Count from one to five.')
        with ThreadPoolExecutor(128) as pool:
            traces = list(pool.map(lambda seed: policy._generate(prompt, 8, seed), range(128)))
        for trace in traces:
            trace['behavior_logprobs'] = trace['old_policy_logprobs']
        groups = [[{'trajectory_id':str(i), 'case_id':str(i//32), 'reward':i%2,
                    'rollout_policy_sha256':before, 'agent_traces':[traces[i]]}
                   for i in range(start,start+32)] for start in range(0,128,32)]
        result = make_updater(policy, cfg).update_groups(groups)
        assert result['optimizer_step'] and result['optimizer_steps'] == 1
        after = adapter_sha256(policy.model)
        assert before != after
        synced = policy.sync_replicas(expected_sha256=after)
        assert len(set(synced['adapter_sha256'])) == 1
        print(json.dumps({'optimizer_steps':1,'groups':4,'synthetic_trajectories':128,
                          'ratio':result['ratio'],'replicas_synchronized':True}),flush=True)


if __name__ == '__main__':
    main()
