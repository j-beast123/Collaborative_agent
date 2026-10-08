"""Held-out evaluation of one Agent adapter outside training (read-only diagnostic): the milestone evaluator of
run_tree.py (full Doctor-Agent episodes on the configured split, plain Agent prompt) on the initial adapter
(--initial: the seeded fresh LoRA of policy.lora.init_seed, equal to the base model) or on a saved checkpoint directory
(e.g. a run's checkpoints/checkpoint_000000). No training state is touched.

The checkpoint identity (checkpoint_sha256, the reuse key of cases.jsonl) is the hash of the adapter weights actually
loaded, so --initial with seed S and a run's checkpoint_000000 made with seed S share it, and a partial evaluation
resumes from its completed cases. An --out holding rows of another checkpoint or protocol is refused, never appended to."""
import argparse, json, os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))


def foreign_rows(out: Path, key: tuple) -> int:
    """Number of rows in out/cases.jsonl (and out/result.json) that belong to another (checkpoint, protocol)."""
    n = 0
    if (out / 'cases.jsonl').exists():
        for line in (out / 'cases.jsonl').read_text().splitlines():
            r = json.loads(line); n += (r.get('checkpoint_sha256'), r.get('eval_protocol_sha256')) != key
    if (out / 'result.json').exists():
        r = json.loads((out / 'result.json').read_text()); n += (r.get('checkpoint_sha256'), r.get('eval_protocol_sha256')) != key
    return n


def reusable_rows(out: Path, key: tuple) -> int:
    if not (out / 'cases.jsonl').exists(): return 0
    return len({r['case_id'] for r in map(json.loads, (out / 'cases.jsonl').read_text().splitlines())
                if (r.get('checkpoint_sha256'), r.get('eval_protocol_sha256')) == key})


def run_initial_sha256(checkpoint: Path):
    """initial_policy_sha256 of the run that owns checkpoint_000000 (None for any other directory)."""
    man = checkpoint.parent.parent / 'run_manifest.json'
    if checkpoint.name != 'checkpoint_000000' or not man.exists(): return None
    return json.loads(man.read_text()).get('initial_policy_sha256')


def main():
    a = argparse.ArgumentParser(); a.add_argument('--config', required=True); a.add_argument('--out', required=True)
    g = a.add_mutually_exclusive_group(required=True); g.add_argument('--initial', action='store_true'); g.add_argument('--checkpoint')
    a.add_argument('--launch', action='store_true'); x = a.parse_args()
    import run_tree as RT
    from src.online_rl.environment import final_outcome
    from src.online_rl.integrated.continuation import adapter_sha256
    from src.online_rl.integrated.records import dir_sha256
    cfg = json.loads((ROOT / x.config).read_text()); out = ROOT / x.out
    if any(p in out.resolve().parents or p == out.resolve() for p in [(ROOT / cfg['output_dir']).resolve() / d for d in ('cases', 'checkpoints')]):
        raise SystemExit('--out must not be inside a run\'s cases/checkpoints')
    seed = (cfg['policy'].get('lora') or {}).get('init_seed')
    if x.initial and (cfg['policy'].get('init_adapter') or not isinstance(seed, int) or isinstance(seed, bool)):
        raise SystemExit('--initial needs policy.lora.init_seed (integer) and no init_adapter')
    print(json.dumps({'config': x.config, 'split': cfg['evaluation']['split'], 'cases_file': cfg['evaluation']['cases_file'], 'out': x.out,
                      'lora_init_seed': seed, 'checkpoint': x.checkpoint or 'initial'}), flush=True)
    if not x.launch: raise SystemExit('not running (pass --launch)')
    os.environ.setdefault('API_USAGE_LOG', str(out / 'api_usage.jsonl'))
    policy = RT.make_policy(cfg, RT.load_key())
    if x.checkpoint:
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        set_peft_model_state_dict(policy.model, load_file(str(Path(x.checkpoint) / 'adapter' / 'adapter_model.safetensors'))); getattr(policy, 'mark_dirty', lambda: None)()
    sha = adapter_sha256(policy.model)
    want = run_initial_sha256(Path(x.checkpoint)) if x.checkpoint else None
    if want and sha != want: raise SystemExit(f'loaded adapter {sha[:12]} differs from the run\'s initial_policy_sha256 {want[:12]}')
    ev = RT.milestone_evaluator(cfg, policy, final_outcome)
    if ev is None: raise SystemExit('the config has no evaluation block')
    key = (sha, ev.info['eval_protocol_sha256'])
    if foreign_rows(out, key): raise SystemExit(f'{x.out} holds results of another checkpoint or protocol; use a new --out')
    print(json.dumps({'checkpoint_sha256': sha, 'eval_protocol_sha256': key[1], 'reused_cases': reusable_rows(out, key)}), flush=True)
    prov = {'milestone': 0 if x.initial else None,
            'checkpoint': f'initial adapter (fresh LoRA, init_seed {seed}, equal to the base model)' if x.initial else x.checkpoint,
            'checkpoint_sha256': sha, 'checkpoint_dir_sha256': None if x.initial else dir_sha256(Path(x.checkpoint)), 'policy_sha256': sha,
            'lora_init_seed': seed, 'prompt_id': None, 'prompt_sha256': None, 'config_sha256': None}
    r = ev(milestone=prov['milestone'] or 0, checkpoint=Path(prov['checkpoint']), out_dir=out, provenance=prov)
    print(json.dumps({k: r[k] for k in ('accuracy', 'n_cases', 'split', 'policy_sha256', 'eval_protocol_sha256')}), flush=True)


if __name__ == '__main__': sys.path.insert(0, str(ROOT / 'scripts/online_rl')); main()
