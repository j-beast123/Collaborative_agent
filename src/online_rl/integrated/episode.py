"""Case schedule, pre-action state realisation and same-prompt group checks."""
from __future__ import annotations
import hashlib, json, random


def schedule(case_ids: list[str], seed: int, step: int, per_step: int) -> list[str]:
    """Epoch-wise seeded permutation; step g takes the g-th slice, so a resumed run continues identically."""
    per_epoch = len(case_ids) // per_step
    epoch, idx = divmod(step, per_epoch)
    order = sorted(case_ids); random.Random(seed * 1_000_003 + epoch).shuffle(order)
    return order[idx * per_step:(idx + 1) * per_step]


def sample_turn(reached: int, rule: str, rng: random.Random) -> int:
    if reached < 1: raise ValueError('rollout reached no Agent turn')
    if rule == 'uniform_reached': return rng.randint(1, reached)
    if rule == 'last_reached': return reached
    raise ValueError(f'unknown turn sampling rule {rule}')


def pre_action_prefix(dialogue: list[dict], agent_index: int) -> list[dict]:
    """History strictly before the Agent turn (ends with the Doctor message of that round)."""
    if dialogue[agent_index]['speaker'] != 'agent': raise ValueError('pre_action_prefix needs an Agent turn index')
    return [{k: d[k] for k in ('speaker', 'stage', 'text', 'round') if k in d} for d in dialogue[:agent_index]]


def prompt_hash(trace: dict) -> str:
    ids = trace.get('prompt_input_ids')
    return hashlib.sha256((json.dumps(ids) if ids is not None else trace['prompt_text']).encode()).hexdigest()


def assert_single_prompt(records: list[dict]) -> str:
    """One completion per record and one identical policy prompt for the whole group; otherwise no loss."""
    if not records: raise ValueError('empty GRPO group')
    if any(len(r['agent_traces']) != 1 for r in records): raise RuntimeError('each record must hold exactly one completion')
    hashes = {prompt_hash(r['agent_traces'][0]) for r in records}
    if len(hashes) != 1: raise RuntimeError(f'GRPO group mixes {len(hashes)} prompt hashes; refusing to compute the loss')
    if len({r['case_id'] for r in records}) != 1: raise RuntimeError('GRPO group key differs inside one group')
    return hashes.pop()
