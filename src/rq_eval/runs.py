"""Load runs for RQ1-RQ3: verified provenance, prompt lineage, milestone evaluations and an evaluation reply function.
Case-tree runs (load_tree_run, tree_*) and the earlier integrated runs (load_run, prompt_lineage, onpolicy_rubrics) are
both supported. Provenance comes from run_manifest.json + run_state.json (never from directory names) and is checked
against the files."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Callable
from src.online_rl.integrated.conditions import unmatched_keys
from src.online_rl.integrated.records import dir_sha256, read_json, sha256_json, sha256_text
from src.online_rl.integrated.rubric import STATE_SCHEMA_SHA256


def load_run(run_dir: Path) -> dict:
    run_dir = Path(run_dir); m = read_json(run_dir / 'run_manifest.json'); s = read_json(run_dir / 'run_state.json'); cfg = m['config']
    problems = []
    if sha256_json(cfg) != s['config_sha256'] or m['config_sha256'] != s['config_sha256']: problems.append('config hash mismatch')
    if dir_sha256(Path(s['checkpoint'])) != s['checkpoint_sha256']: problems.append('checkpoint files changed')
    if sha256_text((run_dir / 'prompts' / f"{s['prompt_sha256']}.txt").read_text()) != s['prompt_sha256']: problems.append('prompt text mismatch')
    if m['state_schema']['sha256'] != STATE_SCHEMA_SHA256: problems.append('state schema differs from the current code')
    if problems: raise RuntimeError(f'{run_dir}: ' + '; '.join(problems))
    return {'run_dir': str(run_dir), 'condition': m['condition'], 'prompt_mode': m['prompt_mode'], 'base_model': cfg['policy']['base_model'],
            'checkpoint': s['checkpoint'], 'checkpoint_sha256': s['checkpoint_sha256'], 'adapter_sha256': s['adapter_sha256'],
            'step': s['step'], 'optimizer_steps': s['optimizer_steps'], 'finished': s['finished'], 'seed': s['run_seed'],
            'initial_prompt_sha256': m['initial_prompt']['sha256'], 'prompt_id': s['prompt_id'], 'prompt_sha256': s['prompt_sha256'],
            'accepted_prompts': s['history'], 'state_schema_version': m['state_schema']['version'], 'state_schema_sha256': m['state_schema']['sha256'],
            'reward': {'w_O': cfg['reward']['w_O'], 'w_A': cfg['reward']['w_A']}, 'advantage_mode': m['advantage_mode'],
            'config_sha256': s['config_sha256'], 'config': cfg}


def base_provenance(cfg: dict) -> dict:
    """The untrained Base condition: no adapter, no prompt."""
    return {'condition': 'base', 'prompt_mode': 'none', 'base_model': cfg['policy']['base_model'], 'adapter_sha256': None, 'checkpoint': None, 'step': 0}


def public(prov: dict) -> dict:
    return {k: v for k, v in prov.items() if k != 'config'}


def prompt_lineage(run_dir: Path) -> list[dict]:
    """One row per prompt version of every committed event (immutable lineage)."""
    run_dir = Path(run_dir); cursor = read_json(run_dir / 'run_state.json')['lineage_cursor']; rows = []
    for e in range(1, cursor + 1):
        ev = read_json(run_dir / 'prompt_events' / f'event_{e:04d}.json')
        for v in ev['versions']:
            rows.append({'event': e, 'step': ev['step'], 'parent_prompt_id': v['parent_prompt_id'], 'prompt_id': v['prompt_id'], 'sha256': v['sha256'],
                         'depth': v['depth'], 'rank': v['rank'], 'status': v['status'], 'accepted': v['status'] == 'accepted',
                         'mean_paired_Q_gain': (v['acceptance'] or {}).get('mean_paired_Q_gain'), 'rejection_reason': v['rejection_reason']})
    return rows


def onpolicy_rubrics(run_dir: Path) -> list[dict]:
    """Committed training rubrics (for theme analysis only; never an evaluation criterion)."""
    run_dir = Path(run_dir); cursor = read_json(run_dir / 'run_state.json')['evidence_cursor']; out = []
    for i in range(cursor):
        e = read_json(run_dir / 'evidence' / f'step_{i:06d}.json')
        if e['rubric']['valid']: out.append({'step': i, 'prompt_id': e['prompt_id'], 'case_id': e['case_id'], 'turn': e['turn'], 'rubric': e['rubric']['rubric']})
    return out


def rq3_pair_problems(fixed: dict, evolving: dict) -> list[str]:
    """Fixed and evolving must share P0, state schema and every setting except prompt evolution."""
    if fixed.get('format') == 'tree' or evolving.get('format') == 'tree': return rq3_tree_pair_problems(fixed, evolving)
    out = []
    if (fixed['prompt_mode'], evolving['prompt_mode']) != ('fixed', 'evolving'): out.append('arms are not fixed/evolving')
    for k in ('initial_prompt_sha256', 'state_schema_sha256', 'base_model', 'seed', 'advantage_mode', 'reward'):
        if fixed[k] != evolving[k]: out.append(f'{k} differs')
    diff = unmatched_keys(fixed['config'], evolving['config'])
    if diff: out.append(f'config differs at {sorted(diff)}')
    return out


RQ2_AGENT_PROTOCOL = 'precommit_v1'


def require_rq2_protocol(cfg: dict) -> None:
    """Every RQ2 policy (Base and trained) acts with the finalized shared Agent protocol it is trained and VAL50-evaluated with."""
    got = (cfg.get('rollout') or {}).get('agent_protocol')
    if got != RQ2_AGENT_PROTOCOL: raise RuntimeError(f'RQ2 requires rollout.agent_protocol = {RQ2_AGENT_PROTOCOL!r}, got {got!r}')


def policy_reply_fn(policy, seed_fn: Callable) -> Callable:
    """Evaluation reply with the shared precommit_v1 Agent action (src/online_rl/agent_protocol.py), identical for Base and every
    trained policy: a private Stage-A assessment blind to the dialogue, then the visible Stage-B reply. Only the visible reply is
    returned, so judges and continuations never see the private assessment; no rubric, R_A or s_j is involved. The seed depends
    only on (case, history), so same-history paired comparisons are unchanged: (case, prefix, turn) -> text."""
    if not getattr(policy, 'precommit', False): raise RuntimeError(f'RQ2 replies require the {RQ2_AGENT_PROTOCOL} Agent protocol')
    return lambda case, prefix, turn: policy.agent_text(case, prefix, turn, seed_fn(case['case_id'], json.dumps([d['text'] for d in prefix])))


# ---------- case-tree runs ----------
TREE_VERSION = 'sherpa_tree_v1'


def tree_milestones(run_dir: Path) -> list[dict]:
    """Completed milestone evaluations (eval/<checkpoint>/result.json), each checked against its retained checkpoint."""
    run_dir = Path(run_dir); out = []
    for p in sorted((run_dir / 'eval').glob('checkpoint_*/result.json')):
        r = read_json(p)
        if r.get('status') != 'complete': continue
        if Path(r['checkpoint']).exists() and dir_sha256(Path(r['checkpoint'])) != r['checkpoint_sha256']:
            raise RuntimeError(f'{p}: milestone checkpoint files changed')
        out.append(r)
    return sorted(out, key=lambda r: r['milestone'])


def load_tree_run(run_dir: Path) -> dict:
    """Verified provenance of a case-tree run: the committed checkpoint (directory hash), its adapter hash and the active prompt."""
    run_dir = Path(run_dir); m = read_json(run_dir / 'run_manifest.json'); s = read_json(run_dir / 'run_state.json'); cfg = m['config']
    if m.get('version') != TREE_VERSION: raise RuntimeError(f'{run_dir} is not a case-tree run')
    problems = []
    if sha256_json(cfg) != s['config_sha256'] or m['config_sha256'] != s['config_sha256']: problems.append('config hash mismatch')
    if sha256_text((run_dir / 'prompts' / f"{s['prompt_sha256']}.txt").read_text()) != s['prompt_sha256']: problems.append('prompt text mismatch')
    if not s['checkpoint'] or not Path(s['checkpoint']).exists(): problems.append('committed checkpoint missing')
    if problems: raise RuntimeError(f'{run_dir}: ' + '; '.join(problems))
    from src.online_rl.tree.conditions import arm_of
    arm = arm_of(cfg)
    return {'run_dir': str(run_dir), 'format': 'tree', 'condition': f'tree_{arm}', 'prompt_mode': arm, 'base_model': cfg['policy']['base_model'],
            'checkpoint': s['checkpoint'], 'checkpoint_sha256': dir_sha256(Path(s['checkpoint'])), 'adapter_sha256': s['policy_sha256'],
            'case_cursor': s['case_cursor'], 'total_cases': cfg['schedule']['cases'], 'finished': s['case_cursor'] >= cfg['schedule']['cases'],
            'seed': s['run_seed'], 'initial_prompt_sha256': m['initial_prompt_sha256'], 'prompt_id': s['prompt_id'], 'prompt_sha256': s['prompt_sha256'],
            'accepted_prompts': s['history'], 'protocol_sha256': m['protocol_sha256'], 'reward': dict(cfg.get('agent_reward') or {}),
            'advantage_mode': (cfg.get('advantage_normalization') or {}).get('mode'), 'milestones': [public_milestone(r) for r in tree_milestones(run_dir)],
            'config_sha256': s['config_sha256'], 'config': cfg}


def public_milestone(r: dict) -> dict:
    return {k: r[k] for k in ('milestone', 'checkpoint', 'checkpoint_sha256', 'policy_sha256', 'prompt_sha256', 'split', 'manifest_sha256',
                              'eval_protocol_sha256', 'accuracy', 'n_cases', 'completed_at')}


def rq1_milestone_inputs(run_dir: Path, label: str) -> tuple[dict, dict]:
    """(results, provenance) for src.rq_eval.main_table.table, one condition per milestone: '<label>@<milestone>'."""
    results, prov = {}, {}
    for r in tree_milestones(run_dir):
        c = f"{label}@{r['milestone']:06d}"
        results[c] = {r['split']: [{'case_id': x['case_id'], 'correct': x['correct']} for x in r['per_case']]}
        prov[c] = {'format': 'tree', 'run_dir': str(run_dir), **public_milestone(r)}
    return results, prov


def tree_prompt_lineage(run_dir: Path) -> list[dict]:
    """One row per prompt version of every committed tree prompt event, with the directional scores behind J."""
    run_dir = Path(run_dir); s = read_json(run_dir / 'run_state.json'); rows = []
    for e in range(1, s['lineage_cursor'] + 1):
        ev = read_json(run_dir / 'prompt_events' / f'event_{e:04d}.json'); shas = {ev['parent_prompt_id']: ev['parent_sha256']}
        shas.update({v['prompt_id']: v['sha256'] for v in ev['versions']})
        for v in ev['versions']:
            sc = (v['metrics'] or {}).get('search') or {}; acc = v['acceptance'] or {}
            rows.append({'event': e, 'case_cursor': ev['case_cursor'], 'prompt_id': v['prompt_id'], 'sha256': v['sha256'] or None,
                         'parent_prompt_id': v['parent_prompt_id'], 'parent_sha256': shas.get(v['parent_prompt_id']), 'depth': v['depth'],
                         'status': v['status'], 'accepted': v['status'] == 'accepted',
                         'search': {k: sc.get(k) for k in ('S_success', 'S_failure', 'J', 'n_success', 'n_failure', 'invalid', 'no_match')},
                         'acceptance': {'parent': {k: (acc.get('parent') or {}).get(k) for k in ('S_success', 'S_failure', 'J')},
                                        'candidate': {k: (acc.get('candidate') or {}).get(k) for k in ('S_success', 'S_failure', 'J')},
                                        'observed_dJ': acc.get('observed_dJ'), 'p_gain': acc.get('p_gain'),
                                        'n_success': acc.get('n_success'), 'n_failure': acc.get('n_failure')} if acc else None,
                         'rerun_policy_sha256': ev.get('rerun_policy_sha256'), 'rejection_reason': v['rejection_reason']})
    return rows


def rq3_tree_pair_problems(fixed: dict, evolving: dict) -> list[str]:
    """Fixed (P0 throughout) and Evolving (J / bootstrap prompt updates) tree runs must match in everything else."""
    out = []
    if (fixed.get('format'), evolving.get('format')) != ('tree', 'tree'): return ['both RQ3 arms must be case-tree runs']
    if (fixed['prompt_mode'], evolving['prompt_mode']) != ('fixed', 'evolving'): out.append('arms are not fixed/evolving')
    if len(fixed['accepted_prompts']) != 1: out.append('the fixed arm changed its prompt')
    for k in ('initial_prompt_sha256', 'base_model', 'seed', 'advantage_mode', 'reward', 'protocol_sha256', 'total_cases'):
        if fixed[k] != evolving[k]: out.append(f'{k} differs')
    diff = {k for k in unmatched_keys(fixed['config'], evolving['config']) if k != 'name'}
    if diff: out.append(f'config differs at {sorted(diff)}')
    return out


def load_tree_policy(cfg: dict, checkpoint: Path, adapter_sha256: str, key: str = ''):
    """Agent policy of a tree checkpoint for evaluation (RQ2): the base model plus the checkpoint adapter, verified by hash.
    Replies come from policy_reply_fn (precommit_v1 action); no training rubric or signal is involved."""
    require_rq2_protocol(cfg)
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file
    from src.online_rl.integrated.continuation import adapter_sha256 as adapter_hash
    from src.online_rl.runtime import SharedPolicy
    policy = SharedPolicy(cfg, key)
    set_peft_model_state_dict(policy.model, load_file(str(Path(checkpoint) / 'adapter' / 'adapter_model.safetensors')))
    getattr(policy, 'mark_dirty', lambda: None)()
    if adapter_hash(policy.model) != adapter_sha256: raise RuntimeError(f'{checkpoint}: adapter hash differs from the recorded policy')
    policy.model.eval(); return policy


def load_base_policy(cfg: dict, key: str = ''):
    """The untrained Base policy for RQ2: the seeded initial adapter (LoRA B = 0, i.e. the base model's function), acting with the
    same precommit_v1 Agent protocol as the trained policies."""
    require_rq2_protocol(cfg)
    if cfg['policy'].get('init_adapter'): raise RuntimeError('Base must start from the fresh seeded adapter, not policy.init_adapter')
    from src.online_rl.runtime import SharedPolicy
    policy = SharedPolicy(cfg, key)
    if any(float(p.abs().sum()) != 0 for n, p in policy.model.named_parameters() if 'lora_B' in n):
        raise RuntimeError('Base adapter is not the identity (non-zero lora_B)')
    policy.model.eval(); return policy
