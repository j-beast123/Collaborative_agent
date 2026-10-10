"""PASS/FAIL checks of a 5-case pilot smoke run (infrastructure only; accuracy is never read as evidence).
    python scripts/dev_opt/verify_pilot_smoke.py <run dir> [--gpu-log outputs/logs/smoke_gpu_mem.log]"""
from __future__ import annotations
import argparse, glob, json, math, re, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
EPS = 1e-9


def main():
    a = argparse.ArgumentParser(); a.add_argument('run'); a.add_argument('--gpu-log'); x = a.parse_args(); run = Path(x.run)
    from src.online_rl.tree.team import compose_reward
    man = json.loads((run / 'run_manifest.json').read_text()); cfg = man['config']; credit = cfg['agent_reward'].get('credit', 'shared')
    w = {k: cfg['agent_reward'][k] for k in ('w_outcome', 'w_rubric')}; think = bool(cfg['rollout'].get('agent_thinking'))
    recs = [json.load(open(f)) for f in sorted(glob.glob(str(run / 'cases' / 'case_*.json')))]; C = {}; notes = {}
    C['5 cases committed'] = len(recs) == 5 and json.loads((run / 'run_state.json').read_text())['case_cursor'] == 5
    segs = [(n, s) for r in recs for n in r['tree']['nodes'] for s in n['trace']['segments']]; nodes = [n for r in recs for n in r['tree']['nodes']]
    C['thinking: every Agent prompt is rendered with thinking on (no pre-filled empty think block)'] = think and all('<think>\n\n</think>' not in s['prompt_text'] for _, s in segs)
    C['thinking: reasoning tokens are in the trained completion'] = all(s['think_tokens'] > 0 for _, s in segs) and all(('</think>' in s['completion_text']) == s['thinking_closed'] for _, s in segs)
    notes['think_tokens mean / max'] = [round(sum(s['think_tokens'] for _, s in segs) / len(segs)), max(s['think_tokens'] for _, s in segs)]
    notes['unclosed thinking segments'] = f"{sum(not s['thinking_closed'] for _, s in segs)} / {len(segs)}"
    C['old logprobs: behaviour logprobs cover every token of both segments'] = all(len(n['trace']['behavior_logprobs']) == sum(len(s['completion_ids']) for s in n['trace']['segments']) for n in nodes)
    leak = lambda t: ('<think>' in (t or '')) or ('</think>' in (t or ''))
    C['stripping: no think markup in the answers used downstream'] = not any(leak(s['answer_text']) for _, s in segs) and not any(leak(n['response']) or leak(n['trace']['private_assessment']) for n in nodes)
    C['stripping: no think markup anywhere in the dialogues'] = not any(leak(t[3]) for r in recs for l in r['tree']['leaves'] for t in l['dialogue']) and not any(leak(t[3]) for n in nodes for t in n['prefix'])
    hidden = [(n, n['trace']['segments'][1]['prompt_text'].split('[Hidden context: never mention it]')[-1]) for n in nodes]
    C['stripping: the hidden block carries the private answer only'] = all(not leak(h) and (n['trace']['private_assessment'] in h) for n, h in hidden)
    # the visible reply must not contain the reasoning of its own completion (checked on the longest reasoning sentence)
    def reasoning_leaks(s):
        think = s['completion_text'].split('</think>')[0].replace('<think>', ''); sent = max((p.strip() for p in re.split(r'[.\n]', think)), key=len, default='')
        return len(sent) > 40 and sent in s['answer_text']
    C['stripping: reasoning text does not reappear in the consumed answer'] = not any(reasoning_leaks(s) for _, s in segs)
    blob = json.dumps([{k: r.get(k) for k in ('feedback', 'trajectory_inputs', 'trajectory_rubrics', 'response_scores')} for r in recs])
    C['stripping: no think markup in rubric / feedback / scorer inputs'] = not leak(blob); notes['rubric pipeline artifacts present'] = any(r.get('feedback') for r in recs)
    C['parser: private option recorded on every action'] = all('private_option' in n['trace'] and 'private_outcome' in n['trace'] for n in nodes)
    notes['private option parsed'] = f"{sum(n['trace']['private_option'] is not None for n in nodes)} / {len(nodes)}"
    notes['Doctor FINAL parsed'] = f"{sum(l['final_option'] not in (None, '') for r in recs for l in r['tree']['leaves'])} / {sum(len(r['tree']['leaves']) for r in recs)}"
    ok_reward = True; used_tokens = 0; used = 0
    for r in recs:
        t = r['agent_targets']; nd = {n['node_id']: n for n in r['tree']['nodes']}
        for nid, v in t['nodes'].items():
            exp = float(nd[nid]['trace']['private_outcome'] or 0.0) if credit == 'decision_only' else compose_reward(v['outcome_component'], v['R_A'], **w)
            ok_reward &= (v['reward'] is None and exp is None) or (v['reward'] is not None and exp is not None and abs(v['reward'] - exp) < EPS)
        ok_reward &= (t.get('loss_segments') == [0] and t.get('credit') == 'decision_only') if credit == 'decision_only' else ('loss_segments' not in t)
        for ids in t['groups'].values():
            rew = [t['nodes'][i]['reward'] for i in ids if t['nodes'][i]['reward'] is not None]
            if len(rew) >= 2 and max(rew) - min(rew) > EPS:
                used += 1; used_tokens += sum(len(s['completion_ids']) for i in ids if t['nodes'][i]['reward'] is not None for k, s in enumerate(nd[i]['trace']['segments']) if credit != 'decision_only' or k == 0)
    C[f"reward path: {'R_D = private option correct, private segment only' if credit == 'decision_only' else 'configured Outcome/R_A mixture on both segments'}"] = ok_reward
    if credit == 'decision_only': C['E2: rubric pipeline did not run'] = not any(r.get('feedback') or r.get('response_scores') for r in recs)
    bu = recs[-1].get('batch_update') or {}; notes['batch_update'] = {k: bu.get(k) for k in ('optimizer_step', 'skip_reason', 'grad_norm', 'kl_loss', 'used_agent_completions', 'groups_used', 'groups_skipped')}
    notes['groups with nonzero advantage (expected used)'] = used
    if bu.get('optimizer_step'):
        C['on-policy ratio: exactly 1 at the update'] = bu['ratio']['max_abs_deviation'] <= 1e-3
        C['gradient: finite norm'] = math.isfinite(bu['grad_norm']) and bu['grad_norm'] > 0
        C['token mask: optimized token count = tokens of the intended segments of the used groups'] = bu['ratio']['count'] == used_tokens; notes['optimized tokens'] = [bu['ratio']['count'], used_tokens]
    else:
        C['update skipped only because no group had a nonzero advantage'] = used == 0 and bu.get('skip_reason') is not None
    ck = run / 'checkpoints' / 'checkpoint_000005'
    C['checkpoint: written with adapter, trainer state and receipt'] = all((ck / p).exists() for p in ('adapter/adapter_model.safetensors', 'trainer_state.pt', 'COMPLETE.json'))
    if (ck / 'COMPLETE.json').exists():
        from safetensors.torch import load_file
        sd = load_file(str(ck / 'adapter' / 'adapter_model.safetensors')); rec = json.loads((ck / 'COMPLETE.json').read_text())
        C['checkpoint: reloadable and its receipt names the final policy'] = len(sd) > 0 and rec['policy_sha256'] == json.loads((run / 'run_state.json').read_text())['policy_sha256'] and rec['completed_cases'] == 5
        nB = sum(float((v.float() ** 2).sum()) for k, v in sd.items() if 'lora_B' in k) ** .5; notes['||B|| after the smoke'] = round(nB, 5)
        if bu.get('optimizer_step'): C['checkpoint: the adapter moved from theta0 (B != 0)'] = nB > 0
    if x.gpu_log and Path(x.gpu_log).exists():
        peak = max(int(v) for l in open(x.gpu_log) for v in l.split()[1:]); notes['peak GPU memory MiB (sampled)'] = peak; C['GPU memory: below 46 GiB'] = peak < 46000
    sup = (run / 'SUPERVISOR.log').read_text() if (run / 'SUPERVISOR.log').exists() else ''; C['supervisor: completed without restart'] = 'complete: cursor 5' in sup and sup.count(' launch ') == 1
    print(json.dumps({'run': str(run), 'credit': credit, 'PASS': all(C.values()), 'checks': C, 'notes': notes}, indent=1))


if __name__ == '__main__':
    main()
