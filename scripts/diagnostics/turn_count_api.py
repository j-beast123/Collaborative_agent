"""Small API-only turn-count diagnostic using the current rollout and prompt code.

This measures API base models, not a trained local adapter or a full G=4 tree.
No gold, outcomes or archived conversations are sent to either provider.
"""
from __future__ import annotations
import argparse
from collections import Counter
from dataclasses import asdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.belief import build_initial_agent_response_prompt, build_plain_agent_response_prompt
from src.collaboration.generator import dialogue_text, final_answer_option, final_decision_prompt, stop_reason
from src.online_rl.agent_protocol import ASSESS_MAX_TOKENS, ASSESS_SYSTEM, HIDDEN_BLOCK, blind_case_prompt, stage_seeds
from src.online_rl.cases import load_cases
from src.online_rl.doctor import halt_reason, split_status, turn_prompt
from src.online_rl.integrated.records import atomic_write_json, sha256_json
from src.online_rl.tree.cache import CallCache
from src.online_rl.tree.rollout import build_tree


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', default='/opt/anonymous/sherpa/data/interim/medmcqa/train_random500_seed20261006.jsonl')
    p.add_argument('--config', default='configs/online_rl/tree_v21_witness.json')
    p.add_argument('--out', default='outputs/diagnostics/api_turn_count_10')
    p.add_argument('--cases', type=int, default=10)
    p.add_argument('--seed', type=int, default=20261008)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--agent-model', default='qwen/qwen3-8b')
    p.add_argument('--launch', action='store_true')
    args = p.parse_args()
    cfg = json.loads((ROOT / args.config).read_text())
    ro = cfg['rollout']
    doctor_system = (ROOT / ro['doctor_system_prompt_file']).read_text()
    agent_system = (ROOT / ro.get('agent_system_prompt_file', 'prompts/agent.txt')).read_text()
    cases = random.Random(args.seed).sample(load_cases(args.data), args.cases)
    # Explicit allowlist: answers and other dataset metadata never reach API code.
    cases = [{k: c[k] for k in ('case_id', 'scenario', 'question', 'options')} for c in cases]
    ident = dict(doctor_model='google/gemma-3-12b-it', agent_model=args.agent_model,
                 endpoint='https://openrouter.ai/api/v1/chat/completions',
                 config=args.config, config_sha256=sha256_json(cfg), seed=args.seed,
                 max_agent_turns=ro['rounds'], min_agent_turns=ro['halting']['min_agent_turns'],
                 halting=ro['halting'], agent_protocol=ro['agent_protocol'], branching=1,
                 doctor_prompt_sha256=hashlib.sha256(doctor_system.encode()).hexdigest(),
                 agent_prompt_sha256=hashlib.sha256(agent_system.encode()).hexdigest(),
                 case_ids=[c['case_id'] for c in cases],
                 data_sha256=hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
                 deviations=['API base Agent without LoRA; API Doctor instead of pinned local Q8_0.',
                             'One sampled path per case, not the full G=4 interaction tree.',
                             'Fresh Doctor opening; pinned shared opening table is unavailable.',
                             'Provider decoding/tokenization can differ from local model.'],
                 outbound_fields=['question', 'scenario', 'options', 'generated dialogue', 'role prompts'])
    print(json.dumps(ident, indent=2), flush=True)
    if not args.launch:
        return
    key = os.environ.get('OPENROUTER_API_KEY')
    if not key:
        for line in Path('/opt/anonymous/sherpa/.env').read_text().splitlines():
            if line.startswith('OPENROUTER_API_KEY='):
                key = line.split('=', 1)[1].strip().strip('\"\'')
    if not key:
        raise RuntimeError('OPENROUTER_API_KEY unavailable')
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'manifest.json').exists() and json.loads((out / 'manifest.json').read_text()) != ident:
        raise RuntimeError('Output belongs to a different experiment')
    atomic_write_json(out / 'manifest.json', ident)
    cache = CallCache(out / 'calls.jsonl')

    def chat(model, system, user, seed, max_tokens, temperature, top_p, role):
        messages = ([{'role': 'user', 'content': system + '\n\n' + user}]
                    if role == 'doctor' else [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}])
        payload = dict(model=model, messages=messages, max_tokens=max_tokens, temperature=temperature,
                       top_p=top_p, seed=seed, provider={'require_parameters': True})
        if role == 'agent' and '/qwen3-' in model:
            payload['reasoning'] = {'enabled': False}
        def invoke():
            for attempt in range(4):
                try:
                    req = urllib.request.Request(ident['endpoint'], data=json.dumps(payload).encode(),
                          headers={'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'})
                    with urllib.request.urlopen(req, timeout=60) as f:
                        body = json.load(f)
                    if body.get('error'):
                        raise RuntimeError(str(body['error']))
                    choice = body['choices'][0]
                    text = choice['message'].get('content')
                    if not text or not text.strip():
                        raise RuntimeError(f'{model}: empty reply')
                    if choice.get('finish_reason') == 'length':
                        raise RuntimeError(f'{model}: truncated reply; do not count as a completed sample')
                    return dict(text=text.strip(), usage=body.get('usage'), provider=body.get('provider'),
                                actual_model=body.get('model'), finish_reason=choice.get('finish_reason'))
                except urllib.error.HTTPError as e:
                    if e.code not in (429, 500, 502, 503, 504) or attempt == 3:
                        raise RuntimeError(f'{model}: HTTP {e.code}: {e.read().decode()[:300]}') from e
                    print(f'{model}: HTTP {e.code}; retry {attempt + 1}/3', flush=True)
                except (urllib.error.URLError, TimeoutError):
                    if attempt == 3:
                        raise
                time.sleep(min(15 * (attempt + 1), 45))
        return cache.call('api_chat', {'experiment': sha256_json(ident), 'payload': payload}, invoke)['text']

    class Doctor:
        status_protocol = True
        min_agent_turns = ro['halting']['min_agent_turns']
        def turn(self, case, dialogue, round_, rounds, seed):
            raw = chat(ident['doctor_model'], doctor_system, turn_prompt(case, dialogue, round_, rounds, True),
                       seed, ro['max_tokens_doctor'], ro['doctor_backend']['options']['temperature'], 1.0, 'doctor')
            visible, status, error = split_status(raw, case['options'])
            if round_ == 1:
                status, error = None, None
            return dict(raw=raw, visible=visible, status=status, status_error=error, seed=seed)
        def final(self, case, dialogue, seed):
            return chat(ident['doctor_model'], doctor_system, final_decision_prompt(case, dialogue), seed,
                        ro['max_tokens_doctor'], ro['doctor_backend']['options']['temperature'], 1.0, 'doctor')
        def halt(self, **kw):
            return halt_reason(**kw, min_agent_turns=ro['halting']['min_agent_turns']) if ro['halting']['enabled'] else None

    class Agent:
        def sample(self, case, prefix, turn, seed):
            sa, sb = stage_seeds(seed, case['case_id'])
            assessment = chat(ident['agent_model'], ASSESS_SYSTEM, blind_case_prompt(case), sa,
                              ASSESS_MAX_TOKENS, ro['temperature'], ro['top_p'], 'agent')
            if turn == 1:
                user = build_initial_agent_response_prompt(scenario=case['scenario'], question=case['question'],
                         options=case['options'], clinician_message=prefix[-1]['text'])
            else:
                user = build_plain_agent_response_prompt(scenario=case['scenario'], question=case['question'],
                         options=case['options'], dialogue_history=dialogue_text(prefix), current_round=turn,
                         max_rounds=ro['rounds'], is_final_round=turn == ro['rounds'])
            user += HIDDEN_BLOCK.format(assessment=assessment)
            reply = chat(ident['agent_model'], agent_system, user, sb, ro['max_completion_length'],
                         ro['temperature'], ro['top_p'], 'agent')
            return dict(completion_text=reply, private_assessment=assessment)

    def one(case):
        dest = out / 'cases' / f"{case['case_id']}.json"
        if dest.exists():
            return json.loads(dest.read_text())
        try:
            tree = build_tree(case, max_agent_turns=ro['rounds'], branching=1, run_seed=args.seed,
                      policy_sha256=sha256_json(ident), protocol_sha256=sha256_json(ident),
                      agent=Agent(), doctor=Doctor(), stop_rule=stop_reason, final_option=final_answer_option,
                      outcome_fn=lambda c, o: None, cache=cache)
            leaf = tree.leaves[0]
            row = dict(case_id=case['case_id'], status='complete', agent_turns=len(leaf.path),
                       doctor_discussion_turns=sum(t[0] == 'doctor' and t[1] != 'FINAL' for t in leaf.dialogue),
                       stop_reason=leaf.stop_reason, final_option=leaf.final_option, tree=asdict(tree),
                       status_parse_errors=[n.next_doctor_status['status_error'] for n in tree.nodes
                           if n.next_doctor_status and n.next_doctor_status.get('status_error')])
            atomic_write_json(dest, row)
            print(json.dumps({k: row[k] for k in ('case_id', 'agent_turns', 'stop_reason')}) , flush=True)
            return row
        except Exception as e:
            print(json.dumps({'case_id': case['case_id'], 'status': 'error', 'error': str(e)}), flush=True)
            return dict(case_id=case['case_id'], status='error', error=str(e))

    with ThreadPoolExecutor(args.workers) as pool:
        results = list(pool.map(one, cases))
    good = [r for r in results if r['status'] == 'complete']
    calls = [r['value'] for r in cache.store.values() if r['kind'] == 'api_chat']
    summary = dict(requested=len(cases), completed=len(good), errors=[r for r in results if r['status'] == 'error'],
                   mean_agent_turns=sum(r['agent_turns'] for r in good)/len(good) if good else None,
                   turn_distribution=dict(Counter(r['agent_turns'] for r in good)),
                   stop_reasons=dict(Counter(r['stop_reason'] for r in good)),
                   status_parse_errors=sum(len(r['status_parse_errors']) for r in good),
                   mean_doctor_discussion_turns=sum(r['doctor_discussion_turns'] for r in good)/len(good) if good else None,
                   api_calls=len(calls), api_cost_usd=sum((r.get('usage') or {}).get('cost') or 0 for r in calls),
                   cases=[{k:v for k,v in r.items() if k != 'tree'} for r in results], limitations=ident['deviations'])
    atomic_write_json(out / 'summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
