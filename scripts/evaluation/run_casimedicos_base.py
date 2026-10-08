"""One unassisted five-round Qwen/Gemma trajectory per case, sharded by GPU."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.belief.local_judgment import build_initial_agent_response_prompt, build_plain_agent_response_prompt
from src.collaboration.generator import dialogue_text
from src.online_rl.cases import load_cases
from src.online_rl.environment import discussion_prompt
from src.online_rl.integrated.records import atomic_write_json, sha256_json
from src.online_rl.tree.backends import ApiDoctor, default_final_judge
from src.online_rl.tree.ollama_doctor import OllamaDoctorCall
from src.online_rl.tree.ollama_policy import OllamaPolicy
from src.online_rl.tree.posthoc import case_view


def shard(cases, rank, world_size):
    return [(index, case) for index, case in enumerate(cases) if index % world_size == rank]


def agent_prompt(view, dialogue, turn, rounds):
    common = {'scenario': view.get('scenario', ''), 'question': view['question'], 'options': view['options']}
    if turn == 1:
        return build_initial_agent_response_prompt(**common, clinician_message=dialogue[-1]['text'])
    return build_plain_agent_response_prompt(**common, dialogue_history=dialogue_text(dialogue),
        current_round=turn, max_rounds=rounds, is_final_round=turn == rounds)


def run_case(case, doctor, agent, rounds, seed, checkpoint):
    view = case_view(case)
    dialogue = json.loads(checkpoint.read_text())['dialogue'] if checkpoint.exists() else []
    for turn in range(1, rounds + 1):
        for speaker in ('doctor', 'agent'):
            stage = ('D' if speaker == 'doctor' else 'A') + str(turn)
            if any(d['stage'] == stage for d in dialogue):
                continue
            started = time.monotonic()
            if speaker == 'doctor':
                text = doctor.turn(view, dialogue, turn, rounds)
            else:
                turn_seed = int(hashlib.sha256(f'{seed}:{case["case_id"]}:{turn}'.encode()).hexdigest()[:8], 16)
                text = agent.agent(agent_prompt(view, dialogue, turn, rounds), turn_seed)
            dialogue.append({'speaker': speaker, 'stage': stage, 'round': turn, 'text': text,
                             'duration_s': time.monotonic() - started})
            atomic_write_json(checkpoint, {'case_id': case['case_id'], 'dialogue': dialogue})
    if not any(d['stage'] == 'FINAL' for d in dialogue):
        started = time.monotonic()
        text = doctor.final(view, dialogue)
        dialogue.append({'speaker': 'doctor', 'stage': 'FINAL', 'text': text, 'duration_s': time.monotonic() - started})
        atomic_write_json(checkpoint, {'case_id': case['case_id'], 'dialogue': dialogue})
    judged = default_final_judge(case, dialogue[-1]['text'])
    return {'case_id': case['case_id'], 'source_split': case['metadata']['source_split'],
            'case': case, 'dialogue': dialogue, 'agent_turns': rounds,
            'status': 'complete' if judged['outcome'] in (0.0, 1.0) else 'unparseable_final',
            **judged, 'termination_reason': 'fixed_five_rounds', 'seed': seed,
            'protocol': 'plain_collaboration_no_belief_no_rubric_no_feedback_v1'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    cfg = json.loads((ROOT / args.config).read_text())
    out = ROOT / cfg['output_dir']; out.mkdir(parents=True, exist_ok=True)
    os.environ['API_USAGE_LOG'] = str(out / 'api_usage.jsonl')  # local timing log only; no API calls
    cases = load_cases(ROOT / cfg['data']['all'])
    assert hashlib.sha256((ROOT / cfg['data']['all']).read_bytes()).hexdigest() == cfg['data']['sha256']
    assigned = shard(cases, cfg['rank'], cfg['world_size'])
    identity = {'config': cfg, 'case_ids': [c['case_id'] for _, c in assigned],
                'prompt_sha256': {name: hashlib.sha256((ROOT / 'prompts' / name).read_bytes()).hexdigest()
                                  for name in ('agent.txt', 'user_simulator.txt')}}
    manifest = out / 'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text()) != identity:
        raise RuntimeError('Baseline config, cases or prompts changed; use a new output directory')
    atomic_write_json(manifest, identity)
    policy = OllamaPolicy(cfg, '')
    call = OllamaDoctorCall(cfg['doctor_backend'])
    doctor = ApiDoctor('', cfg['rollout']['doctor_model'], policy.doctor_system,
                       cfg['rollout']['max_tokens_doctor'], call=call)
    atomic_write_json(out / 'models.json', {'agent': policy.identity(), 'doctor': cfg['doctor_backend']})
    for index, case in assigned:
        result_path = out / 'cases' / f'case_{index:06d}.json'
        checkpoint = out / 'checkpoints' / f'case_{index:06d}.json'
        if result_path.exists():
            continue
        for attempt in range(3):
            try:
                result = run_case(case, doctor, policy, cfg['rollout']['rounds'], cfg['seed'], checkpoint)
                result['source_index'] = index
                atomic_write_json(result_path, result)
                print(json.dumps({'index': index, 'case_id': case['case_id'], 'status': result['status'],
                                  'outcome': result['outcome']}, ensure_ascii=False), flush=True)
                break
            except Exception as error:
                print(json.dumps({'index': index, 'attempt': attempt + 1, 'technical_error':
                                  f'{type(error).__name__}: {error}'}), flush=True)
                if attempt < 2:
                    time.sleep(1)
                else:
                    atomic_write_json(out / 'errors' / f'case_{index:06d}.json',
                                      {'case_id': case['case_id'], 'error': str(error), 'status': 'technical_error'})
    print('Shard finished', flush=True)


if __name__ == '__main__':
    main()
