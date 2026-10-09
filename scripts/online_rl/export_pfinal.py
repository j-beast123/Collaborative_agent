"""Export Pfinal: the last Rubric-Generator prompt adopted by a completed five-GPU evolving run, with provenance.
Writes prompts/sherpa/pfinal/<run_id>/{prompt.txt,pfinal.json}; set matrix.json pfinal_artifact to the json path."""
import argparse
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.online_rl.integrated.prompt_search import PromptStore
from src.online_rl.integrated.records import sha256_json
from src.online_rl.tree import experiment as X

PFINAL_DIR = 'prompts/sherpa/pfinal'


def adoption_history(run: Path, p0: str, final: str) -> list[dict]:
    """Chain of batch prompt events; every parent must be the previous winner."""
    events = [json.loads(p.read_text()) for p in sorted((run / 'prompt_events').glob('batch_*.json'))]
    if not events: raise ValueError(f'{run}: no batch prompt events')
    current, out = p0, []
    for e in events:
        if e['parent_prompt']['sha256'] != current: raise ValueError(f"{run}: batch {e['batch']} parent breaks the prompt chain")
        d = e['decision']
        out.append({'batch': e['batch'], 'parent_sha256': current, 'candidate_sha256': e['candidate_prompt']['sha256'],
                    'accepted': d['accepted'], 'reason': d['reason'], 'parent_J': d['parent'].get('J'),
                    'candidate_J': d['candidate'].get('J'), 'next_sha256': e['next_prompt']['sha256']})
        current = e['next_prompt']['sha256']
    if current != final: raise ValueError(f'{run}: last event winner differs from run_state prompt')
    return out


def rel(p: Path) -> str:
    return str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)


def export(run: Path, dest_root: Path = ROOT / PFINAL_DIR) -> Path:
    run = (ROOT / run).resolve()
    manifest = json.loads((run / 'run_manifest.json').read_text()); state = json.loads((run / 'run_state.json').read_text())
    cfg = manifest['config']
    if not cfg['prompt_update']['enabled']: raise ValueError(f'{run}: not an evolving run')
    mode = (cfg.get('experiment') or {}).get('prompt_mode', 'evolving')
    if state['case_cursor'] != cfg['schedule']['cases']: raise ValueError(f"{run}: incomplete ({state['case_cursor']}/{cfg['schedule']['cases']} cases)")
    history = adoption_history(run, cfg['rubric']['prompt_sha256'], state['prompt_sha256'])
    adopted = [h for h in history if h['accepted']]
    if not adopted: raise ValueError(f'{run}: no prompt was ever adopted; Pfinal would equal P0')
    last = adopted[-1]; text = PromptStore(run / 'prompts').get(last['candidate_sha256'])
    rid = (cfg.get('experiment') or {}).get('run_id') or run.name
    dest = Path(dest_root) / rid; dest.mkdir(parents=True, exist_ok=True)
    prompt_path = dest / 'prompt.txt'
    if prompt_path.exists() and prompt_path.read_text() != text: raise ValueError(f'{prompt_path} exists with different text')
    prompt_path.write_text(text)
    meta = {'schema': X.PFINAL_SCHEMA, 'source_run_id': rid, 'source_output_dir': rel(run),
            'source_checkpoint': state['checkpoint'], 'source_batch': last['batch'], 'source_completed_cases': state['case_cursor'],
            'source_final_policy_sha256': state['policy_sha256'], 'prompt_sha256': last['candidate_sha256'],
            'prompt_text_path': rel(prompt_path), 'parent_prompt_sha256': last['parent_sha256'],
            'p0_sha256': cfg['rubric']['prompt_sha256'], 'adoption_history': history,
            'source_config_sha256': sha256_json(cfg), 'source_run_config_sha256': state['config_sha256'],
            'source_protocol_sha256': state['protocol_sha256'], 'source_prompt_mode': mode, **X.source_identity(cfg),
            'source_theta0_policy_sha256': json.loads((run / 'checkpoints/checkpoint_000000/COMPLETE.json').read_text())['policy_sha256']}
    path = dest / 'pfinal.json'
    path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + '\n')
    X.load_pfinal(rel(path))
    return path


def main(argv=None):
    a = argparse.ArgumentParser(); a.add_argument('--run', required=True, help='completed evolving run output directory')
    print(rel(export(Path(a.parse_args(argv).run))))


if __name__ == '__main__': main()
