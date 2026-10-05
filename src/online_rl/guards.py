"""Evaluation-only data (Prompt-Test, VAL50 / RQ3 probe pool, TEST100, CasiMedicos) must never reach training or prompt
evolution. The shipped manifest holds only ids and normalized-question sha256 (no question text); ids are matched where
reliable and question hashes catch the same item across id spaces."""
from __future__ import annotations
import argparse, hashlib, json, unicodedata
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / 'data/splits/protected_eval_v1.json'


def question_sha256(question: str) -> str:
    return hashlib.sha256(' '.join(unicodedata.normalize('NFKC', str(question)).split()).casefold().encode()).hexdigest()


def _manifest() -> dict:
    """Missing manifest raises: a guard must not silently stop guarding."""
    return json.loads(MANIFEST.read_text())['splits']


def protected_ids() -> dict[str, set]:
    return {name: set(s['ids']) for name, s in _manifest().items() if s['ids']}


def protected_question_hashes() -> dict[str, set]:
    return {name: set(s['question_sha256']) for name, s in _manifest().items() if s['question_sha256']}


def assert_not_protected(cases_or_ids, purpose: str) -> None:
    ids, qh = set(), set()
    for c in cases_or_ids:
        if isinstance(c, dict):
            ids |= {c.get('case_id'), str(c.get('source_question_id'))}
            if c.get('question'): qh.add(question_sha256(c['question']))
        else: ids.add(c)
    for name, prot in protected_ids().items():
        if ids & prot: raise RuntimeError(f'{purpose}: {len(ids & prot)} ids belong to {name}, which is evaluation-only')
    for name, prot in protected_question_hashes().items():
        if qh & prot: raise RuntimeError(f'{purpose}: {len(qh & prot)} questions belong to {name}, which is evaluation-only')


def build_manifest(medmcqa_splits: dict[str, Path], casimedicos_dir: Path, probe_pool: Path) -> dict:
    """Rebuild the manifest from local evaluation files (MedMCQA-style JSONL, CasiMedicos en_*_casimedicos.jsonl)."""
    rows = lambda p: [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]
    splits = {}
    for name, path in medmcqa_splits.items():
        r = rows(path)
        splits[name] = {'ids': sorted({x['case_id'] for x in r} | {str(x['source_question_id']) for x in r if x.get('source_question_id')}),
                        'question_sha256': sorted({question_sha256(x['question']) for x in r})}
    for p in sorted(Path(casimedicos_dir).glob('en_*_casimedicos.jsonl')):
        splits[f'casimedicos_{p.stem.split("_")[1]}'] = {'ids': [], 'question_sha256': sorted({question_sha256(x['full_question']) for x in rows(p)})}
    pool = json.loads(Path(probe_pool).read_text())
    splits['rq3_probe_pool'] = {'ids': sorted(set(pool['case_ids']) | set(pool['source_question_ids'])), 'question_sha256': []}
    return {'version': 'protected_eval_v1', 'normalization': 'NFKC, whitespace-collapsed, casefolded question text', 'splits': splits}


if __name__ == '__main__':
    a = argparse.ArgumentParser(description='Rebuild data/splits/protected_eval_v1.json from local evaluation files.')
    a.add_argument('--val50', required=True); a.add_argument('--test100', required=True); a.add_argument('--prompt-test', required=True)
    a.add_argument('--casimedicos-dir', required=True); a.add_argument('--probe-pool', required=True); x = a.parse_args()
    m = build_manifest({'val50': x.val50, 'test100': x.test100, 'prompt_test': x.prompt_test}, x.casimedicos_dir, x.probe_pool)
    MANIFEST.write_text(json.dumps(m, indent=1) + '\n'); print(f'wrote {MANIFEST}')
