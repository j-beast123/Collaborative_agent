"""AUDIT ONLY: residual answer-specific content in VALID v4 feedback. Never used by training: it does not change feedback
validity, rubric mapping, witness verification, KEEP, reward, caches or any protocol hash (the production guard is the exact
lexical posthoc.leakage_problems). For each valid feedback and each answer option it reports
  exact      : the option text as a whole phrase (the production guard's own rule; should be empty for valid feedback)
  normalized : the option phrase after case / whitespace / hyphen / simple-plural normalisation
  paraphrase : candidates for a human: content words of the option (>= 4 letters, simple stem) present in the feedback
and, for every hit, the transcript evidence: the pre-FINAL dialogue turns (speaker, label) in which the same normalised words
occur, so a reader can tell a restatement of the visible discussion from information that could only come from the hidden
gold / outcome (which the v4 writer never receives). The gold option is shown for the reader only.
Usage: feedback_residual_audit.py <arm_dir> --pool POOL.jsonl --out PREFIX"""
import argparse, json, re, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))

STOP = {'with', 'from', 'that', 'this', 'than', 'into', 'over', 'under', 'test', 'tests', 'type', 'both', 'none', 'above', 'below', 'only',
        'other', 'which', 'their', 'there', 'joint', 'level', 'cell', 'cells', 'acute', 'chronic', 'syndrome', 'disease', 'deficiency'}


def norm(t: str) -> str:
    t = t.lower().replace('-', ' ').replace('/', ' ')
    t = re.sub(r'[^a-z0-9 ]+', ' ', t)
    return ' '.join(w[:-2] if w.endswith('es') and len(w) > 4 else w[:-1] if w.endswith('s') and len(w) > 3 else w for w in t.split())


def stem(w: str) -> str:
    for suf in ('ing', 'ed', 'es', 's', 'al', 'ic', 'y'):
        if w.endswith(suf) and len(w) - len(suf) >= 4: return w[:-len(suf)]
    return w


def content_words(t: str) -> list[str]:
    return [w for w in norm(t).split() if len(w) >= 4 and w not in STOP]


def evidence(words: list[str], turns: list[tuple[str, str]]) -> list[str]:
    out = []
    for label, text in turns:
        st = {stem(w) for w in norm(text).split()}
        hit = [w for w in words if stem(w) in st]
        if hit: out.append(f'{label}: {", ".join(hit)}')
    return out


def main():
    a = argparse.ArgumentParser(); a.add_argument('arm'); a.add_argument('--pool', required=True); a.add_argument('--out', required=True); x = a.parse_args()
    from src.online_rl.cases import load_cases
    pool = {c['case_id']: c for c in load_cases(x.pool)}; rows = []; n_valid = 0
    for f in sorted((Path(x.arm) / 'cases').glob('case_*.json')):
        rec = json.loads(f.read_text()); case = pool[rec['case_id']]
        gold = (case.get('ground_truth') or {}).get('correct_option') or case.get('correct_option')
        leaves = {l['leaf_id']: l for l in rec['tree']['leaves']}
        for fb in rec['feedback']:
            if not fb.get('valid', True): continue
            n_valid += 1; text = fb['text']; nt = norm(text); st = {stem(w) for w in nt.split()}
            k = 0; turns = []
            for sp, stage, _, t in leaves[fb['leaf_id']]['dialogue']:
                if stage == 'FINAL': continue
                if sp == 'agent': k += 1; turns.append((f'Agent reply {k}', t))
                else: turns.append((f'Physician {stage}', t))
            for opt, otext in sorted(case['options'].items()):
                no = norm(otext); cw = content_words(otext)
                exact = bool(re.search(r'(?<![\w])' + re.escape(otext.lower().strip(' .')) + r'(?![\w])', text.lower())) and len(otext.strip()) >= 4
                normalized = bool(no) and len(no) >= 4 and re.search(r'(?<![a-z0-9])' + re.escape(no) + r'(?![a-z0-9])', nt) is not None
                para = [w for w in cw if stem(w) in st]
                if not (exact or normalized or para): continue
                ev = evidence(para or cw, turns)
                rows.append({'case_id': rec['case_id'], 'leaf_id': fb['leaf_id'], 'outcome': fb['outcome'], 'option': opt, 'option_text': otext,
                             'is_gold': opt == gold, 'exact': exact, 'normalized': normalized, 'paraphrase_words': para,
                             'transcript_evidence': ev, 'source': 'transcript restatement' if ev else 'NOT IN TRANSCRIPT (review)',
                             'feedback_excerpt': text[:400]})
    Path(x.out + '.json').write_text(json.dumps({'valid_feedback': n_valid, 'rows': rows}, indent=1, ensure_ascii=False))
    L = ['# Audit-only residual answer-specific content in valid v4 feedback', '',
         'Audit only: no effect on validity, KEEP, reward, caches or protocol. Production guard = exact lexical posthoc.leakage_problems.', '',
         f'valid feedback: {n_valid}; feedback with any hit: {len({r["leaf_id"] for r in rows})}; '
         f'exact: {sum(r["exact"] for r in rows)}; normalized: {sum(r["normalized"] for r in rows)}; paraphrase-only: '
         f'{sum(not r["exact"] and not r["normalized"] for r in rows)}; not in transcript: {sum(r["source"].startswith("NOT") for r in rows)}', '',
         '| case | leaf | outcome | option | gold? | exact | normalized | paraphrase words | transcript evidence | source |', '|---|---|---|---|---|---|---|---|---|---|']
    for r in rows:
        L.append(f"| {r['case_id'][-8:]} | {r['leaf_id'].split('/')[-1]} | {'S' if r['outcome'] == 1.0 else 'F'} | {r['option']}: {r['option_text']} | "
                 f"{'yes' if r['is_gold'] else ''} | {r['exact']} | {r['normalized']} | {', '.join(r['paraphrase_words'])} | "
                 f"{'; '.join(r['transcript_evidence'][:3])}{' …' if len(r['transcript_evidence']) > 3 else ''} | {r['source']} |")
    Path(x.out + '.md').write_text('\n'.join(L) + '\n'); print('\n'.join(L[:6]))


if __name__ == '__main__': main()
