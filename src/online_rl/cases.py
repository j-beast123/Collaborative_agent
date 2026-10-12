"""Case loading for online RL / evaluation; normalises the gold field (some splits use top-level correct_option)."""
from __future__ import annotations
import hashlib, json
from pathlib import Path


def normalize(case: dict) -> dict:
    c = dict(case)
    if not isinstance(c.get('ground_truth'), dict):
        g = c.get('correct_option')
        if g is None: raise ValueError(f"case {c.get('case_id')} has no gold option")
        c['ground_truth'] = {'correct_option': str(g), 'correct_text': c['options'].get(str(g))}
    c.setdefault('scenario', '')
    return c


def load_cases(path: str | Path) -> list[dict]:
    return [normalize(json.loads(l)) for l in Path(path).read_text().splitlines() if l.strip()]


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def history_key(case_id: str, prefix: list[dict]) -> str:
    """Stable key of a pre-action history (cache key for rubric generation / R_E)."""
    payload = json.dumps([case_id, [(d.get('speaker'), d.get('text')) for d in prefix]], ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()
