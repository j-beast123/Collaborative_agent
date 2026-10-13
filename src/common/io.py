"""JSONL reading and writing shared by the CLIs and library modules."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


def read_jsonl(path, *, missing_ok: bool = False) -> list[dict]:
    path = Path(path)
    if missing_ok and not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path, rows: Iterable[dict]) -> None:
    Path(path).write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
