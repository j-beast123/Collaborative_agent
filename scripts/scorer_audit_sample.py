"""Draw a stratified sample of scored responses for an independent audit of the rubric scorer."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from src.common.io import read_jsonl

ROOT = Path(__file__).resolve().parents[1]


def units(rows: list[dict]) -> list[dict]:
    """One unit per (response, rubric) from a reward log (rubric_scores per stage) or from gap_scores.jsonl."""
    found = []
    for row in rows:
        scored = row.get("rubric_scores") or ({"-": {"status": row.get("status"), "labels": row.get("labels"), "input": row.get("input"),
                                                      "result": {"probabilities": row.get("probabilities"), "scorer": (row.get("scorer") or {}).get("name")}}}
                                              if "labels" in row else {})
        for stage, score in scored.items():
            if score.get("status") != "scored" or not score.get("input"):
                continue
            rubrics = {r["rubric_id"]: r for r in score["input"]["rubrics"]}
            result = score.get("result") or {}
            for rubric_id, label in score["labels"].items():
                found.append({"unit_id": f"{row.get('sample_id') or row.get('item_id')}:{row.get('generation', '')}:{stage}:{rubric_id}",
                              "case_id": row.get("case_id"), "stage": row.get("stage", stage), "rubric_id": rubric_id,
                              "scorer": result.get("scorer"), "scorer_label": label,
                              "scorer_probabilities": (result.get("probabilities") or {}).get(rubric_id),
                              "input": {k: v for k, v in score["input"].items() if k != "rubrics"}, "rubric": rubrics[rubric_id],
                              "auditor_label": None, "auditor": None, "auditor_note": None})
    return found


def sample(found: list[dict], per_stratum: int, seed: int) -> list[dict]:
    strata = defaultdict(list)
    for unit in sorted(found, key=lambda u: u["unit_id"]):
        strata[(unit["rubric_id"], str(unit["scorer_label"]))].append(unit)
    rng = random.Random(seed)
    return [unit for key in sorted(strata) for unit in sorted(rng.sample(strata[key], min(per_stratum, len(strata[key]))),
                                                                 key=lambda u: u["unit_id"])]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log", required=True)
    parser.add_argument("--per-stratum", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    out = ROOT / args.output
    if out.exists():
        raise SystemExit(f"{out} exists: an audit sample is drawn once")
    found = units(read_jsonl(ROOT / args.log))
    chosen = sample(found, args.per_stratum, args.seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(u, ensure_ascii=False) + "\n" for u in chosen), encoding="utf-8")
    print(json.dumps({"units": len(found), "sampled": len(chosen), "strata": len({(u["rubric_id"], str(u["scorer_label"])) for u in found}),
                      "output": args.output}, indent=2))


if __name__ == "__main__":
    main()
