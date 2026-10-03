"""Candidate automatic equivalence judges on the pairs of the human adjudication sheet, and their agreement with the human."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from src.judges.llm import CachedChat, ChatRequest, build_chat_backend, parse_json_object

ROOT = Path(__file__).resolve().parents[2]
OPENROUTER = {"backend": "openai_compatible", "base_url": "https://openrouter.ai/api/v1", "api_key_env": "OPENROUTER_API_KEY"}
CANDIDATES = [
    {"name": "gpt-4o-mini_v4", "model_id": "openai/gpt-4o-mini", "prompt": "prompts/agentclinic/diagnosis_equivalence_v4.txt", "version": "diagnosis-equivalence-v4"},
    {"name": "gpt-6.1-sol_v4", "model_id": "openai/gpt-6.1-sol", "prompt": "prompts/agentclinic/diagnosis_equivalence_v4.txt", "version": "diagnosis-equivalence-v4"},
    {"name": "gpt-4o-mini_v3", "model_id": "openai/gpt-4o-mini", "prompt": "prompts/agentclinic/diagnosis_equivalence.txt", "version": "diagnosis-equivalence-v3"},
]
PARAMS = {"temperature": 0, "max_tokens": 200}
HELDOUT, SPLIT_SEED = 131, "0:equivalence_judge_split"


def pair_id(row: dict) -> str:
    return f"{row['correct_diagnosis']} <> {row['predicted_diagnosis']}"


def split(sheet: Path, out: Path) -> dict:
    target = out / "judge_split_manifest.json"
    if target.exists():
        raise SystemExit(f"{target} exists: the split is made once")
    rows = rows_of(sheet)
    if any(row["human_decision(equivalent y/n)"].strip() for row in rows):
        raise SystemExit("the sheet already has human decisions: the split must be fixed before they are entered")
    strata: dict[tuple, list[str]] = {}
    for row in rows:
        strata.setdefault((row["reason_listed"], row["automatic_decision"]), []).append(pair_id(row))
    exact = {key: HELDOUT * len(members) / len(rows) for key, members in strata.items()}
    quota = {key: int(value) for key, value in exact.items()}
    for key in sorted(strata, key=lambda k: (-(exact[k] - quota[k]), k))[: HELDOUT - sum(quota.values())]:
        quota[key] += 1
    heldout = []
    for key, members in sorted(strata.items()):
        ordered = sorted(members, key=lambda pair: hashlib.sha256(f"{SPLIT_SEED}:{pair}".encode()).hexdigest())
        heldout += ordered[: quota[key]]
    selection = sorted(set(pair_id(r) for r in rows) - set(heldout))
    manifest = {"sheet": str(sheet.relative_to(ROOT)), "sheet_sha256_at_split": hashlib.sha256(sheet.read_bytes()).hexdigest(), "pairs": len(rows),
                "made_before_any_human_decision": True, "seed": SPLIT_SEED,
                "rule": "stratified by (reason a pair was listed, automatic decision); proportional quotas by largest remainder; "
                        "inside a stratum the order of sha256('<seed>:<pair>')",
                "strata": {f"{reason} / automatic {decision}": {"pairs": len(strata[(reason, decision)]), "heldout": quota[(reason, decision)]}
                           for reason, decision in sorted(strata)},
                "use": {"selection": "the judge (model and prompt) is chosen by agreement with the human on these pairs only",
                        "heldout": "evaluated once for the chosen judge after the choice; the judge is not changed afterwards"},
                "population_note": "the pairs are hard cases selected on purpose (conflicting automatic decisions, lowered-certainty and "
                                   "non-committal wordings); agreement on them is a validation on difficult equivalence decisions, not the accuracy on all pairs",
                "selection": selection, "heldout": sorted(heldout), "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    manifest["manifest_sha256"] = hashlib.sha256(json.dumps(manifest, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    out.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    target.chmod(0o444)
    return {"selection": len(selection), "heldout": len(heldout), "strata": manifest["strata"], "manifest_sha256": manifest["manifest_sha256"]}


def load_split(out: Path) -> dict:
    manifest = json.loads((out / "judge_split_manifest.json").read_text(encoding="utf-8"))
    body = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    if hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False).encode()).hexdigest() != manifest["manifest_sha256"]:
        raise SystemExit("the split manifest was edited after it was written")
    return manifest


def human_decisions(sheet: Path) -> tuple[dict, dict]:
    human, reason = {}, {}
    for row in rows_of(sheet):
        value = row["human_decision(equivalent y/n)"].strip().lower()
        if value not in ("y", "n"):
            raise SystemExit(f"no human decision for: {pair_id(row)}")
        human[pair_id(row)] = value == "y"
        reason[pair_id(row)] = row["reason_listed"]
    return human, reason


def agreement(candidate: dict, out: Path, pairs: set, human: dict, reason: dict) -> dict:
    decisions = [d for d in json.loads((out / f"decisions_{candidate['name']}.json").read_text(encoding="utf-8")) if pair_id(d) in pairs]
    agree, missed, extra, by_reason = 0, 0, 0, Counter()
    for d in decisions:
        key = pair_id(d)
        ok = d["equivalent"] == human[key]
        agree += ok
        missed += human[key] and d["equivalent"] is not True
        extra += (not human[key]) and d["equivalent"] is True
        by_reason[(reason[key], ok)] += 1
    return {"model": candidate["model_id"], "prompt": candidate["prompt"], "params": PARAMS, "pairs": len(decisions), "agreement": agree / len(decisions),
            "human_equivalent_judge_not": missed, "human_not_judge_equivalent": extra,
            "agreement_by_reason": {r: by_reason[(r, True)] / (by_reason[(r, True)] + by_reason[(r, False)]) for r in sorted({k[0] for k in by_reason})}}


def rows_of(sheet: Path) -> list[dict]:
    with sheet.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def decide(sheet: Path, out: Path) -> dict:
    rows, result = rows_of(sheet), {}
    for candidate in CANDIDATES:
        chat = CachedChat(build_chat_backend(dict(OPENROUTER, model_id=candidate["model_id"], revision="openrouter-unpinned")),
                          out / f"cache_{candidate['name']}.jsonl")
        prompt = (ROOT / candidate["prompt"]).read_text(encoding="utf-8")

        def one(row: dict) -> dict:
            user = json.dumps({"reference_diagnosis": row["correct_diagnosis"], "predicted_diagnosis": row["predicted_diagnosis"]}, ensure_ascii=False)
            answer = chat.complete(ChatRequest(prompt, user, candidate["version"], PARAMS))
            parsed, _ = parse_json_object(answer.text) if answer.status == "ok" else (None, None)
            return {"correct_diagnosis": row["correct_diagnosis"], "predicted_diagnosis": row["predicted_diagnosis"],
                    "equivalent": (parsed or {}).get("equivalent"), "reason": (parsed or {}).get("reason")}

        with ThreadPoolExecutor(8) as pool:
            decisions = list(pool.map(one, rows))
        (out / f"decisions_{candidate['name']}.json").write_text(json.dumps(decisions, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        result[candidate["name"]] = {"pairs": len(decisions), "equivalent": sum(d["equivalent"] is True for d in decisions),
                                     "undecided": sum(not isinstance(d["equivalent"], bool) for d in decisions)}
    return result


def compare(sheet: Path, out: Path) -> dict:
    """Agreement of every candidate with the human on the judge-selection set only."""
    human, reason = human_decisions(sheet)
    manifest = load_split(out)
    pairs = set(manifest["selection"])
    result = {"set": "judge-selection", "pairs": len(pairs), "human_equivalent": sum(human[p] for p in pairs), "split_manifest_sha256": manifest["manifest_sha256"],
              "population_note": manifest["population_note"],
              "candidates": {c["name"]: agreement(c, out, pairs, human, reason) for c in CANDIDATES}}
    result["highest_agreement"] = max(result["candidates"], key=lambda name: result["candidates"][name]["agreement"])
    (out / "judge_agreement_with_human.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def heldout(sheet: Path, out: Path, judge: str) -> dict:
    """The chosen judge on the held-out set, once."""
    target = out / "judge_heldout_agreement.json"
    if target.exists():
        raise SystemExit(f"{target} exists: the held-out set is evaluated once")
    human, reason = human_decisions(sheet)
    manifest = load_split(out)
    candidate = next(c for c in CANDIDATES if c["name"] == judge)
    pairs = set(manifest["heldout"])
    result = {"set": "held-out judge-audit", "judge": judge, "human_equivalent": sum(human[p] for p in pairs), "split_manifest_sha256": manifest["manifest_sha256"],
              "population_note": manifest["population_note"], **agreement(candidate, out, pairs, human, reason),
              "evaluated_once": time.strftime("%Y-%m-%d %H:%M:%S"), "the_judge_is_not_changed_after_this": True}
    target.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    target.chmod(0o444)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("split", "decide", "compare", "heldout"))
    parser.add_argument("--sheet", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--judge", help="heldout: the candidate that was chosen on the selection set")
    args = parser.parse_args()
    out = ROOT / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    sheet = ROOT / args.sheet
    result = heldout(sheet, out, args.judge) if args.command == "heldout" else {"split": split, "decide": decide, "compare": compare}[args.command](sheet, out)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
