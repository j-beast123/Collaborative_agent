"""Secondary diagnostic of the backbone pilot's final diagnoses: how many strict failures are near misses?
Analysis only. The primary result stays the run's strict diagnosis-equivalence judgment (majority-v1 mapping); nothing here
changes a label, the 2-of-3 criterion, a rubric selection or any run.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from scripts.diagnostics.failure_audit import GENERIC, names
from src.agentclinic.outcome import normalize
from src.agentclinic.runtime import build_judge
from src.common.config import read_config
from src.common.io import read_jsonl
from src.judges.llm import CachedChat, ChatRequest, build_chat_backend, parse_json_object

ROOT = Path(__file__).resolve().parents[2]
VERSION = "near-miss-v1"
LABELS = ("STRICT_CORRECT", "NEAR_MISS", "CLEARLY_WRONG")
PROMPT = """You compare a predicted final diagnosis with the gold diagnosis of a clinical case, for a research analysis.
Separately from any strict exact or equivalence scoring, say how close the prediction is to the gold diagnosis clinically
and semantically. You see only the two strings.

- "equivalent": the prediction names the same diagnosis (synonym, abbreviation, or the same entity in other words).
- "near_miss": not the same diagnosis, but the prediction substantively contains the gold's core disease process or
  diagnostic concept, or is a clinically very close broader or narrower diagnosis of it.
- "clearly_wrong": the core disease concept differs; or a gold element is mentioned but the prediction's primary
  diagnosis is substantively another one.

Also give the relation of the prediction to the gold: "same", "broader" (the gold is a kind or a cause-specific form of
the prediction), "narrower" (the prediction is a kind of the gold), "overlapping" (they share the core process but neither
contains the other), or "different".

Return exactly one JSON object and no Markdown:
{"label": "near_miss", "relation": "broader", "reason": "one sentence"}"""


ERROR_PROMPT = """You classify why a predicted final diagnosis of a clinical case is not the gold diagnosis, for a research analysis.
You see only the two strings. Choose exactly one type:

- "broader_than_gold": the prediction is a broader category, process or syndrome that contains the gold diagnosis
  (e.g. naming the general process instead of its specific named form).
- "narrower_or_manifestation": the prediction names a part, a manifestation, a complication or one localized form of the
  gold diagnosis instead of the gold diagnosis itself.
- "associated_syndrome_substitution": the prediction names a different but associated syndrome or entity that includes,
  accompanies or is classically linked to the gold diagnosis.
- "multiple_or_weak_commitment": the prediction lists several diagnoses, hedges between alternatives, or contains the gold
  concept only as a secondary or possible item next to another primary diagnosis.
- "genuinely_wrong": a different disease, without one of the relations above.

Return exactly one JSON object and no Markdown:
{"error_type": "broader_than_gold", "reason": "one sentence"}"""
ERROR_TYPES = ("broader_than_gold", "narrower_or_manifestation", "associated_syndrome_substitution", "multiple_or_weak_commitment",
               "genuinely_wrong")


def content_words(text: str) -> set:
    return {w for w in normalize(text).split() if w not in GENERIC and len(w) > 2}


def rows_of(backbone_dir: Path, persistent_dir: Path | None) -> list[dict]:
    rows = [r for path in sorted(backbone_dir.glob("rollouts_*.jsonl")) for r in read_jsonl(path)]
    if persistent_dir:
        for r in read_jsonl(persistent_dir / "persistent_rollouts.jsonl"):
            rows.append({"backbone": "4b", "condition": "persistent", "case_id": r["case_id"], "rubric_id": r["rubric_id"],
                         "rerun": r["rerun"], "status": r["status"], "final_diagnosis": r["final_diagnosis"], "correct": r["correct"],
                         "from_turn": r["intervention_turn_id"], "dialogue": r["dialogue"], "reused_from": str(persistent_dir)})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backbone-pilot", required=True)
    parser.add_argument("--persistent-pilot")
    args = parser.parse_args()
    base, out = ROOT / args.backbone_pilot, ROOT / args.backbone_pilot / "near_miss"
    out.mkdir(parents=True, exist_ok=True)
    targeted = json.loads((base / "run.sh").read_text(encoding="utf-8").split("--targeted-pilot ")[1].split()[0].join(['"', '"']))
    config = dict(json.loads((ROOT / targeted / "resolved_config.json").read_text(encoding="utf-8")), output_dir=str(out.relative_to(ROOT)))
    judge = build_judge(config, ROOT, read_config(ROOT / config["environment"]["config"]))
    gold_of = {t["case_id"]: t["ground_truth"]["correct_diagnosis"] for t in read_jsonl(ROOT / targeted / "base/trajectories.jsonl")}
    chat = CachedChat(build_chat_backend(config["roles"]["rubric_manager"]), out / "cache_reader.jsonl")
    rows = rows_of(base, ROOT / args.persistent_pilot if args.persistent_pilot else None)

    pairs = sorted({(gold_of[r["case_id"]], r["final_diagnosis"] or "") for r in rows if not r["correct"]})

    def read(pair: tuple) -> tuple:
        gold, predicted = pair
        if not predicted:
            return pair, {"label": "clearly_wrong", "relation": "different", "reason": "no final diagnosis"}
        answer = chat.complete(ChatRequest(PROMPT, json.dumps({"gold_diagnosis": gold, "predicted_final_diagnosis": predicted}, ensure_ascii=False),
                                           VERSION, {"temperature": 0, "max_tokens": 300}))
        parsed, _ = parse_json_object(answer.text) if answer.status == "ok" else (None, None)
        return pair, parsed or {"label": None, "relation": None, "reason": "unread"}

    def classify(pair: tuple) -> tuple:
        gold, predicted = pair
        if not predicted:
            return pair, {"error_type": "genuinely_wrong", "reason": "no final diagnosis"}
        answer = chat.complete(ChatRequest(ERROR_PROMPT, json.dumps({"gold_diagnosis": gold, "predicted_final_diagnosis": predicted}, ensure_ascii=False),
                                           "near-miss-error-type-v1", {"temperature": 0, "max_tokens": 300}))
        parsed, _ = parse_json_object(answer.text) if answer.status == "ok" else (None, None)
        return pair, parsed if parsed and parsed.get("error_type") in ERROR_TYPES else {"error_type": None, "reason": "unread"}

    with ThreadPoolExecutor(8) as pool:
        reading = dict(pool.map(read, pairs))
        error_type = dict(pool.map(classify, pairs))
    equivalent = lambda gold, text: bool(text) and bool(judge.equivalent(gold, text)["equivalent"])  # noqa: E731
    results = []
    for r in rows:
        gold, predicted = gold_of[r["case_id"]], r["final_diagnosis"] or ""
        dialogue = r["dialogue"]
        first = lambda speaker, field: next((m["stage"] for m in dialogue if m["speaker"] == speaker  # noqa: E731
                                             and equivalent(gold, m.get(field) or (m.get("final_diagnosis") if speaker == "doctor" else ""))), None)
        seen = reading.get((gold, predicted), {})
        label = "STRICT_CORRECT" if r["correct"] else ("NEAR_MISS" if seen.get("label") in ("near_miss", "equivalent") else
                                                         "CLEARLY_WRONG" if seen.get("label") == "clearly_wrong" else "UNREAD")
        words = content_words(gold)
        results.append({"backbone": r["backbone"], "condition": r["condition"], "case_id": r["case_id"], "rerun": r["rerun"] + 1,
                        "gold": gold, "predicted": r["final_diagnosis"], "strict_correct": bool(r["correct"]), "secondary_label": label,
                        "reader_label": seen.get("label"), "reader_equivalent": (not r["correct"]) and seen.get("label") == "equivalent",
                        "relation": seen.get("relation"), "reason": seen.get("reason"),
                        "error_type": None if r["correct"] else error_type.get((gold, predicted), {}).get("error_type"),
                        "error_type_reason": None if r["correct"] else error_type.get((gold, predicted), {}).get("reason"),
                        "gold_string_in_prediction": normalize(gold) in normalize(predicted),
                        "gold_term_overlap": round(len(words & content_words(predicted)) / len(words), 3) if words else None,
                        "gold_named_in_dialogue": any(names(gold, m["text"]) for m in dialogue),
                        "first_agent_turn_at_gold": first("agent", "assessment"),
                        "first_doctor_turn_at_gold": first("doctor", "provisional_diagnosis")})
    (out / "near_miss.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in results), encoding="utf-8")
    with (out / "near_miss.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    summary = {}
    for key in sorted({(x["backbone"], x["condition"]) for x in results}):
        group = [x for x in results if (x["backbone"], x["condition"]) == key]
        count = Counter(x["secondary_label"] for x in group)
        summary["_".join(key)] = {"reruns": len(group), **{k: count.get(k, 0) for k in (*LABELS, "UNREAD")},
                                  "strict_plus_near_miss": count.get("STRICT_CORRECT", 0) + count.get("NEAR_MISS", 0),
                                  "reader_equivalent_on_strict_failure": sum(x["reader_equivalent"] for x in group),
                                  "near_miss_cases": sorted({x["case_id"][-4:] for x in group if x["secondary_label"] == "NEAR_MISS"}),
                                  "agent_reached_gold": sum(bool(x["first_agent_turn_at_gold"]) for x in group),
                                  "doctor_reached_gold": sum(bool(x["first_doctor_turn_at_gold"]) for x in group),
                                  "gold_named_in_dialogue": sum(x["gold_named_in_dialogue"] for x in group),
                                  "error_types": dict(Counter(x["error_type"] for x in group if not x["strict_correct"]))}
    (out / "near_miss_summary.json").write_text(json.dumps({"summary": summary, "reader": config["roles"]["rubric_manager"]["model_id"]
                                                            + " (analysis only, blind to backbone and condition; not a ground truth)",
                                                            "note": "primary result = strict majority-v1; this is secondary"},
                                                           ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    for x in results:
        if x["secondary_label"] != "CLEARLY_WRONG":
            print(x["backbone"], x["condition"], x["case_id"][-4:], x["rerun"], x["secondary_label"], x["relation"], "|", x["gold"], "<-", x["predicted"])


if __name__ == "__main__":
    main()
