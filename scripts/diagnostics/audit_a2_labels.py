"""Audit of the A2 Effectiveness labels before the joint predictor is trained on them."""

from __future__ import annotations

import argparse
import inspect
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

from src.agentclinic.adapter import doctor_view, load_cases, render
from src.agentclinic.outcome import normalize
from src.agentclinic.protocol import doctor_prompt
from src.common.config import read_config
from src.common.io import read_jsonl
from src.effectiveness.pair_data import load_pairs, to_examples

ROOT = Path(__file__).resolve().parents[2]
MARKERS = ("candidate", "intervention", "fixed", "guideline", "Correct_Diagnosis", "behavior", "compliance")


def label_statistics(labels: list[dict], k: int) -> dict:
    by_history = defaultdict(list)
    for row in labels:
        by_history[row["history_id"]].append(row)
    pairs = sum(len({r["after"]["successes"] for r in rows}) > 1 for rows in by_history.values())
    return {"histories": len(by_history), "pairs": len(labels),
            "nonzero_delta_pairs": sum(r["delta_effectiveness"] != 0 for r in labels),
            "histories_with_a_nonzero_delta": sum(any(r["delta_effectiveness"] != 0 for r in rows) for rows in by_history.values()),
            "before_floor_or_ceiling_share": mean(rows[0]["before"]["successes"] in (0, k) for rows in by_history.values()) if by_history else None,
            "after_floor_or_ceiling_share": mean(r["after"]["successes"] in (0, k) for r in labels) if labels else None,
            "histories_with_candidate_ordering": pairs,
            "within_history_ordering_pairs": sum(sum(a["after"]["successes"] != b["after"]["successes"]
                                                     for i, a in enumerate(rows) for b in rows[i + 1:]) for rows in by_history.values()),
            "mean_abs_delta": mean(abs(r["delta_effectiveness"]) for r in labels) if labels else None,
            "delta_by_candidate": {name: {"n": len(rows), "nonzero": sum(r["delta_effectiveness"] != 0 for r in rows),
                                          "mean_delta": mean(r["delta_effectiveness"] for r in rows),
                                          "min_delta": min(r["delta_effectiveness"] for r in rows),
                                          "max_delta": max(r["delta_effectiveness"] for r in rows)}
                                   for name in sorted({r["candidate_id"] for r in labels})
                                   if (rows := [r for r in labels if r["candidate_id"] == name])}}


def audit(labels_config: dict, predictor_config_path: Path) -> dict:
    from scripts import train_joint
    from src.effectiveness.train import load_predictor_config

    out, k = ROOT / labels_config["output_dir"], labels_config["k"]
    names = labels_config["candidates"]["names"]
    histories = {h["history_id"]: h for h in read_jsonl(out / "histories_a2.jsonl")}
    before, after = read_jsonl(out / "before_rollouts.jsonl"), read_jsonl(out / "after_rollouts.jsonl")
    labels = read_jsonl(out / "labels.jsonl")
    candidates = {(c["history_id"], c["candidate_id"]): c for c in read_jsonl(out / "candidates.jsonl")}
    compliance = {(c["history_id"], c["candidate_id"]): c for c in read_jsonl(out / "candidate_compliance.jsonl")}
    cases = {c["case_id"]: c for c in load_cases(ROOT / labels_config["dataset"]["path"],
                                                 (labels_config.get("doctor_view") or {}).get("patient_fields"))}
    labelled = {(r["history_id"], r["candidate_id"]): r for r in labels}
    problems: dict[str, list] = defaultdict(list)

    # 2, 3, 4: what was held fixed and what was sampled
    before_by, after_by = defaultdict(list), defaultdict(list)
    for row in before:
        before_by[row["history_id"]].append(row)
    for row in after:
        after_by[(row["history_id"], row["candidate_id"], row.get("candidate_attempt", 1))].append(row)
    for row in before + after:
        if row["status"] == "completed" and [m["text"] for m in row["dialogue"][:3]] != [m["text"] for m in histories[row["history_id"]]["history"]]:
            problems["2_history_identical_across_candidates"].append(row["history_id"])
    for (history_id, name), label in labelled.items():
        rows = after_by[(history_id, name, label["candidate_attempt"])]
        texts = {r["dialogue"][3]["text"] for r in rows if r["status"] == "completed"}
        if texts != {label["response"]} or len(rows) != k or not all(r["dialogue"][3].get("fixed") for r in rows):
            problems["3_after_uses_one_fixed_candidate_text"].append(f"{history_id}/{name}")
    natural_unique = {hid: len({r["dialogue"][3]["text"] for r in rows if r["status"] == "completed"}) for hid, rows in before_by.items()}
    for history_id, rows in before_by.items():
        if any(r["dialogue"][3].get("fixed") for r in rows if r["status"] == "completed"):
            problems["4_before_samples_the_natural_A2"].append(history_id)
    # a natural A2 that came out identical in all 9 samples is possible for a peaked policy; all histories identical is not
    if natural_unique and mean(v > 1 for v in natural_unique.values()) < 0.5:
        problems["4_before_samples_the_natural_A2"].append("most histories have one A2 text in all BEFORE continuations")

    # 5, 6: one shared BEFORE per history, one label per pair
    counts = defaultdict(set)
    for row in labels:
        counts[row["history_id"]].add((row["before"]["successes"], row["before"]["trials"]))
        done = [r for r in before_by[row["history_id"]] if r["status"] == "completed" and r["success"] is not None]
        if (sum(r["success"] for r in done), len(done)) != (row["before"]["successes"], row["before"]["trials"]) or len(done) != k:
            problems["5_one_shared_before_per_history"].append(row["history_id"])
    problems["5_one_shared_before_per_history"] += [hid for hid, values in counts.items() if len(values) != 1]
    if len(labelled) != len(labels):
        problems["6_one_label_per_pair"].append("duplicate (history, candidate) rows")
    for row in labels:
        if row["after"]["trials"] != k or row["before"]["trials"] != k or \
                abs(row["delta_effectiveness"] - (row["after"]["successes"] - row["before"]["successes"]) / k) > 1e-12:
            problems["6_one_label_per_pair"].append(f"{row['history_id']}/{row['candidate_id']}")

    # 7: predictor inputs (exactly the texts the training script builds)
    predictor = load_predictor_config(predictor_config_path)
    spec = predictor["continuation_spec"]
    pairs = load_pairs(out / "labels.jsonl", protocol_hash=spec.protocol_hash(), expected_k=spec.k)
    examples = to_examples(pairs, predictor["predictor"]["input_format"])
    responses = {(r["history_id"], r["candidate_id"]): r["response"] for r in labels}
    for example in examples:
        own = responses.get((example.history_id, example.response_id), "")
        text = example.text.replace(own, "") if own else example.text
        for row in histories[example.history_id]["history"]:  # what a speaker said is dialogue, not metadata
            text = text.replace(row["text"], "")
        if any(name in text for name in names) or "candidate" in text.lower() or "Intervention turn: A2" not in example.text:
            problems["7_no_candidate_metadata_in_predictor_input"].append(f"{example.history_id}/{example.response_id}")
    if len(examples) != len(labels) + len(counts):
        problems["6_one_label_per_pair"].append("training examples are not one BEFORE per history plus one AFTER per pair")

    # 8: only compliant candidates are labelled
    for key in labelled:
        if not compliance.get(key, {}).get("behavior_compliant"):
            problems["8_only_compliant_candidates_labelled"].append("/".join(key))
    coverage = {name: {"histories": len(histories), "generated": sum((h, name) in candidates for h in histories),
                       "compliant": sum(bool(compliance.get((h, name), {}).get("behavior_compliant")) for h in histories),
                       "labelled": sum((h, name) in labelled for h in histories),
                       "mean_generations": mean(candidates[(h, name)].get("attempt", 1) for h in histories if (h, name) in candidates)
                       if any((h, name) in candidates for h in histories) else None} for name in names}

    # 10: the Doctor's FINAL prompt, rebuilt as the protocol builds it, with the stored dialogue.
    skeletons = set()
    for row in after[:: max(1, len(after) // 400)] + before[:: max(1, len(before) // 100)]:
        if row["status"] != "completed":
            continue
        case = cases[row["case_id"]]
        prompt = doctor_prompt(case, row["dialogue"][:4], "FINAL")
        skeleton = prompt.replace(render(doctor_view(case)), "<DOCTOR INFORMATION>")
        for message in row["dialogue"][:4]:
            skeleton = skeleton.replace(message["text"], "<MESSAGE>")
        skeletons.add(skeleton)
        if normalize(case["gold"]) in normalize(prompt.split("Conversation so far:")[0]) or "Correct_Diagnosis" in prompt:
            problems["10_final_prompt_is_the_normal_dialogue"].append(f"{row['history_id']}: correct diagnosis in the Doctor's information")
    if len(skeletons) != 1 or any(marker.lower() in skeleton.lower() for skeleton in skeletons for marker in MARKERS):
        problems["10_final_prompt_is_the_normal_dialogue"].append(f"{len(skeletons)} prompt forms, or a marker in the fixed text")
    if "fixed" in inspect.getsource(doctor_prompt) or "candidate" in inspect.getsource(doctor_prompt):
        problems["10_final_prompt_is_the_normal_dialogue"].append("doctor_prompt reads candidate fields")

    # 11: decoding of the A1 label run
    source = read_config(ROOT / "configs/effectiveness/v5_agentclinic_full.json")
    same_protocol = all(labels_config[key] == source[key] for key in ("models", "decoding", "prompts", "k", "split", "dataset"))
    if not same_protocol:
        problems["11_same_decoding_as_the_A1_labels"].append("models, decoding, prompts, K, split or dataset differ from the A1 label run")
    diversity = {"final_raw_outputs_per_pair": [], "final_diagnoses_per_pair": []}
    for rows in after_by.values():
        done = [r for r in rows if r["status"] == "completed"]
        if len(done) == k:
            diversity["final_raw_outputs_per_pair"].append(len({r["dialogue"][4]["text"] for r in done}))
            diversity["final_diagnoses_per_pair"].append(len({normalize(r["final_diagnosis"] or "") for r in done}))

    # 1: what the training script reads, from its source
    usage = {
        "cross_validation_reads": "TRAIN" if 'load_examples(config, "train")' in inspect.getsource(train_joint.cross_validate) else "?",
        "final_training_reads": "TRAIN" if 'load_examples(config, "train")' in inspect.getsource(train_joint.final) else "?",
        "calibration_fitted_on": "out-of-fold TRAIN logits (inside cross_validate)",
        "dev_read_by": [name for name in ("cross_validate", "final", "evaluate", "run_fit", "build_model")
                        if '"dev"' in inspect.getsource(getattr(train_joint, name))],
        "final_monitoring_set": "TRAIN fold 0",
    }
    final_prompt_form = next(iter(skeletons)) if len(skeletons) == 1 else None
    if usage["dev_read_by"] != ["evaluate"] or "?" in (usage["cross_validation_reads"], usage["final_training_reads"]):
        problems["1_dev_only_in_the_final_evaluation"].append(str(usage))
    splits = Counter(r["split"] for r in labels)
    by_split = {split: label_statistics([r for r in labels if r["split"] == split], k) for split in ("train", "dev")}

    required = ["1_dev_only_in_the_final_evaluation", "2_history_identical_across_candidates", "3_after_uses_one_fixed_candidate_text",
                "4_before_samples_the_natural_A2", "5_one_shared_before_per_history", "6_one_label_per_pair",
                "7_no_candidate_metadata_in_predictor_input", "8_only_compliant_candidates_labelled",
                "10_final_prompt_is_the_normal_dialogue", "11_same_decoding_as_the_A1_labels"]
    checks = {name: not problems[name] for name in required}
    return {
        "passed": all(checks.values()), "checks": checks, "problems": {k: v[:10] for k, v in problems.items() if v},
        "data_usage": usage, "labels_by_split": dict(splits),
        "before_natural_a2": {"histories": len(natural_unique), "mean_distinct_a2_texts_of_9": mean(natural_unique.values()) if natural_unique else None,
                              "histories_with_one_text_only": sum(v == 1 for v in natural_unique.values())},
        "candidate_coverage": coverage,
        "statistics_all_candidates": {"pooled": label_statistics(labels, k), **by_split},
        "statistics_without_request_evidence": {"pooled": label_statistics([r for r in labels if r["candidate_id"] != "request_evidence"], k),
                                                **{s: label_statistics([r for r in labels if r["split"] == s and r["candidate_id"] != "request_evidence"], k)
                                                   for s in ("train", "dev")}},
        "final_output_diversity_k9": {name: {"mean": mean(values), "pairs_with_one_value": sum(v == 1 for v in values), "pairs": len(values)}
                                      for name, values in diversity.items() if values},
        "doctor_decoding": labels_config["decoding"]["doctor"], "agent_decoding": labels_config["decoding"]["agent"],
        "final_prompt_form": final_prompt_form,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels-config", required=True)
    parser.add_argument("--predictor-config", required=True)
    args = parser.parse_args()
    config = read_config(args.labels_config)
    result = audit(config, Path(args.predictor_config))
    out = ROOT / config["output_dir"]
    (out / "a2_label_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    pooled, without = result["statistics_all_candidates"]["pooled"], result["statistics_without_request_evidence"]["pooled"]
    lines = ["# A2 label audit before joint training", "", f"Passed: {result['passed']}", "",
             "| check | holds |", "|---|---|", *[f"| {name} | {ok} |" for name, ok in result["checks"].items()], "",
             f"Data usage: {json.dumps(result['data_usage'])}", "",
             f"Labels: {result['labels_by_split']}. BEFORE: {result['before_natural_a2']}.", "",
             "| | histories | pairs | nonzero delta | histories with ordering | ordering pairs | BEFORE at 0 or K | AFTER at 0 or K |", "|---|---|---|---|---|---|---|---|",
             *[f"| {name} | {s['histories']} | {s['pairs']} | {s['nonzero_delta_pairs']} | {s['histories_with_candidate_ordering']} | "
               f"{s['within_history_ordering_pairs']} | {s['before_floor_or_ceiling_share']:.2f} | {s['after_floor_or_ceiling_share']:.2f} |"
               for name, s in (("all candidates", pooled), ("without request_evidence", without))], "",
             f"Candidate coverage: {json.dumps(result['candidate_coverage'])}", "",
             f"FINAL diversity over K = 9: {json.dumps(result['final_output_diversity_k9'])}", ""]
    (out / "a2_label_audit.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in ("candidate_coverage",)}, indent=2))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
