"""Audit of the main pipeline on the real data, without any model or API call."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path

from src.agentclinic.adapter import (GOLD_FIELD, PRIVATE_FIELDS, agent_view, assert_no_leakage, doctor_view, leaf_values,
                                     load_cases, render, render_private)
from src.agentclinic.outcome import normalize
from src.agentclinic.protocol import agent_prompt, doctor_prompt, standalone_prompt, state_prompt
from src.agentclinic.runtime import frozen_histories
from src.agentclinic.selection import split_eligible_cases
from src.agentclinic.trajectory import to_trajectory
from src.common.config import read_config
from src.common.io import read_jsonl
from src.common.main_config import load_main_config
from src.effectiveness import calibration
from src.effectiveness.history import format_input
from src.grpo.dataset import grpo_rows
from src.grpo.environment import ClinicEnvironment
from src.judges.failure_analysis import analyzer_input
from src.judges.rubric_scoring import build_scoring_items

ROOT = Path(__file__).resolve().parents[1]
PLACEHOLDER = [
    {"speaker": "doctor", "stage": "D1", "text": "[doctor message one]\nProvisional diagnosis: [doctor diagnosis]",
     "provisional_diagnosis": "[doctor diagnosis]"},
    {"speaker": "agent", "stage": "A1", "text": "[agent message one]", "assessment": "[agent diagnosis]"},
    {"speaker": "doctor", "stage": "D2", "text": "[doctor message two]\nProvisional diagnosis: [doctor diagnosis]",
     "provisional_diagnosis": "[doctor diagnosis]"},
    {"speaker": "agent", "stage": "A2", "text": "[agent message two]", "assessment": "[agent diagnosis]"},
    {"speaker": "doctor", "stage": "FINAL", "text": '{"final_diagnosis": "[final diagnosis]"}', "final_diagnosis": "[final diagnosis]"},
]
STATE = {"state_status": "elicited", "agent_current_assessment": "[agent diagnosis]",
         "doctor_current_assessment": "[doctor diagnosis]", "better_supported_position": "unclear",
         "evidence_sufficient": False, "supporting_evidence_summary": "[summary]"}
RUBRIC = {"rubric_id": "audit", "name": "n", "applicability": {"user_side_context": "c", "agent_side_state": "s"},
          "scores": {level: {"user_side_context": "c", "agent_side_state": "s", "behavior": f"In this situation, the agent {level}"}
                     for level in ("0", "1", "2")}}


class NoJudge:
    """Outcome judge stand-in: the audit builds trajectories without judging anything."""

    def equivalent(self, reference: str, predicted: str) -> dict:
        return {"equivalent": False, "method": "audit", "reason": ""}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def inventory(folders: list[str]) -> dict:
    files, in_progress = [], []
    for folder in folders:
        for path in sorted((ROOT / folder).rglob("*")):
            if not path.is_file() or "/audit/" in f"/{path.relative_to(ROOT)}":
                continue
            try:
                before = path.stat()
                value = sha256(path)
                after = path.stat()
            except FileNotFoundError:  # removed while the inventory ran (a run in progress)
                in_progress.append(str(path.relative_to(ROOT)))
                continue
            if (before.st_size, before.st_mtime) != (after.st_size, after.st_mtime):
                in_progress.append(str(path.relative_to(ROOT)))
                continue
            files.append({"path": str(path.relative_to(ROOT)), "size": after.st_size, "sha256": value,
                          "modified": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(after.st_mtime))})
    return {"created": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "root": str(ROOT), "folders": folders,
            "files": files, "in_progress": in_progress}


def verify(manifest_path: Path, running: tuple = ()) -> dict:
    """Every file of an earlier manifest against the disk now."""
    earlier = json.loads(manifest_path.read_text(encoding="utf-8"))
    result = {"manifest": str(manifest_path), "manifest_created": earlier.get("created") or earlier.get("created_utc"),
              "files": len(earlier["files"]), "unchanged": 0, "changed": [], "missing": [], "of_a_run_in_progress": []}
    for entry in earlier["files"]:
        path = next((p for p in (ROOT / entry["path"], ROOT.parent / entry["path"]) if p.is_file()), None)
        if any(fragment in entry["path"] for fragment in running):
            result["of_a_run_in_progress"].append(entry["path"])
        elif path is None:
            result["missing"].append(entry["path"])
        elif sha256(path) != entry["sha256"]:
            result["changed"].append(entry["path"])
        else:
            result["unchanged"] += 1
    result["preserved"] = not result["changed"] and not result["missing"]
    return result


def contains(text: str, needle: str) -> bool:
    """Whole-phrase match on normalized text (lower case, punctuation removed)."""
    needle = normalize(needle)
    return bool(needle) and re.search(rf"(?<![a-z0-9]){re.escape(needle)}(?![a-z0-9])", normalize(text)) is not None


def surfaces(case: dict, environment: ClinicEnvironment, agent_system: str) -> dict[str, str]:
    """Every text a model receives for this case on the main path, by name."""
    d1, a1, d2, a2, _ = PLACEHOLDER
    history_id = f"{case['case_id']}:A1"
    trajectory = to_trajectory(case, PLACEHOLDER, trajectory_id=f"{history_id}:audit", standalone="[agent diagnosis]",
                               states={"A1": STATE, "A2": STATE}, judge=NoJudge())
    predictor = environment.predictor_history(history_id, [d1], "A1")
    row, = grpo_rows({case["case_id"]: dict(case, split="train")}, [{"history_id": history_id, "case_id": case["case_id"],
                                                                      "history": [d1]}], agent_system)
    scoring = build_scoring_items(trajectory, [RUBRIC], include_initial_turn=True)
    return {
        "doctor_prompt_D1": doctor_prompt(case, [], "D1"),
        "doctor_prompt_D2": doctor_prompt(case, [d1, a1], "D2"),
        "doctor_prompt_FINAL": doctor_prompt(case, [d1, a1, d2, a2], "FINAL"),
        "agent_prompt_A1": agent_prompt(case, [d1], "A1"),
        "agent_prompt_A2": agent_prompt(case, [d1, a1, d2], "A2"),
        "agent_standalone_prompt": standalone_prompt(case),
        "state_elicitation_before_A1": state_prompt(case, [d1]),
        "state_elicitation_before_A2": state_prompt(case, [d1, a1, d2]),
        "grpo_policy_prompt": json.dumps(row["prompt"], ensure_ascii=False),
        "effectiveness_input_before": format_input(predictor),
        "effectiveness_input_after": format_input(predictor, a1["text"]),
        "rubric_generation_input": json.dumps(analyzer_input(trajectory), ensure_ascii=False),
        "rubric_scoring_input_A1": json.dumps(scoring[0].payload(), ensure_ascii=False),
        "rubric_scoring_input_A2": json.dumps(scoring[1].payload(), ensure_ascii=False),
    }


def leakage(config: dict) -> dict:
    source = config["resolved_environment"]
    raw = {c["case_id"]: c for c in load_cases(ROOT / source["dataset"]["path"], (source.get("doctor_view") or {}).get("patient_fields"))}
    osce = [json.loads(line)["OSCE_Examination"] for line in (ROOT / source["dataset"]["path"]).read_text(encoding="utf-8").splitlines() if line.strip()]
    cases = split_eligible_cases(source, ROOT)
    agent_system = (ROOT / source["prompts"]["consultant"]).read_text(encoding="utf-8")
    histories = [{"history_id": f"{cid}:A1", "case_id": cid, "history": [PLACEHOLDER[0]]} for cid in cases]
    environment = ClinicEnvironment(cases, histories, None, None)

    gold_hits: dict[str, list[str]] = {}
    private_in_doctor, boundary_errors, evaluator_fields, keys, mentions = [], [], [], [], []
    for cid, case in sorted(cases.items()):
        try:
            assert_no_leakage(case)
        except AssertionError as error:
            boundary_errors.append(str(error))
        texts = surfaces(case, environment, agent_system)
        for name, text in texts.items():
            if contains(text, case["gold"]):
                gold_hits.setdefault(name, []).append(cid)
            if GOLD_FIELD in text or "ground_truth" in text or "correct_diagnosis" in text:
                evaluator_fields.append(f"{cid}: {name}")
        # the Doctor's own information section must hold no consultant-only finding the patient did not report
        patient_text = "\n".join(leaf_values(case["doctor_view"])).lower()
        for name in ("doctor_prompt_D1", "doctor_prompt_D2", "doctor_prompt_FINAL"):
            information = texts[name].split("Conversation so far:")[0].lower()
            leaked = [leaf for field in PRIVATE_FIELDS for leaf in leaf_values(case["private_evidence"][field])
                      if len(leaf) > 25 and leaf.lower() in information and leaf.lower() not in patient_text]
            if leaked or "consultant only" in information:  # the label render() puts on the consultant-only sections
                private_in_doctor.append(f"{cid}: {name}")
        # informational: an Objective can say that findings exist ("... and test results") without stating any
        if any(field.replace("_", " ").lower() in texts["doctor_prompt_D1"].split("Conversation so far:")[0].lower()
               for field in PRIVATE_FIELDS):
            mentions.append(cid)
        # fields of the source case that reach no prompt at all
        other = {k: v for k, v in osce[case["source_index"]].items()
                 if k not in ("Objective_for_Doctor", "Patient_Actor", *PRIVATE_FIELDS, GOLD_FIELD)}
        for field, value in other.items():
            for leaf in leaf_values(value):
                if len(leaf) > 25 and any(leaf in text for text in texts.values()):
                    evaluator_fields.append(f"{cid}: unused field {field} appears in a prompt")
        generation = json.loads(texts["rubric_generation_input"])
        keys += [f"{cid}: {k}" for k in ("ground_truth", "is_correct", "collaboration_state", "agent_initial_correct",
                                         "doctor_initial_correct", "outcome_method") if k in generation]
        # the Agent does get the consultant-only findings (the audit would be vacuous otherwise)
        if render_private(case).splitlines()[1].strip() not in texts["agent_prompt_A1"]:
            boundary_errors.append(f"{cid}: the Agent prompt lacks the consultant-only findings")

    # where the gold string does occur, is it already in the source text the role is allowed to see?
    explained = {}
    for name, ids in gold_hits.items():
        allowed = doctor_view if name.startswith("doctor_prompt") else agent_view
        explained[name] = {"cases": len(ids), "case_ids": ids,
                           "all_from_the_role_s_own_case_text": all(contains(render(allowed(cases[c])), cases[c]["gold"]) for c in ids)}

    splits = {name: sorted(c for c, case in cases.items() if case["split"] == name) for name in ("train", "dev", "test")}
    label_dir = ROOT / source["output_dir"]
    used = {name: sorted({row["case_id"] for row in read_jsonl(label_dir / name, missing_ok=True)})
            for name in ("histories.jsonl", "candidates.jsonl", "before_rollouts.jsonl", "after_rollouts.jsonl",
                         "labels_train.jsonl", "labels_dev.jsonl")}
    test = set(splits["test"])
    manifest = json.loads((ROOT / config["data"]["split_manifest"]).read_text(encoding="utf-8"))
    manifest_split = {json.loads(key)[1]: {"eval": "test"}.get(member["split"], member["split"])
                      for key, member in manifest["members"].items()}
    split_checks = {
        "sizes": {name: len(ids) for name, ids in splits.items()},
        "pairwise_disjoint": not (set(splits["train"]) & set(splits["dev"])) and not (set(splits["train"]) & test)
        and not (set(splits["dev"]) & test),
        "split_manifest_matches_the_computed_split": manifest_split == {c: case["split"] for c, case in cases.items()},
        "test_cases_in_label_run_files": {name: len(set(ids) & test) for name, ids in used.items()},
        "train_labels_only_train_cases": set(used["labels_train.jsonl"]) <= set(splits["train"]),
        "dev_labels_only_dev_cases": set(used["labels_dev.jsonl"]) <= set(splits["dev"]),
        "frozen_histories_without_test": not ({h["case_id"] for h in frozen_histories(config, ROOT, cases)} & test),
        "grpo_prompts_are_train_only": {cases[h["case_id"]]["split"] for h in frozen_histories(config, ROOT, cases)
                                        if h["history_id"] in {r["sample_id"] for r in grpo_rows(
                                            cases, frozen_histories(config, ROOT, cases), agent_system)}} == {"train"},
    }
    gold_ok = all(v["all_from_the_role_s_own_case_text"] for v in explained.values())
    checks = {
        "no_boundary_error": not boundary_errors,
        "no_consultant_only_finding_in_a_doctor_information_section": not private_in_doctor,
        "no_evaluator_only_field_in_any_prompt": not evaluator_fields,
        "no_outcome_or_gold_key_in_rubric_generation_input": not keys,
        "gold_string_never_added_by_the_pipeline": gold_ok,
        "splits_disjoint_and_test_untouched": split_checks["pairwise_disjoint"] and split_checks["split_manifest_matches_the_computed_split"]
        and not any(split_checks["test_cases_in_label_run_files"].values()) and split_checks["train_labels_only_train_cases"]
        and split_checks["dev_labels_only_dev_cases"] and split_checks["frozen_histories_without_test"]
        and split_checks["grpo_prompts_are_train_only"],
    }
    return {"source_cases": len(raw), "eligible_cases": len(cases), "surfaces_per_case": 14, "checks": checks,
            "passed": all(checks.values()), "boundary_errors": boundary_errors, "private_in_doctor_information": private_in_doctor,
            "evaluator_only_fields": evaluator_fields, "outcome_keys_in_rubric_generation_input": keys,
            "gold_string_occurrences": explained,
            "objectives_that_mention_that_findings_exist_without_stating_any": mentions, "splits": split_checks}


def input_parity(config: dict) -> dict:
    """Train / serve consistency of what the models read, on the real label files and the real tokenizer."""
    source = config["resolved_environment"]
    cases = split_eligible_cases(source, ROOT)
    histories = frozen_histories(config, ROOT, cases)
    by_history = {h["history_id"]: h for h in histories}
    environment = ClinicEnvironment(cases, histories, None, None)
    label_dir = ROOT / source["output_dir"]
    compared = differing = 0
    for name in ("labels_train.jsonl", "labels_dev.jsonl"):
        for row in read_jsonl(label_dir / name):
            served = environment.predictor_history(row["history_id"], by_history[row["history_id"]]["history"], "A1")
            compared += 1
            differing += (format_input(row["history"]) != format_input(served)
                          or format_input(row["history"], row["response"]) != format_input(served, row["response"]))
    section = config["reward"]["effectiveness"]
    checkpoint = json.loads((ROOT / section["checkpoint"] / "predictor_config.json").read_text(encoding="utf-8"))
    parameters = calibration.load(ROOT / section["calibration"])
    result = {
        "effectiveness_input": {"label_pairs_compared": compared, "pairs_whose_training_text_differs_from_the_grpo_text": differing},
        "predictor_artifacts": {
            "checkpoint": section["checkpoint"], "checkpoint_id": checkpoint["trainable_state_sha256"][:16],
            "architecture": checkpoint.get("architecture", "single"), "checkpoint_epoch": checkpoint["metadata"].get("epoch"),
            "protocol_hash_matches_config": checkpoint["metadata"].get("continuation_protocol_hash") == section["expected_continuation_protocol_hash"],
            "calibration": parameters, "temperature": 1 / parameters["a"],
            "temperature_within_0.005_of_expected": abs(1 / parameters["a"] - section["expected_temperature"]) <= 0.005,
            "calibration_is_pure_temperature": parameters["b"] == 0.0 and parameters["method"] == "temperature"},
    }
    agent_system = (ROOT / source["prompts"]["consultant"]).read_text(encoding="utf-8")
    rows = grpo_rows(cases, histories, agent_system)
    same_messages = all(row["prompt"][0]["content"] == agent_system and row["prompt"][1]["content"]
                        == agent_prompt(cases[by_history[row["sample_id"]]["case_id"]], by_history[row["sample_id"]]["history"], "A1")
                        for row in rows)
    result["grpo_prompt"] = {"train_prompts": len(rows), "messages_equal_the_label_pipeline_s_A1_prompt": same_messages}
    tokenizer_dir = Path(config["policy"].get("checkpoint") or "")
    if tokenizer_dir.is_dir():
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
        # TRL renders the prompt with chat_template_kwargs = {"enable_thinking": False}; the label pipeline's
        # Agent (BatchedAgent / agent_call) renders (system, user) the same way
        trl = [tokenizer.apply_chat_template(row["prompt"], tokenize=False, add_generation_prompt=True, enable_thinking=False)
               for row in rows[:20]]
        label = [tokenizer.apply_chat_template([{"role": "system", "content": agent_system}, {"role": "user", "content": row["prompt"][1]["content"]}],
                                               tokenize=False, add_generation_prompt=True, enable_thinking=False) for row in rows[:20]]
        lengths = [len(tokenizer(text)["input_ids"]) for text in trl]
        result["grpo_prompt"].update(rendered_prompts_compared=len(trl), rendered_text_identical=trl == label,
                                     no_thinking_block_opened=all("<think>\n\n</think>" in t or "<think>" not in t for t in trl),
                                     prompt_tokens_max_of_sample=max(lengths))
    result["passed"] = differing == 0 and same_messages and result["predictor_artifacts"]["protocol_hash_matches_config"] \
        and result["predictor_artifacts"]["temperature_within_0.005_of_expected"] \
        and result["grpo_prompt"].get("rendered_text_identical", True)
    return result


def markdown(report: dict) -> str:
    lines = ["# Leakage and split audit", "",
             f"{report['eligible_cases']} eligible cases of {report['source_cases']}; {report['surfaces_per_case']} prompt / input surfaces per case, "
             "built by the production builders with a neutral placeholder dialogue. No model was called.", "",
             f"Result: {'PASSED' if report['passed'] else 'FAILED'}", "", "| check | result |", "|---|---|"]
    lines += [f"| {name.replace('_', ' ')} | {'ok' if ok else 'FAILED'} |" for name, ok in report["checks"].items()]
    lines += ["", "## Where the correct-diagnosis string occurs", "",
              "A case text can itself contain the diagnosis words (for example a symptom description). These are occurrences "
              "in the source text the role is allowed to read, not something the pipeline added.", "",
              "| surface | cases | all from the role's own case text |", "|---|---|---|"]
    lines += [f"| {name} | {v['cases']} | {v['all_from_the_role_s_own_case_text']} |" for name, v in sorted(report["gold_string_occurrences"].items())] \
        or ["| (none) | 0 | |"]
    lines += ["", "## Objectives that mention findings", "",
              f"{len(report['objectives_that_mention_that_findings_exist_without_stating_any'])} case(s) have a Doctor objective "
              "that says findings exist (e.g. \"... and test results\") without stating any finding: "
              + ", ".join(report["objectives_that_mention_that_findings_exist_without_stating_any"]) + ". This is source text, not a leak."]
    lines += ["", "## Splits", "", "```json", json.dumps(report["splits"], indent=2), "```", ""]
    for title, key in (("Boundary errors", "boundary_errors"), ("Consultant-only findings in a Doctor information section", "private_in_doctor_information"),
                       ("Evaluator-only fields in a prompt", "evaluator_only_fields"),
                       ("Outcome keys in the rubric-generation input", "outcome_keys_in_rubric_generation_input")):
        lines += [f"## {title}", "", *(f"- {item}" for item in report[key][:50]), *([] if report[key] else ["none"]), ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--verify-against", action="append", default=[], help="an earlier artifact manifest")
    parser.add_argument("--running", action="append", default=[],
                        help="path fragment of a run that is still writing (its files are not expected to be unchanged)")
    parser.add_argument("--skip-inventory", action="store_true")
    args = parser.parse_args()
    config = load_main_config(args.config, ROOT)
    out = ROOT / "outputs" / "audit" / f"audit_{time.strftime('%Y%m%d-%H%M%S')}"
    out.mkdir(parents=True)  # a new directory every time
    summary = {"audit_dir": str(out.relative_to(ROOT))}
    if not args.skip_inventory:
        manifest = inventory(["outputs", "artifacts"])
        (out / "artifact_manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
        summary["inventory"] = {"files": len(manifest["files"]), "in_progress": len(manifest["in_progress"]),
                                "bytes": sum(f["size"] for f in manifest["files"])}
    if args.verify_against:
        preservation = [verify(Path(p), tuple(args.running)) for p in args.verify_against]
        (out / "preservation.json").write_text(json.dumps(preservation, indent=1), encoding="utf-8")
        summary["preservation"] = [{k: (v if not isinstance(v, list) else len(v)) for k, v in p.items()} for p in preservation]
    report = leakage(config)
    (out / "leakage_audit.json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    (out / "leakage_audit.md").write_text(markdown(report), encoding="utf-8")
    parity = input_parity(config)
    (out / "input_parity.json").write_text(json.dumps(parity, indent=1, ensure_ascii=False), encoding="utf-8")
    summary.update(leakage_passed=report["passed"], leakage_checks=report["checks"], input_parity_passed=parity["passed"],
                   input_parity=parity)
    (out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    preserved = all(p["preserved"] for p in summary.get("preservation", []))
    if not (report["passed"] and parity["passed"] and preserved):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
