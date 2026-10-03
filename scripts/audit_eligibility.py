"""Eligibility audit of all AgentClinic-MedQA cases for the main experiment, from the raw case content only."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import time
from pathlib import Path

from src.agentclinic.adapter import PRIVATE_FIELDS, diagnosis_named_in_private_evidence, doctor_view, eligibility, load_cases, render
from src.agentclinic.selection import eligible_cases, split_cases
from src.common.config import read_config

ROOT = Path(__file__).resolve().parents[1]
GENERIC = {"disease", "syndrome", "acute", "chronic", "of", "the", "and", "with", "due", "to", "type", "primary", "secondary",
           "disorder", "infection", "induced", "in", "a"}
# an interpretive clause appended to a finding; cut from its start to the end of the sentence
CLAUSE = re.compile(r"(?:[,;]?\s+(?:which\s+(?:is|are)\s+)?(?:consistent with|compatible with|suggestive of|suggests|indicative of|"
                    r"supportive of|associated with|confirming|confirms|in keeping with)\b[^.]*)", re.I)
SENTENCE = re.compile(r"(?:^|(?<=\.)\s*)(?:These|The) findings (?:are|is) [^.]*\.?", re.I)
NOT_PERFORMED = re.compile(r"\bnot (?:performed|done|available|obtained)\b", re.I)
DANGLING = {"with", "features", "of", "for", "findings", "lesions", "and", "the", "a"}
# a clause that starts with a finite verb and no comma takes the sentence's verb with it ("alignment suggests X")
VERB_CUT = re.compile(r"[^,;]\s+(?:suggests|confirms)\b", re.I)
# judgments the fixed rule cannot make, from reading the finding; raw case content only
REVIEWED = {
    "0196": ("unusable_leakage", "the genetic result is stated only through the diagnosis ('mutations associated with ...'); "
                                 "the cut leaves 'Positive for mutations', which is not a finding"),
}
SPLIT_SEED_SUFFIX = "eligibility_v2"


def words(text: str) -> list[str]:
    return re.sub(r"[^a-z0-9 ]+", " ", text.lower().replace("’", "'").replace("'s", "")).split()


def leaves(value, path: str = "") -> list[tuple[str, str]]:
    if isinstance(value, dict):
        return [leaf for key, item in value.items() for leaf in leaves(item, f"{path}/{key}")]
    if isinstance(value, list):
        return [leaf for index, item in enumerate(value) for leaf in leaves(item, f"{path}[{index}]")]
    return [(path, str(value))]


def exposure(gold: str, path: str, text: str) -> str | None:
    """How a finding (with its field name) states the diagnosis; None when it does not."""
    full = " ".join(words(gold))
    match = re.match(r"^(.*?)\s*\((.*?)\)\s*$", gold.strip())
    variants = {full} | ({" ".join(words(match.group(1)))} if match else set())
    haystack = " ".join(words(path.replace("_", " ") + " " + text))
    if any(v and v in haystack for v in variants):
        return "the diagnosis string"
    if match and re.search(r"\b" + re.escape(match.group(2)) + r"\b", path + " " + text):
        return "the abbreviation of the diagnosis"
    content = [t for t in words(re.sub(r"\(.*?\)", "", gold)) if t not in GENERIC and len(t) > 2]
    if len(content) >= 2 and all(t in set(haystack.split()) for t in content):
        return "all content words of the diagnosis"
    return None


def redact(text: str) -> str:
    return re.sub(r"\s+", " ", CLAUSE.sub("", SENTENCE.sub("", text))).strip(" ,;")


def audit_case(case: dict, min_tests: int) -> dict:
    gold = case["gold"] or ""
    private = {name: leaves(case["private_evidence"][name], name) for name in PRIVATE_FIELDS}
    tests = private["Test_Results"]
    real_tests = [(p, t) for p, t in tests if not NOT_PERFORMED.search(t)]
    exam = private["Physical_Examination_Findings"]
    exposed = [(p, t, how) for name in PRIVATE_FIELDS for p, t in private[name] if (how := exposure(gold, p, t))]
    # the Doctor's side: only the diagnosis string or its abbreviation counts; shared symptom words are reported apart
    doctor_hits = [(p, how) for p, t in leaves(case["doctor_view"]) if (how := exposure(gold, "", t))]
    doctor_exposed = any(how != "all content words of the diagnosis" for _, how in doctor_hits)
    basic = eligibility(case)  # no diagnosis, no patient block, no test results, no examination findings

    # redaction: every exposing finding must lose the diagnosis through the fixed rule and stay a finding
    recoverable, transformations = bool(exposed), []
    for path, text, how in exposed:
        cut = redact(text)
        still = exposure(gold, path, cut)
        natural = (len(cut.split()) >= 2 and cut.split()[-1].lower().strip(".") not in DANGLING
                   and not VERB_CUT.search(text))
        in_field_name = exposure(gold, path, "") is not None
        transformations.append({"field": path, "original": text, "after_rule": cut, "exposure_left": bool(still),
                                "still_a_finding": natural, "exposure_in_field_name": in_field_name})
        recoverable = recoverable and not still and natural and not in_field_name and cut != text
    remaining_tests = len([1 for p, t in real_tests if not any(p == q for q, _, _ in exposed)]) + sum(
        1 for tr in transformations if tr["field"].startswith("Test_Results") and tr["still_a_finding"] and not tr["exposure_left"])

    current = basic[:]
    if not current and len(tests) < min_tests:
        current = [f"fewer than {min_tests} test-result values"]
    if not current and diagnosis_named_in_private_evidence(case):
        current = ["diagnosis named in consultant-only evidence"]

    has_private = bool(real_tests) or bool(exam)
    kind, review = None, REVIEWED.get(case["case_id"][-4:])
    if not gold.strip() or not case["doctor_view"].get("Patient_Actor"):
        status = "unusable_incomplete_case"
    elif not has_private:
        status = "unusable_no_private_evidence"
    elif doctor_exposed:
        status = "unusable_leakage"
    elif exposed:
        status = "potentially_recoverable_by_redaction" if recoverable else "unusable_leakage"
    elif len(real_tests) >= min_tests and exam:
        status = "clean_eligible"
    else:
        status, kind = "eligible_low_information", "one_real_test_result" if real_tests else "examination_findings_only"
    if review:
        status = review[0]
    return {
        "case_id": case["case_id"], "correct_diagnosis": gold, "current_rule_eligible": not current,
        "current_exclusion_reason": "; ".join(current) or None,
        "private_test_results": len(tests), "private_real_test_results": len(real_tests),
        "private_examination_findings": len(exam),
        "test_not_performed": [t for _, t in tests if NOT_PERFORMED.search(t)],
        "diagnosis_exposed_in_private_evidence": bool(exposed),
        "exposure": [{"field": p, "how": how, "text": t} for p, t, how in exposed],
        "doctor_side_information_exposes_the_diagnosis": doctor_exposed,
        "doctor_side_shares_all_content_words_with_the_diagnosis": [p for p, _ in doctor_hits],
        "low_information_kind": kind, "manual_review": review[1] if review else None,
        "at_least_one_genuine_agent_only_evidence_item": has_private,
        "redaction": transformations, "real_test_results_left_after_redaction": remaining_tests if exposed else None,
        "recommended_status": status,
    }


def split_sizes(pool: int, test: int = 60) -> list[dict]:
    return [{"TRAIN": pool - test - dev, "DEV": dev, "TEST": test} for dev in (20, 22, 25) if pool - test - dev > 0]


def diagnosis_key(gold: str) -> str:
    return " ".join(words(re.sub(r"\(.*?\)", "", gold)))


def propose_split(pool: list[dict], seed: str, test: int, dev: int, grouped: bool) -> dict:
    """TEST keeps its clean cases; the rest is filled by a hash order. Uses case ids, diagnoses and the status only."""
    low = lambda r: r["recommended_status"] == "eligible_low_information"
    units: dict[str, list[dict]] = {}
    for row in pool:
        units.setdefault(diagnosis_key(row["correct_diagnosis"]) if grouped else row["case_id"], []).append(row)
    assigned, kept = {}, [key for key, rows in units.items() if any(r["current_split"] == "test" for r in rows)]
    for key in kept:
        assigned.update({r["case_id"]: "test" for r in units[key]})
    share = sum(map(low, pool)) / len(pool)
    room = {"test": test - len(assigned), "dev": dev}
    low_room = {"test": round(test * share) - sum(low(r) for r in pool if r["case_id"] in assigned), "dev": round(dev * share)}
    order = sorted((k for k in units if k not in kept), key=lambda k: hashlib.sha256(f"{seed}:{k}".encode()).hexdigest())
    for key in order:
        rows, n_low = units[key], sum(map(low, units[key]))
        target = next((s for s in ("test", "dev") if len(rows) <= room[s] and n_low <= low_room[s]
                       and len(rows) - n_low <= room[s] - low_room[s]), "train")
        if target != "train":
            room[target] -= len(rows)
            low_room[target] -= n_low
        assigned.update({r["case_id"]: target for r in rows})
    by = {s: sorted(c for c, v in assigned.items() if v == s) for s in ("train", "dev", "test")}
    keys = {s: {diagnosis_key(r["correct_diagnosis"]) for r in pool if assigned[r["case_id"]] == s} for s in by}
    return {
        "grouped_by_diagnosis": grouped, "seed": seed, "sizes": {s: len(v) for s, v in by.items()},
        "low_information_cases": {s: sum(low(r) for r in pool if assigned[r["case_id"]] == s) for s in by},
        "TEST_composition": {
            "kept_from_current_TEST": sum(r["current_split"] == "test" and assigned[r["case_id"]] == "test" for r in pool),
            "moved_from_current_TRAIN": sum(r["current_split"] == "train" and assigned[r["case_id"]] == "test" for r in pool),
            "moved_from_current_DEV": sum(r["current_split"] == "dev" and assigned[r["case_id"]] == "test" for r in pool),
            "never_run_before": sum(r["current_split"] is None and assigned[r["case_id"]] == "test" for r in pool)},
        "diagnoses_shared_between_TEST_and_TRAIN_or_DEV": len(keys["test"] & (keys["train"] | keys["dev"])),
        "TEST_cases_whose_diagnosis_is_also_in_TRAIN_or_DEV": sum(
            assigned[r["case_id"]] == "test" and diagnosis_key(r["correct_diagnosis"]) in (keys["train"] | keys["dev"]) for r in pool),
        "case_ids": by,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="the label run's config (dataset path, current rule, current split)")
    args = parser.parse_args()
    config = read_config(args.config)
    source = load_cases(ROOT / config["dataset"]["path"], (config.get("doctor_view") or {}).get("patient_fields"))
    min_tests = config["dataset"]["min_test_findings"]
    rows = [audit_case(case, min_tests) for case in source]
    eligible, reasons = eligible_cases(source, min_tests, bool(config["dataset"].get("exclude_diagnosis_named_in_private_evidence")))
    current_split = split_cases([c["case_id"] for c in eligible], config["split"]["seed"], config["split"]["fractions"])
    if {r["case_id"] for r in rows if r["current_rule_eligible"]} != set(current_split):
        raise SystemExit("the audit's reading of the current rule differs from src/agentclinic/selection.py")
    for row in rows:
        row["current_split"] = current_split.get(row["case_id"])

    def count(status: str, among=rows) -> int:
        return sum(r["recommended_status"] == status for r in among)

    excluded = [r for r in rows if not r["current_rule_eligible"]]
    included = [r for r in rows if r["current_rule_eligible"]]
    strict = count("clean_eligible")
    low_rows = [r for r in rows if r["recommended_status"] == "eligible_low_information"]
    exam_only = sum(r["low_information_kind"] == "examination_findings_only" for r in low_rows)
    low, redaction = len(low_rows) - exam_only, count("potentially_recoverable_by_redaction")
    pool_b = [r for r in rows if r["recommended_status"] in ("clean_eligible", "eligible_low_information")]
    seed = f'{config["split"]["seed"]}:{SPLIT_SEED_SUFFIX}'
    proposals = {f"{'grouped' if g else 'case_level'}_dev{d}": propose_split(pool_b, seed, 60, d, g) for g in (True, False) for d in (22,)}
    summary = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"), "dataset": config["dataset"]["path"],
        "dataset_sha256": hashlib.sha256((ROOT / config["dataset"]["path"]).read_bytes()).hexdigest(),
        "total_raw_cases": len(rows), "current_rule": {"eligible": len(included), "excluded": len(excluded), "reasons": reasons,
                                                       "split": {s: sum(v == s for v in current_split.values()) for s in ("train", "dev", "test")}},
        "status_counts_all_cases": {s: count(s) for s in sorted({r["recommended_status"] for r in rows})},
        "status_counts_of_the_cases_the_current_rule_excludes": {s: count(s, excluded) for s in sorted({r["recommended_status"] for r in excluded})},
        "currently_eligible_cases_with_a_diagnosis_exposure_the_exact_string_rule_missed": [
            {"case_id": r["case_id"], "current_split": r["current_split"], "correct_diagnosis": r["correct_diagnosis"],
             "exposure": r["exposure"], "recommended_status": r["recommended_status"]}
            for r in included if r["diagnosis_exposed_in_private_evidence"]],
        "low_information_kinds": {"one_real_test_result": low, "examination_findings_only": exam_only},
        "doctor_side_shares_all_content_words_with_the_diagnosis": [
            {"case_id": r["case_id"], "current_split": r["current_split"], "correct_diagnosis": r["correct_diagnosis"]}
            for r in rows if r["doctor_side_shares_all_content_words_with_the_diagnosis"]],
        "manual_review": {r["case_id"]: r["manual_review"] for r in rows if r["manual_review"]},
        "pools": {
            "A_strict_clean": strict,
            "B_minus_examination_only": strict + low,
            "B_strict_clean_plus_valid_low_information": strict + low + exam_only,
            "C_B_plus_potentially_recoverable_by_redaction": strict + low + exam_only + redaction,
        },
        "clean_cases_of_the_current_TEST_split": sum(r["current_split"] == "test" and r["recommended_status"] == "clean_eligible" for r in rows),
        "possible_splits_with_TEST_60": {name: split_sizes(n) for name, n in (
            ("A", strict), ("B_minus_examination_only", strict + low), ("B", strict + low + exam_only),
            ("C", strict + low + exam_only + redaction))},
        "proposed_split_not_applied": {name: {k: v for k, v in p.items() if k != "case_ids"} for name, p in proposals.items()},
        "not_used": "no model output, outcome, trajectory length, Effectiveness value or rubric result",
    }
    out = ROOT / "outputs/audit" / f"eligibility_{time.strftime('%Y%m%d-%H%M%S')}"
    out.mkdir(parents=True)
    (out / "all_cases.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
    (out / "excluded_cases.json").write_text(json.dumps(excluded, indent=1, ensure_ascii=False), encoding="utf-8")
    (out / "proposed_split.json").write_text(json.dumps(proposals, indent=1), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    with (out / "excluded_cases.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["case_id", "correct_diagnosis", "current_exclusion_reason", "private_real_test_results", "private_examination_findings",
                         "diagnosis_exposed", "where", "how", "exposing_text", "after_redaction_rule", "doctor_side_exposes",
                         "genuine_agent_only_evidence", "low_information_kind",
                         "real_test_results_left_after_redaction", "manual_review", "recommended_status"])
        for r in excluded:
            writer.writerow([r["case_id"], r["correct_diagnosis"], r["current_exclusion_reason"], r["private_real_test_results"],
                             r["private_examination_findings"], r["diagnosis_exposed_in_private_evidence"],
                             " | ".join(e["field"] for e in r["exposure"]), " | ".join(e["how"] for e in r["exposure"]),
                             " | ".join(e["text"] for e in r["exposure"]), " | ".join(t["after_rule"] for t in r["redaction"]),
                             r["doctor_side_information_exposes_the_diagnosis"], r["at_least_one_genuine_agent_only_evidence_item"],
                             r["low_information_kind"], r["real_test_results_left_after_redaction"], r["manual_review"],
                             r["recommended_status"]])
    print(json.dumps(dict(summary, output_dir=str(out.relative_to(ROOT))), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
