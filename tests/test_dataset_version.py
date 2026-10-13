"""The frozen dataset version of the main experiment: eligibility and the case-level split."""

import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest

from scripts.audit_eligibility import audit_case, diagnosis_key, exposure, redact
from scripts.freeze_dataset import allocate, make_split
from src.agentclinic.adapter import assert_no_leakage, load_cases
from src.agentclinic.selection import load_manifest, select_eligible, split_eligible_cases
from src.common.config import read_config

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data/splits/agentclinic_main_v2.json"
DATA = ROOT / "data/raw/agentclinic/agentclinic_medqa_extended.jsonl"
pytestmark = pytest.mark.skipif(not (MANIFEST.is_file() and DATA.is_file()), reason="needs the dataset and its frozen version")


def short(case_id: str) -> str:
    return case_id[-4:]


def test_exposure_and_redaction_rules():
    assert exposure("Hirschsprung disease", "Barium_Enema/Findings", "A transition zone, compatible with Hirschsprung disease") == "the diagnosis string"
    assert exposure("Benign Paroxysmal Positional Vertigo (BPPV)", "x", "suggestive of BPPV") == "the abbreviation of the diagnosis"
    assert exposure("Coarctation of the aorta", "Echo", "Narrowing of the aorta, consistent with coarctation") == "all content words of the diagnosis"
    assert exposure("Varicella", "Varicella_Specific_Tests/IgM", "Positive") == "the diagnosis string"  # in a field name
    assert exposure("Myasthenia gravis", "Blood", "Acetylcholine receptor antibodies elevated") is None
    assert redact("Infiltrates in the right lower lobe consistent with pneumonia") == "Infiltrates in the right lower lobe"
    assert redact("Hyperkeratosis. These findings are consistent with actinic keratosis.") == "Hyperkeratosis."
    assert allocate(27, {"a": 1.5, "b": 0.5, "c": 0.0}) == {"a": 20, "b": 7, "c": 0} and sum(allocate(31, {"a": 3, "b": 4, "c": 5}).values()) == 31


def test_the_frozen_version_is_the_approved_pool_and_split():
    manifest = load_manifest(MANIFEST, DATA)
    cases, out = manifest["cases"], manifest["not_in_main"]
    assert (manifest["raw_cases"], len(cases), len(out)) == (214, 181, 33) and not set(cases) & set(out)
    assert Counter(c["split"] for c in cases.values()) == {"train": 90, "dev": 31, "test": 60}
    assert Counter(c["status"] for c in cases.values()) == {"clean_eligible": 169, "eligible_low_information": 12}
    assert Counter(c["status"] for c in out.values()) == {"potentially_recoverable_by_redaction": 20, "unusable_leakage": 13}
    assert {short(c) for c in out} >= {"0033", "0053", "0207"}  # eligible under the first rule; the same rule now applies to every case
    # the eligible cases of the first TEST split stay in TEST; no case of it moves to TRAIN or DEV
    first_test = [c for c in cases.values() if c["split_in_the_first_label_run"] == "test"]
    assert len(first_test) == 33 and all(c["split"] == "test" for c in first_test)
    # every case has Agent-only evidence; low-information cases are spread over the splits
    assert all(c["private_real_test_results"] + c["private_examination_findings"] >= 1 for c in cases.values())
    assert manifest["split"]["low_information_cases"] == {"train": 6, "dev": 2, "test": 4}
    # case level: a diagnosis may occur on both sides, and the flag for a non-overlap subset is there
    flags = [c["diagnosis_also_in_train_or_dev"] for c in cases.values() if c["split"] == "test"]
    assert all(flag in (True, False) for flag in flags) and 0 < sum(flags) < 60
    seen = {c["diagnosis_key"] for c in cases.values() if c["split"] != "test"}
    assert all((c["diagnosis_key"] in seen) == c["diagnosis_also_in_train_or_dev"] for c in cases.values() if c["split"] == "test")


def test_the_split_is_reproducible_from_the_raw_cases_alone():
    manifest = load_manifest(MANIFEST, DATA)
    first = read_config(ROOT / "configs/effectiveness/v5_agentclinic_full.json")
    earlier = {cid: c["split"] for cid, c in split_eligible_cases(first, ROOT).items()}
    assert Counter(earlier.values()) == {"train": 103, "dev": 34, "test": 35}  # the first run's rule and split are unchanged
    rows = []
    for case in load_cases(DATA):
        row = audit_case(case, 2)
        if row["recommended_status"] in manifest["eligibility"]["statuses_in_main"]:
            rows.append(dict(row, earlier_split=earlier.get(case["case_id"])))
    repeated = {k for k, n in Counter(diagnosis_key(r["correct_diagnosis"]) for r in rows).items() if n > 1}
    for row in rows:
        row["stratum"] = (("low_information" if row["recommended_status"] == "eligible_low_information" else "standard") + "/"
                          + ("repeated_diagnosis" if diagnosis_key(row["correct_diagnosis"]) in repeated else "single_diagnosis"))
    assert make_split(rows, manifest["split"]["seed"], 60, 31) == {cid: c["split"] for cid, c in manifest["cases"].items()}


def test_a_config_with_a_manifest_reads_it_and_an_edited_manifest_is_refused(tmp_path):
    config = read_config(ROOT / "configs/effectiveness/v8_agentclinic_max7.json")
    eligible, splits, reasons = select_eligible(config, ROOT)
    assert len(eligible) == 181 and reasons == {"potentially_recoverable_by_redaction": 20, "unusable_leakage": 13}
    for case in eligible:
        assert_no_leakage(case)  # also the cases without a test result
    assert sum(not c["private_evidence"]["Test_Results"] for c in eligible) == 4
    edited = json.loads(MANIFEST.read_text())
    moved = next(cid for cid, c in edited["cases"].items() if c["split"] == "test")
    edited["cases"][moved]["split"] = "train"
    (tmp_path / "m.json").write_text(json.dumps(edited))
    with pytest.raises(ValueError, match="edited"):
        load_manifest(tmp_path / "m.json", DATA)
    (tmp_path / "other.jsonl").write_text(DATA.read_text() + "\n")
    with pytest.raises(ValueError, match="not the dataset file"):
        load_manifest(MANIFEST, tmp_path / "other.jsonl")
    assert hashlib.sha256(DATA.read_bytes()).hexdigest() == json.loads(MANIFEST.read_text())["dataset_sha256"]
