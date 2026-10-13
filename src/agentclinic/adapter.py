"""AgentClinic-MedQA cases as three views of a collaborative diagnosis."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SOURCE_DATASET = "agentclinic_medqa_extended"
DOCTOR_FIELDS = ("Objective_for_Doctor", "Patient_Actor")
PRIVATE_FIELDS = ("Physical_Examination_Findings", "Test_Results")
GOLD_FIELD = "Correct_Diagnosis"


def load_cases(path: str | Path, doctor_patient_fields: list[str] | None = None,
               *, case_ids: set[str] | None = None) -> list[dict]:
    """[{case_id, source_index, doctor_view, private_evidence, gold}] in file order."""
    cases = []
    lines = [line for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    for index, line in enumerate(lines):
        if case_ids is not None and f"{SOURCE_DATASET}_{index:04d}" not in case_ids:
            continue  # filter before deserializing any held-out case content
        osce = json.loads(line)["OSCE_Examination"]
        cases.append({
            "case_id": f"{SOURCE_DATASET}_{index:04d}", "source_index": index,
            "doctor_view": {name: osce.get(name) for name in DOCTOR_FIELDS},
            "private_evidence": {name: osce.get(name) or {} for name in PRIVATE_FIELDS},
            "gold": osce.get(GOLD_FIELD), "doctor_patient_fields": doctor_patient_fields,
        })
    return cases


def _visible_patient(patient: dict, fields: list[str] | None) -> dict:
    if fields is None:
        return patient
    visible: dict = {}
    for key, value in patient.items():  # source order
        if key in fields:
            visible[key] = value
        elif isinstance(value, dict):
            nested = {k: v for k, v in value.items() if f"{key}.{k}" in fields}
            if nested:
                visible[key] = nested
    return visible


def doctor_view(case: dict) -> dict:
    return {"case_id": case["case_id"], "Objective_for_Doctor": case["doctor_view"]["Objective_for_Doctor"],
            "Patient_Actor": _visible_patient(case["doctor_view"]["Patient_Actor"] or {}, case.get("doctor_patient_fields"))}


def agent_view(case: dict) -> dict:
    return {"case_id": case["case_id"], **case["doctor_view"], **case["private_evidence"]}


def evaluator_view(case: dict) -> dict:
    return {"case_id": case["case_id"], "Correct_Diagnosis": case["gold"]}


def _lines(value: Any, indent: int = 0) -> list[str]:
    pad = "  " * indent
    if isinstance(value, dict):
        out = []
        for key, item in value.items():
            label = str(key).replace("_", " ")
            if isinstance(item, (dict, list)):
                out += [f"{pad}{label}:"] + _lines(item, indent + 1)
            else:
                out.append(f"{pad}{label}: {item}")
        return out
    if isinstance(value, list):
        return [line for item in value for line in (_lines(item, indent) if isinstance(item, (dict, list))
                                                    else [f"{pad}- {item}"])]
    return [f"{pad}{value}"]


def render(view: dict) -> str:
    """Plain-text rendering of a doctor_view or agent_view for a prompt."""
    parts = [f"Objective: {view['Objective_for_Doctor']}", "", "Patient-reported information:"]
    parts += _lines(view["Patient_Actor"], 1)
    for name, title in (("Physical_Examination_Findings", "Physical examination findings (consultant only):"),
                        ("Test_Results", "Test results (consultant only):")):
        if name in view:
            parts += ["", title] + (_lines(view[name], 1) or ["  (none recorded)"])
    return "\n".join(parts)


def render_private(case: dict) -> str:
    """The consultant-only evidence alone (for judges that check a response against it)."""
    return "\n".join(line for name in PRIVATE_FIELDS
                     for line in [name.replace("_", " ") + ":"] + (_lines(case["private_evidence"][name], 1) or ["  (none)"]))


def leaf_values(value: Any) -> list[str]:
    if isinstance(value, dict):
        return [leaf for item in value.values() for leaf in leaf_values(item)]
    if isinstance(value, list):
        return [leaf for item in value for leaf in leaf_values(item)]
    return [str(value)]


def eligibility(case: dict) -> list[str]:
    """Reasons a case cannot be used; empty when it can. Uses the case content only, never a model outcome."""
    reasons = []
    if not (case["gold"] or "").strip():
        reasons.append("no Correct_Diagnosis")
    if not case["doctor_view"].get("Patient_Actor"):
        reasons.append("no Patient_Actor")
    if not case["private_evidence"]["Test_Results"]:
        reasons.append("no Test_Results")
    if not case["private_evidence"]["Physical_Examination_Findings"]:
        reasons.append("no Physical_Examination_Findings")
    return reasons


def assert_no_leakage(case: dict) -> None:
    """The rendered views must respect the information boundaries (also checked in tests on every case)."""
    doctor_text = render(doctor_view(case))
    patient_text = "\n".join(leaf_values(case["doctor_view"])).lower()
    for name in PRIVATE_FIELDS:
        if name in doctor_view(case):
            raise AssertionError(f"{case['case_id']}: {name} in doctor_view")
        for leaf in leaf_values(case["private_evidence"][name]):
            # a finding the patient also reports (e.g. "Normal") is not a leak
            if len(leaf) > 25 and leaf.lower() in doctor_text.lower() and leaf.lower() not in patient_text:
                raise AssertionError(f"{case['case_id']}: private finding in the Doctor's context: {leaf[:60]}")
    if "consultant only" in doctor_text:
        raise AssertionError(f"{case['case_id']}: consultant-only section rendered for the Doctor")
    for view in (doctor_view(case), agent_view(case)):
        if GOLD_FIELD in view or "gold" in view:
            raise AssertionError(f"{case['case_id']}: Correct_Diagnosis in a prompt view")
    agent_leaves = set(leaf_values(agent_view(case)))
    if any(leaf not in agent_leaves for leaf in leaf_values(doctor_view(case))):
        raise AssertionError(f"{case['case_id']}: the Agent must see everything the Doctor sees")


def diagnosis_named_in_private_evidence(case: dict) -> bool:
    """The Correct_Diagnosis string appears in the consultant-only evidence (field names included)."""
    gold = (case["gold"] or "").strip().lower()
    return bool(gold) and gold in render_private(case).lower()


def diagnosis_stated_in_context(case: dict) -> dict[str, bool]:
    """Whether the gold diagnosis string already appears in a view (reported, not an error: a test
    result may name the disease, but the Doctor's starting context should not)."""
    gold = (case["gold"] or "").strip().lower()
    return {"doctor_view": bool(gold) and gold in render(doctor_view(case)).lower(),
            "agent_view": bool(gold) and gold in render(agent_view(case)).lower()}
