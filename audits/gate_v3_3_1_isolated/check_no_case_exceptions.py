#!/usr/bin/env python3
"""Verify that v3.3.1 runtime contracts contain no frozen clinical cases."""
from __future__ import annotations

import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNTIME = [
    HERE / "compact_quality_gate_v3_3_1.py",
    HERE / "legacy_v33_stage_adapter_v1.py",
    HERE / "prompts/blind_source_v3_3_1.txt",
    HERE / "prompts/applicability_v3_3_1.txt",
    HERE / "contracts/public_evidence_contract_v4.schema.json",
    HERE / "contracts/applicability_contract_v3_3_1.schema.json",
]
FORBIDDEN = [
    r"C3/0\.0", r"C4/0\.3", r"medmcqa_[0-9a-f-]+", r"diaphragmatic",
    r"cataract", r"pyloric", r"aciduria", r"timing compatibility with this case",
]


def main() -> None:
    findings = []
    for path in RUNTIME:
        text = path.read_text()
        for pattern in FORBIDDEN:
            if re.search(pattern, text, re.I):
                findings.append({"path": str(path), "pattern": pattern})
    result = {
        "schema": "case_exception_search_v1",
        "runtime_files": [str(path) for path in RUNTIME],
        "forbidden_patterns": FORBIDDEN,
        "findings": findings,
        "passed": not findings,
        "note": "Case IDs are permitted only in replay/test selection data and are excluded from this runtime search.",
    }
    path = HERE / "reports/case_exception_search.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if findings:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
