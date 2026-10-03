"""Diagnostic pilot (not the method, not read by any run): one decision-focused rubric per audit-recoverable failure, for
condition D of scripts/diagnostics/backbone_capacity_pilot.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.diagnostics.targeted_rubric_pilot import _CLEAN, CONTROLS, PASSING, QUESTIONS, check, rubric_text
from src.common.io import read_jsonl
from src.judges.agent_state import resolve_include_initial_turn
from src.judges.failure_analysis import analyze_failures, load_latest
from src.judges.jev import JevClient
from src.judges.llm import CachedChat, build_chat_backend

ROOT = Path(__file__).resolve().parents[2]
ADDENDUM = "prompts/agentclinic/rubric_decision_focused_addendum.txt"
VERSION = "+decision-focused-v1"
INVALID = "decision_rubric_invalid"
DECISION = {
    "E": ("Does the rubric's score 2 require the Agent to compare the hypotheses under discussion against specific evidence, "
          "saying which hypothesis the evidence supports or weakens?",
          {"compares": "Yes. Score 2 requires an explicit comparison of the hypotheses against the evidence.",
           "no_comparison": "No. Score 2 can be reached without comparing hypotheses against evidence."}),
    "F": ("Does the rubric's score 2 require the Agent to explicitly propose whether the shared diagnosis should be maintained, "
          "challenged or revised?",
          {"decides": "Yes. Score 2 requires an explicit proposal to maintain, challenge or revise the shared diagnosis.",
           "no_decision": "No. Score 2 can be reached without an explicit proposal about the shared diagnosis."}),
}
DECISION_PASSING = {"E": "compares", "F": "decides"}
_DECISION_CONTROL = dict(
    _CLEAN, name="Updating the shared hypothesis with conflicting evidence",
    score_2="The agent states the finding it holds that conflicts with the collaborator's hypothesis, compares how well the "
            "collaborator's hypothesis and the alternative explain that finding, and explicitly proposes to keep or revise the "
            "shared diagnosis, naming the evidence that would settle it if it is missing.",
    ideal_response_behavior="Surface the conflicting finding, compare both hypotheses against it, and propose explicitly whether "
                            "the shared diagnosis should be kept or revised.")
# expected choices of the checker on the controls: the A-D controls of the targeted pilot (none of them is decision-focused)
# and one clean decision-focused rubric
EXPECTED = {**{name: (gold, text, {**({k: PASSING[k] for k in PASSING} if clean else {}), "E": "no_comparison", "F": "no_decision"})
               for name, gold, text, clean in CONTROLS if clean},
            "clean_decision_update": ("Boerhaave syndrome", _DECISION_CONTROL, {**PASSING, **DECISION_PASSING})}


def decide(client: JevClient, text: dict) -> dict:
    answers, _ = client.decide(json.dumps({"rubric": text}, ensure_ascii=False),
                               {k: {"type": "choice", "instructions": q, "criteria": c} for k, (q, c) in {**QUESTIONS, **DECISION}.items()},
                               purpose="decision_rubric_check")
    return {k: {"choice": a["value"], "probabilities": a["probabilities"]} for k, a in answers.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--targeted-pilot", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    pilot, out = ROOT / args.targeted_pilot, ROOT / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    config = dict(json.loads((pilot / "resolved_config.json").read_text(encoding="utf-8")), output_dir=args.output_dir)
    trajectories = {t["trajectory_id"]: t for t in read_jsonl(pilot / "base/trajectories.jsonl")}
    analyzer = config["roles"]["failure_analyzer"]
    prompt = (ROOT / analyzer["prompt"]).read_text(encoding="utf-8")
    if analyzer.get("prompt_addendum"):
        prompt += (ROOT / analyzer["prompt_addendum"]).read_text(encoding="utf-8")
    prompt += (ROOT / ADDENDUM).read_text(encoding="utf-8")
    print("generation", analyze_failures(list(trajectories.values()), CachedChat(build_chat_backend(analyzer), out / "cache_failure_analyzer.jsonl"),
                                         system_prompt=prompt, prompt_version=analyzer["prompt_version"] + VERSION, params=analyzer["params"],
                                         output_path=out / "failure_analyses.jsonl", include_initial_turn=resolve_include_initial_turn(config)))
    from src.grpo.trainer import build_scorer

    scorer = build_scorer(config, ROOT)
    settings = config["rubrics"]["scorer"]
    client = JevClient(endpoint=settings.get("endpoint"), model=settings.get("model_id"), api_key_env=settings.get("api_key_env"))
    controls = []
    for name, (gold, text, expected) in EXPECTED.items():
        answer = decide(client, text)
        controls.append({"control": name, "expected": expected, "answers": answer,
                         "as_expected": all(answer[k]["choice"] == v for k, v in expected.items())})
    (out / "checker_controls.json").write_text(json.dumps(controls, ensure_ascii=False, indent=1), encoding="utf-8")
    if not all(c["as_expected"] for c in controls):
        raise SystemExit(f"the checker misjudges the controls {[c['control'] for c in controls if not c['as_expected']]}")
    rows = []
    for record in load_latest(out / "failure_analyses.jsonl").values():
        trajectory = trajectories[record["source_trajectory_id"]]
        if record["analysis_status"] != "valid":
            rows.append({"case_id": record["source_case_id"], "rubric_id": None, "status": INVALID, "reason": record["validation_problems"]})
            continue
        base = check(record, trajectory, client, scorer)  # A-D, the same function as the targeted pilot
        extra = decide(client, rubric_text(record))
        row = {**base, "E_evidence_hypothesis_comparison": extra["E"]["choice"] == "compares",
               "F_explicit_decision_update": extra["F"]["choice"] == "decides", "decision_probabilities": {k: extra[k] for k in DECISION}}
        row["passed"] = base["passed"] and row["E_evidence_hypothesis_comparison"] and row["F_explicit_decision_update"]
        row["status"] = "valid" if row["passed"] else INVALID
        rows.append(row)
    (out / "decision_rubric_checks.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    for r in rows:
        print(r["case_id"][-4:], r["status"], {k: r.get(k) for k in ("A_interaction_specific", "B_no_diagnostic_knowledge", "C_applicable_in_failed_state",
                                                                       "D_reusable", "E_evidence_hypothesis_comparison", "F_explicit_decision_update")})


if __name__ == "__main__":
    main()
