"""CAUSAL DECISION-ALIGNED RUBRIC diagnostic on MedQA (not the method, not read by any run). Only failure localization,
rubric generation and the score-2 compliance check differ from the original rubric branch; the causal validation (one
frozen score-2 response, 3 reruns, pass at >= 2 exact-option successes) is the original one.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import re
import shutil
from collections import Counter
from pathlib import Path

from scripts.diagnostics.medqa_mechanism_pilot import OptionJudge, classes, frozen_cases, letter_of, patch_views
from src.agentclinic.pipeline import Appender
from src.common.io import read_jsonl

ROOT = Path(__file__).resolve().parents[2]
LOCALIZATION = "prompts/medqa_causal/localization_v1.txt"
GENERATOR = "prompts/medqa_causal/rubric_generator_v1.txt"
BEHAVIORS = ("PREMATURE_AGREEMENT", "UNSUPPORTED_BELIEF_REVISION", "FAILURE_TO_COMPARE_HYPOTHESES", "FAILURE_TO_SURFACE_COUNTEREVIDENCE",
             "FAILURE_TO_CHALLENGE", "FAILURE_TO_REQUEST_DISCRIMINATING_EVIDENCE", "PREMATURE_UNCERTAINTY_BOTH_SIDES_HEDGING", "OTHER")
DECISION_ACTION = {
    "compares": ("Does the reply directly compare the Agent's own current hypothesis with the collaborator's hypothesis against "
                 "specific evidence?", {"yes": "Yes, an explicit comparison of the two hypotheses on evidence.",
                                        "no": "No, it does not compare the two hypotheses on evidence."}),
    "decides": ("Does the reply explicitly state one decision about the shared conclusion: keep its hypothesis, switch to the other, "
                "or declare the disagreement unresolved?", {"yes": "Yes, one of these decisions is stated explicitly.",
                                                            "no": "No explicit decision of this kind is stated."}),
    "commits": ("Does the reply explicitly commit to a primary conclusion, or, if it declares the question unresolved, name the "
                "specific evidence that would resolve it?", {"yes": "Yes.", "no": "No: it hedges, defers or only lists considerations."}),
}
CONTROLS = [  # replies written here, not from any case: (reply, expected answers)
    ("The collaborator's suggestion is reasonable. We should consider both hypotheses and check the remaining details, and keep "
     "an open mind as more information comes in.", {"compares": "no", "decides": "no", "commits": "no"}),
    ("Comparing the two: the finding we both have is explained well by my hypothesis but poorly by yours, and nothing new in your "
     "last message weakens mine. I therefore keep my hypothesis as the primary diagnosis; I would switch only if a result "
     "showed the feature that only yours predicts.", {"compares": "yes", "decides": "yes", "commits": "yes"}),
]
PLACEHOLDER = re.compile(r"HYPOTHESIS_[A-Z0-9]+")


def sha(path: str | Path) -> str:
    return hashlib.sha256(Path(ROOT / path).read_bytes()).hexdigest()


def labels_of(case: dict) -> dict:
    p, c = case["agent_private"][0], case["doctor_private"][0]
    others = [k for k in "ABCD" if k not in (p, c)]
    return {p: "HYPOTHESIS_P", c: "HYPOTHESIS_C", **{k: f"HYPOTHESIS_O{i + 1}" for i, k in enumerate(others)}}


def abstract(text: str, case: dict) -> str:
    """The text with every option (letter references and option texts) replaced by its label."""
    labels = labels_of(case)
    for letter, option in sorted(case["options"].items(), key=lambda kv: -len(kv[1])):
        text = re.sub(re.escape(option), labels[letter], text or "", flags=re.I)
    return re.sub(r"\b([A-D])\s*[\.\)]\s+(?=HYPOTHESIS_)", "", re.sub(r"\(([A-D])\)", lambda m: labels[m.group(1)], text))


def transition(trajectory: dict, case: dict) -> int:
    agent = [(i, letter_of(m.get("assessment") or "", case)) for i, m in enumerate(trajectory["turns"]) if m["speaker"] == "agent"]
    first = next((i for i, letter in agent if letter and letter != case["agent_private"][0]), None)
    return first if first is not None else agent[-1][0]


def first_failed(rows: list[dict], case_id: str) -> dict:
    return sorted((r for r in rows if r["case_id"] == case_id and not r["correct"]), key=lambda r: r["rerun"])[0]["trajectory"]


def clinic_for(args, cases):
    from src.agentclinic.runtime import build_clinic, load_policy

    config = dict(json.loads((ROOT / args.native_run / "resolved_config.json").read_text(encoding="utf-8")), output_dir=args.output_dir)
    patch_views()
    model, tokenizer = load_policy(config["policy"], None, trainable=False, device=config["devices"]["policy"])
    return config, dataclasses.replace(build_clinic(config, ROOT, model, tokenizer), cases=cases, judge=OptionJudge())


def jev(config):
    from src.judges.jev import JevClient

    s = config["rubrics"]["scorer"]
    return JevClient(endpoint=s.get("endpoint"), model=s.get("model_id"), api_key_env=s.get("api_key_env"))


def decision_action(client, reply: str) -> dict:
    answers, _ = client.decide(json.dumps({"reply": reply}, ensure_ascii=False),
                               {k: {"type": "choice", "instructions": q, "criteria": c} for k, (q, c) in DECISION_ACTION.items()},
                               purpose="decision_action_check")
    return {k: a["value"] for k, a in answers.items()}


def branch(args, cases: dict, rows: list[dict], case_ids: list[str], out: Path) -> None:
    """New localization -> new rubric -> new compliance -> the original causal validation, for case_ids."""
    from scripts.diagnostics.decision_rubric_pilot import decide as rubric_checks
    from src.agentclinic.trajectory import intervention_message, rerun_from_turn
    from src.collaboration.rerun import intervention_instruction
    from src.collaboration.runtime import run_with_retries
    from src.grpo.trainer import build_scorer, spec_of
    from src.judges.llm import CachedChat, ChatRequest, build_chat_backend, parse_json_object
    from src.judges.rubric_schema import build_levels, validate_rubric
    from src.judges.rubric_scoring import build_scoring_items, rubric_view
    from src.judges.rubric_validation import GENERATION_FAILURE, validate_candidates

    out.mkdir(parents=True, exist_ok=True)
    config, clinic = clinic_for(args, cases)
    spec, scorer, client = spec_of(clinic.source), build_scorer(config, ROOT), jev(config)
    for reply, expected in CONTROLS:
        got = decision_action(client, reply)
        if got != expected:
            raise SystemExit(f"the decision-action check misjudges its control: {got} != {expected}")
    reader = CachedChat(build_chat_backend(config["roles"]["rubric_manager"]), out / "cache_sol.jsonl")
    local_prompt, gen_prompt = (ROOT / LOCALIZATION).read_text(encoding="utf-8"), (ROOT / GENERATOR).read_text(encoding="utf-8")
    localized = Appender(out / "localization.jsonl", key=lambda r: r["case_id"])
    generated = Appender(out / "rubrics.jsonl", key=lambda r: r["case_id"])
    trajectories, records = {}, []
    for case_id in case_ids:
        case, t = cases[case_id], first_failed(rows, case_id)
        trajectories[t["trajectory_id"]] = t
        turn = transition(t, case)
        labels = labels_of(case)
        view = [{"turn_id": i, "speaker": m["speaker"], "stage": m["stage"], "text": abstract(m["text"], case),
                 "stated": labels.get(letter_of((m.get("assessment") if m["speaker"] == "agent" else
                                                 (m.get("provisional_diagnosis") or m.get("final_diagnosis"))) or "", case) or "", None)}
                for i, m in enumerate(t["turns"])]
        if case_id not in localized.keys:
            answer = reader.complete(ChatRequest(local_prompt, json.dumps({"agent_private": "HYPOTHESIS_P", "collaborator_private": "HYPOTHESIS_C",
                                                                           "dialogue": view, "transition_turn_id": turn}, ensure_ascii=False),
                                                 "medqa-localization-v1", {"temperature": 0, "max_tokens": 1500}))
            parsed, _ = parse_json_object(answer.text) if answer.status == "ok" else (None, None)
            localized.add({"case_id": case_id, "trajectory_id": t["trajectory_id"], "transition_turn_id": turn,
                           "transition_stage": t["turns"][turn]["stage"], **(parsed or {"primary_causal_behavior": None})})
        loc = next(r for r in localized.rows if r["case_id"] == case_id)
        if case_id not in generated.keys:
            answer = reader.complete(ChatRequest(gen_prompt, json.dumps({"dialogue_up_to_transition": view[:turn + 1], "transition_turn_id": turn,
                                                                         "agent_turn_states": loc.get("agent_turn_states"),
                                                                         "primary_causal_behavior": loc.get("primary_causal_behavior"),
                                                                         "abstract_transition": loc.get("abstract_transition")}, ensure_ascii=False),
                                                 "medqa-decision-action-rubric-v1", {"temperature": 0, "max_tokens": 2000}))
            g, _ = parse_json_object(answer.text) if answer.status == "ok" else (None, None)
            g = g or {}
            rubric = {"name": g.get("name"), "applicability": {"user_side_context": g.get("user_side_context"), "agent_side_state": g.get("agent_side_state")},
                      "scores": build_levels(g.get("user_side_context") or "", g.get("agent_side_state") or "",
                                             {lv: g.get(f"score_{lv}") or "" for lv in "012"})}
            problems = validate_rubric(rubric) + [f"missing {k}" for k in ("trigger", "comparison_action", "decision_rule", "explicit_commitment",
                                                                           "revision_condition", "uncertainty_action", "ideal_response_behavior") if not g.get(k)]
            text = json.dumps(g, ensure_ascii=False)
            leaked = [o for o in case["options"].values() if o.lower() in text.lower()] + PLACEHOLDER.findall(text)
            checks = rubric_checks(client, {"name": rubric["name"], "applicability": rubric["applicability"], "score_0": g.get("score_0"),
                                            "score_1": g.get("score_1"), "score_2": g.get("score_2"),
                                            "ideal_response_behavior": g.get("ideal_response_behavior")}) if not problems else {}
            valid = not problems and not leaked and checks["B"]["choice"] == "none" and checks["E"]["choice"] == "compares" \
                and checks["F"]["choice"] == "decides"
            generated.add({"case_id": case_id, "rubric_id": "dar-" + hashlib.sha256(text.encode()).hexdigest()[:12], "generated": g,
                           "candidate_rubric": rubric, "problems": problems, "leaked_option_terms": leaked,
                           "checks": {k: v["choice"] for k, v in checks.items()}, "status": "valid" if valid else "rubric_generation_invalid"})
        gen = next(r for r in generated.rows if r["case_id"] == case_id)
        if gen["status"] == "valid":
            records.append({"rubric_id": gen["rubric_id"], "source_case_id": case_id, "source_trajectory_id": t["trajectory_id"],
                            "analysis_status": "valid", "candidate_rubric": gen["candidate_rubric"],
                            "analysis": {"intervention_turn_id": turn, "ideal_response_behavior": gen["generated"]["ideal_response_behavior"]}})
    frozen = Appender(out / "interventions.jsonl", key=lambda r: r["rubric_id"])

    def intervene(record, trajectory, instruction):
        stored = next((r for r in frozen.rows if r["rubric_id"] == record["rubric_id"]), None)
        if stored:
            return stored
        turn, view_, tried = record["analysis"]["intervention_turn_id"], rubric_view(record), []
        for attempt in range(1, 4):
            message = intervention_message(trajectory, turn, instruction, clinic.cases[trajectory["case_id"]], clinic.runtime, spec)
            row = {"attempt": attempt, "text": message["text"], "assessment": message["assessment"], "label": None, "decision_action": None}
            tried.append(row)
            if not message["text"]:
                continue
            answered = dict(trajectory, turns=trajectory["turns"][:turn] + [dict(trajectory["turns"][turn], text=message["text"])])
            item, = [i for i in build_scoring_items(answered, [view_], include_initial_turn=True) if i.turn_id == turn]
            result = scorer.score(item)
            row.update(label=result.labels.get(record["rubric_id"]), probabilities=result.probabilities.get(record["rubric_id"]))
            if row["label"] == 2:
                row["decision_action"] = decision_action(client, message["text"])
                if all(v == "yes" for v in row["decision_action"].values()):
                    break
        chosen = tried[-1] if tried[-1]["label"] == 2 and all(v == "yes" for v in (tried[-1]["decision_action"] or {"x": "no"}).values()) else None
        outcome = {"rubric_id": record["rubric_id"], "status": "frozen" if chosen else GENERATION_FAILURE, "text": chosen and chosen["text"],
                   "assessment": chosen and chosen["assessment"], "intervention_turn_id": turn, "stage": trajectory["turns"][turn]["stage"],
                   "compliance": {"criterion": "rubric scorer label 2 and compares/decides/commits all yes", "passed": bool(chosen),
                                  "accepted_attempt": chosen and chosen["attempt"]}, "attempts": tried, "instruction": instruction}
        frozen.add(outcome)
        return outcome

    rerun = lambda trajectory, turn, instruction, index, fixed=None: run_with_retries(  # noqa: E731  the original causal rerun
        lambda: rerun_from_turn(trajectory, turn, instruction, clinic.cases[trajectory["case_id"]], clinic.runtime, clinic.judge, spec, fixed=fixed),
        format_attempts=config["validation"].get("max_rerun_attempts", 1))
    print(validate_candidates(records, trajectories, rerun, num_reruns=config["validation"]["num_reruns"],
                              min_successes=config["validation"]["min_successes"], output_path=out / "validations.jsonl", intervene=intervene))


def original(args, cases: dict, rows: list[dict], case_ids: list[str], out: Path) -> None:
    """The original analyzer, candidate and frozen-intervention validation on the same failed collaborations."""
    import src.agentclinic.runtime as runtime
    from scripts import build_rubrics
    from src.judges.agent_state import resolve_include_initial_turn
    from src.judges.failure_analysis import analyze_failures, load_latest
    from src.judges.llm import CachedChat, build_chat_backend
    from src.judges.rubric_validation import validate_candidates

    out.mkdir(parents=True, exist_ok=True)
    config = dict(json.loads((ROOT / args.native_run / "resolved_config.json").read_text(encoding="utf-8")), output_dir=args.output_dir)
    patch_views()
    build = runtime.build_clinic
    runtime.build_clinic = lambda *a, **k: dataclasses.replace(build(*a, **k), cases=cases, judge=OptionJudge())
    trajectories = {t["trajectory_id"]: t for t in (first_failed(rows, k) for k in case_ids)}
    analyzer = config["roles"]["failure_analyzer"]
    prompt = (ROOT / analyzer["prompt"]).read_text(encoding="utf-8") + (ROOT / analyzer["prompt_addendum"]).read_text(encoding="utf-8")
    analyze_failures(list(trajectories.values()), CachedChat(build_chat_backend(analyzer), out / "cache_failure_analyzer.jsonl"), system_prompt=prompt,
                     prompt_version=analyzer["prompt_version"], params=analyzer["params"], output_path=out / "failure_analyses.jsonl",
                     include_initial_turn=resolve_include_initial_turn(config))
    records = [r for r in load_latest(out / "failure_analyses.jsonl").values() if r["analysis_status"] == "valid"]
    rerun, intervene = build_rubrics.clinic_rerun(config, 0, out)[:2]
    print(validate_candidates(records, trajectories, rerun, num_reruns=config["validation"]["num_reruns"],
                              min_successes=config["validation"]["min_successes"], output_path=out / "validations.jsonl", intervene=intervene))


def clear_ids(pilot: Path) -> list[str]:
    return sorted(r["case_id"] for r in read_jsonl(pilot / "attribution_audit.jsonl", missing_ok=True) if r.get("category") == "CLEAR_INTERACTION_ATTRIBUTABLE")


def freeze(args) -> None:
    out = ROOT / args.output_dir / "frozen"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "freeze_manifest.json"
    if path.exists():
        raise SystemExit(f"{path} exists: the method is frozen once")
    for name in (LOCALIZATION, GENERATOR, "scripts/diagnostics/medqa_causal_rubric_pilot.py"):
        shutil.copyfile(ROOT / name, out / Path(name).name)
    path.write_text(json.dumps({"localization_prompt": {"path": LOCALIZATION, "sha256": sha(LOCALIZATION)},
                                "rubric_generator_prompt": {"path": GENERATOR, "sha256": sha(GENERATOR)},
                                "compliance_criterion": {"questions": DECISION_ACTION, "controls": CONTROLS,
                                                         "sha256": hashlib.sha256(json.dumps([DECISION_ACTION, CONTROLS]).encode()).hexdigest()},
                                "code": {"path": "scripts/diagnostics/medqa_causal_rubric_pilot.py", "sha256": sha("scripts/diagnostics/medqa_causal_rubric_pilot.py")},
                                "native_run_config": args.native_run, "note": "frozen after the two development cases; not changed during the unseen evaluation"},
                               indent=1), encoding="utf-8")
    print(path.read_text())


def text_letter(final: str, case: dict) -> str | None:
    """Secondary only: the option whose text the final answer gives after its letter (a Doctor sometimes writes one option's
    letter with another option's text). The outcome stays the letter's exact match."""
    from src.agentclinic.outcome import normalize

    body = re.sub(r"^\s*\(?[A-D]\)?\s*[\.\):]\s*", "", final or "")
    return next((k for k, o in case["options"].items() if normalize(o) == normalize(body)), None)


def report(args) -> None:
    out, unseen = ROOT / args.output_dir, ROOT / args.unseen
    cases = frozen_cases(unseen)
    rows = read_jsonl(unseen / "collaborations.jsonl")
    ids = clear_ids(unseen)
    loc = {r["case_id"]: r for r in read_jsonl(out / "unseen_new/localization.jsonl", missing_ok=True)}
    gen = {r["case_id"]: r for r in read_jsonl(out / "unseen_new/rubrics.jsonl", missing_ok=True)}

    def by_case(path: Path) -> dict:
        result = {}
        for v in read_jsonl(path, missing_ok=True):
            result[v["source_trajectory_id"].split(":")[0]] = v
        return result

    new, old = by_case(out / "unseen_new/validations.jsonl"), by_case(out / "unseen_original/validations.jsonl")
    table = []
    for k in ids:
        n, o = new.get(k, {}), old.get(k, {})
        table.append({"id": k, "transition": (loc.get(k) or {}).get("transition_stage"), "primary_behavior": (loc.get(k) or {}).get("primary_causal_behavior"),
                      "new_rubric_status": (gen.get(k) or {}).get("status"), "new_rubric": ((gen.get(k) or {}).get("generated") or {}).get("name"),
                      "original_feasible": (o.get("intervention") or {}).get("status") == "frozen", "original_success": o.get("successes") if o.get("reruns") else None,
                      "original_status": o.get("validation_status"), "original_turn": (o.get("intervention") or {}).get("stage"),
                      "new_feasible": (n.get("intervention") or {}).get("status") == "frozen", "new_success": n.get("successes") if n.get("reruns") else None,
                      "new_status": n.get("validation_status") or (gen.get(k) or {}).get("status"),
                      "new_final": [x.get("final_diagnosis") for x in n.get("reruns", [])], "original_final": [x.get("final_diagnosis") for x in o.get("reruns", [])],
                      "secondary_letter_text_mismatch": {side: sum(1 for x in v.get("reruns", []) if text_letter(x.get("final_diagnosis"), cases[k])
                                                                   and text_letter(x.get("final_diagnosis"), cases[k]) != letter_of(x.get("final_diagnosis") or ""))
                                                         for side, v in (("new", n), ("original", o))},
                      "secondary_text_correct": {side: sum(1 for x in v.get("reruns", []) if text_letter(x.get("final_diagnosis"), cases[k]) == cases[k]["gold_letter"])
                                                 for side, v in (("new", n), ("original", o))}})
    screening = read_jsonl(unseen / "screening.jsonl")
    cls = classes(rows)
    summary = {"unseen_screening": {"screened": len(screening), "agent_correct": sum(r["agent_correct"] for r in screening),
                                    "candidates": len(cases), "classes": dict(Counter(v["class"] for v in cls.values())), "clear": len(ids)},
               "original_causal_pass": f"{sum(r['original_status'] == 'pass' for r in table)} / {len(ids)}",
               "new_causal_pass": f"{sum(r['new_status'] == 'pass' for r in table)} / {len(ids)}",
               "original_feasible": sum(r["original_feasible"] for r in table), "new_feasible": sum(r["new_feasible"] for r in table),
               "new_rubric_valid": sum(r["new_rubric_status"] == "valid" for r in table), "cases": table}
    (out / "unseen_comparison.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=("dev", "freeze", "new", "original", "report"), required=True)
    parser.add_argument("--pilot", help="dev: the first MedQA pilot directory")
    parser.add_argument("--unseen", help="new / original / report: the unseen MedQA pilot directory")
    parser.add_argument("--native-run", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    out = ROOT / args.output_dir
    if args.stage == "dev":
        pilot = ROOT / args.pilot
        branch(args, frozen_cases(pilot), read_jsonl(pilot / "collaborations.jsonl"), clear_ids(pilot), out / "dev")
    elif args.stage == "freeze":
        freeze(args)
    elif args.stage in ("new", "original"):
        if args.stage == "new" and not (out / "frozen/freeze_manifest.json").exists():
            raise SystemExit("freeze first")
        unseen = ROOT / args.unseen
        (branch if args.stage == "new" else original)(args, frozen_cases(unseen), read_jsonl(unseen / "collaborations.jsonl"), clear_ids(unseen),
                                                      out / ("unseen_new" if args.stage == "new" else "unseen_original"))
    else:
        report(args)


if __name__ == "__main__":
    main()
