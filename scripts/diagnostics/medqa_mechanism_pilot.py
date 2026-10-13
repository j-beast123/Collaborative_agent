"""MedQA clean mechanism pilot (diagnostic; not the method, not read by any run): when the Agent privately commits to the
correct option and the Doctor privately to another, and the collaboration still ends wrong, does the ORIGINAL SERL rubric
branch recover it causally? MedQA TRAIN only (GBaker/MedQA-USMLE-4-options-hf train.json at a pinned revision).
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import random
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from src.agentclinic.outcome import normalize
from src.agentclinic.pipeline import Appender
from src.common import usage
from src.common.io import read_jsonl

ROOT = Path(__file__).resolve().parents[2]
DATA = "data/raw/medqa/train.json"
DATA_REVISION = "GBaker/MedQA-USMLE-4-options-hf@17af9355ef89fba60de966eabaeba797c695f86e"
SEED, CHUNK, TARGET, MAX_SCREEN, REPEATS = 20261003, 40, 25, 400, 3
LETTERS = "ABCD"
DX_QUESTION = re.compile(r"most likely diagnosis\s*\??\s*$", re.I)
TASK = ("Answer the question with one of the options. Whenever you state a diagnosis (a provisional diagnosis, your current "
        "assessment or the final diagnosis), give the option letter and its text, for example 'C. <option text>'.")
COMMIT = ("No one has commented on this case yet. Choose the single best answer. You may reason briefly.\n"
          "End with one line of the form: FINAL ANSWER: <option letter>")
AUDIT_PROMPT = """You audit failed collaborations on one multiple-choice clinical question for a research analysis. Before the
collaboration the Agent privately chose the correct option and the Doctor privately chose another; each started the
collaboration knowing only its own private answer. The final answer (the Doctor's) was wrong. You receive the question, the
options, the correct option, both private answers and the failed collaborations with turn ids and the Agent's stated
assessment per turn.

Use only what is written. Answer over the collaborations together (say "in k of n" where they differ):
- agent_left_correct_answer_at: the turn id where the Agent's stated assessment first left its correct private answer, per
  collaboration (null when it never did)
- stance_shift_right_after_doctor_claim: "yes" / "no" / "some"
- premature_agreement, surfaced_own_evidence, compared_or_checked_doctor_evidence, challenged_disagreement: "yes" / "no" / "some"
- final_answer_matches_doctor_private_answer: "yes" / "no" / "some"
- category: "CLEAR_INTERACTION_ATTRIBUTABLE" (the Agent held the correct answer in the collaboration and an observable
  interaction action lost it: it yielded to the Doctor, did not surface or defend its evidence, or the Doctor overrode a
  correct answer the Agent failed to carry), "INTERACTION_ASSOCIATED_BUT_AMBIGUOUS", or "NOT_INTERACTION_ATTRIBUTABLE" (the
  Agent itself reasoned to another answer in the collaboration without interaction pressure)
- confidence: "high", "medium" or "low"
- reason: two sentences

Return exactly one JSON object and no Markdown with these keys."""


def load_questions() -> tuple[list[dict], dict]:
    rows = [json.loads(line) for line in (ROOT / DATA).read_text(encoding="utf-8").splitlines() if line.strip()]
    kept, excluded, seen = [], Counter(), set()
    for r in rows:
        options = [r.get(f"ending{i}", "") for i in range(4)]
        stem = (r.get("sent1") or "") + (r.get("sent2") or "")
        if not DX_QUESTION.search(stem.strip()):
            excluded["not_a_most_likely_diagnosis_question"] += 1
            continue
        if any(not o.strip() for o in options) or len({normalize(o) for o in options}) < 4 or r.get("label") not in range(4):
            excluded["malformed_options_or_label"] += 1
            continue
        if normalize(stem) in seen:
            excluded["duplicate_stem"] += 1
            continue
        seen.add(normalize(stem))
        kept.append({"case_id": f"medqa_{r['id']}", "source_id": r["id"], "stem": stem.strip(),
                     "options": dict(zip(LETTERS, options)), "gold_letter": LETTERS[r["label"]],
                     "gold": f"{LETTERS[r['label']]}. {options[r['label']]}", "split": "train"})
    return kept, {"source": DATA_REVISION, "file_sha256": hashlib.sha256((ROOT / DATA).read_bytes()).hexdigest(), "rows": len(rows),
                  "eligible": len(kept), "excluded": dict(excluded), "rule": DX_QUESTION.pattern, "seed": SEED}


def view(case: dict, own: str | None) -> dict:
    return {"case_id": case["case_id"], "Question": case["stem"], "Options": {k: v for k, v in case["options"].items()}, "Task": TASK,
            **({"Your_initial_independent_answer": own} if own else {})}


def patch_views() -> None:
    """In this process: the Doctor and the Agent see the MedQA views (each with only its own private answer)."""
    import src.agentclinic.adapter as adapter
    import src.agentclinic.protocol as protocol
    import src.agentclinic.trajectory as trajectory

    doctor = lambda case: view(case, case.get("doctor_private"))  # noqa: E731
    agent = lambda case: view(case, case.get("agent_private"))  # noqa: E731
    original = adapter.render

    def render(v: dict) -> str:
        if "Question" not in v:
            return original(v)
        return "\n".join([f"Question:\n{v['Question']}", "", "Options:", *[f"{k}. {o}" for k, o in v["Options"].items()], "",
                          f"Task: {v['Task']}", *([f"\nYour initial independent answer: {v['Your_initial_independent_answer']}"]
                                                  if v.get("Your_initial_independent_answer") else [])])

    for module in (adapter, protocol, trajectory):
        module.doctor_view, module.agent_view, module.render = doctor, agent, render


def letter_of(text: str, case: dict | None = None) -> str | None:
    text = (text or "").strip()
    found = re.findall(r"FINAL ANSWER:\s*\**\s*\(?([A-D])\b", text, flags=re.I)
    if found:
        return found[-1].upper()
    m = re.match(r"^\(?([A-D])\)?(?:[\.\):]|\s|$)", text)
    if m:
        return m.group(1).upper()
    if case:
        for letter, option in case["options"].items():
            if normalize(option) == normalize(text):
                return letter
    return None


class OptionJudge:
    """Exact match of the official option: reference is 'L. text'; the prediction's letter (leading letter, or a
    'FINAL ANSWER: L' line, or an option text given exactly) must be L. No model is called."""

    def equivalent(self, reference: str, predicted: str) -> dict:
        gold_letter, gold_text = reference[0], reference[3:]
        letter = letter_of(predicted)
        if letter is None and normalize(predicted or "") == normalize(gold_text):
            letter = gold_letter
        return {"equivalent": letter == gold_letter if (predicted or "").strip() else False, "method": "option_exact",
                "reason": f"parsed option {letter}"}


def setup(args, with_model: bool):
    native = ROOT / args.native_run
    config = dict(json.loads((native / "resolved_config.json").read_text(encoding="utf-8")), output_dir=args.output_dir)
    patch_views()
    clinic = None
    if with_model:
        from src.agentclinic.runtime import build_clinic, load_policy

        model, tokenizer = load_policy(config["policy"], None, trainable=False, device=config["devices"]["policy"])
        clinic = dataclasses.replace(build_clinic(config, ROOT, model, tokenizer), judge=OptionJudge())
    return config, clinic


def screen(args) -> None:
    out = ROOT / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    questions, manifest = load_questions()
    random.Random(SEED).shuffle(questions)
    target, max_screen = args.target or TARGET, args.max_screen or MAX_SCREEN
    if args.exclude_from:  # a later screening: the questions an earlier one screened are not used again (same order otherwise)
        used = set(json.loads((ROOT / args.exclude_from / "dataset_manifest.json").read_text())["screened_ids"])
        questions = [q for q in questions if q["source_id"] not in used]
        manifest.update(excluded_screened_by=args.exclude_from, excluded_screened=len(used))
    config, clinic = setup(args, with_model=True)
    import src.agentclinic.adapter as adapter

    render = adapter.render
    rows = Appender(out / "screening.jsonl", key=lambda r: r["case_id"])

    def one(case):
        if case["case_id"] in rows.keys:
            return
        prompt = f"Clinical information available to you:\n{render(view(case, None))}\n\n{COMMIT}"
        with usage.case(case["case_id"]):
            agent = (clinic.runtime.agent_greedy or clinic.runtime.agent)(clinic.runtime.agent_system, prompt)
            doctor = clinic.runtime.doctor(clinic.runtime.doctor_system, prompt)
        a, d = letter_of(agent, case), letter_of(doctor, case)
        rows.add({**{k: case[k] for k in ("case_id", "source_id", "gold_letter")}, "agent_answer": a, "doctor_answer": d,
                  "agent_correct": a == case["gold_letter"], "doctor_correct": d == case["gold_letter"],
                  "agent_raw": agent, "doctor_raw": doctor})

    sample = []
    while len(sample) < min(max_screen, len(questions)):
        chunk = questions[len(sample):len(sample) + CHUNK]
        with clinic.generation(), ThreadPoolExecutor(config["rollout"]["workers"]) as pool:
            list(pool.map(one, chunk))
        sample += chunk
        done = {r["case_id"]: r for r in rows.rows}
        if sum(done[c["case_id"]]["agent_correct"] and done[c["case_id"]]["doctor_answer"] is not None
               and not done[c["case_id"]]["doctor_correct"] for c in sample) >= target:
            break
    manifest.update(screened=len(sample), screened_ids=[q["source_id"] for q in sample], chunk=CHUNK, target=target, max_screen=max_screen)
    (out / "dataset_manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    by_id = {r["case_id"]: r for r in rows.rows}
    frozen = [dict(case, agent_private=f"{by_id[case['case_id']]['agent_answer']}. {case['options'][by_id[case['case_id']]['agent_answer']]}",
                   doctor_private=f"{by_id[case['case_id']]['doctor_answer']}. {case['options'][by_id[case['case_id']]['doctor_answer']]}")
              for case in sample if by_id[case["case_id"]]["agent_correct"] and by_id[case["case_id"]]["doctor_answer"]
              and not by_id[case["case_id"]]["doctor_correct"]]
    path = out / "frozen_candidates.json"
    if path.exists():
        raise SystemExit(f"{path} exists: the candidates are frozen once")
    path.write_text(json.dumps({"frozen_before_collaboration": True, "rule": "A0 correct and D0 parsed and wrong",
                                "candidates": frozen}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(len(sample), Counter((r["agent_correct"], r["doctor_correct"]) for r in rows.rows), "candidates", len(frozen))


def frozen_cases(out: Path) -> dict:
    return {c["case_id"]: c for c in json.loads((out / "frozen_candidates.json").read_text(encoding="utf-8"))["candidates"]}


def collaborate_stage(args) -> None:
    from src.agentclinic.trajectory import collaborate
    from src.grpo.trainer import spec_of

    out = ROOT / args.output_dir
    config, clinic = setup(args, with_model=True)
    cases = frozen_cases(out)
    rows = Appender(out / "collaborations.jsonl", key=lambda r: (r["case_id"], r["rerun"]))

    def one(job):
        case_id, index = job
        if (case_id, index) in rows.keys:
            return
        case = cases[case_id]
        t = collaborate(case, [], clinic.runtime, clinic.judge, spec=spec_of(clinic.source), max_attempts=3,
                        standalone=case["agent_private"], trajectory_id=f"{case_id}:14b:collab:{index}")
        rows.add({"case_id": case_id, "rerun": index, "final_diagnosis": t["final_diagnosis"],
                  "final_letter": letter_of(t["final_diagnosis"] or "", case), "correct": bool(t["is_correct"]),
                  "trajectory_status": t["trajectory_status"], "num_turns": t.get("num_turns"), "trajectory": t})

    with clinic.generation(), ThreadPoolExecutor(config["rollout"]["workers"]) as pool:
        list(pool.map(one, [(k, i) for k in cases for i in range(REPEATS)]))


def classes(rows: list[dict]) -> dict:
    result = {}
    for case_id in sorted({r["case_id"] for r in rows}):
        mine = [r for r in rows if r["case_id"] == case_id]
        success = sum(r["correct"] for r in mine)
        result[case_id] = {"collaborations": len(mine), "success": success,
                           "class": "RESISTS" if success == 3 else "UNSTABLE" if success == 2 else "INTERACTION_INDUCED_FAILURE"}
    return result


def audit(args) -> None:
    from src.judges.llm import CachedChat, ChatRequest, build_chat_backend, parse_json_object

    out = ROOT / args.output_dir
    config, _ = setup(args, with_model=False)
    cases, rows = frozen_cases(out), read_jsonl(out / "collaborations.jsonl")
    reader = CachedChat(build_chat_backend(config["roles"]["rubric_manager"]), out / "cache_reader.jsonl")
    audits = Appender(out / "attribution_audit.jsonl", key=lambda r: r["case_id"])

    def one(case_id):
        if case_id in audits.keys:
            return
        case = cases[case_id]
        failed = [r["trajectory"] for r in rows if r["case_id"] == case_id and not r["correct"]]
        user = json.dumps({"question": case["stem"], "options": case["options"], "correct_option": case["gold"],
                           "agent_private_answer": case["agent_private"], "doctor_private_answer": case["doctor_private"],
                           "failed_collaborations": [{"final_answer": t["final_diagnosis"], "turns": [
                               {"turn_id": i, "speaker": m["speaker"], "stage": m["stage"], "text": m["text"],
                                **({"assessment": m.get("assessment")} if m["speaker"] == "agent" else
                                   {"stated_diagnosis": m.get("provisional_diagnosis") or m.get("final_diagnosis")})}
                               for i, m in enumerate(t["turns"])]} for t in failed]}, ensure_ascii=False)
        answer = reader.complete(ChatRequest(AUDIT_PROMPT, user, "medqa-attribution-v1", {"temperature": 0, "max_tokens": 900}))
        parsed, _ = parse_json_object(answer.text) if answer.status == "ok" else (None, None)
        audits.add({"case_id": case_id, **(parsed or {"category": None})})

    induced = [k for k, v in classes(rows).items() if v["class"] == "INTERACTION_INDUCED_FAILURE"]
    with ThreadPoolExecutor(6) as pool:
        list(pool.map(one, induced))


def serl(args) -> None:
    import src.agentclinic.runtime as runtime
    import src.agentclinic.selection as selection
    from scripts import build_rubrics

    out = ROOT / args.output_dir
    config, _ = setup(args, with_model=False)
    cases, rows = frozen_cases(out), read_jsonl(out / "collaborations.jsonl")
    audits = {r["case_id"]: r for r in read_jsonl(out / "attribution_audit.jsonl", missing_ok=True)}
    clear = [k for k, a in sorted(audits.items()) if a.get("category") == "CLEAR_INTERACTION_ATTRIBUTABLE"]
    run = out / "serl_run"
    rubrics = run / "iter0/rubrics"
    rubrics.mkdir(parents=True, exist_ok=True)
    (run / "base").mkdir(exist_ok=True)
    trajectories = [sorted((r for r in rows if r["case_id"] == k and not r["correct"]), key=lambda r: r["rerun"])[0]["trajectory"] for k in clear]
    (run / "base/trajectories.jsonl").write_text("".join(json.dumps(t, ensure_ascii=False) + "\n" for t in trajectories), encoding="utf-8")
    config = dict(config, output_dir=str(run.relative_to(ROOT)),
                  diagnostic_pilot={"what": "original rubric branch on MedQA clear interaction-attributable failures", "cases": clear})
    (run / "resolved_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=1), encoding="utf-8")
    if not trajectories:
        print("no clear interaction-attributable failure: nothing to run")
        return
    build_clinic, eligible = runtime.build_clinic, selection.split_eligible_cases
    runtime.build_clinic = lambda *a, **k: dataclasses.replace(build_clinic(*a, **k), cases=cases, judge=OptionJudge())
    selection.split_eligible_cases = lambda *a, **k: cases
    build_rubrics.split_filter = lambda config: (None, None)  # every trajectory here is a MedQA TRAIN question
    rerun_fn, intervene_fn = build_rubrics.clinic_rerun(config, 0, rubrics)[:2]
    observe_fn = build_rubrics.gap_observer(config, 0, run / "base/trajectories.jsonl", rubrics)
    print(json.dumps(build_rubrics.run(config, run / "base/trajectories.jsonl", rubrics, 0, rerun_fn=rerun_fn,
                                       intervene_fn=intervene_fn, observe_fn=observe_fn), ensure_ascii=False, indent=1))


def report(args) -> None:
    out = ROOT / args.output_dir
    screening = read_jsonl(out / "screening.jsonl")
    cases = frozen_cases(out)
    rows = read_jsonl(out / "collaborations.jsonl", missing_ok=True)
    cls = classes(rows)
    audits = {r["case_id"]: r for r in read_jsonl(out / "attribution_audit.jsonl", missing_ok=True)}
    validations = {}
    path = out / "serl_run/iter0/rubrics/validations.jsonl"
    if path.exists():
        for line in path.read_text().splitlines():
            v = json.loads(line)
            validations[v["source_trajectory_id"].split(":")[0]] = v
    rubric_set = json.loads((out / "serl_run/iter0/rubrics/rubric_set.json").read_text())["rubrics"] if (out / "serl_run/iter0/rubrics/rubric_set.json").exists() else []
    table = json.loads((out / "serl_run/iter0/rubrics/rubric_table.json").read_text())["rows"] if (out / "serl_run/iter0/rubrics/rubric_table.json").exists() else []
    n = len(screening)
    pct = lambda k, d=n: {"count": k, "percent": round(100 * k / d, 1) if d else None}  # noqa: E731
    ids = lambda pred: sorted(k for k in cls if pred(k))  # noqa: E731
    clear = sorted(k for k, a in audits.items() if a.get("category") == "CLEAR_INTERACTION_ATTRIBUTABLE")
    feasible = sorted(k for k, v in validations.items() if (v.get("intervention") or {}).get("status") == "frozen")
    passed = sorted(k for k, v in validations.items() if v["validation_status"] == "pass")
    funnel = {"screened": pct(n), "agent_standalone_correct": pct(sum(r["agent_correct"] for r in screening)),
              "doctor_standalone_wrong": pct(sum(not r["doctor_correct"] for r in screening)),
              "frozen_disagreement_candidates": dict(pct(len(cases)), ids=sorted(cases)),
              "RESISTS": dict(pct(len(ids(lambda k: cls[k]["class"] == "RESISTS")), len(cases)), ids=ids(lambda k: cls[k]["class"] == "RESISTS")),
              "UNSTABLE": dict(pct(len(ids(lambda k: cls[k]["class"] == "UNSTABLE")), len(cases)), ids=ids(lambda k: cls[k]["class"] == "UNSTABLE")),
              "INTERACTION_INDUCED_FAILURE": dict(pct(len(ids(lambda k: cls[k]["class"] == "INTERACTION_INDUCED_FAILURE")), len(cases)),
                                                  ids=ids(lambda k: cls[k]["class"] == "INTERACTION_INDUCED_FAILURE")),
              "CLEAR_INTERACTION_ATTRIBUTABLE": dict(pct(len(clear), len(cases)), ids=clear),
              "intervention_feasible": dict(pct(len(feasible), len(clear)), ids=feasible),
              "causal_pass": dict(pct(len(passed), len(clear)), ids=passed),
              "positive_effectiveness_gap": sum(1 for r in table if r.get("level") == "category" and (r.get("effectiveness_gap") or 0) > 0),
              "retained_rubrics": len(rubric_set)}
    case_rows = []
    for k in sorted(cases):
        v = validations.get(k, {})
        case_rows.append({"id": k, "gold": cases[k]["gold_letter"], "agent_A0": cases[k]["agent_private"][0], "doctor_D0": cases[k]["doctor_private"][0],
                          "collab_success": cls.get(k, {}).get("success"), "class": cls.get(k, {}).get("class"),
                          "final_letters": [r["final_letter"] for r in rows if r["case_id"] == k],
                          "attribution": (audits.get(k) or {}).get("category"), "attribution_reason": (audits.get(k) or {}).get("reason"),
                          "rubric": v and json.loads(json.dumps(v)).get("rubric_id"),
                          "intervention_scores": [a.get("label") for a in (v.get("intervention") or {}).get("attempts", [])],
                          "causal_success": v.get("successes") if v.get("reruns") else None, "causal_status": v.get("validation_status"),
                          "rerun_final": [x.get("final_diagnosis") for x in v.get("reruns", [])]})
    summary = {"funnel": funnel, "screening": {"screened": n, "agent_accuracy": round(sum(r["agent_correct"] for r in screening) / n, 3),
                                               "doctor_accuracy": round(sum(r["doctor_correct"] for r in screening) / n, 3),
                                               "agent_correct_doctor_wrong": len(cases),
                                               "unparsed": {"agent": sum(r["agent_answer"] is None for r in screening),
                                                            "doctor": sum(r["doctor_answer"] is None for r in screening)}},
               "cases": case_rows, "retained": rubric_set,
               "note": "diagnostic; outcome = official option exact match; original rubric branch unchanged"}
    (out / "final_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "causal_results.json").write_text(json.dumps(validations, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("funnel", "screening")}, ensure_ascii=False, indent=1))
    for c in case_rows:
        print(c["id"], c["gold"], c["agent_A0"], c["doctor_D0"], c["collab_success"], c["class"], c["final_letters"], c["attribution"],
              c["intervention_scores"], c["causal_success"], c["causal_status"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=("screen", "collaborate", "audit", "serl", "report"), required=True)
    parser.add_argument("--native-run", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--exclude-from", help="screen: skip the questions screened in that pilot directory")
    parser.add_argument("--target", type=int, help=f"screen: stop at this many candidates (default {TARGET})")
    parser.add_argument("--max-screen", type=int, help=f"screen: at most this many questions (default {MAX_SCREEN})")
    args = parser.parse_args()
    {"screen": screen, "collaborate": collaborate_stage, "audit": audit, "serl": serl, "report": report}[args.stage](args)


if __name__ == "__main__":
    main()
