"""Diagnostic of the causal validation's reruns: did the frozen score-2 intervention change anything downstream short of
the final diagnosis? Analysis only; no rule, threshold or selection of the method reads this.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from scripts.diagnostics.failure_audit import names
from src.agentclinic.adapter import agent_view, render
from src.agentclinic.outcome import normalize
from src.agentclinic.pipeline import QUESTION
from src.agentclinic.runtime import build_judge
from src.agentclinic.selection import split_eligible_cases
from src.common.config import read_config
from src.common.io import read_jsonl
from src.judges.llm import CachedChat, ChatRequest, build_chat_backend, parse_json_object
from src.judges.rubric_validation import load_validations

ROOT = Path(__file__).resolve().parents[2]
VERSION = "intervention-local-effect-v1"
PROMPT = """You compare two continuations of one clinical collaboration for a research analysis. A consulting Agent (who alone
has the examination and test findings) talks with a primary Doctor (who has only the history); the Doctor gives the final
diagnosis. Both continuations start from the same dialogue and differ from one Agent reply onward. In both the final
diagnosis was judged not equivalent to the correct diagnosis. You are told the correct diagnosis. You are not told how
either reply was produced; judge only what is written.

"Closer to the correct diagnosis" means: names it, lists it among the possibilities, moves toward the organ system,
mechanism or disease class it belongs to, drops a wrong diagnosis in its favor, or asks for or weighs the finding that
separates it from the diagnosis under discussion. Wording, politeness and length do not count.

Answer:
- doctor_next_message: in which continuation is the Doctor's first message after the Agent's reply closer to the correct
  diagnosis? "X", "Y" or "same"
- final_diagnosis: which final diagnosis is closer to the correct diagnosis? "X", "Y" or "same"
- discussion_after: over everything after the Agent's reply, in which continuation does the discussion come closer to
  the correct diagnosis? "X", "Y" or "same"
- correct_raised_in_x / correct_raised_in_y: is the correct diagnosis (or an equivalent name) raised by the Doctor or
  the Agent anywhere from the Agent's reply onward? true / false
- reason: one sentence.

Return exactly one JSON object and no Markdown:
{"doctor_next_message": "same", "final_diagnosis": "same", "discussion_after": "same", "correct_raised_in_x": false,
 "correct_raised_in_y": false, "reason": "..."}"""


def after(dialogue: list[dict], turn: int) -> list[dict]:
    return [{"speaker": m["speaker"], "stage": m["stage"], "text": m["text"]} for m in dialogue[turn:]]


def doctor_diagnoses(dialogue: list[dict], turn: int) -> list[str]:
    """The Doctor's stated diagnoses after the Agent turn at index turn, in order (provisional ones, then the final)."""
    return [m.get("provisional_diagnosis") or m.get("final_diagnosis") or "" for m in dialogue[turn + 1:] if m["speaker"] == "doctor"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True)
    parser.add_argument("--iteration", type=int, default=0)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--no-predictor", action="store_true")
    parser.add_argument("--any-reruns", action="store_true",
                        help="every validation row with reruns, not only those with a frozen intervention (a diagnostic rollout "
                             "without one, e.g. scripts/diagnostics/persistent_rubric_pilot.py; use with --no-predictor)")
    args = parser.parse_args()
    run, out = ROOT / args.run, ROOT / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    config = json.loads((run / "resolved_config.json").read_text(encoding="utf-8"))
    source = read_config(ROOT / config["environment"]["config"])
    cases = split_eligible_cases(source, ROOT, allowed_splits=("train",))
    rubrics = run / f"iter{args.iteration}" / "rubrics"
    trajectories = {t["trajectory_id"]: t for t in
                    read_jsonl(run / ("base" if args.iteration == 0 else f"iter{args.iteration - 1}") / "trajectories.jsonl")}
    validations = [v for v in load_validations(rubrics / "validations.jsonl").values()
                   if (v["reruns"] if args.any_reruns else (v.get("intervention") or {}).get("status") == "frozen")]
    # the judge is the run's (same prompt, policy and frozen mapping); its call cache of this analysis stays in the output directory
    judge = build_judge(dict(config, output_dir=args.output_dir), ROOT, source)
    chat = CachedChat(build_chat_backend(config["roles"]["rubric_manager"]), out / "cache_reader.jsonl")
    correct = lambda gold, text: bool(text) and bool(judge.equivalent(gold, text)["equivalent"])  # noqa: E731
    provider = None
    if not args.no_predictor:
        from src.effectiveness.service import build_effectiveness
        from src.grpo.trainer import effectiveness_settings

        section = effectiveness_settings(config, args.iteration)
        provider = build_effectiveness(section, 1.0, ROOT)

    def effect(v: dict) -> dict:
        t, turn = trajectories[v["source_trajectory_id"]], v["intervention_turn_id"]
        stage = t["turns"][turn]["stage"]
        history = {"history_id": f"{v['rubric_id']}:local", "intervention_turn": stage, "question": QUESTION, "options": {},
                   "scenario": render(agent_view(cases[t["case_id"]])),
                   "turns": [{"speaker": m["speaker"], "text": m["text"]} for m in t["turns"][:turn]]}
        original = provider.predict(history, t["turns"][turn]["text"], stage, trajectory_id=history["history_id"])
        frozen = provider.predict(history, v["intervention"]["text"], stage, trajectory_id=history["history_id"])
        return {"p_before": original.p_before, "original_delta": original.delta, "frozen_delta": frozen.delta,
                "frozen_minus_original": frozen.delta - original.delta}

    predicted = {v["rubric_id"]: effect(v) for v in validations} if provider else {}  # the predictor is called from one thread

    def compare(job: tuple) -> dict:
        v, index = job
        t, turn, rerun = trajectories[v["source_trajectory_id"]], v["intervention_turn_id"], v["reruns"][index]
        gold = t["ground_truth"]["correct_diagnosis"]
        base, new = doctor_diagnoses(t["turns"], turn), doctor_diagnoses(rerun["dialogue"], turn)
        agent_after = [m.get("assessment") or "" for m in rerun["dialogue"][turn + 1:] if m["speaker"] == "agent"]
        row = {"case_id": t["case_id"], "rubric_id": v["rubric_id"], "failed_turn": t["turns"][turn]["stage"], "rerun": index + 1,
               "correct_diagnosis": gold,
               "original_doctor_next_diagnosis": base[0] if base else None, "rerun_doctor_next_diagnosis": new[0] if new else None,
               "doctor_next_diagnosis_differs_from_original": bool(base and new) and normalize(base[0]) != normalize(new[0]),
               "doctor_next_diagnosis_correct": correct(gold, new[0]) if new else False,
               "any_doctor_diagnosis_correct_after": any(correct(gold, d) for d in set(new)),
               "any_agent_assessment_correct_after": any(correct(gold, a) for a in set(agent_after) if a),
               "original_final_diagnosis": t.get("final_diagnosis"), "rerun_final_diagnosis": rerun.get("final_diagnosis"),
               "final_diagnosis_differs_from_original": normalize(t.get("final_diagnosis") or "") != normalize(rerun.get("final_diagnosis") or ""),
               "correct_named_after_in_original": any(names(gold, m["text"]) for m in t["turns"][turn:]),
               "correct_named_after_in_rerun": any(names(gold, m["text"]) for m in rerun["dialogue"][turn:]),
               "original_agent_turns": sum(m["speaker"] == "agent" for m in t["turns"]),
               "rerun_agent_turns": sum(m["speaker"] == "agent" for m in rerun["dialogue"])}
        # the reader does not know which continuation is the rerun: the order is fixed by a hash of the rerun
        rerun_is_x = int(hashlib.sha256(f"{v['rubric_id']}:{index}".encode()).hexdigest(), 16) % 2 == 0
        branches = {"original": {"continuation": after(t["turns"], turn), "final_diagnosis": t.get("final_diagnosis")},
                    "rerun": {"continuation": after(rerun["dialogue"], turn), "final_diagnosis": rerun.get("final_diagnosis")}}
        user = json.dumps({"correct_diagnosis": gold,
                           "shared_dialogue_before": [{"speaker": m["speaker"], "stage": m["stage"], "text": m["text"]} for m in t["turns"][:turn]],
                           "X": branches["rerun" if rerun_is_x else "original"], "Y": branches["original" if rerun_is_x else "rerun"]},
                          ensure_ascii=False)
        answer = chat.complete(ChatRequest(PROMPT, user, VERSION, {"temperature": 0, "max_tokens": 500}))
        parsed, _ = parse_json_object(answer.text) if answer.status == "ok" else (None, None)
        side = {"X": "rerun" if rerun_is_x else "original", "Y": "original" if rerun_is_x else "rerun", "same": "same"}
        for key in ("doctor_next_message", "final_diagnosis", "discussion_after"):
            row[f"reader_{key}_closer"] = side.get((parsed or {}).get(key))
        row["reader_correct_raised_in_rerun"] = (parsed or {}).get("correct_raised_in_x" if rerun_is_x else "correct_raised_in_y")
        row["reader_correct_raised_in_original"] = (parsed or {}).get("correct_raised_in_y" if rerun_is_x else "correct_raised_in_x")
        row["reader_reason"] = (parsed or {}).get("reason")
        row["reader_status"] = "ok" if parsed else "unread"
        return row

    with ThreadPoolExecutor(8) as pool:
        rows = list(pool.map(compare, [(v, i) for v in validations for i in range(len(v["reruns"]))]))
    readings = ("reader_doctor_next_message_closer", "reader_final_diagnosis_closer", "reader_discussion_after_closer")
    flags = ("doctor_next_diagnosis_differs_from_original", "doctor_next_diagnosis_correct", "any_doctor_diagnosis_correct_after",
             "any_agent_assessment_correct_after", "final_diagnosis_differs_from_original", "correct_named_after_in_rerun",
             "correct_named_after_in_original")

    def tally(group: list[dict]) -> dict:
        return {"reruns": len(group), **{f: sum(bool(r[f]) for r in group) for f in flags},
                **{k: dict(Counter(str(r[k]) for r in group)) for k in readings},
                "reader_correct_raised_in_rerun": sum(bool(r["reader_correct_raised_in_rerun"]) for r in group),
                "reader_correct_raised_in_original": sum(bool(r["reader_correct_raised_in_original"]) for r in group)}

    candidates = []
    for v in validations:
        group = [r for r in rows if r["rubric_id"] == v["rubric_id"]]
        # per candidate, the same 2-of-3 reading as the validation uses for outcomes (a description, not a criterion)
        majority = {k: next((s for s in ("rerun", "original", "same") if sum(r[k] == s for r in group) >= 2), "split") for k in readings}
        candidates.append({"case_id": group[0]["case_id"], "rubric_id": v["rubric_id"], "failed_turn": group[0]["failed_turn"],
                           **tally(group), **{f"majority_{k}": s for k, s in majority.items()}, **predicted.get(v["rubric_id"], {})})
    summary = {"run": args.run, "version": VERSION, "candidates": len(validations), **tally(rows),
               "candidates_majority": {k: dict(Counter(c[f"majority_{k}"] for c in candidates)) for k in readings},
               "reader": config["roles"]["rubric_manager"]["model_id"] + " (analysis only, blind to which continuation is the rerun; not a ground truth)",
               "note": "nothing here is read by the method; the validation criterion (>= 2 of 3 correct final diagnoses) is unchanged"}
    if predicted:
        deltas = [c["frozen_minus_original"] for c in candidates]
        summary["effectiveness_model"] = {"frozen_delta_mean": sum(c["frozen_delta"] for c in candidates) / len(candidates),
                                          "original_delta_mean": sum(c["original_delta"] for c in candidates) / len(candidates),
                                          "frozen_minus_original_mean": sum(deltas) / len(deltas),
                                          "frozen_higher_than_original": sum(d > 0 for d in deltas)}
    (out / "local_effect.json").write_text(json.dumps({"summary": summary, "candidates": candidates, "reruns": rows}, ensure_ascii=False, indent=1),
                                           encoding="utf-8")
    with (out / "local_effect.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    for c in candidates:
        print(c["case_id"][-4:], c["failed_turn"], "| next", c["reader_doctor_next_message_closer"], "| final", c["reader_final_diagnosis_closer"],
              "| after", c["reader_discussion_after_closer"], "| raised rerun/orig", c["reader_correct_raised_in_rerun"], c["reader_correct_raised_in_original"],
              "| R_E orig/frozen", *(round(c[k], 3) for k in ("original_delta", "frozen_delta") if k in c))


if __name__ == "__main__":
    main()
