"""Generate candidate responses per history with a given agent (e.g. Base or SERL)."""

from __future__ import annotations

import argparse
import json
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from src.collaboration.rerun import agent_response_at, intervention_instruction
from src.collaboration.runtime import build_runtime, run_with_retries
from src.common.io import read_jsonl
from src.effectiveness.labels import collect_histories, resolve_labels_section
from src.effectiveness.train import load_predictor_config
from src.evaluation.behavior_judge import BEHAVIOR_CLASSES, BehaviorJudge
from src.judges.llm import openrouter_chat
from src.judges.option_extraction import openrouter_extractor

ROOT = Path(__file__).resolve().parents[1]
STANCES = {
    "challenge": ("Examine the clinician's current position critically. If you think it is wrong or not well "
                  "supported, say so directly, name the specific point you disagree with, and give the evidence from "
                  "the question and established medical knowledge that supports a different option. If after "
                  "examining it you agree, state the strongest counter-argument you considered and why it fails."),
    "accept": ("Take the clinician's current position as your working answer. Acknowledge their reasoning, state "
               "the same option as the clinician, and add points that support it."),
}
ADVOCATE = ("Argue that option {key} ({text}) is the best answer. State option {key} explicitly as your current answer, "
            "give the strongest reasons for it from the question and established medical knowledge, and respond "
            "directly to the clinician's reasoning where it points to a different option.")
OWN = "Your own answer to this question is option {key} ({text}). "
# The controlled behaviors are the behavior classes of the main pipeline (src/evaluation/behavior_judge.py:
# ACCEPT_REVISE, MAINTAIN_CHALLENGE, CONSIDER_REQUEST_EVIDENCE; OTHER is not a behavior to produce), with ...
BEHAVIORS = {
    "evidence_grounded_maintain": (
        OWN + "Take option {key} as your answer. Tell the clinician specifically where you disagree with their "
        "reasoning and give the evidence, from the question and established medical knowledge, that supports option {key}.",
        "MAINTAIN_CHALLENGE", True),
    "unsupported_maintain": (
        OWN + "Reply in at most two sentences. Say only that you keep option {key} as your answer and that you are "
        "not persuaded. Give no reason, no evidence and no explanation of any kind, and do not discuss the "
        "clinician's reasoning or the other options.",
        "MAINTAIN_CHALLENGE", False),
    "accept_revise": (
        "Adopt the clinician's current position as your answer: state the same option as the clinician and say "
        "that you agree with their reasoning.",
        "ACCEPT_REVISE", None),
    "request_evidence": (
        OWN + "Do not commit to an option in this response. Ask the clinician for the specific evidence or reasoning "
        "that would decide between their position and option {key}.",
        "CONSIDER_REQUEST_EVIDENCE", None),
}


assert all(expected in BEHAVIOR_CLASSES for _, expected, _ in BEHAVIORS.values())


def candidate_kinds(config: dict, a: argparse.Namespace) -> dict:
    """The config's candidates section, unless kind flags are given on the command line."""
    flags = {"response_id": a.response_id, "samples": a.samples, "behaviors": a.behaviors,
             "stances": a.stance, "advocate_options": a.advocate_options}
    kinds = {k: v for k, v in flags.items() if v} or dict(config.get("candidates") or {})
    if not any(kinds.get(k) for k in flags):
        raise SystemExit("no candidate kind: set the config's candidates section or pass a kind flag")
    return kinds


def doctor_consistency(screening_row: dict) -> float | None:
    """Uncertainty proxy: share of the doctor's screening samples that favor its most common option."""
    stated = [o for o in screening_row.get("doctor_options", []) if o]
    return Counter(stated).most_common(1)[0][1] / len(stated) if stated else None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--trajectories", required=True)
    p.add_argument("--agent-model", required=True, help="agent model/checkpoint that writes the candidate")
    p.add_argument("--output", required=True)
    p.add_argument("--screening", help="scripts/screen_cases.py output: the agent's standalone answer per case")
    p.add_argument("--parallel", type=int, default=1, help="histories answered concurrently (batched on the GPU)")
    p.add_argument("--only-disagreement", action="store_true",
                   help="keep histories where the agent's standalone answer differs from the clinician's option")
    p.add_argument("--max-histories", type=int, help="after --only-disagreement; e.g. a signal pilot")
    p.add_argument("--response-id")
    p.add_argument("--samples", type=int, default=0)
    p.add_argument("--behaviors", action="store_true")
    p.add_argument("--stance", action="append", choices=sorted(STANCES), default=[])
    p.add_argument("--advocate-options", action="store_true")
    a = p.parse_args()

    config = load_predictor_config(a.config)
    spec = config["continuation_spec"]
    labels = resolve_labels_section(config)
    kinds = candidate_kinds(config, a)
    needs_positions = bool(kinds.get("behaviors") or a.only_disagreement)
    if needs_positions and not a.screening:
        raise SystemExit("behaviors / --only-disagreement need --screening (the agent's standalone answer per case)")

    trajectories = read_jsonl(a.trajectories)
    by_id = {str(t["trajectory_id"]): t for t in trajectories}
    histories = [h for h, _ in collect_histories(
        trajectories, turn_scope=labels.get("turn_scope", "first_collaboration_turn"),
        include_initial_turn=labels["include_initial_turn"], max_histories=labels.get("max_histories"))]
    output = Path(a.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    # Positions before the response: the agent's standalone answer and the clinician's current option.
    screening = {str(r["case_id"]): r for r in read_jsonl(a.screening)} if a.screening else {}
    positions = {}
    if needs_positions:
        extractor = openrouter_extractor(spec.doctor_model.model_id,
                                         output.with_name(output.stem + "_option_cache.jsonl"), ROOT)

        def position(history):
            trajectory = by_id[history["source_trajectory_id"]]
            own = (screening.get(str(trajectory["case_id"])) or {}).get("agent_option") or ""
            return history["history_id"], {"agent_initial_answer": own if own in trajectory["options"] else "",
                                           "doctor_claim": extractor.stated_option(trajectory, history["turns"][-1]["text"]) or ""}

        with ThreadPoolExecutor(8) as pool:
            positions = dict(pool.map(position, histories))

    def disagrees(history) -> bool:
        at = positions.get(history["history_id"], {})
        return bool(at.get("agent_initial_answer") and at.get("doctor_claim")
                    and at["agent_initial_answer"] != at["doctor_claim"])

    if a.only_disagreement:
        histories = [h for h in histories if disagrees(h)]
    histories = histories[: a.max_histories]

    def variants(history) -> list[tuple[str, str, str, str | None]]:
        """(response_id, candidate_type, guideline, intended behavior name) for one history."""
        options = by_id[history["source_trajectory_id"]]["options"]
        found = [(kinds["response_id"], "natural", "", None)] if kinds.get("response_id") else []
        found += [(f"sample_{i}", "natural", "", None) for i in range(1, int(kinds.get("samples") or 0) + 1)]
        found += [(name, "controlled", STANCES[name], None) for name in kinds.get("stances") or []]
        if kinds.get("advocate_options"):
            found += [(f"advocate_{key}", "controlled", ADVOCATE.format(key=key, text=text), None)
                      for key, text in options.items()]
        if kinds.get("behaviors") and disagrees(history):  # the agent has something to maintain or give up
            own = positions[history["history_id"]]["agent_initial_answer"]
            found += [(name, "controlled", guideline.format(key=own, text=options[own]), name)
                      for name, (guideline, _, _) in BEHAVIORS.items()]
        return found

    done = {(r["history_id"], r["response_id"]) for r in read_jsonl(output, missing_ok=True)}
    jobs = [(h, *variant) for h in histories for variant in variants(h) if (h["history_id"], variant[0]) not in done]
    if not jobs:
        print(json.dumps({"histories": len(histories), "candidates_written": 0, "already_done": len(done)}))
        return

    judge = None
    if kinds.get("compliance_judge"):
        settings = kinds["compliance_judge"]
        judge = BehaviorJudge(openrouter_chat(settings["model_id"], output.with_name(output.stem + "_judge_cache.jsonl")),
                              system_prompt=(ROOT / settings["prompt"]).read_text(encoding="utf-8"),
                              prompt_version=settings["prompt_version"], params=settings["params"])
    decoding = spec.decoding or {}
    runtime = build_runtime(agent_model=a.agent_model, doctor_model=spec.doctor_model.model_id,
                            reconstructor_model=spec.reconstructor_model.model_id,
                            max_tokens=(decoding.get("agent") or {}).get("max_new_tokens", 512), root=ROOT,
                            agent_decoding=decoding.get("agent"), doctor_decoding=decoding.get("doctor"),
                            agent_batch_size=a.parallel, doctor_prompt=spec.doctor_prompt)
    lock = threading.Lock()
    counts = Counter()
    attempts = int(labels.get("max_rollout_attempts") or 1)

    def observe(history, intended, response) -> dict:
        """What the response did, by the behavior judge; compliance for controlled behaviors."""
        if judge is None:
            return {"observed_behavior": None, "gives_evidence": None, "compliance": None}
        at = positions.get(history["history_id"], {})
        judgment = judge.classify(
            agent_prior_position=f"option {at['agent_initial_answer']}" if at.get("agent_initial_answer") else "unknown",
            clinician_claim_option=at.get("doctor_claim") or "unknown",
            clinician_message=history["turns"][-1]["text"], response=response)
        observed = judgment.behavior_class if judgment.status == "scored" else None
        compliance = None
        if intended and observed:
            _, expected_class, expected_evidence = BEHAVIORS[intended]
            compliance = observed == expected_class and expected_evidence in (None, judgment.gives_evidence)
        return {"observed_behavior": observed, "gives_evidence": judgment.gives_evidence, "compliance": compliance}

    def run(job):
        history, response_id, candidate_type, guideline, intended = job
        trajectory = by_id[history["source_trajectory_id"]]
        index = int(history["history_id"].rsplit(":turn", 1)[1])
        instruction = intervention_instruction(guideline) if guideline else ""
        result = run_with_retries(
            lambda: agent_response_at(trajectory, index, runtime, instruction=instruction, allow_initial_turn=True),
            format_attempts=attempts)
        if result["status"] != "completed":
            with lock:
                counts["format_failures"] += 1
            return
        at = positions.get(history["history_id"], {})
        truth = trajectory["ground_truth"]["correct_option"]  # analysis metadata only; never in a prompt
        row = {"history_id": history["history_id"], "response_id": response_id, "response": result["response"],
               "source": a.agent_model, "instruction": instruction or None, "candidate_type": candidate_type,
               "intended_behavior": intended,
               "intended_behavior_class": BEHAVIORS[intended][1] if intended else None,
               **observe(history, intended, result["response"]),
               "meta": {**at,
                        "agent_initial_correctness": at["agent_initial_answer"] == truth
                        if at.get("agent_initial_answer") else None,
                        "doctor_initial_correctness": at["doctor_claim"] == truth if at.get("doctor_claim") else None,
                        "doctor_consistency": doctor_consistency(screening.get(str(trajectory["case_id"])) or {})}}
        with lock, output.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            counts["candidates_written"] += 1
            counts[f"compliance_{row['compliance']}"] += row["compliance"] is not None

    with ThreadPoolExecutor(max(1, a.parallel)) as pool:
        list(pool.map(run, jobs))
    print(json.dumps({"histories": len(histories), **counts, "already_done": len(done)}, indent=2))


if __name__ == "__main__":
    main()
