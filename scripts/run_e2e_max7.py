"""End-to-end check of the main protocol (max 7 turns, early stopping) on 1, 2 or 5 TRAIN cases, without any weight update."""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from dataclasses import asdict
from pathlib import Path

from scripts.run_e2e_pilot import FIXTURE, information_section, recompute_appropriateness, recorded
from src.agentclinic.adapter import PRIVATE_FIELDS, assert_no_leakage, leaf_values
from src.agentclinic.outcome import normalize
from src.agentclinic.protocol import (A2_RULE, NEUTRAL_AGENT_TASK, NEXT_STEP, ProtocolSpec, agent_prompt, continue_dialogue,
                                      parse_transition)
from src.agentclinic.runtime import build_clinic, frozen_histories, load_policy
from src.agentclinic.trajectory import rerun_from_turn
from src.common import usage
from src.common.config import IntegrationUnavailable
from src.common.io import read_jsonl, write_jsonl
from src.common.main_config import load_main_config, open_run
from src.common.run_dir import write_new
from src.grpo.dataset import grpo_rows
from src.grpo.environment import ClinicEnvironment, turn_samples
from src.grpo.reward import SerlReward
from src.grpo.trainer import build_provider, build_scorer, spec_of, training_stages
from src.training.reward_pipeline import load_reward_spec

ROOT = Path(__file__).resolve().parents[1]
INSTRUCTION = "State the single examination or test finding that bears most on the diagnosis and say what it implies."


def effectiveness_check(config: dict, row: dict, stage: str, parts: dict, used: dict) -> dict:
    """Without a connected predictor: R_E is unavailable, not 0, and the other weights are not rescaled."""
    section, values = config["reward"]["effectiveness"], row["effectiveness"]
    if not (stage in section["stages"] and config["reward"]["weights"]["effectiveness"] > 0):
        return {"effectiveness_unavailable_not_zero": set(values) == {stage} and values[stage].get("available") is False
                and parts["effectiveness"]["value"] is None and "effectiveness" in row["unavailable_components"] and "effectiveness" not in used}
    turn = values.get(stage) or {}
    delta = turn.get("delta")
    return {"effectiveness_is_the_predictor_delta_of_the_trained_turn": set(values) == {stage} and turn.get("available") is True
            and delta is not None and -1 <= delta <= 1 and abs(delta - (turn["after"] - turn["before"])) < 1e-6
            and parts["effectiveness"]["value"] is not None and abs(parts["effectiveness"]["value"] - delta) < 1e-6
            and "effectiveness" in used and "effectiveness" not in (row.get("unavailable_components") or [])}


def select_cases(base: list[dict], n: int) -> list[dict]:
    """The fixed selection rule of the module docstring, on valid TRAIN base collaborations."""
    train = sorted((t for t in base if t["split"] == "train" and t["trajectory_status"] == "valid"), key=lambda t: t["case_id"])
    if not train:
        raise IntegrationUnavailable("no valid TRAIN base collaboration")
    chosen = [(max(train, key=lambda t: (t["num_turns"], [-ord(c) for c in t["case_id"]])), "longest base collaboration")]
    if n >= 2:
        early = next((t for t in train if t["num_turns"] == 1 and t is not chosen[0][0]), None) or next(t for t in train if t is not chosen[0][0])
        chosen.append((early, "first case in id order whose base collaboration stopped after A1"))
    if n >= 5:
        used_lengths = {t["num_turns"] for t, _ in chosen}
        for length in sorted({t["num_turns"] for t in train}, reverse=True):
            if len(chosen) == 5:
                break
            if length not in used_lengths:
                chosen.append((next(t for t in train if t["num_turns"] == length), f"first case in id order with {length} Agent turns"))
                used_lengths.add(length)
        for t in train:  # fewer distinct lengths than cases: fill in id order
            if len(chosen) == 5:
                break
            if all(t is not c for c, _ in chosen):
                chosen.append((t, "next case in id order"))
    return [{"trajectory": t, "reason": why} for t, why in chosen[:n]]


def run(config: dict, n_cases: int, rubric_path: str) -> dict:
    label = {1: "e2e_max7_1case", 2: "e2e_max7_2case", 5: "e2e_max7_5case"}[n_cases]
    config = open_run(config, ROOT, label=label, purpose=f"end-to-end check of the max-7 early-stopping protocol on {n_cases} "
                                                         "TRAIN case(s); no weight update; fixture rubrics; no Effectiveness predictor")
    run_dir = ROOT / config["output_dir"]
    os.environ[usage.ENV] = str(run_dir / "api_usage.jsonl")
    os.environ[usage.STAGE_ENV] = "e2e_max7"
    import torch

    random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    (run_dir / "rubrics").mkdir()
    shutil.copyfile(ROOT / rubric_path, run_dir / "rubrics" / "rubric_set.json")
    rubric_set = json.loads((ROOT / rubric_path).read_text(encoding="utf-8"))

    model, tokenizer = load_policy(config["policy"], None, trainable=False, device=config["devices"]["policy"])
    clinic = build_clinic(config, ROOT, model, tokenizer)
    spec = spec_of(clinic.source)
    if not spec.early_stopping:
        raise IntegrationUnavailable("this check is for the early-stopping protocol; use scripts/run_e2e_pilot.py for the fixed two-turn one")
    prompts: list[dict] = []
    clinic.runtime.doctor = recorded(clinic.runtime.doctor, "doctor", prompts)
    clinic.runtime.agent = recorded(clinic.runtime.agent, "agent", prompts)
    clinic.runtime.agent_greedy = recorded(clinic.runtime.agent_greedy, "agent_greedy (state elicitation)", prompts)
    base = [t for t in read_jsonl(ROOT / config["environment"]["base_trajectories"]) if t["case_id"] in clinic.cases]
    selected = select_cases(base, n_cases)
    trajectories = [s["trajectory"] for s in selected]
    frozen = {h["history_id"]: h for h in frozen_histories(config, ROOT, clinic.cases)}
    histories = [frozen[t["history_id"]] for t in trajectories]
    stages = training_stages(config, spec)
    later = turn_samples(trajectories, clinic.cases, "train", stages)
    a1_states = {t["history_id"]: {k: v for k, v in t["turns"][1]["pre_response_state"].items() if k != "clinical_information_available_to_agent"}
                 for t in trajectories}
    environment = ClinicEnvironment(clinic.cases, histories, clinic.runtime, clinic.judge, a2=later, a1_states=a1_states, spec=spec,
                                    collaboration_states={t["history_id"]: t.get("collaboration_state") for t in trajectories},
                                    max_attempts=config["rollout"]["max_attempts"])
    # the Effectiveness predictor of the run when the config connects one (full-reward check); otherwise R_E is unavailable
    connected = bool(config["reward"]["effectiveness"]["stages"]) and config["reward"]["weights"]["effectiveness"] > 0
    provider = build_provider(config, 0, ROOT) if connected else None
    reward = SerlReward(environment, load_reward_spec(config), rubric_set=rubric_set, scorer=build_scorer(config, ROOT), provider=provider,
                        log_path=run_dir / "rewards.jsonl", iteration=0, scored_stages=tuple(config["reward"]["scored_stages"]),
                        effectiveness_stages=tuple(config["reward"]["effectiveness"]["stages"]), workers=1, generation=clinic.generation,
                        no_applicable_rubric_policy=config["reward"]["no_applicable_rubric_policy"],
                        credit_assignment=config["reward"]["credit_assignment"],
                        rubric_weight_strategy=config["rubric_importance"]["strategy"])
    rows = {r["sample_id"]: r for r in grpo_rows(clinic.cases, histories, clinic.runtime.agent_system, a2=later, stages=stages, spec=spec)}
    sampled = clinic.runtime.agent

    def tag(mark: int, case_id: str, unit: str) -> None:
        for position, entry in enumerate(prompts[mark:]):
            entry.update(case_id=case_id, unit=unit, order=position)

    records, rolled, extras = [], [], []
    for trajectory in trajectories:
        case = clinic.cases[trajectory["case_id"]]
        if case["split"] != "train":
            raise IntegrationUnavailable("the end-to-end check uses TRAIN cases only")
        assert_no_leakage(case)
        sample_ids = [trajectory["history_id"]] + [s["history_id"] for s in later if s["source_history_id"] == trajectory["history_id"]]
        for sample_id in sample_ids:
            with usage.case(case["case_id"]), clinic.generation():
                mark = len(prompts)
                system, user = (m["content"] for m in rows[sample_id]["prompt"])
                completion = sampled(system, user)  # A_t from the GRPO prompt
                prompts[-1]["role"] = "agent (policy sample)"
            pair, = reward.rollout_and_score([sample_id], [completion])
            tag(mark, case["case_id"], sample_id)
            rolled.append(pair)
            records.append({"sample_id": sample_id, "case_id": case["case_id"], "stage": pair[0].stage,
                            "episode": asdict(pair[0]), "rubric_scores": pair[1]})
        # rubric-validation rerun at the last Agent turn the base collaboration reached
        agent_turns = [i for i, t in enumerate(trajectory["turns"]) if t["speaker"] == "agent"]
        with usage.case(case["case_id"]), clinic.generation():
            mark = len(prompts)
            rerun = rerun_from_turn(trajectory, agent_turns[-1], INSTRUCTION, case, clinic.runtime, clinic.judge, spec)
            tag(mark, case["case_id"], f"{trajectory['history_id']}/rerun")
            mark = len(prompts)
            forced = continue_dialogue(case, frozen[trajectory["history_id"]]["history"], clinic.runtime,
                                       spec=ProtocolSpec(max_turns=1, early_stopping=True))
            tag(mark, case["case_id"], f"{trajectory['history_id']}/forced_final")
        extras.append({"case_id": case["case_id"], "history_id": trajectory["history_id"],
                       "rerun": {"turn_index": agent_turns[-1], "stage": trajectory["turns"][agent_turns[-1]]["stage"], **rerun},
                       "forced_final": forced})
    rewards = reward.finish(rolled, [r["sample_id"] for r in records])
    clinic.close()

    write_new(run_dir / "cases.json", json.dumps([{"case_id": s["trajectory"]["case_id"], "history_id": s["trajectory"]["history_id"],
                                                   "split": "train", "why": s["reason"], "base_num_turns": s["trajectory"]["num_turns"],
                                                   "base_stop": s["trajectory"]["stop"],
                                                   "collaboration_state_of_the_base_agent": s["trajectory"]["collaboration_state"]}
                                                  for s in selected], indent=2) + "\n")
    write_jsonl(run_dir / "base_trajectories.jsonl", trajectories)
    write_jsonl(run_dir / "samples.jsonl", [{"history_id": t["history_id"], "case_id": t["case_id"], "stage": "A1",
                                             "history": frozen[t["history_id"]]["history"]} for t in trajectories] + later)
    write_jsonl(run_dir / "prompts.jsonl", prompts)
    write_jsonl(run_dir / "rollouts.jsonl", records)
    write_jsonl(run_dir / "interventions.jsonl", extras)
    log = read_jsonl(run_dir / "rewards.jsonl")
    checks = verify(config, run_dir, log, spec)
    (run_dir / "checks.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    ledger = run_dir / "api_usage.jsonl"
    cost = usage.summarize([ledger], json.loads((ROOT / "configs/pricing.json").read_text(encoding="utf-8"))["prices"]) if ledger.is_file() else None
    (run_dir / "api_cost.json").write_text(json.dumps(cost, indent=2), encoding="utf-8")
    (run_dir / "run_summary.md").write_text(summarize(config, log, extras, checks, cost), encoding="utf-8")
    return {"run_id": config["run_id"], "samples": len(records), "rewards": rewards, "checks_passed": checks["passed"],
            "failed_checks": checks["failed"]}


def verify(config: dict, run: Path, log: list[dict], spec: ProtocolSpec) -> dict:
    """The properties this check exists for, evaluated on what the run actually produced."""
    from src.agentclinic.selection import split_eligible_cases
    from src.common.config import read_config

    prompts = read_jsonl(run / "prompts.jsonl")
    samples = {s["history_id"]: s for s in read_jsonl(run / "samples.jsonl")}
    base = {t["history_id"]: t for t in read_jsonl(run / "base_trajectories.jsonl")}
    extras = read_jsonl(run / "interventions.jsonl")
    selected = json.loads((run / "cases.json").read_text())
    rubric_set = json.loads((run / "rubrics" / "rubric_set.json").read_text(encoding="utf-8"))
    rubric_weight = {r["rubric_id"]: r["weight"] if config["rubric_importance"]["strategy"] in ("failure_frequency", "effectiveness_gap")
                     else 1.0 / len(rubric_set["rubrics"]) for r in rubric_set["rubrics"]}
    tau_rubric, weights = config["rubric_aggregation"]["temperature"], config["reward"]["weights"]
    cases = split_eligible_cases(read_config(ROOT / config["environment"]["config"]), ROOT, allowed_splits=("train", "dev"))
    system = (ROOT / config["resolved_environment"]["prompts"]["consultant"]).read_text(encoding="utf-8")
    order = spec.turn_stages()

    def unit(name: str) -> list[dict]:
        return sorted((p for p in prompts if p.get("unit") == name), key=lambda p: p["order"])

    def doctor_checks(calls: list[dict], case: dict, limit: int) -> dict:
        """What every Doctor call of a continuation must satisfy. limit: the turn after which FINAL is mandatory."""
        gold = normalize(case["gold"])
        patient = "\n".join(leaf_values(case["doctor_view"])).lower()
        private = [leaf.lower() for field in PRIVATE_FIELDS for leaf in leaf_values(case["private_evidence"][field])
                   if len(leaf) > 25 and leaf.lower() not in patient]
        doctor = [p for p in calls if p["role"] == "doctor"]
        decisions = [p for p in doctor if p["user"].endswith(NEXT_STEP)]
        finals = [p for p in doctor if "Give your final diagnosis" in p["user"]]

        def consultant_messages(p: dict) -> int:
            return p["user"].split("Conversation so far:")[1].count("\n\nConsultant: ")

        return {
            "doctor_sees_no_correct_diagnosis_and_no_consultant_only_finding": bool(doctor) and not any(
                (gold and gold in normalize(information_section(p["user"]))) or "consultant only" in information_section(p["user"]).lower()
                or any(f in information_section(p["user"]).lower() for f in private) for p in doctor),
            # the decision prompt is the same text at every turn and says nothing about the turn or how many are left
            "decision_prompt_is_turn_independent": all(not any(w in p["user"].split("Conversation so far:")[1].split("\n\n")[-1].lower()
                                                               for w in ("more time", "at most", "remaining", "turn")) for p in decisions),
            "every_decision_is_valid_json_of_the_schema": all(parse_transition(p["output"]) is not None for p in decisions),
            # a choice is offered after every Agent turn before the limit; the final diagnosis is demanded only after the limit
            "final_is_forced_only_after_the_last_allowed_turn": all(consultant_messages(p) >= limit for p in finals)
            and all(consultant_messages(p) < limit for p in decisions),
        }

    results = {}
    for row in log:
        name, case, sample = row["sample_id"], cases[row["case_id"]], samples[row["sample_id"]]
        calls, stage, dialogue = unit(name), row["stage"], row["dialogue"]
        stages_seen = [m["stage"] for m in dialogue]
        agent_calls = [p for p in calls if p["role"] in ("agent (policy sample)", "agent")]
        state_calls = [p for p in calls if "state elicitation" in p["role"]]
        scores, parts, used = row.get("rubric_scores") or {}, row.get("components") or {}, row.get("weights_used") or {}
        expected_r_a = recompute_appropriateness(scores[stage]["labels"], rubric_weight, tau_rubric)[0] if stage in scores else None
        recomputed = sum(used[k] * parts[k]["value"] for k in used) if row["status"] == "completed" and parts else None
        n_history = len(sample["history"])
        continued = sum(m["speaker"] == "agent" for m in dialogue) - sum(m["speaker"] == "agent" for m in sample["history"]) - 1
        results[name] = {
            "collaboration_completed": row["status"] == "completed",
            "stages_follow_the_protocol_and_end_with_FINAL": stages_seen[:-1] == list(order[: len(stages_seen) - 1])
            and stages_seen[-1] == "FINAL" and len(stages_seen) % 2 == 1 and sum(s[0] == "A" for s in stages_seen) <= spec.max_turns,
            "prompt_history_is_kept_unchanged": [m["text"] for m in dialogue[:n_history]] == [m["text"] for m in sample["history"]]
            and dialogue[n_history]["stage"] == stage and dialogue[n_history].get("fixed") is True,
            "history_is_the_base_collaboration_up_to_the_turn": [m["text"] for m in sample["history"]]
            == [t["text"] for t in base[name.split("/")[0]]["turns"][:n_history]],
            "trained_turn_prompt_is_the_grpo_prompt": bool(agent_calls) and agent_calls[0]["role"] == "agent (policy sample)"
            and agent_calls[0]["system"] == system and agent_calls[0]["user"] == agent_prompt(case, sample["history"], stage, spec=spec),
            # one neutral task at every Agent turn; no rule about keeping or changing a position
            "agent_prompts_are_neutral": all(NEUTRAL_AGENT_TASK in p["user"] and A2_RULE not in p["user"] for p in agent_calls),
            # every Agent turn the continuation wrote was preceded by its own state call (a rerun after a format failure included)
            "one_state_call_before_each_later_agent_turn": len(state_calls) == len(agent_calls) - 1 >= continued,
            "state_output_absent_from_response_and_doctor_prompts": not any(
                out["output"].strip() and out["output"].strip() in p["user"] for out in state_calls for p in agent_calls + [c for c in calls if c["role"] == "doctor"]),
            "state_fields_absent_from_response_prompts": not any(
                marker in p["user"] for p in agent_calls for marker in ("better_supported_position", "evidence_sufficient", "supporting_evidence_summary")),
            **doctor_checks(calls, case, spec.max_turns),
            "stop_recorded": row["stop"] in ("early", "max_turns") and row["num_turns"] == sum(s[0] == "A" for s in stages_seen)
            and (row["stop"] == "max_turns") == (row["num_turns"] == spec.max_turns and dialogue[-1].get("action") != "finalize"),
            # turn-local credit: only the trained turn is scored with an empty rubric set (an iteration whose
            # candidates all failed validation) nothing is scored and Appropriateness is unavailable, not 0
            "only_the_trained_turn_is_scored": (set(scores) == {stage} and row["scored_stages"] == [stage]
                                                and scores[stage]["status"] == "scored" and set(scores[stage]["labels"]) == set(rubric_weight))
            if rubric_weight else (scores == {} and row["scored_stages"] == [stage] and row["raw_R_A"] is None
                                   and "appropriateness" in row["unavailable_components"] and "appropriateness" not in used),
            "scorer_saw_only_turns_up_to_its_target": all(
                [t["turn_id"] for t in s["input"]["dialogue_through_target_turn"]] == list(range(s["input"]["turn_id"] + 1))
                and s["input"]["turn_id"] == n_history for s in scores.values()),
            "correct_diagnosis_in_no_scorer_input": not any(
                normalize(case["gold"]) in normalize(s["input"]["agent_pre_response_state"]["clinical_information_available_to_agent"])
                or "final_diagnosis" in json.dumps(s["input"]) or "ground_truth" in json.dumps(s["input"]) for s in scores.values()),
            "appropriateness_is_the_weighted_softmin_of_the_trained_turn": (expected_r_a is None and row["raw_R_A"] is None)
            or (expected_r_a is not None and row["raw_R_A"] is not None and abs(expected_r_a - row["raw_R_A"]) < 1e-12),
            **effectiveness_check(config, row, stage, parts, used),
            "weights_from_config_without_rescaling": bool(used) and all(abs(used[k] - weights[k]) < 1e-12 for k in used),
            "outcome_binary_from_the_final_diagnosis": row["outcome"] in (0.0, 1.0) and row["final_diagnosis"] == dialogue[-1].get("final_diagnosis"),
            "composite_reward_recomputes": recomputed is not None and abs(recomputed - row["reward"]) < 1e-9
            and abs(row["raw_composite"] - row["reward_passed_to_trl"]) < 1e-12,
        }
    for extra in extras:
        case, trajectory = cases[extra["case_id"]], base[extra["history_id"]]
        rerun, forced = extra["rerun"], extra["forced_final"]
        index = rerun["turn_index"]
        rerun_calls, forced_calls = unit(f"{extra['history_id']}/rerun"), unit(f"{extra['history_id']}/forced_final")
        reached = [t["stage"] for t in trajectory["turns"] if t["speaker"] == "agent"]
        results[f"{extra['history_id']} (case)"] = {
            # one GRPO sample per Agent turn the base collaboration reached, and none for a turn it did not reach
            "one_sample_per_reached_agent_turn": sorted(r["stage"] for r in log if r["sample_id"].split("/")[0] == extra["history_id"]) == reached,
            "base_collaboration_is_valid_and_within_the_limit": trajectory["trajectory_status"] == "valid"
            and trajectory["num_turns"] == len(reached) <= spec.max_turns and trajectory["turns"][-1]["stage"] == "FINAL",
            # an intervention at any turn: the history is kept, the turn is regenerated under the instruction, the collaboration resumes
            "intervention_keeps_the_history_and_resumes": rerun["status"] in ("completed", "outcome_unjudged")
            and [m["text"] for m in rerun["dialogue"][:index]] == [t["text"] for t in trajectory["turns"][:index]]
            and rerun["dialogue"][index]["stage"] == rerun["stage"] and rerun["dialogue"][-1]["stage"] == "FINAL"
            and any(INSTRUCTION in p["user"] for p in rerun_calls if p["role"] == "agent")
            and sum(INSTRUCTION in p["user"] for p in rerun_calls) == 1,
            **{f"intervention_{k}": v for k, v in doctor_checks(rerun_calls, case, spec.max_turns).items()},
            # the turn limit: after the last allowed turn the controller asks for the final diagnosis and offers no choice
            "forced_final_after_the_last_allowed_turn": forced["status"] == "completed" and forced["stop"] == "max_turns"
            and [m["stage"] for m in forced["dialogue"]] == ["D1", "A1", "FINAL"]
            and [("Give your final diagnosis" in p["user"], p["user"].endswith(NEXT_STEP)) for p in forced_calls if p["role"] == "doctor"] == [(True, False)],
        }
    lengths = sorted(t["num_turns"] for t in base.values())
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    overall = {
        "train_cases_only": bool(selected) and all(c["split"] == "train" and cases[c["case_id"]]["split"] == "train" for c in selected),
        "records_complete": len(log) == sum(lengths) and {r["case_id"] for r in log} == {c["case_id"] for c in selected},
        "an_early_stop_path_was_exercised": len(selected) < 2 or min(lengths) == 1,
        "a_multi_turn_path_was_exercised": max(lengths) >= 2 or all(t["num_turns"] == 1 for t in read_jsonl(ROOT / config["environment"]["base_trajectories"])
                                                                     if t["split"] == "train"),
        "samples_of_different_turns_were_exercised": len({r["stage"] for r in log}) == max(lengths),
        "no_weight_update": not any(run.glob("**/*.safetensors")) and not any(run.glob("**/*.pt")),
        "run_files_present": all((run / name).is_file() for name in (
            "resolved_config.json", "environment.json", "manifest.json", "base_trajectories.jsonl", "samples.jsonl", "rewards.jsonl",
            "prompts.jsonl", "rollouts.jsonl", "interventions.jsonl", "rubrics/rubric_set.json", "cases.json")),
        "run_manifest_has_identity": all(manifest.get(k) is not None for k in ("run_id", "config_hash", "seed", "created", "git")),
    }
    failed = [f"{name}: {check}" for name, checks in results.items() for check, ok in checks.items() if not ok] \
        + [check for check, ok in overall.items() if not ok]
    return {"passed": not failed, "failed": failed, "per_unit": results, "run": overall,
            "paths": {"base_lengths": lengths, "trained_stages": sorted({r["stage"] for r in log}),
                      "continuation_lengths": sorted({r["num_turns"] for r in log}),
                      "stop_reasons": sorted({str(r["stop"]) for r in log}),
                      "reached_the_turn_limit": any(r["num_turns"] == spec.max_turns for r in log)},
            "paid_api_calls_recorded": len(read_jsonl(run / "api_usage.jsonl", missing_ok=True))}


def summarize(config: dict, log: list[dict], extras: list[dict], checks: dict, cost) -> str:
    total = sum(len(v) for v in checks["per_unit"].values()) + len(checks["run"])
    lines = [f"# End-to-end check of the max-7 protocol: {config['run_id']}", "",
             f"{len(extras)} TRAIN case(s), {len(log)} GRPO samples. No weight update. Rubrics: fixture set. No Effectiveness predictor.",
             f"Checks: {total} evaluated, passed: {checks['passed']}" + ("" if checks["passed"] else f" (failed: {checks['failed']})"),
             f"Paths: {json.dumps(checks['paths'])}", ""]
    for row in log:
        lines += [f"- {row['sample_id']} (trained turn {row['stage']}): continuation ended after A{row['num_turns']} ({row['stop']}); "
                  f"final {row['final_diagnosis']!r}, outcome {row['outcome']}; R_A {row['raw_R_A']}; reward {row['reward']:.3f}"]
    lines += ["", f"API cost: {cost['total_usd'] if cost else 'n/a'} USD over {cost['calls'] if cost else 0} calls", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--cases", type=int, choices=(1, 2, 5), default=1)
    parser.add_argument("--prerequisite-run", help="2 cases: a passed 1-case run id; 5 cases: a passed 2-case run id")
    parser.add_argument("--rubric-set", default=FIXTURE)
    parser.add_argument("--write-run-id")
    args = parser.parse_args()
    config = load_main_config(args.config, ROOT)
    needed = {2: 1, 5: 2}.get(args.cases)
    if needed:
        if not args.prerequisite_run or Path(args.prerequisite_run).name != args.prerequisite_run:
            raise SystemExit(f"the {args.cases}-case check requires --prerequisite-run (a passed {needed}-case run)")
        prior = ROOT / config["output_root"] / args.prerequisite_run
        if not json.loads((prior / "checks.json").read_text())["passed"] or len(json.loads((prior / "cases.json").read_text())) != needed:
            raise SystemExit(f"the prerequisite must be a passed {needed}-case run")
    result = run(config, args.cases, args.rubric_set)
    if args.write_run_id:
        Path(args.write_run_id).write_text(result["run_id"], encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    if not result["checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
