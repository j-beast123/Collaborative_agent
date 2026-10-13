"""End-to-end check of the main pipeline on one or two TRAIN cases, without any weight update."""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from src.agentclinic.adapter import assert_no_leakage
from src.agentclinic.protocol import elicit_state
from src.agentclinic.runtime import build_clinic, frozen_histories, load_policy, protocol_identity
from src.agentclinic.trajectory import from_prepared, to_trajectory
from src.common import usage
from src.common.config import IntegrationUnavailable, read_config
from src.common.io import read_jsonl, write_jsonl
from src.common.main_config import load_main_config, open_run
from src.common.run_dir import write_new
from src.effectiveness import calibration
from src.effectiveness.service import build_effectiveness
from src.grpo.dataset import grpo_rows
from src.grpo.environment import ClinicEnvironment, Episode, a2_samples
from src.grpo.reward import SerlReward
from src.grpo.trainer import build_scorer, effectiveness_settings, training_stages
from src.training.reward_pipeline import load_reward_spec

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = "configs/pilot/e2e_rubric_fixture.json"


class DeferredEffectiveness:
    """Stands in for the predictor during the rollout stage, which must not call it."""

    def __init__(self, supported_turns: tuple) -> None:
        self.supported_turns = supported_turns

    def identity(self) -> dict:
        return {"source": "deferred to the reward stage"}

    def predict(self, *args, **kwargs):
        raise IntegrationUnavailable("the rollout stage must not call the Effectiveness predictor")

    def close(self) -> None:
        pass


def recorded(function, role: str, log: list, lock=None):
    """Wrap a (system, user) -> text model call so that its prompt, output and latency are kept."""
    def call(system: str, user: str) -> str:
        started = time.perf_counter()
        text = function(system, user)
        log.append({"role": role, "system": system, "user": user, "output": text,
                    "latency_ms": (time.perf_counter() - started) * 1000, "time": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
        return text
    return call


def select_cases(config: dict, clinic, histories: list[dict], n: int) -> list[dict]:
    source = clinic.source
    diagnostics = {r["case_id"]: r for r in read_jsonl(ROOT / source["output_dir"] / "prepare_diagnostics.jsonl")}
    train = sorted((h for h in histories if clinic.cases[h["case_id"]]["split"] == "train"), key=lambda h: h["case_id"])
    if n == 1:
        chosen = [(train[0], "first TRAIN case in id order")]
    else:
        def first(state: str):
            return next(h for h in train if diagnostics[h["case_id"]]["collaboration_state"] == state)
        chosen = [(first("Challenge"), "first TRAIN case with initial Doctor-Agent disagreement (Challenge)"),
                  (first("Preserve"), "first TRAIN case where Doctor and Agent initially agree and are right (Preserve)")]
    return [{"history": h, "reason": why, "collaboration_state_of_the_base_agent": diagnostics[h["case_id"]]["collaboration_state"]}
            for h, why in chosen]


def build_reward(config: dict, environment, rubric_set: dict, provider, run: Path, generation) -> SerlReward:
    return SerlReward(environment, load_reward_spec(config), rubric_set=rubric_set, scorer=build_scorer(config, ROOT),
                      provider=provider, log_path=run / "rewards.jsonl", iteration=0,
                      scored_stages=tuple(config["reward"]["scored_stages"]),
                      effectiveness_stages=tuple(config["reward"]["effectiveness"]["stages"]),
                      workers=1, generation=generation,
                      no_applicable_rubric_policy=config["reward"]["no_applicable_rubric_policy"],
                      credit_assignment=config["reward"]["credit_assignment"],
                      rubric_weight_strategy=config["rubric_importance"]["strategy"])


def rollout(config: dict, n_cases: int, rubric_path: str) -> dict:
    label = "e2e_dryrun" if n_cases == 1 else "e2e_pilot"
    config = open_run(config, ROOT, label=label,
                      purpose=f"end-to-end pipeline check on {n_cases} TRAIN case(s); no weight update; fixture rubrics")
    run = ROOT / config["output_dir"]
    os.environ[usage.ENV] = str(run / "api_usage.jsonl")
    os.environ[usage.STAGE_ENV] = "rollout"
    import torch

    random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    for name in ("rubrics", "grpc_logs", "evaluation", "checkpoints"):
        (run / name).mkdir()
    write_new(run / "checkpoints" / "README.txt", "No weight update in this run: the policy is the base Agent "
              f"({config['policy']['model_id']} @ {config['policy']['revision']}) without an adapter.\n")
    rubric_set = json.loads((ROOT / rubric_path).read_text(encoding="utf-8"))
    shutil.copyfile(ROOT / rubric_path, run / "rubrics" / "rubric_set.json")

    concurrent = config["devices"]["placement"] == "concurrent"
    if concurrent:  # the service is up: it must agree with the offline evaluation before anything is rolled out
        parity_report = parity(config, config.get("pilot", {}).get("parity_histories", 6))
        if not parity_report["passed"]:
            raise IntegrationUnavailable("offline/gRPC parity failed; rollout was not started")
    model, tokenizer = load_policy(config["policy"], None, trainable=False, device=config["devices"]["policy"])
    clinic = build_clinic(config, ROOT, model, tokenizer)
    prompts: list[dict] = []
    clinic.runtime.doctor = recorded(clinic.runtime.doctor, "doctor", prompts)
    clinic.runtime.agent = recorded(clinic.runtime.agent, "agent", prompts)
    clinic.runtime.agent_greedy = recorded(clinic.runtime.agent_greedy, "agent_greedy (state elicitation)", prompts)
    histories = frozen_histories(config, ROOT, clinic.cases)
    selected = select_cases(config, clinic, histories, n_cases)
    chosen = [s["history"] for s in selected]
    stages, identity = training_stages(config), protocol_identity(config, ROOT)
    second = []
    if "A2" in stages:  # the base Agent's frozen collaborations with their elicited states: the A2 samples of iteration 0
        base = []
        for item in selected:
            history, mark = item["history"], len(prompts)
            with usage.case(history["case_id"]), clinic.generation():
                base.append(from_prepared(clinic.cases[history["case_id"]], history, clinic.runtime, clinic.judge,
                                          iteration=0, checkpoint_id="base", protocol_identity=identity))
            for position, entry in enumerate(prompts[mark:]):
                entry.update(case_id=history["case_id"], sample_id=f"{history['history_id']}/base", order=position)
        second = a2_samples(base, clinic.cases, "train")
        write_jsonl(run / "base_trajectories.jsonl", base)
        write_jsonl(run / "a2_samples.jsonl", second)
    environment = ClinicEnvironment(clinic.cases, chosen, clinic.runtime, clinic.judge, a2=second,
                                    collaboration_states={s["history"]["history_id"]: s["collaboration_state_of_the_base_agent"]
                                                          for s in selected},
                                    max_attempts=config["rollout"]["max_attempts"])
    provider = build_effectiveness(effectiveness_settings(config, 0), config["reward"]["weights"]["effectiveness"], ROOT) \
        if concurrent else DeferredEffectiveness(tuple(config["reward"]["effectiveness"]["stages"]))
    reward = build_reward(config, environment, rubric_set, provider, run, clinic.generation)
    rows = {r["sample_id"]: r for r in grpo_rows(clinic.cases, chosen, clinic.runtime.agent_system, a2=second, stages=stages)}
    sampled = clinic.runtime.agent  # the policy with the configured sampling, as the label pipeline's Agent

    records, trajectories, states = [], [], []
    for item in selected:
        history = item["history"]
        case = clinic.cases[history["case_id"]]
        sample_id = history["history_id"]
        if case["split"] != "train":
            raise IntegrationUnavailable("the end-to-end check uses TRAIN cases only")
        assert_no_leakage(case)
        with usage.case(case["case_id"]), clinic.generation():
            mark = len(prompts)
            state_a1 = elicit_state(case, history["history"], clinic.runtime)  # before A1; A1's prompt never sees it
            environment.a1_states[sample_id] = state_a1
            system, user = (m["content"] for m in rows[sample_id]["prompt"])
            completion = sampled(system, user)  # A1 from the GRPO prompt
            prompts[-1]["role"] = "agent (A1, policy sample)"
        (episode, scores), = reward.rollout_and_score([sample_id], [completion])
        for position, entry in enumerate(prompts[mark:]):
            entry.update(case_id=case["case_id"], sample_id=sample_id, order=position)
        records.append({"sample_id": sample_id, "episode": asdict(episode), "rubric_scores": scores})
        if episode.status == "completed":
            standalone = history.get("agent_standalone_diagnosis")
            trajectories.append(to_trajectory(case, episode.dialogue, trajectory_id=f"{sample_id}:{episode.rollout_id}",
                                              standalone=standalone, states=episode.states, judge=clinic.judge,
                                              history_id=sample_id, iteration=0, checkpoint_id="base",
                                              protocol_identity=identity))
        states += [{"case_id": case["case_id"], "sample_id": sample_id, "before": stage, "state": state}
                   for stage, state in (episode.states or {"A1": state_a1}).items()]
    for sample in second:  # A2 samples: the policy writes A2 to D1 -> A1 -> D2, the Doctor's FINAL follows
        case, sample_id = clinic.cases[sample["case_id"]], sample["history_id"]
        assert_no_leakage(case)
        with usage.case(case["case_id"]), clinic.generation():
            mark = len(prompts)
            system, user = (m["content"] for m in rows[sample_id]["prompt"])
            completion = sampled(system, user)  # A2 from the GRPO prompt
            prompts[-1]["role"] = "agent (A2, policy sample)"
        (episode, scores), = reward.rollout_and_score([sample_id], [completion])
        for position, entry in enumerate(prompts[mark:]):
            entry.update(case_id=case["case_id"], sample_id=sample_id, order=position)
        records.append({"sample_id": sample_id, "episode": asdict(episode), "rubric_scores": scores})
        states += [{"case_id": case["case_id"], "sample_id": sample_id, "before": stage, "state": state}
                   for stage, state in episode.states.items()]
    clinic.close()
    provider.close()

    write_new(run / "cases.json", json.dumps([{"case_id": s["history"]["case_id"], "history_id": s["history"]["history_id"],
                                               "split": "train", "why": s["reason"],
                                               "collaboration_state_of_the_base_agent": s["collaboration_state_of_the_base_agent"]}
                                              for s in selected], indent=2) + "\n")
    write_jsonl(run / "prompts.jsonl", prompts)
    write_jsonl(run / "rollouts.jsonl", records)
    write_jsonl(run / "trajectories.jsonl", trajectories)
    write_jsonl(run / "states.jsonl", states)
    write_jsonl(run / "rubric_scores.jsonl", [{"sample_id": r["sample_id"], **r["rubric_scores"]} for r in records])
    return {"run_id": config["run_id"], "run_dir": config["output_dir"], "cases": [r["sample_id"] for r in records],
            "statuses": [r["episode"]["status"] for r in records]}


def parity(config: dict, n_histories: int) -> dict:
    """The served predictor against the offline evaluation on DEV label inputs."""
    run = ROOT / config["output_dir"]
    section = effectiveness_settings(config, 0)
    client = build_effectiveness(section, 1.0, ROOT)
    labels_dir = ROOT / read_config(ROOT / config["environment"]["config"])["output_dir"]
    offline = json.loads((labels_dir / "model_lora/dev_logits.json").read_text(encoding="utf-8"))
    stored = {(e["kind"], e["history_id"], e["response_id"]): z
              for e, z in zip(offline["examples"], offline["checkpoints"]["epoch3"]["logits"])}
    parameters = calibration.load(ROOT / section["calibration"])
    rows, seen = [], []
    for label in read_jsonl(labels_dir / "labels_dev.jsonl"):
        if label["history_id"] not in seen:
            if len(seen) == n_histories:
                break
            seen.append(label["history_id"])
        prediction = client.predict(label["history"], label["response"], "A1", trajectory_id=f"parity:{label['history_id']}")
        z_before, z_after = stored[("before", label["history_id"], None)], stored[("after", label["history_id"], label["response_id"])]
        p_before, p_after = calibration.apply_logits(parameters, [z_before, z_after])
        rows.append({"history_id": label["history_id"], "response_id": label["response_id"],
                     "service": {"logit_before": prediction.logit_before, "logit_after": prediction.logit_after,
                                 "p_before": prediction.p_before, "p_after": prediction.p_after, "delta": prediction.delta},
                     "offline": {"logit_before": z_before, "logit_after": z_after, "p_before": p_before, "p_after": p_after,
                                 "delta": p_after - p_before},
                     "self_consistency": max(abs(prediction.p_before - calibration.apply_logits(parameters, [prediction.logit_before])[0]),
                                             abs(prediction.p_after - calibration.apply_logits(parameters, [prediction.logit_after])[0]))})
    def worst(name: str) -> float:
        return max(abs(r["service"][name] - r["offline"][name]) for r in rows)

    if not rows:
        raise IntegrationUnavailable("no DEV examples available for parity")
    tolerance = config["reward"]["effectiveness"]["parity_tolerance"]
    report = {"service": client.identity(), "pairs": len(rows), "dev_histories": len(seen),
              "max_abs_diff": {name: worst(name) for name in ("logit_before", "logit_after", "p_before", "p_after", "delta")},
              "max_abs_diff_between_returned_probability_and_sigmoid_of_returned_logit_over_T": max(r["self_consistency"] for r in rows),
              "calibration": parameters, "probability_tolerance": tolerance, "rows": rows}
    report["passed"] = bool(worst("p_before") <= tolerance and worst("p_after") <= tolerance and worst("delta") <= tolerance
                            and max(worst("logit_before"), worst("logit_after")) <= config["reward"]["effectiveness"]["parity_logit_tolerance"]
                            and report["max_abs_diff_between_returned_probability_and_sigmoid_of_returned_logit_over_T"] < 1e-9)
    write_new(run / "evaluation" / "effectiveness_parity.json", json.dumps(report, indent=2))
    client.close()
    return {k: v for k, v in report.items() if k != "rows"}


def finish(config: dict) -> dict:
    run = ROOT / config["output_dir"]
    os.environ[usage.ENV] = str(run / "api_usage.jsonl")
    os.environ[usage.STAGE_ENV] = "reward"
    section = effectiveness_settings(config, 0)
    provider = build_effectiveness(section, config["reward"]["weights"]["effectiveness"], ROOT)
    rubric_set = json.loads((run / "rubrics" / "rubric_set.json").read_text(encoding="utf-8"))
    if (run / "rewards.jsonl").exists():
        raise IntegrationUnavailable("reward stage has already written results; use a new run")
    records = read_jsonl(run / "rollouts.jsonl")
    source = read_config(ROOT / config["environment"]["config"])
    from src.agentclinic.selection import split_eligible_cases

    cases = split_eligible_cases(source, ROOT)
    histories = [h for h in frozen_histories(config, ROOT, cases) if h["history_id"] in {r["sample_id"] for r in records}]
    selected = {c["history_id"]: c["collaboration_state_of_the_base_agent"] for c in json.loads((run / "cases.json").read_text())}
    environment = ClinicEnvironment(cases, histories, None, None, a2=read_jsonl(run / "a2_samples.jsonl", missing_ok=True),
                                    collaboration_states=selected)  # no model call in this stage
    from contextlib import nullcontext

    reward = build_reward(config, environment, rubric_set, provider, run, nullcontext)
    rolled = [(Episode(**r["episode"]), r["rubric_scores"]) for r in records]
    rewards = reward.finish(rolled, [r["sample_id"] for r in records])
    log = read_jsonl(run / "rewards.jsonl")[-len(records):]
    write_jsonl(run / "effectiveness.jsonl", [{"sample_id": row["sample_id"], "case_id": row["case_id"],
                                               "per_turn": row["effectiveness"], "predictor": provider.identity() if provider else None}
                                              for row in log])
    (run / "evaluation" / "outcomes.json").write_text(json.dumps(
        [{"sample_id": row["sample_id"], "final_diagnosis": row["final_diagnosis"], "outcome": row["outcome"],
          "outcome_method": row["outcome_method"]} for row in log], indent=2), encoding="utf-8")

    # the service writes its request log into this run (--log-dir <run>/grpc_logs); a service started before the
    # run existed logs elsewhere (pilot.service_request_log) and its rows of this run are copied in
    request_log = (config.get("pilot") or {}).get("service_request_log")
    if request_log and not (run / "grpc_logs" / "effectiveness_requests.jsonl").exists():
        ids = {r["effectiveness"]["A1"]["request_id"] for r in log if "A1" in (r["effectiveness"] or {})}
        write_jsonl(run / "grpc_logs" / "effectiveness_requests.jsonl",
                    [r for r in read_jsonl(Path(request_log)) if r.get("request_id") in ids])
    provider.close()
    checks = verify(config, run, log, records)
    (run / "checks.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    ledger = run / "api_usage.jsonl"
    cost = usage.summarize([ledger], json.loads((ROOT / "configs/pricing.json").read_text(encoding="utf-8"))["prices"]) if ledger.is_file() else None
    (run / "api_cost.json").write_text(json.dumps(cost, indent=2), encoding="utf-8")
    summary = summarize(config, run, log, checks, cost)
    (run / "run_summary.md").write_text(summary, encoding="utf-8")
    return {"run_id": config["run_id"], "rewards": rewards, "checks_passed": checks["passed"], "failed_checks": checks["failed"]}


def information_section(prompt: str) -> str:
    """The part of a prompt that the pipeline supplies as the role's own information (before the conversation).
    The conversation itself is not checked: what a speaker wrote may name any diagnosis or finding."""
    return prompt.split("Conversation so far:")[0]


def recompute_appropriateness(labels: dict, weights: dict, tau: float) -> tuple:
    """(R_A, applicable rubric ids) of one response from the scorer's labels and the rubric weights, independently of
    src/judges/reward.py: the weighted soft-min -tau * log(sum_k alpha_k * exp(-s_k / tau)) over the applicable ...
    """
    import math

    applicable = {rid: {0: 0.0, 1: 0.5, 2: 1.0}[label] for rid, label in labels.items() if label != "NA"}
    total = sum(weights[rid] for rid in applicable)
    if not total:
        return None, sorted(applicable)
    return -tau * math.log(sum(weights[rid] / total * math.exp(-score / tau) for rid, score in applicable.items())), sorted(applicable)


def verify(config: dict, run: Path, log: list[dict], records: list[dict]) -> dict:
    """The properties this check exists for, evaluated on what the run actually produced."""
    import math

    from src.agentclinic.adapter import PRIVATE_FIELDS, agent_view, leaf_values, render
    from src.agentclinic.outcome import normalize
    from src.agentclinic.protocol import agent_prompt
    from src.agentclinic.selection import split_eligible_cases
    from src.effectiveness.history import format_input

    prompts = read_jsonl(run / "prompts.jsonl")
    states = read_jsonl(run / "states.jsonl")
    rubric_set = json.loads((run / "rubrics" / "rubric_set.json").read_text(encoding="utf-8"))
    rubric_weight = {r["rubric_id"]: r["weight"] if config["rubric_importance"]["strategy"] == "failure_frequency"
                     else 1.0 / len(rubric_set["rubrics"]) for r in rubric_set["rubrics"]}
    tau_rubric = config["rubric_aggregation"]["temperature"]
    covered = tuple(config["reward"]["effectiveness"]["stages"])  # the turns the Effectiveness predictor covers
    parity_path = run / "evaluation" / "effectiveness_parity.json"
    parity_report = json.loads(parity_path.read_text(encoding="utf-8")) if parity_path.is_file() else None
    section = config["reward"]["effectiveness"]
    parameters = calibration.load(ROOT / section["calibration"])
    weights, tau = config["reward"]["weights"], config["reward"]["softmin_temperature"]
    selected = json.loads((run / "cases.json").read_text())
    cases = split_eligible_cases(read_config(ROOT / config["environment"]["config"]), ROOT)
    frozen = {h["history_id"]: h for h in frozen_histories(config, ROOT, cases)}
    system = (ROOT / config["resolved_environment"]["prompts"]["consultant"]).read_text(encoding="utf-8")
    second = {s["history_id"]: s for s in read_jsonl(run / "a2_samples.jsonl", missing_ok=True)}
    stages = training_stages(config)
    grpo_prompt = {r["sample_id"]: r["prompt"] for r in grpo_rows(cases, list(frozen.values()), system,
                                                                  a2=list(second.values()), stages=stages)}
    concurrent = config["devices"]["placement"] == "concurrent"

    def soft_min(values: list) -> float:
        return -tau * math.log(sum(math.exp(-v / tau) for v in values) / len(values))

    results = {}
    for row, record in zip(log, records):
        name, case = row["sample_id"], cases[row["case_id"]]
        gold = normalize(case["gold"])
        calls = sorted((p for p in prompts if p.get("sample_id") == name), key=lambda p: p["order"])
        roles = [p["role"] for p in calls]
        doctor_calls = [p for p in calls if p["role"] == "doctor"]
        agent_calls = [p for p in calls if p["role"].startswith("agent (A") or p["role"] == "agent"]
        state_calls = [p for p in calls if "state elicitation" in p["role"]]
        patient = "\n".join(leaf_values(case["doctor_view"])).lower()
        private = [leaf.lower() for field in PRIVATE_FIELDS for leaf in leaf_values(case["private_evidence"][field])
                   if len(leaf) > 25 and leaf.lower() not in patient]
        a1, a2 = (row["effectiveness"] or {}).get("A1") or {}, (row["effectiveness"] or {}).get("A2") or {}
        parts = row.get("components") or {}
        scores = row.get("rubric_scores") or {}
        per_turn = {stage: recompute_appropriateness(scores[stage]["labels"], rubric_weight, tau_rubric) for stage in ("A1", "A2") if stage in scores}
        available = [value for value, _ in per_turn.values() if value is not None]
        expected_r_a = soft_min(available) if available else None
        logged_r_a = row.get("raw_R_A")  # the response's own value, before any group-mean imputation
        used = row.get("weights_used") or {}
        recomputed = sum(used[k] * parts[k]["value"] for k in used) if row["status"] == "completed" and parts else None
        request = (a1.get("request") or {})
        shared = {  # properties of every training sample, whichever Agent turn it trains
            "collaboration_completed": row["status"] == "completed",
            "stage_order_D1_A1_D2_A2_FINAL": [m["stage"] for m in row["dialogue"]] == ["D1", "A1", "D2", "A2", "FINAL"],
            "d1_is_the_frozen_history": row["dialogue"][0]["text"] == frozen[name.split("/")[0]]["history"][0]["text"],
            "trained_turn_prompt_is_the_grpo_prompt": bool(agent_calls) and agent_calls[0]["system"] == grpo_prompt[name][0]["content"]
            and agent_calls[0]["user"] == grpo_prompt[name][1]["content"],
            "state_fields_absent_from_response_prompts": not any(
                marker in p["user"] for p in agent_calls for marker in ("better_supported_position", "evidence_sufficient", "supporting_evidence_summary")),
            "correct_diagnosis_in_no_information_section": not any(
                gold and gold in normalize(information_section(p["user"])) for p in calls),
            "doctor_information_has_no_consultant_only_finding": bool(doctor_calls) and not any(
                "consultant only" in information_section(p["user"]).lower()
                or any(finding in information_section(p["user"]).lower() for finding in private) for p in doctor_calls),
            "correct_diagnosis_in_no_scorer_or_predictor_input": not any(
                gold in normalize(text) for text in
                [s["input"]["agent_pre_response_state"]["clinical_information_available_to_agent"] for s in scores.values()]
                + [json.dumps(s["input"]["rubrics"]) for s in scores.values()] + [request.get("history", {}).get("scenario", "")])
            and all("final_diagnosis" not in json.dumps(s["input"]) and "ground_truth" not in json.dumps(s["input"]) for s in scores.values()),
            "scorer_saw_only_turns_up_to_its_target": all(
                [t["turn_id"] for t in s["input"]["dialogue_through_target_turn"]] == list(range(s["input"]["turn_id"] + 1))
                for s in scores.values()),
            "appropriateness_per_turn_is_the_weighted_softmin_of_applicable_rubrics": all(
                (parts["appropriateness"]["metadata"]["per_turn"][stage]["value"] is None and value is None)
                or (value is not None and abs(parts["appropriateness"]["metadata"]["per_turn"][stage]["value"] - value) < 1e-12)
                for stage, (value, _) in per_turn.items()) if parts else False,
            "no_match_is_not_scored_zero": all(
                rid not in parts["appropriateness"]["metadata"]["per_turn"][stage]["applicable_rubrics"]
                for stage, s in scores.items() for rid, label in s["labels"].items() if label == "NA") if parts else False,
            "turn_without_applicable_rubric_is_flagged_not_zero": all(
                parts["appropriateness"]["metadata"]["per_turn"][stage]["no_applicable_rubric"] == (value is None)
                for stage, (value, _) in per_turn.items()) if parts else False,
            "appropriateness_is_the_weighted_softmin_of_the_response_s_rubrics": (expected_r_a is None and logged_r_a is None)
            or (expected_r_a is not None and logged_r_a is not None and abs(expected_r_a - logged_r_a) < 1e-12),
            # a response without an applicable rubric: never 0; its group's mean when the group has one, else unavailable
            "no_applicable_rubric_is_imputed_or_unavailable_never_zero": row.get("no_applicable") == (expected_r_a is None) and (
                expected_r_a is not None or (parts.get("appropriateness", {}).get("value") == row.get("imputed_R_A")
                                             and (row.get("imputed_R_A") is not None) != ("appropriateness" in row["unavailable_components"]))),
            "outcome_binary_from_the_final_diagnosis": row["outcome"] in (0.0, 1.0) and row["final_diagnosis"] == row["dialogue"][-1].get("final_diagnosis"),
            "components_have_value_available_source": bool(parts) and all(
                {"value", "available", "source", "model_version", "metadata"} <= set(part) for part in parts.values()),
            "composite_reward_recomputes": recomputed is not None and abs(recomputed - row["reward"]) < 1e-9,
            # an unavailable component adds nothing and the other weights are the configured ones, not rescaled
            "weights_from_config_without_rescaling": bool(used) and all(abs(used[k] - weights[k]) < 1e-12 for k in used)
            and set(used) == {k for k in weights if weights[k] > 0 and parts[k]["value"] is not None},
            "raw_composite_is_the_reward_passed_to_trl": row["raw_composite"] is not None
            and abs(row["raw_composite"] - row["reward_passed_to_trl"]) < 1e-12 and row["reward"] == row["reward_passed_to_trl"],
            "appropriateness_turn_log_is_consistent": set(row["raw_R_A_by_turn"]) == set(scores) == set(row["imputed_R_A_by_turn"])
            and all(row["turn_applicability"][s] == (row["raw_R_A_by_turn"][s] is not None) for s in scores)
            and all((per_turn[s][0] is None and row["raw_R_A_by_turn"][s] is None)
                    or (per_turn[s][0] is not None and abs(per_turn[s][0] - row["raw_R_A_by_turn"][s]) < 1e-12) for s in scores)
            and row["final_R_A"] == parts["appropriateness"]["value"],
        }
        if row.get("stage") == "A2":
            sample = second[name]
            final_calls = [p for p in calls if p["role"] == "doctor"]
            results[name] = {**shared, **{
                "prompt_is_D1_A1_D2_of_the_base_collaboration": [m["text"] for m in row["dialogue"][:3]] == [m["text"] for m in sample["history"]]
                and [m["stage"] for m in sample["history"]] == ["D1", "A1", "D2"]
                and agent_calls[0]["user"] == agent_prompt(case, row["dialogue"][:3], "A2"),
                # the policy writes A2 and only the Doctor's FINAL follows (repeated when it is not valid JSON)
                "only_the_final_diagnosis_follows_A2": roles[:1] == ["agent (A2, policy sample)"] and len(roles) > 1
                and set(roles[1:]) == {"doctor"} and all("Give your final diagnosis" in p["user"] for p in final_calls),
                "state_before_A2_is_the_stored_one": any(s["before"] == "A2" and s["state"] == sample["states"]["A2"]
                                                         and s["state"].get("state_status") == "elicited"
                                                         for s in states if s["sample_id"] == name),
                "only_A2_is_scored": set(scores) == {"A2"} and scores["A2"]["status"] == "scored"
                and set(scores["A2"]["labels"]) == set(rubric_weight) and scores["A2"]["input"]["turn_id"] == 3,
                # no predictor covers A2: no R_E at all (not 0, not the A1 predictor). A predictor that covers A2: its delta.
                "effectiveness_of_the_A2_sample_is_A2s_own_or_unavailable": set(row["effectiveness"]) == {"A2"} and (
                    (a2.get("available") is True and abs(parts["effectiveness"]["value"] - a2["delta"]) < 1e-12
                     and abs(a2["delta"] - (a2["after"] - a2["before"])) < 1e-12 and a2["prediction"]["intervention_turn"] == "A2"
                     and [t["text"] for t in a2["request"]["history"]["turns"]] == [m["text"] for m in row["dialogue"][:3]]
                     and a2["request"]["response"] == row["dialogue"][3]["text"])
                    if "A2" in covered else
                    (a2.get("available") is False and "delta" not in a2 and parts["effectiveness"]["value"] is None
                     and parts["effectiveness"]["available"] is False and "effectiveness" in row["unavailable_components"]
                     and "effectiveness" not in used)),
            }}
            continue
        results[name] = {**shared, **{
            # the order in which the models were actually called for this sample
            # state(A1), A1, then per attempt D2, state(A2), A2, FINAL (a FINAL that is not valid JSON repeats the attempt)
            "calls_in_protocol_order": roles[:2] == ["agent_greedy (state elicitation)", "agent (A1, policy sample)"]
            and len(roles) > 2 and (len(roles) - 2) % 4 == 0
            and all(roles[k:k + 4] == ["doctor", "agent_greedy (state elicitation)", "agent", "doctor"] for k in range(2, len(roles), 4)),
            "a2_prompt_is_the_label_pipeline_prompt": len(agent_calls) >= 2
            and agent_calls[-1]["user"] == agent_prompt(case, row["dialogue"][:3], "A2"),
            "state_elicited_before_A1_and_A2": sorted(s["before"] for s in states if s["sample_id"] == name) == ["A1", "A2"]
            and all(s["state"].get("state_status") == "elicited" for s in states if s["sample_id"] == name),
            "state_output_absent_from_response_and_doctor_prompts": bool(agent_calls) and len(state_calls) >= 2 and not any(
                out["output"].strip() and out["output"].strip() in p["user"] for out in state_calls for p in agent_calls + doctor_calls),
            "agent_prompts_hold_the_consultant_only_findings": all("consultant only" in information_section(p["user"]).lower()
                                                                   for p in agent_calls + state_calls),
            # turn-local credit: only A1 is scored and predicted for an A1 sample; the continuation's A2 is not part of its reward
            "only_A1_enters_the_reward_of_the_A1_sample": set(scores) == {"A1"} == set(row["effectiveness"])
            and row["scored_stages"] == ["A1"] and len(row["turn_effectiveness"]) == 1,
            "rubric_labels_complete_and_valid": set(scores) == {"A1"} and all(
                s["status"] == "scored" and set(s["labels"]) == set(rubric_weight) and set(s["labels"].values()) <= {"NA", 0, 1, 2}
                for s in scores.values()),
            "effectiveness_A1_available": a1.get("available") is True and a1.get("source") == "grpc",
            "effectiveness_A1_probabilities_are_temperature_scaled": a1.get("available") is True and all(
                abs(p - calibration.apply_logits(parameters, [z])[0]) < 1e-12 for p, z in
                ((a1["prediction"]["p_before"], a1["prediction"]["logit_before"]), (a1["prediction"]["p_after"], a1["prediction"]["logit_after"])))
            and abs(a1["calibration_temperature"] - parameters["temperature"]) < 1e-12
            and abs(a1["prediction"]["p_after"] - 1 / (1 + math.exp(-a1["prediction"]["logit_after"]))) > 1e-6,
            "effectiveness_A1_delta_is_after_minus_before": a1.get("available") is True
            and abs(a1["delta"] - (a1["after"] - a1["before"])) < 1e-12,
            "effectiveness_A1_input_is_the_history_before_A1_and_A1": request.get("response") == row["dialogue"][1]["text"]
            and [t["text"] for t in request.get("history", {}).get("turns", [])] == [row["dialogue"][0]["text"]]
            and request.get("history", {}).get("scenario") == render(agent_view(case))
            and format_input(request["history"], request["response"]).count("[AGENT RESPONSE]") == 1,
            "effectiveness_A1_model_is_the_configured_one": a1.get("model_version") == section["model_version"]
            and a1.get("checkpoint_id") == json.loads((ROOT / section["checkpoint"] / "predictor_config.json").read_text())["trainable_state_sha256"][:16],
            "effectiveness_reward_uses_A1_only": parts.get("effectiveness", {}).get("value") is not None
            and abs(parts["effectiveness"]["value"] - a1.get("delta", float("nan"))) < 1e-12,
            "effectiveness_called_between_A1_and_D2": (not concurrent) or bool(a1.get("request_id")),
        }}
    request_log = run / "grpc_logs" / "effectiveness_requests.jsonl"
    served = {r.get("request_id") for r in read_jsonl(request_log, missing_ok=True) if r.get("status") == "OK"}
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    overall = {
        "train_cases_only": bool(selected) and all(c.get("split") == "train" and cases.get(c["case_id"], {}).get("split") == "train" for c in selected),
        "case_records_complete": len(log) == len(records) == len(selected) * len(stages) and bool(log)
        and sorted(r.get("stage") for r in log) == sorted(list(stages) * len(selected))
        and {r["case_id"] for r in log} == {c["case_id"] for c in selected},
        "effectiveness_parity_passed": bool(parity_report and parity_report["passed"]),
        "every_effectiveness_request_is_in_the_service_log": bool(served) and all(
            ((r["effectiveness"] or {}).get("A1") or {}).get("request_id") in served for r in log if r.get("stage") == "A1"),
        "no_weight_update": not any((run / "checkpoints").glob("**/*.safetensors")) and not any((run / "checkpoints").glob("**/*.pt")),
        "run_files_present": all((run / name).is_file() for name in (
            "resolved_config.json", "environment.json", "manifest.json", "trajectories.jsonl", "states.jsonl", "rewards.jsonl",
            "prompts.jsonl", "rollouts.jsonl", "rubric_scores.jsonl", "effectiveness.jsonl", "rubrics/rubric_set.json",
            "evaluation/outcomes.json", "evaluation/effectiveness_parity.json")),
        "run_manifest_has_identity": all(manifest.get(k) is not None for k in ("run_id", "config_hash", "seed", "created", "git")),
    }
    failed = [f"{name}: {check}" for name, checks in results.items() for check, ok in checks.items() if not ok] \
        + [check for check, ok in overall.items() if not ok]
    return {"passed": not failed, "failed": failed, "per_case": results, "run": overall,
            "paid_api_calls_recorded": len(read_jsonl(run / "api_usage.jsonl", missing_ok=True))}


def summarize(config: dict, run: Path, log: list[dict], checks: dict, cost: Optional[dict]) -> str:
    per_case = sum(len(v) for v in checks["per_case"].values())
    lines = [f"# End-to-end pipeline check: {config['run_id']}", "",
             f"Cases: {len(log)} TRAIN case(s). No weight update. Rubrics: fixture set (not research rubrics).",
             f"Device placement: {config['devices']['placement']} (physical GPUs {config['devices'].get('physical_gpus')}).",
             f"Checks: {per_case + len(checks['run'])} evaluated, passed: {checks['passed']}"
             + ("" if checks["passed"] else f" (failed: {checks['failed']})"), ""]
    for row in log:
        parts = row.get("components") or {}
        a1 = (row["effectiveness"] or {}).get("A1") or {}
        lines += [f"## {row['sample_id']} (trained turn: {row.get('stage')})", "",
                  f"- status: {row['status']}; final diagnosis: {row['final_diagnosis']}; outcome: {row['outcome']} ({row['outcome_method']})",
                  f"- A1 Effectiveness: p_before {a1.get('before')}, p_after {a1.get('after')}, delta {a1.get('delta')} "
                  f"(model {a1.get('model_version')}, T = {a1.get('calibration_temperature')}, {a1.get('source')})",
                  f"- A2 Effectiveness: {(row['effectiveness'] or {}).get('A2')}",
                  f"- rubric labels: {json.dumps({stage: s.get('labels') for stage, s in row['rubric_scores'].items()})}",
                  f"- appropriateness per turn: {row['turn_appropriateness']}; raw R_A {row.get('raw_R_A')}, "
                  f"imputed {row.get('imputed_R_A')}, used {parts.get('appropriateness', {}).get('value')}",
                  f"- R_O {parts.get('outcome', {}).get('value')}, R_E {parts.get('effectiveness', {}).get('value')}; "
                  f"weights used {row.get('weights_used')}; reward {row['reward']}", ""]
    lines += ["## Cost", "", f"API cost of this run: {cost['total_usd'] if cost else 'n/a'} USD over {cost['calls'] if cost else 0} calls", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=("run", "rollout", "parity", "reward"))
    parser.add_argument("--prerequisite-run", help="required for 2-case pilot: a passed 1-case run id")
    parser.add_argument("--config", required=True)
    parser.add_argument("--cases", type=int, choices=(1, 2), default=1)
    parser.add_argument("--run-id")
    parser.add_argument("--rubric-set", default=FIXTURE)
    parser.add_argument("--parity-histories", type=int, default=6)
    parser.add_argument("--set", action="append", default=[], metavar="a.b=JSON")
    parser.add_argument("--write-run-id", help="file that receives the new run id (for scripts/run_e2e_pilot.sh)")
    args = parser.parse_args()
    config = load_main_config(args.config, ROOT, args.set)
    if args.stage in ("run", "rollout"):
        if args.cases == 2:
            if not args.prerequisite_run or Path(args.prerequisite_run).name != args.prerequisite_run:
                raise SystemExit("2-case pilot requires --prerequisite-run")
            prior = ROOT / config["output_root"] / args.prerequisite_run
            check = json.loads((prior / "checks.json").read_text())
            previous_cases = json.loads((prior / "cases.json").read_text())
            if not check.get("passed") or len(previous_cases) != 1:
                raise SystemExit("prerequisite must be a passed 1-case dry-run")
            open_run(config, ROOT, run_id=args.prerequisite_run, resume=True)
        result = rollout(config, args.cases, args.rubric_set)
        if args.write_run_id:
            Path(args.write_run_id).write_text(result["run_id"], encoding="utf-8")
        if args.stage == "run":
            config = open_run(config, ROOT, run_id=result["run_id"], resume=True)
            result = finish(config)
    else:
        if not args.run_id:
            raise SystemExit(f"{args.stage} needs --run-id")
        config = open_run(config, ROOT, run_id=args.run_id, resume=True)
        result = parity(config, args.parity_histories) if args.stage == "parity" else finish(config)
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    if args.stage in ("run", "reward") and not result["checks_passed"]:
        raise SystemExit(1)
    if args.stage == "parity" and not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
