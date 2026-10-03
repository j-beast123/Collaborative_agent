"""Effectiveness labels on AgentClinic-MedQA (v5): inspect, pilot, and (gated) full labelling."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

from src.agentclinic.adapter import (agent_view, assert_no_leakage, diagnosis_named_in_private_evidence,
                                     diagnosis_stated_in_context, doctor_view, evaluator_view, load_cases)
from src.agentclinic.candidates import CANDIDATES, ComplianceJudge, compliance_system_prompt
from src.agentclinic.outcome import DiagnosisJudge
from src.agentclinic.pipeline import (build_labels, check_frozen, freeze_histories, generate_candidates,
                                     integrity_check, prepare, prepare_diagnostics, rescore, select_label_histories)
from src.agentclinic.protocol import ClinicRuntime, agent_prompt, doctor_prompt
from src.agentclinic.report import build_report
from src.agentclinic.adapter import SOURCE_DATASET
from src.agentclinic.selection import sample_pilot, select_eligible, split_fractions, split_manifest
from src.common.config import IntegrationUnavailable, read_config
from src.common.io import read_jsonl, write_jsonl
from src.judges.llm import ChatRequest, openrouter_chat

ROOT = Path(__file__).resolve().parents[1]
MODEL_STAGES = ("prepare", "freeze", "rescore", "candidates", "labels", "report")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select(config: dict, scope: str) -> tuple[list[dict], dict]:
    """(cases of the scope, dataset summary). Selection uses case content and ids only."""
    source = load_cases(ROOT / config["dataset"]["path"], (config.get("doctor_view") or {}).get("patient_fields"))
    eligible, splits, reasons = select_eligible(config, ROOT)
    if scope == "full":
        if config.get("run_full_labeling") is not True:
            raise IntegrationUnavailable("full labelling is disabled: review the pilot's signal_report.md, then set "
                                         "run_full_labeling to true in the config by hand")
        cases = [c for c in eligible if splits[c["case_id"]] in ("train", "dev")]
    elif scope == "screen":  # every TRAIN case, for the prepare diagnostic only
        cases = [c for c in eligible if splits[c["case_id"]] == "train"]
    else:
        pilot = config["pilot"]
        if pilot.get("same_cases_as"):  # a variant on the cases of another run, for a paired comparison
            wanted = {row["case_id"] for row in read_jsonl(ROOT / pilot["same_cases_as"] / "cases.jsonl")}
            cases = [c for c in eligible if c["case_id"] in wanted]
            if len(cases) != len(wanted):
                raise ValueError("same_cases_as names cases that are not eligible here")
        else:
            cases = sample_pilot(eligible, splits, split=pilot["split"], n=pilot["cases"], seed=pilot["seed"])
    for case in cases:
        case["split"] = splits[case["case_id"]]
        assert_no_leakage(case)
    summary = {"source_cases": len(source), "eligible": len(eligible), "sampled": len(cases),
               "ineligible_reasons": reasons, "selection": config["dataset"]["selection_notes"],
               "split_sizes": {name: sum(s == name for s in splits.values()) for name in ("train", "dev", "test")},
               "splits": splits,
               "gold_named_in_private_evidence": sum(diagnosis_named_in_private_evidence(c) for c in cases),
               "cases_with_diagnosis_named_in_private_evidence": [c["case_id"] for c in source
                                                                  if diagnosis_named_in_private_evidence(c)]}
    return cases, summary


def render(cases: list[dict], n: int) -> None:
    doctor_system = (ROOT / "prompts/agentclinic/doctor.txt").read_text(encoding="utf-8")
    agent_system = (ROOT / "prompts/agentclinic/consultant.txt").read_text(encoding="utf-8")
    print("#" * 30, "DOCTOR system prompt (all cases)", "#" * 30, "\n" + doctor_system)
    print("#" * 30, "CONSULTANT AGENT system prompt (all cases)", "#" * 30, "\n" + agent_system)
    for case in cases[:n]:
        placeholder = [{"speaker": "doctor", "stage": "D1", "text": "<the Doctor's D1 message>"}]
        print("\n" + "=" * 100 + f"\nCASE {case['case_id']}  (split {case['split']})")
        print("-" * 30, "DOCTOR, user prompt at D1", "-" * 30, "\n" + doctor_prompt(case, [], "D1"))
        print("-" * 30, "CONSULTANT AGENT, user prompt at A1", "-" * 30, "\n" + agent_prompt(case, placeholder, "A1"))
        print("-" * 30, "EVALUATOR VIEW (never in a prompt)", "-" * 30, "\n" + json.dumps(evaluator_view(case)))


def leakage(config: dict) -> dict:
    source = load_cases(ROOT / config["dataset"]["path"], (config.get("doctor_view") or {}).get("patient_fields"))
    for case in source:
        assert_no_leakage(case)
        assert "Correct_Diagnosis" not in doctor_view(case) and "Correct_Diagnosis" not in agent_view(case)
    named = [diagnosis_stated_in_context(c) for c in source]
    return {"cases_checked": len(source), "doctor_view_contains_examination_or_test_findings": 0,
            "prompt_views_contain_correct_diagnosis_field": 0,
            "gold_string_appears_in_doctor_view": sum(n["doctor_view"] for n in named),
            "gold_string_appears_in_agent_only_evidence": sum(n["agent_view"] and not n["doctor_view"] for n in named)}


def estimate(config: dict, n_cases: int) -> dict:
    k, n_candidates = config["k"], len(CANDIDATES)
    continuations = n_cases * (1 + n_candidates) * k
    doctor_calls = n_cases * (1 + 2) + continuations * 2  # D1; natural D2 + FINAL; each continuation D2 + FINAL
    judge_calls = n_cases * n_candidates + continuations * 3  # compliance; outcome, revision and stance comparisons
    calls = doctor_calls + judge_calls
    cost = calls * (1500 * 0.15 + 200 * 0.60) / 1e6  # gpt-4o-mini list price per 1M tokens: 0.15 in, 0.60 out
    return {"cases": n_cases, "k": k, "before_continuations": n_cases * k,
            "after_continuations": n_cases * n_candidates * k, "local_agent_generations": continuations + n_cases * k
            + n_cases * (2 + n_candidates), "doctor_api_calls": doctor_calls,
            "judge_api_calls_upper_bound": judge_calls,
            "cost_usd_upper_bound": round(cost, 2),
            "cost_assumption": "about 1,500 input and 200 output tokens per call at gpt-4o-mini list prices; judge "
                               "calls are cached per distinct input, so the real number is lower"}


def protocol_identity(config: dict) -> dict:
    """The hash under which src/effectiveness/train.py accepts these labels (the predictor config's
    ContinuationSpec), when the run names a predictor config."""
    if not config.get("predictor_config"):
        return {}
    from src.effectiveness.train import load_predictor_config

    spec = load_predictor_config(ROOT / config["predictor_config"])["continuation_spec"]
    if spec.k != config["k"]:
        raise ValueError("predictor config K differs from the labelling K")
    return {"continuation_protocol_hash": spec.protocol_hash()}


def write_split_files(out: Path, dataset: dict, config: dict) -> dict:
    """labels_train.jsonl / labels_dev.jsonl for training, and the case-level split manifest."""
    labels = read_jsonl(out / "labels.jsonl", missing_ok=True)
    counts = {}
    for split in ("train", "dev"):
        rows = [r for r in labels if r["split"] == split]
        if rows:
            write_jsonl(out / f"labels_{split}.jsonl", rows)
        counts[split] = len(rows)
    seed, fractions = split_fractions(config, dataset["splits"])
    manifest = split_manifest(dataset["splits"], source_dataset=SOURCE_DATASET,
                              dataset_version="sha256:" + sha(ROOT / config["dataset"]["path"])[:16], seed=seed, fractions=fractions)
    (out / "split_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return counts


def build_runtime(config: dict):
    from src.collaboration.generator import doctor_call
    from src.collaboration.runtime import load_local_agent

    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise SystemExit("OPENROUTER_API_KEY is not set")
    out = ROOT / config["output_dir"]
    models, decoding, prompts = config["models"], config["decoding"], config["prompts"]
    generate, close = load_local_agent(models["agent"]["checkpoint"] or models["agent"]["model_id"],
                                       config["labels"]["agent_batch_size"])
    agent = dict(decoding["agent"])
    tokens = agent.pop("max_new_tokens")
    runtime = ClinicRuntime(
        doctor=lambda system, user: doctor_call(key, models["doctor"]["model_id"], system, user,
                                                decoding["doctor"]["max_tokens"], decoding["doctor"]["temperature"]),
        agent=lambda system, user: generate(system, user, tokens, **agent),
        agent_greedy=lambda system, user: generate(system, user, tokens),
        doctor_system=(ROOT / prompts["doctor"]).read_text(encoding="utf-8"),
        agent_system=(ROOT / prompts["consultant"]).read_text(encoding="utf-8"), close=close)
    judge_model = models["judge"]["model_id"]
    from src.agentclinic.outcome import frozen_equivalence

    diagnosis = DiagnosisJudge(openrouter_chat(judge_model, out / "cache_diagnosis_judge.jsonl"),
                               (ROOT / prompts["diagnosis_equivalence"]).read_text(encoding="utf-8"),
                               params=config.get("diagnosis_judge_params"), frozen=frozen_equivalence(config, ROOT),
                               policy=config.get("diagnosis_judge_policy"))
    compliance = compliance_judge(config, out)
    return runtime, diagnosis, compliance


def compliance_judge(config: dict, out: Path):
    """The judge of the candidate responses: label_roles.compliance_judge when it is configured (backend jev: typed
    decisions with probabilities), else the judge model of the run with the written class definitions."""
    settings = (config.get("label_roles") or {}).get("compliance_judge") or {}
    if settings.get("backend") == "jev" and settings.get("model_id"):
        from src.agentclinic.candidates import JevComplianceJudge
        from src.judges.jev import JevClient

        return JevComplianceJudge(JevClient(endpoint=settings.get("endpoint"), model=settings["model_id"],
                                            api_key_env=settings.get("api_key_env")))
    return ComplianceJudge(openrouter_chat(config["models"]["judge"]["model_id"], out / "cache_compliance_judge.jsonl"),
                           compliance_system_prompt(ROOT))


def candidate_generator(config: dict):
    """(system, user) -> text of the model that writes the candidate responses, when label_roles.candidate_generator
    names one; None when the Agent model writes them.
    """
    settings = (config.get("label_roles") or {}).get("candidate_generator") or {}
    if not settings.get("model_id"):
        return None
    from src.judges.llm import ChatRequest, build_chat_backend

    backend = build_chat_backend(settings)
    params = settings.get("params") or {}

    def generate(system: str, user: str) -> str:
        result = backend.complete(ChatRequest(system, user, "candidate_generator", params))
        if result.status != "ok":
            raise OSError(f"candidate generator call failed: {result.error}")
        return result.text

    return generate


def evidence_checker(config: dict):
    chat = openrouter_chat(config["models"]["judge"]["model_id"], ROOT / config["output_dir"] / "cache_evidence_check.jsonl")
    prompt = (ROOT / config["prompts"]["evidence_check"]).read_text(encoding="utf-8")

    def new_evidence(dialogue: list, index: int):
        earlier = "\n\n".join(f"{m['speaker'].title()}: {m['text']}" for m in dialogue[:index])
        user = json.dumps({"discussion_so_far": earlier, "new_clinician_message": dialogue[index]["text"]}, ensure_ascii=False)
        result = chat.complete(ChatRequest(prompt, user, "evidence-check-v1", {"temperature": 0, "max_tokens": 3}))
        reply = (result.text or "").strip().upper()
        return None if result.status != "ok" or reply[:3] not in ("YES", "NO") else reply.startswith("YES")

    return new_evidence


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--scope", choices=("pilot", "screen", "full"), default="pilot")
    p.add_argument("--n", type=int, default=3, help="render: number of cases")
    p.add_argument("stages", nargs="+", choices=("render", "leakage", "estimate") + MODEL_STAGES)
    a = p.parse_args()
    config = read_config(a.config)
    if a.scope == "screen" and set(a.stages) - {"prepare", "render", "leakage", "estimate"}:
        raise SystemExit("--scope screen runs prepare only; candidates and K rollouts need --scope pilot or full")
    cases, dataset = select(config, a.scope)
    by_id = {c["case_id"]: c for c in cases}
    out = ROOT / config["output_dir"]
    results = {}
    if "render" in a.stages:
        render(cases, a.n)
    if "leakage" in a.stages:
        results["leakage"] = leakage(config)
    if "estimate" in a.stages:
        results["estimate"] = estimate(config, len(cases))

    if any(stage in MODEL_STAGES for stage in a.stages):
        out.mkdir(parents=True, exist_ok=True)
        write_jsonl(out / "cases.jsonl", [{"case_id": c["case_id"], "source_index": c["source_index"], "split": c["split"],
                                           "doctor_view": doctor_view(c), "agent_view": agent_view(c),
                                           "evaluator_view": evaluator_view(c)} for c in cases])
        (out / "run_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
        (out / "run_manifest.json").write_text(json.dumps({
            "version": config["version"], "scope": a.scope, "config": a.config,
            "dataset": {k: v for k, v in dataset.items() if k != "splits"},
            "dataset_sha256": sha(ROOT / config["dataset"]["path"]),
            "prompt_sha256": {name: sha(ROOT / path) for name, path in config["prompts"].items()},
            "rq2_behavior_prompt_sha256": sha(ROOT / "prompts/behavior_judge.txt"),
            "case_ids": [c["case_id"] for c in cases], "candidates": list(CANDIDATES), "k": config["k"],
            "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        needs_agent = any(stage in ("prepare", "candidates", "labels") for stage in a.stages)
        runtime = diagnosis = compliance = None
        if needs_agent:
            runtime, diagnosis, compliance = build_runtime(config)
        else:
            diagnosis = DiagnosisJudge(openrouter_chat(config["models"]["judge"]["model_id"], out / "cache_diagnosis_judge.jsonl"),
                                       (ROOT / config["prompts"]["diagnosis_equivalence"]).read_text(encoding="utf-8"))
        workers = config["labels"]["workers"]
        try:
            if "prepare" in a.stages:
                shared = (config.get("screen") or {}).get("standalone_file") if a.scope == "screen" else None
                results["prepare"] = prepare(cases, runtime, diagnosis, out, workers,
                                             standalone_file=ROOT / shared if shared else None)
                results["prepare_diagnostics"] = prepare_diagnostics(out, diagnosis)
            if "freeze" in a.stages:
                results["freeze"] = {"histories_sha256": freeze_histories(out)}
            check_frozen(out)
            if "rescore" in a.stages:
                results["rescore"] = rescore(out, diagnosis)
                results["prepare_diagnostics"] = prepare_diagnostics(out, diagnosis)
            histories = [h for h in read_jsonl(out / "histories.jsonl", missing_ok=True) if h["case_id"] in by_id]
            if a.scope == "pilot" and config["pilot"].get("label_selection") and histories \
                    and any(stage in ("candidates", "labels") for stage in a.stages):
                rule = config["pilot"]["label_selection"]
                selection = select_label_histories(histories, diagnosis, n=rule["cases"], seed=rule["seed"])
                (out / "pilot_label_cases.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
                histories = [h for h in histories if h["case_id"] in set(selection["selected_case_ids"])]
                results["label_histories"] = {"pool": len(selection["pool_case_ids"]), "selected": len(histories)}
            if "candidates" in a.stages:
                results["candidates"] = generate_candidates(by_id, histories, runtime, compliance, out, workers,
                                                            max_attempts=config["candidates"]["max_attempts"])
            if "labels" in a.stages:
                results["labels"] = build_labels(
                    by_id, histories, runtime, diagnosis, out, k=config["k"], workers=workers,
                    require_compliance=config["labels"]["require_compliance"],
                    max_attempts=config["labels"]["max_rollout_attempts"],
                    metadata={"version": config["version"], "k": config["k"], "models": config["models"],
                              "decoding": config["decoding"], **protocol_identity(config),
                              "split": cases[0]["split"] if a.scope == "pilot" else "mixed"})
                results["label_files"] = write_split_files(out, dataset, config)
        finally:
            if runtime and runtime.close:
                runtime.close()
        if "report" in a.stages:
            report = build_report(out, {k: v for k, v in dataset.items() if k != "splits"}, diagnosis,
                                  evidence_checker(config))
            results["report"] = {"signal": report["signal"], "file": str(out / "signal_report.md")}
            if config.get("integrity"):  # full runs: training starts only if the labels pass
                results["integrity"] = integrity_check(out, k=config["k"], splits=dataset["splits"],
                                                       thresholds=config["integrity"])
                if not results["integrity"]["passed"]:
                    print(json.dumps(results, indent=2, ensure_ascii=False, default=str))
                    raise SystemExit("label integrity check failed: " + ", ".join(results["integrity"]["failures"]))
    if results:
        print(json.dumps(results, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
