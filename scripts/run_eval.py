"""Evaluation pipeline (EVAL split): RQ1, RQ2 (states, controlled, effectiveness), RQ3, figures."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Optional

from analysis.plot_rq2_behavior import plot_bubble, plot_effectiveness, plot_rq2_behavior
from src.agentclinic.trajectory import PROTOCOL
from src.common.config import IntegrationUnavailable, read_config
from src.common.main_config import load_main_config
from src.common.io import read_jsonl
from src.effectiveness.continuation import ContinuationSpec
from src.evaluation.behavior_judge import BehaviorJudge
from src.evaluation.collaboration_states import accuracy_table, appropriateness_summary, first_collaboration_turn, state_rows
from src.evaluation.controlled_history import EvidenceBuilder, build_items, controlled_rows
from src.evaluation.figure_contract import export_rows
from src.evaluation.matched_continuation import generate_responses, run_continuations, summarize
from src.evaluation.rq3 import bubble_rows, probe_transitions
from src.judges.llm import CachedChat, build_chat_backend
from src.judges.option_extraction import OptionExtractor, with_initial_options
from src.judges.rubric_consolidation import write_json

ROOT = Path(__file__).resolve().parents[1]
META_FIELDS = ("run_id", "source_dataset", "dataset_version", "split_manifest_hash", "reference_protocol_version")


def require_meta(config: dict) -> dict:
    missing = [k for k in META_FIELDS if not config["meta"].get(k)]
    for name, system in config["systems"].items():
        missing += [f"systems.{name}.{k}" for k in ("model_family", "model_id", "checkpoint") if system.get(k) in (None, "")]
        missing += [f"systems.{name}.{k}" for k in ("iteration", "training_seed", "response_seed")
                    if type(system.get(k)) is not int]
    if missing:
        raise IntegrationUnavailable("Evaluation metadata unresolved: " + ", ".join(missing))
    return config["meta"]


def chat(config: dict, settings: dict, name: str) -> CachedChat:
    return CachedChat(build_chat_backend(settings), ROOT / config["cache_dir"] / f"{name}.jsonl")


def make_judge(config: dict, chat_factory=chat) -> BehaviorJudge:
    settings = config["behavior_judge"]
    return BehaviorJudge(chat_factory(config, settings, "behavior_judge"),
                         system_prompt=(ROOT / settings["prompt"]).read_text(encoding="utf-8"),
                         prompt_version=settings["prompt_version"], params=settings["params"])


def evidence_validator(config: dict):
    """The validator of the controlled evidence: a chat model, or Jev (backend "jev": typed decisions, independent of the generator)."""
    settings = config["evidence"]["validator"]
    if settings.get("backend") == "jev":
        from src.agentclinic.evaluation import JevEvidenceValidator
        from src.judges.jev import JevClient

        return JevEvidenceValidator(JevClient(endpoint=settings.get("endpoint"), model=settings["model_id"], api_key_env=settings.get("api_key_env")),
                                    ROOT / config["cache_dir"] / "evidence_validator_jev.jsonl")
    return chat(config, settings, "evidence_validator")


def make_extractor(config: dict, chat_factory=chat) -> OptionExtractor:
    settings = config["option_extractor"]
    return OptionExtractor(chat_factory(config, settings, "option_extractor"),
                           system_prompt=(ROOT / settings["prompt"]).read_text(encoding="utf-8"),
                           prompt_version=settings["prompt_version"], params=settings["params"])


def reread_initial_options(trajectories: list[dict], extractor) -> list[dict]:
    """extractor None (tests with hand-written fields): the trajectories' own fields are used."""
    return [with_initial_options(t, extractor) for t in trajectories] if extractor else trajectories


def real_runtime(config: dict, system_name: str):
    from src.collaboration.runtime import build_runtime

    collab = config["collaboration"]
    model_path = config["systems"][system_name].get("model_path")
    missing = [k for k, v in {"model_path": model_path, "doctor_model": collab.get("doctor_model"),
                              "reconstructor_model": collab.get("reconstructor_model")}.items() if not v]
    if missing:
        raise IntegrationUnavailable(f"Runtime for {system_name} needs: {', '.join(missing)}")
    return build_runtime(agent_model=model_path, doctor_model=collab["doctor_model"],
                         reconstructor_model=collab["reconstructor_model"], max_tokens=collab["max_tokens"], root=ROOT)


def is_clinic(config: dict) -> bool:
    """An AgentClinic evaluation config: the fixed two-turn protocol or the main protocol (max turns, early stopping)."""
    return str(config.get("protocol") or "").startswith("agentclinic")


def clinic_for(config: dict, system_name: str):
    """The AgentClinic models with one system's Agent (base model, plus its adapter when it has one)."""
    from src.agentclinic.runtime import build_clinic, load_policy

    run = dict(load_main_config(ROOT / config["run_config"], ROOT), output_dir=config["cache_dir"])
    adapter = config["systems"][system_name].get("adapter")
    model, tokenizer = load_policy(run["policy"], ROOT / adapter if adapter else None, trainable=False,
                                   device=run["devices"]["policy"])
    return build_clinic(run, ROOT, model, tokenizer)


def clinic_classifier(config: dict, chat_factory=chat):
    from src.agentclinic.candidates import ComplianceJudge, compliance_system_prompt
    from src.agentclinic.evaluation import BehaviorClassifier
    from src.agentclinic.selection import split_eligible_cases

    run = load_main_config(ROOT / config["run_config"], ROOT)
    cases = split_eligible_cases(read_config(ROOT / run["environment"]["config"]), ROOT)
    judge = ComplianceJudge(chat_factory(config, config["behavior_judge"], "behavior_judge"), compliance_system_prompt(ROOT))
    same_diagnosis = None
    if (config.get("behavior_judge") or {}).get("position_change_rule"):  # frozen in the evaluation config
        from src.agentclinic.runtime import build_judge

        diagnosis = build_judge(dict(run, output_dir=config["cache_dir"]), ROOT, read_config(ROOT / run["environment"]["config"]))
        # the equivalence judge compares a predicted diagnosis with a reference (a more specific prediction can pass, a less
        # specific one cannot); two positions are the same diagnosis only when they are equivalent in both directions
        same_diagnosis = lambda before, after: (diagnosis.equivalent(before, after)["equivalent"] is True  # noqa: E731
                                                and diagnosis.equivalent(after, before)["equivalent"] is True)
    return BehaviorClassifier(cases, judge, same_diagnosis)


def _close(runtime) -> None:
    if getattr(runtime, "close", None):
        runtime.close()


def standalone_rows(trajectories: list[dict]) -> Optional[list[dict]]:
    """Standalone accuracy rows from trajectories that carry the Agent's judged standalone assessment."""
    rows = [{"case_id": t["case_id"], "is_correct": t["agent_initial_correct"]} for t in trajectories
            if t.get("agent_initial_correct") is not None]
    return rows or None


def run_rq1(runs: list[dict], output_dir: Path, extractor=None) -> list[dict]:
    loaded = []
    for run in runs:
        trajectories = reread_initial_options(read_jsonl(run["trajectories"]), extractor)
        loaded.append(dict(run, trajectories=trajectories,
                           standalone=read_jsonl(run["standalone"]) if run.get("standalone") else standalone_rows(trajectories)))
    table = accuracy_table(loaded)
    write_json(output_dir / "rq1_accuracy.json", table)
    return table


def run_rq2_states(config: dict, trajectories_by_system: dict[str, list[dict]], judge, output_dir: Path,
                   extractor=None, classify=None) -> list[dict]:
    meta = require_meta(config)
    rows, excluded = [], {}
    for system_name, trajectories in trajectories_by_system.items():
        trajectories = reread_initial_options(trajectories, extractor)
        system = config["systems"][system_name]
        system_meta = dict(meta, system=system_name, **{k: system[k] for k in (
            "model_family", "model_id", "checkpoint", "iteration", "training_seed", "response_seed")})
        produced = state_rows(trajectories, judge, system_meta, config["expected_actions"]["states"], extractor,
                              classify)
        excluded[system_name] = len(trajectories) - len(produced)
        rows += produced
    export_rows(rows, "state", output_dir / "figure_data/state_results.csv", synthetic=meta["synthetic"])
    summary = {"appropriateness": appropriateness_summary(rows), "state_undefined_excluded": excluded,
               "status_counts": _status_counts(rows)}
    if is_clinic(config):  # situation assessment: the state elicited before A1, separate from the response
        from src.agentclinic.evaluation import situation_assessment_rows, situation_assessment_summary

        assessment = situation_assessment_rows(trajectories_by_system)
        write_json(output_dir / "rq2_situation_assessment_rows.json", assessment)
        summary["situation_assessment"] = situation_assessment_summary(assessment, rows)
    write_json(output_dir / "rq2_states_summary.json", summary)
    return rows


def run_rq2_controlled(config: dict, reference: list[dict], evidence_builder, judge, output_dir: Path,
                       runtime_factory=real_runtime, extractor=None) -> list[dict]:
    meta = require_meta(config)
    items = build_items(reread_initial_options(reference, extractor), evidence_builder)
    write_json(output_dir / "rq2_controlled_items.json",
               [{k: v for k, v in item.items() if k != "trajectory"} for item in items])
    rows, log = [], []
    for system_name, system in config["systems"].items():
        runtime = runtime_factory(config, system_name)
        try:
            produced, produced_log = controlled_rows(items, system_name, runtime, system, judge, meta,
                                                     config["expected_actions"]["controlled"])
        finally:
            _close(runtime)
        rows += produced
        log += produced_log
    export_rows(rows, "controlled", output_dir / "figure_data/controlled_results.csv", synthetic=meta["synthetic"])
    write_json(output_dir / "rq2_controlled_log.json", log)
    write_json(output_dir / "rq2_controlled_summary.json",
               {"appropriateness": appropriateness_summary(rows), "status_counts": _status_counts(rows),
                "evidence_rejected": sum(i["validation_status"] != "accepted" for i in items)})
    return rows


def run_clinic_controlled(config: dict, reference: list[dict], builder, classifier, diagnosis_judge, output_dir: Path,
                          clinic_factory=clinic_for) -> list[dict]:
    """rq2-controlled on AgentClinic: every system's A1 to the same controlled D1 messages."""
    from src.agentclinic.evaluation import controlled_items, controlled_rows

    meta = require_meta(config)
    items = controlled_items(reference, classifier.cases, builder, diagnosis_judge)
    write_json(output_dir / "rq2_controlled_items.json",
               [{k: v for k, v in item.items() if k != "trajectory"} for item in items])
    rows, log = [], []
    for system_name, system in config["systems"].items():
        clinic = clinic_factory(config, system_name)
        try:
            produced, produced_log = controlled_rows(items, system_name, clinic, system, classifier, meta,
                                                     config["expected_actions"]["controlled"])
        finally:
            clinic.close()
        rows += produced
        log += produced_log
    export_rows(rows, "controlled", output_dir / "figure_data/controlled_results.csv", synthetic=meta["synthetic"])
    write_json(output_dir / "rq2_controlled_log.json", log)
    write_json(output_dir / "rq2_controlled_summary.json",
               {"appropriateness": appropriateness_summary(rows), "status_counts": _status_counts(rows),
                "evidence_rejected": sum(i["validation_status"] != "accepted" for i in items)})
    return rows


def run_clinic_effectiveness(config: dict, trajectories_by_system: dict[str, list[dict]], output_dir: Path,
                             clinic_factory=clinic_for) -> dict:
    """rq2-effectiveness on AgentClinic: each system's A1 to the same D1, fixed, continued K times."""
    from src.agentclinic.evaluation import matched_continuations, matched_responses

    settings = config["matched_continuation"]
    histories, responses = matched_responses(trajectories_by_system)
    if settings.get("max_histories"):
        histories = histories[: settings["max_histories"]]
    clinic = clinic_factory(config, settings["continuation_system"])
    try:
        rows = matched_continuations(histories, responses, clinic, k=settings["k"],
                                     max_attempts=settings.get("max_rollout_attempts", 1),
                                     progress_path=output_dir / "rq2_effectiveness_progress.jsonl")
    finally:
        clinic.close()
    summary = dict(summarize(rows), k=settings["k"], continuation_system=settings["continuation_system"])
    write_json(output_dir / "rq2_effectiveness_rows.json", rows)
    write_json(output_dir / "rq2_effectiveness_summary.json", summary)
    return summary


def run_rq2_effectiveness(config: dict, reference: list[dict], output_dir: Path, runtime_factory=real_runtime,
                          set_seed=None) -> dict:
    settings = config["matched_continuation"]
    spec = ContinuationSpec.from_dict({"k": settings["k"], "seed_policy": settings["seed_policy"]})
    histories = [(t, first_collaboration_turn(t)) for t in reference
                 if t.get("trajectory_status") == "valid" and first_collaboration_turn(t) is not None]
    if settings.get("max_histories"):
        histories = histories[: settings["max_histories"]]
    attempts = settings.get("max_rollout_attempts", 1)
    output_dir.mkdir(parents=True, exist_ok=True)
    responses = {}
    for system_name in config["systems"]:
        saved = output_dir / f"rq2_effectiveness_responses_{system_name}.json"
        if saved.exists():  # an interrupted run does not load this system's model again
            responses[system_name] = json.loads(saved.read_text(encoding="utf-8"))
            continue
        runtime = runtime_factory(config, system_name)
        try:
            responses[system_name] = generate_responses(histories, runtime, system_name, format_attempts=attempts)
        finally:
            _close(runtime)
        write_json(saved, responses[system_name])
    continuation = runtime_factory(config, settings["continuation_system"])
    try:
        rows = run_continuations(histories, responses, continuation, k=spec.k,
                                 seeds_for=lambda history_id: spec.rollout_seeds(history_id, history_id),
                                 set_seed=set_seed, format_attempts=attempts,
                                 progress_path=output_dir / "rq2_effectiveness_progress.jsonl")
    finally:
        _close(continuation)
    summary = dict(summarize(rows), k=spec.k, continuation_system=settings["continuation_system"],
                   note="Doctor sampling through an API may not be seedable; paired seeds apply to local sampling.")
    write_json(output_dir / "rq2_effectiveness_rows.json", rows)
    write_json(output_dir / "rq2_effectiveness_summary.json", summary)
    return summary


def run_rq3(registry: list[dict], output_dir: Path, probe_results=None, top_keywords: int | None = 8) -> dict:
    result = {"bubble": bubble_rows(registry, top_k=top_keywords)}
    write_json(output_dir / "rq3_bubble.json", result["bubble"])
    if probe_results is not None:
        result["transitions"] = probe_transitions(probe_results)
        write_json(output_dir / "rq3_probe_transitions.json", result["transitions"])
    return result


def run_plots(output_dir: Path, synthetic: bool = False) -> list[str]:
    made = []
    data = output_dir / "figure_data"
    if (data / "state_results.csv").exists() and (data / "controlled_results.csv").exists():
        plot_rq2_behavior(data / "state_results.csv", data / "controlled_results.csv",
                          output_dir / "fig_rq2_behavior.png", synthetic=synthetic)
        made.append("fig_rq2_behavior.png")
    if (output_dir / "rq2_effectiveness_rows.json").exists():
        plot_effectiveness(output_dir / "rq2_effectiveness_rows.json", output_dir / "fig_rq2_effectiveness.png")
        made.append("fig_rq2_effectiveness.png")
    if (output_dir / "rq3_bubble.json").exists():
        plot_bubble(output_dir / "rq3_bubble.json", output_dir / "fig_rq3_bubble.png")
        made.append("fig_rq3_bubble.png")
    return made


def _status_counts(rows: list[dict]) -> dict:
    counts = {}
    for row in rows:
        key = f"{row['system']}:{row['evaluation_status']}"
        counts[key] = counts.get(key, 0) + 1
    return counts


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=("rq1", "rq2-states", "rq2-controlled", "rq2-effectiveness", "rq3", "plots"))
    p.add_argument("--config")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--runs")
    p.add_argument("--trajectories", action="append", default=[], help="system=path (repeatable)")
    p.add_argument("--reference", help="reference EVAL trajectories (shared histories)")
    p.add_argument("--registry")
    p.add_argument("--probe-results")
    p.add_argument("--top-keywords", type=int, default=8, help="categories shown in the bubble plot; the rest -> OTHER")
    a = p.parse_args()
    output_dir = Path(a.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = read_config(a.config) if a.config else None
    if config and isinstance(config.get("systems"), dict):  # a system set to null in a config that extends another is not evaluated
        config["systems"] = {name: system for name, system in config["systems"].items() if system}
    by_system = {name: read_jsonl(path) for name, path in (item.split("=", 1) for item in a.trajectories)}
    if a.command == "rq1":
        if config is None:
            raise SystemExit("rq1 needs --config: per-state accuracy re-reads the initial options (option_extractor)")
        result = run_rq1(json.loads(Path(a.runs).read_text(encoding="utf-8")), output_dir,
                         None if is_clinic(config) else make_extractor(config))
    elif a.command == "rq2-states" and is_clinic(config):
        classifier = clinic_classifier(config)
        result = {"rows": len(run_rq2_states(config, by_system, classifier, output_dir, classify=classifier))}
    elif a.command == "rq2-controlled" and is_clinic(config):
        from src.agentclinic.evaluation import ClinicEvidenceBuilder
        from src.agentclinic.runtime import build_judge

        ev, run = config["evidence"], dict(load_main_config(ROOT / config["run_config"], ROOT), output_dir=config["cache_dir"])
        builder = ClinicEvidenceBuilder(chat(config, ev["generator"], "evidence_generator"),
                                        evidence_validator(config),
                                        generator_prompt=(ROOT / ev["generator_prompt"]).read_text(encoding="utf-8"),
                                        validator_prompt=(ROOT / ev["validator_prompt"]).read_text(encoding="utf-8"),
                                        prompt_version=ev["prompt_version"], params=ev["params"],
                                        max_attempts=int(ev.get("max_attempts", 1)), retry_params=ev.get("retry_params"),
                                        validator_prompt_version=ev.get("validator_prompt_version"),
                                        pair_prompt=(ROOT / ev["pair_validator_prompt"]).read_text(encoding="utf-8")
                                        if ev.get("pair_validator_prompt") else None)
        diagnosis_judge = build_judge(run, ROOT, read_config(ROOT / run["environment"]["config"]))
        result = {"rows": len(run_clinic_controlled(config, read_jsonl(a.reference), builder,
                                                    clinic_classifier(config), diagnosis_judge, output_dir))}
    elif a.command == "rq2-effectiveness" and is_clinic(config):
        result = run_clinic_effectiveness(config, by_system, output_dir)
    elif a.command == "rq2-states":
        result = run_rq2_states(config, by_system, make_judge(config), output_dir, make_extractor(config))
        result = {"rows": len(result)}
    elif a.command == "rq2-controlled":
        ev = config["evidence"]
        builder = EvidenceBuilder(chat(config, ev["generator"], "evidence_generator"),
                                  chat(config, ev["validator"], "evidence_validator"),
                                  generator_prompt=(ROOT / ev["generator_prompt"]).read_text(encoding="utf-8"),
                                  validator_prompt=(ROOT / ev["validator_prompt"]).read_text(encoding="utf-8"),
                                  prompt_version=ev["prompt_version"], params=ev["params"])
        result = {"rows": len(run_rq2_controlled(config, read_jsonl(a.reference), builder, make_judge(config),
                                                 output_dir, extractor=make_extractor(config)))}
    elif a.command == "rq2-effectiveness":
        result = run_rq2_effectiveness(config, read_jsonl(a.reference), output_dir, set_seed=_seed_everything)
    elif a.command == "rq3":
        registry = json.loads(Path(a.registry).read_text(encoding="utf-8"))
        result = run_rq3(registry, output_dir, read_jsonl(a.probe_results) if a.probe_results else None,
                         top_keywords=a.top_keywords)
        result = {k: len(v) for k, v in result.items()}
    else:
        result = run_plots(output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
