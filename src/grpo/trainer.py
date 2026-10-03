"""One SERL iteration of policy training with TRL's GRPOTrainer, and the trajectories of its checkpoint."""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from src.agentclinic.runtime import (build_clinic, check_identity, frozen_histories, load_policy, opening_histories,
                                      protocol_identity)
from src.agentclinic.trajectory import from_prepared, generate_trajectories, summary
from src.common.config import IntegrationUnavailable
from src.common.main_config import training_device_problems
from src.common.io import read_jsonl
from src.common.run_dir import write_new
from src.data.schema import digest
from src.effectiveness.service import build_effectiveness
from src.judges.llm import CachedChat, build_chat_backend
from src.judges.rubric_scoring import ChatRubricScorer, JevRubricScorer, RubricScorer
from src.training.backend import CheckpointMetadata
from src.training.reward_pipeline import load_reward_spec

from .dataset import grpo_rows
from src.agentclinic.protocol import ProtocolSpec, protocol_spec

from .environment import ClinicEnvironment, turn_samples
from .reward import SerlReward

NO_RUBRICS = {"rubrics": [], "rubric_set_version": None}


def iteration_dir(config: dict, root: Path, iteration: int) -> Path:
    return root / config["output_dir"] / f"iter{iteration}"


def trajectories_path(config: dict, root: Path, iteration: int) -> Path:
    """TRAIN trajectories of theta_iteration: the base Agent's for 0, else those of the previous checkpoint."""
    return (root / config["output_dir"] / "base" if iteration == 0 else iteration_dir(config, root, iteration - 1)) \
        / "trajectories.jsonl"


def spec_of(source: dict) -> ProtocolSpec:
    """The collaboration protocol of a label / trajectory run's config."""
    return protocol_spec(source.get("protocol"))


def training_stages(config: dict, spec: Optional[ProtocolSpec] = None) -> tuple:
    """The Agent turns trained as GRPO completions (training_samples.stages; every turn of the protocol by default)."""
    allowed = (spec or spec_of(config.get("resolved_environment") or {})).agent_stages()
    stages = tuple((config.get("training_samples") or {}).get("stages") or allowed)
    if not stages or set(stages) - set(allowed):
        raise IntegrationUnavailable(f"training_samples.stages must be a non-empty subset of {allowed}")
    return stages


def rubric_set_template(config: dict) -> Optional[str]:
    """Path of the rubric set the reward uses."""
    registry = config.get("rubric_registry") or {}
    if registry.get("schedule", "evolving") == "fixed":
        if not registry.get("fixed_rubric_set"):
            raise IntegrationUnavailable("rubric_registry.schedule = fixed needs rubric_registry.fixed_rubric_set")
        return registry["fixed_rubric_set"]
    return config["rubrics"].get("rubric_set")


def load_rubric_set(config: dict, root: Path, iteration: int) -> dict:
    """Active rubrics of the iteration; rubric_set is a path with an {iteration} placeholder."""
    template = rubric_set_template(config)
    if ((config.get("reward") or {}).get("weights") or {}).get("appropriateness", 1) == 0:
        return NO_RUBRICS  # an ablation without Appropriateness reads no rubric set
    if not template:
        raise IntegrationUnavailable("w_A > 0 requires rubrics.rubric_set")
    relative = Path(template.format(iteration=iteration))
    in_run = root / config["output_dir"] / relative  # this run's own rubric set; else a path from the repository root
    result = json.loads((in_run if in_run.is_file() else root / relative).read_text(encoding="utf-8"))
    if result.get("fixture"):
        raise IntegrationUnavailable("fixture rubrics are for integration checks only; GRPO requires validated research rubrics")
    registry = config.get("rubric_registry") or {}
    unfiltered = registry.get("schedule") == "no_filtering"
    if unfiltered != (result.get("rubric_schedule") == "no_filtering"):
        raise IntegrationUnavailable(f"{relative} does not belong to rubric_registry.schedule = {registry.get('schedule')}")
    cap = None if unfiltered else registry.get("max_active_categories")  # the unfiltered set is not capped
    categories = [r.get("category_id", r["rubric_id"]) for r in result["rubrics"]]
    gap_method = (registry.get("selection") or {}).get("method") == "effectiveness_gap"
    if gap_method:  # weights are effectiveness gaps (Ours, Static) or 1 per raw candidate (No-filtering): never zero or negative
        from src.judges.effectiveness_gap import assert_positive_weights

        problem = assert_positive_weights(result)
        if problem:
            raise IntegrationUnavailable(f"{relative}: {problem} (a rubric with a non-positive effectiveness gap is removed, not weighted)")
    if cap is not None and len(set(categories)) > cap:
        raise IntegrationUnavailable(f"{relative} holds {len(set(categories))} active categories; "
                                     f"rubric_registry.max_active_categories is {cap}")
    if (cap is not None or (gap_method and not unfiltered)) and len(set(categories)) != len(categories):
        raise IntegrationUnavailable(f"{relative} holds more than one scoring rubric for a category; "
                                     "the scorer gets one canonical rubric per active category")
    return result


def build_scorer(config: dict, root: Path) -> RubricScorer:
    section = config["rubrics"]
    settings = section["scorer"]
    if settings["backend"] == "jev":
        return JevRubricScorer(settings["name"], settings)
    prompt = (root / section["prompt"]).read_text(encoding="utf-8")
    if section.get("prompt_addendum"):
        prompt += (root / section["prompt_addendum"]).read_text(encoding="utf-8")
    chat = CachedChat(build_chat_backend(settings), root / config["output_dir"] / f"cache_scorer_{settings['name']}.jsonl")
    return ChatRubricScorer(settings["name"], chat, system_prompt=prompt, prompt_version=section["prompt_version"],
                            params=settings.get("params", {}), max_rubrics_per_call=settings.get("max_rubrics_per_call"))


def effectiveness_settings(config: dict, iteration: int) -> dict:
    """reward.effectiveness for an iteration."""
    section = dict(config["reward"].get("effectiveness") or {})
    override = (section.get("checkpoint_by_iteration") or {}).get(str(iteration))
    return dict(section, checkpoint=override) if override else section


def build_provider(config: dict, iteration: int = 0, root: Optional[Path] = None):
    """The reward's Effectiveness predictor (src/effectiveness/service.py); None when w_E = 0. Nothing here
    falls back: a missing service, a mismatching model or a CPU placement is an error."""
    return build_effectiveness(effectiveness_settings(config, iteration), config["reward"]["weights"]["effectiveness"], root)


def previous_adapter(config: dict, root: Path, iteration: int) -> Optional[Path]:
    """The adapter of theta_iteration: none for the base Agent, else the one iteration - 1 trained."""
    if iteration == 0:
        return None
    adapter = iteration_dir(config, root, iteration - 1) / "adapter"
    if not (adapter / "adapter_config.json").is_file():
        raise IntegrationUnavailable(f"iteration {iteration} continues from {adapter}, which does not exist")
    return adapter


def a1_states(config: dict, root: Path, iteration: int) -> dict[str, dict]:
    """{history_id: theta_iteration's elicited state before A1} from its TRAIN trajectories."""
    return {t["history_id"]: {k: v for k, v in t["turns"][1]["pre_response_state"].items()
                              if k != "clinical_information_available_to_agent"}
            for t in read_jsonl(trajectories_path(config, root, iteration), missing_ok=True)
            if len(t["turns"]) > 1 and t["turns"][1]["speaker"] == "agent"}


def library_versions() -> dict:
    import accelerate
    import peft
    import torch
    import transformers
    import trl

    return {name: module.__version__ for name, module in
            (("trl", trl), ("transformers", transformers), ("peft", peft), ("accelerate", accelerate), ("torch", torch))}


def base_trajectories(config: dict, root: Path, *, policy=None, judge_chat=None) -> dict:
    """theta_0's trajectories of trajectories.split from the label run's prepared collaborations."""
    if trajectories_path(config, root, 0).exists():
        raise IntegrationUnavailable(f"{trajectories_path(config, root, 0)} exists; base trajectories are written once per run")
    source_file = config["environment"].get("base_trajectories")
    if source_file:  # the main protocol: the base Agent's collaborations were generated once, with their states; they are taken as they are
        split, identity = config["trajectories"]["split"], protocol_identity(config, root)
        rows = [dict(t, source_protocol_identity=t.get("protocol_identity"), protocol_identity=identity)
                for t in read_jsonl(root / source_file) if t["split"] == split]
        if not rows:
            raise IntegrationUnavailable(f"{source_file} holds no {split} collaboration")
        path = trajectories_path(config, root, 0)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_new(path, "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
        return dict(summary(rows), split=split, checkpoint_id="base", path=str(path.relative_to(root)), source=source_file)
    model, tokenizer = policy or load_policy(config["policy"], None, trainable=False, device=config["devices"]["policy"])
    clinic = build_clinic(config, root, model, tokenizer, doctor=lambda system, user: "", judge_chat=judge_chat)
    split, identity = config["trajectories"]["split"], protocol_identity(config, root)
    prepared = [row for row in frozen_histories(config, root, clinic.cases)
                if clinic.cases[row["case_id"]]["split"] == split]
    with clinic.generation(), ThreadPoolExecutor(config["rollout"]["workers"]) as pool:
        rows = list(pool.map(lambda row: from_prepared(clinic.cases[row["case_id"]], row, clinic.runtime, clinic.judge,
                                                       iteration=0, checkpoint_id="base"), prepared))
    path = trajectories_path(config, root, 0)
    path.parent.mkdir(parents=True, exist_ok=True)
    for row in rows:
        row["protocol_identity"] = identity
    write_new(path, "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    return dict(summary(rows), split=split, checkpoint_id="base", path=str(path.relative_to(root)))


def train_iteration(config: dict, root: Path, iteration: int, *, resume: bool = False, policy=None,
                    doctor=None, judge_chat=None, scorer=None, provider=None) -> dict:
    """Train theta_iteration with R_iteration and save the adapter and its metadata. The keyword
    arguments replace the configured models (tests and smoke runs)."""
    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer

    from .callbacks import JsonlLog

    out = iteration_dir(config, root, iteration)
    if (out / "checkpoint.json").exists():
        raise IntegrationUnavailable(f"iteration {iteration} of this run is already trained ({out / 'checkpoint.json'}); "
                                     "results are never overwritten, start a new run to train it again")
    if policy is None and provider is None:  # the configured models: both must be placed, on two GPUs
        problems = training_device_problems(config)
        if problems:
            raise IntegrationUnavailable("GRPO training cannot start: " + "; ".join(problems))
    out.mkdir(parents=True, exist_ok=True)
    spec = load_reward_spec(config)
    initial = previous_adapter(config, root, iteration)
    rubric_set = load_rubric_set(config, root, iteration)
    model, tokenizer = policy or load_policy(config["policy"], initial, trainable=True, device=config["devices"]["policy"])

    check_identity(read_jsonl(trajectories_path(config, root, iteration), missing_ok=True),
                   protocol_identity(config, root), trajectories_path(config, root, iteration))
    if policy is None:
        import torch

        if not torch.cuda.is_available():
            raise IntegrationUnavailable("GRPO training of the configured policy needs a GPU; none is visible")
    clinic = build_clinic(config, root, model, tokenizer, doctor=doctor, judge_chat=judge_chat)
    histories = frozen_histories(config, root, clinic.cases)
    protocol = spec_of(clinic.source)
    stages, split = training_stages(config, protocol), config["environment"]["train_split"]
    own = read_jsonl(trajectories_path(config, root, iteration), missing_ok=True)  # theta_iteration's collaborations
    second = turn_samples(own, clinic.cases, split, stages)  # one sample per later Agent turn that was reached
    if set(stages) - {"A1"} and not second:
        raise IntegrationUnavailable(f"A2 training samples come from {trajectories_path(config, root, iteration)}, "
                                     "which holds no collaboration that reached D2")
    environment = ClinicEnvironment(clinic.cases, histories, clinic.runtime, clinic.judge,
                                    a1_states=a1_states(config, root, iteration), a2=second,
                                    collaboration_states={t["history_id"]: t.get("collaboration_state") for t in own},
                                    max_attempts=config["rollout"]["max_attempts"], spec=protocol)
    if spec.weights["appropriateness"] > 0 and rubric_set["rubrics"] and scorer is None:
        scorer = build_scorer(config, root)
    if spec.weights["effectiveness"] > 0 and provider is None:
        provider = build_provider(config, iteration, root)
    reward = SerlReward(environment, spec, rubric_set=rubric_set, scorer=scorer, provider=provider,
                        log_path=out / "reward_log.jsonl", iteration=iteration,
                        scored_stages=tuple(config["reward"]["scored_stages"]),
                        effectiveness_stages=tuple(config["reward"]["effectiveness"]["stages"]),
                        workers=config["rollout"]["workers"], generation=clinic.generation,
                        max_failed_share=config["rollout"].get("max_failed_share", 0.5),
                        no_applicable_rubric_policy=config["reward"].get("no_applicable_rubric_policy", "group_mean"),
                        credit_assignment=config["reward"].get("credit_assignment", "turn_local"),
                        rubric_weight_strategy=(config.get("rubric_importance") or {}).get("strategy", "failure_frequency"))

    rows = grpo_rows(clinic.cases, histories, clinic.runtime.agent_system, split=split, a2=second, stages=stages, spec=protocol)
    arguments = GRPOConfig(output_dir=str(out / "trainer"), report_to="none",
                           chat_template_kwargs={"enable_thinking": False}, **config["grpo"])
    trainer = GRPOTrainer(model=model, reward_funcs=reward, args=arguments, train_dataset=Dataset.from_list(rows),
                          processing_class=tokenizer, callbacks=[JsonlLog(out / "train_log.jsonl", iteration)])
    checkpoints = sorted((out / "trainer").glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
    if resume and checkpoints:
        record_resume(out, checkpoints[-1])
        allow_own_checkpoint_state()
    result = trainer.train(resume_from_checkpoint=bool(resume and checkpoints))
    trainer.save_model(str(out / "adapter"))

    identity = reward.identity()
    metadata = CheckpointMetadata(
        checkpoint_id=f"{config['run_id']}-iter{iteration}", model_id=config["policy"]["model_id"],
        model_revision=config["policy"]["revision"], iteration=iteration,
        rubric_version=str(rubric_set.get("rubric_set_version")), reward_version=digest(identity)[:16])
    result = {**asdict(metadata), "adapter": str((out / "adapter").relative_to(root)),
              "initial_adapter": str(initial.relative_to(root)) if initial else None,
              "train_prompts": len(rows), "global_step": trainer.state.global_step,
              "train_prompts_by_stage": {stage: sum(r["stage"] == stage for r in rows) for stage in stages},
              "training_loss": result.training_loss, "reward": identity, "grpo": config["grpo"],
              "protocol_identity": protocol_identity(config, root), "seed": config["grpo"].get("seed"),
              "rubric_set": (rubric_set_template(config) or "").format(iteration=iteration) or None,
              "rubric_schedule": (config.get("rubric_registry") or {}).get("schedule", "evolving"),
              "effectiveness_checkpoint": effectiveness_settings(config, iteration).get("checkpoint"),
              "library_versions": library_versions()}
    write_new(out / "run_config.json", json.dumps(config, indent=2, ensure_ascii=False, default=str) + "\n")
    write_new(out / "checkpoint.json", json.dumps(result, indent=2, ensure_ascii=False, default=str) + "\n")
    return result


def allow_own_checkpoint_state() -> None:
    """transformers refuses torch.load on torch < 2.6 (CVE-2025-32434), and the optimizer, scheduler and RNG state of a
    trainer checkpoint are torch-pickled files, so a resume would fail with the installed torch.
    """
    import transformers.trainer

    from transformers.utils import is_torch_greater_or_equal

    if not is_torch_greater_or_equal("2.6"):
        transformers.trainer.check_torch_load_is_safe = lambda: None


def record_resume(out: Path, checkpoint: Path) -> dict:
    """Before training continues from a trainer checkpoint (policy adapter, optimizer, scheduler, global step, RNG
    state; the trainer replays the sampler up to that step): the steps the interrupted process had logged after that ...
    """
    import socket
    import time

    step = json.loads((checkpoint / "trainer_state.json").read_text(encoding="utf-8"))["global_step"]
    stamp = time.strftime("%Y%m%d-%H%M%S")
    moved = {}
    # reward rows carry the step that was being computed (0-based); train-log rows the number of completed steps
    for name, keep in (("reward_log.jsonl", lambda r: r.get("global_step") is None or r["global_step"] < step),
                       ("reward_timing.jsonl", lambda r: r.get("global_step") is None or r["global_step"] < step),
                       ("train_log.jsonl", lambda r: r.get("global_step", 0) <= step)):
        path = out / name
        if not path.is_file():
            continue
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        replayed = [r for r in rows if not keep(r)]
        if replayed:
            (out / f"replayed_{stamp}").mkdir(exist_ok=True)
            (out / f"replayed_{stamp}" / name).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in replayed), encoding="utf-8")
            path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows if keep(r)), encoding="utf-8")
            moved[name] = len(replayed)
    try:
        import torch

        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:  # noqa: BLE001 - the record is informational
        gpu = None
    row = {"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "resumed_from": checkpoint.name, "global_step": step,
           "host": socket.gethostname(), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "gpu": gpu,
           "rows_moved_to_replayed": moved}
    with (out / "resume_log.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")
    return row


def checkpoint_trajectories(config: dict, root: Path, iteration: Optional[int], *, split: Optional[str] = None,
                            output: Optional[Path] = None, policy=None, doctor=None, judge_chat=None, repeats: int = 1) -> dict:
    """Natural collaborations of a checkpoint: the adapter trained in `iteration`, or the base Agent for None."""
    checkpoint_id = "base" if iteration is None else \
        json.loads((iteration_dir(config, root, iteration) / "checkpoint.json").read_text(encoding="utf-8"))["checkpoint_id"]
    adapter = None if iteration is None else iteration_dir(config, root, iteration) / "adapter"
    model, tokenizer = policy or load_policy(config["policy"], adapter, trainable=False, device=config["devices"]["policy"])
    clinic = build_clinic(config, root, model, tokenizer, doctor=doctor, judge_chat=judge_chat)
    split = split or config["trajectories"]["split"]
    output = output or iteration_dir(config, root, iteration) / "trajectories.jsonl"
    identity = protocol_identity(config, root)
    check_identity(read_jsonl(output, missing_ok=True), identity, output)
    workers = config["rollout"]["workers"]
    with clinic.generation():
        histories = [row for row in frozen_histories(config, root, clinic.cases)
                     if clinic.cases[row["case_id"]]["split"] == split] \
            or opening_histories(clinic, split, output.parent / f"histories_{split}.jsonl", workers)
        for repeat in range(repeats):
            rows = generate_trajectories(clinic.cases, histories, clinic.runtime, clinic.judge, output,
                                         suffix=checkpoint_id if repeats == 1 else f"{checkpoint_id}:r{repeat}", workers=workers,
                                         max_attempts=config["rollout"]["max_attempts"],
                                         iteration=0 if iteration is None else iteration + 1, checkpoint_id=checkpoint_id,
                                         protocol_identity=identity, spec=spec_of(clinic.source),
                                         **({"rollout_index": repeat} if repeats > 1 else {}))
    return dict(summary(rows), split=split, checkpoint_id=checkpoint_id, repeats=repeats)
