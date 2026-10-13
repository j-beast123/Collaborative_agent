"""The main SERL configuration (AgentClinic two-turn protocol): loading, overrides and validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional

from src.common.config import IntegrationUnavailable, read_config
from src.common.run_dir import create_run

PIPELINE = "serl_agentclinic_main"
PROTOCOL = "agentclinic_two_turn"
AGENT_TURNS = ["A1", "A2"]


def protocol_of(config: dict, root: Path):
    """The collaboration protocol of a main config: that of its environment config (the label / trajectory run)."""
    from src.agentclinic.protocol import TWO_TURN, protocol_spec

    path = root / str((config.get("environment") or {}).get("config") or "")
    return protocol_spec(read_config(path).get("protocol")) if path.is_file() else TWO_TURN


def override(config: dict, assignments: Iterable[str]) -> dict:
    """--set a.b=JSON overrides; the key must exist."""
    for assignment in assignments:
        path, _, value = assignment.partition("=")
        *parents, leaf = path.split(".")
        section = config
        for name in parents:
            section = section[name]
        if leaf not in section:
            raise SystemExit(f"--set {path}: no such config value")
        section[leaf] = json.loads(value)
    return config


def problems_of(config: dict, root: Path) -> list[str]:
    found = []

    def need(condition: bool, message: str) -> None:
        if not condition:
            found.append(message)

    need(config.get("pipeline") == PIPELINE, f"pipeline must be {PIPELINE!r} (a legacy or unrelated config was given)")
    collaboration = config.get("collaboration") or {}
    spec = protocol_of(config, root)
    turns = list(spec.agent_stages())
    need(collaboration.get("protocol") == spec.name, f"collaboration.protocol must be {spec.name} (the protocol of environment.config)")
    need(collaboration.get("max_rounds") == spec.max_turns, f"collaboration.max_rounds must be {spec.max_turns}")
    if spec.early_stopping:
        need(bool((config.get("environment") or {}).get("base_trajectories")),
             "environment.base_trajectories is required: the base Agent's collaborations under this protocol")
    need(collaboration.get("final_decision_role") == "doctor", "collaboration.final_decision_role must be doctor")
    need((config.get("scoring") or {}).get("include_initial_turn") is True, "scoring.include_initial_turn must be true (A1 is scored)")
    need(type(config.get("seed")) is int, "seed must be an integer")
    need(bool(config.get("output_root")), "output_root is required (runs are written to <output_root>/<run_id>)")
    need(str((config.get("devices") or {}).get("policy") or "").startswith("cuda"), "devices.policy must name a GPU (cuda:N)")

    environment = config.get("environment") or {}
    for key in ("config", "histories", "state_prompt"):
        need(bool(environment.get(key)) and (root / str(environment.get(key))).is_file(), f"environment.{key} must be an existing file")
    need(environment.get("train_split") == "train", "environment.train_split must be train")
    need((config.get("trajectories") or {}).get("split") == "train", "trajectories.split must be train (rubric discovery uses TRAIN)")

    validation = config.get("validation") or {}
    reruns, keep = validation.get("num_reruns"), validation.get("min_successes")
    need(type(reruns) is int and type(keep) is int and 1 <= keep <= reruns, "validation needs integers 1 <= min_successes <= num_reruns")
    manager = (config.get("roles") or {}).get("rubric_manager") or {}
    registry = config.get("rubric_registry") or {}
    cap = registry.get("max_active_categories")
    selection = registry.get("selection") or {}
    gap_method = selection.get("method") == "effectiveness_gap"
    need(selection.get("method", "failure_frequency_top_n") in ("failure_frequency_top_n", "effectiveness_gap"),
         "rubric_registry.selection.method must be effectiveness_gap (the main method) or failure_frequency_top_n")
    need(gap_method or not spec.early_stopping,
         "the main protocol uses rubric_registry.selection.method = effectiveness_gap (per-iteration rubrics, no persistent active set)")
    if gap_method:
        need(cap is None and selection.get("engineering_safety_cap") is None,
             "effectiveness_gap selection has no cap: rubric_registry.max_active_categories and selection.engineering_safety_cap must be null")
        need(selection.get("carry_over") is False, "rubric_registry.selection.carry_over must be false (no rubric of an earlier iteration is reused)")
        need((selection.get("high_score"), selection.get("low_score")) == (2, 0),
             "rubric_registry.selection: high adherence is score 2 and low adherence is score 0")
        need(type(selection.get("min_group_n")) is int and selection.get("min_group_n") >= 1,
             "rubric_registry.selection.min_group_n must be an integer >= 1 (both adherence groups need responses)")
        need(selection.get("population") in ("policy_train_turns",), "rubric_registry.selection.population must be policy_train_turns")
    else:
        need(type(cap) is int and cap >= 1, "rubric_registry.max_active_categories must be a positive integer (cap of the active set)")
    need(manager.get("max_active_rubrics") is None, "roles.rubric_manager.max_active_rubrics must be null "
         "(the cap of the main pipeline is rubric_registry.max_active_categories)")
    need(registry.get("schedule") in ("evolving", "fixed", "no_filtering"),
         "rubric_registry.schedule must be evolving, fixed or no_filtering")
    initial = registry.get("initial_rubrics")
    need(initial is None or (root / str(initial) / "failure_analyses.jsonl").is_file(),
         "rubric_registry.initial_rubrics must be the iter0/rubrics directory of the run whose iteration-0 rubric step is shared")
    need(registry.get("schedule") != "fixed" or (bool(registry.get("fixed_rubric_set"))
                                                 and (root / str(registry.get("fixed_rubric_set"))).is_file()),
         "rubric_registry.schedule = fixed needs rubric_registry.fixed_rubric_set: the existing R_0 file of the "
         "Evolving run (a fixed run generates no rubrics)")
    stages = (config.get("training_samples") or {}).get("stages")
    need(isinstance(stages, list) and bool(stages) and set(stages) <= set(turns) and len(set(stages)) == len(stages),
         f"training_samples.stages must be a non-empty subset of {turns} (every turn for the main method)")
    need(manager.get("retention_policy") == ("current_only" if gap_method else "reactivation"),
         "roles.rubric_manager.retention_policy must be current_only with effectiveness_gap selection (reactivation otherwise)")
    need((config.get("rubric_importance") or {}).get("strategy") in (("effectiveness_gap",) if gap_method else ("failure_frequency", "uniform")),
         "rubric_importance.strategy must be effectiveness_gap with effectiveness_gap selection (failure_frequency or uniform otherwise)")
    aggregation = config.get("rubric_aggregation") or {}
    need(aggregation.get("method") == "weighted_softmin" and isinstance(aggregation.get("temperature"), (int, float))
         and aggregation.get("temperature", 0) > 0,
         "rubric_aggregation must be {method: weighted_softmin, temperature > 0} (the rubrics of one response)")

    reward = config.get("reward") or {}
    weights = reward.get("weights") or {}
    need(set(weights) == {"outcome", "appropriateness", "effectiveness"} and all(
        isinstance(w, (int, float)) and w >= 0 for w in weights.values()) and sum(weights.values()) > 0,
        "reward.weights needs non-negative outcome, appropriateness, effectiveness with a positive sum")
    need(isinstance(reward.get("softmin_temperature"), (int, float)) and reward.get("softmin_temperature", 0) > 0,
         "reward.softmin_temperature must be positive")
    need((config.get("grpo") or {}).get("seed") == config.get("seed"), "grpo.seed must equal seed")
    need(reward.get("scored_stages") == turns, f"reward.scored_stages must be {turns}")
    need(reward.get("no_applicable_rubric_policy") == "group_mean",
         "reward.no_applicable_rubric_policy must be group_mean (a turn without an applicable rubric gets its group's mean R_A of that turn)")
    need(reward.get("credit_assignment") == "turn_local",
         "reward.credit_assignment must be turn_local (a response is rewarded for its own turn; no soft-min across A1 and A2)")
    need(reward.get("missing_component_policy") == "omit",
         "reward.missing_component_policy must be omit (an unavailable component adds nothing; the other weights are not rescaled)")
    effectiveness = reward.get("effectiveness") or {}
    need(effectiveness.get("transform") == "identity", "main effectiveness.transform must be identity (raw calibrated delta)")
    if not spec.early_stopping:  # the fixed two-turn protocol: Effectiveness v1 covers A1 only
        need(effectiveness.get("available") == {"A1": True, "A2": False}, "only the A1 predictor exists; A2 must be unavailable")
    devices = config.get("devices") or {}
    need(devices.get("placement") in ("concurrent", "sequential"), "devices.placement must be concurrent or sequential")
    need(str(devices.get("effectiveness_service") or "").startswith("cuda"), "devices.effectiveness_service must name a GPU (cuda:N)")
    need(set(devices.get("physical_gpus") or {}) == {"policy", "effectiveness_service"},
         "devices.physical_gpus must name policy and effectiveness_service (null until a GPU is assigned)")
    stages = effectiveness.get("stages") or []
    need(set(stages) <= set(turns) and (len(stages) > 0 or spec.early_stopping),
         f"reward.effectiveness.stages must be a subset of {turns} (empty only while this protocol has no predictor)")
    available = effectiveness.get("available") or {}
    need(set(available) == set(turns) and all(type(v) is bool for v in available.values()),
         f"reward.effectiveness.available must say true/false for each of {turns}")
    need(sorted(stages) == sorted(t for t, ok in available.items() if ok),
         "reward.effectiveness.stages must be exactly the turns marked available")
    if weights.get("effectiveness", 0) > 0 and stages:
        need(effectiveness.get("provider") in ("grpc", "local"), "reward.effectiveness.provider must be grpc or local")
        need(bool(effectiveness.get("model_version")), "reward.effectiveness.model_version is required")
        need(bool(effectiveness.get("checkpoint")), "reward.effectiveness.checkpoint is required")
        need(bool(effectiveness.get("calibration")), "reward.effectiveness.calibration is required (calibrated probabilities)")
        need(isinstance(effectiveness.get("expected_temperature"), (int, float)), "reward.effectiveness.expected_temperature is required")
        need(effectiveness.get("transform") is not None, "reward.effectiveness.transform is required")
        if effectiveness.get("provider") == "grpc":
            need(bool(effectiveness.get("endpoint")), "reward.effectiveness.endpoint is required for provider grpc")
        if effectiveness.get("provider") == "local":
            need(str(effectiveness.get("device") or "").startswith("cuda") or bool(effectiveness.get("allow_cpu")),
                 "reward.effectiveness.device must name a GPU for provider local")
    return found


def stand_in_roles(config: dict, roles: Iterable[str]) -> list[str]:
    """Of the given roles, those whose decided model (role_models.<role>) is not connected yet
    (role_models.connected.<model> is not true): the code path behind them still holds a stand-in model.
    """
    models = config.get("role_models") or {}
    connected = models.get("connected") or {}
    return [f"{role} = {models[role]}" for role in roles if role in models and models[role] in connected
            and connected[models[role]] is not True]


def refuse_stand_ins(config: dict, roles: Iterable[str], what: str, allowed: bool) -> None:
    """Main artifacts are not made with a stand-in model unless that is asked for explicitly."""
    missing = stand_in_roles(config, roles)
    if missing and not allowed:
        raise SystemExit(f"{what}: the decided model of {', '.join(missing)} is not connected; the code would use its stand-in. "
                         "Connect the model (role_models.connected) or pass --allow-stand-ins for a wiring check.")


def training_device_problems(config: dict) -> list[str]:
    """Why GRPO training with the Effectiveness reward cannot run on the configured devices; empty when it can.
    The policy and the predictor are needed at the same time, each on its own physical GPU."""
    if (config.get("reward") or {}).get("weights", {}).get("effectiveness", 0) == 0:
        return []
    devices = config.get("devices") or {}
    gpus = devices.get("physical_gpus") or {}
    found = []
    if ((config.get("reward") or {}).get("effectiveness") or {}).get("stages") == []:
        return ["w_E > 0 but no Effectiveness predictor covers a turn of this protocol (reward.effectiveness.stages is empty): "
                "train the predictor of this protocol first, or run an ablation config with w_E = 0"]
    if devices.get("placement") != "concurrent":
        found.append("devices.placement must be concurrent: GRPO with w_E > 0 needs the policy and the Effectiveness "
                     "service at the same time (sequential placement is for the pipeline check only)")
    if gpus.get("policy") is None or gpus.get("effectiveness_service") is None or gpus.get("policy") == gpus.get("effectiveness_service"):
        found.append("devices.physical_gpus must name two different GPUs for policy and effectiveness_service "
                     f"(got {gpus}); the two Qwen3-8B models do not fit on one 24 GB GPU")
    return found


def validate_main_config(config: dict, root: Path) -> dict:
    found = problems_of(config, root)
    if found:
        raise IntegrationUnavailable("main config is not valid:\n  - " + "\n  - ".join(found))
    return config


def load_main_config(path, root: Path, assignments: Iterable[str] = ()) -> dict:
    from src.data.schema import digest
    config = validate_main_config(override(read_config(path), assignments), root)
    source = read_config(root / config["environment"]["config"])
    dependencies = {config["environment"]["state_prompt"]}
    dependencies.update(source.get("prompts", {}).values())
    for key in ("prompt", "prompt_addendum"):
        if config.get("rubrics", {}).get(key):
            dependencies.add(config["rubrics"][key])
    config["resolved_environment"] = source
    config["input_hashes"] = {p: digest((root / p).read_text()) for p in sorted(dependencies)}
    return config


def stored_run_config(config: dict, root: Path, run_id: str) -> dict:
    """A run that is continued keeps the configuration it was created with (its resolved_config.json), also when the
    config files were edited since: another role connected, a predictor added, a note changed.
    """
    import sys

    from src.common.run_dir import config_hash

    path = root / config["output_root"] / run_id / "resolved_config.json"
    if not path.is_file():
        return config
    stored = json.loads(path.read_text(encoding="utf-8"))
    if config_hash(stored) == config_hash(config):
        return stored
    changed = sorted(name for name in set(stored.get("input_hashes") or {}) | set(config.get("input_hashes") or {})
                     if (stored.get("input_hashes") or {}).get(name) != (config.get("input_hashes") or {}).get(name))
    if changed:
        raise IntegrationUnavailable(f"run {run_id} cannot be continued: prompt files changed since it was created: {changed}")
    print(f"run {run_id}: the config files changed since the run was created; it continues with its own resolved_config.json",
          file=sys.stderr)
    return stored


def open_run(config: dict, root: Path, *, run_id: Optional[str] = None, resume: bool = False, purpose: str = "",
             label: Optional[str] = None) -> dict:
    """The configuration bound to a run directory: run_id and output_dir (relative to root) are filled in.
    Without run_id a new run is created; with one, the existing run is continued (same configuration only)."""
    if run_id and resume:
        config = stored_run_config(config, root, run_id)
    run = create_run(root / config["output_root"], label or config["run_label"], config, root=root, seed=config["seed"],
                     purpose=purpose, run_id=run_id, resume=resume)
    return dict(config, run_id=run.name, output_dir=str(run.relative_to(root)))
