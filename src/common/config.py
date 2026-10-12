"""Offline configuration guards; unresolved research inputs stay unresolved."""
import json
from pathlib import Path

class IntegrationUnavailable(RuntimeError):
    pass


def local_model_path(path):
    """Where a configured model directory is on this machine."""
    import os

    text = os.path.expandvars(str(path)) if path is not None else path  # ${MODEL_ROOT}/... in the configs
    for pair in filter(None, (os.environ.get("MODEL_PATH_MAP") or "").split(",")):
        old, _, new = pair.partition("=")
        if text and old and (text == old or text.startswith(old.rstrip("/") + "/")):
            return new.rstrip("/") + text[len(old.rstrip("/")):]
    return text


def read_config(path) -> dict:
    """JSON config; "extends": "<file next to it>" loads that file first and overrides it key by key."""
    path = Path(path)
    config = json.loads(path.read_text(encoding="utf-8"))
    parent = config.pop("extends", None)
    return _merge(read_config(path.parent / parent), config) if parent else config


def _merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        merged[key] = _merge(base[key], value) if isinstance(value, dict) and isinstance(base.get(key), dict) else value
    return merged

def load_config(path):
    config = json.loads(Path(path).read_text())
    if config.get("mode") not in ("smoke", "production"):
        raise ValueError("mode must be smoke or production")
    predictor = config.get("predictor", {})
    if predictor.get("training_mode") not in ("frozen_backbone_scalar_head", "lora_scalar_head"):
        raise ValueError("Invalid predictor training_mode")
    if predictor["training_mode"] == "lora_scalar_head" and not predictor.get("pilot_decision"):
        raise ValueError("LoRA requires a recorded pilot decision")
    if config["mode"] == "smoke" and config.get("dataset_adapter") != "synthetic_fixture":
        raise ValueError("Offline smoke requires the synthetic fixture adapter")
    if config["mode"] == "production" and config.get("dataset_adapter") == "synthetic_fixture":
        raise ValueError("Synthetic adapter is forbidden in production")
    models = config.get("models")
    if not isinstance(models, dict) or not models:
        raise ValueError("Agent model registry is required")
    if any(m.get("role") != "agent_backbone" or type(m.get("enabled")) is not bool for m in models.values()):
        raise ValueError("models must contain agent_backbone entries with boolean enabled")
    if predictor.get("role") != "effectiveness_predictor":
        raise ValueError("Predictor must have its own effectiveness_predictor role")
    roles = {"behavior_judge", "rubric_scorer", "evidence_judge", "rubric_generator", "collaborator_simulator"}
    services = config.get("services", {})
    if set(services) != roles or any(v.get("role") != k or v.get("backend") != "llm" for k,v in services.items()):
        raise ValueError("Core services must declare separate LLM roles")
    # Jev is the score LLM candidate (compared against GPT); it scores rubric
    # adherence only and is never the effectiveness predictor or an agent.
    experiment = config.get("optional_experiments", {}).get("jev_judge", {})
    if type(experiment.get("enabled")) is not bool or experiment.get("role") != "rubric_scorer_candidate":
        raise ValueError("Jev must be declared as a rubric_scorer_candidate with boolean enabled")
    return config

def require_execution_ready(config):
    required = ("dataset_adapter", "dataset_version", "split_manifest_hash",
                "training_backend", "reward_protocol", "continuation_protocol")
    missing = [k for k in required if not config.get(k)]
    for name, model in config.get("models", {}).items():
        if model.get("enabled", True) and (not model.get("model_id") or not model.get("revision")):
            missing.append("models." + name + ".model_id/revision")
    for field in ("model_id", "revision"):
        if not config.get("predictor", {}).get(field):
            missing.append("predictor." + field)
    for name, service in config.get("services", {}).items():
        if not service.get("model_id") or not service.get("revision"):
            missing.append("services." + name + ".model_id/revision")
    if config.get("mode") != "production" or missing:
        raise IntegrationUnavailable("Real execution unavailable; resolve: " + ", ".join(missing or ["production mode"]))
    raise IntegrationUnavailable("Phase 0/1 has no real execution backend; integrate the verified backend later")
