"""Build effectiveness labels by continuation rollouts (TRAIN or DEV only)."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from src.common.config import IntegrationUnavailable
from src.common.io import read_jsonl
from src.data.schema import identity
from src.data.split import read_manifest
from src.effectiveness.labels import (build_pair_labels, collect_histories, config_hash, label_metadata,
                                      load_candidates, original_candidates, resolve_labels_section)
from src.effectiveness.train import load_predictor_config

ROOT = Path(__file__).resolve().parents[1]


def check_split(config: dict, trajectories: list[dict], split: str) -> None:
    data = config.get("data", {})
    if config["mode"] != "production":
        return
    if not data.get("split_manifest") or not data.get("source_dataset"):
        raise IntegrationUnavailable("Label building needs data.split_manifest and data.source_dataset (TODO)")
    manifest = read_manifest(data["split_manifest"])
    for t in trajectories:
        member = manifest["members"].get(identity(data["source_dataset"], str(t["case_id"])))
        if member is None or member["split"] != split:
            raise ValueError(f"case {t['case_id']} is not in the {split} split; EVAL cases never get labels")


def plan(histories, candidates, k) -> dict:
    with_candidates = {c.history_id for c in candidates}
    before = len([h for h, _ in histories if h["history_id"] in with_candidates])
    return {"histories": before, "pairs": len(candidates), "k": k,
            "before_rollouts": before * k, "after_rollouts": len(candidates) * k,
            "total_rollouts": (before + len(candidates)) * k}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--trajectories", required=True)
    p.add_argument("--split", choices=("train", "dev"), required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--candidates", action="append", default=[], help="extra candidate responses JSONL (repeatable)")
    p.add_argument("--only-candidate-histories", action="store_true",
                   help="label only histories that have a candidate in the files (e.g. a signal pilot)")
    p.add_argument("--plan-only", action="store_true", help="print the rollout budget without running anything")
    a = p.parse_args()

    config = load_predictor_config(a.config)  # production requires a fully resolved ContinuationSpec
    spec = config["continuation_spec"]
    labels = resolve_labels_section(config)
    trajectories = read_jsonl(a.trajectories)
    check_split(config, trajectories, a.split)
    histories = collect_histories(trajectories, turn_scope=labels.get("turn_scope", "first_collaboration_turn"),
                                  include_initial_turn=labels["include_initial_turn"],
                                  max_histories=labels.get("max_histories"))
    candidates = original_candidates(trajectories, histories) if labels.get("include_original_response", True) else []
    for path in list(labels.get("candidate_files") or []) + a.candidates:
        candidates += load_candidates(path, require_compliance=bool(labels.get("require_compliance")))
    if a.only_candidate_histories:
        listed = {c.history_id for c in candidates if c.response_id != "original"}
        histories = [(h, gt) for h, gt in histories if h["history_id"] in listed]
        candidates = [c for c in candidates if c.history_id in listed]
    budget = plan(histories, candidates, spec.k)
    print(json.dumps({"plan": budget}, indent=2))
    if a.plan_only:
        return

    from src.collaboration.runtime import build_runtime
    import torch

    decoding = spec.decoding or {}
    parallel = int(labels.get("parallel_rollouts") or 1)
    runtime = build_runtime(
        agent_model=spec.agent_model.checkpoint or spec.agent_model.model_id,
        doctor_model=spec.doctor_model.model_id, reconstructor_model=spec.reconstructor_model.model_id,
        max_tokens=(decoding.get("agent") or {}).get("max_new_tokens", 512), root=ROOT,
        agent_decoding=decoding.get("agent"), doctor_decoding=decoding.get("doctor"),
        agent_batch_size=int(labels.get("agent_batch_size") or 1) if parallel > 1 else 1,
        doctor_prompt=spec.doctor_prompt, continuation_guideline=spec.continuation_guideline,
    )

    def set_seed(seed: int) -> None:
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    seed_control = {"agent": "torch.manual_seed per rollout (local sampling)" if parallel == 1 else
                    f"not seeded per rollout: {parallel} concurrent rollouts share the sampling RNG (batched)",
                    "doctor": "not controlled (API sampling); paired seeds are not assumed for the doctor"}
    counts = build_pair_labels(histories, candidates, runtime, spec, split=a.split, output_path=a.output,
                               metadata=label_metadata(spec, config_hash(config), seed_control),
                               set_seed=set_seed if parallel == 1 else None,
                               max_rollout_attempts=labels.get("max_rollout_attempts", 1),
                               parallel_rollouts=parallel)
    print(json.dumps({"plan": budget, "result": counts}, indent=2))


if __name__ == "__main__":
    main()
