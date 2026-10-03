"""Two-case multi-turn GRPO smoke test for the MedMCQA collaboration.

The script samples four full conversations per case, scores every Agent turn
with Jev, aggregates rubric adherence with rubric-level and dialogue-level
soft-mins, and performs one LoRA policy-gradient/GRPO-style update. Doctor
messages never enter the policy loss. Exact prompt/completion token IDs and
generation log-probabilities are stored in the trajectory JSONL.

This is deliberately a small smoke path, not the final high-throughput
trainer. It exists to verify that full multi-turn rollout, reward aggregation,
token traces, KL regularization, and one parameter update are connected. With
`--trajectories`, it reuses a saved JSONL collection without calling Doctor or
Jev again; stored completion log-probabilities are the old-policy values.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from run_medmcqa_full_trajectory_smoke import (
        generate_trajectory,
        load_cases,
        resolve_generation_top_k,
        score_agent_turn,
        visible_case,
    )
    from run_medmcqa_grpo_pilot import load_rubrics, rubric_set_metadata
except ModuleNotFoundError:
    from scripts.run_medmcqa_full_trajectory_smoke import (
        generate_trajectory,
        load_cases,
        resolve_generation_top_k,
        score_agent_turn,
        visible_case,
    )
    from scripts.run_medmcqa_grpo_pilot import load_rubrics, rubric_set_metadata
from src.judges.jev import JevClient
from src.judges.reward import soft_min, weighted_soft_min
from src.judges.rubric_schema import normalize_score
from src.judges.rubric_scoring import TWO_STAGE_JEV_PROTOCOL


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/interim/medmcqa/medicine_train_random300_seed20261004.jsonl")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--num-cases", type=int, default=None)
    parser.add_argument("--rollouts-per-case", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--output", default="outputs/medmcqa_full_grpo_smoke")
    parser.add_argument(
        "--trajectories",
        default=None,
        help="reuse saved full-trajectory JSONL; skips Doctor/Jev generation and uses stored old log-probs",
    )
    parser.add_argument("--model", default="/home/kai/models/Qwen3-8B")
    parser.add_argument("--rubric-set", default="configs/rubrics/rubric_guidance_set_11_observable_v2.json")
    parser.add_argument("--doctor-model", default="google/gemma-3-12b-it")
    parser.add_argument("--jev-endpoint", default="https://openrouter.ai/api/alpha/decisions")
    parser.add_argument("--jev-model", default="typesafe/jev-1.13")
    parser.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    parser.add_argument("--max-completion-length", type=int, default=400)
    parser.add_argument("--max-tokens-doctor", type=int, default=512)
    parser.add_argument("--generation-temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="generation top-k; when omitted, use the model generation_config value",
    )
    parser.add_argument("--rubric-temperature", type=float, default=0.5)
    parser.add_argument("--rubric-weight", type=float, default=0.3)
    parser.add_argument("--outcome-weight", type=float, default=0.4)
    parser.add_argument("--beta", type=float, default=0.04)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument(
        "--pre-update-ratio-tolerance",
        type=float,
        default=1e-3,
        help="integrity tolerance for the initial new/old policy ratio; not a training reward threshold",
    )
    parser.add_argument("--jev-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--allow-unvalidated-rubrics", action="store_true")
    parser.add_argument("--allow-unversioned-rubric-scores", action="store_true")
    parser.add_argument(
        "--allow-legacy-rubric-scoring-protocol",
        action="store_true",
        help="allow saved rubric scores produced by a missing or different scorer protocol",
    )
    parser.add_argument(
        "--allow-legacy-sampling-metadata",
        action="store_true",
        help="allow old traces without explicit temperature/top-p/top-k and use command-line values",
    )
    parser.add_argument(
        "--allow-zero-advantage",
        action="store_true",
        help="complete a connectivity smoke test even when every within-case reward is identical",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def outcome_value(case: dict, final_option: str) -> float | None:
    """Return the observed final-answer outcome without treating missing data as failure."""
    truth = case.get("ground_truth") or {}
    correct_option = str(truth.get("correct_option") or "").strip().upper()
    option = str(final_option or "").strip().upper()
    if not correct_option or not option:
        return None
    return float(option == correct_option)


def require_outcome(outcome: float | None, outcome_weight: float, identifier: str) -> None:
    if outcome_weight > 0 and outcome is None:
        raise ValueError(
            f"outcome-weight is positive but outcome is unavailable for {identifier}; "
            "ground_truth.correct_option and final_option are both required"
        )


def _label_for_rubric(row: dict, rubric_id: str) -> object:
    labels = row.get("normalized_labels") or row.get("labels") or {}
    if rubric_id not in labels:
        raise ValueError(
            f"turn {row.get('turn_id')} is missing selected rubric {rubric_id}; "
            "the saved trajectory was not scored with a compatible rubric set"
        )
    value = labels[rubric_id]
    if isinstance(value, str) and value in {"0", "1", "2"}:
        return int(value)
    return value


def selected_turn_adherence(row: dict, rubric_ids: set[str], rubric_temperature: float) -> float | None:
    applicable = []
    for rubric_id in sorted(rubric_ids):
        value = _label_for_rubric(row, rubric_id)
        if value == "NA":
            continue
        applicable.append(normalize_score(value))
    return weighted_soft_min(
        applicable,
        [1.0] * len(applicable),
        rubric_temperature,
    ) if applicable else None


def prepare_turn_scores(turn_scores: list[dict], rubric_ids: set[str] | None,
                        rubric_temperature: float) -> list[dict]:
    prepared = []
    for row in turn_scores:
        updated = dict(row)
        if rubric_ids is not None:
            updated["raw_rubric_adherence_all"] = row.get("raw_rubric_adherence")
            updated["raw_rubric_adherence"] = selected_turn_adherence(
                row, rubric_ids, rubric_temperature,
            )
            updated["selected_rubric_ids"] = sorted(rubric_ids)
        prepared.append(updated)
    return prepared


def refresh_reward(record: dict, rubric_weight: float, outcome_weight: float,
                   rubric_temperature: float, rubric_ids: set[str]) -> dict:
    """Recompute reward fields from saved turn scores without calling Jev."""
    turn_scores = prepare_turn_scores(record.get("turn_scores") or [], rubric_ids, rubric_temperature)
    available = [row["raw_rubric_adherence"] for row in turn_scores
                 if row.get("raw_rubric_adherence") is not None]
    conversation_value = soft_min(available, rubric_temperature) if available else None
    outcome = outcome_value(record, record.get("final_option", ""))
    require_outcome(outcome, outcome_weight, str(record.get("trajectory_id")))
    scaled_rubric_reward = None if conversation_value is None else rubric_weight * conversation_value
    scaled_outcome_reward = None if outcome is None else outcome_weight * outcome
    reward_parts = [value for value in (scaled_rubric_reward, scaled_outcome_reward)
                    if value is not None]
    refreshed = dict(record)
    refreshed.update(
        turn_scores=turn_scores,
        rubric_ids_used=sorted(rubric_ids),
        conversation_rubric_adherence=conversation_value,
        scaled_rubric_reward=scaled_rubric_reward,
        outcome=outcome,
        scaled_outcome_reward=scaled_outcome_reward,
        reward=sum(reward_parts) if reward_parts else None,
    )
    return refreshed


def load_saved_records(path: Path, rubric_weight: float, outcome_weight: float,
                       rubric_temperature: float, expected_rollouts: int,
                       sampling_temperature: float, sampling_top_p: float,
                       sampling_top_k: int, rubric_ids: set[str],
                       expected_rubric_set_version: str | None,
                       expected_rubric_content_sha256: str | None,
                       expected_parent_rubric_set_version: str | None,
                       expected_parent_rubric_content_sha256: str | None,
                       allow_unversioned_rubric_scores: bool,
                       allow_legacy_sampling_metadata: bool,
                       expected_scoring_protocol: str,
                       allow_legacy_rubric_scoring_protocol: bool) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"trajectory file is empty: {path}")
    counts = defaultdict(int)
    rollout_indices = defaultdict(set)
    seen_ids = set()
    for row in rows:
        required = {
            "trajectory_id", "case_id", "rollout_index", "ground_truth",
            "final_option", "turn_scores", "agent_traces",
        }
        missing = sorted(required - set(row))
        if missing:
            raise ValueError(f"saved trajectory {row.get('trajectory_id')} is missing: {missing}")
        if row["trajectory_id"] in seen_ids:
            raise ValueError(f"duplicate trajectory_id in saved file: {row['trajectory_id']}")
        seen_ids.add(row["trajectory_id"])
        saved_version = row.get("rubric_set_version")
        saved_content_sha256 = row.get("rubric_set_content_sha256")
        saved_scoring_protocol = row.get("scoring_protocol")
        parent_subset_reuse = bool(
            saved_version and expected_parent_rubric_set_version
            and saved_version == expected_parent_rubric_set_version
            and saved_content_sha256 and expected_parent_rubric_content_sha256
            and saved_content_sha256 == expected_parent_rubric_content_sha256
        )
        if rubric_weight > 0 and expected_rubric_set_version:
            if saved_version is None and not allow_unversioned_rubric_scores:
                raise ValueError(
                    f"saved trajectory has unversioned rubric scores: {row['trajectory_id']}; "
                    "use a versioned collection or explicitly pass --allow-unversioned-rubric-scores"
                )
            if (saved_version is not None and saved_version != expected_rubric_set_version
                    and not parent_subset_reuse):
                raise ValueError(
                    f"saved rubric version mismatch at {row['trajectory_id']}: "
                    f"saved={saved_version} requested={expected_rubric_set_version}"
                )
        if rubric_weight > 0 and expected_rubric_content_sha256:
            if saved_content_sha256 is None and not allow_unversioned_rubric_scores:
                raise ValueError(
                    f"saved trajectory has no rubric content hash: {row['trajectory_id']}; "
                    "use a fully versioned collection or explicitly pass "
                    "--allow-unversioned-rubric-scores"
                )
            if (saved_content_sha256 is not None
                    and saved_content_sha256 != expected_rubric_content_sha256
                    and not parent_subset_reuse):
                raise ValueError(
                    f"saved rubric content hash mismatch at {row['trajectory_id']}: "
                    f"saved={saved_content_sha256} requested={expected_rubric_content_sha256}"
                )
        if rubric_weight > 0 and saved_scoring_protocol != expected_scoring_protocol:
            if not allow_legacy_rubric_scoring_protocol:
                raise ValueError(
                    f"saved rubric scoring protocol mismatch at {row['trajectory_id']}: "
                    f"saved={saved_scoring_protocol!r} requested={expected_scoring_protocol!r}; "
                    "rescore the trajectories or explicitly pass "
                    "--allow-legacy-rubric-scoring-protocol"
                )
        if not row["agent_traces"]:
            raise ValueError(f"saved trajectory has no Agent traces: {row['trajectory_id']}")
        counts[row["case_id"]] += 1
        rollout_indices[row["case_id"]].add(int(row["rollout_index"]))
        for trace in row["agent_traces"]:
            if not trace.get("prompt_input_ids") or "completion_ids" not in trace or "completion_logprobs" not in trace:
                raise ValueError(f"saved trajectory has incomplete token trace: {row['trajectory_id']}")
            if not trace["completion_ids"]:
                raise ValueError(f"saved trajectory has an empty completion trace: {row['trajectory_id']}")
            if len(trace["completion_ids"]) != len(trace["completion_logprobs"]):
                raise ValueError(
                    f"saved trajectory has mismatched token/logprob lengths: {row['trajectory_id']}"
                )
            missing_sampling = [
                field for field in ("sampling_temperature", "sampling_top_p", "sampling_top_k")
                if field not in trace
            ]
            if missing_sampling and not allow_legacy_sampling_metadata:
                raise ValueError(
                    f"saved trajectory has no exact sampling metadata at {row['trajectory_id']}: "
                    f"missing={missing_sampling}; recollect it or explicitly pass "
                    "--allow-legacy-sampling-metadata"
                )
            trace.setdefault("sampling_temperature", sampling_temperature)
            trace.setdefault("sampling_top_p", sampling_top_p)
            trace.setdefault("sampling_top_k", sampling_top_k)
    wrong = {case_id: count for case_id, count in counts.items() if count != expected_rollouts}
    if wrong:
        raise ValueError(f"each saved case must have exactly {expected_rollouts} rollouts: {wrong}")
    expected_indices = set(range(expected_rollouts))
    malformed = {
        case_id: sorted(indices)
        for case_id, indices in rollout_indices.items()
        if indices != expected_indices
    }
    if malformed:
        raise ValueError(f"saved case rollout indices are incomplete or duplicated: {malformed}")
    return [refresh_reward(row, rubric_weight, outcome_weight, rubric_temperature, rubric_ids) for row in rows]


def record_trajectory(case: dict, case_index: int, rollout_index: int, trajectory: dict,
                      turn_scores: list[dict], rubric_weight: float, outcome_weight: float,
                      rubric_temperature: float, rubric_ids: set[str],
                      rubric_metadata: dict | None = None) -> dict:
    turn_scores = prepare_turn_scores(turn_scores, rubric_ids, rubric_temperature)
    available = [row["raw_rubric_adherence"] for row in turn_scores
                 if row["raw_rubric_adherence"] is not None]
    conversation_value = soft_min(available, rubric_temperature) if available else None
    outcome = outcome_value(case, trajectory["final_option"])
    require_outcome(outcome, outcome_weight, f"{case['case_id']}:rollout_{rollout_index}")
    scaled_rubric_reward = None if conversation_value is None else rubric_weight * conversation_value
    scaled_outcome_reward = None if outcome is None else outcome_weight * outcome
    reward_parts = [value for value in (scaled_rubric_reward, scaled_outcome_reward)
                    if value is not None]
    total_reward = sum(reward_parts) if reward_parts else None
    return {
        "trajectory_id": f"{case['case_id']}:rollout_{rollout_index}",
        "case_id": case["case_id"],
        "case_index": case_index,
        "rollout_index": rollout_index,
        "case_context": visible_case(case),
        "ground_truth": case.get("ground_truth"),
        "dialogue": trajectory["dialogue"],
        "agent_traces": trajectory["agent_traces"],
        "turn_scores": turn_scores,
        "rubric_ids_used": sorted(rubric_ids),
        **(rubric_metadata or {}),
        "scoring_protocol": TWO_STAGE_JEV_PROTOCOL,
        "conversation_rubric_adherence": conversation_value,
        "scaled_rubric_reward": scaled_rubric_reward,
        "outcome": outcome,
        "scaled_outcome_reward": scaled_outcome_reward,
        "reward": total_reward,
        "final_decision_text": trajectory["final_decision_text"],
        "final_option": trajectory["final_option"],
    }


def sequence_logprobs(model, prompt_ids: list[int], completion_ids: list[int], *, grad: bool,
                      temperature: float, top_p: float, top_k: int):
    import torch
    import torch.nn.functional as F
    from transformers.generation.logits_process import (
        TemperatureLogitsWarper,
        TopKLogitsWarper,
        TopPLogitsWarper,
    )

    if not completion_ids:
        return torch.empty(0, device=next(model.parameters()).device)
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")
    if top_k < 0:
        raise ValueError("top_k must be non-negative")
    full_ids = torch.tensor([prompt_ids + completion_ids], dtype=torch.long, device=next(model.parameters()).device)
    attention = torch.ones_like(full_ids)
    context = torch.enable_grad() if grad else torch.no_grad()
    with context:
        logits = model(input_ids=full_ids, attention_mask=attention, use_cache=False).logits[:, :-1]
        targets = full_ids[:, 1:]
        # `generate(..., temperature, top_k, top_p)` stores transition scores after these
        # same warpers. Apply them here so current and rollout log-probs share a
        # distribution before the GRPO update.
        flat_logits = logits.reshape(-1, logits.shape[-1])
        warper_input_ids = torch.zeros((flat_logits.shape[0], 1), dtype=torch.long, device=flat_logits.device)
        if temperature != 1.0:
            flat_logits = TemperatureLogitsWarper(float(temperature))(warper_input_ids, flat_logits)
        if top_k > 0:
            flat_logits = TopKLogitsWarper(int(top_k))(warper_input_ids, flat_logits)
        if top_p < 1.0:
            flat_logits = TopPLogitsWarper(float(top_p))(warper_input_ids, flat_logits)
        processed_logits = flat_logits.reshape_as(logits)
        token_logprobs = F.log_softmax(processed_logits, dim=-1).gather(
            -1, targets.unsqueeze(-1),
        ).squeeze(-1)
    start = len(prompt_ids) - 1
    end = start + len(completion_ids)
    return token_logprobs[0, start:end]


def reference_logprobs(model, prompt_ids: list[int], completion_ids: list[int], *,
                       temperature: float, top_p: float, top_k: int):
    if not hasattr(model, "disable_adapter"):
        raise RuntimeError("the LoRA model does not expose disable_adapter for KL reference scoring")
    with model.disable_adapter():
        return sequence_logprobs(
            model, prompt_ids, completion_ids, grad=False,
            temperature=temperature, top_p=top_p, top_k=top_k,
        )


def compute_grpo_loss(model, records: list[dict], beta: float, clip_epsilon: float, *,
                      temperature: float, top_p: float, top_k: int,
                      backward: bool = False):
    import torch

    grouped = defaultdict(list)
    for record in records:
        grouped[record["case_id"]].append(record)
    advantages = {}
    group_reward_stats = {}
    masked_no_reward = 0
    for case_id, members in grouped.items():
        available_members = [member for member in members if member["reward"] is not None]
        masked_no_reward += len(members) - len(available_members)
        if len(available_members) < 2:
            group_reward_stats[case_id] = {
                "available": len(available_members),
                "masked": len(members) - len(available_members),
                "mean": None,
                "std": None,
                "has_comparative_signal": False,
            }
            continue
        rewards = torch.tensor(
            [float(member["reward"]) for member in available_members],
            dtype=torch.float32,
            device=next(model.parameters()).device,
        )
        mean = rewards.mean()
        std = rewards.std(unbiased=False)
        normalized = (rewards - mean) / (std + 1e-8) if float(std) > 0 else torch.zeros_like(rewards)
        group_reward_stats[case_id] = {
            "available": len(available_members),
            "masked": len(members) - len(available_members),
            "mean": float(mean.detach().cpu()),
            "std": float(std.detach().cpu()),
            "has_comparative_signal": bool(float(std) > 0),
        }
        for member, advantage in zip(available_members, normalized):
            advantages[member["trajectory_id"]] = advantage

    trajectory_token_counts = {
        record["trajectory_id"]: sum(len(trace.get("completion_ids") or []) for trace in record["agent_traces"])
        for record in records
        if record["trajectory_id"] in advantages
    }
    empty_trajectories = sorted(
        trajectory_id for trajectory_id, count in trajectory_token_counts.items() if count < 1
    )
    if empty_trajectories:
        raise RuntimeError(f"eligible trajectories have no Agent completion tokens: {empty_trajectories}")
    eligible_trajectory_count = len(trajectory_token_counts)
    jobs = []
    for record in records:
        if record["trajectory_id"] not in advantages:
            continue
        advantage = advantages[record["trajectory_id"]]
        for trace in record["agent_traces"]:
            jobs.append((
                record["trajectory_id"], trace, advantage,
                trajectory_token_counts[record["trajectory_id"]],
            ))
    if not jobs:
        raise RuntimeError(
            "no GRPO groups had at least two trajectories with usable rewards; "
            "all-NA rubric rewards are masked rather than imputed"
        )

    policy_contributions = []
    kl_contributions = []
    used = 0
    ratio_sum = 0.0
    ratio_sq_sum = 0.0
    ratio_min = float("inf")
    ratio_max = float("-inf")
    ratio_count = 0
    for trajectory_id, trace, advantage, trajectory_token_count in jobs:
        trace_temperature = float(trace.get("sampling_temperature", temperature))
        trace_top_p = float(trace.get("sampling_top_p", top_p))
        trace_top_k = int(trace.get("sampling_top_k", top_k))
        current = sequence_logprobs(
            model, trace["prompt_input_ids"], trace["completion_ids"], grad=True,
            temperature=trace_temperature, top_p=trace_top_p, top_k=trace_top_k,
        )
        old = torch.tensor(trace["completion_logprobs"], dtype=current.dtype, device=current.device)
        if current.numel() == 0:
            raise RuntimeError(f"empty Agent completion tokens: {trajectory_id}:{trace.get('stage')}")
        if old.numel() != current.numel():
            raise RuntimeError(
                f"old/new token logprob length mismatch at {trajectory_id}:{trace.get('stage')}: "
                f"old={old.numel()} new={current.numel()}"
            )
        if not torch.isfinite(current).all().item() or not torch.isfinite(old).all().item():
            raise RuntimeError(f"non-finite old/new log-probability at {trajectory_id}:{trace.get('stage')}")
        ratio = torch.exp(current - old)
        if not torch.isfinite(ratio).all().item():
            raise RuntimeError(f"non-finite new/old policy ratio at {trajectory_id}:{trace.get('stage')}")
        ratio_values = ratio.detach().float()
        ratio_sum += float(ratio_values.sum())
        ratio_sq_sum += float((ratio_values * ratio_values).sum())
        ratio_min = min(ratio_min, float(ratio_values.min()))
        ratio_max = max(ratio_max, float(ratio_values.max()))
        ratio_count += ratio_values.numel()
        clipped = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
        surrogate = torch.minimum(ratio * advantage, clipped * advantage)
        policy_sum = -surrogate.sum()

        ref = reference_logprobs(
            model, trace["prompt_input_ids"], trace["completion_ids"],
            temperature=trace_temperature, top_p=trace_top_p, top_k=trace_top_k,
        )
        log_ratio = ref - current
        kl_sum = (torch.exp(log_ratio) - log_ratio - 1.0).sum()
        if not torch.isfinite(policy_sum).item() or not torch.isfinite(kl_sum).item():
            raise RuntimeError(f"non-finite loss term at {trajectory_id}:{trace.get('stage')}")
        # All Agent tokens in one sampled conversation form one trajectory.
        # Average inside that trajectory, then average trajectories, so a
        # longer dialogue cannot receive more weight merely because it has
        # more Agent turns or tokens.
        denominator = eligible_trajectory_count * trajectory_token_count
        policy_contribution = policy_sum / denominator
        kl_contribution = kl_sum / denominator
        if backward:
            (policy_contribution + beta * kl_contribution).backward()
        policy_contributions.append(policy_contribution.detach())
        kl_contributions.append(kl_contribution.detach())
        used += 1
        del current, old, ref, ratio, clipped, surrogate, policy_sum, kl_sum
        del policy_contribution, kl_contribution
    if not policy_contributions:
        raise RuntimeError("no Agent completion tokens had matching old log-probabilities")
    policy_loss = torch.stack(policy_contributions).sum()
    kl_loss = torch.stack(kl_contributions).sum() if kl_contributions else torch.zeros_like(policy_loss)
    total_loss = policy_loss + beta * kl_loss
    ratio_mean = ratio_sum / ratio_count if ratio_count else None
    ratio_variance = max(ratio_sq_sum / ratio_count - ratio_mean * ratio_mean, 0.0) if ratio_count else None
    ratio_stats = {
        "count": ratio_count,
        "mean": ratio_mean,
        "std": ratio_variance ** 0.5 if ratio_variance is not None else None,
        "min": ratio_min if ratio_count else None,
        "max": ratio_max if ratio_count else None,
        "max_abs_deviation": max(
            abs(ratio_min - 1.0), abs(ratio_max - 1.0),
        ) if ratio_count else None,
        "eligible_trajectories": eligible_trajectory_count,
        "masked_no_reward_trajectories": masked_no_reward,
        "nonzero_advantage_trajectories": sum(
            abs(float(value.detach().cpu())) > 1e-8 for value in advantages.values()
        ),
        "group_rewards": group_reward_stats,
        "objective_weighting": "mean_agent_tokens_within_trajectory_then_mean_trajectories",
    }
    if not torch.isfinite(policy_loss).item() or not torch.isfinite(kl_loss).item() or not torch.isfinite(total_loss).item():
        raise RuntimeError("non-finite GRPO loss")
    return total_loss, policy_loss, kl_loss, used, ratio_stats


def main() -> None:
    args = parse_args()
    if args.rollouts_per_case != 4:
        raise ValueError("the smoke test uses exactly four rollouts per case")
    if args.num_cases is not None and args.num_cases < 1:
        raise ValueError("num-cases must be positive")
    if args.rounds < 1:
        raise ValueError("rounds must be positive")
    if args.pre_update_ratio_tolerance < 0:
        raise ValueError("pre-update-ratio-tolerance must be non-negative")
    if args.rubric_weight < 0 or args.outcome_weight < 0 or args.rubric_weight + args.outcome_weight <= 0:
        raise ValueError("rubric-weight and outcome-weight must be non-negative with a positive sum")
    source_path = ROOT / args.trajectories if args.trajectories else None
    if source_path is not None and source_path.resolve() == (ROOT / args.output / "trajectories.jsonl").resolve():
        raise ValueError("--trajectories must be different from the training output trajectories.jsonl")

    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output directory is not empty: {output}; use --overwrite")
    if args.overwrite:
        for name in ("trajectories.jsonl", "summary.json", "train_log.jsonl", "config.json"):
            (output / name).unlink(missing_ok=True)

    rubric_path = ROOT / args.rubric_set
    rubrics = load_rubrics(rubric_path)
    rubric_metadata = rubric_set_metadata(rubric_path)
    validation_status = rubric_metadata.get("rubric_set_validation_status")
    if validation_status and validation_status not in {"validated", "passed"} and not args.allow_unvalidated_rubrics:
        raise ValueError(
            f"rubric set is not validated for training: {validation_status}; "
            "revalidate it first or use --allow-unvalidated-rubrics only for a deliberate smoke test"
        )
    rubric_ids = {rubric["rubric_id"] for rubric in rubrics}
    key = None
    client = None
    if source_path is None:
        num_cases = args.num_cases if args.num_cases is not None else 2
        key = os.environ.get(args.api_key_env)
        if not key:
            raise RuntimeError(f"{args.api_key_env} is not set")
        cases = load_cases(ROOT / args.input, args.start_index, num_cases)
        doctor_system = (ROOT / "prompts/user_simulator.txt").read_text(encoding="utf-8")
        agent_system = (ROOT / "prompts/agent.txt").read_text(encoding="utf-8")
    else:
        cases = []
        num_cases = args.num_cases

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; launch this smoke test on the GPU server")
    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, trust_remote_code=True)
    model = get_peft_model(model, LoraConfig(
        task_type="CAUSAL_LM", r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], use_rslora=True,
    ))
    model.to(torch.device("cuda:0"))
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    args.top_k = resolve_generation_top_k(model, args.top_k)
    model.eval()
    trajectory_path = output / "trajectories.jsonl"
    if source_path is not None:
        records = load_saved_records(
            source_path, args.rubric_weight, args.outcome_weight, args.rubric_temperature,
            args.rollouts_per_case, args.generation_temperature, args.top_p, args.top_k, rubric_ids,
            rubric_metadata.get("rubric_set_version"),
            rubric_metadata.get("rubric_set_content_sha256"),
            rubric_metadata.get("rubric_set_parent_version"),
            rubric_metadata.get("rubric_set_parent_content_sha256"),
            args.allow_unversioned_rubric_scores,
            args.allow_legacy_sampling_metadata,
            TWO_STAGE_JEV_PROTOCOL,
            args.allow_legacy_rubric_scoring_protocol,
        )
        case_ids = sorted({row["case_id"] for row in records})
        if num_cases is not None and len(case_ids) != num_cases:
            raise ValueError(f"saved file contains {len(case_ids)} cases, not --num-cases={num_cases}")
        num_cases = len(case_ids)
        for record in records:
            source_identity = {
                key: record.get(key)
                for key in (
                    "rubric_set_version", "rubric_set_sha256",
                    "rubric_set_content_sha256", "rubric_set_validation_status",
                )
            }
            record["source_rubric_score_identity"] = source_identity
            record.update(rubric_metadata)
            record["scoring_protocol"] = TWO_STAGE_JEV_PROTOCOL
        with trajectory_path.open("w", encoding="utf-8") as handle:
            for index, record in enumerate(records, 1):
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                if index % args.rollouts_per_case == 0 or index == len(records):
                    print(f"LOADED TRAJECTORIES: {index}/{len(records)} | Jev calls=0", flush=True)
    else:
        client = JevClient(endpoint=args.jev_endpoint, model=args.jev_model, api_key_env=args.api_key_env)
        records = []
        with trajectory_path.open("w", encoding="utf-8") as handle:
            for case_number, case in enumerate(cases, 1):
                for rollout_index in range(args.rollouts_per_case):
                    trajectory = generate_trajectory(
                        model, tokenizer, case, doctor_system, agent_system, key, args,
                    )
                    agent_indices = [index for index, turn in enumerate(trajectory["dialogue"])
                                     if turn["speaker"] == "agent"]
                    with ThreadPoolExecutor(max_workers=min(args.jev_workers, len(agent_indices))) as pool:
                        futures = [pool.submit(
                            score_agent_turn, client, case, trajectory["dialogue"], index,
                            rubrics, args.rubric_temperature, rubric_metadata,
                        ) for index in agent_indices]
                        turn_scores = [future.result() for future in futures]
                    record = record_trajectory(
                        case, args.start_index + case_number - 1, rollout_index,
                        trajectory, turn_scores, args.rubric_weight, args.outcome_weight,
                        args.rubric_temperature, rubric_ids, rubric_metadata,
                    )
                    records.append(record)
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    handle.flush()
                    print(f"TRAJECTORIES: {len(records)}/{len(cases) * 4} | Jev calls={client.calls}", flush=True)

    # Keep dropout disabled while recomputing the rollout policy. This makes
    # the pre-update ratio comparable to the deterministic generation pass.
    model.eval()
    model.config.use_cache = False
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate)
    optimizer.zero_grad(set_to_none=True)
    total_loss, policy_loss, kl_loss, used_completions, ratio_stats = compute_grpo_loss(
        model, records, args.beta, args.clip_epsilon,
        temperature=args.generation_temperature, top_p=args.top_p, top_k=args.top_k, backward=True,
    )
    finite_stats = {
        "loss": total_loss,
        "policy_loss": policy_loss,
        "kl_loss": kl_loss,
    }
    if any(not torch.isfinite(value).item() for value in finite_stats.values()):
        raise RuntimeError("refusing optimizer step because a GRPO loss is non-finite")
    if not ratio_stats["count"] or ratio_stats["max_abs_deviation"] is None:
        raise RuntimeError("refusing optimizer step because no pre-update policy ratios were measured")
    if not math.isfinite(ratio_stats["max_abs_deviation"]):
        raise RuntimeError("refusing optimizer step because the pre-update ratio is non-finite")
    if ratio_stats["max_abs_deviation"] > args.pre_update_ratio_tolerance:
        raise RuntimeError(
            "refusing optimizer step because new/old policy ratio is not near 1: "
            f"max_abs_deviation={ratio_stats['max_abs_deviation']:.8g}, "
            f"tolerance={args.pre_update_ratio_tolerance:.8g}"
        )
    if ratio_stats["nonzero_advantage_trajectories"] == 0 and not args.allow_zero_advantage:
        raise RuntimeError(
            "refusing optimizer step because all within-case GRPO advantages are zero; "
            "collect reward-diverse rollouts or pass --allow-zero-advantage only for a connectivity smoke test"
        )
    grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
    if not torch.isfinite(grad_norm).item():
        raise RuntimeError("refusing optimizer step because the gradient norm is non-finite")
    optimizer.step()
    model.save_pretrained(output / "adapter")
    tokenizer.save_pretrained(output / "adapter")

    train_row = {
        "global_step": 1,
        "loss": float(total_loss.detach().cpu()),
        "policy_loss": float(policy_loss.detach().cpu()),
        "kl_loss": float(kl_loss.detach().cpu()),
        "beta": args.beta,
        "learning_rate": args.learning_rate,
        "used_agent_completions": used_completions,
        "pre_update_ratio": ratio_stats,
        "sampling": {
            "temperature": args.generation_temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
        },
        "pre_update_ratio_tolerance": args.pre_update_ratio_tolerance,
        "gradient_norm_before_step": float(grad_norm.detach().cpu()),
        "cases": num_cases,
        "trajectories": len(records),
    }
    (output / "train_log.jsonl").write_text(json.dumps(train_row, ensure_ascii=False) + "\n", encoding="utf-8")
    summary = {
        "status": "completed",
        "implementation": "custom_multi_turn_grpo_smoke",
        "cases": num_cases,
        "trajectories": len(records),
        "agent_turns": sum(len(row["agent_traces"]) for row in records),
        "jev_calls": client.calls if client is not None else 0,
        "data_source": "saved_trajectories" if source_path is not None else "online_generation",
        "loss": train_row,
        "reward": {"rubric": args.rubric_weight, "outcome": args.outcome_weight, "effectiveness": 0.0},
        "signal_semantics": {
            "rubric": "rubric_adherence_not_full_behavioral_appropriateness",
            "outcome": "final_task_success",
            "effectiveness": "not_computed",
        },
        "scoring_protocol": TWO_STAGE_JEV_PROTOCOL,
        **rubric_metadata,
        "aggregation": {"rubric": "weighted_softmin", "turns": "soft_min"},
        "output": str(output.relative_to(ROOT)),
    }
    (output / "config.json").write_text(json.dumps({
        "input": args.input, "trajectory_source": args.trajectories,
        "start_index": args.start_index, "num_cases": num_cases,
        "rollouts_per_case": args.rollouts_per_case, "rounds": args.rounds,
        "rubric_set": args.rubric_set, "rubrics": len(rubrics),
        **rubric_metadata,
        "rubric_ids": sorted(rubric_ids),
        "rubric_temperature": args.rubric_temperature, "rubric_weight": args.rubric_weight,
        "outcome_weight": args.outcome_weight,
        "effectiveness_weight": 0.0,
        "scoring_protocol": TWO_STAGE_JEV_PROTOCOL,
        "beta": args.beta, "learning_rate": args.learning_rate,
        "clip_epsilon": args.clip_epsilon, "training": True,
        "sampling": {
            "temperature": args.generation_temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
        },
        "pre_update_ratio_tolerance": args.pre_update_ratio_tolerance,
        "allow_unvalidated_rubrics": args.allow_unvalidated_rubrics,
        "allow_unversioned_rubric_scores": args.allow_unversioned_rubric_scores,
        "allow_legacy_rubric_scoring_protocol": args.allow_legacy_rubric_scoring_protocol,
        "allow_legacy_sampling_metadata": args.allow_legacy_sampling_metadata,
        "allow_zero_advantage": args.allow_zero_advantage,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
