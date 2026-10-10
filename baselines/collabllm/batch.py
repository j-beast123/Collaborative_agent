"""Adapt a deferred five-case batch before the existing GRPO update barrier."""
import copy

from src.online_rl.integrated.records import from_dict
from src.online_rl.tree.records import Tree
from src.online_rl.tree.five_gpu import update_batch
from src.online_rl.tree.reward import sibling_group_diagnostics, interaction_diagnostics
from baselines.collabllm.pilot import frozen_targets
from baselines.collabllm.reward import METHOD


def prepare_batch(records, reports, *, frozen_weights, data_receipt, expected_policy):
    """Return new records; do not mutate the original tree, trace, or config.

    The parent runner should call this after collecting a deferred outcome-only
    rollout batch and before update_batch. A saved rollout from a different
    policy cannot be used as on-policy training data for a subsequent batch.
    """
    if not data_receipt["all_verified"]:
        raise ValueError("Original train, order and test must all be verified before training")
    if len(records) != 5 or len(reports) != 5:
        raise ValueError("Preserve the original five-case batch")
    pilot = frozen_weights["pilot"]
    for field, role in (("train_sha256", "train"), ("train_order_sha256", "train_order")):
        if pilot[field] != data_receipt["files"][role]["actual_sha256"]:
            raise ValueError("Training data identity differs from frozen pilot")
    if pilot["reference_config_sha256"] != data_receipt["reference_config_sha256"]:
        raise ValueError("Reference configuration differs from frozen pilot")
    start = records[0]["case_cursor"]
    if start < 0 or start % 5 or [r["case_cursor"] for r in records] != list(range(start, start + 5)):
        raise ValueError("Case cursors must preserve the five-case batch boundary")
    expected_ids = data_receipt["train_case_order"][start:start + 5]
    if [r["case_id"] for r in records] != expected_ids:
        raise ValueError("Batch differs from original training order")
    prepared = []
    for record, report in zip(records, reports):
        if record["policy_sha256"] != expected_policy:
            raise ValueError("Rollout policy differs from frozen batch policy")
        update = record.get("update") or {}
        if update.get("optimizer_steps", 0) or update.get("optimizer_step", False):
            raise ValueError("Cannot replace rewards after an optimizer update")
        if any(record.get(k) for k in ("feedback", "trajectory_rubrics", "effectiveness", "response_scores")):
            raise ValueError("Use deferred outcome-only rollouts, with SHERPA rubric pipeline disabled")
        tree = from_dict(Tree, record["tree"])
        if tree.policy_sha256 != expected_policy or tree.case_id != record["case_id"]:
            raise ValueError("Tree identity differs from batch record")
        out = copy.deepcopy(record)
        out["agent_targets"] = frozen_targets(tree, report, frozen_weights)
        out["method_label"] = METHOD
        out["collabllm"] = copy.deepcopy(report)
        out["group_diagnostics"] = sibling_group_diagnostics(tree, out["agent_targets"])
        out["interaction_diagnostics"] = interaction_diagnostics(tree, out["agent_targets"])
        prepared.append(out)
    return prepared


def update_prepared_batch(updater, records, reports, *, frozen_weights, data_receipt,
                          expected_policy, actual_policy):
    """Delegate unchanged grouping, action traces and optimizer to SHERPA."""
    prepared = prepare_batch(records, reports, frozen_weights=frozen_weights,
                             data_receipt=data_receipt, expected_policy=expected_policy)
    result = update_batch(updater, prepared, expected_policy, actual_policy)
    return prepared, result
