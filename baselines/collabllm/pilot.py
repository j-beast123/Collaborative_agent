"""Train-only component distributions and an explicit length-weight freeze.

The pilot uses predeclared 5/20/50-case prefixes of the original train order at
one initial frozen policy, without updates. Five cases cannot freeze a weight.
No coefficient is selected automatically from test data or mean token length.
"""
from __future__ import annotations

import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

from src.online_rl.integrated.records import sha256_json
from src.online_rl.integrated.metrics import advantages
from baselines.collabllm.reward import METHOD, PROTOCOL, component_targets, make_targets, number


def pilot_plan():
    return json.loads(Path(__file__).with_name("adaptation_spec.json").read_text())["pilot_plan"]


def distribution(values):
    xs = sorted(x for x in values if x is not None)
    if not xs:
        return {"count": 0}
    def percentile(q):
        pos = (len(xs) - 1) * q
        lo = int(pos)
        return xs[lo] + (xs[min(lo + 1, len(xs) - 1)] - xs[lo]) * (pos - lo)
    return {"count": len(xs), "mean": statistics.mean(xs), "std": statistics.pstdev(xs),
            "min": xs[0], "p05": percentile(.05), "p50": percentile(.5),
            "p95": percentile(.95), "max": xs[-1]}


def summarize_pilot(trees_and_reports, data_receipt, *, initial_policy_sha256):
    pairs = list(trees_and_reports)
    if not data_receipt["train_verified"]:
        raise ValueError("Original train bytes and order must be verified before pilot")
    plan = pilot_plan()
    expected = data_receipt["train_case_order"][:len(pairs)]
    if len(pairs) not in plan["stage_sizes"] or [tree.case_id for tree, _ in pairs] != expected:
        raise ValueError("Pilot must use a predeclared 5/20/50-case prefix of the original train order")
    if not initial_policy_sha256 or any(tree.policy_sha256 != initial_policy_sha256 for tree, _ in pairs):
        raise ValueError("Pilot must use the same verified initial policy without updates")
    judge_ids = {sha256_json(report["judge"]) for _, report in pairs}
    token_ids = {sha256_json(report["tokenizer"]) for _, report in pairs}
    if len(judge_ids) != 1 or len(token_ids) != 1:
        raise ValueError("Mixed judge or tokenizer in pilot")
    leaf_values = {k: [] for k in ("success", "interactivity", "agent_ktokens")}
    node_values = {k: [] for k in leaf_values}
    centered = {k: [] for k in leaf_values}
    group_components = []
    groups = 0
    case_distributions = []
    for tree, report in pairs:
        before = {k: len(v) for k, v in centered.items()}
        targets = component_targets(tree, report)
        for row in report["leaves"].values():
            if row["success"] is not None:
                for key in leaf_values:
                    leaf_values[key].append(row[key])
        for ids in targets["groups"].values():
            groups += 1
            group_components.append([
                {"success": targets["nodes"][nid]["outcome_component"],
                 "interactivity": targets["nodes"][nid]["interactivity"],
                 "agent_ktokens": targets["nodes"][nid]["agent_ktokens"]}
                for nid in ids])
            for key in leaf_values:
                field = "outcome_component" if key == "success" else key
                values = [targets["nodes"][nid][field] for nid in ids
                          if targets["nodes"][nid]["outcome_component"] is not None]
                node_values[key].extend(values)
                if len(values) >= 2:
                    avg = statistics.mean(values)
                    centered[key].extend(v - avg for v in values)
        case_distributions.append({"case_id": tree.case_id,
            "terminal": {k: distribution([row[k] for row in report["leaves"].values()
                                           if row["success"] is not None]) for k in leaf_values},
            "sibling_centered": {k: distribution(v[before[k]:]) for k, v in centered.items()}})
    # Compare equally weighted case-level signal scales; leaf count is not the sample size.
    stability = {"minimum_cases_met": len(pairs) >= plan["minimum_freeze_cases"], "components": {}}
    for component in centered:
        per_case = [r["sibling_centered"][component].get("std", 0.0) for r in case_distributions]
        early, later = per_case[:5], per_case[5:]
        first = statistics.mean(early)
        last = statistics.mean(later) if later else None
        relative = abs(last-first) / max(abs(first), abs(last), 1e-8) if last is not None else None
        means = [r["terminal"][component].get("mean") for r in case_distributions]
        mean_shift = (abs(statistics.mean(means[5:]) - statistics.mean(means[:5]))
                      if len(means) > 5 and all(v is not None for v in means) else None)
        stable = (relative is not None and relative <= plan["relative_std_change_threshold"]
                  and mean_shift is not None and mean_shift <= plan["absolute_mean_shift_threshold"])
        stability["components"][component] = {"first5_mean_case_std": first,
            "additional_cases_mean_case_std": last, "relative_change": relative,
            "case_mean_shift": mean_shift,
            "zero_signal": all(v == 0 for v in per_case), "stable": stable}
    stable = stability["minimum_cases_met"] and all(v["stable"] for v in stability["components"].values())
    stability["ready_for_weight_review"] = stable
    stability["next_stage_cases"] = (None if stable else next((n for n in plan["stage_sizes"] if n > len(pairs)), None))
    return {"schema": "collabllm_train_pilot_v2", "method_label": METHOD, "protocol": PROTOCOL,
            "case_ids": expected, "initial_policy_sha256": initial_policy_sha256,
            "pilot_plan": plan, "case_distributions": case_distributions, "stability": stability,
            "train_sha256": data_receipt["files"]["train"]["actual_sha256"],
            "train_order_sha256": data_receipt["files"]["train_order"]["actual_sha256"],
            "reference_config_sha256": data_receipt["reference_config_sha256"],
            "judge": pairs[0][1]["judge"], "tokenizer": pairs[0][1]["tokenizer"],
            "report_sha256": [sha256_json(report) for _, report in pairs],
            "leaf_distributions": {k: distribution(v) for k, v in leaf_values.items()},
            "node_distributions": {k: distribution(v) for k, v in node_values.items()},
            "sibling_centered_distributions": {k: distribution(v) for k, v in centered.items()},
            "group_components": group_components,
            "sibling_groups": groups, "optimizer_steps": 0,
            "test_data_used_by_this_function": False}


def freeze_length_weight(pilot, *, length_weight, rationale, path):
    lam = number(length_weight, "length_weight")
    if (pilot.get("schema") != "collabllm_train_pilot_v2"
            or pilot.get("test_data_used_by_this_function") is not False
            or pilot["leaf_distributions"]["success"]["count"] == 0):
        raise ValueError("A valid train pilot must precede weight selection")
    if pilot.get("pilot_plan") != pilot_plan() or not pilot.get("stability", {}).get("ready_for_weight_review"):
        raise ValueError("Five cases or unstable component distributions cannot freeze lambda; follow the pilot expansion plan")
    if not isinstance(rationale, str) or not rationale.strip():
        raise ValueError("Record the rationale based on the train component distributions")
    rewards, centered = [], []
    for group in pilot["group_components"]:
        values = [None if row["success"] is None else
                  row["success"] + row["interactivity"] - lam * row["agent_ktokens"] for row in group]
        rewards.extend(values)
        centered.extend(advantages(values, "mean_center"))
    record = {"schema": "collabllm_frozen_weights_v2", "method_label": METHOD,
              "protocol": PROTOCOL, "length_weight": lam, "interactivity_weight": 1.0,
              "task_success_weight": 1.0, "pilot_sha256": sha256_json(pilot),
              "pilot": pilot, "rationale": rationale,
              "selected_reward_distribution": distribution(rewards),
              "selected_advantage_distribution": distribution(centered),
              "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
              "selection_rule": "train_component_distributions_before_test_result_inspection"}
    record["sha256"] = sha256_json(record)
    # Immutable receipt: changing lambda requires a new artifact/run identity.
    with Path(path).open("x") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    return record


def frozen_targets(tree, report, frozen_weights):
    frozen = dict(frozen_weights)
    digest = frozen.pop("sha256")
    if digest != sha256_json(frozen) or frozen.get("schema") != "collabllm_frozen_weights_v2":
        raise ValueError("Invalid frozen weight artifact")
    if frozen["protocol"] != PROTOCOL or frozen["pilot_sha256"] != sha256_json(frozen["pilot"]):
        raise ValueError("Frozen pilot/protocol identity mismatch")
    if frozen["task_success_weight"] != 1.0 or frozen["interactivity_weight"] != 1.0:
        raise ValueError("Only the length weight is calibrated in this protocol")
    if any(report[k] != frozen["pilot"][k] for k in ("judge", "tokenizer")):
        raise ValueError("Judge/tokenizer differs from the frozen pilot")
    result = make_targets(tree, report, length_weight=frozen["length_weight"],
                          interactivity_weight=frozen["interactivity_weight"])
    result["frozen_weights_sha256"] = digest
    return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Freeze an explicitly chosen train-pilot length weight; never reads test results.')
    parser.add_argument('--pilot', required=True)
    parser.add_argument('--length-weight', type=float, required=True)
    parser.add_argument('--rationale', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    result = freeze_length_weight(json.loads(Path(args.pilot).read_text()), length_weight=args.length_weight,
                                  rationale=args.rationale, path=args.out)
    print(json.dumps({'sha256':result['sha256'],'length_weight':result['length_weight']},indent=2))
