"""CPU-only checks with synthetic trees and judges; no API or dataset access."""
import copy
import json
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path

from src.judges.llm import ChatResult
from src.online_rl.integrated.metrics import advantages
from src.online_rl.tree.cache import CallCache
from src.online_rl.tree.records import AgentNode, Leaf, Tree
from src.online_rl.tree.team import node_credit
from baselines.collabllm.reward import (
    PROTOCOL, InteractivityJudge, score_tree, make_targets, visible_discussion,
)
from baselines.collabllm.pilot import summarize_pilot, freeze_length_weight, frozen_targets
from baselines.collabllm.batch import prepare_batch, update_prepared_batch
from baselines.collabllm.data import audit_data


class Counter:
    identity = {"encoding": "synthetic_character_counter_NOT_production"}

    def __call__(self, text):
        return len(text)


class Judge:
    def __init__(self, score=.75):
        self.identity = {"model": "synthetic-judge", "revision": "test", "score": score}
        self.score, self.calls, self.inputs = score, 0, []

    def __call__(self, messages):
        self.calls += 1
        self.inputs.append(copy.deepcopy(messages))
        return {"interactivity": self.score, "thought": "fixture"}


def toy(case="case0"):
    # One short success and two deeper failure leaves: uniform-leaf mean = 1/3.
    nodes = []
    specs = [("a", None, 1), ("b", None, 1), ("a0", "a", 2),
             ("a1", "a", 2), ("a10", "a1", 3), ("a11", "a1", 3)]
    for nid, parent, depth in specs:
        nodes.append(AgentNode(nid, parent, case, depth, (("doctor", "D1", 1, "question"),),
            "reply", {"agent_protocol": "precommit_v1", "state_prompt_hash": parent or "root",
                      "segments": [{"private": "DO_NOT_LEAK_PRIVATE"}]}, "theta0", 0,
            "stop" if nid in ("b", "a0", "a10", "a11") else None,
            nid in ("b", "a0", "a10", "a11"), None))
    def leaf(nid, path, outcome):
        dialog = (("doctor", "D1", 1, "Please explain the distinction."),
                  ("agent", "A1", 1, "A useful distinction."),
                  ("doctor", "FINAL", 0, "DO_NOT_LEAK_FINAL"))
        return Leaf(nid, case, tuple(path), dialog, "DO_NOT_LEAK_FINAL", "A", outcome, "stop")
    return Tree(case, "theta0", 4, 4, tuple(nodes), (
        leaf("a0", ["a", "a0"], 1.), leaf("a10", ["a", "a1", "a10"], 0.),
        leaf("a11", ["a", "a1", "a11"], 0.), leaf("b", ["b"], None)))


def scored(tree):
    return score_tree(tree, judge=Judge(), cache=CallCache(None), token_counter=Counter())


def fixture_pilot(count=20):
    trees = [toy(f"case{i}") for i in range(count)]
    reports = [scored(t) for t in trees]
    receipt = {"train_verified": True, "all_verified": True,
               "train_case_order": [t.case_id for t in trees], "reference_config_sha256": "config",
               "files": {"train": {"actual_sha256": "train"}, "train_order": {"actual_sha256": "order"}}}
    pilot = summarize_pilot(zip(trees, reports), receipt, initial_policy_sha256="theta0")
    return trees, reports, receipt, pilot


class RewardTests(unittest.TestCase):
    def test_success_only_exactly_matches_existing_rewards_and_advantages(self):
        tree = toy()
        targets = make_targets(tree, scored(tree), length_weight=0, interactivity_weight=0)
        expected = node_credit(1., 0.)(tree, {}, {})
        self.assertEqual(targets["groups"], expected["groups"])
        self.assertEqual(targets["nodes"]["a"]["reward"], 1/3)  # not 1/2, no path reweighting
        for ids in targets["groups"].values():
            actual = [targets["nodes"][i]["reward"] for i in ids]
            original = [expected["nodes"][i]["reward"] for i in ids]
            self.assertEqual(actual, original)
            self.assertEqual(advantages(actual, "mean_center"), advantages(original, "mean_center"))

    def test_one_judgment_for_duplicate_dialogues_and_restart(self):
        tree, judge = toy(), Judge()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"cache.jsonl"
            report = score_tree(tree, judge=judge, cache=CallCache(path), token_counter=Counter())
            self.assertEqual(judge.calls, 1)
            self.assertEqual(report["cost"]["new_successful_judgments"], 1)
            self.assertEqual(report["cost"]["additional_conversations"], 0)
            targets = make_targets(tree, report, length_weight=.4)
            self.assertEqual(judge.calls, 1)  # reuse at every ancestor
            self.assertEqual(targets["nodes"]["a"]["descendant_leaves"], 3)
            again = score_tree(tree, judge=judge, cache=CallCache(path), token_counter=Counter())
            self.assertEqual(judge.calls, 1)
            self.assertEqual(again["cost"]["new_successful_judgments"], 0)
            changed = Judge(.5)
            score_tree(tree, judge=changed, cache=CallCache(path), token_counter=Counter())
            self.assertEqual(changed.calls, 1)  # judge identity invalidates cache

    def test_no_final_or_private_state_in_judge_and_length(self):
        tree, judge = toy(), Judge()
        report = score_tree(tree, judge=judge, cache=CallCache(None), token_counter=Counter())
        self.assertNotIn("DO_NOT_LEAK", json.dumps(judge.inputs))
        self.assertEqual(report["leaves"]["a0"]["agent_visible_tokens"], len("A useful distinction."))
        self.assertEqual(visible_discussion(tree.leaves[0])[0]["role"], "user")

    def test_none_outcomes_excluded_from_all_components(self):
        tree = toy()
        report = scored(tree)
        self.assertEqual(report["leaves"]["b"]["status"], "unparsed_final")
        result = make_targets(tree, report, length_weight=.4)
        self.assertIsNone(result["nodes"]["b"]["reward"])
        row = result["nodes"]["a"]
        self.assertAlmostEqual(row["reward"], 1/3 + .75 - .4 * len("A useful distinction.") / 1000)

    def test_invalid_judge_score_never_cached_or_zero_filled(self):
        for bad in (float("nan"), 1.1, -0.1, True):
            cache = CallCache(None)
            with self.assertRaises(ValueError):
                score_tree(toy(), judge=Judge(bad), cache=cache, token_counter=Counter())
            self.assertEqual(cache.store, {})

    def test_tree_and_weight_guards(self):
        tree = toy()
        report = scored(tree)
        for weight in (None, True, -.1, float("inf")):
            with self.assertRaises(ValueError):
                make_targets(tree, report, length_weight=weight)
        with self.assertRaises(ValueError):
            make_targets(replace(tree, policy_sha256="other"), report, length_weight=0)

    def test_judge_error_is_audited_before_abort(self):
        class Backend:
            def identity(self):
                return {"model": "test", "revision": "fixed"}
            def complete(self, request):
                return ChatResult("ok", '{"thought":"bad","interactivity":2}', "test", "test", "fixed", {"cost": .01}, .1, 1)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"attempts.jsonl"
            judge = InteractivityJudge(Backend(), path)
            with self.assertRaises(ValueError):
                judge(visible_discussion(toy().leaves[0]))
            self.assertEqual(json.loads(path.read_text())["result"]["usage"]["cost"], .01)


class PilotTests(unittest.TestCase):
    def test_train_order_and_policy_required(self):
        trees, reports, receipt, pilot = fixture_pilot()
        self.assertEqual(pilot["leaf_distributions"]["success"]["count"], 60)
        self.assertIn("agent_ktokens", pilot["sibling_centered_distributions"])
        for pairs, data, policy in ((list(zip(trees, reports))[::-1], receipt, "theta0"),
                                    (zip(trees, reports), {**receipt, "train_verified": False}, "theta0"),
                                    (zip(trees, reports), receipt, "wrong")):
            with self.assertRaises(ValueError):
                summarize_pilot(pairs, data, initial_policy_sha256=policy)

    def test_freeze_is_immutable_and_tampering_rejected(self):
        trees, reports, _, pilot = fixture_pilot()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"weights.json"
            frozen = freeze_length_weight(pilot, length_weight=.2, rationale="synthetic test only", path=path)
            self.assertEqual(frozen_targets(trees[0], reports[0], frozen)["reward_weights"]["agent_ktokens"], -.2)
            with self.assertRaises(FileExistsError):
                freeze_length_weight(pilot, length_weight=.3, rationale="changed", path=path)
            with self.assertRaises(ValueError):
                frozen_targets(trees[0], reports[0], {**frozen, "length_weight": .9})

    def test_original_batch_updater_receives_only_original_traces(self):
        trees, reports, receipt, pilot = fixture_pilot()
        trees, reports = trees[:5], reports[:5]
        records = [{"case_id": t.case_id, "case_cursor": i, "policy_sha256": "theta0",
                    "tree": asdict(t), "update": {"optimizer_step": False}}
                   for i, t in enumerate(trees)]
        original = copy.deepcopy(records)
        class Updater:
            def update_groups(self, groups):
                self.groups = groups
                return {"optimizer_steps": 1}
        with tempfile.TemporaryDirectory() as temp:
            frozen = freeze_length_weight(pilot, length_weight=.2, rationale="fixture", path=Path(temp)/"weights.json")
            updater = Updater()
            prepared, result = update_prepared_batch(updater, records, reports, frozen_weights=frozen,
                data_receipt=receipt, expected_policy="theta0", actual_policy=lambda: "theta0")
            self.assertEqual(records, original)
            self.assertEqual(result["optimizer_steps"], 1)
            self.assertEqual(sum(len(g) for g in updater.groups), sum(len(t.nodes) for t in trees))
            for old, new in zip(records, prepared):
                self.assertEqual(old["tree"], new["tree"])
            with self.assertRaises(ValueError):
                prepare_batch(records, reports, frozen_weights=frozen,
                    data_receipt={**receipt, "all_verified": False}, expected_policy="theta0")

    def test_wrong_original_data_hash_cannot_pass(self):
        with tempfile.TemporaryDirectory() as temp:
            bad = Path(temp)/"data.jsonl"
            bad.write_text('{}\n')
            result = audit_data(train=bad, train_order=bad, test=bad)
            self.assertFalse(result["all_verified"])
            self.assertFalse(result["train_verified"])
            self.assertEqual({r["status"] for r in result["files"].values()}, {"mismatch"})


if __name__ == "__main__":
    unittest.main()
