"""Synthetic CPU integration; no clinical data, provider calls or CUDA training."""
import copy
import json
import tempfile
import unittest
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from src.online_rl.integrated import grpo
from src.online_rl.integrated.continuation import adapter_sha256
from src.online_rl.tree.cache import CallCache
from src.online_rl.tree.records import AgentNode, Leaf, Tree
from src.online_rl.tree.team import node_credit
from baselines.collabllm.launch import configuration, RewardHook, isolated_runner, bind_run, reuse_first_pilot_batch
from baselines.collabllm.pilot import summarize_pilot, freeze_length_weight
from baselines.collabllm.reward import Cl100kCounter, score_tree
from baselines.collabllm.test_reward import toy, Judge, Counter, fixture_pilot


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.adapter = torch.nn.Parameter(torch.zeros(8))
        self.config = SimpleNamespace(use_cache=True)
        self.disabled = False

    @contextmanager
    def disable_adapter(self):
        self.disabled = True
        try:
            yield
        finally:
            self.disabled = False

    def forward(self, input_ids, attention_mask, use_cache, logits_to_keep):
        logits = self.adapter * (0 if self.disabled else 1)
        return SimpleNamespace(logits=logits.reshape(1, 1, -1).expand(1, logits_to_keep, -1))


class VariedJudge(Judge):
    def __init__(self):
        super().__init__()
        self.identity = {"model": "synthetic", "revision": "cpu_test"}

    def __call__(self, messages):
        self.calls += 1
        return {"thought": "synthetic only", "interactivity": int(messages[-1]["content"][-1]) / 3}


def learning_fixture(model):
    policy = adapter_sha256(model)
    trees = []
    for case in range(20):
        nodes, leaves = [], []
        for sibling in range(4):
            nid = str(sibling)
            trace = {"agent_protocol": "precommit_v1", "state_prompt_hash": "same_state",
                     "segments": [{"prompt_input_ids": [1], "completion_ids": [sibling + 1]},
                                  {"prompt_input_ids": [1, sibling + 1], "completion_ids": [sibling + 1]}]}
            trace["behavior_logprobs"] = grpo.behavior_logprobs(model, [trace], 16)[0]
            dialogue = (("doctor", "D1", 1, "Synthetic question"), ("agent", "A1", 1, f"reply{sibling}"))
            nodes.append(AgentNode(nid, None, f"case{case}", 1, dialogue[:1], f"reply{sibling}",
                                   trace, policy, 0, "synthetic_stop", True, None))
            leaves.append(Leaf(nid, f"case{case}", (nid,), dialogue, "A", "A", 1., "synthetic_stop"))
        trees.append(Tree(f"case{case}", policy, 4, 4, tuple(nodes), tuple(leaves)))
    judge, counter = VariedJudge(), Counter()
    reports = [score_tree(t, judge=judge, cache=CallCache(None), token_counter=counter) for t in trees]
    receipt = {"train_verified": True, "all_verified": True, "reference_config_sha256": "synthetic",
               "train_case_order": [t.case_id for t in trees],
               "files": {k: {"actual_sha256": "synthetic_" + k, "path": "/synthetic/" + k}
                         for k in ("train", "train_order", "test")}}
    pilot = summarize_pilot(zip(trees, reports), receipt, initial_policy_sha256=policy)
    return trees, reports, receipt, pilot, judge, counter


def cpu_smoke():
    model = TinyModel()
    trees, reports, receipt, pilot, judge, counter = learning_fixture(model)
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        weights = freeze_length_weight(pilot, length_weight=.2, rationale="synthetic test, NOT a production lambda", path=root / "weights.json")
        cfg, audit = configuration(output=root / "run", receipt=receipt, judge=judge, counter=counter,
                                   weights=weights, initial_policy_sha256=adapter_sha256(model), base_model_revision="synthetic")
        hook = RewardHook(cfg, receipt, judge, counter, weights)
        runner = isolated_runner(hook)
        records = [{"case_id": t.case_id, "case_cursor": i, "policy_sha256": t.policy_sha256,
                    "prompt_sha256": cfg["rubric"]["prompt_sha256"], "tree": asdict(t),
                    "agent_targets": node_credit(1., 0.)(t, {}, {}), "update": {"optimizer_step": False}}
                   for i, t in enumerate(trees[:5])]
        updater = grpo.IntegratedGRPOUpdater(SimpleNamespace(torch=torch, model=model), cfg)
        before = adapter_sha256(model)
        baseline = runner.update_batch(updater, records, before, lambda: adapter_sha256(model))
        assert not baseline["optimizer_step"]
        runner.experiment.check_batch(cfg, records)
        captured = []
        original = grpo.grpo_loss
        def capture(m, rows, adv, *args):
            captured.extend(adv)
            return original(m, rows, adv, *args)
        with patch.object(grpo, "grpo_loss", side_effect=capture):
            result = runner.update_batch(updater, records, before, lambda: adapter_sha256(model))
        assert result["optimizer_step"] and result["optimizer_steps"] == 1
        assert adapter_sha256(model) != before and len(captured) == 20
        assert all(abs(a - b) < 1e-12 for a, b in zip(captured[:4], [-.5, -1/6, 1/6, .5]))
        bind_run(cfg, create=True)
        bind_run(copy.deepcopy(cfg))
        rejected = []
        for key in ("weights_sha256", "judge", "tokenizer", "initial_policy_sha256", "base_model_revision"):
            changed = copy.deepcopy(cfg)
            changed["collabllm"][key] = "changed"
            try:
                bind_run(changed)
            except RuntimeError:
                rejected.append(key)
            else:
                raise AssertionError("Changed resume identity accepted")
        return {"scope": "synthetic CPU; original launcher hook and GRPO backward/optimizer, not Qwen/GPU training",
                "outcome_only": baseline, "collabllm": result, "first_group_advantages": captured[:4],
                "adapter_changed": True, "resume_same_identity_accepted": True,
                "resume_changed_fields_rejected": rejected, "common_settings": audit}


class LaunchTests(unittest.TestCase):
    def test_pilot_reuse_is_limited_to_first_initial_policy_batch(self):
        model = TinyModel()
        trees, reports, receipt, pilot, judge, counter = learning_fixture(model)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            weights = freeze_length_weight(pilot, length_weight=.2, rationale="synthetic", path=root / "weights.json")
            args = dict(receipt=receipt, judge=judge, counter=counter,
                        initial_policy_sha256=adapter_sha256(model), base_model_revision="synthetic")
            pcfg, _ = configuration(output=root / "pilot", pilot_cases=20, **args)
            bind_run(pcfg, create=True)
            (root / "pilot/cases").mkdir()
            for i, tree in enumerate(trees[:5]):
                record = {"case_cursor": i, "case_id": tree.case_id, "tree": asdict(tree),
                          "policy_sha256": tree.policy_sha256, "collabllm": reports[i],
                          "batch_update": {"optimizer_steps": 0}}
                (root / "pilot/cases" / f"case_{i:06d}.json").write_text(json.dumps(record))
            cfg, _ = configuration(output=root / "main", weights=weights, **args)
            hook = RewardHook(cfg, receipt, judge, counter, weights)
            runner = isolated_runner(hook)
            reuse_first_pilot_batch(runner, hook, root / "pilot")
            def run_many(*args):
                return "fresh_generation"
            run_many.__module__ = "src.online_rl.tree.five_gpu_worker"
            executor = runner.ThreadPoolExecutor(max_workers=1)
            job = {"index": 0, "case": {"case_id": "case0"}, "worker": 0, "lane": 0}
            try:
                result = executor.submit(run_many, [job], "checkpoint", adapter_sha256(model)).result()
                self.assertEqual(result[0]["tree"], json.loads(json.dumps(asdict(trees[0]))))
                self.assertEqual(executor.submit(run_many, [{**job, "index": 5}], "checkpoint", "updated").result(), "fresh_generation")
                with self.assertRaises(RuntimeError):
                    executor.submit(run_many, [job], "checkpoint", "updated")
            finally:
                executor.shutdown()

    def test_actual_grpo_backward_and_resume_binding(self):
        self.assertTrue(cpu_smoke()["adapter_changed"])

    def test_actual_pinned_tokenizer_excludes_hidden_and_doctor(self):
        report = score_tree(toy(), judge=Judge(), cache=CallCache(None), token_counter=Cl100kCounter())
        self.assertEqual(report["leaves"]["a0"]["agent_visible_tokens"], 4)

    def test_five_cases_cannot_freeze(self):
        _, _, _, pilot = fixture_pilot(5)
        self.assertFalse(pilot["stability"]["ready_for_weight_review"])
        self.assertEqual(pilot["stability"]["next_stage_cases"], 20)
        with tempfile.TemporaryDirectory() as temp, self.assertRaises(ValueError):
            freeze_length_weight(pilot, length_weight=.2, rationale="test", path=Path(temp) / "weights.json")


if __name__ == "__main__":
    unittest.main()
