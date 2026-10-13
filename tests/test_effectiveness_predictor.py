"""Effectiveness pipeline, offline: labels -> pair data -> model / training -> metrics -> provider."""

import json
import math
from pathlib import Path

import pytest

from src.common.config import IntegrationUnavailable
from src.effectiveness.continuation import ContinuationSpec
from src.effectiveness.evaluate import evaluate_predictions
from src.effectiveness.history import format_input, history_at
from src.effectiveness.labels import (Candidate, build_pair_labels, collect_histories, label_metadata,
                                      original_candidates)
from src.effectiveness.metrics import binomial_nll as binomial_nll_metric
from src.effectiveness.pair_data import (Example, assert_case_disjoint, assert_manifest_splits, to_examples,
                                         validate_pair)
from test_serl_steps import fake_runtime, rich_trajectory

torch = pytest.importorskip("torch")

from src.effectiveness.model import CollaborationValueModel, binomial_nll  # noqa: E402
from src.effectiveness.providers import LearnedPredictorProvider  # noqa: E402
from src.effectiveness.synthetic import (SYNTHETIC_MODEL_ID, SYNTHETIC_REVISION,  # noqa: E402
                                         load_synthetic_backbone, synthetic_pair_rows)
from src.effectiveness.train import load_predictor_config, run_training  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SMOKE_CONFIG = ROOT / "configs/effectiveness/predictor_smoke.json"
SPEC = ContinuationSpec.from_dict({"k": 2, "max_rounds": 3, "final_decision_role": "doctor",
                                   "seed_policy": {"mode": "paired_by_history", "base_seed": 0}})


def outcome_runtime():
    """Continuations end correct only if the dialogue contains a response with 'GOOD'."""
    runtime, calls = fake_runtime(agent_text="BASE continuation reply.")
    calls["finals"] = 0

    def doctor(system, user):
        if "This is the final decision" in user:
            calls["finals"] += 1
            return "Final answer: option A - a" if "GOOD" in user else "Final answer: option B - b"
        return "Let us keep discussing."
    runtime.doctor = doctor
    return runtime, calls


def build(tmp_path, candidates=None, spec=SPEC, split="train"):
    trajectory = rich_trajectory("t1")
    histories = collect_histories([trajectory], turn_scope="first_collaboration_turn", include_initial_turn=False)
    candidates = candidates if candidates is not None else original_candidates([trajectory], histories)
    runtime, calls = outcome_runtime()
    counts = build_pair_labels(histories, candidates, runtime, spec, split=split, output_path=tmp_path / "labels.jsonl",
                               metadata=label_metadata(spec, "cfg", {"agent": "test", "doctor": "test"}))
    return counts, calls, histories


# ── labels ─────────────────────────────────────────────────────────────────────
def test_before_after_labels_and_before_cache(tmp_path):
    history_id = "t1:turn3"
    candidates = [Candidate(history_id, "base", "I simply agree.", "test"),
                  Candidate(history_id, "serl", "GOOD: let me verify the evidence first.", "test")]
    counts, calls, _ = build(tmp_path, candidates)
    assert counts["labels_written"] == 2 and counts["before_computed"] == 1 and counts["before_cached"] == 1
    assert calls["finals"] == SPEC.k * (1 + 2)  # BEFORE once per history, shared by both candidates
    rows = [json.loads(l) for l in (tmp_path / "labels.jsonl").read_text().splitlines()]
    by = {r["response_id"]: r for r in rows}
    assert by["base"]["before"] == by["serl"]["before"]
    assert (by["base"]["before"]["successes"], by["base"]["before"]["trials"]) == (0, 2)
    assert (by["serl"]["after"]["successes"], by["base"]["after"]["successes"]) == (2, 0)
    assert "ground_truth" not in json.dumps(by["serl"]["history"])
    assert by["serl"]["metadata"]["continuation_protocol_hash"] == SPEC.protocol_hash()
    rollouts = [json.loads(l) for l in (tmp_path / "labels.jsonl.rollouts.jsonl").read_text().splitlines()]
    assert len(rollouts) == 6 and all(r["dialogue"] for r in rollouts)  # raw rollouts preserved
    assert set(by["serl"]["after"]["rollout_ids"]) <= {r["rollout_id"] for r in rollouts}


def test_parallel_rollouts_give_the_same_labels(tmp_path):
    history_id = "t1:turn3"
    candidates = [Candidate(history_id, "base", "I simply agree.", "test"),
                  Candidate(history_id, "serl", "GOOD: let me verify the evidence first.", "test")]
    trajectory = rich_trajectory("t1")
    histories = collect_histories([trajectory], turn_scope="first_collaboration_turn", include_initial_turn=False)
    rows = {}
    for parallel in (1, 4):
        runtime, _ = outcome_runtime()
        out = tmp_path / f"p{parallel}" / "labels.jsonl"
        counts = build_pair_labels(histories, candidates, runtime, SPEC, split="train", output_path=out,
                                   metadata={}, parallel_rollouts=parallel)
        assert counts["labels_written"] == 2 and counts["before_computed"] == 1 and counts["before_cached"] == 1
        rows[parallel] = {r["response_id"]: (r["before"], r["after"])
                          for r in map(json.loads, out.read_text().splitlines())}
    assert rows[1] == rows[4]
    duplicated = build_pair_labels(histories, candidates + candidates[:1], outcome_runtime()[0], SPEC, split="train",
                                   output_path=tmp_path / "dup" / "labels.jsonl", metadata={}, parallel_rollouts=4)
    assert duplicated["labels_written"] == 2 and duplicated["skipped_done"] == 1  # duplicate row labelled once
    with pytest.raises(ValueError, match="seeding"):
        build_pair_labels(histories, candidates, outcome_runtime()[0], SPEC, split="train",
                          output_path=tmp_path / "x.jsonl", metadata={}, set_seed=lambda s: None, parallel_rollouts=2)


def test_labels_resume_and_split_guard(tmp_path):
    build(tmp_path)
    counts, calls, _ = build(tmp_path)
    assert counts["skipped_done"] == 1 and calls["finals"] == 0
    with pytest.raises(ValueError, match="never EVAL"):
        build(tmp_path / "e", split="eval")


def test_incomplete_rollouts_are_not_labels(tmp_path):
    trajectory = rich_trajectory("t1")
    histories = collect_histories([trajectory], turn_scope="first_collaboration_turn", include_initial_turn=False)
    runtime, _ = fake_runtime(final_text="no parsable answer")
    counts = build_pair_labels(histories, original_candidates([trajectory], histories), runtime, SPEC, split="train",
                               output_path=tmp_path / "l.jsonl", metadata={})
    assert counts["incomplete"] == 1 and counts["labels_written"] == 0
    assert (tmp_path / "l.jsonl.incomplete.jsonl").exists()


def test_history_and_input_format():
    t = rich_trajectory("t1")
    t["turns"][1]["reconstruction"] = {"secret": 1}
    history = history_at(t, 3)
    assert history["history_id"] == "t1:turn3" and history["round"] == 2
    assert all(set(turn) == {"speaker", "text", "round"} for turn in history["turns"])
    before, after = format_input(history), format_input(history, "My response.")
    assert "[STATE] before_response" in before and "[AGENT RESPONSE]" not in before
    assert "[STATE] after_response" in after and after.endswith("[AGENT RESPONSE]\nMy response.")
    assert "secret" not in after and "correct_option" not in after


# ── pair data, K, leakage ───────────────────────────────────────────────────────
def pair(case="c1", history="h1", response="r1", before=(1, 4), after=(3, 4), split="train", protocol="p"):
    return {"case_id": case, "original_case_id": case, "history_id": history, "response_id": response, "split": split,
            "history": {"turns": [{"speaker": "doctor", "text": "x", "round": 1}], "options": {}, "question": "q"},
            "response": "resp " + response,
            "before": {"successes": before[0], "trials": before[1]}, "after": {"successes": after[0], "trials": after[1]},
            "metadata": {"continuation_protocol_hash": protocol}}


def test_pair_validation_variable_k_and_counts():
    validate_pair(pair())
    for bad in (pair(after=(5, 4)), pair(after=(-1, 4)), pair(before=(1, 0)), pair(before=(1, 4), after=(1, 2))):
        with pytest.raises(ValueError):
            validate_pair(bad)
    with pytest.raises(ValueError, match="continuation k"):
        validate_pair(pair(), expected_k=2)
    with pytest.raises(ValueError, match="different continuation protocol"):
        validate_pair(pair(), protocol_hash="other")
    with pytest.raises(ValueError, match="EVAL"):
        validate_pair(pair(split="eval"))
    # Variable K across pairs is allowed when not pinned to the spec.
    examples = to_examples([pair(), pair(history="h2", before=(1, 2), after=(2, 2))])
    assert sorted({e.num_rollouts for e in examples}) == [2, 4]
    # BEFORE is one example per history, shared by its candidates; inconsistent counts are rejected.
    examples = to_examples([pair(), pair(response="r2", after=(0, 4))])
    assert [e.kind for e in examples] == ["before", "after", "after"]
    with pytest.raises(ValueError, match="inconsistent BEFORE"):
        to_examples([pair(), pair(response="r2", before=(2, 4))])


def test_case_level_leakage_detection():
    assert_case_disjoint({"train": [pair(case="a")], "dev": [pair(case="b", split="dev")]})
    with pytest.raises(ValueError, match="appears in"):
        assert_case_disjoint({"train": [pair(case="a")], "dev": [pair(case="a", split="dev")]})
    from src.data.schema import NormalizedCase
    from src.data.split import make_manifest

    cases = [NormalizedCase(case_id=str(i), original_case_id=str(i), source_dataset="src", dataset_version="1",
                            question="q", answer_format="text", synthetic=True) for i in range(10)]
    manifest = make_manifest(cases, seed=3, proportions={"train": 0.6, "dev": 0.2, "eval": 0.2})
    split_of = {json.loads(k)[1]: m["split"] for k, m in manifest["members"].items()}
    by = {s: [c for c, v in split_of.items() if v == s][0] for s in ("train", "dev", "eval")}
    assert_manifest_splits({"train": [pair(case=by["train"])], "dev": [pair(case=by["dev"], split="dev")]}, manifest, "src")
    with pytest.raises(ValueError, match="EVAL"):
        assert_manifest_splits({"train": [pair(case=by["eval"])]}, manifest, "src")


# ── model, loss, training ───────────────────────────────────────────────────────
def tiny_model(**kwargs):
    backbone, tokenizer = load_synthetic_backbone(SYNTHETIC_MODEL_ID, SYNTHETIC_REVISION)
    return CollaborationValueModel(backbone, tokenizer, base_model_id=SYNTHETIC_MODEL_ID,
                                   base_model_revision=SYNTHETIC_REVISION, max_length=64, **kwargs)


def test_binomial_nll_variable_k():
    logits = torch.tensor([0.3, -1.2])
    m, k = torch.tensor([3.0, 1.0]), torch.tensor([4.0, 2.0])
    p = torch.sigmoid(logits)
    manual = -(m * p.log() + (k - m) * (1 - p).log()).sum() / k.sum()
    assert torch.isclose(binomial_nll(logits, m, k), manual)
    assert math.isclose(binomial_nll_metric(p.tolist(), [3, 1], [4, 2]), float(manual), rel_tol=1e-5)


def test_frozen_backbone_and_head_gradient():
    model = tiny_model()
    assert model.training_mode == "frozen_backbone_scalar_head" and model.backbone_frozen
    assert {n for n, p in model.named_parameters() if p.requires_grad} == {"value_head.weight", "value_head.bias"}
    model.train()
    assert not model.backbone.training
    loss = model.binomial_nll(model(["a b", "c"]), torch.tensor([1.0, 0.0]), torch.tensor([2.0, 2.0]))
    loss.backward()
    assert model.value_head.weight.grad is not None and model.value_head.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.backbone.parameters())


def test_lora_is_an_optional_fallback(tmp_path):
    lora = {"r": 4, "alpha": 8, "dropout": 0.0, "target_modules": ["q"]}
    with pytest.raises(ValueError):
        tiny_model(training_mode="lora_scalar_head")  # needs lora settings
    with pytest.raises(ValueError):
        tiny_model(lora=lora)  # lora settings only with lora_scalar_head
    import importlib.util

    if importlib.util.find_spec("peft") is None:
        with pytest.raises(IntegrationUnavailable, match="peft"):
            tiny_model(training_mode="lora_scalar_head", lora=lora)
    raw = json.loads(SMOKE_CONFIG.read_text())
    raw["predictor"]["training_mode"] = "lora_scalar_head"
    (tmp_path / "c.json").write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="pilot_decision"):
        load_predictor_config(tmp_path / "c.json")


def test_checkpoint_round_trip_and_tamper(tmp_path):
    model = tiny_model()
    with torch.no_grad():
        model.value_head.weight.normal_()
    texts = ["alpha beta", "gamma"]
    model.save_pretrained(tmp_path / "ckpt", metadata={"continuation_protocol_hash": "p"})
    reloaded = CollaborationValueModel.from_pretrained(tmp_path / "ckpt", backbone_loader=load_synthetic_backbone)
    assert reloaded.predict(texts, 2) == pytest.approx(model.predict(texts, 2))
    config = json.loads((tmp_path / "ckpt/predictor_config.json").read_text())
    assert config["backbone_frozen"] and not (tmp_path / "ckpt/backbone").exists()
    config["trainable_state_sha256"] = "0" * 64
    (tmp_path / "ckpt/predictor_config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="checksum"):
        CollaborationValueModel.from_pretrained(tmp_path / "ckpt", backbone_loader=load_synthetic_backbone)


def write_labels(tmp_path, k=2):
    spec = load_predictor_config(SMOKE_CONFIG)["continuation_spec"]
    tmp_path.mkdir(parents=True, exist_ok=True)
    paths = {}
    for split, n, seed in (("train", 6, 1), ("dev", 3, 2)):
        paths[split] = tmp_path / f"{split}.jsonl"
        paths[split].write_text("\n".join(json.dumps(r) for r in synthetic_pair_rows(
            n, k, spec.protocol_hash(), seed=seed, prefix=split, split=split)) + "\n")
    return paths


def test_training_end_to_end_resume_guards_and_metrics(tmp_path):
    paths = write_labels(tmp_path)
    raw = json.loads(SMOKE_CONFIG.read_text())
    raw["training"]["epochs"] = 2
    config = tmp_path / "config.json"
    config.write_text(json.dumps(raw))
    summary = run_training(config, paths["train"], paths["dev"], tmp_path / "run")
    assert summary["checkpoint_round_trip"] == "pass" and summary["train_pairs"] == 12
    metrics = summary["dev_metrics_best_checkpoint"]
    assert set(metrics) == {"absolute", "ranking", "delta"}
    assert set(metrics["absolute"]) == {"all", "before", "after"} and metrics["delta"]["count"] == 6
    with pytest.raises(ValueError, match="--resume"):
        run_training(config, paths["train"], paths["dev"], tmp_path / "run")
    raw["training"]["learning_rate"] = 0.5
    config.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="Refusing to resume"):
        run_training(config, paths["train"], paths["dev"], tmp_path / "run", resume=True)
    # K in the labels must match the spec unless explicitly relaxed.
    other = write_labels(tmp_path / "k4", k=4)
    with pytest.raises(ValueError, match="continuation k"):
        run_training(SMOKE_CONFIG, other["train"], other["dev"], tmp_path / "k4run")


def test_train_dev_case_overlap_is_rejected(tmp_path):
    paths = write_labels(tmp_path)
    dev_rows = [json.loads(l) for l in paths["dev"].read_text().splitlines()]
    train_first = json.loads(paths["train"].read_text().splitlines()[0])
    dev_rows[0].update(case_id=train_first["case_id"], original_case_id=train_first["original_case_id"])
    paths["dev"].write_text("\n".join(json.dumps(r) for r in dev_rows) + "\n")
    with pytest.raises(ValueError, match="appears in"):
        run_training(SMOKE_CONFIG, paths["train"], paths["dev"], tmp_path / "run")


def test_metrics_separate_absolute_ranking_delta():
    examples = [Example("before", "h1", None, "c", "c", "b", 1, 4),
                Example("after", "h1", "r1", "c", "c", "a1", 3, 4),
                Example("after", "h1", "r2", "c", "c", "a2", 1, 4)]
    good = evaluate_predictions([0.25, 0.75, 0.25], examples)
    assert good["ranking"]["within_history_pairwise_accuracy"] == 1.0
    assert good["delta"]["mae"] == pytest.approx(0.0) and good["delta"]["sign_agreement_nonzero"] == 1.0
    assert good["delta"]["nonzero_count"] == 1  # r2 has rollout delta 0 and is left out of sign agreement
    swapped = evaluate_predictions([0.25, 0.25, 0.75], examples)
    assert swapped["ranking"]["within_history_pairwise_accuracy"] == 0.0
    assert swapped["absolute"]["before"]["mae"] == pytest.approx(0.0)


# ── provider ───────────────────────────────────────────────────────────────────
def test_learned_provider_before_after_delta(tmp_path):
    model = tiny_model()
    with torch.no_grad():
        model.value_head.weight.normal_()
    model.save_pretrained(tmp_path / "ckpt", metadata={"continuation_protocol_hash": "p"})
    provider = LearnedPredictorProvider.from_checkpoint(str(tmp_path / "ckpt"), expected_protocol_hash="p",
                                                        backbone_loader=load_synthetic_backbone)
    history = history_at(rich_trajectory("t1"), 3)
    before = provider.predict_before(history)
    after = provider.predict_after(history, "I verify the evidence first.")
    assert before == pytest.approx(model.predict([format_input(history)], 1)[0])
    assert provider.predict_delta(history, "I verify the evidence first.") == pytest.approx(after - before)
    assert 0 <= before <= 1 and before != after


def test_dev_report_baselines_intervals_and_ties():
    from src.effectiveness.metrics import grouped_pairwise_ranking
    from src.effectiveness.report import baselines, brier_noise_floor, dev_report, label_diagnostics

    # A predictor that cannot separate candidates scores chance (0.5), not 0.
    assert grouped_pairwise_ranking([0.4, 0.4], [1.0, 0.0], ["h", "h"])["accuracy"] == 0.5
    train = [Example("before", "t1", None, "a", "a", "x", 1, 4), Example("after", "t1", "r", "a", "a", "y", 3, 4)]
    dev = []
    for i in range(6):
        dev += [Example("before", f"h{i}", None, f"c{i}", f"c{i}", "b", 2, 4),
                Example("after", f"h{i}", "good", f"c{i}", f"c{i}", "g", 4, 4),
                Example("after", f"h{i}", "bad", f"c{i}", f"c{i}", "x", 0, 4)]
    model = [0.5, 0.95, 0.05] * 6
    base = baselines(model, dev, train)
    assert set(base["train_mean"]) == {0.5} and base["train_mean_by_kind"][:3] == [0.25, 0.75, 0.75]
    assert base["no_response_effect"][:3] == [0.5, 0.5, 0.5]  # AFTER copies the model's BEFORE -> delta 0
    assert set(base["dev_mean_reference"]) == {0.5}  # (2 + 4 + 0) / 12 per history
    report = dev_report(model, dev, train, n_boot=200)
    ranking = report["comparison"]["ranking.within_history_pairwise_accuracy"]
    assert ranking["model"] == 1.0 and ranking["no_response_effect"]["value"] == 0.5 and ranking["chance"] == 0.5
    brier = report["comparison"]["absolute.after.brier"]["train_mean"]
    assert brier["model_minus_baseline"] < 0 and brier["ci_high"] < 0  # model clearly better on this toy set
    assert report["label_diagnostics"]["dev"]["histories_where_candidates_differ"] == 6
    assert label_diagnostics(dev)["after"]["share_all_fail"] == 0.5
    assert brier_noise_floor(dev) == pytest.approx((6 * (0.25 / 3)) / 18)


def test_format_failures_are_retried_outcomes_are_not(tmp_path):
    trajectory = rich_trajectory("t1")
    histories = collect_histories([trajectory], turn_scope="first_collaboration_turn", include_initial_turn=False)
    runtime, calls = outcome_runtime()
    finals = {"n": 0}
    good_doctor = runtime.doctor

    def flaky_doctor(system, user):
        if "This is the final decision" in user:
            finals["n"] += 1
            if finals["n"] % 2 == 1:  # every first attempt returns an unparsable final answer
                return "I cannot decide."
        return good_doctor(system, user)
    runtime.doctor = flaky_doctor
    counts = build_pair_labels(histories, original_candidates([trajectory], histories), runtime, SPEC, split="train",
                               output_path=tmp_path / "l.jsonl", metadata={}, max_rollout_attempts=2)
    assert counts["labels_written"] == 1 and counts["incomplete"] == 0
    rollouts = [json.loads(l) for l in (tmp_path / "l.jsonl.rollouts.jsonl").read_text().splitlines()]
    assert len(rollouts) == 2 * SPEC.k * 2 and {r["attempt"] for r in rollouts} == {1, 2}
    row = json.loads((tmp_path / "l.jsonl").read_text())
    assert row["before"]["trials"] == row["after"]["trials"] == SPEC.k  # K stays K; failed attempts are not trials


def test_lora_micro_batches_match_full_batch_and_adapters_train(tmp_path):
    pytest.importorskip("peft")
    import torch

    from src.effectiveness.train import fit

    lora = {"r": 4, "alpha": 8, "dropout": 0.0, "target_modules": ["q"]}
    examples = [Example("after", f"h{i}", "r", f"c{i}", f"c{i}", f"text number {i} " * (1 + i % 3), i % 5, 4)
                for i in range(10)]
    training = {"seed": 0, "epochs": 2, "learning_rate": 0.05, "lora_learning_rate": 0.01, "weight_decay": 0.0,
                "batch_size": 8, "grad_clip": 1.0, "selection_metric": "binomial_nll", "cache_features": True}
    states = {}
    for micro in (8, 3):
        torch.manual_seed(0)
        model = tiny_model(training_mode="lora_scalar_head", lora=lora)
        initial = model.trainable_state()
        fit(model, examples, examples[:4], dict(training, micro_batch_size=micro), tmp_path / f"m{micro}",
            fingerprint="f", resume=False)
        states[micro] = model.trainable_state()
    adapter_names = [n for n in states[8] if "lora_" in n]
    assert adapter_names and any(not torch.equal(states[8][n], initial[n]) for n in adapter_names)  # adapters moved
    for name in states[8]:  # accumulating over chunks gives the full-batch update
        assert torch.allclose(states[8][name], states[3][name], atol=1e-4), name
    # LoRA checkpoint round trip: adapters + head are restored on a freshly loaded backbone.
    texts = [e.text for e in examples]
    expected = model.predict(texts, 4)
    model.save_pretrained(tmp_path / "ckpt")
    reloaded = CollaborationValueModel.from_pretrained(tmp_path / "ckpt", backbone_loader=load_synthetic_backbone)
    assert reloaded.predict(texts, 4) == pytest.approx(expected, abs=1e-6)
    frozen = tiny_model()
    assert frozen.predict(texts, 4) != pytest.approx(expected, abs=1e-3)  # the reload is not just the bare backbone


def test_reports_split_natural_and_controlled_and_rank_same_answer_behaviors():
    from src.effectiveness.report import is_natural, same_answer_behavior_ranking, subset_report

    def history(i, case, evidence=4, plain=0):
        return [Example("before", f"h{i}", None, case, case, "b", 2, 4),
                Example("after", f"h{i}", "original", case, case, "o", 2, 4),
                Example("after", f"h{i}", "sample_1", case, case, "s", 4, 4),
                Example("after", f"h{i}", "evidence_grounded_maintain", case, case, "e", evidence, 4),
                Example("after", f"h{i}", "unsupported_maintain", case, case, "u", plain, 4)]

    dev = [e for i in range(6) for e in history(i, f"d{i}")] + history(6, "d6", evidence=4, plain=4)
    train = [e for i in range(6) for e in history(i, f"t{i}")]
    assert [is_natural(e) for e in dev[:5]] == [True, True, True, False, False]
    predictions = [0.5, 0.5, 0.9, 0.9, 0.1] * 7
    natural = subset_report(predictions, dev, train, natural=True, n_boot=100)
    controlled = subset_report(predictions, dev, train, natural=False, n_boot=100)
    assert natural["label_diagnostics"]["dev"]["after"]["count"] == 14  # original + sample_1 per history
    assert controlled["label_diagnostics"]["dev"]["after"]["count"] == 14  # the two maintain candidates
    assert natural["label_diagnostics"]["dev"]["before"]["count"] == controlled["label_diagnostics"]["dev"]["before"]["count"] == 7
    assert natural["comparison"]["ranking.within_history_pairwise_accuracy"]["model"] == 1.0
    ranking = same_answer_behavior_ranking(predictions, dev)
    assert ranking == {"pair": ["evidence_grounded_maintain", "unsupported_maintain"], "accuracy": 1.0,
                       "informative_pairs": 6, "pairs_with_equal_outcome": 1, "chance": 0.5}
    assert same_answer_behavior_ranking([0.5] * len(dev), dev)["accuracy"] == 0.5  # a predictor that cannot tell them apart
    only_natural = [e for e in dev if is_natural(e)]
    assert subset_report([0.5] * len(only_natural), only_natural, train, natural=False, n_boot=100) is None


def test_continuation_guideline_only_after_the_continued_turn_and_label_fields(tmp_path):
    guideline = "Continue consistently with your previous response."
    trajectory = rich_trajectory("t1")
    histories = collect_histories([trajectory], turn_scope="first_collaboration_turn", include_initial_turn=False)
    prompts = []
    runtime, _ = outcome_runtime()
    agent = runtime.agent
    runtime.agent = lambda system, user, tokens: (prompts.append((tokens, user)), agent(system, user, tokens))[1]
    runtime.continuation_guideline = guideline
    candidate = Candidate("t1:turn3", "evidence_grounded_maintain", "GOOD: the lab result supports A.", "test",
                          {"candidate_type": "controlled", "intended_behavior": "evidence_grounded_maintain",
                           "observed_behavior": "MAINTAIN_CHALLENGE", "compliance": True,
                           "meta": {"agent_initial_answer": "A", "doctor_claim": "B"}})
    build_pair_labels(histories, [candidate], runtime, SPEC, split="train", output_path=tmp_path / "labels.jsonl",
                      metadata={})
    responses = [user for tokens, user in prompts if tokens == runtime.max_tokens]
    with_guideline = [guideline in user for user in responses]
    # BEFORE: the turn being continued is written without the guideline, the turn after it with it.
    # AFTER: the continued turn is the fixed response, so every generated agent turn comes after it.
    assert with_guideline.count(False) == SPEC.k and with_guideline.count(True) == 2 * SPEC.k
    row = json.loads((tmp_path / "labels.jsonl").read_text())
    assert (row["before_rate"], row["after_rate"], row["delta_effectiveness"]) == (0.0, 1.0, 1.0)
    assert row["candidate_type"] == "controlled" and row["compliance"] is True
    assert row["intended_behavior"] == "evidence_grounded_maintain" and row["meta"]["doctor_claim"] == "B"
    assert "ground_truth" not in json.dumps(row["history"])


def test_checkpoint_selection_can_use_natural_responses_only(tmp_path):
    from src.effectiveness.train import fit

    train = [Example("after", f"h{i}", "original", f"c{i}", f"c{i}", f"text {i}", i % 5, 4) for i in range(8)]
    dev = [Example("after", "d1", "original", "d", "d", "natural text", 4, 4),
           Example("after", "d1", "advocate_B", "d", "d", "guided text", 0, 4)]
    training = {"seed": 0, "epochs": 2, "learning_rate": 0.05, "weight_decay": 0.0, "batch_size": 8, "grad_clip": 1.0,
                "selection_metric": "binomial_nll", "cache_features": True}
    best = {}
    for subset in ("all", "natural"):
        result = fit(tiny_model(), train, dev, dict(training, selection_subset=subset), tmp_path / subset,
                     fingerprint="f", resume=False)
        epochs = result["history"]
        assert all(e["dev_natural"]["after"]["count"] == 1 and e["dev"]["absolute"]["after"]["count"] == 2 for e in epochs)
        key = (lambda e: e["dev_natural"]["all"]["binomial_nll"]) if subset == "natural" \
            else (lambda e: e["dev"]["absolute"]["all"]["binomial_nll"])
        assert result["best"]["binomial_nll"] == pytest.approx(min(key(e) for e in epochs))
        best[subset] = result["best"]["binomial_nll"]
    assert best["all"] != best["natural"]


def test_last_collaboration_turn_scope_is_the_final_round_only():
    trajectory = rich_trajectory("t1")  # agent turns at rounds 1 and 2, max_discussion_rounds 3
    assert collect_histories([trajectory], turn_scope="last_collaboration_turn", include_initial_turn=False) == []
    trajectory["turns"] += [{"speaker": "doctor", "text": "Still B.", "round": 3},
                            {"speaker": "agent", "text": "A.", "round": 3, "local_judgment_status": "valid"}]
    histories = collect_histories([trajectory], turn_scope="last_collaboration_turn", include_initial_turn=False)
    assert [h["history_id"] for h, _ in histories] == ["t1:turn5"] and histories[0][0]["round"] == 3


def test_history_batches_ranking_pairs_and_chunked_ranking_loss():
    import torch

    from src.effectiveness.pair_data import Example
    from src.effectiveness.train import history_batches, ranking_loss, ranking_pairs

    def example(history, kind, successes, response=None):
        return Example(kind=kind, history_id=history, response_id=response, case_id=history, original_case_id=history,
                       text="t", num_success=successes, num_rollouts=9)

    examples = [example("h1", "before", 5), example("h1", "after", 9, "a"), example("h1", "after", 7, "b"),
                example("h1", "after", 8, "c"), example("h2", "before", 0), example("h2", "after", 0, "a"),
                example("h2", "after", 9, "b")]
    batches = history_batches(examples, [5, 0, 6, 1, 4, 2, 3], batch_size=4)
    assert [sorted(b) for b in batches] == [[4, 5, 6], [0, 1, 2, 3]]  # a history is never split
    index = [0, 1, 2, 3]
    # only AFTER pairs whose rollout rates differ by at least 2/9: (9 vs 7); 9 vs 8 and 8 vs 7 are too close
    assert ranking_pairs(examples, index, 2 / 9) == [(1, 2)]
    assert ranking_pairs(examples, [4, 5, 6], 2 / 9) == [(2, 1)] and ranking_pairs(examples, index, 1.5) == []

    logits = torch.tensor([0.0, 0.2, 1.0, -0.3], requires_grad=True)
    pairs = [(1, 2), (3, 0)]
    full = ranking_loss(logits, logits.detach(), pairs, [0, 1, 2, 3])
    full.backward()
    expected = logits.grad.clone()
    logits.grad = None
    for members in ([0, 1], [2, 3]):  # the same batch in two chunks, the partner's logit detached
        ranking_loss(logits[members], logits.detach(), pairs, members).backward()
    assert torch.allclose(logits.grad, expected) and float(full) > 0
    assert ranking_loss(logits[[0]], logits.detach(), [(1, 2)], [0]) is None


def test_monotonic_calibration_keeps_order_and_lowers_nll():
    from src.effectiveness import calibration
    from src.effectiveness.metrics import binomial_nll
    from src.effectiveness.pair_data import Example

    truth = [0.1, 0.3, 0.5, 0.7, 0.9] * 8
    overconfident = [0.001, 0.05, 0.5, 0.95, 0.999] * 8  # right order, probabilities pushed to the extremes
    successes = [round(9 * t) for t in truth]
    for method in calibration.METHODS:
        parameters = calibration.fit(overconfident, successes, [9] * 40, method)
        fixed = calibration.apply(parameters, overconfident)
        assert parameters["a"] < 1 and sorted(range(40), key=lambda i: (fixed[i], i)) == sorted(range(40), key=lambda i: (overconfident[i], i))
        assert binomial_nll(fixed, successes, [9] * 40) < binomial_nll(overconfident, successes, [9] * 40)
    examples = [Example(kind="after", history_id=f"h{i % 10}", response_id=str(i), case_id=f"c{i % 10}",
                        original_case_id=f"c{i % 10}", text="t", num_success=s, num_rollouts=9) for i, s in enumerate(successes)]
    crossed = calibration.cross_fitted(overconfident, examples, "temperature", folds=5)
    assert None not in crossed and binomial_nll(crossed, successes, [9] * 40) < binomial_nll(overconfident, successes, [9] * 40)
    with pytest.raises(ValueError):
        calibration.fit(overconfident, successes, [9] * 40, "isotonic")


def test_ordering_breakdown_counts_histories_once_and_groups_by_state():
    from src.effectiveness.evaluate import ordering_breakdown
    from src.effectiveness.pair_data import Example

    def after(history, response, successes):
        return Example(kind="after", history_id=history, response_id=response, case_id=history, original_case_id=history,
                       text="t", num_success=successes, num_rollouts=9)

    examples = [after("h1", "a", 9), after("h1", "b", 0), after("h1", "c", 5),     # three informative pairs, all right
                after("h2", "a", 9), after("h2", "b", 0),                            # one informative pair, wrong
                after("h3", "a", 4), after("h3", "b", 4)]                            # no informative pair
    predictions = [0.9, 0.1, 0.5, 0.2, 0.8, 0.5, 0.6]
    result = ordering_breakdown(predictions, examples, {"h1": "Challenge", "h2": "Accept", "h3": "Preserve"}.get)
    assert result["overall"] == {"accuracy": 0.75, "pairs": 4, "histories": 2}
    assert result["informative_histories"] == {"accuracy": 0.5, "histories": 2}
    assert result["by_group"] == {"Accept": {"accuracy": 0.0, "pairs": 1, "histories": 1},
                                  "Challenge": {"accuracy": 1.0, "pairs": 3, "histories": 1}}


def test_training_keeps_every_epoch_and_supports_ranking_and_head_init(tmp_path):
    paths = write_labels(tmp_path)
    raw = json.loads(SMOKE_CONFIG.read_text())
    raw["training"]["epochs"] = 3
    config = tmp_path / "config.json"
    config.write_text(json.dumps(raw))
    run_training(config, paths["train"], paths["dev"], tmp_path / "frozen")
    run = tmp_path / "frozen"
    assert sorted(p.name for p in (run / "epochs").iterdir()) == ["epoch1", "epoch2", "epoch3"]
    listing = json.loads((run / "checkpoints.json").read_text())
    assert [r["epoch"] for r in listing["epochs"]] == [1, 2, 3] and listing["selection_metric"] == "binomial_nll"
    assert listing["best_selection"]["binomial_nll"] == min(r["binomial_nll"] for r in listing["epochs"])
    assert (run / "best_ordering" / "trainable_state.pt").is_file() and listing["best_ordering"]["epoch"] in (1, 2, 3)
    for name, epoch in (("best", listing["best_selection"]["epoch"]), ("best_ordering", listing["best_ordering"]["epoch"])):
        saved = json.loads((run / name / "predictor_config.json").read_text())
        assert saved["metadata"]["epoch"] == epoch

    # ranking loss with history-grouped batches, head started from the first run's best checkpoint (both paths:
    # cached features, and chunked forward passes with micro_batch_size)
    for label, extra in (("cached", {}), ("chunked", {"cache_features": False, "micro_batch_size": 2})):
        raw["training"].update(ranking={"lambda": 0.2, "min_after_gap": 0.5}, init_head_from=str(run / "best"),
                               learning_rate=0.0, **extra)
        config.write_text(json.dumps(raw))
        summary = run_training(config, paths["train"], paths["dev"], tmp_path / label)
        # with a zero learning rate the head stays the initial one: same DEV NLL as the checkpoint it came from
        assert summary["best"]["binomial_nll"] == pytest.approx(listing["best_selection"]["binomial_nll"], abs=1e-6)


def test_residual_predictor_starts_at_before_and_learns_the_response_effect(tmp_path):
    import torch

    from src.effectiveness.model import CollaborationValueModel
    from src.effectiveness.pair_data import load_pairs, to_examples
    from src.effectiveness.providers import LearnedPredictorProvider
    from src.effectiveness.train import residual_step

    paths = write_labels(tmp_path)
    raw = json.loads(SMOKE_CONFIG.read_text())
    raw["predictor"]["architecture"] = "residual"
    raw["training"].update(epochs=3, ranking={"lambda": 0.2, "min_after_gap": 0.5}, residual={"regularization": 0.01})
    config = tmp_path / "config.json"
    config.write_text(json.dumps(raw))

    model = tiny_model(architecture="residual")
    spec = ContinuationSpec.from_dict(raw["continuation"])
    train = to_examples(load_pairs(paths["train"], split="train", protocol_hash=spec.protocol_hash(), expected_k=spec.k))
    history = train[0].history_id
    group = [i for i, e in enumerate(train) if e.history_id == history]
    probabilities = model.predict_examples([train[i] for i in group], 4)
    assert len(set(round(p, 6) for p in probabilities)) == 1  # G = 0: P(success | h, r) = P(success | h)
    successes = torch.tensor([e.num_success for e in train], dtype=torch.float32)
    counts = torch.tensor([e.num_rollouts for e in train], dtype=torch.float32)
    after_only = [i for i in group if train[i].kind == "after"]
    before = [i for i in group if train[i].kind == "before"]
    model.zero_grad()
    residual_step(model, train, before + after_only, successes, counts, micro=1, pairs=[], scale=0.0, regularization=0.0)
    full = model.value_head.weight.grad.clone()
    model.zero_grad()
    residual_step(model, train, before, successes, counts, micro=1, pairs=[], scale=0.0, regularization=0.0)
    # B(h) gets its gradient from the BEFORE example alone; the AFTER terms reach only G
    assert torch.allclose(full, model.value_head.weight.grad) and model.effect_head.weight.grad is None

    summary = run_training(config, paths["train"], paths["dev"], tmp_path / "run")
    assert summary["checkpoint_round_trip"] == "pass" and summary["epochs_completed"] == 3
    saved = json.loads((tmp_path / "run/best/predictor_config.json").read_text())
    assert saved["architecture"] == "residual"
    state = torch.load(tmp_path / "run/best/trainable_state.pt")
    assert "effect_head.weight" in state and bool(state["effect_head.weight"].abs().sum() > 0)
    assert sorted(p.name for p in (tmp_path / "run/epochs").iterdir()) == ["epoch1", "epoch2", "epoch3"]

    from src.effectiveness.synthetic import load_synthetic_backbone
    provider = LearnedPredictorProvider.from_checkpoint(str(tmp_path / "run/best"), backbone_loader=load_synthetic_backbone)
    history_input = {"question": "q", "options": {}, "scenario": "s", "turns": [{"speaker": "doctor", "text": "d"}]}
    before_p, after_p = provider.predict_before(history_input), provider.predict_after(history_input, "a response")
    assert 0 < before_p < 1 and 0 < after_p < 1 and provider.predict_delta(history_input, "a response") == pytest.approx(after_p - before_p)
    reloaded = CollaborationValueModel.from_pretrained(tmp_path / "run/best", backbone_loader=load_synthetic_backbone)
    with pytest.raises(ValueError, match="BEFORE example"):
        reloaded.example_logits([e for e in train if e.kind == "after"][:2], 2)


def test_residual_with_frozen_before_keeps_b_identical_and_trains_only_g(tmp_path):
    pytest.importorskip("peft")
    import torch

    from src.effectiveness.pair_data import load_pairs, to_examples
    from src.effectiveness.train import fit

    paths = write_labels(tmp_path)
    spec = ContinuationSpec.from_dict(json.loads(SMOKE_CONFIG.read_text())["continuation"])
    train, dev = (to_examples(load_pairs(paths[name], split=name, protocol_hash=spec.protocol_hash(), expected_k=spec.k))
                  for name in ("train", "dev"))
    lora = {"r": 4, "alpha": 8, "dropout": 0.0, "target_modules": ["q"]}

    torch.manual_seed(0)
    baseline = tiny_model()  # the frozen baseline: bare backbone + head
    with torch.no_grad():
        baseline.value_head.weight.normal_(0, 0.5)
    baseline_logits = baseline.example_logits(dev, 8)
    (tmp_path / "baseline_logits.json").write_text(json.dumps(baseline_logits))

    model = tiny_model(training_mode="lora_scalar_head", lora=lora, architecture="residual_frozen_before")
    model.value_head.load_state_dict(baseline.value_head.state_dict())
    head_before = model.value_head.weight.detach().clone()
    initial = model.trainable_state()
    assert "value_head.weight" in initial and not model.value_head.weight.requires_grad  # stored, not trained
    assert all(not n.startswith("value_head") for n, p in model.named_parameters() if p.requires_grad)
    start = model.example_logits(dev, 8)
    before = [i for i, e in enumerate(dev) if e.kind == "before"]
    assert [start[i] for i in before] == [baseline_logits[i] for i in before]  # B is the frozen baseline, exactly
    b_of = {dev[i].history_id: start[i] for i in before}
    assert all(start[i] == b_of[e.history_id] for i, e in enumerate(dev) if e.kind == "after")  # G = 0

    training = {"seed": 0, "epochs": 2, "learning_rate": 0.05, "lora_learning_rate": 0.01, "weight_decay": 0.0,
                "batch_size": 8, "micro_batch_size": 3, "grad_clip": 1.0, "selection_metric": "binomial_nll",
                "cache_features": True, "ranking": {"lambda": 0.2, "min_after_gap": 0.5},
                "residual": {"regularization": 0.01}, "frozen_before_baseline_logits": str(tmp_path / "baseline_logits.json")}
    fit(model, train, dev, training, tmp_path / "run", fingerprint="f", resume=False)
    sanity = json.loads((tmp_path / "run/sanity.json").read_text())
    assert sanity["passed"] and sanity["max_abs_diff_to_frozen_baseline_before_logits"] == 0
    assert sanity["max_abs_diff_before_logits_with_random_adapters"] == 0 and sanity["adapter_path_changes_with_random_adapters"]
    assert sanity["optimizer_parameter_kinds"] == ["effect_head", "lora"]

    end = model.example_logits(dev, 8)
    state = model.trainable_state()
    assert [end[i] for i in before] == [baseline_logits[i] for i in before]  # B did not move by a single bit
    assert torch.equal(model.value_head.weight, head_before)
    assert any(not torch.equal(state[n], initial[n]) for n in state if "lora_" in n)  # the adapters learned
    assert bool(state["effect_head.weight"].abs().sum() > 0)
    assert any(end[i] != b_of[e.history_id] for i, e in enumerate(dev) if e.kind == "after")  # and G is no longer 0

    reloaded = CollaborationValueModel.from_pretrained(tmp_path / "run/best", backbone_loader=load_synthetic_backbone)
    assert reloaded.architecture == "residual_frozen_before" and torch.equal(reloaded.value_head.weight.cpu(), head_before.cpu())

    # a baseline that does not match stops the run before training
    (tmp_path / "other.json").write_text(json.dumps([z + 0.5 for z in baseline_logits]))
    fresh = tiny_model(training_mode="lora_scalar_head", lora=lora, architecture="residual_frozen_before")
    with pytest.raises(RuntimeError, match="differs from the frozen baseline"):
        fit(fresh, train, dev, dict(training, frozen_before_baseline_logits=str(tmp_path / "other.json")),
            tmp_path / "bad", fingerprint="f", resume=False)


def test_gap_weighted_ranking_pairs_and_train_only_selection(tmp_path, monkeypatch):
    pytest.importorskip("peft")
    import torch

    from scripts import train_residual_cv
    from src.effectiveness.train import pair_weight_total, ranking_loss, ranking_pairs

    def example(history, kind, successes, response=None):
        return Example(kind=kind, history_id=history, response_id=response, case_id=history, original_case_id=history,
                       text="t", num_success=successes, num_rollouts=9)

    examples = [example("h", "after", 9, "a"), example("h", "after", 7, "b"), example("h", "after", 2, "c")]
    pairs = ranking_pairs(examples, [0, 1, 2], 2 / 9, gap_weighted=True)
    assert [(a, b) for a, b, _ in pairs] == [(0, 1), (0, 2), (1, 2)]
    assert [w for _, _, w in pairs] == pytest.approx([2 / 9, 7 / 9, 5 / 9]) and pair_weight_total(pairs) == pytest.approx(14 / 9)
    logits = torch.zeros(3)
    weighted, plain = ranking_loss(logits, logits, pairs, [0, 1, 2]), ranking_loss(logits, logits, [(a, b) for a, b, _ in pairs], [0, 1, 2])
    assert float(weighted) == pytest.approx(float(plain) * (14 / 9) / 3)  # softplus(0) each, times the gaps

    # the selection rule: stay near the best AFTER NLL, then take the best ordering
    candidates = [{"lambda_rank": 0.2, "epoch": 1, "after_nll": 0.60, "ordering": 0.70, "delta_sign": 0.7},
                  {"lambda_rank": 0.5, "epoch": 2, "after_nll": 0.63, "ordering": 0.78, "delta_sign": 0.7},
                  {"lambda_rank": 1.0, "epoch": 3, "after_nll": 0.90, "ordering": 0.85, "delta_sign": 0.8}]
    assert train_residual_cv.choose(candidates, 0.05)["lambda_rank"] == 0.5  # 1.0 ranks best but ruins the NLL
    assert train_residual_cv.choose(candidates, 0.5)["lambda_rank"] == 1.0

    # cross-validation and the final run on the synthetic backbone; DEV labels are never opened
    paths = write_labels(tmp_path)
    torch.manual_seed(0)
    baseline = tiny_model()
    baseline.save_pretrained(tmp_path / "frozen_best")
    raw = json.loads(SMOKE_CONFIG.read_text())
    raw["predictor"].update(training_mode="lora_scalar_head", architecture="residual_frozen_before", pilot_decision="test",
                            lora={"r": 4, "alpha": 8, "dropout": 0.0, "target_modules": ["q"]})
    raw["training"].update(epochs=2, lora_learning_rate=0.01, micro_batch_size=3, init_head_from=str(tmp_path / "frozen_best"),
                           ranking={"lambda": 0.2, "lambda_candidates": [0.2, 1.0], "min_after_gap": 0.5, "gap_weighted": True},
                           residual={"regularization": 0.01}, selection={"folds": 2, "seed": 0, "nll_tolerance": 0.05})
    (tmp_path / "config.json").write_text(json.dumps(raw))
    opened = []
    real = train_residual_cv.load_pairs
    monkeypatch.setattr(train_residual_cv, "load_pairs", lambda path, **kw: opened.append(str(path)) or real(path, **kw))
    out = tmp_path / "v22"
    for command in ("cv", "final"):
        monkeypatch.setattr("sys.argv", ["x", command, "--config", str(tmp_path / "config.json"),
                                         "--train-data", str(paths["train"]), "--output-dir", str(out)])
        train_residual_cv.main()
    summary = json.loads((out / "cv_summary.json").read_text())
    assert len(summary["candidates"]) == 4 and summary["dev_used"] is False and summary["chosen"] in summary["candidates"]
    assert not any((out / "cv").rglob("*.pt"))  # cross-validation weights are not kept
    final = json.loads((out / "final_summary.json").read_text())
    assert final["lambda_rank"] == summary["chosen"]["lambda_rank"] and final["epochs"] == summary["chosen"]["epoch"]
    assert (Path(final["checkpoint"]) / "trainable_state.pt").is_file()
    assert opened == [str(paths["train"])] * 2
