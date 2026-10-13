"""Fit the effectiveness model on rollout-derived TRAIN pair labels."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path
from typing import Any, Optional

from src.common.config import IntegrationUnavailable, read_config
from src.data.schema import digest

from .continuation import ContinuationSpec
from .evaluate import evaluate_predictions
from .report import dev_report, is_natural, same_answer_behavior_ranking, subset_report
from .pair_data import assert_case_disjoint, assert_manifest_splits, load_pairs, to_examples

BACKBONE_SOURCES = ("huggingface", "synthetic_tiny")
LOWER_IS_BETTER = ("binomial_nll", "brier", "ece", "mae", "rmse")
TRAINING_FIELDS = (
    "seed", "epochs", "learning_rate", "weight_decay", "batch_size",
    "grad_clip", "selection_metric", "cache_features",
)


def load_predictor_config(path: str | Path) -> dict[str, Any]:
    config = read_config(path)
    mode = config.get("mode")
    if mode not in ("smoke", "production"):
        raise ValueError("mode must be smoke or production")
    predictor = config.get("predictor", {})
    if predictor.get("role") != "effectiveness_predictor":
        raise ValueError("predictor.role must be effectiveness_predictor")
    if predictor.get("backbone_source") not in BACKBONE_SOURCES:
        raise ValueError(f"predictor.backbone_source must be one of {BACKBONE_SOURCES}")
    if (predictor["backbone_source"] == "synthetic_tiny") != (mode == "smoke"):
        raise ValueError("The synthetic backbone is required in smoke mode and forbidden in production")
    if predictor.get("training_mode") not in ("frozen_backbone_scalar_head", "lora_scalar_head"):
        raise ValueError("Invalid predictor.training_mode")
    if predictor["training_mode"] == "lora_scalar_head" and not predictor.get("pilot_decision"):
        raise ValueError("lora_scalar_head is a fallback: record pilot_decision (why the frozen head was insufficient)")
    training = config.get("training", {})
    missing = [name for name in TRAINING_FIELDS if training.get(name) is None]
    if missing:
        raise ValueError(f"Missing training settings: {missing}")
    if training["selection_metric"] not in LOWER_IS_BETTER:
        raise ValueError(f"selection_metric must be one of {LOWER_IS_BETTER}")
    spec = ContinuationSpec.from_dict(config.get("continuation", {}))
    if spec.k is None:
        raise ValueError("continuation.k is required to validate rollout-derived targets")
    if mode == "production":
        spec.require_resolved()
    config["continuation_spec"] = spec
    return config


def _atomic_torch_save(obj, path: Path) -> None:
    import torch

    temporary = path.with_name(path.name + ".tmp")
    torch.save(obj, temporary)
    os.replace(temporary, path)


def _encode_all(model, examples, batch_size):
    import torch

    features = []
    for start in range(0, len(examples), batch_size):
        features.append(model.encode([e.text for e in examples[start : start + batch_size]]))
    return torch.cat(features)


def history_batches(examples, order: list[int], batch_size: int) -> list[list[int]]:
    """Batches that keep the examples of one history together (in the shuffled order of first appearance), so that
    same-history ranking pairs fall inside a batch.
    """
    groups: dict[str, list[int]] = {}
    for index in order:
        groups.setdefault(examples[index].history_id, []).append(index)
    batches, current = [], []
    for group in groups.values():
        if current and len(current) + len(group) > batch_size:
            batches.append(current)
            current = []
        current = current + group
    return batches + ([current] if current else [])


def turn_balanced_batches(examples, order: list[int], batch_size: int, rng: random.Random) -> list[list[int]]:
    """Batches that hold the same number of examples of every intervention turn (A1, A2), so that neither turn's labels
    dominate an update.
    """
    by_turn: dict[str, list[int]] = {}
    for index in order:
        by_turn.setdefault(examples[index].intervention_turn, []).append(index)
    if len(by_turn) < 2:
        return [order[start: start + batch_size] for start in range(0, len(order), batch_size)]
    per_turn, longest = batch_size // len(by_turn), max(len(v) for v in by_turn.values())
    if per_turn < 1:
        raise ValueError("batch_size is smaller than the number of intervention turns")
    streams = {turn: members + [members[rng.randrange(len(members))] for _ in range(longest - len(members))]
               for turn, members in sorted(by_turn.items())}
    return [[i for stream in streams.values() for i in stream[start: start + per_turn]]
            for start in range(0, longest, per_turn)]


def ranking_pairs(examples, index: list[int], min_gap: float, gap_weighted: bool = False) -> list[tuple]:
    """(position of the better response, position of the worse one) within a batch: AFTER examples of the same history
    whose rollout success rates differ by at least min_gap.
    """
    after = [(position, examples[i]) for position, i in enumerate(index) if examples[i].kind == "after"]
    pairs = [(a, b, first.empirical_value - second.empirical_value) for a, first in after for b, second in after
             if first.history_id == second.history_id and first.empirical_value - second.empirical_value >= min_gap - 1e-9]
    return pairs if gap_weighted else [(a, b) for a, b, _ in pairs]


def pair_weight_total(pairs: list[tuple]) -> float:
    return sum(p[2] if len(p) > 2 else 1.0 for p in pairs)


def ranking_loss(logits, reference, pairs: list[tuple], members: list[int]):
    """Pairwise logistic loss softplus(-(z_better - z_worse)), times the pair's weight when it has one, summed over the
    pairs that have a member in `members` (batch positions whose logits carry gradients; logits[k] belongs to ...
    """
    import torch

    at = {position: k for k, position in enumerate(members)}
    terms = [(weight[0] if weight else 1.0) * torch.nn.functional.softplus(
                 -((logits[at[a]] if a in at else reference[a]) - (logits[at[b]] if b in at else reference[b])))
             for a, b, *weight in pairs if a in at or b in at]
    return torch.stack(terms).sum() if terms else None


def residual_step(model, train, index: list[int], successes, counts, micro: int, pairs, scale: float,
                  regularization: float) -> float:
    """Gradients of one batch of whole histories for the residual model; returns the loss value."""
    import torch

    texts = [train[i].text for i in index]
    after = torch.tensor([train[i].kind == "after" for i in index])
    with torch.no_grad():
        features = torch.cat([model.encode(texts[s : s + micro]) for s in range(0, len(index), micro)])
        own, effect = model.head_logits(features), model.effect_logits(features)
    b_of = {train[i].history_id: own[position] for position, i in enumerate(index) if train[i].kind == "before"}
    base = torch.stack([b_of[train[i].history_id] for i in index])  # every history's BEFORE example is in its batch
    reference = torch.where(after.to(own.device), base + effect, own)
    totals = {True: counts[index][after].sum().clamp_min(1), False: counts[index][~after].sum().clamp_min(1)}
    n_after = int(after.sum())
    total = 0.0
    for start in range(0, len(index), micro):
        members = list(range(start, min(start + micro, len(index))))
        chunk = [index[m] for m in members]
        features = model.encode([train[i].text for i in chunk])
        loss = features.new_zeros(())
        for kind in (False, True):
            rows = [k for k, m in enumerate(members) if bool(after[m]) == kind]
            if not rows:
                continue
            ids = [chunk[k] for k in rows]
            if kind:
                e = model.effect_logits(features[rows])
                logits = base[[members[k] for k in rows]] + e
                loss = loss + regularization * (e ** 2).sum() / n_after
            else:
                logits = model.head_logits(features[rows])
            weight = float(counts[ids].sum() / totals[kind])
            loss = loss + weight * model.binomial_nll(logits, successes[ids], counts[ids])
            if kind and pairs:
                rank = ranking_loss(logits, reference, pairs, [members[k] for k in rows])
                if rank is not None:
                    loss = loss + scale * rank
        loss.backward()
        total += float(loss)
    return total


def frozen_before_step(model, train, index: list[int], base, successes, counts, micro: int, pairs, scale: float,
                       regularization: float) -> float:
    """Gradients of one batch of AFTER examples for residual_frozen_before; returns the loss value."""
    import torch

    reference = None
    if pairs:  # partner logits of ranking pairs that span chunks
        with torch.no_grad():
            reference = base + torch.cat([model.effect_logits(model.encode([train[i].text for i in index[s : s + micro]]))
                                          for s in range(0, len(index), micro)])
    total_rollouts = counts[index].sum().clamp_min(1)
    total = 0.0
    for start in range(0, len(index), micro):
        members = list(range(start, min(start + micro, len(index))))
        chunk = [index[m] for m in members]
        e = model.effect_logits(model.encode([train[i].text for i in chunk]))
        logits = base[members] + e
        loss = float(counts[chunk].sum() / total_rollouts) * model.binomial_nll(logits, successes[chunk], counts[chunk]) \
            + regularization * (e ** 2).sum() / len(index)
        if pairs:
            rank = ranking_loss(logits, reference, pairs, members)
            if rank is not None:
                loss = loss + scale * rank
        loss.backward()
        total += float(loss)
    return total


def frozen_before_sanity(model, dev, optimizer, batch_size: int, baseline_logits: Optional[list[float]],
                         tolerance: float = 1e-6) -> dict[str, Any]:
    """Checks before a residual_frozen_before run; raises when one fails."""
    import torch

    before = [i for i, e in enumerate(dev) if e.kind == "before"]
    after = [i for i, e in enumerate(dev) if e.kind == "after"]
    logits = model.example_logits(dev, batch_size)
    b_of = {dev[i].history_id: logits[i] for i in before}
    report: dict[str, Any] = {"dev_before_examples": len(before)}
    if baseline_logits is not None:
        report["max_abs_diff_to_frozen_baseline_before_logits"] = max(abs(logits[i] - baseline_logits[i]) for i in before)
    report["max_abs_diff_after_minus_before_at_initialisation"] = max(abs(logits[i] - b_of[dev[i].history_id]) for i in after)

    names = {id(p): n for n, p in model.named_parameters()}
    optimized = sorted(names[id(p)] for group in optimizer.param_groups for p in group["params"])
    report["optimizer_parameter_count"] = len(optimized)
    report["optimizer_parameter_kinds"] = sorted({"effect_head" if n.startswith("effect_head") else "lora" if "lora_" in n
                                                  else n for n in optimized})
    report["optimizer_holds_value_head_or_base_weights"] = any(
        not (n.startswith("effect_head") or "lora_" in n) for n in optimized)

    adapters = [p for n, p in model.named_parameters() if "lora_" in n]
    probe = [dev[i] for i in before[:batch_size]]
    if adapters:
        saved = [p.detach().clone() for p in adapters]
        reference_b = model.example_logits(probe, batch_size)
        was_training = model.training
        model.eval()
        with torch.no_grad():
            enabled = model.encode([e.text for e in probe])
            for p in adapters:
                p.copy_(torch.randn_like(p) * 0.05)
            report["adapter_path_changes_with_random_adapters"] = bool(
                (model.encode([e.text for e in probe]) - enabled).abs().max() > 1e-4)
            perturbed_b = model.example_logits(probe, batch_size)
            for p, value in zip(adapters, saved):
                p.copy_(value)
        model.train(was_training)
        report["max_abs_diff_before_logits_with_random_adapters"] = max(abs(x - y) for x, y in zip(reference_b, perturbed_b))
        report["max_abs_diff_before_logits_after_restore"] = max(
            abs(x - y) for x, y in zip(reference_b, model.example_logits(probe, batch_size)))
    failures = []
    if report.get("max_abs_diff_to_frozen_baseline_before_logits", 0.0) > tolerance:
        failures.append("p_before differs from the frozen baseline")
    if report["max_abs_diff_after_minus_before_at_initialisation"] > 0:
        failures.append("p_after differs from p_before although G = 0")
    if report.get("max_abs_diff_before_logits_with_random_adapters", 0.0) > 0:
        failures.append("B changes when the adapter weights change")
    if adapters and not report["adapter_path_changes_with_random_adapters"]:
        failures.append("the adapter perturbation had no visible effect, so check 3 proves nothing")
    if report["optimizer_holds_value_head_or_base_weights"]:
        failures.append("the optimizer holds the B head or base weights")
    report["passed"], report["failures"] = not failures, failures
    return report


def fit(model, train, dev, training: dict[str, Any], output_dir: str | Path, *, fingerprint: str, resume: bool,
        checkpoint_metadata: dict[str, Any] | None = None, on_epoch=None) -> dict[str, Any]:
    """Train trainable parameters; keep every epoch's checkpoint and a resumable state."""
    import torch

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    state_path = output / "trainer_state.pt"
    batch_size = training["batch_size"]
    residual = model.architecture == "residual"
    frozen_before = model.architecture == "residual_frozen_before"
    # The heads use learning_rate; LoRA adapters use lora_learning_rate when it is set.
    head = list(model.effect_head.parameters()) if frozen_before else \
        list(model.value_head.parameters()) + (list(model.effect_head.parameters()) if residual else [])
    head_ids = {id(p) for p in head}
    adapters = [p for p in model.trainable_parameters() if id(p) not in head_ids]
    groups = [{"params": head, "lr": training["learning_rate"]}]
    if adapters:
        groups.append({"params": adapters, "lr": training.get("lora_learning_rate") or training["learning_rate"]})
    optimizer = torch.optim.AdamW(groups, lr=training["learning_rate"], weight_decay=training["weight_decay"])
    # micro_batch_size < batch_size: one optimizer step per batch, gradients accumulated over
    # chunks (needed when the backbone is trained and a full batch does not fit in GPU memory).
    micro = int(training.get("micro_batch_size") or batch_size)
    start_epoch, best, history, best_ordering = 0, math.inf, [], (-math.inf, math.inf)
    ranking = training.get("ranking") or {}
    if state_path.exists():
        if not resume:
            raise ValueError(f"{state_path} exists; pass --resume or choose a new output directory")
        from .model import safe_torch_load

        state = safe_torch_load(state_path, map_location=model.device)
        if state["fingerprint"] != fingerprint:
            raise ValueError("Refusing to resume: configuration or data changed since the saved state")
        model.load_trainable_state(state["trainable"])
        optimizer.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["torch_rng"].cpu())
        start_epoch, best, history = state["epoch"], state["best"], state["history"]
        best_ordering = tuple(state.get("best_ordering", best_ordering))

    cached = training["cache_features"] and model.backbone_frozen and not residual and not frozen_before
    if frozen_before:
        baseline = training.get("frozen_before_baseline_logits")
        sanity = frozen_before_sanity(model, dev, optimizer, batch_size,
                                      json.loads(Path(baseline).read_text(encoding="utf-8")) if baseline else None)
        (output / "sanity.json").write_text(json.dumps(sanity, indent=2), encoding="utf-8")
        print("sanity", json.dumps(sanity), flush=True)
        if not sanity["passed"]:
            raise RuntimeError(f"residual_frozen_before sanity checks failed: {sanity['failures']}")
        # B(h) of every TRAIN history and the DEV BEFORE logits, fixed for the whole run
        train_before = [e for e in train if e.kind == "before"]
        b_train = dict(zip([e.history_id for e in train_before], model.example_logits(train_before, batch_size)))
        dev_before_reference = [z for z, e in zip(model.example_logits(dev, batch_size), dev) if e.kind == "before"]
    if cached:
        # The frozen backbone is deterministic, so encode each example once.
        train_features = _encode_all(model, train, batch_size)
    successes = torch.tensor([e.num_success for e in train], dtype=torch.float32)
    counts = torch.tensor([e.num_rollouts for e in train], dtype=torch.float32)
    metric = training["selection_metric"]

    for epoch in range(start_epoch, training["epochs"]):
        order = [i for i in range(len(train)) if not frozen_before or train[i].kind == "after"]
        random.Random(training["seed"] * 1_000_003 + epoch).shuffle(order)
        model.train()
        if training.get("turn_balanced"):
            batches = turn_balanced_batches(train, order, batch_size, random.Random(training["seed"] * 7_919 + epoch))
        else:
            batches = history_batches(train, order, batch_size) if ranking or residual or frozen_before else \
                [order[start : start + batch_size] for start in range(0, len(order), batch_size)]
        for index in batches:
            optimizer.zero_grad()
            pairs = ranking_pairs(train, index, ranking["min_after_gap"], ranking.get("gap_weighted", False)) if ranking else []
            # lambda times the (weighted) mean over the batch's pairs
            scale = ranking["lambda"] / pair_weight_total(pairs) if pairs else 0.0
            if frozen_before:
                base = torch.tensor([b_train[train[i].history_id] for i in index], dtype=torch.float32, device=model.device)
                loss = frozen_before_step(model, train, index, base, successes, counts, micro, pairs, scale,
                                          training["residual"]["regularization"])
            elif residual:
                loss = residual_step(model, train, index, successes, counts, micro, pairs, scale,
                                     training["residual"]["regularization"])
            elif cached:
                logits = model.head_logits(train_features[index])
                loss = model.binomial_nll(logits, successes[index], counts[index])
                if pairs:
                    loss = loss + scale * ranking_loss(logits, logits.detach(), pairs, list(range(len(index))))
                loss.backward()
            else:
                reference = None
                if pairs:  # the partner logits of pairs that span chunks
                    with torch.no_grad():
                        reference = torch.cat([model.forward([train[i].text for i in index[s : s + micro]])
                                               for s in range(0, len(index), micro)])
                loss = 0.0  # equals the full-batch binomial NLL: each chunk is weighted by its rollouts
                for chunk_start in range(0, len(index), micro):
                    chunk = index[chunk_start : chunk_start + micro]
                    weight = float(counts[chunk].sum() / counts[index].sum().clamp_min(1))
                    logits = model.forward([train[i].text for i in chunk])
                    chunk_loss = weight * model.binomial_nll(logits, successes[chunk], counts[chunk])
                    if pairs:
                        members = list(range(chunk_start, chunk_start + len(chunk)))
                        rank = ranking_loss(logits, reference, pairs, members)
                        if rank is not None:
                            # a pair inside one chunk is complete here; a pair across chunks gets one side's
                            # gradient here and the other side's in the other chunk
                            chunk_loss = chunk_loss + scale * rank
                    chunk_loss.backward()
                    loss += float(chunk_loss)
            torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), training["grad_clip"])
            optimizer.step()
        if on_epoch is not None:
            on_epoch(epoch + 1, model)
        predictions = model.predict_examples(dev, batch_size)
        dev_metrics = evaluate_predictions(predictions, dev)
        if frozen_before:
            # B must be exactly what it was before training; anything else is an architecture fault
            drift = max(abs(a - b) for a, b in zip(
                [z for z, e in zip(model.example_logits(dev, batch_size), dev) if e.kind == "before"], dev_before_reference))
            print(f"epoch {epoch + 1}: DEV BEFORE NLL {dev_metrics['absolute']['before']['binomial_nll']:.6f}, "
                  f"max change of the BEFORE logits since the start {drift:.3g}", flush=True)
            if drift > 0:
                raise RuntimeError(f"B(h) changed during training (max logit change {drift}); stopping")
        natural = [i for i, e in enumerate(dev) if is_natural(e)]
        natural_metrics = evaluate_predictions([predictions[i] for i in natural], [dev[i] for i in natural])
        history.append({"epoch": epoch + 1, "train_loss_last_batch": float(loss), "dev": dev_metrics,
                        "dev_natural": natural_metrics["absolute"]})
        # Checkpoint selection on absolute DEV quality: every candidate, or natural responses only.
        selected = natural_metrics if training.get("selection_subset") == "natural" else dev_metrics
        score = selected["absolute"]["all"][metric]
        metadata = {"epoch": epoch + 1, "selection_metric": metric, "dev": dev_metrics, "fingerprint": fingerprint,
                    **(checkpoint_metadata or {})}
        if training.get("keep_epoch_checkpoints", True):
            model.save_pretrained(output / "epochs" / f"epoch{epoch + 1}", metadata=metadata)
        if score < best:
            best = score
            model.save_pretrained(output / "best", metadata=metadata)
        ordering = dev_metrics["ranking"]["within_history_pairwise_accuracy"]
        if ordering is not None and (ordering, -score) > (best_ordering[0], -best_ordering[1]):
            best_ordering = (ordering, score)
            model.save_pretrained(output / "best_ordering", metadata=metadata)
        rows = [{"epoch": h["epoch"], metric: h["dev"]["absolute"]["all"][metric],
                 "ordering": h["dev"]["ranking"]["within_history_pairwise_accuracy"]} for h in history]
        (output / "checkpoints.json").write_text(json.dumps({
            "epochs": rows,
            "best_selection": min(rows, key=lambda r: r[metric]),
            "best_ordering": max((r for r in rows if r["ordering"] is not None),
                                 key=lambda r: (r["ordering"], -r[metric]), default=None),
            "selection_metric": metric, "selection_subset": training.get("selection_subset") or "all"}, indent=2),
            encoding="utf-8")
        _atomic_torch_save(
            {
                "epoch": epoch + 1,
                "best": best,
                "best_ordering": list(best_ordering),
                "history": history,
                "fingerprint": fingerprint,
                "trainable": model.trainable_state(),
                "optimizer": optimizer.state_dict(),
                "torch_rng": torch.get_rng_state(),
            },
            state_path,
        )
    return {"best": {metric: best}, "history": history}


def build_backbone(predictor: dict[str, Any]):
    if predictor["backbone_source"] == "synthetic_tiny":
        from .synthetic import SYNTHETIC_MODEL_ID, SYNTHETIC_REVISION, load_synthetic_backbone

        return SYNTHETIC_MODEL_ID, SYNTHETIC_REVISION, load_synthetic_backbone, {}
    from .model import load_hf_backbone

    options = {"torch_dtype": predictor["torch_dtype"], "trust_remote_code": predictor["trust_remote_code"]}
    return predictor["model_id"], predictor["revision"], load_hf_backbone, options


def run_training(config_path, train_data, dev_data, output_dir, *, resume=False, split_manifest=None) -> dict[str, Any]:
    import torch

    from .model import CollaborationValueModel, safe_torch_load

    config = load_predictor_config(config_path)
    spec: ContinuationSpec = config["continuation_spec"]
    predictor, training = config["predictor"], config["training"]
    protocol_hash = spec.protocol_hash()
    expected_k = spec.k if training.get("require_k_equals_spec", True) else None
    pairs = {
        name: load_pairs(path, split=name, protocol_hash=protocol_hash, expected_k=expected_k)
        for name, path in (("train", train_data), ("dev", dev_data))
    }
    if not pairs["train"] or not pairs["dev"]:
        raise ValueError("Train and dev effectiveness labels must be non-empty")
    assert_case_disjoint(pairs)  # the same original clinical case never spans TRAIN and DEV
    manifest_path = split_manifest or config.get("data", {}).get("split_manifest")
    if config["mode"] == "production":
        source_dataset = config.get("data", {}).get("source_dataset")
        if not manifest_path or not source_dataset:
            raise IntegrationUnavailable("Production training requires data.split_manifest and data.source_dataset")
        from src.data.split import read_manifest

        assert_manifest_splits(pairs, read_manifest(manifest_path), source_dataset)
    splits = {name: to_examples(rows) for name, rows in pairs.items()}

    torch.manual_seed(training["seed"])
    random.seed(training["seed"])
    model_id, revision, loader, options = build_backbone(predictor)
    backbone, tokenizer = loader(model_id, revision, **options)
    lora = predictor.get("lora") if predictor["training_mode"] == "lora_scalar_head" else None
    model = CollaborationValueModel(
        backbone,
        tokenizer,
        base_model_id=model_id,
        base_model_revision=revision,
        max_length=predictor["max_length"],
        training_mode=predictor["training_mode"],
        lora=lora,
        backbone_options=options,
        architecture=predictor.get("architecture", "single"),
    )
    if model.architecture == "residual_frozen_before" and not training.get("init_head_from"):
        raise ValueError("residual_frozen_before needs training.init_head_from (the frozen baseline whose head is B)")
    if training.get("init_head_from"):
        # Start the scalar head from another checkpoint's head (e.g. the frozen-backbone baseline).
        head = safe_torch_load(Path(training["init_head_from"]) / "trainable_state.pt", map_location="cpu")
        model.value_head.load_state_dict({name.split("value_head.", 1)[1]: value for name, value in head.items()
                                          if name.startswith("value_head.")})
    if lora and training.get("gradient_checkpointing"):
        # Trades compute for memory. The frozen embeddings must emit a tensor that requires
        # grad, otherwise checkpointed blocks see no differentiable input and adapters get no gradient.
        model.backbone.gradient_checkpointing_enable()
        model.backbone.enable_input_require_grads()
    # epochs is excluded so an interrupted run can resume toward the same or a longer schedule.
    fingerprint = digest({
        "predictor": predictor,
        "training": {key: value for key, value in training.items() if key != "epochs"},
        "continuation_protocol_hash": protocol_hash,
        "data": {
            name: [(e.kind, e.history_id, e.response_id, digest(e.text), e.num_success, e.num_rollouts) for e in rows]
            for name, rows in splits.items()
        },
    })
    result = fit(model, splits["train"], splits["dev"], training, output_dir, fingerprint=fingerprint, resume=resume,
                 checkpoint_metadata={"continuation_protocol_hash": protocol_hash, "k": spec.k})

    # Round trip: predictions from the in-memory best weights must match a fresh load of the saved checkpoint
    # (backbone re-loaded by ID/revision).
    best_dir = Path(output_dir) / "best"
    model.load_trainable_state(safe_torch_load(best_dir / "trainable_state.pt", map_location="cpu"))
    dev_predictions = model.predict_examples(splits["dev"], training["batch_size"])
    backbone_frozen = model.backbone_frozen
    del model, backbone, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    reloaded = CollaborationValueModel.from_pretrained(best_dir, backbone_loader=loader)
    if any(abs(a - b) > 1e-5 for a, b in zip(dev_predictions, reloaded.predict_examples(splits["dev"], training["batch_size"]))):
        raise RuntimeError("Checkpoint round-trip predictions differ")
    report = dev_report(dev_predictions, splits["dev"], splits["train"],
                        n_boot=training.get("report_bootstrap", 1000), seed=training["seed"])
    for name, natural in (("natural_candidates", True), ("controlled_candidates", False)):
        subset = subset_report(dev_predictions, splits["dev"], splits["train"], natural,
                               n_boot=training.get("report_bootstrap", 1000), seed=training["seed"])
        if subset:
            report[name] = subset
    report["same_answer_behavior_ranking"] = same_answer_behavior_ranking(dev_predictions, splits["dev"])
    (Path(output_dir) / "dev_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    summary = {
        "mode": config["mode"],
        "synthetic": config["mode"] == "smoke",
        "checkpoint": str(best_dir),
        "training_mode": predictor["training_mode"],
        "backbone_frozen": backbone_frozen,
        "continuation_protocol_hash": protocol_hash,
        "k": spec.k,
        "train_pairs": len(pairs["train"]),
        "dev_pairs": len(pairs["dev"]),
        "train_examples": len(splits["train"]),
        "dev_examples": len(splits["dev"]),
        "best": result["best"],
        "dev_metrics_best_checkpoint": evaluate_predictions(dev_predictions, splits["dev"]),
        "epochs_completed": len(result["history"]),
        "checkpoint_round_trip": "pass",
        "dev_vs_baselines": report["comparison"],
        "label_diagnostics": report["label_diagnostics"],
        "brier_noise_floor": report["brier_noise_floor"],
    }
    (Path(output_dir) / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--dev-data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split-manifest")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    summary = run_training(
        args.config, args.train_data, args.dev_data, args.output_dir,
        resume=args.resume, split_manifest=args.split_manifest,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
