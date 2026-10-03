"""Effectiveness predictor: P(final success | history, candidate response)."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any, Callable, Optional

import torch

from src.common.config import IntegrationUnavailable

TRAINING_MODES = ("frozen_backbone_scalar_head", "lora_scalar_head")
ARCHITECTURES = ("single", "residual", "residual_frozen_before")
# single                  one head on the example's own input (BEFORE or AFTER) residual                B(h) and
# G(h, r) on one trainable backbone: P(after) = sigmoid(B + G) residual_frozen_before  the same sum, but B(h) is the ...
RESIDUAL = ("residual", "residual_frozen_before")
CHECKPOINT_FORMAT = "effectiveness-predictor-v1"
POOLING = "last_non_padding_token"
LORA_FIELDS = ("r", "alpha", "dropout", "target_modules")


def binomial_nll(logits, successes, rollout_counts):
    """Mean per-rollout NLL of m successes out of K; K may differ per example."""
    successes = successes.to(logits.device, torch.float32)
    rollout_counts = rollout_counts.to(logits.device, torch.float32)
    logits = logits.float()
    log_p = torch.nn.functional.logsigmoid(logits)
    log_not_p = torch.nn.functional.logsigmoid(-logits)
    nll = -(successes * log_p + (rollout_counts - successes) * log_not_p)
    return nll.sum() / rollout_counts.sum().clamp_min(1)


def safe_torch_load(path, map_location=None):
    """torch.load restricted to tensors/containers where torch supports it (>=1.13)."""
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def state_checksum(state: dict[str, Any]) -> str:
    sha = hashlib.sha256()
    for name in sorted(state):
        sha.update(name.encode("utf-8"))
        sha.update(state[name].detach().float().cpu().numpy().tobytes())
    return sha.hexdigest()


def load_hf_backbone(
    model_id: Optional[str],
    revision: Optional[str],
    *,
    torch_dtype: str = "bfloat16",
    trust_remote_code: bool = False,
    device: Optional[str] = None,
):
    """Load a pinned Hugging Face backbone and tokenizer. Never guesses an ID."""
    if not model_id or not revision:
        raise IntegrationUnavailable("Predictor backbone needs an explicit model_id and revision")
    from transformers import AutoModel, AutoTokenizer

    from src.common.config import local_model_path

    model_id = local_model_path(model_id)  # the same directory on another machine (MODEL_PATH_MAP)
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, revision=revision, trust_remote_code=trust_remote_code
    )
    backbone = AutoModel.from_pretrained(
        model_id,
        revision=revision,
        torch_dtype=getattr(torch, torch_dtype),
        trust_remote_code=trust_remote_code,
    )
    backbone.to(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    return backbone, tokenizer


def _apply_lora(backbone, lora: dict[str, Any]):
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as exc:
        raise IntegrationUnavailable("lora_scalar_head requires the peft package") from exc
    config = LoraConfig(
        r=lora["r"],
        lora_alpha=lora["alpha"],
        lora_dropout=lora["dropout"],
        target_modules=list(lora["target_modules"]),
        use_rslora=bool(lora.get("use_rslora", False)),
        bias="none",
        task_type="FEATURE_EXTRACTION",
    )
    return get_peft_model(backbone, config)


class CollaborationValueModel(torch.nn.Module):
    def __init__(
        self,
        backbone,
        tokenizer,
        *,
        base_model_id: str,
        base_model_revision: str,
        max_length: int,
        training_mode: str = "frozen_backbone_scalar_head",
        lora: Optional[dict[str, Any]] = None,
        backbone_options: Optional[dict[str, Any]] = None,
        architecture: str = "single",
    ) -> None:
        super().__init__()
        if training_mode not in TRAINING_MODES:
            raise ValueError(f"training_mode must be one of {TRAINING_MODES}")
        if architecture not in ARCHITECTURES:
            raise ValueError(f"architecture must be one of {ARCHITECTURES}")
        self.architecture = architecture
        self.input_format = "v1"  # how the input text is built (history.format_input); set by the trainer, kept in the checkpoint
        self.supported_turns = ["A1"]
        if training_mode == "lora_scalar_head":
            if not lora or any(lora.get(key) is None for key in LORA_FIELDS):
                raise ValueError(f"lora_scalar_head requires lora settings {LORA_FIELDS}")
        elif lora is not None:
            raise ValueError("lora settings are only valid with training_mode=lora_scalar_head")
        self.torch = torch
        self.base_model_id = base_model_id
        self.base_model_revision = base_model_revision
        self.max_length = max_length
        self.training_mode = training_mode
        self.lora = lora
        self.backbone_options = backbone_options or {}

        self.tokenizer = tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Pooling reads the last non-padding position, which requires right padding.
        self.tokenizer.padding_side = "right"

        for parameter in backbone.parameters():
            parameter.requires_grad = False
        self.backbone = _apply_lora(backbone, lora) if lora else backbone
        self.hidden_size = int(backbone.config.hidden_size)
        device = next(backbone.parameters()).device
        self.value_head = torch.nn.Linear(self.hidden_size, 1).float().to(device)
        if architecture == "residual_frozen_before":
            for parameter in self.value_head.parameters():
                parameter.requires_grad = False  # B's head is the frozen baseline's head; it is loaded, never trained
        if architecture in RESIDUAL:
            # G(h, r): starts at exactly 0, so P(success | h, r) starts at P(success | h)
            self.effect_head = torch.nn.Linear(self.hidden_size, 1).float().to(device)
            torch.nn.init.zeros_(self.effect_head.weight)
            torch.nn.init.zeros_(self.effect_head.bias)

    @property
    def device(self):
        return self.value_head.weight.device

    @property
    def backbone_frozen(self) -> bool:
        return self.training_mode == "frozen_backbone_scalar_head"

    def train(self, mode: bool = True):
        super().train(mode)
        if self.backbone_frozen:
            # A frozen backbone always runs deterministically (no dropout).
            self.backbone.eval()
        return self

    def trainable_parameters(self):
        return [parameter for parameter in self.parameters() if parameter.requires_grad]

    def encode(self, texts: list[str]):
        """Final non-padding token hidden state for each text, as float32."""
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        inputs = {key: encoded[key].to(self.device) for key in ("input_ids", "attention_mask")}
        if self.backbone_frozen:
            with torch.no_grad():
                output = self.backbone(**inputs)
        else:
            output = self.backbone(**inputs)
        last_indices = inputs["attention_mask"].long().sum(dim=1).clamp_min(1) - 1
        rows = torch.arange(last_indices.size(0), device=self.device)
        return output.last_hidden_state[rows, last_indices].float()

    def base_encode(self, texts: list[str]):
        """encode() by the base backbone alone: adapters switched off, no gradients. The result does not
        depend on the adapter weights, so B(h) of residual_frozen_before stays what the frozen baseline gives."""
        with torch.no_grad():
            if hasattr(self.backbone, "disable_adapter"):
                with self.backbone.disable_adapter():
                    return self.encode(texts)
            return self.encode(texts)

    def head_logits(self, features):
        return self.value_head(features).squeeze(-1)

    def effect_logits(self, features):
        return self.effect_head(features).squeeze(-1)

    def forward(self, texts: list[str]):
        return self.head_logits(self.encode(texts))

    def example_logits(self, examples, batch_size: int) -> list[float]:
        """Success logit of every example (kind, history_id, text), without gradients."""
        was_training = self.training
        self.eval()
        own, effect = [], []
        with torch.no_grad():
            for start in range(0, len(examples), batch_size):
                texts = [e.text for e in examples[start : start + batch_size]]
                if self.architecture == "residual_frozen_before":
                    # the frozen baseline's own computation: same weights, same batches, adapters off
                    own.extend(float(v) for v in self.head_logits(self.base_encode(texts)).cpu())
                    effect.extend(float(v) for v in self.effect_logits(self.encode(texts)).cpu())
                    continue
                features = self.encode(texts)
                own.extend(float(v) for v in self.head_logits(features).cpu())
                if self.architecture == "residual":
                    effect.extend(float(v) for v in self.effect_logits(features).cpu())
        self.train(was_training)
        if self.architecture not in RESIDUAL:
            return own
        before = {e.history_id: own[i] for i, e in enumerate(examples) if e.kind == "before"}
        missing = sorted({e.history_id for e in examples if e.kind == "after"} - set(before))
        if missing:
            raise ValueError(f"residual prediction needs the BEFORE example of each history; missing for {missing[:3]}")
        return [own[i] if e.kind == "before" else before[e.history_id] + effect[i] for i, e in enumerate(examples)]

    def predict_examples(self, examples, batch_size: int) -> list[float]:
        return [float(torch.sigmoid(torch.tensor(z, dtype=torch.float64))) for z in self.example_logits(examples, batch_size)]

    def residual_after_probability(self, before_text: str, after_text: str) -> float:
        """sigmoid(B(h) + G(h, r)) for one history input and its history + response input."""
        was_training = self.training
        self.eval()
        with torch.no_grad():
            if self.architecture == "residual_frozen_before":
                b = self.head_logits(self.base_encode([before_text]))
                value = torch.sigmoid(b + self.effect_logits(self.encode([after_text])))
            else:
                features = self.encode([before_text, after_text])
                value = torch.sigmoid(self.head_logits(features[:1]) + self.effect_logits(features[1:]))
        self.train(was_training)
        return float(value.cpu()[0])

    def probabilities(self, texts: list[str]):
        return torch.sigmoid(self.forward(texts))

    def binomial_nll(self, logits, successes, rollout_counts):
        return binomial_nll(logits, successes, rollout_counts)

    def predict(self, texts: list[str], batch_size: int) -> list[float]:
        was_training = self.training
        self.eval()
        values = []
        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                values.extend(float(v) for v in self.probabilities(texts[start : start + batch_size]).cpu())
        self.train(was_training)
        return values

    def _saved(self, name: str, parameter) -> bool:
        # the frozen B head is part of a residual_frozen_before checkpoint although it is not trained
        return parameter.requires_grad or (self.architecture == "residual_frozen_before" and name.startswith("value_head."))

    def trainable_state(self) -> dict[str, Any]:
        """The weights a checkpoint stores: everything trained, plus the frozen B head where there is one."""
        return {name: p.detach().cpu().clone() for name, p in self.named_parameters() if self._saved(name, p)}

    def load_trainable_state(self, state: dict[str, Any]) -> None:
        params = dict(self.named_parameters())
        expected = {name for name, p in params.items() if self._saved(name, p)}
        if set(state) != expected:
            raise ValueError("Trainable parameter names do not match the checkpoint")
        with torch.no_grad():
            for name, value in state.items():
                params[name].copy_(value.to(params[name].device))

    def config_dict(self, metadata: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        return {
            "format": CHECKPOINT_FORMAT,
            "base_model_id": self.base_model_id,
            "base_model_revision": self.base_model_revision,
            "backbone_options": self.backbone_options,
            "max_length": self.max_length,
            "training_mode": self.training_mode,
            "backbone_frozen": self.backbone_frozen,
            "lora": self.lora,
            "architecture": self.architecture,
            "pooling": POOLING,
            "hidden_size": self.hidden_size,
            "head": "Linear(hidden_size, 1) -> sigmoid",
            "metadata": metadata or {},
            # written only for a turn-conditioned predictor, so an A1-only checkpoint keeps its recorded form
            **({} if self.input_format == "v1" else {"input_format": self.input_format,
                                                     "supported_turns": list(self.supported_turns)}),
        }

    def save_pretrained(self, output_dir: str | Path, metadata: Optional[dict[str, Any]] = None) -> None:
        """Save only trainable weights; the frozen backbone is re-loaded by ID/revision."""
        path = Path(output_dir)
        staging = path.with_name(path.name + ".tmp")
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True)
        state = self.trainable_state()
        torch.save(state, staging / "trainable_state.pt")
        config = self.config_dict(metadata)
        config["trainable_state_sha256"] = state_checksum(state)
        (staging / "predictor_config.json").write_text(
            json.dumps(config, indent=2, sort_keys=True), encoding="utf-8"
        )
        if path.exists():
            shutil.rmtree(path)
        os.replace(staging, path)

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_dir: str | Path,
        *,
        backbone_loader: Callable = load_hf_backbone,
    ) -> "CollaborationValueModel":
        path = Path(checkpoint_dir)
        config = json.loads((path / "predictor_config.json").read_text(encoding="utf-8"))
        if config.get("format") != CHECKPOINT_FORMAT:
            raise ValueError("Unsupported effectiveness predictor checkpoint format")
        backbone, tokenizer = backbone_loader(
            config["base_model_id"], config["base_model_revision"], **config["backbone_options"]
        )
        model = cls(
            backbone,
            tokenizer,
            base_model_id=config["base_model_id"],
            base_model_revision=config["base_model_revision"],
            max_length=config["max_length"],
            training_mode=config["training_mode"],
            lora=config["lora"],
            backbone_options=config["backbone_options"],
            architecture=config.get("architecture", "single"),
        )
        if model.hidden_size != config["hidden_size"]:
            raise ValueError("Backbone hidden size differs from the checkpoint")
        state = safe_torch_load(path / "trainable_state.pt", map_location=model.device)
        if state_checksum(state) != config["trainable_state_sha256"]:
            raise ValueError("Checkpoint weights do not match their recorded checksum")
        model.load_trainable_state(state)
        model.input_format = config.get("input_format", "v1")
        model.supported_turns = config.get("supported_turns", ["A1"])
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad = False
        return model
