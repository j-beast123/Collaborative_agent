"""Qwen-backed scalar value model for collaboration-state success probability."""

from __future__ import annotations

from typing import Any
import json
from pathlib import Path

import torch


class CollaborationValueModel(torch.nn.Module):
    def __init__(self, model_name: str, *, max_length: int = 2048, use_lora: bool = True, freeze_backbone: bool = False) -> None:
        from transformers import AutoModel, AutoTokenizer

        super().__init__()
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.backbone = AutoModel.from_pretrained(
            model_name, trust_remote_code=True, torch_dtype=torch.bfloat16
        )
        hidden_size = int(self.backbone.config.hidden_size)
        self.value_head = torch.nn.Linear(hidden_size, 1, dtype=torch.float32)
        self.max_length = max_length
        self.tokenizer.padding_side = "right"
        self.backbone.config.pad_token_id = self.tokenizer.pad_token_id
        self.base_model_name = model_name
        self.use_lora = use_lora
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.backbone.to(self.device)
        self.value_head.to(self.device)
        if use_lora:
            from peft import LoraConfig, get_peft_model

            config = LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                use_rslora=True,
                bias="none",
                task_type="FEATURE_EXTRACTION",
            )
            self.backbone = get_peft_model(self.backbone, config)
        if not use_lora or freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

    def trainable_parameters(self):
        return [parameter for parameter in self.parameters() if parameter.requires_grad]

    def forward(self, texts: list[str]):
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        output = self.backbone(**encoded)
        mask = encoded["attention_mask"].bool()
        last_indices = mask.long().sum(dim=1).clamp_min(1) - 1
        rows = self.torch.arange(output.last_hidden_state.size(0), device=self.device)
        hidden = output.last_hidden_state[rows, last_indices]
        return self.value_head(hidden.float()).squeeze(-1)

    def binomial_nll(self, logits, successes, rollout_counts):
        successes = successes.to(logits.device).float()
        rollout_counts = rollout_counts.to(logits.device).float()
        targets = (successes / rollout_counts.clamp_min(1)).to(logits.dtype)
        return self.torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)

    def probabilities(self, texts: list[str]):
        return self.torch.sigmoid(self.forward(texts))

    def save_pretrained(self, output_dir: str) -> None:
        import torch
        from pathlib import Path

        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save_pretrained(path)
        self.backbone.save_pretrained(path / "backbone")
        torch.save(self.value_head.state_dict(), path / "value_head.pt")
        (path / "value_config.json").write_text(json.dumps({
            "base_model_name": self.base_model_name,
            "max_length": self.max_length,
            "use_lora": self.use_lora,
        }), encoding="utf-8")

    @classmethod
    def from_pretrained(cls, checkpoint_dir: str):
        path = Path(checkpoint_dir)
        config = json.loads((path / "value_config.json").read_text(encoding="utf-8"))
        model = cls(config["base_model_name"], max_length=config["max_length"], use_lora=False, freeze_backbone=True)
        if config["use_lora"]:
            from peft import PeftModel
            model.backbone = PeftModel.from_pretrained(model.backbone, path / "backbone", is_trainable=False)
        model.value_head.load_state_dict(torch.load(path / "value_head.pt", map_location=model.device))
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad = False
        return model
