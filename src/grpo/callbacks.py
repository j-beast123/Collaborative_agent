"""Training log of a GRPO iteration: every Trainer log record (GRPO loss, reward mean/std,
serl/* reward components, kl, entropy) as one JSONL row with its iteration and step."""

from __future__ import annotations

import json
from pathlib import Path

from transformers import TrainerCallback


class JsonlLog(TrainerCallback):
    def __init__(self, path: Path, iteration: int) -> None:
        self.path, self.iteration = Path(path), iteration

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs or not state.is_world_process_zero:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"iteration": self.iteration, "global_step": state.global_step, **logs},
                                    default=str) + "\n")
