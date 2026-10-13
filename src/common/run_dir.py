"""Run directories: every run writes under its own versioned directory and never over another run."""

from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Optional

from src.common.config import IntegrationUnavailable
from src.data.schema import digest

RUN_FILES = ("resolved_config.json", "environment.json", "manifest.json")


def config_hash(config: dict) -> str:
    return digest(json.loads(json.dumps(config, default=str)))[:16]


def git_revision(root: Path) -> dict[str, Any]:
    def git(*args: str) -> Optional[str]:
        try:
            done = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, timeout=20)
        except (OSError, subprocess.SubprocessError):
            return None
        return done.stdout.strip() if done.returncode == 0 else None

    status = git("status", "--porcelain")
    return {"head": git("rev-parse", "HEAD"), "uncommitted_changes": bool(status) if status is not None else None}


def environment_info() -> dict[str, Any]:
    info: dict[str, Any] = {"host": socket.gethostname(), "python": sys.version.split()[0], "platform": platform.platform(),
                            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "libraries": {}}
    for name in ("torch", "transformers", "peft", "accelerate", "trl", "datasets", "grpc"):
        try:
            info["libraries"][name] = __import__(name).__version__
        except Exception:  # noqa: BLE001 - a library that is not installed is simply not listed
            info["libraries"][name] = None
    try:
        import torch

        info["cuda_available"] = torch.cuda.is_available()
        info["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if info["cuda_available"] else []
    except Exception:  # noqa: BLE001
        info["cuda_available"] = None
    return info


def create_run(output_root: Path, label: str, config: dict, *, root: Path, seed: Optional[int] = None,
               purpose: str = "", run_id: Optional[str] = None, resume: bool = False) -> Path:
    """The run directory, created with its three run files. See the module docstring for the rules."""
    output_root = Path(output_root)
    if run_id is None:
        run_id = f"{label}_{time.strftime('%Y%m%d-%H%M%S')}_{uuid.uuid4().hex[:8]}"
        if (output_root / run_id).exists():
            raise IntegrationUnavailable(f"run directory {output_root / run_id} already exists; start again in a second")
    if not run_id or Path(run_id).name != run_id or run_id in (".", ".."):
        raise IntegrationUnavailable("run_id must be a single directory name")
    run = output_root / run_id
    if resume and not run.is_dir():
        raise IntegrationUnavailable(f"cannot resume missing run {run}")
    if run.exists():
        if not resume:
            raise IntegrationUnavailable(f"{run} exists: runs are never overwritten. Use a new run id, or resume it explicitly")
        manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
        if manifest["config_hash"] != config_hash(config):
            raise IntegrationUnavailable(f"{run} was created with another configuration ({manifest['config_hash']}); "
                                         "it cannot be continued with this one")
        return run
    run.mkdir(parents=True)
    write_new(run / "resolved_config.json", json.dumps(config, indent=2, ensure_ascii=False, default=str) + "\n")
    write_new(run / "environment.json", json.dumps(environment_info(), indent=2) + "\n")
    write_new(run / "manifest.json", json.dumps({
        "run_id": run_id, "label": label, "purpose": purpose, "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "config_hash": config_hash(config), "seed": seed, "git": git_revision(root)}, indent=2) + "\n")
    return run


def write_new(path: Path, text: str) -> None:
    """Write a file that must not exist yet (results are never overwritten)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(text)
