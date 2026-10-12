"""Read-only reproduction evaluator for the packaged reported checkpoint LoRA checkpoint.

This command never constructs an optimizer and never writes into the adapter
directory. Results are resumable in a separate output directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

ARTIFACT = ROOT / "artifacts/reported_checkpoint"
BASE_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
ADAPTER_FILE_SHA256 = "e81c3975dab9f9cc08965563aacd1b1cfce5dae90cdbb22e5a83849e3acae178"
DATA_SHA256 = "e42077d006c4e8227564f99b06050b66e7df9989f9b2b17f5fc5d9dd083feb56"
POLICY_SHA256 = "bbbfc75889508d0438094559c6c3066759c1d4a35548f75f86125c56b68e8299"
DOCTOR_DIGEST = "997a7c2c09753e9011ec18a3fcb498e27c0ba330599dfb26a2e05b58cfbdc1bc"


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def paths(artifact: Path) -> dict[str, Path]:
    return {
        "config": artifact / "config/tree_v21_five_gpu_shared5_random10_promptv7_match8.json",
        "adapter": artifact / "adapter",
        "weights": artifact / "adapter/adapter_model.safetensors",
        "data": artifact / "data/all_en_622.jsonl",
        "receipt": artifact / "provenance/checkpoint_COMPLETE.json",
    }


def prepared_config(artifact: Path, doctor_endpoint: str) -> tuple[dict, dict[str, Path]]:
    p = paths(artifact)
    missing = [str(v) for v in p.values() if not v.exists()]
    if missing:
        raise RuntimeError(f"missing reported checkpoint artifact files: {missing}")
    if file_sha256(p["weights"]) != ADAPTER_FILE_SHA256:
        raise RuntimeError("reported checkpoint adapter file SHA-256 mismatch (Git LFS object missing or corrupted)")
    if file_sha256(p["data"]) != DATA_SHA256:
        raise RuntimeError("reported checkpoint evaluation data SHA-256 mismatch")
    receipt = json.loads(p["receipt"].read_text())
    if receipt.get("policy_sha256") != POLICY_SHA256 or receipt.get("completed_cases") != 30:
        raise RuntimeError("reported checkpoint receipt mismatch")
    cfg = json.loads(p["config"].read_text())
    cfg["policy"]["base_revision"] = BASE_REVISION
    cfg["policy"]["init_adapter"] = str(p["adapter"])
    cfg["rollout"]["doctor_backend"]["endpoint"] = doctor_endpoint
    return cfg, p


def summarize(case_dir: Path, total: int) -> dict:
    rows = [json.loads(p.read_text()) for p in sorted(case_dir.glob("case_*.json"))]
    return {
        "completed_cases": 30,
        "dataset_cases": total,
        "evaluated": len(rows),
        "correct": sum(r["outcome"] == 1.0 for r in rows),
        "unparsed": sum(r["outcome"] is None for r in rows),
        "accuracy": (sum(r["outcome"] == 1.0 for r in rows) / len(rows)) if rows else None,
        "mean_agent_turns": (sum(r["num_discussion_rounds"] for r in rows) / len(rows)) if rows else None,
        "status": "complete" if len(rows) == total else "pending",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact-dir", type=Path, default=ARTIFACT)
    ap.add_argument("--out", type=Path, default=ROOT / "outputs/checkpoint_eval")
    ap.add_argument("--doctor-endpoint", default="http://127.0.0.1:11535")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--launch", action="store_true")
    args = ap.parse_args()

    cfg, p = prepared_config(args.artifact_dir.resolve(), args.doctor_endpoint)
    preflight = {
        "base_model": cfg["policy"]["base_model"],
        "base_revision": BASE_REVISION,
        "adapter": str(p["adapter"]),
        "adapter_file_sha256": ADAPTER_FILE_SHA256,
        "expected_policy_sha256": POLICY_SHA256,
        "doctor_model": cfg["rollout"]["doctor_backend"]["model"],
        "doctor_digest": DOCTOR_DIGEST,
        "doctor_endpoint": args.doctor_endpoint,
        "eval_data": str(p["data"]),
        "eval_data_sha256": DATA_SHA256,
        "rounds": cfg["rollout"]["rounds"],
        "halting": cfg["rollout"]["halting"],
        "launch": args.launch,
    }
    print(json.dumps(preflight, indent=2), flush=True)
    if not args.launch:
        return

    # This evaluator uses one read-only Agent instance. External API credentials
    # are unnecessary because the Doctor is the pinned local Ollama service.
    os.environ["SHERPA_AGENT_REPLICAS"] = "1"
    from src.online_rl.cases import load_cases
    from src.online_rl.environment import final_outcome
    from src.online_rl.integrated.continuation import adapter_sha256, derive_seed
    from src.online_rl.integrated.records import atomic_write_json
    from src.online_rl.runtime import SharedPolicy

    cases = load_cases(p["data"])
    if len(cases) != 622 or len({c["case_id"] for c in cases}) != 622:
        raise RuntimeError("reported checkpoint evaluation set must contain 622 unique cases")
    policy = SharedPolicy(cfg, "", device=args.device)
    doctor_problems = policy.doctor.verify()["problems"]
    if doctor_problems:
        raise RuntimeError("Doctor identity mismatch: " + "; ".join(doctor_problems))
    loaded_policy = adapter_sha256(policy.model)
    if loaded_policy != POLICY_SHA256:
        raise RuntimeError(f"loaded adapter policy hash {loaded_policy} != expected {POLICY_SHA256}")

    out = args.out.resolve()
    case_dir = out / "cases"
    case_dir.mkdir(parents=True, exist_ok=True)
    identity = {**preflight, "loaded_policy_sha256": loaded_policy, "launch": True}
    manifest = out / "manifest.json"
    if manifest.exists() and json.loads(manifest.read_text()) != identity:
        raise RuntimeError("output directory belongs to another evaluation identity")
    atomic_write_json(manifest, identity)

    for index, case in enumerate(cases):
        dest = case_dir / f"case_{index:06d}.json"
        if dest.exists():
            old = json.loads(dest.read_text())
            if old.get("case_id") != case["case_id"] or old.get("policy_sha256") != POLICY_SHA256:
                raise RuntimeError(f"foreign result at {dest}")
            continue
        seed = derive_seed(0, case["case_id"], "eval", 0)
        started = time.monotonic()
        episode = policy.generate(case, seed)
        row = {
            "index": index,
            "case_id": case["case_id"],
            "seed": seed,
            "policy_sha256": POLICY_SHA256,
            **episode,
            "outcome": final_outcome(case, episode["final_option"]),
            "seconds": time.monotonic() - started,
        }
        atomic_write_json(dest, row)
        summary = summarize(case_dir, len(cases))
        atomic_write_json(out / "summary.json", summary)
        print(json.dumps({"stage": "checkpoint_eval", "index": index + 1, "total": len(cases), **summary}), flush=True)

    atomic_write_json(out / "summary.json", summarize(case_dir, len(cases)))


if __name__ == "__main__":
    main()
