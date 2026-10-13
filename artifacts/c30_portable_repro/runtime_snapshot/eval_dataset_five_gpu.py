"""Read-only five-GPU evaluation of a saved Agent adapter on an arbitrary case JSONL.

This does not touch training state or the checkpoint evaluation queue. Each GPU owns
one frozen Agent replica and uses its already-pinned local Doctor endpoint.
"""
import argparse
import json
import multiprocessing
import os
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

_POLICY = None
_OUT = None


def _init(config_path: str, checkpoint: str, out: str, worker: int, expected_sha: str, key: str):
    global _POLICY, _OUT
    os.environ["CUDA_VISIBLE_DEVICES"] = str(worker)
    os.environ["SHERPA_AGENT_REPLICAS"] = "1"
    # OllamaDoctor uses one queue token per configured endpoint. Repeating the
    # same local endpoint intentionally exposes multiple concurrency slots to
    # the server, whose OLLAMA_NUM_PARALLEL setting performs the actual batch.
    doctor_slots = max(1, int(os.environ.get("SHERPA_DOCTOR_SLOTS", "1")))
    from src.online_rl.runtime import SharedPolicy
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file
    from src.online_rl.integrated.continuation import adapter_sha256

    cfg = json.loads(Path(config_path).read_text())
    _OUT = Path(out)
    cfg["output_dir"] = out
    doctor_endpoint = cfg["five_gpu"]["doctor_urls"][worker]
    os.environ["SHERPA_DOCTOR_ENDPOINTS"] = ",".join([doctor_endpoint] * doctor_slots)
    cfg["rollout"]["doctor_backend"]["endpoint"] = doctor_endpoint
    _POLICY = SharedPolicy(cfg, key)
    set_peft_model_state_dict(
        _POLICY.model,
        load_file(str(Path(checkpoint) / "adapter" / "adapter_model.safetensors")),
    )
    _POLICY.mark_dirty()
    _POLICY.model.eval()
    got = adapter_sha256(_POLICY.model)
    if got != expected_sha:
        raise RuntimeError(f"GPU{worker} adapter mismatch: {got} != {expected_sha}")
    problems = _POLICY.doctor.verify()["problems"]
    if problems:
        raise RuntimeError(f"GPU{worker} Doctor identity: " + "; ".join(problems))


def _play(index_case):
    index, case, seed = index_case
    from src.online_rl.environment import final_outcome
    episode = _POLICY.generate(case, seed)
    return {
        "index": index,
        "case_id": case["case_id"],
        "dialogue": episode["dialogue"],
        "final_option": episode["final_option"],
        "num_discussion_rounds": episode["num_discussion_rounds"],
        "outcome": final_outcome(case, episode["final_option"]),
    }


def _play_many(jobs, lanes):
    """Run several episodes concurrently so Agent requests coalesce in Batcher
    and the local Doctor server can continuously batch concurrent requests."""
    rows = []
    with ThreadPoolExecutor(max_workers=lanes) as pool:
        futures = [pool.submit(_play, job) for job in jobs]
        for future in as_completed(futures):
            row = future.result()
            # Unique index paths make concurrent worker writes independent.
            _atomic(_OUT / "cases" / f"case_{row['index']:06d}.json", row)
            rows.append(row)
    return rows


def _atomic(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(tmp, path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--cases", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=20261013)
    p.add_argument("--workers", type=int, default=5)
    p.add_argument("--gpu-ids", default="0,1,2,3,4",
                   help="Comma-separated physical GPU IDs owned by this evaluation")
    p.add_argument("--lanes-per-gpu", type=int, default=1)
    p.add_argument("--chunk-size", type=int, default=8)
    args = p.parse_args()
    gpu_ids = [int(x) for x in args.gpu_ids.split(",") if x.strip()]
    if not gpu_ids or len(set(gpu_ids)) != len(gpu_ids) or any(x < 0 or x > 4 for x in gpu_ids):
        raise SystemExit("--gpu-ids must contain unique IDs from 0..4")
    if args.workers != len(gpu_ids):
        raise SystemExit("--workers must equal the number of --gpu-ids")
    if args.lanes_per_gpu < 1 or args.chunk_size < args.lanes_per_gpu:
        raise SystemExit("invalid lane/chunk configuration")

    from scripts.online_rl.run_tree import load_key
    from src.online_rl.cases import load_cases, file_sha256
    from src.online_rl.integrated.continuation import adapter_sha256
    from src.online_rl.runtime import SharedPolicy
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file

    config_path = (ROOT / args.config).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    cases_path = Path(args.cases).resolve()
    out = Path(args.out).resolve()
    cases = load_cases(cases_path)
    if not cases:
        raise SystemExit("No evaluation cases")

    # Compute the exact loaded-adapter identity once on an owned GPU construction,
    # then release it before worker replicas are spawned.
    cfg = json.loads(config_path.read_text())
    probe_gpu = gpu_ids[-1]
    cfg["rollout"]["doctor_backend"]["endpoint"] = cfg["five_gpu"]["doctor_urls"][probe_gpu]
    key = load_key()
    probe = SharedPolicy(cfg, key, device=f"cuda:{probe_gpu}")
    set_peft_model_state_dict(probe.model, load_file(str(checkpoint / "adapter" / "adapter_model.safetensors")))
    probe.mark_dirty()
    expected_sha = adapter_sha256(probe.model)
    del probe
    import torch
    torch.cuda.empty_cache()

    identity = {
        "evaluator_protocol": "concurrent_trajectory_v3_doctor_slots",
        "doctor_slots": max(1, int(os.environ.get("SHERPA_DOCTOR_SLOTS", "1"))),
        "config": str(config_path),
        "checkpoint": str(checkpoint),
        "policy_sha256": expected_sha,
        "cases": str(cases_path),
        "cases_sha256": file_sha256(cases_path),
        "case_ids": [c["case_id"] for c in cases],
        "seed": args.seed,
        "workers": args.workers,
        "gpu_ids": gpu_ids,
        "lanes_per_gpu": args.lanes_per_gpu,
        "chunk_size": args.chunk_size,
    }
    manifest = out / "manifest.json"
    if manifest.exists() and json.loads(manifest.read_text()) != identity:
        raise SystemExit("Output directory belongs to a different evaluation identity")
    _atomic(manifest, identity)
    done = {int(x.stem.split("_")[-1]) for x in (out / "cases").glob("case_*.json")}
    todo = [(i, c, args.seed + i) for i, c in enumerate(cases) if i not in done]

    ctx = multiprocessing.get_context("spawn")
    executors = [
        ProcessPoolExecutor(
            max_workers=1,
            mp_context=ctx,
            initializer=_init,
            initargs=(str(config_path), str(checkpoint), str(out), gpu, expected_sha, key),
        )
        for gpu in gpu_ids
    ]
    futures = {}
    try:
        per_gpu = [[] for _ in executors]
        for pos, job in enumerate(todo): per_gpu[pos % len(executors)].append(job)
        for worker, jobs in enumerate(per_gpu):
            for start in range(0, len(jobs), args.chunk_size):
                chunk = jobs[start:start + args.chunk_size]
                fut = executors[worker].submit(_play_many, chunk, args.lanes_per_gpu)
                futures[fut] = [x[0] for x in chunk]
        for fut in as_completed(futures):
            batch_rows = fut.result()
            for row in batch_rows:
                _atomic(out / "cases" / f"case_{row['index']:06d}.json", row)
                done.add(row["index"])
            rows = [json.loads(x.read_text()) for x in sorted((out / "cases").glob("case_*.json"))]
            summary = {
                "evaluated": len(rows),
                "total": len(cases),
                "correct": sum(r["outcome"] == 1 for r in rows),
                "unparsed": sum(r["outcome"] is None for r in rows),
                "accuracy": sum(r["outcome"] == 1 for r in rows) / len(rows),
                "mean_agent_turns": sum(r["num_discussion_rounds"] for r in rows) / len(rows),
                "status": "complete" if len(rows) == len(cases) else "running",
            }
            _atomic(out / "summary.json", summary)
            if len(rows) % 10 == 0 or len(rows) == len(cases):
                print(json.dumps(summary), flush=True)
    finally:
        for ex in executors:
            ex.shutdown(wait=False, cancel_futures=True)


if __name__ == "__main__":
    main()
