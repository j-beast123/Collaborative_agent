"""Can one Effectiveness service (one predictor on one GPU) serve several GRPO runs at once? Measured, not assumed."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from src.common.io import read_jsonl
from src.common.main_config import load_main_config
from src.effectiveness.service import LocalEffectiveness, build_effectiveness

ROOT = Path(__file__).resolve().parents[2]
TOLERANCE = 1e-4


def request_set(labels: Path, n: int) -> list[dict]:
    """The first n TRAIN pairs taken turn by turn in rotation, so that every intervention turn is in the set."""
    by_turn: dict[str, list] = {}
    for row in read_jsonl(labels):
        by_turn.setdefault(row["intervention_turn"], []).append(row)
    chosen, index = [], 0
    while len(chosen) < n and any(index < len(rows) for rows in by_turn.values()):
        for turn in sorted(by_turn):
            if index < len(by_turn[turn]) and len(chosen) < n:
                row = by_turn[turn][index]
                chosen.append({"key": f"{row['history_id']}/{row['response_id']}", "history": row["history"], "response": row["response"],
                               "intervention_turn": row["intervention_turn"]})
        index += 1
    return chosen


def gpu_sampler(gpu: int, samples: list, stop: threading.Event) -> None:
    while not stop.is_set():
        try:
            used, util = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits",
                                                  "-i", str(gpu)], text=True).strip().split(", ")
            samples.append((int(used), int(util)))
        except Exception:  # noqa: BLE001 - a missed sample is not a result
            pass
        stop.wait(1.0)


def run_setting(section: dict, requests: list[dict], clients: int, reference: dict, gpu: int | None) -> dict:
    samples, stop = [], threading.Event()
    sampler = threading.Thread(target=gpu_sampler, args=(gpu, samples, stop), daemon=True) if gpu is not None else None

    def client(index: int) -> list[dict]:
        provider = build_effectiveness(section, 1.0, ROOT)  # the client a GRPO run builds: own channel, identity checked
        rows = []
        for request in requests:
            sent, started = uuid.uuid4().hex, time.time()
            try:
                prediction = provider.predict(request["history"], request["response"], request["intervention_turn"],
                                              request_id=sent, trajectory_id=f"benchmark:{clients}:{index}")
                rows.append({"key": request["key"], "latency_s": time.time() - started, "ok": True, "same_request_id": prediction.request_id == sent,
                             "p_before": prediction.p_before, "p_after": prediction.p_after, "delta": prediction.delta})
            except Exception as error:  # noqa: BLE001 - counted and reported
                rows.append({"key": request["key"], "latency_s": time.time() - started, "ok": False, "error": f"{type(error).__name__}: {str(error)[:200]}"})
        return rows

    if sampler:
        sampler.start()
    started = time.time()
    with ThreadPoolExecutor(clients) as pool:
        results = [row for rows in pool.map(client, range(clients)) for row in rows]
    elapsed = time.time() - started
    stop.set()
    ok = [r for r in results if r["ok"]]
    latencies = sorted(r["latency_s"] for r in ok)
    differences = [max(abs(r[k] - reference[r["key"]][k]) for k in ("p_before", "p_after", "delta")) for r in ok] if reference else []
    return {"clients": clients, "requests_sent": len(results), "completed": len(ok), "errors": len(results) - len(ok),
            "error_examples": sorted({r["error"] for r in results if not r["ok"]})[:3], "dropped_requests": len(results) - len(ok),
            "wall_seconds": elapsed, "requests_per_second": len(ok) / elapsed if elapsed else None,
            "mean_latency_s": statistics.mean(latencies) if latencies else None,
            "p95_latency_s": latencies[max(0, int(0.95 * len(latencies)) - 1)] if latencies else None,
            "max_latency_s": latencies[-1] if latencies else None,
            "results_under_another_request_id": sum(not r["same_request_id"] for r in ok),
            "max_abs_difference_from_offline": max(differences) if differences else None,
            "results_differing_from_offline": sum(d > TOLERANCE for d in differences) if reference else None,
            "service_gpu_memory_mib_max": max((s[0] for s in samples), default=None),
            "service_gpu_utilization_mean": statistics.mean(s[1] for s in samples) if samples else None,
            "service_gpu_utilization_max": max((s[1] for s in samples), default=None)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--requests", type=int, default=60)
    parser.add_argument("--clients", type=int, nargs="+", default=[1, 2, 5])
    parser.add_argument("--service-gpu", type=int, help="physical GPU of the running service, for utilization and memory")
    parser.add_argument("--no-offline", action="store_true", help="skip the offline predictor (no GPU for it): no parity columns")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = load_main_config(ROOT / args.config, ROOT)
    section = config["reward"]["effectiveness"]
    labels = ROOT / Path(section["checkpoint"]).parent / "labels" / "labels_train.jsonl"
    requests = request_set(labels, args.requests)
    reference = {}
    if not args.no_offline:
        local = LocalEffectiveness.from_checkpoint(str(ROOT / section["checkpoint"]), model_version=section["model_version"],
                                                   calibration=str(ROOT / section["calibration"]), device="cuda:0",
                                                   supported_turns=tuple(section["stages"]),
                                                   expected_protocol_hash=section["expected_continuation_protocol_hash"])
        for request in requests:
            p = local.predict(request["history"], request["response"], request["intervention_turn"])
            reference[request["key"]] = {"p_before": p.p_before, "p_after": p.p_after, "delta": p.delta}
        offline_identity = local.identity()
        del local
    settings = [run_setting(section, requests, clients, reference, args.service_gpu) for clients in args.clients]
    result = {"config": args.config, "endpoint": section.get("endpoint"), "request_set": {"pairs": len(requests), "source": str(labels.relative_to(ROOT)),
                                                                                           "turns": sorted({r["intervention_turn"] for r in requests})},
              "offline_reference": None if args.no_offline else {k: offline_identity.get(k) for k in ("model_version", "checkpoint_id", "calibration_id", "calibration_temperature")},
              "tolerance": TOLERANCE, "settings": settings, "service_handles_one_request_at_a_time": True, "time": time.strftime("%Y-%m-%d %H:%M:%S"),
              "correct": all(s["errors"] == 0 and s["results_under_another_request_id"] == 0
                             and (args.no_offline or s["results_differing_from_offline"] == 0) for s in settings)}
    Path(ROOT / args.output).parent.mkdir(parents=True, exist_ok=True)
    (ROOT / args.output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
