"""Serve the Effectiveness Model over gRPC, in its own process and on its own device."""

from __future__ import annotations

import argparse
import json
import signal
import threading
from pathlib import Path

from src.effectiveness.service import LocalEffectiveness, start_server


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", help="main config: checkpoint, calibration, model version, expected temperature and "
                        "protocol hash, turns, device and port are read from reward.effectiveness and devices")
    parser.add_argument("--iteration", type=int, default=0, help="with --config: the iteration whose predictor is served")
    parser.add_argument("--checkpoint")
    parser.add_argument("--calibration", help="monotonic calibration file ({method, a, b}); omit for raw probabilities")
    parser.add_argument("--model-version")
    parser.add_argument("--expected-temperature", type=float)
    parser.add_argument("--expected-protocol-hash")
    parser.add_argument("--intervention-turn", action="append", choices=tuple(f"A{t}" for t in range(1, 8)),
                        help="turn(s) this checkpoint was trained on; other turns are refused")
    parser.add_argument("--device", help="e.g. cuda:0")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int)
    parser.add_argument("--log-dir")
    args = parser.parse_args()
    if args.config:
        from src.common.main_config import load_main_config
        from src.grpo.trainer import effectiveness_settings

        root = Path(__file__).resolve().parents[1]
        config = load_main_config(args.config, root)
        section = effectiveness_settings(config, args.iteration)
        host, _, port = section["endpoint"].rpartition(":")
        args.checkpoint = args.checkpoint or section["checkpoint"]
        args.calibration = args.calibration or section["calibration"]
        args.model_version = args.model_version or section["model_version"]
        args.expected_temperature = args.expected_temperature if args.expected_temperature is not None else section["expected_temperature"]
        args.expected_protocol_hash = args.expected_protocol_hash or section["expected_continuation_protocol_hash"]
        args.intervention_turn = args.intervention_turn or list(section["stages"])
        args.device = args.device or config["devices"]["effectiveness_service"]
        args.port = args.port if args.port is not None else int(port)
        args.host = host or args.host
    missing = [name for name in ("checkpoint", "model_version", "intervention_turn", "device", "port") if getattr(args, name) in (None, [])]
    if missing:
        parser.error(f"missing {missing}: give them, or --config")

    predictor = LocalEffectiveness.from_checkpoint(
        args.checkpoint, model_version=args.model_version, calibration=args.calibration, device=args.device,
        supported_turns=tuple(args.intervention_turn), allow_cpu=args.allow_cpu,
        expected_protocol_hash=args.expected_protocol_hash, expected_temperature=args.expected_temperature)
    log_path = Path(args.log_dir) / "effectiveness_requests.jsonl" if args.log_dir else None
    server, port = start_server(predictor, port=args.port, host=args.host, log_path=log_path)
    print(json.dumps(dict(predictor.identity(), endpoint=f"{args.host}:{port}", status="SERVING")), flush=True)

    stop = threading.Event()
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, lambda *_: stop.set())
    stop.wait()
    server.stop(grace=30).wait()
    print(json.dumps({"status": "STOPPED"}), flush=True)


if __name__ == "__main__":
    main()
