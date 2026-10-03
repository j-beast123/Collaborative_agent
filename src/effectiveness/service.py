"""The Effectiveness Model as a component: one prediction schema, local or over gRPC."""

from __future__ import annotations

import json
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

from src.common.config import IntegrationUnavailable
from src.data.schema import digest

from . import calibration as calibration_module
from .history import format_input

SERVICE = "serl.effectiveness.v1.Effectiveness"
SCHEMA_VERSION = "effectiveness-prediction-v1"
from .history import TURNS  # noqa: E402  (A1, A2, ...: the Agent turns a predictor may cover)
IDENTITY_FIELDS = ("model_version", "checkpoint_id", "calibration_temperature", "continuation_protocol_hash",
                   "supported_turns", "calibration_id")


class EffectivenessUnavailable(RuntimeError):
    """No predictor covers the requested intervention turn."""


@dataclass
class EffectivenessPrediction:
    p_before: float
    p_after: Optional[float]          # None when no response was given (P(success | h) only)
    delta: Optional[float]
    logit_before: float               # raw logits, before calibration
    logit_after: Optional[float]
    intervention_turn: str
    model_version: str
    checkpoint_id: str
    calibration_temperature: Optional[float]
    calibration: Optional[dict]
    device: str
    source: str                       # "local" or "grpc"
    request_id: str
    trajectory_id: Optional[str] = None
    latency_ms: Optional[float] = None
    schema_version: str = SCHEMA_VERSION
    protocol_hash: Optional[str] = None
    calibration_type: Optional[str] = None
    raw_logit_before: Optional[float] = None
    raw_logit_after: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_history(history: Any) -> None:
    if not isinstance(history, dict) or not isinstance(history.get("turns"), list) or not all(
            isinstance(t, dict) and isinstance(t.get("speaker"), str) and isinstance(t.get("text"), str) for t in history["turns"]):
        raise ValueError("history needs turns: [{speaker, text}]")
    if "scenario" not in history:
        raise ValueError("history needs scenario (the information the Agent has)")


class LocalEffectiveness:
    """Direct inference with a loaded checkpoint and its calibration."""

    source = "local"

    def __init__(self, model, *, checkpoint: str, model_version: str, calibration: Optional[dict],
                 supported_turns: tuple = ("A1",), cache_size: int = 4096) -> None:
        if set(supported_turns) - set(TURNS) or not supported_turns:
            raise ValueError(f"supported_turns must be a non-empty subset of {TURNS}")
        self.model, self.checkpoint, self.model_version = model, checkpoint, model_version
        self.calibration = calibration
        self.supported_turns = tuple(supported_turns)
        self._before: OrderedDict[str, float] = OrderedDict()  # raw BEFORE logit per history input
        self._cache_size, self._lock = cache_size, threading.Lock()
        config = json.loads((Path(checkpoint) / "predictor_config.json").read_text(encoding="utf-8"))
        self.checkpoint_id = config["trainable_state_sha256"][:16]
        self.metadata = config.get("metadata", {})
        self.architecture = config.get("architecture", "single")
        self.base_model = f"{config['base_model_id']}@{config['base_model_revision']}"
        self.input_format = config.get("input_format", "v1")
        trained = config.get("supported_turns", ["A1"])
        if set(self.supported_turns) - set(trained):
            raise IntegrationUnavailable(f"the checkpoint was trained for turns {trained}, not for {list(self.supported_turns)}")
        if self.input_format == "v1" and self.supported_turns != ("A1",):
            raise IntegrationUnavailable("an A1-only predictor (input_format v1) cannot serve another turn")
        if calibration and calibration.get("method") == calibration_module.BY_TURN \
                and set(self.supported_turns) - set(calibration["by_turn"]):
            raise IntegrationUnavailable("the calibration has no temperature for every supported turn")

    @classmethod
    def from_checkpoint(cls, checkpoint: Optional[str], *, model_version: Optional[str], calibration: Optional[str],
                        device: Optional[str], supported_turns: tuple = ("A1",), allow_cpu: bool = False,
                        expected_protocol_hash: Optional[str] = None, expected_temperature: Optional[float] = None,
                        backbone_loader=None) -> "LocalEffectiveness":
        """Everything is checked here, before the first request: checkpoint files, protocol hash,
        calibration file and temperature, and the device the model really ended up on."""
        if not checkpoint or not model_version:
            raise IntegrationUnavailable("the Effectiveness predictor needs checkpoint and model_version")
        path = Path(checkpoint)
        for name in ("predictor_config.json", "trainable_state.pt"):
            if not (path / name).is_file():
                raise IntegrationUnavailable(f"Effectiveness checkpoint {path} is missing {name}")
        metadata = json.loads((path / "predictor_config.json").read_text(encoding="utf-8")).get("metadata", {})
        if expected_protocol_hash and metadata.get("continuation_protocol_hash") != expected_protocol_hash:
            raise IntegrationUnavailable("the Effectiveness checkpoint was trained under a different continuation protocol")
        parameters = calibration_module.load(calibration) if calibration else None
        if expected_temperature is not None:
            actual = parameters and parameters.get("temperature")
            if actual is None or abs(actual - expected_temperature) > 0.005:
                raise IntegrationUnavailable(f"calibration temperature is {actual}, expected {expected_temperature}")
        import torch

        from .model import CollaborationValueModel, load_hf_backbone

        if device is None:
            raise IntegrationUnavailable("the Effectiveness predictor needs an explicit device (e.g. cuda:0)")
        if device.startswith("cpu") and not allow_cpu:
            raise IntegrationUnavailable("the Effectiveness predictor is not placed on CPU unless allow_cpu is set")
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise IntegrationUnavailable(f"device {device} was requested but no GPU is visible")
        loader = backbone_loader or (lambda model_id, revision, **options: load_hf_backbone(
            model_id, revision, **{**options, "device": device}))
        model = CollaborationValueModel.from_pretrained(path, backbone_loader=loader)
        if backbone_loader is None and torch.device(model.device) != torch.device(device):
            raise IntegrationUnavailable(f"the predictor is on {model.device}, not on the requested {device}")
        return cls(model, checkpoint=str(path), model_version=model_version, calibration=parameters,
                   supported_turns=supported_turns)

    def identity(self) -> dict[str, Any]:
        return {"schema_version": SCHEMA_VERSION, "model_version": self.model_version, "checkpoint": self.checkpoint,
                "checkpoint_id": self.checkpoint_id, "base_model": self.base_model, "architecture": self.architecture,
                "calibration": self.calibration, "calibration_id": digest(self.calibration),
                "calibration_temperature": (self.calibration or {}).get("temperature"),
                "calibration_temperature_by_turn": {turn: self._temperature(turn) for turn in self.supported_turns},
                "input_format": self.input_format,
                "continuation_protocol_hash": self.metadata.get("continuation_protocol_hash"),
                "supported_turns": list(self.supported_turns), "device": str(self.model.device), "source": self.source}

    def _probability(self, logit: float, turn: str) -> float:
        return calibration_module.apply_logits(self.calibration or {"a": 1.0, "b": 0.0}, [logit], turn)[0]

    def _temperature(self, turn: str) -> Optional[float]:
        return calibration_module.for_turn(self.calibration, turn).get("temperature") if self.calibration else None

    def _logits(self, before_text: str, after_text: Optional[str]) -> tuple[float, Optional[float]]:
        """Raw logits; every text is encoded on its own, so a prediction does not depend on what it is batched with."""
        key = digest(before_text)
        with self._lock:
            cached = self._before.get(key)
            if cached is None:
                cached = self.model.example_logits([SimpleNamespace(kind="before", history_id="h", text=before_text)], 1)[0]
                self._before[key] = cached
                while len(self._before) > self._cache_size:
                    self._before.popitem(last=False)
            if after_text is None:
                return cached, None
            if self.architecture == "single":
                after = self.model.example_logits([SimpleNamespace(kind="after", history_id="h", text=after_text)], 1)[0]
            else:  # residual: B(h) + G(h, r)
                after = self.model.example_logits([SimpleNamespace(kind="before", history_id="h", text=before_text),
                                                   SimpleNamespace(kind="after", history_id="h", text=after_text)], 1)[1]
            return cached, after

    def predict(self, history: dict, response: Optional[str], intervention_turn: str = "A1", *,
                request_id: Optional[str] = None, trajectory_id: Optional[str] = None) -> EffectivenessPrediction:
        if intervention_turn not in self.supported_turns:
            raise EffectivenessUnavailable(f"no Effectiveness predictor for turn {intervention_turn} "
                                           f"(this one covers {list(self.supported_turns)})")
        validate_history(history)
        if response is not None and (not isinstance(response, str) or not response.strip()):
            raise ValueError("response must be a non-empty string, or null for P(success | h) only")
        started = time.perf_counter()
        z_before, z_after = self._logits(
            format_input(history, None, self.input_format, intervention_turn),
            format_input(history, response, self.input_format, intervention_turn) if response is not None else None)
        p_before = self._probability(z_before, intervention_turn)
        p_after = self._probability(z_after, intervention_turn) if z_after is not None else None
        return EffectivenessPrediction(
            p_before=p_before, p_after=p_after, delta=None if p_after is None else p_after - p_before,
            logit_before=z_before, logit_after=z_after, intervention_turn=intervention_turn,
            model_version=self.model_version, checkpoint_id=self.checkpoint_id,
            calibration_temperature=self._temperature(intervention_turn), calibration=self.calibration,
            device=str(self.model.device), source=self.source, request_id=request_id or uuid.uuid4().hex,
            trajectory_id=trajectory_id, latency_ms=(time.perf_counter() - started) * 1000,
            protocol_hash=self.metadata.get("continuation_protocol_hash"),
            calibration_type=(self.calibration or {}).get("method", "identity"),
            raw_logit_before=z_before, raw_logit_after=z_after)


def check_identity(identity: dict, expected: dict) -> None:
    """Fail when the predictor is not the one the configuration names (expected: a subset of IDENTITY_FIELDS)."""
    problems = []
    for name, value in expected.items():
        actual = identity.get(name)
        if value is None:
            continue
        if name == "calibration_temperature":
            if actual is None or abs(actual - value) > 0.005:
                problems.append(f"calibration_temperature is {actual}, expected {value}")
        elif name == "supported_turns":
            if set(value) != set(actual or []):
                problems.append(f"supported_turns are {actual}, expected {value}")
        elif actual != value:
            problems.append(f"{name} is {actual!r}, expected {value!r}")
    if problems:
        raise IntegrationUnavailable("Effectiveness predictor mismatch: " + "; ".join(problems))


# ---------------------------------------------------------------- gRPC

def _dump(message: dict) -> bytes:
    return json.dumps(message, ensure_ascii=False).encode("utf-8")


def _load(data: bytes) -> Any:
    return json.loads(data.decode("utf-8"))


class EffectivenessServicer:
    """gRPC handlers around one loaded predictor. Requests are served one at a time (one model, one GPU)."""

    def __init__(self, predictor: LocalEffectiveness, log_path: Optional[Path] = None) -> None:
        self.predictor, self.log_path = predictor, Path(log_path) if log_path else None
        self.lock, self.requests, self.errors = threading.Lock(), 0, 0

    def _log(self, row: dict) -> None:
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    def predict(self, request: Any, context) -> dict:
        import grpc

        request_id = request.get("request_id") if isinstance(request, dict) else None
        row = {"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "request_id": request_id,
               "trajectory_id": request.get("trajectory_id") if isinstance(request, dict) else None}

        def fail(code, message: str):
            self.errors += 1
            self._log(dict(row, status=code.name, error=message))
            context.abort(code, message)

        if not isinstance(request, dict) or "history" not in request:
            fail(grpc.StatusCode.INVALID_ARGUMENT, "request must be an object with history, response, intervention_turn")
        try:
            with self.lock:
                self.requests += 1
                prediction = self.predictor.predict(request["history"], request.get("response"),
                                                    request.get("intervention_turn", "A1"), request_id=request_id,
                                                    trajectory_id=request.get("trajectory_id"))
        except EffectivenessUnavailable as error:
            fail(grpc.StatusCode.FAILED_PRECONDITION, str(error))
        except (ValueError, KeyError, TypeError) as error:
            fail(grpc.StatusCode.INVALID_ARGUMENT, f"{type(error).__name__}: {error}")
        except Exception as error:  # noqa: BLE001 - reported to the caller, never replaced by a value
            out_of_memory = type(error).__name__ == "OutOfMemoryError"
            fail(grpc.StatusCode.RESOURCE_EXHAUSTED if out_of_memory else grpc.StatusCode.INTERNAL,
                 f"{type(error).__name__}: {error}")
        result = dict(prediction.to_dict(), source="grpc")
        self._log(dict(row, status="OK", intervention_turn=prediction.intervention_turn, p_before=prediction.p_before,
                       p_after=prediction.p_after, delta=prediction.delta, latency_ms=prediction.latency_ms,
                       request_id=prediction.request_id, request=request, response=result))
        return result

    def identity(self, request: Any, context) -> dict:
        return dict(self.predictor.identity(), source="grpc")

    def health(self, request: Any, context) -> dict:
        return {"status": "SERVING", "device": str(self.predictor.model.device), "requests": self.requests,
                "errors": self.errors, "checkpoint_id": self.predictor.checkpoint_id}


def start_server(predictor: LocalEffectiveness, *, port: int, host: str = "127.0.0.1", log_path: Optional[Path] = None):
    """(server, bound port). The caller keeps the server and stops it with server.stop(grace)."""
    from concurrent import futures

    import grpc

    servicer = EffectivenessServicer(predictor, log_path)
    handlers = {name: grpc.unary_unary_rpc_method_handler(function, request_deserializer=_load, response_serializer=_dump)
                for name, function in (("Predict", servicer.predict), ("Identity", servicer.identity), ("Health", servicer.health))}
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4),
                         options=[("grpc.max_receive_message_length", 32 * 1024 * 1024)])
    server.add_generic_rpc_handlers((grpc.method_handlers_generic_handler(SERVICE, handlers),))
    bound = server.add_insecure_port(f"{host}:{port}")
    if bound == 0:
        raise IntegrationUnavailable(f"could not bind {host}:{port}")
    server.start()
    return server, bound


class GrpcEffectivenessClient:
    """The served predictor behind the interface of LocalEffectiveness."""

    source = "grpc"

    def __init__(self, endpoint: Optional[str], *, expected: Optional[dict] = None, timeout_s: float = 120.0,
                 retries: int = 2, retry_wait_s: float = 2.0) -> None:
        import grpc

        if not endpoint:
            raise IntegrationUnavailable("the gRPC Effectiveness predictor needs reward.effectiveness.endpoint")
        self.endpoint, self.timeout_s, self.retries, self.retry_wait_s = endpoint, timeout_s, retries, retry_wait_s
        self.channel = grpc.insecure_channel(endpoint, options=[("grpc.max_send_message_length", 32 * 1024 * 1024)])
        self._calls = {name: self.channel.unary_unary(f"/{SERVICE}/{name}", request_serializer=_dump,
                                                      response_deserializer=_load)
                       for name in ("Predict", "Identity", "Health")}
        try:
            self._identity = self._call("Identity", {})
            check_identity(self._identity, expected or {})
        except Exception:
            self.channel.close()
            raise
        self.supported_turns = tuple(self._identity["supported_turns"])
        self.model_version, self.checkpoint_id = self._identity["model_version"], self._identity["checkpoint_id"]

    def _call(self, name: str, message: dict) -> dict:
        import grpc

        for attempt in range(self.retries + 1):
            try:
                return self._calls[name](message, timeout=self.timeout_s)
            except grpc.RpcError as error:
                code = error.code()
                if code == grpc.StatusCode.FAILED_PRECONDITION:
                    raise EffectivenessUnavailable(error.details()) from None
                if code == grpc.StatusCode.INVALID_ARGUMENT:
                    raise ValueError(error.details()) from None
                retryable = code in (grpc.StatusCode.UNAVAILABLE, grpc.StatusCode.DEADLINE_EXCEEDED)
                if not retryable or attempt == self.retries:
                    raise IntegrationUnavailable(
                        f"Effectiveness service {self.endpoint} failed ({code.name}): {error.details()}") from None
                time.sleep(self.retry_wait_s)
        raise AssertionError("unreachable")

    def identity(self) -> dict[str, Any]:
        return dict(self._identity, endpoint=self.endpoint)

    def health(self) -> dict:
        return self._call("Health", {})

    def predict(self, history: dict, response: Optional[str], intervention_turn: str = "A1", *,
                request_id: Optional[str] = None, trajectory_id: Optional[str] = None) -> EffectivenessPrediction:
        if intervention_turn not in self.supported_turns:  # the same answer as the local predictor, without a call
            raise EffectivenessUnavailable(f"no Effectiveness predictor for turn {intervention_turn} "
                                           f"(this one covers {list(self.supported_turns)})")
        message = self._call("Predict", {"history": history, "response": response, "intervention_turn": intervention_turn,
                                         "request_id": request_id or uuid.uuid4().hex, "trajectory_id": trajectory_id})
        by_turn = self._identity.get("calibration_temperature_by_turn") or {}
        check_identity(message, {"calibration_temperature": by_turn.get(intervention_turn, self._identity["calibration_temperature"]),
                                 **{k: self._identity[k] for k in ("model_version", "checkpoint_id", "schema_version")}})
        if message.get("calibration") != self._identity.get("calibration"):
            raise IntegrationUnavailable("Effectiveness calibration changed during the run")
        return EffectivenessPrediction(**message)

    def close(self) -> None:
        self.channel.close()


class ProviderEffectiveness:
    """An EffectivenessProvider (predict_before / predict_after, src/effectiveness/providers.py) behind the
    same interface, for the rollout provider and for tests."""

    source = "provider"

    def __init__(self, provider, *, supported_turns: tuple = ("A1",), model_version: Optional[str] = None) -> None:
        self.provider, self.supported_turns = provider, tuple(supported_turns)
        self.model_version = model_version or provider.identity().get("provider", "provider")
        self.checkpoint_id = str(provider.identity().get("checkpoint"))
        self._before: dict[str, float] = {}  # P(success | h) is shared by the responses to one history

    def identity(self) -> dict[str, Any]:
        return {"schema_version": SCHEMA_VERSION, "model_version": self.model_version, "checkpoint_id": self.checkpoint_id,
                "calibration_temperature": None, "supported_turns": list(self.supported_turns), "source": self.source,
                "provider": self.provider.identity()}

    def predict(self, history: dict, response: Optional[str], intervention_turn: str = "A1", *,
                request_id: Optional[str] = None, trajectory_id: Optional[str] = None) -> EffectivenessPrediction:
        if intervention_turn not in self.supported_turns:
            raise EffectivenessUnavailable(f"no Effectiveness predictor for turn {intervention_turn} "
                                           f"(this one covers {list(self.supported_turns)})")
        started = time.perf_counter()
        key = digest({"scenario": history.get("scenario"), "turns": [t["text"] for t in history["turns"]]})
        if key not in self._before:
            self._before[key] = self.provider.predict_before(history)
        before = self._before[key]
        after = self.provider.predict_after(history, response) if response is not None else None
        return EffectivenessPrediction(
            p_before=before, p_after=after, delta=None if after is None else after - before, logit_before=float("nan"),
            logit_after=None, intervention_turn=intervention_turn, model_version=self.model_version,
            checkpoint_id=self.checkpoint_id, calibration_temperature=None, calibration=None, device="n/a",
            source=self.source, request_id=request_id or uuid.uuid4().hex, trajectory_id=trajectory_id,
            latency_ms=(time.perf_counter() - started) * 1000)


def build_effectiveness(section: dict, weight: float, root: Optional[Path] = None):
    """The reward's Effectiveness predictor from reward.effectiveness; None when w_E = 0.
        root: the repository root that checkpoint and calibration paths are relative to (default: the working directory).
    """
    if weight == 0:
        return None
    kind = (section or {}).get("provider")
    stages = tuple(section.get("stages") or ())
    expected = {"model_version": section.get("model_version"),
                "calibration_id": section.get("expected_calibration_id"),
                "checkpoint_id": section.get("expected_checkpoint_id"),
                "calibration_temperature": section.get("expected_temperature"),
                "continuation_protocol_hash": section.get("expected_continuation_protocol_hash"),
                "supported_turns": list(stages)}
    root = Path(root) if root else Path(".")
    checkpoint_path = str(root / section["checkpoint"]) if section.get("checkpoint") else None
    calibration_path = str(root / section["calibration"]) if section.get("calibration") else None
    if checkpoint_path:
        checkpoint = json.loads((Path(checkpoint_path) / "predictor_config.json").read_text(encoding="utf-8"))
        actual_id = checkpoint["trainable_state_sha256"][:16]
        if expected.get("checkpoint_id") is not None and expected["checkpoint_id"] != actual_id:
            raise IntegrationUnavailable("configured checkpoint id differs from checkpoint artifact")
        expected["checkpoint_id"] = actual_id
    if calibration_path:
        actual_id = digest(calibration_module.load(calibration_path))
        if expected.get("calibration_id") is not None and expected["calibration_id"] != actual_id:
            raise IntegrationUnavailable("configured calibration id differs from calibration artifact")
        expected["calibration_id"] = actual_id
    if kind == "grpc":
        return GrpcEffectivenessClient(section.get("endpoint"), expected=expected,
                                       timeout_s=section.get("timeout_s", 120.0), retries=section.get("retries", 2))
    if kind == "local":
        predictor = LocalEffectiveness.from_checkpoint(
            checkpoint_path, model_version=section.get("model_version"), calibration=calibration_path,
            device=section.get("device"), supported_turns=stages, allow_cpu=bool(section.get("allow_cpu")),
            expected_protocol_hash=section.get("expected_continuation_protocol_hash"),
            expected_temperature=section.get("expected_temperature"))
        check_identity(predictor.identity(), expected)
        return predictor
    raise IntegrationUnavailable("w_E > 0 requires reward.effectiveness.provider: grpc or local")
