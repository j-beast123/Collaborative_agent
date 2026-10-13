from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
from src.common.config import IntegrationUnavailable

@dataclass(frozen=True)
class CheckpointMetadata:
    checkpoint_id: str
    model_id: str
    model_revision: str
    iteration: int
    rubric_version: str
    reward_version: str

@dataclass(frozen=True)
class TrainingRequest:
    run_id: str
    split_manifest_hash: str
    initial_checkpoint: CheckpointMetadata
    seed: int
    resume_checkpoint: Optional[str] = None

@dataclass(frozen=True)
class TrajectoryRequest:
    checkpoint: CheckpointMetadata
    case_ids: tuple
    protocol_hash: str
    seed: int

class TrainingBackend:
    """Unavailable until integration. Resume must restore optimizer/scheduler/RNG."""
    capabilities = frozenset()
    def train(self, request: TrainingRequest) -> CheckpointMetadata:
        raise IntegrationUnavailable("Production GRPO backend missing; the TRL smoke script is not a backend")
    def load_checkpoint(self, checkpoint_id: str) -> CheckpointMetadata:
        raise IntegrationUnavailable("Checkpoint loading requires a verified training backend")
    def generate_trajectories(self, request: TrajectoryRequest) -> list:
        raise IntegrationUnavailable("Trajectory generation requires a configured backend")
    def get_checkpoint_metadata(self, checkpoint_id: str) -> CheckpointMetadata:
        raise IntegrationUnavailable("Checkpoint metadata requires a configured backend")
