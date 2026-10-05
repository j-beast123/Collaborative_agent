from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any, Optional
import json
import hashlib

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()

def identity(source, value):
    return json.dumps([source, value], separators=(",", ":"), ensure_ascii=False)

@dataclass(frozen=True)
class NormalizedCase:
    case_id: str
    original_case_id: str
    source_dataset: str
    dataset_version: str
    question: str
    answer_format: str
    choices: Optional[dict] = None
    correct_answer: Any = None
    evidence: Optional[list] = None
    rationale: Optional[str] = None
    language: Optional[str] = None
    domain: Optional[str] = None
    metadata: dict = field(default_factory=dict)
    synthetic: bool = False

    def __post_init__(self):
        for key in ("case_id", "original_case_id", "source_dataset", "dataset_version", "question", "answer_format"):
            if not isinstance(getattr(self, key), str) or not getattr(self, key).strip():
                raise ValueError("Missing/non-string " + key)
        if type(self.synthetic) is not bool:
            raise ValueError("synthetic must be a boolean")
        for name, expected in (("choices", dict), ("evidence", list), ("rationale", str), ("language", str), ("domain", str)):
            value = getattr(self, name)
            if value is not None and not isinstance(value, expected):
                raise ValueError("Invalid normalized field type: " + name)
        if not isinstance(self.metadata, dict):
            raise ValueError("metadata must be a mapping")
        digest(asdict(self))

    @property
    def key(self):
        return identity(self.source_dataset, self.case_id)

    @property
    def original_key(self):
        return identity(self.source_dataset, self.original_case_id)

@dataclass(frozen=True)
class AdapterMetadata:
    source: str
    version: Optional[str] = None
    license_status: Optional[str] = None
    fields_available: tuple = ()
    evidence_availability: Optional[str] = None
    evidence_provenance: Optional[str] = None
    rationale_provenance: Optional[str] = None
    answer_format: Optional[str] = None
    normalization_version: str = "1"

@dataclass(frozen=True)
class DerivedRecord:
    record_id: str
    source_dataset: str
    case_id: str
    original_case_id: str
    split: str
    synthetic: bool
    parent_record_id: Optional[str] = None
