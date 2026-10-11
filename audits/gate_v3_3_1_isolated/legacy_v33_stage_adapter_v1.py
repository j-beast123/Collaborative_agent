"""Read-only adapter from validated v3.3 stage records to v3.3.1 records.

This module never infers an absence requirement from the word ``absent`` or
from clause prose.  A legacy absence needs an explicit, hash-bound migration
review record.  The adapter emits provenance for every transformed clause.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

ADAPTER_VERSION = "legacy_v33_stage_adapter_v1"
HERE = Path(__file__).resolve().parent
REVIEW_PATH = HERE / "migration_reviews.json"


class MigrationReviewRequired(ValueError):
    pass


class LegacyStageError(ValueError):
    pass


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def adapter_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _reviews() -> dict[str, dict]:
    rows = json.loads(REVIEW_PATH.read_text())["reviews"]
    return {row["clause_sha256"]: row for row in rows}


def adapt_applicability(record: dict, *, visible_prefix: list[dict]) -> tuple[dict, dict]:
    """Convert one already-validated legacy result without semantic guessing."""
    if set(record) != {"answer", "record"}:
        raise LegacyStageError("Legacy applicability packet has an unexpected shape")
    answer = record["answer"]
    validation = record["record"].get("validation")
    if not isinstance(validation, dict):
        raise LegacyStageError("Legacy parser validation is absent")
    if validation.get("applicability", {}).get("gate_v3_app_candidate") != answer:
        raise LegacyStageError("Legacy parser validation does not preserve the saved answer")

    catalog = {row["turn_id"]: row["text"] for row in visible_prefix}
    complete_ids = [row["turn_id"] for row in visible_prefix]
    reviews = _reviews()
    clauses, provenance = [], []
    for side in ("doctor_condition", "agent_condition"):
        for index, old in enumerate(answer["conditions"][side]):
            status = old["status"]
            clause_hash = sha256_text(old["clause"])
            review = None
            if status == "supported":
                requirement_type = "presence"
                new_status = "supported"
            elif status == "not_required":
                requirement_type = "not_required"
                new_status = "not_required"
            elif status == "absent":
                review = reviews.get(clause_hash)
                if not review or review.get("requirement_type") != "absence":
                    raise MigrationReviewRequired(
                        "Legacy absent clause lacks an explicit hash-bound polarity review"
                    )
                if old.get("evidence"):
                    raise LegacyStageError("Confirmed absence has positive evidence")
                if old.get("inspected") != complete_ids:
                    raise LegacyStageError("Confirmed absence did not inspect the complete visible prefix")
                if answer.get("applicable") is not True:
                    raise LegacyStageError("Satisfied legacy absence conflicts with aggregate applicability")
                requirement_type = "absence"
                new_status = "absent"
            else:
                # An unmet/uncertain legacy row has no safe polarity migration.
                raise MigrationReviewRequired(
                    f"Legacy status {status!r} needs an explicit polarity review"
                )

            evidence_rows = old.get("evidence", [])
            evidence_quote = None
            if evidence_rows:
                first = evidence_rows[0]
                if first.get("turn_id") not in catalog or first.get("quote") not in catalog[first["turn_id"]]:
                    raise LegacyStageError("Legacy applicability evidence is not literal prefix evidence")
                evidence_quote = {"turn_id": first["turn_id"], "quote": first["quote"]}

            scope = (
                "complete_visible_prefix"
                if old.get("inspected") == complete_ids
                else "active_history"
            )
            clause = {
                "clause_id": f"{side}:{index}",
                "clause": old["clause"],
                "requirement_type": requirement_type,
                "status": new_status,
                "evidence_scope": scope,
                "evidence_quote": evidence_quote,
                "inspected_turn_ids": copy.deepcopy(old.get("inspected", [])),
                "satisfied": True,
            }
            clauses.append(clause)
            provenance.append({
                "clause_id": clause["clause_id"],
                "clause_sha256": clause_hash,
                "legacy_status": status,
                "new_requirement_type": requirement_type,
                "new_status": new_status,
                "review": copy.deepcopy(review),
                "transformation": "explicit_status_mapping" if review is None else "hash_bound_human_review",
            })

    return {"clauses": clauses, "applicable": True}, {
        "adapter_version": ADAPTER_VERSION,
        "adapter_sha256": adapter_sha256(),
        "migration_review_sha256": hashlib.sha256(REVIEW_PATH.read_bytes()).hexdigest(),
        "clauses": provenance,
    }
