"""Deterministic original-case grouping, explicit aliases, immutable manifests."""
from dataclasses import asdict
from pathlib import Path
import json
import math
import os
import tempfile

from .schema import digest, identity

SPLITS = ("train", "dev", "eval")
ALGORITHM = "sha256-group-order-v1"
# A manifest that records assignments made elsewhere (e.g. src/agentclinic/selection.py). It is checked
# like any other manifest, except that it cannot be regenerated from the cases by make_manifest.
EXTERNAL_ASSIGNMENT = "external-assignment-v1"


def _check_proportions(seed, proportions):
    if type(seed) is not int or not isinstance(proportions, dict) or set(proportions) != set(SPLITS):
        raise ValueError("Explicit integer seed and train/dev/eval proportions required")
    values = proportions.values()
    if any(type(v) not in (float, int) or not math.isfinite(v) or v <= 0 for v in values):
        raise ValueError("Proportions must be finite, positive and sum to one")
    if not math.isclose(sum(values), 1):
        raise ValueError("Proportions must be finite, positive and sum to one")


def make_manifest(cases, *, seed, proportions, duplicate_groups=()):
    cases = list(cases)
    _check_proportions(seed, proportions)
    keys = [c.key for c in cases]
    if not cases or len(set(keys)) != len(keys):
        raise ValueError("Empty input or duplicate case identity")
    if len({c.synthetic for c in cases}) != 1:
        raise ValueError("Mixed synthetic and real cases")

    # Union-find over original-case identities; explicit aliases merge groups.
    parents = {c.original_key: c.original_key for c in cases}

    def find(key):
        while parents[key] != key:
            key = parents[key]
        return key

    aliases = []
    for group in duplicate_groups:
        group = sorted(set(group))
        if len(group) < 2 or any(k not in parents for k in group):
            raise ValueError("Alias groups require at least two known original identities")
        aliases.append(group)
        for key in group[1:]:
            a, b = find(group[0]), find(key)
            parents[max(a, b)] = min(a, b)

    groups = sorted({find(k) for k in parents}, key=lambda k: (digest([seed, k]), k))
    n = len(groups)
    # Largest-remainder allocation of whole groups; ties follow SPLITS order.
    counts = {s: int(n * proportions[s]) for s in SPLITS}
    remaining = n - sum(counts.values())
    order = sorted(SPLITS, key=lambda s: (-(n * proportions[s] - counts[s]), SPLITS.index(s)))
    for s in order[:remaining]:
        counts[s] += 1
    if any(v == 0 for v in counts.values()):
        raise ValueError("Insufficient independent groups for the requested proportions")

    assignments = {}
    start = 0
    for s in SPLITS:
        assignments.update({g: s for g in groups[start:start + counts[s]]})
        start += counts[s]

    ordered = sorted(cases, key=lambda c: c.key)
    members = {
        c.key: {
            "original_key": c.original_key,
            "group": find(c.original_key),
            "split": assignments[find(c.original_key)],
            "dataset_version": c.dataset_version,
            "synthetic": c.synthetic,
        }
        for c in ordered
    }
    payload = {
        "schema_version": 1,
        "algorithm": ALGORITHM,
        "seed": seed,
        "proportions": proportions,
        "source_fingerprint": digest([asdict(c) for c in ordered]),
        "duplicate_groups": sorted(aliases),
        "deduplication_limitations": "Only explicit aliases are resolved; unreviewed duplicates may remain.",
        "group_counts": counts,
        "members": members,
    }
    payload["manifest_hash"] = digest(payload)
    return payload


def _check_member(key, row):
    try:
        identities = (json.loads(key), json.loads(row["original_key"]), json.loads(row["group"]))
        for ident in identities:
            if not isinstance(ident, list) or len(ident) != 2 or any(not isinstance(v, str) or not v for v in ident):
                raise ValueError("Invalid namespaced identity")
        if identities[0][0] != identities[1][0]:
            raise ValueError("Invalid namespaced identity")
        if not isinstance(row["dataset_version"], str) or not row["dataset_version"]:
            raise ValueError("Missing dataset version")
        if row["split"] not in SPLITS or type(row["synthetic"]) is not bool:
            raise ValueError("Invalid split/provenance")
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("Invalid manifest member") from exc


def validate_manifest(manifest, cases=None):
    payload = dict(manifest)
    claimed = payload.pop("manifest_hash", None)
    if claimed != digest(payload) or payload.get("algorithm") not in (ALGORITHM, EXTERNAL_ASSIGNMENT) \
            or payload.get("schema_version") != 1:
        raise ValueError("Manifest hash/version mismatch")
    try:
        _check_proportions(payload.get("seed"), payload.get("proportions"))
    except ValueError as exc:
        raise ValueError("Invalid manifest seed/proportions") from exc
    counts = payload.get("group_counts", {})
    if set(counts) != set(SPLITS) or any(type(v) is not int or v <= 0 for v in counts.values()):
        raise ValueError("Manifest requires nonempty train/dev/eval groups")
    members = payload.get("members", {})
    if not isinstance(members, dict) or not members:
        raise ValueError("Empty manifest membership")
    for key, row in members.items():
        _check_member(key, row)

    groups, originals, provenance = {}, {}, set()
    for row in members.values():
        provenance.add(row["synthetic"])
        if row["group"] in groups and groups[row["group"]] != row["split"]:
            raise ValueError("Duplicate group crosses splits")
        if row["original_key"] in originals and originals[row["original_key"]] != row["group"]:
            raise ValueError("Original case crosses groups")
        groups[row["group"]] = row["split"]
        originals[row["original_key"]] = row["group"]
    split_counts = {s: list(groups.values()).count(s) for s in SPLITS}
    if len(provenance) != 1 or split_counts != counts:
        raise ValueError("Manifest group counts/provenance mismatch")
    if any(group not in originals for group in groups):
        raise ValueError("Unknown group representative")
    aliases = payload.get("duplicate_groups")
    if not isinstance(aliases, list):
        raise ValueError("Manifest duplicate_groups must be a list")
    for alias in aliases:
        if any(k not in originals for k in alias) or len({originals[k] for k in alias}) != 1:
            raise ValueError("Alias group is not preserved")

    if cases is not None:
        if payload["algorithm"] == EXTERNAL_ASSIGNMENT:
            raise ValueError("An external-assignment manifest cannot be regenerated from cases")
        expected = make_manifest(
            cases, seed=payload["seed"], proportions=payload["proportions"], duplicate_groups=aliases
        )
        if expected != manifest:
            raise ValueError("Manifest does not match source cases/configuration")
    return manifest


def write_manifest(path, manifest):
    validate_manifest(manifest)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if read_manifest(path) != manifest:
            raise ValueError("Refusing to replace an existing split manifest")
        return
    fd, temporary = tempfile.mkstemp(dir=str(path.parent), prefix=".manifest-")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(manifest, f, sort_keys=True, indent=2)
            f.flush()
            os.fsync(f.fileno())
        try:
            # Hard link: atomic create that fails if another writer got there first.
            os.link(temporary, str(path))
        except FileExistsError:
            if read_manifest(path) != manifest:
                raise ValueError("Concurrent manifest differs; refusing overwrite")
    finally:
        os.unlink(temporary)


def read_manifest(path, cases=None):
    return validate_manifest(json.loads(Path(path).read_text()), cases)


def validate_lineage(records, manifest):
    validate_manifest(manifest)
    records = list(records)
    by_id = {r.record_id: r for r in records}
    if len(by_id) != len(records) or any(not r.record_id for r in records):
        raise ValueError("Duplicate/empty derived record identity")
    for r in records:
        member = manifest["members"].get(identity(r.source_dataset, r.case_id))
        if (
            member is None
            or member["original_key"] != identity(r.source_dataset, r.original_case_id)
            or member["split"] != r.split
            or type(r.synthetic) is not bool
            or member["synthetic"] != r.synthetic
        ):
            raise ValueError("Derived record violates original-case split/provenance")
        seen = {r.record_id}
        current = r
        while current.parent_record_id is not None:
            parent = by_id.get(current.parent_record_id)
            if parent is None or parent.record_id in seen:
                raise ValueError("Unknown or cyclic parent lineage")
            if (parent.source_dataset, parent.original_case_id, parent.split) != (r.source_dataset, r.original_case_id, r.split):
                raise ValueError("Parent belongs to a different original case/split")
            seen.add(parent.record_id)
            current = parent
