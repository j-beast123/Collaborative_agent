"""Shards of the Effectiveness label work: several processes (one per GPU) label disjoint sets of histories, and their
files are merged back into the label directory of the single run.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path

from src.common.io import read_jsonl

from .candidates import CANDIDATES
from .outcome import normalize

KEYS = {
    "candidate_attempts.jsonl": lambda r: (r["history_id"], r["candidate_id"], r.get("attempt", 1)),
    "candidates.jsonl": lambda r: (r["history_id"], r["candidate_id"]),
    "candidate_compliance.jsonl": lambda r: (r["history_id"], r["candidate_id"]),
    "before_rollouts.jsonl": lambda r: (r["history_id"], r["rollout_index"]),
    "after_rollouts.jsonl": lambda r: (r["history_id"], r["candidate_id"], r.get("candidate_attempt", 1), r["rollout_index"]),
    "labels.jsonl": lambda r: (r["history_id"], r["response_id"]),
}
SORTED_BY_KEY = ("candidate_attempts.jsonl", "candidates.jsonl", "candidate_compliance.jsonl")  # rewritten sorted by the single run
JUDGE_CACHE = "cache_diagnosis_judge.jsonl"
MARKER = "shard.json"


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def assign(histories: list[dict], shards: int) -> dict[str, int]:
    """history_id -> shard index (0-based)."""
    if shards < 1:
        raise ValueError("the number of shards must be at least 1")
    last = max(int(h["intervention_turn"][1:]) for h in histories)
    units = defaultdict(list)
    for row in histories:
        units[normalize(row["gold"]) or row["case_id"]].append(row)

    def weight(members: list[dict]) -> int:  # turns a continuation can still run from each history
        return sum(last - int(h["intervention_turn"][1:]) + 1 for h in members)

    load, shard_of = [0] * shards, {}
    for _, members in sorted(units.items(), key=lambda item: (-weight(item[1]), item[0])):
        index = min(range(shards), key=lambda i: (load[i], i))
        load[index] += weight(members)
        for row in members:
            if row["history_id"] in shard_of:
                raise ValueError(f"history {row['history_id']} appears twice")
            shard_of[row["history_id"]] = index
    return shard_of


def assignment_digest(shard_of: dict[str, int], shards: int) -> str:
    return hashlib.sha256(json.dumps([shards, sorted(shard_of.items())]).encode()).hexdigest()


def shard_dir(labels_dir: Path, index: int, shards: int) -> Path:
    return Path(labels_dir) / "shards" / f"n{shards:02d}" / f"shard{index:02d}"


def write_rows(path: Path, rows: list[dict]) -> None:
    Path(path).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def seed(labels_dir: Path, index: int, shards: int, shard_of: dict[str, int], extra: dict | None = None) -> Path:
    """The directory of shard `index`, created with the rows the label directory has for its histories. A directory that
    was seeded before is checked (same assignment) and left as it is."""
    labels_dir, target = Path(labels_dir), shard_dir(labels_dir, index, shards)
    digest = assignment_digest(shard_of, shards)
    mine = sorted(h for h, i in shard_of.items() if i == index)
    marker = target / MARKER
    if marker.is_file():
        known = json.loads(marker.read_text(encoding="utf-8"))
        if (known["assignment_sha256"], known["shard"], known["shards"]) != (digest, index, shards):
            raise ValueError(f"{target} was made for another assignment of histories")
        return target
    target.mkdir(parents=True, exist_ok=True)
    seeded = {}
    for name, key in KEYS.items():
        source = labels_dir / name
        if not source.is_file():
            continue
        rows = [r for r in read_jsonl(source) if shard_of.get(r["history_id"]) == index]
        write_rows(target / name, rows)
        seeded[name] = {"source_sha256": sha(source), "rows": len(rows)}
    if (labels_dir / JUDGE_CACHE).is_file():
        shutil.copyfile(labels_dir / JUDGE_CACHE, target / JUDGE_CACHE)
        seeded[JUDGE_CACHE] = {"source_sha256": sha(labels_dir / JUDGE_CACHE), "rows": sum(1 for _ in (target / JUDGE_CACHE).open())}
    marker.write_text(json.dumps({"shard": index, "shards": shards, "assignment_sha256": digest, "histories": len(mine),
                                  "history_ids": mine, "seeded_from": seeded, "seeded_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                                  **(extra or {})}, indent=2) + "\n", encoding="utf-8")
    return target


def progress(directory: Path, histories: list[dict], names: tuple, k: int, require_compliance: bool = True) -> dict:
    """Rows of `directory` for `histories` against what the label run must hold when it is complete."""
    ids = {h["history_id"] for h in histories}
    rows = {name: [r for r in read_jsonl(Path(directory) / name, missing_ok=True) if r["history_id"] in ids] for name in KEYS}
    duplicates = {name: sum(n - 1 for n in Counter(map(key, rows[name])).values() if n > 1) for name, key in KEYS.items()}
    chosen = {(c["history_id"], c["candidate_id"]): c.get("attempt", 1) for c in rows["candidates.jsonl"]}
    judged = {(c["history_id"], c["candidate_id"]): bool(c["behavior_compliant"]) for c in rows["candidate_compliance.jsonl"]}
    expected = {(h, name) for h in ids for name in names}
    eligible = {pair for pair in expected if pair in chosen and (judged.get(pair) or not require_compliance)}

    def done(row: dict) -> bool:
        return row["status"] == "completed" and row["success"] is not None

    before = Counter(r["history_id"] for r in rows["before_rollouts.jsonl"] if done(r))
    after = Counter((r["history_id"], r["candidate_id"]) for r in rows["after_rollouts.jsonl"]
                    if done(r) and r.get("candidate_attempt", 1) == chosen.get((r["history_id"], r["candidate_id"])))
    labels = {(r["history_id"], r["response_id"]) for r in rows["labels.jsonl"]}
    missing = {"candidates": sorted(expected - (set(chosen) & set(judged))),
               "before": sorted(h for h in ids if before[h] != k),
               "after": sorted(pair for pair in eligible if after[pair] != k),
               "labels": sorted(eligible - labels)}
    failed = sum(not done(r) for name in ("before_rollouts.jsonl", "after_rollouts.jsonl") for r in rows[name])
    return {"histories": len(ids), "candidate_pairs": len(set(chosen) & set(judged)), "candidate_pairs_expected": len(expected),
            "compliant_pairs": len(eligible), "before": sum(min(n, k) for n in before.values()), "before_expected": len(ids) * k,
            "after": sum(min(after[pair], k) for pair in eligible), "after_expected_for_the_compliant_pairs_so_far": len(eligible) * k,
            "labels": len(labels & eligible), "continuation_rows": len(rows["before_rollouts.jsonl"]) + len(rows["after_rollouts.jsonl"]),
            "continuations_not_completed": failed, "duplicate_keys": {n: d for n, d in duplicates.items() if d},
            "missing": {name: len(items) for name, items in missing.items()},
            "missing_examples": {name: [list(i) if isinstance(i, tuple) else i for i in items[:5]] for name, items in missing.items() if items},
            "complete": not any(missing.values()) and not any(duplicates.values())}


def merge(labels_dir: Path, shards: int, histories: list[dict], names: tuple, k: int, *, require_compliance: bool = True,
          dry_run: bool = False) -> dict:
    """Put the shard files back into the label directory. Raises ValueError with every problem found; with dry_run the
    checks run and nothing is written."""
    labels_dir = Path(labels_dir)
    shard_of = assign(histories, shards)
    digest = assignment_digest(shard_of, shards)
    order = {h["history_id"]: i for i, h in enumerate(histories)}
    candidate_order = {name: i for i, name in enumerate(CANDIDATES)}
    problems, rows, markers = [], {name: [] for name in KEYS}, []
    for index in range(shards):
        directory = shard_dir(labels_dir, index, shards)
        if not (directory / MARKER).is_file():
            problems.append(f"shard {index}: {directory} was never started")
            continue
        marker = json.loads((directory / MARKER).read_text(encoding="utf-8"))
        markers.append(marker)
        if marker["assignment_sha256"] != digest:
            problems.append(f"shard {index}: made for another assignment of histories")
        for name in KEYS:
            for row in read_jsonl(directory / name, missing_ok=True):
                if shard_of.get(row["history_id"]) != index:
                    problems.append(f"shard {index}: {name} has a row of {row['history_id']}, which is not its history")
                rows[name].append(row)
        for name, info in marker["seeded_from"].items():  # the label directory must be what the shards started from
            if not (labels_dir / name).is_file() or sha(labels_dir / name) != info["source_sha256"]:
                problems.append(f"shard {index}: {name} of the label directory changed after the shard was seeded")
    existing = {name: read_jsonl(labels_dir / name, missing_ok=True) for name in KEYS}
    for name, key in KEYS.items():
        counts = Counter(map(key, rows[name]))
        if any(n > 1 for n in counts.values()):
            problems.append(f"{name}: {sum(n - 1 for n in counts.values() if n > 1)} duplicate keys")
        merged = {key(r): r for r in rows[name]}
        lost = [key(r) for r in existing[name] if merged.get(key(r)) != r]
        if lost:
            problems.append(f"{name}: {len(lost)} rows of the label directory are missing or changed in the shards, e.g. {lost[:2]}")
    if not problems:
        union = labels_dir / "shards" / f"n{shards:02d}" / "merge_staging"
        if union.exists():
            shutil.rmtree(union)
        union.mkdir(parents=True)
        for name, key in KEYS.items():
            had = {key(r) for r in existing[name]}
            new = [r for r in rows[name] if key(r) not in had]
            if name in SORTED_BY_KEY:  # the single run writes these sorted by (history, candidate[, attempt])
                final = sorted(rows[name], key=key)
            elif name == "labels.jsonl":  # the single run appends labels history by history, candidate by candidate
                final = existing[name] + sorted(new, key=lambda r: (order[r["history_id"]], candidate_order[r["response_id"]]))
            else:  # continuations are appended as they finish; here: by history, candidate, rollout index
                final = existing[name] + sorted(new, key=lambda r: (order[r["history_id"]], candidate_order.get(r.get("candidate_id"), -1),
                                                                    r.get("candidate_attempt", 1), r["rollout_index"]))
            write_rows(union / name, final)
        state = progress(union, histories, names, k, require_compliance)
        if not state["complete"]:
            problems.append(f"incomplete: missing {state['missing']}, duplicate keys {state['duplicate_keys']}, e.g. {state['missing_examples']}")
    if problems:
        raise ValueError("the shards cannot be merged:\n  " + "\n  ".join(problems[:40]))
    # the diagnosis-judge cache: the lines that were there, then the decisions the shards added (a key is kept once)
    cache_lines = (labels_dir / JUDGE_CACHE).read_text(encoding="utf-8").splitlines() if (labels_dir / JUDGE_CACHE).is_file() else []
    seen = {json.loads(line)["key"]: json.loads(line)["result"].get("text") for line in cache_lines if line.strip()}
    conflicts = 0
    for index in range(shards):
        path = shard_dir(labels_dir, index, shards) / JUDGE_CACHE
        for line in path.read_text(encoding="utf-8").splitlines() if path.is_file() else []:
            if not line.strip():
                continue
            entry = json.loads(line)
            if entry["key"] not in seen:
                seen[entry["key"]] = entry["result"].get("text")
                cache_lines.append(line)
            elif seen[entry["key"]] != entry["result"].get("text"):
                conflicts += 1
    (union / JUDGE_CACHE).write_text("".join(line + "\n" for line in cache_lines if line.strip()), encoding="utf-8")
    known = {r["history_id"]: r for r in read_jsonl(labels_dir / "histories_turns.jsonl", missing_ok=True)}
    write_rows(union / "histories_turns.jsonl", sorted({**known, **{h["history_id"]: h for h in histories}}.values(), key=lambda r: r["history_id"]))
    result = {"shards": shards, "assignment_sha256": digest, "histories": len(histories), "progress": state,
              "histories_by_shard": [m["histories"] for m in markers], "judge_cache_conflicting_keys": conflicts,
              "rows_before_merge": {name: len(existing[name]) for name in KEYS},
              "files": {p.name: {"rows": sum(1 for _ in p.open(encoding="utf-8")), "sha256": sha(p)} for p in sorted(union.glob("*.jsonl"))}}
    if dry_run:
        shutil.rmtree(union)
        return dict(result, written=False)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    backup = labels_dir / f"pre_shard_merge_{stamp}"
    backup.mkdir()
    for path in sorted(union.glob("*.jsonl")):
        if (labels_dir / path.name).is_file():
            shutil.copy2(labels_dir / path.name, backup / path.name)
    for path in sorted(union.glob("*.jsonl")):
        path.replace(labels_dir / path.name)
    union.rmdir()
    result.update(written=True, merged_at=stamp, previous_files=str(backup.name))
    (labels_dir / "shard_merge_manifest.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result
