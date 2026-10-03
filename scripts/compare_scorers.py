"""Score the same DEV (turn, rubric) units with every enabled score LLM and compare."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from src.common.config import IntegrationUnavailable
from src.common.io import read_jsonl
from src.data.schema import identity
from src.data.split import read_manifest
from src.judges.agent_state import resolve_include_initial_turn
from src.judges.llm import CachedChat, build_chat_backend
from src.judges.rubric_consolidation import write_json
from src.judges.rubric_scoring import ChatRubricScorer, JevRubricScorer, build_scoring_items, final_rubric_view
from src.judges.scorer_comparison import compare_scorers

ROOT = Path(__file__).resolve().parents[1]


def dev_only(config: dict, trajectories: list[dict]) -> list[dict]:
    data = config.get("data", {})
    if not data.get("split_manifest"):
        if config.get("mode") == "production":
            raise IntegrationUnavailable("Scorer selection must use DEV cases: set data.split_manifest/source_dataset")
        return trajectories
    manifest = read_manifest(data["split_manifest"])
    selected = []
    for trajectory in trajectories:
        member = manifest["members"].get(identity(data["source_dataset"], str(trajectory["case_id"])))
        if member is None or member["split"] == "eval":
            raise ValueError(f"Case {trajectory['case_id']} is held-out EVAL or unknown; scorer selection uses DEV")
        if member["split"] == "dev":
            selected.append(trajectory)
    return selected


def build_scorers(config: dict, chat_factory=None) -> tuple[list, dict]:
    """Enabled scorers plus {name: reason} for those that cannot run yet."""
    section = config["scorers"]
    system_prompt = (ROOT / section["prompt"]).read_text(encoding="utf-8")
    if section.get("prompt_addendum"):
        system_prompt += (ROOT / section["prompt_addendum"]).read_text(encoding="utf-8")
    chat_factory = chat_factory or (lambda settings, name: CachedChat(
        build_chat_backend(settings), ROOT / config["cache_dir"] / f"scorer_{name}.jsonl"))
    scorers, unavailable = [], {}
    for name, settings in sorted(section["candidates"].items()):
        if not settings.get("enabled"):
            continue
        if settings["backend"] == "jev":
            scorers.append(JevRubricScorer(name, settings))
            continue
        try:
            chat = chat_factory(settings, name)
        except IntegrationUnavailable as exc:
            unavailable[name] = str(exc)
            continue
        scorers.append(ChatRubricScorer(name, chat, system_prompt=system_prompt,
                                        prompt_version=section["prompt_version"], params=settings.get("params", {})))
    return scorers, unavailable


def run(config: dict, trajectories: list[dict], rubric_set: dict, output_dir: Path,
        human_rows=None, max_items=None, chat_factory=None, audit_size=50) -> dict:
    rubrics = [final_rubric_view(r) for r in rubric_set["rubrics"]]
    if not rubrics:
        raise ValueError("Rubric set is empty")
    include_initial_turn = resolve_include_initial_turn(config)  # METHOD_TBD, explicit in production
    items = [item for t in dev_only(config, trajectories) if t.get("trajectory_status") == "valid"
             for item in build_scoring_items(t, rubrics, include_initial_turn=include_initial_turn)]
    items = sorted(items, key=lambda i: i.item_id)[:max_items]
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    scorers, unavailable = build_scorers(config, chat_factory)
    for scorer in scorers:
        path = output_dir / f"scores_{scorer.name}.jsonl"
        rows = []
        try:
            for item in items:
                rows.append(scorer.score(item))
        except IntegrationUnavailable as exc:
            unavailable[scorer.name] = str(exc)
            continue
        path.write_text("\n".join(json.dumps(r.to_dict(), ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
        results[scorer.name] = rows
    human = {(row["item_id"], row["rubric_id"]): row["label"] for row in human_rows or []}
    report = compare_scorers(results, len(items) * len(rubrics), human or None) if results else {}
    report.update(unavailable_scorers=unavailable, items=len(items), rubrics=len(rubrics),
                  rubric_set_version=rubric_set.get("rubric_set_version"))
    # Random audit sample of units for human labelling (labels hidden from annotators).
    units = [(item, r["rubric_id"]) for item in items for r in rubrics]
    sample = random.Random(0).sample(units, min(audit_size, len(units)))
    (output_dir / "audit_units.jsonl").write_text("\n".join(json.dumps({
        "item_id": item.item_id, "rubric_id": rubric_id, "payload": item.payload(), "label": None,
    }, ensure_ascii=False) for item, rubric_id in sample) + "\n", encoding="utf-8")
    write_json(output_dir / "comparison.json", report)
    return report


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--trajectories", required=True)
    p.add_argument("--rubric-set", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--human-labels")
    p.add_argument("--max-items", type=int)
    a = p.parse_args()
    config = json.loads(Path(a.config).read_text(encoding="utf-8"))
    report = run(config, read_jsonl(Path(a.trajectories)), json.loads(Path(a.rubric_set).read_text(encoding="utf-8")),
                 Path(a.output_dir), read_jsonl(Path(a.human_labels)) if a.human_labels else None, a.max_items)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
