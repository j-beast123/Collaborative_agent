"""RQ3 category x iteration data for a method whose rubrics are rebuilt in every iteration: a post-hoc, analysis-only
taxonomy of the iterations' rubrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from src.common.main_config import load_main_config
from src.evaluation.rq3 import bubble_rows
from src.judges.llm import CachedChat, ChatRequest, build_chat_backend, parse_json_object

ROOT = Path(__file__).resolve().parents[1]
PROMPT = "prompts/rq3_taxonomy.txt"
PROMPT_VERSION = "rq3-taxonomy-v1"


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def rubric_items(run: Path) -> list[dict]:
    """The consolidated rubrics of every iteration of the run that has them."""
    items = []
    for directory in sorted(run.glob("iter*"), key=lambda p: int(p.name[4:])):
        path = directory / "rubrics" / "consolidated_rubrics.json"
        if not path.is_file():
            continue
        iteration = int(directory.name[4:])
        for rubric in json.loads(path.read_text(encoding="utf-8"))["rubrics"]:
            items.append({"key": f"iter{iteration}:{rubric['category_id']}", "iteration": iteration,
                          "category_id": rubric["category_id"], "category_name": rubric.get("category_name"),
                          "name": rubric["name"], "applicability": rubric.get("applicability"), "scores": rubric.get("scores"),
                          "failure_count": int(rubric.get("failure_count") or 0)})
    return items


def build_taxonomy(items: list[dict], chat, params: dict, system_prompt: str) -> dict:
    """{themes, assignments}; ValueError unless every rubric is in exactly one existing theme."""
    shown = [{"key": i["key"], "iteration": i["iteration"], "name": i["name"], "applies_when": i["applicability"],
              "full_score_response": (i["scores"] or {}).get("2") if isinstance(i["scores"], dict) else i["scores"]} for i in items]
    result = chat.complete(ChatRequest(system_prompt, json.dumps({"rubrics": shown}, ensure_ascii=False), PROMPT_VERSION, params))
    parsed, _ = parse_json_object(result.text) if result.status == "ok" else (None, "call_failed")
    if not parsed or not isinstance(parsed.get("themes"), list) or not isinstance(parsed.get("assignments"), dict):
        raise ValueError(f"the taxonomy call did not return themes and assignments ({result.error or 'unparsable output'})")
    themes = {t["theme_id"]: t for t in parsed["themes"]}
    assigned = parsed["assignments"]
    keys = {i["key"] for i in items}
    problems = [f"not assigned: {sorted(keys - set(assigned))[:5]}"] if keys - set(assigned) else []
    problems += [f"unknown rubric keys: {sorted(set(assigned) - keys)[:5]}"] if set(assigned) - keys else []
    problems += [f"unknown themes: {sorted(set(assigned.values()) - set(themes))[:5]}"] if set(assigned.values()) - set(themes) else []
    if problems or len(themes) != len(parsed["themes"]):
        raise ValueError("the taxonomy is not a partition of the rubrics: " + "; ".join(problems or ["duplicate theme ids"]))
    return {"themes": [t for t in parsed["themes"] if t["theme_id"] in set(assigned.values())], "assignments": assigned}


def registry_of(items: list[dict], taxonomy: dict) -> list[dict]:
    """The themes as a category registry for bubble_rows: per iteration the failures consolidated into the theme's rubrics."""
    rows = {t["theme_id"]: {"category_id": t["theme_id"], "name": t["name"], "definition": t.get("definition"), "status": "analysis",
                            "frequency_history": {}, "rubrics": []} for t in taxonomy["themes"]}
    for item in items:
        row = rows[taxonomy["assignments"][item["key"]]]
        row["frequency_history"][str(item["iteration"])] = row["frequency_history"].get(str(item["iteration"]), 0) + item["failure_count"]
        row["rubrics"].append({"key": item["key"], "name": item["name"], "failure_count": item["failure_count"]})
    return list(rows.values())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-keywords", type=int, default=8, help="themes shown in the bubble data; the rest -> OTHER (visualization only)")
    args = parser.parse_args()
    run, out = ROOT / args.run, ROOT / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    items = rubric_items(run)
    if not items:
        raise SystemExit(f"{run} has no iter<i>/rubrics/consolidated_rubrics.json yet")
    inputs = digest([{k: i[k] for k in ("key", "name", "applicability", "scores")} for i in items])
    target = out / "taxonomy.json"
    known = json.loads(target.read_text(encoding="utf-8")) if target.is_file() else None
    if known and known["inputs_sha256"] == inputs:
        taxonomy = known
    else:
        config = json.loads((run / "resolved_config.json").read_text(encoding="utf-8")) if (run / "resolved_config.json").is_file() \
            else load_main_config(ROOT / "configs/serl_agentclinic_qwen4b_max7_train30.json", ROOT)
        manager = config["roles"]["rubric_manager"]
        chat = CachedChat(build_chat_backend(manager), out / "cache_taxonomy.jsonl")
        system_prompt = (ROOT / PROMPT).read_text(encoding="utf-8")
        built = build_taxonomy(items, chat, dict(manager["params"], max_tokens=4000), system_prompt)
        taxonomy = {"analysis_only": True, "never_used_for": "training, reward, rubric selection, filtering or weights",
                    "run": args.run, "iterations": sorted({i["iteration"] for i in items}), "inputs_sha256": inputs,
                    "source": "iter<i>/rubrics/consolidated_rubrics.json of the run (TRAIN failures of each iteration)",
                    "model": manager["model_id"], "prompt": PROMPT, "prompt_version": PROMPT_VERSION,
                    "prompt_sha256": hashlib.sha256(system_prompt.encode()).hexdigest(), "frozen": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "themes": built["themes"], "assignments": built["assignments"],
                    "rubrics": [{k: i[k] for k in ("key", "iteration", "category_id", "category_name", "name", "failure_count")} for i in items]}
        if known:
            (out / "history").mkdir(exist_ok=True)
            target.rename(out / "history" / f"taxonomy_{time.strftime('%Y%m%d-%H%M%S')}.json")
        target.write_text(json.dumps(taxonomy, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    registry = registry_of(items, taxonomy)
    (out / "registry.json").write_text(json.dumps(registry, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    bubble = bubble_rows(registry, top_k=args.top_keywords)
    (out / "rq3_bubble.json").write_text(json.dumps(bubble, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"rubrics": len(items), "iterations": taxonomy["iterations"], "themes": len(taxonomy["themes"]),
                      "bubble_rows": len(bubble), "output": args.output_dir}, indent=2))


if __name__ == "__main__":
    main()
