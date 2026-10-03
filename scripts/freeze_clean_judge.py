"""Freeze the automatic diagnosis judge of the clean environment from its agreement with the human adjudication."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--comparison", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    comparison = json.loads((ROOT / args.comparison).read_text(encoding="utf-8"))
    path = ROOT / args.config
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("diagnosis_judge_frozen"):
        raise SystemExit("the config already names a frozen judge")
    prompt = config["prompts"]["diagnosis_equivalence"]
    eligible = {name: c for name, c in comparison["candidates"].items() if c["prompt"] == prompt}
    chosen = max(eligible, key=lambda name: eligible[name]["agreement"])
    judge = eligible[chosen]
    config.setdefault("models", {})["judge"] = {"model_id": judge["model"], "revision": "openrouter-unpinned"}
    config["diagnosis_judge_params"] = judge["params"]
    record = {"chosen": chosen, "model": judge["model"], "prompt": prompt, "prompt_sha256": hashlib.sha256((ROOT / prompt).read_bytes()).hexdigest(),
              "params": judge["params"], "agreement_with_human_on_the_selection_set": judge["agreement"], "selection_pairs": comparison["pairs"],
              "split_manifest_sha256": comparison.get("split_manifest_sha256"), "population_note": comparison.get("population_note"),
              "heldout_agreement": "judge_heldout_agreement.json (evaluated once, after this choice)",
              "human_equivalent_judge_not": judge["human_equivalent_judge_not"], "human_not_judge_equivalent": judge["human_not_judge_equivalent"],
              "all_candidates": {name: c["agreement"] for name, c in comparison["candidates"].items()}, "frozen": time.strftime("%Y-%m-%d %H:%M:%S"),
              "rule": "highest agreement with the human adjudication on the judge-selection set among the candidates with the clean prompt; "
                      "fixed before the held-out set is evaluated and before any DEV or TEST result of the clean run"}
    config["diagnosis_judge_frozen"] = record
    path.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (ROOT / args.comparison).with_name("judge_frozen.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
