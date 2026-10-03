"""The majority mapping of the outcome judge for the pairs of an environment: policy majority-v1 applied to existing votes."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from src.agentclinic.outcome import MAJORITY_POLICY, DiagnosisJudge, normalize
from src.common.config import read_config
from src.common.io import read_jsonl
from src.judges.llm import openrouter_chat

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--votes", nargs=2, required=True)
    parser.add_argument("--labels-dir", required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    config, out = read_config(args.config), ROOT / args.output_dir
    target = out / "majority_mapping_v1.jsonl"
    if target.exists():
        raise SystemExit(f"{target} exists: the mapping is written once")
    runs = [{(r["reference"], r["predicted"]): r for r in read_jsonl(ROOT / path)} for path in args.votes]
    if set(runs[0]) != set(runs[1]):
        raise SystemExit("the two runs did not decide the same pairs")
    use = Counter()
    for name in ("before_rollouts.jsonl", "after_rollouts.jsonl"):
        for r in read_jsonl(ROOT / args.labels_dir / name):
            use[(normalize(r["gold"]), normalize(r["final_diagnosis"] or ""), "label")] += 1
    for t in read_jsonl(ROOT / args.base):
        for stated in (t.get("final_diagnosis"), t.get("agent_initial_position"), t.get("doctor_initial_claim")):
            use[(normalize(t["ground_truth"]["correct_diagnosis"]), normalize(stated or ""), "base")] += 1
    judge = DiagnosisJudge(openrouter_chat(config["models"]["judge"]["model_id"], None),
                           (ROOT / config["prompts"]["diagnosis_equivalence"]).read_text(encoding="utf-8"), params=config.get("diagnosis_judge_params"))

    def decide(key):
        first = runs[0][key]
        votes = [bool(runs[0][key]["equivalent"]), bool(runs[1][key]["equivalent"])]
        extra = 1 if votes[0] == votes[1] else 3   # three votes; five when the first two differ
        for _ in range(extra):
            for _ in range(3):
                vote = judge.vote(first["raw_reference"], first["raw_predicted"], cached=False)["equivalent"]
                if isinstance(vote, bool):
                    votes.append(vote)
                    break
            else:
                raise SystemExit(f"the judge gave no vote for {key}")
        return {"reference": key[0], "predicted": key[1], "equivalent": sum(votes) * 2 > len(votes), "source": "judge_majority", "policy": MAJORITY_POLICY,
                "votes": votes, "raw_reference": first["raw_reference"], "raw_predicted": first["raw_predicted"],
                "label_continuations": use[(key[0], key[1], "label")], "base_uses": use[(key[0], key[1], "base")]}

    with ThreadPoolExecutor(16) as pool:
        rows = list(pool.map(decide, sorted(runs[0])))
    out.mkdir(parents=True, exist_ok=True)
    (out / "votes.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    target.write_text("".join(json.dumps({k: r[k] for k in ("reference", "predicted", "equivalent", "source", "policy", "votes", "raw_reference", "raw_predicted")},
                                         ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    target.chmod(0o444)
    prompt = config["prompts"]["diagnosis_equivalence"]
    split = Counter("5 votes" if len(r["votes"]) == 5 else "3 votes, unanimous" if len(set(r["votes"])) == 1 else "3 votes, 2-1" for r in rows)
    weight = Counter()
    for r in rows:
        weight["5 votes" if len(r["votes"]) == 5 else "3 votes, unanimous" if len(set(r["votes"])) == 1 else "3 votes, 2-1"] += r["label_continuations"]
    summary = {"mapping": str(target.relative_to(ROOT)), "sha256": hashlib.sha256(target.read_bytes()).hexdigest(), "outcome_judge_policy": MAJORITY_POLICY,
               "policy": "votes of one judge (same model, prompt, decoding): the majority of three; when the first two votes differ, the majority of five. "
                         "No other model votes or breaks a tie. A pair first seen later is decided the same way (two calls; three more when they differ).",
               "judge": {"model": config["models"]["judge"]["model_id"], "prompt": prompt, "prompt_sha256": hashlib.sha256((ROOT / prompt).read_bytes()).hexdigest(),
                         "params": config.get("diagnosis_judge_params")},
               "vote_sources": {"vote 1": args.votes[0], "vote 2": args.votes[1], "further votes": "new calls of this script",
                                "note": "vote 1 was made with max_tokens 120, the others with the frozen params"},
               "pairs": len(rows), "equivalent": sum(r["equivalent"] for r in rows), "pairs_by_vote": dict(split), "label_continuations_by_vote": dict(weight),
               "decisions_differing_from_vote_2": sum(r["equivalent"] != r["votes"][1] for r in rows),
               "five_vote_pairs": [{k: r[k] for k in ("raw_reference", "raw_predicted", "votes", "equivalent", "label_continuations")} for r in rows if len(r["votes"]) == 5],
               "frozen": time.strftime("%Y-%m-%d %H:%M:%S"), "not_to_be_changed_after": "any DEV or TEST result of the clean run"}
    (out / "majority_mapping_v1.summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "five_vote_pairs"}, indent=2))
    for r in sorted(summary["five_vote_pairs"], key=lambda r: -r["label_continuations"])[:10]:
        print(r["label_continuations"], "|", r["raw_reference"], "<>", r["raw_predicted"][:70], "|", r["votes"], "->", r["equivalent"])


if __name__ == "__main__":
    main()
