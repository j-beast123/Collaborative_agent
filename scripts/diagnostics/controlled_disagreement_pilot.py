"""Controlled interaction-failure pilot (diagnostic; not the method, not read by any run): does the original rubric branch work
on failures that the interaction induces? Qwen3-14B Agent; DEV and TEST are never read.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from scripts.diagnostics.train_pool_audit import POOL_ENV, collaboration, row_of, setup
from src.agentclinic.pipeline import Appender
from src.common.config import read_config
from src.common.io import read_jsonl

ROOT = Path(__file__).resolve().parents[2]
REPEATS = 3


def candidates(pool: list[dict]) -> tuple[list[str], list[str]]:
    first = {r["case_id"]: r for r in pool if r["rerun"] == 0}
    standalone = sorted(k for k, r in first.items() if r["standalone_correct"])
    return standalone, [k for k in standalone if not first[k]["doctor_initial_correct"]]


def screen(args) -> None:
    config, cases, histories, clinic = setup(args, with_model=True)
    out = ROOT / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    pool = read_jsonl(ROOT / args.pool_audit / "pool_collaborations.jsonl")
    _, usable = candidates(pool)
    rows = Appender(out / "collaborations.jsonl", key=lambda r: (r["case_id"], r["rerun"]))
    for r in pool:  # the pool audit's collaborations of these cases, as they are
        if r["case_id"] in usable and (r["case_id"], r["rerun"]) not in rows.keys:
            rows.add(dict(r, source=f"{args.pool_audit}/pool_collaborations.jsonl"))

    def one(job):
        case_id, index = job
        if (case_id, index) not in rows.keys:
            rows.add(row_of(collaboration(clinic, cases[case_id], histories[case_id], f"{case_id}:14b:pool:{index}"), "generated", index))

    with clinic.generation(), ThreadPoolExecutor(config["rollout"]["workers"]) as pool_:
        list(pool_.map(one, [(k, i) for k in usable for i in range(REPEATS)]))


def classes(rows: list[dict]) -> dict:
    result = {}
    for case_id in sorted({r["case_id"] for r in rows}):
        mine = [r for r in rows if r["case_id"] == case_id and r["rerun"] < REPEATS]
        success = sum(r["collaboration_correct"] for r in mine)
        result[case_id] = {"collaborations": len(mine), "success": success,
                           "class": "RESISTS" if success == REPEATS else "INTERACTION_INDUCED_FAILURE" if len(mine) - success >= 2 else "UNSTABLE"}
    return result


def serl(args) -> None:
    import src.agentclinic.runtime as runtime
    import src.agentclinic.selection as selection
    from scripts import build_rubrics

    config, cases, histories, _ = setup(args, with_model=False)
    out = ROOT / args.output_dir
    rows = read_jsonl(out / "collaborations.jsonl")
    induced = [k for k, v in classes(rows).items() if v["class"] == "INTERACTION_INDUCED_FAILURE"]
    run = out / "serl_run"
    rubrics = run / "iter0/rubrics"
    rubrics.mkdir(parents=True, exist_ok=True)
    trajectories = []
    for case_id in induced:
        failed = sorted((r for r in rows if r["case_id"] == case_id and not r["collaboration_correct"]), key=lambda r: r["rerun"])
        trajectories.append(failed[0]["trajectory"])
    (run / "base").mkdir(exist_ok=True)
    (run / "base/trajectories.jsonl").write_text("".join(json.dumps(t, ensure_ascii=False) + "\n" for t in trajectories), encoding="utf-8")
    # data.split_manifest of the run is already the case-level split of all of agentclinic_main_v2 (TRAIN90): kept as it is
    config = dict(config, output_dir=str(run.relative_to(ROOT)),
                  diagnostic_pilot={"what": "original rubric branch on interaction-induced failures; widened: the case table (TRAIN90 cases)",
                                    "induced_failure_cases": induced})
    (run / "resolved_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=1), encoding="utf-8")
    # this process only: the clinic and the case table also hold the TRAIN90 cases (TRAIN only)
    build_clinic, eligible = runtime.build_clinic, selection.split_eligible_cases

    def clinic_with_pool(*a, **k):
        clinic = build_clinic(*a, **k)
        clinic.cases.update(cases)
        return clinic

    def eligible_with_pool(source, root, **k):
        found = eligible(source, root, **k)
        return {**found, **{key: c for key, c in cases.items() if key not in found}}

    runtime.build_clinic, selection.split_eligible_cases = clinic_with_pool, eligible_with_pool
    rerun_fn, intervene_fn = build_rubrics.clinic_rerun(config, 0, rubrics)[:2]
    observe_fn = build_rubrics.gap_observer(config, 0, run / "base/trajectories.jsonl", rubrics)
    summary = build_rubrics.run(config, run / "base/trajectories.jsonl", rubrics, 0, rerun_fn=rerun_fn, intervene_fn=intervene_fn,
                                observe_fn=observe_fn)
    print(json.dumps(summary, ensure_ascii=False, indent=1))


def report(args) -> None:
    out = ROOT / args.output_dir
    pool = read_jsonl(ROOT / args.pool_audit / "pool_collaborations.jsonl")
    standalone, usable = candidates(pool)
    rows = read_jsonl(out / "collaborations.jsonl")
    cls = classes(rows)
    summary_path = out / "serl_run/iter0/rubrics/summary.json"
    validations = [json.loads(line) for line in (out / "serl_run/iter0/rubrics/validations.jsonl").read_text().splitlines() if line.strip()] \
        if (out / "serl_run/iter0/rubrics/validations.jsonl").exists() else []
    result = {"standalone_correct_candidates": len(standalone),
              "usable_wrong_doctor_d1": len(usable), "doctor_d1_correct_not_usable": len(standalone) - len(usable),
              "classes": dict(Counter(v["class"] for v in cls.values())),
              "per_case": {k[-4:]: v for k, v in cls.items()},
              "interaction_induced_case_ids": [k[-4:] for k, v in cls.items() if v["class"] == "INTERACTION_INDUCED_FAILURE"],
              "serl_summary": json.loads(summary_path.read_text()) if summary_path.exists() else None,
              "candidates": [{"case": v["source_trajectory_id"].split(":")[0][-4:], "status": v["validation_status"], "successes": v["successes"],
                              "attempt_scores": [a.get("label") for a in (v.get("intervention") or {}).get("attempts", [])],
                              "rerun_final_diagnoses": [x.get("final_diagnosis") for x in v["reruns"]]} for v in validations],
              "note": "diagnostic; strict majority-v1; original rubric branch unchanged"}
    (out / "controlled_disagreement.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=("screen", "serl", "report"), required=True)
    parser.add_argument("--pool-audit", required=True)
    parser.add_argument("--native-run", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    {"screen": screen, "serl": serl, "report": report}[args.stage](args)


if __name__ == "__main__":
    main()
