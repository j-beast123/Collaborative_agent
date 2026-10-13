"""Shards of the Effectiveness label run: every history in one shard, written work reused, and a merge that gives the
files of the single run or refuses."""
import json

import pytest

from src.agentclinic.pipeline import build_labels, generate_candidates, turn_histories
from src.agentclinic.protocol import continue_dialogue
from src.agentclinic.shards import KEYS, MARKER, assign, merge, progress, seed, shard_dir
from src.agentclinic.trajectory import collaborate
from src.common.io import read_jsonl
from test_agentclinic import GOLD, cases, compliance_judge, diagnosis_judge  # noqa: F401 (cases is a fixture)
from test_protocol_max7 import MAX7, runtime

NAMES = ("evidence_grounded_maintain", "unsupported_maintain")
K = 3


def environment(cases, tmp_path):
    golds = [GOLD, "Diagnosis B", "Diagnosis C", GOLD, "Diagnosis E"]  # cases 0 and 3 have the same correct diagnosis
    by_id = {c["case_id"]: dict(c, split="train" if i < 3 else "dev", gold=golds[i]) for i, c in enumerate(cases[:5])}
    calls = []

    def transition(said):
        user = calls[-1][1]
        if "antibody result" in user or said >= 3:
            return json.dumps({"action": "finalize", "final_diagnosis": GOLD if "antibody result" in user else "Viral illness"})
        return json.dumps({"action": "continue", "provisional_assessment": "Viral illness", "message_to_agent": "Tell me more."})

    clinic = runtime(0, calls, transition)
    plain = clinic.agent

    def agent(system, user):
        if "stating the finding itself" in user:
            return "Current assessment: MG\nResponse: The antibody result is elevated, which supports MG."
        if "Reply in at most two sentences" in user:
            return "Current assessment: MG\nResponse: I do not agree."
        return plain(system, user)

    clinic.agent = clinic.agent_greedy = agent
    judge, _ = diagnosis_judge(tmp_path / "base_judge.jsonl")
    trajectories = []
    for case in by_id.values():
        d1 = continue_dialogue(case, [], clinic, spec=MAX7)["dialogue"][:1]
        trajectories.append(collaborate(case, d1, clinic, judge, trajectory_id=f"{case['case_id']}:A1:base", spec=MAX7,
                                        standalone="MG", history_id=f"{case['case_id']}:A1"))
    return by_id, turn_histories(trajectories), clinic, calls


def run(by_id, histories, clinic, out):
    """What scripts/run_agentclinic_turns.py does for the stages candidates and labels in the directory `out`."""
    out.mkdir(parents=True, exist_ok=True)
    judge, _ = diagnosis_judge(out / "cache_diagnosis_judge.jsonl")
    generate_candidates(by_id, histories, clinic, compliance_judge(), out, workers=2, max_attempts=1, stage=None, names=NAMES)
    return build_labels(by_id, histories, clinic, judge, out, k=K, workers=2, require_compliance=True, stage=None, spec=MAX7,
                        metadata={"version": "t", "k": K, "split": "mixed", "continuation_protocol_hash": "h"})


def run_shards(by_id, histories, clinic, labels, shards):
    shard_of = assign(histories, shards)
    for index in range(shards):
        mine = [h for h in histories if shard_of[h["history_id"]] == index]
        run(by_id, mine, clinic, seed(labels, index, shards, shard_of))
    return shard_of


def without_order(rows):
    return sorted(json.dumps(r, sort_keys=True) for r in rows)


def test_every_history_is_in_exactly_one_shard_and_the_assignment_is_deterministic(cases, tmp_path):
    by_id, histories, _, _ = environment(cases, tmp_path)
    for shards in (1, 2, 3, 4):
        shard_of = assign(histories, shards)
        assert sorted(shard_of) == sorted(h["history_id"] for h in histories) and set(shard_of.values()) <= set(range(shards))
        assert shard_of == assign(list(reversed(histories)), shards)  # the order of the input does not matter
        by_case, by_gold = {}, {}
        for h in histories:
            by_case.setdefault(h["case_id"], set()).add(shard_of[h["history_id"]])
            by_gold.setdefault(h["gold"], set()).add(shard_of[h["history_id"]])
        assert all(len(v) == 1 for v in by_case.values())  # all turns of a case
        assert all(len(v) == 1 for v in by_gold.values())  # and all cases with the same correct diagnosis
        assert {by_id[h["case_id"]]["split"] for h in histories} == {"train", "dev"}  # the split of a case is not touched
    assert len(set(assign(histories, 3).values())) == 3
    with pytest.raises(ValueError):
        assign(histories + histories[:1], 2)


def test_merged_shards_are_the_files_of_a_single_run_and_reuse_what_was_written(cases, tmp_path):
    by_id, histories, clinic, calls = environment(cases, tmp_path)
    single = tmp_path / "single"
    first_case = [h for h in histories if h["case_id"] == histories[0]["case_id"]]
    run(by_id, first_case, clinic, single)  # the "smoke": one case is labelled before the full run
    labels = tmp_path / "labels"
    labels.mkdir()
    for path in single.iterdir():
        (labels / path.name).write_bytes(path.read_bytes())
    (labels / "histories_turns.jsonl").write_text("".join(json.dumps(h) + "\n" for h in sorted(first_case, key=lambda r: r["history_id"])))
    built = run(by_id, histories, clinic, single)  # the single full run continues in its directory
    assert built["incomplete_pairs"] == 0

    before_calls = len(calls)
    shard_of = run_shards(by_id, histories, clinic, labels, 3)
    seeded = [h for h in first_case]
    owner = shard_dir(labels, shard_of[seeded[0]["history_id"]], 3)
    kept = [r for r in read_jsonl(owner / "before_rollouts.jsonl") if r["history_id"] == seeded[0]["history_id"]]
    assert kept == [r for r in read_jsonl(labels / "before_rollouts.jsonl") if r["history_id"] == seeded[0]["history_id"]]  # not made again
    assert len(calls) > before_calls
    for index in range(3):  # shard by shard: BEFORE is K continuations per history, made once, shared by its candidates
        mine = [h for h in histories if shard_of[h["history_id"]] == index]
        state = progress(shard_dir(labels, index, 3), mine, NAMES, K)
        assert state["complete"] and state["before"] == len(mine) * K and not state["duplicate_keys"]
        assert {r["history_id"] for r in read_jsonl(shard_dir(labels, index, 3) / "labels.jsonl")} <= {h["history_id"] for h in mine}

    checked = merge(labels, 3, histories, NAMES, K, dry_run=True)
    assert checked["written"] is False and not (labels / "shard_merge_manifest.json").exists()
    result = merge(labels, 3, histories, NAMES, K)
    assert result["written"] and result["progress"]["complete"] and result["judge_cache_conflicting_keys"] == 0
    for name in KEYS:
        ours, theirs = read_jsonl(labels / name), read_jsonl(single / name)
        assert [set(r) for r in ours[:1]] == [set(r) for r in theirs[:1]]  # same fields
        if name in ("before_rollouts.jsonl", "after_rollouts.jsonl"):  # continuations are appended as they finish
            assert without_order(ours) == without_order(theirs)
        else:
            assert ours == theirs  # same rows in the same order as the single run
    assert read_jsonl(labels / "histories_turns.jsonl") == sorted(histories, key=lambda r: r["history_id"])
    assert len(read_jsonl(labels / "before_rollouts.jsonl")) == len(histories) * K
    previous = labels / result["previous_files"]
    assert read_jsonl(previous / "labels.jsonl") == read_jsonl(labels / "labels.jsonl")[:len(read_jsonl(previous / "labels.jsonl"))]
    with pytest.raises(ValueError, match="changed after the shard was seeded"):  # a second merge would start from other files
        merge(labels, 3, histories, NAMES, K)


def test_merge_refuses_missing_duplicate_and_foreign_rows(cases, tmp_path):
    by_id, histories, clinic, _ = environment(cases, tmp_path)
    labels = tmp_path / "labels"
    labels.mkdir()
    shard_of = assign(histories, 2)
    first = seed(labels, 0, 2, shard_of)
    run(by_id, [h for h in histories if shard_of[h["history_id"]] == 0], clinic, first)
    with pytest.raises(ValueError, match="never started"):
        merge(labels, 2, histories, NAMES, K)
    second = seed(labels, 1, 2, shard_of)
    run(by_id, [h for h in histories if shard_of[h["history_id"]] == 1], clinic, second)
    assert merge(labels, 2, histories, NAMES, K, dry_run=True)["progress"]["complete"]

    path = second / "after_rollouts.jsonl"
    original = path.read_text()
    lines = original.splitlines(keepends=True)
    path.write_text("".join(lines[1:]))  # a continuation is missing
    assert progress(second, [h for h in histories if shard_of[h["history_id"]] == 1], NAMES, K)["missing"]["after"] == 1
    with pytest.raises(ValueError, match="incomplete"):
        merge(labels, 2, histories, NAMES, K)
    path.write_text(original + lines[0])  # the same continuation twice
    with pytest.raises(ValueError, match="duplicate"):
        merge(labels, 2, histories, NAMES, K)
    path.write_text(original)
    foreign = (first / "before_rollouts.jsonl").read_text().splitlines(keepends=True)[0]
    (second / "before_rollouts.jsonl").write_text((second / "before_rollouts.jsonl").read_text() + foreign)
    with pytest.raises(ValueError, match="not its history"):
        merge(labels, 2, histories, NAMES, K)
    with pytest.raises(ValueError, match="another assignment"):  # a directory made for 2 shards is not shard 0 of another plan
        marker = json.loads((first / MARKER).read_text())
        (first / MARKER).write_text(json.dumps(dict(marker, assignment_sha256="x")))
        seed(labels, 0, 2, shard_of)
    assert not (labels / "labels.jsonl").exists()  # nothing was written into the label directory by a refused merge
