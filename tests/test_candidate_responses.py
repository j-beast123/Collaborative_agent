import json
import sys

import pytest

import scripts.generate_candidate_responses as candidates_script
from test_serl_steps import fake_runtime, rich_trajectory


def test_stance_candidates_parallel_and_resume(tmp_path, monkeypatch):
    prompts = []
    runtime, _ = fake_runtime(agent_text="stance reply")
    plain_agent = runtime.agent

    def agent(system, user, tokens):
        prompts.append(user)
        return plain_agent(system, user, tokens)

    runtime.agent = agent
    monkeypatch.setattr(candidates_script, "build_runtime", lambda **kwargs: runtime)
    trajectories = tmp_path / "traj.jsonl"
    trajectories.write_text("\n".join(json.dumps(rich_trajectory(f"t{i}")) for i in range(3)) + "\n")
    out = tmp_path / "cand.jsonl"
    argv = ["x", "--config", "configs/effectiveness/predictor_frozen.json", "--trajectories", str(trajectories),
            "--agent-model", "m", "--stance", "challenge", "--stance", "accept", "--parallel", "4",
            "--output", str(out)]
    monkeypatch.setattr(sys, "argv", argv)
    candidates_script.main()
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    assert sorted((r["history_id"], r["response_id"]) for r in rows) == sorted(
        (f"t{i}:turn3", s) for i in range(3) for s in ("challenge", "accept"))
    assert all("interaction guideline" in r["instruction"] for r in rows)
    assert sum("clinician's current position critically" in p for p in prompts) == 3
    assert sum("state the same option as the clinician" in p for p in prompts) == 3
    candidates_script.main()  # resume: nothing new
    assert len(out.read_text().splitlines()) == 6


def test_advocate_options_one_candidate_per_option_without_ground_truth(tmp_path, monkeypatch):
    prompts = []
    runtime, _ = fake_runtime(agent_text="advocate reply")
    plain_agent = runtime.agent
    runtime.agent = lambda system, user, tokens: (prompts.append(user), plain_agent(system, user, tokens))[1]
    monkeypatch.setattr(candidates_script, "build_runtime", lambda **kwargs: runtime)
    trajectory = rich_trajectory("t1")
    trajectories = tmp_path / "traj.jsonl"
    trajectories.write_text(json.dumps(trajectory) + "\n")
    out = tmp_path / "cand.jsonl"
    monkeypatch.setattr(sys, "argv", ["x", "--config", "configs/effectiveness/predictor_frozen.json", "--trajectories",
                                      str(trajectories), "--agent-model", "m", "--advocate-options", "--output", str(out)])
    candidates_script.main()
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    assert sorted(r["response_id"] for r in rows) == sorted(f"advocate_{k}" for k in trajectory["options"])
    for key, text in trajectory["options"].items():
        assert sum(f"Argue that option {key} ({text})" in p for p in prompts) == 1  # every option, gold or not


def test_v4_candidates_behaviors_samples_compliance_and_metadata(tmp_path, monkeypatch):
    from src.judges.llm import CachedChat, ScriptedBackend

    prompts = []
    runtime, _ = fake_runtime(agent_text="reply")
    plain_agent = runtime.agent

    def agent(system, user, tokens):  # the response depends on the guideline; the agent ignores request_evidence
        prompts.append(user)
        if tokens != runtime.max_tokens:
            return plain_agent(system, user, tokens)
        for marker, text in (("give the evidence", "MAINTAIN with evidence"), ("Give no reason", "MAINTAIN bare"),
                             ("Adopt the clinician", "ACCEPT"), ("Do not commit", "ACCEPT instead of asking")):
            if marker in user:
                return text
        return "natural sample"

    runtime.agent = agent
    monkeypatch.setattr(candidates_script, "build_runtime", lambda **kwargs: runtime)

    class Extractor:  # the clinician favors option A in every history
        def stated_option(self, trajectory, message):
            return "A"

    def judged(request):
        response = json.loads(request.user)["agent_response"]
        label = "MAINTAIN_CHALLENGE" if response.startswith("MAINTAIN") else "ACCEPT_REVISE" \
            if response.startswith("ACCEPT") else "OTHER"
        return json.dumps({"behavior_class": label, "gives_evidence": "with evidence" in response, "confidence": 1.0})

    monkeypatch.setattr(candidates_script, "openrouter_extractor", lambda model, cache, root: Extractor())
    monkeypatch.setattr(candidates_script, "openrouter_chat", lambda model, cache: CachedChat(ScriptedBackend(judged), None))
    trajectories = [rich_trajectory(f"t{i}") for i in range(3)]
    for i, t in enumerate(trajectories):
        t["case_id"] = f"case{i}"
    (tmp_path / "traj.jsonl").write_text("\n".join(json.dumps(t) for t in trajectories) + "\n")
    # standalone answer: differs from the clinician (case0, and it is wrong), same (case1), unparsed (case2)
    (tmp_path / "screen.jsonl").write_text("\n".join(json.dumps(
        {"case_id": f"case{i}", "agent_option": own, "doctor_options": ["A", "A", "B"]})
        for i, own in enumerate(["B", "A", ""])) + "\n")
    out = tmp_path / "cand.jsonl"
    argv = ["x", "--config", "configs/effectiveness/predictor_frozen.json", "--trajectories", str(tmp_path / "traj.jsonl"),
            "--agent-model", "m", "--screening", str(tmp_path / "screen.jsonl"), "--output", str(out), "--parallel", "2"]
    monkeypatch.setattr(sys, "argv", argv)
    candidates_script.main()
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    by_history = {}
    for r in rows:
        by_history.setdefault(r["history_id"], {})[r["response_id"]] = r
    samples = ["sample_1", "sample_2", "sample_3"]
    assert sorted(by_history["t0:turn3"]) == sorted(list(candidates_script.BEHAVIORS) + samples)
    for history in ("t1:turn3", "t2:turn3"):  # nothing to maintain or give up: natural samples only
        assert sorted(by_history[history]) == samples
    guided = by_history["t0:turn3"]
    assert {name: guided[name]["compliance"] for name in candidates_script.BEHAVIORS} == {
        "evidence_grounded_maintain": True, "unsupported_maintain": True, "accept_revise": True,
        "request_evidence": False}  # observed ACCEPT_REVISE, intended CONSIDER_REQUEST_EVIDENCE
    assert guided["request_evidence"]["observed_behavior"] == "ACCEPT_REVISE"
    assert (guided["evidence_grounded_maintain"]["gives_evidence"], guided["unsupported_maintain"]["gives_evidence"]) == (True, False)
    assert guided["sample_1"]["candidate_type"] == "natural" and guided["sample_1"]["compliance"] is None
    assert guided["sample_1"]["instruction"] is None and guided["accept_revise"]["candidate_type"] == "controlled"
    meta = guided["unsupported_maintain"]["meta"]
    assert (meta["agent_initial_answer"], meta["doctor_claim"]) == ("B", "A")
    assert (meta["agent_initial_correctness"], meta["doctor_initial_correctness"]) == (False, True)  # gold is A
    assert meta["doctor_consistency"] == pytest.approx(2 / 3)
    assert "Your own answer to this question is option B" in guided["evidence_grounded_maintain"]["instruction"]

    # No ground truth reaches a prompt: with the other option as ground truth every prompt is the same,
    # and only the correctness metadata changes.
    first_prompts = sorted(prompts)
    prompts.clear()
    for t in trajectories:
        t["ground_truth"] = {"correct_option": "B"}
    (tmp_path / "traj.jsonl").write_text("\n".join(json.dumps(t) for t in trajectories) + "\n")
    monkeypatch.setattr(sys, "argv", argv[:-4] + ["--output", str(tmp_path / "other_gold.jsonl"), "--parallel", "2"])
    candidates_script.main()
    assert sorted(prompts) == first_prompts
    flipped = {(r["history_id"], r["response_id"]): r for r in map(json.loads, (tmp_path / "other_gold.jsonl").read_text().splitlines())}
    assert flipped[("t0:turn3", "unsupported_maintain")]["meta"]["agent_initial_correctness"] is True

    # labels keep only compliant controlled candidates when the config asks for it
    from src.effectiveness.labels import load_candidates
    kept = {(c.history_id, c.response_id) for c in load_candidates(out, require_compliance=True)}
    assert ("t0:turn3", "request_evidence") not in kept and ("t0:turn3", "accept_revise") in kept
    assert len(load_candidates(out)) == len(rows) == len(kept) + 1

    # --only-disagreement --max-histories: the signal pilot subset
    monkeypatch.setattr(sys, "argv", argv[:-4] + ["--output", str(tmp_path / "pilot.jsonl"), "--only-disagreement",
                                                  "--max-histories", "5"])
    candidates_script.main()
    assert {json.loads(l)["history_id"] for l in (tmp_path / "pilot.jsonl").read_text().splitlines()} == {"t0:turn3"}
