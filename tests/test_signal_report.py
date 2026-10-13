import json
import sys

import scripts.effectiveness_signal_report as signal_report
from src.effectiveness.labels import Candidate, build_pair_labels, collect_histories
from src.judges.llm import CachedChat, ScriptedBackend
from test_effectiveness_predictor import SPEC
from test_serl_steps import fake_runtime, rich_trajectory


def test_signal_report_metrics_and_examples(tmp_path, monkeypatch):
    """Doctor revises only after an evidence-grounded response; the continuation agent then gives up on B."""
    trajectories = [rich_trajectory(f"t{i}") for i in range(2)]  # ground truth A; doctor claims B at round 2
    histories = collect_histories(trajectories, turn_scope="first_collaboration_turn", include_initial_turn=False)
    runtime, _ = fake_runtime(agent_text="I now agree with option B.")

    def doctor(system, user):
        if "This is the final decision" in user:
            return "Final answer: option A - a" if "EVIDENCE" in user else "Final answer: option B - b"
        return "I still think it is B."

    runtime.doctor = doctor
    meta = {"agent_initial_answer": "A", "doctor_claim": "B", "agent_initial_correctness": True,
            "doctor_initial_correctness": False}
    texts = {"evidence_grounded_maintain": "EVIDENCE: the lab result supports option A.",
             "unsupported_maintain": "It is option A.", "sample_1": "I agree with option B."}
    rows = [{"history_id": h["history_id"], "response_id": name, "response": text, "source": "test",
             "candidate_type": "natural" if name.startswith("sample") else "controlled",
             "intended_behavior": None if name.startswith("sample") else name, "observed_behavior": "MAINTAIN_CHALLENGE",
             "compliance": None if name.startswith("sample") else True, "meta": meta}
            for h, _ in histories for name, text in texts.items()]
    (tmp_path / "candidates.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    candidates = [Candidate(r["history_id"], r["response_id"], r["response"], "test",
                            {k: r[k] for k in ("candidate_type", "intended_behavior", "compliance", "meta")}) for r in rows]
    build_pair_labels(histories, candidates, runtime, SPEC, split="train", output_path=tmp_path / "labels.jsonl",
                      metadata={})

    class Extractor:
        def stated_option(self, case, message):
            return "A" if "option A" in message else "B" if "option B" in message else ""

    monkeypatch.setattr(signal_report, "openrouter_extractor", lambda model, cache, root: Extractor())
    monkeypatch.setattr(signal_report, "openrouter_chat", lambda model, cache: CachedChat(ScriptedBackend(lambda r: "NO"), None))
    monkeypatch.setattr(sys, "argv", ["x", "--labels", str(tmp_path / "labels.jsonl"), "--candidates",
                                      str(tmp_path / "candidates.jsonl"), "--judge-model", "m", "--output-dir", str(tmp_path)])
    signal_report.main()
    report = json.loads((tmp_path / "signal_report.json").read_text())
    assert report["histories"] == 2 and report["pairs"] == 6 and report["k"] == SPEC.k
    assert report["nonzero_delta"] == {"pairs": 2 / 6, "histories_with_any": 1.0}
    assert report["within_history_variation"]["mean"] == 1.0
    same = report["same_answer_behavior_effect"]
    assert (same["n"], same["mean"], same["evidence_better"]) == (2, 1.0, 2)
    revision = report["doctor_revision_rate"]
    assert revision["evidence_grounded_maintain"]["doctor_revised"] == 1.0
    assert revision["evidence_grounded_maintain"]["revised_to_agent_answer"] == 1.0
    assert revision["unsupported_maintain"]["doctor_revised"] == 0.0 and revision["(before)"]["doctor_revised"] == 0.0
    reversal = report["agent_reversal_rate"]
    assert reversal["unsupported_maintain"] == {"rollouts": 2 * SPEC.k, "agent_reversed": 1.0,
                                                "reversed_without_new_evidence": 1.0}
    assert reversal["natural"]["agent_reversed"] == 0.0  # the sample already said B
    assert report["before_saturation"]["before = 0"]["histories"] == 2
    assert report["compliance"]["evidence_grounded_maintain"]["True"] == 2
    markdown = (tmp_path / "signal_report.md").read_text()
    for heading in ("Non-zero delta", "Within-history variation", "Same-answer behavior effect", "Doctor revision rate",
                    "Agent reversal rate", "BEFORE saturation", "Recommendation"):
        assert f"## {heading}" in markdown
    assert len((tmp_path / "doctor_revision_examples.jsonl").read_text().splitlines()) == 2 * SPEC.k
    example = json.loads((tmp_path / "agent_reversal_examples.jsonl").read_text().splitlines()[0])
    assert example["new_evidence"] is False and any(t.get("intervention") for t in example["dialogue"])
