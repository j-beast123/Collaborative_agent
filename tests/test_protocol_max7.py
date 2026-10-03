"""The main protocol: at most 7 Doctor-Agent collaboration turns, the Doctor may stop early."""

from conftest import AGENTCLINIC, require_env_histories, require_local  # noqa: F401
import json
from pathlib import Path

import pytest

from src.agentclinic.protocol import (A2_RULE, NEUTRAL_AGENT_TASK, NEXT_STEP, PROVISIONAL, TWO_TURN, ClinicRuntime, ProtocolSpec,
                                      agent_prompt, continue_dialogue, doctor_prompt, parse_transition, protocol_spec)
from src.agentclinic.trajectory import collaborate, dialogue_of, rerun_from_turn
from src.common.config import read_config
from test_agentclinic import GOLD, cases, diagnosis_judge  # noqa: F401 (cases is a fixture)

ROOT = Path(__file__).resolve().parents[1]
MAX7 = ProtocolSpec(max_turns=7, early_stopping=True)
STATE = {"agent_current_assessment": "MG", "doctor_current_assessment": "Viral illness", "better_supported_position": "agent",
         "evidence_sufficient": True, "supporting_evidence_summary": "Antibodies are elevated."}


def runtime(finalize_after: int, calls: list, transition=None):
    """The Doctor finalizes once the conversation holds `finalize_after` consultant messages (0: never by itself)."""
    def doctor(system, user):
        calls.append(("doctor", user))
        said = user.count("\n\nConsultant: ") + user.count("\nConsultant: ") - user.count("\n\nConsultant: ")
        if "Decide what to do next" in user:
            if transition:
                return transition(said)
            if finalize_after and said >= finalize_after:
                return json.dumps({"action": "finalize", "final_diagnosis": GOLD if "ANTIBODY" in user else "Viral illness", "brief_reason": "r"})
            return json.dumps({"action": "continue", "provisional_assessment": "Viral illness", "message_to_agent": f"Tell me more ({said})."})
        if "Give your final diagnosis" in user:
            return json.dumps({"final_diagnosis": "Viral illness", "brief_reason": "forced"})
        return f"I think this is viral.\n{PROVISIONAL} Viral illness"

    def agent(system, user):
        calls.append(("agent", user))
        if "return the JSON object" in user:
            return json.dumps(STATE)
        return "Current assessment: MG\nResponse: Still my view."

    return ClinicRuntime(doctor=doctor, agent=agent, agent_greedy=agent, doctor_system="D", agent_system="A", state_system="S")


def stages(result) -> list[str]:
    return [m["stage"] for m in result["dialogue"]]


def test_doctor_stops_early_continues_or_is_forced_to_finalize_at_the_seventh_turn(cases):
    case = cases[0]
    one = continue_dialogue(case, [], runtime(1, []), spec=MAX7)
    assert stages(one) == ["D1", "A1", "FINAL"] and one["status"] == "completed" and (one["turns"], one["stop"]) == (1, "early")
    assert one["dialogue"][-1]["action"] == "finalize" and one["final_diagnosis"] == "Viral illness"

    calls = []
    three = continue_dialogue(case, [], runtime(3, calls), spec=MAX7)
    assert stages(three) == ["D1", "A1", "D2", "A2", "D3", "A3", "FINAL"] and (three["turns"], three["stop"]) == (3, "early")
    d2 = three["dialogue"][2]  # the Doctor's continue message reaches the Agent in the form of D1
    assert d2["text"] == f"Tell me more (1).\n{PROVISIONAL} Viral illness" and d2["provisional_diagnosis"] == "Viral illness" and d2["action"] == "continue"
    # the Agent has one neutral task at every turn: no rule about keeping or changing its position
    agent_prompts = [user for who, user in calls if who == "agent"]
    assert len(agent_prompts) == 3 and all(NEUTRAL_AGENT_TASK in p and A2_RULE not in p and "Do not reverse" not in p for p in agent_prompts)
    assert len({p.split("Conversation so far:")[1].split("\n\n")[-1] for p in agent_prompts}) == 1  # the same instruction text
    # the Doctor decides by the same criterion at every turn; it is not told the turn or how many are left
    decisions = [user for who, user in calls if who == "doctor" and "Decide what to do next" in user]
    assert len(decisions) == 3 and all(d.endswith(NEXT_STEP) for d in decisions)
    assert not any(word in NEXT_STEP.lower() for word in ("more time", "at most", "remaining", "turn", "left"))

    calls = []
    seven = continue_dialogue(case, [], runtime(0, calls), spec=MAX7)  # the Doctor never finalizes by itself
    assert stages(seven) == [f"{r}{t}" for t in range(1, 8) for r in "DA"] + ["FINAL"] and (seven["turns"], seven["stop"]) == (7, "max_turns")
    doctor_calls = [user for who, user in calls if who == "doctor"]
    assert sum("Decide what to do next" in u for u in doctor_calls) == 6  # after A1 ... A6
    assert "Give your final diagnosis" in doctor_calls[-1] and "Decide what to do next" not in doctor_calls[-1]  # after A7: mandatory
    with pytest.raises(ValueError):
        continue_dialogue(case, seven["dialogue"], runtime(0, []), spec=MAX7)  # a finished collaboration is not continued


def test_early_stopping_uses_only_what_the_doctor_sees(cases):
    case = cases[0]
    calls = []
    continue_dialogue(case, [], runtime(2, calls), spec=MAX7)
    for who, user in calls:
        if who == "doctor":  # the Doctor's own information: no correct diagnosis, no consultant-only finding, no evaluator field
            information = user.split("Conversation so far:")[0]
            assert GOLD not in information and "ANTIBODY-RESULT" not in information and "PTOSIS-FINDING" not in information
            assert "Correct_Diagnosis" not in user and "SECRET-MANAGEMENT-PLAN" not in user and "state" not in information.lower()
    main_system = (ROOT / "prompts/agentclinic/consultant_main.txt").read_text()
    for directive in ("Maintain consistency", "Do not agree", "Do not disagree", "explicitly communicate"):
        assert directive not in main_system  # the main protocol's Agent prompt names no interaction behavior
    assert "do not\ninvent findings" in main_system and "Maintain consistency" in (ROOT / "prompts/agentclinic/consultant.txt").read_text()
    decision = doctor_prompt(case, [{"speaker": "doctor", "stage": "D1", "text": "x"}, {"speaker": "agent", "stage": "A1", "text": "y"}], "NEXT")
    assert '"action": "finalize"' in decision and '"action": "continue"' in decision
    assert parse_transition('{"action": "finalize", "final_diagnosis": " MG "}') == {"action": "finalize", "final_diagnosis": "MG"}
    assert parse_transition('{"action": "continue", "provisional_assessment": "MG", "message_to_agent": "why?"}')["action"] == "continue"
    for broken in ('{"action": "continue", "message_to_agent": "why?"}', '{"action": "finalize"}', '{"action": "wait"}', "I will continue."):
        assert parse_transition(broken) is None  # neither form: a format failure, never read as a decision


def test_invalid_decision_is_a_format_failure_and_the_collaboration_is_rerun(cases, tmp_path):
    case = cases[0]
    case["split"] = "train"
    attempts = []

    def transition(said):
        attempts.append(said)
        return "Let me think about it." if len(attempts) == 1 else json.dumps({"action": "finalize", "final_diagnosis": "Viral illness"})

    broken = continue_dialogue(case, [], runtime(0, [], lambda said: "not json"), spec=MAX7)
    assert broken["status"] == "doctor_transition_parse_failure" and broken["final_diagnosis"] is None and stages(broken) == ["D1", "A1"]
    judge, _ = diagnosis_judge(tmp_path / "judge.jsonl")
    d1 = continue_dialogue(case, [], runtime(1, []), spec=MAX7)["dialogue"][:1]
    trajectory = collaborate(case, d1, runtime(0, [], transition), judge, trajectory_id="t", spec=MAX7, standalone="MG", history_id="h")
    assert trajectory["status"] == "completed" and trajectory["attempts"] == 2 and trajectory["trajectory_status"] == "valid"
    never = collaborate(case, d1, runtime(0, [], lambda said: "not json"), judge, trajectory_id="t2", spec=MAX7, standalone="MG", max_attempts=2)
    assert never["status"] == "doctor_transition_parse_failure" and never["trajectory_status"] == "invalid" and never["attempts"] == 2


def test_trajectory_of_a_variable_length_collaboration_and_intervention_at_any_turn(cases, tmp_path):
    case = cases[0]
    case["split"] = "train"
    judge, _ = diagnosis_judge(tmp_path / "judge.jsonl")
    calls = []
    clinic = runtime(4, calls)
    d1 = continue_dialogue(case, [], clinic, spec=MAX7)["dialogue"][:1]
    del calls[:]
    trajectory = collaborate(case, d1, clinic, judge, trajectory_id="t", spec=MAX7, standalone="MG", history_id="h")
    assert [t["stage"] for t in trajectory["turns"]] == ["D1", "A1", "D2", "A2", "D3", "A3", "D4", "A4", "FINAL"]
    assert [t["round"] for t in trajectory["turns"]] == [1, 1, 2, 2, 3, 3, 4, 4, 4] and trajectory["turns"][-1]["final"] is True
    assert (trajectory["protocol"], trajectory["num_turns"], trajectory["stop"], trajectory["max_discussion_rounds"], trajectory["early_stopping"]) == \
        ("agentclinic_max7_early_stop", 4, "early", 7, True)
    # the Agent-side state is asked before every Agent turn, by its own call, and no response prompt holds it
    state_calls = [user for who, user in calls if who == "agent" and "return the JSON object" in user]
    response_calls = [user for who, user in calls if who == "agent" and "return the JSON object" not in user]
    assert len(state_calls) == 4 == len(response_calls) and all("better_supported_position" not in p for p in response_calls)
    assert all(t["pre_response_state"]["state_status"] == "elicited" for t in trajectory["turns"] if t["speaker"] == "agent")
    assert GOLD not in json.dumps(trajectory["turns"]) and trajectory["ground_truth"] == {"correct_diagnosis": GOLD}

    # an intervention at A3: the history through D3 is kept, the fixed A3 is inserted, the same early-stopping Doctor continues
    h3 = dialogue_of(trajectory, 5)
    assert [m["stage"] for m in h3] == ["D1", "A1", "D2", "A2", "D3"]
    fixed = continue_dialogue(case, h3, clinic, fixed={"text": "The ANTIBODY result is elevated.", "assessment": "MG"}, spec=MAX7)
    assert stages(fixed)[:6] == ["D1", "A1", "D2", "A2", "D3", "A3"] and fixed["dialogue"][5]["fixed"] is True
    assert [m["text"] for m in fixed["dialogue"][:5]] == [m["text"] for m in h3] and stages(fixed)[-1] == "FINAL" and fixed["turns"] == 4
    # rubric validation at any turn: the turn is regenerated under the instruction and the collaboration resumes
    rerun = rerun_from_turn(trajectory, 3, "Report the antibody finding.", case, clinic, judge, spec=MAX7)
    assert rerun["status"] == "completed" and [m["stage"] for m in rerun["dialogue"]][:4] == ["D1", "A1", "D2", "A2"]
    # with a frozen intervention the turn is that text as it is; only the turns after it are generated
    frozen = rerun_from_turn(trajectory, 3, "Report the antibody finding.", case, clinic, judge, spec=MAX7,
                             fixed={"text": "The ANTIBODY result is elevated.", "assessment": "MG"})
    assert frozen["intervention_response"] == "The ANTIBODY result is elevated." and frozen["dialogue"][3]["fixed"] is True
    assert [m["text"] for m in frozen["dialogue"][:3]] == [m["text"] for m in dialogue_of(trajectory, 3)]
    with pytest.raises(ValueError):
        continue_dialogue(case, h3[:4], clinic, fixed={"text": "x"}, spec=MAX7)  # a fixed message needs a Doctor message before it
    # the protocol of the first labels is unchanged
    assert stages(continue_dialogue(case, [], runtime(1, []), spec=TWO_TURN)) == ["D1", "A1", "D2", "A2", "FINAL"]
    assert TWO_TURN.name == "agentclinic_two_turn" and MAX7.agent_stages() == tuple(f"A{t}" for t in range(1, 8))


def test_main_run_uses_the_frozen_dataset_version_and_reuses_nothing():
    require_local(AGENTCLINIC, why="needs the AgentClinic dataset (data/raw, not tracked)")
    from scripts.run_base_collaborations import identity, provenance
    from src.agentclinic.selection import split_eligible_cases

    config, draft = (read_config(ROOT / f"configs/effectiveness/{name}.json") for name in ("v8_agentclinic_max7", "v7_agentclinic_max7"))
    for key in ("protocol", "prompts", "models", "decoding"):
        assert config[key] == draft[key], key  # the protocol implementation and prompts are those already fixed
    assert config["dataset"]["manifest"] == "data/splits/agentclinic_main_v2.json" and config["source_histories"] is None
    assert config["output_dir"] != draft["output_dir"] and identity(config) != identity(draft)
    origin = provenance(config)
    assert origin["source_run"] is None and origin["d1"]["reused_from_source"] is False
    assert origin["agent_standalone_assessment"]["reused_from_source"] is False
    cases = split_eligible_cases(config, ROOT)
    assert sorted(sum(c["split"] == s for c in cases.values()) for s in ("train", "dev", "test")) == [31, 60, 90]
    main = read_config(ROOT / "configs/serl_agentclinic_max7.json")
    assert main["environment"]["config"] == "configs/effectiveness/v8_agentclinic_max7.json"
    assert all(main["environment"][key].startswith(config["output_dir"] + "/") for key in ("histories", "base_trajectories"))


def test_main_protocol_config_and_turn_reach_report():
    require_local("outputs/effectiveness/v5_agentclinic_full/histories.jsonl", "outputs/effectiveness/v7_agentclinic_max7", why="needs earlier environment runs (outputs/, not tracked)")
    from scripts.run_base_collaborations import identity, reach

    config = read_config(ROOT / "configs/effectiveness/v7_agentclinic_max7.json")
    first = read_config(ROOT / "configs/effectiveness/v5_agentclinic_full.json")
    assert protocol_spec(config["protocol"]) == MAX7 and protocol_spec(first.get("protocol")) == TWO_TURN
    for key in ("dataset", "split", "models", "decoding", "k"):
        assert config[key] == first[key], key  # the same cases, split, models and decoding as the first label run
    assert config["output_dir"] != first["output_dir"] and len(identity(config)) == 16
    assert config["prompts"]["doctor"] == first["prompts"]["doctor"] and config["prompts"]["consultant"].endswith("consultant_main.txt")
    from scripts.run_base_collaborations import provenance
    if (ROOT / config["source_histories"]).is_file():  # D1 may be reused (same Doctor); the standalone assessment may not (other Agent prompt)
        origin = provenance(config)
        assert origin["d1"]["reused_from_source"] is True and all(origin["d1"]["identical"].values())
        assert origin["agent_standalone_assessment"]["reused_from_source"] is False
        assert origin["agent_standalone_assessment"]["identical"] == {"model": True, "decoding": True, "system_prompt": False}

    def row(turns, correct, status="valid", stop="early"):
        return {"trajectory_status": status, "status": "completed" if status == "valid" else "doctor_transition_parse_failure",
                "num_turns": turns, "is_correct": correct, "stop": stop, "attempts": 1}

    result = reach([row(1, True), row(1, False), row(3, True), row(7, False, stop="max_turns"), row(2, False, status="invalid")], 7)
    assert result["reached"] == {"A1": 4, "A2": 2, "A3": 2, "A4": 1, "A5": 1, "A6": 1, "A7": 1}
    assert result["stopped_after"] == {"A1": 2, "A2": 0, "A3": 1, "A4": 0, "A5": 0, "A6": 0, "A7": 1}
    assert (result["valid"], result["invalid"], result["invalid_rate"], result["final_accuracy"], result["mean_turns"]) == (4, 1, pytest.approx(0.2), 0.5, 3.0)
    assert result["stop_reason"] == {"early": 3, "max_turns": 1} and result["final_accuracy_by_stop_turn"]["A1"] == {"n": 2, "accuracy": 0.5}


def test_protocol_freeze_follows_the_train_reach_report_and_refuses_a_degenerate_stopping_rule(tmp_path):
    from scripts.run_base_collaborations import FROZEN, freeze, identity

    config = read_config(ROOT / "configs/effectiveness/v8_agentclinic_max7.json")
    (tmp_path / "base_trajectories.jsonl").write_text("{}\n")
    report = {"protocol": "agentclinic_max7_early_stop", "protocol_identity": [identity(config)], "train_cases": 90, "train_collaborations": 90,
              "degeneracy": {"almost_all_stop_after_A1": True, "almost_all_run_to_the_last_turn": False}}
    (tmp_path / "turn_reach_report.json").write_text(json.dumps(report))
    with pytest.raises(SystemExit, match="degenerate"):
        freeze(config, tmp_path)
    report["degeneracy"]["almost_all_stop_after_A1"] = False
    (tmp_path / "turn_reach_report.json").write_text(json.dumps(dict(report, train_collaborations=80)))
    with pytest.raises(SystemExit, match="every TRAIN case"):
        freeze(config, tmp_path)
    (tmp_path / "turn_reach_report.json").write_text(json.dumps(report))
    frozen = freeze(config, tmp_path)
    assert frozen["protocol_identity"] == identity(config) and (tmp_path / FROZEN).is_file()
    assert freeze(config, tmp_path) == json.loads((tmp_path / FROZEN).read_text())  # freezing again changes nothing
    with pytest.raises(SystemExit, match="another protocol identity"):
        freeze(dict(config, decoding=dict(config["decoding"], doctor={"temperature": 1.5})), tmp_path)


def test_grpo_samples_and_turn_local_reward_for_every_reached_turn(cases, tmp_path):
    """A collaboration that stopped after A3 gives A1, A2 and A3 samples and no others; each is rewarded for its own
    turn, and its continuation runs under the same early-stopping Doctor."""
    from src.grpo.dataset import grpo_rows
    from src.grpo.environment import ClinicEnvironment, turn_samples
    from src.grpo.reward import SerlReward
    from src.judges.rubric_scoring import RubricScorer, ScoreResult
    from src.training.reward_pipeline import load_reward_spec
    from test_judges import schema_rubric

    by_id = {c["case_id"]: dict(c, split="train") for c in cases[:2]}
    judge, _ = diagnosis_judge(tmp_path / "judge.jsonl")
    calls = []
    clinic = runtime(3, calls)  # the Doctor finalizes after three consultant messages
    trajectories = []
    for case in by_id.values():
        d1 = continue_dialogue(case, [], clinic, spec=MAX7)["dialogue"][:1]
        trajectories.append(collaborate(case, d1, clinic, judge, trajectory_id=f"{case['case_id']}:A1:base", spec=MAX7,
                                        standalone="MG", history_id=f"{case['case_id']}:A1"))
    assert all(t["num_turns"] == 3 for t in trajectories)
    stages = MAX7.agent_stages()
    later = turn_samples(trajectories, by_id, "train", stages)
    assert sorted(s["stage"] for s in later) == ["A2", "A2", "A3", "A3"]  # A4 ... A7 were not reached: no sample is made up
    third = next(s for s in later if s["stage"] == "A3")
    assert third["history_id"].endswith(":A1/A3") and [m["stage"] for m in third["history"]] == ["D1", "A1", "D2", "A2", "D3"]
    assert set(third["states"]) == {"A1", "A2", "A3"}
    histories = [{"history_id": t["history_id"], "case_id": t["case_id"], "history": dialogue_of(t, 1)} for t in trajectories]
    rows = grpo_rows(by_id, histories, "consultant system", a2=later, stages=stages, spec=MAX7)
    assert sorted(r["stage"] for r in rows) == ["A1", "A1", "A2", "A2", "A3", "A3"]
    a3_prompt = next(r for r in rows if r["sample_id"] == third["history_id"])["prompt"][1]["content"]
    assert "Tell me more (2)." in a3_prompt and NEUTRAL_AGENT_TASK in a3_prompt and A2_RULE not in a3_prompt and GOLD not in a3_prompt
    a1_prompt = next(r for r in rows if r["stage"] == "A1")["prompt"][1]["content"]
    assert a1_prompt.split("Conversation so far:")[1].split("\n\n")[-1] == a3_prompt.split("Conversation so far:")[1].split("\n\n")[-1]
    # the fixed two-turn protocol keeps its own wording (the first labels were made with it)
    assert A2_RULE in agent_prompt(next(iter(by_id.values())), third["history"][:3], "A2") and NEUTRAL_AGENT_TASK not in agent_prompt(
        next(iter(by_id.values())), third["history"][:1], "A1")

    class Scorer(RubricScorer):
        name = "fake"

        def __init__(self):
            self.items = []

        def score(self, item):
            self.items.append(item)
            return ScoreResult(item.item_id, self.name, "scored", {"r1": 2 if "ANTIBODY" in item.context_turns[-1]["text"] else 0})

    environment = ClinicEnvironment(by_id, histories, clinic, judge, a2=later, spec=MAX7, max_attempts=2)
    config = {"reward": {"weights": {"outcome": 0.4, "appropriateness": 0.3, "effectiveness": 0.0}, "softmin_temperature": 0.5,
                         "missing_component_policy": "omit"}, "rubric_aggregation": {"method": "weighted_softmin", "temperature": 0.5}}
    scorer = Scorer()
    reward = SerlReward(environment, load_reward_spec(config), rubric_set={"rubric_set_version": "v", "rubrics": [dict(schema_rubric(), rubric_id="r1", weight=1.0)]},
                        scorer=scorer, provider=None, log_path=tmp_path / "reward_log.jsonl", iteration=0, scored_stages=stages, workers=2)
    good = "Current assessment: MG\nResponse: The ANTIBODY result is elevated."
    a1_sample = histories[0]["history_id"]
    rewards = reward(prompts=[None] * 3, completions=[[{"role": "assistant", "content": c}] for c in (good, good, "Current assessment: MG\nResponse: Nothing new.")],
                     completion_ids=[[]] * 3, sample_id=[a1_sample, third["history_id"], third["history_id"]])
    rows = [json.loads(line) for line in (tmp_path / "reward_log.jsonl").read_text().splitlines()]
    assert [r["stage"] for r in rows] == ["A1", "A3", "A3"] and [r["scored_stages"] for r in rows] == [["A1"], ["A3"], ["A3"]]
    # the A1 sample continues until the Doctor finalizes (after A3 here); the A3 sample continues from A3 (one more decision)
    assert [m["stage"] for m in rows[0]["dialogue"]] == ["D1", "A1", "D2", "A2", "D3", "A3", "FINAL"] and rows[0]["num_turns"] == 3
    assert [m["stage"] for m in rows[1]["dialogue"]] == ["D1", "A1", "D2", "A2", "D3", "A3", "FINAL"] and rows[1]["stop"] == "early"
    assert [m["text"] for m in rows[1]["dialogue"][:5]] == [m["text"] for m in third["history"]]  # the prompt's history is kept
    # each response is scored on its own turn only (turn-local): one scorer call per response, on A1 or on A3
    assert sorted(item.item_id.split(":")[-1] for item in scorer.items) == ["A1", "A3", "A3"]
    assert rewards == pytest.approx([0.4 * 1.0 + 0.3 * 1.0, 0.4 * 1.0 + 0.3 * 1.0, 0.3 * 0.0])  # the outcome follows the antibody finding
    assert reward.identity()["trained_stages"] == ["A1", "A2", "A3"] and environment.name == "agentclinic_max7_early_stop"


def test_a_collaboration_that_runs_all_seven_turns(cases, tmp_path):
    """Synthetic seven-turn collaboration (a Doctor that never finalizes by itself): the A7 history, an intervention at
    A7, the A7 GRPO sample and its reward, the A7 input of the turn-conditioned predictor, and the forced final ...
    """
    from src.agentclinic.pipeline import turn_histories
    from src.effectiveness.history import TURNS, format_input
    from src.grpo.dataset import grpo_rows
    from src.grpo.environment import ClinicEnvironment, turn_samples
    from src.grpo.reward import SerlReward
    from src.judges.rubric_scoring import RubricScorer, ScoreResult
    from src.training.reward_pipeline import load_reward_spec
    from test_judges import schema_rubric

    case = dict(cases[0], split="train")
    by_id = {case["case_id"]: case}
    judge, _ = diagnosis_judge(tmp_path / "judge.jsonl")
    calls = []
    clinic = runtime(0, calls)  # never finalizes: every collaboration runs to the seventh turn
    d1 = continue_dialogue(case, [], clinic, spec=MAX7)["dialogue"][:1]
    trajectory = collaborate(case, d1, clinic, judge, trajectory_id=f"{case['case_id']}:A1:base", spec=MAX7, standalone="MG",
                             history_id=f"{case['case_id']}:A1")
    order = [f"{r}{t}" for t in range(1, 8) for r in "DA"]
    assert (trajectory["num_turns"], trajectory["stop"], trajectory["trajectory_status"]) == (7, "max_turns", "valid")
    assert [t["stage"] for t in trajectory["turns"]] == order + ["FINAL"]
    assert all(t.get("pre_response_state") for t in trajectory["turns"] if t["speaker"] == "agent")
    assert json.loads(json.dumps(trajectory)) == trajectory  # the whole trajectory is plain JSON

    # A7 history: the dialogue D1 ... D7 as it was, in the label histories and in the GRPO samples
    labels = turn_histories([trajectory])
    assert [h["intervention_turn"] for h in labels] == [f"A{t}" for t in range(1, 8)]
    seventh = labels[-1]
    assert [m["stage"] for m in seventh["history"]] == order[:13] and seventh["history_id"].endswith(":A7")
    assert seventh["history"][-1]["text"] == f"Tell me more (6).\n{PROVISIONAL} Viral illness" and seventh["agent_assessment_before"] == "MG"
    stages = MAX7.agent_stages()
    later = turn_samples([trajectory], by_id, "train", stages)
    assert [s["stage"] for s in later] == [f"A{t}" for t in range(2, 8)]
    sample = later[-1]
    assert sample["history_id"].endswith(":A1/A7") and [m["text"] for m in sample["history"]] == [m["text"] for m in seventh["history"]]
    assert set(sample["states"]) == {f"A{t}" for t in range(1, 8)}

    # A7 intervention: the Agent's seventh turn is regenerated under an instruction; the final diagnosis follows at once
    calls.clear()
    rerun = rerun_from_turn(trajectory, 13, "Report the ANTIBODY result.", case, clinic, judge, spec=MAX7)
    assert [m["stage"] for m in rerun["dialogue"]] == order + ["FINAL"] and rerun["status"] == "completed"
    assert [m["text"] for m in rerun["dialogue"][:13]] == [m["text"] for m in seventh["history"]]
    agent_calls = [user for who, user in calls if who == "agent" and "return the JSON object" not in user]
    doctor_calls = [user for who, user in calls if who == "doctor"]
    assert len(agent_calls) == 1 and "Report the ANTIBODY result." in agent_calls[0]
    assert len(doctor_calls) == 1 and "Give your final diagnosis" in doctor_calls[0] and "Decide what to do next" not in doctor_calls[0]
    assert GOLD not in doctor_calls[0] and "ANTIBODY" not in doctor_calls[0].split("Conversation so far:")[0]

    # max-turn finalization from a label continuation: a fixed A7 response, then the mandatory final diagnosis
    calls.clear()
    after = continue_dialogue(case, seventh["history"], clinic, fixed={"text": "The ANTIBODY result is elevated.", "assessment": "MG"}, spec=MAX7)
    assert [m["stage"] for m in after["dialogue"][13:]] == ["A7", "FINAL"] and (after["turns"], after["stop"], after["status"]) == (7, "max_turns", "completed")
    assert [who for who, _ in calls] == ["doctor"] and "Give your final diagnosis" in calls[0][1]

    # A7 GRPO sample: its prompt ends with D7 and the neutral task; its reward is for A7 alone
    histories = [{"history_id": trajectory["history_id"], "case_id": case["case_id"], "history": dialogue_of(trajectory, 1)}]
    rows = grpo_rows(by_id, histories, "consultant system", a2=later, stages=stages, spec=MAX7)
    assert [r["stage"] for r in rows] == [f"A{t}" for t in range(1, 8)]
    prompt = next(r for r in rows if r["sample_id"] == sample["history_id"])["prompt"][1]["content"]
    assert "Tell me more (6)." in prompt and NEUTRAL_AGENT_TASK in prompt and GOLD not in prompt and "seventh" not in prompt.lower()

    class Scorer(RubricScorer):
        name = "fake"

        def __init__(self):
            self.items = []

        def score(self, item):
            self.items.append(item)
            return ScoreResult(item.item_id, self.name, "scored", {"r1": 2})

    environment = ClinicEnvironment(by_id, histories, clinic, judge, a2=later, spec=MAX7, max_attempts=2)
    config = {"reward": {"weights": {"outcome": 0.4, "appropriateness": 0.3, "effectiveness": 0.3}, "softmin_temperature": 0.5,
                         "missing_component_policy": "omit", "effectiveness": {"transform": "identity"}}, "rubric_aggregation": {"method": "weighted_softmin", "temperature": 0.5}}
    scorer = Scorer()
    reward = SerlReward(environment, load_reward_spec(config), rubric_set={"rubric_set_version": "v", "rubrics": [dict(schema_rubric(), rubric_id="r1", weight=1.0)]},
                        scorer=scorer, provider=None, log_path=tmp_path / "reward_log.jsonl", iteration=0, scored_stages=stages,
                        effectiveness_stages=(), workers=1)
    calls.clear()
    completion = "Current assessment: MG\nResponse: The ANTIBODY result is elevated."
    value, = reward(prompts=[None], completions=[[{"role": "assistant", "content": completion}]], completion_ids=[[]], sample_id=[sample["history_id"]])
    row, = [json.loads(line) for line in (tmp_path / "reward_log.jsonl").read_text().splitlines()]
    assert (row["stage"], row["scored_stages"], row["num_turns"], row["stop"]) == ("A7", ["A7"], 7, "max_turns")
    assert [m["stage"] for m in row["dialogue"]] == order + ["FINAL"] and row["dialogue"][13]["text"] == "The ANTIBODY result is elevated."
    assert [item.item_id.split(":")[-1] for item in scorer.items] == ["A7"] and len(scorer.items[0].context_turns) == 14
    # no Doctor decision follows A7, no Agent turn is generated beyond the completion, and R_E is missing rather than 0
    assert [who for who, user in calls if "return the JSON object" not in user] == ["doctor"]
    assert row["effectiveness"]["A7"]["available"] is False and row["weights_used"] == {"outcome": 0.4, "appropriateness": 0.3}
    assert value == pytest.approx(0.4 * 0.0 + 0.3 * 1.0)  # the forced final diagnosis of this Doctor is wrong; the rubric is satisfied

    # A7 input of the turn-conditioned predictor: BEFORE is the history, AFTER adds the response; the turn is named
    assert "A7" in TURNS
    history = environment.predictor_history(sample["history_id"], row["dialogue"][:13], "A7")
    before, after_text = format_input(history, input_format="turn_v1"), format_input(history, "The ANTIBODY result is elevated.", input_format="turn_v1")
    assert before.splitlines()[:2] == ["[STATE] before_response", "Intervention turn: A7"] and after_text.splitlines()[1] == "Intervention turn: A7"
    dialogue_block = before.split("[HISTORY]\n")[1]
    assert dialogue_block.count("Doctor: ") == 7 and dialogue_block.count("Agent: ") == 6 and "[AGENT RESPONSE]" not in before
    assert after_text == before.replace("[STATE] before_response", "[STATE] after_response") + "\n[AGENT RESPONSE]\nThe ANTIBODY result is elevated."
    assert GOLD not in before and format_input(history, input_format="turn_v1", turn="A7") == before


def test_effectiveness_labels_at_every_reached_turn_with_the_same_early_stopping_continuation(cases, tmp_path):
    """Label histories exist for the turns the base collaboration reached and for no others; BEFORE and AFTER continue
    under the same early-stopping Doctor, so a response can also change how long the collaboration goes on."""
    from src.agentclinic.pipeline import build_labels, generate_candidates, turn_histories
    from src.common.io import read_jsonl
    from src.effectiveness.history import format_input
    from test_agentclinic import compliance_judge

    by_id = {c["case_id"]: dict(c, split="train") for c in cases[:2]}
    judge, _ = diagnosis_judge(tmp_path / "judge.jsonl")
    calls = []

    def transition(said):  # the Doctor finalizes as soon as it was told the antibody result, else after three messages
        user = calls[-1][1]
        if "antibody result" in user or said >= 3:
            return json.dumps({"action": "finalize", "final_diagnosis": GOLD if "antibody result" in user else "Viral illness"})
        return json.dumps({"action": "continue", "provisional_assessment": "Viral illness", "message_to_agent": "Tell me more."})

    clinic = runtime(0, calls, transition)
    plain = clinic.agent

    def agent(system, user):  # candidates echo their guideline (test_agentclinic.fake_runtime); natural replies say nothing new
        if "stating the finding itself" in user:
            return "Current assessment: MG\nResponse: The antibody result is elevated, which supports MG."
        if "Reply in at most two sentences" in user:
            return "Current assessment: MG\nResponse: I do not agree."
        return plain(system, user)

    clinic.agent = clinic.agent_greedy = agent
    trajectories = []
    for case in by_id.values():
        d1 = continue_dialogue(case, [], clinic, spec=MAX7)["dialogue"][:1]
        trajectories.append(collaborate(case, d1, clinic, judge, trajectory_id=f"{case['case_id']}:A1:base", spec=MAX7,
                                        standalone="MG", history_id=f"{case['case_id']}:A1"))
    histories = turn_histories(trajectories)
    assert sorted(h["intervention_turn"] for h in histories) == ["A1", "A1", "A2", "A2", "A3", "A3"]  # stopped after A3: no A4 history
    third = next(h for h in histories if h["intervention_turn"] == "A3")
    assert [m["stage"] for m in third["history"]] == ["D1", "A1", "D2", "A2", "D3"] and third["history_id"].endswith(":A3")
    out = tmp_path / "labels"
    out.mkdir()
    names = ("evidence_grounded_maintain", "unsupported_maintain")
    generate_candidates(by_id, histories, clinic, compliance_judge(), out, workers=2, max_attempts=1, stage=None, names=names)
    built = build_labels(by_id, histories, clinic, judge, out, k=3, workers=2, require_compliance=True, stage=None, spec=MAX7,
                         metadata={"version": "t", "k": 3, "split": "mixed", "continuation_protocol_hash": "h"})
    assert built["labels"] == 12 and built["incomplete_pairs"] == 0
    labels = read_jsonl(out / "labels.jsonl")
    assert sorted({r["intervention_turn"] for r in labels}) == ["A1", "A2", "A3"]
    first = next(r for r in labels if r["intervention_turn"] == "A1" and r["candidate_id"] == "evidence_grounded_maintain")
    assert (first["before_rate"], first["after_rate"], first["delta_effectiveness"]) == (0.0, 1.0, 1.0)
    assert "Intervention turn: A1" in format_input(first["history"], first["response"], "turn_v1")
    assert "Intervention turn: A3" in format_input(next(r for r in labels if r["intervention_turn"] == "A3")["history"], "x", "turn_v1")
    # the candidate changes the length of the collaboration: with the evidence the Doctor finalizes right after A1,
    # without it the conversation goes on to the third turn. BEFORE (natural A1) also runs to the third turn.
    after = [r for r in read_jsonl(out / "after_rollouts.jsonl") if r["history_id"] == first["history_id"]]
    assert {r["turns"] for r in after if r["candidate_id"] == "evidence_grounded_maintain"} == {1}
    assert {r["turns"] for r in after if r["candidate_id"] == "unsupported_maintain"} == {3}
    assert {r["turns"] for r in read_jsonl(out / "before_rollouts.jsonl") if r["history_id"] == first["history_id"]} == {3}
    assert all(r["dialogue"][-1]["stage"] == "FINAL" and r["stop"] == "early" for r in after)


LABELS = ROOT / "outputs/effectiveness/v5_agentclinic_full"


def test_end_to_end_check_of_the_main_protocol_with_fake_models(tmp_path, monkeypatch):
    require_local(AGENTCLINIC, why="needs the AgentClinic dataset (data/raw, not tracked)")
    """scripts/run_e2e_max7.py with scripted models on the real cases: collaborations of different lengths, one GRPO
    sample per reached turn, an intervention, the forced final diagnosis, and the verification itself."""
    from contextlib import nullcontext

    from scripts import run_e2e_max7 as e2e
    from src.agentclinic.outcome import DiagnosisJudge
    from src.agentclinic.runtime import Clinic
    from src.agentclinic.selection import split_eligible_cases
    from src.common.io import read_jsonl
    from src.data.schema import digest
    from src.judges.llm import CachedChat, ScriptedBackend
    from src.judges.rubric_scoring import RubricScorer, ScoreResult

    root = tmp_path / "repo"
    run_dir = root / "outputs/effectiveness/v8_agentclinic_max7"  # where the main config expects the D1 histories and the base collaborations
    run_dir.mkdir(parents=True)
    for name in ("configs", "prompts", "data"):
        (root / name).symlink_to(ROOT / name)
    cases = split_eligible_cases(read_config(ROOT / "configs/effectiveness/v8_agentclinic_max7.json"), ROOT)
    consultant_system = (ROOT / "prompts/agentclinic/consultant_main.txt").read_text()

    def doctor(system, user):
        said = user.split("Conversation so far:")[1].count("\n\nConsultant: ")
        if user.endswith(NEXT_STEP):
            target = int(digest(user.split("Conversation so far:")[0])[:2], 16) % 3 + 1  # 1, 2 or 3 turns, fixed per case
            if said >= target:
                return json.dumps({"action": "finalize", "final_diagnosis": "Sarcoidosis", "brief_reason": "r"})
            return json.dumps({"action": "continue", "provisional_assessment": "Sarcoidosis", "message_to_agent": "What else do you see?"})
        if "Give your final diagnosis" in user:
            return json.dumps({"final_diagnosis": "Sarcoidosis", "brief_reason": "forced"})
        return f"I will keep my view for now.\n{PROVISIONAL} Sarcoidosis"

    def agent(system, user):
        if "return the JSON object" in user:
            return json.dumps(STATE)
        return "Current assessment: Sarcoidosis\nResponse: The findings fit your diagnosis."

    def clinic(config, root_, model_, tokenizer_, **_):
        fake = ClinicRuntime(doctor=doctor, agent=agent, agent_greedy=agent, doctor_system="doctor system",
                             agent_system=consultant_system, state_system="state system")
        judge = DiagnosisJudge(CachedChat(ScriptedBackend(lambda r: json.dumps({"equivalent": False, "reason": "x"})), None), "j")
        return Clinic(cases, fake, judge, nullcontext, config["resolved_environment"], lambda: None)

    class Scorer(RubricScorer):
        name = "fake"

        def score(self, item):
            first, second = (r["rubric_id"] for r in item.rubrics)
            return ScoreResult(item.item_id, self.name, "scored", {first: 2, second: "NA" if item.turn_id == 1 else 0})

    # the base collaborations of the protocol for twelve TRAIN cases, written where the main config expects them
    built = clinic({"resolved_environment": {}}, root, None, None)
    from scripts.run_base_collaborations import opening_histories
    train = opening_histories({c["case_id"]: c for c in sorted(cases.values(), key=lambda c: c["case_id"]) if c["split"] == "train"},
                              built.runtime, run_dir / "histories.jsonl", workers=2)[:12]
    base = [collaborate(cases[h["case_id"]], h["history"], built.runtime, built.judge, trajectory_id=f"{h['history_id']}:base", spec=MAX7,
                        standalone="Sarcoidosis", history_id=h["history_id"]) for h in train]
    assert {t["num_turns"] for t in base} == {1, 2, 3}
    (run_dir / "base_trajectories.jsonl").write_text("".join(json.dumps(t) + "\n" for t in base))
    (root / "check.json").write_text(json.dumps({"extends": "configs/serl_agentclinic_max7.json"}))
    monkeypatch.setattr(e2e, "ROOT", root)
    monkeypatch.setattr(e2e, "load_policy", lambda *a, **k: (None, None))
    monkeypatch.setattr(e2e, "build_clinic", clinic)
    monkeypatch.setattr(e2e, "build_scorer", lambda config, root_: Scorer())
    monkeypatch.setenv("OPENROUTER_API_KEY", "unused")

    def stage(*arguments):
        monkeypatch.setattr("sys.argv", ["run_e2e_max7", "--config", str(root / "check.json"), *arguments])
        try:
            e2e.main()
        except SystemExit as stop:
            return stop.code
        return 0

    assert stage("--cases", "2") != 0  # needs a passed 1-case run
    assert stage("--cases", "1", "--write-run-id", str(root / "id1")) == 0
    first = root / "outputs/serl" / (root / "id1").read_text()
    checks = json.loads((first / "checks.json").read_text())
    assert checks["passed"], checks["failed"]
    log = read_jsonl(first / "rewards.jsonl")
    # the longest base collaboration (3 turns): one sample per reached turn, each rewarded for its own turn, no R_E
    assert [r["stage"] for r in log] == ["A1", "A2", "A3"] and [r["scored_stages"] for r in log] == [["A1"], ["A2"], ["A3"]]
    assert all(r["effectiveness"][r["stage"]]["available"] is False and r["weights_used"] == {"outcome": 0.4, "appropriateness": 0.3} for r in log)
    assert all(len(r["dialogue"]) == 7 and r["dialogue"][-1]["stage"] == "FINAL" and r["stop"] == "early" for r in log)
    intervention, = read_jsonl(first / "interventions.jsonl")
    assert intervention["rerun"]["stage"] == "A3" and [m["stage"] for m in intervention["forced_final"]["dialogue"]] == ["D1", "A1", "FINAL"]
    assert len(checks["per_unit"]) == 4 and all(all(v.values()) for v in checks["per_unit"].values()) and all(checks["run"].values())

    assert stage("--cases", "2", "--prerequisite-run", first.name, "--write-run-id", str(root / "id2")) == 0
    second = root / "outputs/serl" / (root / "id2").read_text()
    assert stage("--cases", "5", "--prerequisite-run", first.name) != 0  # the 5-case check needs a passed 2-case run
    assert stage("--cases", "5", "--prerequisite-run", second.name, "--write-run-id", str(root / "id5")) == 0
    five = json.loads((root / "outputs/serl" / (root / "id5").read_text() / "checks.json").read_text())
    assert five["passed"], five["failed"]
    assert five["paths"]["base_lengths"] == [1, 1, 1, 2, 3] or sorted(set(five["paths"]["base_lengths"])) == [1, 2, 3]
    assert five["paths"]["trained_stages"] == ["A1", "A2", "A3"] and five["paths"]["stop_reasons"] == ["early"]

    # the verification is not a formality: a Doctor prompt that leaks the diagnosis or names the turn budget fails it
    run = root / "outputs/serl" / (root / "id1").read_text()
    prompts = read_jsonl(run / "prompts.jsonl")
    case = cases[prompts[0]["case_id"]]
    for p in prompts:
        if p["role"] == "doctor" and p["user"].endswith(NEXT_STEP):
            p["user"] = p["user"].replace("Conversation so far:", f"Confirmed: {case['gold']}\n\nConversation so far:", 1)
    (run / "prompts.jsonl").write_text("".join(json.dumps(p) + "\n" for p in prompts))
    config = e2e.load_main_config(root / "check.json", root)
    broken = e2e.verify(dict(config, run_id=run.name, output_dir=str(run.relative_to(root))), run, log, MAX7)
    assert not broken["passed"] and any("doctor_sees_no_correct_diagnosis" in f for f in broken["failed"])
