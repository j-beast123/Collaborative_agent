"""Doctor-Agent environment of a GRPO step (AgentClinic protocols, protocol.ProtocolSpec)."""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from typing import Optional

from src.agentclinic.adapter import agent_view, render
from src.agentclinic.outcome import DiagnosisJudge
from src.agentclinic.protocol import TWO_TURN, ClinicRuntime, ProtocolSpec, continue_dialogue, elicit_state, parse_agent
from src.agentclinic.trajectory import QUESTION, agent_state, dialogue_of
from src.judges.rubric_scoring import ScoringItem

AGENT_STAGES = ("A1", "A2")


def turn_sample_id(history_id: str, stage: str) -> str:
    return f"{history_id}/{stage}"


def a2_sample_id(history_id: str) -> str:
    return turn_sample_id(history_id, "A2")


def turn_samples(trajectories: list[dict], cases: dict[str, dict], split: str, stages: tuple) -> list[dict]:
    """Training samples of the later Agent turns (A2, A3, ...) from the iteration's collaborations: for every `split`
    trajectory and every Agent turn A_t in `stages` that it reached, the dialogue D1 -> ...
    """
    samples = []
    for trajectory in sorted(trajectories, key=lambda t: t["history_id"]):
        turns = trajectory["turns"]
        if cases[trajectory["case_id"]].get("split") != split:
            continue
        states = {t["stage"]: {k: v for k, v in t["pre_response_state"].items() if k != "clinical_information_available_to_agent"}
                  for t in turns if t["speaker"] == "agent" and t.get("pre_response_state")}
        for index, turn in enumerate(turns):
            stage = turn["stage"]
            if turn["speaker"] != "agent" or stage == "A1" or stage not in stages:
                continue
            if any(not t["text"] for t in turns[:index]):  # the history must be a real conversation up to D_t
                continue
            samples.append({"history_id": turn_sample_id(trajectory["history_id"], stage),
                            "source_history_id": trajectory["history_id"], "case_id": trajectory["case_id"], "stage": stage,
                            "history": dialogue_of(trajectory, index),
                            "collaboration_state": trajectory.get("collaboration_state"), "states": states})
    return samples


def a2_samples(trajectories: list[dict], cases: dict[str, dict], split: str) -> list[dict]:
    """The A2 samples (fixed two-turn protocol)."""
    return turn_samples(trajectories, cases, split, ("A2",))


@dataclass
class Episode:
    sample_id: str
    case_id: str
    completion: str
    # completed / invalid_response (the policy's fault) / rollout_failed / outcome_unjudged / scoring_failed / api_failed
    status: str
    response: str = ""
    assessment: str = ""
    dialogue: list = field(default_factory=list)
    effectiveness: dict = field(default_factory=dict)
    states: dict = field(default_factory=dict)  # {stage: elicited Agent-side state before that turn}
    final_diagnosis: Optional[str] = None
    outcome: Optional[float] = None
    outcome_method: Optional[str] = None
    rollout_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])  # distinguishes responses to one history
    stage: str = "A1"  # the Agent turn this response is (the sample's completion)
    num_turns: Optional[int] = None  # Agent turns of the continued collaboration (early stopping: where the Doctor ended it)
    stop: Optional[str] = None       # early / max_turns / fixed_length


class ClinicEnvironment:
    def __init__(self, cases: dict[str, dict], histories: list[dict], runtime: ClinicRuntime, judge: DiagnosisJudge,
                 *, a1_states: Optional[dict[str, dict]] = None, a2: Optional[list[dict]] = None,
                 collaboration_states: Optional[dict[str, str]] = None, max_attempts: int = 3,
                 spec: ProtocolSpec = TWO_TURN) -> None:
        """histories: the A1 samples (frozen D1 histories)."""
        self.cases, self.runtime, self.judge, self.max_attempts = cases, runtime, judge, max_attempts
        self.spec, self.name = spec, spec.name
        a2 = list(a2 or [])
        self.histories = {row["history_id"]: row for row in list(histories) + a2}  # sample id -> sample
        self.a1_states = dict(a1_states or {})
        self.a1_states.update({row["history_id"]: row["states"][row["stage"]] for row in a2 if row["stage"] in row["states"]})
        self.collaboration_states = dict(collaboration_states or {})
        self.collaboration_states.update({row["history_id"]: row.get("collaboration_state") for row in a2})
        self._lock = threading.Lock()
        self.after_agent = None
        missing = sorted({row["case_id"] for row in histories} - set(cases))
        if missing:
            raise ValueError(f"histories of unknown cases: {missing[:5]}")

    def case(self, sample_id: str) -> dict:
        return self.cases[self.histories[sample_id]["case_id"]]

    def stage(self, sample_id: str) -> str:
        return self.histories[sample_id].get("stage", "A1")

    def a1_state(self, sample_id: str) -> dict:
        """The stored state before the sample's trained turn (A1, or A_t of an A_t sample)."""
        with self._lock:
            state = self.a1_states.get(sample_id)
        if state is None:
            state = elicit_state(self.case(sample_id), self.histories[sample_id]["history"], self.runtime)
            with self._lock:
                self.a1_states[sample_id] = state
        return state

    def rollout(self, sample_id: str, completion: str) -> Episode:
        row, case = self.histories[sample_id], self.case(sample_id)
        assessment, message = parse_agent(completion)
        stage = self.stage(sample_id)
        episode = Episode(sample_id, row["case_id"], completion, "invalid_response", message, assessment, stage=stage)
        if not message:
            return episode
        for _ in range(self.max_attempts):  # a FINAL that is not valid JSON is a format failure, rerun
            # stored states of the prompt's turns and of the trained turn; a turn the continuation writes gets its own
            # state on the continued dialogue (a stored state of the base collaboration's later turn is another dialogue's)
            earlier = {k: v for k, v in row.get("states", {}).items() if int(k[1:]) < int(stage[1:])}
            episode.states = {**earlier, stage: self.a1_state(sample_id)}
            def after_agent(stage, dialogue):
                if self.after_agent is not None:
                    self.after_agent(episode, stage, dialogue)
            result = continue_dialogue(case, row["history"], self.runtime,
                                       fixed={"text": message, "assessment": assessment},
                                       states=episode.states, after_agent=after_agent, spec=self.spec)
            if result["status"] == "completed":
                break
        episode.dialogue, episode.final_diagnosis = result["dialogue"], result["final_diagnosis"]
        episode.num_turns, episode.stop = result.get("turns"), result.get("stop")
        if result["status"] != "completed":
            episode.status = "rollout_failed"
            return episode
        verdict = self.judge.equivalent(case["gold"], result["final_diagnosis"])
        episode.outcome_method = verdict["method"]
        if verdict["equivalent"] is None:
            episode.status = "outcome_unjudged"
            return episode
        episode.status, episode.outcome = "completed", float(verdict["equivalent"])
        return episode

    def predictor_history(self, sample_id: str, turns: list[dict], intervention_turn: str) -> dict:
        """The Effectiveness Model's history input before an Agent turn (turns: the dialogue so far),
        as in the label rows of src/agentclinic/pipeline.py. intervention_turn is A1 or A2."""
        return {"history_id": sample_id, "intervention_turn": intervention_turn, "question": QUESTION, "options": {},
                "scenario": render(agent_view(self.case(sample_id))),
                "turns": [{"speaker": m["speaker"], "text": m["text"]} for m in turns]}

    def scoring_items(self, episode: Episode, rubrics: tuple, stages: tuple) -> dict[str, ScoringItem]:
        """{stage: scorer input} for the Agent turns of an episode."""
        turns = [{"turn_id": i, "speaker": m["speaker"], "text": m["text"]} for i, m in enumerate(episode.dialogue)]
        case = self.case(episode.sample_id)
        return {m["stage"]: ScoringItem(
            item_id=f"{episode.sample_id}:{m['stage']}", case_id=episode.case_id, trajectory_id=episode.sample_id,
            turn_id=i, context_turns=tuple(turns[: i + 1]), rubrics=rubrics,
            pre_response_state=agent_state(case, episode.states[m["stage"]]))
            for i, m in enumerate(episode.dialogue) if m["speaker"] == "agent" and m["stage"] in stages}
