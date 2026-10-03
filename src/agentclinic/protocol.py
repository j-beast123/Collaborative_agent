"""The Doctor-Agent collaboration. A Doctor-Agent pair is one collaboration turn (D1-A1 is turn 1, D2-A2 is turn 2)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Optional

from src.judges.llm import parse_json_object

from .adapter import agent_view, doctor_view, render

PROVISIONAL = "Provisional diagnosis:"
ASSESSMENT = "Current assessment:"
RESPONSE = "Response:"
UNCERTAIN = ("uncertain", "undecided")
AGENT_FORMAT = (f"Write your reply in exactly this format:\n{ASSESSMENT} <your single primary diagnosis: one disease "
                f"name, without alternatives or qualifiers>\n{RESPONSE} <your message to the Doctor>\n"
                "In the message you may express uncertainty and reasoning naturally.")
# The Doctor's decision after an Agent turn (early stopping).
NEXT_STEP = (
    "Decide what to do next. If the information you now have is enough to commit to a diagnosis, give your final "
    "diagnosis. If an issue is still unresolved and a further reply from the consultant would materially help you settle "
    "it, continue the discussion: write your next message to the consultant and state your current provisional diagnosis.\n"
    "Reply with exactly one JSON object and nothing else, in one of these two forms:\n"
    '{"action": "finalize", "final_diagnosis": "<one diagnosis>", "brief_reason": "<one or two sentences>"}\n'
    '{"action": "continue", "provisional_assessment": "<one diagnosis>", "message_to_agent": "<your message to the consultant>"}')
# The Agent's task in the main protocol: the same neutral instruction at every turn A1 ...
NEUTRAL_AGENT_TASK = ("Respond to the Doctor's latest message. Form your current assessment from all the clinical information "
                      "you have and the conversation so far, and write the reply that is most useful to the Doctor.")
# agent_first: the Agent's first message, before the Doctor has said anything
AGENT_FIRST_OPENING = ("No one has commented on this case yet. Open the discussion with the Doctor: form your current assessment "
                       "from all the clinical information you have and write the message that is most useful to the Doctor.")
A2_RULE = ("Stay consistent with your previous message. Do not reverse your position solely because the Doctor "
           "disagrees. Revise it only if the Doctor's latest message contains new substantive evidence, a clinically "
           "relevant correction, or a valid reasoning error in your previous message; if you revise, say what changed your mind.")


@dataclass(frozen=True)
class ProtocolSpec:
    """How long a collaboration may be and who ends it."""

    max_turns: int = 2
    early_stopping: bool = False  # True: after an Agent turn the Doctor chooses between FINAL and another message
    agent_first: bool = False  # True: the Agent opens (A1, D1, A2, D2, ...); the Doctor gives no diagnosis before A1
    doctor_view: str = "history"  # history: the patient history; full: the whole case; none: only the dialogue (agent-mediated)

    @property
    def doctor_full_view(self) -> bool:
        return self.doctor_view == "full"

    def __post_init__(self):
        if type(self.max_turns) is not int or self.max_turns < 1:
            raise ValueError("max_turns must be a positive integer")
        if self.doctor_view not in ("history", "full", "none"):
            raise ValueError("doctor_view must be history, full or none")
        if self.doctor_view == "none" and not self.agent_first:
            raise ValueError("a Doctor without case information needs the agent-first order")

    @property
    def name(self) -> str:
        if self.agent_first:
            return f"agentclinic_agentfirst_max{self.max_turns}" + ("_early_stop" if self.early_stopping else "") \
                + {"full": "_doctor_full_view", "none": "_agent_mediated"}.get(self.doctor_view, "")
        if self.early_stopping:
            return f"agentclinic_max{self.max_turns}_early_stop"
        return "agentclinic_two_turn" if self.max_turns == 2 else f"agentclinic_fixed_{self.max_turns}_turn"

    def turn_stages(self) -> tuple:
        """D1, A1, ..., Dn, An (FINAL follows whenever the Doctor finalizes); agent_first: A1, D1, ..., An (D_t answers A_t)."""
        if self.agent_first:
            return tuple(s for turn in range(1, self.max_turns + 1) for s in ((f"A{turn}", f"D{turn}") if turn < self.max_turns else (f"A{turn}",)))
        return tuple(f"{role}{turn}" for turn in range(1, self.max_turns + 1) for role in ("D", "A"))

    def agent_stages(self) -> tuple:
        return tuple(f"A{turn}" for turn in range(1, self.max_turns + 1))

    @property
    def neutral_agent_task(self) -> bool:
        """The main protocol gives the Agent one neutral task at every turn; the fixed two-turn protocol of the first
        labels keeps its own wording (with the consistency rule at A2)."""
        return self.early_stopping


TWO_TURN = ProtocolSpec()


def protocol_spec(section: Optional[dict]) -> ProtocolSpec:
    """The protocol of a config's `protocol` section ({"max_turns", "early_stopping"}); without one, the fixed two turns."""
    return ProtocolSpec(int(section["max_turns"]), bool(section["early_stopping"]), bool(section.get("agent_first", False)),
                        section.get("doctor_view") or "history") if section else TWO_TURN


@dataclass
class ClinicRuntime:
    """Model calls of a collaboration; tests pass fakes."""

    doctor: Callable[[str, str], str]       # (system, user) -> Doctor text
    agent: Callable[[str, str], str]        # (system, user) -> Consultant Agent text (sampled)
    doctor_system: str
    agent_system: str
    close: Optional[Callable[[], None]] = None
    agent_greedy: Optional[Callable[[str, str], str]] = None  # standalone assessment and state; falls back to agent
    state_system: str = ""  # system prompt of the Agent-side state elicitation


def dialogue_text(dialogue: list[dict]) -> str:
    names = {"doctor": "Doctor", "agent": "Consultant"}
    return "\n\n".join(f"{names[m['speaker']]}: {m['text']}" for m in dialogue) or "(no messages yet)"


def doctor_prompt(case: dict, dialogue: list[dict], stage: str, view="history") -> str:
    """view: history (the Doctor's own case information), full (the whole case) or none (no case information: everything
    the Doctor knows comes from the consultant's messages). True / False are accepted for full / history."""
    view = {True: "full", False: "history"}.get(view, view)
    if view == "none":
        context = ("You have no case information of your own. Everything you know about the patient comes from the "
                   "consultant's messages in the conversation below.")
    else:
        context = f"Clinical information available to you:\n{render(agent_view(case) if view == 'full' else doctor_view(case))}"
    if stage == "D1":
        task = ("Write your first message to the consultant. State your provisional diagnosis and briefly explain "
                f"its key basis.\nEnd with a line of the form: {PROVISIONAL} <one diagnosis>")
    elif stage == "D2":
        task = ("Respond to the consultant's message. Update or keep your provisional diagnosis based on the evidence "
                "the consultant communicated, and say explicitly which new evidence, if any, changed your assessment.\n"
                f"End with a line of the form: {PROVISIONAL} <one diagnosis>")
    elif stage == "FINAL":
        task = ("Give your final diagnosis based on the whole conversation. Reply with exactly one JSON object and "
                'nothing else:\n{"final_diagnosis": "<one diagnosis>", "brief_reason": "<one or two sentences>"}')
    elif stage == "NEXT":  # early stopping: finalize or continue, decided from the conversation alone
        task = NEXT_STEP
    else:
        raise ValueError(f"unknown Doctor stage {stage!r}")
    return f"{context}\n\nConversation so far:\n{dialogue_text(dialogue)}\n\n{task}"


def agent_prompt(case: dict, dialogue: list[dict], stage: str, guideline: str = "", spec: "ProtocolSpec" = None) -> str:
    context = f"Clinical information available to you:\n{render(agent_view(case))}"
    if spec is not None and spec.neutral_agent_task:
        if not (stage[:1] == "A" and stage[1:].isdigit()):
            raise ValueError(f"unknown Agent stage {stage!r}")
        task = AGENT_FIRST_OPENING if spec.agent_first and not dialogue else NEUTRAL_AGENT_TASK
    elif stage == "A1":
        task = "Respond to the Doctor's message."
    elif stage[:1] == "A" and stage[1:].isdigit():  # every later Agent turn has the task of A2
        task = f"Respond to the Doctor's latest message. {A2_RULE}"
    else:
        raise ValueError(f"unknown Agent stage {stage!r}")
    task += f"\n{AGENT_FORMAT}"
    if guideline:
        task += f"\n\n{guideline}"
    return f"{context}\n\nConversation so far:\n{dialogue_text(dialogue)}\n\n{task}"


def standalone_prompt(case: dict) -> str:
    return (f"Clinical information available to you:\n{render(agent_view(case))}\n\n"
            f"No one has commented on this case yet. State your single primary diagnosis.\n"
            f"Reply with one line and nothing else: {ASSESSMENT} <one disease name, without alternatives or qualifiers>")


def stated(text: str, marker: str) -> str:
    """The diagnosis after the last `marker` line of a message; "" when absent or uncertain."""
    found = re.findall(re.escape(marker) + r"\s*\**\s*(.+)", text or "", flags=re.I)
    value = found[-1].strip().strip("*.` ") if found else ""
    return "" if value.lower().startswith(UNCERTAIN) else value


def parse_agent(text: str) -> tuple[str, str]:
    """(assessment, message shown to the Doctor) of an Agent reply in AGENT_FORMAT.
    Without a Response part the whole reply minus the assessment line is the message."""
    head, marker, body = (text or "").partition(RESPONSE)
    message = body if marker else re.sub(re.escape(ASSESSMENT) + r".*", "", text or "", flags=re.I)
    first = re.search(re.escape(ASSESSMENT) + r"\s*\**\s*(.+)", head if marker else (text or ""), flags=re.I)
    value = first.group(1).strip().strip("*.` ") if first else ""
    return ("" if value.lower().startswith(UNCERTAIN) else value), message.strip()


def final_diagnosis(text: str) -> str:
    parsed, _ = parse_json_object(text)
    value = (parsed or {}).get("final_diagnosis")
    return value.strip() if isinstance(value, str) else ""


def doctor_opening(case: dict, runtime: ClinicRuntime) -> list[dict]:
    """D1: the history of the effectiveness labels (everything before A1)."""
    text = runtime.doctor(runtime.doctor_system, doctor_prompt(case, [], "D1"))
    return [{"speaker": "doctor", "stage": "D1", "text": text, "provisional_diagnosis": stated(text, PROVISIONAL)}]


def agent_message(case: dict, dialogue: list[dict], stage: str, runtime: ClinicRuntime, guideline: str = "",
                  spec: "ProtocolSpec" = None) -> dict:
    raw = runtime.agent(runtime.agent_system, agent_prompt(case, dialogue, stage, guideline, spec))
    assessment, message = parse_agent(raw)
    return {"speaker": "agent", "stage": stage, "text": message, "assessment": assessment, "raw": raw}


def continue_from_a1(case: dict, history: list[dict], runtime: ClinicRuntime, fixed_a1: Optional[str] = None,
                     fixed_assessment: str = "") -> dict:
    """history (D1) -> A1 (natural, or the fixed message with its assessment field) -> D2 -> A2 -> FINAL."""
    dialogue = list(history)
    if fixed_a1 is None:
        dialogue.append(agent_message(case, dialogue, "A1", runtime))
    else:
        dialogue.append({"speaker": "agent", "stage": "A1", "text": fixed_a1, "assessment": fixed_assessment,
                         "fixed": True})
    d2 = runtime.doctor(runtime.doctor_system, doctor_prompt(case, dialogue, "D2"))
    dialogue.append({"speaker": "doctor", "stage": "D2", "text": d2, "provisional_diagnosis": stated(d2, PROVISIONAL)})
    dialogue.append(agent_message(case, dialogue, "A2", runtime))
    final = runtime.doctor(runtime.doctor_system, doctor_prompt(case, dialogue, "FINAL"))
    diagnosis = final_diagnosis(final)
    dialogue.append({"speaker": "doctor", "stage": "FINAL", "text": final, "final_diagnosis": diagnosis})
    return {"status": "completed" if diagnosis else "final_diagnosis_parse_failure", "final_diagnosis": diagnosis or None,
            "dialogue": dialogue}


STAGES = ("D1", "A1", "D2", "A2", "FINAL")
STATE_FIELDS = ("agent_current_assessment", "doctor_current_assessment", "better_supported_position",
                "evidence_sufficient", "supporting_evidence_summary")
POSITIONS = ("agent", "doctor", "aligned", "unclear")


def state_prompt(case: dict, dialogue: list[dict]) -> str:
    return (f"Clinical information available to you:\n{render(agent_view(case))}\n\n"
            f"Conversation so far:\n{dialogue_text(dialogue)}\n\n"
            "Assess the situation before your next message and return the JSON object.")


def elicit_state(case: dict, dialogue: list[dict], runtime: ClinicRuntime) -> dict:
    """Agent-side state before the Agent turn that follows dialogue: an independent call on what the Agent sees (its
    information and the conversation so far).
    """
    # an empty dialogue is the state before A1 of the agent-first protocol (no Doctor message yet)
    if dialogue and (dialogue[-1]["speaker"] != "doctor" or dialogue[-1]["stage"] == "FINAL"):
        raise ValueError("the state is elicited after a Doctor message (or before an agent-first A1) and before an Agent turn")
    raw = (runtime.agent_greedy or runtime.agent)(runtime.state_system, state_prompt(case, dialogue))
    parsed, _ = parse_json_object(raw)
    valid = bool(parsed) and parsed.get("better_supported_position") in POSITIONS \
        and type(parsed.get("evidence_sufficient")) is bool \
        and all(isinstance(parsed.get(k), str) and parsed[k].strip()
                for k in ("agent_current_assessment", "doctor_current_assessment", "supporting_evidence_summary"))
    if not valid:
        return {"state_status": "elicitation_failed", **{k: None for k in STATE_FIELDS}}
    return {"state_status": "elicited", **{k: parsed[k] for k in STATE_FIELDS}}


def parse_transition(text: str) -> Optional[dict]:
    """The Doctor's decision after an Agent turn: {"action": "finalize", "final_diagnosis"} or
    {"action": "continue", "provisional_assessment", "message_to_agent"}; None when the output is neither."""
    parsed, _ = parse_json_object(text)
    if not isinstance(parsed, dict):
        return None

    def filled(key: str) -> bool:
        return isinstance(parsed.get(key), str) and bool(parsed[key].strip())

    if parsed.get("action") == "finalize" and filled("final_diagnosis"):
        return {"action": "finalize", "final_diagnosis": parsed["final_diagnosis"].strip()}
    if parsed.get("action") == "continue" and filled("message_to_agent") and filled("provisional_assessment"):
        return {"action": "continue", "provisional_assessment": parsed["provisional_assessment"].strip(),
                "message_to_agent": parsed["message_to_agent"].strip()}
    return None


def continue_dialogue(case: dict, dialogue: list[dict], runtime: ClinicRuntime, *, guideline: str = "",
                      fixed: Optional[dict] = None, states: Optional[dict] = None,
                      after_agent: Optional[Callable] = None, spec: ProtocolSpec = TWO_TURN,
                      persistent_guideline: bool = False) -> dict:
    """Continue a collaboration from any prefix D1 -> A1 -> D2 -> ... to the Doctor's final diagnosis."""
    dialogue = list(dialogue)
    order = spec.turn_stages()
    if [m["stage"] for m in dialogue] != list(order[: len(dialogue)]):
        raise ValueError(f"dialogue must be a prefix of {' -> '.join(order)} without the final diagnosis")
    if (guideline or fixed) and not ((dialogue and dialogue[-1]["speaker"] == "doctor") or (spec.agent_first and not dialogue)):
        raise ValueError("a guideline or a fixed message applies to an Agent turn")

    def result(status: str, diagnosis: Optional[str], **extra) -> dict:
        return {"status": status, "final_diagnosis": diagnosis or None, "dialogue": dialogue,
                "turns": sum(m["speaker"] == "agent" for m in dialogue), **extra}

    while True:
        last = dialogue[-1]["stage"] if dialogue else None
        if last is None and not spec.agent_first:
            dialogue += doctor_opening(case, runtime)
        elif last is None or last[0] == "D":
            stage = "A1" if last is None else f"A{int(last[1:]) + 1}" if spec.agent_first else f"A{last[1:]}"
            if states is not None and stage not in states:
                states[stage] = elicit_state(case, dialogue, runtime)
            if fixed:
                dialogue.append({"speaker": "agent", "stage": stage, "text": fixed["text"],
                                 "assessment": fixed.get("assessment") or "", "fixed": True})
            else:
                dialogue.append(agent_message(case, dialogue, stage, runtime, guideline, spec))
            guideline, fixed = guideline if persistent_guideline else "", None  # persistent_guideline: diagnostic only
            if after_agent is not None:
                after_agent(stage, list(dialogue))
        else:  # after an Agent turn: the Doctor continues or finalizes
            turn = int(last[1:])
            if turn >= spec.max_turns:  # the last allowed turn: the final diagnosis is mandatory
                text = runtime.doctor(runtime.doctor_system, doctor_prompt(case, dialogue, "FINAL", spec.doctor_view))
                diagnosis = final_diagnosis(text)
                dialogue.append({"speaker": "doctor", "stage": "FINAL", "text": text, "final_diagnosis": diagnosis})
                return result("completed" if diagnosis else "final_diagnosis_parse_failure", diagnosis,
                              stop="max_turns" if spec.early_stopping else "fixed_length")
            if not spec.early_stopping:
                text = runtime.doctor(runtime.doctor_system, doctor_prompt(case, dialogue, "D2", spec.doctor_view))
                dialogue.append({"speaker": "doctor", "stage": f"D{turn if spec.agent_first else turn + 1}", "text": text,
                                 "provisional_diagnosis": stated(text, PROVISIONAL)})
                continue
            text = runtime.doctor(runtime.doctor_system, doctor_prompt(case, dialogue, "NEXT", spec.doctor_view))
            decision = parse_transition(text)
            if decision is None:
                return result("doctor_transition_parse_failure", None, invalid_output=text)
            if decision["action"] == "finalize":
                dialogue.append({"speaker": "doctor", "stage": "FINAL", "text": text,
                                 "final_diagnosis": decision["final_diagnosis"], "action": "finalize"})
                return result("completed", decision["final_diagnosis"], stop="early")
            # the Agent reads a message in the form of D1: the Doctor's words, then its provisional diagnosis
            dialogue.append({"speaker": "doctor", "stage": f"D{turn if spec.agent_first else turn + 1}", "action": "continue",
                             "text": f"{decision['message_to_agent']}\n{PROVISIONAL} {decision['provisional_assessment']}",
                             "provisional_diagnosis": decision["provisional_assessment"]})
