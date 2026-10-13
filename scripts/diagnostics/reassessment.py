"""Private re-assessment before an intervention response (diagnostic only; validation.intervention.reassessment)."""

from __future__ import annotations

import json


def private_reassessment(trajectory: dict, turn: int, case: dict, runtime, prompt: str) -> dict:
    """validation.intervention.reassessment: before the intervention response the Agent re-assesses the case privately
    (greedy, like the Agent-side state).
    """
    from src.agentclinic.adapter import agent_view, render
    from src.judges.llm import parse_json_object

    masked = masked_dialogue(trajectory["turns"][:turn])
    user = (f"Clinical information available to you:\n{render(agent_view(case))}\n\nConversation so far (the Doctor's diagnostic "
            f"conclusions withheld):\n" + "\n".join(f"{m['speaker'].capitalize()}: {m['text']}" for m in masked) + f"\n\n{prompt}")
    raw = (runtime.agent_greedy or runtime.agent)(runtime.agent_system, user)
    parsed, _ = parse_json_object(raw)
    valid = isinstance(parsed, dict) and parsed.get("decision") in ("MAINTAIN", "REVISE", "UNRESOLVED")
    return {"parsed": parsed if valid else None, "raw": raw, "masked_turns": [m["text"] for m in masked if m["speaker"] == "doctor"]}


def masked_dialogue(turns: list[dict]) -> list[dict]:
    """The Doctor's turns without its diagnostic conclusions: the stated provisional / final diagnosis is replaced in the
    text and the "Provisional diagnosis:" line is dropped. The Agent's own turns are unchanged."""
    import re

    from src.agentclinic.protocol import PROVISIONAL

    out = []
    for m in turns:
        text = m.get("text") or ""
        if m["speaker"] == "doctor":
            text = re.sub(r"^\s*" + re.escape(PROVISIONAL) + r".*$", "", text, flags=re.I | re.M)
            for stated in {m.get("provisional_diagnosis"), m.get("final_diagnosis")} - {None, ""}:
                text = re.sub(re.escape(stated), "[the Doctor's conclusion, withheld]", text, flags=re.I)
        out.append({"speaker": m["speaker"], "text": text.strip()})
    return out


def reassessment_guidance(parsed: dict) -> str:
    return ("Your private re-assessment before this reply (the Doctor does not see it):\n" + json.dumps(parsed, ensure_ascii=False)
            + "\nReply accordingly. If MAINTAIN, keep your current assessment and give the evidence for it. If REVISE, revise it "
              "and name the new evidence that changed it. If UNRESOLVED, do not agree prematurely; say what is unresolved and "
              "request the discriminating evidence.")
