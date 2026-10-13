"""GRPO prompts: the Agent's prompt before the turn it is trained on, as chat messages."""

from __future__ import annotations

from src.agentclinic.adapter import assert_no_leakage
from src.agentclinic.protocol import TWO_TURN, ProtocolSpec, agent_prompt


def grpo_rows(cases: dict[str, dict], histories: list[dict], agent_system: str, *, split: str = "train",
              a2: list[dict] = (), stages: tuple = ("A1",), spec: ProtocolSpec = TWO_TURN) -> list[dict]:
    """histories: frozen D1 histories (A1 samples). a2: samples of the later turns (environment.turn_samples)."""
    rows = []
    for row in sorted(histories, key=lambda r: r["history_id"]) if "A1" in stages else []:
        case = cases[row["case_id"]]
        if case.get("split") != split:
            continue
        assert_no_leakage(case)
        rows.append({"sample_id": row["history_id"], "stage": "A1",
                     "prompt": [{"role": "system", "content": agent_system},
                                {"role": "user", "content": agent_prompt(case, row["history"], "A1", spec=spec)}]})
    for row in sorted(a2, key=lambda r: r["history_id"]):
        if row["stage"] not in stages:
            continue
        case = cases[row["case_id"]]
        if case.get("split") != split:
            raise ValueError(f"{row['stage']} sample {row['history_id']} is not a {split} case")
        assert_no_leakage(case)
        rows.append({"sample_id": row["history_id"], "stage": row["stage"],
                     "prompt": [{"role": "system", "content": agent_system},
                                {"role": "user", "content": agent_prompt(case, row["history"], row["stage"], spec=spec)}]})
    if not rows:
        raise ValueError(f"no {split} samples for GRPO (stages {list(stages)})")
    return rows
