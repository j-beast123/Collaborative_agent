"""Rubric scoring (score LLM) behind one interface, so Jev and GPT are comparable."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Optional

from src.common.config import IntegrationUnavailable

from .agent_state import is_collaboration_turn, pre_response_state
from .llm import CachedChat, ChatRequest, parse_json_object
from .rubric_schema import validate_rubric

LABELS = ("NA", 0, 1, 2)


@dataclass(frozen=True)
class ScoringItem:
    item_id: str
    case_id: str
    trajectory_id: str
    turn_id: int
    context_turns: tuple
    rubrics: tuple
    pre_response_state: Optional[dict] = None
    case_context: Optional[dict] = None

    def payload(self) -> dict[str, Any]:
        payload = {
            "trajectory_id": self.trajectory_id,
            "turn_id": self.turn_id,
            "dialogue_through_target_turn": list(self.context_turns),
            "target_turn": self.context_turns[-1],
            "agent_pre_response_state": self.pre_response_state,
            "rubrics": list(self.rubrics),
        }
        if self.case_context is not None:
            payload["case_context"] = self.case_context
        return payload


@dataclass
class ScoreResult:
    item_id: str
    scorer: str
    status: str  # scored / invalid / call_failed
    labels: dict = field(default_factory=dict)  # rubric_id -> label
    rationales: dict = field(default_factory=dict)
    probabilities: dict = field(default_factory=dict)  # rubric_id -> {label: p}, if the scorer provides them
    problems: list = field(default_factory=list)
    call: dict = field(default_factory=dict)
    raw_response: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def scorer_rubric(rubric_id: str, rubric: dict) -> dict[str, Any]:
    """Scorer-facing rubric: id, name, applicability and self-contained levels 0/1/2."""
    problems = validate_rubric(rubric)
    if problems:
        raise ValueError(f"Rubric {rubric_id} does not follow the schema: {problems}")
    return {"rubric_id": rubric_id, "name": rubric["name"], "applicability": rubric["applicability"],
            "scores": rubric["scores"]}


def rubric_view(analysis_record: dict) -> dict[str, Any]:
    """Scorer-facing rubric from a validated failure-analysis record (a candidate)."""
    if analysis_record.get("analysis_status") != "valid":
        raise ValueError("Only validated rubrics can be scored")
    return scorer_rubric(analysis_record["rubric_id"], analysis_record["candidate_rubric"])


def final_rubric_view(rubric: dict) -> dict[str, Any]:
    """Scorer-facing view of an active rubric from rubric_set.json."""
    return scorer_rubric(rubric["rubric_id"], rubric)


def build_scoring_items(trajectory: dict, rubrics: list[dict], *, include_initial_turn: bool = False) -> list[ScoringItem]:
    """One item per scored agent turn; each carries all rubrics (the scorer decides NA)."""
    for rubric in rubrics:
        if validate_rubric(rubric) or not rubric.get("rubric_id"):
            raise ValueError("Rubric does not follow the schema")
    turns = [
        {"turn_id": index, "speaker": turn.get("speaker"), "text": turn.get("text", "")}
        for index, turn in enumerate(trajectory.get("turns", []))
    ]
    return [
        ScoringItem(
            item_id=f"{trajectory['trajectory_id']}:turn{turn['turn_id']}",
            case_id=str(trajectory["case_id"]),
            trajectory_id=str(trajectory["trajectory_id"]),
            turn_id=turn["turn_id"],
            context_turns=tuple(turns[: turn["turn_id"] + 1]),
            rubrics=tuple(rubrics),
            pre_response_state=pre_response_state(trajectory, turn["turn_id"]),
        )
        for turn in turns
        if is_collaboration_turn(trajectory, turn["turn_id"], include_initial_turn=include_initial_turn)
    ]


class RubricScorer:
    name = "abstract"

    def identity(self) -> dict[str, Any]:
        return {"name": self.name}

    def score(self, item: ScoringItem) -> ScoreResult:
        raise NotImplementedError


class ChatRubricScorer(RubricScorer):
    """Generative scorer (e.g. GPT) using prompts/rubric_scorer.txt."""

    def __init__(self, name: str, chat: CachedChat, *, system_prompt: str, prompt_version: str, params: dict,
                 max_rubrics_per_call: Optional[int] = None) -> None:
        self.name = name
        self.chat = chat
        self.system_prompt = system_prompt
        self.prompt_version = prompt_version
        self.params = params
        self.max_rubrics_per_call = max_rubrics_per_call  # None: every rubric of the item in one call
        self.calls = 0

    def identity(self) -> dict[str, Any]:
        return {"name": self.name, "backend": self.chat.backend.identity(), "prompt_version": self.prompt_version,
                **({"max_rubrics_per_call": self.max_rubrics_per_call} if self.max_rubrics_per_call else {})}

    def score(self, item: ScoringItem) -> ScoreResult:
        """One call, or one call per max_rubrics_per_call rubrics in the set's order: a rubric is scored on its own
        applicability and levels, so the split changes the number of calls and not what a label means."""
        size = self.max_rubrics_per_call
        if not size or len(item.rubrics) <= size:
            return self._score(item)
        parts = [self._score(replace(item, rubrics=item.rubrics[start:start + size])) for start in range(0, len(item.rubrics), size)]
        failed = next((p for p in parts if p.status != "scored"), None)
        merged = ScoreResult(item.item_id, self.name, failed.status if failed else "scored",
                             problems=[problem for p in parts for problem in p.problems],
                             call={"calls": len(parts), "rubrics_per_call": size, "parts": [p.call for p in parts]})
        if not failed:
            for part in parts:
                merged.labels.update(part.labels)
                merged.rationales.update(part.rationales)
        return merged

    def _score(self, item: ScoringItem) -> ScoreResult:
        self.calls += 1
        user = "Score the target agent turn against every rubric.\n\n" + json.dumps(item.payload(), ensure_ascii=False)
        result = self.chat.complete(ChatRequest(self.system_prompt, user, self.prompt_version, self.params))
        call = {k: v for k, v in result.to_dict().items() if k != "text"}
        if result.status != "ok":
            return ScoreResult(item.item_id, self.name, "call_failed", problems=[result.error or "call failed"], call=call)
        parsed, parse_status = parse_json_object(result.text)
        if parsed is None:
            return ScoreResult(item.item_id, self.name, "invalid", problems=[parse_status], call=call, raw_response=result.text)
        labels, rationales, problems = parse_chat_scores(parsed, item)
        status = "scored" if not problems else "invalid"
        return ScoreResult(item.item_id, self.name, status, labels if not problems else {}, rationales,
                           problems=problems, call=call, raw_response=result.text)


def parse_chat_scores(parsed: dict, item: ScoringItem) -> tuple[dict, dict, list]:
    expected = [r["rubric_id"] for r in item.rubrics]
    scores = parsed.get("scores")
    problems, labels, rationales = [], {}, {}
    if parsed.get("turn_id") != item.turn_id:
        problems.append("turn_id mismatch")
    if not isinstance(scores, list):
        return {}, {}, problems + ["scores is not a list"]
    for entry in scores:
        rubric_id = entry.get("rubric_id") if isinstance(entry, dict) else None
        if rubric_id not in expected or rubric_id in labels:
            problems.append(f"unexpected or duplicate rubric_id {rubric_id}")
            continue
        applicable, score = entry.get("applicable"), entry.get("score")
        if type(applicable) is not bool or type(score) is not int or score not in (0, 1, 2):
            problems.append(f"invalid applicable/score for {rubric_id}")
            continue
        labels[rubric_id] = score if applicable else "NA"
        rationales[rubric_id] = entry.get("rationale")
    missing = set(expected) - set(labels)
    if missing:
        problems.append(f"missing rubric scores: {sorted(missing)}")
    return labels, rationales, problems


class JevRubricScorer(RubricScorer):
    """Jev (TypeSafe AI) structured scorer."""

    def __init__(self, name: str, settings: dict[str, Any], client=None) -> None:
        self.name = name
        self.settings = settings
        self.client = client  # src/judges/jev.py: JevClient; built from the settings on first use

    def identity(self) -> dict[str, Any]:
        return {"name": self.name, "backend": "jev", "model_id": self.settings.get("model_id"),
                "revision": self.settings.get("revision")}

    def request_schema(self, item: ScoringItem) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                rubric["rubric_id"]: {"enum": [str(label) for label in LABELS]} for rubric in item.rubrics
            },
            "required": [rubric["rubric_id"] for rubric in item.rubrics],
        }

    def questions(self, item: ScoringItem) -> dict[str, dict]:
        """One choice question per rubric: NA when the rubric's situation does not hold before the target turn,
        else the level whose behavior the target turn shows."""
        return {rubric["rubric_id"]: {
            "type": "choice",
            "instructions": (
                f"Rubric: {rubric['name']}. Judge only the last agent turn of dialogue_through_target_turn. The rubric applies only "
                f"if both conditions hold before that turn. Clinician side: {rubric['applicability']['user_side_context']} "
                f"Agent side (see agent_pre_response_state): {rubric['applicability']['agent_side_state']} "
                "If either condition does not hold, choose NA. Otherwise choose the level whose behavior the turn shows."),
            "criteria": {"NA": "The rubric's situation does not hold before the target turn.",
                         **{level: rubric["scores"][level]["behavior"] for level in ("0", "1", "2")}}}
            for rubric in item.rubrics}

    def score(self, item: ScoringItem) -> ScoreResult:
        if self.client is None:
            from .jev import JevClient

            self.client = JevClient(endpoint=self.settings.get("endpoint"), model=self.settings.get("model_id"),
                                    api_key_env=self.settings.get("api_key_env"))
        state = json.dumps({k: v for k, v in item.payload().items() if k != "rubrics"}, ensure_ascii=False)
        size = self.settings.get("max_rubrics_per_call") or len(item.rubrics) or 1  # Jev's context is 32K tokens
        questions, outputs, calls = self.questions(item), {}, []
        names = list(questions)
        for start in range(0, len(names), size):
            answers, call = self.client.decide(state, {n: questions[n] for n in names[start:start + size]}, purpose="rubric_scorer")
            calls.append(call)
            outputs.update({rid: {"label": a["value"], "probabilities": a["probabilities"], "confidence": a["confidence"]}
                            for rid, a in answers.items()})
        result = normalize_categorical(item, self.name, outputs, calls[0] if len(calls) == 1 else {"calls": len(calls), "parts": calls})
        result.rationales = {}  # Jev writes none
        return result


def normalize_categorical(item: ScoringItem, scorer: str, outputs: dict[str, dict], call: dict) -> ScoreResult:
    """Convert per-rubric {label, probabilities} outputs (e.g. Jev) to a ScoreResult."""
    labels, probabilities, problems = {}, {}, []
    for rubric in item.rubrics:
        output = outputs.get(rubric["rubric_id"])
        label = None if output is None else output.get("label")
        label = int(label) if label in ("0", "1", "2") else label
        if label not in LABELS:
            problems.append(f"invalid label for {rubric['rubric_id']}")
            continue
        labels[rubric["rubric_id"]] = label
        if output.get("probabilities") is not None:
            probabilities[rubric["rubric_id"]] = output["probabilities"]
    status = "scored" if not problems else "invalid"
    return ScoreResult(item.item_id, scorer, status, labels if not problems else {},
                       probabilities=probabilities, problems=problems, call=call)
