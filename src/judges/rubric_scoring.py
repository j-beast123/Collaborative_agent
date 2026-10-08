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
TWO_STAGE_JEV_PROTOCOL = "two_stage_jev_strict_applicability_then_expected_adherence_v5"


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
            "rubrics": list(self.rubrics),
        }
        if self.pre_response_state is not None:
            payload["agent_pre_response_state"] = self.pre_response_state
        if self.case_context is not None:
            payload["case_context"] = self.case_context
        return payload

    def pre_target_payload(self) -> dict[str, Any]:
        """Visible state before the target response; used only for applicability."""
        payload = {
            "trajectory_id": self.trajectory_id,
            "turn_id": self.turn_id,
            "dialogue_before_target_turn": list(self.context_turns[:-1]),
        }
        if self.pre_response_state is not None:
            payload["agent_pre_response_state"] = self.pre_response_state
        if self.case_context is not None:
            payload["case_context"] = self.case_context
        return payload

    def target_payload(self) -> dict[str, Any]:
        """Visible state through the target response; used only after applicability is established."""
        payload = {
            "trajectory_id": self.trajectory_id,
            "turn_id": self.turn_id,
            "dialogue_before_target_turn": list(self.context_turns[:-1]),
            "target_turn": self.context_turns[-1],
        }
        if self.pre_response_state is not None:
            payload["agent_pre_response_state"] = self.pre_response_state
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
    applicability_labels: dict = field(default_factory=dict)  # rubric_id -> Y|N
    applicability_probabilities: dict = field(default_factory=dict)
    behavior_scores: dict = field(default_factory=dict)  # applicable rubric_id -> 0|1|2
    behavior_probabilities: dict = field(default_factory=dict)
    expected_behavior_scores: dict = field(default_factory=dict)  # applicable rubric_id -> E[0,1,2]
    expected_behavior_adherence: dict = field(default_factory=dict)  # applicable rubric_id -> E[score]/2
    problems: list = field(default_factory=list)
    call: dict = field(default_factory=dict)
    call_count: int = 0
    scoring_protocol: Optional[str] = None
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


def validate_visible_rubric(rubric: dict[str, Any]) -> list[str]:
    """Problems with an observable-dialogue rubric used by the two-stage scorer."""
    problems = []
    for field_name in ("rubric_id", "name", "applicability_condition"):
        if not isinstance(rubric.get(field_name), str) or not rubric[field_name].strip():
            problems.append(f"missing {field_name}")
    scores = rubric.get("scores")
    if not isinstance(scores, dict) or set(scores) != {"0", "1", "2"}:
        return problems + ["scores must have exactly string levels 0, 1, 2"]
    if any(not isinstance(scores[level], str) or not scores[level].strip() for level in ("0", "1", "2")):
        problems.append("scores 0, 1, and 2 must be non-empty strings")
    return problems


def visible_guidance_rubric(rubric: dict[str, Any]) -> dict[str, Any]:
    """Scorer-facing view of an observable-only guidance rubric."""
    problems = validate_visible_rubric(rubric)
    if problems:
        raise ValueError(f"Visible rubric {rubric.get('rubric_id')} does not follow the schema: {problems}")
    return {
        "rubric_id": rubric["rubric_id"],
        "name": rubric["name"],
        "applicability_condition": rubric["applicability_condition"],
        "scores": {level: rubric["scores"][level] for level in ("0", "1", "2")},
    }


def _visible_scorer_rubric(rubric: dict[str, Any]) -> dict[str, Any]:
    """Normalize either supported rubric schema for Jev's observable questions."""
    if "applicability_condition" in rubric:
        return visible_guidance_rubric(rubric)
    problems = validate_rubric(rubric)
    if problems or not rubric.get("rubric_id"):
        raise ValueError(f"Legacy rubric {rubric.get('rubric_id')} does not follow the schema: {problems}")
    return {
        "rubric_id": rubric["rubric_id"],
        "name": rubric["name"],
        "applicability_condition": (
            f"Clinician-side context: {rubric['applicability']['user_side_context']} "
            f"Agent pre-response assessment: {rubric['applicability']['agent_side_state']}"
        ),
        "scores": {
            level: rubric["scores"][level]["behavior"]
            for level in ("0", "1", "2")
        },
    }


def rubric_view(analysis_record: dict) -> dict[str, Any]:
    """Scorer-facing rubric from a validated failure-analysis record (a candidate)."""
    if analysis_record.get("analysis_status") != "valid":
        raise ValueError("Only validated rubrics can be scored")
    return scorer_rubric(analysis_record["rubric_id"], analysis_record["candidate_rubric"])


def final_rubric_view(rubric: dict) -> dict[str, Any]:
    """Scorer-facing view of an active rubric from rubric_set.json."""
    if "applicability_condition" in rubric:
        return visible_guidance_rubric(rubric)
    return scorer_rubric(rubric["rubric_id"], rubric)


def build_scoring_items(trajectory: dict, rubrics: list[dict], *, include_initial_turn: bool = False) -> list[ScoringItem]:
    """Build turn-local scoring items without exposing ground truth or future turns."""
    visible_only = ["applicability_condition" in rubric for rubric in rubrics]
    if any(visible_only) and not all(visible_only):
        raise ValueError("observable-only and legacy state-conditioned rubrics cannot be mixed in one scoring item")
    for rubric in rubrics:
        problems = validate_visible_rubric(rubric) if "applicability_condition" in rubric else validate_rubric(rubric)
        if problems or not rubric.get("rubric_id"):
            raise ValueError("Rubric does not follow the schema")
    if len({rubric["rubric_id"] for rubric in rubrics}) != len(rubrics):
        raise ValueError("rubric_id values must be unique")
    turns = [
        {"turn_id": index, "speaker": turn.get("speaker"), "text": turn.get("text", "")}
        for index, turn in enumerate(trajectory.get("turns", []))
    ]
    stored_case = trajectory.get("case_context")
    case_source = stored_case if isinstance(stored_case, dict) else trajectory
    case_context = {
        key: case_source.get(key)
        for key in ("question", "scenario", "options")
    }
    return [
        ScoringItem(
            item_id=f"{trajectory['trajectory_id']}:turn{turn['turn_id']}",
            case_id=str(trajectory["case_id"]),
            trajectory_id=str(trajectory["trajectory_id"]),
            turn_id=turn["turn_id"],
            context_turns=tuple(turns[: turn["turn_id"] + 1]),
            rubrics=tuple(rubrics),
            # Observable-only rubric applicability must not receive belief or
            # original agent-side state. Legacy rubrics retain their explicit
            # state-conditioned contract for backwards compatibility.
            pre_response_state=(None if all(visible_only) else pre_response_state(trajectory, turn["turn_id"])),
            case_context=case_context,
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
                             call={"calls": len(parts), "rubrics_per_call": size, "parts": [p.call for p in parts]},
                             call_count=sum(part.call_count for part in parts))
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
            return ScoreResult(item.item_id, self.name, "call_failed", problems=[result.error or "call failed"],
                               call=call, call_count=1)
        parsed, parse_status = parse_json_object(result.text)
        if parsed is None:
            return ScoreResult(item.item_id, self.name, "invalid", problems=[parse_status], call=call,
                               call_count=1, raw_response=result.text)
        labels, rationales, problems = parse_chat_scores(parsed, item)
        status = "scored" if not problems else "invalid"
        return ScoreResult(item.item_id, self.name, status, labels if not problems else {}, rationales,
                           problems=problems, call=call, call_count=1, raw_response=result.text)


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

    def visible_rubrics(self, item: ScoringItem) -> list[dict]:
        return [_visible_scorer_rubric(rubric) for rubric in item.rubrics]

    def score(self, item: ScoringItem) -> ScoreResult:
        if self.client is None:
            from .jev import JevClient

            self.client = JevClient(endpoint=self.settings.get("endpoint"), model=self.settings.get("model_id"),
                                    api_key_env=self.settings.get("api_key_env"))
        size = self.settings.get("max_rubrics_per_call") or len(item.rubrics) or 1  # Jev's context is 32K tokens
        scored = two_stage_jev_score(
            self.client,
            history_state=item.pre_target_payload(),
            target_state=item.target_payload(),
            rubrics=self.visible_rubrics(item),
            purpose="rubric_scorer",
            max_rubrics_per_call=size,
        )
        return ScoreResult(
            item_id=item.item_id,
            scorer=self.name,
            status="scored",
            labels=scored["labels"],
            rationales={},
            probabilities=scored["probabilities"],
            applicability_labels=scored["applicability_labels"],
            applicability_probabilities=scored["applicability_probabilities"],
            behavior_scores=scored["behavior_scores"],
            behavior_probabilities=scored["behavior_probabilities"],
            expected_behavior_scores=scored["expected_behavior_scores"],
            expected_behavior_adherence=scored["expected_behavior_adherence"],
            call=scored["calls"],
            call_count=scored["call_count"],
            scoring_protocol=TWO_STAGE_JEV_PROTOCOL,
        )


def applicability_questions(rubrics: list[dict]) -> dict[str, dict]:
    """Y/N questions over pre-response history only."""
    return {
        rubric["rubric_id"]: {
            "type": "choice",
            "instructions": (
                "Apply a strict conjunctive gate using only the supplied dialogue before the target Agent response. "
                "Every distinct requirement and relationship in the complete applicability condition must be "
                "directly supported by visible dialogue. Choose N when any required actor, premise, comparison, "
                "proxy-target distinction, exception structure, inconsistency, or missing discriminator is absent, "
                "only implied, or ambiguous. Do not fill a missing requirement with medical knowledge. "
                "Do not infer applicability from the target response, future turns, final correctness, or ground "
                "truth. Generic agreement, disagreement, confidence, option ranking, elimination, or discussion of "
                "multiple alternatives is not enough by itself. A premise that directly answers the requested "
                "target is not a proxy merely because it is used to compare options. An ordinary best-answer or "
                "EXCEPT question is not a categorical-exception situation unless the condition's specific contrast "
                "between relative weakness and category membership is visible. Similarity to the original failure "
                "is not sufficient. Choose Y only for a complete visible match; otherwise choose N. "
                f"Rubric: {rubric['name']}. "
                f"Applicability condition: {rubric['applicability_condition']}"
            ),
            "criteria": {
                "Y": (
                    "Every required component and relationship in the condition is directly and unambiguously "
                    "supported by the pre-response dialogue."
                ),
                "N": (
                    "One or more required components or relationships is absent, unsupported, merely implied, "
                    "ambiguous, or only generically similar."
                ),
            },
        }
        for rubric in rubrics
    }


def behavior_questions(rubrics: list[dict]) -> dict[str, dict]:
    """0/1/2 questions used only after the corresponding applicability answer is Y."""
    return {
        rubric["rubric_id"]: {
            "type": "choice",
            "instructions": (
                "The applicability condition was assessed separately and found to hold. Judge only what the target "
                "Agent response observably performs. Select the highest level whose entire description is satisfied; "
                "if any required part of a level is missing, do not award that level. In particular, level 2 requires "
                "every check, comparison, reconciliation, and conclusion-calibration operation named in level 2 to "
                "be carried out explicitly enough to inspect. Merely repeating the collaborator's premise or ranking, "
                "calling it established, saying that no contrary evidence is present, acknowledging uncertainty, or "
                "promising to examine something does not count as testing support, applying a criterion, or weighing "
                "a competing basis. Do not reward correctness, agreement, confidence, verbosity, or a changed answer "
                "unless the rubric level itself requires the corresponding observable reasoning. Do not use future "
                "turns, final correctness, or ground truth. "
                f"Rubric: {rubric['name']}."
            ),
            "criteria": {
                level: rubric["scores"][level]
                for level in ("0", "1", "2")
            },
        }
        for rubric in rubrics
    }


def _batched_jev_decisions(client, state: dict, questions: dict[str, dict], *, purpose: str,
                           max_rubrics_per_call: Optional[int]) -> tuple[dict, list[dict]]:
    if not questions:
        return {}, []
    size = max_rubrics_per_call or len(questions)
    if size < 1:
        raise ValueError("max_rubrics_per_call must be positive")
    names = list(questions)
    answers, calls = {}, []
    encoded_state = json.dumps(state, ensure_ascii=False)
    for start in range(0, len(names), size):
        selected = {name: questions[name] for name in names[start:start + size]}
        part, call = client.decide(encoded_state, selected, purpose=purpose)
        answers.update(part)
        calls.append(call)
    return answers, calls


def _call_record(calls: list[dict], max_rubrics_per_call: Optional[int]) -> dict | None:
    if not calls:
        return None
    if len(calls) == 1:
        return calls[0]
    return {
        "calls": len(calls),
        "rubrics_per_call": max_rubrics_per_call,
        "parts": calls,
    }


def two_stage_jev_score(client, *, history_state: dict, target_state: dict, rubrics: list[dict],
                        purpose: str, max_rubrics_per_call: Optional[int] = None) -> dict[str, Any]:
    """Assess applicability without the target response, then score only applicable target behaviors."""
    if "target_turn" in history_state or "dialogue_through_target_turn" in history_state:
        raise ValueError("applicability state must not include the target Agent response")
    if "target_turn" not in target_state or "dialogue_before_target_turn" not in target_state:
        raise ValueError("behavior state needs dialogue_before_target_turn and target_turn")
    if "dialogue_through_target_turn" in target_state:
        raise ValueError("behavior state must not duplicate the target response in dialogue_through_target_turn")
    if len({rubric.get("rubric_id") for rubric in rubrics}) != len(rubrics):
        raise ValueError("rubric_id values must be unique")
    for rubric in rubrics:
        problems = validate_visible_rubric(rubric)
        if problems:
            raise ValueError(f"invalid visible rubric {rubric.get('rubric_id')}: {problems}")
    app_answers, app_calls = _batched_jev_decisions(
        client,
        history_state,
        applicability_questions(rubrics),
        purpose=f"{purpose}_applicability",
        max_rubrics_per_call=max_rubrics_per_call,
    )
    expected_applicability = {rubric["rubric_id"] for rubric in rubrics}
    if set(app_answers) != expected_applicability:
        missing = sorted(expected_applicability - set(app_answers))
        extra = sorted(set(app_answers) - expected_applicability)
        raise ValueError(f"incomplete Jev applicability answers: missing={missing} extra={extra}")
    applicability_labels = {rid: answer["value"] for rid, answer in app_answers.items()}
    invalid_applicability = {
        rid: value for rid, value in applicability_labels.items()
        if value not in {"Y", "N"}
    }
    if invalid_applicability:
        raise ValueError(f"invalid Jev applicability labels: {invalid_applicability}")
    applicable_ids = {rid for rid, value in applicability_labels.items() if value == "Y"}
    applicable_rubrics = [rubric for rubric in rubrics if rubric["rubric_id"] in applicable_ids]
    behavior_answers, behavior_calls = _batched_jev_decisions(
        client,
        target_state,
        behavior_questions(applicable_rubrics),
        purpose=f"{purpose}_behavior",
        max_rubrics_per_call=max_rubrics_per_call,
    )
    if set(behavior_answers) != applicable_ids:
        missing = sorted(applicable_ids - set(behavior_answers))
        extra = sorted(set(behavior_answers) - applicable_ids)
        raise ValueError(f"incomplete Jev behavior answers: missing={missing} extra={extra}")
    invalid_behavior = {
        rid: answer.get("value") for rid, answer in behavior_answers.items()
        if str(answer.get("value")) not in {"0", "1", "2"}
    }
    if invalid_behavior:
        raise ValueError(f"invalid Jev behavior scores: {invalid_behavior}")
    behavior_scores = {rid: int(answer["value"]) for rid, answer in behavior_answers.items()}
    labels = {
        rubric["rubric_id"]: (
            behavior_scores[rubric["rubric_id"]]
            if rubric["rubric_id"] in applicable_ids
            else "NA"
        )
        for rubric in rubrics
    }
    applicability_probabilities = {
        rid: answer.get("probabilities")
        for rid, answer in app_answers.items()
    }
    behavior_probabilities = {
        rid: answer.get("probabilities")
        for rid, answer in behavior_answers.items()
    }
    expected_behavior_scores = {}
    expected_behavior_adherence = {}
    for rid, probabilities in behavior_probabilities.items():
        if not isinstance(probabilities, dict):
            raise ValueError(f"Jev returned no behavior probabilities for {rid}")
        try:
            values = {level: float(probabilities[level]) for level in ("0", "1", "2")}
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid Jev behavior probabilities for {rid}: {probabilities}") from exc
        if any(value < 0 for value in values.values()) or sum(values.values()) <= 0:
            raise ValueError(f"invalid Jev behavior probability mass for {rid}: {probabilities}")
        total = sum(values.values())
        expected = (values["1"] + 2.0 * values["2"]) / total
        expected_behavior_scores[rid] = expected
        expected_behavior_adherence[rid] = expected / 2.0
    probabilities = {
        rid: (
            behavior_probabilities.get(rid)
            if rid in applicable_ids
            else {"NA": (applicability_probabilities.get(rid) or {}).get("N")}
        )
        for rid in labels
    }
    return {
        "labels": labels,
        "probabilities": probabilities,
        "applicability_labels": applicability_labels,
        "applicability_probabilities": applicability_probabilities,
        "behavior_scores": behavior_scores,
        "behavior_probabilities": behavior_probabilities,
        "expected_behavior_scores": expected_behavior_scores,
        "expected_behavior_adherence": expected_behavior_adherence,
        "call_count": len(app_calls) + len(behavior_calls),
        "calls": {
            "applicability": _call_record(app_calls, max_rubrics_per_call),
            "behavior": _call_record(behavior_calls, max_rubrics_per_call),
        },
    }


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
