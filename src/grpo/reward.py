"""SERL reward as a GRPOTrainer reward function.

R_t = w_O R_O + w_A R_A,t + w_E R_E,t (turn-local credit). A missing component is omitted; the other weights are not
renormalized. R_A combines the active rubrics scored on the response; R_E is the predictor's P(success|h,r) - P(success|h).
"""

from __future__ import annotations

import http.client
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path
from collections import defaultdict
from statistics import mean
from typing import Any, Callable, Optional

from src.common import usage
from src.common.config import IntegrationUnavailable
from src.effectiveness.service import EffectivenessUnavailable, ProviderEffectiveness
from src.judges.reward import (RewardSpec, TurnSignals, compute_reward, group_mean_imputation, rubric_weights, soft_min,
                               weighted_reward)
from src.judges.rubric_scoring import RubricScorer, final_rubric_view

from .environment import AGENT_STAGES, ClinicEnvironment, Episode


class Timers:
    """Wall time spent in the reward's model calls, summed over the threads that make them. A call's time includes its
    waiting: an API retry after a rate limit, or a place in the local model's batch."""

    def __init__(self) -> None:
        self._lock, self.seconds, self.calls = threading.Lock(), defaultdict(float), defaultdict(int)

    def wrap(self, name: str, function: Callable) -> Callable:
        def timed(*args, **kwargs):
            started = time.perf_counter()
            try:
                return function(*args, **kwargs)
            finally:
                with self._lock:
                    self.seconds[name] += time.perf_counter() - started
                    self.calls[name] += 1
        return timed

    def take(self) -> dict:
        with self._lock:
            row = {"seconds": dict(self.seconds), "calls": dict(self.calls)}
            self.seconds.clear()
            self.calls.clear()
        return row


CREDIT_ASSIGNMENTS = ("turn_local", "from_turn_on")
WEIGHT_STRATEGIES = ("effectiveness_gap", "failure_frequency", "uniform")  # which weights the aggregation uses (rubric_importance)
SET_WEIGHTS = ("effectiveness_gap", "failure_frequency")  # the weights written in the rubric set; uniform ignores them
NO_APPLICABLE_POLICIES = ("group_mean", "none")  # none: no imputation, each response keeps its own turns (ablation)


def completion_text(completion: Any) -> str:
    if isinstance(completion, list):  # chat completions: [{"role": "assistant", "content": ...}]
        return "".join(str(m.get("content") or "") for m in completion if isinstance(m, dict))
    return str(completion or "")


class SerlReward:
    __name__ = "serl_reward"

    def __init__(self, environment: ClinicEnvironment, spec: RewardSpec, *, rubric_set: dict,
                 scorer: Optional[RubricScorer], provider, log_path: Path,
                 iteration: int, scored_stages: tuple = AGENT_STAGES, effectiveness_stages: tuple = ("A1",),
                 workers: int = 8, generation: Callable = nullcontext, max_failed_share: float = 0.5,
                 no_applicable_rubric_policy: str = "group_mean", credit_assignment: str = "turn_local",
                 rubric_weight_strategy: str = "failure_frequency") -> None:
        if credit_assignment not in CREDIT_ASSIGNMENTS:
            raise ValueError(f"credit_assignment must be one of {CREDIT_ASSIGNMENTS}")
        if rubric_weight_strategy not in WEIGHT_STRATEGIES:
            raise ValueError(f"rubric_weight_strategy must be one of {WEIGHT_STRATEGIES}")
        self.credit_assignment, self.rubric_weight_strategy = credit_assignment, rubric_weight_strategy
        self.max_failed_share = max_failed_share
        if no_applicable_rubric_policy not in NO_APPLICABLE_POLICIES:
            raise ValueError(f"no_applicable_rubric_policy must be one of {NO_APPLICABLE_POLICIES}")
        self.no_applicable_rubric_policy = no_applicable_rubric_policy
        allowed = environment.spec.agent_stages()
        if set(scored_stages) - set(allowed) or "A1" not in scored_stages:
            raise ValueError(f"scored_stages must contain A1 and only stages of {allowed}")
        if set(effectiveness_stages) - set(scored_stages):
            raise ValueError("effectiveness_stages must be scored stages")
        self.rubrics = tuple(final_rubric_view(r) for r in rubric_set.get("rubrics", []))
        if spec.weights["appropriateness"] > 0 and self.rubrics and scorer is None:
            raise IntegrationUnavailable("w_A > 0 with active rubrics requires a rubric scorer")
        if spec.weights["effectiveness"] > 0 and provider is None and effectiveness_stages:
            raise IntegrationUnavailable("w_E > 0 requires an effectiveness provider")
        if provider is not None and not hasattr(provider, "predict"):  # predict_before / predict_after interface
            provider = ProviderEffectiveness(provider, supported_turns=tuple(effectiveness_stages))
        if provider is not None and set(effectiveness_stages) - set(provider.supported_turns):
            raise IntegrationUnavailable(f"effectiveness.stages {list(effectiveness_stages)} are not all covered by the "
                                         f"predictor ({list(provider.supported_turns)})")
        self.environment, self.spec, self.scorer, self.provider = environment, spec, scorer, provider
        self.rubric_set_version = rubric_set.get("rubric_set_version")
        self.active_rubric_ids = [r["rubric_id"] for r in self.rubrics]
        # failure_frequency: the weights the rubric set carries (its categories' shares of the failures); uniform: equal
        self.rubric_weights = rubric_weights(rubric_set) if rubric_weight_strategy in SET_WEIGHTS else \
            ({rid: 1.0 / len(self.active_rubric_ids) for rid in self.active_rubric_ids} or None)
        self.rubric_frequencies = {r["rubric_id"]: {"category_id": r.get("category_id"), "failure_count": r.get("failure_count"),
                                                    "weight": r.get("weight")} for r in rubric_set.get("rubrics", [])}
        self.scored_stages, self.effectiveness_stages = tuple(scored_stages), tuple(effectiveness_stages)
        self.log_path, self.iteration, self.workers, self.generation = Path(log_path), iteration, workers, generation
        self._lock = threading.Lock()
        # where a reward call spends its time (reward_timing.jsonl next to the reward log); nothing else changes
        self.timers = Timers()
        runtime = getattr(environment, "runtime", None)
        if runtime is not None:
            runtime.doctor = self.timers.wrap("doctor", runtime.doctor)
            runtime.agent = self.timers.wrap("agent_continuation", runtime.agent)
            if getattr(runtime, "agent_greedy", None):
                runtime.agent_greedy = self.timers.wrap("agent_state", runtime.agent_greedy)
        if getattr(environment, "judge", None) is not None:
            environment.judge.equivalent = self.timers.wrap("outcome_judge", environment.judge.equivalent)
        if self.scorer is not None:
            self.scorer.score = self.timers.wrap("rubric_scorer", self.scorer.score)
        if self.provider is not None:
            self.provider.predict = self.timers.wrap("effectiveness", self.provider.predict)
        if provider is not None and getattr(provider, "source", None) in ("grpc", "local"):
            environment.after_agent = self._predict_after_agent

    def stages_for(self, episode: Episode) -> tuple:
        """The Agent turns that enter a response's reward. turn_local: the response itself. from_turn_on:
        also the turns the continuation wrote after it."""
        if self.credit_assignment == "turn_local":
            return (episode.stage,) if episode.stage in self.scored_stages else ()
        return tuple(s for s in self.scored_stages if int(s[1:]) >= int(episode.stage[1:]))

    def _predict_after_agent(self, episode, stage, dialogue):
        if stage in self.stages_for(episode) and self.effectiveness_available(stage):
            history = self.environment.predictor_history(episode.sample_id, dialogue[:-1], stage)
            prediction = self.provider.predict(history, dialogue[-1]["text"], stage,
                                               trajectory_id=f"{episode.sample_id}:{episode.rollout_id}")
            episode.effectiveness[stage] = prediction.to_dict()

    def identity(self) -> dict[str, Any]:
        return {"weights": self.spec.weights, "softmin_temperature": self.spec.tau,
                "softmin_temperature_effectiveness": self.spec.tau_effectiveness or self.spec.tau,
                "effectiveness_coverage": "all scored turns" if set(self.effectiveness_stages) == set(self.scored_stages)
                else f"{'+'.join(self.effectiveness_stages)} only (ablation: the other scored turns have no R_E)",
                "rubric_frequencies": self.rubric_frequencies,
                "missing_component_policy": self.spec.missing_component_policy,
                "no_applicable_rubric_policy": self.no_applicable_rubric_policy,
                "credit_assignment": self.credit_assignment, "rubric_weight_strategy": self.rubric_weight_strategy,
                "rubric_aggregation": self.spec.rubric_aggregation, "rubric_temperature": self.spec.tau_rubric,
                "trained_stages": sorted({self.environment.stage(s) for s in self.environment.histories}),
                "scored_stages": self.scored_stages, "effectiveness_stages": self.effectiveness_stages,
                "rubric_set_version": self.rubric_set_version, "active_rubric_ids": self.active_rubric_ids,
                "rubric_weights": self.rubric_weights,
                "scorer": self.scorer.identity() if self.scorer and self.rubrics else None,
                "effectiveness_provider": self.provider.identity() if self.provider else None,
                "environment": self.environment.name}

    def _collaborate(self, sample_id: str, completion: str) -> tuple[Episode, dict]:
        """Environment rollout and rubric scores of one response (API-bound; runs in a thread)."""
        with usage.case(self.environment.histories[sample_id]["case_id"]):
            try:
                episode = self.environment.rollout(sample_id, completion)
            except (OSError, http.client.HTTPException) as error:
                # the Doctor or judge API stayed unreachable after its retries: this response has no computed
                # reward (group mean, logged). A predictor failure is not caught here; it stops the run.
                episode = Episode(sample_id, self.environment.histories[sample_id]["case_id"], completion, "api_failed",
                                  stage=self.environment.stage(sample_id))
                episode.error = f"{type(error).__name__}: {error}"
                return episode, {}
            scores = {}
            if episode.status == "completed" and self.rubrics and self.spec.weights["appropriateness"] > 0:
                for stage, item in self.environment.scoring_items(episode, self.rubrics, self.stages_for(episode)).items():
                    result = self.scorer.score(item)
                    complete = result.status == "scored" and set(result.labels) == set(self.active_rubric_ids)
                    scores[stage] = {"status": result.status if complete or result.status != "scored" else "incomplete",
                                     "labels": result.labels if complete else {}, "problems": result.problems,
                                     "input": item.payload(), "result": result.to_dict()}
                    if not complete:  # never read as No Match: the response has no computed reward
                        episode.status = "scoring_failed"
            return episode, scores

    def effectiveness_available(self, stage: str) -> bool:
        """True when the reward has an Effectiveness value for this Agent turn (a predictor covers it)."""
        return self.spec.weights["effectiveness"] > 0 and stage in self.effectiveness_stages and self.provider is not None

    def _effectiveness(self, episode: Episode) -> dict:
        """{stage: prediction or the reason there is none}. A turn without a predictor is recorded as
        unavailable; no value is substituted for it. The predictor is called from one thread only."""
        values = {}
        for index, message in enumerate(episode.dialogue):
            if message["speaker"] != "agent" or message["stage"] not in self.stages_for(episode):
                continue
            stage = message["stage"]
            if not self.effectiveness_available(stage):
                values[stage] = {"available": False, "reason": "w_E = 0" if self.spec.weights["effectiveness"] == 0
                                 else f"no Effectiveness predictor for {stage}"}
                continue
            history = self.environment.predictor_history(episode.sample_id, episode.dialogue[:index], stage)
            try:
                from src.effectiveness.service import EffectivenessPrediction
                prediction = (EffectivenessPrediction(**episode.effectiveness[stage]) if stage in episode.effectiveness
                              else self.provider.predict(history, message["text"], stage,
                                                         trajectory_id=f"{episode.sample_id}:{episode.rollout_id}"))
            except EffectivenessUnavailable as error:  # configured as covered but the predictor refuses: an error
                raise IntegrationUnavailable(str(error)) from None
            values[stage] = {"available": True, "before": prediction.p_before, "after": prediction.p_after,
                             "delta": prediction.delta, "model_version": prediction.model_version,
                             "checkpoint_id": prediction.checkpoint_id,
                             "calibration_temperature": prediction.calibration_temperature,
                             "source": prediction.source, "request_id": prediction.request_id,
                             "latency_ms": prediction.latency_ms, "request": {"history": history, "response": message["text"]},
                             "prediction": prediction.to_dict()}
        return values

    def components(self, episode: Episode, result: dict) -> dict:
        """Every reward component with value, available, source, model_version and metadata; a missing
        component is available = False with value None, never 0."""
        parts = result.get("components") or {}
        effectiveness = result.get("effectiveness") or {}
        details = dict(zip(result.get("scored_stages") or [], result.get("turn_appropriateness_details") or []))
        scorer = self.scorer.identity() if self.scorer and self.rubrics else None
        predictor = self.provider.identity() if self.provider else None
        return {
            "outcome": {"value": parts.get("outcome"), "available": parts.get("outcome") is not None,
                        "source": "diagnosis_equivalence_judge", "model_version": episode.outcome_method,
                        "metadata": {"final_diagnosis": episode.final_diagnosis}},
            "appropriateness": {"value": parts.get("appropriateness"), "available": parts.get("appropriateness") is not None,
                                "source": "group_mean_imputation" if result.get("no_applicable") and result.get("imputed_turns")
                                else "rubric_scorer",
                                "model_version": scorer and scorer.get("name"),
                                "metadata": {"rubric_set_version": self.rubric_set_version,
                                             "aggregation": self.spec.rubric_aggregation, "rubric_temperature": self.spec.tau_rubric,
                                             "rubric_weight_strategy": self.rubric_weight_strategy,
                                             "credit_assignment": self.credit_assignment,
                                             "raw_value": result.get("raw_R_A"), "imputed": bool(result.get("imputed_turns")),
                                             "imputed_turns": result.get("imputed_turns"),
                                             "value_by_turn_after_imputation": result.get("imputed_R_A_by_turn"),
                                             "group_turn_missing": result.get("group_turn_missing"),
                                             "group_no_applicable_rate": result.get("group_no_applicable_rate"),
                                             "softmin_temperature": self.spec.tau,
                                             "per_turn": {stage: {"value": d["appropriateness"], "available": not d["no_applicable_rubric"],
                                                                  "no_applicable_rubric": d["no_applicable_rubric"],
                                                                  "applicable_rubrics": d["applicable_rubrics"]}
                                                          for stage, d in details.items()}}},
            "effectiveness": {"value": parts.get("effectiveness"), "available": parts.get("effectiveness") is not None,
                              "source": predictor and predictor.get("source"),
                              "model_version": predictor and predictor.get("model_version"),
                              "metadata": {"checkpoint_id": predictor and predictor.get("checkpoint_id"),
                                           "calibration_temperature": predictor and predictor.get("calibration_temperature"),
                                           "aggregation": "soft_min over the available turns",
                                           "softmin_temperature": self.spec.tau_effectiveness or self.spec.tau,
                                           "per_turn": effectiveness}},
        }

    def _reward(self, episode: Episode, scores: dict) -> dict:
        if episode.status != "completed":
            return {"reward": 0.0 if episode.status == "invalid_response" else None, "components": None}
        effectiveness = self._effectiveness(episode)
        turns = [TurnSignals(turn_id=index, rubric_labels=(scores.get(m["stage"]) or {}).get("labels") or {},
                             delta_effectiveness=(effectiveness.get(m["stage"]) or {}).get("delta"),
                             effectiveness_applicable=bool((effectiveness.get(m["stage"]) or {}).get("available")))
                 for index, m in enumerate(episode.dialogue)
                 if m["speaker"] == "agent" and m["stage"] in self.stages_for(episode)]
        result = compute_reward(turns, episode.outcome, self.spec, self.rubric_weights, combine=False)
        result.update(effectiveness=effectiveness, scored_stages=[episode.dialogue[t.turn_id]["stage"] for t in turns])
        return result

    def _combine_groups(self, sample_id: list[str], results: list[dict]) -> None:
        """Per group (the responses to one sample), turn by turn: under group_mean a turn without a value gets the mean
        of the same turn over the group's responses that have one; a turn nobody in the group has a value for stays ...
        """
        groups = defaultdict(list)
        for sid, result in zip(sample_id, results):
            if result.get("components"):
                groups[sid].append(result)
        for members in groups.values():
            stages = members[0]["scored_stages"]  # one sample, so one trained turn and the same scored turns
            own = [dict(zip(r["scored_stages"], r["turn_appropriateness"])) for r in members]
            columns = {stage: [turns.get(stage) for turns in own] for stage in stages}
            filled = {stage: group_mean_imputation(values) if self.no_applicable_rubric_policy == "group_mean" else values
                      for stage, values in columns.items()}
            raw = [r["components"]["appropriateness"] for r in members]
            for index, result in enumerate(members):
                used = {stage: filled[stage][index] for stage in stages}
                final = soft_min([v for v in used.values() if v is not None], self.spec.tau)
                imputed = [stage for stage in stages if own[index].get(stage) is None and used[stage] is not None]
                result.update(raw_R_A=raw[index], no_applicable=raw[index] is None,
                              group_no_applicable_rate=sum(v is None for v in raw) / len(raw),
                              raw_R_A_by_turn=own[index], imputed_R_A_by_turn=used, imputed_turns=imputed,
                              turn_applicability={stage: own[index].get(stage) is not None for stage in stages},
                              group_turn_missing={stage: all(v is None for v in columns[stage]) for stage in stages},
                              final_R_A=final, imputed_R_A=final if imputed else None)
                if self.spec.weights["appropriateness"] > 0:
                    result["components"] = dict(result["components"], appropriateness=final)
                active = [name for name, weight in self.spec.weights.items() if weight > 0]
                if all(result["components"][name] is None for name in active):
                    result.update(reward=None, weights_used={}, unavailable_components=active)
                else:
                    result.update(weighted_reward(result["components"], self.spec))

    def rollout_and_score(self, sample_id: list[str], texts: list[str]) -> list[tuple[Episode, dict]]:
        """Stage 1 of a reward: the environment continues every response (D2 -> A2 -> FINAL, outcome, Agent-side
        states) and the rubric scorer scores its Agent turns. Needs the policy; does not call the predictor."""
        with self.generation():  # the policy writes A2 of every continued collaboration
            with ThreadPoolExecutor(self.workers) as pool:
                return list(pool.map(self._collaborate, sample_id, texts))

    def __call__(self, prompts, completions, sample_id, trainer_state=None, log_metric=None, **_) -> list[float]:
        started = time.perf_counter()
        rolled = self.rollout_and_score(sample_id, [completion_text(c) for c in completions])
        rollout_seconds = time.perf_counter() - started
        rewards = self.finish(rolled, sample_id, trainer_state=trainer_state, log_metric=log_metric)
        self.log_timing(rolled, rollout_seconds, time.perf_counter() - started, trainer_state)
        return rewards

    def log_timing(self, rolled, rollout_seconds: float, total_seconds: float, trainer_state=None) -> None:
        """One row per reward call: its wall time, and the time and number of each kind of model call inside it
        (summed over the concurrent continuations, so the sum can exceed the wall time)."""
        spent = self.timers.take()
        row = {"global_step": getattr(trainer_state, "global_step", None), "responses": len(rolled), "workers": self.workers,
               "reward_wall_seconds": total_seconds, "rollout_and_score_wall_seconds": rollout_seconds,
               "sum_seconds": spent["seconds"], "calls": spent["calls"],
               "continuation_agent_turns": [sum(m["speaker"] == "agent" for m in episode.dialogue) - int(episode.stage[1:])
                                            for episode, _ in rolled if episode.dialogue]}
        with self._lock, self.log_path.with_name("reward_timing.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")

    def finish(self, rolled: list[tuple[Episode, dict]], sample_id: list[str], trainer_state=None, log_metric=None) -> list[float]:
        """Stage 2 of a reward: the Effectiveness predictions, the composite reward and the reward log."""
        results = [self._reward(episode, scores) for episode, scores in rolled]
        self._combine_groups(sample_id, results)

        rewards = [r["reward"] for r in results]
        failed = sum(r["reward"] is None for r in results)
        if results and failed / len(results) > self.max_failed_share:
            statuses = sorted({episode.status for (episode, _), r in zip(rolled, results) if r["reward"] is None})
            raise IntegrationUnavailable(f"{failed} of {len(results)} responses have no computed reward ({statuses}); "
                                         f"more than rollout.max_failed_share = {self.max_failed_share}")
        for index, reward in enumerate(rewards):
            if reward is None:
                group = [r["reward"] for sid, r in zip(sample_id, results) if sid == sample_id[index] and r["reward"] is not None]
                rewards[index] = mean(group) if group else 0.0

        step = getattr(trainer_state, "global_step", None)
        seen: dict[str, int] = {}
        rows = []
        for (episode, scores), result, reward in zip(rolled, results, rewards):
            components = result["components"] or {}
            seen[episode.sample_id] = seen.get(episode.sample_id, 0) + 1
            rows.append({
                "iteration": self.iteration, "global_step": step, "sample_id": episode.sample_id,
                "case_id": episode.case_id, "stage": episode.stage,
                "collaboration_state": self.environment.collaboration_states.get(episode.sample_id),
                "generation": seen[episode.sample_id] - 1, "status": episode.status,
                "reward": reward, "reward_source": "computed" if result["reward"] is not None else "group_mean",
                # raw_composite: the weighted sum of this response's available components; reward: what GRPOTrainer gets
                # (the same, or the group's mean for a response without a computed reward)
                "raw_composite": result["reward"], "reward_passed_to_trl": reward,
                "r_outcome": components.get("outcome"), "r_appropriateness": components.get("appropriateness"),
                "r_effectiveness": components.get("effectiveness"), "weights_used": result.get("weights_used"),
                "raw_R_A": result.get("raw_R_A"), "imputed_R_A": result.get("imputed_R_A"),
                "raw_R_A_by_turn": result.get("raw_R_A_by_turn"), "imputed_R_A_by_turn": result.get("imputed_R_A_by_turn"),
                "turn_applicability": result.get("turn_applicability"), "group_turn_missing": result.get("group_turn_missing"),
                "final_R_A": result.get("final_R_A"),
                "no_applicable": result.get("no_applicable"),
                "group_no_applicable_rate": result.get("group_no_applicable_rate"),
                "unavailable_components": result.get("unavailable_components"),
                "active_rubric_ids": self.active_rubric_ids, "rubric_set_version": self.rubric_set_version,
                "components": self.components(episode, result) if result.get("components") else None,
                "rubric_scores": scores, "effectiveness": result.get("effectiveness"),
                "scored_stages": result.get("scored_stages"),
                "turn_appropriateness": result.get("turn_appropriateness"),
                "turn_appropriateness_details": result.get("turn_appropriateness_details"),
                "agent_states": episode.states,
                "turn_effectiveness": result.get("turn_effectiveness"),
                "completion": episode.completion, "response": episode.response, "assessment": episode.assessment,
                "error": getattr(episode, "error", None),
                "num_turns": episode.num_turns, "stop": episode.stop,
                "final_diagnosis": episode.final_diagnosis, "outcome": episode.outcome,
                "outcome_method": episode.outcome_method, "dialogue": episode.dialogue})
        with self._lock:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as handle:
                handle.writelines(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)

        if log_metric:
            for name in ("r_outcome", "r_appropriateness", "r_effectiveness"):
                values = [row[name] for row in rows if row[name] is not None]
                if values:
                    log_metric(f"serl/{name}", mean(values))
            log_metric("serl/completed_share", mean(row["status"] == "completed" for row in rows))
            scored = [row["no_applicable"] for row in rows if row["no_applicable"] is not None]
            if scored:
                log_metric("serl/no_applicable_share", mean(scored))
            # GRPO learns from differences inside a group: a group whose rewards are all equal gives no signal
            groups = defaultdict(list)
            for row in rows:
                groups[row["sample_id"]].append(row["reward"])
            log_metric("serl/groups_with_reward_variance", mean(max(g) - min(g) > 1e-12 for g in groups.values()))
        return rewards
