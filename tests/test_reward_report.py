import pytest

from scripts.reward_report import report

WEIGHTS = {"outcome": 0.4, "appropriateness": 0.3, "effectiveness": 0.3}


def row(sample, outcome, appropriateness, after, reward, *, uncovered=False, source="computed"):
    return {"sample_id": sample, "global_step": 0, "status": "completed", "reward_source": source, "reward": reward,
            "rubric_scores": {"A1": {"labels": {"r1": 2 if after >= 0.8 else 0, "r2": "NA"}}},
            "weights_used": {"outcome": 0.4, "appropriateness": 0.3, "effectiveness": 0.3},
            "r_outcome": outcome, "r_appropriateness": appropriateness, "r_effectiveness": after - 0.5,
            "unavailable_components": ["appropriateness"] if uncovered else [], "scored_stages": ["A1", "A2"],
            "turn_appropriateness_details": [{"no_applicable_rubric": uncovered}, {"no_applicable_rubric": uncovered}],
            "effectiveness": {"A1": {"available": True, "after": after, "delta": after - 0.5}, "A2": {"available": False}}}


def test_reward_report_reads_spread_coverage_and_predictor_fit():
    rows = [row("h1", 1.0, 1.0, 0.8, 0.79), row("h1", 1.0, None, 0.6, 0.61, uncovered=True),
            row("h2", 0.0, 0.5, 0.2, 0.06), row("h2", 1.0, 0.5, 0.4, 0.52),
            row("h2", 0.0, 0.5, 0.9, 0.3, source="group_mean")]
    result = report(rows, WEIGHTS)
    assert result["responses"] == 5 and result["with_computed_reward"] == 4  # the group-mean row is not a measurement
    outcome = result["components"]["outcome"]
    assert outcome["groups"] == 2 and outcome["groups_without_variation"] == 1  # h1: both correct
    assert abs(outcome["weighted_within_group_std"] - 0.4 * 0.25) < 1e-9
    assert abs(result["components"]["effectiveness"]["within_group_std"] - 0.1) < 1e-9
    coverage = result["rubric_coverage"]
    assert coverage["no_applicable_rubric_share"] == {"A1": 0.25, "A2": 0.25}
    assert coverage["responses_without_appropriateness"] == 1 and coverage["mean_reward_without_appropriateness"] == 0.61
    predictor = result["predictor"]
    assert predictor["n"] == 4 and predictor["observed_success_rate"] == 0.75
    # diagnostics of a rubric: success rate and Effectiveness delta by its score, and their score-2-minus-score-0 gaps
    r1 = result["rubric_diagnostics"]["r1"]
    assert r1["by_score"]["2"]["turns"] == 1 and r1["by_score"]["0"]["turns"] == 3 and "r2" not in result["rubric_diagnostics"]
    assert r1["task_success_gap"] == pytest.approx(1.0 - 2 / 3) and r1["effectiveness_gap"] == pytest.approx(0.3 - (0.1 - 0.3 - 0.1) / 3)
    assert result["rubric_coverage"]["rubric_applicability_rate"] == {"r1": 1.0, "r2": 0.0}
    assert result["ranking"]["groups"] == 2 and set(result["by_stage"]) == {"A1"}
    # every candidate basis of a rubric weight side by side (for the decision after the rubric pilot)
    rubric_set = {"rubrics": [{"rubric_id": "r1", "category_id": "cat-1", "name": "n", "failure_count": 3, "weight": 0.75},
                              {"rubric_id": "r2", "category_id": "cat-2", "name": "m", "failure_count": 1, "weight": 0.25}]}
    selection = {"categories": [{"category_id": "cat-1", "historical_frequency": 5, "validation_success_rate": 0.67, "current_frequency": 3},
                                {"category_id": "cat-2", "historical_frequency": 1, "validation_success_rate": 1.0, "current_frequency": 1}]}
    weights = report(rows, WEIGHTS, rubric_set, selection)["rubric_weight_diagnostics"]
    assert weights["r1"] == {"category_id": "cat-1", "name": "n", "applicability_rate": 1.0, "task_success_gap": pytest.approx(1 / 3),
                             "effectiveness_gap": pytest.approx(0.4), "current_failure_frequency": 3, "historical_frequency": 5,
                             "validation_success_rate": 0.67, "weight_failure_frequency": 0.75, "weight_uniform": 0.5}
    assert weights["r2"]["applicability_rate"] == 0.0 and weights["r2"]["task_success_gap"] is None
    assert abs(predictor["brier"] - (0.04 + 0.16 + 0.04 + 0.36) / 4) < 1e-9
