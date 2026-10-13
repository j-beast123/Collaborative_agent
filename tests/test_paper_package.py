import copy
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load(name):
    return json.loads((ROOT / "configs" / "online_rl" / name).read_text())


def test_reported_and_full_configs_share_the_paper_method():
    reported = load("paper_reproduction.json")
    full = load("paper_full_120.json")
    normalized = copy.deepcopy(full)
    normalized["name"] = reported["name"]
    normalized["output_dir"] = reported["output_dir"]
    normalized["schedule"]["cases"] = reported["schedule"]["cases"]
    normalized["notes"].pop("paper_full_120")
    assert normalized == reported
    assert reported["schedule"]["cases"] == 30
    assert full["schedule"]["cases"] == 120


def test_paper_contract_is_explicit():
    cfg = load("paper_reproduction.json")
    assert cfg["condition"] == "rubric_only"
    assert cfg["agent_reward"] == {
        "w_outcome": 0.0,
        "w_rubric": 1.0,
        "r_a_weighting": "equal",
    }
    assert cfg["rollout"]["branching"] == 4
    assert cfg["rollout"]["rounds"] == 4
    assert cfg["rollout"]["halting"]["min_agent_turns"] == 2
    assert cfg["five_gpu"]["batch_cases"] == 5
    assert cfg["optimizer"]["learning_rate"] == 1e-5
    assert cfg["optimizer"]["beta"] == 0.04
    assert cfg["optimizer"]["clip_epsilon"] == 0.2
    assert cfg["rubric"]["generator_model"] == "google/gemini-3.8-flash-20260902"
    assert cfg["feedback"]["model"] == "anthropic/claude-4.5-haiku-20251001"
    assert cfg["judges"]["gate_model"] == "openai/gpt-5.6-luna"
    assert cfg["judges"]["scorer_model"] == "openai/gpt-5.6-luna"
    assert cfg["prompt_update"]["refiner_model"] == "anthropic/claude-sonnet-5.5-20260928"


def test_packaged_reported_result():
    summary = json.loads((ROOT / "artifacts" / "reported_checkpoint" /
                          "saved_evaluation" / "summary.json").read_text())
    assert summary["evaluated"] == 622
    assert summary["correct"] == 442
    assert summary["unparsed"] == 0
    assert abs(summary["accuracy"] - 442 / 622) < 1e-12
