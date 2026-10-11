from pathlib import Path

from src.online_rl.tree.experiment import load_config


ROOT = Path(__file__).resolve().parents[1]
BASE = "configs/experiments/train120_max4_direct_luna_v9_1.json"
CONSOLIDATED = "configs/experiments/train120_max4_consolidation8_allmembers_v1.json"


def test_consolidation_config_changes_only_identity_assignment_and_witness_scope():
    base = load_config(BASE)
    cfg = load_config(CONSOLIDATED)

    # The requested feature must not silently change the actual learning/model contract.
    for section in (
        "policy", "rollout", "feedback",
        "agent_reward", "optimizer", "schedule", "data", "intervention",
        "five_gpu",
    ):
        assert cfg[section] == base[section], section

    assert cfg["api_provider"] == "openrouter"
    assert cfg["judges"] == {**base["judges"], "api_provider": "openai_direct"}

    base_rubric = dict(base["rubric"]); base_generation=dict(base_rubric.pop("generation"))
    cfg_rubric = dict(cfg["rubric"]); cfg_generation=dict(cfg_rubric.pop("generation"))
    assert cfg_rubric.pop("generator_model") == "google/gemini-3.8-flash-20260902"
    assert cfg_rubric.pop("assignment_protocol") is None
    assert cfg_rubric.pop("match_batch_size") == 8
    base_rubric.pop("assignment_protocol"); base_rubric.pop("generator_model")
    assert cfg_rubric == base_rubric
    assert cfg_generation == {**base_generation, "api_provider": "openrouter"}

    assert cfg["verification"]["mode"] == "all_members_witness_v1"
    assert cfg["verification"]["max_valid_supports"] is None
    assert cfg["verification"]["reruns_per_support"] == 3
    assert cfg["verification"]["support_sampling"] == "topology"


def test_consolidation_config_is_a_fresh_run_identity():
    base = load_config(BASE)
    cfg = load_config(CONSOLIDATED)
    assert cfg["experiment"]["output_root"] != base["experiment"]["output_root"]
    assert cfg["experiment"]["run_label"] == "train120_max4_consolidation8_allmembers_v1"
