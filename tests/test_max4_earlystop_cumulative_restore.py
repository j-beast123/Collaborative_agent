from src.online_rl.tree.experiment import load_config


CONFIG = "configs/experiments/train120_max4_earlystop_exact_cumulative_restore_v1.json"


def test_restored_run_keeps_max4_early_stop_and_exact_only_bank_contract():
    cfg = load_config(CONFIG)

    assert cfg["rollout"]["rounds"] == 4
    assert cfg["rollout"]["halting"] == {
        "enabled": True,
        "min_agent_turns": 1,
        "repeat_agent_reply_after_turns": 4,
        "default_final_after_agent_turns": 2,
    }

    assert cfg["rubric"]["assignment_protocol"] == "exact_generated_source_case_v1"
    bank = cfg["rubric"]["cumulative_bank"]
    assert bank["protocol"] == "cumulative_keep_bank_v1"
    assert bank["scoring_scope"] == "all"
    assert cfg["policy"]["init_adapter"] is None
    assert cfg["supervision_handoff"] is None

    # The stable public M/C path remains one version end-to-end. Semantic
    # consolidation is not introduced by this restore configuration.
    assert cfg["judges"]["gate_protocol"] == "public_mc_gate_v3"
    assert cfg["judges"]["scorer_protocol"] == "public_mc_evidence_v3"
    assert cfg["judges"]["compact_runtime"] == "v7_5_v7_6"
