from pathlib import Path
import importlib.util


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("eval_checkpoint", ROOT / "scripts/online_rl/eval_checkpoint.py")
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


def test_packaged_artifacts_and_frozen_config():
    cfg, paths = MOD.prepared_config(ROOT / "artifacts/reported_checkpoint", "http://127.0.0.1:11535")
    assert paths["weights"].is_file()
    assert cfg["policy"]["base_revision"] == MOD.BASE_REVISION
    assert cfg["policy"]["init_adapter"] == str(ROOT / "artifacts/reported_checkpoint/adapter")
    assert cfg["rollout"]["rounds"] == 4
    assert cfg["rollout"]["halting"]["min_agent_turns"] == 2
    assert cfg["rollout"]["doctor_backend"]["model_digest"] == MOD.DOCTOR_DIGEST


def test_saved_case_result_counts():
    rows = (ROOT / "artifacts/reported_checkpoint/saved_evaluation/case_results.jsonl").read_text().splitlines()
    import json
    records = [json.loads(row) for row in rows]
    assert len(records) == len({r["case_id"] for r in records}) == 622
    assert sum(r["correct"] for r in records) == 442


def test_base_revision_changes_protocol_identity():
    import json
    from src.online_rl.integrated.continuation import protocol_sha256
    cfg = json.loads((ROOT / "configs/online_rl/paper_reproduction.json").read_text())
    pinned = protocol_sha256(cfg, ROOT)
    del cfg["policy"]["base_revision"]
    assert protocol_sha256(cfg, ROOT) != pinned
