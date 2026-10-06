import json
from pathlib import Path
from src.online_rl.tree.checkpoints import save_periodic_snapshot


def test_periodic_model_prompt_and_optimizer_are_saved_every_30_cases(tmp_path):
    source=tmp_path/'checkpoints/recovery/case_000030'
    source.mkdir(parents=True)
    (source/'adapter').mkdir();(source/'adapter/weights').write_text('weights')
    (source/'trainer_state.pt').write_text('optimizer')
    state={'case_cursor':29,'checkpoint':str(source),'policy_sha256':'policy','prompt_sha256':'prompt'}
    assert save_periodic_snapshot(tmp_path,state,30,'P0') is None
    state['case_cursor']=30
    result=save_periodic_snapshot(tmp_path,state,30,'P0')
    assert result==tmp_path/'checkpoints/periodic/case_000030'
    assert (result/'model/trainer_state.pt').read_text()=='optimizer'
    assert (result/'rubric_generator_prompt.txt').read_text()=='P0'
    assert json.loads((result/'run_state.json').read_text())==state
    assert save_periodic_snapshot(tmp_path,state,30,'P0')==result
    assert json.loads((result.parent/'latest.json').read_text())['completed_cases']==30
    state['case_cursor']=50
    assert save_periodic_snapshot(tmp_path,state,30,'P0') is None
    state['case_cursor']=60
    assert save_periodic_snapshot(tmp_path,state,30,'P0').name=='case_000060'


def test_training_config_uses_30_cases():
    root=Path(__file__).resolve().parents[1]
    assert json.loads((root/'configs/online_rl/tree_v1.json').read_text())['checkpoint']['every_cases']==30
