import json
from pathlib import Path

import pytest

from scripts.diagnostics.restart_shared_pool import frozen_batch
from src.online_rl.integrated.records import sha256_json


@pytest.fixture
def frozen_run(tmp_path):
    cfg={'five_gpu':{'execution_mode':'shared_ready_tasks_v1'}}
    checkpoint=tmp_path/'checkpoint';checkpoint.mkdir()
    state={'case_cursor':0,'batch_cursor':0,'config_sha256':sha256_json(cfg),
           'policy_sha256':'theta0','checkpoint':str(checkpoint)}
    (checkpoint/'COMPLETE.json').write_text(json.dumps({'completed_cases':0,'policy_sha256':'theta0'}))
    (tmp_path/'run_state.json').write_text(json.dumps(state))
    batch=tmp_path/'batches/batch_000001';case=batch/'case_000000';case.mkdir(parents=True)
    (batch/'manifest.json').write_text(json.dumps({'source_policy_sha256':'theta0','assignments':[{'index':0}]}))
    (tmp_path/'train.log').write_text('loading model\n'+json.dumps({'stage':'prompt_P0'})+'\n')
    return cfg,tmp_path,case,state


def test_restart_preserves_same_pending_batch_and_checkpoint(frozen_run):
    cfg,run,case,state=frozen_run
    assert frozen_batch(cfg,run)==(state,[case])


@pytest.mark.parametrize('phase',['P1.json','union.json'])
def test_restart_refuses_later_phase(frozen_run,phase):
    cfg,run,case,_=frozen_run
    (case/phase).write_text('{}')
    with pytest.raises(RuntimeError,match='unfinished P0'):frozen_batch(cfg,run)


def test_restart_refuses_update_and_bad_checkpoint(frozen_run):
    cfg,run,_,state=frozen_run
    (run/'train.log').write_text(json.dumps({'stage':'agent_update'})+'\n')
    with pytest.raises(RuntimeError,match='frozen P0'):frozen_batch(cfg,run)
    (run/'train.log').write_text(json.dumps({'stage':'prompt_P0'})+'\n')
    checkpoint=Path(state['checkpoint'])
    (checkpoint/'COMPLETE.json').write_text(json.dumps({'completed_cases':5,'policy_sha256':'theta1'}))
    with pytest.raises(RuntimeError,match='Checkpoint'):frozen_batch(cfg,run)
