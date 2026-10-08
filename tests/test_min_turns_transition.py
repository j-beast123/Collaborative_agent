import json
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.diagnostics import transition_min_turns as migration
from src.online_rl.integrated.records import sha256_json


def test_transition_preserves_checkpoint_and_separates_eval(tmp_path,monkeypatch):
    monkeypatch.setattr(migration,'ROOT',tmp_path)
    old={'output_dir':'old','rollout':{'halting':{'min_agent_turns':3}},'five_gpu':{'opening_identity':{},'opening_mode':'x'},'schedule':{'cases':500}}
    new={**old,'output_dir':'new','rollout':{'halting':{'min_agent_turns':2}}}
    for name,cfg in [('old.json',old),('new.json',new)]:
        (tmp_path/name).write_text(json.dumps(cfg))
    source=tmp_path/'old';ckpt=source/'checkpoints/recovery_000004';ckpt.mkdir(parents=True)
    receipt={'completed_cases':4,'policy_sha256':'weights','protocol_sha256':'old-protocol'}
    (ckpt/'COMPLETE.json').write_text(json.dumps(receipt));(ckpt/'optimizer.pt').write_bytes(b'unchanged optimizer')
    state={'case_cursor':4,'batch_cursor':1,'config_sha256':sha256_json(old),'policy_sha256':'weights','protocol_sha256':'old-protocol',
           'checkpoint':str(ckpt),'prompt_sha256':'winner','prompt_id':'P1'}
    (source/'run_state.json').write_text(json.dumps(state))
    (source/'evaluation').mkdir();(source/'evaluation/old_result.json').write_text('{}')
    with patch('src.online_rl.integrated.continuation.protocol_sha256',return_value='new-protocol'):
        target=migration.transition('old.json','new.json')
    resumed=json.loads((target/'run_state.json').read_text())
    assert resumed['case_cursor']==4 and resumed['prompt_sha256']=='winner'
    assert Path(resumed['checkpoint']).joinpath('optimizer.pt').read_bytes()==b'unchanged optimizer'
    assert not (target/'evaluation').exists()
    assert json.loads((ckpt/'COMPLETE.json').read_text())==receipt
    assert json.loads((target/'protocol_transition.json').read_text())['remaining_cases_under_min2']==496
    with pytest.raises(RuntimeError,match='fresh'):
        migration.transition('old.json','new.json')
