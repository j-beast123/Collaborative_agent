from src.online_rl.tree import offline_chunk_bank as O,cumulative_run as C
from src.online_rl.tree.experiment import load_config
from pathlib import Path


def test_all_eligible_trajectories_include_every_leaf_above_48():
    ids=[f'case/{i:03}' for i in range(73)]
    selected,metadata=O.select(list(reversed(ids)),0,'case',cap=None)
    assert selected==ids and metadata['selected_ids']==ids
    assert metadata['cap'] is None and not metadata['sampled'] and metadata['not_selected_ids']==[]
    assert [len(chunk) for chunk in O.chunks(selected)]==[8]*9+[1]
    old,legacy=O.select(ids,0,'case')
    assert len(old)==48 and legacy['sampled']


def test_config_and_bank_identity_declare_unlimited_discovery():
    root=Path(__file__).resolve().parents[1]
    cfg=load_config('configs/online_openai20.json')
    identity=C.config_identity(cfg,root)
    assert identity['discovery_trajectory_cap'] is None
    assert identity['generation_assignment']=='case_chunk8_single_generator_call_all_eligible_v2'
    assert cfg['agent_reward']['w_rubric']==cfg['agent_reward']['w_outcome']==0.5
    assert load_config('configs/online_rubric_only_openai20.json')['rubric']['cumulative_bank']['discovery_trajectory_cap'] is None


def test_monitor_counts_all_eligible_targets(tmp_path):
    import json
    from src.common.bank_monitor import BankCounters
    (tmp_path/'bank_rollout.json').write_text(json.dumps({'feedback':[{'valid':True}]*73}))
    result=BankCounters().read(tmp_path,discovery_cap=None)
    assert result['selected']==73 and result['chunks_total']==10
