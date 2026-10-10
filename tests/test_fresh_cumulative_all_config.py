"""A fresh full-bank experiment must not silently resume a trained policy/bank."""
import copy

from scripts.online_rl import run_five_gpu
from src.online_rl.tree import cumulative_run
from src.online_rl.tree.experiment import load_config


CONFIG = 'configs/experiments/train120_max2_fixed_p0_v23_cumulative_all_fresh.json'
BASE = 'configs/experiments/train120_max2_fixed_p0_v23.json'
OLD_RUN = 'configs/experiments/train120_max2_fixed_p0_v23_cumulative_bank.json'


def test_fresh_base_empty_bank_and_distinct_output():
    cfg = load_config(CONFIG)
    prior = load_config(OLD_RUN)
    base = load_config(BASE)
    assert cfg['policy']['base_model'] == 'Qwen/Qwen3-8B'
    assert cfg['policy']['init_adapter'] is None
    assert cfg.get('supervision_handoff') is None
    spec = cfg['rubric']['cumulative_bank']
    assert spec['scoring_scope'] == 'all'
    assert spec['retrieval_top_k'] == 8  # Pre-generation reuse search only.
    assert not spec.get('initial_snapshot_file')
    assert not spec.get('initial_snapshot_sha256')
    bank = cumulative_run.initial_snapshot(cfg, run_five_gpu.ROOT)
    assert bank['entries'] == [] and bank['version'] == 0
    assert cfg['output_dir'] not in (prior['output_dir'], base['output_dir'])
    assert cfg['experiment']['output_root'] in cfg['output_dir']
    assert cfg['schedule']['cases'] == 120
    assert cfg['rollout']['rounds'] == 2
    assert cfg['five_gpu']['batch_cases'] == 5
    assert cfg['five_gpu']['checkpoint_every_cases'] == 30
    assert cfg['agent_reward']['w_outcome'] == cfg['agent_reward']['w_rubric'] == 0.5


def test_fresh_scoring_change_preserves_existing_experiment_contracts():
    cfg, base = load_config(CONFIG), load_config(BASE)
    for section in ('policy', 'data', 'schedule', 'optimizer', 'advantage_normalization',
                    'feedback', 'judges', 'intervention', 'verification', 'agent_reward',
                    'prompt_update'):
        assert cfg[section] == base[section]
    rubric = copy.deepcopy(cfg['rubric'])
    rubric.pop('cumulative_bank')
    assert rubric == base['rubric']
    rollout = copy.deepcopy(cfg['rollout'])
    rollout['doctor_backend']['endpoint'] = base['rollout']['doctor_backend']['endpoint']
    assert rollout == base['rollout']
    five_gpu = copy.deepcopy(cfg['five_gpu'])
    five_gpu['doctor_urls'] = base['five_gpu']['doctor_urls']
    assert five_gpu == base['five_gpu']
    assert not set(cfg['five_gpu']['doctor_urls']) & set(base['five_gpu']['doctor_urls'])
    assert run_five_gpu.preflight(cfg) == []
