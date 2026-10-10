"""A saved joint bank must contain each member's own frozen execution contract."""
import copy

import pytest

from src.online_rl.integrated.records import to_dict
from src.online_rl.tree.cache import CallCache
from src.online_rl.tree.member_intervention import MEMBER_INTERVENTION_VERSION
from src.online_rl.tree.records import Tree
from src.online_rl.tree.trainer import TreeTrainer
from test_member_intervention_bank import CASE, PROMPT, functions, sources
from src.online_rl.tree import rubric_bank


def saved_bank():
    leaves, nodes, tins = sources()
    generate, match, _ = functions(False)
    canon, assignments = rubric_bank.build_banks(CASE, leaves, tins, PROMPT, 'prompt', generate, match, None, CallCache(None))
    trainer = object.__new__(TreeTrainer)
    trainer.reuse = True
    trainer.member_intervention_version = MEMBER_INTERVENTION_VERSION
    record = {'tree': to_dict(Tree(CASE['case_id'], 'policy', 2, 4, tuple(nodes.values()), tuple(leaves))),
              'feedback': [{'leaf_id': leaf.leaf_id, 'valid': True} for leaf in leaves],
              'trajectory_rubrics': [to_dict(r) for r in canon.values()],
              'rubric_bank': {'member_intervention_version': MEMBER_INTERVENTION_VERSION,
                             'assignments': [to_dict(a) for a in assignments],
                             'verification_pending': False, 'witness': []}}
    return trainer, record


def test_completed_new_bank_can_resume_without_regeneration():
    trainer, record = saved_bank()
    before = copy.deepcopy(record)
    trainer.validate_member_interventions(record)
    assert trainer.finish_deferred_verification(record) is record
    assert record == before


@pytest.mark.parametrize('missing', ['version', 'instruction', 'spec'])
def test_saved_phase_rejects_old_or_incomplete_joint_contract_even_when_marked_complete(missing):
    trainer, record = saved_bank()
    if missing == 'version': record['rubric_bank'].pop('member_intervention_version')
    elif missing == 'instruction': record['trajectory_rubrics'][0]['intervention_instruction'] = None
    else: record['rubric_bank']['assignments'][1].pop('intervention_spec')
    with pytest.raises((RuntimeError, ValueError), match='instruction|intervention_spec'):
        trainer.finish_deferred_verification(record)


def test_rollout_protocol_separates_member_instructions_from_old_joint_generation():
    from scripts.online_rl.run_tree import ROOT
    from src.online_rl.tree.experiment import load_config
    from src.online_rl.integrated.continuation import protocol_sha256
    CONFIG = 'configs/experiments/train120_max2_fixed_p0.json'
    cfg = load_config(CONFIG)
    previous = copy.deepcopy(cfg)
    previous['rubric']['generation']['output_contract'] = 'trajectory_analysis_candidate_v2'
    assert protocol_sha256(cfg, ROOT) != protocol_sha256(previous, ROOT)


@pytest.mark.parametrize('change', ['omit_member', 'duplicate_member', 'omit_feedback'])
def test_completed_bank_requires_full_member_coverage(change):
    trainer, record = saved_bank()
    if change == 'omit_member': record['rubric_bank']['assignments'].pop()
    elif change == 'duplicate_member': record['rubric_bank']['assignments'].append(record['rubric_bank']['assignments'][0])
    else: record['feedback'].pop()
    with pytest.raises(RuntimeError, match='exactly once'):
        trainer.finish_deferred_verification(record)
