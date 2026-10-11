from dataclasses import dataclass

import pytest

from src.online_rl.tree.trainer import sample_rubric_sources, RUBRIC_SOURCE_SAMPLING_PROTOCOL


@dataclass(frozen=True)
class StubLeaf:
    leaf_id: str


CFG = {'protocol': RUBRIC_SOURCE_SAMPLING_PROTOCOL, 'threshold': 16, 'max_trajectories': 8}


def leaves(n):
    return [StubLeaf(f'case/{i:03d}') for i in range(n)]


def test_at_or_below_threshold_uses_every_leaf():
    selected, audit = sample_rubric_sources(leaves(16), CFG, run_seed=7, case_id='case')
    assert [x.leaf_id for x in selected] == [x.leaf_id for x in leaves(16)]
    assert audit['eligible'] == audit['selected'] == 16


def test_above_threshold_selects_exactly_eight_deterministically_and_order_independently():
    source = leaves(48)
    first, audit = sample_rubric_sources(source, CFG, run_seed=7, case_id='case')
    second, again = sample_rubric_sources(list(reversed(source)), CFG, run_seed=7, case_id='case')
    assert first == second
    assert audit == again
    assert len(first) == len(set(x.leaf_id for x in first)) == 8
    assert audit['eligible'] == 48 and audit['selected'] == 8
    assert audit['selected_leaf_ids'] == [x.leaf_id for x in first]


def test_seed_or_case_identity_changes_sample_and_invalid_contract_fails():
    source = leaves(48)
    a, _ = sample_rubric_sources(source, CFG, run_seed=7, case_id='case-a')
    b, _ = sample_rubric_sources(source, CFG, run_seed=8, case_id='case-a')
    c, _ = sample_rubric_sources(source, CFG, run_seed=7, case_id='case-b')
    assert {x.leaf_id for x in a} != {x.leaf_id for x in b}
    assert {x.leaf_id for x in a} != {x.leaf_id for x in c}
    with pytest.raises(ValueError, match='source_sampling'):
        sample_rubric_sources(source, {**CFG, 'max_trajectories': 0}, run_seed=0, case_id='case')
