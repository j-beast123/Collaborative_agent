import pytest
from pathlib import Path
from src.online_rl.clean.halting import halt_reason
from src.online_rl.doctor import OpenAIDoctor
from src.online_rl.tree.experiment import load_config

ST = {'current_option': 'A', 'open_issue': False, 'collaborator_objection': False,
      'more_info_could_change_decision': False}


def decision(status=ST, prev=ST, turns=2):
    return halt_reason(agent_turns=turns, status=status, prev_status=prev, options=['A', 'B'])


def test_only_resolved_stable_status_halts_after_two_agent_turns():
    assert decision() == 'doctor_halt'
    assert decision(turns=1) is None
    assert decision(turns=0) is None
    assert decision(prev={**ST, 'open_issue': True}) == 'doctor_halt'
    assert decision(prev={**ST, 'current_option': 'B'}) is None


@pytest.mark.parametrize('flag', ['open_issue', 'collaborator_objection', 'more_info_could_change_decision'])
def test_each_unresolved_flag_blocks_even_late(flag):
    assert decision(status={**ST, flag: True}, turns=4) is None


@pytest.mark.parametrize('bad', [None, {}, {**ST, 'current_option': 'none'},
                                 {**ST, 'current_option': 'C'}, {**ST, 'open_issue': 0},
                                 {**ST, 'extra': False}])
def test_both_status_objects_must_be_valid(bad):
    assert decision(status=bad) is None
    assert decision(prev=bad) is None


def test_active_online_doctor_ignores_agent_option_without_relaxing_flags():
    cfg = load_config('configs/online.json')
    ro = cfg['rollout']
    root = Path(__file__).resolve().parents[1]
    doctor = OpenAIDoctor(ro['doctor_backend'], ro['halting'],
                          (root/ro['doctor_system_prompt_file']).read_text(), ro['max_tokens_doctor'])
    assert doctor.min_agent_turns == 2
    kw = dict(agent_turns=2, status=ST, prev_status=ST, last_agent='I choose option B.', options=['A','B'])
    assert doctor.halt(**kw) == 'doctor_halt'
    assert doctor.halt(**{**kw, 'agent_turns': 1}) is None
    assert doctor.halt(**{**kw, 'status': {**ST, 'collaborator_objection': True}}) is None
    assert 'halting_rule_sha256' in doctor.identity
