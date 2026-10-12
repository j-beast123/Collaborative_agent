import pytest

import json

from scripts.online_rl.pilot_doctor_reflections import grounded_reflection,reflection_sections


def test_private_sections_do_not_accept_a_new_decision():
    text='What helped: None identified.\nWhat hindered: Repeating agreement.\nWhat I would have wanted: A premise check.'
    assert reflection_sections(text)['What I would have wanted']=='A premise check.'
    for invalid in ('Discussion: Final answer: A.',text+'\nSTATUS: Resolved.',text.replace('What hindered:','Discussion:')):
        with pytest.raises(ValueError):reflection_sections(invalid)


def test_grounded_reflection_uses_only_exact_assistant_passages():
    t={'dialogue':[{'speaker':'doctor','stage':'D1','text':'A premise.'},
                  {'speaker':'agent','stage':'A1','text':'I agree without checking.'}]}
    value={'helpful':{'evidence_id':'none','reflection':'None identified'},
           'hindering':{'evidence_id':'a1_p1','reflection':'The assistant did not check my premise.'},
           'wished_for':{'evidence_id':'a1_p1','reflection':'I would have wanted a specific premise check.'}}
    text,_,evidence=grounded_reflection(json.dumps(value),t)
    assert evidence['hindering']['quote']=='I agree without checking.'
    assert evidence['helpful'] is None
    value['hindering']['evidence_id']='doctor_turn_1'
    with pytest.raises(ValueError,match='assistant passage'):grounded_reflection(json.dumps(value),t)


def test_whole_turn_evidence_preserves_complete_speech():
    from scripts.online_rl.pilot_doctor_reflections import reflection_evidence
    text='The lack of other symptoms makes a direct cause more likely. '*10
    t={'dialogue':[{'speaker':'agent','stage':'A1','text':text}]}
    assert reflection_evidence(t,True)=={'a1':{'assistant_turn':1,'quote':text}}
