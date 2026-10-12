import pytest

from src.online_rl.sherpa.behavior_analysis import (
    associations, decode_annotations, dialogue_view, evidence_passages, validate_annotations)


def test_dialogue_view_hides_final_and_only_numbers_assistant_turns():
    trajectory={'outcome':1,'dialogue':[
        {'speaker':'doctor','stage':'D1','text':'A premise.'},
        {'speaker':'agent','stage':'A1','text':'Check that premise.'},
        {'speaker':'doctor','stage':'FINAL','text':'Final answer: correct option'}]}
    view=dialogue_view(trajectory)
    assert view==[{'speaker':'doctor','text':'A premise.'},
                  {'speaker':'agent','text':'Check that premise.','assistant_turn':1}]


def test_annotation_cannot_borrow_physician_or_other_dialogue_evidence():
    dialogues={'one':[{'speaker':'doctor','text':'Check that premise.'},
                     {'speaker':'agent','assistant_turn':1,'text':'I agree.'}]}
    annotation={'annotations':[{'dialogue_id':'one','behaviors':[
        {'tag':'contradiction_flag','assistant_turn':1,'quote':'Check that premise.'}]}]}
    with pytest.raises(ValueError,match='exactly quote'):
        validate_annotations(annotation,dialogues)
    annotation['annotations'][0]['behaviors'][0]={'tag':'agreement_only','assistant_turn':1,'quote':'I agree.'}
    assert validate_annotations(annotation,dialogues)


def test_repetition_and_agreement_only_constraints():
    dialogues={'one':[{'speaker':'agent','assistant_turn':1,'text':'I agree. Compare the alternatives.'}]}
    row={'dialogue_id':'one','behaviors':[{'tag':'repeated_agreement','assistant_turn':1,'quote':'I agree.'}]}
    with pytest.raises(ValueError,match='later assistant turn'):
        validate_annotations({'annotations':[row]},dialogues)
    row['behaviors']=[{'tag':tag,'assistant_turn':1,'quote':'Compare the alternatives.'}
                      for tag in ('agreement_only','alternative_comparison')]
    with pytest.raises(ValueError,match='contradicts'):
        validate_annotations({'annotations':[row]},dialogues)


def test_case_adjustment_detects_confounded_raw_association():
    rows=[]
    def add(cid,tagged,successes,failures):
        for outcome,count in ((1,successes),(0,failures)):
            rows.extend({'case_id':cid,'outcome':outcome,'tags':['alternative_comparison'] if tagged else []}
                        for _ in range(count))
    add('easy',True,9,1);add('easy',False,1,0)
    add('hard',True,0,1);add('hard',False,1,9)
    add('uniform',True,1,0);add('uniform',False,1,0)
    stat=next(r for r in associations(rows) if r['tag']=='alternative_comparison')
    assert stat['tagged_success_rate']>stat['untagged_success_rate']
    assert stat['within_case_mean_difference']==pytest.approx(-.2/3)
    assert stat['mixed_within_case_mean_difference']==pytest.approx(-.1)
    assert stat['within_case_count']==3 and stat['mixed_within_case_count']==2
    assert stat['mantel_haenszel_odds_ratio']==0


def test_evidence_ids_resolve_to_exact_original_assistant_fragments():
    text='This is a long assistant explanation. '*20
    dialogue=[{'speaker':'doctor','text':'A different claim.'},
              {'speaker':'agent','assistant_turn':1,'text':text}]
    spans=evidence_passages(dialogue)
    assert spans and all(s['quote'] in text and len(s['quote'])<=160 for s in spans.values())
    raw={'annotations':[{'dialogue_id':'one','behaviors':[{'tag':'alternative_comparison','evidence_id':'a1_p2'}]}]}
    decoded=decode_annotations(raw,{'one':dialogue})
    assert decoded['annotations'][0]['behaviors'][0]['quote']==spans['a1_p2']['quote']
    raw['annotations'][0]['behaviors'][0]['evidence_id']='a99_p1'
    with pytest.raises(ValueError,match='must belong'):
        decode_annotations(raw,{'one':dialogue})


def test_sparse_review_cannot_silently_add_or_drop_candidates():
    from scripts.online_rl.verify_behavior_tags import apply_review
    row={'tags':['repeated_agreement','contradiction_flag'],'evidence':[
        {'tag':'repeated_agreement','quote':'I agree.','assistant_turn':2},
        {'tag':'contradiction_flag','quote':'A different statement.','assistant_turn':1}]}
    result=apply_review(row,[],['contradiction_flag'])
    assert result['tags']==['repeated_agreement'] and row['tags']!=result['tags']
    with pytest.raises(ValueError,match='exactly once'):apply_review(row,[],[])
    with pytest.raises(ValueError,match='exactly once'):apply_review(row,[{'tag':'unverified_fact_assertion'}],['contradiction_flag'])


def test_direct_contrast_does_not_invent_a_within_case_comparison():
    from src.online_rl.sherpa.behavior_analysis import direct_tag_contrast
    rows=[{'case_id':'first','outcome':1,'tags':['contradiction_flag']},
          {'case_id':'first','outcome':0,'tags':[]},
          {'case_id':'second','outcome':1,'tags':['agreement_only']}]
    r=direct_tag_contrast(rows)
    assert r['mixed_comparable_cases']==0 and r['mixed_case_mean_difference'] is None
    assert r['first_success_rate']==r['second_success_rate']==1
