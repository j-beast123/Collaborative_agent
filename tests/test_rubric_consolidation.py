from dataclasses import replace
import pytest
from tests.test_tree_selected_protocol import source
from src.online_rl.tree.consolidation import consolidate
from src.online_rl.tree.cache import CallCache


def test_existing_representative_and_safe_inputs():
    _,_,_,r=source();other=replace(r,rubric_id='other',leaf_id='other-leaf')
    def generate(system,payload):
        assert set(payload)=={'case_id','candidates'}
        assert all(set(c)=={'rubric_id','candidate_rubric'} for c in payload['candidates'])
        return {'groups':[{'representative_id':r.rubric_id,'member_ids':[r.rubric_id,'other'],'reason':'Same conditions and all levels'}]}
    result=consolidate('synthetic',[r,other],generate,CallCache(None))
    assert result['representatives'][0] is r
    assert result['originals']==[r,other]
    assert result['groups'][0]['source_targets'][1]['leaf_id']=='other-leaf'


@pytest.mark.parametrize('group',[
    {'representative_id':'invented','member_ids':['other'],'reason':'x'},
    {'representative_id':'other','member_ids':['other','other'],'reason':'x'},
    {'representative_id':'other','member_ids':['other','unknown'],'reason':'x'},
])
def test_invalid_clusters_rejected(group):
    _,_,_,r=source();other=replace(r,rubric_id='other')
    with pytest.raises(ValueError):consolidate('synthetic',[r,other],lambda *a:{'groups':[group]},CallCache(None))


def test_foreign_case_rejected_before_call():
    _,_,_,r=source();foreign=replace(r,rubric={**r.rubric,'source_case_id':'foreign'})
    with pytest.raises(ValueError):consolidate('synthetic',[foreign],lambda *a:pytest.fail('must not call'),CallCache(None))
