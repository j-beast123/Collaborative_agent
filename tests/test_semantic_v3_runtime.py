import copy
from pathlib import Path
from src.online_rl.tree.experiment import load_config
from src.online_rl.tree.luna import LunaDecisionClient
from scripts.online_rl.run_tree import blockers,make_scorer,make_gate,evidence_gate_identity
ROOT=Path(__file__).resolve().parents[1]
def config(v):return load_config(ROOT/('configs/experiments/quality_public_mc_semantic_v3.json' if v==3 else 'configs/experiments/quality_public_mc_isolated.json'))
def client(c,role):
 j=c['judges'];return LunaDecisionClient(model=j[role+'_model'],key='',system=(ROOT/j[role+'_prompt_file']).read_text(),max_tokens=j[role+'_max_tokens'],post=lambda *_a,**_k: (_ for _ in ()).throw(AssertionError('Offline test must not call API')))
def test_v3_real_config_resolves_without_blockers():assert blockers(config(3))==[]
def test_v3_gate_identity_equals_actual_factory():
 c=config(3);assert evidence_gate_identity(c)==make_gate(c,client(c,'gate'),[]).settings
def test_versioned_judges_have_distinct_cache_identities():
 a,b=config(1),config(3);x,y=make_scorer(a,client(a,'scorer')),make_scorer(b,client(b,'scorer'))
 assert x.protocol!=y.protocol
 assert x.review_identity!=y.review_identity
 assert evidence_gate_identity(a)!=evidence_gate_identity(b)
def test_mixed_v3_generation_and_v1_judges_fail_before_launch():
 c=config(3);c['judges']['gate_protocol']='public_mc_gate_v1'
 assert any('v3' in x for x in blockers(c))
