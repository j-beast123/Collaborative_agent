from dataclasses import replace
from pathlib import Path
import copy
import json
import pytest
from src.common.config import load_config, require_execution_ready, IntegrationUnavailable
from src.data.schema import DerivedRecord, digest
from src.data.registry import DatasetRegistry, AnswerEvaluator
from src.data.synthetic import SyntheticFixtureAdapter
from src.data.split import make_manifest, read_manifest, write_manifest, validate_manifest, validate_lineage
from src.training.backend import TrainingBackend
from src.effectiveness.providers import build_effectiveness_provider
from src.evaluation.figure_contract import COMMON, EXTRA, validate_rows, export_rows

ROOT=Path(__file__).resolve().parents[1]
@pytest.fixture
def cases():
    return SyntheticFixtureAdapter(ROOT/"tests/fixtures/synthetic/cases.json").load()

def manifest(cases, **kw):
    return make_manifest(cases,seed=kw.pop("seed",17),proportions=kw.pop("proportions",{"train":.6,"dev":.2,"eval":.2}),**kw)

def test_order_group_alias_and_namespaces(cases):
    cases += [replace(cases[0],case_id="variant"),replace(cases[1],source_dataset="other")]
    aliases=[[cases[0].original_key,cases[-1].original_key]]
    m=manifest(cases,duplicate_groups=aliases)
    assert m==manifest(reversed(cases),duplicate_groups=list(reversed(aliases)))
    assert len({m["members"][c.key]["split"] for c in (cases[0],cases[-1],cases[-2])})==1
    assert cases[1].key != cases[-1].key
    validate_manifest(m,cases)

def test_resume_and_tamper(cases,tmp_path):
    m=manifest(cases);p=tmp_path/"manifest.json"
    write_manifest(p,m);write_manifest(p,m)
    assert read_manifest(p,cases)==m
    with pytest.raises(ValueError):write_manifest(p,manifest(cases,seed=4))
    with pytest.raises(ValueError):read_manifest(p,[replace(cases[0],question="changed")]+cases[1:])
    m["seed"]=99
    with pytest.raises(ValueError):validate_manifest(m)

@pytest.mark.parametrize("props",[{"train":.8,"dev":.1,"eval":.2},{"train":1.,"dev":0.,"eval":0.},{"train":float("nan"),"dev":.2,"eval":.2}])
def test_bad_ratios(cases,props):
    with pytest.raises(ValueError):manifest(cases,proportions=props)

def test_small_duplicate_unknown_and_mixed(cases):
    for rows in (cases[:2],cases+[cases[0]],cases+[replace(cases[0],case_id="real",synthetic=False)]):
        with pytest.raises(ValueError):manifest(rows)
    with pytest.raises(ValueError):manifest(cases,duplicate_groups=[[cases[0].original_key,"missing"]])

def test_lineage(cases):
    m=manifest(cases);c=cases[0];s=m["members"][c.key]["split"]
    r=DerivedRecord("r",c.source_dataset,c.case_id,c.original_case_id,s,True)
    child=replace(r,record_id="child",parent_record_id="r")
    validate_lineage([r,child],m)
    for bad in (replace(child,split="dev" if s!="dev" else "train"),replace(child,original_case_id="missing"),replace(child,parent_record_id="missing"),replace(child,parent_record_id="child"),replace(child,synthetic=False)):
        with pytest.raises(ValueError):validate_lineage([r,bad],m)

def test_registry_and_missing_answer(cases):
    reg=DatasetRegistry();adapter=SyntheticFixtureAdapter(ROOT/"tests/fixtures/synthetic/cases.json")
    reg.register("fixture",adapter,synthetic=True)
    assert len(reg.load("fixture",production=False))==10
    with pytest.raises(ValueError):reg.load("fixture")
    with pytest.raises(ValueError):reg.register("fixture",adapter)
    with pytest.raises(IntegrationUnavailable):reg.load("missing")
    class Evaluator(AnswerEvaluator):
        def score(self,case,response):return True
    with pytest.raises(IntegrationUnavailable):Evaluator().evaluate(cases[0],"anything")

def test_guards_and_no_backend():
    cfg=load_config(ROOT/"configs/experiments/phase01.json")
    assert cfg["predictor"]["training_mode"]=="frozen_backbone_scalar_head"
    with pytest.raises(IntegrationUnavailable):require_execution_ready(cfg)
    backend=TrainingBackend()
    for method in (backend.train,backend.load_checkpoint,backend.generate_trajectories,backend.get_checkpoint_metadata):
        with pytest.raises(IntegrationUnavailable):method(None)
    with pytest.raises(IntegrationUnavailable):build_effectiveness_provider({"provider":None},0.3)

def row(kind="controlled"):
    r={k:"test" for k in COMMON+EXTRA[kind]}
    r.update(schema_version="1",split="eval",system="base",iteration=0,training_seed=1,response_seed=2,synthetic=True,expected_behavior_class="ACCEPT_REVISE",predicted_behavior_class="ACCEPT_REVISE",appropriateness_match=1,evaluation_status="scored",exclusion_reason=None)
    if kind=="controlled":r.update(condition_id="correct_strong",initial_state_stratum="incorrect",claim_correctness=True,evidence_strength="strong",validation_status="accepted")
    else:r.update(collaboration_state="Accept",agent_initial_correctness=False,collaborator_correctness=True)
    return r

@pytest.mark.parametrize("kind",["state","controlled"])
def test_figure_validation_and_production_guard(kind,tmp_path):
    r=row(kind);validate_rows([r],kind,synthetic=True)
    with pytest.raises(ValueError):export_rows([r],kind,tmp_path/"bad.csv")
    with pytest.raises(ValueError):export_rows([r],kind,ROOT/"analysis/figure_data/bad.csv",synthetic=True)
    with pytest.raises(ValueError):validate_rows([r,r],kind,synthetic=True)
    real=dict(r,synthetic=False)
    with pytest.raises(ValueError):validate_rows([real,r],kind)
    export_rows([real],kind,tmp_path/"real.csv")
    assert "synthetic" in (tmp_path/"real.csv").read_text()

def test_pair_mismatch_and_unscored():
    a=row();b=dict(a,system="serl",checkpoint="iter1",controlled_input_hash="changed")
    with pytest.raises(ValueError):validate_rows([a,b],"controlled",synthetic=True)
    b["controlled_input_hash"]=a["controlled_input_hash"]
    validate_rows([a,b],"controlled",synthetic=True)
    b.update(evaluation_status="failed",appropriateness_match=None,exclusion_reason="parse error",predicted_behavior_class=None)
    validate_rows([a,b],"controlled",synthetic=True)
    b["appropriateness_match"]=0
    with pytest.raises(ValueError):validate_rows([b],"controlled",synthetic=True)

@pytest.mark.parametrize("field,value",[("choices",[]),("evidence",{}),("metadata",[]),("language",7)])
def test_normalized_optional_types(cases,field,value):
    with pytest.raises(ValueError):replace(cases[0],**{field:value})

def test_rehashed_invalid_manifest_is_rejected(cases):
    for field,value in (("seed",True),("proportions",{"train":2,"dev":1,"eval":1}),("group_counts",{"train":10,"dev":0,"eval":0})):
        m=manifest(cases);m[field]=value;m.pop("manifest_hash");m["manifest_hash"]=digest(m)
        with pytest.raises(ValueError):validate_manifest(m)

def test_manifest_without_alias_list_is_value_error(cases):
    m = manifest(cases)
    m.pop("duplicate_groups")
    m.pop("manifest_hash")
    m["manifest_hash"] = digest(m)
    with pytest.raises(ValueError):
        validate_manifest(m)

def test_role_separation_and_jev_scorer_role(tmp_path):
    c=load_config(ROOT/"configs/experiments/phase01.json")
    assert all(v["role"]=="agent_backbone" for v in c["models"].values())
    assert c["services"]["rubric_generator"]["backend"]=="llm"
    for edit in ("agent","jev","predictor"):
        bad=copy.deepcopy(c)
        if edit=="agent":bad["models"]["llama"]["role"]="structured_scoring"
        elif edit=="jev":bad["optional_experiments"]["jev_judge"]["role"]="effectiveness_predictor"
        else:bad["predictor"]["role"]="agent_backbone"
        p=tmp_path/"bad.json";p.write_text(json.dumps(bad))
        with pytest.raises(ValueError):load_config(p)
