import hashlib
import json
from pathlib import Path


ROOT=Path(__file__).resolve().parents[1]


def read(path):
    return (ROOT/path).read_text()


def test_v3_generator_keeps_bank_contract_and_adds_outcome_conditioned_levels():
    old=read('prompts/sherpa/trajectory_rubric_generator_P0_aoc_v2.txt')
    new=read('prompts/sherpa/trajectory_rubric_generator_P0_aoc_v3.txt')
    marker='[Assign or create]'
    assert marker in old and marker in new
    assert old.split(marker,1)[1]==new.split(marker,1)[1]
    assert len(new.split())<900
    assert 'SUCCESS:' in new and 'FAILURE:' in new
    assert 'removes ONLY that helpful behavior' in new
    assert 'partial correction' in new
    assert '"user_side_context"' in new and '"agent_side_state"' in new


def test_v5_feedback_and_prompt_diagnosis_match_live_evidence_contract():
    feedback=read('prompts/sherpa/behavior_feedback_v5.txt')
    diagnosis=read('prompts/sherpa/tree_prompt_feedback_v3.txt')
    refiner=read('prompts/sherpa/tree_prompt_refiner_v2.txt')
    assert all(term in feedback for term in ('explicit request','handled that request','maintaining a position','not a claim of causal impact'))
    assert all(term in diagnosis for term in ('uniformly without replacement','target_reply_id','intervention_reply',
                                               'Doctor FINAL transcripts','NOT_VALIDATED'))
    assert 'still uses one of the sampled trajectory slots' in diagnosis
    assert '10 valid supports' not in diagnosis
    assert all(term in refiner for term in ('[Assign or create]','user_side_context','agent_side_state',
                                             'target_reply_id','under 900 words'))


def test_same_evidence_refiner_receives_diagnostic_bundle_and_stays_case_neutral():
    from src.online_rl.integrated.prompt_search import SearchBackend, _propose
    from src.online_rl.tree.cache import CallCache
    from src.online_rl.tree.prompt_search import cached_backend
    evidence=[{'target_reply_id':2,'rubric':{'scores':{'0':'behavior'}},
               'interventions':[{'requested_behavior':'general behavior','intervention_reply':'reply',
                                 'reruns_executed':2,'expected_reruns':3,'early_stopped_before_K':True,
                                 'rerun_outcomes':[1,1]}]}]
    seen={'feedback':None,'refiner':[]}
    def feedback(parent,view,n):
        seen['feedback']=view
        return {'diagnosis':'supported diagnosis','directions':['make level semantics more local']}
    def refine(parent,diagnosis,direction,history,note,evidence_view):
        seen['refiner'].append(evidence_view)
        return parent.replace('VARIANT=0','VARIANT=1')
    backend=SearchBackend(feedback,lambda *args:'legacy path',None,None,None,refine)
    result=_propose('Rubric generator with "name" "applicability_condition" "required_behavior" and "scores". VARIANT=0',
                    evidence,1,[],backend,[],[])
    assert result['proposals'][0]['text'].endswith('VARIANT=1')
    assert seen['feedback'] is evidence and seen['refiner']==[evidence]

    calls=[]
    cached=cached_backend(SearchBackend(feedback,lambda *args:'legacy path',None,None,None,
                                       lambda *args:calls.append(args[-1]) or 'new'),CallCache(None))
    cached.refine_with_evidence('P','D','direction',[],None,evidence)
    cached.refine_with_evidence('P','D','direction',[],None,evidence)
    other=[{'target_reply_id':1}]
    cached.refine_with_evidence('P','D','direction',[],None,other)
    assert calls==[evidence,other]
    assert 'same evidence' in read('prompts/sherpa/tree_prompt_refiner_v3.txt')
    assert 'do not copy or encode any case-specific clinical fact' in read('prompts/sherpa/tree_prompt_refiner_v3.txt')
    assert 'There is no arbitrary word-count cap' in read('prompts/sherpa/tree_prompt_refiner_v3.txt')
    assert 'Never replace this with the legacy flat applicability_condition schema' in read('prompts/sherpa/tree_prompt_refiner_v3.txt')


def test_promptv3_config_hashes_and_isolated_output_dir():
    config=json.loads(read('configs/online_rl/tree_v21_five_gpu_shared5_random10_promptv3.json'))
    prior=json.loads(read('configs/online_rl/tree_v21_five_gpu_shared5_random10_reuse_active.json'))
    assert config['output_dir']!=prior['output_dir']
    for section,field,digest in (
        ('rubric','generator_prompt_file','prompt_sha256'),('feedback','prompt_file','prompt_sha256'),
        ('prompt_update','feedback_prompt_file','feedback_prompt_sha256'),
        ('prompt_update','refiner_prompt_file','refiner_prompt_sha256')):
        assert config[section][digest]==hashlib.sha256(read(config[section][field]).encode()).hexdigest()
    assert config['feedback']['protocol']=='behavior_observer_v5'
    assert config['rubric']['state_addendum_sha256']==hashlib.sha256(read(config['rubric']['state_addendum_file']).encode()).hexdigest()


def test_promptv4_config_pins_same_evidence_refiner_and_fresh_output():
    config=json.loads(read('configs/online_rl/tree_v21_five_gpu_shared5_random10_promptv4.json'))
    previous=json.loads(read('configs/online_rl/tree_v21_five_gpu_shared5_random10_promptv3.json'))
    file=config['prompt_update']['refiner_prompt_file']
    assert config['output_dir']!=previous['output_dir']
    assert file=='prompts/sherpa/tree_prompt_refiner_v3.txt'
    assert config['prompt_update']['refiner_prompt_sha256']==hashlib.sha256(read(file).encode()).hexdigest()
    assert config['prompt_update']['reflection_model']=='anthropic/claude-sonnet-5.5-20260928'
    assert config['prompt_update']['refiner_model']=='anthropic/claude-sonnet-5.5-20260928'
    assert config['five_gpu']['rollout_gpus']==[0,1,2,3,4]


def test_prompt_guard_accepts_both_flat_and_observed_context_rubric_contracts():
    from src.prompt_opt.reflect import structural_problems
    flat='"name" "applicability_condition" "required_behavior" "scores"'
    observed=('"name" "applicability" "user_side_context" "agent_side_state" '
              '"required_behavior" "scores"')
    assert structural_problems(flat)==[]
    assert structural_problems(observed)==[]
    assert 'missing output key "applicability_condition"' in structural_problems(
        '"name" "required_behavior" "scores"')
    assert 'missing observed-context output key "agent_side_state"' in structural_problems(
        '"name" "applicability" "user_side_context" "required_behavior" "scores"')


def test_observed_tree_prompt_guard_requires_both_sides_and_has_no_word_cap():
    from src.prompt_opt.reflect import structural_problems
    flat='"name" "applicability_condition" "required_behavior" "scores"'
    observed=('{"name":"x","applicability":{"user_side_context":"u",'
              '"agent_side_state":"a"},"required_behavior":"r","scores":{'
              '"0":{"user_side_context":"u","agent_side_state":"a","behavior":"b0"},'
              '"1":{"user_side_context":"u","agent_side_state":"a","behavior":"b1"},'
              '"2":{"user_side_context":"u","agent_side_state":"a","behavior":"b2"}}}')
    assert 'missing output key "applicability"' in structural_problems(flat, require_observed_context=True)
    assert structural_problems(observed, require_observed_context=True)==[]
    assert structural_problems(observed + (' extra' * 1200), require_observed_context=True)==[]


def test_candidate_guard_allows_targeted_reuse_edit_but_preserves_all_bank_contracts():
    from src.online_rl.tree.batch_prompt import _candidate_section_problem
    parent=read('prompts/sherpa/trajectory_rubric_generator_P0_aoc_v3.txt')
    suffix=parent.split('[Assign or create]',1)[1]
    updated=suffix.replace(
        'reinterpretation;',
        'reinterpretation; Compare the target reply and feedback against each level as written;')
    candidate=parent.split('[Assign or create]',1)[0]+'[Assign or create]'+updated
    assert _candidate_section_problem(candidate,parent,'tighten the reuse check') is None
    assert 'outside a reuse-focused direction' in _candidate_section_problem(candidate,parent,'clarify score levels')
    broken=candidate.replace('same_construct:','same_construct_removed:')
    assert 'same_construct:' in _candidate_section_problem(broken,parent,'tighten the reuse check')
    legacy=parent.replace('"applicability"','"applicability_condition"')
    assert 'observed-context rubric schema' in _candidate_section_problem(legacy,parent,'tighten the reuse check')
