"""Five-question, frozen-Agent prompt pilot: four optimization cases, one held-out comparison."""
import argparse
import json
import os
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.online_rl import run_tree as RT
from src.online_rl.integrated.records import atomic_write_json,sha256_json,to_dict


def preflight(cfg,config_path='configs/online_rl/tree_pilot5.json'):
    problems=RT.blockers(cfg)+RT.setup_problems(cfg,config_path)
    if not cfg.get('pilot',{}).get('prompt_only'):
        problems.append('Pilot requires prompt_only=true')
    heldout=ROOT/cfg['data']['heldout']
    if not heldout.is_file(): problems.append('held-out file missing')
    try:
        from src.online_rl.cases import load_cases
        train=load_cases(ROOT/cfg['data']['train']); test=load_cases(heldout)
        from src.online_rl.guards import question_sha256,assert_not_protected
        assert_not_protected(train+test,'pilot data')
        if len(train)!=4 or len(test)!=1: problems.append('Pilot requires exactly four optimization and one held-out questions')
        if {question_sha256(c['question']) for c in train}&{question_sha256(c['question']) for c in test}:
            problems.append('Optimization/held-out overlap')
    except (OSError,ValueError,RuntimeError) as exc:
        problems.append(str(exc))
    try:
        RT.load_key()
    except (OSError,KeyError):
        problems.append('OPENROUTER_API_KEY not configured')
    try:
        if cfg.get('pilot',{}).get('backend') == 'ollama':
            import urllib.request
            with urllib.request.urlopen(cfg['pilot']['ollama_url']+'/api/tags',timeout=10) as response:
                models=json.load(response)['models']
            if not any(m['name']==cfg['pilot']['ollama_model'] for m in models): problems.append('Ollama model missing')
            if cfg.get('doctor_backend', {}).get('backend') == 'ollama':
                from src.online_rl.tree.ollama_doctor import OllamaDoctorCall
                OllamaDoctorCall(cfg['doctor_backend'])
        else:
            import torch
            if not torch.cuda.is_available(): problems.append('CUDA unavailable in this execution environment')
            import transformers,peft
    except Exception as exc:
        problems.append(f'Policy backend: {type(exc).__name__}: {str(exc)[:180]}')
    return problems


def compare_heldout(tr):
    from src.online_rl.cases import load_cases
    from src.online_rl.tree.rollout import build_tree
    from src.online_rl.tree.posthoc import doctor_feedback,freeze_trajectory_input
    from src.online_rl.tree.prompt_search import EvidenceUnit,ScoringDeps,score_prompt,unit_values,objective,evidence_id
    case=load_cases(ROOT/tr.cfg['data']['heldout'])[0]; env=tr.env
    policy=tr.state.policy_sha256
    tree=build_tree(case,max_agent_turns=tr.cfg['rollout']['rounds'],branching=tr.cfg['rollout']['branching'],
                    run_seed=tr.state.run_seed,policy_sha256=policy,protocol_sha256=tr.protocol_sha,
                    agent=env.agent,doctor=env.doctor,stop_rule=env.stop_rule,final_option=env.final_option,
                    outcome_fn=env.outcome_fn,cache=tr.cache)
    nodes={n.node_id:n for n in tree.nodes}; units=[]; feedback=[]
    for leaf in tree.leaves:
        if leaf.outcome is None: continue
        fb=doctor_feedback(case,leaf,env.feedback_fn,tr.fb_sha,tr.cache);feedback.append(to_dict(fb))
        tin=freeze_trajectory_input(case,leaf,fb,nodes=nodes)
        units.append(EvidenceUnit(case['case_id'],leaf,tuple(nodes[n] for n in leaf.path),tin,policy))
    deps=ScoringDeps(3,tr.state.run_seed,env.generate_fn,env.gate_fn,env.scorer,env.writer,env.continuation,policy,tr.st,tr.cache)
    initial=tr.cfg['rubric']['prompt_sha256']; selected=tr.state.prompt_sha256
    rows=[]
    for label,sha in [('P0',initial),('selected',selected)]:
        results=score_prompt(tr.store.get(sha),sha,units,{case['case_id']:case},deps)
        scores=objective(unit_values(results,tr.st),{evidence_id(u):u.tin.outcome for u in units})
        rows.append({'prompt_id':label,'prompt_sha256':sha,'split':'heldout','policy_sha256':policy,**scores})
    gain=rows[1]['J']-rows[0]['J'] if all(r['J'] is not None for r in rows) else None
    result={'case':case,'tree':to_dict(tree),'doctor_feedback':feedback,'prompt_scores':rows,'J_gain':gain,
            'note':'Untouched final diagnostic; never used for prompt feedback, acceptance, or replacement.'}
    atomic_write_json(tr.out/'heldout/result.json',result)
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',default='configs/online_rl/tree_pilot5.json')
    parser.add_argument('--launch',action='store_true');args=parser.parse_args()
    cfg=json.loads((ROOT/args.config).read_text());problems=preflight(cfg,args.config)
    print(json.dumps({'config':args.config,'questions':5,'agent_updates':False,'blockers':problems},indent=2),flush=True)
    if problems: raise SystemExit('Pilot not launched: preflight failed')
    if not args.launch: return
    os.environ['API_USAGE_LOG']=str(ROOT/cfg['output_dir']/'api_usage.jsonl')
    tr=RT.build(cfg);tr.run()
    frozen=tr.policy_sha(); result=compare_heldout(tr)
    if tr.policy_sha()!=frozen: raise RuntimeError('Agent policy changed during held-out evaluation')
    from src.online_rl.tree.artifacts import export_run
    print(json.dumps({'review':export_run(tr.out),'heldout_scores':result['prompt_scores'],'heldout_J_gain':result['J_gain']},indent=2),flush=True)

if __name__=='__main__':main()
