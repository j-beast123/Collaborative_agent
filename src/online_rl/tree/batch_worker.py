"""One spawned process/model per rollout GPU, three concurrent dialogue lanes.
Only local Agent generation takes the device lock; Doctor/API waiting overlaps.
"""
import copy
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import threading
from src.online_rl.integrated.records import atomic_write_json, from_dict, to_dict
from src.online_rl.tree.records import Tree, TrajectoryRubricInput
from src.online_rl.tree.cache import CallCache
from src.online_rl.tree.batch import optimize_case
ROOT=Path(__file__).resolve().parents[3]
CTX=None


def initialize(cfg, worker, key):
    global CTX
    from src.online_rl.runtime import SharedPolicy
    from src.online_rl.integrated.continuation import adapter_sha256
    class LanePolicy(SharedPolicy):
        def policy_sha(self):
            return self.fixed_sha
        def agent_user_prompt(self,case,prefix,turn):
            if self.cfg.get('belief',{}).get('enabled'):
                from src.online_rl.tree.belief import prepare
                return prepare(self,case,prefix,turn)[0]
            return super().agent_user_prompt(case,prefix,turn)
        def agent(self,user,seed,*,system=None,greedy=False,max_tokens=None):
            from src.online_rl.tree.batched_generation import Request
            prompt=self.ST.chat_prompt(self.tok,system or self.agent_system,user)
            trace=self.batcher.submit(Request(prompt,seed,max_tokens or self.args.max_completion_length,greedy))
            return trace['completion_text']
    policy=LanePolicy(cfg,key,device=cfg['gpu']['rollout_devices'][worker],trainable=False)
    policy.lock=threading.Lock();policy.fixed_sha=adapter_sha256(policy.model)
    policy.judgment_system=(ROOT/'prompts/local_epistemic_judgment.txt').read_text()
    policy.local=threading.local()
    # Each lane gets its own cache; belief preparation accesses it on the lane.
    type(policy).cache=property(lambda self:self.local.cache)
    from src.online_rl.tree.batched_generation import GenerationBatcher,decode_batch
    def generate(requests):
        with policy.lock,policy.torch.cuda.device(policy.device):
            started=__import__('time').monotonic()
            traces=decode_batch(policy.model,policy.tok,requests,temperature=policy.args.generation_temperature,
                                top_p=policy.args.top_p,top_k=policy.args.top_k)
            print(__import__('json').dumps({'stage':'agent_generation_batch','gpu':worker,'batch_size':len(requests),
                  'tokens':sum(len(t['completion_ids']) for t in traces),'seconds':round(__import__('time').monotonic()-started,3)}),flush=True)
            return traces
    policy.batcher=GenerationBatcher(generate,cfg['batch'].get('agent_batch_size',3),cfg['batch'].get('batch_wait_ms',15))
    CTX={'cfg':cfg,'worker':worker,'key':key,'policy':policy}


def load_checkpoint(checkpoint, expected_sha):
    from src.online_rl.integrated.continuation import adapter_sha256
    policy=CTX['policy']
    with policy.lock:
        policy.reload_adapter(checkpoint)
        policy.fixed_sha=adapter_sha256(policy.model)
    if policy.fixed_sha!=expected_sha:raise RuntimeError('Rollout replica differs from training adapter')
    return policy.fixed_sha


def components(case,out):
    from src.online_rl.tree.backends import ApiDoctor, PolicyInterventions, api_feedback, api_rubric_generator, posthoc_gates, two_stage_scorer
    from src.online_rl.tree.ollama_doctor import OllamaDoctorCall
    from src.judges.openrouter_choice import OpenRouterChoiceJudge
    from src.online_rl.tree.team import team_strategies
    from src.online_rl.integrated.prompt_search import sol_backend
    cfg=CTX['cfg'];key=CTX['key'];policy=CTX['policy'];cache=CallCache(out/'calls.jsonl');policy.local.cache=cache
    settings=copy.deepcopy(cfg['doctor_backend']);settings['url']=cfg['batch']['doctor_urls'][CTX['worker']]
    doctor=ApiDoctor(key,cfg['rollout']['doctor_model'],policy.doctor_system,cfg['rollout']['max_tokens_doctor'],call=OllamaDoctorCall(settings))
    judge=OpenRouterChoiceJudge(key,cfg['judge']['model'],cfg['judge']['max_tokens'],cfg['judge']['batch_size'],
                               max_retries=cfg['judge'].get('max_retries',2),audit_path=out/'judge_calls_raw.jsonl')
    text=lambda f:(ROOT/f).read_text();iv=PolicyInterventions(policy,text(cfg['intervention']['score0_template_file']),text(cfg['intervention']['score2_template_file']),doctor=doctor)
    generate=api_rubric_generator(key,cfg['pilot']['prompt_model'],cfg['pilot']['generator_max_tokens'])
    scorer=two_stage_scorer(judge);gate=posthoc_gates(judge,{case['case_id']:case});pu=cfg['prompt_update']
    backend=sol_backend(key,text(pu['feedback_prompt_file']),text(pu['refiner_prompt_file']),generate,gate,scorer,
                        feedback_model=pu['feedback_model'],refiner_model=pu['refiner_model'])
    return cache,doctor,iv,generate,scorer,gate,backend,team_strategies(None,cfg['agent_reward']['w_outcome'],cfg['agent_reward']['w_rubric'],cfg['agent_reward'].get('credit','shared'))


def rollout_one(job):
    from src.collaboration.generator import stop_reason
    from src.online_rl.tree.backends import parse_final_option,api_feedback
    from src.online_rl.environment import final_outcome
    from src.online_rl.tree.rollout import build_tree
    from src.online_rl.tree.posthoc import doctor_feedback,freeze_trajectory_input
    from src.online_rl.integrated.continuation import protocol_sha256
    case,out=job['case'],Path(job['out']);cfg=CTX['cfg'];policy=CTX['policy'];out.mkdir(parents=True,exist_ok=True)
    cache,doctor,*_=components(case,out)
    class Agent:
        def sample(self,case,prefix,turn,seed):
            packet=None
            if cfg.get('belief',{}).get('enabled'):
                from src.online_rl.tree.belief import prepare
                prompt,packet=prepare(policy,case,prefix,turn)
            else:prompt=policy.agent_user_prompt(case,prefix,turn)
            from src.online_rl.tree.batched_generation import Request
            chat=policy.ST.chat_prompt(policy.tok,policy.agent_system,prompt)
            tr=policy.batcher.submit(Request(chat,seed,policy.args.max_completion_length))
            return {**tr,'prompt_text':chat,'seed':seed,'pre_response_belief':packet}
    def behavior_scores(traces):
        from src.online_rl.integrated.grpo import behavior_logprobs
        with policy.lock,policy.torch.cuda.device(policy.device):
            return behavior_logprobs(policy.model,traces,cfg['optimizer']['logprob_token_chunk_size'])
    tree=build_tree(case,max_agent_turns=cfg['rollout']['rounds'],branching=cfg['rollout']['branching'],
           run_seed=cfg['schedule']['seed'],policy_sha256=policy.fixed_sha,protocol_sha256=protocol_sha256(cfg,ROOT),
           agent=Agent(),doctor=doctor,stop_rule=stop_reason,final_option=lambda text:parse_final_option(text,dict(case['options'])) or '',
           outcome_fn=final_outcome,cache=cache,behavior_logprobs=behavior_scores)
    feedback_fn=api_feedback(CTX['key'],cfg['rollout']['doctor_model'],(ROOT/cfg['feedback']['prompt_file']).read_text(),cfg['feedback']['max_tokens'],call=doctor.call)
    nodes={n.node_id:n for n in tree.nodes};inputs=[];feedbacks=[]
    for leaf in tree.leaves:
        if leaf.outcome is None:continue
        fb=doctor_feedback(case,leaf,feedback_fn,cfg['feedback']['prompt_sha256'],cache);feedbacks.append(fb)
        inputs.append(freeze_trajectory_input(case,leaf,fb,nodes))
    rec={'tree':to_dict(tree),'trajectory_inputs':[to_dict(t) for t in inputs],'feedback':[to_dict(f) for f in feedbacks]}
    atomic_write_json(out/'source_rollout.json',rec)
    return {'index':job['index'],'case_id':case['case_id'],'policy_sha256':policy.fixed_sha}


def rollout_many(jobs,checkpoint,expected_sha):
    load_checkpoint(checkpoint,expected_sha)
    with ThreadPoolExecutor(max_workers=CTX['cfg']['batch']['cases_per_gpu']) as pool:
        return list(pool.map(rollout_one,jobs))


def evaluate(job,prompt_id,prompt_text,history,event):
    from src.online_rl.integrated.prompt_search import PromptStore
    from src.online_rl.tree.prompt_search import ScoringDeps
    out=Path(job['out']);source=__import__('json').loads((out/'source_rollout.json').read_text());policy=CTX['policy'];cfg=CTX['cfg']
    cache,doctor,iv,generate,scorer,gate,backend,strategies=components(job['case'],out)
    deps=ScoringDeps(3,cfg['schedule']['seed'],generate,gate,scorer,iv.write,iv.continue_,policy.fixed_sha,strategies,cache)
    result=optimize_case(case=job['case'],tree=from_dict(Tree,source['tree']),
          inputs=[from_dict(TrajectoryRubricInput,t) for t in source['trajectory_inputs']],
          prompt_id=prompt_id,prompt_text=prompt_text,store=PromptStore(out/'prompts'),backend=backend,deps=deps,
          cfg=cfg,event=event,case_index=job['index'],history=history,out_dir=out)
    result['feedback']=source['feedback'];atomic_write_json(out/'case.json',result)
    return result
