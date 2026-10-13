"""Prefix-only reconstruction and greedy Agent Q1/Q2, shared by sibling actions and reruns."""
import json
from src.online_rl.integrated.records import sha256_json
from src.online_rl.tree.posthoc import case_view
from src.collaboration.generator import reconstruct_call, RECONSTRUCTION_FIELD_ORDER, dialogue_text
from src.belief import build_local_judgment_prompt,parse_local_judgment,map_judgment_to_roles,build_agent_response_prompt
from src.online_rl.integrated.rubric import split_state


def prepare(policy, case, prefix, turn):
    settings=policy.cfg['belief']; view=case_view(case)
    identity={'protocol':'tree_prefix_belief_v1','policy':policy.policy_sha(),'case':view['case_id'],
              'prefix_sha256':sha256_json(prefix),'turn':turn,'settings':settings}
    def measure():
        rec=reconstruct_call(policy.key,settings['reconstructor_model'],view,prefix,settings['max_tokens'])
        if rec.get('parse_status')!='valid':raise ValueError('Belief reconstruction invalid; no plain-response fallback')
        judgment=None
        if turn==1:
            rec.update(agent_current_position='',agent_claims_so_far=[],agent_evidence_so_far=[])
        else:
            question=build_local_judgment_prompt(question=view['question'],options=view['options'],scenario=view.get('scenario',''),
                        reconstruction={k:rec[k] for k in RECONSTRUCTION_FIELD_ORDER},claim_order='doctor_first')
            for attempt in range(3):
                raw=policy.agent(question if not attempt else question+'\nReturn only the requested JSON with string answers and nonempty reasons.',0,
                                 system=policy.judgment_system,greedy=True,max_tokens=settings['judgment_max_tokens'])
                judgment=parse_local_judgment(raw)
                if judgment.get('parse_status')=='valid':break
            if judgment.get('parse_status')!='valid':raise ValueError('Agent Q1/Q2 invalid; no silent default belief')
        state=split_state(rec,judgment,prefix,turn)
        return {'reconstruction':rec,'judgment':judgment,'state':state,'prefix_sha256':identity['prefix_sha256'],
                'measurement_policy_sha256':identity['policy'],'agent_turn':turn}
    packet=policy.cache.call('agent_belief',identity,measure)
    if turn==1:
        from src.prompt_opt.harness import Env
        prompt=Env.agent_user_prompt(policy,view,prefix,turn)
    else:
        prompt=build_agent_response_prompt(question=view['question'],options=view['options'],scenario=view.get('scenario',''),
                reconstructed_context=json.dumps({k:packet['reconstruction'][k] for k in RECONSTRUCTION_FIELD_ORDER},ensure_ascii=False),
                judgment=map_judgment_to_roles(packet['judgment'],'doctor_first'),dialogue_history=dialogue_text(prefix),
                current_round=turn,max_rounds=policy.args.rounds,is_final_round=turn==policy.args.rounds)
    return prompt,packet
