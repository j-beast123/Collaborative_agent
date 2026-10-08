"""Committed-batch migration to the opt-in fourth-turn stable-choice rule."""
import copy
import json
from pathlib import Path
import shutil
import time

from src.online_rl.integrated.records import atomic_write_json, sha256_json
from src.online_rl.integrated.continuation import protocol_sha256
from src.online_rl.tree.team import RUBRIC_ONLY_NA_POLICY
from src.online_rl.tree.effectiveness import RERUN_CONTINUATION_PROTOCOL, RERUN_OUTCOME_POLICY

ROOT = Path(__file__).resolve().parents[2]


def validate_config_change(old, new):
    """Allow the agreed halt, visible-context rubric, first-finisher P1, bounded execution and random-10 changes."""
    expected = copy.deepcopy(old)
    # A pre-commit cache-reuse bridge may pin the original tree-call identity;
    # the destination starts fresh tree identities under the new rollout policy.
    expected.pop('rollout_protocol_sha256_override',None)
    expected['output_dir'] = new['output_dir']
    expected['rollout']['halting']['stable_choice_no_objection_after_turns'] = 4
    if new['rollout']['halting'].get('repeat_agent_reply_after_turns'):
        expected['rollout']['halting']['repeat_agent_reply_after_turns']=4
    if new['rubric'].get('schema'):
        from src.online_rl.tree.agent_state import SCHEMA
        from src.online_rl.integrated.records import sha256_text
        path = 'prompts/sherpa/observed_context_behavior_v1.txt'
        expected['rubric'].update(schema=SCHEMA, state_addendum_file=path,
                                 state_addendum_sha256=sha256_text((ROOT/path).read_text()))
    if new['prompt_update'].get('candidate_seed'):
        expected['prompt_update']['candidate_seed'] = 'first_completed_P0_v1'
    if new['rubric'].get('match_batch_size'):
        expected['rubric']['match_batch_size']=8
    if new.get('execution', {}).get('member_workers') is not None:
        if new['execution']['member_workers'] != 2:
            raise RuntimeError('member_workers transition is limited to the approved value 2')
        expected.setdefault('execution', {})['member_workers'] = 2
    if new.get('verification', {}).get('mode') == 'random10_witness_v1':
        expected['verification'].update(
            mode='random10_witness_v1', max_valid_supports=10,
            support_sampling='uniform_random_without_replacement_v1',
            support_order='uniform random without replacement, deterministic per run seed + case ID + rubric ID; sample at most 10 same-ID trajectories',
            keep='KEEP at the first passing sampled trajectory; NOT_VALIDATED after the 10 sampled same-ID trajectories are exhausted')
        expected.setdefault('notes', {})['verification'] = (
            'random10_witness_v1: uniformly sample up to 10 same-ID trajectories without replacement using a seed derived from run seed, case ID, and rubric ID; test each at its own feedback locus with Stage-1 applicability and K=3 (2-of-3 short-circuit). KEEP at the first witness, otherwise NOT_VALIDATED after the sample is exhausted. Prompt-search J reuses these witness outcomes and compares the shared tested-trajectory intersection; it performs no additional member reruns.')
        expected['prompt_update']['aggregation']='mean_sampled_witness_effect_per_direction_product_v1'
    if (new != expected or old['rollout']['halting'].get('stable_choice_no_objection_after_turns') is not None or
            old['five_gpu'].get('execution_mode') != 'shared_ready_tasks_v1'):
        raise RuntimeError('Only the fourth-turn rule, visible-context rubric contract, first-finisher common P1, batch8 matching, member_workers=2, random10 witness protocol and output directory may change')


def transition(old_config, new_config):
    old=json.loads((ROOT/old_config).read_text());new=json.loads((ROOT/new_config).read_text())
    validate_config_change(old,new)
    source=(ROOT/old['output_dir']).resolve();target=(ROOT/new['output_dir']).resolve()
    if source==target or target.exists():raise RuntimeError('Transition requires a fresh output directory')
    state=json.loads((source/'run_state.json').read_text())
    if not 0<state['case_cursor']<=old['schedule']['cases'] or state['case_cursor']%5:
        raise RuntimeError('Transition requires a committed five-case batch')
    if state['config_sha256']!=sha256_json(old):raise RuntimeError('Source config mismatch')
    checkpoint=Path(state['checkpoint']);receipt=json.loads((checkpoint/'COMPLETE.json').read_text())
    if receipt['completed_cases']!=state['case_cursor'] or receipt['policy_sha256']!=state['policy_sha256']:
        raise RuntimeError('Source checkpoint does not match committed state')
    if any(not (source/'cases'/f'case_{i:06d}.json').exists() for i in range(state['case_cursor'])):
        raise RuntimeError('Committed case record missing')
    protocol=sha256_json({'base':protocol_sha256(new,ROOT),'D1':new['five_gpu']['opening_identity'],
                         'mode':new['five_gpu']['opening_mode'],'na':RUBRIC_ONLY_NA_POLICY,
                         'execution_mode':'shared_ready_tasks_v1','batch_cases':5})
    audit={'source_run':str(source),'source_run_state':state,'source_receipt':receipt,'time':time.time(),
           'completed_cases_with_original_halting':state['case_cursor'],
           'stable_choice_no_objection_after_turns':4,'repeat_agent_reply_after_turns':new['rollout']['halting'].get('repeat_agent_reply_after_turns'),
           'repeat_agent_reply_policy':'normalized_exact_prior_reply_v1','new_protocol_sha256':protocol,
           'incomplete_source_batches':'retained in source; never reused under the new protocol',
           'pool_submission_workers':{'agent':20,'doctor':5}}
    if old.get('verification', {}).get('mode') != new.get('verification', {}).get('mode'):
        audit.update(training_verification_transition={
            'source_protocol':old['verification']['mode'],
            'destination_protocol':new['verification']['mode'],
            'cases_before_transition_keep_source_protocol':state['case_cursor'],
            'cases_after_transition_use_destination_protocol':True,
            'prompt_search_effectiveness_reuses_witness_sample':True,
            'prompt_search_additional_member_reruns':0})
    if new['rubric'].get('schema'):
        audit.update(rubric_schema=new['rubric']['schema'],agent_side_definition='visible earlier Agent behavior; no hidden belief',
                     completed_cases_with_original_rubrics=state['case_cursor'])
    if new['prompt_update'].get('candidate_seed'):
        audit.update(candidate_seed=new['prompt_update']['candidate_seed'],selection_scope='all_batch_trajectories_v1')
    if new['rubric'].get('match_batch_size'):
        audit.update(match_batch_size=8,match_protocol='snapshot_match8_new_recheck_v1')
    if new.get('execution', {}).get('member_workers') is not None:
        audit['member_workers']=new['execution']['member_workers']
        audit['member_parallelism']='independent rubric witness checks; all same-ID members and K=3 early decision unchanged'
    pending=target.with_name(target.name+'.pending')
    if pending.exists():raise RuntimeError('Incomplete transition exists; inspect before retry')
    pending.mkdir(parents=True)
    try:
        # Optimizer, LoRA weights, RNG, winning prompt and all committed evidence.
        for ckpt in sorted((source/'checkpoints').glob('*')):
            proof_path=ckpt/'COMPLETE.json'
            if not proof_path.exists():continue
            proof=json.loads(proof_path.read_text())
            if proof['completed_cases']>state['case_cursor']:continue
            dest=pending/'checkpoints'/ckpt.name
            shutil.copytree(ckpt,dest)
            atomic_write_json(dest/'COMPLETE.json',{**proof,'protocol_sha256':protocol,
                              'transition_source_protocol_sha256':proof['protocol_sha256']})
        if not (pending/'checkpoints'/checkpoint.name/'COMPLETE.json').exists():
            raise RuntimeError('Current checkpoint was not copied')
        for name in ('prompts','prompt_events','cases','openings'):
            if (source/name).exists():shutil.copytree(source/name,pending/name)
        for batch in sorted((source/'batches').glob('batch_*')):
            if int(batch.name.split('_')[-1])<=state['batch_cursor']:
                shutil.copytree(batch,pending/'batches'/batch.name)
        # Reuse only the owned server PID registrations; their logs stay in source.
        for pidfile in (source/'doctor_servers').glob('gpu*.pid'):
            dest=pending/'doctor_servers'/pidfile.name;dest.parent.mkdir(exist_ok=True)
            shutil.copy2(pidfile,dest)
        for name in ('launch.json','api_usage.jsonl','prompt_calls.jsonl'):
            if (source/name).exists():shutil.copy2(source/name,pending/name)
        resumed={**state,'config_sha256':sha256_json(new),'protocol_sha256':protocol,
                 'checkpoint':str(target/'checkpoints'/checkpoint.name)}
        atomic_write_json(pending/'protocol_transition.json',audit)
        atomic_write_json(pending/'run_manifest.json',{'config':new,'protocol_sha256':protocol,
                          'rubric_only_na_policy':RUBRIC_ONLY_NA_POLICY,
                          'rerun_outcome_policy':RERUN_OUTCOME_POLICY,
                          'rerun_continuation_protocol':RERUN_CONTINUATION_PROTOCOL,'protocol_transition':audit})
        atomic_write_json(pending/'run_state.json',resumed)
        pending.rename(target)
    except BaseException:
        shutil.rmtree(pending,ignore_errors=True)
        raise
    return target
