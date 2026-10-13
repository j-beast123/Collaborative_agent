"""Explicit, auditable protocol transition at a fully committed training batch."""
import copy
import json
from pathlib import Path
import shutil
import time
ROOT=Path(__file__).resolve().parents[2]


def transition(old_config,new_config):
    from src.online_rl.integrated.records import atomic_write_json,sha256_json
    from src.online_rl.integrated.continuation import protocol_sha256
    from src.online_rl.tree.team import RUBRIC_ONLY_NA_POLICY
    from src.online_rl.tree.effectiveness import RERUN_OUTCOME_POLICY
    old=json.loads((ROOT/old_config).read_text());new=json.loads((ROOT/new_config).read_text())
    expected=copy.deepcopy(old);expected['output_dir']=new['output_dir'];expected['rollout']['halting']['min_agent_turns']=2
    if new!=expected or old['rollout']['halting']['min_agent_turns']!=3:raise RuntimeError('Only min-turns 3 -> 2 and output directory may change')
    source=ROOT/old['output_dir'];target=ROOT/new['output_dir']
    if target.resolve()==source.resolve() or target.exists():raise RuntimeError('Transition requires a fresh output directory')
    state=json.loads((source/'run_state.json').read_text())
    if state['case_cursor']<4 or state['case_cursor']%4:raise RuntimeError('Transition requires a committed batch')
    if sha256_json(old)!=state['config_sha256']:raise RuntimeError('Source config mismatch')
    checkpoint=Path(state['checkpoint']);receipt=json.loads((checkpoint/'COMPLETE.json').read_text())
    if receipt['completed_cases']!=state['case_cursor'] or receipt['policy_sha256']!=state['policy_sha256']:raise RuntimeError('Source receipt mismatch')
    protocol=sha256_json({'base':protocol_sha256(new,ROOT),'D1':new['five_gpu']['opening_identity'],'mode':new['five_gpu']['opening_mode'],'na':RUBRIC_ONLY_NA_POLICY})
    audit={'source_run':str(source),'source_checkpoint':str(checkpoint),'source_run_state':state,'source_receipt':receipt,
           'completed_cases_under_min3':state['case_cursor'],'remaining_cases_under_min2':new['schedule']['cases']-state['case_cursor'],
           'old_min_agent_turns':3,'new_min_agent_turns':2,'new_protocol_sha256':protocol,'time':time.time(),
           'base_evaluation':'restart all 622 under minimum 2; previous minimum-3 results remain in source run'}
    pending=target.with_name(target.name+'.pending')
    if pending.exists():raise RuntimeError('Incomplete transition exists; inspect before retry')
    pending.mkdir(parents=True)
    try:
        destination=pending/'checkpoints'/f'checkpoint_{state["case_cursor"]:06d}'
        shutil.copytree(checkpoint,destination)
        atomic_write_json(destination/'COMPLETE.json',{**receipt,'protocol_sha256':protocol,'transition_source_protocol_sha256':state['protocol_sha256']})
        for name in ('prompts','prompt_events','cases','openings'):
            if (source/name).exists():shutil.copytree(source/name,pending/name)
        resumed={**state,'config_sha256':sha256_json(new),'protocol_sha256':protocol,
                 'checkpoint':str(target/'checkpoints'/destination.name)}
        atomic_write_json(pending/'protocol_transition.json',audit)
        atomic_write_json(pending/'run_manifest.json',{'config':new,'protocol_sha256':protocol,'rubric_only_na_policy':RUBRIC_ONLY_NA_POLICY,
                              'rerun_outcome_policy':RERUN_OUTCOME_POLICY,'protocol_transition':audit})
        atomic_write_json(pending/'run_state.json',resumed)
        pending.rename(target)
    except BaseException:
        shutil.rmtree(pending,ignore_errors=True);raise
    return target
