"""Five-case shared ready-work pool; GPU4 updates after the complete batch barrier."""
import argparse
import copy
import fcntl
import json
import multiprocessing
import os
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.online_rl.integrated.records import atomic_write_json, sha256_json
from src.online_rl.tree.five_gpu import (EvaluationQueue, batch_slots, cache_openings,
    evaluation_adapter, load_checkpoint, save_checkpoint, training_order, update_batch, wait_with_evaluation)
from src.online_rl.tree import experiment
from src.online_rl.tree.devices import physical_gpus, record_runtime


def preflight(cfg):
    from scripts.online_rl.run_tree import blockers, PROMPT_HASHES, file_sha
    from src.online_rl.cases import load_cases, file_sha256
    errors = blockers(cfg)
    from src.online_rl.tree.cumulative_run import config_identity as bank_identity
    try:
        bank_identity(cfg, ROOT)
    except (ValueError, RuntimeError, OSError, KeyError) as exc:
        errors.append('Cumulative bank: ' + str(exc))
    for section, file, digest in PROMPT_HASHES:
        if file not in cfg.get(section, {}) or digest not in cfg.get(section, {}): continue
        try:
            actual_digest = file_sha(cfg[section][file])
        except (OSError, TypeError):
            errors.append(f'Missing or unreadable prompt artifact: {section}.{file}')
            continue
        if cfg[section][digest] != actual_digest:
            errors.append(f'Prompt hash differs: {section}.{file}')
    for section, path_key, hash_key in ((cfg['data'],'train','train_sha256'),
                                       (cfg['five_gpu'],'eval_file','eval_sha256')):
        path = ROOT/section[path_key]
        if not path.exists(): errors.append(f'Data missing: {path}')
        elif file_sha256(path) != section[hash_key]: errors.append(f'Data hash differs: {path}')
    fg = cfg['five_gpu']
    from src.online_rl.tree.gpu_pool import MODE as POOL_MODE
    shared = fg.get('execution_mode') == POOL_MODE
    if cfg['prompt_update'].get('candidate_seed') not in (None,'first_completed_P0_v1','all_completed_P0_v1'):
        errors.append('Unknown common P1 seed mode')
    if cfg['prompt_update'].get('candidate_seed') and not shared:
        errors.append('The staged batch P0/P1 workflow requires shared five-GPU execution')
    batch_cases = 5 if shared else 4
    if shared and fg.get('base_evaluation', {}).get('enabled'):
        errors.append('Shared five-GPU checkpoint evaluation cannot run BASE evaluation concurrently')
    if shared and (fg.get('agent_batch_size') != 4 or fg.get('doctor_concurrency') != 1 or
                   os.environ.get('SHERPA_AGENT_BATCH_SIZE','4') != '4' or os.environ.get('SHERPA_DOCTOR_CONCURRENCY','1') != '1'):
        errors.append('Shared mode requires the measured Agent4 / Doctor1 configuration')
    if fg['rollout_gpus'] != (list(range(5)) if shared else [0,1,2,3]) or fg['update_gpu'] != 4 or fg['lanes_per_gpu'] != 1 or fg.get('batch_cases') != batch_cases:
        errors.append(f'Need {batch_cases} case coordinators, one lane each, updater GPU4')
    if fg['checkpoint_every_cases'] != (30 if shared else 24) or len(fg['doctor_urls']) != 5 or len(set(fg['doctor_urls'])) != 5:
        errors.append('Need the declared snapshot interval and five distinct Doctor endpoints')
    if cfg.get('evaluation'):
        errors.append('Use the asynchronous evaluation queue in this runner')
    if cfg['prompt_update']['enabled']:
        from src.online_rl.tree.batch_prompt import MODE, FIVE_MODE
        pu=cfg['prompt_update']
        if pu.get('mode')!=(FIVE_MODE if shared else MODE) or pu.get('candidate_count')!=1 or pu.get('refinement_depth')!=1 or pu.get('every_n_cases')!=batch_cases:
            errors.append(f'Batch prompt evolution needs one P1 candidate, one refinement, every {batch_cases} cases')
        if cfg['acceptance'].get('rule')!='strictly_higher_product_v1':
            errors.append('Batch prompt acceptance must use the strictly higher directional product')
        if pu['reruns_per_intervention']!=cfg['intervention']['reruns_per_intervention']:
            errors.append('P0/P1 must use the same training rerun count')
        verification = cfg.get('verification', {})
        mode = verification.get('mode')
        if mode == 'all_members_witness_v1' and verification.get('max_valid_supports') is None:
            pass
        elif (mode == 'random10_witness_v1' and verification.get('max_valid_supports') == 10 and
              verification.get('support_sampling') == 'uniform_random_without_replacement_v1'):
            pass
        else:
            errors.append('Batch prompt evolution needs all-member witness or reproducible random-10 same-ID witness verification')
        expected_aggregation=('mean_sampled_witness_effect_per_direction_product_v1'
                              if mode=='random10_witness_v1' else 'mean_trajectory_effect_per_direction_product_v1')
        required={'aggregation':expected_aggregation,'tie':'keep_parent',
                  'missing_direction':'keep_parent','training_rubrics':'union_both_kept_deduplicated_v1'}
        if any(pu.get(k)!=v for k,v in required.items()):
            errors.append('Batch prompt aggregation, tie, missing-direction or union policy differs')
    if cfg['schedule']['cases']%batch_cases:
        errors.append(f'Training case count must be a multiple of {batch_cases}')
    opening=fg.get('opening_identity',{})
    if opening.get('system_prompt_sha256')!=cfg['rollout'].get('doctor_system_prompt_sha256'):
        errors.append('D1 identity must use the current Doctor prompt hash')
    if opening.get('options')!={**cfg['rollout']['doctor_backend']['options'],
                                'num_predict':cfg['rollout']['max_tokens_doctor']}:
        errors.append('D1 identity must use the current Doctor generation options')
    if cfg['rollout'].get('doctor_opening') or fg.get('opening_mode') != 'immutable_runtime_D1_v1':
        errors.append('This runner uses the declared immutable runtime D1 protocol')
    if cfg['rollout']['halting']['min_agent_turns'] not in (1,2,3):
        errors.append('Minimum Agent turns must be one, two or three')
    if (ROOT/cfg['data']['train']).exists() and (ROOT/fg['eval_file']).exists():
        train = load_cases(ROOT/cfg['data']['train']); test = load_cases(ROOT/fg['eval_file'])
        if len(test) != fg['eval_cases'] or len({c['case_id'] for c in test}) != len(test):
            errors.append('Evaluation size or unique case IDs differ')
        def question(c): return ' '.join(c['question'].split()).casefold()
        if {question(c) for c in train} & {question(c) for c in test}:
            errors.append('Training/evaluation question overlap')
        from src.online_rl.guards import assert_not_protected
        try: assert_not_protected(train, 'five-GPU training')
        except (ValueError, RuntimeError) as e: errors.append(str(e))
        try: training_order(cfg,train,ROOT)
        except (ValueError,OSError) as e: errors.append(str(e))
    try: physical_gpus()
    except ValueError as e: errors.append(str(e))
    if cfg.get('supervision_handoff') is not None and not (ROOT/cfg['output_dir']/'run_state.json').exists():
        from src.online_rl.tree.supervision_handoff import validate_handoff
        try: validate_handoff(cfg, ROOT/cfg['output_dir'])
        except (ValueError, RuntimeError, KeyError, OSError) as e:
            errors.append('Checkpoint supervision handoff: ' + str(e))
    return errors


def prompt_round(batch,jobs,state,collect,store,backend,cache,history,batchdir):
    """Run P0 generation, P0 verification, P1 generation, P1 verification, then union scoring."""
    from src.online_rl.tree.batch_prompt import choose, propose_candidate, MODE
    parent={'text':store.get(state['prompt_sha256']),'sha256':state['prompt_sha256'],'id':state['prompt_id']}
    print(json.dumps({'stage':'P0_rubric_generation','batch':batch,'prompt_sha256':parent['sha256']}),flush=True)
    collect('P0_prepare',parent)
    print(json.dumps({'stage':'P0_rerun_verification','batch':batch}),flush=True)
    p0=collect('P0_verify',parent)

    # P1 uses the evidence from the complete five-case P0 verification batch.
    evidence_sha=sha256_json([{'case_id':r['case_id'],'tree':sha256_json(r['tree']),
                               'evaluation':r['prompt_evaluation']} for r in p0])
    candidate_path=batchdir/'prompt_candidate.json'
    candidate=None
    if candidate_path.exists():
        saved=json.loads(candidate_path.read_text())
        if saved.get('parent_sha256')==parent['sha256'] and saved.get('p0_evidence_sha256')==evidence_sha:
            candidate=saved['candidate']
    if candidate is None:
        candidate=propose_candidate(p0,parent['text'],store,backend,cache,[j['case'] for j in jobs],history)
        candidate['id']=f'B{batch}-P1-{candidate["sha256"][:12]}'
        atomic_write_json(candidate_path,{'parent_sha256':parent['sha256'],'p0_evidence_sha256':evidence_sha,
                                          'candidate':candidate})
    p1={'text':candidate['text'],'sha256':candidate['sha256'],'id':candidate['id']}
    print(json.dumps({'stage':'P1_rubric_generation','batch':batch,'prompt_sha256':p1['sha256']}),flush=True)
    collect('P1_prepare',p1)
    print(json.dumps({'stage':'P1_rerun_verification','batch':batch}),flush=True)
    c1=collect('P1_verify',p1)
    decision=choose(p0,c1)
    if not candidate['valid_candidate']:
        decision.update(accepted=False,winner='P0',reason='no_valid_distinct_candidate')
    winner=p1 if decision['accepted'] else parent
    print(json.dumps({'stage':'final_union_rubric_scoring','batch':batch}),flush=True)
    records=collect('union',{**p1,'parent_sha256':parent['sha256']})
    next_prompt={'id':winner['id'],'sha256':winner['sha256']}
    event={'mode':store.batch_mode if hasattr(store,'batch_mode') else MODE,'batch':batch,'case_ids':[j['case']['case_id'] for j in jobs],
           'rerun_policy_sha256':state['policy_sha256'],'parent_prompt':{k:parent[k] for k in ('id','sha256')},
           'candidate_prompt':{k:p1[k] for k in ('id','sha256')},'proposal':candidate['proposal'],
           'decision':decision,'next_prompt':next_prompt,'training_pool':'union_both_kept_deduplicated_v1'}
    event.update(candidate_evidence='all_P0_verified_cases_v1',p0_evidence_sha256=evidence_sha,
                 workflow='P0_generate_all_then_verify_all__P1_generate_all_then_verify_all__union_score_once_v1',
                 selection_scope='all_batch_trajectories_v1')
    return records,event


def run(cfg, max_batches=None, eval_only=False):
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() < 5:
        raise RuntimeError('CUDA GPUs 0..4 must be visible')
    from scripts.online_rl.run_tree import load_key
    from src.online_rl.runtime import SharedPolicy
    from src.online_rl.integrated.grpo import make_updater
    from src.online_rl.integrated.continuation import adapter_sha256, protocol_sha256
    from src.online_rl.cases import load_cases
    from src.online_rl.environment import final_outcome
    from src.online_rl.tree.team import RUBRIC_ONLY_NA_POLICY
    from src.online_rl.tree.effectiveness import RERUN_CONTINUATION_PROTOCOL, RERUN_OUTCOME_POLICY
    from src.online_rl.tree import five_gpu_worker as worker
    fg = cfg['five_gpu']; out = (ROOT/cfg['output_dir']).resolve(); out.mkdir(parents=True,exist_ok=True)
    from src.online_rl.tree.gpu_pool import MODE as POOL_MODE, create_pool, assert_idle
    shared = fg.get('execution_mode') == POOL_MODE
    evaluation_enabled = fg.get('checkpoint_evaluation_enabled', True)
    if eval_only and not evaluation_enabled:
        raise RuntimeError('Evaluation is disabled in this training configuration')
    lock = (out/'runner.lock').open('a')
    try: fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError: raise RuntimeError('Another runner owns this output directory/GPU4')
    # Lock the updater's physical GPU.
    gpu_lock = Path(f"/tmp/sherpa_tree_physical_gpu{physical_gpus()[fg['update_gpu']]}.lock").open('a')
    try: fcntl.flock(gpu_lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close(); gpu_lock.close()
        raise RuntimeError('GPU4 is already owned by another five-GPU runner')
    # Keep both locks open throughout training and evaluation.
    os.environ['SHERPA_AGENT_REPLICAS'] = '1'
    os.environ.pop('SHERPA_DOCTOR_ENDPOINTS', None)
    os.environ['API_USAGE_LOG'] = str(out/'api_usage.jsonl')
    os.environ['SHERPA_API_WORKERS'] = str(fg.get('api_workers',3))
    record_runtime(out,cfg)
    key = load_key(cfg)
    guardian_call = None
    guardian_spec = ((cfg.get('rubric') or {}).get('cumulative_bank') or {}).get('guardian')
    if guardian_spec is not None:
        from src.online_rl.tree.bank_guardian import gemini_model_call
        guardian_call = gemini_model_call(key, model=guardian_spec['model'],
                                           max_tokens=guardian_spec['max_tokens'])
    local = copy.deepcopy(cfg)
    local['rollout']['doctor_backend']['endpoint'] = fg['doctor_urls'][4]
    torch.manual_seed(cfg['schedule']['seed'])
    from src.common.live_gpu import configure, activity, note
    from src.online_rl.tree.idle_eval_hooks import register_worker, training_guard
    register_worker(4)
    configure(out,4)
    with training_guard(4, 'model_load', release_memory=True), activity('model_load'):
        policy = SharedPolicy(local,key,device='cuda:4')
    problems = policy.doctor.verify()['problems']
    if problems: raise RuntimeError('GPU4 Doctor identity: '+'; '.join(problems))
    if not shared:
        cache_openings(policy.doctor,out/'openings/eval.jsonl',fg['opening_identity'])
    updater = make_updater(policy,cfg)
    # Enabling checkpoint evaluation is a safe runtime-only resume change: the
    # trained policy, data, optimizer and rollout protocol remain identical.
    resume_cfg = copy.deepcopy(cfg)
    resume_cfg.setdefault('five_gpu', {})['checkpoint_evaluation_enabled'] = False
    config_sha = sha256_json(resume_cfg)
    evolving=cfg['prompt_update']['enabled']
    if evolving:
        from src.online_rl.integrated.prompt_search import PromptStore, sol_backend
        from src.online_rl.tree.cache import CallCache
        store=PromptStore(out/'prompts')
        store.batch_mode = cfg['prompt_update']['mode']
        initial_prompt=store.put((ROOT/cfg['rubric']['generator_prompt_file']).read_text())
        if initial_prompt!=cfg['rubric']['prompt_sha256']:raise RuntimeError('Initial generator prompt hash mismatch')
        pu=cfg['prompt_update']
        backend=sol_backend(key,(ROOT/pu['feedback_prompt_file']).read_text(),(ROOT/pu['refiner_prompt_file']).read_text(),
                            None,None,None,reflection_model=pu['reflection_model'],refiner_model=pu['refiner_model'])
        prompt_cache=CallCache(out/'prompt_calls.jsonl')
    protocol_parts = {'base':protocol_sha256(cfg,ROOT),'D1':fg['opening_identity'],
                      'mode':fg['opening_mode'],'na':RUBRIC_ONLY_NA_POLICY,
                      **({'execution_mode':POOL_MODE,'batch_cases':5} if shared else {})}
    if cfg.get('rubric', {}).get('generation', {}).get('output_contract') == 'trajectory_analysis_candidate_v2_with_intervention':
        from src.online_rl.tree.member_intervention import MEMBER_INTERVENTION_VERSION
        # Bind even when the rollout-only protocol has an explicit frozen-tree override.
        protocol_parts['member_intervention_protocol'] = MEMBER_INTERVENTION_VERSION
    if cfg.get('feedback', {}).get('protocol') == 'physician_post_final_reflection_v1':
        # Imported rollout identity can stay frozen while supervision has a new
        # research identity. Never describe observer and Doctor reflection as the same protocol.
        protocol_parts['doctor_reflection'] = {k: cfg['feedback'][k]
            for k in ('protocol', 'model', 'prompt_sha256', 'max_tokens')}
        protocol_parts['fixed_rubric_prompt_sha256'] = cfg['rubric']['prompt_sha256']
    if cfg.get('rubric', {}).get('generation', {}).get('feedback_check_protocol') is not None:
        from src.online_rl.tree.backends import generation_settings
        # The dialogue policy stays frozen, but a new reflection-output contract
        # must never share the old supervision protocol identity.
        protocol_parts['feedback_claim_audit'] = generation_settings(cfg['rubric'])
    from scripts.online_rl.run_tree import evidence_gate_identity, intervention_audit_identity
    from src.online_rl.tree.applicability import APPLICABILITY_CONTRACT
    from src.online_rl.tree.scoring import SCORE_RECORD_CONTRACT
    gate_identity = evidence_gate_identity(cfg, ROOT)
    if gate_identity is not None:
        protocol_parts['rubric_gate'] = gate_identity
    if cfg.get('rubric', {}).get('enabled', True):
        protocol_parts['applicability_contract'] = APPLICABILITY_CONTRACT
        protocol_parts['score_record_contract'] = SCORE_RECORD_CONTRACT
    audit_identity = intervention_audit_identity(cfg, ROOT)
    if audit_identity is not None:
        protocol_parts['intervention_instruction_audit'] = audit_identity
    from src.online_rl.tree import cumulative_run
    bank_identity = cumulative_run.config_identity(cfg, ROOT)
    if bank_identity is not None:
        protocol_parts['cumulative_bank'] = bank_identity
    # Case workers call run_tree.build, which binds the same configured audit callback.
    # Explicit research identities prevent importing old decisions with a frozen rollout hash.
    protocol = sha256_json({**protocol_parts,
                            'rubric_pipeline_mode':'phase_barrier_reruns_then_union_score_v1'})
    path = out/'run_state.json'
    theta0 = adapter_sha256(policy.model)
    imported_rollout = out/'rollout_import.json'
    if imported_rollout.exists():
        provenance = json.loads(imported_rollout.read_text())
        if provenance.get('source_policy_sha256') != theta0:
            raise RuntimeError('Imported rollout policy differs from the freshly initialized Agent; refusing cached replay')
    experiment.check_pfinal_source(cfg,theta0,protocol)
    if path.exists():
        state = json.loads(path.read_text())
        if state['config_sha256'] != config_sha:
            raise RuntimeError('Run configuration/protocol changed; use a new output directory')
        if state['protocol_sha256'] != protocol:
            # One explicit, auditable transition: an old prompt-pipeline run may
            # adopt the new phase-barrier workflow only at a committed 5-case
            # batch boundary. Rollout, reward, data and policy identities must
            # still match; all other protocol mismatches remain fatal.
            old_pipeline_protocol = sha256_json(protocol_parts)
            cursor = state.get('case_cursor', -1)
            batch_cursor = state.get('batch_cursor', -1)
            checkpoint = Path(state.get('checkpoint', ''))
            receipt_path = checkpoint/'COMPLETE.json'
            receipt = json.loads(receipt_path.read_text()) if receipt_path.is_file() else {}
            handoff_protocol_carried = (
                receipt.get('protocol_sha256') == state.get('protocol_sha256') or
                any(row.get('from') == receipt.get('protocol_sha256') and
                    row.get('to') == state.get('protocol_sha256') and
                    row.get('completed_cases') == cursor
                    for row in state.get('protocol_transitions', []) if isinstance(row, dict))
            )
            verified_handoff = bool(
                cfg.get('supervision_handoff') and
                state.get('supervision_handoff_sha256') and
                receipt.get('supervision_handoff_sha256') == state.get('supervision_handoff_sha256') and
                handoff_protocol_carried
            )
            boundary_ok = (
                shared and isinstance(cursor, int) and cursor > 0 and cursor % 5 == 0 and
                batch_cursor == cursor//5 and receipt.get('completed_cases') == cursor and
                receipt.get('policy_sha256') == state.get('policy_sha256')
            )
            phase_transition = (
                os.environ.get('SHERPA_ALLOW_P0P1_PHASE_RESUME') == '1' and
                # The phase-barrier transition is independent of prompt
                # optimization. Fixed-P0 runs need the same verified boundary
                # resume when their supervision pipeline adopts this workflow.
                boundary_ok and (state.get('protocol_sha256') == old_pipeline_protocol or verified_handoff) and
                (receipt.get('protocol_sha256') == old_pipeline_protocol or verified_handoff)
            )
            # A validated execution-contract repair can change a scorer's
            # protocol identity without changing policy, data, rewards already
            # committed, or optimizer state.  Permit that transition only via
            # an explicit environment flag at a verified checkpoint boundary,
            # with the checkpoint receipt pinned to the exact previous protocol.
            contract_repair_transition = (
                os.environ.get('SHERPA_ALLOW_EXECUTION_CONTRACT_RESUME') == '1' and
                boundary_ok and receipt.get('protocol_sha256') == state.get('protocol_sha256')
            )
            approved = phase_transition or contract_repair_transition
            if not approved:
                raise RuntimeError('Run protocol changed. P0/P1 phase-resume is allowed only with '
                                   'SHERPA_ALLOW_P0P1_PHASE_RESUME=1 at a verified 5-case checkpoint boundary. '
                                   f'diagnostics={{"env":{os.environ.get("SHERPA_ALLOW_P0P1_PHASE_RESUME") == "1"},'
                                   f'"shared":{shared},"legacy":{state.get("protocol_sha256") == old_pipeline_protocol},'
                                   f'"handoff":{verified_handoff},"cursor":{cursor},"batch":{batch_cursor},'
                                   f'"receipt_cases":{receipt.get("completed_cases")},'
                                   f'"receipt_policy_match":{receipt.get("policy_sha256") == state.get("policy_sha256")}}}')
            previous_protocol = state['protocol_sha256']
            state['protocol_sha256'] = protocol
            state.setdefault('protocol_transitions', []).append({
                'from': previous_protocol, 'to': protocol,
                'completed_cases': cursor,
                'transition': ('old_prompt_pipeline_to_phase_barrier_reruns_then_union_score_v1'
                               if phase_transition else 'compact_execution_contract_repair_v1')
            })
            atomic_write_json(path, state)
            manifest_path = out/'run_manifest.json'
            if manifest_path.exists():
                run_manifest = json.loads(manifest_path.read_text())
                run_manifest.setdefault('protocol_transitions', []).append(state['protocol_transitions'][-1])
                run_manifest['protocol_sha256'] = protocol
                atomic_write_json(manifest_path, run_manifest)
        experiment.check_resume(cfg,out,state,theta0)
        with training_guard(4, 'resume_checkpoint', release_memory=True):
            load_checkpoint(updater,state['checkpoint'])
        if adapter_sha256(policy.model) != state['policy_sha256']: raise RuntimeError('Resume adapter mismatch')
        manifest_path = out/'run_manifest.json'
        if manifest_path.exists():
            run_manifest = json.loads(manifest_path.read_text())
            old_enabled = run_manifest.get('config', {}).get('five_gpu', {}).get('checkpoint_evaluation_enabled', False)
            if old_enabled != evaluation_enabled:
                run_manifest.setdefault('runtime_changes', []).append({
                    'checkpoint_evaluation_enabled': {'from': old_enabled, 'to': evaluation_enabled},
                    'note': 'Evaluation-only change; training config identity is normalized to disabled.'})
                run_manifest['config'] = cfg
                atomic_write_json(manifest_path, run_manifest)
    elif cfg.get('supervision_handoff') is not None:
        if eval_only: raise RuntimeError('--eval-only requires an existing run')
        from src.online_rl.tree.supervision_handoff import prepare_handoff
        print(json.dumps({'stage':'checkpoint_handoff',
                          'source_checkpoint':cfg['supervision_handoff']['source_checkpoint']}),flush=True)
        state = prepare_handoff(updater,cfg,out,config_sha,protocol,theta0,
            lambda:adapter_sha256(policy.model),
            manifest_payload={'config':cfg,'protocol_sha256':protocol,
                              'rubric_only_na_policy':RUBRIC_ONLY_NA_POLICY,
                              'rerun_outcome_policy':RERUN_OUTCOME_POLICY,
                              'rerun_continuation_protocol':RERUN_CONTINUATION_PROTOCOL})
        print(json.dumps({'stage':'checkpoint_handoff_complete','completed_cases':state['case_cursor'],
                          'batch':state['batch_cursor'],'optimizer_steps':updater.steps,
                          'policy_sha256':state['policy_sha256']}),flush=True)
    else:
        if eval_only: raise RuntimeError('--eval-only requires an existing run')
        if (out/'checkpoints').exists(): raise RuntimeError('Checkpoint artifacts without committed run state')
        initial = out/'checkpoints/checkpoint_000000'
        sha = adapter_sha256(policy.model)
        save_checkpoint(updater,initial,{'completed_cases':0,'policy_sha256':sha,'protocol_sha256':protocol,
                                         **experiment.receipt_fields(cfg)})
        state = {'case_cursor':0,'batch_cursor':0,'config_sha256':config_sha,'protocol_sha256':protocol,
                 'policy_sha256':sha,'checkpoint':str(initial),**experiment.state_fields(cfg,theta0)}
        if evolving:state.update(prompt_sha256=initial_prompt,prompt_id=cfg['rubric']['prompt_id'])
        atomic_write_json(out/'run_manifest.json',{'config':cfg,'protocol_sha256':protocol,
                            'rubric_only_na_policy':RUBRIC_ONLY_NA_POLICY,'rerun_outcome_policy':RERUN_OUTCOME_POLICY,
                            'rerun_continuation_protocol':RERUN_CONTINUATION_PROTOCOL})
        atomic_write_json(path,state)
    if bank_identity is not None:
        state = cumulative_run.initialize_bank_state(cfg, ROOT, out, state)
    eval_cases = load_cases(ROOT/fg['eval_file']) if evaluation_enabled or fg.get('base_evaluation',{}).get('enabled') else []
    base_evaluation=None
    if fg.get('base_evaluation',{}).get('enabled'):
        from src.online_rl.tree.base_evaluation import BaseEvaluation
        base_evaluation=BaseEvaluation(out/'evaluation/BASE_raw_no_precommit',eval_cases,policy,cfg)
    queue = EvaluationQueue(out/'evaluation',eval_cases,{'seed':fg['eval_seed'],
        'data_sha256':fg['eval_sha256'],'protocol_sha256':protocol,'dataset':'CasiMedicos all_en_622'}) if evaluation_enabled else None
    for ckpt in sorted((out/'checkpoints').glob('checkpoint_*')):
        receipt = json.loads((ckpt/'COMPLETE.json').read_text())
        cursor = receipt['completed_cases']
        should_evaluate = cursor > 0 and (cursor % fg['checkpoint_every_cases'] == 0 or cursor == cfg['schedule']['cases'])
        if queue is not None and should_evaluate and cursor <= state['case_cursor']:
            queue.register(ckpt,receipt['completed_cases'],receipt['policy_sha256'])

    def play(manifest,case,seed):
        if shared:
            policy.pool_client.context = {'case_id':case['case_id'], 'phase':f"eval_{manifest['completed_cases']}"}
            episode = policy.generate(case,seed)
        else:
            with evaluation_adapter(policy,manifest['checkpoint']):
                if adapter_sha256(policy.model) != manifest['policy_sha256']:
                    raise RuntimeError('Evaluation checkpoint adapter mismatch')
                episode = policy.generate(case,seed)
        return {'dialogue':episode['dialogue'],'final_option':episode['final_option'],
                'num_discussion_rounds':episode['num_discussion_rounds'],
                'outcome':final_outcome(case,episode['final_option'])}

    def drain_checkpoint_evaluations():
        if queue is None:
            return False
        if shared:
            if len(executors) != 5 or parent_actor is None:
                raise RuntimeError('Shared checkpoint evaluation requires all five GPU actors')
            def load_eval_snapshot(manifest):
                assert_idle(pool_spec)
                prepared=[executors[w].submit(worker.prepare,manifest['checkpoint'],manifest['policy_sha256']) for w in range(5)]
                wait_with_evaluation(prepared,lambda:False)
                if adapter_sha256(policy.model) != manifest['policy_sha256']:
                    raise RuntimeError('Evaluation policy differs from its committed checkpoint')
            def progress(manifest, done, total):
                print(json.dumps({'stage':'checkpoint_eval','checkpoint_cases':manifest['completed_cases'],
                                  'evaluated':done,'total':total,'gpus':fg['rollout_gpus']},ensure_ascii=False),flush=True)
            try:
                return queue.run_all_parallel(play,workers=25,progress=progress,before_job=load_eval_snapshot)
            finally:
                # Restore the latest committed training weights and actor hashes,
                # even when catching up an older interrupted evaluation fails.
                prepared=[executors[w].submit(worker.prepare,state['checkpoint'],state['policy_sha256']) for w in range(5)]
                wait_with_evaluation(prepared,lambda:False)
                if adapter_sha256(policy.model) != state['policy_sha256']:
                    raise RuntimeError('Failed to restore the current training checkpoint after evaluation')
                assert_idle(pool_spec)
        return queue.run_all_parallel(play,workers=1)
    def evaluate_one():
        if base_evaluation is not None and base_evaluation.step():return True
        return queue.step(play) if queue is not None else False
    executors = []
    manager = pool_spec = parent_actor = None
    try:
        if not eval_only or (shared and evaluation_enabled):
            cases = load_cases(ROOT/cfg['data']['train']); byid = {c['case_id']:c for c in cases}
            order = training_order(cfg,cases,ROOT)
            if shared:
                manager = multiprocessing.get_context('spawn').Manager()
                pool_spec = create_pool(manager, list(range(5)), fg['agent_batch_size'], fg['doctor_concurrency'])
                worker.initialize(cfg, 4, key, pool_spec, existing_policy=policy)
                parent_actor = policy.pool_actor
            for w in range(4):
                executors.append(ProcessPoolExecutor(max_workers=1,mp_context=multiprocessing.get_context('spawn'),
                                 initializer=worker.initialize,initargs=(cfg,w,key,pool_spec)))
            if shared:
                executors.append(ThreadPoolExecutor(max_workers=1,thread_name_prefix='gpu4-case'))
        if not eval_only:
            if queue is not None:drain_checkpoint_evaluations()
            completed = 0
            while state['case_cursor'] < cfg['schedule']['cases'] and (max_batches is None or completed < max_batches):
                batch = state['batch_cursor']+1
                batchdir = out/'batches'/f'batch_{batch:06d}'
                slots = batch_slots(state['case_cursor'],cfg['schedule']['cases'],
                                    interval=fg['checkpoint_every_cases'],
                                    workers=len(fg['rollout_gpus']),lanes=fg['lanes_per_gpu'])
                jobs = [{**s,'case':byid[order[s['index']]],
                         'out':str(batchdir/f'case_{s["index"]:06d}')} for s in slots]
                manifest={'source_policy_sha256':state['policy_sha256'],
                          'assignments':slots,'case_ids':[j['case']['case_id'] for j in jobs]}
                manifest_path=batchdir/'manifest.json'
                if manifest_path.exists():
                    if json.loads(manifest_path.read_text())!=manifest:
                        raise RuntimeError('Saved pending batch assignment/policy mismatch')
                else:
                    atomic_write_json(manifest_path,manifest)
                print(json.dumps({'stage':'rollout','batch':batch,'cases':len(jobs),'gpus':fg['rollout_gpus']}),flush=True)
                def collect(phase=None,prompt=None,selected_case_cursors=None):
                    if shared:
                        assert_idle(pool_spec)
                        prepared=[executors[w].submit(worker.prepare,state['checkpoint'],state['policy_sha256']) for w in range(5)]
                        wait_with_evaluation(prepared,lambda:False)
                    futures=[]
                    selected = set(selected_case_cursors) if selected_case_cursors is not None else None
                    for w in range(len(executors)):
                        assigned=[j for j in jobs if j['worker']==w and (selected is None or j['index'] in selected)]
                        if not assigned:continue
                        args=(assigned,state['checkpoint'],state['policy_sha256'])
                        if phase is not None:args+= (phase,prompt)
                        futures.append(executors[w].submit(worker.run_many,*args))
                    results=wait_with_evaluation(futures,lambda:False) if shared else wait_with_evaluation(futures,evaluate_one)
                    if shared:assert_idle(pool_spec)
                    return sorted([r for result in results for r in result],key=lambda r:r['case_cursor'])
                next_prompt=None
                next_bank=None
                if evolving:
                    history=[]
                    for p in sorted((out/'prompt_events').glob('batch_*.json')):
                        if int(p.stem.split('_')[-1])>=batch:continue
                        past=json.loads(p.read_text()); d=past['decision']
                        history.append({'batch':past['batch'],'accepted':d['accepted'],'reason':d['reason'],
                                        'parent_J':d['parent']['J'],'candidate_J':d['candidate']['J'],
                                        'winner_sha256':past['next_prompt']['sha256']})
                    records,event=prompt_round(batch,jobs,state,collect,store,backend,prompt_cache,history,batchdir)
                    atomic_write_json(out/'prompt_events'/f'batch_{batch:06d}.json',event)
                    next_prompt=event['next_prompt']
                elif bank_identity is not None:
                    records,next_bank=cumulative_run.round_records(cfg,ROOT,out,state,batch,batchdir,collect,
                                                                   guardian_call=guardian_call)
                else:
                    records=collect()
                if shared:
                    assert_idle(pool_spec)
                    with parent_actor.changing_policy():pass
                reward_rows = experiment.check_batch(cfg,records)
                if reward_rows:
                    atomic_write_json(out/'reward_components'/f'batch_{batch:06d}.json',reward_rows)
                digest = sha256_json(records); cursor = state['case_cursor']+len(records)
                milestone = cursor%fg['checkpoint_every_cases'] == 0 or cursor == cfg['schedule']['cases']
                ckpt = out/'checkpoints'/(f'checkpoint_{cursor:06d}' if milestone else f'recovery_{batch:06d}')
                if ckpt.exists():
                    proof = json.loads((ckpt/'COMPLETE.json').read_text())
                    experiment.check_receipt(cfg,proof,'Recovery checkpoint')
                    if proof['records_sha256'] != digest or proof['source_policy_sha256'] != state['policy_sha256']:
                        raise RuntimeError('Recovery receipt differs from replayed batch')
                    if evolving and proof.get('next_prompt')!=next_prompt:
                        raise RuntimeError('Recovery receipt prompt winner differs')
                    if next_bank is not None and any(proof.get(k)!=v for k,v in cumulative_run.bank_fields(next_bank).items()):
                        raise RuntimeError('Recovery receipt cumulative bank differs')
                    with training_guard(4, 'recovery_checkpoint', release_memory=True):
                        load_checkpoint(updater,ckpt)
                    update = proof['update']
                else:
                    print(json.dumps({'stage':'agent_update','gpu':4,'batch':batch,'cases':len(records)}),flush=True)
                    # The optional independent evaluator must release GPU4's
                    # model residency before backward can allocate memory.
                    with training_guard(4, 'agent_update', release_memory=True):
                        policy.doctor.http(fg['doctor_urls'][4]+'/api/generate',
                            {'model':cfg['rollout']['doctor_backend']['model'],'keep_alive':0},60)
                        note(phase='agent_update',batch=batch)
                        with activity('agent_update'):
                            update = update_batch(updater,records,state['policy_sha256'],lambda:adapter_sha256(policy.model))
                        proof = {'records_sha256':digest,'source_policy_sha256':state['policy_sha256'],
                            'policy_sha256':adapter_sha256(policy.model),'completed_cases':cursor,
                            'protocol_sha256':protocol,'update':update,**experiment.receipt_fields(cfg)}
                        if evolving:proof['next_prompt']=next_prompt
                        if next_bank is not None:proof.update(cumulative_run.bank_fields(next_bank))
                        save_checkpoint(updater,ckpt,proof)
                        if os.environ.get('SHERPA_IDLE_EVAL_PRIORITY_DIR'):
                            torch.cuda.synchronize(policy.model.device)
                            torch.cuda.empty_cache()
                previous = Path(state['checkpoint'])
                state = {**state,'case_cursor':cursor,'batch_cursor':batch,
                         'checkpoint':str(ckpt),'policy_sha256':adapter_sha256(policy.model)}
                if evolving:state.update(prompt_sha256=next_prompt['sha256'],prompt_id=next_prompt['id'])
                if next_bank is not None:state.update(cumulative_run.bank_fields(next_bank))
                for record in records:
                    record['batch_update'] = update
                    atomic_write_json(out/'cases'/f'case_{record["case_cursor"]:06d}.json',record)
                atomic_write_json(path,state)
                if milestone and queue is not None:
                    queue.register(ckpt,cursor,state['policy_sha256'])
                    drain_checkpoint_evaluations()
                if previous.name.startswith('recovery_'):
                    import shutil
                    shutil.rmtree(previous)
                completed += 1
                print(json.dumps({'stage':'batch_committed','completed_cases':cursor,'batch':batch}),flush=True)
        # A deliberately bounded diagnostic returns promptly; normal training drains every queued snapshot.
        if eval_only or max_batches is None:
            if shared and queue is not None:
                drain_checkpoint_evaluations()
            else:
                while evaluate_one(): pass
    finally:
        for executor in executors: executor.shutdown(wait=True,cancel_futures=True)
        if parent_actor is not None:parent_actor.close()
        if manager is not None:manager.shutdown()
        if pool_spec is not None:
            import shutil
            shutil.rmtree(pool_spec['directory'],ignore_errors=True)
        gpu_lock.close()
        lock.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',default='configs/online_rl/tree_v21_five_gpu_shared5.json')
    parser.add_argument('--launch',action='store_true')
    parser.add_argument('--eval-only',action='store_true')
    parser.add_argument('--max-batches',type=int)
    args = parser.parse_args()
    cfg = experiment.load_config(args.config)
    errors = preflight(cfg)
    print(json.dumps({'training_cases':cfg['schedule']['cases'],'rollout_gpus':cfg['five_gpu']['rollout_gpus'],'lanes_per_gpu':cfg['five_gpu']['lanes_per_gpu'],'batch_cases':cfg['five_gpu']['batch_cases'],'update_gpu':4,
                      'checkpoint_every_cases':cfg['five_gpu']['checkpoint_every_cases'],'evaluation_cases':cfg['five_gpu']['eval_cases'] if cfg['five_gpu'].get('checkpoint_evaluation_enabled',True) else 0,
                      'execution_mode':cfg['five_gpu'].get('execution_mode','fixed_case_v1'),
                      'output_dir':cfg['output_dir'],'experiment':cfg.get('experiment'),
                      'blockers':errors,'launch':args.launch},ensure_ascii=False,indent=2),flush=True)
    if errors: raise SystemExit(1)
    if args.max_batches is not None and args.max_batches < 1: parser.error('--max-batches must be positive')
    if args.launch: run(cfg,args.max_batches,args.eval_only)

if __name__ == '__main__': main()
