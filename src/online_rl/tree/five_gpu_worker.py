"""Case coordinators and optional shared frozen GPU actors."""
import copy
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

_CFG = _POLICY = None


def initialize(cfg, worker, key, pool_spec=None, existing_policy=None):
    global _CFG, _POLICY
    if existing_policy is None:
        from src.online_rl.tree.devices import physical_gpus
        os.environ['CUDA_VISIBLE_DEVICES'] = physical_gpus()[worker]
    os.environ['SHERPA_AGENT_REPLICAS'] = '1'
    os.environ.pop('SHERPA_DOCTOR_ENDPOINTS', None)
    if cfg.get('api_provider') == 'openai_direct':
        os.environ['OPENAI_API_KEY'] = key
        os.environ.pop('OPENROUTER_API_KEY', None)
    else:
        os.environ['OPENROUTER_API_KEY'] = key
    from scripts.online_rl.run_tree import ROOT
    from src.online_rl.tree.idle_eval_hooks import register_worker, training_guard
    register_worker(worker)
    _CFG = copy.deepcopy(cfg)
    # The parent coordinator owns batch prompt evolution; no per-case prompt event.
    _CFG['prompt_update']['enabled'] = False
    _CFG['output_dir'] = str((ROOT/cfg['output_dir']).resolve())
    from src.common.live_gpu import configure, activity, note
    configure(_CFG['output_dir'],worker)
    _CFG['rollout']['doctor_backend']['endpoint'] = cfg['five_gpu']['doctor_urls'][worker]
    from src.online_rl.runtime import SharedPolicy
    from src.online_rl.tree.five_gpu import cache_openings
    if existing_policy is None:
        with training_guard(worker, 'model_load', release_memory=True), activity('model_load'):
            _POLICY = SharedPolicy(_CFG, key)
    else:
        _POLICY = existing_policy
    note(phase='ready')
    # Each generation seeds under gen_lock; episode-level reseeds would race across lanes.
    _POLICY.seed_all = lambda seed: None
    # Execution-only request capacity is independent of the number of training cases.
    concurrency=int(os.environ.get('SHERPA_DOCTOR_CONCURRENCY',cfg['five_gpu'].get('doctor_concurrency',cfg['five_gpu']['lanes_per_gpu'])))
    if concurrency not in (1,2,4,8):raise ValueError('Doctor concurrency must be 1, 2, 4 or 8')
    _CFG.setdefault('execution',{})['doctor_workers']=concurrency
    _CFG['execution']['agent_workers']=int(os.environ.get('SHERPA_AGENT_BATCH_SIZE',cfg['five_gpu'].get('agent_batch_size',1)))
    for _ in range(concurrency-1):
        _POLICY.doctor._free.put(_CFG['rollout']['doctor_backend']['endpoint'])
    cache_openings(_POLICY.doctor, Path(_CFG['output_dir'])/'openings'/f'train_gpu{worker}.jsonl',
                   cfg['five_gpu']['opening_identity'])
    if pool_spec is not None:
        from src.online_rl.tree.gpu_pool import attach
        attach(_POLICY, _CFG, pool_spec, worker)


def prepare(checkpoint, expected):
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file
    from src.online_rl.integrated.continuation import adapter_sha256
    def load():
        with _POLICY.gen_lock:
            set_peft_model_state_dict(_POLICY.model, load_file(str(Path(checkpoint)/'adapter/adapter_model.safetensors')))
            _POLICY.mark_dirty()
            _POLICY.model.eval()
            if adapter_sha256(_POLICY.model) != expected:
                raise RuntimeError('Rollout worker adapter mismatch')
    actor = getattr(_POLICY, 'pool_actor', None)
    if actor is not None:
        from src.online_rl.tree.idle_eval_hooks import training_guard
        with training_guard(actor.gpu, 'prepare'), actor.changing_policy():
            load()
            actor.expected = expected
            _POLICY.pool_client.expected = expected
    else:
        load()
    return expected


def run_many(jobs, checkpoint, expected, phase='fixed', prompt=None):
    from src.online_rl.integrated.continuation import adapter_sha256
    if not hasattr(_POLICY, 'pool_actor'):
        prepare(checkpoint, expected)
    elif _POLICY.pool_actor.expected != expected:
        raise RuntimeError('Shared GPU worker was not prepared at the batch barrier')
    with ThreadPoolExecutor(max_workers=_CFG['five_gpu']['lanes_per_gpu']) as pool:
        records=list(pool.map(lambda job: run_case(job, phase, prompt), jobs))
    if adapter_sha256(_POLICY.model)!=expected:
        raise RuntimeError('Worker phase changed the frozen Agent policy')
    return records


def run_case(job, phase='fixed', prompt=None):
    from src.common.live_gpu import note, activity
    note(phase=phase,case_index=job['index'],case_id=job['case']['case_id'])
    if hasattr(_POLICY, 'pool_client'):
        _POLICY.pool_client.context = {'case_index':job['index'], 'case_id':job['case']['case_id'], 'phase':phase}
    from scripts.online_rl.run_tree import build
    from src.online_rl.integrated.records import atomic_write_json
    cfg = copy.deepcopy(_CFG)
    cfg['output_dir'] = job['out']
    cfg['schedule']['cases'] = 1
    cfg['schedule'].pop('positions', None)
    cfg['schedule']['calibration_only_cursors'] = []
    trainer = build(cfg, policy=_POLICY, cases=[job['case']], defer_update=True)
    if phase in ('bank_prepare', 'bank_verify', 'bank_score', 'bank_online'):
        from src.online_rl.tree import cumulative_bank as CB
        from src.online_rl.integrated.records import sha256_text
        snapshot = CB.validate_snapshot(prompt['snapshot'])
        if sha256_text(prompt['text']) != prompt['sha256']:
            raise RuntimeError('Cumulative bank phase prompt text/hash mismatch')
        base = Path(job['out']); path = base / f'{phase}.json'

        def validate_phase(record, name, expected_snapshot=None):
            if (record['prompt_sha256'] != prompt['sha256'] or
                    record['policy_sha256'] != trainer.state.policy_sha256 or
                    record['case_id'] != job['case']['case_id']):
                raise RuntimeError('Saved cumulative phase prompt/policy/case mismatch')
            trainer.validate_member_interventions(record)
            if name == 'bank_score':
                scoring = record.get('cumulative_bank_scoring') or {}
                actual = CB.validate_snapshot(scoring.get('snapshot'))
                if scoring.get('snapshot_sha256') != actual['sha256']:
                    raise RuntimeError('Saved cumulative scoring snapshot identity mismatch')
            else:
                actual = CB.validate_snapshot(record.get('cumulative_bank_input'))
            if expected_snapshot is not None and actual['sha256'] != expected_snapshot['sha256']:
                raise RuntimeError('Saved cumulative phase snapshot changed')
            if name not in ('bank_prepare',) and record['rubric_bank'].get('verification_pending'):
                raise RuntimeError('Saved cumulative verification/scoring is incomplete')
            return record

        if path.exists():
            return validate_phase(json.loads(path.read_text()), phase, snapshot)
        phase_started = time.perf_counter()
        if phase == 'bank_online':
            with activity('bank_online.trajectory_match_generate_verify'):
                record = trainer.step(prompt_text=prompt['text'], prompt_id=prompt['id'],
                                      score_responses=False, commit=False, apply_update=False,
                                      defer_verification=False, cumulative_snapshot=snapshot,
                                      online_working_bank=True)
        elif phase == 'bank_prepare':
            frozen = None
            seeds = (cfg.get('five_gpu') or {}).get('first_batch_frozen_records') or {}
            seed = seeds.get(str(job['index']))
            if seed is not None:
                import hashlib
                seed_path = Path(seed['file'])
                if not seed_path.is_absolute(): seed_path = Path(__file__).resolve().parents[3] / seed_path
                if hashlib.sha256(seed_path.read_bytes()).hexdigest() != seed['sha256']:
                    raise RuntimeError('Frozen first-batch record hash changed')
                frozen = json.loads(seed_path.read_text())
                if (frozen.get('case_id') != job['case']['case_id'] or
                        frozen.get('policy_sha256') != trainer.state.policy_sha256 or
                        not frozen.get('tree') or not frozen.get('feedback')):
                    raise RuntimeError('Frozen first-batch tree/feedback identity changed')
            with activity('bank_prepare.tree_rubric_reuse'):
                record = trainer.step(frozen_record=frozen, prompt_text=prompt['text'], prompt_id=prompt['id'],
                                      score_responses=False, commit=False, apply_update=False,
                                      defer_verification=True, cumulative_snapshot=snapshot)
            if frozen is not None:
                record['frozen_rollout_source'] = {'file': str(seed_path), 'sha256': seed['sha256'],
                                                   'reused_fields': ['tree', 'feedback']}
        elif phase == 'bank_verify':
            prepared = validate_phase(json.loads((base / 'bank_prepare.json').read_text()), 'bank_prepare', snapshot)
            with activity('bank_verify.witness_verification'):
                record = trainer.finish_deferred_verification(prepared)
        else:
            # The input snapshot predates new KEEP additions; its identity is
            # independently validated, while all five scores use this new set.
            source = base / ('bank_online.json' if (base / 'bank_online.json').exists() else 'bank_verify.json')
            verified = validate_phase(json.loads(source.read_text()), source.stem)
            with activity('bank_score.frozen_cumulative_scoring'):
                record = trainer.score_cumulative(verified, snapshot)
        record.setdefault('timings', {})[phase + '_wall_seconds'] = time.perf_counter() - phase_started
        record.update(case_cursor=job['index'], rollout_gpu=job['worker'], dialogue_lane=job['lane'])
        for decision in record.get('online_working_bank_decisions', []):
            decision['case_cursor'] = job['index']
        validate_phase(record, phase, snapshot)
        atomic_write_json(path, record)
        note(phase='waiting_at_batch_barrier', finished_phase=phase)
        return record
    if phase != 'fixed':
        from src.online_rl.tree.batch_prompt import evaluate_prompt_record, score_union
        base = Path(job['out'])
        path = base/f'{phase}.json'
        if path.exists():
            record = json.loads(path.read_text())
            expected_sha = prompt['sha256'] if phase in ('P0_prepare','P0_verify','P1_prepare','P1_verify') else prompt['parent_sha256']
            if record['prompt_sha256'] != expected_sha or record['policy_sha256'] != trainer.state.policy_sha256:
                raise RuntimeError('Saved batch phase prompt/policy mismatch')
            trainer.validate_member_interventions(record)
            return record
        if phase in ('P0_prepare','P1_prepare'):
            # The phase-barrier workflow stores verified records under the
            # explicit phase name; P1 must reuse the frozen P0 tree/feedback.
            frozen = json.loads((base/'P0_verify.json').read_text()) if phase == 'P1_prepare' else None
            with activity(phase + '.tree_rubric_witness'):
                record = trainer.step(frozen_record=frozen, prompt_text=prompt['text'], prompt_id=prompt['id'],
                                      score_responses=False, commit=False, apply_update=False,
                                      defer_verification=True)
        elif phase in ('P0_verify','P1_verify'):
            prepared = json.loads((base/f'{phase.removesuffix("_verify")}_prepare.json').read_text())
            if prepared['prompt_sha256'] != prompt['sha256']:
                raise RuntimeError('Prepared rubric bank prompt mismatch')
            with activity(phase + '.witness_verification'):
                record = trainer.finish_deferred_verification(prepared)
            with activity(phase + '.full_member_effect'):
                record['prompt_evaluation'] = evaluate_prompt_record(trainer, record)
        elif phase == 'union':
            parent = json.loads((base/'P0_verify.json').read_text())
            candidate = json.loads((base/'P1_verify.json').read_text())
            if candidate['prompt_sha256'] != prompt['sha256']:
                raise RuntimeError('Union candidate prompt mismatch')
            with activity('union_scoring'):
                record = score_union(trainer, parent, candidate, prompt['text'])
        else:
            raise ValueError(f'Unknown worker phase {phase}')
        record.update(case_cursor=job['index'], rollout_gpu=job['worker'], dialogue_lane=job['lane'])
        atomic_write_json(path,record)
        note(phase='waiting_at_batch_barrier',finished_phase=phase)
        return record
    path = Path(job['out'])/'cases/case_000000.json'
    record = json.loads(path.read_text()) if trainer.state.case_cursor else trainer.step()
    trainer.validate_member_interventions(record)
    record.update(case_cursor=job['index'], rollout_gpu=job['worker'], dialogue_lane=job['lane'])
    atomic_write_json(Path(job['out'])/'result.json', record)
    return record
