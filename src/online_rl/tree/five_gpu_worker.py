"""Case coordinators and optional shared frozen GPU actors."""
import copy
import json
import os
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
    os.environ['OPENROUTER_API_KEY'] = key
    from scripts.online_rl.run_tree import ROOT
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
        with activity('model_load'):_POLICY = SharedPolicy(_CFG, key)
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
        with actor.changing_policy():
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
    if phase != 'fixed':
        from src.online_rl.tree.batch_prompt import evaluate_prompt_record, score_union
        base = Path(job['out'])
        path = base/f'{phase}.json'
        if path.exists():
            record = json.loads(path.read_text())
            expected_sha = prompt['sha256'] if phase in ('P0','P1') else prompt['parent_sha256']
            if record['prompt_sha256'] != expected_sha or record['policy_sha256'] != trainer.state.policy_sha256:
                raise RuntimeError('Saved batch phase prompt/policy mismatch')
            return record
        if phase in ('P0','P1'):
            frozen = json.loads((base/'P0.json').read_text()) if phase == 'P1' else None
            with activity(phase + '.tree_rubric_witness'):
                record = trainer.step(frozen_record=frozen, prompt_text=prompt['text'], prompt_id=prompt['id'],
                                      score_responses=True, commit=False, apply_update=False)
            with activity(phase + '.full_member_effect'):
                record['prompt_evaluation'] = evaluate_prompt_record(trainer, record)
        elif phase == 'union':
            parent = json.loads((base/'P0.json').read_text())
            candidate = json.loads((base/'P1.json').read_text())
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
    record.update(case_cursor=job['index'], rollout_gpu=job['worker'], dialogue_lane=job['lane'])
    atomic_write_json(Path(job['out'])/'result.json', record)
    return record
