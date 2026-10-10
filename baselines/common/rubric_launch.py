"""Explicit preflight / train / final evaluation; never starts work on import."""
import argparse
import copy
import importlib.metadata
import json
import os
import re
import shutil
from pathlib import Path
from src.online_rl.integrated.records import atomic_write_json, from_dict, sha256_json
from src.online_rl.tree import experiment
from src.online_rl.tree.records import Tree
from baselines.common.data import ROOT, audit_data, _cases
from baselines.common.runner import isolated_runner, source_hash
from baselines.common.protocol import apply_common, identity
from baselines.common.local_settings import resolve_runtime, recipe, common_record
from baselines.common.final_eval import evaluate_completed_run
from baselines.common.rubrics import RubricEngine
from baselines.common.backend import AuditedBackend
from baselines.common.credentials import read_api_key,credential_status,API_KEY_ENV

from baselines.common.topology import is_four,assistant_devices,owned_devices

REFERENCE = 'configs/experiments/reward_prompt_matrix/reward_sweep/r0_o1_outcome_only.json'
PACKAGES = ('torch','transformers','peft','accelerate','safetensors','huggingface-hub')


def configuration(method, output, receipt, engine, initial_policy_sha256, base_model_revision, *, spec=None, runtime=None):
    cfg = apply_common(experiment.load_config(REFERENCE))
    spec = spec or json.loads((ROOT/'baselines'/method/'method.json').read_text())
    runtime = runtime or {}
    cfg.update(name=spec['label'],condition=spec['label'],output_dir=str(Path(output).resolve()))
    for section,key,role in (('data','train','train'),('five_gpu','train_order_file','train_order'),('five_gpu','eval_file','test')):
        cfg[section][key] = str(Path(receipt['files'][role]['path']).resolve())
    if runtime.get('base_model_path'):cfg['policy']['base_model']=str(Path(runtime['base_model_path']).resolve())
    if runtime.get('init_adapter'):cfg['policy']['init_adapter']=str(Path(runtime['init_adapter']).resolve())
    for key in ('ollama_binary','ollama_models_dir'):
        if runtime.get(key):cfg['five_gpu'][key]=runtime[key]
    cfg['five_gpu']['doctor_urls']=[f'http://127.0.0.1:{spec["doctor_port_start"]+i}' for i in range(2 if is_four(spec) else 5)]
    if is_four(spec):
        cfg['four_gpu']={k:spec[k] for k in ('doctor_physical_gpus','assistant_physical_gpus','eval_workers')}
        cfg['four_gpu']['execution']='doctor2_assistant2_v1'
        cfg['five_gpu'].update(rollout_gpus=[0,1],update_gpu=1)
    cfg['rollout']['doctor_backend']['endpoint']=cfg['five_gpu']['doctor_urls'][0]
    # Workers run only the original outcome tree. The batch hook replaces targets
    # after all five frozen-policy trees arrive, before the original updater.
    cfg['comparison_method']={'method_spec':spec,'method':method,'source_sha256':source_hash(),
        'engine':engine.identity if engine else None,'actual_reward_weights':engine.weights if engine else ({'w_outcome':1.,'w_rubric':0.} if method=='outcome_only' else None),
        'initial_policy_sha256':initial_policy_sha256,'base_model_revision':base_model_revision,
        'common_identity':identity(cfg),'package_versions':runtime.get('package_versions',{}),
        'data_sha256':{k:v['actual_sha256'] for k,v in receipt['files'].items()},
        'local_model_identity':runtime.get('local_model_identity'),'initialization_registry':runtime.get('initialization_registry'),
        'authority':'existing_repository_config_and_code',
        'doctor_identity':{k:cfg['rollout']['doctor_backend'][k] for k in ('model','model_digest','ollama_version')},
        'doctor_change_record':'baselines/common/doctor_change.json'}
    return cfg


class RubricHook:
    def __init__(self,cfg,receipt,engine,cases):
        self.cfg,self.receipt,self.engine,self.cases=cfg,receipt,engine,cases
        self.fingerprint=sha256_json(cfg['comparison_method'])
    def __getattr__(self,name):return getattr(experiment,name)
    def receipt_fields(self,cfg):return {'experiment_fingerprint':self.fingerprint,'method_label':cfg['comparison_method']['method']}
    def state_fields(self,cfg,theta0):return {**self.receipt_fields(cfg),'theta0_policy_sha256':theta0}
    def check_receipt(self,cfg,receipt,what):
        if any(receipt.get(k)!=v for k,v in self.receipt_fields(cfg).items()):raise RuntimeError(f'{what}: method identity changed')
    def check_pfinal_source(self,cfg,theta0,protocol):
        from baselines.common.initialization import record_initial
        record_initial(cfg,theta0)
        ref=getattr(self.engine,'reference',None)
        if hasattr(ref,'observe_initial'):ref.observe_initial(theta0)
    def check_resume(self,cfg,out,state,theta0):
        self.check_receipt(cfg,state,'Run state');self.check_pfinal_source(cfg,theta0,None)
        checkpoint=Path(state['checkpoint']).resolve()
        if checkpoint.parent!=(Path(out)/'checkpoints').resolve() or state.get('theta0_policy_sha256')!=theta0:
            raise RuntimeError('Checkpoint/initial adapter differs')
        self.check_receipt(cfg,json.loads((checkpoint/'COMPLETE.json').read_text()),'Checkpoint')
    def check_batch(self,cfg,records):
        raw_rows=experiment.check_batch(cfg,records)
        start=records[0]['case_cursor']
        if (len(records)!=5 or start%5 or [r['case_cursor'] for r in records]!=list(range(start,start+5))
                or [r['case_id'] for r in records]!=self.receipt['train_case_order'][start:start+5]):
            raise RuntimeError('Original five-case order differs')
        if len({r['policy_sha256'] for r in records})!=1:raise RuntimeError('Mixed frozen policies')
        if self.engine is None:return raw_rows
        rows=[]
        for record in records:
            if record.get('update',{}).get('optimizer_step'):raise RuntimeError('Already updated record')
            tree=from_dict(Tree,record['tree'])
            targets,report=self.engine.score(tree,self.cases[record['case_id']])
            from src.online_rl.tree.reward import sibling_group_diagnostics, interaction_diagnostics
            record['agent_targets']=targets
            record['group_diagnostics']=sibling_group_diagnostics(tree,targets)
            record['interaction_diagnostics']=interaction_diagnostics(tree,targets)
            record['method_label']=cfg['comparison_method']['method'];record['rubric_adaptation']=report
            atomic_write_json(Path(cfg['output_dir'])/'method_reports'/f"case_{record['case_cursor']:06d}.json",report)
            rows.append({'case_id':tree.case_id,'nodes':targets['nodes']})
        return rows


def bind(cfg, *, create=True):
    path=Path(cfg['output_dir'])/'method_binding.json';value=cfg['comparison_method']
    if path.exists():
        if json.loads(path.read_text())!=value:raise RuntimeError('Resume method settings changed; use the original config/source/cache')
    elif (Path(cfg['output_dir'])/'run_state.json').exists():raise RuntimeError('Missing method binding')
    elif create:
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('x') as stream:json.dump(value,stream,indent=2)


def setup(method,spec,runtime):
    """Read-only preparation from existing local config/model; no manifest prerequisite."""
    runtime,errors=resolve_runtime(runtime)
    if spec.get('doctor_digest')!=recipe()['rollout']['doctor_backend']['model_digest']:errors.append('Method config Doctor pin differs; use the new isolated output/cache settings')
    if spec['method']!=method or spec['schema']!='direct_candidate_comparison_v2':raise ValueError('Wrong method config')
    if any(spec[k] for k in ('partner_reflection','verification','prompt_optimization')):raise ValueError('Baseline exclusions changed')
    if spec['current_reference_pairs']!=(1 if method=='online_rubrics' else 0):raise ValueError('Wrong reference pair count')
    receipt=audit_data(train=runtime.get('train_file'),train_order=runtime.get('train_order_file'),test=runtime.get('test_file'))
    cfg=configuration(method,spec['output_dir'],receipt,None,runtime.get('initial_policy_sha256'),runtime.get('base_model_revision'),spec=spec,runtime=runtime)
    engine=None;reference=None
    if not receipt['all_verified']:errors.append('Train/order/test verification incomplete')
    gpus=assistant_devices(spec,runtime)
    if is_four(spec):
        owned=owned_devices(spec,runtime)
        if len(owned)!=4 or len(set(owned))!=4 or any(not str(g).isdigit() for g in owned):errors.append('Four distinct physical GPUs required')
        if len(gpus)!=2 or len(spec['doctor_physical_gpus'])!=2:errors.append('Doctor2 + Assistant2 required')
    elif len(gpus)!=5 or len(set(gpus))!=5:errors.append('Specify five distinct physical training GPU IDs')
    if method!='outcome_only':
        weights={'w_outcome':0.,'w_rubric':1.}
        if spec.get('reward_weights',weights)!=weights:errors.append('RaR/OnlineRubrics require rubric-only reward')
        common=recipe();roles={}
        for role in ('generator','grader'):
            entry=runtime[role]
            roles[role]=AuditedBackend(audit_path=Path(spec['output_dir'])/'reward_api_attempts.jsonl',
                base_url=runtime['api_base_url'],model=entry['model'],request_model=entry['request_model'],
                revision=entry['revision'],accepted_response_models=entry['accepted_response_models'],
                api_key_env=runtime['api_key_env'],max_retries=2)
        if method=='online_rubrics':
            if not is_four(spec):raise ValueError('Use the configured four-GPU OnlineRubrics launcher')
            from baselines.common.pooled_reference import PooledInitialReference
            reference=PooledInitialReference(cfg,Path(spec['output_dir'])/'rewards',base_revision=runtime.get('base_model_revision'),initial_policy_sha256=runtime.get('initial_policy_sha256'))
        gen=common['rubric']['generation']
        gen_params={'temperature':gen['temperature'],'max_tokens':gen['max_tokens'],
            'reasoning':{'effort':gen['reasoning_effort']},'response_format':{'type':'json_object'}}
        grade_params={'temperature':0.,'max_tokens':4096,'response_format':{'type':'json_object'}}
        engine=RubricEngine(method,roles['generator'],roles['grader'],Path(spec['output_dir'])/'rewards',
            weights=weights,seed=cfg['schedule']['seed'],reference=reference,generator_params=gen_params,
            grader_params=grade_params,initial_rubric_source=spec.get('initial_rubric_source'),initial_registry=spec.get('initial_rubric_registry'))
        cfg=configuration(method,spec['output_dir'],receipt,engine,runtime.get('initial_policy_sha256'),runtime.get('base_model_revision'),spec=spec,runtime=runtime)
    cfg['comparison_method']['common_settings']=common_record(runtime)
    return cfg,receipt,engine,reference,errors


def resource_checks(cfg,runtime,method,spec,*,gpu):
    errors=[]
    from baselines.common.doctors import assert_endpoints_unused
    try:assert_endpoints_unused(cfg)
    except (OSError,ValueError,RuntimeError) as exc:errors.append(str(exc))
    binary=runtime.get('ollama_binary')
    if not binary or not shutil.which(binary):errors.append('Set ollama_binary to the Ollama 0.12.5 executable')
    models=runtime.get('ollama_models_dir')
    digest=cfg['rollout']['doctor_backend']['model_digest']
    import hashlib
    tag=cfg['rollout']['doctor_backend']['model'];name,version=tag.split(':')
    manifest=Path(models or '/nonexistent')/'manifests/registry.ollama.ai/library'/name/version
    if not manifest.is_file() or hashlib.sha256(manifest.read_bytes()).hexdigest()!=digest:errors.append('Pinned Doctor model manifest missing/different')
    try:read_api_key()
    except RuntimeError as exc:errors.append(str(exc))
    from baselines.common.model_cache import check_snapshot
    errors+=check_snapshot(runtime['base_model_path'])
    if gpu:
        import torch,subprocess
        if not torch.cuda.is_available() or torch.cuda.device_count()!=len(assistant_devices(spec,runtime)):errors.append('Assistant CUDA visibility differs from topology')
        if binary and shutil.which(binary):
            version=subprocess.run([binary,'--version'],capture_output=True,text=True)
            if '0.12.5' not in version.stdout+version.stderr:errors.append('Ollama version differs from 0.12.5')
        if is_four(spec) or method=='online_rubrics':
            found=subprocess.run(['nvidia-smi','--query-gpu=index','--format=csv,noheader'],capture_output=True,text=True)
            if not set(owned_devices(spec,runtime))<=set(found.stdout.split()):errors.append('Requested physical GPUs unavailable')
    return errors


def main(method):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default=f'baselines/{method}/method.json')
    p.add_argument('--runtime',default='baselines/runtime.local.json')
    p.add_argument('--action',choices=('preflight','train','evaluate','stop-doctors','audit-existing'),default='preflight')
    p.add_argument('--cpu-only',action='store_true');p.add_argument('--max-batches',type=int);p.add_argument('--existing-run')
    a=p.parse_args()
    if a.cpu_only and a.action!='preflight':p.error('--cpu-only is for preflight only')
    if a.max_batches is not None and (a.action!='train' or a.max_batches<1):p.error('--max-batches must be positive, for train only')
    spec=json.loads(Path(a.config).read_text())
    if not Path(a.runtime).is_file():p.error('Missing runtime JSON; copy baselines/runtime.example.json and set local resource paths')
    runtime,local_errors=resolve_runtime(json.loads(Path(a.runtime).read_text()))
    if runtime.get('hf_hub_cache'):os.environ['HF_HUB_CACHE']=str(Path(runtime['hf_hub_cache']).resolve())
    os.environ['CUDA_VISIBLE_DEVICES']=','.join(map(str,assistant_devices(spec,runtime)))
    if runtime.get('ollama_binary'):os.environ['SHERPA_OLLAMA_BINARY']=runtime['ollama_binary']
    if runtime.get('ollama_models_dir'):os.environ['SHERPA_OLLAMA_MODELS']=runtime['ollama_models_dir']
    os.environ['SHERPA_AGENT_REPLICAS']='1'
    if a.action=='stop-doctors':
        from baselines.common.doctors import stop
        stop(spec,runtime);return
    cfg,receipt,engine,reference,errors=setup(method,spec,runtime)
    errors=list(dict.fromkeys(local_errors+errors))
    if a.action!='audit-existing':errors+=resource_checks(cfg,runtime,method,spec,gpu=not a.cpu_only)
    bind(cfg,create=False)
    print(json.dumps({'method':spec['label'],'action':a.action,'cpu_only':a.cpu_only,'blockers':errors,
        'data_verified':receipt['all_verified'],'reward_weights':cfg['comparison_method']['actual_reward_weights'],
        'credentials':credential_status(),'python_executable':__import__('sys').executable,
        'doctor':cfg['comparison_method']['doctor_identity'],'opening_doctor_digest':cfg['five_gpu']['opening_identity']['model_digest'],
        'output':cfg['output_dir'],'training_cases':cfg['schedule']['cases'],'live_work_started':False,'physical_gpu_roles':cfg.get('four_gpu'),
        'memory_note':'No GPU memory smoke performed. BF16 base fits only if KV/cache/activations also fit; no automatic quantization or batch reduction.'},indent=2))
    if errors:raise SystemExit(1)
    if a.action=='audit-existing':
        if method!='outcome_only' or not a.existing_run:p.error('audit-existing requires outcome_only and --existing-run')
        from baselines.common.existing import audit
        print(json.dumps(audit(a.existing_run,cfg),indent=2));return
    cases={c['case_id']:c for c in _cases(receipt['files']['train']['path'])}
    hook=RubricHook(cfg,receipt,engine,cases);runner=isolated_runner(hook)
    errors=runner.preflight(cfg)
    if errors:raise RuntimeError('; '.join(errors))
    if a.action=='preflight':return
    statepath=Path(cfg['output_dir'])/'run_state.json'
    if a.action=='evaluate':
        if not statepath.is_file() or json.loads(statepath.read_text())['case_cursor']!=120:raise RuntimeError('Complete train120 before evaluating test')
    os.environ['HF_HUB_OFFLINE']='1'
    # Same resolver as preflight; inherited by workers and the original runner.
    os.environ[API_KEY_ENV]=read_api_key()
    bind(cfg)
    atomic_write_json(Path(cfg['output_dir'])/'resolved_launch_config.json',cfg)
    from baselines.common.doctors import start,stop,assert_endpoints_unused
    # Own every assigned physical device before starting Doctors. Disjoint
    # four-GPU methods can run concurrently; overlapping assignments cannot.
    import fcntl
    locks=[]
    try:
        for device in sorted(set(owned_devices(spec,runtime))):
            lock=open(f'/tmp/comparison_launcher_gpu{device}.lock','a')
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:
                lock.close();raise RuntimeError('Another comparison launcher owns GPU '+device)
            locks.append(lock)
    except Exception:
        for lock in locks:lock.close()
        raise
    doctor_start_attempted=False
    try:
        # Check again after taking GPU locks. A failed read-only guard must not
        # enter cleanup that could terminate a previously running server.
        assert_endpoints_unused(cfg)
        doctor_start_attempted=True
        start(cfg)
        if a.action=='train':runner.run(cfg,max_batches=a.max_batches)
        else:evaluate_completed_run(runner,cfg,method_identity=hook.fingerprint)
    finally:
        if reference:reference.close()
        try:
            if doctor_start_attempted:stop(spec,runtime)
        finally:
            for lock in locks:lock.close()
        from baselines.common.costs import summarize
        atomic_write_json(Path(cfg['output_dir'])/'costs.json',summarize([cfg['output_dir']]))
