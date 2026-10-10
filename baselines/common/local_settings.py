"""Existing repository recipe and local artifacts; no external manifest required."""
import copy
import hashlib
import importlib.metadata
import json
import os
import re
import shutil
from pathlib import Path
from baselines.common.protocol import ROOT,apply_common
from src.online_rl.tree.experiment import load_config
from src.online_rl.integrated.records import sha256_json

COMMON_CONFIG='configs/online_rl/tree_v21_five_gpu_shared5_random10_promptv7.json'
LOCAL_MODEL='/convei_nas2/huggingface_cache/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218'
PACKAGES=('torch','transformers','peft','accelerate','safetensors','huggingface-hub')


def recipe():
    cfg=load_config(COMMON_CONFIG)
    # Existing no-prompt-optimization comparison, with the same train120 overlay.
    cfg['prompt_update']['enabled']=False
    return apply_common(cfg)


def model_identity(path):
    from baselines.common.model_cache import check_snapshot
    root=Path(path).resolve();errors=check_snapshot(root)
    if errors:raise ValueError('; '.join(errors))
    config=json.loads((root/'config.json').read_text())
    if config.get('model_type')!='qwen3' or config.get('hidden_size')!=4096 or config.get('num_hidden_layers')!=36:
        raise ValueError('Local model is not the expected Qwen3-8B architecture')
    small={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in root.glob('*.json')}
    index=json.loads((root/'model.safetensors.index.json').read_text())
    shards={name:{'bytes':(root/name).stat().st_size,'blob':(root/name).resolve().name,
        'mtime_ns':(root/name).stat().st_mtime_ns} for name in sorted(set(index['weight_map'].values()))}
    revision=root.name if re.fullmatch('[0-9a-f]{40}',root.name) else None
    return {'path':str(root),'snapshot_revision':revision,'json_sha256':small,'shards':shards,
        'identity_sha256':sha256_json({'json':small,'shards':shards}),
        'weight_verification':'Existing HF blob identifiers, sizes and mtimes recorded; full shard bytes not rehashed.'}


def resolve_runtime(value):
    runtime=copy.deepcopy(value);cfg=recipe()
    runtime.setdefault('base_model_path',LOCAL_MODEL)
    runtime.setdefault('initialization_registry','outputs/common_initialization_v4')
    # Comparisons use exactly the terminal environment key, never a config value
    # or a .env file. Both preflight and execution use the same resolver.
    runtime['api_key_env']='OPENROUTER_API_KEY';runtime['env_file']=None
    runtime.setdefault('api_base_url','https://openrouter.ai/api/v1')
    runtime.setdefault('training_physical_gpus',['0','1','2','3','4'])
    runtime['ollama_binary']=runtime.get('ollama_binary') or os.environ.get('SHERPA_OLLAMA_BINARY') or cfg['five_gpu'].get('ollama_binary','ollama')
    runtime['ollama_models_dir']=runtime.get('ollama_models_dir') or os.environ.get('SHERPA_OLLAMA_MODELS') or str(Path.home()/'.ollama/models')
    runtime['package_versions']={}
    errors=[]
    expected_doctor=cfg['rollout']['doctor_backend']['model_digest']
    if runtime.get('doctor_model_digest',expected_doctor)!=expected_doctor:errors.append('Doctor digest differs from the user-approved pin')
    runtime['doctor_model_digest']=expected_doctor
    for package in PACKAGES:
        try:runtime['package_versions'][package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:errors.append('Required package missing: '+package)
    try:
        runtime['local_model_identity']=model_identity(runtime['base_model_path'])
        runtime['base_model_path']=runtime['local_model_identity']['path']
        runtime['base_model_revision']=runtime['local_model_identity']['snapshot_revision'] or runtime['local_model_identity']['identity_sha256']
    except (OSError,ValueError,KeyError) as exc:errors.append('Local Qwen snapshot: '+str(exc))
    # The repository's seeded fresh LoRA is reused. No external adapter artifact/hash required.
    if runtime.get('init_adapter') not in (None,cfg['policy'].get('init_adapter')):errors.append('Keep the existing seeded fresh LoRA initialization; custom init_adapter is not part of this recipe')
    runtime['init_adapter']=cfg['policy'].get('init_adapter')
    for role,model in [('generator',cfg['rubric']['generator_model']),('grader',cfg['judges']['scorer_model'])]:
        entry=runtime.get(role) or {}
        request=entry.get('request_model') or model
        if entry.get('model') not in (None,model) or request!=model:errors.append(role+' must reuse the existing configured model '+model)
        accepted=[model]
        if role=='grader':
            canonical=cfg['judges'].get('luna_canonical_slug','').split(' ',1)[0]
            if canonical and canonical not in accepted:accepted.append(canonical)
        runtime[role]={'model':model,'request_model':request,'revision':entry.get('revision') or model,
            'accepted_response_models':[entry['revision']] if entry.get('revision') and entry['revision']!=model else accepted}
    return runtime,errors


def common_record(runtime):
    cfg=recipe()
    from src.online_rl import agent_protocol as ap
    prompt_files=[cfg['rollout'].get('agent_system_prompt_file','prompts/agent.txt'),cfg['rollout']['doctor_system_prompt_file']]
    generation={}
    if runtime.get('local_model_identity'):
        generation=json.loads((Path(runtime['base_model_path'])/'generation_config.json').read_text())
    return {'authority':'existing_repository_config_and_code; no external SHERPA manifest required',
        'source_config':COMMON_CONFIG,'source_config_sha256':hashlib.sha256((ROOT/COMMON_CONFIG).read_bytes()).hexdigest(),
        'policy':cfg['policy'],'rollout':cfg['rollout'],'optimizer':cfg['optimizer'],'advantage_normalization':cfg['advantage_normalization'],
        'schedule':cfg['schedule'],'batch_cases':cfg['five_gpu']['batch_cases'],'test':{k:cfg['five_gpu'][k] for k in ('eval_file','eval_sha256','eval_seed','eval_cases')},
        'prompt_sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in prompt_files},
        'private_assessment_protocol':ap.IDENTITY,'local_model':runtime.get('local_model_identity'),
        'effective_agent_top_k':cfg['rollout']['top_k'] if cfg['rollout']['top_k'] is not None else generation.get('top_k'),
        'packages':runtime['package_versions'],'initial_adapter_hash':'measured at user-run startup; compared across methods and on resume',
        'reward_weights_are_method_specific':True,'rar_weights':{'w_outcome':0.,'w_rubric':1.},'online_rubrics_weights':{'w_outcome':0.,'w_rubric':1.},
        'model_loader_compatibility':'For Transformers <4.56, translate dtype=bfloat16 to torch_dtype=bfloat16; same model and dtype.'}
