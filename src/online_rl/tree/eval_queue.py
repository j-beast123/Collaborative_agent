"""Durable ordered queue for immutable, fully published evaluation snapshots."""
from pathlib import Path
import json
from src.online_rl.integrated.records import atomic_write_json,sha256_json


def enqueue(out,checkpoint,settings,prompts):
    out,checkpoint=Path(out),Path(checkpoint)
    info=json.loads((checkpoint/'COMPLETE.json').read_text())
    state=json.loads((checkpoint/'run_state.json').read_text())
    model=checkpoint/'model'
    if not info['model_saved'] or not (model/'COMPLETE.json').is_file():
        raise RuntimeError('Evaluation requires a fully published model checkpoint')
    if sha256_json(state)!=info['run_state_sha256'] or state['policy_sha256']!=info['policy_sha256']:
        raise RuntimeError('Checkpoint state identity mismatch')
    count=info['completed_cases'];queue=out/'evaluation/queue';queue.mkdir(parents=True,exist_ok=True)
    job={'completed_cases':count,'checkpoint':str(checkpoint.resolve()),'model_checkpoint':str(model.resolve()),
         'policy_sha256':info['policy_sha256'],'settings':settings,'prompts':prompts}
    job['job_sha256']=sha256_json(job);path=queue/f'case_{count:06d}.json'
    if path.exists():
        if json.loads(path.read_text())!=job:raise RuntimeError('Evaluation job identity changed')
    else:atomic_write_json(path,job)
    return path


def pending_jobs(out):
    out=Path(out)
    for path in sorted((out/'evaluation/queue').glob('case_*.json')):
        job=json.loads(path.read_text());copy=dict(job);digest=copy.pop('job_sha256')
        if sha256_json(copy)!=digest:raise RuntimeError('Evaluation job was modified')
        done=out/'evaluation/results'/path.stem/'summary.json'
        if done.exists():
            if json.loads(done.read_text())['job_sha256']!=digest:raise RuntimeError('Evaluation result belongs to another job')
            continue
        yield path,job


def register_ready(out,settings,root):
    """Discovery is independent of the trainer; even a stopped trainer can drain."""
    out,root=Path(out),Path(root)
    from src.online_rl.integrated.continuation import protocol_sha256
    for checkpoint in sorted((out/'checkpoints/periodic').glob('case_*')):
        if not (checkpoint/'COMPLETE.json').is_file():continue
        if (out/'evaluation/queue'/f'{checkpoint.name}.json').exists():continue
        manifest=json.loads((checkpoint/'run_manifest.json').read_text())
        if protocol_sha256(manifest['config'],root)!=manifest['protocol_sha256']:
            raise RuntimeError('Evaluation prompt files differ from the training protocol')
        prompts={'agent':(root/'prompts/agent.txt').read_text(),'doctor':(root/'prompts/user_simulator.txt').read_text()}
        enqueue(out,checkpoint,settings,prompts)
