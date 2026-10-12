"""Batch barriers and durable, case-at-a-time evaluation for one GPU4 owner."""
import json
import shutil
from contextlib import contextmanager
from pathlib import Path
from src.online_rl.integrated.records import atomic_write_json, from_dict, sha256_json
from src.online_rl.tree.records import Tree


def batch_slots(cursor, total, interval=24, workers=4, lanes=1):
    """Assign a bounded batch, allowing either uniform or per-worker lanes."""
    lane_counts = ([lanes] * workers if isinstance(lanes, int) else list(lanes))
    if (min(interval, workers, *lane_counts) < 1 or len(lane_counts) != workers
            or not 0 <= cursor <= total):
        raise ValueError('Invalid batch dimensions or cursor')
    capacity = sum(lane_counts)
    size = min(total-cursor, interval-cursor % interval, capacity)
    placement = [(worker, lane) for lane in range(max(lane_counts))
                 for worker, count in enumerate(lane_counts) if lane < count]
    return [{'index': cursor+i, 'worker': placement[i][0], 'lane': placement[i][1]}
            for i in range(size)]



def training_order(cfg, cases, root):
    """A pinned unique order permits 500 training cases while reserving calibration IDs."""
    from src.online_rl.cases import file_sha256
    from src.online_rl.integrated.episode import schedule
    byid={c['case_id']:c for c in cases}
    if len(byid)!=len(cases):raise ValueError('Training case IDs must be unique')
    forbidden=set(cfg['data'].get('excluded_calibration_case_ids',[]))
    if forbidden & set(byid):raise ValueError('Reserved calibration IDs occur in training pool')
    spec=cfg['five_gpu'].get('train_order_file')
    if spec:
        path=Path(root)/spec
        if file_sha256(path)!=cfg['five_gpu']['train_order_sha256']:
            raise ValueError('Training order hash differs')
        order=json.loads(path.read_text())
        if len(order)!=len(set(order)) or set(order)!=set(byid):
            raise ValueError('Training order must contain each pool case exactly once')
    else:
        excluded=set(cfg['schedule'].get('calibration_only_cursors',[]))
        order=[schedule(sorted(byid),cfg['schedule']['seed'],i,1)[0]
               for i in range(len(byid)) if i not in excluded]
    if not 1 <= cfg['schedule']['cases'] <= len(order):
        raise ValueError('Training count exceeds non-calibration pool')
    return order


def update_batch(updater, records, expected, actual):
    if actual() != expected:
        raise RuntimeError('Policy changed before update barrier')
    from src.online_rl.integrated.episode import assert_single_prompt
    groups = []
    for record in records:
        tree = from_dict(Tree, record['tree'])
        if tree.policy_sha256 != expected or record['policy_sha256'] != expected:
            raise RuntimeError('Mixed source policies in batch')
        nodes = {n.node_id: n for n in tree.nodes}; targets = record['agent_targets']
        if targets.get('mode') == 'trajectory_level':
            from src.online_rl.integrated.episode import assert_trajectory_group
            group=[]
            for leaf in tree.leaves:
                target=targets['trajectories'][leaf.leaf_id]
                group.append({'trajectory_id':leaf.leaf_id,'case_id':tree.case_id,
                              'reward':target['reward'],'rollout_policy_sha256':tree.policy_sha256,
                              'agent_traces':[nodes[nid].trace for nid in leaf.path]})
            assert_trajectory_group(group); groups.append(group)
        else:
            for parent, ids in sorted(targets['groups'].items()):
                group = [{'trajectory_id': nid, 'case_id': f'{tree.case_id}|{parent}',
                          'reward': targets['nodes'][nid]['reward'],
                          'rollout_policy_sha256': nodes[nid].policy_sha256,
                          'agent_traces': [nodes[nid].trace]} for nid in ids]
                assert_single_prompt(group)
                groups.append(group)
    return updater.update_groups(groups)


def save_checkpoint(updater, path, receipt):
    """Publish all weights, optimizer and receipt together; never expose partial saves."""
    path = Path(path)
    if path.exists():
        old = json.loads((path/'COMPLETE.json').read_text())
        if old != receipt:
            raise RuntimeError('Checkpoint receipt differs')
        return str(path)
    pending = path.with_name(path.name+'.pending')
    shutil.rmtree(pending, ignore_errors=True)
    try:
        updater.save(pending)
        torch = updater.policy.torch
        if torch.cuda.is_available():
            torch.save(torch.cuda.get_rng_state_all(), pending/'cuda_rng.pt')
        atomic_write_json(pending/'COMPLETE.json', receipt)
        pending.rename(path)
    except BaseException:
        shutil.rmtree(pending, ignore_errors=True)
        raise
    return str(path)


def load_checkpoint(updater, path):
    updater.load(str(path))
    torch = updater.policy.torch
    rng = Path(path)/'cuda_rng.pt'
    if rng.exists() and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(torch.load(rng, weights_only=True, map_location='cpu'))


@contextmanager
def evaluation_adapter(policy, checkpoint):
    """Borrow the live GPU4 model; restore weights, RNG and model mode even on error."""
    from peft import get_peft_model_state_dict, set_peft_model_state_dict
    from safetensors.torch import load_file
    from src.online_rl.integrated.continuation import adapter_sha256
    torch = policy.torch
    original_sha = adapter_sha256(policy.model)
    weights = {k: v.detach().cpu().clone() for k,v in get_peft_model_state_dict(policy.model).items()}
    cpu = torch.get_rng_state()
    gpu = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    training = policy.model.training
    use_cache = policy.model.config.use_cache
    try:
        set_peft_model_state_dict(policy.model, load_file(str(Path(checkpoint)/'adapter/adapter_model.safetensors')))
        policy.mark_dirty()
        policy.model.eval()
        yield
    finally:
        set_peft_model_state_dict(policy.model, weights)
        policy.mark_dirty()
        policy.model.train(training)
        policy.model.config.use_cache = use_cache
        torch.set_rng_state(cpu)
        if gpu is not None: torch.cuda.set_rng_state_all(gpu)
        if adapter_sha256(policy.model) != original_sha:
            raise RuntimeError('Evaluation changed training weights')


class EvaluationQueue:
    """Immutable job identity, per-case commit and restartable FIFO progress."""
    def __init__(self, root, cases, settings):
        self.root = Path(root)
        self.cases = cases
        self.settings = settings
        self.root.mkdir(parents=True, exist_ok=True)

    def register(self, checkpoint, cursor, policy_sha):
        identity = {'checkpoint': str(Path(checkpoint).resolve()), 'completed_cases': cursor,
                    'policy_sha256': policy_sha, 'settings': self.settings,
                    'case_ids': [c['case_id'] for c in self.cases]}
        path = self.root/f'checkpoint_{cursor:06d}'
        path.mkdir(exist_ok=True)
        manifest = path/'manifest.json'
        if manifest.exists():
            if json.loads(manifest.read_text()) != identity:
                raise RuntimeError('Evaluation job identity changed')
        else:
            atomic_write_json(manifest, identity)
        return path

    def next_case(self):
        for job in sorted(self.root.glob('checkpoint_*')):
            manifest = json.loads((job/'manifest.json').read_text())
            if manifest['settings'] != self.settings or manifest['case_ids'] != [c['case_id'] for c in self.cases]:
                raise RuntimeError('Evaluation dataset or protocol changed')
            for i, case in enumerate(self.cases):
                if not (job/'cases'/f'case_{i:06d}.json').exists():
                    return job, manifest, i, case
            summary = job/'summary.json'
            if not summary.exists() or json.loads(summary.read_text()).get('status') != 'complete':
                self._summarize(job, manifest)
        return None

    def step(self, play):
        pending = self.next_case()
        if pending is None: return False
        job, manifest, i, case = pending
        result = play(manifest, case, self.settings['seed']+i)
        atomic_write_json(job/'cases'/f'case_{i:06d}.json',
                          {'case_id': case['case_id'], 'index': i,
                           'job_sha256': sha256_json(manifest), **result})
        self._summarize(job, manifest)
        return True

    def run_all_parallel(self, play, workers=25, progress=None, before_job=None):
        """Drain every registered snapshot before training resumes.

        Work is claimed up front and result files are committed by this thread,
        so retries after interruption only rerun missing cases and concurrent
        workers cannot overwrite one another.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        if workers < 1:
            raise ValueError('Evaluation worker count must be positive')
        completed_any = False
        for job in sorted(self.root.glob('checkpoint_*')):
            manifest = json.loads((job/'manifest.json').read_text())
            if (manifest['settings'] != self.settings or
                    manifest['case_ids'] != [c['case_id'] for c in self.cases]):
                raise RuntimeError('Evaluation dataset or protocol changed')
            if before_job:
                before_job(manifest)
            job_sha = sha256_json(manifest)
            todo = [(i, case) for i, case in enumerate(self.cases)
                    if not (job/'cases'/f'case_{i:06d}.json').exists()]
            (job/'cases').mkdir(exist_ok=True)
            if todo:
                with ThreadPoolExecutor(max_workers=workers, thread_name_prefix='checkpoint-eval') as pool:
                    futures = {pool.submit(play, manifest, case, self.settings['seed']+i): (i, case)
                               for i, case in todo}
                    done = 0
                    for future in as_completed(futures):
                        i, case = futures[future]
                        result = future.result()
                        atomic_write_json(job/'cases'/f'case_{i:06d}.json',
                            {'case_id': case['case_id'], 'index': i, 'job_sha256': job_sha, **result})
                        done += 1
                        completed_any = True
                        if progress and (done % 25 == 0 or done == len(todo)):
                            self._summarize(job, manifest)
                            progress(manifest, len(self.cases)-len(todo)+done, len(self.cases))
            self._summarize(job, manifest)
        return completed_any

    def _summarize(self, job, manifest):
        rows = [json.loads(p.read_text()) for p in sorted((job/'cases').glob('case_*.json'))]
        for row in rows:
            if row['job_sha256'] != sha256_json(manifest) or row['case_id'] != self.cases[row['index']]['case_id']:
                raise RuntimeError('Evaluation result belongs to a different job/case')
        atomic_write_json(job/'summary.json', {'completed_cases': manifest['completed_cases'],
            'dataset_cases': len(self.cases), 'evaluated': len(rows),
            'correct': sum(r['outcome'] == 1 for r in rows),
            'unparsed': sum(r['outcome'] is None for r in rows),
            'accuracy': sum(r['outcome'] == 1 for r in rows)/len(rows),
            'mean_agent_turns': sum(r['num_discussion_rounds'] for r in rows)/len(rows),
            'status': 'complete' if len(rows) == len(self.cases) else 'pending'})


def wait_with_evaluation(futures, evaluate_one):
    """Update takes priority at each evaluation case boundary; failures propagate."""
    from concurrent.futures import wait, FIRST_COMPLETED
    while not all(f.done() for f in futures):
        for f in futures:
            if f.done(): f.result()
        if not evaluate_one():
            wait([f for f in futures if not f.done()], timeout=1, return_when=FIRST_COMPLETED)
    return [f.result() for f in futures]


def cache_openings(doctor, path, identity):
    """D1 is independent of Agent weights; one durable opening per case/protocol."""
    from src.online_rl.tree.cache import CallCache
    cache = CallCache(path)
    turn = doctor.turn
    def cached(case, dialogue, round_, rounds, seed, followup_role=None):
        if round_ != 1:
            return turn(case, dialogue, round_, rounds, seed, followup_role=followup_role)
        if followup_role is not None:
            raise ValueError('Doctor follow-up role cannot alter the shared D1 opening')
        view = {k: case.get(k) for k in ('case_id','scenario','question','options')}
        key = {'case': view, 'protocol': identity, 'rounds': rounds}
        # Stable seed independent of checkpoint, lane and rollout branch.
        fixed_seed = int(sha256_json(key)[:8],16) % (2**31-1)
        return cache.call('shared_D1', key, lambda: turn(case, [], 1, rounds, fixed_seed))
    doctor.turn = cached
    return doctor
