"""One immutable common P1 from the first completed P0, evaluated as cases become ready."""
import json
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
from src.online_rl.integrated.records import atomic_write_json, sha256_json

SEED_MODE = 'first_completed_P0_v1'


def evaluate(jobs, parent, submit, propose, batchdir, policy_sha256, eager_union=False):
    """submit(phase, job, prompt) returns a Future. Policy preparation must finish BEFORE this call.

    The seed case and candidate are persisted before P1 work. A resumed batch uses
    exactly that seed/candidate, regardless of which cached P0 happens to finish first.
    """
    batchdir.mkdir(parents=True, exist_ok=True)
    seed_path = batchdir / 'prompt_seed.json'
    proposal_path = batchdir / 'prompt_candidate.json'
    identity = {'parent_prompt_sha256': parent['sha256'], 'policy_sha256': policy_sha256,
                'case_ids': [j['case']['case_id'] for j in jobs], 'seed_mode': SEED_MODE}
    seed = json.loads(seed_path.read_text()) if seed_path.exists() else None
    candidate = json.loads(proposal_path.read_text()) if proposal_path.exists() else None
    if seed and any(seed.get(k) != v for k, v in identity.items()): raise RuntimeError('Saved prompt seed differs from this batch')
    if candidate and (seed is None or candidate.get('seed_sha256') != sha256_json(seed)):
        raise RuntimeError('Saved candidate differs from its prompt seed')
    # An earlier version could persist the unchanged P0 as an invalid P1 after
    # a local guard rejection. Re-run refinement against the same saved P0
    # evidence; the call cache reuses the API response while the corrected guard
    # revalidates it. Never treat that fallback as a real P1 on resume.
    if candidate and not candidate.get('valid_candidate', True):
        candidate = None
    events = Queue(); p0 = {}; p1 = {}; unions = {}; submitted = set(); proposing = False

    def watch(kind, index, future):
        future.add_done_callback(lambda done: events.put((kind, index, done)))

    def start_p1(index):
        if index in submitted: return
        submitted.add(index)
        prompt = {'text': candidate['text'], 'sha256': candidate['sha256'], 'id': candidate['id']}
        watch('P1', index, submit('P1', jobs[index], prompt))

    with ThreadPoolExecutor(max_workers=1, thread_name_prefix='common-P1-proposal') as proposal_pool:
        for i, job in enumerate(jobs): watch('P0', i, submit('P0', job, parent))
        while len(unions if eager_union else p1) < len(jobs):
            kind, index, future = events.get()
            value = future.result()  # errors propagate; no changed policy or silent fallback
            if kind == 'P0':
                if value['case_id'] != jobs[index]['case']['case_id'] or value['policy_sha256'] != policy_sha256:
                    raise RuntimeError('P0 case or frozen policy mismatch')
                p0[index] = value
                if seed is None:
                    seed = {**identity, 'seed_case_id': value['case_id'], 'seed_job_index': index,
                            'seed_record_sha256': sha256_json(value)}
                    atomic_write_json(seed_path, seed)
                    print(json.dumps({'stage':'prompt_seed_selected','seed_case_id':seed['seed_case_id'],
                                      'candidate_seed':SEED_MODE}),flush=True)
                if index == seed['seed_job_index']:
                    if sha256_json(value) != seed['seed_record_sha256']: raise RuntimeError('Saved seed P0 evidence changed')
                    if candidate is None and not proposing:
                        proposing = True
                        watch('proposal', index, proposal_pool.submit(propose, [value]))
                if candidate is not None: start_p1(index)
            elif kind == 'proposal':
                candidate = {**value, 'id': f'P1-{value["sha256"][:12]}', 'seed_sha256': sha256_json(seed)}
                atomic_write_json(proposal_path, candidate)
                print(json.dumps({'stage':'prompt_P1_fixed','seed_case_id':seed['seed_case_id'],
                                  'prompt_sha256':candidate['sha256']}),flush=True)
                for i in sorted(p0): start_p1(i)
            elif kind == 'P1':
                p1[index] = value
                if eager_union:
                    prompt = {'text': candidate['text'], 'sha256': candidate['sha256'], 'id': candidate['id'],
                              'parent_sha256': parent['sha256']}
                    watch('union', index, submit('union', jobs[index], prompt))
            elif kind == 'union':
                unions[index] = value
            else: raise RuntimeError(f'Unknown pipeline completion {kind}')
    result = ([p0[i] for i in range(len(jobs))], [p1[i] for i in range(len(jobs))], candidate, seed)
    return (*result, [unions[i] for i in range(len(jobs))]) if eager_union else result
