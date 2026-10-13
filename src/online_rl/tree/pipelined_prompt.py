"""One immutable common P1 proposed from all five completed P0 records."""
import json
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
from src.online_rl.integrated.records import atomic_write_json, sha256_json

SEED_MODE = 'all_p0_before_p1_v1'


def evaluate(jobs, parent, submit, propose, batchdir, policy_sha256, eager_union=False):
    """submit(phase, job, prompt) returns a Future. Policy preparation must finish BEFORE this call.

    P1 generation waits for every P0 record. The ordered five-record evidence set
    and candidate are persisted before P1 work; resume validates the same evidence.
    Production waits for every P1 rubric/KEEP set before returning; final union scoring
    is dispatched by the caller only after this barrier.
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

    def propose_from_all_p0():
        rows = [p0[i] for i in range(len(jobs))]
        if seed is None:
            evidence = {'record_sha256': [sha256_json(row) for row in rows]}
            seed_value = {**identity, 'p0_evidence_sha256': sha256_json(evidence),
                          'p0_record_sha256': evidence['record_sha256']}
            atomic_write_json(seed_path, seed_value)
            return seed_value, proposal_pool.submit(propose, rows)
        expected = seed.get('p0_record_sha256')
        actual = [sha256_json(row) for row in rows]
        if expected != actual:
            raise RuntimeError('Saved P0 evidence set changed')
        return seed, proposal_pool.submit(propose, rows)

    with ThreadPoolExecutor(max_workers=1, thread_name_prefix='common-P1-proposal') as proposal_pool:
        for i, job in enumerate(jobs): watch('P0', i, submit('P0', job, parent))
        while len(unions if eager_union else p1) < len(jobs):
            kind, index, future = events.get()
            value = future.result()  # errors propagate; no changed policy or silent fallback
            if kind == 'P0':
                if value['case_id'] != jobs[index]['case']['case_id'] or value['policy_sha256'] != policy_sha256:
                    raise RuntimeError('P0 case or frozen policy mismatch')
                p0[index] = value
                if len(p0) == len(jobs):
                    if candidate is None and not proposing:
                        proposing = True
                        seed, proposal_future = propose_from_all_p0()
                        print(json.dumps({'stage':'prompt_seed_selected','p0_cases':len(p0),
                                          'p0_evidence_sha256':seed['p0_evidence_sha256'],
                                          'candidate_seed':SEED_MODE}),flush=True)
                        watch('proposal', -1, proposal_future)
                    elif candidate is not None:
                        expected = seed.get('p0_record_sha256') if seed else None
                        if expected != [sha256_json(p0[i]) for i in range(len(jobs))]:
                            raise RuntimeError('Saved P0 evidence set changed')
                        for ready_index in range(len(jobs)): start_p1(ready_index)
            elif kind == 'proposal':
                candidate = {**value, 'id': f'P1-{value["sha256"][:12]}', 'seed_sha256': sha256_json(seed)}
                atomic_write_json(proposal_path, candidate)
                print(json.dumps({'stage':'prompt_P1_fixed','p0_cases':len(p0),
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
