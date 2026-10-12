"""Concurrent validation of the independent CREATE groups of one chunk must give exactly the sequential result:
same assignments (and their order fields), same bank transitions, same decision journal, same report."""
import hashlib, json, random, time
from types import SimpleNamespace as NS
import pytest
from src.online_rl.tree import offline_chunk_bank as O


def run(monkeypatch, workers):
    monkeypatch.setenv('SHERPA_CHUNK_WORKERS', str(workers))
    ids = [f'c/{i:02d}' for i in range(20)]                      # 3 chunks: 8, 8, 4
    plan = {}
    for k, tid in enumerate(ids):                                 # per chunk: several CREATE groups with peers, plus bank reuse later
        j = k % 8
        if j in (0, 2, 5): plan[tid] = ('CREATE', None)
        elif j in (1, 3): plan[tid] = ('PEER', ids[k - 1])
        elif j == 6: plan[tid] = ('PEER', ids[k - 1])
        else: plan[tid] = ('REUSE_BANK', None)
    valid = lambda tid: int(tid[-2:]) % 5 != 0                    # some representatives are rejected
    keep = lambda tid: int(tid[-2:]) % 3 != 2                     # some valid rubrics get no witness
    calls = []

    def chunk_generator(context, matcher_text):
        def generate(package, tins, snapshot):
            rows = {}
            for t in tins:
                action, rep = plan[t.leaf_id]
                if action == 'REUSE_BANK' and not snapshot['entries']: action, rep = 'CREATE', None
                rows[t.leaf_id] = {'action': action, 'layer': t.outcome, 'generation': {'x': 1} if action == 'CREATE' else None,
                                   'normalized': {'name': t.leaf_id}, 'parse_error': None,
                                   'peer': {'peer_representative_id': rep} if action == 'PEER' else None,
                                   'bank_reuse': {k: None for k in O.CB.DECISION_FIELDS} | {'bank_id': snapshot['entries'][0]['bank_id'],
                                                  'selected_agent_turn_id': 1} if action == 'REUSE_BANK' else None}
            calls.append([t.leaf_id for t in tins])
            return {'decisions': rows, 'api_calls': 1, 'bank_sha256': snapshot['sha256']}
        generate.model = 'm'; generate.settings = {}
        return generate
    monkeypatch.setattr(O, 'api_chunk_generator', chunk_generator)

    def generate_rubric(create_text, prompt_sha, tin, preset, cache, gate_fn, source_stage1=None):
        time.sleep(random.random() * 0.03)                        # completion order differs from submission order
        return NS(rubric_id='r-' + tin.leaf_id, valid=valid(tin.leaf_id), problems=() if valid(tin.leaf_id) else ('gate',),
                  gate={'passed': valid(tin.leaf_id)}, rubric={'name': tin.leaf_id}, source_check={'status': 'PASS'})
    monkeypatch.setattr(O, 'generate_rubric', generate_rubric)
    monkeypatch.setattr(O.RB, 'prompt_sections', lambda text: (text, text))
    monkeypatch.setattr(O.RB, '_member_spec', lambda rub, tin, kind, decision: {'leaf': tin.leaf_id, 'kind': kind})
    monkeypatch.setattr(O.RB, '_validate_member_decision', lambda decision, bank, tin: decision)
    monkeypatch.setattr(O.RB, 'match_problem', lambda decision, bank: '' if decision['ok'] else 'decision CREATE_NEW')
    monkeypatch.setattr(O.RB, 'bank_view', lambda bank: [r.rubric_id for r in bank])
    monkeypatch.setattr(O.CB, 'validate_snapshot', lambda s: s)
    monkeypatch.setattr(O.CB, 'validate_reuse', lambda tin, snapshot, match: match)
    monkeypatch.setattr(O.CB, '_payload', lambda tin, entries: {'t': tin.leaf_id})
    monkeypatch.setattr(O.CB, 'content_hash', lambda rubric: rubric['name'])
    monkeypatch.setattr(O, 'to_dict', lambda x: x if isinstance(x, (dict, str)) else getattr(x, '__dict__', str(x)))
    monkeypatch.setattr(O, 'to_dialogue', lambda turns: [list(t) for t in turns])

    def extend(snapshot, records):
        entries = snapshot['entries'] + [{'sources': [], 'bank_id': 'bank_' + r['rubric']['name'] if isinstance(r, dict) and 'rubric' in r else 'bank_x'}
                                         for r in records[0]['trajectory_rubrics']]
        return {'entries': entries, 'sha256': hashlib.sha256(json.dumps([e['bank_id'] for e in entries]).encode()).hexdigest()}
    monkeypatch.setattr(O.CB, 'extend_snapshot', extend)

    class Cache:
        def call(self, kind, identity, fn): return fn()
    def batch(package, tins, anchors):
        time.sleep(random.random() * 0.02)
        return {'decisions': {t.leaf_id: {'ok': int(t.leaf_id[-2:]) % 7 != 3} for t in tins}}
    def verify(case, tree, feedback, members, canon, assignments):
        time.sleep(random.random() * 0.03); rid = next(iter(canon))
        return {rid: NS(decision='KEEP' if keep(rid) else 'NOT_VALIDATED', stop_reason='s', num_valid_supports=1, num_reruns_executed=2,
                        validated_by_trajectory_id=members[0].leaf_id)} if canon[rid].valid else {}
    trainer = NS(env=NS(generate_fn=NS(chunk_context={'model': 'm', 'settings': {}}), gate_fn=None, assign_fn=NS(batch=batch, model='m', settings={}),
                        scorer=NS(applicability=None)),
                 cache=Cache(), cumulative_bank={'matcher_prompt_file': 'prompts/sherpa/cumulative_bank_match_public_mc_v1.txt'}, source_consistency=False,
                 cfg={'judges': {}}, _verify_bounded=verify,
                 _witness_selection=lambda canon, wit: [NS(rubric_id=r, retained=w.decision == 'KEEP') for r, w in wit.items()])
    tin_of = {i: NS(leaf_id=i, sha256='s' + i, outcome='success' if int(i[-2:]) % 2 else 'failure', feedback='f', dialogue=[('doctor', 'D1', 1, 'd'), ('agent', 'A1', 1, 'a')])
              for i in ids}
    ruled = [NS(leaf_id=i, path=(i,)) for i in ids]
    out = O.process_case(trainer, case={'case_id': 'c'}, state=NS(run_seed=0, case_cursor=0, prompt_sha256='p', policy_sha256='pol'),
                         tree=NS(nodes=[]), feedback=[], ruled=ruled, tin_of=tin_of, prompt_text='prompt',
                         snapshot={'entries': [], 'sha256': 'empty'})
    for chunk in out['summary']['chunks']: chunk.pop('timing_seconds', None)     # wall-clock metadata only; never part of a decision
    key = lambda a: (a.leaf_id, a.stratum, a.order, a.canonical_rubric_id, a.assignment)
    return {'decisions': out['decisions'], 'summary': out['summary'], 'assignments': [key(a) for a in out['assignments']],
            'working': out['working'], 'reuse': [m['leaf_id'] for m in out['cumulative_reuse']], 'calls': calls}


def test_concurrent_groups_equal_the_sequential_result(monkeypatch):
    random.seed(1); sequential = run(monkeypatch, 1)
    for trial in range(5):
        random.seed(100 + trial); concurrent = run(monkeypatch, 4)
        assert json.dumps(concurrent, sort_keys=True, default=str) == json.dumps(sequential, sort_keys=True, default=str)
    chunks = sequential['summary']['chunks']
    assert [c['trajectories'] for c in chunks] == [8, 8, 4] and sequential['summary']['generator_calls'] == 3
    assert all(a['bank_after_sha256'] == b['bank_before_sha256'] for a, b in zip(chunks, chunks[1:]))   # the next chunk saw the updated bank
    assert sequential['working']['entries'] and any(d['action'] == 'GENERATED_REMOVE' for d in sequential['decisions'])
