"""Offline rubric-bank construction: ONE Rubric-Generator call per chunk of at most eight trajectories.

For one case, the eligible trajectories (valid Doctor reflection, decided
outcome) are capped at 48 by a reproducible sample and processed in leaf-id order in chunks of at most eight. Each chunk makes
one call with the CURRENT persistent KEEP bank; its rows are applied immediately, so the next chunk sees the updated bank.

Nothing downstream is redefined. A CREATE row carries the unchanged single-trajectory generation object and goes through the
existing normalisation, source consistency, Gate and bounded witness (random-5, 2-of-3) code; a REUSE_BANK row is the unchanged
cumulative-bank reuse decision; a PEER row is, as in the existing batch contract, only a generation-saving proposal: once the
representative's rubric exists and passed its Gate, the proposed peers are re-checked against that actual rubric with the
unchanged batch matcher (rubric_bank / api_rubric_matcher.batch) and attached only on its REUSE_EXISTING decision. A trajectory
whose representative is rejected or whose re-check is not a reuse gets no rubric in this protocol (no further primary call).
A peer re-check that stays execution-invalid after the local retry budget confirms no peer (EXECUTION_UNRESOLVED); other judge
errors propagate as in the per-trajectory path. Layers never mix."""
from __future__ import annotations
import copy, hashlib, json, random, re, time
from dataclasses import replace
from pathlib import Path
from src.online_rl.integrated.records import sha256_json, sha256_text, to_dict
from src.online_rl.tree import cumulative_bank as CB
from src.online_rl.tree import rubric_bank as RB
from src.online_rl.tree.agent_state import visible_rubric
from src.online_rl.tree.posthoc import TRANSCRIPT_FORMAT, generate_rubric, trajectory_message
from src.online_rl.tree.records import RubricAssignment, to_dialogue

VERSION = 'offline_chunk8_single_call_bank_v1'
CHUNK, CAP = 8, 48
ROOT = Path(__file__).resolve().parents[3]
PROMPT = ROOT / 'prompts/sherpa/offline_chunk8_bank_generation_v1.txt'
ACTIONS = ('REUSE_BANK', 'CREATE', 'PEER')
ROW_KEYS = {'trajectory_id', 'action', 'bank_reuse', 'generation', 'peer'}
PEER_KEYS = {'peer_representative_id', 'checks', 'rationale'}


def select(leaf_ids, run_seed, case_id):
    """At most CAP trajectories for rubric discovery: all of them, or a reproducible uniform sample kept in leaf-id order."""
    ordered = sorted(leaf_ids)
    seed = int(hashlib.sha256(f'{VERSION}|{run_seed}|{case_id}'.encode()).hexdigest()[:16], 16)
    chosen = ordered if len(ordered) <= CAP else sorted(random.Random(seed).sample(ordered, CAP))
    taken = set(chosen)
    return chosen, {'eligible': len(ordered), 'cap': CAP, 'sampled': len(ordered) > CAP, 'sampling_seed': seed,
                    'selected_ids': chosen, 'not_selected_ids': [x for x in ordered if x not in taken]}


def chunks(selected):
    return [selected[i:i + CHUNK] for i in range(0, len(selected), CHUNK)]


def entry_layer(entry):
    layer = ((entry['rubric'].get('rubric') or {}).get('source_provenance') or {}).get('source_outcome')
    if layer not in ('success', 'failure'): raise ValueError('Bank entry lacks its success/failure source layer')
    return layer


def bank_view(snapshot):
    return [{'bank_id': e['bank_id'], 'layer': entry_layer(e),
             'rubric': {k: v for k, v in visible_rubric(e['rubric']['rubric']).items() if k != 'rubric_id'}}
            for e in snapshot['entries']]


def output_schema(generation_schema, bank_checks):
    from src.online_rl.tree.backends import MATCH_CHECKS
    checks = lambda names: {'type': 'object', 'additionalProperties': False, 'required': list(names),
                            'properties': {c: {'type': 'boolean'} for c in names}}
    reuse = {'type': ['object', 'null'], 'additionalProperties': False, 'required': sorted(CB.DECISION_FIELDS),
             'properties': {'reuse': {'type': 'boolean'}, 'bank_id': {'type': ['string', 'null']},
                            'selected_agent_turn_id': {'type': ['integer', 'null'], 'minimum': 1},
                            'doctor_quote': {'type': ['string', 'null']}, 'agent_quote': {'type': ['string', 'null']},
                            'rationale': {'type': 'string'}, 'checks': checks(bank_checks)}}
    peer = {'type': ['object', 'null'], 'additionalProperties': False, 'required': sorted(PEER_KEYS),
            'properties': {'peer_representative_id': {'type': 'string'}, 'checks': checks(MATCH_CHECKS), 'rationale': {'type': 'string'}}}
    row = {'type': 'object', 'additionalProperties': False, 'required': sorted(ROW_KEYS),
           'properties': {'trajectory_id': {'type': 'string'}, 'action': {'type': 'string', 'enum': list(ACTIONS)},
                          'bank_reuse': reuse, 'generation': {'anyOf': [generation_schema, {'type': 'null'}]}, 'peer': peer}}
    return {'type': 'object', 'additionalProperties': False, 'required': ['decisions'],
            'properties': {'decisions': {'type': 'array', 'items': row}}}


def api_chunk_generator(context, matcher_prompt_text):
    """context = api_rubric_generator(...).chunk_context: the unchanged generator key / model / settings / schema / normalise."""
    from src.online_rl.tree import backends as B
    key, model, base = context['key'], context['model'], context['settings']
    contract = PROMPT.read_text()
    settings = {**base, 'response_format': None, 'max_tokens': int(base['max_tokens']) * CHUNK, 'chunk_protocol': VERSION,
                'chunk_prompt_sha256': sha256_text(contract), 'matcher_prompt_sha256': sha256_text(matcher_prompt_text)}
    generator_v1 = base.get('selected_turn_policy') == 'generator_v1'

    def validate(raw, tins, snapshot, payloads, lenient):
        try: parsed = json.loads(re.sub(r'^```(?:json)?\s*|\s*```$', '', (raw or '').strip()))
        except (json.JSONDecodeError, AttributeError) as exc: raise B.RubricOutputIncomplete(f'chunk output is not JSON: {exc}') from exc
        rows = parsed.get('decisions') if isinstance(parsed, dict) and set(parsed) == {'decisions'} else None
        by_tin = {t.leaf_id: t for t in tins}; position = {t.leaf_id: i for i, t in enumerate(tins)}
        if not isinstance(rows, list) or len(rows) != len(tins): raise B.RubricOutputIncomplete('chunk trajectory coverage differs')
        by_bank = {e['bank_id']: e for e in snapshot['entries']}; out = {}; hard = []; soft = []
        for row in rows:
            if not isinstance(row, dict) or set(row) != ROW_KEYS: raise B.RubricOutputIncomplete('malformed chunk row')
            tid = row['trajectory_id']
            if not isinstance(tid, str) or tid not in by_tin or tid in out: raise B.RubricOutputIncomplete('missing, duplicate or unknown trajectory_id')
            tin, action = by_tin[tid], row['action']
            filled = [k for k in ('bank_reuse', 'generation', 'peer') if row[k] is not None]
            expected = {'REUSE_BANK': 'bank_reuse', 'CREATE': 'generation', 'PEER': 'peer'}.get(action)
            if expected is None or filled != [expected]:
                hard.append(f'{tid}: action {action!r} needs exactly its own object'); out[tid] = None; continue
            value = {'trajectory_id': tid, 'action': action, 'layer': tin.outcome, 'bank_reuse': None, 'generation': None, 'peer': None,
                     'normalized': None, 'parse_error': None}
            if action == 'REUSE_BANK':
                decision = row['bank_reuse']
                try:
                    if not isinstance(decision, dict) or decision.get('reuse') is not True: raise ValueError('bank_reuse needs reuse = true')
                    CB._validate_decision({k: decision.get(k) for k in CB.DECISION_FIELDS} if set(decision) == CB.DECISION_FIELDS else decision, payloads[tid])
                    if entry_layer(by_bank[decision['bank_id']]) != tin.outcome: raise ValueError('bank entry of the other success/failure layer')
                except ValueError as exc: hard.append(f'{tid}: {exc}')
                value['bank_reuse'] = decision
            elif action == 'CREATE':
                generation = row['generation']
                if not isinstance(generation, dict): hard.append(f'{tid}: generation must be an object')
                else:
                    rubric, err = context['normalize'](json.dumps(generation, ensure_ascii=False), copy.deepcopy(generation), None, tin)
                    value.update(generation=generation, normalized=rubric, parse_error=err)
                    if err: soft.append(f'{tid}: {err}')
            else:
                peer = row['peer']
                if (not isinstance(peer, dict) or set(peer) != PEER_KEYS or not isinstance(peer['checks'], dict) or
                        set(peer['checks']) != set(B.MATCH_CHECKS) or not all(v is True for v in peer['checks'].values()) or
                        not isinstance(peer['rationale'], str)):
                    hard.append(f'{tid}: malformed peer object (a representative, all six peer checks true and a rationale are required)')
                value['peer'] = peer
            out[tid] = value
        for tid, value in out.items():
            if value is None or value['action'] != 'PEER' or not isinstance(value['peer'], dict): continue
            rep = value['peer'].get('peer_representative_id'); leader = out.get(rep)
            if (leader is None or leader['action'] != 'CREATE' or position[rep] >= position[tid] or leader['layer'] != value['layer']):
                hard.append(f'{tid}: peer_representative_id must be an earlier CREATE trajectory of the same layer')
        if hard: raise B.RubricOutputIncomplete('; '.join(hard))
        if soft and not lenient: raise B.RubricOutputIncomplete('generation objects failed deterministic validation: ' + '; '.join(soft))
        return out

    def generate(package, tins, snapshot):
        if not 1 <= len(tins) <= CHUNK or len({t.leaf_id for t in tins}) != len(tins): raise ValueError('A chunk holds 1..8 distinct trajectories')
        CB.validate_snapshot(snapshot)
        payloads = {t.leaf_id: CB._payload(t, snapshot['entries']) for t in tins}
        bank_checks = CB._decision_checks(payloads[tins[0].leaf_id]) if snapshot['entries'] else CB.CHECKS
        schema = output_schema(context['schema'], bank_checks)
        system = (context['system'](package) + '\n\n' + contract + '\n\nBANK REUSE CONTRACT (unchanged cumulative-bank matcher text; it '
                  'defines the content of bank_reuse; its candidates are the supplied current_bank):\n' + matcher_prompt_text +
                  '\n\nOUTPUT JSON SCHEMA OF THIS CALL:\n' + json.dumps(schema, ensure_ascii=False))
        user = json.dumps({'current_bank': bank_view(snapshot), 'trajectories': [
            {'trajectory_id': t.leaf_id, 'layer': t.outcome,
             'input': trajectory_message(replace(t, target_reply_id=None) if generator_v1 else t)} for t in tins]}, ensure_ascii=False)
        attempts = []; request = user
        for attempt in range(2):
            raw, meta = B._generator_json(key, model, system, request, settings, schema, 'offline_chunk_bank', 'rubric_generator')
            try: value = validate(raw, tins, snapshot, payloads, lenient=bool(attempt))
            except B.RubricOutputIncomplete as exc:
                attempts.append({'attempt': attempt + 1, 'raw': raw, 'meta': meta, 'error': str(exc)})
                if attempt:
                    failure = B.RubricOutputIncomplete(f'{model}: chunk bank output invalid after one correction: {exc}')
                    failure.audit = {'contract': VERSION, 'settings': settings, 'attempts': attempts}
                    raise failure from exc
                request = (user + '\n\nCHUNK JSON CORRECTION: The previous response failed a deterministic validation. Re-evaluate only the '
                           'SAME bank and trajectories. Correct only what is invalid, keep grounded content, and return the complete JSON '
                           'object with one row per supplied trajectory_id. Do not turn a format error into a different decision or invent '
                           'evidence.\n' + json.dumps({'validation_error': str(exc), 'invalid_previous_output': raw}, ensure_ascii=False))
                continue
            attempts.append({'attempt': attempt + 1, 'raw': raw, 'meta': meta, 'error': None})
            return {'decisions': value, 'attempts': attempts, 'api_calls': len(attempts), 'system_sha256': sha256_text(system),
                    'user_sha256': sha256_text(user), 'bank_sha256': snapshot['sha256'], 'usage': [a['meta'].get('usage') for a in attempts]}
    generate.model = model; generate.settings = settings
    return generate


def _preset(context, chunk_sha, row):
    """The chunk row as the result of the normal per-trajectory generation call, so generate_rubric runs unchanged."""
    joint = context['settings'].get('output_contract') == 'trajectory_analysis_candidate_v2_with_intervention'
    def generate(prompt_text, tin):
        raw = json.dumps(row['generation'], ensure_ascii=False)
        return {'raw': raw, 'raw_attempts': [raw], 'rubric': row['normalized'],
                'intervention_instruction': row['generation'].get('intervention_instruction') if joint and not row['parse_error'] else None,
                'parse_error': row['parse_error'], 'usage': None, 'usage_attempts': [], 'chunk_generation_sha256': chunk_sha}
    generate.model = context['model']
    generate.settings = {**context['settings'], 'chunk_protocol': VERSION, 'chunk_generation_sha256': chunk_sha}
    return generate


def _ordered_map(fn, items, workers):
    """Results in the order of `items`, whatever the completion order; workers <= 1 is the plain sequential loop."""
    if workers <= 1 or len(items) <= 1: return [fn(item) for item in items]
    from concurrent.futures import ThreadPoolExecutor
    from contextvars import copy_context
    with ThreadPoolExecutor(min(workers, len(items))) as pool:
        futures = [pool.submit(copy_context().run, fn, item) for item in items]
        return [future.result() for future in futures]


class Speculation:
    """Candidate-local, in-memory results of speculative requests (execution only).

    `speculative` is the cache handed to speculative requests: it may read results that are already persistent, but a newly
    computed result goes only into this object. `logical` is the cache handed to the unchanged sequential logic: when that
    logic itself requests a judgment, a speculative result with exactly that identity is committed to the persistent cache
    (as if computed at that moment) and returned; everything else falls through to the persistent cache. A speculative result
    the logic never requests is dropped with this object: it is never written and can never become a later cache hit."""
    def __init__(self, persistent):
        import threading
        self.persistent, self.pending, self.lock, self.used = persistent, {}, threading.Lock(), set()
        outer = self

        class Speculative:
            path = None
            def call(self, kind, identity, fn):
                key = sha256_json([kind, identity])
                stored = getattr(outer.persistent, 'store', {}).get(key)
                if stored is not None: return stored['value']
                with outer.lock:
                    if key in outer.pending: return outer.pending[key]
                value = fn(); json.dumps(value)
                with outer.lock: return outer.pending.setdefault(key, value)

        class Logical:
            path = getattr(persistent, 'path', None)
            def call(self, kind, identity, fn):
                key = sha256_json([kind, identity])
                with outer.lock: ready = key in outer.pending; value = outer.pending.get(key)
                if ready:
                    outer.used.add(key)
                    return outer.persistent.call(kind, identity, lambda: value)
                return outer.persistent.call(kind, identity, fn)
        self.speculative, self.logical = Speculative(), Logical()

    @property
    def discarded(self): return len(set(self.pending) - self.used)


def speculate(tasks, workers):
    """Start independent, side-effect-free judge requests of one fixed candidate together (execution only). Each task only
    fills the candidate-local Speculation; nothing is decided or persisted here. The unchanged logic afterwards asks for its
    judgments in its original order and never sees a result it would not have requested. A failing task is ignored: the
    original call repeats it and reports the error exactly as before."""
    def quiet(task):
        try: task()
        except Exception: pass
    _ordered_map(quiet, [t for t in tasks if t is not None], workers)


PEER_EXECUTION_UNRESOLVED = 'EXECUTION_UNRESOLVED: peer re-check invalid after every local retry'


def recheck_peers(request, bound):
    """The unchanged peer re-check request, repeated locally while its output is execution-invalid. Returns
    (decisions, calls, reasons); decisions is None when the budget is exhausted (no peer is confirmed, none is rejected on
    its merits). Without a configured budget the original behaviour is kept: the error propagates."""
    from src.online_rl.tree.backends import RubricOutputIncomplete
    bound = int(bound or 0); reasons = []
    for attempt in range(max(bound, 1)):
        try: return request(), attempt + 1, reasons
        except RubricOutputIncomplete as exc:
            if not bound: raise
            reasons.append(str(exc)[:200])
    return None, bound, reasons


def process_case(trainer, *, case, state, tree, feedback, ruled, tin_of, prompt_text, snapshot):
    """Sequential chunk transactions of one case against the latest working bank; returns the pieces of the online-bank record."""
    from src.online_rl.tree import witness as W
    env, cache, cid = trainer.env, trainer.cache, case['case_id']
    context = getattr(env.generate_fn, 'chunk_context', None)
    if context is None: raise RuntimeError('The rubric generator does not expose its chunk context')
    create_text, package = RB.prompt_sections(prompt_text)
    chunk_generate = api_chunk_generator(context, (ROOT / trainer.cumulative_bank['matcher_prompt_file']).read_text())
    nodes = {n.node_id: n for n in tree.nodes}; leaf_of = {l.leaf_id: l for l in ruled}
    stage1 = (lambda leaf: W.source_stage1_for(case, nodes, leaf, tin_of[leaf.leaf_id].target_reply_id, env.scorer.applicability, cache)
              ) if trainer.source_consistency else None
    selected, sampling = select(list(leaf_of), state.run_seed, cid)
    import os
    workers = int(os.environ.get('SHERPA_CHUNK_WORKERS', '1'))            # execution only: concurrent independent CREATE groups
    speculative = os.environ.get('SHERPA_SPECULATIVE_VALIDATION', '0') == '1'   # execution only: source check, Gate and peer re-check together
    working = CB.validate_snapshot(snapshot)
    canon_all, assignments_all, witness_all, witness_objects, selection_all = {}, [], [], {}, []
    reuse_rows, decisions, chunk_reports = [], [], []
    uncovered = list(sampling['not_selected_ids']); orders = {'success': 0, 'failure': 0}
    for index, ids in enumerate(chunks(selected)):
        tins = [tin_of[i] for i in ids]; before = working
        identity = {'protocol': VERSION, 'case': cid, 'chunk_index': index, 'prompt': state.prompt_sha256,
                    'tins': [[t.leaf_id, t.sha256] for t in tins], 'bank_sha256': before['sha256'], 'transcript': TRANSCRIPT_FORMAT,
                    'model': chunk_generate.model, 'settings': chunk_generate.settings}
        def primary_call():                                      # ONE primary generator call; an execution-invalid response
            from src.online_rl.tree.backends import RubricOutputIncomplete   # (not JSON / contract violation after its correction)
            bound = int((trainer.cfg.get('judges') or {}).get('execution_retry_attempts') or 1)   # repeats the SAME request, bounded
            for attempt in range(bound):
                try:
                    out = chunk_generate(package, tins, before); out['execution_retries'] = attempt
                    return out
                except RubricOutputIncomplete as exc:
                    if attempt == bound - 1: raise
                    print(json.dumps({'event': 'generator_execution_retry', 'case_id': cid, 'chunk_index': index, 'attempt': attempt + 1,
                                      'of': bound, 'error': str(exc)[:160]}), flush=True)
        chunk_started = time.monotonic()
        result = cache.call('offline_chunk_generation', identity, primary_call)
        if result.get('bank_sha256') != before['sha256']: raise RuntimeError('Chunk result belongs to another bank')
        rows = result['decisions']; chunk_sha = sha256_json(identity); groups = {}; report = []; timing = {}
        primary_seconds = round(time.monotonic() - chunk_started, 1); phase_started = time.monotonic()
        # Every CREATE group of this chunk was decided against the same frozen pre-chunk bank, so its validation calls are
        # independent of the other groups: they run concurrently (bounded; execution only) and are then applied below in the
        # original trajectory order, which alone determines assignment order, the bank transitions and the decision journal.
        creators = [tid for tid in ids if rows[tid]['action'] == 'CREATE']

        def admit(tid):                                          # generation -> source consistency -> Gate -> peer re-check
            row, tin, leaf, layer = rows[tid], tin_of[tid], leaf_of[tid], tin_of[tid].outcome
            started = time.monotonic()
            proposed = [pid for pid in ids if rows[pid]['action'] == 'PEER' and rows[pid]['peer']['peer_representative_id'] == tid]
            peer_tins = [tin_of[pid] for pid in proposed]

            spec = Speculation(cache); logical = spec.logical if speculative else cache
            stage1_on = lambda store: (W.source_stage1_for(case, nodes, leaf, tin.target_reply_id, env.scorer.applicability, store)
                                       if trainer.source_consistency else None)

            def match_request(anchors, store):                   # the unchanged batch matcher against the ACTUAL generated rubric
                match_identity = {'version': RB.BATCH_VERSION, 'case': cid, 'stratum': layer, 'prompt': state.prompt_sha256,
                                  'assignment_cache_version': RB.ASSIGNMENT_CACHE_VERSION, 'tins': [[t.leaf_id, t.sha256] for t in peer_tins],
                                  'bank': sha256_json(RB.bank_view(anchors)), 'transcript': TRANSCRIPT_FORMAT,
                                  'model': getattr(env.assign_fn, 'model', None), 'settings': getattr(env.assign_fn, 'settings', None),
                                  'chunk_protocol': VERSION}
                def checked(value):
                    if not isinstance(value, dict) or not isinstance(value.get('decisions'), dict) or set(value['decisions']) != set(proposed):
                        raise RuntimeError('Batch match returned incomplete trajectory IDs')
                    for pid in proposed: RB._validate_member_decision(value['decisions'][pid], anchors, tin_of[pid])
                    return value
                return checked(store.call('rubric_match_batch', match_identity,
                                          lambda: checked(env.assign_fn.batch(package, peer_tins, anchors))))['decisions']

            speculated = 0
            if speculative and row['normalized'] is not None and not row['parse_error']:
                # The candidate is fixed here. Its source check, Gate and peer re-check read only (trajectory, rubric): none of
                # them consumes another's output, so their requests are sent together with the identities the unchanged code
                # below uses (rubric id / scores normalisation as posthoc.generate_rubric; no repair exists for generator_v1).
                from src.online_rl.tree.posthoc import rubric_problems
                from src.online_rl.tree.records import TrajectoryRubric
                rid = f'{state.prompt_sha256[:12]}-{tin.sha256[:12]}'
                rubric = {**row['normalized'], 'rubric_id': rid, 'scores': {str(a): b for a, b in (row['normalized'].get('scores') or {}).items()}}
                generator_v1 = (rubric.get('source_provenance') or {}).get('selected_turn_policy') == 'generator_v1'
                if not rubric_problems(rubric) and (generator_v1 or stage1 is None):
                    gate_identity = {'tin': tin.sha256, 'rubric': sha256_json(rubric), 'model': getattr(env.gate_fn, 'model', None),
                                     'protocol': getattr(env.gate_fn, 'protocol', None)}
                    if getattr(env.gate_fn, 'settings', None) is not None: gate_identity['settings'] = env.gate_fn.settings
                    provisional = TrajectoryRubric(rid, tin.leaf_id, state.prompt_sha256, json.dumps(row['generation'], ensure_ascii=False), rubric,
                                                   True, (), None, None, _preset(context, chunk_sha, row)(None, tin)['intervention_instruction'])
                    quiet = spec.speculative
                    tasks = [(lambda: stage1_on(quiet)(rubric)) if stage1 else None,
                             (lambda: quiet.call('posthoc_gate', gate_identity, lambda: env.gate_fn(tin, rubric))) if env.gate_fn else None,
                             (lambda: match_request([provisional], quiet)) if proposed else None]
                    speculate(tasks, 3); speculated = sum(t is not None for t in tasks)
            rub = generate_rubric(create_text, state.prompt_sha256, tin, _preset(context, chunk_sha, row), logical, env.gate_fn,
                                  source_stage1=stage1_on(logical))
            confirmed = {}; matching_calls = 0
            if rub.valid and proposed:                           # as rubric_bank.batched_stratum: followers are checked against the ACTUAL rubric
                confirmed, matching_calls, reasons = recheck_peers(lambda: match_request([rub], logical),
                                                                   (trainer.cfg.get('judges') or {}).get('execution_retry_attempts'))
                if confirmed is None:                            # EXECUTION_UNRESOLVED: no re-check verdict, so no peer is confirmed
                    from src.online_rl.tree.applicability import log_execution_unresolved
                    log_execution_unresolved(cache, stage='peer_reconfirmation', case_id=cid, node_id=tid, rubric_id=rub.rubric_id,
                                             attempts=matching_calls, reasons=reasons)
            timing[tid] = {'admit_s': round(time.monotonic() - started, 1), 'speculative_requests': speculated,
                           'speculative_results_discarded': spec.discarded}
            return rub, proposed, confirmed, matching_calls
        admitted = dict(zip(creators, _ordered_map(admit, creators, workers)))
        admit_seconds = round(time.monotonic() - phase_started, 1); phase_started = time.monotonic()

        for tid in creators:                                     # assignments in the original order (deterministic `order` fields)
            rub, proposed, confirmed, matching_calls = admitted[tid]; tin, layer = tin_of[tid], tin_of[tid].outcome
            kind = 'CREATED_NEW' if rub.valid else 'CREATE_REJECTED'
            note = '' if rub.valid else 'generated rubric rejected: ' + ', '.join(rub.problems or RB._gate_reasons(rub.gate))
            canon = {rub.rubric_id: rub}; members = [leaf_of[tid]]; accepted = {}; refused = {}
            assignments = [RubricAssignment(tid, cid, layer, orders[layer], rub.rubric_id, kind, None, note, sha256_json(tin.feedback),
                                            RB._member_spec(rub, tin, kind, None))]
            orders[layer] += 1
            for pid in proposed:
                if not rub.valid: refused[pid] = 'representative rubric rejected'; continue
                if confirmed is None: refused[pid] = PEER_EXECUTION_UNRESOLVED; continue
                decision = confirmed[pid]; problem = RB.match_problem(decision, [rub])
                if problem: refused[pid] = problem; continue
                assignments.append(RubricAssignment(pid, cid, layer, orders[layer], rub.rubric_id, 'REUSED_EXISTING', decision, '',
                    sha256_json(tin_of[pid].feedback), RB._member_spec(rub, tin_of[pid], 'REUSED_EXISTING', decision)))
                orders[layer] += 1; members.append(leaf_of[pid]); accepted[pid] = True
            groups[tid] = {'rub': rub, 'canon': canon, 'members': members, 'assignments': assignments, 'accepted': accepted, 'refused': refused,
                           'proposed': proposed, 'matching_calls': matching_calls}
        witnessed = dict(zip(creators, _ordered_map(
            lambda tid: trainer._verify_bounded(case, tree, feedback, groups[tid]['members'], groups[tid]['canon'], groups[tid]['assignments']),
            creators, workers)))
        witness_seconds = round(time.monotonic() - phase_started, 1)

        for tid in ids:                                          # apply in the original trajectory order
            row, tin, leaf, layer, input_sha = rows[tid], tin_of[tid], leaf_of[tid], tin_of[tid].outcome, working['sha256']
            base = {'case_cursor': state.case_cursor, 'case_id': cid, 'leaf_id': tid}
            if row['action'] == 'REUSE_BANK':
                decision = {k: row['bank_reuse'].get(k) for k in CB.DECISION_FIELDS}
                entry = next(e for e in before['entries'] if e['bank_id'] == decision['bank_id'])
                positions = [i for i, turn in enumerate(tin.dialogue) if turn[0] == 'agent']
                match = CB.validate_reuse(tin, before, {**copy.deepcopy(decision), 'leaf_id': tid, 'sourceproof': copy.deepcopy(entry['sources']),
                    'snapshot_sha256': before['sha256'], 'retrieval_ids': [e['bank_id'] for e in before['entries']],
                    'trajectory_sha256': sha256_json(CB._payload(tin, [])), 'chunk_protocol': VERSION, 'chunk_index': index,
                    'prefix_sha256': sha256_json(to_dialogue(tin.dialogue[:positions[decision['selected_agent_turn_id'] - 1]]))})
                reuse_rows.append(match)
                decisions.append({**base, 'action': 'REUSED_EXISTING', 'input_bank_sha256': input_sha, 'output_bank_sha256': input_sha,
                                  'bank_id': decision['bank_id'], 'rubric_id': None})
                report.append({'trajectory_id': tid, 'layer': layer, 'decision': 'REUSE', 'bank_id': decision['bank_id']})
            elif row['action'] == 'CREATE':
                group = groups[tid]; rub, canon, wit = group['rub'], group['canon'], witnessed[tid]
                chosen = trainer._witness_selection(canon, wit)
                kept = any(d.retained for d in chosen)
                if kept:
                    working = CB.extend_snapshot(working, [{'case_id': cid, 'case_cursor': state.case_cursor, 'policy_sha256': state.policy_sha256,
                        'prompt_sha256': state.prompt_sha256, 'trajectory_rubrics': [to_dict(r) for r in canon.values()],
                        'selection': [to_dict(d) for d in chosen], 'rubric_bank': {'witness': [to_dict(w) for w in wit.values()]}}])
                bank_id = 'bank_' + CB.content_hash(rub.rubric) if kept else None
                group.update(rubric_id=rub.rubric_id, kept=kept, bank_id=bank_id)
                canon_all.update(canon); assignments_all.extend(group['assignments']); witness_all.extend(to_dict(w) for w in wit.values())
                witness_objects.update(wit); selection_all.extend(chosen)
                decisions.append({**base, 'action': 'GENERATED_KEEP' if kept else 'GENERATED_REMOVE', 'input_bank_sha256': input_sha,
                                  'output_bank_sha256': working['sha256'], 'bank_id': bank_id, 'rubric_id': rub.rubric_id})
                w = wit.get(rub.rubric_id)
                report.append({'trajectory_id': tid, 'layer': layer, 'decision': 'CREATE', 'rubric_id': rub.rubric_id, 'name': (rub.rubric or {}).get('name'),
                               'structure_valid': not rub.problems, 'problems': list(rub.problems or ()),
                               'gate_passed': (rub.gate or {}).get('passed') if rub.gate is not None else None, 'peers': sorted(group['accepted']),
                               'witness_decision': getattr(w, 'decision', None), 'witness_stop_reason': getattr(w, 'stop_reason', None),
                               'witness_valid_supports': getattr(w, 'num_valid_supports', None), 'witness_reruns': getattr(w, 'num_reruns_executed', None),
                               'validated_by': getattr(w, 'validated_by_trajectory_id', None), 'keep': kept, 'bank_id': bank_id,
                               'source_check': (rub.source_check or {}).get('status'), 'proposed_peers': group['proposed'],
                               'peer_matching_calls': group['matching_calls'], 'refused_peers': group['refused']})
            else:
                rep = row['peer']['peer_representative_id']; group = groups[rep]; member = tid in group['accepted']
                if not member: uncovered.append(tid)
                reused = member and group['kept']
                decisions.append({**base, 'action': 'REUSED_EXISTING' if reused else 'GENERATED_REMOVE', 'input_bank_sha256': input_sha,
                                  'output_bank_sha256': input_sha, 'bank_id': group['bank_id'] if reused else None, 'rubric_id': group['rubric_id']})
                report.append({'trajectory_id': tid, 'layer': layer, 'decision': 'PEER', 'peer_of': rep, 'member': member,
                               'refused': group['refused'].get(tid), 'representative_keep': group['kept'], 'bank_id': group['bank_id'] if reused else None})
        chunk_reports.append({'chunk_index': index, 'trajectory_ids': ids, 'trajectories': len(ids), 'generator_calls': 1,
                              'generator_api_requests': result['api_calls'], 'generator_correction_requests': result['api_calls'] - 1,
                              'generator_execution_retries': result.get('execution_retries', 0),
                              'timing_seconds': {'primary_generation': primary_seconds, 'admission_wall': admit_seconds, 'witness_wall': witness_seconds,
                                                 'chunk_wall': round(time.monotonic() - chunk_started, 1), 'groups': timing},
                              'peer_matching_calls': sum(r.get('peer_matching_calls', 0) for r in report), 'bank_before_sha256': before['sha256'],
                              'bank_before_size': len(before['entries']), 'bank_after_sha256': working['sha256'],
                              'bank_after_size': len(working['entries']), 'rows': report})
    for earlier, later in zip(chunk_reports, chunk_reports[1:]):
        if earlier['bank_after_sha256'] != later['bank_before_sha256']: raise RuntimeError('A chunk did not start from the previous chunk bank')
    if len(chunk_reports) > CAP // CHUNK: raise RuntimeError('More than six generator calls for one case')
    summary = {'protocol': VERSION, **sampling, 'uncovered_ids': sorted(uncovered), 'chunks': chunk_reports,
               'generator_calls': len(chunk_reports), 'generator_api_requests': sum(c['generator_api_requests'] for c in chunk_reports),
               'generator_correction_requests': sum(c['generator_correction_requests'] for c in chunk_reports),
               'peer_matching_calls': sum(c['peer_matching_calls'] for c in chunk_reports),
               'input_bank_sha256': snapshot['sha256'], 'output_bank_sha256': working['sha256']}
    return {'working': working, 'canon': canon_all, 'assignments': assignments_all, 'witness': witness_all, 'witness_objects': witness_objects,
            'selection': selection_all, 'cumulative_reuse': reuse_rows, 'decisions': decisions, 'summary': summary}
