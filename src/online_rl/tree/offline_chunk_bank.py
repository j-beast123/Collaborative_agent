"""Offline rubric-bank construction: ONE Rubric-Generator call per chunk of at most eight trajectories.

Collaborator-authorized protocol (2026-10-11). For one case, the eligible trajectories (valid Doctor reflection, decided
outcome) are capped at 48 by a reproducible sample and processed in leaf-id order in chunks of at most eight. Each chunk makes
one call with the CURRENT persistent KEEP bank; its rows are applied immediately, so the next chunk sees the updated bank.

Nothing downstream is redefined. A CREATE row carries the unchanged single-trajectory generation object and goes through the
existing normalisation, source consistency, Gate and bounded witness (random-5, 2-of-3) code; a REUSE_BANK row is the unchanged
cumulative-bank reuse decision; a PEER row is the unchanged member decision (own turn + instruction) for the rubric created in
the same call. Compatibility adaptation: a peer is attached on the peer checks of that same call instead of a second matching
call against the generated rubric (a second call would break the one-call-per-chunk contract). Layers never mix."""
from __future__ import annotations
import copy, hashlib, json, random, re
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
PEER_KEYS = {'peer_representative_id', 'checks', 'rationale', 'selected_agent_turn_id', 'intervention_instruction'}


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
            'properties': {'peer_representative_id': {'type': 'string'}, 'checks': checks(MATCH_CHECKS), 'rationale': {'type': 'string'},
                           'selected_agent_turn_id': {'type': 'integer', 'minimum': 1},
                           'intervention_instruction': {'type': 'string', 'minLength': 1, 'pattern': r'\S'}}}
    row = {'type': 'object', 'additionalProperties': False, 'required': sorted(ROW_KEYS),
           'properties': {'trajectory_id': {'type': 'string'}, 'action': {'type': 'string', 'enum': list(ACTIONS)},
                          'bank_reuse': reuse, 'generation': {'anyOf': [generation_schema, {'type': 'null'}]}, 'peer': peer}}
    return {'type': 'object', 'additionalProperties': False, 'required': ['decisions'],
            'properties': {'decisions': {'type': 'array', 'items': row}}}


def api_chunk_generator(context, matcher_prompt_text):
    """context = api_rubric_generator(...).chunk_context: the unchanged generator key / model / settings / schema / normalise."""
    from src.online_rl.tree import backends as B
    key, model, base = context['key'], context['model'], context['settings']
    member = B.MEMBER_MATCH_INSTRUCTIONS
    if base.get('public_criterion_protocol') is not None: member += '\n' + B.PUBLIC_MEMBER_ALIGNMENT_INSTRUCTIONS
    elif base.get('instruction_alignment_protocol') is not None: member += '\n' + B.MEMBER_ALIGNMENT_INSTRUCTIONS
    contract = PROMPT.read_text()
    settings = {**base, 'response_format': None, 'max_tokens': int(base['max_tokens']) * CHUNK, 'chunk_protocol': VERSION,
                'chunk_prompt_sha256': sha256_text(contract), 'matcher_prompt_sha256': sha256_text(matcher_prompt_text),
                'member_prompt_sha256': sha256_text(member)}
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
                peer = row['peer']; agents = sum(turn[0] == 'agent' for turn in tin.dialogue)
                if (not isinstance(peer, dict) or set(peer) != PEER_KEYS or not isinstance(peer['checks'], dict) or
                        set(peer['checks']) != set(B.MATCH_CHECKS) or not all(v is True for v in peer['checks'].values()) or
                        not isinstance(peer['rationale'], str) or isinstance(peer['selected_agent_turn_id'], bool) or
                        not isinstance(peer['selected_agent_turn_id'], int) or not 1 <= peer['selected_agent_turn_id'] <= agents or
                        not isinstance(peer['intervention_instruction'], str) or not peer['intervention_instruction'].strip()):
                    hard.append(f'{tid}: malformed peer object (all six checks true, an actual Agent reply id and an instruction are required)')
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
                  '\n\nPEER MEMBER CONTRACT (unchanged member text; it defines selected_agent_turn_id and intervention_instruction of a '
                  'peer; the canonical rubric is the one generated for the peer representative in this call):\n' + member +
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
    working = CB.validate_snapshot(snapshot)
    canon_all, assignments_all, witness_all, witness_objects, selection_all = {}, [], [], {}, []
    reuse_rows, decisions, chunk_reports = [], [], []
    uncovered = list(sampling['not_selected_ids']); orders = {'success': 0, 'failure': 0}
    for index, ids in enumerate(chunks(selected)):
        tins = [tin_of[i] for i in ids]; before = working
        identity = {'protocol': VERSION, 'case': cid, 'chunk_index': index, 'prompt': state.prompt_sha256,
                    'tins': [[t.leaf_id, t.sha256] for t in tins], 'bank_sha256': before['sha256'], 'transcript': TRANSCRIPT_FORMAT,
                    'model': chunk_generate.model, 'settings': chunk_generate.settings}
        result = cache.call('offline_chunk_generation', identity, lambda: chunk_generate(package, tins, before))
        if result.get('bank_sha256') != before['sha256']: raise RuntimeError('Chunk result belongs to another bank')
        rows = result['decisions']; chunk_sha = sha256_json(identity); groups = {}; report = []
        for tid in ids:
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
                rub = generate_rubric(create_text, state.prompt_sha256, tin, _preset(context, chunk_sha, row), cache, env.gate_fn,
                                      source_stage1=stage1(leaf) if stage1 else None)
                kind = 'CREATED_NEW' if rub.valid else 'CREATE_REJECTED'
                note = '' if rub.valid else 'generated rubric rejected: ' + ', '.join(rub.problems or RB._gate_reasons(rub.gate))
                canon = {rub.rubric_id: rub}; members = [leaf]; accepted = {}; refused = {}
                assignments = [RubricAssignment(tid, cid, layer, orders[layer], rub.rubric_id, kind, None, note, sha256_json(tin.feedback),
                                                RB._member_spec(rub, tin, kind, None))]
                orders[layer] += 1
                for pid in ids:
                    peer = rows[pid]['peer'] if rows[pid]['action'] == 'PEER' else None
                    if peer is None or peer['peer_representative_id'] != tid: continue
                    if not rub.valid: refused[pid] = 'representative rubric rejected'; continue
                    decision = {'decision': 'REUSE_EXISTING', 'rubric_id': rub.rubric_id, 'checks': dict(peer['checks']), 'rationale': peer['rationale'],
                                'selected_agent_turn_id': peer['selected_agent_turn_id'], 'intervention_instruction': peer['intervention_instruction']}
                    try: spec = RB._member_spec(rub, tin_of[pid], 'REUSED_EXISTING', decision)
                    except ValueError as exc: refused[pid] = f'member instruction invalid: {exc}'; continue
                    assignments.append(RubricAssignment(pid, cid, layer, orders[layer], rub.rubric_id, 'REUSED_EXISTING',
                        {**decision, 'peer_representative_id': tid, 'chunk_protocol': VERSION}, '', sha256_json(tin_of[pid].feedback), spec))
                    orders[layer] += 1; members.append(leaf_of[pid]); accepted[pid] = True
                wit = trainer._verify_bounded(case, tree, feedback, members, canon, assignments)
                chosen = trainer._witness_selection(canon, wit)
                kept = any(d.retained for d in chosen)
                if kept:
                    working = CB.extend_snapshot(working, [{'case_id': cid, 'case_cursor': state.case_cursor, 'policy_sha256': state.policy_sha256,
                        'prompt_sha256': state.prompt_sha256, 'trajectory_rubrics': [to_dict(r) for r in canon.values()],
                        'selection': [to_dict(d) for d in chosen], 'rubric_bank': {'witness': [to_dict(w) for w in wit.values()]}}])
                bank_id = 'bank_' + CB.content_hash(rub.rubric) if kept else None
                groups[tid] = {'rubric_id': rub.rubric_id, 'kept': kept, 'bank_id': bank_id, 'accepted': accepted, 'refused': refused}
                canon_all.update(canon); assignments_all.extend(assignments); witness_all.extend(to_dict(w) for w in wit.values())
                witness_objects.update(wit); selection_all.extend(chosen)
                decisions.append({**base, 'action': 'GENERATED_KEEP' if kept else 'GENERATED_REMOVE', 'input_bank_sha256': input_sha,
                                  'output_bank_sha256': working['sha256'], 'bank_id': bank_id, 'rubric_id': rub.rubric_id})
                w = wit.get(rub.rubric_id)
                report.append({'trajectory_id': tid, 'layer': layer, 'decision': 'CREATE', 'rubric_id': rub.rubric_id, 'name': (rub.rubric or {}).get('name'),
                               'structure_valid': not rub.problems, 'problems': list(rub.problems or ()),
                               'gate_passed': (rub.gate or {}).get('passed') if rub.gate is not None else None, 'peers': sorted(accepted),
                               'witness_decision': getattr(w, 'decision', None), 'witness_stop_reason': getattr(w, 'stop_reason', None),
                               'witness_valid_supports': getattr(w, 'num_valid_supports', None), 'witness_reruns': getattr(w, 'num_reruns_executed', None),
                               'validated_by': getattr(w, 'validated_by_trajectory_id', None), 'keep': kept, 'bank_id': bank_id})
            else:
                rep = row['peer']['peer_representative_id']; group = groups[rep]; member = tid in group['accepted']
                if not member: uncovered.append(tid)
                reused = member and group['kept']
                decisions.append({**base, 'action': 'REUSED_EXISTING' if reused else 'GENERATED_REMOVE', 'input_bank_sha256': input_sha,
                                  'output_bank_sha256': input_sha, 'bank_id': group['bank_id'] if reused else None, 'rubric_id': group['rubric_id']})
                report.append({'trajectory_id': tid, 'layer': layer, 'decision': 'PEER', 'peer_of': rep, 'member': member,
                               'refused': group['refused'].get(tid), 'representative_keep': group['kept'], 'bank_id': group['bank_id'] if reused else None})
        chunk_reports.append({'chunk_index': index, 'trajectory_ids': ids, 'trajectories': len(ids), 'generator_calls': 1,
                              'generator_api_requests': result['api_calls'], 'bank_before_sha256': before['sha256'],
                              'bank_before_size': len(before['entries']), 'bank_after_sha256': working['sha256'],
                              'bank_after_size': len(working['entries']), 'rows': report})
    for earlier, later in zip(chunk_reports, chunk_reports[1:]):
        if earlier['bank_after_sha256'] != later['bank_before_sha256']: raise RuntimeError('A chunk did not start from the previous chunk bank')
    if len(chunk_reports) > CAP // CHUNK: raise RuntimeError('More than six generator calls for one case')
    summary = {'protocol': VERSION, **sampling, 'uncovered_ids': sorted(uncovered), 'chunks': chunk_reports,
               'generator_calls': len(chunk_reports), 'generator_api_requests': sum(c['generator_api_requests'] for c in chunk_reports),
               'input_bank_sha256': snapshot['sha256'], 'output_bank_sha256': working['sha256']}
    return {'working': working, 'canon': canon_all, 'assignments': assignments_all, 'witness': witness_all, 'witness_objects': witness_objects,
            'selection': selection_all, 'cumulative_reuse': reuse_rows, 'decisions': decisions, 'summary': summary}
