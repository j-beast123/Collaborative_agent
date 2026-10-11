"""Opt-in v7 compact judge adapter; no legacy cache/criterion reinterpretation."""
import copy
import json
from pathlib import Path
from src.common.config import IntegrationUnavailable
from src.online_rl.integrated.records import sha256_json
from src.online_rl.tree.applicability import ApplicabilityUncertain, condition_state, condition_items, public_client_identity
from src.online_rl.tree.pilot_evidence_scorer import public_behavior_items
from src.online_rl.tree.compact_audit import prepare, validate_response
from src.online_rl.tree.backends import with_role

PROTOCOL = 'compact_runtime_v7_1'
APP_CONTRACT = 'compact_condition_v7_1'
QUALITY_PROTOCOL = 'compact_runtime_v7_3'
QUALITY_APP_CONTRACT = 'compact_condition_v7_3'
REFERENT_PROTOCOL = 'compact_runtime_v7_5_v7_6'
REFERENT_APP_CONTRACT = 'compact_condition_v7_5'
REVISIONS = ('v7_1', 'v7_3', 'v7_5_v7_6')
ROOT = Path(__file__).resolve().parents[3]


def upgrade(payload, *, revision='v7_1'):
    if revision not in REVISIONS:
        raise ValueError('Unknown compact scorer revision')
    source_packet = json.loads(payload['messages'][1]['content'])
    kind = 'behavior' if 'responses' in source_packet else 'applicability'
    q = prepare(payload, semantic=True, quality=revision != 'v7_1',
                relation=revision == 'v7_5_v7_6' and kind == 'applicability')
    packet = json.loads(q['messages'][1]['content'])
    old = (ROOT / f'prompts/sherpa/compact_{kind}_semantics_v1.txt').read_text()
    new = (ROOT / f'prompts/sherpa/compact_{kind}_semantics_v{2 if revision == "v7_1" else 3}.txt').read_text()
    q['messages'][0]['content'] = q['messages'][0]['content'].replace(old, new)
    for item in packet['rubric_items' if kind == 'behavior' else 'items'].values():
        item['instructions'] = item['instructions'].replace(old, new)
    q['messages'][1]['content'] = json.dumps(packet, ensure_ascii=False)
    if revision == 'v7_5_v7_6':
        names = (('compact_applicability_relation_v7_4.txt',
                  'compact_applicability_referent_v7_5.txt') if kind == 'applicability'
                 else ('compact_behavior_referent_v7_6.txt',))
        # Exactly the pilot's system-only supplements.  The public rubric,
        # user packet and behavior schema stay unchanged.
        for name in names:
            q['messages'][0]['content'] += '\n\n' + (ROOT/'prompts/sherpa'/name).read_text()
    return q


def compact_scorer(client, app_client, response_batch_size=4, *, revision='v7_1', execution_retry_attempts=None):
    if revision not in REVISIONS:
        raise ValueError('Unknown compact scorer revision')
    app_contract = {'v7_1': APP_CONTRACT, 'v7_3': QUALITY_APP_CONTRACT,
                    'v7_5_v7_6': REFERENT_APP_CONTRACT}[revision]
    protocol = {'v7_1': PROTOCOL, 'v7_3': QUALITY_PROTOCOL,
                'v7_5_v7_6': REFERENT_PROTOCOL}[revision]
    def invoke(judge, packet, role):
        payload = upgrade({'model': judge.model, 'max_tokens': judge.max_tokens,
            'messages': [{'role': 'system', 'content': judge.system},
                         {'role': 'user', 'content': json.dumps(packet, ensure_ascii=False)}]}, revision=revision)
        body = with_role(role, lambda: judge._chat(payload['messages'],
            payload['response_format']['json_schema']['schema'], role, return_body=True))
        # _chat uses the same schema with its fixed name; preserve actual wire name.
        payload['response_format']['json_schema']['name'] = 'decisions'
        result = validate_response(payload, body)
        audit = {'requests': [{'messages': payload['messages'], 'messages_sha256': sha256_json(payload['messages']),
                  'payload': payload}], 'decisions': [body], 'protocol': protocol,
                 'validation': result, 'exact_quotes_validated': result['status'] == 'valid_execution_contract',
                 'semantic_entailment': 'model_assessment_not_independent_proof'}
        if result['status'] != 'valid_execution_contract':
            error = IntegrationUnavailable('Compact judge execution failed: ' + str(result))
            error.audit = audit
            raise error
        return result, audit

    def invoke_bounded(judge, packet, role):
        """The unchanged request, repeated locally while its response is execution-invalid; (None, record) when the configured
        budget is exhausted (EXECUTION_UNRESOLVED: no verdict, never a score). Without a budget: the original single call."""
        if not execution_retry_attempts: return invoke(judge, packet, role)
        import time
        failed = []
        for attempt in range(execution_retry_attempts):
            try: return invoke(judge, packet, role)
            except IntegrationUnavailable as exc:
                validation = (getattr(exc, 'audit', None) or {}).get('validation') or {}
                if validation.get('status') not in ('execution_error', 'execution_incomplete'): raise
                failed.append({k: str(validation.get(k))[:160] for k in ('status', 'kind', 'error')})
                if attempt == execution_retry_attempts - 1:
                    return None, {'execution_unresolved': True, 'total_attempts': len(failed), 'reasons_by_attempt': failed}
                print(json.dumps({'event': 'judge_execution_retry', 'stage': 'behavior', 'attempt': attempt + 1, 'of': execution_retry_attempts,
                                  'status': validation.get('status'), 'kind': validation.get('kind'), 'error': str(validation.get('error'))[:120]}), flush=True)
                time.sleep(min(2 * (attempt + 1), 20))

    def review(view, prefix, rubrics):
        result = {}
        # Stable singleton request, shared by source/witness/sibling cache.
        for rubric in rubrics:
            rid = rubric['rubric_id']
            packet = {'stage': 'applicability', 'state': condition_state(view, prefix),
                      'items': condition_items([rubric])}
            row, audit = invoke(app_client, packet, 'rubric_applicability')
            item = row['applicability'][rid]
            if item['applicable'] is None and revision == 'v7_1':
                error = IntegrationUnavailable('Unresolved compact applicability')
                error.audit = audit
                raise error
            raw = json.loads(audit['decisions'][0]['choices'][0]['message']['content'])['answers'][rid]
            result[rid] = {'applicable': item['applicable'], 'stage1_contract': app_contract,
                          'stage1': raw, 'applicability_status': item['status'],
                          'request_state': item.get('request_state'), 'evidence_validation': audit}
        return result

    def behavior_batch(view, prefix, rubrics, responses):
        ids = [r['response_id'] for r in responses]
        if len(ids) != len(set(ids)) or len(ids) > response_batch_size:
            raise ValueError('Invalid sibling batch')
        out = {nid: {} for nid in ids}
        # One response/criterion per call bounds evidence output at the original token limit.
        for response in responses:
            nid = response['response_id']
            for rubric in rubrics:
                rid = rubric['rubric_id']
                packet = {'stage': 'level', 'shared_pre_reply_state': condition_state(view, prefix),
                    'rubric_items': public_behavior_items([rubric], client.system),
                    'responses': [response]}
                result, audit = invoke_bounded(client, packet, 'rubric_scorer')
                if result is None:                               # only this response x criterion pair is left without a judgment
                    out[nid][rid] = {'applicable': True, 'probabilities': None, 'hard': None, 'scoring_status': 'execution_unresolved',
                                     'stage2': None, 'total_attempts': audit['total_attempts'], 'reasons_by_attempt': audit['reasons_by_attempt']}
                    continue
                item = result['answers'][nid][rid]
                out[nid][rid] = {'applicable': True, 'probabilities': None, 'hard': item['score'],
                    'scoring_status': item['scoring_status'], 'stage2': item['evidence'],
                    'evidence_validation': audit}
        return out

    def behavior(view, prefix, rubrics, reply, matches=None):
        return behavior_batch(view, prefix, rubrics, [{'response_id': 'target', 'reply': reply}])['target']

    def score(view, prefix, rubrics, reply):
        app = review(view, prefix, rubrics)
        if any(row['applicable'] is None for row in app.values()):
            error = ApplicabilityUncertain('Uncertain compact applicability; no rubric reward')
            error.audit = {'applicability': app}
            raise error
        on = [r for r in rubrics if app[r['rubric_id']]['applicable']]
        judged = behavior(view, prefix, on, reply)
        return {r['rubric_id']: {**app[r['rubric_id']], **judged.get(r['rubric_id'], {
            'hard': None, 'probabilities': None, 'stage2': None, 'scoring_status': 'not_applicable'})} for r in rubrics}

    # The legacy protocol hashed literal source bytes. Keep its frozen bytes
    # available inside this isolated copy so selecting v7_1 retains the exact
    # old cache identity even though this opt-in module also implements v7_3.
    source_root = ROOT if revision != 'v7_1' else ROOT/'protocol_baseline/v7_1'
    file_paths = [
        source_root/'src/online_rl/tree/compact_runtime.py',
        source_root/'src/online_rl/tree/compact_audit.py',
        *[ROOT/f'prompts/sherpa/compact_{kind}_{suffix}.txt' for kind in ('applicability','behavior')
          for suffix in ('contract_v1','semantics_v1',f'semantics_v{2 if revision == "v7_1" else 3}')]]
    if revision == 'v7_5_v7_6':
        file_paths += [ROOT/'prompts/sherpa'/name for name in (
            'compact_applicability_relation_v7_4.txt', 'compact_applicability_referent_v7_5.txt',
            'compact_behavior_referent_v7_6.txt')]
    files = {str(p.relative_to(source_root if p.is_relative_to(source_root) else ROOT)): p.read_text()
             for p in file_paths}
    review.protocol = sha256_json({'contract': app_contract, 'client': public_client_identity(app_client), 'files': files})
    review.model = app_client.model
    review.applicability_contract = app_contract
    score.review_state = score.applicability = review
    score.score_behavior = behavior
    score.score_behavior_batch = behavior_batch
    score.response_batch_size = response_batch_size
    # Exposed for no-API preflight to prove the configured arm is selected.
    score.compact_revision = revision
    score.review_identity = {'contract': app_contract, 'protocol': review.protocol}
    score.model = client.model
    score.protocol = sha256_json({'contract': protocol, 'client': public_client_identity(client),
                                 'app': review.protocol, 'files': files})
    score.behavior_cache_contract = 'public_behavior_content_alias_v1'
    return score
