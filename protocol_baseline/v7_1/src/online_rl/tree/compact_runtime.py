"""Opt-in v7 compact judge adapter; no legacy cache/criterion reinterpretation."""
import copy
import json
from pathlib import Path
from src.common.config import IntegrationUnavailable
from src.online_rl.integrated.records import sha256_json
from src.online_rl.tree.applicability import condition_state, condition_items, public_client_identity
from src.online_rl.tree.pilot_evidence_scorer import public_behavior_items
from src.online_rl.tree.compact_audit import prepare, validate_response
from src.online_rl.tree.backends import with_role

PROTOCOL = 'compact_runtime_v7_1'
APP_CONTRACT = 'compact_condition_v7_1'
ROOT = Path(__file__).resolve().parents[3]


def upgrade(payload):
    q = prepare(payload, semantic=True)
    packet = json.loads(q['messages'][1]['content'])
    kind = 'behavior' if 'responses' in packet else 'applicability'
    old = (ROOT / f'prompts/sherpa/compact_{kind}_semantics_v1.txt').read_text()
    new = (ROOT / f'prompts/sherpa/compact_{kind}_semantics_v2.txt').read_text()
    q['messages'][0]['content'] = q['messages'][0]['content'].replace(old, new)
    for item in packet['rubric_items' if kind == 'behavior' else 'items'].values():
        item['instructions'] = item['instructions'].replace(old, new)
    q['messages'][1]['content'] = json.dumps(packet, ensure_ascii=False)
    return q


def compact_scorer(client, app_client, response_batch_size=4):
    def invoke(judge, packet, role):
        payload = upgrade({'model': judge.model, 'max_tokens': judge.max_tokens,
            'messages': [{'role': 'system', 'content': judge.system},
                         {'role': 'user', 'content': json.dumps(packet, ensure_ascii=False)}]})
        body = with_role(role, lambda: judge._chat(payload['messages'],
            payload['response_format']['json_schema']['schema'], role, return_body=True))
        # _chat uses the same schema with its fixed name; preserve actual wire name.
        payload['response_format']['json_schema']['name'] = 'decisions'
        result = validate_response(payload, body)
        audit = {'requests': [{'messages': payload['messages'], 'messages_sha256': sha256_json(payload['messages']),
                  'payload': payload}], 'decisions': [body], 'protocol': PROTOCOL,
                 'validation': result, 'exact_quotes_validated': result['status'] == 'valid_execution_contract',
                 'semantic_entailment': 'model_assessment_not_independent_proof'}
        if result['status'] != 'valid_execution_contract':
            error = IntegrationUnavailable('Compact judge execution failed: ' + str(result))
            error.audit = audit
            raise error
        return result, audit

    def review(view, prefix, rubrics):
        result = {}
        # Stable singleton request, shared by source/witness/sibling cache.
        for rubric in rubrics:
            rid = rubric['rubric_id']
            packet = {'stage': 'applicability', 'state': condition_state(view, prefix),
                      'items': condition_items([rubric])}
            row, audit = invoke(app_client, packet, 'rubric_applicability')
            item = row['applicability'][rid]
            if item['applicable'] is None:
                error = IntegrationUnavailable('Unresolved compact applicability')
                error.audit = audit
                raise error
            raw = json.loads(audit['decisions'][0]['choices'][0]['message']['content'])['answers'][rid]
            result[rid] = {'applicable': item['applicable'], 'stage1_contract': APP_CONTRACT,
                          'stage1': raw, 'evidence_validation': audit}
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
                result, audit = invoke(client, packet, 'rubric_scorer')
                item = result['answers'][nid][rid]
                out[nid][rid] = {'applicable': True, 'probabilities': None, 'hard': item['score'],
                    'scoring_status': item['scoring_status'], 'stage2': item['evidence'],
                    'evidence_validation': audit}
        return out

    def behavior(view, prefix, rubrics, reply, matches=None):
        return behavior_batch(view, prefix, rubrics, [{'response_id': 'target', 'reply': reply}])['target']

    def score(view, prefix, rubrics, reply):
        app = review(view, prefix, rubrics)
        on = [r for r in rubrics if app[r['rubric_id']]['applicable']]
        judged = behavior(view, prefix, on, reply)
        return {r['rubric_id']: {**app[r['rubric_id']], **judged.get(r['rubric_id'], {
            'hard': None, 'probabilities': None, 'stage2': None, 'scoring_status': 'not_applicable'})} for r in rubrics}

    files = {str(p.relative_to(ROOT)): p.read_text() for p in [
        ROOT/'src/online_rl/tree/compact_runtime.py', ROOT/'src/online_rl/tree/compact_audit.py',
        *[ROOT/f'prompts/sherpa/compact_{kind}_{suffix}.txt' for kind in ('applicability','behavior')
          for suffix in ('contract_v1','semantics_v1','semantics_v2')]]}
    review.protocol = sha256_json({'contract': APP_CONTRACT, 'client': public_client_identity(app_client), 'files': files})
    review.model = app_client.model
    review.applicability_contract = APP_CONTRACT
    score.review_state = score.applicability = review
    score.score_behavior = behavior
    score.score_behavior_batch = behavior_batch
    score.response_batch_size = response_batch_size
    score.review_identity = {'contract': APP_CONTRACT, 'protocol': review.protocol}
    score.model = client.model
    score.protocol = sha256_json({'contract': PROTOCOL, 'client': public_client_identity(client),
                                 'app': review.protocol, 'files': files})
    score.behavior_cache_contract = 'public_behavior_content_alias_v1'
    return score
