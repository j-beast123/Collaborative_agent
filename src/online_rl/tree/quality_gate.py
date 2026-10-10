"""Evidence-backed admission checks for a generated interaction rubric.

This gate does not measure outcome effectiveness. It grades the source reply
without receiving its outcome; the expected source anchor is compared locally.
Invalid judge output is an operational error, never a negative quality label.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

from src.common.config import IntegrationUnavailable
from src.online_rl.integrated.records import sha256_json, sha256_text
from src.online_rl.tree.agent_state import rubric_problems, visible_rubric
from src.online_rl.tree.records import to_dialogue


PROTOCOL = 'evidence_gate_v2'
INPUT_VERSION = 'selected_prefix_and_separate_source_reply_v1'
FIELDS = {
    'answer_presupposition': {'type': 'boolean'},
    'answer_presupposition_quote': {'type': 'string'},
    'answer_presupposition_reason': {'type': 'string'},
    'new_domain_premise': {'type': 'boolean'},
    'new_domain_premise_quote': {'type': 'string'},
    'new_domain_premise_reason': {'type': 'string'},
    'observable': {'type': 'boolean'},
    'observable_quote': {'type': 'string'},
    'observable_reason': {'type': 'string'},
    'score_boundaries_clear': {'type': 'boolean'},
    'score_boundaries_quote': {'type': 'string'},
    'score_boundaries_other_quote': {'type': 'string'},
    'score_boundaries_reason': {'type': 'string'},
    'source_level': {'type': 'string', 'enum': ['0', '1', '2', 'UNSCORABLE']},
    'source_minimum_quote': {'type': 'string'},
    'source_completion_quote': {'type': 'string'},
    'source_missing_or_conflicting': {'type': 'string'},
    'source_reason': {'type': 'string'},
}
ITEM = {
    'task': 'Review the visible candidate rubric and independently assess the separate source Agent reply.',
    'checks': {
        'answer_presupposition': 'True only if satisfying an anchor requires a predetermined answer, factual conclusion, agreement or disagreement. An open-ended examination is not itself a predetermined conclusion.',
        'new_domain_premise': 'True if the rubric asserts or requires a particular domain fact not present in its admissible evidence. Asking to evaluate a relationship or identify a missing fact without supplying the answer is not itself a new factual assertion.',
        'observable': 'Every anchor must be decidable from visible local text without gold, hidden beliefs, final outcome or later turns.',
        'score_boundaries_clear': 'The actual written 0/1/2 anchors must separate absent minimum contribution, partial performance with a named remaining gap, and adequate completion of that same behavior, without overlap, uncovered coherent responses, or adding an unrelated requirement.',
        'source_level': 'Independently assess the source reply from its text and prefix against the written anchors. No expected score is provided. Use UNSCORABLE if a unique supported score cannot be assigned.',
    },
    'evidence_contract': 'Give nonempty reasons for all checks. Nonempty rubric quotes must be verbatim contiguous substrings of one visible rubric field. Every blocking check needs a rubric quote. Quote both conflicting anchors for overlap. Source minimum/completion quotes must be verbatim substrings of source_agent_reply.text, never substituted from the prefix. Grade the whole reply; do not repair missing relations yourself.',
}
RUBRIC_QUOTE_PROTOCOL = 'visible_field_enum_v1'
RUBRIC_QUOTE_FIELDS = (
    'answer_presupposition_quote', 'new_domain_premise_quote', 'observable_quote',
    'score_boundaries_quote', 'score_boundaries_other_quote',
)
RUBRIC_QUOTE_POLICY = {
    'protocol': RUBRIC_QUOTE_PROTOCOL,
    'fields': list(RUBRIC_QUOTE_FIELDS),
    'catalog': 'Empty string followed by unique nonempty complete string field values of visible_rubric, excluding rubric_id, in traversal order; preserve literal text.',
    'validation': 'Require exact catalog membership, then apply unchanged local evidence and source-anchor validation.',
}
RUBRIC_QUOTE_INSTRUCTION = (
    ' For the five rubric quote fields, select either the empty string or exactly one complete '
    'visible rubric field value from the output schema enum. Do not rewrite, shorten, join, '
    'or normalize its text. This changes only how rubric evidence is quoted; keep the same '
    'independent checks, decisions, and reasons. Source reply evidence quotes remain verbatim '
    'substrings of source_agent_reply.text and do not use this rubric catalog.'
)


def _text_values(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            if key != 'rubric_id':
                yield from _text_values(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _text_values(item)


def _quote_catalog_contract(state):
    """Each call owns its schema; concurrent candidates cannot share quote enums."""
    fields, item = copy.deepcopy(FIELDS), copy.deepcopy(ITEM)
    catalog = list(dict.fromkeys([''] + [text for text in _text_values(state['candidate_rubric'])
                                       if text.strip()]))
    for name in RUBRIC_QUOTE_FIELDS:
        fields[name]['enum'] = list(catalog)
    item['evidence_contract'] += RUBRIC_QUOTE_INSTRUCTION
    return fields, item


def gate_state(tin, rubric):
    """Whitelist source-local evidence; never expose provenance, feedback or outcome."""
    turn = ((rubric.get('source_provenance') or {}).get('analysis') or {}).get('selected_agent_turn_id')
    if type(turn) is not int or turn < 1:
        raise ValueError('evidence gate requires an actual integer selected_agent_turn_id')
    prefix, target, number = [], None, 0
    for item in to_dialogue(tin.dialogue):
        if item['stage'] == 'FINAL':
            break
        if item['speaker'] == 'agent':
            number += 1
            if number == turn:
                target = item
                break
        prefix.append(item)
    if target is None or not target['text'].strip():
        raise ValueError('evidence gate selected Agent reply is missing or empty')
    return {
        'case_context': {'scenario': tin.scenario, 'question': tin.question, 'options': dict(tin.options)},
        'selected_agent_turn_id': turn,
        'dialogue_before_target_turn': prefix,
        'source_agent_reply': target,
        'candidate_rubric': visible_rubric(rubric),
    }


def validate_evidence(answer, state):
    """Validate traceability and output consistency; not the semantic judgement."""
    if not isinstance(answer, dict) or set(answer) != set(FIELDS):
        raise ValueError('gate answer fields differ from the requested contract')
    for key, spec in FIELDS.items():
        if spec['type'] == 'boolean':
            if type(answer[key]) is not bool:
                raise ValueError(f'{key} must be a boolean')
        elif not isinstance(answer[key], str):
            raise ValueError(f'{key} must be a string')
    for key in ('answer_presupposition_reason', 'new_domain_premise_reason', 'observable_reason',
                'score_boundaries_reason', 'source_reason'):
        if not answer[key].strip():
            raise ValueError(f'{key} must be nonempty')
    rubric_strings = list(_text_values(state['candidate_rubric']))
    for key in ('answer_presupposition_quote', 'new_domain_premise_quote', 'observable_quote',
                'score_boundaries_quote', 'score_boundaries_other_quote'):
        quote = answer[key]
        if quote and (not quote.strip() or not any(quote in text for text in rubric_strings)):
            raise ValueError(f'{key} must be an exact quote from one visible rubric field')
    for check, blocked in (('answer_presupposition', True), ('new_domain_premise', True),
                           ('observable', False), ('score_boundaries_clear', False)):
        quote_key = 'score_boundaries_quote' if check == 'score_boundaries_clear' else check + '_quote'
        if answer[check] is blocked and not answer[quote_key].strip():
            raise ValueError(f'{check} requires a quoted rubric clause supporting rejection')
    level = answer['source_level']
    if level not in FIELDS['source_level']['enum']:
        raise ValueError('invalid source_level')
    reply = state['source_agent_reply']['text']
    for key in ('source_minimum_quote', 'source_completion_quote'):
        quote = answer[key]
        if quote and (not quote.strip() or quote not in reply):
            raise ValueError(f'{key} must be an exact quote from the source Agent reply')
    if level in ('1', '2') and not answer['source_minimum_quote'].strip():
        raise ValueError('source levels 1/2 require a minimum contribution quote')
    if level == '0' and (answer['source_minimum_quote'] or answer['source_completion_quote']):
        raise ValueError('source level 0 cannot include minimum/completion evidence')
    if level == '2':
        if not answer['source_completion_quote'].strip() or answer['source_missing_or_conflicting'].strip():
            raise ValueError('source level 2 requires completion evidence and no remaining gap')
    elif not answer['source_missing_or_conflicting'].strip():
        raise ValueError('source below 2 or UNSCORABLE requires a missing/conflicting explanation')


def evidence_posthoc_gate(client, cases, observed_context=False, rubric_quote_protocol=None):
    """One quality+blind-anchor call, with at most one evidence correction.

The client separately has one JSON-format repair. A quality rejection never
triggers a re-vote. The correction only repairs a malformed evidence record.
"""
    if not callable(getattr(client, 'judge', None)):
        raise ValueError('evidence_gate_v2 requires a structured judge client')
    if rubric_quote_protocol not in (None, RUBRIC_QUOTE_PROTOCOL):
        raise ValueError('Unsupported rubric quote protocol: ' + str(rubric_quote_protocol))
    settings = {
        'protocol': PROTOCOL, 'input_version': INPUT_VERSION,
        'model': client.model, 'prompt_sha256': sha256_text(client.system),
        'client_protocol_sha256': getattr(client, 'protocol_sha256', None),
        'max_tokens': client.max_tokens, 'timeout_s': getattr(client, 'timeout_s', None),
        'observed_context': bool(observed_context),
        'contract_sha256': sha256_json({'fields': FIELDS, 'item': ITEM}),
        'evidence_corrections': 1,
    }
    if rubric_quote_protocol is not None:
        item = copy.deepcopy(ITEM)
        item['evidence_contract'] += RUBRIC_QUOTE_INSTRUCTION
        settings['rubric_quote_protocol'] = rubric_quote_protocol
        settings['contract_sha256'] = sha256_json({
            'fields': FIELDS, 'item': item, 'dynamic_rubric_quote_policy': RUBRIC_QUOTE_POLICY,
        })

    def gate(tin, rubric):
        from src.online_rl.tree.backends import with_role
        if tin.case_id not in cases:
            raise ValueError('Gate source case is not in this run')
        problems = rubric_problems(rubric)
        result = {
            'protocol': PROTOCOL, 'schema': not problems, 'passed': False,
            'reasons': [], 'unresolved': [], 'leakage_deterministic': [],
        }
        if problems:
            result['reasons'] = ['schema: ' + str(problems)]
            return result
        if tin.outcome not in ('success', 'failure'):
            raise ValueError('Gate source outcome must be an observed success or failure')
        state = gate_state(tin, rubric)
        # No unconditional option-string veto: neutral references are distinguishable
        # from answer steering by the evidence-backed answer_presupposition check.
        attempts, requests, errors = [], [], []
        current = copy.deepcopy(state)
        for attempt in range(2):
            fields, item = (_quote_catalog_contract(state) if rubric_quote_protocol is not None
                            else (FIELDS, ITEM))
            user = json.dumps({'stage': PROTOCOL, 'state': current, 'items': {'candidate': item}}, ensure_ascii=False)
            messages = [{'role': 'system', 'content': client.system}, {'role': 'user', 'content': user}]
            requests.append({'messages': messages, 'messages_sha256': sha256_json(messages)})
            if rubric_quote_protocol is not None:
                requests[-1].update(output_fields=copy.deepcopy(fields), output_fields_sha256=sha256_json(fields))
            answers, record = with_role('rubric_gate', lambda: client.judge(
                PROTOCOL, current, {'candidate': item}, fields, 'tree_posthoc_rubric_gate_v2'))
            attempts.append(copy.deepcopy(record))
            answer = answers.get('candidate') if isinstance(answers, dict) else None
            try:
                if not isinstance(answers, dict) or set(answers) != {'candidate'}:
                    raise ValueError('gate response must contain exactly candidate')
                validate_evidence(answer, state)
                if rubric_quote_protocol is not None:
                    for name in RUBRIC_QUOTE_FIELDS:
                        if answer[name] not in fields[name]['enum']:
                            raise ValueError(f'{name} must select one complete literal visible rubric field from its enum')
                break
            except ValueError as exc:
                errors.append(str(exc))
                if attempt == 1:
                    error = IntegrationUnavailable('Invalid evidence_gate_v2 evidence after one correction: ' + str(exc))
                    error.audit = {'requests': requests, 'decisions': attempts, 'validation_errors': errors}
                    raise error from exc
                current = {**copy.deepcopy(state), 'evidence_correction': {
                    'previous_answer': answer, 'error': str(exc),
                    'instruction': 'Correct only the evidence/output-contract error against the same input. Do not infer an expected source score or force acceptance. Use exact quotes and independently reassess any claim lacking support.',
                }}
        expected = '2' if tin.outcome == 'success' else '0'
        reasons = []
        for key, block in (('answer_presupposition', True), ('new_domain_premise', True),
                           ('observable', False), ('score_boundaries_clear', False)):
            if answer[key] is block:
                reasons.append(key if block else 'not_' + key)
        if answer['source_level'] == 'UNSCORABLE':
            reasons.append('source_unscorable')
        elif answer['source_level'] != expected:
            reasons.append('source_anchor_mismatch')
        result.update({
            **{k: answer[k] for k in ('answer_presupposition', 'new_domain_premise', 'observable', 'score_boundaries_clear')},
            'passed': not reasons, 'reasons': reasons, 'assessment': answer,
            'source_anchor': {'expected_level': expected, 'observed_level': answer['source_level'],
                              'matches': answer['source_level'] == expected},
            'selected_agent_turn_id': state['selected_agent_turn_id'],
            'input_sha256': sha256_json(state),
            'evidence': {'requests': requests, 'validation_errors': errors, 'attempts': len(attempts)},
            'decisions': attempts,
        })
        return result

    gate.model, gate.protocol, gate.settings = client.model, sha256_json(settings), settings
    return gate


PUBLIC_GATE_PROTOCOL = 'public_mc_gate_v1'
PUBLIC_GATE_PROTOCOL_V2 = 'public_mc_gate_v2'
PUBLIC_GATE_PROTOCOL_V3 = 'public_mc_gate_v3'
PUBLIC_FIELD_REFERENCE_PROTOCOL = 'input_field_refs_v2'
QUALITY_CATEGORIES = ('mc_anchor_alignment', 'meaningful_partial', 'boundary_coverage', 'source_grounding',
                      'unsupported_premise', 'source_claim_strength', 'concise_sufficiency',
                      'conclusion_neutrality', 'observability')
PROBE_TYPES = ('below_minimum', 'minimum_only', 'complete', 'alternate_partial',
               'concise_complete', 'alternative_conclusion')
_DIALOGUE_QUOTE = {'turn_id': {'type': 'string'}, 'speaker': {'type': 'string'}, 'quote': {'type': 'string'}}
_QUALITY_CHECK = {
    'passed': {'type': 'boolean'}, 'rubric_quotes': {'type': 'array', 'items': {'type': 'string'}},
    'dialogue_evidence': {'type': 'array', 'items': {'type': 'object', 'additionalProperties': False,
                                                  'required': list(_DIALOGUE_QUOTE), 'properties': _DIALOGUE_QUOTE}},
    'reason': {'type': 'string'},
}
_PROBE_ANCHOR = {'level': {'type': 'string', 'enum': ['0', '1', '2']}, 'quote': {'type': 'string'}}
_PROBE = {
    'kind': {'type': 'string', 'enum': ['synthetic_probe']},
    'probe_type': {'type': 'string', 'enum': list(PROBE_TYPES)}, 'response': {'type': 'string'},
    'matching_levels': {'type': 'array', 'items': {'type': 'string', 'enum': ['0', '1', '2']}},
    'anchor_quotes': {'type': 'array', 'items': {'type': 'object', 'additionalProperties': False,
                                              'required': list(_PROBE_ANCHOR), 'properties': _PROBE_ANCHOR}},
    'boundary_problem': {'type': 'string', 'enum': ['none', 'overlap', 'gap']}, 'reason': {'type': 'string'},
}
PUBLIC_GATE_FIELDS = {
    'checks': {'type': 'object', 'additionalProperties': False, 'required': list(QUALITY_CATEGORIES),
               'properties': {category: {'type': 'object', 'additionalProperties': False,
                                        'required': list(_QUALITY_CHECK), 'properties': copy.deepcopy(_QUALITY_CHECK)}
                              for category in QUALITY_CATEGORIES}},
    'score_boundaries_clear': {'type': 'boolean'},
    'boundary_probes': {'type': 'array', 'minItems': len(PROBE_TYPES), 'maxItems': len(PROBE_TYPES),
                       'items': {'type': 'object', 'additionalProperties': False,
                                 'required': list(_PROBE), 'properties': _PROBE}},
}


def public_gate_state(tin, rubric):
    from src.online_rl.tree.applicability import condition_state
    state = gate_state(tin, rubric)
    evidence = condition_state(state['case_context'], state['dialogue_before_target_turn'])
    state['dialogue_before_target_turn'] = evidence['dialogue_before_target_reply']
    state['case_evidence'] = evidence['case_evidence']
    state['source_agent_reply'] = {**state['source_agent_reply'], 'turn_id': 'source'}
    return state


def validate_public_quality(answer, state, *, quote_catalog=False):
    """Check literal evidence and expose contradictions as blocking findings."""
    from src.online_rl.tree.pilot_evidence_scorer import anchor_text
    if not isinstance(answer, dict) or set(answer) != set(PUBLIC_GATE_FIELDS):
        raise ValueError('Public Gate fields differ')
    if type(answer['score_boundaries_clear']) is not bool or not isinstance(answer['checks'], dict) or set(answer['checks']) != set(QUALITY_CATEGORIES):
        raise ValueError('Public Gate categories or boundary flag differ')
    strings = list(_text_values(state['candidate_rubric']))
    catalog = {row['turn_id']: row for row in state['case_evidence'] + state['dialogue_before_target_turn'] + [state['source_agent_reply']]}
    for name, check in answer['checks'].items():
        if not isinstance(check, dict) or set(check) != set(_QUALITY_CHECK) or type(check['passed']) is not bool:
            raise ValueError(f'{name}: invalid quality check')
        if not isinstance(check['reason'], str) or not check['reason'].strip():
            raise ValueError(f'{name}: substantive reason required')
        if not isinstance(check['rubric_quotes'], list) or not isinstance(check['dialogue_evidence'], list):
            raise ValueError(f'{name}: evidence arrays required')
        for quote in check['rubric_quotes']:
            if not isinstance(quote, str) or not quote.strip() or not any(quote in text for text in strings):
                raise ValueError(f'{name}: quote not found in one public criterion field')
            if quote_catalog and quote not in strings:
                raise ValueError(f'{name}: rubric quote must select a complete literal visible field')
        if not check['passed'] and not check['rubric_quotes']:
            raise ValueError(f'{name}: rejection must quote its actual criterion clause')
        for quote in check['dialogue_evidence']:
            if (not isinstance(quote, dict) or set(quote) != set(_DIALOGUE_QUOTE)
                    or any(not isinstance(value, str) or not value.strip() for value in quote.values())):
                raise ValueError(f'{name}: malformed dialogue evidence')
            source = catalog.get(quote['turn_id'])
            if source is None or source['speaker'] != quote['speaker'] or quote['quote'] not in source['text']:
                raise ValueError(f'{name}: source quote speaker/turn/text differs')
        if name == 'source_grounding' and check['passed'] and not check['dialogue_evidence']:
            raise ValueError('Passing source grounding requires actual source-local evidence')
    probes = answer['boundary_probes']
    if not isinstance(probes, list) or len(probes) != len(PROBE_TYPES):
        raise ValueError('All six explicit synthetic boundary probes are required')
    seen, contradictions = set(), []
    for probe in probes:
        if not isinstance(probe, dict) or set(probe) != set(_PROBE):
            raise ValueError('Synthetic probe fields differ')
        if probe['kind'] != 'synthetic_probe' or probe['probe_type'] not in PROBE_TYPES or probe['probe_type'] in seen:
            raise ValueError('Synthetic probe kind/type is missing or repeated')
        seen.add(probe['probe_type'])
        if any(not isinstance(probe[key], str) or not probe[key].strip() for key in ('response', 'reason')):
            raise ValueError('Synthetic probe response and reason must be explicit')
        levels = probe['matching_levels']
        if not isinstance(levels, list) or any(level not in ('0', '1', '2') for level in levels) or len(levels) != len(set(levels)):
            raise ValueError('Probe matching levels must be distinct written score levels')
        anchors = probe['anchor_quotes']
        if not isinstance(anchors, list) or not anchors:
            raise ValueError('Probe must quote the anchors supporting a match or the identified gap')
        quoted_levels = set()
        for row in anchors:
            if not isinstance(row, dict) or set(row) != set(_PROBE_ANCHOR) or row['level'] not in ('0', '1', '2'):
                raise ValueError('Probe anchor quote requires its actual level')
            if row['quote'] != anchor_text(state['candidate_rubric'], row['level']):
                raise ValueError('Probe anchor quote does not equal the written anchor')
            quoted_levels.add(row['level'])
        if not set(levels) <= quoted_levels:
            raise ValueError('Each probe matching level needs its exact corresponding anchor')
        actual_problem = 'gap' if not levels else ('overlap' if len(levels) > 1 else 'none')
        if probe['boundary_problem'] not in _PROBE['boundary_problem']['enum']:
            raise ValueError('Unknown boundary problem')
        if actual_problem != 'none':
            contradictions.append('boundary_probe_' + actual_problem)
        if probe['boundary_problem'] != actual_problem:
            contradictions.append('boundary_probe_self_contradiction')
    if answer['score_boundaries_clear'] and (contradictions or not answer['checks']['boundary_coverage']['passed']):
        contradictions.append('boundary_flag_contradicts_evidence')
    return list(dict.fromkeys(contradictions))


def public_rubric_field_catalog(rubric):
    """Stable literal visible-field references, without source/provenance data."""
    result = {}
    def visit(value, path):
        if isinstance(value, str) and value.strip():
            result[path] = value
        elif isinstance(value, dict):
            for key, child in value.items():
                if key != 'rubric_id':
                    visit(child, f'{path}.{key}' if path else key)
    visit(visible_rubric(rubric), '')
    return result


def public_gate_reference_fields(catalog):
    fields = copy.deepcopy(PUBLIC_GATE_FIELDS)
    for check in fields['checks']['properties'].values():
        check['properties'].pop('rubric_quotes')
        check['properties']['rubric_refs'] = {'type': 'array', 'items': {'type': 'string', 'enum': list(catalog)}}
        check['required'] = list(check['properties'])
    probe = fields['boundary_probes']['items']
    probe['properties'].pop('anchor_quotes')
    probe['properties']['anchor_levels'] = {'type': 'array', 'minItems': 1,
                                          'items': {'type': 'string', 'enum': ['0', '1', '2']}}
    probe['required'] = list(probe['properties'])
    return fields


def materialize_public_quality(answer, state):
    """Resolve references to exact rubric/anchor text before normal validation."""
    from src.online_rl.tree.pilot_evidence_scorer import anchor_text
    catalog = public_rubric_field_catalog(state['candidate_rubric'])
    fields = public_gate_reference_fields(catalog)
    if not isinstance(answer, dict) or set(answer) != set(fields):
        raise ValueError('V2 Gate fields differ')
    resolved = copy.deepcopy(answer)
    checks = resolved.get('checks')
    if not isinstance(checks, dict) or set(checks) != set(QUALITY_CATEGORIES):
        raise ValueError('V2 Gate categories differ')
    for name, check in checks.items():
        if not isinstance(check, dict) or set(check) != set(fields['checks']['properties'][name]['properties']):
            raise ValueError(f'{name}: V2 check fields differ')
        refs = check.pop('rubric_refs')
        if not isinstance(refs, list) or any(not isinstance(ref, str) or ref not in catalog for ref in refs):
            raise ValueError(f'{name}: unknown visible rubric field reference')
        check['rubric_quotes'] = [catalog[ref] for ref in refs]
    probes = resolved.get('boundary_probes')
    if not isinstance(probes, list):
        raise ValueError('V2 boundary probes must be an array')
    for probe in probes:
        if not isinstance(probe, dict) or set(probe) != set(fields['boundary_probes']['items']['properties']):
            raise ValueError('V2 probe fields differ')
        levels = probe.pop('anchor_levels')
        if (not isinstance(levels, list) or not levels or any(level not in ('0', '1', '2') for level in levels)
                or len(set(levels)) != len(levels)):
            raise ValueError('V2 probe anchor levels must be distinct actual levels')
        probe['anchor_quotes'] = [{'level': level, 'quote': anchor_text(state['candidate_rubric'], level)}
                                  for level in levels]
    contradictions = validate_public_quality(resolved, state, quote_catalog=True)
    return resolved, contradictions


def public_mc_posthoc_gate(client, cases, observed_context=False, rubric_quote_protocol=None,
                           *, protocol=PUBLIC_GATE_PROTOCOL, instructions_override=None, blind_scorer_override=None):
    """Quality/probes and source performance are independent input packets.

    Source anchoring remains success=2/failure=0. A legitimate source 1 is
    preserved and rejects that candidate; it is never coerced to satisfy 0.
    """
    from src.online_rl.tree.pilot_evidence_scorer import (evidence_scorer, PUBLIC_EVIDENCE_CONTRACT,
                                                       PUBLIC_EVIDENCE_CONTRACT_V2, PUBLIC_EVIDENCE_CONTRACT_V3)
    if protocol not in (PUBLIC_GATE_PROTOCOL, PUBLIC_GATE_PROTOCOL_V2, PUBLIC_GATE_PROTOCOL_V3):
        raise ValueError('Unsupported public Gate protocol')
    v2 = protocol in (PUBLIC_GATE_PROTOCOL_V2, PUBLIC_GATE_PROTOCOL_V3)
    v3 = protocol == PUBLIC_GATE_PROTOCOL_V3
    allowed = (None, PUBLIC_FIELD_REFERENCE_PROTOCOL) if v2 else (None, RUBRIC_QUOTE_PROTOCOL)
    if rubric_quote_protocol not in allowed:
        raise ValueError('Unsupported public Gate rubric quote protocol')
    if v2:
        rubric_quote_protocol = PUBLIC_FIELD_REFERENCE_PROTOCOL
    instructions = (Path(__file__).resolve().parents[3] /
                    f'prompts/sherpa/gate_public_mc_v{3 if v3 else 2 if v2 else 1}.txt').read_text()
    blind_scorer = evidence_scorer(client,
        evidence_contract=PUBLIC_EVIDENCE_CONTRACT_V3 if v3 else PUBLIC_EVIDENCE_CONTRACT_V2 if v2 else PUBLIC_EVIDENCE_CONTRACT)
    instructions = instructions_override or instructions
    blind_scorer = blind_scorer_override or blind_scorer
    settings = {'protocol': protocol, 'input_version': 'separate_quality_and_blind_source_v1',
                'model': client.model, 'prompt_sha256': sha256_text(client.system),
                'client_protocol_sha256': getattr(client, 'protocol_sha256', None),
                'max_tokens': client.max_tokens, 'timeout_s': getattr(client, 'timeout_s', None),
                'observed_context': bool(observed_context), 'rubric_quote_protocol': rubric_quote_protocol,
                'contract_sha256': sha256_json({'fields': PUBLIC_GATE_FIELDS, 'instructions': instructions}),
                'blind_source_protocol': blind_scorer.protocol, 'quality_calls': 1, 'blind_source_calls': 1,
                'source_anchor_policy': 'success2_failure0_unchanged', 'evidence_corrections': 1}
    if v2:
        settings['input_version'] = 'separate_quality_and_blind_source_semantic_v3' if v3 else 'separate_quality_and_blind_source_semantic_v2'
        settings['reference_schema_sha256'] = sha256_json(public_gate_reference_fields({'visible_field': ''}))

    def gate(tin, rubric):
        from src.online_rl.tree.backends import with_role
        from src.online_rl.tree.public_criterion import content_hash
        if tin.case_id not in cases or tin.outcome not in ('success', 'failure'):
            raise ValueError('Gate requires a declared source case and observed outcome')
        problems = rubric_problems(rubric)
        result = {'protocol': protocol, 'schema': not problems, 'passed': False,
                  'public_criterion_version': rubric.get('criterion_version'),
                  'reasons': [], 'unresolved': [], 'leakage_deterministic': []}
        if problems:
            result['reasons'] = ['schema: ' + str(problems)]
            return result
        result['criterion_sha256'] = content_hash(rubric)
        state = public_gate_state(tin, rubric)
        current, requests, decisions, errors = copy.deepcopy(state), [], [], []
        item = {'instructions': instructions, 'probe_evidence_status': 'synthetic_diagnostics_not_independent_gold'}
        if v3:
            item['criterion_contract'] = ('public_mc_definitions' if 'criterion_version' in rubric
                                           else 'legacy_written_anchors')
        fields = copy.deepcopy(PUBLIC_GATE_FIELDS)
        if v2:
            catalog = public_rubric_field_catalog(state['candidate_rubric'])
            current['rubric_field_catalog'] = catalog
            fields = public_gate_reference_fields(catalog)
        elif rubric_quote_protocol is not None:
            catalog = list(dict.fromkeys(text for text in _text_values(state['candidate_rubric']) if text.strip()))
            for check in fields['checks']['properties'].values():
                check['properties']['rubric_quotes']['items']['enum'] = catalog
        for attempt in range(2):
            messages = [{'role': 'system', 'content': client.system}, {'role': 'user', 'content': json.dumps(
                {'stage': protocol, 'state': current, 'items': {'candidate': item}}, ensure_ascii=False)}]
            requests.append({'messages': messages, 'messages_sha256': sha256_json(messages),
                             'output_fields': copy.deepcopy(fields)})
            values, record = with_role('rubric_gate', lambda: client.judge(
                protocol, current, {'candidate': item}, fields, 'public_criterion_quality'))
            decisions.append(copy.deepcopy(record))
            answer = values.get('candidate') if isinstance(values, dict) else None
            try:
                if not isinstance(values, dict) or set(values) != {'candidate'}:
                    raise ValueError('Public Gate response must contain exactly candidate')
                if v2:
                    answer, contradictions = materialize_public_quality(answer, state)
                else:
                    contradictions = validate_public_quality(answer, state, quote_catalog=rubric_quote_protocol is not None)
                break
            except (ValueError, TypeError) as exc:
                errors.append(str(exc))
                if attempt:
                    error = IntegrationUnavailable('Invalid public Gate evidence after one correction: ' + str(exc))
                    error.audit = {'requests': requests, 'decisions': decisions, 'validation_errors': errors}
                    raise error from exc
                current = {**copy.deepcopy(state), 'evidence_correction': {'previous_answer': answer, 'error': str(exc)}}
                if v2:
                    current['rubric_field_catalog'] = catalog
        # No quality checks, probes, outcomes, expected levels, analysis or
        # reflection are included in this separate blind behavioral assessment.
        visible = {**state['candidate_rubric'], 'rubric_id': 'source_criterion'}
        source = blind_scorer.score_behavior(state['case_context'], state['dialogue_before_target_turn'],
                                             [visible], state['source_agent_reply']['text'])['source_criterion']
        observed = source['stage2']['level']
        expected = '2' if tin.outcome == 'success' else '0'
        reasons = [name for name, check in answer['checks'].items() if not check['passed']]
        reasons += contradictions
        if not answer['score_boundaries_clear']:
            reasons.append('not_score_boundaries_clear')
        if observed == 'UNSCORABLE':
            reasons.append('source_unscorable')
        elif observed != expected:
            reasons.append('source_anchor_mismatch')
        result.update(passed=not reasons, reasons=list(dict.fromkeys(reasons)), assessment=answer,
                      observable=answer['checks']['observability']['passed'],
                      score_boundaries_clear=answer['score_boundaries_clear'] and not contradictions
                          and answer['checks']['boundary_coverage']['passed'],
                      source_anchor={'expected_level': expected, 'observed_level': observed, 'matches': observed == expected,
                          'generator_declared_level': ((rubric.get('source_provenance') or {}).get('analysis') or {}).get('source_score')},
                      blind_source_assessment=source['stage2'], selected_agent_turn_id=state['selected_agent_turn_id'],
                      input_sha256=sha256_json(state),
                      evidence={'requests': requests, 'validation_errors': errors, 'attempts': len(decisions),
                                'blind_source': source['evidence_validation'],
                                'synthetic_probes_are_independent_gold': False,
                                'exact_quotes_are_semantic_proof': False}, decisions=decisions)
        return result

    gate.model, gate.protocol, gate.settings = client.model, sha256_json(settings), settings
    return gate
