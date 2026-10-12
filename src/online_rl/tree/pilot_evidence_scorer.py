"""Behavior scoring with exact response evidence, enabled explicitly for pilot or training.

Evidence validation checks provenance and internal consistency, not medical truth
or whether a quoted passage semantically satisfies a rubric.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

from src.common.config import IntegrationUnavailable
from src.online_rl.integrated.records import sha256_json, sha256_text
from src.online_rl.tree.agent_state import rubric_problems
from src.online_rl.tree.backends import stage2_items, two_stage_structured_scorer, with_role


PROTOCOL = 'pilot_response_evidence_v1'
FIELDS = {
    'level': {'type': 'string', 'enum': ['0', '1', '2', 'UNSCORABLE']},
    'anchor_basis': {'type': 'string'},
    'minimum_contribution_quote': {'type': 'string'},
    'completion_quote': {'type': 'string'},
    'missing_or_conflicting': {'type': 'string'},
}
INSTRUCTIONS = """Assess the target response alone against the supplied 0/1/2 anchors.
Before awarding 1 or 2, identify the minimum useful contribution required by
those anchors and quote the exact target-response passage performing it in
minimum_contribution_quote. A bare position, acknowledgement, or promise does
not establish an explanation/comparison that the anchor requires. For 2,
completion_quote must quote the exact target-response passage completing the
remaining mandatory behavior and anchor_basis must explain how every required
part is visibly satisfied. Quote a contiguous span, including multiple sentences
if necessary. Do not manufacture a link between separate assertions; identify
any missing required connection or unresolved contradiction in
missing_or_conflicting. Only 2 has an empty missing_or_conflicting field; for 1,
name the actual required gap remaining for 2. For 0, explain why the minimum
contribution is absent. Use only requirements actually in the anchors; do not
add an independent correctness criterion. Never substitute facts from the
prefix, medical knowledge, the rubric, or another response for target-response
evidence. A quotation does not by itself prove that the requirement is satisfied.
If no complete anchor covers the response, return UNSCORABLE with the criterion
gap explained in missing_or_conflicting, rather than forcing a numeric score or
changing applicability. For absent evidence return an empty quote. All nonempty
quotes must be verbatim substrings of target_agent_reply, with no ellipses,
paraphrases, or added quotation marks. anchor_basis must be nonempty.
"""


def validate_evidence(answer: dict, reply: str) -> None:
    """Reject invalid evidence without coercing it to a lower score or NA."""
    if not isinstance(answer, dict) or set(answer) != set(FIELDS):
        raise ValueError('Evidence fields differ from the requested scoring contract')
    if any(not isinstance(answer[k], str) for k in FIELDS):
        raise ValueError('All evidence fields must be strings')
    level = answer['level']
    if level not in FIELDS['level']['enum']:
        raise ValueError('Invalid evidence level')
    if not answer['anchor_basis'].strip():
        raise ValueError('anchor_basis must explain the decision')
    for key in ('minimum_contribution_quote', 'completion_quote'):
        quote = answer[key]
        if quote and (not quote.strip() or quote not in reply):
            raise ValueError(f'{key} must be an exact nonblank substring of the target reply')
    if level in ('1', '2') and not answer['minimum_contribution_quote'].strip():
        raise ValueError('Levels 1 and 2 require evidence of the minimum contribution')
    if level == '0' and answer['minimum_contribution_quote']:
        raise ValueError('Level 0 cannot simultaneously report a minimum contribution quote')
    if level == '0' and answer['completion_quote']:
        raise ValueError('Level 0 cannot simultaneously report a completion quote')
    missing = answer['missing_or_conflicting'].strip()
    if level == '2':
        if not answer['completion_quote'].strip():
            raise ValueError('Level 2 requires an exact completion quote')
        if missing:
            raise ValueError('Level 2 cannot simultaneously report missing/conflicting required behavior')
    elif not missing:
        raise ValueError('Below 2 or UNSCORABLE must explain the remaining gap or conflict')


def evidence_scorer(client, response_batch_size: int = 1, *, evidence_contract=None, applicability_client=None):
    """Return an evidence scorer exposing separate applicability and behavior stages.

The client retains its existing one format repair. In addition, this adapter
allows one evidence-validation correction with the same input and raw answer;
both judge records remain in client.log. A repeated failure is an operational
error, never a numeric score, NA, or a fabricated response.
"""
    if evidence_contract is not None:
        if evidence_contract not in (PUBLIC_EVIDENCE_CONTRACT, PUBLIC_EVIDENCE_CONTRACT_V2, PUBLIC_EVIDENCE_CONTRACT_V3):
            raise ValueError('Unknown behavioral evidence contract: ' + str(evidence_contract))
        return public_evidence_scorer(client, response_batch_size, applicability_client=applicability_client,
                                      evidence_contract=evidence_contract)
    if applicability_client is not None:
        raise ValueError('A separate applicability client requires the public evidence contract')
    base = two_stage_structured_scorer(client, observed_context=True, response_batch_size=response_batch_size)

    def visible_prefix(prefix):
        return [{key: turn[key] for key in ('speaker', 'stage', 'round', 'text') if key in turn}
                for turn in prefix]

    def review_state(view, prefix, rubrics):
        return base.review_state(view, visible_prefix(prefix), rubrics)

    def result(value, attempts, errors):
        return {'applicable': True, 'probabilities': None,
                'hard': None if value['level'] == 'UNSCORABLE' else int(value['level']),
                'scoring_status': 'rubric_unscorable' if value['level'] == 'UNSCORABLE' else 'scored',
                'stage2': value, 'evidence_validation': {
                    'protocol': PROTOCOL, 'attempts': attempts, 'errors': list(errors),
                    'exact_quotes_validated': True}}

    def score_behavior(view, prefix, rubrics, reply, matches=None):
        if not rubrics:
            return {}
        for rubric in rubrics:
            if rubric_problems(rubric):
                raise ValueError(f"invalid visible rubric {rubric.get('rubric_id')}")
        state = {
            'case_context': {'question': view['question'], 'options': view['options']},
            'dialogue_before_target_reply': visible_prefix(prefix),
            'target_agent_reply': reply,
        }
        items = stage2_items(rubrics)
        for item in items.values():
            item['instructions'] += '\n' + INSTRUCTIONS
        errors = []
        for attempt in range(1):
            values, _ = with_role('rubric_scorer', lambda: client.judge(
                'level', state, items, FIELDS, 'pilot_response_behavior_evidence'))
            try:
                if set(values) != set(items):
                    raise ValueError('Returned rubric IDs differ from the requested set')
                for value in values.values():
                    validate_evidence(value, reply)
                return {rid: result(value, attempt + 1, errors) for rid, value in values.items()}
            except (ValueError, TypeError) as exc:
                errors.append(str(exc))
                if True:
                    raise IntegrationUnavailable(
                        f'Pilot scoring evidence invalid after one correction: {errors[-1]}') from exc
                items = copy.deepcopy(items)
                correction = ('\nYour previous evidence was invalid: ' + str(exc) +
                              '\nPrevious answer: ' + json.dumps(values, ensure_ascii=False) +
                              '\nReassess the same target reply against the unchanged anchors. '
                              'Return valid literal evidence and a consistent level; do not invent a quote.')
                for item in items.values():
                    item['instructions'] += correction

    def score_behavior_batch(view, prefix, rubrics, responses):
        """Validate every quote against its own sibling response; malformed evidence never becomes a score."""
        if not responses or not rubrics:
            return {}
        response_ids = [row['response_id'] for row in responses]
        if len(response_ids) != len(set(response_ids)) or len(responses) > response_batch_size:
            raise ValueError('Evidence batch needs unique response IDs within the configured batch size')
        if not callable(getattr(client, 'judge_batch', None)):
            return {row['response_id']: score_behavior(view, prefix, rubrics, row['reply']) for row in responses}
        for rubric in rubrics:
            if rubric_problems(rubric):
                raise ValueError(f"invalid visible rubric {rubric.get('rubric_id')}")
        state = {'case_context': {'question': view['question'], 'options': view['options']},
                 'dialogue_before_target_reply': visible_prefix(prefix)}
        replies = {row['response_id']: row['reply'] for row in responses}
        batch_rows = [{'response_id': rid, 'target_agent_reply': reply} for rid, reply in replies.items()]
        items = stage2_items(rubrics)
        for item in items.values():
            item['instructions'] += '\n' + INSTRUCTIONS
        errors = []
        for attempt in range(1):
            values, _ = with_role('rubric_scorer', lambda: client.judge_batch(
                'level', state, batch_rows, items, FIELDS, 'response_behavior_evidence_batch'))
            try:
                if not isinstance(values, dict) or set(values) != set(replies):
                    raise ValueError('Returned response IDs differ from the requested set')
                for response_id, row in values.items():
                    if not isinstance(row, dict) or set(row) != set(items):
                        raise ValueError(f'{response_id}: returned rubric IDs differ from the requested set')
                    for value in row.values():
                        try:
                            validate_evidence(value, replies[response_id])
                        except (ValueError, TypeError) as exc:
                            raise ValueError(f'{response_id}: {exc}') from exc
                return {response_id: {rid: result(value, attempt + 1, errors) for rid, value in row.items()}
                        for response_id, row in values.items()}
            except (ValueError, TypeError) as exc:
                errors.append(str(exc))
                if True:
                    raise IntegrationUnavailable(
                        f'Batch scoring evidence invalid after one correction: {errors[-1]}') from exc
                items = copy.deepcopy(items)
                correction = ('\nYour previous evidence was invalid: ' + str(exc) +
                              '\nPrevious answer: ' + json.dumps(values, ensure_ascii=False) +
                              '\nReassess each unchanged target reply against the unchanged anchors. '
                              'Return literal evidence from that response ID only and a consistent level. '
                              'Never transfer evidence between responses.')
                for item in items.values():
                    item['instructions'] += correction

    def score(view, prefix, rubrics, reply):
        app = review_state(view, prefix, rubrics)
        on = [rubric for rubric in rubrics if app[rubric['rubric_id']]['applicable']]
        behavior = score_behavior(view, prefix, on, reply)
        return {
            rubric['rubric_id']: {
                **app[rubric['rubric_id']],
                **behavior.get(rubric['rubric_id'], {
                    'hard': None, 'stage2': None, 'scoring_status': 'not_applicable'}),
            } for rubric in rubrics}

    score.review_state = review_state
    score.score_behavior = score_behavior
    score.score_behavior_batch = score_behavior_batch
    score.response_batch_size = response_batch_size
    score.model = getattr(client, 'model', None)
    score.protocol = sha256_text(f'{base.protocol}|{PROTOCOL}|{INSTRUCTIONS}|{json.dumps(FIELDS, sort_keys=True)}')
    if response_batch_size > 1:
        score.protocol = sha256_text(score.protocol + '|exact_response_evidence_batch_v1')
    score.review_identity = base.review_identity
    applicability = lambda view, prefix, rubrics: review_state(view, prefix, rubrics)
    applicability.model = score.model
    applicability.protocol = score.review_identity['protocol']
    score.applicability = applicability
    return score


PUBLIC_EVIDENCE_CONTRACT = 'public_mc_evidence_v1'
PUBLIC_EVIDENCE_CONTRACT_V2 = 'public_mc_evidence_v2'
PUBLIC_EVIDENCE_CONTRACT_V3 = 'public_mc_evidence_v3'
LEDGER_SCHEMA_REVISION = 'typed_criterion_ledger_v2'
LEDGER_SCHEMA_REVISION_V2 = 'referenced_criterion_ledger_v3'
LEDGER_SCHEMA_REVISION_V3 = 'requested_task_scope_ledger_v4'
TASK_OPERATORS = ('possibility_any', 'frequency_typicality', 'comparison', 'relation',
                  'identity_membership', 'timing', 'mechanism', 'other')
_TASK_QUOTE = {'turn_id': {'type': 'string'}, 'speaker': {'type': 'string'}, 'quote': {'type': 'string'}}
_TARGET_ASSESSMENT = {
    'target': {'type': 'string'},
    'coverage': {'type': 'string', 'enum': ['addressed', 'explicit_limitation', 'not_addressed']},
    'response_operators': {'type': 'array', 'items': {'type': 'string', 'enum': list(TASK_OPERATORS) + ['information_limit']}},
    'response_quotes': {'type': 'array', 'items': {'type': 'string'}},
}
TASK_FRAME_FIELDS = {
    'requested_operator': {'type': 'string', 'enum': list(TASK_OPERATORS)},
    'requested_targets': {'type': 'array', 'minItems': 1, 'maxItems': 16, 'items': {'type': 'string'}},
    'reference_targets': {'type': 'array', 'maxItems': 16, 'items': {'type': 'string'}},
    'request_quotes': {'type': 'array', 'minItems': 1, 'items': {'type': 'object', 'additionalProperties': False,
        'required': list(_TASK_QUOTE), 'properties': _TASK_QUOTE}},
    'target_assessments': {'type': 'array', 'minItems': 1, 'maxItems': 16,
        'items': {'type': 'object', 'additionalProperties': False, 'required': list(_TARGET_ASSESSMENT),
                  'properties': _TARGET_ASSESSMENT}},
    'scope_met': {'type': 'boolean'},
    'reason': {'type': 'string'},
}
_REQUIREMENT_FIELDS = {
    'condition': {'type': 'string', 'enum': ['minimum_performance', 'completion_condition', 'written_anchor']},
    'definition_quote': {'type': 'string'}, 'met': {'type': 'boolean'},
    'evidence_type': {'type': 'string', 'enum': ['response_spans', 'absence', 'conflict']},
    'response_quotes': {'type': 'array', 'items': {'type': 'string'}},
    'entailment': {'type': 'string'},
}
PUBLIC_FIELDS = {
    'level': {'type': 'string', 'enum': ['0', '1', '2', 'UNSCORABLE']},
    'anchor_quote': {'type': 'string'}, 'anchor_basis': {'type': 'string'},
    'requirements': {'type': 'array', 'minItems': 1, 'maxItems': 12,
                     'items': {'type': 'object', 'additionalProperties': False,
                               'required': list(_REQUIREMENT_FIELDS), 'properties': _REQUIREMENT_FIELDS}},
    'remaining_gap': {'type': 'string'}, 'criterion_defect': {'type': 'string'},
}


def anchor_text(rubric, level):
    value = rubric['scores'][str(level)]
    return value['behavior'] if isinstance(value, dict) else value


def public_behavior_items(rubrics, instructions):
    """Public behavior definition only; no applicability or source provenance."""
    items = {}
    for rubric in rubrics:
        if rubric_problems(rubric):
            raise ValueError(f"Invalid visible rubric {rubric.get('rubric_id')}")
        item = {'rubric': rubric['name'], 'required_behavior': rubric['required_behavior'],
                'score_anchors': {str(level): anchor_text(rubric, level) for level in range(3)},
                'evidence_kind': ('public_mc_definitions' if 'criterion_version' in rubric
                                  else 'legacy_written_anchors'),
                'instructions': instructions}
        if 'criterion_version' in rubric:
            item.update({key: rubric[key] for key in
                         ('criterion_version', 'minimum_performance', 'completion_condition')})
        items[rubric['rubric_id']] = item
    return items


def public_evidence_fields(rubrics, *, evidence_contract=PUBLIC_EVIDENCE_CONTRACT):
    """Constrain output labels without inventing definitions for legacy criteria."""
    fields = copy.deepcopy(PUBLIC_FIELDS)
    kinds = {'criterion_version' in rubric for rubric in rubrics}
    if kinds == {False}:
        labels = ['written_anchor']
    elif kinds == {True}:
        labels = ['minimum_performance', 'completion_condition']
    else:
        labels = list(_REQUIREMENT_FIELDS['condition']['enum'])
    fields['requirements']['items']['properties']['condition']['enum'] = labels
    if evidence_contract in (PUBLIC_EVIDENCE_CONTRACT_V2, PUBLIC_EVIDENCE_CONTRACT_V3):
        fields.pop('anchor_quote')
        fields['anchor_ref'] = {'type': 'string', 'enum': ['none'] + [f'score_anchors.{i}' for i in range(3)]}
        fields['anchor_matches'] = {'type': 'array', 'items': {'type': 'string', 'enum': ['0', '1', '2']}}
        fields['criterion_status'] = {'type': 'string', 'enum': ['coherent', 'ambiguous', 'contradictory']}
        rows = fields['requirements']['items']
        rows['properties'].pop('definition_quote')
        references = ([f'score_anchors.{i}' for i in range(3)] if kinds == {False}
                      else ['minimum_performance', 'completion_condition'] if kinds == {True}
                      else ['minimum_performance', 'completion_condition'] + [f'score_anchors.{i}' for i in range(3)])
        rows['properties']['definition_ref'] = {'type': 'string', 'enum': references}
        rows['required'] = list(rows['properties'])
        if evidence_contract == PUBLIC_EVIDENCE_CONTRACT_V3:
            fields['task_frame'] = {'type': 'object', 'additionalProperties': False,
                                    'required': list(TASK_FRAME_FIELDS), 'properties': copy.deepcopy(TASK_FRAME_FIELDS)}
    elif evidence_contract != PUBLIC_EVIDENCE_CONTRACT:
        raise ValueError('Unknown public behavioral evidence contract')
    return fields


def validate_task_frame(frame, reply, state, *, level):
    """Check scope evidence and internal consistency, never infer semantic facts.

    Operator labels and target resolution are explicit model judgments, not a
    lexical classifier. A mismatch cannot silently become a different reward.
    """
    if not isinstance(frame, dict) or set(frame) != set(TASK_FRAME_FIELDS):
        raise ValueError('Task-frame fields differ')
    if frame['requested_operator'] not in TASK_OPERATORS or type(frame['scope_met']) is not bool:
        raise ValueError('Invalid requested operator or scope flag')
    if not isinstance(frame['reason'], str) or not frame['reason'].strip():
        raise ValueError('Task scope requires an explicit entailment reason')
    targets, references = frame['requested_targets'], frame['reference_targets']
    for values in (targets, references):
        if (not isinstance(values, list) or len(values) > 16 or
                any(not isinstance(v, str) or not v.strip() for v in values) or len(set(values)) != len(values)):
            raise ValueError('Task target sets must contain unique nonblank labels')
    if not targets or set(targets) & set(references):
        raise ValueError('Requested targets must be nonempty and exclude reference-only targets')
    if state is None:
        raise ValueError('Task scope requires the actual visible pre-response state')
    from src.online_rl.tree.applicability import condition_state
    evidence = condition_state(state['case_context'], state['dialogue_before_target_reply'])
    catalog = {r['turn_id']: r for r in evidence['case_evidence'] + evidence['dialogue_before_target_reply']}
    quotes = frame['request_quotes']
    if not isinstance(quotes, list) or not quotes:
        raise ValueError('Requested task needs exact pre-response evidence')
    for q in quotes:
        if not isinstance(q, dict) or set(q) != set(_TASK_QUOTE) or any(not isinstance(v, str) or not v.strip() for v in q.values()):
            raise ValueError('Task request quote fields differ')
        source = catalog.get(q['turn_id'])
        if source is None or source['speaker'] != q['speaker'] or q['quote'] not in source['text']:
            raise ValueError('Task request quote must come from its actual pre-response turn')
    assessments = frame['target_assessments']
    if not isinstance(assessments, list) or len(assessments) != len(targets):
        raise ValueError('Each requested target needs exactly one assessment')
    seen, met = set(), []
    for row in assessments:
        if not isinstance(row, dict) or set(row) != set(_TARGET_ASSESSMENT):
            raise ValueError('Target assessment fields differ')
        target = row['target']
        if not isinstance(target, str) or target not in targets or target in seen:
            raise ValueError('Cannot count a reference-only, unrequested, or duplicate target')
        seen.add(target)
        coverage, operators, spans = row['coverage'], row['response_operators'], row['response_quotes']
        if coverage not in _TARGET_ASSESSMENT['coverage']['enum']:
            raise ValueError('Unknown target coverage status')
        if (not isinstance(operators, list) or any(x not in TASK_OPERATORS + ('information_limit',) for x in operators)
                or len(set(operators)) != len(operators)):
            raise ValueError('Unknown or duplicate response operator')
        if not isinstance(spans, list) or any(not isinstance(q, str) or not q.strip() or q not in reply for q in spans):
            raise ValueError('Task response evidence must occur in its own reply')
        if coverage == 'not_addressed':
            if spans or operators:
                raise ValueError('Unaddressed target uses no invented response evidence')
            met.append(False)
        elif coverage == 'explicit_limitation':
            if not spans or operators != ['information_limit']:
                raise ValueError('An explicit information limit requires its own exact reply evidence')
            met.append(True)
        else:
            if not spans or not operators or 'information_limit' in operators:
                raise ValueError('Addressed target requires performed-operation evidence')
            # The response may perform several operations. Topic overlap or a
            # frequency claim alone cannot stand in for a possibility operation.
            met.append(frame['requested_operator'] in operators)
    if frame['scope_met'] != all(met):
        raise ValueError('Task scope flag contradicts the requested operator/target coverage evidence')
    if level == '2' and not frame['scope_met']:
        raise ValueError('Level2 cannot leave the criterion-relevant requested scope unmet')


def materialize_public_evidence(answer, reply, rubric, *, evidence_contract=PUBLIC_EVIDENCE_CONTRACT_V2,
                                state=None):
    """Resolve checked input references; never repair a semantic level or ledger.

    Raw responses remain in the request decision audit. Returned evidence uses
    the existing literal-text representation for downstream scoring consumers.
    """
    fields = public_evidence_fields([rubric], evidence_contract=evidence_contract)
    if not isinstance(answer, dict) or set(answer) != set(fields):
        raise ValueError('V2 behavioral evidence fields differ')
    matches = answer['anchor_matches']
    if (not isinstance(matches, list) or any(x not in ('0', '1', '2') for x in matches)
            or len(set(matches)) != len(matches)):
        raise ValueError('Anchor matches must be distinct written levels')
    status = answer['criterion_status']
    if status not in ('coherent', 'ambiguous', 'contradictory'):
        raise ValueError('Unknown criterion status')
    level = answer['level']
    if level not in PUBLIC_FIELDS['level']['enum']:
        raise ValueError('Invalid behavior level')
    if level == 'UNSCORABLE':
        if status == 'coherent' or answer['anchor_ref'] != 'none':
            raise ValueError('UNSCORABLE needs an actual scale ambiguity/contradiction, not a numeric anchor')
    elif (status != 'coherent' or matches != [level] or
          answer['anchor_ref'] != f'score_anchors.{level}'):
        raise ValueError('Numeric judgment must uniquely match its unchanged coherent written anchor')
    resolved = copy.deepcopy(answer)
    if evidence_contract == PUBLIC_EVIDENCE_CONTRACT_V3:
        validate_task_frame(resolved.pop('task_frame'), reply, state, level=level)
    for key in ('anchor_matches', 'criterion_status', 'anchor_ref'):
        resolved.pop(key)
    resolved['anchor_quote'] = '' if level == 'UNSCORABLE' else anchor_text(rubric, level)
    references = {f'score_anchors.{i}': anchor_text(rubric, i) for i in range(3)}
    public = 'criterion_version' in rubric
    if public:
        references.update({key: rubric[key] for key in ('minimum_performance', 'completion_condition')})
    if not isinstance(resolved['requirements'], list):
        raise ValueError('Requirement evidence must be an array')
    for row in resolved['requirements']:
        expected_fields = (set(_REQUIREMENT_FIELDS) - {'definition_quote'}) | {'definition_ref'}
        if not isinstance(row, dict) or set(row) != expected_fields:
            raise ValueError('V2 requirement fields differ')
        ref = row.pop('definition_ref')
        if not isinstance(ref, str) or ref not in references:
            raise ValueError('Unknown criterion definition reference')
        if public and ref != row['condition']:
            raise ValueError('Public M/C reference must point to its exact declared condition')
        if not public and not ref.startswith('score_anchors.'):
            raise ValueError('Legacy evidence can reference only written score anchors')
        row['definition_quote'] = references[ref]
    validate_public_evidence(resolved, reply, rubric,
        allow_legacy_failure_conflict=evidence_contract == PUBLIC_EVIDENCE_CONTRACT_V3)
    return resolved


def validate_public_evidence(answer, reply, rubric, *, allow_legacy_failure_conflict=False):
    """Literal traceability and MC consistency, not a semantic truth oracle."""
    if not isinstance(answer, dict) or set(answer) != set(PUBLIC_FIELDS):
        raise ValueError('Public behavioral evidence fields differ')
    if any(not isinstance(answer[k], str) for k in PUBLIC_FIELDS if k != 'requirements'):
        raise ValueError('Behavior level/anchor/gap fields must be strings')
    level = answer['level']
    anchors = {str(i): anchor_text(rubric, i) for i in range(3)}
    if level not in PUBLIC_FIELDS['level']['enum'] or not answer['anchor_basis'].strip():
        raise ValueError('Invalid behavior level or empty anchor explanation')
    if level == 'UNSCORABLE':
        if not answer['criterion_defect'].strip() or not answer['remaining_gap'].strip():
            raise ValueError('UNSCORABLE needs the actual criterion defect and unresolved gap')
        if answer['anchor_quote'] and answer['anchor_quote'] not in anchors.values():
            raise ValueError('UNSCORABLE anchor quote must remain an exact written anchor')
    elif answer['anchor_quote'] != anchors[level] or answer['criterion_defect'].strip():
        raise ValueError('Numeric score requires the exact selected anchor and no criterion defect')
    if level == '2':
        if answer['remaining_gap'].strip():
            raise ValueError('Score 2 cannot report an unresolved required gap')
    elif not answer['remaining_gap'].strip():
        raise ValueError('Below 2 or UNSCORABLE must identify the concrete remaining gap')
    requirements = answer['requirements']
    if not isinstance(requirements, list) or not 1 <= len(requirements) <= 12:
        raise ValueError('A bounded nonempty requirement evidence ledger is required')
    public = 'criterion_version' in rubric
    by_condition = {}
    for row in requirements:
        if not isinstance(row, dict) or set(row) != set(_REQUIREMENT_FIELDS):
            raise ValueError('Requirement evidence fields differ')
        if type(row['met']) is not bool or any(not isinstance(row[k], str) for k in
                                               ('condition', 'definition_quote', 'evidence_type', 'entailment')):
            raise ValueError('Requirement evidence types differ')
        if row['condition'] not in _REQUIREMENT_FIELDS['condition']['enum'] or row['evidence_type'] not in _REQUIREMENT_FIELDS['evidence_type']['enum']:
            raise ValueError('Unknown requirement/evidence type')
        if not row['definition_quote'].strip() or not row['entailment'].strip():
            raise ValueError('Requirement and entailment must be substantive')
        quotes = row['response_quotes']
        if not isinstance(quotes, list) or any(not isinstance(q, str) or not q.strip() or q not in reply for q in quotes):
            raise ValueError('Every evidence quote must occur exactly in its own target response')
        if row['evidence_type'] == 'absence':
            if quotes or (public and row['met']):
                raise ValueError('Absence uses no fabricated quote and cannot establish performed M/C')
        elif not quotes:
            raise ValueError('Performed/conflicting behavior requires response spans')
        legacy_failure_match = (allow_legacy_failure_conflict and not public and level == '0'
                                and row['condition'] == 'written_anchor'
                                and row['definition_quote'] in anchors['0'])
        if row['evidence_type'] == 'conflict' and row['met'] and not legacy_failure_match:
            raise ValueError('Unresolved conflicting evidence cannot establish its condition')
        if public:
            condition = row['condition']
            if condition not in ('minimum_performance', 'completion_condition') or condition in by_condition:
                raise ValueError('Public criteria require exactly one M and one C evidence row')
            if row['definition_quote'] != rubric[condition]:
                raise ValueError('The public M/C definition must not be changed or inferred from analysis')
            by_condition[condition] = row
        elif row['condition'] != 'written_anchor' or not any(row['definition_quote'] in anchor for anchor in anchors.values()):
            raise ValueError('Legacy evidence must quote written anchors; no hidden M/C backfill')
    if public:
        if set(by_condition) != {'minimum_performance', 'completion_condition'}:
            raise ValueError('Public evidence must assess both M and C')
        minimum = by_condition['minimum_performance']['met']
        completion = by_condition['completion_condition']['met']
        expected = '0' if not minimum else ('2' if completion else '1')
        if level != 'UNSCORABLE' and level != expected:
            raise ValueError('Score contradicts its own public M/C evidence')
    if level in ('1', '2') and not any(row['met'] and row['response_quotes'] for row in requirements):
        raise ValueError('Score 1/2 requires actual positive response evidence under its written criterion')


def public_evidence_scorer(client, response_batch_size=1, *, applicability_client=None,
                           evidence_contract=PUBLIC_EVIDENCE_CONTRACT):
    from src.online_rl.tree.applicability import (condition_evidence_reviewer,
        CONDITION_EVIDENCE_CONTRACT, CONDITION_EVIDENCE_CONTRACT_V2, CONDITION_EVIDENCE_CONTRACT_V3)
    if response_batch_size not in range(1, 9):
        raise ValueError('response_batch_size must be between 1 and 8')
    if evidence_contract not in (PUBLIC_EVIDENCE_CONTRACT, PUBLIC_EVIDENCE_CONTRACT_V2, PUBLIC_EVIDENCE_CONTRACT_V3):
        raise ValueError('Unknown public behavioral evidence contract')
    v2 = evidence_contract in (PUBLIC_EVIDENCE_CONTRACT_V2, PUBLIC_EVIDENCE_CONTRACT_V3)
    v3 = evidence_contract == PUBLIC_EVIDENCE_CONTRACT_V3
    revision = LEDGER_SCHEMA_REVISION_V3 if v3 else LEDGER_SCHEMA_REVISION_V2 if v2 else LEDGER_SCHEMA_REVISION
    instructions = (Path(__file__).resolve().parents[3] /
                    f'prompts/sherpa/scorer_public_mc_v{3 if v3 else 2 if v2 else 1}.txt').read_text()
    review = condition_evidence_reviewer(applicability_client or client,
        contract=CONDITION_EVIDENCE_CONTRACT_V3 if v3 else CONDITION_EVIDENCE_CONTRACT_V2 if v2 else CONDITION_EVIDENCE_CONTRACT)

    def state_before(view, prefix):
        state = {'case_context': {key: view[key] for key in ('scenario', 'question', 'options') if key in view},
                'dialogue_before_target_reply': [{key: row[key] for key in ('speaker', 'stage', 'round', 'text') if key in row}
                                                 for row in prefix]}
        if v3:
            from src.online_rl.tree.applicability import condition_state
            state = condition_state(state['case_context'], state['dialogue_before_target_reply'])
        return state

    def assess(view, prefix, rubrics, responses, *, batched):
        if not rubrics or not responses:
            return {}
        ids = [r['response_id'] for r in responses]
        if any(not isinstance(rid, str) or not rid for rid in ids) or len(set(ids)) != len(ids) or len(ids) > response_batch_size:
            raise ValueError('Response evidence batch IDs or batch size are invalid')
        definitions = {rubric['rubric_id']: rubric for rubric in rubrics}
        if len(definitions) != len(rubrics):
            raise ValueError('Duplicate rubric IDs')
        items = public_behavior_items(rubrics, instructions)
        fields = public_evidence_fields(rubrics, evidence_contract=evidence_contract)
        before = state_before(view, prefix)
        replies = {row['response_id']: row['reply'] for row in responses}
        request_rows = [{'response_id': rid, 'target_agent_reply': reply} for rid, reply in replies.items()]
        requests, decisions, errors = [], [], []
        for attempt in range(1):
            if batched:
                packet = {'stage': 'level', 'shared_pre_reply_state': before, 'rubric_items': items, 'responses': request_rows}
                system = client.system + '\nScore every response independently. The shared state is identical for all responses; only the target reply changes. Never transfer evidence or a score between response IDs.'
                messages = [{'role': 'system', 'content': system},
                            {'role': 'user', 'content': json.dumps(packet, ensure_ascii=False)}]
                requests.append({'messages': messages, 'messages_sha256': sha256_json(messages),
                                 'output_fields': copy.deepcopy(fields)})
                values, record = with_role('rubric_scorer', lambda: client.judge_batch(
                    'level', before, request_rows, items, fields, 'public_response_evidence_batch'))
            else:
                state = {**before, 'target_agent_reply': responses[0]['reply']}
                packet = {'stage': 'level', 'state': state, 'items': items}
                messages = [{'role': 'system', 'content': client.system},
                            {'role': 'user', 'content': json.dumps(packet, ensure_ascii=False)}]
                requests.append({'messages': messages, 'messages_sha256': sha256_json(messages),
                                 'output_fields': copy.deepcopy(fields)})
                row, record = with_role('rubric_scorer', lambda: client.judge(
                    'level', state, items, fields, 'public_response_evidence'))
                values = {ids[0]: row}
            decisions.append(copy.deepcopy(record))
            try:
                if not isinstance(values, dict) or set(values) != set(replies):
                    raise ValueError('Behavior response IDs differ')
                normalized = copy.deepcopy(values)
                for rid, rows in values.items():
                    if not isinstance(rows, dict) or set(rows) != set(definitions):
                        raise ValueError(f'{rid}: behavior rubric IDs differ')
                    for rubric_id, answer in rows.items():
                        if v2:
                            normalized[rid][rubric_id] = materialize_public_evidence(answer, replies[rid], definitions[rubric_id],
                                evidence_contract=evidence_contract, state=before)
                        else:
                            validate_public_evidence(answer, replies[rid], definitions[rubric_id])
                return {rid: {rubric_id: {
                    'applicable': True, 'probabilities': None,
                    'hard': None if answer['level'] == 'UNSCORABLE' else int(answer['level']),
                    'scoring_status': 'rubric_unscorable' if answer['level'] == 'UNSCORABLE' else 'scored',
                    'stage2': answer, 'evidence_validation': {'protocol': evidence_contract,
                        'ledger_schema_revision': revision,
                        'attempts': attempt + 1, 'errors': errors, 'requests': requests, 'decisions': decisions,
                        **({'task_frame': copy.deepcopy(values[rid][rubric_id]['task_frame'])} if v3 else {}),
                        'exact_quotes_validated': True, 'semantic_entailment': 'model_assessment_not_independent_proof'}}
                    for rubric_id, answer in rows.items()} for rid, rows in normalized.items()}
            except (ValueError, TypeError) as exc:
                errors.append(str(exc))
                if True:
                    error = IntegrationUnavailable('Invalid public behavioral evidence after one correction: ' + str(exc))
                    error.audit = {'requests': requests, 'decisions': decisions, 'validation_errors': errors}
                    raise error from exc
                items = copy.deepcopy(items)
                correction = ('\nCorrect the evidence contract error against the same unchanged responses/criteria: ' + str(exc)
                              + '\nPrevious output: ' + json.dumps(values, ensure_ascii=False)
                              + '\nNever move evidence between responses or infer a caller-expected score.')
                for item in items.values():
                    item['instructions'] += correction

    def behavior(view, prefix, rubrics, reply, matches=None):
        return assess(view, prefix, rubrics, [{'response_id': 'target', 'reply': reply}], batched=False).get('target', {})

    def behavior_batch(view, prefix, rubrics, responses):
        ids = [row['response_id'] for row in responses]
        if len(ids) != len(set(ids)) or not 1 <= len(ids) <= response_batch_size:
            raise ValueError('Response evidence batch IDs or batch size are invalid')
        if not callable(getattr(client, 'judge_batch', None)):
            return {row['response_id']: behavior(view, prefix, rubrics, row['reply']) for row in responses}
        return assess(view, prefix, rubrics, responses, batched=True)

    def score(view, prefix, rubrics, reply):
        app = review(view, prefix, rubrics)
        on = [r for r in rubrics if app[r['rubric_id']]['applicable']]
        judged = behavior(view, prefix, on, reply)
        return {r['rubric_id']: {**app[r['rubric_id']], **judged.get(r['rubric_id'], {
            'hard': None, 'probabilities': None, 'stage2': None, 'scoring_status': 'not_applicable'})} for r in rubrics}

    score.review_state = score.applicability = review
    score.score_behavior, score.score_behavior_batch = behavior, behavior_batch
    score.model = getattr(client, 'model', None)
    score.response_batch_size = response_batch_size
    score.review_identity = {'protocol': review.protocol, 'contract': review.applicability_contract}
    from src.online_rl.tree.applicability import public_client_identity
    score.protocol = sha256_json({'contract': evidence_contract, 'prompt': sha256_text(instructions),
        'ledger_schema_revision': revision,
        'fields': PUBLIC_FIELDS, 'client': public_client_identity(client),
        'applicability_protocol': review.protocol, 'response_batch_size': response_batch_size})
    if v2:
        score.protocol = sha256_json({'base': score.protocol, 'field_reference_contract': 'input_field_refs_v2',
            'fields': public_evidence_fields([], evidence_contract=evidence_contract)})
    if v3:
        score.protocol = sha256_json({'base': score.protocol,
            'legacy_failure_conflict': 'literal_legacy_zero_anchor_may_match_observed_conflict_v1'})
    score.behavior_cache_contract = 'public_behavior_content_alias_v1'
    return score
