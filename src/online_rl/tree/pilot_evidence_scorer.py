"""Behavior scoring with exact response evidence, enabled explicitly for pilot or training.

Evidence validation checks provenance and internal consistency, not medical truth
or whether a quoted passage semantically satisfies a rubric.
"""
from __future__ import annotations

import copy
import json

from src.common.config import IntegrationUnavailable
from src.online_rl.integrated.records import sha256_text
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


def evidence_scorer(client, response_batch_size: int = 1):
    """Return an evidence scorer exposing separate applicability and behavior stages.

The client retains its existing one format repair. In addition, this adapter
allows one evidence-validation correction with the same input and raw answer;
both judge records remain in client.log. A repeated failure is an operational
error, never a numeric score, NA, or a fabricated response.
"""
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
        for attempt in range(2):
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
                if attempt == 1:
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
        for attempt in range(2):
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
                if attempt == 1:
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
