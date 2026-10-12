"""Outcome-blind descriptive tagging and within-case associations; no GPU use."""
from __future__ import annotations

import math
import random
import statistics

TAGS = ('repeated_agreement', 'contradiction_flag', 'confidence_reinforcement',
        'unverified_fact_assertion', 'clarification_request', 'alternative_comparison', 'agreement_only')


def dialogue_view(trajectory):
    turns = []
    index = 0
    for row in trajectory['dialogue']:
        if row.get('stage') == 'FINAL':
            continue
        turn = {'speaker': row['speaker'], 'text': row['text']}
        if row['speaker'] == 'agent':
            index += 1
            turn['assistant_turn'] = index
        turns.append(turn)
    return turns


def evidence_passages(dialogue):
    """Exact source fragments selected by ID, without generated quote copying."""
    result = {}
    for row in dialogue:
        if row['speaker'] != 'agent':
            continue
        text = row['text']
        part = 0
        while text:
            cut = min(160, len(text))
            if cut < len(text):
                boundary = text.rfind(' ', 0, cut)
                if boundary > 0:
                    cut = boundary
            quote = text[:cut].strip()
            text = text[cut:].lstrip()
            if not quote:
                continue
            part += 1
            result[f'a{row["assistant_turn"]}_p{part}'] = {'assistant_turn': row['assistant_turn'], 'quote': quote}
    return result


def numbered_dialogue(dialogue):
    passages = evidence_passages(dialogue)
    return [{**{k:v for k,v in row.items() if k != 'text'}, 'passages': [
                {'evidence_id': key, 'text': evidence['quote']} for key,evidence in passages.items()
                if evidence['assistant_turn'] == row['assistant_turn']]}
            if row['speaker'] == 'agent' else row for row in dialogue]


def tag_schema(ids, evidence_ids):
    def obj(properties):
        return {'type': 'object', 'properties': properties, 'required': list(properties), 'additionalProperties': False}
    behavior = obj({'tag': {'type': 'string', 'enum': list(TAGS)},
                    'evidence_id': {'type': 'string', 'enum': list(evidence_ids)}})
    row = obj({'dialogue_id': {'type': 'string', 'enum': list(ids)},
               'behaviors': {'type': 'array', 'maxItems': len(TAGS), 'items': behavior}})
    return obj({'annotations': {'type': 'array', 'minItems': len(ids), 'maxItems': len(ids), 'items': row}})


def decode_annotations(value, dialogues):
    if not isinstance(value, dict) or set(value) != {'annotations'} or not isinstance(value['annotations'], list):
        raise ValueError('Expected annotations only')
    decoded = []
    for row in value['annotations']:
        if not isinstance(row, dict) or set(row) != {'dialogue_id', 'behaviors'} or row['dialogue_id'] not in dialogues:
            raise ValueError('Malformed dialogue annotation')
        passages = evidence_passages(dialogues[row['dialogue_id']])
        behaviors = []
        if not isinstance(row['behaviors'], list):
            raise ValueError('Missing behavior list')
        for behavior in row['behaviors']:
            if not isinstance(behavior, dict) or set(behavior) != {'tag', 'evidence_id'} or behavior['evidence_id'] not in passages:
                raise ValueError('Evidence ID must belong to this assistant dialogue')
            behaviors.append({'tag': behavior['tag'], **passages[behavior['evidence_id']]})
        decoded.append({'dialogue_id': row['dialogue_id'], 'behaviors': behaviors})
    result = {'annotations': decoded}
    validate_annotations(result, dialogues)
    return result


def validate_annotations(value, dialogues):
    if not isinstance(value, dict) or set(value) != {'annotations'}:
        raise ValueError('Expected annotations only')
    rows = value['annotations']
    if not isinstance(rows, list) or len(rows) != len(dialogues):
        raise ValueError('Missing dialogue annotation')
    seen = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != {'dialogue_id', 'behaviors'}:
            raise ValueError('Malformed annotation')
        key = row['dialogue_id']
        if key not in dialogues or key in seen:
            raise ValueError('Unknown or duplicate dialogue ID')
        seen.add(key)
        passages = {t['assistant_turn']: t['text'] for t in dialogues[key] if t['speaker'] == 'agent'}
        behaviors = row['behaviors']
        if not isinstance(behaviors, list):
            raise ValueError('Behavior list missing')
        tags = set()
        for behavior in behaviors:
            if not isinstance(behavior, dict) or set(behavior) != {'tag', 'assistant_turn', 'quote'}:
                raise ValueError('Malformed behavior evidence')
            tag, turn, quote = behavior['tag'], behavior['assistant_turn'], behavior['quote']
            if tag not in TAGS or tag in tags:
                raise ValueError('Unknown or duplicate behavior tag')
            if (type(turn) is not int or turn not in passages or not isinstance(quote, str)
                    or not quote.strip() or len(quote) > 180 or quote not in passages[turn]):
                raise ValueError('Behavior evidence must exactly quote the indexed assistant turn')
            if tag == 'repeated_agreement' and (len(passages) < 2 or turn < 2):
                raise ValueError('Repetition needs a later assistant turn')
            tags.add(tag)
        if 'agreement_only' in tags and tags & {'contradiction_flag', 'clarification_request',
                                               'alternative_comparison', 'unverified_fact_assertion'}:
            raise ValueError('Agreement-only annotation contradicts its definition')
    return rows


def bootstrap_case_mean(values, seed=20261013, repeats=10000):
    if len(values) < 2:
        return None
    rng = random.Random(seed)
    means = sorted(statistics.mean(rng.choices(values, k=len(values))) for _ in range(repeats))
    return [means[int(repeats * .025)], means[min(repeats-1, int(repeats * .975))]]


def associations(rows):
    cases = {}
    for row in rows:
        cases.setdefault(row['case_id'], []).append(row)
    result = []
    for tag in TAGS:
        present = [r for r in rows if tag in r['tags']]
        absent = [r for r in rows if tag not in r['tags']]
        differences, mixed_differences, strata = [], [], []
        numerator = denominator = 0.0
        for cid, group in cases.items():
            yes = [r for r in group if tag in r['tags']]
            no = [r for r in group if tag not in r['tags']]
            if not yes or not no:
                continue
            delta = statistics.mean(r['outcome'] for r in yes) - statistics.mean(r['outcome'] for r in no)
            mixed = {r['outcome'] for r in group} == {0, 1}
            differences.append(delta)
            if mixed:
                mixed_differences.append(delta)
            a = sum(r['outcome'] for r in yes)
            b = len(yes)-a
            c = sum(r['outcome'] for r in no)
            d = len(no)-c
            numerator += a*d/len(group)
            denominator += b*c/len(group)
            strata.append({'case_id': cid, 'mixed': mixed, 'tagged': len(yes), 'untagged': len(no),
                           'tagged_successes': a, 'untagged_successes': c, 'success_rate_difference': delta})
        result.append({'tag': tag, 'tagged': len(present), 'untagged': len(absent),
                       'tagged_success_rate': statistics.mean(r['outcome'] for r in present) if present else None,
                       'untagged_success_rate': statistics.mean(r['outcome'] for r in absent) if absent else None,
                       'within_case_count': len(differences),
                       'within_case_mean_difference': statistics.mean(differences) if differences else None,
                       'within_case_bootstrap_95_interval': bootstrap_case_mean(differences),
                       'mixed_within_case_count': len(mixed_differences),
                       'mixed_within_case_mean_difference': statistics.mean(mixed_differences) if mixed_differences else None,
                       'mixed_within_case_bootstrap_95_interval': bootstrap_case_mean(mixed_differences),
                       'mantel_haenszel_odds_ratio': numerator/denominator if denominator else None,
                       'mh_zero_denominator': denominator == 0, 'case_strata': strata})
    return result


def direct_tag_contrast(rows, first='contradiction_flag', second='agreement_only'):
    """Compare the two named groups directly, rather than using their complements."""
    first_rows=[r for r in rows if first in r['tags']]
    second_rows=[r for r in rows if second in r['tags']]
    cases={}
    for r in rows:cases.setdefault(r['case_id'],[]).append(r)
    differences=[]
    for group in cases.values():
        a=[r for r in group if first in r['tags']]
        b=[r for r in group if second in r['tags']]
        if a and b and {r['outcome'] for r in group}=={0,1}:
            differences.append(statistics.mean(r['outcome'] for r in a)-statistics.mean(r['outcome'] for r in b))
    return {'first':first,'second':second,'first_count':len(first_rows),'second_count':len(second_rows),
            'first_success_rate':statistics.mean(r['outcome'] for r in first_rows) if first_rows else None,
            'second_success_rate':statistics.mean(r['outcome'] for r in second_rows) if second_rows else None,
            'mixed_comparable_cases':len(differences),
            'mixed_case_mean_difference':statistics.mean(differences) if differences else None,
            'mixed_case_bootstrap_95_interval':bootstrap_case_mean(differences)}
