"""Sampling, checklist contracts, lexical answer guard and tied-rank AUC."""
from __future__ import annotations

import math
import random
import re
import unicodedata

ASSISTANT_BEHAVIOR_INSTRUCTION = (
    "Each rubric must evaluate the assistant's observable behavior. The partner's "
    "state may appear only as the condition (e.g., when the physician is uncertain "
    "between two options, the assistant compares their supporting evidence). "
    "Never score the partner's confidence, decision, agreement or change of mind "
    "as the behavior, or require a particular partner response for success."
)

LEAKAGE_INSTRUCTION = (
    'Never mention or imply a specific diagnosis, medical answer, correct option, '
    'option letter or option text in any rubric. Rubrics must describe collaboration '
    'behaviors only: questions, uncertainty, checking assumptions, resolving objections '
    'and comparing reasoning. State when the behavior is needed and what observable '
    'behavior satisfies it. Do not reward agreement, answer correctness or stating a diagnosis.'
)


def obj(properties):
    return {'type': 'object', 'properties': properties, 'required': list(properties),
            'additionalProperties': False}


TEXT = {'type': 'string', 'minLength': 1}
RUBRIC_SCHEMA = obj({'name': TEXT, 'situation': TEXT, 'behavior': TEXT})
SET_SCHEMA = obj({'rubrics': {'type': 'array', 'minItems': 5, 'maxItems': 6,
                             'items': RUBRIC_SCHEMA}})


def checklist_schema(count):
    return obj({'scores': {'type': 'array', 'minItems': count, 'maxItems': count,
                          'items': {'type': 'integer', 'enum': [0, 1]}}})


def parse_rubrics(raw):
    import json
    result = json.loads(raw)
    if not isinstance(result, dict) or set(result) != {'rubrics'}:
        raise ValueError('Expected exactly one rubrics field')
    rows = result['rubrics']
    if not isinstance(rows, list) or not 5 <= len(rows) <= 6:
        raise ValueError('Expected five or six rubrics')
    for row in rows:
        if not isinstance(row, dict) or set(row) != {'name', 'situation', 'behavior'}:
            raise ValueError('Rubric needs name, situation and behavior only')
        if any(not isinstance(v, str) or not v.strip() for v in row.values()):
            raise ValueError('Rubric fields must be nonempty strings')
    if len({normalize(row['name']) for row in rows}) != len(rows):
        raise ValueError('Rubric names must be distinct')
    return rows


def normalize(text):
    return ' '.join(unicodedata.normalize('NFKC', str(text)).casefold().split())


def leakage_matches(case, rubric):
    """Case-sensitive standalone gold letters; normalized option/diagnosis phrases.

    Single-letter labels are not matched against English articles. Explicit
    option references are case insensitive. The prompt also bans implicit clues.
    """
    text = ' '.join(rubric.values())
    canonical = ' '.join(re.findall(r'\w+', normalize(text)))
    gold = str(case['ground_truth']['correct_option']).strip()
    matches = []
    if gold and (re.search(r'(?<!\w)' + re.escape(gold) + r'(?!\w)', text)
                 or re.search(r'\b(?:option|choice|answer)\s*[:(\-]?\s*'
                              + re.escape(gold) + r'(?!\w)', text, re.I)):
        matches.append({'kind': 'correct_option_label', 'value': gold})
    phrases = list(case['options'].values())
    truth = case.get('ground_truth') or {}
    phrases.extend(truth.get(key) for key in ('correct_text', 'diagnosis') if truth.get(key))
    names = case.get('diagnosis_names', [])
    if isinstance(names, list):
        phrases.extend(names)
    elif isinstance(names, dict):
        phrases.extend(names.values())
    elif isinstance(names, str):
        phrases.append(names)
    for phrase in phrases:
        if not isinstance(phrase, str):
            continue
        needle = ' '.join(re.findall(r'\w+', normalize(phrase)))
        if needle and re.search(r'(?<!\w)' + re.escape(needle) + r'(?!\w)', canonical):
            matches.append({'kind': 'option_or_diagnosis_text', 'value': phrase})
    return matches


def filter_leakage(case, rubrics, *, method='sherpa'):
    if method == 'rar':
        return list(rubrics), []
    kept, removed = [], []
    for rubric in rubrics:
        matches = leakage_matches(case, rubric)
        if matches:
            removed.append({'rubric': rubric, 'matches': matches})
        else:
            kept.append(rubric)
    return kept, removed


def select_reflections(trajectories, seed):
    """Four per outcome, or four total if every outcome is identical."""
    rng = random.Random(seed)
    positives = [row for row in trajectories if row['outcome'] == 1]
    negatives = [row for row in trajectories if row['outcome'] == 0]
    if positives and negatives:
        selected = rng.sample(positives, min(4, len(positives)))
        selected += rng.sample(negatives, min(4, len(negatives)))
    else:
        selected = rng.sample(trajectories, min(4, len(trajectories)))
    rng.shuffle(selected)
    return selected


def auc(outcomes, rewards):
    """Mann–Whitney AUC, giving tied success/failure rewards half credit."""
    if len(outcomes) != len(rewards) or not outcomes:
        raise ValueError('AUC inputs must be nonempty and aligned')
    if any(type(v) not in (int, float) or v not in (0, 1) for v in outcomes):
        raise ValueError('AUC requires binary outcomes')
    if any(not math.isfinite(value) for value in rewards):
        raise ValueError('AUC requires finite rewards')
    good = [r for y, r in zip(outcomes, rewards) if y == 1]
    bad = [r for y, r in zip(outcomes, rewards) if y == 0]
    if not good or not bad:
        return None
    return sum((p > n) + 0.5 * (p == n) for p in good for n in bad) / (len(good) * len(bad))
