"""After a trajectory's FINAL and outcome: Doctor post-hoc feedback, the frozen post-hoc rubric input, and the ONE
rubric per trajectory. Neither the Doctor feedback nor the Rubric Generator receives the gold answer."""
from __future__ import annotations
import re
from typing import Callable, Optional
from src.online_rl.integrated.records import sha256_json
from src.online_rl.tree.cache import CallCache
from src.online_rl.tree.records import DoctorFeedback, Leaf, TrajectoryRubric, TrajectoryRubricInput, to_dialogue

LEVELS = ('0', '1', '2')
TRANSCRIPT_FORMAT = 'agent_reply_labels_v1'           # how feedback / generator transcripts are rendered; part of their cache keys
RUBRIC_KEYS = ('name', 'applicability_condition', 'required_behavior', 'scores')


def outcome_label(outcome: float) -> str:
    return 'success' if outcome == 1.0 else 'failure'


def case_view(case: dict) -> dict:
    """The case as the Doctor and the Rubric Generator see it: no gold answer."""
    return {'case_id': case['case_id'], 'scenario': case.get('scenario') or '', 'question': case['question'], 'options': dict(case['options'])}


def doctor_feedback(case: dict, leaf: Leaf, feedback_fn: Callable, prompt_sha256: str, cache: CallCache, model: str | None = None) -> DoctorFeedback:
    """feedback_fn(case_view, dialogue_with_final, outcome_label) -> text; runs only after the outcome is known. `model` (the
    Feedback Generator model id) is part of the cache key, so feedback from another model is never reused."""
    if leaf.outcome is None: raise ValueError(f'{leaf.leaf_id}: no outcome, so no post-hoc feedback')
    model = model if model is not None else getattr(feedback_fn, 'model', None)
    label = outcome_label(leaf.outcome)
    value = cache.call('feedback', {'leaf': leaf.leaf_id, 'dialogue': sha256_json(leaf.dialogue), 'outcome': label, 'prompt': prompt_sha256,
                                    'transcript': getattr(feedback_fn, 'transcript', TRANSCRIPT_FORMAT), 'model': model},
                       lambda: feedback_fn(case_view(case), to_dialogue(leaf.dialogue), label))
    if isinstance(value, dict):                                     # feedback v3 / v4: text without the locus line + the behavioral locus
        return DoctorFeedback(leaf.leaf_id, leaf.outcome, value['text'], prompt_sha256, value.get('target_reply_id'),
                              {k: value.get(k) for k in ('status', 'raw', 'leakage')}, bool(value.get('valid', True)),
                              value.get('kind', 'physician_after_decision'))
    return DoctorFeedback(leaf.leaf_id, leaf.outcome, value, prompt_sha256)


BEHAVIOR_TRANSCRIPT = 'behavior_dialogue_v1'           # v4 feedback input: labelled Doctor / Agent turns, no FINAL, no case text


def behavior_message(dialogue: list[dict]) -> tuple[str, int]:
    """v4 feedback input: only the labelled dialogue before the FINAL (no question, options, final decision or outcome)."""
    turns = [d for d in dialogue if d.get('stage') != 'FINAL']
    out, k = [], 0
    for d in turns:
        if d['speaker'] == 'agent': k += 1; out.append(f"Agent reply {k}: {d['text']}")
        else: out.append(f"Physician: {d['text']}")
    return (f'The discussion. The Agent gave {k} replies, labelled Agent reply 1 to Agent reply {k}:\n' + '\n\n'.join(out) +
            '\n\nGive your feedback on the Agent\'s collaborative behavior.'), k


GENERIC_OPTIONS = {'all of the above', 'none of the above', 'both', 'none', 'all', 'neither', 'both a and b', 'all of these', 'none of these'}
OPTION_LABEL = re.compile(r'\b(?:option|choice|answer)\s*\(?[A-E]\)?(?![\w])|\([A-E]\)|(?<![\w])[A-E]\)', re.I)
CORRECTNESS = re.compile(r'\b(?:correct|correctly|incorrect|incorrectly|accurate|accurately|inaccurate|inaccurately|mistaken|mistakes?|'
                         r'misconceptions?|erroneous|erroneously|wrong|wrongly|gold)\b|\bright (?:answer|option|choice|diagnosis)\b|'
                         r'\b(?:the )?answer (?:is|was)\b|\bdiagnosis (?:is|was)\b', re.I)


def leakage_problems(text: str, view: dict) -> list[str]:
    """Deterministic answer-leakage guard for v4 feedback: option labels, option texts, correctness wording."""
    out = [f'option label "{m.group(0)}"' for m in OPTION_LABEL.finditer(text)]
    low = re.sub(r'\s+', ' ', text.lower())
    for k, t in sorted((view.get('options') or {}).items()):
        t = re.sub(r'\s+', ' ', str(t).lower()).strip(' .')
        if len(t) >= 4 and t not in GENERIC_OPTIONS and re.search(r'(?<![\w])' + re.escape(t) + r'(?![\w])', low): out.append(f'option text "{t}"')
    out += [f'correctness wording "{m.group(0)}"' for m in CORRECTNESS.finditer(text)]
    return sorted(set(out))


LOCUS_LINE = re.compile(r'^\s*\**\s*behavio(?:u)?ral locus\s*\**\s*:\s*\**\s*agent reply\s+(\d+)\s*\**\s*\.?\s*$', re.I | re.M)


def split_locus(text: str, n_replies: int) -> tuple[str, Optional[int], Optional[str]]:
    """(feedback text without the locus line, locus reply id or None, problem or None). Exactly one locus line, last, in range."""
    found = list(LOCUS_LINE.finditer(text))
    if not found: return text.strip(), None, 'missing'
    if len(found) > 1: return text.strip(), None, 'multiple'
    m = found[0]; n = int(m.group(1))
    if text[m.end():].strip(): return text.strip(), None, 'not_last'
    clean = text[:m.start()].rstrip()
    if not 1 <= n <= n_replies: return clean, None, 'out_of_range'
    return clean, n, None


def freeze_trajectory_input(case: dict, leaf: Leaf, feedback: DoctorFeedback) -> TrajectoryRubricInput:
    if leaf.outcome is None or feedback.leaf_id != leaf.leaf_id: raise ValueError(f'{leaf.leaf_id}: outcome and matching feedback are required')
    if not leaf.dialogue or leaf.dialogue[-1][1] != 'FINAL': raise ValueError(f'{leaf.leaf_id}: trajectory must end with the Doctor FINAL')
    v = case_view(case); kind = getattr(feedback, 'kind', 'physician_after_decision')
    payload = {'leaf_id': leaf.leaf_id, 'case_id': v['case_id'], 'scenario': v['scenario'], 'question': v['question'],
               'options': sorted(v['options'].items()), 'dialogue': [list(t) for t in leaf.dialogue], 'outcome': outcome_label(leaf.outcome),
               'feedback': feedback.text, **({'feedback_kind': kind} if kind != 'physician_after_decision' else {})}
    return TrajectoryRubricInput(leaf.leaf_id, v['case_id'], v['scenario'], v['question'], tuple(tuple(o) for o in payload['options']),
                                 leaf.dialogue, payload['outcome'], feedback.text, sha256_json(payload), kind)


def rubric_problems(rubric: Optional[dict]) -> list[str]:
    if not isinstance(rubric, dict): return ['no rubric object']
    out = [f'missing {k}' for k in RUBRIC_KEYS if not rubric.get(k)]
    scores = rubric.get('scores')
    if isinstance(scores, dict) and set(map(str, scores)) != set(LEVELS): out.append('scores must define levels 0, 1 and 2')
    return out


def generate_rubric(prompt_text: str, prompt_sha256: str, tin: TrajectoryRubricInput, generate_fn: Callable,
                    cache: CallCache, gate_fn: Optional[Callable] = None) -> TrajectoryRubric:
    """One Rubric-Generator call per trajectory: generate_fn(prompt_text, tin) -> {'raw', 'rubric'} gives the ONE rubric r_tau.
    gate_fn(tin, rubric) -> {'passed', ...} runs on a structurally complete rubric; it is valid iff structure and gate pass."""
    gid = {'tin': tin.sha256, 'prompt': prompt_sha256, 'transcript': TRANSCRIPT_FORMAT, 'model': getattr(generate_fn, 'model', None)}
    if getattr(generate_fn, 'settings', None) is not None: gid['settings'] = generate_fn.settings     # reasoning / token budget / output format
    g = cache.call('rubric_generation', gid,
                   lambda: generate_fn(prompt_text, tin))
    rubric = g.get('rubric'); problems = rubric_problems(rubric)
    rid = f'{prompt_sha256[:12]}-{tin.sha256[:12]}'
    if rubric is not None: rubric = {**rubric, 'rubric_id': rid, 'scores': {str(a): b for a, b in (rubric.get('scores') or {}).items()}}
    gate = cache.call('posthoc_gate', {'tin': tin.sha256, 'rubric': sha256_json(rubric), 'model': getattr(gate_fn, 'model', None),
                                       'protocol': getattr(gate_fn, 'protocol', None)},
                      lambda: gate_fn(tin, rubric)) if (gate_fn and not problems) else None
    valid = not problems and (gate is None or bool(gate['passed']))
    return TrajectoryRubric(rid, tin.leaf_id, prompt_sha256, g.get('raw'), rubric, valid, tuple(problems), gate)


def trajectory_message(tin: TrajectoryRubricInput) -> str:
    """Generator user message: case, full trajectory with the FINAL, outcome and Doctor feedback (no gold answer)."""
    from src.collaboration.generator import case_text
    view = {'scenario': tin.scenario, 'question': tin.question, 'options': dict(tin.options)}
    transcript, _ = labeled_dialogue(to_dialogue(tin.dialogue))                  # same Agent reply labels as the Doctor feedback
    return (f'{case_text(view)}\n\nCompleted discussion, including the physician\'s final decision:\n{transcript}\n\n'
            f'Outcome: the final decision was a {tin.outcome}.\n\n{FEEDBACK_HEADERS[getattr(tin, "feedback_kind", "physician_after_decision")]}\n{tin.feedback}\n\n'
            'Write ONE behavioral rubric for the Agent\'s replies in this discussion.')


FEEDBACK_HEADERS = {'physician_after_decision': 'Physician feedback after the decision:',
                    'behavior_observer_v4': 'Observer feedback on the Agent\'s collaborative behavior (written from the dialogue only, without the question, options, final decision or outcome):'}

ORDINALS = {w: i for i, w in enumerate(('first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh', 'eighth', 'ninth', 'tenth'), 1)}


def labeled_dialogue(dialogue: list[dict]) -> tuple[str, int]:
    """Transcript with stable labels: 'Agent reply k' for the k-th Agent turn, so the Doctor never counts turns itself."""
    out, k = [], 0
    for d in dialogue:
        if d['speaker'] == 'agent': k += 1; out.append(f"Agent reply {k}: {d['text']}")
        else: out.append(f"Doctor{' (final decision)' if d.get('stage') == 'FINAL' else ''}: {d['text']}")
    return '\n\n'.join(out), k


def feedback_label_problems(text: str, n_replies: int) -> list[str]:
    """Agent-reply references in feedback that do not exist in the transcript (label numbers or ordinal words)."""
    nums = [int(x) for x in re.findall(r'Agent reply (\d+)', text, re.I)]
    nums += [ORDINALS[w.lower()] for w in re.findall(r'\b(' + '|'.join(ORDINALS) + r')\s+(?:agent\s+)?(?:reply|response|turn)\b', text, re.I)]
    return sorted({f'reply {n}' for n in nums if not 1 <= n <= n_replies})


def feedback_message(view: dict, dialogue: list[dict], outcome: str) -> str:
    from src.collaboration.generator import case_text
    transcript, n = labeled_dialogue(dialogue)
    return (f'{case_text(view)}\n\nThe discussion you just had with the Agent, including your final decision. The Agent gave {n} '
            f'replies, labelled Agent reply 1 to Agent reply {n}:\n{transcript}\n\n'
            f'Your final decision turned out to be a {outcome}. Give your feedback on the Agent\'s behavior.')
