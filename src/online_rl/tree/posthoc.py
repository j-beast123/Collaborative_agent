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


def doctor_feedback(case: dict, leaf: Leaf, feedback_fn: Callable, prompt_sha256: str, cache: CallCache) -> DoctorFeedback:
    """feedback_fn(case_view, dialogue_with_final, outcome_label) -> text; runs only after the outcome is known."""
    if leaf.outcome is None: raise ValueError(f'{leaf.leaf_id}: no outcome, so no post-hoc feedback')
    label = outcome_label(leaf.outcome)
    text = cache.call('feedback', {'leaf': leaf.leaf_id, 'dialogue': sha256_json(leaf.dialogue), 'outcome': label, 'prompt': prompt_sha256,
                                   'transcript': TRANSCRIPT_FORMAT},
                      lambda: feedback_fn(case_view(case), to_dialogue(leaf.dialogue), label))
    return DoctorFeedback(leaf.leaf_id, leaf.outcome, text, prompt_sha256)


def freeze_trajectory_input(case: dict, leaf: Leaf, feedback: DoctorFeedback) -> TrajectoryRubricInput:
    if leaf.outcome is None or feedback.leaf_id != leaf.leaf_id: raise ValueError(f'{leaf.leaf_id}: outcome and matching feedback are required')
    if not leaf.dialogue or leaf.dialogue[-1][1] != 'FINAL': raise ValueError(f'{leaf.leaf_id}: trajectory must end with the Doctor FINAL')
    v = case_view(case)
    payload = {'leaf_id': leaf.leaf_id, 'case_id': v['case_id'], 'scenario': v['scenario'], 'question': v['question'],
               'options': sorted(v['options'].items()), 'dialogue': [list(t) for t in leaf.dialogue], 'outcome': outcome_label(leaf.outcome),
               'feedback': feedback.text}
    return TrajectoryRubricInput(leaf.leaf_id, v['case_id'], v['scenario'], v['question'], tuple(tuple(o) for o in payload['options']),
                                 leaf.dialogue, payload['outcome'], feedback.text, sha256_json(payload))


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
    g = cache.call('rubric_generation', {'tin': tin.sha256, 'prompt': prompt_sha256, 'transcript': TRANSCRIPT_FORMAT}, lambda: generate_fn(prompt_text, tin))
    rubric = g.get('rubric'); problems = rubric_problems(rubric)
    rid = f'{prompt_sha256[:12]}-{tin.sha256[:12]}'
    if rubric is not None: rubric = {**rubric, 'rubric_id': rid, 'scores': {str(a): b for a, b in (rubric.get('scores') or {}).items()}}
    gate = cache.call('posthoc_gate', {'tin': tin.sha256, 'rubric': sha256_json(rubric)}, lambda: gate_fn(tin, rubric)) if (gate_fn and not problems) else None
    valid = not problems and (gate is None or bool(gate['passed']))
    return TrajectoryRubric(rid, tin.leaf_id, prompt_sha256, g.get('raw'), rubric, valid, tuple(problems), gate)


def trajectory_message(tin: TrajectoryRubricInput) -> str:
    """Generator user message: case, full trajectory with the FINAL, outcome and Doctor feedback (no gold answer)."""
    from src.collaboration.generator import case_text
    view = {'scenario': tin.scenario, 'question': tin.question, 'options': dict(tin.options)}
    transcript, _ = labeled_dialogue(to_dialogue(tin.dialogue))                  # same Agent reply labels as the Doctor feedback
    return (f'{case_text(view)}\n\nCompleted discussion, including the physician\'s final decision:\n{transcript}\n\n'
            f'Outcome: the final decision was a {tin.outcome}.\n\nPhysician feedback after the decision:\n{tin.feedback}\n\n'
            'Write ONE behavioral rubric for the Agent\'s replies in this discussion.')


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
