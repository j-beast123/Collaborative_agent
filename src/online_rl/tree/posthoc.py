"""After a trajectory's FINAL and outcome: Doctor post-hoc feedback, the frozen post-hoc rubric input, and the ONE
rubric per trajectory. Neither the Doctor feedback nor the Rubric Generator receives the gold answer."""
from __future__ import annotations
import re
import json
from typing import Callable, Optional
from src.online_rl.integrated.records import sha256_json
from src.online_rl.tree.cache import CallCache
from src.online_rl.tree.records import DoctorFeedback, Leaf, TrajectoryRubric, TrajectoryRubricInput, to_dialogue

LEVELS = ('0', '1', '2')
TRANSCRIPT_FORMAT = 'agent_reply_node_ids_v2'           # how feedback / generator transcripts are rendered; part of their cache keys
RUBRIC_KEYS = ('name', 'applicability', 'scores')
CONTEXT_KEYS = ('user_side_context', 'agent_side_state')


def outcome_label(outcome: float) -> str:
    if outcome not in (0.0, 1.0):
        raise ValueError(f'Expected a valid binary outcome, got {outcome!r}')
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


def freeze_trajectory_input(case: dict, leaf: Leaf, feedback: DoctorFeedback, nodes: dict | None = None) -> TrajectoryRubricInput:
    if leaf.outcome is None or feedback.leaf_id != leaf.leaf_id: raise ValueError(f'{leaf.leaf_id}: outcome and matching feedback are required')
    if not leaf.dialogue or leaf.dialogue[-1][1] != 'FINAL': raise ValueError(f'{leaf.leaf_id}: trajectory must end with the Doctor FINAL')
    if case['case_id'] != leaf.case_id:
        raise ValueError('Case and leaf case_id differ')
    if leaf.dialogue[-1][0] != 'doctor':
        raise ValueError('FINAL must be a Doctor turn')
    n_agents = sum(t[0] == 'agent' for t in leaf.dialogue)
    if n_agents != len(leaf.path) or len(set(leaf.path)) != len(leaf.path):
        raise ValueError('Leaf.path must identify each Agent reply once, in dialogue order')
    if not leaf.path or leaf.path[-1] != leaf.leaf_id:
        raise ValueError('Leaf.path must end at the terminal leaf node')
    v = case_view(case)
    payload = {'leaf_id': leaf.leaf_id, 'case_id': v['case_id'], 'scenario': v['scenario'], 'question': v['question'],
               'options': sorted(v['options'].items()), 'dialogue': [list(t) for t in leaf.dialogue], 'outcome': outcome_label(leaf.outcome),
               'feedback': feedback.text, 'agent_node_ids': list(leaf.path)}
    states = tuple({'agent_turn_id': nid, **nodes[nid].trace['pre_response_belief']} for nid in leaf.path if nodes and nodes[nid].trace.get('pre_response_belief'))
    if states:
        payload['agent_pre_response_states'] = states
    return TrajectoryRubricInput(leaf.leaf_id, v['case_id'], v['scenario'], v['question'], tuple(tuple(o) for o in payload['options']),
                                 leaf.dialogue, payload['outcome'], feedback.text, sha256_json(payload), tuple(leaf.path), agent_pre_response_states=states)


def rubric_problems(rubric: Optional[dict]) -> list[str]:
    """Validate the fixed user-side / Agent-side / behavior structure."""
    if not isinstance(rubric, dict): return ['no rubric object']
    candidate = rubric.get('candidate_rubric', rubric)
    if not isinstance(candidate, dict): return ['candidate_rubric must be an object']
    out = [f'missing {k}' for k in RUBRIC_KEYS if not candidate.get(k)]
    if not isinstance(candidate.get('name'), str) or not candidate.get('name', '').strip():
        out.append('name must be nonempty text')
    app = candidate.get('applicability')
    if not isinstance(app, dict):
        out.append('applicability must contain user_side_context and agent_side_state')
        app = {}
    for k in CONTEXT_KEYS:
        if not isinstance(app.get(k), str) or not app.get(k, '').strip():
            out.append(f'applicability.{k} must be nonempty text')
    scores = candidate.get('scores')
    if not isinstance(scores, dict):
        return out + ['scores must be an object']
    if set(scores) != set(LEVELS): out.append('scores must define string levels 0, 1 and 2')
    for level in LEVELS:
        score = scores.get(level)
        if not isinstance(score, dict):
            out.append(f'score {level} must contain contexts and behavior')
            continue
        for k in CONTEXT_KEYS:
            if score.get(k) != app.get(k) or k not in app:
                out.append(f'score {level}.{k} must equal applicability.{k}')
        if not isinstance(score.get('behavior'), str) or not score.get('behavior', '').strip():
            out.append(f'score {level}.behavior must be nonempty text')
    return out


def selected_agent_node_id(rubric: dict, tin: TrajectoryRubricInput) -> tuple[Optional[str], list[str]]:
    """Only exact node IDs printed in this trajectory are accepted; no numeric guessing."""
    n = sum(t[0] == 'agent' for t in tin.dialogue)
    ids = tin.agent_node_ids
    if len(ids) != n or not ids or len(set(ids)) != len(ids) or ids[-1] != tin.leaf_id:
        return None, ['invalid or missing input Agent-node mapping']
    analysis = rubric.get('analysis')
    if not isinstance(analysis, dict):
        return None, ['missing analysis.selected_agent_turn_id']
    selected = analysis.get('selected_agent_turn_id')
    if not isinstance(selected, str) or not selected:
        return None, ['selected_agent_turn_id must be an exact Agent node ID string']
    if selected not in ids:
        return None, ['selected_agent_turn_id is not an Agent turn of this trajectory']
    return selected, []


def _generation_payload(g: dict) -> Optional[dict]:
    """Keep the full generator envelope, including analysis, before downstream extraction."""
    value = g.get('rubric')
    if isinstance(value, dict) and 'analysis' in value:
        return value
    raw = g.get('raw')
    if isinstance(raw, str):
        try:
            decoded = json.loads(raw)
        except (ValueError, TypeError):
            decoded = None
        if isinstance(decoded, dict) and 'analysis' in decoded:
            return decoded
    return value if isinstance(value, dict) else None


def generate_rubric(prompt_text: str, prompt_sha256: str, tin: TrajectoryRubricInput, generate_fn: Callable,
                    cache: CallCache, gate_fn: Optional[Callable] = None) -> TrajectoryRubric:
    """One generator call. Store canonical selected node ID independently of the behavior rubric.
    gate_fn receives the rubric object as stored here (including candidate_rubric for an envelope).
    """
    rid = f'{prompt_sha256[:12]}-{tin.sha256[:12]}'
    g = cache.call('rubric_generation', {'tin': tin.sha256, 'prompt': prompt_sha256, 'transcript': TRANSCRIPT_FORMAT},
                   lambda: generate_fn(prompt_text, tin))
    if not isinstance(g, dict):
        return TrajectoryRubric(rid, tin.leaf_id, prompt_sha256, None, None, False,
                                ('generator result must be an object',), None, selected_agent_turn_id=None)
    rubric = _generation_payload(g)
    problems = rubric_problems(rubric)
    selected, selection_problems = selected_agent_node_id(rubric, tin) if rubric is not None else (None, [])
    problems.extend(selection_problems)
    if rubric is not None:
        rubric = {**rubric, 'rubric_id': rid}
        expected = {'source_case_id': tin.case_id, 'source_trajectory_id': tin.leaf_id, 'source_outcome': tin.outcome}
        for field, expected_value in expected.items():
            if field in rubric and rubric[field] != expected_value:
                problems.append(f'{field} disagrees with source input')
    gate = cache.call('posthoc_gate', {'tin': tin.sha256, 'rubric': sha256_json(rubric), 'contract': TRANSCRIPT_FORMAT},
                      lambda: gate_fn(tin, rubric)) if (gate_fn and not problems) else None
    valid = not problems and (gate is None or bool(gate.get('passed', False)))
    return TrajectoryRubric(rid, tin.leaf_id, prompt_sha256, g.get('raw'), rubric, valid,
                            tuple(problems), gate, selected_agent_turn_id=selected)


def trajectory_message(tin: TrajectoryRubricInput) -> str:
    """Generator user message: case, full trajectory with the FINAL, outcome and Doctor feedback (no gold answer)."""
    from src.collaboration.generator import case_text
    view = {'scenario': tin.scenario, 'question': tin.question, 'options': dict(tin.options)}
    transcript, _ = labeled_dialogue(to_dialogue(tin.dialogue), tin.agent_node_ids)                  # same Agent reply labels as the Doctor feedback
    return (f'{case_text(view)}\n\nCompleted discussion, including the physician\'s final decision:\n{transcript}\n\n'
            f'Outcome: the final decision was a {tin.outcome}.\n\nPhysician feedback after the decision:\n{tin.feedback}\n\n'
            f'Source case ID: {tin.case_id}\nSource trajectory ID: {tin.leaf_id}\n'
            'The caller assigns the canonical rubric_id after generation; do not infer it from case content.\n'
            'Write ONE behavioral rubric. In analysis.selected_agent_turn_id, copy exactly one Agent turn_id '
            'printed above. Do not return the reply ordinal, a Doctor ID, or a fabricated ID.'
            + ('\n\nRecorded prefix-only Agent reconstruction and judgment for each turn:\n' + json.dumps(tin.agent_pre_response_states, ensure_ascii=False) if tin.agent_pre_response_states else ''))


ORDINALS = {w: i for i, w in enumerate(('first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh', 'eighth', 'ninth', 'tenth'), 1)}


def labeled_dialogue(dialogue: list[dict], agent_node_ids: Optional[tuple[str, ...]] = None) -> tuple[str, int]:
    """Transcript with stable labels: 'Agent reply k' for the k-th Agent turn, so the Doctor never counts turns itself."""
    if agent_node_ids is not None:
        count = sum(d['speaker'] == 'agent' for d in dialogue)
        if len(agent_node_ids) != count or len(set(agent_node_ids)) != count:
            raise ValueError('Agent IDs must map one-to-one to transcript replies')
    out, k = [], 0
    for d in dialogue:
        if d['speaker'] == 'agent':
            k += 1
            label = f' [turn_id={agent_node_ids[k - 1]}]' if agent_node_ids is not None else ''
            out.append(f"Agent reply {k}{label}: {d['text']}")
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
