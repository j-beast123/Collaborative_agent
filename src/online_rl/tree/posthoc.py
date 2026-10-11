"""After a trajectory's FINAL and outcome: Doctor post-hoc feedback, the frozen post-hoc rubric input, and the ONE
rubric per trajectory. Neither the Doctor feedback nor the Rubric Generator receives the gold answer."""
from __future__ import annotations
import json, re
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
    identity = {'leaf': leaf.leaf_id, 'dialogue': sha256_json(leaf.dialogue), 'outcome': label, 'prompt': prompt_sha256,
                'transcript': getattr(feedback_fn, 'transcript', TRANSCRIPT_FORMAT), 'model': model}
    if getattr(feedback_fn, 'protocol', None) is not None:
        identity.update(protocol=feedback_fn.protocol, settings=getattr(feedback_fn, 'settings', None))
    value = cache.call('feedback', identity,
                       lambda: feedback_fn(case_view(case), to_dialogue(leaf.dialogue), label))
    if isinstance(value, dict):                                     # feedback v3 / v4: text without the locus line + the behavioral locus
        return DoctorFeedback(leaf.leaf_id, leaf.outcome, value['text'], prompt_sha256, value.get('target_reply_id'),
                              {k: v for k, v in value.items() if k not in ('text', 'target_reply_id', 'valid', 'kind')}, bool(value.get('valid', True)),
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


def freeze_trajectory_input(case: dict, leaf: Leaf, feedback: DoctorFeedback, include_locus: bool = False) -> TrajectoryRubricInput:
    if leaf.outcome is None or feedback.leaf_id != leaf.leaf_id: raise ValueError(f'{leaf.leaf_id}: outcome and matching feedback are required')
    if not leaf.dialogue or leaf.dialogue[-1][1] != 'FINAL': raise ValueError(f'{leaf.leaf_id}: trajectory must end with the Doctor FINAL')
    v = case_view(case); kind = getattr(feedback, 'kind', 'physician_after_decision')
    payload = {'leaf_id': leaf.leaf_id, 'case_id': v['case_id'], 'scenario': v['scenario'], 'question': v['question'],
               'options': sorted(v['options'].items()), 'dialogue': [list(t) for t in leaf.dialogue], 'outcome': outcome_label(leaf.outcome),
               'feedback': feedback.text, **({'feedback_kind': kind} if kind != 'physician_after_decision' else {})}
    locus = feedback.target_reply_id if include_locus else None                  # v2.1: the generator sees the behavioral locus
    if include_locus: payload['target_reply_id'] = locus
    return TrajectoryRubricInput(leaf.leaf_id, v['case_id'], v['scenario'], v['question'], tuple(tuple(o) for o in payload['options']),
                                 leaf.dialogue, payload['outcome'], feedback.text, sha256_json(payload), kind, locus)


def rubric_problems(rubric: Optional[dict]) -> list[str]:
    if not isinstance(rubric, dict): return ['no rubric object']
    if 'applicability' in rubric:
        from src.online_rl.tree.agent_state import rubric_problems as validate
        return validate(rubric)
    out = [f'missing {k}' for k in RUBRIC_KEYS if not rubric.get(k)]
    scores = rubric.get('scores')
    if isinstance(scores, dict) and set(map(str, scores)) != set(LEVELS): out.append('scores must define levels 0, 1 and 2')
    return out


def generate_rubric(prompt_text: str, prompt_sha256: str, tin: TrajectoryRubricInput, generate_fn: Callable,
                    cache: CallCache, gate_fn: Optional[Callable] = None, source_stage1: Optional[Callable] = None) -> TrajectoryRubric:
    """One Rubric-Generator call per trajectory: generate_fn(prompt_text, tin) -> {'raw', 'rubric'} gives the ONE rubric r_tau.
    gate_fn(tin, rubric) -> {'passed', ...} runs on a structurally complete rubric; it is valid iff structure and gate pass."""
    gid = {'tin': tin.sha256, 'prompt': prompt_sha256, 'transcript': TRANSCRIPT_FORMAT, 'model': getattr(generate_fn, 'model', None)}
    if getattr(generate_fn, 'settings', None) is not None: gid['settings'] = generate_fn.settings     # reasoning / token budget / output format
    g = cache.call('rubric_generation', gid,
                   lambda: generate_fn(prompt_text, tin))
    rubric = g.get('rubric'); problems = rubric_problems(rubric)
    rid = f'{prompt_sha256[:12]}-{tin.sha256[:12]}'
    if rubric is not None: rubric = {**rubric, 'rubric_id': rid, 'scores': {str(a): b for a, b in (rubric.get('scores') or {}).items()}}
    instruction = g.get('intervention_instruction')
    source_check, raw = None, g.get('raw')
    # A structurally valid generator response can honestly report that its selected
    # behavior does not satisfy the source anchor (failure -> 0, success -> 2).
    # This is a normal abstention, not a malformed response and not a Gate reject.
    # Stop here so it cannot consume Gate/rerun calls or enter the active bank.
    provenance = (rubric or {}).get('source_provenance') or {}
    declaration = provenance.get('source_anchor_declaration') or {}
    mismatch = provenance.get('source_anchor_contract_mismatch') is True or declaration.get('matches_expected') is False
    if not problems and mismatch:
        expected = declaration.get('expected_score_from_outcome')
        claimed = declaration.get('generator_claimed_source_score')
        source_check = {'status': 'NO_VALID_CANDIDATE', 'reason': 'SOURCE_ANCHOR_MISMATCH',
                        'expected_source_level': expected, 'observed_source_level': claimed}
        problem = f'no_valid_candidate: source_score={claimed} expected={expected}'
        return TrajectoryRubric(rid, tin.leaf_id, prompt_sha256, raw, rubric, False,
                                (problem,), None, source_check, None)
    if source_stage1 is not None and not problems:                 # v2.1: source-self-applicability before the quality gate
        source_check, rubric, raw, problems = _source_consistency(prompt_text, prompt_sha256, tin, generate_fn, cache, source_stage1, rubric, raw, rid)
        rid = rubric['rubric_id'] if rubric else rid
        if problems: return TrajectoryRubric(rid, tin.leaf_id, prompt_sha256, raw, rubric, False, tuple(problems), None, source_check, instruction)
    gate_identity = {'tin': tin.sha256, 'rubric': sha256_json(rubric), 'model': getattr(gate_fn, 'model', None),
                     'protocol': getattr(gate_fn, 'protocol', None)}
    if getattr(gate_fn, 'settings', None) is not None:
        gate_identity['settings'] = gate_fn.settings
    gate = cache.call('posthoc_gate', gate_identity,
                      lambda: gate_fn(tin, rubric)) if (gate_fn and not problems) else None
    valid = not problems and (gate is None or bool(gate['passed']))
    return TrajectoryRubric(rid, tin.leaf_id, prompt_sha256, raw, rubric, valid, tuple(problems), gate, source_check, instruction)


REPAIR_PRINCIPLE = ('Keep the same behavioral concept and source feedback. Rewrite only what is necessary so that the rubric is genuinely '
                    'applicable at the provided source behavioral locus. Do not broaden the rubric merely to force applicability, and do not '
                    'introduce answer-specific or medical-ground-truth assumptions. Change ONLY the applicability condition: keep the name, the '
                    'required behavior and the 0/1/2 level descriptions exactly as they are.')


def _source_consistency(prompt_text, prompt_sha256, tin, generate_fn, cache, source_stage1, rubric, raw, rid):
    """Admission consistency check only (never causal evidence; KEEP still needs the counterfactual witness). Stage-1
    applicability of the new rubric at its own source locus. Applicable -> PASS. No Match -> ONE repair (same locus, same
    feedback; generate_fn.repair) of which ONLY the applicability condition is taken: name, required behavior, 0/1/2 levels
    and rubric id stay those of the generated rubric; a second Stage-1 check -> REPAIRED_PASS or SOURCE_INCONSISTENT.
    No locus -> NO_SOURCE_LOCUS. The locus is never changed."""
    # A judge contract failure is neither non-applicability nor score zero.  At
    # generation admission it quarantines this candidate so one malformed or
    # uncertain LLM result cannot terminate the whole training batch.
    from src.online_rl.tree.applicability import ApplicabilityUncertain, ScoringContractError
    try:
        first = source_stage1(rubric)
    except (ApplicabilityUncertain, ScoringContractError) as error:
        return ({'status':'SOURCE_APPLICABILITY_ERROR','error_type':type(error).__name__}, rubric, raw,
                [f'source_applicability_error: {type(error).__name__}'])
    if first is None: return {'status': 'NO_SOURCE_LOCUS', 'locus': None}, rubric, raw, ['source_inconsistent: no behavioral locus']
    check = {'locus': first.get('target_reply_id', tin.target_reply_id), 'node_id': first.get('node_id'), 'first': first}
    if first['applicable']: return {**check, 'status': 'PASS'}, rubric, raw, []
    if (rubric.get('source_provenance') or {}).get('selected_turn_policy') == 'generator_v1':
        # Jointly generated instructions are fixed experimental inputs. A separate
        # condition-only rewrite would no longer validate that same contract.
        return {**check, 'status': 'SOURCE_INCONSISTENT'}, rubric, raw, ['source_inconsistent: joint rubric is not applicable at its selected turn']
    rep = cache.call('rubric_repair', {'tin': tin.sha256, 'prompt': prompt_sha256, 'rubric': sha256_json(rubric), 'model': getattr(generate_fn, 'model', None),
                                       'settings': getattr(generate_fn, 'settings', None), 'principle': sha256_json(REPAIR_PRINCIPLE)},
                     lambda: generate_fn.repair(prompt_text, tin, rubric, first.get('stage1') or {}))
    if 'applicability' in rubric:
        cond = (rep.get('rubric') or {}).get('applicability')
        if not isinstance(cond, dict) or set(cond) != {'user_side_context', 'agent_side_state'} or not all(isinstance(v, str) and v.strip() for v in cond.values()):
            return {**check, 'status': 'SOURCE_INCONSISTENT', 'original_rubric': rubric}, rubric, raw, ['source_inconsistent: repair output invalid']
        fixed = {**rubric, 'applicability': cond, 'scores': {k: {**cond, 'behavior': v['behavior']} for k, v in rubric['scores'].items()}, 'source_repair': 1}
        second = source_stage1(fixed); check = {**check, 'second': second, 'original_rubric': rubric}
        if second['applicable']: return {**check, 'status': 'REPAIRED_PASS'}, fixed, rep.get('raw'), []
        return {**check, 'status': 'SOURCE_INCONSISTENT', 'repaired_rubric': fixed}, fixed, rep.get('raw'), ['source_inconsistent: No Match at the source locus after one repair']
    cond = (rep.get('rubric') or {}).get('applicability_condition')
    if not isinstance(cond, str) or not cond.strip():
        return {**check, 'status': 'SOURCE_INCONSISTENT', 'repair_problems': ['no applicability condition'], 'original_rubric': rubric}, rubric, raw, \
               ['source_inconsistent: repair output invalid']
    fixed = {**rubric, 'applicability_condition': cond.strip(), 'source_repair': 1}       # condition only; same id, concept and levels
    second = source_stage1(fixed); check = {**check, 'second': second, 'original_rubric': rubric}
    if second['applicable']: return {**check, 'status': 'REPAIRED_PASS'}, fixed, rep.get('raw'), []
    return {**check, 'status': 'SOURCE_INCONSISTENT', 'repaired_rubric': fixed}, fixed, rep.get('raw'), ['source_inconsistent: No Match at the source locus after one repair']


def trajectory_message(tin: TrajectoryRubricInput) -> str:
    """Generator user message: case, full trajectory with the FINAL, outcome and Doctor feedback (no gold answer)."""
    from src.collaboration.generator import case_text
    view = {'scenario': tin.scenario, 'question': tin.question, 'options': dict(tin.options)}
    # Expose stable source identifiers required by the trajectory-analysis P0. Agent reply IDs
    # are the same 1-based IDs used by feedback.target_reply_id; Doctor turn IDs are 1-based
    # within this transcript. Speaker is part of each ID's identity.
    transcript_lines, agent_id, doctor_id = [], 0, 0
    for turn in to_dialogue(tin.dialogue):
        if turn['speaker'] == 'agent':
            agent_id += 1
            transcript_lines.append(f"Agent reply {agent_id} [turn_id={agent_id}]: {turn['text']}")
        else:
            doctor_id += 1
            stage = ' (final decision)' if turn.get('stage') == 'FINAL' else ''
            transcript_lines.append(f"Doctor turn {doctor_id} [turn_id={doctor_id}]{stage}: {turn['text']}")
    transcript = '\n\n'.join(transcript_lines)
    return (f'Source case ID (JSON value; preserve type): {json.dumps(tin.case_id, ensure_ascii=False)}\n'
            f'Source trajectory ID (JSON value; preserve type): {json.dumps(tin.leaf_id, ensure_ascii=False)}\n\n'
            f'{case_text(view)}\n\nCompleted discussion, including the Doctor FINAL:\n{transcript}\n\n'
            f'Outcome: the final decision was a {tin.outcome}.\n\n{FEEDBACK_HEADERS[getattr(tin, "feedback_kind", "physician_after_decision")]}\n{tin.feedback}\n\n'
            + locus_section(tin) + 'Write ONE behavioral rubric for the Agent\'s replies in this discussion.')


LOCUS_PRINCIPLE = ('The generated rubric must describe the behavioral reason identified in the feedback, and its applicability condition must be '
                   'satisfied at the specified behavioral locus in the source trajectory. Do not define an applicability condition that only holds at '
                   'a different turn or earlier/later conversational state. The rubric stays answer-neutral and behavior-level: it must not presuppose '
                   'the gold answer, the outcome, or which option is correct.')


def locus_section(tin: TrajectoryRubricInput) -> str:
    """v2.1 (tin.target_reply_id set): the behavioral locus, the conversation before it and its reply text, plus the consistency
    principle. Empty for earlier protocols, so their generator input is unchanged."""
    n = getattr(tin, 'target_reply_id', None)
    if n is None: return ''
    turns = to_dialogue(tin.dialogue); k = 0; before = []; reply = None
    for d in turns:
        if d['speaker'] == 'agent':
            k += 1
            if k == n: reply = d['text']; break
        before.append(d)
    if reply is None: raise ValueError(f'{tin.leaf_id}: behavioral locus Agent reply {n} is not in the trajectory')
    context, _ = labeled_dialogue(before)
    return (f'Behavioral locus: Agent reply {n}\n\nConversation before the locus reply:\n{context}\n\n'
            f'Locus reply (Agent reply {n}):\n{reply}\n\n{LOCUS_PRINCIPLE}\n\n')


FEEDBACK_HEADERS = {'physician_after_decision': 'Physician feedback after the decision:',
                    'physician_post_final_reflection_v1': 'The participating Doctor\'s own post-FINAL reflection (the same pinned Doctor model, without ground truth or the outcome label). Treat this as reported experience to investigate, not proof of appropriate behavior or correctness:',
                    'behavior_observer_v4': 'Observer feedback on the Agent\'s collaborative behavior (written from the dialogue only, without the question, options, final decision or outcome):',
                    'behavior_observer_v5': 'Behavioral observation linking the physician\'s visible request, the Agent\'s response, and the physician\'s later response or unresolved need (dialogue only; no question, options, final decision or outcome):'}

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
