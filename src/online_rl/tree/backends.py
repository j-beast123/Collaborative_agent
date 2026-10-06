"""Real components for the case tree (lazy imports: need API credentials / the policy model)."""
from __future__ import annotations


def visible_behavior_rubric(rubric: dict) -> dict:
    """Project the fixed nested schema for existing flat judge/template APIs only.
    The stored/generated rubric keeps user-side / Agent-side / behavior fields.
    Never expose analysis, source outcomes, or feedback to the response judge.
    """
    # score_nodes now projects before invoking the callback. Accept that view
    # idempotently so direct backend callers with canonical rubrics also work.
    if isinstance(rubric, dict) and 'candidate_rubric' not in rubric and 'applicability' not in rubric:
        keys = ('name', 'applicability_condition', 'required_behavior')
        if not all(isinstance(rubric.get(k), str) and rubric[k].strip() for k in keys):
            raise ValueError('Invalid legacy judge rubric view')
        scores = rubric.get('scores')
        if not isinstance(scores, dict) or set(scores) != {'0', '1', '2'} or not all(
                isinstance(value, str) and value.strip() for value in scores.values()):
            raise ValueError('Legacy judge view requires behavior text at all three levels')
        visible = {k: rubric[k] for k in keys}
        visible['scores'] = dict(scores)
        if 'rubric_id' in rubric:
            visible['rubric_id'] = rubric['rubric_id']
        return visible
    from src.online_rl.tree.posthoc import rubric_problems
    problems = rubric_problems(rubric)
    if problems:
        raise ValueError(f"Invalid behavioral rubric: {problems}")
    candidate = rubric.get("candidate_rubric", rubric)
    app = candidate["applicability"]
    scores = {k: candidate["scores"][k]["behavior"] for k in ("0", "1", "2")}
    visible = {
        "name": candidate["name"],
        "applicability_condition": ("[Physician-side context]\n" + app["user_side_context"]
                                    + "\n[Pre-response Agent state]\n" + app["agent_side_state"]),
        "required_behavior": scores["2"],
        "scores": scores,
    }
    if "rubric_id" in rubric:
        visible["rubric_id"] = rubric["rubric_id"]
    return visible



class ApiDoctor:
    """Doctor discussion turns and FINAL through the existing collaboration prompts (OpenRouter)."""

    def __init__(self, key: str, model: str, system: str, max_tokens: int, *, call=None):
        self.key, self.model, self.system, self.max_tokens = key, model, system, max_tokens
        self.call = call

    def turn(self, case: dict, dialogue: list[dict], round_: int, rounds: int) -> str:
        from src.collaboration.generator import doctor_call
        from src.online_rl.environment import discussion_prompt
        return (self.call or doctor_call)(self.key, self.model, self.system, discussion_prompt(case, dialogue, round_, rounds), self.max_tokens)

    def final(self, case: dict, dialogue: list[dict]) -> str:
        from src.collaboration.generator import doctor_call, final_decision_prompt
        return (self.call or doctor_call)(self.key, self.model, self.system, final_decision_prompt(case, dialogue), self.max_tokens)


class PolicyAgent:
    """One traced Agent sample from the current policy with the plain Agent prompt (no rubric, no feedback)."""

    def __init__(self, policy): self.policy = policy

    def sample(self, case: dict, prefix: list[dict], turn: int, seed: int) -> dict:
        tr = self.policy.sample_completions(case, prefix, turn, [seed])[0]
        return {k: tr[k] for k in ('prompt_text', 'prompt_input_ids', 'completion_ids', 'old_policy_logprobs', 'completion_text', 'seed')}


def api_feedback(key: str, model: str, system: str, max_tokens: int, *, call=None):
    """Doctor feedback on the labelled transcript; one bounded retry if it names an Agent reply that does not exist."""
    from src.collaboration.generator import doctor_call
    from src.online_rl.tree.posthoc import feedback_label_problems, feedback_message, labeled_dialogue

    completion = call or doctor_call

    def feedback(view, dialogue, outcome):
        user = feedback_message(view, dialogue, outcome); n = labeled_dialogue(dialogue)[1]
        text = completion(key, model, system, user, max_tokens); bad = feedback_label_problems(text, n)
        if bad:
            note = (f'\n\nYour previous feedback referred to {", ".join(bad)}, which does not exist. The Agent gave only {n} replies '
                    f'(Agent reply 1 to Agent reply {n}). Write the feedback again using only those labels.')
            text = completion(key, model, system, user + note, max_tokens)
        return text
    return feedback


def api_rubric_generator(key: str, model: str | None = None, max_tokens: int | None = None):
    from src.prompt_opt import harness as H
    from src.online_rl.tree.posthoc import trajectory_message

    def generate(prompt_text, tin):
        raw = H.doctor_call(key, model or H.SOL, prompt_text, trajectory_message(tin), max_tokens or H.GEN_MAX_TOKENS, H.GEN_TEMPERATURE)
        import json
        value = raw.strip()
        if value.startswith('```'):
            lines = value.splitlines()
            if len(lines) >= 3 and lines[-1].strip() == '```':
                value = '\n'.join(lines[1:-1])
        try:
            rubric = json.loads(value)
            if not isinstance(rubric, dict):
                raise ValueError('generator JSON must be an object')
            err = None
        except (ValueError, TypeError) as exc:
            rubric, err = None, str(exc)
        return {'raw': raw, 'rubric': rubric, 'parse_error': err}
    return generate


def two_stage_scorer(client):
    """Two-stage Jev scoring: applicability on the pre-reply dialogue, then the 0/1/2 level; 'NA' = No Match."""
    from src.judges.rubric_scoring import two_stage_jev_score

    def score(view, prefix, rubrics, reply):
        hist = {'trajectory_id': f"tree:{view['case_id']}", 'turn_id': len(prefix), 'dialogue_before_target_turn': prefix,
                'case_context': {'question': view['question'], 'options': view['options']}}
        if view.get('agent_pre_response_state'):
            hist['agent_pre_response_state'] = view['agent_pre_response_state']
        r = two_stage_jev_score(client, history_state=hist, target_state={**hist, 'target_turn': {'speaker': 'agent', 'text': reply}},
                                rubrics=[visible_behavior_rubric(x) for x in rubrics], purpose='tree_response_rubric_score')
        return {rid: {'applicable': r['applicability_labels'][rid] == 'Y', 'probabilities': r['behavior_probabilities'].get(rid),
                      'hard': r['behavior_scores'].get(rid)} for rid in (x['rubric_id'] for x in rubrics)}
    score.judge_identity = client.identity() if hasattr(client, 'identity') else None
    return score


POSTHOC_BLOCKING = ('answer_presupposition', 'new_domain_premise')
POSTHOC_UNRESOLVED = ('observable',)


def posthoc_gates(client, cases: dict):
    """Post-hoc rubric gate: schema, deterministic leakage / answer copy (the gold is used only inside this check), and the
    Jev answer_presupposition and new_domain_premise questions on the source discussion without the FINAL. 'observable' is
    recorded but does not block."""
    import json
    from src.judges.rubric_scoring import validate_visible_rubric, _batched_jev_decisions
    from src.online_rl.tree.records import to_dialogue
    from src.prompt_opt import harness as H
    from src.prompt_opt import harness_v11 as V
    Q = json.loads((H.ROOT / 'prompts/prompt_opt/rubric_validator_questions_v1_2.json').read_text())['questions']
    questions = {k: Q[k] for k in POSTHOC_BLOCKING + POSTHOC_UNRESOLVED}

    def gate(tin, rubric) -> dict:
        case = cases[tin.case_id]; g = {'schema': None, 'leakage_deterministic': None, **{k: None for k in questions}, 'reasons': [], 'passed': False,
                                        'unresolved': list(POSTHOC_UNRESOLVED)}
        from src.online_rl.tree.posthoc import rubric_problems
        probs = rubric_problems(rubric)
        if not probs:
            visible = visible_behavior_rubric(rubric)
            probs += validate_visible_rubric(visible)
        g['schema'] = not probs
        if probs: g['reasons'].append(f'schema: {probs}'); return g
        leak = H.deterministic_leakage(visible, case); g['leakage_deterministic'] = leak
        if leak: g['reasons'].append(f'deterministic leakage: {leak}')
        discussion = [d for d in to_dialogue(tin.dialogue) if d['stage'] != 'FINAL']
        state = {'case_context': {'question': tin.question, 'options': dict(tin.options)}, 'dialogue_before_target_turn': discussion,
                 'candidate_rubric': V._visible(visible)}
        ans, _ = _batched_jev_decisions(client, state, questions, purpose='tree_posthoc_rubric_gate', max_rubrics_per_call=None)
        for k in questions: g[k] = ans[k]['value'] == 'Y'
        for k in POSTHOC_BLOCKING:
            if g[k]: g['reasons'].append(k.replace('_', ' '))
        g['passed'] = g['schema'] and not leak and not any(g[k] for k in POSTHOC_BLOCKING)
        return g
    return gate


def parse_final_option(text: str, options: dict) -> str | None:
    """Conservative fallback. Inject the repository's parser/judge when available.
    Accept an explicit FINAL answer marker or a standalone option letter, never
    an arbitrary letter mentioned in reasoning. Ambiguous finals stay incomplete.
    """
    import re
    labels = {str(k).upper() for k in options}
    if not labels or any(len(k) != 1 or not k.isalpha() for k in labels):
        raise ValueError('Default parser requires single-letter options; inject final_judge otherwise')
    text = text.strip().replace('**', '')
    found = re.findall(r'(?im)(?:\bfinal\s+(?:answer|decision)|^\s*answer)\s*:\s*(?:option\s+)?[\(\[]?([A-Z])\b', text)
    if not found:
        plain = re.fullmatch(r'(?:option\s+)?[\(\[]?([A-Z])[\)\].]?\s*', text, re.I)
        found = [plain.group(1)] if plain else []
    found = {x.upper() for x in found}
    return next(iter(found)) if len(found) == 1 and found <= labels else None


def default_final_judge(case: dict, text: str) -> dict:
    """Gold is used AFTER the Doctor call, for evaluation only.
    Only unambiguous gold letter fields are supported. Do not guess index bases.
    """
    options = dict(case['options'])
    gt = case.get('ground_truth')
    candidates = []
    if isinstance(gt, dict) and gt.get('correct_option') is not None:
        candidates.append(gt['correct_option'])
    if case.get('correct_option') is not None:
        candidates.append(case['correct_option'])
    if not candidates or any(not isinstance(x, str) for x in candidates):
        raise ValueError('No explicit gold option letter: inject the repository final_judge callback')
    golds = {x.strip().upper() for x in candidates}
    if len(golds) != 1 or not golds <= {str(k).upper() for k in options}:
        raise ValueError('Invalid/conflicting gold option letter')
    option = parse_final_option(text, options)
    return {'final_option': option or '', 'outcome': float(option in golds) if option else None}


class PolicyInterventions:
    """Fresh policy reply with a Score-0/2 prompt instruction, then Doctor FINAL only.
    No additional Agent/Doctor discussion and no policy.continue_from() call.
    """

    def __init__(self, policy, score0_template: str, score2_template: str | None = None,
                 *, doctor=None, final_judge=None):
        self.policy, self.score0_template, self.score2_template = policy, score0_template, score2_template
        self.doctor = doctor or ApiDoctor(policy.key, policy.args.doctor_model,
                                         policy.doctor_system, policy.args.max_tokens_doctor)
        self.final_judge = final_judge or default_final_judge

    def instruction(self, rubric, level):
        import json
        if level not in ('0', '2'):
            raise ValueError(f'intervention level must be 0 or 2, got {level!r}')
        template = self.score0_template if level == '0' else self.score2_template
        if not isinstance(template, str) or not template.strip():
            raise ValueError(f'Explicit Score-{level} prompt template is required')
        visible = visible_behavior_rubric(rubric)
        # Preserve the existing {rubric_json} template interface. The original
        # template files were not provided: their instructions still need review.
        target = {'name': visible['name'],
                  'applicability_condition': visible['applicability_condition'],
                  'target_level': level, 'target_behavior': visible['scores'][level]}
        return template.format(rubric_json=json.dumps(target, ensure_ascii=False))

    def write(self, case, prefix, turn, rubric, level, seed):
        prompt = self.policy.agent_user_prompt(case, prefix, turn) + '\n\n' + self.instruction(rubric, level)
        return self.policy.agent(prompt, seed)

    def continue_(self, case, prefix, turn, text, seed):
        from copy import deepcopy
        from src.online_rl.tree.posthoc import case_view
        dialogue = deepcopy(prefix)
        dialogue.append({'speaker': 'agent', 'stage': f'A{turn}', 'round': turn, 'text': text})
        # Strip gold before sending case context to the Doctor.
        final_text = self.doctor.final(case_view(case), dialogue)
        judged = self.final_judge(case, final_text)
        if not isinstance(judged, dict):
            raise TypeError('final_judge must return a dict with final_option and outcome')
        outcome = judged.get('outcome')
        final_option = judged.get('final_option') or ''
        complete = bool(final_option) and outcome in (0.0, 1.0)
        return {
            'final_text': final_text, 'final_option': final_option,
            'outcome': float(outcome) if complete else None,
            'status': 'complete' if complete else 'incomplete',
            'termination_reason': 'intervention_doctor_final_only',
            'agent_turns': turn, 'additional_discussion_turns': 0,
            # ApiDoctor.final currently accepts no seed; do not claim seed control
            # of OpenRouter randomness. Agent sampling uses write()'s seed.
            'requested_continuation_seed': seed,
        }
