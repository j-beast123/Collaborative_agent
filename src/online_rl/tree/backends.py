"""Real components for the case tree (lazy imports: need API credentials / the policy model)."""
from __future__ import annotations


class ApiDoctor:
    """Doctor discussion turns and FINAL through the existing collaboration prompts (OpenRouter)."""

    def __init__(self, key: str, model: str, system: str, max_tokens: int):
        self.key, self.model, self.system, self.max_tokens = key, model, system, max_tokens

    def turn(self, case: dict, dialogue: list[dict], round_: int, rounds: int) -> str:
        from src.collaboration.generator import doctor_call
        from src.online_rl.environment import discussion_prompt
        return doctor_call(self.key, self.model, self.system, discussion_prompt(case, dialogue, round_, rounds), self.max_tokens)

    def final(self, case: dict, dialogue: list[dict]) -> str:
        from src.collaboration.generator import doctor_call, final_decision_prompt
        return doctor_call(self.key, self.model, self.system, final_decision_prompt(case, dialogue), self.max_tokens)


class PolicyAgent:
    """One traced Agent sample from the current policy with the plain Agent prompt (no rubric, no feedback)."""

    def __init__(self, policy): self.policy = policy

    def sample(self, case: dict, prefix: list[dict], turn: int, seed: int) -> dict:
        tr = self.policy.sample_completions(case, prefix, turn, [seed])[0]
        return {k: tr[k] for k in ('prompt_text', 'prompt_input_ids', 'completion_ids', 'old_policy_logprobs', 'completion_text', 'seed')}


def api_feedback(key: str, model: str, system: str, max_tokens: int):
    """Doctor feedback on the labelled transcript; one bounded retry if it names an Agent reply that does not exist."""
    from src.collaboration.generator import doctor_call
    from src.online_rl.tree.posthoc import feedback_label_problems, feedback_message, labeled_dialogue

    def feedback(view, dialogue, outcome):
        user = feedback_message(view, dialogue, outcome); n = labeled_dialogue(dialogue)[1]
        text = doctor_call(key, model, system, user, max_tokens); bad = feedback_label_problems(text, n)
        if bad:
            note = (f'\n\nYour previous feedback referred to {", ".join(bad)}, which does not exist. The Agent gave only {n} replies '
                    f'(Agent reply 1 to Agent reply {n}). Write the feedback again using only those labels.')
            text = doctor_call(key, model, system, user + note, max_tokens)
        return text
    return feedback


def api_rubric_generator(key: str):
    from src.prompt_opt import harness as H
    from src.online_rl.tree.posthoc import trajectory_message

    def generate(prompt_text, tin):
        raw = H.doctor_call(key, H.SOL, prompt_text, trajectory_message(tin), H.GEN_MAX_TOKENS, H.GEN_TEMPERATURE)
        rubric, err = H.parse_rubric(raw)
        return {'raw': raw, 'rubric': rubric, 'parse_error': err}
    return generate


def two_stage_scorer(client):
    """Two-stage Jev scoring: applicability on the pre-reply dialogue, then the 0/1/2 level; 'NA' = No Match."""
    from src.judges.rubric_scoring import two_stage_jev_score

    def score(view, prefix, rubrics, reply):
        hist = {'trajectory_id': f"tree:{view['case_id']}", 'turn_id': len(prefix), 'dialogue_before_target_turn': prefix,
                'case_context': {'question': view['question'], 'options': view['options']}}
        r = two_stage_jev_score(client, history_state=hist, target_state={**hist, 'target_turn': {'speaker': 'agent', 'text': reply}},
                                rubrics=rubrics, purpose='tree_response_rubric_score')
        return {rid: {'applicable': r['applicability_labels'][rid] == 'Y', 'probabilities': r['behavior_probabilities'].get(rid),
                      'hard': r['behavior_scores'].get(rid)} for rid in (x['rubric_id'] for x in rubrics)}
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
        probs = validate_visible_rubric(rubric) + ([] if isinstance(rubric.get('required_behavior'), str) and rubric['required_behavior'].strip() else ['missing required_behavior'])
        g['schema'] = not probs
        if probs: g['reasons'].append(f'schema: {probs}'); return g
        leak = H.deterministic_leakage(rubric, case); g['leakage_deterministic'] = leak
        if leak: g['reasons'].append(f'deterministic leakage: {leak}')
        discussion = [d for d in to_dialogue(tin.dialogue) if d['stage'] != 'FINAL']
        state = {'case_context': {'question': tin.question, 'options': dict(tin.options)}, 'dialogue_before_target_turn': discussion,
                 'candidate_rubric': V._visible(rubric)}
        ans, _ = _batched_jev_decisions(client, state, questions, purpose='tree_posthoc_rubric_gate', max_rubrics_per_call=None)
        for k in questions: g[k] = ans[k]['value'] == 'Y'
        for k in POSTHOC_BLOCKING:
            if g[k]: g['reasons'].append(k.replace('_', ' '))
        g['passed'] = g['schema'] and not leak and not any(g[k] for k in POSTHOC_BLOCKING)
        return g
    return gate


class PolicyInterventions:
    """Intervention reply and continuation from the same policy. Both levels use the Env.guided call shape with the configured
    Score-2 / Score-0 template (Env.guided itself when no Score-2 template is given)."""

    def __init__(self, policy, score0_template: str, score2_template: str | None = None):
        self.policy, self.score0_template, self.score2_template = policy, score0_template, score2_template

    def write(self, case, prefix, turn, rubric, level, seed):
        import json
        from src.prompt_opt import harness as H
        if level == '2' and self.score2_template is None: return H.Env.guided(self.policy, case, prefix, turn, rubric, seed)
        if level not in ('0', '2'): raise ValueError(f'intervention level must be 0 or 2, got {level!r}')
        template = self.score0_template if level == '0' else self.score2_template      # same call shape as Env.guided
        visible = {k: rubric[k] for k in ('name', 'applicability_condition', 'required_behavior', 'scores')}
        return self.policy.agent(self.policy.agent_user_prompt(case, prefix, turn) + '\n\n' + template.format(rubric_json=json.dumps(visible, ensure_ascii=False)), seed)

    def continue_(self, case, prefix, turn, text, seed):
        r = self.policy.continue_from(case, prefix, turn, text, seed)
        return {'final_option': r['final_option'], 'outcome': r['correct'], 'termination_reason': r['termination_reason'], 'agent_turns': r['agent_turns']}
