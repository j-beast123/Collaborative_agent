"""Real components for the case tree (lazy imports: need API credentials / the policy model). Every API-backed callable
carries its provider model id as `.model` (part of its cache key), and every call tags its role in the usage ledger."""
from __future__ import annotations
import json


def with_role(role: str, fn):
    """Run fn with the usage-ledger stage set to the pipeline role (doctor, feedback, rubric_generator, ...); thread-local, so
    concurrent Doctor / rerun threads label their own calls."""
    from src.common.usage import _stage
    old = getattr(_stage, 'value', None); _stage.value = role
    try: return fn()
    finally: _stage.value = old


class ApiDoctor:
    """Doctor discussion turns and FINAL through the existing collaboration prompts (OpenRouter)."""

    def __init__(self, key: str, model: str, system: str, max_tokens: int):
        self.key, self.model, self.system, self.max_tokens = key, model, system, max_tokens

    def turn(self, case: dict, dialogue: list[dict], round_: int, rounds: int) -> str:
        from src.collaboration.generator import doctor_call
        from src.online_rl.environment import discussion_prompt
        return with_role('doctor', lambda: doctor_call(self.key, self.model, self.system, discussion_prompt(case, dialogue, round_, rounds), self.max_tokens))

    def final(self, case: dict, dialogue: list[dict]) -> str:
        from src.collaboration.generator import doctor_call, final_decision_prompt
        return with_role('doctor', lambda: doctor_call(self.key, self.model, self.system, final_decision_prompt(case, dialogue), self.max_tokens))


class RoleDoctor:
    """A protocol-v2 Doctor (src/online_rl/doctor.py) with its calls tagged in the usage ledger under `role`."""
    status_protocol = True

    def __init__(self, doctor, role: str = 'doctor'):
        self.doctor, self.role, self.model, self.protocol = doctor, role, doctor.model, doctor.protocol

    def turn(self, case, dialogue, round_, rounds, seed): return with_role(self.role, lambda: self.doctor.turn(case, dialogue, round_, rounds, seed))

    def final(self, case, dialogue, seed): return with_role(self.role, lambda: self.doctor.final(case, dialogue, seed))

    def halt(self, **kw): return self.doctor.halt(**kw)


class PolicyAgent:
    """One traced Agent sample from the current policy with the plain Agent prompt (no rubric, no feedback)."""

    def __init__(self, policy): self.policy = policy

    def sample(self, case: dict, prefix: list[dict], turn: int, seed: int) -> dict:
        tr = self.policy.sample_completions(case, prefix, turn, [seed])[0]
        keep = ('prompt_text', 'prompt_input_ids', 'completion_ids', 'old_policy_logprobs', 'completion_text', 'seed',
                'segments', 'state_prompt_hash', 'private_assessment', 'agent_protocol')        # last four: precommit_v1 joint action
        return {k: tr[k] for k in keep if k in tr}


def api_feedback(key: str, model: str, system: str, max_tokens: int, locus: bool = False):
    """Feedback Generator on the labelled transcript; one bounded retry if it names an Agent reply that does not exist.
    locus=True (feedback v3): the answer also ends with 'Behavioral locus: Agent reply N'; a missing / malformed /
    out-of-range locus joins the same single bounded retry. Returns {'text' (locus line removed), 'target_reply_id' (None
    if still invalid after the retry), 'status' ok | repaired | <problem>, 'raw' [answers]}."""
    from src.collaboration.generator import doctor_call
    from src.online_rl.tree.posthoc import feedback_label_problems, feedback_message, labeled_dialogue, split_locus
    call = lambda user: with_role('feedback', lambda: doctor_call(key, model, system, user, max_tokens))

    def feedback(view, dialogue, outcome):
        user = feedback_message(view, dialogue, outcome); n = labeled_dialogue(dialogue)[1]
        text = call(user); bad = feedback_label_problems(text, n)
        if not locus:
            if bad:
                note = (f'\n\nYour previous feedback referred to {", ".join(bad)}, which does not exist. The Agent gave only {n} replies '
                        f'(Agent reply 1 to Agent reply {n}). Write the feedback again using only those labels.')
                text = call(user + note)
            return text
        raw = [text]; clean, rid, problem = split_locus(text, n)
        if bad or problem:
            parts = []
            if bad: parts.append(f'referred to {", ".join(bad)}, which does not exist')
            if problem: parts.append(f'did not end with exactly one valid line "Behavioral locus: Agent reply N" ({problem})')
            note = (f'\n\nYour previous feedback {" and ".join(parts)}. The Agent gave only {n} replies (Agent reply 1 to Agent reply '
                    f'{n}). Write the feedback again using only those labels and end with exactly one line "Behavioral locus: Agent reply N".')
            text = call(user + note); raw.append(text); clean, rid, problem2 = split_locus(text, n)
            status = 'repaired' if problem2 is None else problem2     # as before, leftover label problems do not block the feedback
        else: status = 'ok'
        return {'text': clean, 'target_reply_id': rid if status in ('ok', 'repaired') else None, 'status': status, 'raw': raw}
    feedback.model = model
    return feedback


def api_behavior_feedback(key: str, model: str, system: str, max_tokens: int):
    """Feedback v4 (behavior observer): the writer sees ONLY the labelled dialogue before the FINAL (no question, options,
    final decision, outcome or gold). The answer ends with 'Behavioral locus: Agent reply N'. Deterministic guards: reply
    labels, locus, and answer leakage (option labels, option texts, correctness wording; posthoc.leakage_problems). Any
    problem -> ONE repair with the same input plus a note naming the problems. Leakage still present -> invalid feedback
    (valid=False, no locus); never silently redacted. A locus problem alone leaves a valid feedback without locus."""
    from src.collaboration.generator import doctor_call
    from src.online_rl.tree.posthoc import behavior_message, feedback_label_problems, leakage_problems, split_locus
    call = lambda user: with_role('feedback', lambda: doctor_call(key, model, system, user, max_tokens))

    def check(text, n, view):
        clean, rid, locus = split_locus(text, n)
        return clean, rid, locus, feedback_label_problems(text, n), leakage_problems(clean, view)

    def feedback(view, dialogue, outcome):                          # view / outcome: used only by the guard, never shown
        user, n = behavior_message(dialogue)
        text = call(user); raw = [text]; clean, rid, locus, labels, leak = check(text, n, view); first_leak = leak
        if locus or labels or leak:
            parts = []
            if leak: parts.append('used answer-specific or correctness wording (' + '; '.join(leak) + '): describe behavior only, refer to positions '
                                  'abstractly, and do not judge whether anything was correct')
            if labels: parts.append(f'referred to {", ".join(labels)}, which does not exist (only Agent reply 1 to Agent reply {n})')
            if locus: parts.append(f'did not end with exactly one valid line "Behavioral locus: Agent reply N" ({locus})')
            text = call(user + '\n\nYour previous feedback ' + ' and '.join(parts) + '. Write the feedback again following every rule.'); raw.append(text)
            clean, rid, locus, labels, leak = check(text, n, view)
        valid = not leak
        status = ('ok' if len(raw) == 1 else 'repaired') if valid and locus is None else ('leakage' if not valid else locus)
        return {'text': clean, 'target_reply_id': rid if valid and locus is None else None, 'status': status, 'raw': raw, 'valid': valid,
                'leakage': {'first': first_leak, 'final': leak}, 'kind': 'behavior_observer_v4'}
    feedback.model = model; feedback.transcript = 'behavior_dialogue_v1'
    return feedback


RUBRIC_JSON_SCHEMA = {'type': 'object', 'additionalProperties': False, 'required': ['name', 'applicability_condition', 'required_behavior', 'scores'],
                      'properties': {'name': {'type': 'string'}, 'applicability_condition': {'type': 'string'}, 'required_behavior': {'type': 'string'},
                                     'scores': {'type': 'object', 'additionalProperties': False, 'required': ['0', '1', '2'],
                                                'properties': {k: {'type': 'string'} for k in '012'}}}}


class RubricOutputIncomplete(RuntimeError):
    """The Rubric Generator stopped before a complete answer (finish_reason != stop) or returned output that does not parse
    under the strict schema. Operational: nothing is cached, the run stops and the call is repeated on resume; it is never
    scored as an invalid rubric (s = 0)."""


def generation_settings(rubric_cfg: dict) -> dict:
    """rubric.generation: max_tokens, temperature and reasoning effort of every Rubric-Generator call (provenance + cache key).
    Gemini 3.8 Flash reasons mandatorily and its reasoning tokens count against max_tokens: at the default (medium) effort
    and 1200 tokens half of the Stage-A outputs were cut off. Pinned: effort low, 4096 tokens, strict JSON-schema output."""
    g = rubric_cfg.get('generation') or {}
    return {'max_tokens': g.get('max_tokens', 1200), 'temperature': g.get('temperature', 0.7), 'reasoning_effort': g.get('reasoning_effort'),
            'response_format': g.get('response_format')}


def _openrouter_json(key: str, model: str, system: str, user: str, settings: dict, schema: dict, name: str, role: str) -> tuple[str, dict]:
    """One Rubric-Generator call with the pinned settings; raises RubricOutputIncomplete unless it finished normally."""
    from src.prompt_opt import harness as H
    extra = {}
    if settings.get('reasoning_effort'): extra['reasoning'] = {'effort': settings['reasoning_effort']}
    if settings.get('response_format') == 'json_schema_strict':
        extra['response_format'] = {'type': 'json_schema', 'json_schema': {'name': name, 'strict': True, 'schema': schema}}
    meta: dict = {}
    if not extra:                                                     # legacy settings: the plain call, as before
        return with_role(role, lambda: H.doctor_call(key, model, system, user, settings['max_tokens'], settings['temperature'])), meta
    raw = with_role(role, lambda: H.doctor_call(key, model, system, user, settings['max_tokens'], settings['temperature'], extra=extra, meta=meta))
    if meta.get('finish_reason') not in (None, 'stop'):
        raise RubricOutputIncomplete(f"{model} stopped with finish_reason={meta.get('finish_reason')} "
                                     f"(native {meta.get('native_finish_reason')}, usage {meta.get('usage')}) before a complete {name}")
    return raw, meta


def api_rubric_generator(key: str, model: str, settings: dict | None = None):
    """CREATE: trajectory + feedback -> one rubric. With strict-schema settings a non-parsing output is an operational error."""
    from src.prompt_opt import harness as H
    from src.online_rl.tree.posthoc import trajectory_message
    settings = settings or {'max_tokens': H.GEN_MAX_TOKENS, 'temperature': H.GEN_TEMPERATURE, 'reasoning_effort': None, 'response_format': None}

    def generate(prompt_text, tin):
        raw, meta = _openrouter_json(key, model, prompt_text, trajectory_message(tin), settings, RUBRIC_JSON_SCHEMA, 'rubric', 'rubric_generator')
        rubric, err = H.parse_rubric(raw)
        if err and settings.get('response_format') == 'json_schema_strict': raise RubricOutputIncomplete(f'{model}: schema output did not parse: {err}')
        return {'raw': raw, 'rubric': rubric, 'parse_error': err, 'usage': meta.get('usage')}
    generate.model = model; generate.settings = settings
    return generate


MATCH_CHECKS = ('same_construct', 'applicability_visible', 'same_required_behavior', 'same_direction', 'score_semantics_apply', 'no_new_premise')
MATCH_JSON_SCHEMA = {'type': 'object', 'additionalProperties': False, 'required': ['decision', 'rubric_id', 'checks', 'rationale'],
                     'properties': {'decision': {'type': 'string', 'enum': ['REUSE_EXISTING', 'CREATE_NEW']}, 'rubric_id': {'type': 'string'},
                                    'checks': {'type': 'object', 'additionalProperties': False, 'required': list(MATCH_CHECKS),
                                               'properties': {c: {'type': 'boolean'} for c in MATCH_CHECKS}},
                                    'rationale': {'type': 'string'}}}


def api_rubric_matcher(key: str, model: str, settings: dict):
    """MATCH-vs-NEW: (prompt package, trajectory + current bank) -> {'decision', 'rubric_id', 'checks', 'rationale'} from the
    Rubric-Generator model; a decision only, never a rubric. Malformed output is an operational error (RubricOutputIncomplete)."""
    import json, re

    def decide(package, user):
        raw, meta = _openrouter_json(key, model, package, user, settings, MATCH_JSON_SCHEMA, 'rubric_match_decision', 'rubric_matcher')
        try: d = json.loads(re.sub(r'^```(?:json)?\s*|\s*```$', '', raw.strip()))
        except json.JSONDecodeError as e: raise RubricOutputIncomplete(f'{model}: match decision is not JSON: {e}') from e
        ok = (isinstance(d, dict) and set(d) == set(MATCH_JSON_SCHEMA['required']) and d['decision'] in ('REUSE_EXISTING', 'CREATE_NEW')
              and isinstance(d['rubric_id'], str) and isinstance(d['rationale'], str) and isinstance(d['checks'], dict)
              and set(d['checks']) == set(MATCH_CHECKS) and all(isinstance(v, bool) for v in d['checks'].values()))
        if not ok: raise RubricOutputIncomplete(f'{model}: match decision does not follow the schema: {raw[:300]}')
        return {**d, 'usage': meta.get('usage')}
    decide.model = model; decide.settings = settings
    return decide


def two_stage_scorer(client):
    """Two-stage Jev scoring: applicability on the pre-reply dialogue, then the 0/1/2 level; 'NA' = No Match."""
    from src.judges.rubric_scoring import two_stage_jev_score

    def score(view, prefix, rubrics, reply):
        hist = {'trajectory_id': f"tree:{view['case_id']}", 'turn_id': len(prefix), 'dialogue_before_target_turn': prefix,
                'case_context': {'question': view['question'], 'options': view['options']}}
        r = with_role('rubric_scorer', lambda: two_stage_jev_score(client, history_state=hist, target_state={**hist, 'target_turn': {'speaker': 'agent', 'text': reply}},
                                                                   rubrics=rubrics, purpose='tree_response_rubric_score'))
        return {rid: {'applicable': r['applicability_labels'][rid] == 'Y', 'probabilities': r['behavior_probabilities'].get(rid),
                      'hard': r['behavior_scores'].get(rid)} for rid in (x['rubric_id'] for x in rubrics)}
    score.model = getattr(client, 'model', None)
    return score


POSTHOC_BLOCKING = ('answer_presupposition', 'new_domain_premise')
POSTHOC_UNRESOLVED = ('observable',)


STAGE2_FORMAT = 'stage2_required_behavior_v1'
STAGE2_NOTE = ("The rubric's required behavior defines the behavioral construct being judged; the score descriptions in the "
               "criteria remain authoritative for assigning the level. Do not infer anything from correctness, the outcome, "
               "later turns or the gold answer.")


def stage2_questions(rubrics: list[dict]) -> dict:
    """Behavior questions of the two-stage scorer plus each rubric's required behavior (no applicability condition: stage 1
    already established it)."""
    from src.judges.rubric_scoring import behavior_questions
    q = behavior_questions(rubrics)
    return {r['rubric_id']: {**q[r['rubric_id']], 'instructions': f"{q[r['rubric_id']]['instructions']} Required behavior: {r['required_behavior']} {STAGE2_NOTE}"}
            for r in rubrics}


def two_stage_hard_scorer(client):
    """Two-stage hard scoring with a decision client that returns labels only (Luna): applicability Y/N on the dialogue
    before the reply (N = No Match), then level 0/1/2 for the applicable rubrics on the reply. Stage 1 asks the two-stage
    Jev applicability questions; stage 2 asks its behavior questions plus each rubric's required behavior; no probabilities. Format-alignment records of the client are kept under '_decisions'."""
    from src.judges.rubric_scoring import _batched_jev_decisions, applicability_questions, validate_visible_rubric
    from src.online_rl.integrated.records import sha256_text

    def score(view, prefix, rubrics, reply):
        for r in rubrics:
            if validate_visible_rubric(r): raise ValueError(f"invalid visible rubric {r.get('rubric_id')}")
        hist = {'trajectory_id': f"tree:{view['case_id']}", 'turn_id': len(prefix), 'dialogue_before_target_turn': prefix,
                'case_context': {'question': view['question'], 'options': view['options']}}
        target = {**hist, 'target_turn': {'speaker': 'agent', 'text': reply}}; n0 = len(getattr(client, 'log', []))
        app, _ = with_role('rubric_scorer', lambda: _batched_jev_decisions(client, json.loads(json.dumps(hist)), applicability_questions(rubrics),
                                                                             purpose='tree_response_rubric_score_applicability', max_rubrics_per_call=None))
        on = [r for r in rubrics if app[r['rubric_id']]['value'] == 'Y']; beh = {}
        if on: beh, _ = with_role('rubric_scorer', lambda: _batched_jev_decisions(client, target, stage2_questions(on),
                                                                                   purpose='tree_response_rubric_score_behavior', max_rubrics_per_call=None))
        out = {r['rubric_id']: {'applicable': r['rubric_id'] in beh, 'probabilities': None,
                                'hard': int(beh[r['rubric_id']]['value']) if r['rubric_id'] in beh else None} for r in rubrics}
        out['_decisions'] = getattr(client, 'log', [])[n0:]
        return out
    score.model = getattr(client, 'model', None)
    score.protocol = sha256_text(f"{getattr(client, 'protocol_sha256', None)}|{STAGE2_FORMAT}|{STAGE2_NOTE}")    # role prompt + stage-2 question format
    return score


SCORER_V2 = 'luna_structured_two_stage_v2_current_situation'
STAGE1_FIELDS = {'applicable': {'type': 'boolean'}, 'evidence_basis': {'type': 'string'}, 'missing_requirement': {'type': 'string'}}
STAGE2_FIELDS = {'level': {'type': 'string', 'enum': ['0', '1', '2']}, 'anchor_basis': {'type': 'string'}}


def stage1_items(rubrics: list[dict]) -> dict:
    """Applicability items: rubric name and applicability condition only (no required behavior, no score anchors)."""
    from src.judges.rubric_scoring import applicability_questions
    q = applicability_questions(rubrics)
    return {r['rubric_id']: {'rubric': r['name'], 'applicability_condition': r['applicability_condition'], 'instructions': q[r['rubric_id']]['instructions']}
            for r in rubrics}


def stage2_items(rubrics: list[dict]) -> dict:
    """Level items: rubric name, required behavior and the rubric's own score anchors (no applicability condition)."""
    from src.judges.rubric_scoring import behavior_questions
    q = behavior_questions(rubrics)
    return {r['rubric_id']: {'rubric': r['name'], 'required_behavior': r['required_behavior'], 'score_anchors': {k: r['scores'][k] for k in ('0', '1', '2')},
                             'instructions': q[r['rubric_id']]['instructions'] + ' ' + STAGE2_NOTE} for r in rubrics}


def two_stage_structured_scorer(client):
    """Scorer v2 (Luna, hard labels with a one-sentence basis per decision). Stage 1 sees the case, the dialogue BEFORE the
    target reply and each rubric's name + applicability condition, and returns applicable + evidence_basis /
    missing_requirement. Stage 2 (applicable rubrics only) adds the target reply and sees name + required behavior + the
    rubric's own 0/1/2 anchors, and returns level + anchor_basis. Neither stage sees outcome, gold, later turns or rubric
    effectiveness. Malformed output: one same-model repair (client.judge), then an operational error."""
    from src.judges.rubric_scoring import validate_visible_rubric
    from src.online_rl.integrated.records import sha256_text

    def score(view, prefix, rubrics, reply):
        for r in rubrics:
            if validate_visible_rubric(r) or not r.get('required_behavior'): raise ValueError(f"invalid visible rubric {r.get('rubric_id')}")
        latest = next((t['text'] for t in reversed(prefix) if t.get('speaker') == 'doctor'), '')
        before = {'case_context': {'question': view['question'], 'options': view['options']}, 'dialogue_before_target_reply': prefix}
        stage1_state = {**before, 'latest_physician_message': latest}      # the situation the target reply must respond to
        n0 = len(getattr(client, 'log', []))
        app, _ = with_role('rubric_scorer', lambda: client.judge('applicability', stage1_state, stage1_items(rubrics), STAGE1_FIELDS,
                                                                  'tree_response_rubric_score_applicability'))
        on = [r for r in rubrics if app[r['rubric_id']]['applicable']]; lev = {}
        if on:
            lev, _ = with_role('rubric_scorer', lambda: client.judge('level', {**before, 'target_agent_reply': reply}, stage2_items(on), STAGE2_FIELDS,
                                                                      'tree_response_rubric_score_behavior'))
        out = {r['rubric_id']: {'applicable': r['rubric_id'] in lev, 'probabilities': None,
                                'hard': int(lev[r['rubric_id']]['level']) if r['rubric_id'] in lev else None,
                                'stage1': app[r['rubric_id']], 'stage2': lev.get(r['rubric_id'])} for r in rubrics}
        out['_decisions'] = getattr(client, 'log', [])[n0:]
        return out
    def applicability(view, prefix, rubrics):
        """Stage 1 only (the same question as score's Stage 1): {rubric_id: {'applicable': bool, 'stage1': decision}}. Used by
        training-time witness verification, which needs no level; Stage 2 is never called here."""
        for r in rubrics:
            if validate_visible_rubric(r) or not r.get('required_behavior'): raise ValueError(f"invalid visible rubric {r.get('rubric_id')}")
        latest = next((t['text'] for t in reversed(prefix) if t.get('speaker') == 'doctor'), '')
        state = {'case_context': {'question': view['question'], 'options': view['options']}, 'dialogue_before_target_reply': prefix,
                 'latest_physician_message': latest}
        app, _ = with_role('rubric_scorer', lambda: client.judge('applicability', state, stage1_items(rubrics), STAGE1_FIELDS,
                                                                  'tree_response_rubric_score_applicability'))
        return {r['rubric_id']: {'applicable': bool(app[r['rubric_id']]['applicable']), 'stage1': app[r['rubric_id']]} for r in rubrics}
    score.model = getattr(client, 'model', None)
    score.protocol = sha256_text(f"{getattr(client, 'protocol_sha256', None)}|{SCORER_V2}|{STAGE2_NOTE}")
    applicability.model = score.model
    applicability.protocol = sha256_text(f"{getattr(client, 'protocol_sha256', None)}|{SCORER_V2}|stage1_only")
    score.applicability = applicability
    return score


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
        n0 = len(getattr(client, 'log', []))
        ans, _ = with_role('rubric_gate', lambda: _batched_jev_decisions(client, state, questions, purpose='tree_posthoc_rubric_gate', max_rubrics_per_call=None))
        g['decisions'] = getattr(client, 'log', [])[n0:]
        for k in questions: g[k] = ans[k]['value'] == 'Y'
        for k in POSTHOC_BLOCKING:
            if g[k]: g['reasons'].append(k.replace('_', ' '))
        g['passed'] = g['schema'] and not leak and not any(g[k] for k in POSTHOC_BLOCKING)
        return g
    gate.model = getattr(client, 'model', None); gate.protocol = getattr(client, 'protocol_sha256', None)
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

    def continue_(self, case, prefix, turn, text, seed, turn_cache=None):
        r = with_role('doctor_rerun', lambda: self.policy.continue_from(case, prefix, turn, text, seed, **({'turn_cache': turn_cache} if turn_cache else {})))
        return {'final_option': r['final_option'], 'outcome': r['correct'], 'termination_reason': r['termination_reason'], 'agent_turns': r['agent_turns']}
