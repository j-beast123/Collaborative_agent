"""Real components for the case tree (lazy imports: need API credentials / the policy model). Every API-backed callable
carries its provider model id as `.model` (part of its cache key), and every call tags its role in the usage ledger."""
from __future__ import annotations
import json


def with_role(role: str, fn):
    """Run fn with the usage-ledger stage set to the pipeline role (doctor, feedback, rubric_generator, ...); thread-local, so
    concurrent Doctor / rerun threads label their own calls."""
    from src.common.usage import _stage
    from src.common.live_gpu import activity
    old = getattr(_stage, 'value', None); _stage.value = role
    try:
        with activity('role.' + role):return fn()
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
        tr = with_role('rollout_agent', lambda: self.policy.sample_completions(case, prefix, turn, [seed])[0])
        keep = ('prompt_text', 'prompt_input_ids', 'completion_ids', 'old_policy_logprobs', 'completion_text', 'seed',
                'sampling_logprobs', 'sampling_temperature', 'sampling_top_p', 'sampling_top_k',
                'logprob_semantics', 'batch_sampling_protocol',
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


def api_behavior_feedback(key: str, model: str, system: str, max_tokens: int, protocol: str = 'behavior_observer_v4'):
    """Behavior-observer feedback: the writer sees ONLY the labelled dialogue before the FINAL (no question, options,
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
                'leakage': {'first': first_leak, 'final': leak}, 'kind': protocol}
    feedback.model = model; feedback.transcript = 'behavior_dialogue_v1'
    return feedback


RUBRIC_JSON_SCHEMA = {'type': 'object', 'additionalProperties': False, 'required': ['name', 'applicability_condition', 'required_behavior', 'scores'],
                      'properties': {'name': {'type': 'string'}, 'applicability_condition': {'type': 'string'}, 'required_behavior': {'type': 'string'},
                                     'scores': {'type': 'object', 'additionalProperties': False, 'required': ['0', '1', '2'],
                                                'properties': {k: {'type': 'string'} for k in '012'}}}}

CONTEXT_PROPERTIES = {k: {'type': 'string'} for k in ('user_side_context', 'agent_side_state')}
OBSERVED_RUBRIC_JSON_SCHEMA = {'type': 'object', 'additionalProperties': False,
    'required': ['name', 'applicability', 'required_behavior', 'scores'],
    'properties': {'name': {'type': 'string'}, 'required_behavior': {'type': 'string'},
        'applicability': {'type': 'object', 'additionalProperties': False,
                         'required': list(CONTEXT_PROPERTIES), 'properties': CONTEXT_PROPERTIES},
        'scores': {'type': 'object', 'additionalProperties': False, 'required': list('012'), 'properties': {
            k: {'type': 'object', 'additionalProperties': False,
                'required': [*CONTEXT_PROPERTIES, 'behavior'],
                'properties': {**CONTEXT_PROPERTIES, 'behavior': {'type': 'string'}}} for k in '012'}}}}

# Full provenance contract used by the fixed P0 supplied by the user. Keep this separate from
# the compact rubric schema: the full response is retained in raw/provenance, while only the
# normalized candidate rubric is exposed to matching, scoring, and training.
_STRING = {'type': 'string'}
_NULLABLE_STRING = {'anyOf': [_STRING, {'type': 'null'}]}
_SOURCE_ID = {'anyOf': [_STRING, {'type': 'integer'}]}
_EVIDENCE = {'type': 'object', 'additionalProperties': False, 'required': ['turn_id', 'speaker', 'quote'],
             'properties': {'turn_id': {'type': 'integer'}, 'speaker': {'type': 'string', 'enum': ['doctor', 'agent']}, 'quote': _STRING}}
_CANDIDATE_CONTEXT = {'type': 'object', 'additionalProperties': False,
                      'required': ['user_side_context', 'agent_side_state'],
                      'properties': {'user_side_context': _STRING, 'agent_side_state': _STRING}}
_CANDIDATE_SCORE = {'type': 'object', 'additionalProperties': False,
                    'required': ['user_side_context', 'agent_side_state', 'behavior'],
                    'properties': {'user_side_context': _STRING, 'agent_side_state': _STRING, 'behavior': _STRING}}
_CANDIDATE = {'type': 'object', 'additionalProperties': False,
              'required': ['name', 'applicability', 'scores'],
              'properties': {'name': _STRING, 'applicability': _CANDIDATE_CONTEXT,
                             'scores': {'type': 'object', 'additionalProperties': False, 'required': ['0', '1', '2'],
                                        'properties': {k: _CANDIDATE_SCORE for k in '012'}}}}

_ANALYSIS_TEXT_FIELDS = ('user_contribution', 'agent_side_state', 'agent_handling', 'interaction_need',
                         'selected_behavior', 'selection_rationale', 'source_behavior_support',
                         'behavior_to_preserve_or_correct')
_ANALYSIS_NULLABLE_TEXT_FIELDS = ('surface_request', 'decision_shaping_contribution', 'surface_handling', 'reasoning_handling')
_SCORE_CONSTRUCTION = {'type': 'object', 'additionalProperties': False,
    'required': ['score_0_basis', 'score_1_component', 'score_1_remaining_gap', 'score_2_basis', 'conclusion_independence'],
    'properties': {k: _STRING for k in ('score_0_basis', 'score_1_component', 'score_1_remaining_gap', 'score_2_basis', 'conclusion_independence')}}
_FEEDBACK_CHECK = {'type': 'object', 'additionalProperties': False, 'required': ['claim', 'assessment', 'dialogue_basis'],
    'properties': {'claim': _NULLABLE_STRING,
                   'assessment': {'type': 'string', 'enum': ['supported', 'partly_supported', 'unsupported', 'contradicted', 'not_provided']},
                   'dialogue_basis': _NULLABLE_STRING}}
_ANALYSIS_REQUIRED = [*_ANALYSIS_NULLABLE_TEXT_FIELDS, *_ANALYSIS_TEXT_FIELDS, 'score_construction_check',
                      'source_score', 'score_0_basis', 'selected_agent_turn_id', 'feedback_check',
                      'evidence_strength', 'limitations', 'evidence_spans']
_ANALYSIS_PROPERTIES = {
    **{k: _NULLABLE_STRING for k in _ANALYSIS_NULLABLE_TEXT_FIELDS},
    **{k: _STRING for k in _ANALYSIS_TEXT_FIELDS},
    'score_construction_check': _SCORE_CONSTRUCTION,
    'source_score': {'type': 'integer', 'enum': [0, 2]},
    'score_0_basis': {'type': 'string', 'enum': ['counterfactual_removal', 'observed_problem']},
    'selected_agent_turn_id': {'type': 'integer'},
    'feedback_check': _FEEDBACK_CHECK,
    'evidence_strength': {'type': 'string', 'enum': ['strong', 'limited']},
    'limitations': {'type': 'array', 'items': _STRING},
    'evidence_spans': {'type': 'array', 'items': _EVIDENCE}}
FULL_TRAJECTORY_RUBRIC_JSON_SCHEMA = {
    'type': 'object', 'additionalProperties': False,
    'required': ['rubric_id', 'source_case_id', 'source_trajectory_id', 'source_outcome', 'analysis', 'candidate_rubric'],
    'properties': {
        'rubric_id': _STRING, 'source_case_id': _SOURCE_ID, 'source_trajectory_id': _SOURCE_ID,
        'source_outcome': {'type': 'string', 'enum': ['success', 'failure']},
        'analysis': {'type': 'object', 'additionalProperties': False, 'required': _ANALYSIS_REQUIRED,
                     'properties': _ANALYSIS_PROPERTIES},
        'candidate_rubric': _CANDIDATE
    }
}

FULL_TRAJECTORY_RUBRIC_WITH_INTERVENTION_JSON_SCHEMA = {
    **FULL_TRAJECTORY_RUBRIC_JSON_SCHEMA,
    'required': [*FULL_TRAJECTORY_RUBRIC_JSON_SCHEMA['required'], 'intervention_instruction'],
    'properties': {**FULL_TRAJECTORY_RUBRIC_JSON_SCHEMA['properties'], 'intervention_instruction': _STRING}
}


class RubricOutputIncomplete(RuntimeError):
    """The Rubric Generator stopped before a complete answer (finish_reason != stop) or returned output that does not parse
    under the strict schema. Operational: nothing is cached, the run stops and the call is repeated on resume; it is never
    scored as an invalid rubric (s = 0)."""


GENERATION_SEMANTIC_PROTOCOL = 'public_mc_generation_v2'
SEMANTIC_CRITERION_VERSION = 'public_behavior_mc_semantic_v2'
GENERATION_SEMANTIC_VERSIONS = {
    GENERATION_SEMANTIC_PROTOCOL: SEMANTIC_CRITERION_VERSION,
    'public_mc_generation_v3': 'public_behavior_mc_semantic_v3',
    'public_mc_generation_refs_v4': 'public_behavior_mc_refs_v4',
    'public_app_alignment_generation_v5': 'public_behavior_app_alignment_v5',
}


def generation_settings(rubric_cfg: dict) -> dict:
    """rubric.generation: max_tokens, temperature and reasoning effort of every Rubric-Generator call (provenance + cache key).
    Gemini 3.8 Flash reasons mandatorily and its reasoning tokens count against max_tokens: at the default (medium) effort
    and 1200 tokens half of the Stage-A outputs were cut off. trajectory_analysis_candidate_v2 pins the complete P0 analysis
    contract. Pinned: effort low, 4096 tokens, strict JSON-schema output."""
    g = rubric_cfg.get('generation') or {}
    from src.online_rl.tree.feedback_grounding import validate_protocol
    feedback_check_protocol = validate_protocol(g.get('feedback_check_protocol'))
    settings = {'max_tokens': g.get('max_tokens', 1200), 'temperature': g.get('temperature', 0.7), 'reasoning_effort': g.get('reasoning_effort'),
            'response_format': g.get('response_format'), 'output_contract': g.get('output_contract'),
            'local_schema_file': g.get('local_schema_file'), 'local_schema_sha256': g.get('local_schema_sha256'),
            'selected_turn_policy': g.get('selected_turn_policy', 'feedback_locus_v1')}
    # Do not add a null/default key to legacy settings: these identify old caches.
    if feedback_check_protocol is not None:
        settings['feedback_check_protocol'] = feedback_check_protocol
    if 'api_provider' in g: settings['api_provider'] = g['api_provider']
    if g.get('strict_provider_parameters'):
        settings['strict_provider_parameters'] = True
    alignment = g.get('instruction_alignment_protocol')
    if alignment is not None:
        from src.online_rl.tree.instruction_audit import ANALYSIS_PROTOCOL
        if alignment != ANALYSIS_PROTOCOL:
            raise ValueError('Unknown instruction alignment analysis protocol')
        settings['instruction_alignment_protocol'] = alignment
    public_protocol = g.get('public_criterion_protocol')
    if public_protocol is not None:
        from src.judges.rubric_schema import PUBLIC_CRITERION_PROTOCOL
        if public_protocol != PUBLIC_CRITERION_PROTOCOL:
            raise ValueError('Unknown public criterion protocol')
        settings['public_criterion_protocol'] = public_protocol
    semantic_protocol = g.get('generation_semantic_protocol')
    if semantic_protocol is not None:
        if semantic_protocol not in GENERATION_SEMANTIC_VERSIONS or public_protocol is None:
            raise ValueError('Unknown or incompatible generation semantic protocol')
        settings['generation_semantic_protocol'] = semantic_protocol
    return settings


def normalize_trajectory_analysis_output(parsed: dict | None, err: str | None, tin, *, require_intervention: bool = False,
                                        enforce_feedback_locus: bool = True,
                                        feedback_check_protocol: str | None = None,
                                        instruction_alignment_protocol: str | None = None,
                                        public_criterion_protocol: str | None = None,
                                        generation_semantic_protocol: str | None = None) -> tuple[dict | None, str | None]:
    """Validate the user's full P0 response and adapt its reusable candidate to the existing rubric pipeline."""
    from src.online_rl.tree.feedback_grounding import validate_protocol, validate_claim_reviews
    validate_protocol(feedback_check_protocol)
    from src.judges.rubric_schema import PUBLIC_CRITERION_PROTOCOL, PUBLIC_CRITERION_FIELDS, validate_public_criterion
    if public_criterion_protocol is not None and public_criterion_protocol != PUBLIC_CRITERION_PROTOCOL:
        raise ValueError('Unknown public criterion protocol')
    observed_source = generation_semantic_protocol is not None
    reference_analysis = generation_semantic_protocol in ('public_mc_generation_refs_v4', 'public_app_alignment_generation_v5')
    if observed_source and (generation_semantic_protocol not in GENERATION_SEMANTIC_VERSIONS or
                            public_criterion_protocol is None or not require_intervention):
        raise ValueError('Generation semantic protocol requires the public joint intervention contract')
    if instruction_alignment_protocol is not None:
        from src.online_rl.tree.instruction_audit import ANALYSIS_PROTOCOL
        if instruction_alignment_protocol != ANALYSIS_PROTOCOL or not require_intervention:
            raise ValueError('Instruction alignment analysis requires the joint intervention contract')
    if err: return None, err
    if not isinstance(parsed, dict) or not isinstance(parsed.get('analysis'), dict) or not isinstance(parsed.get('candidate_rubric'), dict):
        return None, 'missing analysis/candidate_rubric output contract'
    if require_intervention and (not isinstance(parsed.get('intervention_instruction'), str) or not parsed['intervention_instruction'].strip()):
        return None, 'intervention_instruction must be a substantive non-empty string'
    if require_intervention:
        expected_top = {'rubric_id', 'source_case_id', 'source_trajectory_id', 'source_outcome', 'analysis', 'candidate_rubric', 'intervention_instruction'}
        if set(parsed) != expected_top:
            return None, f'joint rubric/intervention output must contain exactly {sorted(expected_top)}'
    analysis, candidate = parsed['analysis'], parsed['candidate_rubric']
    # Validate the complete analysis contract locally as well as at the provider boundary.
    # This keeps permissive/non-strict transports from silently turning missing analysis into
    # apparently valid provenance via inferred turn IDs or synthesized quotations.
    missing = [k for k in _ANALYSIS_REQUIRED if k not in analysis]
    if missing: return None, f'missing required analysis fields: {", ".join(missing)}'
    extra = sorted(set(analysis) - set(_ANALYSIS_PROPERTIES))
    if extra: return None, f'unexpected analysis fields: {", ".join(extra)}'
    for key in _ANALYSIS_TEXT_FIELDS:
        if not isinstance(analysis.get(key), str) or not analysis[key].strip():
            return None, f'analysis.{key} must be a substantive non-empty string'
    for key in _ANALYSIS_NULLABLE_TEXT_FIELDS:
        value = analysis.get(key)
        if value is not None and (not isinstance(value, str) or not value.strip()):
            return None, f'analysis.{key} must be a substantive string or null'
    check = analysis.get('score_construction_check')
    if not isinstance(check, dict): return None, 'analysis.score_construction_check must be an object'
    for key in _SCORE_CONSTRUCTION['required']:
        if not isinstance(check.get(key), str) or not check[key].strip():
            return None, f'analysis.score_construction_check.{key} must be a substantive non-empty string'
    if instruction_alignment_protocol is not None:
        if reference_analysis:
            if 'minimal_contribution' in check or 'complete_behavior' in check:
                return None, 'Reference analysis must not duplicate M/C definitions'
            for key, ref in [('minimum_ref', 'candidate_rubric.minimum_performance'),
                             ('completion_ref', 'candidate_rubric.completion_condition')]:
                if check.get(key) != ref:
                    return None, f'analysis.{key} must reference {ref}'
        else:
            for key in ('minimal_contribution', 'complete_behavior'):
                if not isinstance(check.get(key), str) or not check[key].strip():
                    return None, f'analysis.score_construction_check.{key} must be a substantive non-empty string'
        alignment = check.get('instruction_alignment')
        instruction_field = 'instruction_ref' if reference_analysis else 'instruction_quote'
        if not isinstance(alignment, dict) or set(alignment) != {'target_level', instruction_field, 'rationale'}:
            return None, f'instruction_alignment must contain exactly target_level, {instruction_field}, rationale'
        expected_level = 0 if tin.outcome == 'success' else 2
        if type(alignment['target_level']) is not int or alignment['target_level'] != expected_level:
            return None, 'instruction_alignment.target_level disagrees with the source direction'
        for key in (instruction_field, 'rationale'):
            if not isinstance(alignment[key], str) or not alignment[key].strip():
                return None, f'instruction_alignment.{key} must be substantive'
        if reference_analysis and alignment['instruction_ref'] != 'intervention_instruction':
            return None, 'instruction_alignment must reference stored intervention_instruction'
        if not reference_analysis and alignment['instruction_quote'] not in parsed['intervention_instruction']:
            return None, 'instruction_alignment.instruction_quote must quote the exact stored instruction'
    feedback = analysis.get('feedback_check')
    feedback_fields = {'claim', 'assessment', 'dialogue_basis'}
    if feedback_check_protocol is not None: feedback_fields.add('claim_reviews')
    if not isinstance(feedback, dict) or set(feedback) != feedback_fields:
        return None, ('analysis.feedback_check must contain exactly claim, assessment, dialogue_basis' +
                      (', claim_reviews' if feedback_check_protocol is not None else ''))
    if feedback['assessment'] not in ('supported', 'partly_supported', 'unsupported', 'contradicted', 'not_provided'):
        return None, 'analysis.feedback_check.assessment has an invalid value'
    for key in ('claim', 'dialogue_basis'):
        if feedback[key] is not None and (not isinstance(feedback[key], str) or not feedback[key].strip()):
            return None, f'analysis.feedback_check.{key} must be a substantive string or null'
    if feedback_check_protocol is not None:
        feedback_error = validate_claim_reviews(feedback, tin)
        if feedback_error: return None, feedback_error
    if analysis.get('evidence_strength') not in ('strong', 'limited'):
        return None, 'analysis.evidence_strength must be strong or limited'
    source_score = analysis.get('source_score')
    if observed_source and (type(source_score) is not int or source_score not in (0, 1, 2)):
        return None, 'analysis.source_score must be integer 0, 1 or 2 for the observed source'
    if not observed_source and (isinstance(source_score, bool) or source_score not in (0, 2)):
        return None, 'analysis.source_score must be integer 0 or 2'
    if analysis.get('score_0_basis') not in ('counterfactual_removal', 'observed_problem'):
        return None, 'analysis.score_0_basis has an invalid value'
    selected_id = analysis.get('selected_agent_turn_id')
    if isinstance(selected_id, bool) or not isinstance(selected_id, int):
        return None, 'analysis.selected_agent_turn_id must be an integer Agent reply ID'
    if require_intervention and enforce_feedback_locus and tin.target_reply_id is not None and selected_id != tin.target_reply_id:
        return None, f'analysis.selected_agent_turn_id={selected_id} differs from the feedback target reply ID {tin.target_reply_id}'
    if not isinstance(analysis.get('limitations'), list) or any(not isinstance(x, str) or not x.strip() for x in analysis['limitations']):
        return None, 'analysis.limitations must be an array of substantive strings'
    if not isinstance(analysis.get('evidence_spans'), list) or not analysis['evidence_spans']:
        return None, 'analysis.evidence_spans must contain grounded source quotations'
    for i, span in enumerate(analysis['evidence_spans']):
        if (not isinstance(span, dict) or set(span) != {'turn_id', 'speaker', 'quote'} or
                isinstance(span.get('turn_id'), bool) or not isinstance(span.get('turn_id'), int) or
                span.get('speaker') not in ('doctor', 'agent') or not isinstance(span.get('quote'), str) or not span['quote'].strip()):
            return None, f'analysis.evidence_spans[{i}] must contain integer turn_id, valid speaker, and non-empty quote'
    candidate_fields = {'name', 'applicability', 'scores'}
    if public_criterion_protocol is not None:
        candidate_fields.update(('required_behavior', *PUBLIC_CRITERION_FIELDS))
    if set(candidate) != candidate_fields:
        return None, f'candidate_rubric must contain exactly {sorted(candidate_fields)}'
    if public_criterion_protocol is not None:
        problems = validate_public_criterion(candidate, required=True)
        if problems: return None, '; '.join(problems)
        expected_version = GENERATION_SEMANTIC_VERSIONS[generation_semantic_protocol] if observed_source else PUBLIC_CRITERION_PROTOCOL
        if candidate['criterion_version'] != expected_version:
            return None, 'candidate_rubric.criterion_version differs from the generation contract'
        if not reference_analysis and check.get('minimal_contribution') != candidate['minimum_performance']:
            return None, 'analysis minimal_contribution must exactly repeat public minimum_performance'
    outcome = 'success' if tin.outcome == 'success' else 'failure'
    expected_source_score = 2 if outcome == 'success' else 0
    expected_basis = 'counterfactual_removal' if outcome == 'success' else 'observed_problem'
    if public_criterion_protocol is not None and ((not observed_source and source_score != expected_source_score)
                                                 or analysis['score_0_basis'] != expected_basis):
        return None, 'public source anchor declaration differs from the source direction; do not relabel observed partial behavior'
    if parsed.get('source_case_id') != tin.case_id or parsed.get('source_trajectory_id') != tin.leaf_id:
        return None, 'source case/trajectory ID does not match the input'
    if parsed.get('source_outcome') != outcome:
        return None, f'source_outcome must be {outcome!r}'
    # Legacy fields were outcome annotations. The opt-in semantic contract instead
    # preserves the generator's source-score claim; independent blind scoring and the
    # outcome-derived Gate expectation remain separate. A truthful partial claim is
    # not a malformed output to retry until it becomes the desired source anchor.
    reported_analysis = dict(analysis)
    normalized_analysis = {**analysis, 'source_score': source_score if observed_source else expected_source_score,
                            'score_0_basis': expected_basis}
    adjustments = {k: {'model_reported': reported_analysis.get(k), 'canonical': normalized_analysis[k]}
                   for k in ('source_score', 'score_0_basis') if reported_analysis.get(k) != normalized_analysis[k]}
    scores = candidate.get('scores') or {}
    app = candidate.get('applicability') or {}
    high = scores.get('2')
    if not isinstance(high, dict) or not isinstance(high.get('behavior'), str) or not high['behavior'].strip():
        return None, 'candidate_rubric.scores.2.behavior is required to derive required_behavior'
    for level in ('0', '1', '2'):
        score = scores.get(level)
        if not isinstance(score, dict): return None, f'candidate_rubric.scores.{level} is missing'
        for field in ('user_side_context', 'agent_side_state'):
            if score.get(field) != app.get(field):
                return None, f'candidate_rubric.scores.{level}.{field} differs from applicability'
    # The prompt input assigns integer Agent reply IDs and speaker-scoped Doctor turn IDs.
    agent_text, doctor_text = {}, {}
    ai = di = 0
    for speaker, _stage, _round, body in tin.dialogue:
        if speaker == 'agent': ai += 1; agent_text[ai] = body
        else: di += 1; doctor_text[di] = body
    selected = analysis.get('selected_agent_turn_id')
    if isinstance(selected, bool) or not isinstance(selected, int) or selected not in agent_text:
        return None, (f'analysis.selected_agent_turn_id={selected!r} is not a supplied Agent reply ID; '
                      f'valid integer IDs are {sorted(agent_text)}')
    evidence_spans = analysis.get('evidence_spans')
    for span in evidence_spans:
        source = agent_text if span['speaker'] == 'agent' else doctor_text
        if span['turn_id'] not in source or not span['quote'] or span['quote'] not in source[span['turn_id']]:
            return None, 'an evidence span does not exactly match its supplied speaker/turn'
    normalized = {**candidate, 'required_behavior': candidate['required_behavior'] if public_criterion_protocol is not None else high['behavior'],
                  'source_provenance': {**{k: parsed[k] for k in ('rubric_id', 'source_case_id', 'source_trajectory_id', 'source_outcome')},
                                        'analysis': normalized_analysis, 'model_reported_analysis': reported_analysis,
                                        'deterministic_adjustments': adjustments}}
    if feedback_check_protocol is not None:
        normalized['source_provenance']['feedback_check_protocol'] = feedback_check_protocol
    if instruction_alignment_protocol is not None:
        normalized['source_provenance']['instruction_alignment_protocol'] = instruction_alignment_protocol
    if public_criterion_protocol is not None:
        normalized['source_provenance']['public_criterion_protocol'] = public_criterion_protocol
    if observed_source:
        normalized['source_provenance'].update(
            generation_semantic_protocol=generation_semantic_protocol,
            source_anchor_contract_mismatch=source_score != expected_source_score,
            source_anchor_declaration={'expected_score_from_outcome': expected_source_score,
                                       'generator_claimed_source_score': source_score,
                                       'matches_expected': source_score == expected_source_score,
                                       'independent_measurement': False})
    return normalized, None


def _openrouter_json(key: str, model: str, system: str, user: str, settings: dict, schema: dict, name: str, role: str) -> tuple[str, dict]:
    """One Rubric-Generator call with the pinned settings; raises RubricOutputIncomplete unless it finished normally."""
    from src.prompt_opt import harness as H
    from src.common.llm_provider import model_matches, key_env
    provider = settings.get('api_provider', 'openrouter')
    key_env(provider)
    route = {'api_provider': provider} if provider != 'openrouter' else {}
    extra = {}
    if settings.get('reasoning_effort'):
        extra.update({'reasoning_effort': settings['reasoning_effort']} if provider == 'openai' else
                     {'reasoning': {'effort': settings['reasoning_effort']}})
    if provider == 'openrouter' and settings.get('strict_provider_parameters'):
        extra['provider'] = {'require_parameters': True, 'allow_fallbacks': False}
    if settings.get('response_format') == 'json_schema_strict':
        extra['response_format'] = {'type': 'json_schema', 'json_schema': {'name': name, 'strict': True, 'schema': schema}}
    meta: dict = {}
    if not extra and provider == 'openrouter':                       # legacy settings: the plain call, as before
        return with_role(role, lambda: H.doctor_call(key, model, system, user, settings['max_tokens'], settings['temperature'])), meta
    raw = with_role(role, lambda: H.doctor_call(key, model, system, user, settings['max_tokens'], settings['temperature'], extra=extra, meta=meta, **route))
    if (settings.get('strict_provider_parameters') or provider == 'openai') and not model_matches(model, meta.get('model'), provider):
        raise RuntimeError(f'{role}: returned model {meta.get("model")!r} differs from pinned {model!r}; no substitution allowed')
    if settings.get('strict_provider_parameters') and settings.get('reasoning_effort') == 'none':
        reasoning_tokens = ((meta.get('usage') or {}).get('completion_tokens_details') or {}).get('reasoning_tokens')
        if reasoning_tokens is not None and reasoning_tokens > 0:
            raise RuntimeError(f'{role}: provider reported reasoning tokens despite reasoning effort none')
    if meta.get('finish_reason') not in (None, 'stop'):
        raise RubricOutputIncomplete(f"{model} stopped with finish_reason={meta.get('finish_reason')} "
                                     f"(native {meta.get('native_finish_reason')}, usage {meta.get('usage')}) before a complete {name}")
    return raw, meta


def api_rubric_generator(key: str, model: str, settings: dict | None = None, context_addendum: str | None = None):
    """CREATE: trajectory + feedback -> one rubric. With strict-schema settings a non-parsing output is an operational error."""
    from src.prompt_opt import harness as H
    from src.online_rl.tree.posthoc import trajectory_message
    settings = settings or {'max_tokens': H.GEN_MAX_TOKENS, 'temperature': H.GEN_TEMPERATURE, 'reasoning_effort': None, 'response_format': None}
    output_contract = settings.get('output_contract')
    from src.online_rl.tree.feedback_grounding import validate_protocol, extend_schema
    feedback_check_protocol = validate_protocol(settings.get('feedback_check_protocol'))
    alignment_protocol = settings.get('instruction_alignment_protocol')
    public_protocol = settings.get('public_criterion_protocol')
    semantic_protocol = settings.get('generation_semantic_protocol')
    if semantic_protocol is not None and (semantic_protocol not in GENERATION_SEMANTIC_VERSIONS or public_protocol is None):
        raise ValueError('Unknown or incompatible generation semantic protocol')
    if public_protocol is not None:
        from src.judges.rubric_schema import PUBLIC_CRITERION_PROTOCOL
        if (public_protocol != PUBLIC_CRITERION_PROTOCOL or output_contract != 'trajectory_analysis_candidate_v2_with_intervention'
                or not settings.get('local_schema_file')):
            raise ValueError('Public criterion protocol requires a joint output contract and pinned local schema')
    if alignment_protocol is not None:
        from src.online_rl.tree.instruction_audit import ANALYSIS_PROTOCOL
        if (alignment_protocol != ANALYSIS_PROTOCOL or output_contract != 'trajectory_analysis_candidate_v2_with_intervention'
                or not settings.get('local_schema_file')):
            raise ValueError('Instruction alignment analysis requires a joint output contract and pinned local schema')
    if feedback_check_protocol is not None and output_contract not in ('trajectory_analysis_candidate_v2', 'trajectory_analysis_candidate_v2_with_intervention'):
        raise ValueError('rubric feedback_check_protocol requires a trajectory analysis output contract')
    schema = (FULL_TRAJECTORY_RUBRIC_WITH_INTERVENTION_JSON_SCHEMA if output_contract == 'trajectory_analysis_candidate_v2_with_intervention' else
              FULL_TRAJECTORY_RUBRIC_JSON_SCHEMA if output_contract == 'trajectory_analysis_candidate_v2' else
              OBSERVED_RUBRIC_JSON_SCHEMA if context_addendum else RUBRIC_JSON_SCHEMA)
    if feedback_check_protocol is not None: schema = extend_schema(schema)
    from src.online_rl.integrated.records import sha256_json
    local_schema = None
    if settings.get('local_schema_file'):
        from pathlib import Path
        import hashlib
        path = Path(settings['local_schema_file'])
        if not path.is_absolute(): path = Path(__file__).resolve().parents[3] / path
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != settings.get('local_schema_sha256'):
            raise ValueError('Rubric local schema hash mismatch')
        local_schema = json.loads(payload)
        # Local validation is independent of provider-side response_format.
        import jsonschema
        jsonschema.Draft202012Validator.check_schema(local_schema)
        schema = local_schema
    if semantic_protocol is not None:
        source_schema = schema['properties']['analysis']['properties']['source_score']
        if source_schema.get('type') != 'integer' or source_schema.get('enum') != [0, 1, 2]:
            raise ValueError('Generation semantic protocol requires observed source_score 0/1/2 schema')
        if schema['properties']['candidate_rubric']['properties']['criterion_version'].get('const') != GENERATION_SEMANTIC_VERSIONS[semantic_protocol]:
            raise ValueError('Generation semantic protocol requires its new criterion version schema')
    turn_policy = settings.get('selected_turn_policy', 'feedback_locus_v1')
    if turn_policy not in ('feedback_locus_v1', 'generator_v1'):
        raise ValueError(f'Unknown rubric selected_turn_policy: {turn_policy}')
    settings = {**settings, 'effective_output_schema_sha256': sha256_json(schema)}
    def system(prompt): return prompt + '\n\n' + context_addendum if context_addendum else prompt

    def normalize(raw, parsed, err, tin):
        if output_contract not in ('trajectory_analysis_candidate_v2', 'trajectory_analysis_candidate_v2_with_intervention') or err:
            return parsed, err
        if local_schema is not None:
            import jsonschema
            try: jsonschema.validate(parsed, local_schema)
            except jsonschema.ValidationError as exc: return None, f'{exc.json_path}: {exc.message}'
        # Supplied schemas allow canonical numeric string IDs. Preserve raw JSON;
        # normalize only the locally validated, speaker-scoped transcript IDs.
        import copy, re
        parsed = copy.deepcopy(parsed)
        analysis = parsed.get('analysis')
        if isinstance(analysis, dict):
            owners = [(analysis, 'selected_agent_turn_id')]
            spans = analysis.get('evidence_spans')
            owners += [(s, 'turn_id') for s in spans if isinstance(s, dict)] if isinstance(spans, list) else []
            for owner, field in owners:
                value = owner.get(field)
                if isinstance(value, str) and re.fullmatch(r'[1-9][0-9]*', value): owner[field] = int(value)
        normalized, error = normalize_trajectory_analysis_output(parsed, err, tin,
            require_intervention=output_contract == 'trajectory_analysis_candidate_v2_with_intervention',
            enforce_feedback_locus=turn_policy != 'generator_v1',
            feedback_check_protocol=feedback_check_protocol,
            instruction_alignment_protocol=alignment_protocol,
            public_criterion_protocol=public_protocol,
            generation_semantic_protocol=semantic_protocol)
        if normalized is not None and turn_policy == 'generator_v1':
            normalized['source_provenance']['selected_turn_policy'] = turn_policy
        return normalized, error

    def generate(prompt_text, tin):
        from dataclasses import replace
        user = trajectory_message(replace(tin, target_reply_id=None) if turn_policy == 'generator_v1' else tin)
        raw, meta = _openrouter_json(key, model, system(prompt_text), user, settings, schema, 'rubric', 'rubric_generator')
        parsed_output, err = H.parse_rubric(raw)
        rubric, err = normalize(raw, parsed_output, err, tin)
        raw_attempts = [raw]; usages = [meta.get('usage')]
        # Preserve the fixed prompt verbatim. If its output violates deterministic provenance
        # checks, issue one bounded correction call with the same system prompt and explicit
        # field-level feedback; never silently coerce model output into a rubric. This correction
        # also applies when provider-side JSON Schema is disabled because a provider may omit
        # required nested analysis fields despite claiming schema support.
        if err and output_contract in ('trajectory_analysis_candidate_v2', 'trajectory_analysis_candidate_v2_with_intervention'):
            source_contract = (
                'analysis.source_score is the generator\'s evidence-grounded observation (integer 0, 1 or 2), '
                'not the outcome-derived expected anchor. Preserve a supported partial score 1; never change '
                'M, C, the score boundaries, source quotes or the claimed score merely to make failure become 0 '
                'or success become 2. A mismatch is a substantive Gate issue, not a JSON-format error. '
                'score_0_basis remains counterfactual_removal for success or observed_problem for failure. '
                if semantic_protocol is not None else
                'For success, source_score must be integer 2 and score_0_basis must be counterfactual_removal. '
                'For failure, source_score must be integer 0 and score_0_basis must be observed_problem. ')
            correction = (f'{user}\n\nYour previous JSON failed a deterministic validation: {err}. '
                          'Correct only the invalid output fields while preserving the same grounded candidate rubric. '
                          'The source outcome is given above. ' + source_contract +
                          'source_outcome, source IDs, evidence quotes, and selected Agent reply ID '
                          'must exactly reflect the supplied input. For the intervention contract, also return a substantive '
                          'intervention_instruction that implements only the source-outcome-selected score behavior and contains no answer or outcome objective. '
                          'Do not omit analysis fields, substitute empty analysis, '
                          'or infer missing fields from the supplied locus. Return the complete required JSON object only.\n\n'
                          'Previous JSON:\n' + raw)
            raw, meta = _openrouter_json(key, model, system(prompt_text), correction, settings, schema, 'rubric', 'rubric_generator')
            raw_attempts.append(raw); usages.append(meta.get('usage'))
            parsed_output, err = H.parse_rubric(raw)
            rubric, err = normalize(raw, parsed_output, err, tin)
        if err and (settings.get('response_format') == 'json_schema_strict' or settings.get('strict_provider_parameters')): raise RubricOutputIncomplete(f'{model}: schema output did not parse after bounded correction: {err}')
        return {'raw': raw, 'raw_attempts': raw_attempts, 'rubric': rubric,
                'intervention_instruction': (parsed_output or {}).get('intervention_instruction') if output_contract == 'trajectory_analysis_candidate_v2_with_intervention' and not err else None,
                'parse_error': err,
                'usage': usages[-1], 'usage_attempts': usages}

    def repair(prompt_text, tin, rubric, stage1):
        """v2.1 source-consistency repair (one call): the same trajectory input (with its behavioral locus), the rubric that was
        No Match at that locus and the Stage-1 reason; returns a rewritten rubric under the same schema."""
        import json
        from src.online_rl.tree.posthoc import REPAIR_PRINCIPLE
        from src.online_rl.tree.agent_state import visible_rubric
        shown = visible_rubric(rubric)
        principle = REPAIR_PRINCIPLE
        if output_contract == 'trajectory_analysis_candidate_v2':
            principle = ('For this repair, change only candidate_rubric.applicability.user_side_context and '
                         'candidate_rubric.applicability.agent_side_state so both hold before the specified source-locus reply. '
                         'Repeat the revised strings exactly in all three score contexts. Preserve the rubric name and all three '
                         'behavior descriptions. Return the full trajectory-analysis JSON contract required by the system prompt.')
        elif context_addendum:
            principle = ('Change only the two visible applicability contexts so they hold BEFORE the source locus reply. '
                         'Repeat the revised contexts exactly at all three score levels. Preserve the name, required_behavior '
                         'and each level\'s behavior text exactly. Do not infer hidden beliefs or use the target reply to establish applicability.')
        user = (trajectory_message(tin) + '\n\nYour previous rubric for this trajectory:\n' + json.dumps(shown, ensure_ascii=False, indent=1) +
                f"\n\nAt the behavioral locus (Agent reply {tin.target_reply_id}) its applicability condition was judged NOT met: "
                f"{stage1.get('missing_requirement') or 'the situation before the locus reply does not satisfy the condition'}\n\n{principle}\n"
                'Return ONE rubric in the required JSON format.')
        raw, meta = _openrouter_json(key, model, system(prompt_text), user, settings, schema, 'rubric', 'rubric_generator')
        rubric2, err = H.parse_rubric(raw)
        rubric2, err = normalize(raw, rubric2, err, tin)
        if err and settings.get('response_format') == 'json_schema_strict': raise RubricOutputIncomplete(f'{model}: repair output did not parse: {err}')
        return {'raw': raw, 'rubric': rubric2, 'parse_error': err, 'usage': meta.get('usage')}
    generate.model = model; generate.settings = settings; generate.repair = repair
    if context_addendum:
        from src.online_rl.integrated.records import sha256_text
        generate.settings = {**settings, 'context_contract_sha256': sha256_text(context_addendum)}
    return generate


MATCH_CHECKS = ('same_construct', 'applicability_visible', 'same_required_behavior', 'same_direction', 'score_semantics_apply', 'no_new_premise')
MATCH_JSON_SCHEMA = {'type': 'object', 'additionalProperties': False, 'required': ['decision', 'rubric_id', 'checks', 'rationale'],
                     'properties': {'decision': {'type': 'string', 'enum': ['REUSE_EXISTING', 'CREATE_NEW']}, 'rubric_id': {'type': 'string'},
                                    'checks': {'type': 'object', 'additionalProperties': False, 'required': list(MATCH_CHECKS),
                                               'properties': {c: {'type': 'boolean'} for c in MATCH_CHECKS}},
                                    'rationale': {'type': 'string'}}}
from src.online_rl.tree.member_intervention import MEMBER_INTERVENTION_VERSION
MEMBER_MATCH_JSON_SCHEMA = {
    **MATCH_JSON_SCHEMA,
    'required': [*MATCH_JSON_SCHEMA['required'], 'selected_agent_turn_id', 'intervention_instruction'],
    'properties': {**MATCH_JSON_SCHEMA['properties'],
                   'selected_agent_turn_id': {'type': ['integer', 'null'], 'minimum': 1},
                   'intervention_instruction': {'type': ['string', 'null'], 'minLength': 1, 'pattern': r'\S'}},
}
MEMBER_MATCH_INSTRUCTIONS = """
JOINT MEMBER INTERVENTION CONTRACT (trajectory_member_instruction_v1):
This is a matching call. Do not generate, edit or replace a rubric. For REUSE_EXISTING,
keep the named canonical rubric's applicability, behavioral construct and score anchors
immutable. In this SAME response, select this member trajectory's own actual integer
Agent reply ID as selected_agent_turn_id and generate its own intervention_instruction.
Evaluate the existing applicability condition against the visible prefix BEFORE that
member reply. The feedback locus is reference evidence, not a forced selection. Do not
copy the canonical source trajectory's selected turn or its intervention instruction.
The instruction must realize exactly the existing canonical score-0 omission for a
success member, or exactly the existing canonical score-2 behavior for a failure member.
Do not change the canonical boundaries, add a new requirement, or provide a model answer.
Ground the requested operation in this member's question, choices and dialogue before
the selected reply. The instruction must not contain the target/source response's
answer, later dialogue, Doctor FINAL, feedback, gold/ground-truth information, outcome
labels, score labels, or a desired success/failure result. Outcome selects the 0/2
direction only; it is not part of the instruction shown to the Agent. Full-trajectory
evidence may establish same-reason membership but must never leak into the pre-reply
instruction. Return one fixed instruction that will be reused unchanged across the
three reruns at this member's selected turn; do not write three responses or instructions.
If no actual member turn satisfies the existing rubric, or an instruction would require
changing its construct, return CREATE_NEW. Both extra fields are required: REUSE_EXISTING
requires an integer selected_agent_turn_id >= 1 and a nonblank intervention_instruction;
CREATE_NEW requires selected_agent_turn_id = null and intervention_instruction = null.
"""
MEMBER_ALIGNMENT_INSTRUCTIONS = """
For each matched member, derive the minimum contribution M and completion C from
the immutable anchors before writing its guide. A score-0 guide must omit M itself;
retaining a comparative assertion while removing only its explanation is score 1
when that is what the anchors define. A score-2 guide performs M and C with an open
conclusion. Do not inject a factual classification of dates/metadata or a medical
premise that its pre-response context does not establish. Ask to check support
and limit inference when information is missing. Earlier explicit requests remain
eligible until visibly fulfilled, withdrawn, superseded or closed; absence of a
repeat request is not resolution. A later reflection cannot create a prior trigger.
"""
PUBLIC_MEMBER_ALIGNMENT_INSTRUCTIONS = """
PUBLIC M/C MATCH CONTRACT (public_behavior_mc_v1):
Read the frozen candidate's explicit required_behavior, minimum_performance (M),
completion_condition (C), criterion_version and all score anchors. Never derive
different boundaries from hidden analysis, a source outcome, or general ideals.
same_required_behavior requires the same concrete response operation and M;
score_semantics_apply requires the same M, C and complete 0/1/2 partition without
alteration. Reusing only a similar behavior label is insufficient. A changed
minimum, completion, applicability or score definition means CREATE_NEW, never
an in-place repair or inherited KEEP. Do not silently upgrade a legacy rubric
without public M/C fields when this public contract is enabled.
The member-specific score-0 guide must remove M, not merely C. The score-2 guide
must perform M and C, leaving the conclusion open. Do not introduce a medical
or metadata premise absent from the prefix, demand an unrelated operation, or
copy a source answer. Unresolved earlier requests may remain active without
repetition; later reflection cannot create a prior need. Keep the canonical
criterion_version and fields immutable. Instruction adherence remains a
diagnostic; this matching call does not certify a rerun effect.
"""
MATCH_CORRECTION_CONTRACT = 'same_input_match_schema_correction_v1'
MATCH_SINGLE_INSTRUCTIONS = """
SINGLE MATCH OUTPUT CONTRACT: This is only an assignment decision against the
provided frozen rubric bank. The generator/CREATE sections earlier in this
package describe other calls and are not the output contract for this call.
Return exactly the matcher JSON object requested here, not analysis or a new
candidate rubric. Use decision, rubric_id, all six boolean checks, and a concise
rationale. When the joint member contract applies, also return its required
selected_agent_turn_id and intervention_instruction fields. CREATE_NEW still
means no existing rubric safely matches; it is not a request to generate one.
Return complete JSON without Markdown or text outside the object.
"""


def api_rubric_matcher(key: str, model: str, settings: dict, context_addendum: str | None = None, match_batch_size: int = 1):
    """MATCH-vs-NEW: (prompt package, trajectory + current bank) -> {'decision', 'rubric_id', 'checks', 'rationale'} from the
    Rubric-Generator model. Joint generation also returns the reused member's own turn and instruction in this call;
    no rubric is rewritten. Malformed output is an operational error (RubricOutputIncomplete)."""
    import json, re
    if match_batch_size not in (1,8):raise ValueError('match_batch_size must be 1 or 8')
    member_interventions = settings.get('output_contract') == 'trajectory_analysis_candidate_v2_with_intervention'
    schema = MEMBER_MATCH_JSON_SCHEMA if member_interventions else MATCH_JSON_SCHEMA
    member_instructions = MEMBER_MATCH_INSTRUCTIONS
    public_protocol = settings.get('public_criterion_protocol')
    if public_protocol is not None:
        from src.judges.rubric_schema import PUBLIC_CRITERION_PROTOCOL
        if public_protocol != PUBLIC_CRITERION_PROTOCOL or not member_interventions:
            raise ValueError('Unknown or incompatible matcher public criterion protocol')
    if settings.get('instruction_alignment_protocol') is not None:
        from src.online_rl.tree.instruction_audit import ANALYSIS_PROTOCOL
        if settings['instruction_alignment_protocol'] != ANALYSIS_PROTOCOL or not member_interventions:
            raise ValueError('Unknown or incompatible matcher instruction alignment protocol')
        if public_protocol is None: member_instructions += '\n' + MEMBER_ALIGNMENT_INSTRUCTIONS
    if public_protocol is not None:
        member_instructions += '\n' + PUBLIC_MEMBER_ALIGNMENT_INSTRUCTIONS
    if member_interventions:
        from src.online_rl.integrated.records import sha256_text
        settings = {**settings, 'member_intervention_version': MEMBER_INTERVENTION_VERSION,
                    'member_intervention_prompt_sha256': sha256_text(member_instructions)}
    from src.online_rl.integrated.records import sha256_text
    settings = {**settings, 'match_correction_contract': MATCH_CORRECTION_CONTRACT,
                'match_single_contract_sha256': sha256_text(MATCH_SINGLE_INSTRUCTIONS)}

    def valid_decision(value):
        if not (isinstance(value, dict) and set(value) == set(schema['required']) and
                value['decision'] in ('REUSE_EXISTING', 'CREATE_NEW') and isinstance(value['rubric_id'], str) and
                isinstance(value['rationale'], str) and isinstance(value['checks'], dict) and
                set(value['checks']) == set(MATCH_CHECKS) and all(isinstance(v, bool) for v in value['checks'].values())):
            return False
        if not member_interventions:
            return True
        turn, instruction = value['selected_agent_turn_id'], value['intervention_instruction']
        if value['decision'] == 'CREATE_NEW':
            return turn is None and instruction is None
        return (isinstance(turn, int) and not isinstance(turn, bool) and turn >= 1 and
                isinstance(instruction, str) and bool(instruction.strip()))

    def request_decision(package, user, output_schema, name, validate):
        """One same-input schema correction; failure stays operational and uncached."""
        attempts = []
        request_user = user
        for attempt in range(2):
            raw, meta = None, {}
            try:
                raw, meta = _openrouter_json(key, model, package, request_user, settings,
                                            output_schema, name, 'rubric_matcher')
                try:
                    parsed = json.loads(re.sub(r'^```(?:json)?\s*|\s*```$', '', raw.strip()))
                except (json.JSONDecodeError, AttributeError) as exc:
                    raise RubricOutputIncomplete(f'{model}: matcher JSON schema parse error: {exc}') from exc
                value = validate(parsed)
            except RubricOutputIncomplete as exc:
                attempts.append({'attempt': attempt + 1, 'raw': raw, 'meta': meta, 'error': str(exc)})
                if attempt:
                    failure = RubricOutputIncomplete(f'{model}: matcher schema correction failed: {exc}')
                    failure.audit = {'contract': MATCH_CORRECTION_CONTRACT, 'system': package,
                                     'user': user, 'output_schema': output_schema, 'settings': settings,
                                     'attempts': attempts}
                    raise failure from exc
                request_user = (user + '\n\nMATCH JSON CORRECTION: The previous response did not satisfy the '
                                'output contract. Re-evaluate only the SAME original input and frozen bank. '
                                'Return one complete JSON object conforming to the schema below. Keep all '
                                'required member turn/instruction fields and exact supplied trajectory IDs. '
                                'Do not turn an output-format error into CREATE_NEW or invent missing evidence. '
                                'Keep rationales concise. No Markdown.\n' +
                                json.dumps({'schema': output_schema, 'validation_error': str(exc),
                                            'invalid_previous_output': raw}, ensure_ascii=False))
                attempts[-1]['correction_user'] = request_user
                continue
            attempts.append({'attempt': attempt + 1, 'raw': raw, 'meta': meta, 'error': None})
            result = {**value, 'usage': meta.get('usage')}
            if attempt:
                result['match_correction_audit'] = {'contract': MATCH_CORRECTION_CONTRACT,
                                                   'system': package, 'user': user,
                                                   'output_schema': output_schema, 'settings': settings,
                                                   'attempts': attempts}
            return result

    def decide(package, user):
        if not member_interventions: package += '\n\n' + MATCH_SINGLE_INSTRUCTIONS
        if context_addendum: package += '\n\n' + context_addendum
        if member_interventions: package += '\n\n' + member_instructions
        if member_interventions: package += '\n\n' + MATCH_SINGLE_INSTRUCTIONS
        def validate(d):
            if not valid_decision(d):
                raise RubricOutputIncomplete(f'{model}: match decision does not follow the schema')
            return d
        return request_decision(package, user, schema, 'rubric_match_decision', validate)
    def batch(package, items, bank):
        """One frozen bank and up to eight independent trajectory decisions; exact ID coverage required."""
        from src.online_rl.tree.rubric_bank import bank_view
        from src.online_rl.tree.posthoc import trajectory_message
        from dataclasses import replace
        if not 1 <= len(items) <= 8 or len({t.leaf_id for t in items}) != len(items):
            raise ValueError('Batch matcher requires 1..8 unique trajectory IDs')
        batch_schema={'type':'object','additionalProperties':False,'required':['decisions'],
                'properties':{'decisions':{'type':'array','items':{
                    'type':'object','additionalProperties':False,'required':['trajectory_id',*schema['required']],
                    'properties':{'trajectory_id':{'type':'string'},**schema['properties']}}}}}
        if context_addendum:package += '\n\n'+context_addendum
        package += ('\n\nBATCH MATCH CONTRACT: Judge each numbered trajectory independently against the SAME frozen bank. '
                    'Return {"decisions": [...]} with exactly one entry for each supplied trajectory_id. '
                    'Each entry has trajectory_id, decision, rubric_id, all six boolean checks, and a concise rationale '
                    'identifying its behavioral locus and quoting the relevant visible dialogue. Never reference another '
                    'trajectory as evidence. CREATE_NEW means no existing rubric safely matches; do not invent rubric IDs '
                    'or create rubrics in this call. There is no gold answer. Return only the batch JSON schema.')
        if member_interventions: package += '\n\n' + member_instructions
        user=json.dumps({'frozen_bank':bank_view(bank),'trajectories':[
            {'trajectory_id':t.leaf_id,
             **({} if member_interventions else {'behavioral_locus':t.target_reply_id}),
             'input':trajectory_message(replace(t, target_reply_id=None) if member_interventions else t)} for t in items]},ensure_ascii=False)
        def validate(d):
            rows=d.get('decisions') if isinstance(d,dict) and set(d)=={'decisions'} else None
            expected={t.leaf_id for t in items}
            if not isinstance(rows,list) or len(rows)!=len(expected):raise RubricOutputIncomplete('Batch match ID coverage differs')
            out={}
            for row in rows:
                if not isinstance(row,dict) or set(row)!={'trajectory_id',*schema['required']}:raise RubricOutputIncomplete('Malformed batch match row')
                tid=row['trajectory_id']
                if not isinstance(tid,str) or tid not in expected or tid in out:raise RubricOutputIncomplete('Missing, duplicate or unknown trajectory ID')
                value={k:row[k] for k in schema['required']}
                if not valid_decision(value):raise RubricOutputIncomplete('Malformed batch match decision')
                if member_interventions and value['decision'] == 'REUSE_EXISTING':
                    tin = next(t for t in items if t.leaf_id == tid)
                    agent_count = sum(turn[0] == 'agent' for turn in tin.dialogue)
                    if value['selected_agent_turn_id'] > agent_count:
                        raise RubricOutputIncomplete('Member selected_agent_turn_id is not an actual Agent reply ID')
                out[tid]=value
            return {'decisions':out}
        return request_decision(package, user, batch_schema, 'rubric_match_batch', validate)
    decide.model = model; decide.settings = settings
    decide.batch=batch;batch.model=model;batch.settings=settings
    decide.batch_size=match_batch_size
    decide.member_interventions = batch.member_interventions = member_interventions
    decide.member_intervention_version = batch.member_intervention_version = MEMBER_INTERVENTION_VERSION if member_interventions else None
    if context_addendum:
        from src.online_rl.integrated.records import sha256_text
        decide.settings = {**settings, 'context_contract_sha256': sha256_text(context_addendum)}
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
    from src.online_rl.tree.agent_state import flat_rubric, state_conditioned
    q = applicability_questions([flat_rubric(r) for r in rubrics])
    return {r['rubric_id']: {'rubric': r['name'], 'applicability_condition': flat_rubric(r)['applicability_condition'],
                            **({'applicability': r['applicability']} if state_conditioned(r) else {}),
                            'instructions': q[r['rubric_id']]['instructions'] +
                            (' Evaluate the supplied Agent-side state condition as written, using only evidence observable before the target reply. '
                             'Do not add a requirement for prior Agent behavior unless that specific condition says prior behavior is required. '
                             'Never infer hidden belief. Do not use the unseen target reply or future dialogue.'
                             if state_conditioned(r) else '')}
            for r in rubrics}


def stage2_items(rubrics: list[dict]) -> dict:
    """Level items: rubric name, required behavior and the rubric's own score anchors (no applicability condition)."""
    from src.judges.rubric_scoring import behavior_questions
    from src.online_rl.tree.agent_state import flat_rubric
    q = behavior_questions([flat_rubric(r) for r in rubrics])
    return {r['rubric_id']: {'rubric': r['name'], 'required_behavior': r['required_behavior'], 'score_anchors': {k: r['scores'][k] for k in ('0', '1', '2')},
                             'instructions': q[r['rubric_id']]['instructions'] + ' ' + STAGE2_NOTE} for r in rubrics}


def two_stage_structured_scorer(client, observed_context: bool = False, response_batch_size: int = 4):
    """Scorer v2 (Luna, hard labels with a one-sentence basis per decision). Stage 1 sees the case, the dialogue BEFORE the
    target reply and each rubric's name + applicability condition, and returns applicable + evidence_basis /
    missing_requirement. Stage 2 (applicable rubrics only) adds the target reply and sees name + required behavior + the
    rubric's own 0/1/2 anchors, and returns level + anchor_basis. Neither stage sees outcome, gold, later turns or rubric
    effectiveness. Malformed output: one same-model repair (client.judge), then an operational error."""
    from src.online_rl.tree.agent_state import rubric_problems, state_conditioned
    from src.online_rl.integrated.records import sha256_text
    if response_batch_size not in range(1, 9): raise ValueError('response_batch_size must be between 1 and 8')

    def validate(rubrics):
        for r in rubrics:
            if rubric_problems(r): raise ValueError(f"invalid visible rubric {r.get('rubric_id')}")
            if state_conditioned(r) and not observed_context: raise ValueError('observed-context rubric requires the matching scorer protocol')

    def before_state(view, prefix):
        latest = next((t['text'] for t in reversed(prefix) if t.get('speaker') == 'doctor'), '')
        before = {'case_context': {'question': view['question'], 'options': view['options']}, 'dialogue_before_target_reply': prefix}
        return {**before, 'latest_physician_message': latest}

    def behavior_state(view, prefix, reply=None):
        # Preserve the original Stage-2 contract: unlike applicability, behavior
        # scoring sees the dialogue and case plus the target reply, without the
        # redundant latest-physician-message field.
        state = {'case_context': {'question': view['question'], 'options': view['options']},
                 'dialogue_before_target_reply': prefix}
        if reply is not None: state['target_agent_reply'] = reply
        return state

    def review_state(view, prefix, rubrics):
        validate(rubrics)
        app, _ = with_role('rubric_scorer', lambda: client.judge('applicability', before_state(view, prefix), stage1_items(rubrics), STAGE1_FIELDS,
                                                                  'tree_response_rubric_score_applicability'))
        if not isinstance(app, dict) or set(app) != {r['rubric_id'] for r in rubrics}:
            raise ValueError('Applicability response must contain exactly the requested rubric IDs')
        for row in app.values():
            if not isinstance(row, dict) or type(row.get('applicable')) is not bool:
                raise ValueError('Applicability must be a JSON boolean; malformed results cannot become NA')
        return {rid: {'applicable': row['applicable'], 'stage1': row} for rid, row in app.items()}

    def score_behavior(view, prefix, rubrics, reply, matches=None):
        if not rubrics: return {}
        validate(rubrics)
        state = behavior_state(view, prefix, reply)
        lev, _ = with_role('rubric_scorer', lambda: client.judge('level', state, stage2_items(rubrics), STAGE2_FIELDS,
                                                                  'tree_response_rubric_score_behavior'))
        return {rid: {'applicable': True, 'probabilities': None, 'hard': int(row['level']), 'stage2': row}
                for rid, row in lev.items()}

    def score_behavior_batch(view, prefix, rubrics, responses):
        if not responses or not rubrics: return {}
        validate(rubrics)
        questions = stage2_items(rubrics)
        if not callable(getattr(client, 'judge_batch', None)):
            return {row['response_id']: score_behavior(view, prefix, rubrics, row['reply']) for row in responses}
        batch_rows = [{'response_id': row['response_id'], 'target_agent_reply': row['reply']}
                      for row in responses]
        values, _ = with_role('rubric_scorer', lambda: client.judge_batch('level', behavior_state(view, prefix), batch_rows,
            questions, STAGE2_FIELDS, 'tree_response_rubric_score_behavior_batch'))
        return {response_id: {rid: {'applicable': True, 'probabilities': None, 'hard': int(item['level']), 'stage2': item}
                              for rid, item in rows.items()} for response_id, rows in values.items()}

    def score(view, prefix, rubrics, reply):
        n0 = len(getattr(client, 'log', []))
        app = review_state(view, prefix, rubrics)
        on = [r for r in rubrics if app[r['rubric_id']]['applicable']]
        lev = score_behavior(view, prefix, on, reply) if on else {}
        out = {r['rubric_id']: {'applicable': r['rubric_id'] in lev, 'probabilities': None,
                                'hard': lev[r['rubric_id']]['hard'] if r['rubric_id'] in lev else None,
                                'stage1': app[r['rubric_id']]['stage1'],
                                'stage2': lev.get(r['rubric_id'], {}).get('stage2')} for r in rubrics}
        out['_decisions'] = getattr(client, 'log', [])[n0:]
        return out

    score.model = getattr(client, 'model', None)
    score.protocol = sha256_text(f"{getattr(client, 'protocol_sha256', None)}|{SCORER_V2}|{STAGE2_NOTE}|response_batch_v1:{response_batch_size}")
    score.review_state = review_state
    score.score_behavior = score_behavior
    score.score_behavior_batch = score_behavior_batch
    score.response_batch_size = response_batch_size
    score.review_identity = {'protocol': sha256_text(f"{getattr(client, 'protocol_sha256', None)}|{SCORER_V2}|stage1_only")}
    applicability = lambda view, prefix, rubrics: review_state(view, prefix, rubrics)
    applicability.model = score.model
    applicability.protocol = score.review_identity['protocol']
    if observed_context:
        from src.online_rl.tree.agent_state import SCHEMA
        score.protocol = sha256_text(score.protocol + '|' + SCHEMA)
        applicability.protocol = sha256_text(applicability.protocol + '|' + SCHEMA)
        score.review_identity = {'protocol': applicability.protocol}
    score.applicability = applicability
    return score


def posthoc_gates(client, cases: dict, observed_context: bool = False, protocol: str = 'legacy_v1',
                  rubric_quote_protocol: str | None = None):
    """Post-hoc rubric gate: schema, deterministic leakage / answer copy (the gold is used only inside this check), and the
    Jev answer_presupposition and new_domain_premise questions on the source discussion without the FINAL. 'observable' is
    recorded but does not block."""
    if protocol in ('public_mc_gate_v1', 'public_mc_gate_v2', 'public_mc_gate_v3'):
        from src.online_rl.tree.quality_gate import public_mc_posthoc_gate
        return public_mc_posthoc_gate(client, cases, observed_context=observed_context,
                                     rubric_quote_protocol=rubric_quote_protocol, protocol=protocol)
    if protocol == 'evidence_gate_v2':
        from src.online_rl.tree.quality_gate import evidence_posthoc_gate
        quote_options = {'rubric_quote_protocol': rubric_quote_protocol} if rubric_quote_protocol is not None else {}
        return evidence_posthoc_gate(client, cases, observed_context=observed_context, **quote_options)
    if protocol != 'legacy_v1':
        raise ValueError(f'Unknown judges.gate_protocol: {protocol!r}')
    import json
    from src.judges.rubric_scoring import validate_visible_rubric, _batched_jev_decisions
    from src.online_rl.tree.records import to_dialogue
    from src.prompt_opt import harness as H
    from src.prompt_opt import harness_v11 as V
    from src.online_rl.tree.agent_state import rubric_problems, flat_rubric, visible_rubric
    Q = json.loads((H.ROOT / 'prompts/prompt_opt/rubric_validator_questions_v1_2.json').read_text())['questions']
    questions = {k: Q[k] for k in POSTHOC_BLOCKING + POSTHOC_UNRESOLVED}

    def gate(tin, rubric) -> dict:
        case = cases[tin.case_id]; g = {'schema': None, 'leakage_deterministic': None, **{k: None for k in questions}, 'reasons': [], 'passed': False,
                                        'unresolved': list(POSTHOC_UNRESOLVED)}
        probs = rubric_problems(rubric)
        g['schema'] = not probs
        if probs: g['reasons'].append(f'schema: {probs}'); return g
        leak = H.deterministic_leakage(flat_rubric(rubric), case); g['leakage_deterministic'] = leak
        if leak: g['reasons'].append(f'deterministic leakage: {leak}')
        discussion = [d for d in to_dialogue(tin.dialogue) if d['stage'] != 'FINAL']
        state = {'case_context': {'question': tin.question, 'options': dict(tin.options)}, 'dialogue_before_target_turn': discussion,
                 'candidate_rubric': visible_rubric(rubric)}
        n0 = len(getattr(client, 'log', []))
        ans, _ = with_role('rubric_gate', lambda: _batched_jev_decisions(client, state, questions, purpose='tree_posthoc_rubric_gate', max_rubrics_per_call=None))
        g['decisions'] = getattr(client, 'log', [])[n0:]
        for k in questions: g[k] = ans[k]['value'] == 'Y'
        for k in POSTHOC_BLOCKING:
            if g[k]: g['reasons'].append(k.replace('_', ' '))
        g['passed'] = g['schema'] and not leak and not any(g[k] for k in POSTHOC_BLOCKING)
        return g
    gate.model = getattr(client, 'model', None); gate.protocol = getattr(client, 'protocol_sha256', None)
    if observed_context:
        from src.online_rl.integrated.records import sha256_text
        from src.online_rl.tree.agent_state import SCHEMA
        gate.protocol = sha256_text(str(gate.protocol) + '|' + SCHEMA)
    return gate


class PolicyInterventions:
    """Intervention reply and continuation from the same policy; expose applicability plus only the selected behavior."""

    def __init__(self, policy, score0_template: str, score2_template: str | None = None, intervention_instruction_template: str | None = None,
                 *, guided_protocol: str | None = None, agent_system: str | None = None):
        self.policy, self.score0_template, self.score2_template = policy, score0_template, score2_template
        self.intervention_instruction_template = intervention_instruction_template
        if guided_protocol not in (None, 'joint_instruction_v2_1'):
            raise ValueError(f'Unknown guided protocol: {guided_protocol}')
        if guided_protocol and (not isinstance(agent_system, str) or not agent_system.strip()):
            raise ValueError('Joint intervention requires an explicit common Agent system')
        self.guided_protocol, self.agent_system = guided_protocol, agent_system

    def template_for(self, level, instruction=None):
        if self.guided_protocol or instruction is None:
            return self.score0_template if level == '0' else self.score2_template
        return self.intervention_instruction_template

    def cache_identity(self, level):
        from src.online_rl.integrated.records import sha256_json, sha256_text
        policy = self.policy
        return {'protocol': self.guided_protocol or 'legacy_guided_v1',
                'system_sha256': sha256_text(self.agent_system or getattr(policy, 'agent_system', '')),
                'score_template_sha256': sha256_text(self.template_for(level) or ''),
                'rollout_settings_sha256': sha256_json(getattr(policy, 'cfg', {}).get('rollout', {})),
                'policy_config_sha256': sha256_json(getattr(policy, 'cfg', {}).get('policy', {})),
                'doctor_system_sha256': sha256_text(getattr(policy, 'doctor_system', '')),
                'generation_args': dict(vars(policy.args)) if hasattr(policy, 'args') else None}

    def write(self, case, prefix, turn, rubric, level, seed, *, intervention_instruction=None):
        import json
        from src.prompt_opt import harness as H
        from src.online_rl.tree.agent_state import intervention_rubric
        if level not in ('0', '2'): raise ValueError(f'intervention level must be 0 or 2, got {level!r}')
        if self.guided_protocol and intervention_instruction is None:
            raise ValueError('Joint intervention cannot reuse a legacy rubric without its saved instruction')
        if intervention_instruction is not None:
            if not isinstance(intervention_instruction, str) or not intervention_instruction.strip():
                raise ValueError('intervention_instruction must be a substantive non-empty string')
            template = self.template_for(level, intervention_instruction)
            if template is None: raise ValueError('generated intervention_instruction requires its configured Agent template')
            if template.count('{intervention_instruction}') != 1 or template.replace('{intervention_instruction}', '').count('{'):
                raise ValueError('Agent intervention template must contain only one {intervention_instruction} field')
            if self.guided_protocol:
                system = self.agent_system + '\n\n' + template.format(intervention_instruction=intervention_instruction)
                visible_prefix = [{k: d[k] for k in ('speaker', 'stage', 'round', 'text') if k in d} for d in prefix]
                user = json.dumps({'scenario': case.get('scenario', ''), 'question': case['question'],
                    'options': dict(case['options']), 'dialogue_before_target_reply': visible_prefix,
                    'next_agent_reply_id': turn}, ensure_ascii=False)
                return self.policy.respond_with_system(system, user, seed)
            user = self.policy.agent_user_prompt(case, prefix, turn) + '\n\n' + template.format(intervention_instruction=intervention_instruction)
            return self.policy.agent(user, seed)
        template = self.score0_template if level == '0' else self.score2_template
        if template is None:
            if level == '0': raise ValueError('Score-0 intervention requires a configured guided-response template')
            template = H.GUIDED_TEMPLATE
        visible = intervention_rubric(rubric, level)
        user_prompt = self.policy.agent_user_prompt(case, prefix, turn)
        if template.count('{intervention_instruction}') == 1 and not template.replace('{intervention_instruction}', '').count('{'):
            selected = visible['scores'][level]['behavior']
            body = template.format(intervention_instruction=selected)
        else:
            body = template.format(rubric_json=json.dumps(visible, ensure_ascii=False))
        return self.policy.agent(user_prompt + '\n\n' + body, seed)

    def continue_(self, case, prefix, turn, text, seed, turn_cache=None):
        r = with_role('doctor_rerun', lambda: self.policy.continue_from(case, prefix, turn, text, seed, **({'turn_cache': turn_cache} if turn_cache else {})))
        return {'final_option': r['final_option'], 'outcome': r['correct'], 'termination_reason': r['termination_reason'], 'agent_turns': r['agent_turns']}
