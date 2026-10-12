"""Real model bindings for the anchored group protocol; no applicability or Gate calls."""
import json

from src.online_rl.doctor import doctor_seed
from src.online_rl.integrated.records import sha256_json
from src.online_rl.tree.backends import PolicyAgent, RoleDoctor, _generator_json, with_role
from src.online_rl.tree.group_learning import (GroupEnv, GroupTrainer, candidate_from_source_outcome,
                                             config_problems, quote_in_reply, validate_candidate)


def obj(properties):
    return {'type': 'object', 'properties': properties, 'required': list(properties), 'additionalProperties': False}


TEXT = {'type': 'string', 'minLength': 1}
SOURCE = {'type': 'string', 'enum': ['S0', 'S1', 'S2', 'S3']}
OBSERVATION = obj({'response_id': SOURCE, 'quote': TEXT, 'experience': TEXT})
REFLECTION_SCHEMA = obj({'reflection': TEXT})
COMPARISON_SCHEMA = obj({key: {'type': 'array', 'maxItems': 4, 'items': OBSERVATION} for key in ('helpful', 'harmful')})
CANDIDATE_SCHEMA = obj({'name': TEXT, 'kind': {'type': 'string', 'enum': ['helpful', 'harmful']},
    'source_response_id': SOURCE, 'state': obj({'partner': TEXT, 'assistant': TEXT}),
    'criteria': obj({key: TEXT for key in ('0', '1', '2')}), 'b': TEXT, 'b_star': TEXT, 'evidence_quote': TEXT})
CANDIDATES = {'type': 'array', 'maxItems': 1, 'items': CANDIDATE_SCHEMA}
DECISIVE_CANDIDATE_SCHEMA = obj({'name': TEXT, 'source_response_id': SOURCE,
    'state': obj({'partner': TEXT, 'assistant': TEXT}),
    'criteria': obj({key: TEXT for key in ('observed', 'middle', 'contrast')}),
    'b': TEXT, 'b_star': TEXT, 'evidence_quote': TEXT})
GENERATION_SCHEMA = obj({'candidates': {'type': 'array', 'minItems': 0, 'maxItems': 1,
                                      'items': DECISIVE_CANDIDATE_SCHEMA}})
JUDGE_SCHEMA = obj({'scores': {'type': 'array', 'minItems': 4, 'maxItems': 4,
    'items': obj({'response_id': {'type': 'string', 'enum': ['A0', 'A1', 'A2', 'A3']},
                  'rubric_scores': {'type': 'array', 'minItems': 1, 'maxItems': 2,
                                    'items': {'type': 'integer', 'enum': [0, 1, 2]}}})}})
RANK_JUDGE_SCHEMA = obj({'ranking': {'type': 'array', 'minItems': 1, 'maxItems': 4,
    'items': obj({'response_ids': {'type': 'array', 'minItems': 1, 'maxItems': 4,
                                 'uniqueItems': True,
                                 'items': {'type': 'string', 'enum': ['A0', 'A1', 'A2', 'A3']}},
                  'reason': TEXT})}})


def parsed_envelope(raw, schema, meta=None, request=None):
    import jsonschema
    try:
        parsed = json.loads(raw)
        jsonschema.validate(parsed, schema)
    except (ValueError, TypeError, jsonschema.ValidationError) as exc:
        return {'valid': False, 'parsed': None, 'raw': raw, 'error': str(exc), 'meta': meta, 'request': request}
    return {'valid': True, 'parsed': parsed, 'raw': raw, 'meta': meta, 'request': request}


def make_callbacks(cfg, policy, generation_key, judge_key, root):
    spec = cfg['group_learning']
    prose = spec.get('reflection_format') == 'prose_json_schema_v2'
    automatic_direction = spec.get('candidate_direction') == 'source_outcome_v1'
    rank_reward = spec.get('reward_mode') == 'rubric_rank_v1'
    candidate_schema = {**CANDIDATES, 'maxItems': spec['max_rubrics_per_group']}
    generation_schema = (obj({'comparative_reflection': COMPARISON_SCHEMA, 'candidates': candidate_schema})
                         if prose else obj({'candidates': candidate_schema}))
    prompts = {role: (root / spec[role + '_prompt_file']).read_text() for role in ('reflection', 'generator', 'judge')}
    generation = cfg['rubric']['generation']
    settings = {key: generation.get(key) for key in ('max_tokens', 'temperature', 'reasoning_effort')}
    settings.update(api_provider=generation.get('api_provider', cfg.get('api_provider', 'openrouter')),
                    response_format='json_schema_strict')
    judge_settings = {'max_tokens': cfg['judges']['scorer_max_tokens'], 'temperature': 0.0,
                      'reasoning_effort': cfg['judges'].get('reasoning_effort', 'low'),
                      'response_format': 'json_schema_strict',
                      'api_provider': cfg['judges'].get('api_provider', cfg.get('api_provider', 'openrouter'))}

    def reflection_view(case, group):
        view = {'case': {key: case.get(key) for key in ('scenario', 'question', 'options')},
                'shared_prefix': group['prefix'], 'partner_state': group['partner_state'],
                'siblings': [{'response_label': f'Reply {index + 1}',
                              **{key: member[key] for key in ('response_id', 'reply', 'outcome', 'final_option')}}
                             for index, member in enumerate(group['members'])]}
        if spec['show_gold_to_reflection']: view['gold_diagnosis'] = case.get('ground_truth')
        return view

    def reflect(case, group):
        request = reflection_view(case, group)
        system = policy.doctor.system + '\n\n' + prompts['reflection']
        user = json.dumps(request, ensure_ascii=False)
        attempts = []
        for attempt in range(1 + spec.get('reflection_format_retries', 0)):
            seed = doctor_seed(cfg['schedule']['seed'], sha256_json(request), 'group_reflection', attempt)
            # _chat builds a fresh messages payload, with no persistent chat
            # history. Reflection is returned to this private callback only.
            raw = with_role('partner_group_reflection', lambda: policy.doctor._chat(
                user, seed, spec['reflection_max_tokens'], system,
                **({'format_schema': REFLECTION_SCHEMA} if prose else {})))
            envelope = parsed_envelope(raw, REFLECTION_SCHEMA if prose else COMPARISON_SCHEMA,
                                       {'model': policy.doctor.model, 'seed': seed}, request)
            if envelope['valid'] and prose and not envelope['parsed']['reflection'].strip():
                envelope.update(valid=False, error='Empty partner reflection')
            if envelope['valid'] and not prose:
                validate_comparison(envelope, envelope['parsed'], group)
            attempts.append({'seed': seed, 'raw': raw, 'valid': envelope['valid'], 'error': envelope.get('error')})
            if envelope['valid']:
                break
        envelope['attempts'] = attempts
        return envelope

    def generate(case, group, reflection):
        request = {**reflection_view(case, group), 'partner_reflection': reflection} if prose else {
            **reflection_view(case, group), 'comparative_reflection': reflection}
        schema = generation_schema
        if automatic_direction:
            eligible = [m['response_id'] for m in group['members'] if m['outcome'] in (0.0, 1.0)]
            if not eligible:
                return {'valid': True, 'parsed': {'candidates': []}, 'raw': None, 'request': request,
                        'no_candidate_reason': 'no_resolved_source_outcome', 'attempts': []}
            item_schema = obj({**DECISIVE_CANDIDATE_SCHEMA['properties'],
                               'source_response_id': {'type': 'string', 'enum': eligible}})
            schema = obj({'candidates': {'type': 'array', 'minItems': 0, 'maxItems': 1, 'items': item_schema}})
        attempts = []
        for attempt in range(1 + spec.get('generation_format_retries', 0)):
            effective_request = dict(request)
            if attempts:
                effective_request['format_retry'] = {
                    'instruction': 'Regenerate once, fixing the format/grounding error. Use the same evidence and output schema.',
                    'previous_error': attempts[-1]['error'], 'previous_output': attempts[-1]['raw']}
            raw, meta = _generator_json(generation_key, cfg['rubric']['generator_model'], prompts['generator'],
                                       json.dumps(effective_request, ensure_ascii=False), settings, schema,
                                       'anchored_group_decisive_candidate_v3' if automatic_direction else
                                       'anchored_group_comparison_candidates_v2' if prose else 'anchored_group_candidates',
                                       'group_rubric_generator')
            envelope = parsed_envelope(raw, schema, meta, effective_request)
            if envelope['valid']:
                if automatic_direction:
                    try:
                        candidates = [candidate_from_source_outcome(draft, group)
                                      for draft in envelope['parsed']['candidates']]
                        for candidate in candidates:
                            validate_candidate(candidate, group, reflection, direction_from_outcome=True)
                        envelope['parsed'] = {'candidates': candidates}
                        if not candidates:
                            envelope['no_candidate_reason'] = 'no_reflection_outcome_consistent_source'
                    except (ValueError, TypeError, KeyError) as exc:
                        envelope.update(valid=False, parsed=None, error=str(exc))
                elif prose:
                    validate_comparison(envelope, envelope['parsed']['comparative_reflection'], group)
            attempts.append({'raw': raw, 'meta': meta, 'valid': envelope['valid'], 'error': envelope.get('error')})
            if envelope['valid']:
                break
        envelope['attempts'] = attempts
        return envelope

    def judge(request):
        raw, meta = _generator_json(judge_key, cfg['judges']['scorer_model'], prompts['judge'],
                                   json.dumps(request, ensure_ascii=False), judge_settings,
                                   RANK_JUDGE_SCHEMA if rank_reward else JUDGE_SCHEMA,
                                   'anchored_group_ranking' if rank_reward else 'anchored_group_scores', 'group_reward_judge')
        return parsed_envelope(raw, RANK_JUDGE_SCHEMA if rank_reward else JUDGE_SCHEMA, meta, request)

    def write(case, group, candidate, seed):
        # The same natural policy role and sampling settings; only this reply
        # receives the contrast guide. Future calls use ordinary PolicyAgent.
        system = policy.agent_system + '\n\nFor this reply, follow this behavior:\n' + candidate['b_star']
        user = policy.agent_user_prompt(case, group['prefix'], group['anchor'])
        return with_role('group_contrast_reply', lambda: policy.respond_with_system(system, user, seed))

    return reflect, generate, judge, write


def validate_comparison(envelope, comparison, group):
    """Keep source attribution strict while tolerating quotation typography."""
    sources = {member['response_id']: member['reply'] for member in group['members']}
    for kind in ('helpful', 'harmful'):
        seen = set()
        for item in comparison[kind]:
            if item['response_id'] in seen or not quote_in_reply(item['quote'], sources[item['response_id']]):
                envelope.update(valid=False, error='Comparison quotation missing from specified source or source duplicated')
            seen.add(item['response_id'])


def build(cfg, *, policy=None, cases=None, defer_update=False):
    from functools import partial
    from scripts.online_rl.run_tree import ROOT, load_key, make_policy, runtime_execution
    from src.collaboration.generator import final_answer_option, stop_reason
    from src.online_rl.cases import load_cases
    from src.online_rl.environment import final_outcome
    from src.online_rl.integrated.continuation import adapter_sha256, protocol_sha256
    from src.online_rl.integrated.grpo import behavior_logprobs, make_updater
    from copy import deepcopy
    problems = config_problems(cfg, ROOT)
    if problems: raise ValueError('; '.join(problems))
    cfg = deepcopy(cfg)
    cfg['output_dir'] = str((ROOT / cfg['output_dir']).resolve())
    generation_provider = cfg['rubric']['generation'].get('api_provider', cfg.get('api_provider', 'openrouter'))
    judge_provider = cfg['judges'].get('api_provider', cfg.get('api_provider', 'openrouter'))
    generation_key, judge_key = load_key({'api_provider': generation_provider}), load_key({'api_provider': judge_provider})
    policy = policy if policy is not None else make_policy(cfg, generation_key)
    if (policy.args.rounds, policy.args.doctor_model) != (cfg['rollout']['rounds'], cfg['rollout']['doctor_model']):
        raise RuntimeError('Group policy rollout settings differ')
    problems = policy.doctor.verify()['problems']
    if problems: raise RuntimeError('Pinned group partner unavailable: ' + '; '.join(problems))
    updater = None if defer_update else make_updater(policy, cfg)
    def behavior(traces):
        if updater is not None: return updater.behavior_logprobs(traces)
        if hasattr(policy, 'pool_behavior_logprobs'): return policy.pool_behavior_logprobs(traces)
        with policy.gen_lock:
            return behavior_logprobs(policy.model, traces, cfg['optimizer']['logprob_token_chunk_size'])
    callbacks = make_callbacks(cfg, policy, generation_key, judge_key, ROOT)
    stop_rule = partial(stop_reason, allow_mutual_conclusion=cfg['rollout'].get('halting', {}).get('allow_mutual_conclusion', True))
    env = GroupEnv(PolicyAgent(policy), RoleDoctor(policy.doctor), stop_rule, final_answer_option, final_outcome,
                   *callbacks, behavior_logprobs=behavior)
    cases = cases if cases is not None else load_cases(ROOT / cfg['data']['train'])
    if len(cases) != 1: raise ValueError('Group trainer is a single-case coordinator; use run_five_gpu for batches')
    return GroupTrainer(cfg, cases[0], env, lambda: adapter_sha256(policy.model), protocol_sha256(cfg, ROOT),
                        runtime_execution(cfg, policy), updater)
