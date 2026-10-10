"""Explicit grader/provider change, independently fingerprinted from the old run."""
import copy
from pathlib import Path
from baselines.parallel_api.backend import OpenAIGraderBackend
from baselines.parallel_api.credentials import read_api_key, status

GRADE_PARAMS = {'max_completion_tokens':4096, 'reasoning_effort':'medium',
                'response_format':{'type':'json_object'}}


def apply_profile_config(original):
    cfg = copy.deepcopy(original)
    cfg['comparison_method']['method_spec']['api_profile'] = 'openai_gpt56_luna_v1'
    engine = cfg['comparison_method']['engine']
    engine['grader'] = OpenAIGraderBackend(audit_path=Path(cfg['output_dir'])/'reward_api_attempts.jsonl').identity()
    engine['grader_params'] = copy.deepcopy(GRADE_PARAMS)
    cfg['judges']['scorer_model'] = 'gpt-5.6-luna'
    cfg['judges']['luna_canonical_slug'] = 'gpt-5.6-luna'
    from baselines.parallel_api.launch import execution_identity
    cfg['comparison_method']['api_profile'] = {
        'name':'openai_gpt56_luna_v1', 'grader_provider':'openai', 'grader_model':'gpt-5.6-luna',
        'grader_params':dict(GRADE_PARAMS), 'credential_environment':'OPENAI_API_KEY',
        'execution_source_sha256':execution_identity(20, 100)['execution_source_sha256'],
        'previous_grader':'openai/gpt-5.6-luna via OpenRouter',
        'reason':'User requested the same gpt-5.6-luna family through direct OpenAI with higher scoring concurrency',
        'parameter_change':'max_tokens renamed max_completion_tokens; reasoning medium explicit; temperature omitted for reasoning compatibility; previous OpenRouter effective effort/temperature not independently verified',
        'snapshot_limit':'Alias is pinned as a request ID; provider weight snapshot is not independently verified'}
    return cfg


def install_profile(launcher):
    original_setup = launcher.setup
    original_checks = launcher.resource_checks

    def setup(method, spec, runtime):
        cfg, receipt, engine, reference, errors = original_setup(method, spec, runtime)
        profile = spec.get('api_profile')
        if profile != 'openai_gpt56_luna_v1':
            raise ValueError('Use the explicit OpenAI grader method config; existing OpenRouter runs are unchanged')
        engine.grader = OpenAIGraderBackend(audit_path=Path(spec['output_dir'])/'reward_api_attempts.jsonl')
        engine.grade_params = dict(GRADE_PARAMS)
        engine.identity['grader'] = engine.grader.identity()
        engine.identity['grader_params'] = dict(GRADE_PARAMS)
        cfg = apply_profile_config(cfg)
        return cfg, receipt, engine, reference, errors

    def checks(cfg, runtime, method, spec, *, gpu):
        errors = original_checks(cfg, runtime, method, spec, gpu=gpu)
        try: read_api_key(variable='OPENAI_API_KEY')
        except RuntimeError as exc: errors.append(str(exc))
        return errors

    launcher.setup = setup
    launcher.resource_checks = checks
    launcher.credential_status = status
