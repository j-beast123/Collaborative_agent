"""New-run-only generator ablation; retain original grader and method algorithms."""
import copy
import hashlib
from pathlib import Path
from baselines.luna_generator.backend import LunaGeneratorBackend

PROFILE = 'openrouter_gpt6_luna_none_v1'
MODEL = 'openai/gpt-6-luna'
NAMESPACE = 'd88c2e99b4a44_gen_gpt6luna_none_v1'
ROOT = Path(__file__).resolve().parents[2]
PARAMETER_ADAPTATION = {
    'name': 'omit_unsupported_generator_temperature_v1',
    'previous_temperature': 0.7,
    'effective_temperature': 'provider default; not specified or assumed to equal 0.7',
    'reason': 'OpenRouter GPT-6 Luna endpoints do not advertise temperature support; strict routing returned HTTP 404',
    'endpoint_metadata': 'https://openrouter.ai/api/v1/models/openai/gpt-6-luna/endpoints',
}


def source_identity():
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(__file__).parent.glob('*.py')) if not p.name.startswith('test_')}


def validate_spec(method, spec):
    if spec.get('method') != method or spec.get('generator_profile') != PROFILE:
        raise ValueError('Use the new Luna generator method config')
    if spec.get('api_profile') != 'openai_gpt56_luna_v1':
        raise ValueError('Keep the existing direct OpenAI gpt-5.6-luna grader')
    expected = (ROOT / f'outputs/{method}_train120_{NAMESPACE}').resolve()
    if (ROOT / spec['output_dir']).resolve() != expected:
        raise ValueError('New generator must use its isolated output/checkpoint/cache directory')
    registry = (ROOT / f'outputs/shared_initial_rubrics_{NAMESPACE}').resolve()
    if (ROOT / spec['initial_rubric_registry']).resolve() != registry or spec.get('initial_rubric_source') is not None:
        raise ValueError('Only the new shared initial-rubric registry is allowed; no old imports')
    expected_port = 11741 if method == 'rar' else 11841
    if spec['doctor_port_start'] != expected_port or spec.get('cache_namespace') != NAMESPACE:
        raise ValueError('Use the isolated Luna generator ports and namespace')


def install_profile(launcher):
    original_setup = launcher.setup  # Includes the unchanged OpenAI grader profile.

    def setup(method, spec, runtime):
        validate_spec(method, spec)
        cfg, receipt, engine, reference, errors = original_setup(method, spec, runtime)
        generator = LunaGeneratorBackend(audit_path=Path(spec['output_dir']) / 'reward_api_attempts.jsonl',
            base_url='https://openrouter.ai/api/v1', model=MODEL, request_model=MODEL,
            revision=MODEL, accepted_response_models=[MODEL], api_key_env='OPENROUTER_API_KEY', max_retries=2)
        old = copy.deepcopy(engine.identity['generator'])
        params = {**copy.deepcopy(engine.gen_params), 'reasoning': {'effort': 'none'}}
        params.pop('temperature', None)
        engine.generator, engine.gen_params = generator, params
        engine.identity.update(generator=generator.identity(), generator_params=params)
        cfg['comparison_method']['engine'] = copy.deepcopy(engine.identity)
        cfg['comparison_method']['generator_profile'] = {
            'name': PROFILE, 'generator': generator.identity(), 'generator_params': params,
            'previous_generator': old, 'execution_source_sha256': source_identity(),
            'parameter_adaptation': copy.deepcopy(PARAMETER_ADAPTATION),
            'previous_run': spec.get('previous_run'), 'initialization': 'fresh seeded initial adapter, not previous final checkpoint',
            'changed_roles': ['initial_rubric'] + (['pairwise_extract', 'deduplicate'] if method == 'online_rubrics' else []),
            'unchanged': 'grader, algorithm, prompts, data/order, policy initialization, Doctor, GRPO, evaluation',
            'provider_support': 'not live-tested; require_parameters true; no model or provider fallback'}
        cfg['rubric']['generator_model'] = MODEL
        cfg['rubric']['generation']['reasoning_effort'] = 'none'
        cfg['rubric']['generation'].pop('temperature', None)
        return cfg, receipt, engine, reference, errors
    launcher.setup = setup
