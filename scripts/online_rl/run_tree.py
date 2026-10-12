"""SHERPA case-tree launcher. Without --launch, or with any blocker, it only prints the blockers; every check runs before
credentials, the policy or any API client is loaded. A smoke config (one with a "smoke" block) must pass the smoke size
contract and runs one phase per launch: --phase A = the training case at case_cursor 0, --phase B = the prompt event
that follows it (case_cursor 1 is never run). The research config runs one arm per launch (--condition outcome_only |
fixed | evolving, output in <output_dir>/<arm>). Resumes from <output_dir>/run_state.json and its call cache."""
import argparse, hashlib, json, os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from src.online_rl.integrated.conditions import unresolved

PROMPT_HASHES = (('rubric', 'generator_prompt_file', 'prompt_sha256'), ('feedback', 'prompt_file', 'prompt_sha256'),
                 ('prompt_update', 'feedback_prompt_file', 'feedback_prompt_sha256'), ('prompt_update', 'refiner_prompt_file', 'refiner_prompt_sha256'),
                 ('intervention', 'score0_template_file', 'score0_template_sha256'), ('intervention', 'score2_template_file', 'score2_template_sha256'),
                 ('intervention', 'instruction_template_file', 'instruction_template_sha256'),
                 ('intervention', 'agent_system_prompt_file', 'agent_system_prompt_sha256'),
                 ('judges', 'gate_prompt_file', 'gate_prompt_sha256'), ('judges', 'scorer_prompt_file', 'scorer_prompt_sha256'),
                 ('judges', 'intervention_audit_prompt_file', 'intervention_audit_prompt_sha256'),
                 ('judges', 'applicability_prompt_file', 'applicability_prompt_sha256'))
EVIDENCE_SCORER_PROTOCOL = 'response_evidence_v1'
EVIDENCE_GATE_PROTOCOL = 'evidence_gate_v2'
PUBLIC_SCORER_PROTOCOL = 'public_mc_evidence_v1'
PUBLIC_GATE_PROTOCOL = 'public_mc_gate_v1'
PUBLIC_SCORER_PROTOCOL_V2 = 'public_mc_evidence_v2'
PUBLIC_GATE_PROTOCOL_V2 = 'public_mc_gate_v2'
PUBLIC_SCORER_PROTOCOL_V3 = 'public_mc_evidence_v3'
PUBLIC_GATE_PROTOCOL_V3 = 'public_mc_gate_v3'
PUBLIC_CRITERION_PROTOCOL = 'public_behavior_mc_v1'
INSTRUCTION_AUDIT_PROTOCOL = 'intervention_instruction_audit_v1'
INSTRUCTION_ALIGNMENT_PROTOCOL = 'intervention_alignment_analysis_v1'
JOINT_OUTPUT_CONTRACT = 'trajectory_analysis_candidate_v2_with_intervention'
JOINT_GUIDED_PROTOCOL = 'joint_instruction_v2_1'
SMOKE_DIR = 'outputs/online_rl/tree_smoke'
PILOT_DIR, PILOT_MAX_CASES = 'outputs/online_rl/tree_pilot', 5
# Named exceptions to the pilot cap: stage -> (output_dir, cases). Each is ONE continuous Outcome-only diagnostic (one policy,
# no rubric side); the general cap is unchanged for every other pilot.
PILOT_EXCEPTIONS = {'OUTCOME_ONLY_LOCAL_10_DIAGNOSTIC': ('outputs/online_rl/tree_pilot/outcome_only_local_10', 10)}
SMOKE_LIMITS = ((('schedule', 'cases'), 2), (('rollout', 'branching'), 2), (('rollout', 'rounds'), 2), (('intervention', 'reruns_per_intervention'), 2),
                (('prompt_update', 'candidate_count'), 1), (('prompt_update', 'retain_top_k'), 1), (('prompt_update', 'refinement_depth'), 1))


def file_sha(rel: str) -> str: return hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()


def blockers(cfg: dict) -> list[str]:
    """Unresolved settings plus every approved prompt/template whose file differs from its pinned sha256 (as run_integrated)."""
    from src.online_rl.tree.conditions import active_unresolved
    from src.online_rl.tree.group_learning import enabled, config_problems
    if enabled(cfg):
        from src.online_rl.integrated.grpo import validate_raw_sampling
        problems = config_problems(cfg, ROOT)
        if cfg.get('optimizer', {}).get('require_raw_on_policy_sampling'):
            try: validate_raw_sampling(cfg['rollout'], recorded=False, label='group rollout')
            except ValueError as exc: problems.append(str(exc))
        return problems
    out = active_unresolved(cfg)                                    # only settings the condition uses
    provider = cfg.get('api_provider', 'openrouter')
    judge_provider = cfg.get('judges', {}).get('api_provider', provider)
    generation_provider = (cfg.get('rubric', {}).get('generation') or {}).get('api_provider', provider)
    if provider not in ('openrouter', 'openai_direct'):
        out.append(f'Unknown api_provider {provider!r}')
    if provider == 'openai_direct':
        if judge_provider != 'openai_direct' or generation_provider != 'openai_direct':
            out.append('OpenAI direct run requires direct rubric generation and judges')
        for section, key in (('rubric', 'generator_model'), ('judges', 'gate_model'),
                             ('judges', 'scorer_model'), ('judges', 'intervention_audit_model')):
            if key in cfg.get(section, {}) and cfg[section][key] != 'gpt-6-luna':
                out.append(f'{section}.{key} must be gpt-6-luna for the OpenAI direct run')
    # Mixed transport is supported: e.g. Gemini generation through OpenRouter and
    # Luna Gate/scoring through the direct OpenAI API. Each role loads its own secret.
    raw_sampling = (cfg.get('optimizer') or {}).get('require_raw_on_policy_sampling', False)
    if not isinstance(raw_sampling, bool):
        out.append('optimizer.require_raw_on_policy_sampling must be boolean')
    elif raw_sampling:
        from src.online_rl.integrated.grpo import validate_raw_sampling
        try:
            validate_raw_sampling(cfg.get('rollout', {}), recorded=False, label='rollout config')
        except ValueError as exc:
            out.append(str(exc))
    if cfg['rubric'].get('match_batch_size',1) not in (1,8):out.append('rubric.match_batch_size must be 1 or 8')
    for sec, f, h in PROMPT_HASHES:
        if f not in cfg.get(sec, {}) or h not in cfg.get(sec, {}): continue
        try:
            digest = file_sha(cfg[sec][f])
        except (OSError, TypeError):
            out.append(f'Missing or unreadable prompt artifact: {sec}.{f}')
            continue
        if cfg[sec][h] not in ('TBD', digest): out.append(f'{sec}.{f} differs from {sec}.{h}')
    generation = (cfg.get('rubric') or {}).get('generation') or {}
    schema_file = generation.get('local_schema_file')
    if schema_file:
        try:
            if generation.get('local_schema_sha256') != file_sha(schema_file):
                out.append('rubric.generation.local_schema_file differs from rubric.generation.local_schema_sha256')
        except (OSError, TypeError):
            out.append('Missing or unreadable rubric.generation.local_schema_file')
    if cfg['rubric'].get('schema'):
        from src.online_rl.tree.agent_state import SCHEMA
        if cfg['rubric']['schema'] != SCHEMA: out.append('Unknown rubric.schema')
        path = cfg['rubric'].get('state_addendum_file')
        if not path or not (ROOT / path).exists(): out.append('Missing visible-context rubric contract')
        elif cfg['rubric'].get('state_addendum_sha256') != file_sha(path): out.append('Visible-context rubric contract hash differs')
        if cfg['judges'].get('scorer_protocol') not in ('structured_v2', EVIDENCE_SCORER_PROTOCOL, PUBLIC_SCORER_PROTOCOL, PUBLIC_SCORER_PROTOCOL_V2, PUBLIC_SCORER_PROTOCOL_V3):
            out.append('Visible-context rubric requires structured_v2 or response_evidence_v1 scorer')
        if not cfg['feedback'].get('behavioral_locus') or cfg['rubric'].get('source_consistency') != 'source_locus_v1':
            out.append('Visible-context rubric requires a checked behavioral locus')
    if cfg.get('feedback', {}).get('protocol') == 'physician_post_final_reflection_v1':
        backend = cfg.get('rollout', {}).get('doctor_backend') or {}
        expected = f"ollama/{backend.get('model')}@{str(backend.get('model_digest', ''))[:12]}"
        if backend.get('engine') != 'ollama' or cfg['feedback'].get('model') != expected:
            out.append('Post-FINAL reflection must use the same pinned local Doctor as the dialogue')
    return out + joint_protocol_problems(cfg) + gate_protocol_problems(cfg) + intervention_audit_problems(cfg) + public_criterion_problems(cfg)


def public_criterion_problems(cfg):
    """One opt-in public contract; never silently activate it for existing banks."""
    g=cfg.get('rubric',{}).get('generation',{}); jm=cfg.get('judges',{})
    protocol=g.get('public_criterion_protocol')
    if protocol is None:
        return []
    out=[]
    if protocol != PUBLIC_CRITERION_PROTOCOL:out.append('Unknown public criterion protocol')
    if jm.get('gate_protocol') not in (PUBLIC_GATE_PROTOCOL, PUBLIC_GATE_PROTOCOL_V2, PUBLIC_GATE_PROTOCOL_V3):out.append('Public M/C generation requires a public Gate protocol')
    if jm.get('scorer_protocol') not in (PUBLIC_SCORER_PROTOCOL, PUBLIC_SCORER_PROTOCOL_V2, PUBLIC_SCORER_PROTOCOL_V3):out.append('Public M/C generation requires a public evidence scorer')
    if (jm.get('gate_protocol') == PUBLIC_GATE_PROTOCOL_V2) != (jm.get('scorer_protocol') == PUBLIC_SCORER_PROTOCOL_V2):out.append('Public semantic v2 Gate/scorer must be selected together')
    if (jm.get('gate_protocol') == PUBLIC_GATE_PROTOCOL_V3) != (jm.get('scorer_protocol') == PUBLIC_SCORER_PROTOCOL_V3):out.append('Public semantic v3 Gate/scorer must be selected together')
    if g.get('generation_semantic_protocol') == 'public_mc_generation_v3' and jm.get('gate_protocol') != PUBLIC_GATE_PROTOCOL_V3:out.append('Semantic v3 generation requires semantic v3 independent judges')
    if g.get('generation_semantic_protocol') == 'public_mc_generation_v2' and jm.get('gate_protocol') != PUBLIC_GATE_PROTOCOL_V2:out.append('Semantic v2 generation requires semantic v2 independent judges')
    if not jm.get('applicability_prompt_file') or not jm.get('applicability_prompt_sha256'):
        out.append('Public M/C evaluation requires a separately pinned applicability prompt')
    try:
        schema=json.loads((ROOT/g['local_schema_file']).read_text())
        required=set(schema['properties']['candidate_rubric']['required'])
        if not {'required_behavior','minimum_performance','completion_condition','criterion_version'} <= required:
            out.append('Public criterion schema loses public fields')
    except (OSError,KeyError,TypeError,ValueError):out.append('Public criterion requires a valid local schema')
    return out


def gate_protocol_problems(cfg: dict) -> list[str]:
    """Evidence gating is opt-in and needs an output budget for quoted judgments."""
    jm = cfg.get('judges', {})
    protocol = jm.get('gate_protocol', 'legacy_v1')
    quote_protocol = jm.get('gate_quote_protocol')
    if quote_protocol not in (None, 'visible_field_enum_v1', 'input_field_refs_v2'):
        return [f'Unknown judges.gate_quote_protocol {quote_protocol!r}']
    if quote_protocol == 'input_field_refs_v2' and protocol not in (PUBLIC_GATE_PROTOCOL_V2, PUBLIC_GATE_PROTOCOL_V3):
        return ['input_field_refs_v2 requires public_mc_gate_v2 or public_mc_gate_v3']
    if quote_protocol is not None and protocol not in (EVIDENCE_GATE_PROTOCOL, PUBLIC_GATE_PROTOCOL, PUBLIC_GATE_PROTOCOL_V2, PUBLIC_GATE_PROTOCOL_V3):
        return ['judges.gate_quote_protocol requires evidence_gate_v2 or public_mc_gate_v1']
    if protocol not in ('legacy_v1', EVIDENCE_GATE_PROTOCOL, PUBLIC_GATE_PROTOCOL, PUBLIC_GATE_PROTOCOL_V2, PUBLIC_GATE_PROTOCOL_V3):
        return [f'Unknown judges.gate_protocol {protocol!r}']
    if protocol not in (EVIDENCE_GATE_PROTOCOL, PUBLIC_GATE_PROTOCOL, PUBLIC_GATE_PROTOCOL_V2, PUBLIC_GATE_PROTOCOL_V3):
        return []
    out = []
    if str(jm.get('gate_model', '')).startswith('typesafe/jev'):
        out.append('evidence_gate_v2 requires a structured Luna client')
    budget = jm.get('gate_max_tokens')
    if isinstance(budget, bool) or not isinstance(budget, int) or budget < 2048:
        out.append('evidence_gate_v2 requires judges.gate_max_tokens >= 2048')
    digest = jm.get('gate_prompt_sha256')
    if (not jm.get('gate_prompt_file') or not isinstance(digest, str) or len(digest) != 64
            or any(char not in '0123456789abcdef' for char in digest)):
        out.append('evidence_gate_v2 requires a pinned gate prompt')
    return out


def evidence_gate_identity(cfg: dict, root: Path = ROOT):
    """Post-rollout research identity, independent of a frozen rollout override."""
    jm = cfg.get('judges', {})
    if jm.get('gate_protocol') not in (EVIDENCE_GATE_PROTOCOL, PUBLIC_GATE_PROTOCOL, PUBLIC_GATE_PROTOCOL_V2, PUBLIC_GATE_PROTOCOL_V3):
        return None
    from src.online_rl.tree.luna import LunaDecisionClient
    from src.online_rl.tree.quality_gate import evidence_posthoc_gate
    # Construction is offline: no credentials are read and no judge is called.
    # Derive the exact runtime contract, including fields and input boundaries,
    # instead of maintaining a second weaker manifest-only fingerprint.
    client = LunaDecisionClient(model=jm['gate_model'], key='',
                                system=(root / jm['gate_prompt_file']).read_text(),
                                max_tokens=jm['gate_max_tokens'],
                                provider=jm.get('api_provider', cfg.get('api_provider', 'openrouter')))
    if jm.get('compact_runtime') in ('v7_1', 'v7_5_v7_6'):
        return make_gate(cfg, client, []).settings
    if jm.get('gate_protocol') in (PUBLIC_GATE_PROTOCOL, PUBLIC_GATE_PROTOCOL_V2, PUBLIC_GATE_PROTOCOL_V3):
        from src.online_rl.tree.quality_gate import public_mc_posthoc_gate
        return public_mc_posthoc_gate(client, {}, observed_context=bool(cfg.get('rubric', {}).get('schema')),
                                      rubric_quote_protocol=jm.get('gate_quote_protocol'), protocol=jm['gate_protocol']).settings
    quote_options = {'rubric_quote_protocol': jm['gate_quote_protocol']} if jm.get('gate_quote_protocol') is not None else {}
    return evidence_posthoc_gate(client, {}, observed_context=bool(cfg.get('rubric', {}).get('schema')), **quote_options).settings


def intervention_audit_problems(cfg: dict) -> list[str]:
    """Opt-in instruction validity is pinned independently of blind source scoring."""
    jm = cfg.get('judges', {})
    generation = cfg.get('rubric', {}).get('generation') or {}
    protocol = jm.get('intervention_audit_protocol')
    alignment = generation.get('instruction_alignment_protocol')
    out = []
    if alignment not in (None, INSTRUCTION_ALIGNMENT_PROTOCOL):
        out.append('Unknown rubric.generation.instruction_alignment_protocol')
    if alignment == INSTRUCTION_ALIGNMENT_PROTOCOL:
        if protocol != INSTRUCTION_AUDIT_PROTOCOL:
            out.append('intervention_alignment_analysis_v1 requires intervention_instruction_audit_v1')
        if generation.get('output_contract') != JOINT_OUTPUT_CONTRACT:
            out.append('intervention_alignment_analysis_v1 requires the joint generation contract')
        try:
            schema = json.loads((ROOT / generation['local_schema_file']).read_text())
            check = schema['properties']['analysis']['properties']['score_construction_check']
            referenced = generation.get('generation_semantic_protocol') in (
                'public_reference_generation_v4', 'public_app_alignment_generation_v5')
            required = ({'minimum_ref','completion_ref','instruction_alignment'} if referenced
                        else {'minimal_contribution','complete_behavior','instruction_alignment'})
            if not required <= set(check.get('required', [])):
                raise ValueError('missing mandatory alignment fields')
            fields = check['properties']
            if referenced:
                for key, target in [('minimum_ref','candidate_rubric.minimum_performance'),
                                    ('completion_ref','candidate_rubric.completion_condition')]:
                    if fields[key].get('const') != target:
                        raise ValueError('invalid public definition reference')
            elif any(fields[key].get('type') != 'string' for key in ('minimal_contribution','complete_behavior')):
                raise ValueError('minimum and completion must be strings')
            evidence = fields['instruction_alignment']
            if (evidence.get('type') != 'object' or
                    not {'target_level', 'instruction_ref' if referenced else 'instruction_quote', 'rationale'} <= set(evidence.get('required', [])) or
                    evidence['properties']['target_level'].get('type') != 'integer' or
                    set(evidence['properties']['target_level'].get('enum', [])) != {0, 2}):
                raise ValueError('invalid instruction alignment evidence')
        except (OSError, ValueError, KeyError, TypeError):
            out.append('intervention_alignment_analysis_v1 requires a valid pinned alignment schema')
    if protocol is None:
        if any(key.startswith('intervention_audit_') and value is not None for key, value in jm.items()):
            out.append('Intervention audit settings require judges.intervention_audit_protocol')
        return out
    if protocol != INSTRUCTION_AUDIT_PROTOCOL:
        return out + [f'Unknown judges.intervention_audit_protocol {protocol!r}']
    from src.online_rl.tree.witness import PROTOCOLS as WITNESS_PROTOCOLS
    if not cfg.get('rubric', {}).get('enabled', True):
        out.append('intervention_instruction_audit_v1 requires rubric.enabled')
    if cfg.get('verification', {}).get('mode', 'exhaustive_members_v1') not in WITNESS_PROTOCOLS:
        out.append('intervention_instruction_audit_v1 requires bounded witness verification')
    if (jm.get('scorer_protocol') not in ('structured_v2', EVIDENCE_SCORER_PROTOCOL, PUBLIC_SCORER_PROTOCOL, PUBLIC_SCORER_PROTOCOL_V2, PUBLIC_SCORER_PROTOCOL_V3)
            or str(jm.get('scorer_model', '')).startswith('typesafe/jev')):
        out.append('intervention_instruction_audit_v1 requires a structured scorer with Stage-1 applicability')
    if generation.get('output_contract') != JOINT_OUTPUT_CONTRACT:
        out.append('intervention_instruction_audit_v1 requires joint rubric/instruction generation')
    if jm.get('intervention_audit_structured') is not True:
        out.append('intervention_instruction_audit_v1 requires judges.intervention_audit_structured = true')
    model = jm.get('intervention_audit_model')
    if not isinstance(model, str) or not model.strip() or model.startswith('typesafe/jev'):
        out.append('intervention_instruction_audit_v1 requires a structured Luna model')
    budget = jm.get('intervention_audit_max_tokens')
    if isinstance(budget, bool) or not isinstance(budget, int) or budget < 2048:
        out.append('intervention_instruction_audit_v1 requires judges.intervention_audit_max_tokens >= 2048')
    timeout = jm.get('intervention_audit_timeout_s')
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not 0 < timeout < float('inf'):
        out.append('intervention_instruction_audit_v1 requires a positive finite timeout')
    path, digest = jm.get('intervention_audit_prompt_file'), jm.get('intervention_audit_prompt_sha256')
    if (not isinstance(path, str) or not path or not isinstance(digest, str) or len(digest) != 64
            or any(char not in '0123456789abcdef' for char in digest)):
        out.append('intervention_instruction_audit_v1 requires a pinned instruction audit prompt')
    elif not (ROOT / path).is_file():
        out.append('Missing intervention instruction audit prompt')
    elif file_sha(path) != digest:
        out.append('judges.intervention_audit_prompt_file differs from judges.intervention_audit_prompt_sha256')
    return out


def _intervention_audit_client(cfg: dict, key: str, root: Path):
    from src.online_rl.tree.luna import LunaDecisionClient
    jm = cfg['judges']
    return LunaDecisionClient(model=jm['intervention_audit_model'], key=key,
                              system=(root / jm['intervention_audit_prompt_file']).read_text(),
                              max_tokens=jm['intervention_audit_max_tokens'],
                              timeout_s=jm['intervention_audit_timeout_s'],
                              provider=jm.get('api_provider', cfg.get('api_provider', 'openrouter')))


def intervention_audit_identity(cfg: dict, root: Path = ROOT):
    """Exact runtime audit identity; constructing a client offline does not call it."""
    if cfg.get('judges', {}).get('intervention_audit_protocol') != INSTRUCTION_AUDIT_PROTOCOL:
        return None
    from src.online_rl.tree.instruction_audit import instruction_audit_settings
    return instruction_audit_settings(_intervention_audit_client(cfg, '', root))


def make_intervention_audit(cfg: dict, key: str):
    """One configured callback shared by run_tree and the five-GPU case workers."""
    if cfg.get('judges', {}).get('intervention_audit_protocol') != INSTRUCTION_AUDIT_PROTOCOL:
        return None
    from src.online_rl.tree.instruction_audit import make_instruction_auditor
    auditor = make_instruction_auditor(_intervention_audit_client(cfg, key, ROOT))
    attempts = cfg.get('judges', {}).get('execution_retry_attempts')
    if not attempts:
        return auditor
    # Bounded LOCAL repetition of the unchanged audit request when its response is execution-invalid (malformed / evidence
    # contract / incomplete after the built-in correction). Such a response is never an audit verdict.
    from functools import wraps
    from src.common.config import IntegrationUnavailable

    @wraps(auditor)
    def retry_invalid_audit(*args, **kwargs):
        import json as _json, time as _time
        for attempt in range(attempts):
            try:
                return auditor(*args, **kwargs)
            except IntegrationUnavailable as exc:
                if attempt == attempts - 1:
                    raise
                print(_json.dumps({'event': 'judge_execution_retry', 'stage': 'intervention_audit', 'attempt': attempt + 1,
                                   'of': attempts, 'error': str(exc)[:120]}), flush=True)
                _time.sleep(min(2 * (attempt + 1), 20))
    for name in ('settings', 'model', 'protocol'):
        setattr(retry_invalid_audit, name, getattr(auditor, name))
    return retry_invalid_audit


def joint_protocol_problems(cfg: dict) -> list[str]:
    """Validate the optional joint generator, evidence scorer and guided Agent contract before any API/model load."""
    from string import Formatter
    out = []
    rb, iv, jm = cfg.get('rubric', {}), cfg.get('intervention', {}), cfg.get('judges', {})
    generation = rb.get('generation') or {}
    feedback_check = generation.get('feedback_check_protocol')
    if feedback_check not in (None, 'all_reflection_claims_v1'):
        out.append('Unknown rubric.generation.feedback_check_protocol')
    if feedback_check == 'all_reflection_claims_v1':
        if cfg.get('feedback', {}).get('protocol') != 'physician_post_final_reflection_v1':
            out.append('all_reflection_claims_v1 requires participating Doctor reflection')
        if generation.get('output_contract') != JOINT_OUTPUT_CONTRACT:
            out.append('all_reflection_claims_v1 requires the joint generation contract')
        try:
            schema = json.loads((ROOT / generation['local_schema_file']).read_text())
            check_schema = schema['properties']['analysis']['properties']['feedback_check']
            if ('claim_reviews' not in check_schema.get('required', []) or
                    check_schema['properties'].get('claim_reviews', {}).get('type') != 'array'):
                out.append('all_reflection_claims_v1 requires a schema with mandatory claim_reviews')
        except (OSError, ValueError, KeyError, TypeError):
            out.append('all_reflection_claims_v1 requires a valid pinned feedback audit schema')
    guided = iv.get('guided_protocol')
    if guided not in (None, JOINT_GUIDED_PROTOCOL):
        out.append(f'Unknown intervention.guided_protocol {guided!r}')
    if generation.get('selected_turn_policy') not in (None, 'generator_v1'):
        out.append('Unknown rubric.generation.selected_turn_policy')
    if jm.get('scorer_protocol') in (EVIDENCE_SCORER_PROTOCOL, PUBLIC_SCORER_PROTOCOL, PUBLIC_SCORER_PROTOCOL_V2, PUBLIC_SCORER_PROTOCOL_V3):
        if not rb.get('schema') or str(jm.get('scorer_model', '')).startswith('typesafe/jev'):
            out.append('response_evidence_v1 requires an observed-context rubric and a structured Luna client')
    if generation.get('output_contract') == JOINT_OUTPUT_CONTRACT:
        if not generation.get('local_schema_file') or not generation.get('local_schema_sha256'):
            out.append('Joint generation requires a pinned local JSON schema')
        if not iv.get('instruction_template_file') and guided != JOINT_GUIDED_PROTOCOL:
            out.append('Joint generation requires an instruction template or joint_instruction_v2_1 guided protocol')
    if guided != JOINT_GUIDED_PROTOCOL:
        return out
    if generation.get('output_contract') != JOINT_OUTPUT_CONTRACT:
        out.append('joint_instruction_v2_1 requires joint rubric/instruction generation')
    if jm.get('scorer_protocol') not in (EVIDENCE_SCORER_PROTOCOL, PUBLIC_SCORER_PROTOCOL, PUBLIC_SCORER_PROTOCOL_V2, PUBLIC_SCORER_PROTOCOL_V3):
        out.append('joint_instruction_v2_1 requires response_evidence_v1 scoring')
    if iv.get('instruction_template_file'):
        out.append('joint_instruction_v2_1 uses its level-specific templates; remove intervention.instruction_template_file')
    for key in ('agent_system_prompt', 'score0_template', 'score2_template'):
        if not iv.get(key + '_file') or not iv.get(key + '_sha256'):
            out.append(f'joint_instruction_v2_1 requires pinned intervention.{key}_file')
            continue
        path = ROOT / iv[key + '_file']
        if not path.is_file():
            out.append(f'Missing joint intervention artifact: {path}')
            continue
        if key != 'agent_system_prompt':
            try:
                fields = [name for _, name, _, _ in Formatter().parse(path.read_text()) if name is not None]
                if fields != ['intervention_instruction']:
                    out.append(f'intervention.{key}_file must use only one intervention_instruction field')
            except ValueError:
                out.append(f'intervention.{key}_file has invalid format fields')
    return out


def smoke_problems(cfg: dict) -> list[str]:
    """The smoke launch contract; a config without a "smoke" block is not checked here."""
    if 'smoke' not in cfg: return []
    out = [f'smoke: {s}.{k}={cfg[s][k]!r} exceeds {lim}' for (s, k), lim in SMOKE_LIMITS if not isinstance(cfg[s][k], int) or cfg[s][k] > lim]
    if (ROOT / cfg['output_dir']).resolve() != (ROOT / SMOKE_DIR).resolve(): out.append(f'smoke: output_dir must be {SMOKE_DIR}')
    return out


def pilot_problems(cfg: dict) -> list[str]:
    """A diagnostic pilot (a "pilot" block): hard case cap, never more than PILOT_MAX_CASES, output under PILOT_DIR."""
    if 'pilot' not in cfg: return []
    cap, out = cfg['pilot'].get('max_cases'), []
    exc = PILOT_EXCEPTIONS.get(cfg['pilot'].get('stage'))
    if exc and cap == exc[1]:                                       # the named diagnostic: exactly its own cap, directory and arm
        if (ROOT / cfg['output_dir']).resolve() != (ROOT / exc[0]).resolve(): out.append(f"pilot: {cfg['pilot']['stage']} must write to {exc[0]}")
        if cfg['rubric'].get('enabled', True) or cfg['prompt_update']['enabled'] or cfg.get('condition') != 'outcome_only':
            out.append(f"pilot: {cfg['pilot']['stage']} is an Outcome-only diagnostic (rubric and prompt update off)")
    elif not isinstance(cap, int) or not 1 <= cap <= PILOT_MAX_CASES: out.append(f'pilot: max_cases must be an integer in 1..{PILOT_MAX_CASES}')
    elif cfg['schedule']['cases'] != cap: out.append(f"pilot: schedule.cases={cfg['schedule']['cases']!r} must equal pilot.max_cases={cap}")
    if (ROOT / PILOT_DIR).resolve() not in (ROOT / cfg['output_dir']).resolve().parents: out.append(f'pilot: output_dir must be inside {PILOT_DIR}')
    if 'smoke' in cfg: out.append('pilot and smoke blocks cannot be combined')
    return out


def setup_problems(cfg: dict, config_path: str) -> list[str]:
    """Output collisions with any other config, training data, and the model/backend settings the wiring needs."""
    out, mine = [], (ROOT / cfg['output_dir']).resolve()
    for p in sorted((ROOT / 'configs/online_rl').glob('*.json')):
        if p.resolve() == (ROOT / config_path).resolve(): continue
        other = json.loads(p.read_text()).get('output_dir')
        if other and (ROOT / other).resolve() == mine: out.append(f'output_dir collides with {p.relative_to(ROOT)}')
    if not (ROOT / cfg['data']['train']).exists(): out.append(f"training split missing: {cfg['data']['train']}")
    elif not file_sha(cfg['data']['train']).startswith(cfg['data']['train_sha256']): out.append('training split hash mismatch')
    for sec, k in (('policy', 'base_model'), ('rollout', 'doctor_model'), ('rollout', 'max_tokens_doctor'), ('feedback', 'model'), ('feedback', 'max_tokens'),
                   ('rubric', 'generator_model'), ('judges', 'gate_model'), ('judges', 'scorer_model'), ('prompt_update', 'reflection_model'),
                   ('prompt_update', 'refiner_model'),
                   ('advantage_normalization', 'mode')):
        if not (cfg.get(sec) or {}).get(k): out.append(f'missing {sec}.{k}')
    ro = cfg.get('rollout') or {}
    from src.online_rl.tree.rollout import ROLLOUT_BUDGETS
    if ro.get('budget', 'full_tree') not in ROLLOUT_BUDGETS:
        out.append(f'rollout.budget must be one of {ROLLOUT_BUDGETS}')
    if ro.get('doctor_system_prompt_file') and file_sha(ro['doctor_system_prompt_file']) != ro.get('doctor_system_prompt_sha256'):
        out.append('rollout.doctor_system_prompt_file differs from rollout.doctor_system_prompt_sha256')
    if ro.get('doctor_backend'):
        b = ro['doctor_backend']
        for k in ('engine', 'endpoint', 'model', 'model_digest', 'quantization', 'ollama_version', 'source_model', 'options'):
            if not b.get(k): out.append(f'missing rollout.doctor_backend.{k}')
        if not str(ro.get('doctor_model', '')).startswith('ollama/'): out.append('rollout.doctor_model must name the local Doctor (ollama/<tag>@<digest>)')
        if not isinstance((ro.get('halting') or {}).get('enabled'), bool) or (ro.get('halting') or {}).get('min_agent_turns') not in (1, 2, 3):
            out.append('rollout.halting needs enabled (true/false) and min_agent_turns = 1, 2, or 3')
    rb = cfg.get('rubric') or {}
    if rb.get('reuse'):
        from src.online_rl.tree.rubric_bank import ASSIGNMENT_VERSION, prompt_sections
        if rb['reuse'] != ASSIGNMENT_VERSION: out.append(f"rubric.reuse must be {ASSIGNMENT_VERSION!r}")
        else:
            try: prompt_sections((ROOT / rb['generator_prompt_file']).read_text())
            except (OSError, ValueError) as e: out.append(f'rubric reuse matcher package is invalid: {e}')
    if rb.get('enabled', True) and 'smoke' not in cfg:                 # every pilot / research run (smoke = legacy wiring check only)
        g = rb.get('generation') or {}
        local_joint = (g.get('output_contract') == JOINT_OUTPUT_CONTRACT and g.get('local_schema_file') and
                       g.get('local_schema_sha256') and g.get('response_format') is None)
        if g.get('reasoning_effort') != 'low' or (g.get('response_format') != 'json_schema_strict' and not local_joint) or not isinstance(g.get('max_tokens'), int) or g['max_tokens'] < 2048:
            out.append('rubric.generation must pin reasoning_effort low, response_format json_schema_strict and max_tokens >= 2048 (Gemini truncation fix)')
        if g.get('output_contract') not in (None, 'trajectory_analysis_candidate_v2', JOINT_OUTPUT_CONTRACT):
            out.append(f"unknown rubric.generation.output_contract {g.get('output_contract')!r}")
        if (cfg.get('judges') or {}).get('scorer_protocol') not in ('structured_v2', EVIDENCE_SCORER_PROTOCOL, PUBLIC_SCORER_PROTOCOL, PUBLIC_SCORER_PROTOCOL_V2, PUBLIC_SCORER_PROTOCOL_V3) and not str((cfg.get('judges') or {}).get('scorer_model', '')).startswith('typesafe/jev'):
            out.append('judges.scorer_protocol must be structured_v2 or response_evidence_v1 for the Luna scorer')
    pol = cfg.get('policy') or {}; seed = (pol.get('lora') or {}).get('init_seed')
    if not pol.get('init_adapter') and (not isinstance(seed, int) or isinstance(seed, bool)):
        out.append('missing policy.lora.init_seed (integer): the initial LoRA must not depend on the global RNG')
    ev = cfg.get('evaluation')
    if ev and not unresolved(ev):
        from src.online_rl.guards import MANIFEST
        from src.online_rl.tree.evaluation import eval_cases
        try: eval_cases(ROOT / ev['cases_file'], ev['split'], json.loads(MANIFEST.read_text()))
        except (OSError, ValueError, KeyError) as e: out.append(f'evaluation cases: {e}')
    return out + opening_problems(cfg)


def phase_problems(cfg: dict, phase) -> list[str]:
    smoke = 'smoke' in cfg
    if not smoke and 'condition' not in cfg: return ['research config: --condition outcome_only|rubric_only|fixed|evolving is required']
    if smoke and phase is None: return ['smoke config: --phase A or --phase B is required']
    if not smoke and phase is not None: return ['--phase is only defined for a smoke config']
    if phase is None: return []
    out = ROOT / cfg['output_dir']; sp = out / 'run_state.json'
    st = json.loads(sp.read_text()) if sp.exists() else None
    if phase == 'A':
        return ['phase A already completed (case_cursor >= 1)'] if st and st['case_cursor'] >= 1 else []
    if st is None or st['case_cursor'] < 1 or not (out / 'cases' / 'case_000000.json').exists(): return ['phase B requires a completed phase A in output_dir']
    if st['case_cursor'] > 1: return ['phase B: a training case after the event already ran; refusing']
    return ['phase B already completed (lineage_cursor >= 1)'] if st['lineage_cursor'] >= 1 else []


def load_key(cfg: dict | None = None) -> str:
    provider = (cfg or {}).get('api_provider', 'openrouter')
    if provider == 'openai_direct':
        # The direct API credential is kept in the user's private settings.
        # Never copy it into config files or emit it in diagnostics.
        from src.common.config import IntegrationUnavailable
        import os
        if os.environ.get('OPENAI_API_KEY', '').strip():
            return os.environ['OPENAI_API_KEY'].strip()
        private = Path.home() / '.config/sherpa/openai.env'
        path = Path(os.environ.get('SHERPA_OPENAI_ENV_FILE') or
                    (str(private) if private.is_file() else os.environ.get('SHERPA_ENV_FILE', str(ROOT / '.env')))).expanduser()
        try:
            lines = path.read_text(encoding='utf-8').splitlines()
        except OSError as exc:
            raise IntegrationUnavailable(f'OpenAI secret settings are unreadable: {path}') from exc
        for line in lines:
            name, sep, value = line.strip().removeprefix('export ').partition('=')
            if sep and name.strip() == 'OPENAI_API_KEY':
                key = value.strip().strip('"\'')
                if key: return key
        raise IntegrationUnavailable(f'OPENAI_API_KEY missing in secret settings: {path}')
    if provider != 'openrouter':
        raise ValueError(f'Unknown API provider: {provider}')
    from src.prompt_opt import harness as H
    return H.load_key()


def judge_client(model: str, key: str, system: str, *, max_tokens: int = 2000, provider: str = 'openrouter'):
    """Jev (decisions API, label probabilities) for typesafe/jev models, else a hard-label decision client (Luna)."""
    if provider not in ('openrouter', 'openai_direct'):
        raise ValueError(f'Unknown judge API provider: {provider}')
    if provider == 'openai_direct' and model != 'gpt-6-luna':
        raise ValueError('OpenAI direct judge model must be gpt-6-luna')
    if model.startswith('typesafe/jev'):
        from src.judges.jev import JevClient
        from src.prompt_opt import harness as H
        return JevClient(**{**H.JEV, 'model': model})
    from src.online_rl.tree.luna import LunaDecisionClient
    return LunaDecisionClient(model=model, key=key, system=system, max_tokens=max_tokens, provider=provider)


def make_scorer(cfg: dict, client):
    """Select evidence scoring only for configs that explicitly request it."""
    from src.online_rl.tree.backends import two_stage_scorer, two_stage_structured_scorer, two_stage_hard_scorer
    jm = cfg['judges']
    if jm.get('compact_runtime') in ('v7_1', 'v7_5_v7_6'):
        from functools import wraps
        from src.common.config import IntegrationUnavailable
        from src.online_rl.tree.compact_runtime import compact_scorer
        from src.online_rl.tree.luna import LunaDecisionClient
        app_client = LunaDecisionClient(model=client.model, key=client.key,
            system=(ROOT/jm['applicability_prompt_file']).read_text(), max_tokens=client.max_tokens,
            post=client.post, sleep=client.sleep, timeout_s=client.timeout_s,
            provider=getattr(client, 'provider', 'openrouter'))
        scorer = compact_scorer(client, app_client, rubric_score_batch_size(cfg),
                                revision=jm['compact_runtime'], execution_retry_attempts=jm.get('execution_retry_attempts'),
                                execution_contract_repair=jm.get('execution_contract_repair',False))
        # A malformed judge response is an execution failure, not an NA or a
        # score. Retry the unchanged Stage-1 request once while retaining the
        # failed wire evidence in the eventual cached decision. This wrapper
        # does not alter the criterion or its pinned cache identity.
        # judges.execution_retry_attempts (optional; absent = the historical single retry): bounded LOCAL repetition of the
        # unchanged request when the judge response is execution-invalid (schema / evidence-contract / malformed JSON) or
        # truncated (finish_reason). Such a response is never a TRUE / FALSE / UNCERTAIN verdict, a score or an NA.
        local_attempts = jm.get('execution_retry_attempts')
        if local_attempts is not None and (type(local_attempts) is not int or local_attempts != 1):
            raise ValueError('judges.execution_retry_attempts must be 1 (single attempt; no retries)')
        attempts = 1

        def execution_invalid(validation):
            if validation.get('status') == 'execution_error' and validation.get('kind') in ('schema', 'evidence_contract', 'json_or_response'):
                return True
            return bool(local_attempts) and validation.get('status') == 'execution_incomplete' and validation.get('kind') == 'finish_reason'

        def note_retry(stage, attempt, validation):
            import json as _json, time as _time
            print(_json.dumps({'event': 'judge_execution_retry', 'stage': stage, 'attempt': attempt + 1, 'of': attempts,
                               'status': validation.get('status'), 'kind': validation.get('kind'),
                               'error': str(validation.get('error'))[:120]}), flush=True)
            if local_attempts: _time.sleep(min(2 * (attempt + 1), 20))

        original_review = getattr(scorer, 'review_state', None)
        if callable(original_review):
            @wraps(original_review)
            def retry_invalid_stage1(*args, **kwargs):
                failed = []
                for attempt in range(attempts):
                    try:
                        retry_kwargs=dict(kwargs)
                        if jm.get('execution_contract_repair') and failed:
                            retry_kwargs['repair_audit']=failed[-1]
                        rows = original_review(*args, **retry_kwargs)
                    except IntegrationUnavailable as exc:
                        audit = getattr(exc, 'audit', None) or {}
                        validation = audit.get('validation') or {}
                        if attempt == attempts - 1 or not execution_invalid(validation):
                            if failed:
                                exc.audit = {**audit, 'prior_failed_attempts': failed}
                            raise
                        failed.append(audit)
                        print('Compact applicability execution invalid; retry 1/1', flush=True) if not local_attempts else note_retry('applicability', attempt, validation)
                        continue
                    if failed:
                        for row in rows.values():
                            row['evidence_validation']['prior_failed_attempts'] = failed
                    return rows
                raise AssertionError('unreachable compact applicability retry')

            # With a configured local retry budget, exhausting it yields EXECUTION_UNRESOLVED for that judgment (applicability.py).
            retry_invalid_stage1.execution_unresolved_terminal = bool(local_attempts)
            scorer.review_state = scorer.applicability = retry_invalid_stage1
        # A judge can omit one of the required public M/C checks while still
        # returning valid JSON. Retry that incomplete execution; never turn it
        # into a numeric score, NA, or UNSCORABLE. Keep the scorer protocol and
        # cached successful decisions unchanged for an in-progress batch.
        original_batch = scorer.score_behavior_batch

        def retry_incomplete_checks(*args, **kwargs):
            for attempt in range(1):
                try:
                    return original_batch(*args, **kwargs)
                except IntegrationUnavailable as exc:
                    validation = getattr(exc, 'audit', {}).get('validation', {})
                    if (validation.get('kind') != 'evidence_contract' or
                            validation.get('error') != 'Each public definition needs one check' or
                            attempt == 4):
                        raise
                    print(f'Compact scorer omitted public checks; retry {attempt + 1}/4', flush=True)

        scorer.score_behavior_batch = retry_incomplete_checks
        if local_attempts:                                      # the same bounded local repetition for the behavior judgment
            checked_batch = scorer.score_behavior_batch

            def retry_invalid_behavior(*args, **kwargs):
                for attempt in range(attempts):
                    try:
                        return checked_batch(*args, **kwargs)
                    except IntegrationUnavailable as exc:
                        validation = (getattr(exc, 'audit', None) or {}).get('validation') or {}
                        if attempt == attempts - 1 or not execution_invalid(validation):
                            raise
                        note_retry('behavior', attempt, validation)

            scorer.score_behavior_batch = retry_invalid_behavior
        return scorer
    if jm.get('scorer_protocol') in (PUBLIC_SCORER_PROTOCOL, PUBLIC_SCORER_PROTOCOL_V2, PUBLIC_SCORER_PROTOCOL_V3):
        from src.online_rl.tree.pilot_evidence_scorer import evidence_scorer
        from src.online_rl.tree.luna import LunaDecisionClient
        app_client=LunaDecisionClient(model=client.model,key=client.key,
            system=(ROOT/jm['applicability_prompt_file']).read_text(),max_tokens=client.max_tokens,
            post=client.post,sleep=client.sleep,timeout_s=client.timeout_s,
            provider=getattr(client, 'provider', 'openrouter'))
        return evidence_scorer(client,response_batch_size=rubric_score_batch_size(cfg),
            evidence_contract=jm['scorer_protocol'],applicability_client=app_client)
    if jm.get('scorer_protocol') == EVIDENCE_SCORER_PROTOCOL:
        from src.online_rl.tree.pilot_evidence_scorer import evidence_scorer
        return evidence_scorer(client, response_batch_size=rubric_score_batch_size(cfg))
    if jm['scorer_model'].startswith('typesafe/jev'):
        return two_stage_scorer(client)
    if jm.get('scorer_protocol') == 'structured_v2':
        return two_stage_structured_scorer(client, observed_context=bool(cfg['rubric'].get('schema')),
                                           response_batch_size=rubric_score_batch_size(cfg))
    return two_stage_hard_scorer(client)


def make_gate(cfg: dict, client, cases):
    """Keep legacy gating for existing configs; bind evidence gating only by explicit version."""
    from src.online_rl.tree.backends import posthoc_gates
    if cfg['judges'].get('compact_runtime') in ('v7_1', 'v7_5_v7_6'):
        from functools import wraps
        from src.common.config import IntegrationUnavailable
        from src.online_rl.tree.quality_gate import public_mc_posthoc_gate
        from src.online_rl.tree.luna import LunaDecisionClient
        jm = cfg['judges']
        blind_client = LunaDecisionClient(model=client.model, key=client.key,
            system=(ROOT/jm['scorer_prompt_file']).read_text(), max_tokens=client.max_tokens,
            post=client.post, sleep=client.sleep, timeout_s=client.timeout_s,
            provider=getattr(client, 'provider', 'openrouter'))
        original_gate = public_mc_posthoc_gate(client, {c['case_id']:c for c in cases},
            observed_context=bool(cfg['rubric'].get('schema')), protocol=jm['gate_protocol'],
            rubric_quote_protocol=jm.get('gate_quote_protocol'),
            instructions_override=client.system, blind_scorer_override=make_scorer(cfg, blind_client))

        @wraps(original_gate)
        def gate_without_retry(tin, rubric):
            audits = []
            error_kinds = []
            for attempt in range(1):
                try:
                    return original_gate(tin, rubric)
                except IntegrationUnavailable as exc:
                    audit = getattr(exc, 'audit', None) or {}
                    validation = audit.get('validation') or {}
                    nonliteral_quote = ('Invalid public Gate evidence after one correction:' in str(exc)
                                        and 'source quote speaker/turn/text differs' in str(exc))
                    invalid_source_judgment = (validation.get('status') == 'execution_error'
                        and validation.get('kind') == 'evidence_contract')
                    if not (nonliteral_quote or invalid_source_judgment):
                        raise
                    audits.append(audit)
                    error_kinds.append('gate_quote' if nonliteral_quote else 'blind_source_evidence')
                    print(f'Gate judge evidence invalid ({error_kinds[-1]}); trajectory will be excluded', flush=True)
            # A repeatedly malformed judge response supplies no quality vote.
            # Exclude this candidate with an explicit execution-error record;
            # never admit it or turn it into an observed behavioral score.
            return {'protocol': jm['gate_protocol'], 'schema': True, 'passed': False, 'status': 'execution_error',
                    'reasons': ['gate_execution_error'],
                    'unresolved': error_kinds,
                    'evidence': {'failed_attempts': audits, 'error_kinds': error_kinds},
                    'public_criterion_version': rubric.get('criterion_version')}

        return gate_without_retry
    quote_options = ({'rubric_quote_protocol': cfg['judges']['gate_quote_protocol']}
                     if cfg['judges'].get('gate_quote_protocol') is not None else {})
    return posthoc_gates(client, {case['case_id']: case for case in cases},
                        observed_context=bool(cfg['rubric'].get('schema')),
                        protocol=cfg['judges'].get('gate_protocol', 'legacy_v1'), **quote_options)


def opening_table(cfg: dict, policy, which: str):
    """The frozen shared D1 table ('train' | 'eval') pinned in rollout.doctor_opening, verified at load; None without one."""
    spec = cfg['rollout'].get('doctor_opening')
    if not spec: return None
    from src.online_rl.doctor_openings import OpeningTable
    split = 'train' if which == 'train' else cfg['evaluation']['split']
    return OpeningTable(ROOT / spec[f'{which}_table'], spec[f'{which}_table_sha256'], split, policy.doctor, cfg['rollout']['rounds'])


def opening_problems(cfg: dict) -> list[str]:
    """Research runs with the local Doctor read D1 from the frozen shared tables (doctor_opening_mode = frozen_shared_v1)."""
    ro = cfg.get('rollout') or {}; spec = ro.get('doctor_opening')
    if not ro.get('doctor_backend') or 'smoke' in cfg or 'pilot' in cfg: return []
    if not spec: return ['missing rollout.doctor_opening: research runs use the frozen shared Doctor openings']
    from src.online_rl.doctor_openings import file_sha256, protocol_identity
    try: protocol_identity(spec)
    except ValueError as e: return [str(e)]
    out = []
    for w in ('train', 'eval'):
        f = ROOT / str(spec.get(f'{w}_table'))
        if not f.exists(): out.append(f'doctor opening table missing: {f}')
        elif file_sha256(f) != spec[f'{w}_table_sha256']: out.append(f'doctor opening table {f} differs from its pinned SHA256')
    return out


def make_policy(cfg: dict, key: str):
    from src.online_rl.runtime import SharedPolicy
    return SharedPolicy(cfg, key)


def make_feedback(cfg: dict, policy, key: str):
    """Select explicit observer or participating-Doctor feedback without a silent role fallback."""
    feedback = cfg['feedback']
    system = (ROOT / feedback['prompt_file']).read_text()
    protocol = feedback.get('protocol')
    if protocol == 'physician_post_final_reflection_v1':
        from src.online_rl.tree.doctor_reflection import local_doctor_feedback
        if policy.doctor.model != feedback['model']:
            raise RuntimeError('Reflection model differs from the participating Doctor')
        return local_doctor_feedback(policy.doctor, system, feedback['max_tokens'], run_seed=cfg['schedule']['seed'])
    from src.online_rl.tree.backends import api_behavior_feedback, api_feedback
    if protocol in ('behavior_observer_v4', 'behavior_observer_v5'):
        return api_behavior_feedback(key, feedback['model'], system, feedback['max_tokens'], protocol)
    return api_feedback(key, feedback['model'], system, feedback['max_tokens'],
                        locus=bool(feedback.get('behavioral_locus')))


def build(cfg: dict, *, policy=None, cases=None, defer_update: bool = False):
    """Bind the existing components to the TreeTrainer interfaces. Policy rounds / Doctor model come from cfg['rollout']."""
    from src.online_rl.tree.group_learning import enabled
    if enabled(cfg):
        from src.online_rl.tree.group_backends import build as build_groups
        return build_groups(cfg, policy=policy, cases=cases, defer_update=defer_update)
    from src.collaboration.generator import final_answer_option, stop_reason
    from src.online_rl.cases import load_cases
    from src.online_rl.environment import final_outcome
    from src.online_rl.integrated.continuation import adapter_sha256, protocol_sha256
    from src.online_rl.integrated.grpo import make_updater
    from src.online_rl.integrated.prompt_search import sol_backend
    from src.online_rl.tree.backends import (ApiDoctor, api_behavior_feedback, PolicyAgent, PolicyInterventions, RoleDoctor, api_rubric_matcher, generation_settings,
                                             two_stage_structured_scorer, api_feedback, api_rubric_generator, posthoc_gates,
                                             two_stage_hard_scorer, two_stage_scorer)
    from src.online_rl.tree.team import team_strategies
    from src.online_rl.tree.trainer import TreeEnv, TreeTrainer
    text = lambda rel: (ROOT / rel).read_text()
    observed_context = bool(cfg['rubric'].get('schema'))
    context_addendum = (text(cfg['rubric']['state_addendum_file'])
                        if observed_context and cfg['rubric'].get('state_addendum_enabled', True) else None)
    ro, pu, ar = cfg['rollout'], cfg['prompt_update'], cfg['agent_reward']
    jm = cfg['judges']
    judge_provider = jm.get('api_provider', cfg.get('api_provider', 'openrouter'))
    generation_provider = (cfg.get('rubric', {}).get('generation') or {}).get('api_provider', cfg.get('api_provider','openrouter'))
    key = load_key({'api_provider': generation_provider})
    judge_key = load_key({'api_provider': judge_provider})
    provider_kw = {'provider': judge_provider} if judge_provider != 'openrouter' else {}
    gate_client = judge_client(jm['gate_model'], judge_key, text(jm['gate_prompt_file']),
        **({'max_tokens': jm['gate_max_tokens']} if 'gate_max_tokens' in jm else {}), **provider_kw)
    scorer_client = judge_client(jm['scorer_model'], judge_key, text(jm['scorer_prompt_file']),
        **({'max_tokens': jm['scorer_max_tokens']} if 'scorer_max_tokens' in jm else {}), **provider_kw)
    policy = policy if policy is not None else make_policy(cfg, key)
    if (policy.args.rounds, policy.args.doctor_model) != (ro['rounds'], ro['doctor_model']):
        raise RuntimeError('policy/harness rounds or Doctor model differ from rollout config')
    if ro.get('doctor_backend'):                                 # protocol v2: the policy's local Doctor serves tree, reruns and evaluation
        problems = policy.doctor.verify()['problems']
        if problems: raise RuntimeError('local Doctor does not match its pinned backend: ' + '; '.join(problems))
    updater = None if defer_update else make_updater(policy, cfg)
    cases = cases if cases is not None else load_cases(ROOT / cfg['data']['train'])
    from src.online_rl.integrated.grpo import behavior_logprobs
    def behavior(traces):
        if updater is not None: return updater.behavior_logprobs(traces)
        if hasattr(policy, 'pool_behavior_logprobs'):
            return policy.pool_behavior_logprobs(traces)
        with policy.gen_lock:
            return behavior_logprobs(policy.model, traces, cfg['optimizer']['logprob_token_chunk_size'])
    iv = PolicyInterventions(policy, text(cfg['intervention']['score0_template_file']), text(cfg['intervention']['score2_template_file']),
                             intervention_instruction_template=text(cfg['intervention']['instruction_template_file']) if cfg['intervention'].get('instruction_template_file') else None,
                             guided_protocol=cfg['intervention'].get('guided_protocol'),
                             agent_system=text(cfg['intervention']['agent_system_prompt_file']) if cfg['intervention'].get('agent_system_prompt_file') else None)
    doctor = RoleDoctor(policy.doctor) if ro.get('doctor_backend') else ApiDoctor(key, ro['doctor_model'], policy.doctor_system, ro['max_tokens_doctor'])
    bank_matcher = None
    if cfg['rubric'].get('cumulative_bank'):
        from src.online_rl.tree.cumulative_bank import api_bank_matcher
        bank_cfg = cfg['rubric']['cumulative_bank']
        bank_matcher = api_bank_matcher(key, cfg['rubric']['generator_model'], generation_settings(cfg['rubric']),
                                       text(bank_cfg['matcher_prompt_file']))
    env = TreeEnv(agent=PolicyAgent(policy), doctor=doctor,
                  stop_rule=stop_reason, final_option=final_answer_option, outcome_fn=final_outcome,
                  feedback_fn=make_feedback(cfg, policy, key),
                  generate_fn=api_rubric_generator(key, cfg['rubric']['generator_model'], generation_settings(cfg['rubric']), context_addendum),
                  scorer=make_scorer(cfg, scorer_client),
                  writer=iv.write, continuation=iv.continue_, gate_fn=make_gate(cfg, gate_client, cases),
                  behavior_logprobs=getattr(policy, 'on_primary', lambda f: f)(behavior),
                  assign_fn=api_rubric_matcher(key, cfg['rubric']['generator_model'], generation_settings(cfg['rubric']), context_addendum,
                                              cfg['rubric'].get('match_batch_size',1)) if cfg['rubric'].get('reuse') else None,
                  opening=opening_table(cfg, policy, 'train'),
                  intervention_audit_fn=make_intervention_audit(cfg, judge_key), bank_matcher=bank_matcher)
    # the tree prompt search uses only the reflection / refinement calls of this backend
    backend = sol_backend(key, text(pu['feedback_prompt_file']), text(pu['refiner_prompt_file']), env.generate_fn, env.gate_fn, env.scorer,
                          reflection_model=pu['reflection_model'], refiner_model=pu['refiner_model'])
    strategies = team_strategies(updater, ar['w_outcome'], ar['w_rubric'])
    if defer_update:
        from dataclasses import replace
        strategies = replace(strategies, agent_update=lambda tree, targets: {'optimizer_step': False, 'deferred_to_gpu4': True})
    return TreeTrainer(cfg, out_dir=ROOT / cfg['output_dir'], cases=cases, env=env, strategies=strategies,
                       search_backend=backend, initial_prompt=text(cfg['rubric']['generator_prompt_file']), policy_sha=lambda: adapter_sha256(policy.model),
                       protocol_sha=protocol_sha256(cfg, ROOT), feedback_prompt_sha256=cfg['feedback']['prompt_sha256'], updater=updater,
                       evaluator=milestone_evaluator(cfg, policy, final_outcome), execution=runtime_execution(cfg, policy),
                       before_case=replica_barrier(cfg, policy))


def runtime_execution(cfg: dict, policy) -> dict:
    """Execution-only concurrency, with environment overrides and config defaults, recorded per launch in
    <output_dir>/execution_runtime.jsonl. These thread counts do not alter logical protocol/cache identities."""
    import socket, time
    n = int(getattr(policy, 'n_replicas', 1))
    ex = {'agent_workers': n,
          'api_workers': int(os.environ.get('SHERPA_API_WORKERS', str((cfg.get('five_gpu') or {}).get('api_workers', 1)))),
          'member_workers': int(os.environ.get('SHERPA_MEMBER_WORKERS', str((cfg.get('execution') or {}).get('member_workers', 1)))),
          'parallel_strata': os.environ.get('SHERPA_PARALLEL_STRATA') == '1',
          'rubric_score_batch_size': rubric_score_batch_size(cfg)}
    if cfg['rollout'].get('agent_sampling_protocol')=='per_row_rng_batch_v1':
        ex['agent_workers']=int(os.environ.get('SHERPA_AGENT_BATCH_SIZE',cfg.get('five_gpu',{}).get('agent_batch_size',1)))
    ex.update(getattr(policy,'pool_execution',{}))
    out = ROOT / cfg['output_dir']; out.mkdir(parents=True, exist_ok=True)
    from src.common.api_slots import create as validate_api_limits
    validate_api_limits((cfg.get('execution') or {}).get('api_limits', {}))
    ex.update(chunk_workers=int(os.environ.get('SHERPA_CHUNK_WORKERS', (cfg.get('execution') or {}).get('chunk_workers', 1))),
              api_limits=(cfg.get('execution') or {}).get('api_limits', {}))
    rec = {'time': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'host': socket.gethostname(), 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
           'agent_replicas': n, 'agent_devices': [str(d) for d in getattr(policy, 'devices', [])],
           'doctor_endpoints': getattr(getattr(policy, 'doctor', None), 'endpoints', None), **ex,
           'config_execution': cfg.get('execution'), 'exec_log': os.environ.get('SHERPA_EXEC_LOG')}
    with open(out / 'execution_runtime.jsonl', 'a') as f: f.write(json.dumps(rec) + '\n')
    return ex


def rubric_score_batch_size(cfg: dict) -> int:
    """Number of same-state replies scored in one structured rubric request (1..8)."""
    value = int(os.environ.get('SHERPA_RUBRIC_SCORE_BATCH_SIZE',
                               str((cfg.get('execution') or {}).get('rubric_score_batch_size', 4))))
    if not 1 <= value <= 8:
        raise ValueError('SHERPA_RUBRIC_SCORE_BATCH_SIZE must be between 1 and 8')
    return value


def replica_barrier(cfg: dict, policy):
    """Before every case (and milestone evaluation): sync every Agent replica to the committed policy and verify all adapter
    hashes; each check is appended to <output_dir>/replica_sync.jsonl."""
    if not hasattr(policy, 'sync_replicas'): return None
    import time
    path = ROOT / cfg['output_dir'] / 'replica_sync.jsonl'

    def barrier(expected_sha256: str):
        r = policy.sync_replicas(expected_sha256)
        with open(path, 'a') as f: f.write(json.dumps({'time': time.time(), 'expected': expected_sha256, **r}) + '\n')
    return barrier


def milestone_evaluator(cfg: dict, policy, outcome_fn):
    """Held-out task accuracy of the live policy at each milestone (no evaluation block -> none). The torch RNG state is
    restored after every episode so evaluation leaves training sampling untouched."""
    ev = cfg.get('evaluation')
    if not ev: return None
    from src.online_rl.guards import MANIFEST
    from src.online_rl.integrated.continuation import protocol_sha256
    from src.online_rl.tree.backends import with_role
    from src.online_rl.tree.evaluation import MilestoneEvaluator, eval_cases
    torch = policy.torch; opening = opening_table(cfg, policy, 'eval')      # the same frozen D1 for every checkpoint / arm

    replicas = int(getattr(policy, 'n_replicas', 1))

    def play(case, seed):
        if replicas > 1:                                           # every generation seeds its own replica; no global RNG to restore
            return with_role('doctor_eval', lambda: policy.generate(case, seed, opening=opening) if opening else policy.generate(case, seed))
        cpu = torch.get_rng_state(); gpu = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        try: return with_role('doctor_eval', lambda: policy.generate(case, seed, opening=opening) if opening else policy.generate(case, seed))
        finally:
            torch.set_rng_state(cpu)
            if gpu is not None: torch.cuda.set_rng_state_all(gpu)
    play.workers = replicas                                         # execution only: concurrent independent episodes
    cases = eval_cases(ROOT / ev['cases_file'], ev['split'], json.loads(MANIFEST.read_text()))
    return MilestoneEvaluator(play, outcome_fn, cases, ev['split'], file_sha(str(MANIFEST.relative_to(ROOT))), protocol_sha256(cfg, ROOT), ev['seed'])


def launch(cfg: dict, phase, max_cases=None) -> dict:
    tr = build(cfg)
    if phase == 'A':
        if tr.state.case_cursor != 0: raise RuntimeError('phase A expects case_cursor 0')
        rec = tr.step()
        return {'phase': 'A', 'case_id': rec['case_id'], 'counts': rec['counts'], 'rollout_policy_sha256': rec['policy_sha256'],
                'policy_sha256_after': tr.state.policy_sha256, 'update': rec['update'], 'case_cursor': tr.state.case_cursor}
    if phase == 'B':
        before = tr.state.lineage_cursor; tr._maybe_prompt_event()
        if tr.state.lineage_cursor == before: raise RuntimeError('phase B: the prompt event did not fire (too few decided trajectories?)')
        ev = json.loads(tr._event_path(tr.state.lineage_cursor).read_text())
        return {'phase': 'B', 'event': ev['event'], 'decision': ev['decision'], 'rerun_policy_sha256': ev['rerun_policy_sha256'],
                'evidence_policy_sha256': ev['evidence_policy_sha256'], 'case_cursor': tr.state.case_cursor}
    tr.run(max_cases=max_cases); return {'case_cursor': tr.state.case_cursor, 'eval_cursor': tr.state.eval_cursor}


def main(argv=None):
    a = argparse.ArgumentParser(); a.add_argument('--config', default='configs/online_rl/tree_v1.json'); a.add_argument('--launch', action='store_true')
    a.add_argument('--phase', choices=['A', 'B']); a.add_argument('--condition', choices=['outcome_only', 'rubric_only', 'fixed', 'evolving'])
    a.add_argument('--max-cases', type=int, help='stop after this many more cases (a due milestone evaluation still runs first)'); x = a.parse_args(argv)
    cfg = json.loads((ROOT / x.config).read_text())
    if x.condition:
        from src.online_rl.tree.conditions import arm_config
        cfg = arm_config(cfg, x.condition)
    todo = blockers(cfg) + smoke_problems(cfg) + pilot_problems(cfg) + setup_problems(cfg, x.config) + phase_problems(cfg, x.phase)
    usage_log = os.environ.get('API_USAGE_LOG') or str(ROOT / cfg['output_dir'] / 'api_usage.jsonl')
    print(json.dumps({'config': x.config, 'condition': cfg.get('condition'), 'phase': x.phase, 'output_dir': cfg['output_dir'], 'max_agent_turns': cfg['rollout']['rounds'],
                      'branching': cfg['rollout']['branching'], 'api_usage_log': usage_log, 'blockers': todo}, indent=1), flush=True)
    if todo: raise SystemExit('not launching: ' + '; '.join(todo[:3]) + (' ...' if len(todo) > 3 else ''))
    if not x.launch: raise SystemExit('not launching (pass --launch after approval)')
    os.environ['API_USAGE_LOG'] = usage_log                                   # existing ledger: src/common/usage.py (no credentials)
    if x.max_cases is not None and x.phase is not None: raise SystemExit('--max-cases is not used with --phase')
    print(json.dumps(launch(cfg, x.phase, x.max_cases), indent=1, default=str), flush=True)


if __name__ == '__main__': main()
