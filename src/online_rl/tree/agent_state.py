"""Visible conversation context and behavior; no hidden beliefs or trace inputs."""
SCHEMA = 'observed_context_behavior_v1'
STATE_PROTOCOL = 'visible_pre_reply_history_v1'


def state_conditioned(rubric):
    return 'applicability' in rubric


def visible_rubric(rubric):
    fields = ('name', 'applicability' if state_conditioned(rubric) else 'applicability_condition', 'required_behavior', 'scores')
    return {**({'rubric_id': rubric['rubric_id']} if 'rubric_id' in rubric else {}), **{k: rubric.get(k) for k in fields}}


def flat_rubric(rubric):
    if not state_conditioned(rubric):
        return visible_rubric(rubric)
    a = rubric['applicability']
    return {**({'rubric_id': rubric['rubric_id']} if 'rubric_id' in rubric else {}), 'name': rubric['name'],
            'applicability_condition': f"User-side context: {a['user_side_context']} Agent-side state: {a['agent_side_state']}",
            'required_behavior': rubric['required_behavior'], 'scores': {k: v['behavior'] for k, v in rubric['scores'].items()}}


def intervention_rubric(rubric, level):
    """Return only applicability and the selected level's behavior for a guided rerun."""
    level = str(level)
    if level not in ('0', '2'):
        raise ValueError(f'intervention level must be 0 or 2, got {level!r}')
    applicability = rubric.get('applicability')
    if applicability is None:
        applicability = rubric.get('applicability_condition')
    scores = rubric.get('scores') or {}
    selected = scores.get(level)
    behavior = selected.get('behavior') if isinstance(selected, dict) else selected
    if not isinstance(applicability, (str, dict)) or not applicability:
        raise ValueError('rubric intervention requires applicability')
    if not isinstance(behavior, str) or not behavior.strip():
        raise ValueError(f'rubric intervention requires Score-{level} behavior')
    return {'applicability': applicability, 'scores': {level: {'behavior': behavior}}}


def rubric_problems(rubric):
    from src.judges.rubric_schema import validate_rubric
    from src.judges.rubric_scoring import validate_visible_rubric
    if not isinstance(rubric, dict): return ['no rubric object']
    problems = validate_rubric(rubric) if state_conditioned(rubric) else validate_visible_rubric(rubric)
    if not isinstance(rubric.get('required_behavior'), str) or not rubric['required_behavior'].strip():
        problems.append('missing required_behavior')
    return problems
