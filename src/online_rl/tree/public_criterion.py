"""Pure public criterion identity; no persistent or cross-case bank."""
import unicodedata
from src.online_rl.integrated.records import sha256_json
from src.judges.rubric_schema import PUBLIC_CRITERION_FIELDS, validate_public_criterion

def _text(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError('Public criterion requires nonblank criterion/evidence strings')
    return ' '.join(unicodedata.normalize('NFC', value).split())


def criteria(rubric):
    """Exact normalized behavioral identity; no names, IDs, provenance or guides."""
    problems = validate_public_criterion(rubric)
    if problems: raise ValueError('; '.join(problems))
    app = rubric.get('applicability')
    if not isinstance(app, dict) or set(app) != {'user_side_context', 'agent_side_state'}:
        raise ValueError('Public criterion requires both applicability sides')
    normalized = {key: _text(app[key]) for key in sorted(app)}
    scores = rubric.get('scores')
    if not isinstance(scores, dict) or set(scores) != {'0', '1', '2'}:
        raise ValueError('Public criterion requires all three score boundaries')
    behaviors = {}
    for level in ('0', '1', '2'):
        row = scores[level]
        if not isinstance(row, dict):
            raise ValueError('Public criterion requires structured score boundaries')
        if any(_text(row.get(key)) != normalized[key] for key in normalized):
            raise ValueError('Score-level applicability differs from canonical condition')
        behaviors[level] = _text(row.get('behavior'))
    # Absent public fields are not filled with nulls or inferred from analysis:
    # legacy content hashes and existing immutable bank IDs remain identical.
    return {'applicability': normalized, 'required_behavior': _text(rubric.get('required_behavior')),
            'scores': behaviors, **{key: _text(rubric[key]) for key in PUBLIC_CRITERION_FIELDS if key in rubric}}


def content_hash(rubric):
    return sha256_json(criteria(rubric))
