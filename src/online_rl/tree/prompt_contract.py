"""Tree-specific prompt guards; legacy integrated prompts keep their own schema."""
import json
import re
from src.online_rl.integrated.prompt_search import option_text_violations
from src.prompt_opt.reflect import copy_violations


def structural_problems(text: str) -> list[str]:
    problems = []
    decoder = json.JSONDecoder()
    schema = None
    for match in re.finditer(r'\{', text):
        try:
            value, _ = decoder.raw_decode(text[match.start():])
        except ValueError:
            continue
        if isinstance(value, dict) and 'candidate_rubric' in value:
            schema = value
            break
    if schema is None:
        problems.append('missing nested JSON output schema')
    else:
        for key in ('rubric_id', 'source_case_id', 'source_trajectory_id', 'source_outcome'):
            if key not in schema: problems.append(f'missing output key {key}')
        analysis = schema.get('analysis')
        for key in ('user_contribution', 'agent_side_state', 'agent_handling', 'selected_agent_turn_id', 'evidence_spans'):
            if not isinstance(analysis, dict) or key not in analysis:
                problems.append(f'missing analysis.{key}')
        candidate = schema.get('candidate_rubric')
        if not isinstance(candidate, dict):
            problems.append('candidate_rubric must be an object')
        else:
            if not candidate.get('name'): problems.append('missing candidate_rubric.name')
            app = candidate.get('applicability')
            for key in ('user_side_context', 'agent_side_state'):
                if not isinstance(app, dict) or not app.get(key):
                    problems.append(f'missing applicability.{key}')
            scores = candidate.get('scores')
            if not isinstance(scores, dict) or set(scores) != {'0', '1', '2'}:
                problems.append('scores must define exactly 0, 1, 2')
            else:
                for level, score in scores.items():
                    for key in ('user_side_context', 'agent_side_state', 'behavior'):
                        if not isinstance(score, dict) or not score.get(key):
                            problems.append(f'missing scores.{level}.{key}')
    if len(text.split()) > 1600:
        problems.append(f'too long ({len(text.split())} words; limit 1600)')
    return problems


def prompt_violations(text: str, allowed: list[str], cases: list[dict]) -> list[str]:
    content = [c['question'] for c in cases] + [v for c in cases for v in c['options'].values()]
    copied = copy_violations(text, allowed, content)
    return (([f'copied 4-grams: {copied[:10]}'] if copied else []) + structural_problems(text)
            + [f'option text: {o}' for o in option_text_violations(
                text, allowed, [v for c in cases for v in c['options'].values()])])
