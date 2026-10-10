"""Exactly one always-applicable accuracy criterion, outside generation/verification."""
import json
from pathlib import Path
from src.online_rl.integrated.records import sha256_text
from src.online_rl.tree.records import TrajectoryRubric

ID = 'fixed_medical_accuracy_v1'
ROOT = Path(__file__).resolve().parents[3]


def criterion(cfg):
    settings = cfg['fixed_accuracy']
    text = (ROOT / settings['criterion_file']).read_text()
    if sha256_text(text) != settings['criterion_sha256']: raise ValueError('Fixed accuracy criterion changed')
    rubric = json.loads(text)
    if rubric.get('rubric_id') != ID: raise ValueError('Fixed accuracy ID changed')
    from src.online_rl.tree.agent_state import rubric_problems
    problems = rubric_problems(rubric)
    if problems: raise ValueError(str(problems))
    return TrajectoryRubric(ID, 'fixed_case_criterion', settings['criterion_sha256'], text, rubric, True, (), None)


def scoring_book(partner_kept, cfg):
    if any(r.rubric_id == ID for r in partner_kept): raise ValueError('Duplicate fixed accuracy criterion')
    return [*partner_kept, criterion(cfg)] if cfg.get('fixed_accuracy', {}).get('enabled') else list(partner_kept)


def always_applicable(scorer):
    """Keep the same scorer/behavior requests; never ask an API to waive accuracy."""
    original_review = scorer.review_state
    scorer.public_history_only = True
    def review(view, prefix, rubrics):
        partner = [r for r in rubrics if r['rubric_id'] != ID]
        result = original_review(view, prefix, partner) if partner else {}
        if any(r['rubric_id'] == ID for r in rubrics):
            result[ID] = {'applicable': True, 'stage1': {'applicable': True,
                'evidence_basis': 'Fixed medical accuracy applies to every Agent response.', 'missing_requirement': ''}}
        return result
    scorer.review_state = review
    return scorer
