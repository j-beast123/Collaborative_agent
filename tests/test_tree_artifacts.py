import json
from pathlib import Path
import test_tree_v1 as T
from src.online_rl.tree.artifacts import export_run


def test_review_preserves_separate_scores_and_every_candidate(tmp_path):
    trainer, _, _ = T.trainer(tmp_path, T.cfg(cases=4, every=2, min_traj=4))
    trainer.run()
    review = tmp_path/'review'
    summary = json.loads((review/'summary.json').read_text())
    assert summary['cases'] == 4
    scores = [json.loads(line) for line in (review/'prompt_scores.jsonl').read_text().splitlines()]
    assert {'search','acceptance'} <= {row['split'] for row in scores}
    for row in scores:
        if row.get('J') is not None:
            assert abs(row['J'] - row['S_success'] * row['S_failure']) < 1e-12
    candidates = [json.loads(line) for line in (review/'prompt_optimization/candidates.jsonl').read_text().splitlines()]
    assert candidates
    for candidate in candidates:
        if candidate['sha256']:
            assert (review/'prompts'/f"{candidate['sha256']}.txt").exists()
    assert list((review/'evaluations').glob('*/*/effectiveness.jsonl'))
    assert list((review/'evaluations').glob('*/*/response_scores.jsonl'))
    snapshot=(review/'summary.json').read_text()
    export_run(tmp_path)
    assert (review/'summary.json').read_text() == snapshot
    assert 'S_success' in (review/'report.md').read_text()


def test_review_handles_no_completed_cases(tmp_path):
    trainer, _, _ = T.trainer(tmp_path)
    result=export_run(tmp_path)
    assert result['cases'] == 0
    assert 'No prompt comparison' in (tmp_path/'review/report.md').read_text()
