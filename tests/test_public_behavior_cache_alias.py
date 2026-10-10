"""Content aliases share judgments, not scores across different inputs."""
from dataclasses import replace
from src.online_rl.tree.cache import CallCache
from src.online_rl.tree.scoring import score_nodes
from src.online_rl.tree.records import TrajectoryRubric
from tests.test_public_mc_evidence import Client, scorer, rubric, REPLY, CASE
from tests.test_scoring_batch import node


def run(cache, judge, alias='source-rubric', **changes):
    definition = {**rubric(), 'rubric_id': alias, **changes.pop('criterion', {})}
    kept = TrajectoryRubric(alias, 'source', 'fixed', None, definition, True, (), None)
    target = replace(node(0), response=changes.pop('response', REPLY), **changes)
    return score_nodes(CASE, [target], [kept], judge, cache)


def test_public_source_and_bank_alias_share_both_decisions():
    client = Client(); judge = scorer(client); cache = CallCache(None)
    first = run(cache, judge)
    count = len(client.calls)
    second = run(cache, judge, 'bank-alias')
    assert len(client.calls) == count
    assert first[0].score == second[0].score == 1.0
    assert first[0].rubric_id == 'source-rubric'
    assert second[0].rubric_id == 'bank-alias'
    assert len([c for c in client.calls if c['stage'] == 'level']) == 1
    assert len([c for c in client.calls if c['stage'] == 'applicability']) == 1


def test_public_cache_partitions_context_reply_content_and_settings():
    client = Client(); judge = scorer(client); cache = CallCache(None)
    run(cache, judge)
    # Changed input must not use the earlier alias judgment. The fake judge's
    # answer stays structurally valid; this test measures cache behavior only.
    run(cache, judge, 'bank-alias', prefix=(('doctor', 'D1', 1, 'Different pending request.'),))
    run(cache, judge, 'bank-alias', criterion={'name': 'Changed public criterion name'})
    run(cache, judge, 'bank-alias', trace={'pre_response_belief': {'state': 'different'}})
    run(cache, judge, 'bank-alias', response=REPLY + ' Additional visible response content.')
    judge.protocol = 'different-settings-hash'
    run(cache, judge, 'bank-alias')
    assert len([c for c in client.calls if c['stage'] == 'level']) == 6
