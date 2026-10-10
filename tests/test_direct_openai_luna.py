"""Direct OpenAI judge transport and role routing, without a training or network call."""
import copy
import json
from pathlib import Path

import pytest

from scripts.online_rl import run_tree
from src.common.config import IntegrationUnavailable
from src.online_rl.tree import experiment
from src.online_rl.tree.luna import LunaDecisionClient, OPENAI_URL, URL


def completed(content, *, finish='stop', usage=None):
    return {'id': 'chatcmpl-test', 'model': 'gpt-6-luna',
            'choices': [{'finish_reason': finish, 'message': {'content': content}}],
            'usage': usage or {'prompt_tokens': 12, 'completion_tokens': 8}}


def test_direct_gate_stage1_and_stage2_use_openai_and_keep_contracts():
    sent = []
    answers = iter([
        '{"answers":{"safe":"N"}}',
        '{"answers":{"r1":{"applicable":true,"reason":"Visible."}}}',
        '{"answers":{"reply1":{"r1":{"level":"2","quote":"Asked."}}}}',
    ])

    def post(url, payload, key, timeout):
        sent.append((url, payload, key, timeout))
        return completed(next(answers))

    client = LunaDecisionClient(model='gpt-6-luna', key='fixture-secret', system='Judge independently.',
                                provider='openai_direct', post=post, max_tokens=2048)
    gate, gate_rec = client.decide('{}', {'safe': {'instructions': 'Check.', 'criteria': ['Y', 'N']}}, 'gate')
    stage1, app_rec = client.judge('applicability', {'prefix': ['Doctor asks why']},
                                   {'r1': {'condition': 'Question visible'}},
                                   {'applicable': {'type': 'boolean'}, 'reason': {'type': 'string'}}, 'applicability')
    stage2, score_rec = client.judge_batch('behavior', {'prefix': ['Doctor asks why']},
        [{'response_id': 'reply1', 'target_agent_reply': 'Asked.'}],
        {'r1': {'score_anchors': {'0': 'absent', '1': 'partial', '2': 'complete'}}},
        {'level': {'type': 'string', 'enum': ['0', '1', '2']}, 'quote': {'type': 'string'}}, 'scorer')

    assert gate['safe']['value'] == 'N' and stage1['r1']['applicable'] is True
    assert stage2['reply1']['r1']['level'] == '2'
    assert all(url == OPENAI_URL and key == 'fixture-secret' and timeout == 180 for url, _, key, timeout in sent)
    assert all(payload['model'] == 'gpt-6-luna' and 'max_completion_tokens' in payload
               and 'max_tokens' not in payload for _, payload, _, _ in sent)
    assert all(payload['response_format']['json_schema']['strict'] is True for _, payload, _, _ in sent)
    assert 'target_agent_reply' not in sent[1][1]['messages'][1]['content']
    assert 'target_agent_reply' in sent[2][1]['messages'][1]['content']
    for rec in (gate_rec, app_rec, score_rec):
        assert rec['provider'] == 'openai_direct'
        assert rec['completions'][0]['finish_reason'] == 'stop'
        assert rec['completions'][0]['usage']['completion_tokens'] == 8


def test_direct_incomplete_completion_is_operational_error_and_keeps_evidence():
    sent = []

    def post(url, payload, key, timeout):
        sent.append(url)
        return completed('', finish='length')

    client = LunaDecisionClient(model='gpt-6-luna', key='fixture-secret', system='Judge.',
                                provider='openai_direct', post=post)
    with pytest.raises(IntegrationUnavailable, match='incomplete completion') as exc:
        client.decide('{}', {'q': {'criteria': ['Y', 'N']}}, 'gate')
    assert sent == [OPENAI_URL]
    assert exc.value.audit['finish_reason'] == 'length'
    assert exc.value.audit['usage']['prompt_tokens'] == 12
    assert 'fixture-secret' not in str(exc.value)


def test_direct_repairs_schema_with_same_provider_and_model():
    sent = []
    replies = iter(['not JSON', '{"answers":{"q":"Y"}}'])

    def post(url, payload, key, timeout):
        sent.append((url, payload))
        return completed(next(replies))

    client = LunaDecisionClient(model='gpt-6-luna', key='fixture-secret', system='Judge.',
                                provider='openai_direct', post=post)
    result, record = client.decide('{}', {'q': {'criteria': ['Y', 'N']}}, 'gate')
    assert result['q']['value'] == 'Y' and record['repaired'] and record['repair_ok']
    assert len(record['completions']) == 2
    assert all(url == OPENAI_URL and payload['model'] == 'gpt-6-luna' for url, payload in sent)


def test_direct_secret_settings_and_role_identity(tmp_path, monkeypatch):
    directory = tmp_path / '.config/sherpa'
    directory.mkdir(parents=True)
    (directory / 'openai.env').write_text('OPENAI_API_KEY="fixture-secret"\n')
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    cfg = experiment.load_config('configs/experiments/train120_v9_fresh_source_case_direct_luna.json')
    assert run_tree.load_key(cfg) == 'fixture-secret'
    client = run_tree.judge_client('gpt-6-luna', 'fixture-secret', 'Judge.', provider='openai_direct')
    assert client.provider == 'openai_direct' and client.model == 'gpt-6-luna'
    assert run_tree.evidence_gate_identity(cfg)['client_protocol_sha256'] != run_tree.evidence_gate_identity(
        {**cfg, 'api_provider': 'openrouter', 'judges': {**cfg['judges'], 'api_provider': 'openrouter'}})['client_protocol_sha256']
    assert run_tree.intervention_audit_identity(cfg)['client_protocol_sha256'] != run_tree.intervention_audit_identity(
        {**cfg, 'api_provider': 'openrouter', 'judges': {**cfg['judges'], 'api_provider': 'openrouter'}})['client_protocol_sha256']
    assert not run_tree.blockers(cfg)
    wrong = copy.deepcopy(cfg)
    wrong['judges']['scorer_model'] = 'openai/gpt-5.6-luna'
    assert any('judges.scorer_model must be gpt-6-luna' in problem for problem in run_tree.blockers(wrong))


def test_legacy_openrouter_payload_remains_compatible():
    sent = []

    def post(url, payload, key, timeout):
        sent.append((url, payload))
        return {'choices': [{'message': {'content': json.dumps({'answers': {'q': 'Y'}})}}]}

    client = LunaDecisionClient(model='openai/gpt-5.6-luna', key='fixture-secret', system='Judge.', post=post)
    client.decide('{}', {'q': {'criteria': ['Y', 'N']}}, 'gate')
    assert sent[0][0] == URL and 'max_tokens' in sent[0][1] and 'max_completion_tokens' not in sent[0][1]
