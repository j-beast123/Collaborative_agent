"""compact_runtime v7_7: one Stage-1-only system supplement on top of v7_5_v7_6; nothing else moves."""
import json
from pathlib import Path

from src.online_rl.tree import compact_runtime as CR
from src.online_rl.tree.experiment import load_config
from src.online_rl.tree.luna import LunaDecisionClient
from tests.test_compact_runtime_v7 import CFG, Transport
from tests.test_scoring_batch import CASE, RUBRIC

SUPPLEMENT = Path('prompts/sherpa') / CR.LITERAL_SUPPLEMENT


def scorer(revision):
    cfg = load_config(CFG); t = Transport()
    client = LunaDecisionClient(model=cfg['judges']['scorer_model'], key='OFFLINE', system=Path(cfg['judges']['scorer_prompt_file']).read_text(),
                                max_tokens=4096, post=t)
    app = LunaDecisionClient(model=client.model, key='OFFLINE', system=Path(cfg['judges']['applicability_prompt_file']).read_text(),
                             max_tokens=4096, post=t)
    return t, CR.compact_scorer(client, app, 4, revision=revision)


def payload(kind):
    packet = ({'stage': 'applicability', 'state': {'dialogue_before_target_reply': []}, 'items': {'criterion': {'instructions': ''}}} if kind == 'applicability'
              else {'stage': 'level', 'responses': [], 'rubric_items': {'criterion': {'instructions': ''}}})
    return {'model': 'm', 'max_tokens': 1, 'messages': [{'role': 'system', 'content': 'base'}, {'role': 'user', 'content': json.dumps(packet)}]}


def upgraded(kind, revision):
    return CR.upgrade(payload(kind), revision=revision)


def test_only_the_applicability_system_prompt_gains_the_supplement():
    old, new = upgraded('applicability', 'v7_5_v7_6'), upgraded('applicability', 'v7_7')
    assert new['messages'][0]['content'] == old['messages'][0]['content'] + '\n\n' + SUPPLEMENT.read_text()
    assert new['messages'][1] == old['messages'][1] and {k: v for k, v in new.items() if k != 'messages'} == {k: v for k, v in old.items() if k != 'messages'}
    assert upgraded('behavior', 'v7_7') == upgraded('behavior', 'v7_5_v7_6')          # behavior scoring request is byte-identical


def test_contract_is_unchanged_and_cache_identity_is_distinct():
    _, old = scorer('v7_5_v7_6'); _, new = scorer('v7_7')
    assert new.review_state.applicability_contract == old.review_state.applicability_contract == CR.REFERENT_APP_CONTRACT
    assert new.review_state.protocol != old.review_state.protocol                     # a literal judgment never reuses an older cached one
    assert new.compact_revision == 'v7_7' and old.compact_revision == 'v7_5_v7_6'


def test_supplement_is_generic_and_stage1_only():
    text = SUPPLEMENT.read_text().lower()
    assert 'stage 1 only' in text and 'does not narrow' in text
    for word in ('newborn', 'bronchopulmonary', 'bpd', 'critical_evaluation', 'score 2', 'r_a'):
        assert word not in text


def test_final_online_config_differs_from_the_validated_one_only_in_the_revision_and_labels():
    base = json.loads(Path('configs/experiments/train120_max4_online_chunk8_bank_final120.json').read_text())
    new = json.loads(Path('configs/experiments/train120_max4_online_chunk8_bank_final120_r2.json').read_text())
    def flat(o, path=''):
        if isinstance(o, dict):
            for k, v in o.items(): yield from flat(v, f'{path}/{k}')
        else: yield path, o
    a, b = dict(flat(base)), dict(flat(new))
    changed = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
    assert b['/judges/compact_runtime'] == 'v7_7'
    allowed = ('/judges/compact_runtime', '/judges/compact_prompt_sha256/prompts/sherpa/compact_applicability_literal_v7_7.txt')
    assert all(k in allowed or k.endswith(('output_root', 'run_label', 'execution_speed', 'applicability_literal')) for k in changed), changed
