"""DEV-only OURS optimization arms: production identities unchanged, each arm differs from BASE only in its declared
hyperparameter, the DEV split is disjoint from training, and the scorer's Stage-1 request does not depend on the reply."""
import json
from pathlib import Path

import pytest

from src.online_rl.tree import experiment as X

ROOT = Path(__file__).resolve().parents[1]
PROD = 'configs/experiments/reward_prompt_matrix/production'
DEV = 'configs/experiments/dev_opt'
# run id and resume fingerprint of every production arm, computed with the code before experiment.variant existed (1fb1bbe)
PRODUCTION_IDENTITY = {
    'r03_o07_evolving_300': ('r03_o07__evolving__sonnet46__seed0', 'ceda18172240b144ac179bee9c271bef317e2c885457aa8a3c8e66370788f3fd'),
    'r05_o05_evolving_300': ('r05_o05__evolving__sonnet46__seed0', '093ad76d54316ad9ef3e9fd44c6bdbff76672f2454e0952a67a5728984459637'),
    'r05_o05_p0_fixed_300': ('r05_o05__p0fixed__sonnet46__seed0', '0b039fe8e559e2ebc8ce9f3702e48fa53dd3eaf476a83d708b78a175a75839f9'),
    'r07_o03_evolving_300': ('r07_o03__evolving__sonnet46__seed0', 'cbae241dc7a72faef0a153c58632c62d56076833f04eb1ca32d0e7da4d9c56d1'),
}
ARMS = {'base': {}, 'diversity': {'rollout.top_p': 1.0}, 'optimizer': {'optimizer.learning_rate': 3e-05},
        'both': {'rollout.top_p': 1.0, 'optimizer.learning_rate': 3e-05}}
BOOKKEEPING = {'name', 'condition', 'output_dir', 'notes'}


def flat(d, prefix=''):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict): out.update(flat(v, f'{prefix}{k}.'))
        else: out[f'{prefix}{k}'] = v
    return out


def arm(name):
    return X.load_config(f'{DEV}/{name}.json')


@pytest.mark.parametrize('name', sorted(PRODUCTION_IDENTITY))
def test_production_run_id_and_fingerprint_unchanged(name):
    cfg = X.load_config(f'{PROD}/{name}.json')
    assert (cfg['experiment']['run_id'], X.fingerprint(cfg)) == PRODUCTION_IDENTITY[name]
    assert 'variant' not in cfg['experiment'] and 'training_hyperparameters' not in cfg['experiment']


@pytest.mark.parametrize('name', sorted(ARMS))
def test_arm_differs_from_base_only_in_its_hyperparameter(name):
    base, cfg = flat(arm('base')), flat(arm(name))
    diff = {k for k in set(base) | set(cfg) if base.get(k) != cfg.get(k)}
    expected = {k for k in ARMS[name]}
    research = {k for k in diff if k.split('.')[0] not in BOOKKEEPING and not k.startswith('experiment.')}
    assert research == expected
    assert all(cfg[k] == v for k, v in ARMS[name].items())


@pytest.mark.parametrize('name', sorted(ARMS))
def test_arm_keeps_canonical_ours_semantics(name):
    cfg, prod = arm(name), X.load_config(f'{PROD}/r05_o05_evolving_300.json')
    assert cfg['agent_reward'] == prod['agent_reward'] == {'w_outcome': 0.5, 'w_rubric': 0.5, 'r_a_weighting': 'equal'}
    assert cfg['prompt_update'] == prod['prompt_update'] and cfg['prompt_update']['enabled']
    for section in ('rubric', 'feedback', 'selection', 'verification', 'judges', 'data', 'policy'):
        assert cfg[section] == prod[section], section
    assert {k: v for k, v in cfg['rollout'].items() if k != 'top_p'} == {k: v for k, v in prod['rollout'].items() if k != 'top_p'}
    assert {k: v for k, v in cfg['optimizer'].items() if k != 'learning_rate'} == {k: v for k, v in prod['optimizer'].items() if k != 'learning_rate'}
    assert X.theta0_identity(cfg) == X.theta0_identity(prod)
    assert cfg['five_gpu']['batch_cases'] == 5 and cfg['schedule'] == {**prod['schedule'], 'cases': 60}
    assert cfg['five_gpu']['train_order_sha256'] == prod['five_gpu']['train_order_sha256']
    assert cfg['five_gpu']['checkpoint_evaluation_enabled'] is False          # never evaluates CasiMedicos622
    assert cfg['output_dir'].startswith('outputs/dev_opt/')


def test_arms_have_distinct_run_ids_and_fingerprints():
    cfgs = [arm(n) for n in ARMS]
    assert len({c['experiment']['run_id'] for c in cfgs}) == len({X.fingerprint(c) for c in cfgs}) == len(ARMS)
    prod = {fp for _, fp in PRODUCTION_IDENTITY.values()}
    assert not prod & {X.fingerprint(c) for c in cfgs}


def test_fingerprint_follows_training_hyperparameters():
    cfg = arm('base'); before = X.fingerprint(cfg)
    cfg['optimizer']['learning_rate'] = 2e-05
    assert X.fingerprint(cfg) != before


@pytest.mark.parametrize('bad', ['Dev', 'dev-base', '', 'dev__x', 3])
def test_variant_tag_is_validated(bad):
    raw = X.load_raw(f'{DEV}/base.json'); raw['experiment']['variant'] = bad
    with pytest.raises(ValueError): X.resolve(raw)


def test_dev_eval_override_is_the_dev_split():
    o = json.loads((ROOT / f'{DEV}/dev_eval.json').read_text())
    assert 'casimedicos' not in o['eval_file'].lower() and o['eval_cases'] == 100
    from scripts.dev_opt.make_dev_split import DEV as DEV_FILE, FIRST, LAST, MAX_TRAINED_POSITION
    assert o['eval_file'].endswith(DEV_FILE) and FIRST > MAX_TRAINED_POSITION and LAST - FIRST + 1 == o['eval_cases']


def test_dev_split_disjoint_from_training_order():
    m = ROOT / 'data/interim/medmcqa/five_gpu_train500/dev100_order400_499.manifest.json'
    order = ROOT / 'data/interim/medmcqa/five_gpu_train500/train500_order_seed20261008.json'
    if not m.exists() or not order.exists(): pytest.skip('DEV split not built here (scripts/dev_opt/make_dev_split.py)')
    man, ids = json.loads(m.read_text()), json.loads(order.read_text())
    o = json.loads((ROOT / f'{DEV}/dev_eval.json').read_text())
    assert man['dev_sha256'] == o['eval_sha256'] and man['case_ids'] == ids[400:500]
    assert not set(man['case_ids']) & set(ids[:300])


def test_eval_override_refuses_trained_cases(tmp_path):
    order = ROOT / 'data/interim/medmcqa/five_gpu_train500/train500_order_seed20261008.json'
    if not order.exists(): pytest.skip('train500 data not present')
    from scripts.online_rl import async_eval
    cfg = arm('base'); cfg['schedule']['cases'] = 450                         # would train on DEV positions 400..449
    with pytest.raises(SystemExit): async_eval.apply_eval_override(cfg, f'{DEV}/dev_eval.json')
    assert async_eval.apply_eval_override(arm('base'), f'{DEV}/dev_eval.json')['five_gpu']['eval_cases'] == 100


def test_stage1_scorer_request_does_not_depend_on_the_reply():
    """Siblings share their pre-reply dialogue, so their Stage-1 (applicability) requests are byte-identical; a different
    applicability decision among siblings is therefore scorer nondeterminism, not a context difference."""
    from src.online_rl.tree.backends import two_stage_structured_scorer
    from src.online_rl.tree.luna import LunaDecisionClient
    rubric = json.loads((ROOT / 'tests/fixtures/dev_opt_visible_rubric.json').read_text())   # a kept production rubric
    rid = rubric['rubric_id']
    sent = []

    def post(url, payload, key, timeout):
        sent.append(payload); stage = json.loads(payload['messages'][-1]['content'])['stage']
        ans = {rid: {'applicable': True, 'evidence_basis': 'e', 'missing_requirement': ''}} if stage == 'applicability' else \
              {rid: {'level': '1', 'anchor_basis': 'x'}}
        return {'choices': [{'message': {'content': json.dumps({'answers': ans})}}], 'usage': {}}
    scorer = two_stage_structured_scorer(LunaDecisionClient(model='m', key='k', system='s', post=post), observed_context=True)
    view = {'question': 'q', 'options': {'A': 'a', 'B': 'b'}}
    prefix = [{'speaker': 'doctor', 'stage': 'D1', 'round': 1, 'text': 'I think A.'}]
    scorer(view, prefix, [rubric], 'reply one'); scorer(view, prefix, [rubric], 'a completely different reply')
    stage1 = [p for p in sent if json.loads(p['messages'][-1]['content'])['stage'] == 'applicability']
    stage2 = [p for p in sent if json.loads(p['messages'][-1]['content'])['stage'] != 'applicability']
    assert len(stage1) == 2 and stage1[0] == stage1[1]
    assert len(stage2) == 2 and stage2[0] != stage2[1]
    assert 'temperature' not in stage1[0]                                     # provider-default sampling: decisions may vary
