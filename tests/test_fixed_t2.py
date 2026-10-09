"""FIXED_T2 protocol: three 120-case reward arms that differ only in reward weights; PromptOpt off, two Agent turns, the scorer
request cache and per-response audit records. OURS without PromptOpt is the R05 arm itself."""
import copy
import json
import threading
from pathlib import Path

import pytest

from src.online_rl.tree import experiment as X
from src.online_rl.tree.cache import CallCache
from src.online_rl.tree.luna import REQUEST_CACHE, LunaDecisionClient
from src.online_rl.tree.rollout import build_tree

ROOT = Path(__file__).resolve().parents[1]
T2 = 'configs/experiments/fixed_t2'
ARMS = {'r03_o07': (0.3, 0.7), 'r05_o05': (0.5, 0.5), 'r07_o03': (0.7, 0.3)}
P0 = 'f8d6e39470d7f8a93ce8224cea9e5aa2ed2fa07ced9d982cc5e16d0e495d9bad'
NAMING = {'name', 'condition', 'output_dir', 'experiment.run_id', 'experiment.reward_weight_rubric', 'experiment.reward_weight_outcome'}


def flat(d, prefix=''):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict): out.update(flat(v, f'{prefix}{k}.'))
        else: out[f'{prefix}{k}'] = v
    return out


def arm(name):
    return X.load_config(f'{T2}/{name}.json')


def test_arms_differ_only_in_reward_weights():
    cfgs = {n: flat(arm(n)) for n in ARMS}
    ref = cfgs['r05_o05']
    for n, c in cfgs.items():
        diff = {k for k in set(ref) | set(c) if ref.get(k) != c.get(k)} - NAMING
        assert diff <= {'agent_reward.w_rubric', 'agent_reward.w_outcome'}, (n, diff)
        assert (c['agent_reward.w_rubric'], c['agent_reward.w_outcome']) == ARMS[n]
    assert len({X.fingerprint(arm(n)) for n in ARMS}) == 3


@pytest.mark.parametrize('name', sorted(ARMS))
def test_common_protocol(name):
    cfg, old = arm(name), X.load_config('configs/experiments/reward_prompt_matrix/production/r05_o05_p0_fixed_300.json')
    assert cfg['schedule']['cases'] == 120 and cfg['five_gpu']['batch_cases'] == 5 and cfg['rollout']['rounds'] == 2
    assert cfg['experiment']['prompt_mode'] == 'p0_fixed' and cfg['prompt_update']['enabled'] is False
    assert cfg['rubric']['prompt_sha256'] == P0 and cfg['rubric']['enabled']
    assert cfg['judges']['scorer_request_cache'] == REQUEST_CACHE
    assert cfg['five_gpu']['checkpoint_evaluation_enabled'] is False and cfg['output_dir'].startswith('outputs/fixed_t2_120/')
    # everything else is the frozen reward-matrix P0_FIXED method
    for section in ('rubric', 'feedback', 'selection', 'verification', 'intervention', 'data', 'policy', 'advantage_normalization'):
        assert cfg[section] == old[section], section
    assert {k: v for k, v in cfg['rollout'].items() if k != 'rounds'} == {k: v for k, v in old['rollout'].items() if k != 'rounds'}
    assert {k: v for k, v in cfg['judges'].items() if k != 'scorer_request_cache'} == old['judges']
    assert {k: v for k, v in cfg['optimizer'].items() if k != 'learning_rate'} == {k: v for k, v in old['optimizer'].items() if k != 'learning_rate'}
    assert X.theta0_identity(cfg) == X.theta0_identity(old) and cfg['schedule']['seed'] == old['schedule']['seed'] == 0
    assert cfg['five_gpu']['train_order_sha256'] == old['five_gpu']['train_order_sha256']
    assert cfg['rollout']['temperature'] == 0.7 and cfg['rollout']['top_p'] == 0.9 and cfg['rollout']['top_k'] is None
    assert cfg['rollout']['branching'] == 4


def test_ours_without_prompt_optimization_is_the_r05_arm():
    raw = X.load_raw(f'{T2}/r05_o05.json'); ours = copy.deepcopy(raw); ours['experiment']['prompt_mode'] = 'evolving'
    a, b = flat(X.resolve(ours)), flat(X.resolve(raw))
    diff = {k for k in set(a) | set(b) if a.get(k) != b.get(k)}
    assert diff == {'name', 'condition', 'output_dir', 'experiment.run_id', 'experiment.prompt_mode', 'prompt_update.enabled'}


@pytest.mark.parametrize('lr, value', [('lr1e5', 1e-05), ('lr3e5', 3e-05)])
def test_optimizer_dev_arms_differ_only_in_cases_and_lr(lr, value):
    base, cfg = flat(arm('r05_o05')), flat(arm(f'dev/{lr}'))
    diff = {k for k in set(base) | set(cfg) if base.get(k) != cfg.get(k)}
    research = {k for k in diff if k.split('.')[0] not in ('name', 'condition', 'output_dir', 'notes') and not k.startswith('experiment.')}
    assert research <= {'schedule.cases', 'optimizer.learning_rate'}
    assert cfg['schedule.cases'] == 30 and cfg['optimizer.learning_rate'] == value and cfg['output_dir'].startswith('outputs/fixed_t2_dev/')


def test_frozen_production_identities_unchanged():
    from tests.test_dev_opt import PRODUCTION_IDENTITY, PROD
    for name, ident in PRODUCTION_IDENTITY.items():
        cfg = X.load_config(f'{PROD}/{name}.json')
        assert (cfg['experiment']['run_id'], X.fingerprint(cfg)) == ident and 'scorer_request_cache' not in cfg['judges']


def test_two_turn_tree_never_exceeds_two_agent_turns():
    class Agent:
        def sample(self, case, prefix, turn, seed): return {'completion_text': f'reply at turn {turn} seed {seed}'}
    class Doctor:
        def turn(self, case, dialogue, round_, rounds, *a): assert rounds == 2; return 'Let us keep discussing.'
        def final(self, *a): return 'Final answer: option C'
    tree = build_tree({'case_id': 'c'}, max_agent_turns=arm('r05_o05')['rollout']['rounds'], branching=4, run_seed=0, policy_sha256='p',
                      protocol_sha256='q', agent=Agent(), doctor=Doctor(), stop_rule=lambda *a: None, final_option=lambda t: 'C',
                      outcome_fn=lambda *a: 1.0, cache=CallCache(None))
    assert max(n.depth for n in tree.nodes) == 2 and len(tree.nodes) == 4 + 16 and len(tree.leaves) == 16
    assert all(len(l.path) == 2 and l.stop_reason == 'max_agent_turns' for l in tree.leaves)


def test_halting_cannot_end_a_two_turn_branch_early():
    from src.online_rl.doctor import halt_reason
    cfg = arm('r05_o05')['rollout']['halting']
    assert cfg['min_agent_turns'] == 2 and cfg['repeat_agent_reply_after_turns'] > 2
    stable = {'current_option': 'C', 'open_issue': False, 'collaborator_objection': False, 'more_info_could_change_decision': False}
    # after one Agent turn, a stable and confident Doctor: the branch must continue to the second (final) turn
    assert halt_reason(agent_turns=1, status=stable, prev_status=stable, last_agent='I agree with C.', options={'A': 'a', 'C': 'c'},
                       min_agent_turns=cfg['min_agent_turns']) is None


# ---------- scorer request cache: the five cases of the protocol ----------
def scorer_with_counter(rec):
    from src.online_rl.tree.backends import two_stage_structured_scorer
    calls = []

    def post(url, payload, key, timeout):
        calls.append(payload); stage = json.loads(payload['messages'][-1]['content'])['stage']
        items = json.loads(payload['messages'][-1]['content'])['items']
        ans = {i: ({'applicable': True, 'evidence_basis': 'e', 'missing_requirement': ''} if stage == 'applicability'
                   else {'level': '1', 'anchor_basis': 'x'}) for i in items}
        return {'choices': [{'message': {'content': json.dumps({'answers': ans})}}], 'usage': {}}
    client = LunaDecisionClient(model=rec.get('model', 'm'), key='k', system=rec.get('system', 's'), post=post, request_cache=REQUEST_CACHE)
    return two_stage_structured_scorer(client, observed_context=True), client, calls


RUBRIC = json.loads((ROOT / 'tests/fixtures/dev_opt_visible_rubric.json').read_text())
VIEW = {'question': 'q', 'options': {'A': 'a', 'B': 'b'}}
PREFIX = [{'speaker': 'doctor', 'stage': 'D1', 'round': 1, 'text': 'I think A.'}]


def stage_count(calls, stage):
    return sum(json.loads(p['messages'][-1]['content'])['stage'] == stage for p in calls)


def test_cache_same_request_different_node_one_backend_call():
    score, client, calls = scorer_with_counter({})
    a = score(VIEW, PREFIX, [RUBRIC], 'reply'); b = score(VIEW, PREFIX, [RUBRIC], 'reply')
    assert len(calls) == 2 and {k: v for k, v in a.items() if k != '_decisions'} == {k: v for k, v in b.items() if k != '_decisions'}


def test_cache_different_response_is_a_separate_stage2_call():
    score, client, calls = scorer_with_counter({})
    score(VIEW, PREFIX, [RUBRIC], 'reply one'); score(VIEW, PREFIX, [RUBRIC], 'reply two')
    assert stage_count(calls, 'applicability') == 1 and stage_count(calls, 'level') == 2


def test_cache_different_dialogue_prefix_is_separate():
    score, client, calls = scorer_with_counter({})
    score(VIEW, PREFIX, [RUBRIC], 'reply'); score(VIEW, PREFIX + [{'speaker': 'agent', 'stage': 'A1', 'round': 1, 'text': 'x'}], [RUBRIC], 'reply')
    assert stage_count(calls, 'applicability') == 2


def test_cache_rubric_content_change_with_same_id_is_separate():
    score, client, calls = scorer_with_counter({})
    repaired = {**RUBRIC, 'required_behavior': RUBRIC['required_behavior'] + ' (repaired)'}
    score(VIEW, PREFIX, [RUBRIC], 'reply'); score(VIEW, PREFIX, [repaired], 'reply')
    assert repaired['rubric_id'] == RUBRIC['rubric_id'] and stage_count(calls, 'level') == 2


@pytest.mark.parametrize('change', [{'model': 'other'}, {'system': 'other prompt'}])
def test_cache_model_or_protocol_change_is_separate(change):
    from src.online_rl.tree.luna import request_key
    s1, c1, calls1 = scorer_with_counter({}); s2, c2, calls2 = scorer_with_counter(change)
    s1(VIEW, PREFIX, [RUBRIC], 'reply'); s2(VIEW, PREFIX, [RUBRIC], 'reply')
    assert request_key(calls1[0]) != request_key(calls2[0]) and c1.protocol_sha256 != c2.protocol_sha256 or change == {'model': 'other'}
    assert calls1[0] != calls2[0]


def test_cache_schema_or_max_tokens_change_is_separate():
    from src.online_rl.tree.luna import request_key
    p = {'model': 'm', 'messages': [], 'max_tokens': 10, 'response_format': {'type': 'json_schema', 'json_schema': {'schema': {'a': 1}}}}
    assert request_key(p) != request_key({**p, 'max_tokens': 11})
    assert request_key(p) != request_key({**p, 'response_format': {'type': 'json_schema', 'json_schema': {'schema': {'a': 2}}}})


def test_decision_records_mark_cache_hits():
    score, client, calls = scorer_with_counter({})
    a = score(VIEW, PREFIX, [RUBRIC], 'reply'); b = score(VIEW, PREFIX, [RUBRIC], 'reply')
    assert [r['requests'][0]['request_cache_hit'] for r in a['_decisions']] == [False, False]
    assert [r['requests'][0]['request_cache_hit'] for r in b['_decisions']] == [True, True]
    assert a['_decisions'][0]['requests'][0]['request_sha256'] == b['_decisions'][0]['requests'][0]['request_sha256']


# ---------- audit logging: each response keeps only its own decisions ----------
def test_concurrent_scoring_keeps_per_response_audit_records():
    from src.online_rl.tree.backends import two_stage_structured_scorer
    barrier = threading.Barrier(8)

    def post(url, payload, key, timeout):
        content = json.loads(payload['messages'][-1]['content']); stage = content['stage']
        if stage == 'applicability': barrier.wait(timeout=5)        # all eight responses are inside Stage 1 together
        ans = {i: ({'applicable': True, 'evidence_basis': 'e', 'missing_requirement': ''} if stage == 'applicability'
                   else {'level': '1', 'anchor_basis': content['state']['target_agent_reply']}) for i in content['items']}
        return {'choices': [{'message': {'content': json.dumps({'answers': ans})}}], 'usage': {}}
    client = LunaDecisionClient(model='m', key='k', system='s', post=post)              # no cache: every request is real
    score = two_stage_structured_scorer(client, observed_context=True)
    out = {}

    def one(i):
        prefix = PREFIX + [{'speaker': 'agent', 'stage': 'A1', 'round': 1, 'text': f'p{i}'}]     # distinct requests
        out[i] = score(VIEW, prefix, [RUBRIC], f'reply {i}')
    threads = [threading.Thread(target=one, args=(i,)) for i in range(8)]
    for t in threads: t.start()
    for t in threads: t.join()
    for i, r in out.items():
        assert len(r['_decisions']) == 2 and [d['stage'] for d in r['_decisions']] == ['applicability', 'level']
        assert r['_decisions'][1]['parsed'][RUBRIC['rubric_id']]['anchor_basis'] == f'reply {i}'
    assert len(client.log) == 16


def test_smoke_config_is_r05_with_five_cases():
    base, cfg = flat(arm('r05_o05')), flat(arm('smoke_r05'))
    diff = {k for k in set(base) | set(cfg) if base.get(k) != cfg.get(k)}
    research = {k for k in diff if k.split('.')[0] not in ('name', 'condition', 'output_dir', 'notes') and not k.startswith('experiment.')}
    assert research == {'schedule.cases'} and cfg['schedule.cases'] == 5 and cfg['output_dir'].startswith('outputs/fixed_t2_smoke/')


def test_preflight_families_match_configs():
    from scripts.online_rl.preflight_destination import protocol_family
    for name, cases, root in (('r05_o05', 120, 'outputs/fixed_t2_120/'), ('dev/lr3e5', 30, 'outputs/fixed_t2_dev/'), ('smoke_r05', 5, 'outputs/fixed_t2_smoke/')):
        cfg = arm(name); fam = protocol_family(cfg['experiment']['variant'])
        assert (fam['cases'], fam['rounds'], fam['root']) == (cases, 2, root) and cfg['output_dir'].startswith(root)
        assert fam['scorer_request_cache'] == cfg['judges']['scorer_request_cache']


@pytest.mark.parametrize('name, w', [('dis/o10_r00', (0.0, 1.0)), ('dis/o09_r01', (0.1, 0.9))])
def test_disagreement_arms_change_only_reward_order_cases_and_lr(name, w):
    base, cfg = flat(arm('r05_o05')), flat(arm(name))
    diff = {k for k in set(base) | set(cfg) if base.get(k) != cfg.get(k)}
    research = {k for k in diff if k.split('.')[0] not in ('name', 'condition', 'output_dir', 'notes') and not k.startswith('experiment.')}
    allowed = {'schedule.cases', 'optimizer.learning_rate', 'five_gpu.train_order_file', 'five_gpu.train_order_sha256',
               'agent_reward.w_rubric', 'agent_reward.w_outcome', 'rubric.enabled'}
    assert research <= allowed, research - allowed
    assert (cfg['agent_reward.w_rubric'], cfg['agent_reward.w_outcome']) == w and cfg['schedule.cases'] == 60
    assert cfg['optimizer.learning_rate'] == 3e-05 and cfg['rollout.rounds'] == 2 and cfg['rollout.branching'] == 4
    assert cfg['prompt_update.enabled'] is False and cfg['output_dir'].startswith('outputs/fixed_t2_dis/')
    assert cfg['rubric.enabled'] is (w[0] > 0)


def test_disagreement_order_keeps_dev_positions_and_all_cases():
    d = ROOT / 'data/interim/medmcqa/five_gpu_train500'
    new, old = d / 'train500_order_disagreement_first_v1.json', d / 'train500_order_seed20261008.json'
    if not new.exists() or not old.exists(): pytest.skip('disagreement-first order not built here')
    a, b = json.loads(new.read_text()), json.loads(old.read_text())
    assert sorted(a) == sorted(b) and a[400:] == b[400:]
    man = json.loads((d / 'train500_order_disagreement_first_v1.manifest.json').read_text())
    k = man['disagreement_cases']; flags = {f['case_id']: f for f in man['flags']}
    assert all(flags[c]['disagree'] for c in a[:k]) and not any(flags[c]['disagree'] for c in a[k:400])
    assert all(flags[c]['blind_option'] != flags[c]['doctor_option'] for c in a[:k])
    assert not {'gold', 'correct_option', 'outcome'} & set(man['flags'][0])          # label-free
