"""Integrated online GRPO + prompt evolution: deterministic unit tests and a fully mocked end-to-end run with resume.
No model download, no API, no GPU."""
import dataclasses, hashlib, json, re, statistics as st
from pathlib import Path
import pytest
from src.online_rl import guards
from src.online_rl.integrated import metrics as M
from src.online_rl.integrated.continuation import ContinuationCache, PolicyIdentity, continue_k, derive_seed, in_baseline
from src.online_rl.integrated.prompt_search import (PromptStore, SearchBackend, option_text_violations, parse_reflection, prompt_violations,
                                                    run_event, score_prompt, split_of)
from src.online_rl.integrated.records import GeneratorInput, RunState, StateEvidence, from_dict, read_json, sha256_text, to_dict
from src.online_rl.integrated.rubric import AGENT_SIDE, DOCTOR_SIDE, freeze_generator_input, generate_rubric, split_state
from src.online_rl.integrated.trainer import IntegratedTrainer

CASES = [{'case_id': f'syn{i}', 'question': f'Synthetic integrated test item {i}: which mechanism fits best?', 'scenario': '',
          'options': {k: f'{k.lower()}lpha option {i}' for k in 'ABCD'}, 'ground_truth': {'correct_option': 'A', 'correct_text': f'alpha option {i}'}}
         for i in range(6)]
PROMPT_TEMPLATE = ('Rubric generator for the next Agent reply. Output one JSON object with "name", "applicability_condition", '
                   '"required_behavior" and "scores" (0, 1, 2). ALIGN={align} {tag}')
P0 = PROMPT_TEMPLATE.format(align=0, tag='base')
h = lambda s: hashlib.sha256(s.encode()).hexdigest()


# ---------- mocks ----------
class Weights:
    def __init__(self): self.version = 0


class MockPolicy:
    """Deterministic in (seed, weights); reply quality q in 0..3 drives continuation success."""

    def __init__(self, w: Weights, drift=False):
        self.w, self.drift, self.calls, self.prompts, self.order = w, drift, {'generate': 0, 'sample': 0, 'continue': 0, 'natural': 0}, [], []

    def identity(self): return {'mock': True}

    def generate(self, case, seed):
        self.calls['generate'] += 1; dlg = []
        for t in range(1, 2 + seed % 3):
            dlg += [{'speaker': 'doctor', 'stage': f'D{t}', 'round': t, 'text': f'doctor {t} {seed % 13}'}, {'speaker': 'agent', 'stage': f'A{t}', 'round': t, 'text': f'agent {t}'}]
        return {'dialogue': dlg + [{'speaker': 'doctor', 'stage': 'FINAL', 'text': 'final A', 'final': True}]}

    def sample_completions(self, case, prefix, turn, seeds):
        self.calls['sample'] += 1; self.order.append('sample'); prompt = 'POLICY|' + case['case_id'] + '|' + '||'.join(d['text'] for d in prefix); self.prompts.append(prompt)
        return [{'prompt_text': prompt + (f'|drift{j}' if self.drift and j == 3 else ''), 'completion_text': f'reply q{(s + self.w.version) % 4} s{s % 997}',
                 'completion_ids': [1, 2, 3], 'old_policy_logprobs': [-0.1, -0.2, -0.3], 'seed': s} for j, s in enumerate(seeds)]

    def continue_from(self, case, prefix, turn, action, seed):
        self.calls['continue'] += 1; q = int(re.search(r'q(\d)', action).group(1))
        ok = int(h(f'{action}|{seed}'), 16) % 4 < q
        return {'final_option': 'A' if ok else 'B', 'termination_reason': 'mock', 'dialogue': [{'speaker': 'agent', 'text': action}]}

    def natural(self, case, prefix, turn, seed): self.calls['natural'] += 1; return f'reply q{seed % 4} natural'


class MockUpdater:
    def __init__(self, w: Weights, fail_update_at=None, fail_save_at=None):
        self.w, self.steps, self.updates, self.saves, self.groups = w, 0, 0, 0, []
        self.fail_update_at, self.fail_save_at = fail_update_at, fail_save_at

    def identity(self): return {'mock': True}

    def behavior_logprobs(self, traces):
        return [[-(self.w.version + 1) / 100] * len(t['completion_ids']) for t in traces]

    def update(self, records):
        self.updates += 1
        if self.updates == self.fail_update_at: raise KeyboardInterrupt('mock crash in update')
        if {r['rollout_policy_sha256'] for r in records} != {h(f'w{self.w.version}')}: raise RuntimeError('stale behavior policy')
        if any(r['agent_traces'][0]['behavior_logprobs'][0] != -(self.w.version + 1) / 100 for r in records): raise RuntimeError('stale behavior logprobs')
        self.groups.append(records); rs = [r['reward'] for r in records if r['reward'] is not None]
        if st.pvariance(rs) > 0: self.steps += 1; self.w.version += 1; return {'optimizer_step': True, 'n': len(rs)}
        return {'optimizer_step': False, 'skip_reason': 'no comparative signal'}

    def save(self, path):
        self.saves += 1
        if self.saves == self.fail_save_at: raise KeyboardInterrupt('mock crash in save')
        path.mkdir(parents=True); (path / 'state.json').write_text(json.dumps({'version': self.w.version, 'steps': self.steps})); return str(path)

    def load(self, path):
        d = json.loads((Path(path) / 'state.json').read_text()); self.w.version, self.steps = d['version'], d['steps']


def mock_state(prefix, turn, status=None):
    doctor = {k: f'{k}: {prefix[-1]["text"]}' for k in DOCTOR_SIDE}; agent = {k: f'AGENTSTATE {k} t{turn}' for k in AGENT_SIDE}
    status = status or ('elicited_initial_turn' if turn == 1 else 'elicited')
    if status in ('reconstruction_failure', 'judgment_failure'): return {'status': status, 'doctor_side_context': None, 'agent_side_state': None}
    return {'status': status, 'doctor_side_context': doctor, 'agent_side_state': agent}


class MockElicitor:
    def __init__(self, policy, fail_every=0): self.policy, self.fail_every, self.seen = policy, fail_every, []

    def __call__(self, case_view, prefix, turn):
        assert set(case_view) == {'case_id', 'question', 'options', 'scenario'}             # no gold reaches the elicitor
        self.policy.order.append('elicit'); self.seen.append((case_view['case_id'], turn))
        fail = self.fail_every and len(self.seen) % self.fail_every == 0
        return mock_state(prefix, turn, 'judgment_failure' if fail else None)


def align_of(text): return int(re.search(r'ALIGN=(-?\d)', text).group(1))


class MockBackend:
    """Generator rubric encodes ALIGN from the prompt; the scorer ranks replies by quality * ALIGN. Records every call."""

    def __init__(self, feedback_fails=0, refine_crash_at=None):
        self.gen_inputs, self.scored, self.feedback_calls, self.refine_calls = [], [], 0, 0
        self.feedback_fails, self.refine_crash_at, self.malformed_seen = feedback_fails, refine_crash_at, set()

    def generate(self, prompt_text, gin):
        self.gen_inputs.append((prompt_text, gin))
        if 'MALFORMED_ALWAYS' in prompt_text or ('MALFORMED_ONCE' in prompt_text and gin.sha256 not in self.malformed_seen):
            self.malformed_seen.add(gin.sha256); return {'raw': 'not json', 'rubric': None, 'parse_error': 'no_json_object'}
        a = align_of(prompt_text)
        return {'raw': '{}', 'parse_error': None, 'rubric': {'name': 'n', 'applicability_condition': 'always', 'required_behavior': f'behave align {a}',
                                                              'scores': {'0': 'bad', '1': 'partial', '2': 'good'}}}

    def gate(self, case, gin, cand, rid):
        if cand['rubric'] is None: return {'schema': False, 'passed': False, 'reasons': ['schema']}
        return {'schema': True, 'passed': True, 'leakage_deterministic': [], 'answer_presupposition': False, 'new_domain_premise': False,
                'observable': True, 'C': 0.8, 'reasons': []}

    def score(self, case_view, gin, rubric, reply):
        assert 'ground_truth' not in case_view
        self.scored.append((gin.sha256, reply)); a = int(rubric['required_behavior'].split()[-1]); q = int(re.search(r'q(\d)', reply).group(1))
        if a == 0: return {'probabilities': {'0': 0.2, '1': 0.6, '2': 0.2}, 'hard': 1}
        level = q / 3 if a == 1 else 1 - q / 3
        return {'probabilities': {'0': (1 - level) ** 2, '1': 2 * level * (1 - level), '2': level ** 2}, 'hard': round(2 * level)}

    def feedback(self, parent, view, n):
        self.feedback_calls += 1
        if self.feedback_calls <= self.feedback_fails: raise ValueError('malformed reflection')
        assert all('ground_truth' not in json.dumps(v) for v in view)
        return {'diagnosis': 'diag', 'directions': [f'dir{i}' for i in range(n)]}

    def refine(self, parent, diagnosis, direction, history, note):
        self.refine_calls += 1
        if self.refine_calls == self.refine_crash_at: raise KeyboardInterrupt('mock crash in prompt search')
        i = int(direction[3:]); return PROMPT_TEMPLATE.format(align=[1, -1, 0, 1][i % 4], tag=f'{direction}-{h(parent)[:6]}')

    def bundle(self): return SearchBackend(self.feedback, self.refine, self.generate, self.gate, self.score)


def make_cfg(**over):
    cfg = {'name': 't', 'output_dir': 'unused', 'condition': 'integrated_evolving', 'policy': {'base_model': 'mock-base'}, 'data': {'train': 'mock'},
           'rollout': {'group_size': 4, 'turn_sampling': 'uniform_reached', 'outcome_continuations_K': 3, 'min_usable_responses': 2},
           'optimizer': {'learning_rate': 1e-5}, 'advantage_normalization': {'mode': 'mean_center'},
           'schedule': {'seed': 0, 'stages': 1, 'steps_per_stage': 24}, 'reward': {'w_O': 0.4, 'w_A': 0.3},
           'rubric': {'enabled': True, 'prompt_id': 'P0', 'prompt_sha256': sha256_text(P0), 'max_attempts': 2},
           'metrics': {'epsilon_O': 1e-9, 'epsilon_score': 1e-6},
           'effectiveness': {'baseline_mode': 'diagnostic_subset', 'diagnostic_fraction': 0.5}, 'diagnostics': {'high_score': 0.75, 'severe_effect': 0.6667},
           'prompt_update': {'enabled': True, 'every_n_steps': 8, 'minimum_raw_states': 8, 'minimum_informative_states': 2, 'acceptance_fraction': 0.5,
                             'candidate_count': 4, 'retain_top_k': 2, 'refinement_depth': 2, 'rubric_samples_per_state': 1},
           'acceptance': {'delta_Q': 0.0, 'p_min': 0.5, 'delta_C': 0.05, 'max_validity_drop': 0.05, 'max_leakage_rise': 0.02,
                          'max_new_premise_rise': 0.02, 'max_observability_drop': 0.05, 'bootstrap_n': 200, 'bootstrap_seed': 0}}
    for k, v in over.items():
        sec, key = k.split('__'); cfg[sec] = {**cfg[sec], key: v}
    return cfg


def make_trainer(out, cfg=None, w=None, policy=None, updater=None, backend=None, prompt=P0, cases=CASES, fail_every=0):
    w = w or Weights(); cfg = cfg or make_cfg()
    if prompt != P0: cfg['rubric']['prompt_sha256'] = sha256_text(prompt)
    policy = policy or MockPolicy(w); updater = updater or MockUpdater(w); backend = backend or MockBackend()
    tr = IntegratedTrainer(cfg, out_dir=out, cases=cases, policy=policy, updater=updater, policy_sha=lambda: h(f'w{w.version}'), protocol_sha='proto',
                           outcome_fn=lambda case, opt: None if opt is None else float(opt == case['ground_truth']['correct_option']),
                           backend=backend.bundle(), initial_prompt=prompt, elicit_state=MockElicitor(policy, fail_every))
    return tr, policy, updater, backend


def evidence(out): return [from_dict(StateEvidence, read_json(p)) for p in sorted((Path(out) / 'evidence').glob('step_*.json'))]


# ---------- D, E, F, G, H, I: pure math ----------
def test_r_a_expected_and_hard_label():
    assert M.r_a({'0': 0.2, '1': 0.3, '2': 0.5}) == pytest.approx(0.65)
    assert M.r_a({'0': 2, '1': 3, '2': 5}) == pytest.approx(0.65)
    assert [M.hard_value(x) for x in (0, 1, 2)] == [0.0, 0.5, 1.0]
    with pytest.raises(ValueError): M.hard_value(3)


def test_r_o_and_r_e():
    assert M.r_o([1, 1, 0]) == pytest.approx(2 / 3)
    assert M.r_o([1, None, 0]) is None
    assert M.r_e(M.r_o([1, 1, 0]), M.r_o([1, 0, 0])) == pytest.approx(1 / 3)


def test_q_cov_cases():
    v = [2 / 3, 0, 1, 1 / 3]
    assert M.q_cov([1, 0, 1, 0.5], v, True, 1e-6) == pytest.approx(7 / 48)
    assert M.q_cov([0, 1, 0, 0.5], v, True, 1e-6) == pytest.approx(-7 / 48)
    assert M.q_cov([0.5] * 4, v, True, 1e-6) == 0.0
    assert M.q_cov([1, 0, 1, 0.5], v, False, 1e-6) == 0.0
    assert M.q_cov([1, 0, 1, 0.5], [x - 1 / 3 for x in v], True, 1e-6) == pytest.approx(7 / 48)      # V(h) shift: Cov(R_A,R_E) == Cov(R_A,R_O)
    assert M.q_cov([0.5, 0.5, 0.5, 0.50001], v, True, 1e-3) == 0.0                                 # flat under epsilon_score
    assert M.q_cov([1, None, 1, 0.5], [2 / 3, None, 1, 1 / 3], True, 1e-6) == pytest.approx(M.q_cov([1, 1, 0.5], [2 / 3, 1, 1 / 3], True, 1e-6))
    with pytest.raises(RuntimeError): M.q_cov([1, None, 1, 0.5], v, True, 1e-6)


def test_informative_depends_only_on_outcomes():
    assert not M.informative([1 / 3] * 4, 1e-9) and M.informative([0, 1 / 3, 1 / 3, 1 / 3], 1e-9)
    assert not M.informative([1, None, None, None], 1e-9)


def test_reward_and_no_r_e_double_counting():
    assert M.reward(2 / 3, 0.5, 0.4, 0.3, True) == pytest.approx(0.4 * 2 / 3 + 0.15)
    assert M.reward(2 / 3, None, 0.4, 0.3, False) == pytest.approx(0.4 * 2 / 3)
    assert M.reward(None, 0.5, 0.4, 0.3, True) is None
    R_O, R_A, V, w_E = [0, 1 / 3, 2 / 3, 1], [0.9, 0.2, 0.5, 0.4], 1 / 3, 0.25
    with_e = M.advantages([0.4 * o + 0.3 * a + w_E * (o - V) for o, a in zip(R_O, R_A)], 'mean_center')
    reweighted = M.advantages([(0.4 + w_E) * o + 0.3 * a for o, a in zip(R_O, R_A)], 'mean_center')
    assert with_e == pytest.approx(reweighted)                                                        # adding R_E only reweights R_O


def test_advantage_modes_tied_outcomes():
    rewards = [0.4 / 3 + 0.1 * a for a in (0.8, 0.6, 0.7, 0.9)]
    assert M.advantages(rewards, 'mean_center') == pytest.approx([0.005, -0.015, -0.005, 0.015])
    assert M.advantages(rewards, 'group_zscore') == pytest.approx([0.4472, -1.3416, -0.4472, 1.3416], abs=1e-4)
    assert M.advantages([0.2] * 4, 'group_zscore') == [0.0] * 4 and M.advantages([0.2] * 4, 'mean_center') == [0.0] * 4
    assert M.advantages([0.2, None, None, None], 'mean_center') == [None] * 4
    with pytest.raises(ValueError): M.advantages([0.1, 0.2], 'other')


def test_diagnostics_are_logged_only():
    d = M.state_diagnostics([1, 0, 1, 0.5], [1, 0, 1, 0.5], [2 / 3, 0, 1, 1 / 3], 1e-6)
    assert d['Q_conc'] == pytest.approx(0.95) and d['corr'] > 0 and d['worst_endorsed'] is False and d['hard_histogram'] == {0: 1, 1: 1, 2: 2}
    e = M.effect_diagnostics([0.9, 0.1], [-1.0, 0.0], 0.75, 0.6667)
    assert e['high_score_negative_effect_rate'] == 1.0 and e['severe_harm_endorsement_rate'] == 1.0


def test_acceptance_rule():
    cfg = {'minimum_informative_states': 3, 'delta_Q': 0.01, 'p_min': 0.9, 'bootstrap_n': 500, 'bootstrap_seed': 0, 'delta_C': 0.02,
           'max_validity_drop': 0.05, 'max_leakage_rise': 0.02, 'max_new_premise_rise': 0.02, 'max_observability_drop': 0.05}
    g = {'validity_rate': 1.0, 'leakage_rate': 0.0, 'new_premise_rate': 0.0, 'observable_rate': 1.0, 'C_mean': 0.8}
    parent, better = {f's{i}': 0.0 for i in range(4)}, {f's{i}': 0.05 for i in range(4)}
    assert M.accept(parent, better, g, g, cfg)['accepted']
    assert not M.accept(parent, better, g, {**g, 'C_mean': 0.7}, cfg)['checks']['C']
    assert not M.accept(parent, better, g, {**g, 'leakage_rate': 0.1}, cfg)['accepted']
    lucky = {'s0': 0.2, 's1': -0.02, 's2': -0.02, 's3': -0.02}                                         # one lucky state, mean worse elsewhere
    assert not M.accept(parent, lucky, g, g, cfg)['accepted']
    assert not M.accept({'s0': 0.0, 's1': 0.0}, {'s0': 0.1, 's1': 0.1}, g, g, cfg)['checks']['min_informative']
    with pytest.raises(RuntimeError): M.accept(parent, {'s0': 0.1}, g, g, cfg)


# ---------- B, C, P: generator boundary, ephemeral rubric, malformed policy ----------
def _prefix(turn):
    out = []
    for t in range(1, turn + 1):
        out.append({'speaker': 'doctor', 'stage': f'D{t}', 'round': t, 'text': f'doc {t}'})
        if t < turn: out.append({'speaker': 'agent', 'stage': f'A{t}', 'round': t, 'text': f'agent {t}'})
    return out


def test_generator_input_boundary():
    gin = freeze_generator_input(CASES[0], _prefix(2), 2, mock_state(_prefix(2), 2)); gin.verify()
    blob = json.dumps(to_dict(gin))
    assert 'ground_truth' not in blob and 'correct' not in blob and 'FINAL' not in blob
    assert [f.name for f in dataclasses.fields(GeneratorInput)] == ['case_id', 'scenario', 'question', 'options', 'history', 'turn', 'state_status',
                                                                     'doctor_side_context', 'agent_side_state', 'sha256']
    st2 = mock_state(_prefix(2), 2)
    with pytest.raises(ValueError): freeze_generator_input(CASES[0], _prefix(2) + [{'speaker': 'doctor', 'stage': 'FINAL', 'text': 'A', 'round': 3}], 2, st2)
    with pytest.raises(ValueError): freeze_generator_input(CASES[0], [{**_prefix(1)[0], 'correct': 1}], 1, mock_state(_prefix(1), 1))
    with pytest.raises(ValueError): freeze_generator_input(CASES[0], _prefix(2)[:-1], 2, st2)            # ends with an Agent reply
    with pytest.raises(ValueError): freeze_generator_input(CASES[0], _prefix(3), 2, st2)                 # future turn
    with pytest.raises(ValueError): freeze_generator_input(CASES[0], _prefix(2), 2, {**st2, 'candidate_replies': ['x']})
    with pytest.raises(ValueError): freeze_generator_input(CASES[0], _prefix(2), 2, {**st2, 'agent_side_state': {**st2['agent_side_state'], 'R_O': 1}})
    with pytest.raises(TypeError): GeneratorInput(**{**to_dict(gin), 'responses': ['x']})
    with pytest.raises(RuntimeError): dataclasses.replace(gin, agent_side_state={**gin.agent_side_state, 'focal_claim': 'x'}).verify()
    with pytest.raises(RuntimeError): dataclasses.replace(gin, question='changed').verify()


def test_generator_sees_only_frozen_input_and_rubric_never_reaches_policy(tmp_path):
    tr, pol, _, be = make_trainer(tmp_path); tr.run(max_steps=3)
    ev = evidence(tmp_path)
    assert len(be.gen_inputs) == 3 and all(isinstance(g, GeneratorInput) for _, g in be.gen_inputs)
    for e, (_, g) in zip(ev, be.gen_inputs):
        assert g == e.generator_input and e.rubric.generator_input_sha256 == g.sha256 and e.rubric.state_key == e.state_key
        assert not any(b.response in json.dumps(to_dict(g)) for b in e.branches)
    assert len({e.rubric.rubric_id for e in ev}) == 3                                                    # a new rubric per state
    assert not any('behave' in p for p in pol.prompts)                                                   # rubric text never in a policy prompt
    assert not hasattr(tr, 'active_rubrics') and not list(tmp_path.glob('**/*active*'))


def test_malformed_rubric_retry_then_mask(tmp_path):
    be = MockBackend(); gin = freeze_generator_input(CASES[0], _prefix(1), 1, mock_state(_prefix(1), 1))
    once = PROMPT_TEMPLATE.format(align=1, tag='MALFORMED_ONCE')
    r = generate_rubric(once, 'p', sha256_text(once), gin, 'sk', CASES[0], be.generate, be.gate, 2, 0)
    assert r.valid and len(r.attempts) == 2 and r.failure is None
    bad = PROMPT_TEMPLATE.format(align=1, tag='MALFORMED_ALWAYS')
    r = generate_rubric(bad, 'p', sha256_text(bad), gin, 'sk', CASES[0], be.generate, be.gate, 2, 0)
    assert not r.valid and len(r.attempts) == 2 and r.failure == 'malformed_after_max_attempts' and r.rubric is None
    tr, *_ = make_trainer(tmp_path, prompt=bad); tr.run(max_steps=4)
    ev = evidence(tmp_path); assert len(ev) == 4
    for e in ev:
        assert e.rubric.failure == 'malformed_after_max_attempts' and e.diagnostics['rubric_masked']
        assert all(b.R_A is None and (b.reward is None or b.reward == pytest.approx(0.4 * b.R_O)) for b in e.branches)
        assert e.Q_cov in (None, 0.0)
    log = [json.loads(l) for l in (tmp_path / 'train_log.jsonl').read_text().splitlines()]
    assert all(r['rubric_failure'] == 'malformed_after_max_attempts' for r in log)


# ---------- A: same-state group ----------
def test_same_state_group_and_mixed_prompt_refusal(tmp_path):
    tr, _, up, _ = make_trainer(tmp_path / 'ok'); tr.run(max_steps=2)
    for g in up.groups:
        assert len(g) == 4 and len({r['case_id'] for r in g}) == 1
        assert len({json.dumps(r['agent_traces'][0]['prompt_text']) for r in g}) == 1
    w = Weights(); tr, *_ = make_trainer(tmp_path / 'drift', w=w, policy=MockPolicy(w, drift=True))
    with pytest.raises(RuntimeError, match='mixes'): tr.step()
    assert tr.state.step == 0 and not list((tmp_path / 'drift' / 'evidence').glob('*'))


def test_group_below_min_usable_skips_update(tmp_path):
    class NoAnswer(MockPolicy):
        def continue_from(self, *a): return {**super().continue_from(*a), 'final_option': None}
    w = Weights(); tr, _, up, _ = make_trainer(tmp_path, w=w, policy=NoAnswer(w)); tr.run(max_steps=1)
    e = evidence(tmp_path)[0]
    assert up.updates == 0 and 'min_usable_responses' in e.update['skip_reason'] and all(b.R_O is None for b in e.branches)


# ---------- J: continuation cache ----------
def test_continuation_cache_policy_keyed(tmp_path):
    w = Weights(); pol = MockPolicy(w); cache = ContinuationCache(tmp_path / 'c.jsonl'); seeds = [1, 2, 3]
    ident = PolicyIdentity('base', 'sha-A', 'proto'); args = (CASES[0], 'sk', _prefix(1), 1, 'reply q2 s1', seeds, None, lambda c, o: float(o == 'A'))
    first = continue_k(pol, cache, ident, *args); n = pol.calls['continue']
    again = continue_k(pol, ContinuationCache(tmp_path / 'c.jsonl'), ident, *args)
    assert [r.key for r in again] == [r.key for r in first] and pol.calls['continue'] == n                # hit (also after reload)
    continue_k(pol, cache, PolicyIdentity('base', 'sha-B', 'proto'), *args); assert pol.calls['continue'] == n + 3   # adapter change -> miss
    with pytest.raises(RuntimeError): cache.get(first[0].key, 'sha-B')                                  # never served to another policy
    rows = (tmp_path / 'c.jsonl').read_text().splitlines(); bad = json.loads(rows[0]); bad['policy_sha256'] = 'sha-X'
    (tmp_path / 'bad.jsonl').write_text(json.dumps(bad) + '\n')
    with pytest.raises(RuntimeError): ContinuationCache(tmp_path / 'bad.jsonl')
    nat = continue_k(pol, cache, ident, CASES[0], 'sk', _prefix(1), 1, None, seeds, [7, 8, 9], lambda c, o: float(o == 'A'))
    assert [r.seed for r in nat] == seeds and all(r.response_sha256 == 'NATURAL' and r.natural_response for r in nat)


def test_seeds_and_baseline_subset_are_deterministic():
    assert derive_seed(0, 'k', 'continuation', 1) == derive_seed(0, 'k', 'continuation', 1) != derive_seed(0, 'k', 'natural', 1)
    keys = [f'state{i}' for i in range(400)]
    sel = [in_baseline('diagnostic_subset', 0.25, 0, k) for k in keys]
    assert sel == [in_baseline('diagnostic_subset', 0.25, 0, k) for k in keys] and 0.15 < sum(sel) / 400 < 0.35
    assert not any(in_baseline('off', 1.0, 0, k) for k in keys) and all(in_baseline('every_step', 0.0, 0, k) for k in keys)


def test_r_e_only_on_selected_states_and_paired_with_r_o(tmp_path):
    tr, *_ = make_trainer(tmp_path, cfg=make_cfg(effectiveness__baseline_mode='every_step')); tr.run(max_steps=3)
    for e in evidence(tmp_path):
        assert e.V_h is not None and len(e.baseline_keys) == 3
        assert all(b.R_E == pytest.approx(b.R_O - e.V_h) for b in e.branches if b.R_O is not None)
        cache = ContinuationCache(tmp_path / 'continuations.jsonl')
        assert [cache.records[k].seed for k in e.baseline_keys] == [cache.records[k].seed for k in e.branches[0].continuation_keys]
    tr, *_ = make_trainer(tmp_path / 'off', cfg=make_cfg(effectiveness__baseline_mode='off')); tr.run(max_steps=3)
    assert all(e.V_h is None and not e.baseline_keys for e in evidence(tmp_path / 'off'))


# ---------- K, L, M: paired prompt search and lineage ----------
def _window(tmp_path, steps=16):
    tr, pol, _, be = make_trainer(tmp_path, cfg=make_cfg(prompt_update__enabled=False)); tr.run(max_steps=steps)
    return tr, pol, be, evidence(tmp_path)


def test_paired_scoring_uses_stored_replies_only(tmp_path):
    tr, pol, be, ev = _window(tmp_path); calls = dict(pol.calls)
    states = [e for e in ev if e.informative]; be.scored.clear()
    cand = PROMPT_TEMPLATE.format(align=1, tag='cand')
    res = score_prompt(cand, 'C', sha256_text(cand), states, {c['case_id']: c for c in CASES}, be.bundle(), tr.cfg, 99, stored=False)
    stored = {(e.generator_input.sha256, b.response) for e in states for b in e.branches if b.R_O is not None}
    assert pol.calls == calls and set(be.scored) <= stored and len(res) == len(states)
    bad = PROMPT_TEMPLATE.format(align=1, tag='MALFORMED_ALWAYS')
    res_bad = score_prompt(bad, 'B', sha256_text(bad), states, {c['case_id']: c for c in CASES}, be.bundle(), tr.cfg, 99, stored=False)
    assert set(res_bad) == set(res) and all(r['Q'] == 0.0 for r in res_bad.values())                    # invalid rubric cannot drop a state


def test_prompt_search_bounded_paired_and_lineage(tmp_path):
    tr, pol, be, ev = _window(tmp_path); calls = dict(pol.calls)
    store = PromptStore(tmp_path / 'prompts'); sha = store.put(P0)
    pu = tr.cfg['prompt_update']
    acc_gins = {e.generator_input.sha256 for e in ev if e.informative and split_of(0, e.state_key, pu['acceptance_fraction']) == 'acceptance'}
    be.gen_inputs.clear()
    res = run_event(1, 'P0', sha, ev, {c['case_id']: c for c in CASES}, store, be.bundle(), tr.cfg, 0, 16, [])
    assert pol.calls == calls                                                                           # no Agent/Doctor call
    assert be.feedback_calls == 1 + 2 and be.refine_calls == 4 + 4                                     # depth 1: 4 from P0; depth 2: 2 x 2 from top-2
    vs = res['versions']; assert len(vs) == 8 and {v.depth for v in vs} == {1, 2}
    assert all(v.parent_prompt_id == 'P0' for v in vs if v.depth == 1)
    assert {v.parent_prompt_id for v in vs if v.depth == 2} <= {v.prompt_id for v in vs if v.status in ('retained', 'accepted', 'rejected') and v.depth == 1}
    acc_texts = {p for p, g in be.gen_inputs if g.sha256 in acc_gins}
    best = res['decision']['candidate_prompt_id']; best_v = next(v for v in vs if v.prompt_id == best)
    assert acc_texts == {store.get(best_v.sha256)}                                                      # only the best candidate touches acceptance
    assert res['decision']['accepted'] and align_of(store.get(best_v.sha256)) == 1 and best_v.status == 'accepted'
    assert set(res['search_evidence_ids']).isdisjoint(res['acceptance_evidence_ids'])
    assert all(store.get(v.sha256) for v in vs if v.sha256)
    again = run_event(1, 'P0', sha, ev, {c['case_id']: c for c in CASES}, store, MockBackend().bundle(), tr.cfg, 0, 16, [])
    assert [to_dict(v) for v in again['versions']] == [to_dict(v) for v in vs]                          # deterministic tie-breaks
    deep = run_event(1, 'P0', sha, ev, {c['case_id']: c for c in CASES}, store, MockBackend().bundle(), {**tr.cfg, 'prompt_update': {**pu, 'refinement_depth': 3}}, 0, 16, [])
    assert len(deep['versions']) == 12                                                                  # linear in depth, not 4^depth


def test_prompt_search_without_valid_reflection_keeps_parent(tmp_path):
    tr, _, _, ev = _window(tmp_path)
    store = PromptStore(tmp_path / 'prompts'); sha = store.put(P0)
    res = run_event(1, 'P0', sha, ev, {c['case_id']: c for c in CASES}, store, MockBackend(feedback_fails=2).bundle(), tr.cfg, 0, 16, [])
    assert not res['decision']['accepted'] and res['versions'] == [] and len(res['reflections'][0]['errors']) == 2


def test_prompt_store_and_guards(tmp_path):
    store = PromptStore(tmp_path)
    a = store.put(P0); assert store.put(P0) == a and store.get(a) == P0
    (tmp_path / f'{a}.txt').write_text('tampered')
    with pytest.raises(RuntimeError): store.get(a)
    with pytest.raises(RuntimeError): store.put(P0)
    assert option_text_violations('prefer the alpha option 3 here', [P0], ['alpha option 3', 'B']) == ['alpha option 3']
    assert option_text_violations('prefer the alpha option 3 here', ['alpha option 3 seen'], ['alpha option 3']) == []
    v = prompt_violations(P0 + ' Synthetic integrated test item 3 mechanism', [P0], CASES[3:4])
    assert any('copied 4-grams' in x for x in v)
    assert prompt_violations(P0.replace('"scores"', 'x'), [P0], CASES) == ['missing output key "scores"']
    assert parse_reflection('x {"diagnosis": "d", "directions": ["a", "b"]} y', 2)['directions'] == ['a', 'b']
    for bad in ('no json', '{"diagnosis": "d", "directions": ["a"]}', '{"directions": ["a", "b"]}'):
        with pytest.raises(ValueError): parse_reflection(bad, 2)


# ---------- O: protected data ----------
def test_protected_ids_and_question_hashes(tmp_path, monkeypatch):
    shipped = json.loads(guards.MANIFEST.read_text())
    assert {'val50', 'test100', 'prompt_test', 'rq3_probe_pool', 'casimedicos_test'} <= set(shipped['splits'])
    assert all(set(v) == {'ids', 'question_sha256'} for v in shipped['splits'].values())                    # ids and hashes only
    def write(name, rows):
        (tmp_path / name).write_text(''.join(json.dumps(r) + '\n' for r in rows)); return tmp_path / name
    test = write('test.jsonl', [{'case_id': 'held-1', 'source_question_id': 'q-held', 'question': 'Held-out item about a rare finding?'}])
    other = write('other.jsonl', [{'case_id': 'held-2', 'question': 'Another held-out item?'}])
    (tmp_path / 'casi').mkdir(); write('casi/en_test_casimedicos.jsonl', [{'id': 1, 'full_question': 'An external exam vignette asks what?'}])
    (tmp_path / 'pool.json').write_text(json.dumps({'case_ids': ['held-2'], 'source_question_ids': []}))
    (tmp_path / 'm.json').write_text(json.dumps(guards.build_manifest({'test100': test, 'val50': other, 'prompt_test': other}, tmp_path / 'casi', tmp_path / 'pool.json')))
    monkeypatch.setattr(guards, 'MANIFEST', tmp_path / 'm.json')
    with pytest.raises(RuntimeError, match='test100'): guards.assert_not_protected(['q-held'], 't')
    with pytest.raises(RuntimeError, match='casimedicos'): guards.assert_not_protected([{'case_id': 'new', 'question': '  AN EXTERNAL exam vignette  asks what? '}], 't')
    leaked = {**CASES[0], 'case_id': 'fresh', 'question': 'Held-out item about a rare finding?'}
    with pytest.raises(RuntimeError): make_trainer(tmp_path / 'a', cases=CASES + [leaked])
    tr, _, be, ev = _window(tmp_path / 'b', steps=8)
    cases = {c['case_id']: c for c in CASES}; cases[ev[0].case_id] = {**cases[ev[0].case_id], 'question': 'Another held-out item?'}
    store = PromptStore(tmp_path / 'p'); sha = store.put(P0)
    with pytest.raises(RuntimeError): run_event(1, 'P0', sha, ev, cases, store, be.bundle(), tr.cfg, 0, 8, [])
    guards.assert_not_protected(CASES, 'synthetic')


# ---------- N, Q: mocked end-to-end with prompt events, interruption and resume ----------
def _snapshot(out):
    out = Path(out)
    ev = {p.name: p.read_text() for p in sorted((out / 'evidence').glob('step_*.json'))}
    events = {p.name: p.read_text() for p in sorted((out / 'prompt_events').glob('event_*.json'))}
    st_ = read_json(out / 'run_state.json'); st_['checkpoint'] = Path(st_['checkpoint']).name
    return ev, events, st_


def test_mocked_end_to_end(tmp_path):
    tr, _, up, _ = make_trainer(tmp_path); tr.run()
    s = tr.state; ev = evidence(tmp_path)
    assert s.finished and s.step == s.evidence_cursor == 24 == len(ev) and s.optimizer_steps == up.steps
    events = [read_json(p) for p in sorted((tmp_path / 'prompt_events').glob('event_*.json'))]
    assert s.lineage_cursor == len(events) >= 1 and any(e['decision']['accepted'] for e in events)
    first_acc = next(e for e in events if e['decision']['accepted'])
    assert s.history[1]['prompt_id'] == first_acc['decision']['candidate_prompt_id'] and s.history[0]['prompt_id'] == 'P0'
    switch = first_acc['step']
    assert {e.prompt_sha256 for e in ev if e.step < switch} == {sha256_text(P0)}
    assert {e.prompt_sha256 for e in ev if e.step >= switch} != {sha256_text(P0)}                       # later states use the updated prompt
    for e in events:
        assert set(e['search_evidence_ids']).isdisjoint(e['acceptance_evidence_ids'])
        for v in e['versions']: assert v['status'] in ('accepted', 'rejected', 'retained', 'pruned')
    used = [i for e in events for i in e['search_evidence_ids'] + e['acceptance_evidence_ids']]
    assert len(used) == len(set(used))                                                                  # each state feeds at most one event
    ckpt = Path(s.checkpoint); assert ckpt.exists() and json.loads((ckpt / 'state.json').read_text())['steps'] == s.optimizer_steps
    assert any(e.informative for e in ev) and all((e.Q_cov is None) != e.informative for e in ev)
    assert any(e.V_h is not None for e in ev) and any(e.V_h is None for e in ev)


@pytest.mark.parametrize('crash', [{'fail_update_at': 6}, {'fail_save_at': 4}, {'refine_crash_at': 3}])
def test_interrupt_and_resume_is_identical(tmp_path, crash):
    ref, *_ = make_trainer(tmp_path / 'ref'); ref.run()
    out = tmp_path / 'run'; w = Weights()
    upd = MockUpdater(w, crash.get('fail_update_at'), crash.get('fail_save_at')); be = MockBackend(refine_crash_at=crash.get('refine_crash_at'))
    tr, *_ = make_trainer(out, w=w, updater=upd, backend=be)
    with pytest.raises(KeyboardInterrupt): tr.run()
    w2 = Weights(); tr2, *_ = make_trainer(out, w=w2)
    ck = tr2.state.checkpoint                                                                            # weights restored to the commit point
    assert tr2.state.step == tr.state.step and w2.version == json.loads((Path(ck) / 'state.json').read_text())['version']
    tr2.run()
    assert _snapshot(out) == _snapshot(tmp_path / 'ref')
    if 'fail_save_at' in crash:
        moved = list((out / 'recovered').glob('resume_*/evidence__step_*.json')); assert len(moved) == 1


def test_resume_fails_fast_on_inconsistency(tmp_path):
    tr, *_ = make_trainer(tmp_path); tr.run(max_steps=5)
    st_path = tmp_path / 'run_state.json'; good = st_path.read_text()
    bad = json.loads(good); bad['adapter_sha256'] = 'f' * 64; st_path.write_text(json.dumps(bad))
    with pytest.raises(RuntimeError, match='adapter hash'): make_trainer(tmp_path)
    bad = json.loads(good); bad['optimizer_steps'] += 1; st_path.write_text(json.dumps(bad))
    with pytest.raises(RuntimeError, match='optimizer steps'): make_trainer(tmp_path)
    st_path.write_text(good)
    with pytest.raises(RuntimeError, match='config changed'): make_trainer(tmp_path, cfg=make_cfg(reward__w_A=0.5))
    (tmp_path / 'evidence' / 'step_000002.json').rename(tmp_path / 'gone.json')
    with pytest.raises(FileNotFoundError): make_trainer(tmp_path)
    (tmp_path / 'gone.json').rename(tmp_path / 'evidence' / 'step_000002.json')
    st_path.unlink()
    with pytest.raises(RuntimeError, match='no run_state'): make_trainer(tmp_path)


def test_trainer_refuses_unresolved_config(tmp_path):
    with pytest.raises(ValueError, match='unresolved'): make_trainer(tmp_path, cfg=make_cfg(reward__w_A='CALIBRATION_REQUIRED'))


def test_launcher_blocks_shipped_config():
    import scripts.online_rl.run_integrated as L
    cfg = json.loads((Path(L.ROOT) / L.DEFAULT_CONFIG).read_text()); b = L.blockers(cfg)
    for k in ('rollout.outcome_continuations_K', 'reward.w_A', 'metrics.epsilon_O', 'acceptance.delta_Q', 'prompt_update.minimum_informative_states'):
        assert any(x.startswith(k) for x in b)
    assert cfg['advantage_normalization']['mode'] == 'mean_center' and cfg['effectiveness']['baseline_mode'] == 'diagnostic_subset'
    assert cfg['rubric']['enabled'] and cfg['condition'] == 'integrated_evolving'


def test_run_state_round_trip():
    s = RunState('r', 'c', 0, 3, 1, 'ck', 'cks', 'a', 'P', 'p', 3, 0, 0, False, ({'prompt_id': 'P', 'sha256': 'p', 'event': None, 'step': 0},))
    assert from_dict(RunState, json.loads(json.dumps(to_dict(s)))) == s


# ---------- I: integrated GRPO loss: pi_old / pi_theta / pi_ref (tiny CPU LoRA model) ----------
def _tiny_updater(lr=1e-2, mode='mean_center'):
    torch = pytest.importorskip('torch'); tf = pytest.importorskip('transformers'); peft = pytest.importorskip('peft')
    from types import SimpleNamespace
    from src.online_rl.integrated.grpo import make_updater
    torch.manual_seed(0)
    base = tf.LlamaForCausalLM(tf.LlamaConfig(vocab_size=40, hidden_size=16, intermediate_size=32, num_hidden_layers=1, num_attention_heads=2,
                                              num_key_value_heads=2, max_position_embeddings=64))
    model = peft.get_peft_model(base, peft.LoraConfig(r=2, lora_alpha=4, target_modules=['q_proj', 'v_proj'], lora_dropout=0.0))
    cfg = {'optimizer': {'learning_rate': lr, 'beta': 0.04, 'clip_epsilon': 0.2, 'grad_clip': 1.0, 'logprob_token_chunk_size': 16,
                         'pre_update_ratio_tolerance': 1e-3}, 'advantage_normalization': {'mode': mode}}
    return torch, model, make_updater(SimpleNamespace(torch=torch, model=model), cfg)


def _group(up, rewards, offset=0):
    from src.online_rl.integrated.continuation import adapter_sha256
    traces = [{'prompt_input_ids': [1, 2, 3, 4], 'completion_ids': [5 + j + offset, 9, 11 + j]} for j in range(len(rewards))]
    for t, lp in zip(traces, up.behavior_logprobs(traces)): t['behavior_logprobs'] = lp
    sha = adapter_sha256(up.policy.model)
    return [{'trajectory_id': f't{j}', 'case_id': 's', 'reward': r, 'rollout_policy_sha256': sha, 'agent_traces': [t]} for j, (r, t) in enumerate(zip(rewards, traces))]


def test_two_consecutive_updates_use_the_behavior_policy():
    from src.online_rl.integrated.continuation import adapter_sha256
    from src.online_rl.integrated.grpo import reference_logprobs
    _, model, up = _tiny_updater()
    sha0 = adapter_sha256(model); g1 = _group(up, [0.3, 0.1, 0.3, 0.5]); u1 = up.update(g1); sha1 = adapter_sha256(model)
    assert u1['optimizer_step'] and u1['ratio']['max_abs_deviation'] < 1e-6 and sha1 != sha0
    g2 = _group(up, [0.5, 0.1, 0.2, 0.4], offset=3)
    ref_gap = max(max(abs(a - b) for a, b in zip(r['agent_traces'][0]['behavior_logprobs'],
                  reference_logprobs(model, r['agent_traces'][0]['prompt_input_ids'], r['agent_traces'][0]['completion_ids']).tolist())) for r in g2)
    assert ref_gap > 1e-3                                                                              # pi_old (policy_1) != frozen base
    u2 = up.update(g2); sha2 = adapter_sha256(model)
    assert u2['optimizer_step'] and u2['ratio']['max_abs_deviation'] < 1e-6 and u2['kl_loss'] > 0 and sha2 != sha1 and up.steps == 2
    assert u2['ratio']['max_abs_logprob_gap_to_reference'] > 1e-3                                       # a ratio against pi_ref would refuse this group


def test_loss_matches_explicit_formula():
    torch, model, up = _tiny_updater(mode='group_zscore')
    with torch.no_grad():
        for n, p in model.named_parameters():
            if 'lora_B' in n: p.normal_(0, 0.5)                                                         # pi_theta != pi_ref, pi_old == pi_theta
    from src.online_rl.integrated.grpo import grpo_loss, reference_logprobs, sequence_logprobs
    from src.online_rl.integrated.metrics import advantages
    g = _group(up, [0.3, 0.1, 0.3, 0.5]); adv = advantages([r['reward'] for r in g], 'group_zscore'); beta, eps = 0.04, 0.2
    ours = grpo_loss(model, g, adv, beta, eps, 16)
    pol = kl = 0.0
    with torch.no_grad():
        for r, a in zip(g, adv):
            t = r['agent_traces'][0]; cur = sequence_logprobs(model, t['prompt_input_ids'], t['completion_ids'], grad=False)
            ratio = torch.exp(cur - torch.tensor(t['behavior_logprobs'])); lr = reference_logprobs(model, t['prompt_input_ids'], t['completion_ids']) - cur
            pol += float(-torch.minimum(ratio * a, torch.clamp(ratio, 1 - eps, 1 + eps) * a).mean()) / len(g)
            kl += float((torch.exp(lr) - lr - 1).mean()) / len(g)
    assert ours['policy_loss'] == pytest.approx(pol, abs=1e-6) and ours['kl_loss'] == pytest.approx(kl, rel=1e-5) and kl > 0
    assert ours['ratio']['max_abs_deviation'] < 1e-6 and any(p.grad is not None for p in model.parameters())


def test_stale_behavior_policy_is_rejected():
    from src.online_rl.integrated.continuation import adapter_sha256
    _, model, up = _tiny_updater()
    old = _group(up, [0.3, 0.1, 0.3, 0.5]); up.update(_group(up, [0.3, 0.1, 0.3, 0.5]))
    with pytest.raises(RuntimeError, match='stale behavior policy'): up.update(old)
    for r in old: r['rollout_policy_sha256'] = adapter_sha256(model)                                   # right hash, stale logprobs
    with pytest.raises(RuntimeError, match='ratio check failed'): up.update(old)
    assert up.steps == 1


def test_zero_advantage_group_does_not_update():
    from src.online_rl.integrated.continuation import adapter_sha256
    _, model, up = _tiny_updater()
    sha = adapter_sha256(model); out = up.update(_group(up, [0.4, 0.4, 0.4, 0.4]))
    assert out == {'optimizer_step': False, 'skip_reason': 'zero_advantage_group'} and adapter_sha256(model) == sha and up.steps == 0
    with pytest.raises(ValueError): _tiny_updater(mode='zscore')


def test_resumed_group_uses_resumed_adapter_as_behavior_policy(tmp_path):
    tr, *_ = make_trainer(tmp_path); tr.run(max_steps=3)
    w = Weights(); tr2, _, up2, _ = make_trainer(tmp_path, w=w)
    resumed = w.version; tr2.run(max_steps=1)
    e = evidence(tmp_path)[3]
    assert resumed >= 1 and e.policy_sha256 == h(f'w{resumed}') and set(e.branches[0].behavior_logprobs) == {-(resumed + 1) / 100}
    assert up2.groups and all(r['rollout_policy_sha256'] == h(f'w{resumed}') for r in up2.groups[0])


# ---------- Doctor-side context / Agent-side state ----------
def test_split_state_reuses_existing_decomposition():
    from src.belief import parse_local_judgment
    from src.collaboration.generator import RECONSTRUCTION_FIELD_ORDER, parse_reconstruction
    raw = {k: ([f'{k} item'] if k not in ('focal_claim', 'clinician_current_position', 'agent_current_position') else f'{k} text')
           for k in RECONSTRUCTION_FIELD_ORDER}
    rec = parse_reconstruction(json.dumps(raw)); judgment = parse_local_judgment('{"q1":{"answer":"A1","reason":"r1"},"q2":{"answer":"Y","reason":"r2"}}')
    out = split_state(rec, judgment, _prefix(2), 2)
    assert out['status'] == 'elicited' and set(out['doctor_side_context']) == set(DOCTOR_SIDE) and set(out['agent_side_state']) == set(AGENT_SIDE)
    assert out['doctor_side_context']['latest_doctor_message'] == 'doc 2' and out['doctor_side_context']['clinician_evidence_so_far'] == ['clinician_evidence_so_far item']
    assert out['agent_side_state']['better_supported_position'] == 'clinician' and out['agent_side_state']['evidence_sufficient'] is True
    assert set(RECONSTRUCTION_FIELD_ORDER) <= set(DOCTOR_SIDE) | set(AGENT_SIDE)                       # every reconstruction field lands once
    a1 = split_state(rec, None, _prefix(1), 1)
    assert a1['status'] == 'elicited_initial_turn' and a1['agent_side_state']['better_supported_position'] is None
    assert split_state({'parse_status': 'invalid'}, None, _prefix(2), 2)['status'] == 'reconstruction_failure'
    assert split_state(rec, {'parse_status': 'invalid'}, _prefix(2), 2)['status'] == 'judgment_failure'
    freeze_generator_input(CASES[0], _prefix(2), 2, out).verify()


def test_state_elicited_before_replies_and_never_in_policy_prompt(tmp_path):
    tr, pol, _, be = make_trainer(tmp_path); tr.run(max_steps=4)
    assert pol.order == ['elicit', 'sample'] * 4                                                        # elicitation precedes the G samples
    assert not any('AGENTSTATE' in p or 'latest_doctor_message' in p for p in pol.prompts)            # never in the policy prompt
    for e, (_, gin) in zip(evidence(tmp_path), be.gen_inputs):
        assert gin.state_status in ('elicited', 'elicited_initial_turn') and gin.agent_side_state == e.generator_input.agent_side_state
        assert 'AGENTSTATE' in json.dumps(gin.agent_side_state) and gin.doctor_side_context['latest_doctor_message'].endswith(gin.history[-1][3])


def test_real_generator_message_carries_state_and_failed_state_masks_rubric(tmp_path):
    from src.online_rl.integrated.rubric import generator_message
    gin = freeze_generator_input(CASES[0], _prefix(2), 2, mock_state(_prefix(2), 2))
    msg = generator_message(gin)
    assert 'AGENTSTATE' in msg and 'doctor_side_context' in msg and 'ground_truth' not in msg
    failed = freeze_generator_input(CASES[0], _prefix(2), 2, mock_state(_prefix(2), 2, 'judgment_failure'))
    assert 'Structured pre-action state' not in generator_message(failed)
    tr, *_ , be = make_trainer(tmp_path, fail_every=2); tr.run(max_steps=4)
    ev = evidence(tmp_path); failed_ev = [e for e in ev if e.generator_input.state_status == 'judgment_failure']
    assert len(failed_ev) == 2 and len(be.gen_inputs) == 2                                              # no generator call without a state
    for e in failed_ev:
        assert e.rubric.failure == 'state_elicitation_failed' and all(b.R_A is None for b in e.branches) and e.Q_cov in (None, 0.0)


def test_update_groups_one_step_over_sibling_groups():
    from src.online_rl.integrated.continuation import adapter_sha256
    _, model, up = _tiny_updater()
    sha = adapter_sha256(model)
    g1 = _group(up, [0.3, 0.1, 0.3, 0.5]); g2 = [{**r, 'case_id': 's2'} for r in _group(up, [1.0, 0.0, 0.5, 0.5], offset=7)]
    g3 = [{**r, 'case_id': 's3'} for r in _group(up, [0.4, 0.4, 0.4, 0.4], offset=2)]                       # zero advantage
    out = up.update_groups([g1, g2, g3])
    assert out['optimizer_step'] and out['groups_used'] == 2 and out['groups_skipped'] == 1 and out['used_agent_completions'] == 8
    assert out['ratio']['max_abs_deviation'] < 1e-6 and up.steps == 1 and adapter_sha256(model) != sha
    sha = adapter_sha256(model)
    assert up.update_groups([[{**r, 'rollout_policy_sha256': sha} for r in g3]]) == {'optimizer_step': False, 'skip_reason': 'zero_advantage_group'}
