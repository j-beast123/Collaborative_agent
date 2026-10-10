"""Pilot E1 / E2 switches: native thinking for the Agent and the decision-only credit rule. Both are off by default, and the frozen
loss, reward and experiment fingerprints are unchanged without them."""
from types import SimpleNamespace
import pytest
from src.online_rl import agent_protocol as AP
from src.online_rl.runtime import SharedPolicy
from src.online_rl.tree import experiment as X
from src.online_rl.tree.records import AgentNode, Leaf, Tree
from src.online_rl.tree.rollout import with_private_outcome
from src.online_rl.tree.team import node_credit
from tests.test_agent_precommit import joint_trace, tiny

T2 = 'configs/experiments/fixed_t2'
END = 99


def flat(d, prefix=''):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict): out.update(flat(v, f'{prefix}{k}.'))
        else: out[f'{prefix}{k}'] = v
    return out


def research_diff(a, b):
    skip = lambda k: k in ('notes', 'name', 'condition', 'output_dir') or k.startswith('experiment.')
    return {k for k in set(a) | set(b) if a.get(k) != b.get(k) and not skip(k)}


def test_pilot_configs_differ_only_in_the_intended_switches():
    clean, e1, e2 = (flat(X.load_config(f'{T2}/{n}.json')) for n in ('r05_o05', 'pilot/e1_think_shared', 'pilot/e2_think_decision'))
    assert research_diff(clean, e1) == {'rollout.agent_thinking', 'rollout.agent_think_budget', 'schedule.cases'}
    assert research_diff(e1, e2) == {'agent_reward.credit', 'agent_reward.w_outcome', 'agent_reward.w_rubric', 'rubric.enabled'}
    assert e2['agent_reward.credit'] == 'decision_only' and e2['rubric.enabled'] is False and e1['schedule.cases'] == e2['schedule.cases'] == 60
    assert e1['optimizer.learning_rate'] == e2['optimizer.learning_rate'] == clean['optimizer.learning_rate'] == 1e-05
    assert e1['five_gpu.train_order_sha256'] == e2['five_gpu.train_order_sha256'] == clean['five_gpu.train_order_sha256']
    fps = {X.fingerprint(X.load_config(f'{T2}/{n}.json')) for n in ('r05_o05', 'pilot/e1_think_shared', 'pilot/e2_think_decision')}
    assert len(fps) == 3
    for smoke, full in (('smoke_e1', 'e1_think_shared'), ('smoke_e2', 'e2_think_decision')):       # the smokes are the pilot arms with 5 cases
        assert research_diff(flat(X.load_config(f'{T2}/pilot/{smoke}.json')), flat(X.load_config(f'{T2}/pilot/{full}.json'))) == {'schedule.cases'}


def test_switches_are_rejected_without_a_variant_or_with_a_rubric_term():
    raw = X.load_raw(f'{T2}/pilot/e2_think_decision.json') if hasattr(X, 'load_raw') else None
    if raw is None: pytest.skip('no raw loader')
    bad = {**raw, 'experiment': {**raw['experiment'], 'reward_weight_rubric': 0.5, 'reward_weight_outcome': 0.5}}
    with pytest.raises(ValueError): X.resolve(bad)


def tree_with(private_outcomes, leaf_outcomes):
    nodes = tuple(AgentNode(f'c/{j}', None, 'c', 1, (), f'r{j}', {'private_assessment': 'x', 'private_outcome': po}, 'P', j, 'max_agent_turns', True, None)
                  for j, po in enumerate(private_outcomes))
    leaves = tuple(Leaf(f'c/{j}', 'c', (f'c/{j}',), (), 'Final answer: option A', 'A', o, 'max_agent_turns') for j, o in enumerate(leaf_outcomes))
    return Tree('c', 'P', 1, len(nodes), nodes, leaves)


def test_decision_only_credit_is_private_correctness_on_the_private_segment():
    tree = tree_with([1.0, 0.0, None, 1.0], [0.0, 1.0, 1.0, 0.0])                        # private correctness differs from the final outcome
    lo = {l.leaf_id: l.outcome for l in tree.leaves}; ra = {n.node_id: 1.0 for n in tree.nodes}
    shared = node_credit(0.5, 0.5)(tree, lo, ra); dec = node_credit(1.0, 0.0, 'decision_only')(tree, lo, {})
    assert [shared['nodes'][f'c/{j}']['reward'] for j in range(4)] == [0.5, 1.0, 1.0, 0.5] and 'loss_segments' not in shared     # frozen rule unchanged
    assert [dec['nodes'][f'c/{j}']['reward'] for j in range(4)] == [1.0, 0.0, 0.0, 1.0]   # R_D; no option written counts as wrong
    assert dec['loss_segments'] == [0] and dec['credit'] == 'decision_only' and dec['groups'] == shared['groups']
    assert [dec['nodes'][f'c/{j}']['outcome_component'] for j in range(4)] == [0.0, 1.0, 1.0, 0.0]      # the team outcome is still logged


def test_private_outcome_is_read_from_the_private_assessment():
    fn = lambda case, opt: float(opt == case['gold'])
    assert with_private_outcome({'private_assessment': 'OPTION: B\nKEY_REASON: x'}, {'gold': 'B'}, fn)['private_outcome'] == 1.0
    assert with_private_outcome({'private_assessment': 'OPTION: C'}, {'gold': 'B'}, fn)['private_option'] == 'C'
    t = with_private_outcome({'private_assessment': ''}, {'gold': 'B'}, fn); assert t['private_option'] is None and t['private_outcome'] is None
    assert 'private_outcome' not in with_private_outcome({'completion_text': 'legacy'}, {'gold': 'B'}, fn)


def test_loss_segments_restrict_the_loss_to_the_private_segment_and_default_is_unchanged():
    torch, model, up = tiny()
    from src.online_rl.integrated.grpo import grpo_loss, reference_logprobs, sequence_logprobs
    traces = [joint_trace(j) for j in range(3)]
    for t, lp in zip(traces, up.behavior_logprobs(traces)): t['behavior_logprobs'] = lp
    adv = [1.0, -0.5, -0.5]; grads = lambda: {n: p.grad.clone() for n, p in model.named_parameters() if p.requires_grad}
    model.zero_grad(); grpo_loss(model, [{'trajectory_id': f't{j}', 'agent_traces': [t]} for j, t in enumerate(traces)], adv, 0.04, 0.2, 16); default = grads()
    model.zero_grad(); grpo_loss(model, [{'trajectory_id': f't{j}', 'agent_traces': [t], 'loss_segments': None} for j, t in enumerate(traces)], adv, 0.04, 0.2, 16); explicit_none = grads()
    model.zero_grad(); grpo_loss(model, [{'trajectory_id': f't{j}', 'agent_traces': [t], 'loss_segments': [0, 1]} for j, t in enumerate(traces)], adv, 0.04, 0.2, 16); both = grads()
    for n in default: assert torch.equal(default[n], explicit_none[n]) and torch.allclose(default[n], both[n], atol=1e-7), n
    model.zero_grad(); grpo_loss(model, [{'trajectory_id': f't{j}', 'agent_traces': [t], 'loss_segments': [0]} for j, t in enumerate(traces)], adv, 0.04, 0.2, 16); private = grads()
    model.zero_grad(); total = 0                                                          # reference: token mean over the private segment only
    for t, a in zip(traces, adv):
        g = t['segments'][0]; cur = sequence_logprobs(model, g['prompt_input_ids'], g['completion_ids'], grad=True); ref = reference_logprobs(model, g['prompt_input_ids'], g['completion_ids'])
        old = torch.tensor(t['behavior_logprobs'][:len(g['completion_ids'])], dtype=cur.dtype); ratio = torch.exp(cur - old)
        sur = torch.minimum(ratio * a, torch.clamp(ratio, 0.8, 1.2) * a); lr = ref - cur
        total = total + (-sur.mean() + 0.04 * (torch.exp(lr) - lr - 1).mean()) / 3
    total.backward()
    for n, p in model.named_parameters():
        if p.requires_grad: assert torch.allclose(private[n], p.grad, atol=1e-6), n
    assert any(not torch.allclose(private[n], default[n], atol=1e-7) for n in default)     # the visible segment no longer contributes
    with pytest.raises(RuntimeError): grpo_loss(model, [{'trajectory_id': 't', 'agent_traces': [traces[0]], 'loss_segments': [2]}], [1.0], 0.04, 0.2, 16)


class ThinkTok:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, enable_thinking=False):
        return ('<THINK_ON>' if enable_thinking else '<THINK_OFF>') + ''.join(f"<{m['role']}>{m['content']}" for m in messages) + '<assistant>'
    def convert_tokens_to_ids(self, t): return END
    def decode(self, ids, skip_special_tokens=True): return ' '.join(f'w{i}' for i in ids)


def think_policy(closed=True):
    from src.online_rl import environment as ST
    p = SimpleNamespace(tok=ThinkTok(), ST=ST, agent_system='AGENT SYSTEM', precommit=True, thinking=True, think_budget=2048, calls=[],
                        args=SimpleNamespace(max_completion_length=400, rounds=2))
    p.agent_user_prompt = lambda case, prefix, turn: f'STATE turn {turn}'

    def gen(prompt, max_new, seed, *a):
        p.calls.append((prompt, max_new, seed)); ids = [5, 6, 7] + ([END, 11, 12] if closed else [])
        return {'prompt_input_ids': [1, 2], 'completion_ids': ids, 'old_policy_logprobs': [-1.0] * len(ids), 'completion_text': 'RAW ' + ' '.join(map(str, ids))}
    p._generate = gen
    for name in ('agent_turn', '_agent_turn_thinking', 'answer_after_thinking'): setattr(p, name, (lambda n: lambda *a: getattr(SharedPolicy, n)(p, *a))(name))
    return p


def test_thinking_keeps_the_reasoning_in_the_trained_segments_and_out_of_the_dialogue():
    p = think_policy(); tr = p.agent_turn({'case_id': 'c', 'question': 'q', 'options': {'A': 'a', 'B': 'b'}}, [], 1, 3)
    (pa, na, _), (pb, nb, _) = p.calls
    assert pa.startswith('<THINK_ON>') and pb.startswith('<THINK_ON>') and na == AP.ASSESS_MAX_TOKENS + 2048 and nb == 400 + 2048      # reasoning has its own budget
    assert tr['private_assessment'] == 'w11 w12' and tr['completion_text'] == 'w11 w12'                # only the answer after </think> is consumed
    assert 'w11 w12' in pb and 'w5' not in pb and 'RAW' not in pb                                       # the hidden block never carries the reasoning
    for seg in tr['segments']:                                                                          # the loss sees the full sampled completion
        assert seg['completion_ids'] == [5, 6, 7, END, 11, 12] and seg['think_tokens'] == 3 and seg['thinking_closed'] is True and seg['answer_text'] == 'w11 w12'


def test_unclosed_thinking_yields_no_answer():
    p = think_policy(closed=False); tr = p.agent_turn({'case_id': 'c', 'question': 'q', 'options': {'A': 'a', 'B': 'b'}}, [], 1, 3)
    assert tr['private_assessment'] == '' and tr['completion_text'] == '' and tr['segments'][0]['thinking_closed'] is False
    assert with_private_outcome(tr, {'gold': 'A'}, lambda case, opt: 1.0)['private_outcome'] is None       # counted as wrong by the decision credit


def test_default_rendering_has_thinking_off():
    from src.online_rl import environment as ST
    assert ST.chat_prompt(ThinkTok(), 's', 'u').startswith('<THINK_OFF>') and ST.chat_prompt(ThinkTok(), 's', 'u', thinking=True).startswith('<THINK_ON>')


# ---------------------------------------------------------------- credit-separated SHERPA (E3-private)
def sep_setup(rd, ra, coef=(0.5, 0.5), n_groups=1):
    """n_groups sibling groups of len(rd) joint actions each; returns (torch, model, updater, groups, captured gradient holder)."""
    torch, model, up = tiny(); up.o['grad_clip'] = 1e9                                  # no clipping: the applied gradient is the combined one
    from src.online_rl.integrated.continuation import adapter_sha256
    sha = adapter_sha256(model); groups = []
    for gi in range(n_groups):
        traces = [joint_trace(j + 3 * gi) for j in range(len(rd))]
        for t, lp in zip(traces, up.behavior_logprobs(traces)): t['behavior_logprobs'] = lp
        groups.append([{'trajectory_id': f'g{gi}t{j}', 'case_id': f'c|{gi}', 'reward': None, 'rollout_policy_sha256': sha, 'agent_traces': [t],
                        'branch_rewards': {'decision': rd[j], 'collaboration': ra[j]}, 'branch_coefficients': {'decision': coef[0], 'collaboration': coef[1]}}
                       for j, t in enumerate(traces)])
    cap = {}
    def step():                                                                         # capture the gradient the optimizer would apply; do not move the weights
        cap['g'] = {n: (p.grad.clone() if p.grad is not None else torch.zeros_like(p)) for n, p in model.named_parameters() if p.requires_grad}
    up.optimizer.step = step
    return torch, model, up, groups, cap


def branch_grad(torch, model, recs, adv, seg):
    from src.online_rl.integrated.grpo import grpo_loss
    model.zero_grad(); out = grpo_loss(model, [{**r, 'loss_segments': [seg]} for r in recs], adv, 0.04, 0.2, 16)
    g = {n: (p.grad.clone() if p.grad is not None else torch.zeros_like(p)) for n, p in model.named_parameters() if p.requires_grad}; model.zero_grad(); return g, out


def centred(xs): m = sum(xs) / len(xs); return [x - m for x in xs]


def test_separated_objective_is_half_decision_plus_half_collaboration_applied_once():
    rd, ra = [1.0, 0.0, 0.0, 1.0], [1.0, 0.5, 0.0, 0.5]
    torch, model, up, groups, cap = sep_setup(rd, ra); g = groups[0]
    gD, oD = branch_grad(torch, model, g, centred(rd), 0); gC, oC = branch_grad(torch, model, g, centred(ra), 1)
    out = up.update_groups(groups)
    for n in gD: assert torch.allclose(cap['g'][n], 0.5 * gD[n] + 0.5 * gC[n], atol=1e-7), n          # L = 0.5 L_D + 0.5 L_C, coefficient applied once
    assert out['optimizer_step'] and out['credit'] == 'separated' and abs(out['loss'] - (0.5 * oD['loss'] + 0.5 * oC['loss'])) < 1e-9
    b = out['branches']; assert b['decision']['optimized_tokens'] == 4 * 3 and b['collaboration']['optimized_tokens'] == 4 * 2     # private vs visible completion tokens only
    assert abs(b['decision']['loss_contribution'] - 0.5 * oD['loss']) < 1e-12 and abs(b['collaboration']['loss_contribution'] - 0.5 * oC['loss']) < 1e-12
    norm = lambda d: sum(float((v ** 2).sum()) for v in d.values()) ** .5
    assert abs(b['decision']['grad_norm'] - norm(gD)) < 1e-6 and abs(b['collaboration']['grad_norm'] - norm(gC)) < 1e-6       # logged branch gradients are the uncombined ones
    dot = sum(float((gD[n] * gC[n]).sum()) for n in gD); assert abs(out['grad_cosine_decision_collaboration'] - dot / (norm(gD) * norm(gC))) < 1e-6
    assert out['ratio']['max_abs_deviation'] <= 1e-3                                       # on-policy at the update


@pytest.mark.parametrize('coef, seg, key', [((1.0, 0.0), 0, 'decision'), ((0.0, 1.0), 1, 'collaboration')])
def test_separated_with_one_coefficient_reproduces_the_single_branch(coef, seg, key):
    rd, ra = [1.0, 0.0, 0.0, 1.0], [1.0, 0.5, 0.0, 0.5]
    torch, model, up, groups, cap = sep_setup(rd, ra, coef); g = groups[0]
    ref, _ = branch_grad(torch, model, g, centred(rd if seg == 0 else ra), seg); up.update_groups(groups)
    for n in ref: assert torch.allclose(cap['g'][n], ref[n], atol=1e-7), n
    if key == 'decision':                                                                  # (1, 0) = the decision-only rule of E2 (reward R_D, private segment)
        torch2, model2, up2, groups2, cap2 = sep_setup(rd, ra)
        dec = [[{k: v for k, v in r.items() if not k.startswith('branch_')} | {'reward': rd[j], 'loss_segments': [0]} for j, r in enumerate(groups2[0])]]
        up2.update_groups(dec)
        for n in ref: assert torch.allclose(cap2['g'][n], ref[n], atol=1e-7), n


def test_each_segment_receives_only_its_own_advantage():
    rd = [1.0, 0.0, 0.0, 1.0]
    torch, model, up, groups, cap = sep_setup(rd, [1.0, 0.5, 0.0, 0.5]); a = up.update_groups(groups)['branches']
    torch2, model2, up2, groups2, cap2 = sep_setup(rd, [0.0, 0.0, 1.0, 0.5]); b = up2.update_groups(groups2)['branches']      # other R_A, same R_D
    assert abs(a['decision']['grad_norm'] - b['decision']['grad_norm']) < 1e-7 and abs(a['collaboration']['grad_norm'] - b['collaboration']['grad_norm']) > 1e-7
    torch3, model3, up3, groups3, cap3 = sep_setup([0.0, 1.0, 1.0, 0.0], [1.0, 0.5, 0.0, 0.5]); c = up3.update_groups(groups3)['branches']   # other R_D, same R_A
    assert abs(a['collaboration']['grad_norm'] - c['collaboration']['grad_norm']) < 1e-7
    assert a['decision']['optimized_tokens'] + a['collaboration']['optimized_tokens'] == 4 * (3 + 2)                         # completion tokens only: prompts / hidden block never enter


def test_zero_variance_decision_reward_gives_no_decision_gradient():
    rd, ra = [1.0, 1.0, 1.0, 1.0], [1.0, 0.5, 0.0, 0.5]
    torch, model, up, groups, cap = sep_setup(rd, ra); gC, _ = branch_grad(torch, model, groups[0], centred(ra), 1); out = up.update_groups(groups)
    d = out['branches']['decision']; assert d['grad_norm'] is None and d['nonzero_groups'] == 0 and d['skip_reasons'] == {'zero_variance_reward': 1} and d['optimized_tokens'] == 0
    for n in gC: assert torch.allclose(cap['g'][n], 0.5 * gC[n], atol=1e-7), n               # only the collaboration branch, still with its 0.5
    assert out['grad_cosine_decision_collaboration'] is None


def test_unscored_rubric_only_affects_the_collaboration_branch():
    rd, ra = [1.0, 0.0, 0.0, 1.0], [1.0, None, 0.0, 0.5]
    torch, model, up, groups, cap = sep_setup(rd, ra); g = groups[0]
    gD, _ = branch_grad(torch, model, g, centred(rd), 0); scored = [r for r in g if r['branch_rewards']['collaboration'] is not None]
    gC, _ = branch_grad(torch, model, scored, centred([1.0, 0.0, 0.5]), 1); out = up.update_groups(groups); b = out['branches']
    assert b['decision']['actions'] == 4 and b['collaboration']['actions'] == 3
    for n in gD: assert torch.allclose(cap['g'][n], 0.5 * gD[n] + 0.5 * gC[n], atol=1e-7), n
    torch2, model2, up2, groups2, cap2 = sep_setup(rd, [None, None, None, 0.5]); o2 = up2.update_groups(groups2)             # one scored sibling: no collaboration group
    assert o2['branches']['collaboration']['skip_reasons'] == {'fewer_than_two_rewarded_responses': 1} and o2['branches']['decision']['nonzero_groups'] == 1
    torch3, model3, up3, groups3, cap3 = sep_setup([0.0] * 4, [None] * 4); o3 = up3.update_groups(groups3)
    assert o3['optimizer_step'] is False and o3['skip_reason'] == 'no_group_with_nonzero_advantage_in_either_branch' and 'g' not in cap3


def test_separated_credit_rule_fields_and_missing_option():
    tree = tree_with([1.0, 0.0, None, 1.0], [0.0, 1.0, 1.0, 0.0]); lo = {l.leaf_id: l.outcome for l in tree.leaves}
    t = node_credit(0.5, 0.5, 'separated')(tree, lo, {'c/0': 1.0, 'c/1': 0.5, 'c/3': None})
    assert [t['nodes'][f'c/{j}']['decision_reward'] for j in range(4)] == [1.0, 0.0, 0.0, 1.0]                       # no option written = wrong
    assert [t['nodes'][f'c/{j}']['collab_reward'] for j in range(4)] == [1.0, 0.5, None, None]
    assert t['credit'] == 'separated' and t['branch_coefficients'] == {'decision': 0.5, 'collaboration': 0.5} and 'loss_segments' not in t
    assert [t['nodes'][f'c/{j}']['outcome_component'] for j in range(4)] == [0.0, 1.0, 1.0, 0.0]                      # the team outcome is logged, not a loss input


def test_e3_config_differs_from_e1_only_in_the_credit_rule():
    e1, e3 = (flat(X.load_config(f'{T2}/pilot/{n}.json')) for n in ('e1_think_shared', 'e3_think_separated'))
    assert research_diff(e1, e3) == {'agent_reward.credit'} and e3['agent_reward.credit'] == 'separated'
    assert (e3['agent_reward.w_outcome'], e3['agent_reward.w_rubric']) == (0.5, 0.5) and e3['rubric.enabled'] is True and e3['rollout.agent_thinking'] is True
    assert research_diff(flat(X.load_config(f'{T2}/pilot/smoke_e3.json')), e3) == {'schedule.cases'}
    assert len({X.fingerprint(X.load_config(f'{T2}/pilot/{n}.json')) for n in ('e1_think_shared', 'e2_think_decision', 'e3_think_separated')}) == 3


# ---------------------------------------------------------------- E3-outcome: the same separated path, another decision reward
def test_e3_outcome_differs_from_e3_private_only_in_the_decision_target():
    priv, out = (flat(X.load_config(f'{T2}/pilot/{n}.json')) for n in ('e3_think_separated', 'e3o_think_separated_outcome'))
    assert research_diff(priv, out) == {'agent_reward.decision_target'} and out['agent_reward.decision_target'] == 'outcome'
    assert research_diff(flat(X.load_config(f'{T2}/pilot/smoke_e3o.json')), out) == {'schedule.cases', 'five_gpu.agent_batch_size'}       # the smoke uses the final batch size 1
    assert X.fingerprint(X.load_config(f'{T2}/pilot/e3_think_separated.json')) != X.fingerprint(X.load_config(f'{T2}/pilot/e3o_think_separated_outcome.json'))


def test_decision_target_outcome_credits_the_private_segment_with_the_team_outcome():
    tree = tree_with([1.0, 0.0, None, 1.0], [0.0, 1.0, 1.0, 0.0]); lo = {l.leaf_id: l.outcome for l in tree.leaves}; ra = {'c/0': 1.0, 'c/1': 0.5}
    priv = node_credit(0.5, 0.5, 'separated')(tree, lo, ra); out = node_credit(0.5, 0.5, 'separated', 'outcome')(tree, lo, ra)
    assert [out['nodes'][f'c/{j}']['decision_reward'] for j in range(4)] == [0.0, 1.0, 1.0, 0.0]                      # the final team outcome
    assert [priv['nodes'][f'c/{j}']['decision_reward'] for j in range(4)] == [1.0, 0.0, 0.0, 1.0]                     # private option correctness
    assert out['decision_target'] == 'outcome' and priv['decision_target'] == 'private_option'
    for k in ('groups', 'branch_coefficients', 'credit'): assert out[k] == priv[k]                                    # same routing and grouping
    assert all(out['nodes'][n]['collab_reward'] == priv['nodes'][n]['collab_reward'] and out['nodes'][n]['R_D'] == priv['nodes'][n]['R_D'] for n in out['nodes'])
    with pytest.raises(ValueError): node_credit(0.5, 0.5, 'shared', 'outcome')
    with pytest.raises(ValueError): node_credit(0.5, 0.5, 'separated', 'other')


def test_main120_configs_are_the_pilot_arms_with_120_cases_and_the_stated_coefficients():
    pairs = {'e1_shared_5_5': ('e1_think_shared', (0.5, 0.5)), 'e3_outcome_5_5': ('e3o_think_separated_outcome', (0.5, 0.5)), 'e3_private_5_5': ('e3_think_separated', (0.5, 0.5))}
    for main, (pilot, w) in pairs.items():
        m, p = flat(X.load_config(f'{T2}/main120/{main}.json')), flat(X.load_config(f'{T2}/pilot/{pilot}.json'))
        assert research_diff(m, p) == {'schedule.cases', 'five_gpu.agent_batch_size'} and m['schedule.cases'] == 120 and (m['agent_reward.w_outcome'], m['agent_reward.w_rubric']) == w
        assert m['five_gpu.checkpoint_every_cases'] == 30 and m['output_dir'].startswith('outputs/fixed_t2_main120/')
    base = flat(X.load_config(f'{T2}/main120/e3_private_5_5.json'))
    for main, w in (('e3_private_7_3', (0.7, 0.3)), ('e3_private_3_7', (0.3, 0.7))):                                  # (decision, collaboration) coefficients
        m = flat(X.load_config(f'{T2}/main120/{main}.json'))
        assert research_diff(m, base) == {'agent_reward.w_outcome', 'agent_reward.w_rubric'} and (m['agent_reward.w_outcome'], m['agent_reward.w_rubric']) == w
    assert len({flat(X.load_config(f'{T2}/main120/{n}.json'))['output_dir'] for n in ('e1_shared_5_5', 'e3_outcome_5_5', 'e3_private_5_5', 'e3_private_7_3', 'e3_private_3_7')}) == 5


def test_final_matrix_uses_generation_batch_size_one_and_is_otherwise_matched():
    """FINAL 120 protocol: Agent generation batch size 1 on every host; the five arms share everything but the credit rule / coefficients."""
    names = ('e1_shared_5_5', 'e3_outcome_5_5', 'e3_private_5_5', 'e3_private_7_3', 'e3_private_3_7'); cfgs = {n: flat(X.load_config(f'{T2}/main120/{n}.json')) for n in names}
    ref = cfgs['e3_private_5_5']
    for n, c in cfgs.items():
        assert c['five_gpu.agent_batch_size'] == 1 and c['rollout.agent_thinking'] is True and c['rollout.agent_think_budget'] == 2048, n
        assert (c['schedule.cases'], c['rollout.rounds'], c['rollout.branching'], c['optimizer.learning_rate'], c['five_gpu.batch_cases'], c['five_gpu.checkpoint_every_cases']) == (120, 2, 4, 1e-05, 5, 30), n
        assert research_diff(c, ref) <= {'agent_reward.credit', 'agent_reward.decision_target', 'agent_reward.w_outcome', 'agent_reward.w_rubric'}, (n, research_diff(c, ref))
        assert c['five_gpu.train_order_sha256'] == ref['five_gpu.train_order_sha256'] and c['policy.lora.init_seed'] == ref['policy.lora.init_seed']
    assert flat(X.load_config(f'{T2}/pilot/smoke_e3o.json'))['five_gpu.agent_batch_size'] == 1
