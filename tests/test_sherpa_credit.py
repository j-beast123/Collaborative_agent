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
