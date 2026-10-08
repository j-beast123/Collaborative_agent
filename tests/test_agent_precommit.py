"""Agent protocol precommit_v1 (src/online_rl/agent_protocol.py): per-sibling Doctor-blind private assessment z_i, then the
visible reply r_i conditioned on the state and z_i; joint-action GRPO over both segments. Fakes and a tiny CPU LoRA model."""
import json
from types import SimpleNamespace
import pytest
from src.online_rl import agent_protocol as AP
from src.online_rl.runtime import SharedPolicy

CASE = {'case_id': 'c1', 'question': 'Which finding is characteristic?', 'scenario': '', 'options': {k: f'option text {k}' for k in 'ABCD'}}
PREFIX = [{'speaker': 'doctor', 'stage': 'D1', 'round': 1, 'text': 'DOCTOR SAYS: I am sure it is B because of the striae.'}]


class Tok:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, enable_thinking=False):
        return ''.join(f"<{m['role']}>{m['content']}" for m in messages) + '<assistant>'


def fake_policy(precommit=True):
    """SharedPolicy.agent_turn with generation replaced by a recorder: output depends on (prompt, seed)."""
    from src.online_rl import environment as ST
    p = SimpleNamespace(tok=Tok(), ST=ST, agent_system='AGENT SYSTEM', precommit=precommit, calls=[],
                        args=SimpleNamespace(max_completion_length=400, rounds=5))
    p.agent_user_prompt = lambda case, prefix, turn: f"STATE turn {turn}: " + ' | '.join(d['text'] for d in prefix)

    def gen(prompt, max_new, seed):
        p.calls.append((prompt, max_new, seed)); n = len(prompt) % 7
        text = f'OPTION: {"ABCD"[seed % 4]}\nKEY_REASON: reason {seed}' if max_new == AP.ASSESS_MAX_TOKENS else f'Visible reply {seed}'
        return {'prompt_input_ids': [len(prompt) % 50, n], 'completion_ids': [seed % 30, 3, 4], 'old_policy_logprobs': [-1.0] * 3, 'completion_text': text}
    p._generate = gen
    p.agent_turn = lambda *a: SharedPolicy.agent_turn(p, *a); p.agent_text = lambda *a: SharedPolicy.agent_text(p, *a)
    return p


def test_each_sibling_has_its_own_blind_assessment_and_reply():
    p = fake_policy(); seeds = [11, 22, 33, 44]
    tr = [p.agent_turn(CASE, PREFIX, 1, s) for s in seeds]
    stage_a = [c for c in p.calls if c[1] == AP.ASSESS_MAX_TOKENS]; stage_b = [c for c in p.calls if c[1] == 400]
    assert len(stage_a) == len(stage_b) == 4
    assert len({c[0] for c in stage_a}) == 1 and all('DOCTOR SAYS' not in c[0] and 'STATE' not in c[0] for c in stage_a)   # blind, shared prompt
    assert all('option text A' in c[0] and CASE['question'] in c[0] for c in stage_a)
    assert len({c[2] for c in p.calls}) == 8 and [c[2] for c in stage_a] == [AP.stage_seeds(s, 'c1')[0] for s in seeds]  # 8 distinct seeds
    for t, s in zip(tr, seeds):                                                           # stage B sees only its own z
        z = t['private_assessment']; others = [u['private_assessment'] for u in tr if u is not t]
        assert z in t['prompt_text'] and not any(o in t['prompt_text'] for o in others) and 'DOCTOR SAYS' in t['prompt_text']
        assert [g['role'] for g in t['segments']] == ['private_assessment', 'visible_reply'] and t['completion_text'] == f'Visible reply {AP.stage_seeds(s, "c1")[1]}'
        assert z not in t['completion_text']
    assert len({t['state_prompt_hash'] for t in tr}) == 1 and len({t['prompt_text'] for t in tr}) == 4   # same state, different stage-B prompts


def test_legacy_protocol_is_one_completion():
    p = fake_policy(precommit=False); t = p.agent_turn(CASE, PREFIX, 1, 5)
    assert len(p.calls) == 1 and 'segments' not in t and p.calls[0][2] == 5


def test_visible_reply_contract_forbids_meta_language():
    for phrase in ('my initial assessment', 'I initially thought', 'my earlier', 'third person', 'Return only the natural reply'):
        assert phrase in AP.HIDDEN_BLOCK
    assert 'gold' not in AP.ASSESS_SYSTEM.lower() and 'physician' not in AP.ASSESS_SYSTEM.lower()


def test_group_invariant_is_the_original_state():
    from src.online_rl.integrated.episode import assert_single_prompt
    p = fake_policy(); tr = [p.agent_turn(CASE, PREFIX, 1, s) for s in (1, 2, 3, 4)]
    recs = lambda ts: [{'case_id': 'g', 'agent_traces': [t]} for t in ts]
    assert_single_prompt(recs(tr))                                                         # different stage-B prompts are fine
    other = p.agent_turn(CASE, PREFIX + [{'speaker': 'agent', 'stage': 'A1', 'round': 1, 'text': 'x'}], 2, 5)
    with pytest.raises(RuntimeError, match='mixes'): assert_single_prompt(recs(tr[:3] + [other]))   # different original state
    legacy = fake_policy(precommit=False).agent_turn(CASE, PREFIX, 1, 9)
    with pytest.raises(RuntimeError, match='mixes'): assert_single_prompt(recs(tr[:3] + [legacy]))  # different protocol


def test_protocol_is_part_of_the_rollout_protocol_hash():
    from pathlib import Path
    from src.online_rl.integrated.continuation import protocol_sha256
    root = Path(__file__).resolve().parents[1]; cfg = json.loads((root / 'configs/online_rl/tree_v1.json').read_text())
    a = {**cfg, 'rollout': {k: v for k, v in cfg['rollout'].items() if k != 'agent_protocol'}}
    b = {**cfg, 'rollout': {**cfg['rollout'], 'agent_protocol': 'precommit_v1'}}
    assert protocol_sha256(a, root) != protocol_sha256(b, root)
    with pytest.raises(ValueError): protocol_sha256({**cfg, 'rollout': {**cfg['rollout'], 'agent_protocol': 'other'}}, root)


# ---------- joint-action GRPO on a tiny CPU LoRA model ----------
def tiny():
    torch = pytest.importorskip('torch'); tf = pytest.importorskip('transformers'); peft = pytest.importorskip('peft')
    from src.online_rl.integrated.grpo import make_updater
    torch.manual_seed(0)
    base = tf.LlamaForCausalLM(tf.LlamaConfig(vocab_size=40, hidden_size=16, intermediate_size=32, num_hidden_layers=1, num_attention_heads=2,
                                              num_key_value_heads=2, max_position_embeddings=64))
    model = peft.get_peft_model(base, peft.LoraConfig(r=2, lora_alpha=4, target_modules=['q_proj', 'v_proj'], lora_dropout=0.0))
    torch.manual_seed(1)
    for n, prm in model.named_parameters():
        if 'lora_B' in n: torch.nn.init.normal_(prm, std=0.05)                       # non-trivial adapter
    cfg = {'optimizer': {'learning_rate': 1e-2, 'beta': 0.04, 'clip_epsilon': 0.2, 'grad_clip': 1.0, 'logprob_token_chunk_size': 16,
                         'pre_update_ratio_tolerance': 1e-3}, 'advantage_normalization': {'mode': 'mean_center'}}
    return torch, model, make_updater(SimpleNamespace(torch=torch, model=model), cfg)


def joint_trace(j):
    return {'agent_protocol': 'precommit_v1', 'state_prompt_hash': 'S', 'prompt_input_ids': [7, 8, 9, 30 + j], 'completion_ids': [12 + j, 13],
            'segments': [{'role': 'private_assessment', 'prompt_input_ids': [1, 2, 3], 'completion_ids': [20 + j, 21, 22]},
                         {'role': 'visible_reply', 'prompt_input_ids': [7, 8, 9, 30 + j], 'completion_ids': [12 + j, 13]}]}


def test_joint_loss_covers_both_segments_with_one_shared_advantage():
    torch, model, up = tiny()
    from src.online_rl.integrated.grpo import grpo_loss, reference_logprobs, sequence_logprobs
    traces = [joint_trace(j) for j in range(3)]
    for t, lp in zip(traces, up.behavior_logprobs(traces)):
        t['behavior_logprobs'] = lp; assert len(lp) == 3 + 2                                # both segments, in order
    recs = [{'trajectory_id': f't{j}', 'agent_traces': [t]} for j, t in enumerate(traces)]; adv = [1.0, -0.5, -0.5]
    model.zero_grad(); grpo_loss(model, recs, adv, 0.04, 0.2, 16)
    got = {n: p.grad.clone() for n, p in model.named_parameters() if p.requires_grad}
    model.zero_grad(); total = 0                                                          # reference: per-action token mean over both segments
    for t, a in zip(traces, adv):
        cur = torch.cat([sequence_logprobs(model, g['prompt_input_ids'], g['completion_ids'], grad=True) for g in t['segments']])
        ref = torch.cat([reference_logprobs(model, g['prompt_input_ids'], g['completion_ids']) for g in t['segments']])
        old = torch.tensor(t['behavior_logprobs'], dtype=cur.dtype); ratio = torch.exp(cur - old)
        sur = torch.minimum(ratio * a, torch.clamp(ratio, 0.8, 1.2) * a); lr = ref - cur
        total = total + (-sur.mean() + 0.04 * (torch.exp(lr) - lr - 1).mean()) / 3
    total.backward()
    for n, p in model.named_parameters():
        if p.requires_grad: assert torch.allclose(got[n], p.grad, atol=1e-6), n
    assert any(float(g.abs().sum()) > 0 for g in got.values())


def test_single_segment_loss_is_unchanged():
    torch, model, up = tiny()
    from src.online_rl.integrated.grpo import grpo_loss
    plain = [{'prompt_input_ids': [1, 2, 3, 4], 'completion_ids': [5 + j, 9, 11 + j]} for j in range(2)]
    seg = [{**t, 'segments': [{'role': 'visible_reply', 'prompt_input_ids': t['prompt_input_ids'], 'completion_ids': t['completion_ids']}]} for t in plain]
    out = []
    for traces in (plain, seg):
        for t, lp in zip(traces, up.behavior_logprobs(traces)): t['behavior_logprobs'] = lp
        model.zero_grad(); r = grpo_loss(model, [{'trajectory_id': f't{j}', 'agent_traces': [t]} for j, t in enumerate(traces)], [0.5, -0.5], 0.04, 0.2, 16)
        out.append((r['loss'], [p.grad.clone() for p in model.parameters() if p.requires_grad]))
    assert out[0][0] == pytest.approx(out[1][0]) and all(torch.allclose(a, b) for a, b in zip(out[0][1], out[1][1]))


def test_joint_action_update_step_runs_on_policy():
    torch, model, up = tiny()
    from src.online_rl.integrated.continuation import adapter_sha256
    traces = [joint_trace(j) for j in range(4)]
    for t, lp in zip(traces, up.behavior_logprobs(traces)): t['behavior_logprobs'] = lp
    sha = adapter_sha256(model)
    g = [{'trajectory_id': f't{j}', 'case_id': 's', 'reward': r, 'rollout_policy_sha256': sha, 'agent_traces': [t]} for j, (r, t) in enumerate(zip([1, 0, 0, 1], traces))]
    out = up.update_groups([g])
    assert out['optimizer_step'] and out['ratio']['count'] == 4 * 5 and out['ratio']['max_abs_deviation'] < 1e-3 and adapter_sha256(model) != sha
