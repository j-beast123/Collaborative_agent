"""GRPO update with three distinct policies:
    pi_old   the rollout adapter that sampled the group (teacher-forced logprobs, no grad, recorded before any update)
    pi_theta the trainable current adapter
    pi_ref   the frozen base (adapter disabled), used only for the KL penalty; never the ratio denominator
All three log-probabilities use the same teacher-forced path (sequence_logprobs)."""
from __future__ import annotations
import math
from src.online_rl.integrated.continuation import adapter_sha256
from src.online_rl.integrated.metrics import advantages
from src.online_rl.runtime import GRPOUpdater

MODES = ('group_zscore', 'mean_center')


def sequence_logprobs(model, prompt_ids: list[int], completion_ids: list[int], *, grad: bool,
                      token_chunk_size: int = 16):
    import torch
    import torch.nn.functional as F

    if not completion_ids:
        return torch.empty(0, device=next(model.parameters()).device)
    if token_chunk_size < 1:
        raise ValueError("token_chunk_size must be positive")
    full_ids = torch.tensor([prompt_ids + completion_ids], dtype=torch.long, device=next(model.parameters()).device)
    attention = torch.ones_like(full_ids)
    context = torch.enable_grad() if grad else torch.no_grad()
    with context:
        # Project only completion positions; full-prompt logits exceed a 24 GiB GPU.
        keep = len(completion_ids) + 1
        try:
            outputs = model(
                input_ids=full_ids,
                attention_mask=attention,
                use_cache=False,
                logits_to_keep=keep,
            )
            logits = outputs.logits[:, :-1]
        except TypeError as error:
            if "logits_to_keep" not in str(error):
                raise
            outputs = model(input_ids=full_ids, attention_mask=attention, use_cache=False)
            logits = outputs.logits[:, -keep:-1]

        targets = full_ids[:, len(prompt_ids):]
        if logits.shape[1] != len(completion_ids) or targets.shape[1] != len(completion_ids):
            raise RuntimeError(
                "completion logit alignment failed: "
                f"logits={logits.shape[1]} targets={targets.shape[1]} completion={len(completion_ids)}"
            )

        # Raw policy log-probs (sampling warpers excluded); chunking bounds the float32 vocabulary log-softmax.
        token_logprobs = []
        for start in range(0, len(completion_ids), token_chunk_size):
            end = min(start + token_chunk_size, len(completion_ids))
            chunk_logits = logits[:, start:end].reshape(-1, logits.shape[-1]).float()
            chunk_targets = targets[:, start:end].reshape(-1, 1)
            chunk_logprobs = F.log_softmax(chunk_logits, dim=-1).gather(
                -1, chunk_targets,
            ).squeeze(-1)
            token_logprobs.append(chunk_logprobs)
    return torch.cat(token_logprobs, dim=0)


def reference_logprobs(model, prompt_ids: list[int], completion_ids: list[int], *,
                       token_chunk_size: int = 16):
    if not hasattr(model, "disable_adapter"):
        raise RuntimeError("the LoRA model does not expose disable_adapter for KL reference scoring")
    with model.disable_adapter():
        return sequence_logprobs(
            model, prompt_ids, completion_ids, grad=False,
            token_chunk_size=token_chunk_size,
        )


def behavior_logprobs(model, traces: list[dict], token_chunk_size: int) -> list[list[float]]:
    """log pi_old of each completion through the same teacher-forced path as the loss."""
    model.eval()
    return [sequence_logprobs(model, t['prompt_input_ids'], t['completion_ids'], grad=False, token_chunk_size=token_chunk_size).float().cpu().tolist()
            for t in traces]


def grpo_loss(model, records: list[dict], adv: list[float], beta: float, clip_epsilon: float, token_chunk_size: int) -> dict:
    """Per trajectory: -mean_t min(r_t A, clip(r_t) A) + beta * mean_t KL3(pi_ref || pi_theta), averaged over trajectories;
    r_t = pi_theta / pi_old. Backward is run per trajectory."""
    import torch
    n = len(records); pol = kl = 0.0; ratios, dev_ref = [], 0.0
    for rec, a in zip(records, adv):
        tr = rec['agent_traces'][0]; p, c = tr['prompt_input_ids'], tr['completion_ids']
        cur = sequence_logprobs(model, p, c, grad=True, token_chunk_size=token_chunk_size)
        old = torch.tensor(tr['behavior_logprobs'], dtype=cur.dtype, device=cur.device)
        ref = reference_logprobs(model, p, c, token_chunk_size=token_chunk_size)
        if not (cur.numel() == old.numel() == ref.numel() > 0): raise RuntimeError(f"logprob length mismatch at {rec['trajectory_id']}")
        ratio = torch.exp(cur - old)
        surrogate = torch.minimum(ratio * a, torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * a)
        lr = ref - cur; kl_t = torch.exp(lr) - lr - 1
        loss = (-surrogate.mean() + beta * kl_t.mean()) / n
        if not torch.isfinite(loss).item(): raise RuntimeError(f"non-finite loss at {rec['trajectory_id']}")
        loss.backward()
        pol += float(-surrogate.mean()) / n; kl += float(kl_t.mean()) / n
        ratios += ratio.detach().float().cpu().tolist(); dev_ref = max(dev_ref, float((cur - ref).abs().max()))
    return {'loss': pol + beta * kl, 'policy_loss': pol, 'kl_loss': kl, 'ratio': {
        'count': len(ratios), 'mean': sum(ratios) / len(ratios), 'min': min(ratios), 'max': max(ratios),
        'max_abs_deviation': max(abs(r - 1) for r in ratios), 'clipped_fraction': sum(abs(r - 1) > clip_epsilon for r in ratios) / len(ratios),
        'max_abs_logprob_gap_to_reference': dev_ref, 'denominator': 'behavior_policy_teacher_forced'}}


class IntegratedGRPOUpdater(GRPOUpdater):
    """One update per same-state group; GRPOUpdater supplies the optimizer, save and load."""

    def __init__(self, policy, cfg: dict):
        super().__init__(policy, cfg)
        self.mode = cfg['advantage_normalization']['mode']
        if self.mode not in MODES: raise ValueError(f'advantage_normalization.mode must be one of {MODES}, got {self.mode!r}')

    def identity(self): return {**super().identity(), 'loss': 'src/online_rl/integrated/grpo.grpo_loss', 'advantage_mode': self.mode}

    def behavior_logprobs(self, traces: list[dict]) -> list[list[float]]:
        return behavior_logprobs(self.policy.model, traces, self.o['logprob_token_chunk_size'])

    def update(self, records: list[dict]) -> dict:
        torch, m = self.policy.torch, self.policy.model
        if {r['rollout_policy_sha256'] for r in records} != {adapter_sha256(m)}:
            raise RuntimeError('stale behavior policy: the group was not sampled by the current adapter')
        usable = [r for r in records if r['reward'] is not None]
        adv = advantages([r['reward'] for r in usable], self.mode)
        if len(usable) < 2 or all(a == 0 for a in adv):
            return {'optimizer_step': False, 'skip_reason': 'zero_advantage_group' if len(usable) >= 2 else 'fewer_than_two_rewarded_responses'}
        m.train()
        for mod in m.modules():
            if isinstance(mod, torch.nn.Dropout): mod.eval()
        m.config.use_cache = False; self.optimizer.zero_grad(set_to_none=True)
        out = grpo_loss(m, usable, adv, self.o['beta'], self.o['clip_epsilon'], self.o['logprob_token_chunk_size'])
        if out['ratio']['max_abs_deviation'] > self.o['pre_update_ratio_tolerance']:
            self.optimizer.zero_grad(set_to_none=True); m.config.use_cache = True
            raise RuntimeError(f"on-policy ratio check failed against the behavior policy: {out['ratio']}")
        g = torch.nn.utils.clip_grad_norm_(self.trainable, self.o['grad_clip'])
        if not math.isfinite(float(g)): raise RuntimeError('non-finite gradient norm')
        self.optimizer.step(); self.steps += 1; self.optimizer.zero_grad(set_to_none=True); m.config.use_cache = True
        return {'optimizer_step': True, **out, 'grad_norm': float(g), 'used_agent_completions': len(usable), 'optimizer_steps': self.steps}


def make_updater(policy, cfg: dict) -> IntegratedGRPOUpdater:
    return IntegratedGRPOUpdater(policy, cfg)
