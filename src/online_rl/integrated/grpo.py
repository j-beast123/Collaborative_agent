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


def segments(trace: dict) -> list[tuple[list[int], list[int]]]:
    """(prompt ids, completion ids) of every sampled segment of one action: one for a plain completion, two (private
    assessment, visible reply) for a precommit_v1 joint action, each under its own prompt."""
    if trace.get('segments'): return [(g['prompt_input_ids'], g['completion_ids']) for g in trace['segments']]
    return [(trace['prompt_input_ids'], trace['completion_ids'])]


def behavior_logprobs(model, traces: list[dict], token_chunk_size: int) -> list[list[float]]:
    """log pi_old of each action's tokens (all segments, in order) through the same teacher-forced path as the loss."""
    model.eval()
    return [[x for p, c in segments(t) for x in sequence_logprobs(model, p, c, grad=False, token_chunk_size=token_chunk_size).float().cpu().tolist()]
            for t in traces]


def grpo_loss(model, records: list[dict], adv: list[float], beta: float, clip_epsilon: float, token_chunk_size: int) -> dict:
    """Per action: -mean_t min(r_t A, clip(r_t) A) + beta * mean_t KL3(pi_ref || pi_theta) over ALL its sampled tokens (both
    segments of a precommit_v1 action, each teacher-forced under its own prompt; one segment otherwise), averaged over
    actions; r_t = pi_theta / pi_old. The sibling advantage A credits every token of the action. Backward per segment with
    weight |segment| / |action| (the same gradient as one backward of the per-action mean).
    A record may name the segments that enter the loss (rec['loss_segments'], e.g. [0] = the private assessment only, the
    decision-only diagnostic); the token mean is then over those segments. Absent / None = all segments (unchanged)."""
    import torch
    n = len(records); pol = kl = 0.0; ratios, dev_ref = [], 0.0
    for rec, a in zip(records, adv):
        tr = rec['agent_traces'][0]; segs = segments(tr); use = rec.get('loss_segments')
        use = set(range(len(segs))) if use is None else set(use)
        if not use or not use <= set(range(len(segs))): raise RuntimeError(f"loss_segments {sorted(use)} do not exist at {rec['trajectory_id']}")
        total = sum(len(c) for k, (_, c) in enumerate(segs) if k in use)
        old_all = torch.tensor(tr['behavior_logprobs'], dtype=torch.float32)
        if total == 0 or old_all.numel() != sum(len(c) for _, c in segs): raise RuntimeError(f"logprob length mismatch at {rec['trajectory_id']}")
        offset = 0; pol_i = kl_i = 0.0
        for k, (p, c) in enumerate(segs):                           # same per-token terms over the used segments; mean over their tokens
            if k not in use: offset += len(c); continue
            cur = sequence_logprobs(model, p, c, grad=True, token_chunk_size=token_chunk_size)
            old = old_all[offset:offset + len(c)].to(device=cur.device, dtype=cur.dtype); offset += len(c)
            ref = reference_logprobs(model, p, c, token_chunk_size=token_chunk_size)
            if not (cur.numel() == old.numel() == ref.numel()): raise RuntimeError(f"logprob length mismatch at {rec['trajectory_id']}")
            if cur.numel() == 0: continue
            ratio = torch.exp(cur - old)
            surrogate = torch.minimum(ratio * a, torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * a)
            lr = ref - cur; kl_t = torch.exp(lr) - lr - 1
            loss = (-surrogate.sum() + beta * kl_t.sum()) / total / n      # = the per-action token mean, split by segment
            if not torch.isfinite(loss).item(): raise RuntimeError(f"non-finite loss at {rec['trajectory_id']}")
            loss.backward()
            pol_i += float(-surrogate.sum()) / total; kl_i += float(kl_t.sum()) / total
            ratios += ratio.detach().float().cpu().tolist(); dev_ref = max(dev_ref, float((cur - ref).abs().max()))
        pol += pol_i / n; kl += kl_i / n
    return {'loss': pol + beta * kl, 'policy_loss': pol, 'kl_loss': kl, 'ratio': {
        'count': len(ratios), 'mean': sum(ratios) / len(ratios), 'min': min(ratios), 'max': max(ratios),
        'max_abs_deviation': max(abs(r - 1) for r in ratios), 'clipped_fraction': sum(abs(r - 1) > clip_epsilon for r in ratios) / len(ratios),
        'max_abs_logprob_gap_to_reference': dev_ref, 'denominator': 'behavior_policy_teacher_forced'}}


class IntegratedGRPOUpdater(GRPOUpdater):
    """One optimizer step per call over one or more same-state groups; GRPOUpdater supplies the optimizer, save and load."""

    def __init__(self, policy, cfg: dict):
        super().__init__(policy, cfg)
        self.mode = cfg['advantage_normalization']['mode']
        if self.mode not in MODES: raise ValueError(f'advantage_normalization.mode must be one of {MODES}, got {self.mode!r}')

    def identity(self): return {**super().identity(), 'loss': 'src/online_rl/integrated/grpo.grpo_loss', 'advantage_mode': self.mode}

    def behavior_logprobs(self, traces: list[dict]) -> list[list[float]]:
        return behavior_logprobs(self.policy.model, traces, self.o['logprob_token_chunk_size'])

    def update(self, records: list[dict]) -> dict:
        return self.update_groups([records])

    def update_groups(self, groups: list[list[dict]]) -> dict:
        """One optimizer step over same-prompt groups sampled by the current adapter; advantages are computed inside each
        group, and a group without two rewarded responses or with all-zero advantages contributes nothing."""
        torch, m = self.policy.torch, self.policy.model
        if {r['rollout_policy_sha256'] for g in groups for r in g} != {adapter_sha256(m)}:
            raise RuntimeError('stale behavior policy: the group was not sampled by the current adapter')
        usable, adv, skipped = [], [], []
        for g in groups:
            u = [r for r in g if r['reward'] is not None]; a = advantages([r['reward'] for r in u], self.mode)
            if len(u) < 2: skipped.append('fewer_than_two_rewarded_responses')
            elif all(x == 0 for x in a): skipped.append('zero_advantage_group')
            else: usable += u; adv += a
        if not usable:
            if len(groups) == 1: return {'optimizer_step': False, 'skip_reason': skipped[0]}
            return {'optimizer_step': False, 'skip_reason': 'no_group_with_nonzero_advantage', 'groups_skipped': len(skipped)}
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
        getattr(self.policy, 'mark_dirty', lambda: None)()         # Agent replicas re-sync before the next generation
        return {'optimizer_step': True, **out, 'grad_norm': float(g), 'used_agent_completions': len(usable), 'optimizer_steps': self.steps,
                'groups_used': len(groups) - len(skipped), 'groups_skipped': len(skipped)}


def make_updater(policy, cfg: dict) -> IntegratedGRPOUpdater:
    return IntegratedGRPOUpdater(policy, cfg)
