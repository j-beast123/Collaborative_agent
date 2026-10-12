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

MODES = ('group_zscore', 'group_mean', 'mean_center')
RAW_LOGPROB_SEMANTICS = 'raw_policy_before_sampling_warpers_v1'


def advantage_summary(values):
    values = [a for a in values if a is not None]
    return {'count': len(values), 'positive': sum(a > 0 for a in values),
            'negative': sum(a < 0 for a in values), 'zero': sum(a == 0 for a in values),
            'mean': sum(values) / len(values) if values else None,
            'sum_abs': sum(abs(a) for a in values),
            'mean_abs': sum(abs(a) for a in values) / len(values) if values else None,
            'min': min(values) if values else None, 'max': max(values) if values else None,
            'max_abs': max((abs(a) for a in values), default=None)}


def prepare_advantage_groups(groups, mode):
    """Center over the FULL rewarded group before selecting loss trajectories.

    group_mean drops exact-zero advantages from both policy and KL loss. Legacy
    modes retain their original inclusion semantics for existing experiments.
    """
    usable, adv, skipped, details = [], [], [], []
    for group in groups:
        rewards = [r['reward'] for r in group]
        aligned = advantages(rewards, mode)
        rewarded = sum(r is not None for r in rewards)
        reason = ('fewer_than_two_rewarded_responses' if rewarded < 2 else
                  'zero_advantage_group' if all(a == 0 for a in aligned if a is not None) else None)
        pairs = [(r, a) for r, a in zip(group, aligned)
                 if a is not None and (mode != 'group_mean' or a != 0)]
        if reason:
            skipped.append(reason)
        else:
            usable.extend(r for r, a in pairs)
            adv.extend(a for r, a in pairs)
        outcomes = [r.get('outcome') for r in group]
        kind = ('unknown' if not outcomes or any(y not in (0, 1) for y in outcomes) else
                'all_success' if all(y == 1 for y in outcomes) else
                'all_failure' if all(y == 0 for y in outcomes) else 'mixed')
        summary = advantage_summary(aligned)
        details.append({'group': group[0].get('case_id') if group else None,
                        'group_type': kind, 'outcomes': outcomes,
                        'trajectory_ids': [r['trajectory_id'] for r in group],
                        'rewards': rewards, 'advantages': aligned,
                        'rewarded_responses': rewarded, 'used': reason is None, 'skip_reason': reason,
                        'zero_advantage_trajectories': summary['zero'],
                        'used_trajectory_ids': [r['trajectory_id'] for r, a in pairs] if reason is None else [],
                        'used_assistant_tokens': sum(len(segment['completion_ids']) for r, a in pairs
                            for trace in r['agent_traces'] for segment in (trace.get('segments') or [trace])) if reason is None else 0,
                        'advantage': summary})
    return usable, adv, skipped, details


def signal_by_group_type(details):
    result = {}
    total_abs = sum(g['advantage']['sum_abs'] for g in details)
    for kind in ('mixed', 'all_success', 'all_failure', 'unknown'):
        rows = [g for g in details if g['group_type'] == kind]
        abs_sum = sum(g['advantage']['sum_abs'] for g in rows)
        count = sum(g['advantage']['count'] for g in rows)
        result[kind] = {'groups_total': len(rows), 'groups_used': sum(g['used'] for g in rows),
                        'groups_skipped': sum(not g['used'] for g in rows),
                        'trajectories': count, 'used_trajectories': sum(len(g['used_trajectory_ids']) for g in rows),
                        'zero_advantage_trajectories': sum(g['zero_advantage_trajectories'] for g in rows),
                        'sum_abs_advantage': abs_sum,
                        'share_abs_advantage': abs_sum / total_abs if total_abs else None,
                        'mean_abs_advantage': abs_sum / count if count else None,
                        'max_abs_advantage': max((g['advantage']['max_abs'] for g in rows
                                                  if g['advantage']['max_abs'] is not None), default=None),
                        'used_assistant_tokens': sum(g['used_assistant_tokens'] for g in rows)}
    return result


def validate_raw_sampling(settings: dict, *, recorded: bool, label: str):
    """The optional exact protocol samples from the distribution used by this loss.

    Legacy temperature/truncation sampling remains available, but cannot opt in
    merely by relabeling its old raw log-probabilities as on-policy evidence.
    """
    fields = (('sampling_temperature', 'sampling_top_p', 'sampling_top_k')
              if recorded else ('temperature', 'top_p', 'top_k'))
    expected = (1.0, 1.0, 0)
    if any(key not in settings or isinstance(settings[key], bool) or settings[key] != value
           for key, value in zip(fields, expected)):
        raise ValueError(f'{label}: raw on-policy sampling requires explicit temperature=1, top_p=1, top_k=0')
    if recorded and settings.get('logprob_semantics') != RAW_LOGPROB_SEMANTICS:
        raise ValueError(f'{label}: missing or incompatible raw logprob semantics')
    if recorded:
        raw = settings.get('old_policy_logprobs')
        sampled = settings.get('sampling_logprobs')
        tokens = settings.get('completion_ids', [])
        if not tokens or not isinstance(raw, list) or not isinstance(sampled, list) or not (len(raw) == len(sampled) == len(tokens)):
            raise ValueError(f'{label}: raw on-policy sampling requires aligned raw and sampling logprobs')
        if any(not math.isfinite(p) or not math.isfinite(q) or not math.isclose(p, q, rel_tol=1e-6, abs_tol=1e-5)
               for p, q in zip(raw, sampled)):
            raise ValueError(f'{label}: actual sampling logprobs differ from raw policy logprobs')


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


def teacher_forced_behavior_records(model, records, token_chunk_size):
    """Align pi_old and pi_theta numerically while retaining actual sampling evidence.

    BF16 cached batched decoding and full-sequence scoring use different kernels.
    Re-score the unchanged rollout policy before the update, never pi_ref or an
    already-updated adapter. Original sampled logprobs remain in the source traces.
    """
    import torch
    aligned, gaps, ratio_gaps = [], [], []
    for record in records:
        old = behavior_logprobs(model, record['agent_traces'], token_chunk_size)
        traces = []
        for trace, canonical in zip(record['agent_traces'], old):
            recorded = trace['behavior_logprobs']
            if len(recorded) != len(canonical):
                raise RuntimeError('Teacher-forced behavior token alignment changed')
            delta = torch.tensor(canonical, dtype=torch.float32) - torch.tensor(recorded, dtype=torch.float32)
            gaps.extend(delta.abs().tolist())
            ratio_gaps.extend((delta.exp() - 1).abs().tolist())
            traces.append({**trace, 'behavior_logprobs': canonical})
        aligned.append({**record, 'agent_traces': traces})
    return aligned, {'tokens': len(gaps), 'mean_abs_logprob_gap': sum(gaps) / len(gaps) if gaps else 0,
                     'max_abs_logprob_gap': max(gaps, default=0),
                     'max_abs_ratio_gap': max(ratio_gaps, default=0),
                     'source': 'unchanged_rollout_adapter_teacher_forced',
                     'sampling_evidence_preserved': True}


def grpo_loss(model, records: list[dict], adv: list[float], beta: float, clip_epsilon: float, token_chunk_size: int) -> dict:
    """Average over trajectories, then over all assistant tokens in each trajectory.

    One case-level advantage credits every assistant turn. Each turn retains
    its own prompt and behavior log-probabilities; Doctor tokens are excluded.
    """
    import torch
    if not records or len(records) != len(adv):
        raise ValueError('GRPO records and advantages must be nonempty and aligned')
    if any(not math.isfinite(a) for a in adv):
        raise ValueError('GRPO advantages must be finite')
    n = len(records); pol = kl = 0.0; ratios, dev_ref = [], 0.0
    for rec, a in zip(records, adv):
        traces = rec['agent_traces']
        segs = [segment for trace in traces for segment in segments(trace)]
        total = sum(len(c) for _, c in segs)
        old_all = torch.tensor([value for trace in traces for value in trace['behavior_logprobs']], dtype=torch.float32)
        if total == 0 or old_all.numel() != total: raise RuntimeError(f"logprob length mismatch at {rec['trajectory_id']}")
        offset = 0; pol_i = kl_i = 0.0
        for p, c in segs:                                           # same per-token terms over both segments; mean over the whole action
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
            pol_i += float(-surrogate.detach().sum()) / total; kl_i += float(kl_t.detach().sum()) / total
            ratios += ratio.detach().float().cpu().tolist(); dev_ref = max(dev_ref, float((cur.detach() - ref).abs().max()))
        pol += pol_i / n; kl += kl_i / n
    return {'loss': pol + beta * kl, 'policy_loss': pol, 'kl_loss': kl, 'ratio': {
        'count': len(ratios), 'mean': sum(ratios) / len(ratios), 'min': min(ratios), 'max': max(ratios),
        'max_abs_deviation': max(abs(r - 1) for r in ratios), 'clipped_fraction': sum(abs(r - 1) > clip_epsilon for r in ratios) / len(ratios),
        'max_abs_logprob_gap_to_reference': dev_ref, 'denominator': 'behavior_policy_teacher_forced',
        'measured_at': 'before_optimizer_step', 'logprob_distribution': 'raw_policy_without_sampling_warpers'}}


class IntegratedGRPOUpdater(GRPOUpdater):
    """One optimizer step per call over one or more same-state groups; GRPOUpdater supplies the optimizer, save and load."""

    def __init__(self, policy, cfg: dict):
        required = cfg['optimizer'].get('require_raw_on_policy_sampling', False)
        if not isinstance(required, bool):
            raise ValueError('optimizer.require_raw_on_policy_sampling must be a boolean')
        if required:
            validate_raw_sampling(cfg.get('rollout', {}), recorded=False, label='rollout config')
        super().__init__(policy, cfg)
        self.require_raw_on_policy_sampling = required
        self.mode = cfg['advantage_normalization']['mode']
        if self.mode not in MODES: raise ValueError(f'advantage_normalization.mode must be one of {MODES}, got {self.mode!r}')

    def identity(self): return {**super().identity(), 'loss': 'src/online_rl/integrated/grpo.grpo_loss', 'advantage_mode': self.mode}

    def behavior_logprobs(self, traces: list[dict]) -> list[list[float]]:
        return behavior_logprobs(self.policy.model, traces, self.o['logprob_token_chunk_size'])

    def update(self, records: list[dict]) -> dict:
        return self.update_groups([records])

    def update_groups(self, groups: list[list[dict]]) -> dict:
        """One optimizer step over same-case trajectory groups sampled by the current adapter; advantages are computed inside each
        group, and a group without two rewarded responses or with all-zero advantages contributes nothing."""
        torch, m = self.policy.torch, self.policy.model
        if {r['rollout_policy_sha256'] for g in groups for r in g} != {adapter_sha256(m)}:
            raise RuntimeError('stale behavior policy: the group was not sampled by the current adapter')
        require_raw = getattr(self, 'require_raw_on_policy_sampling', False)
        if require_raw:
            for group in groups:
                for record in group:
                    for turn, trace in enumerate(record['agent_traces']):
                        for i, segment in enumerate(trace.get('segments') or [trace]):
                            validate_raw_sampling(segment, recorded=True,
                                                  label=f"{record['trajectory_id']} turn {turn} segment {i}")
        usable, adv, skipped, group_details = prepare_advantage_groups(groups, self.mode)
        all_adv = [a for g in group_details for a in g['advantages'] if a is not None]
        zscore_adv = [a for g in group_details for a in advantages(g['rewards'], 'group_zscore') if a is not None]
        telemetry = {
            'groups_total': len(groups), 'groups_used': len(groups) - len(skipped),
            'groups_skipped': len(skipped),
            'skip_counts': {reason: skipped.count(reason) for reason in sorted(set(skipped))},
            'used_trajectories': len(usable),
            'used_agent_completions': sum(len(record['agent_traces']) for record in usable),
            'optimizer_steps': getattr(self, 'steps', 0),
            'kl_scope': ('nonzero_advantage_trajectories_only' if self.mode == 'group_mean'
                         else 'used_nonzero_advantage_groups_only'),
            'sampling_policy_alignment': ('raw_full_support_verified' if require_raw
                                          else 'legacy_raw_loss_sampling_not_enforced'),
            'advantage_diagnostics': {
                'mode': self.mode, 'groups': group_details,
                'exclude_zero_advantage_trajectories': self.mode == 'group_mean',
                'all': advantage_summary(all_adv), 'used': advantage_summary(adv),
                'same_rewards_group_zscore': advantage_summary(zscore_adv),
                'by_group_type': signal_by_group_type(group_details),
            },
        }
        if not usable:
            return {**telemetry, 'optimizer_step': False,
                    'skip_reason': skipped[0] if len(groups) == 1 else 'no_group_with_nonzero_advantage'}
        if require_raw:
            usable, alignment = teacher_forced_behavior_records(m, usable, self.o['logprob_token_chunk_size'])
            telemetry['behavior_alignment'] = alignment
            telemetry['sampling_policy_alignment'] = 'raw_sampling_verified_teacher_forced_ratio'
        m.train()
        for mod in m.modules():
            if isinstance(mod, torch.nn.Dropout): mod.eval()
        original_use_cache = m.config.use_cache
        m.config.use_cache = False; self.optimizer.zero_grad(set_to_none=True)
        try:
            out = grpo_loss(m, usable, adv, self.o['beta'], self.o['clip_epsilon'], self.o['logprob_token_chunk_size'])
            if out['ratio']['max_abs_deviation'] > self.o['pre_update_ratio_tolerance']:
                raise RuntimeError(f"rollout-policy ratio check failed against the behavior policy: {out['ratio']}")
            g = torch.nn.utils.clip_grad_norm_(self.trainable, self.o['grad_clip'])
            if not math.isfinite(float(g)): raise RuntimeError('non-finite gradient norm')
            self.optimizer.step(); self.steps += 1
        finally:
            self.optimizer.zero_grad(set_to_none=True)
            m.config.use_cache = original_use_cache
        getattr(self.policy, 'mark_dirty', lambda: None)()         # Agent replicas re-sync before the next generation
        return {**telemetry, 'optimizer_step': True, **out, 'grad_norm': float(g), 'optimizer_steps': self.steps}


def make_updater(policy, cfg: dict) -> IntegratedGRPOUpdater:
    return IntegratedGRPOUpdater(policy, cfg)
