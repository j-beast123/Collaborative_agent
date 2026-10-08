"""Seeds, policy/protocol identity, and the policy-keyed continuation cache.

The Doctor API is not seedable: a seed fixes only local Agent sampling, so cached continuations are immutable observed
samples, not reproducible draws."""
from __future__ import annotations
import hashlib, json, threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional
from src.online_rl.integrated.records import ContinuationRecord, from_dict, sha256_json, sha256_text, to_dict

NATURAL = 'NATURAL'


def derive_seed(run_seed: int, key: str, role: str, index: int) -> int:
    """The single seed rule: H(run_seed, key, role, index)."""
    return int(hashlib.sha256(f'{run_seed}|{key}|{role}|{index}'.encode()).hexdigest()[:8], 16)


def unit_hash(run_seed: int, key: str, role: str) -> float:
    return derive_seed(run_seed, key, role, 0) / 2 ** 32


def in_baseline(mode: str, fraction: float, run_seed: int, state_key: str) -> bool:
    """Diagnostic-subset membership is a state-key hash, never an outcome."""
    if mode == 'off': return False
    if mode == 'every_step': return True
    if mode == 'diagnostic_subset': return unit_hash(run_seed, state_key, 'baseline') < fraction
    raise ValueError(f'unknown effectiveness.baseline_mode {mode}')


def protocol_sha256(cfg: dict, root: Path) -> str:
    ro = cfg['rollout']
    out = {'base_model': cfg['policy']['base_model'],
           'rollout': {k: ro[k] for k in ('rounds', 'doctor_model', 'max_tokens_doctor', 'max_completion_length', 'temperature', 'top_p', 'top_k')},
           'agent_prompt': sha256_text((root / ro.get('agent_system_prompt_file', 'prompts/agent.txt')).read_text()),
           'doctor_prompt': sha256_text(doctor_system := (root / ro.get('doctor_system_prompt_file', 'prompts/user_simulator.txt')).read_text())}
    if ro.get('agent_protocol'):                       # Agent action protocol (src/online_rl/agent_protocol.py)
        from src.online_rl.agent_protocol import IDENTITY
        if ro['agent_protocol'] != IDENTITY['version']: raise ValueError(f"unknown rollout.agent_protocol {ro['agent_protocol']!r}")
        out['agent_protocol'] = IDENTITY
    if ro.get('doctor_backend'):                       # local Doctor protocol v2 (src/online_rl/doctor.py): backend, prompts, seeds, halting
        from src.online_rl.doctor import OllamaDoctor
        out['doctor'] = OllamaDoctor(ro['doctor_backend'], ro.get('halting'), doctor_system, ro['max_tokens_doctor']).protocol
    method = {k: v for k, v in (('verification', cfg.get('verification')),
                                ('r_a_weighting', (cfg.get('agent_reward') or {}).get('r_a_weighting')),
                                ('feedback_behavioral_locus', (cfg.get('feedback') or {}).get('behavioral_locus')),
                                ('feedback_protocol', (cfg.get('feedback') or {}).get('protocol'))) if v}
    if method:                                         # training-method protocol (bounded witness verification, equal-weight R_A,
        out['method'] = method                         # feedback locus): a run under it never shares keys with the earlier protocol
    if ro.get('doctor_opening'):                       # frozen shared D1 tables (src/online_rl/doctor_openings.py)
        from src.online_rl.doctor_openings import protocol_identity
        out['doctor_opening'] = protocol_identity(ro['doctor_opening'])
    return sha256_json(out)


def adapter_sha256(model) -> str:
    """Hash of the trainable (LoRA) tensors: any optimizer step changes it."""
    h = hashlib.sha256()
    for name, p in sorted((n, p) for n, p in model.named_parameters() if p.requires_grad):
        h.update(name.encode()); h.update(p.detach().float().cpu().numpy().tobytes())
    return h.hexdigest()


@dataclass(frozen=True)
class PolicyIdentity:
    base_model: str
    policy_sha256: str
    protocol_sha256: str


def continuation_key(ident: PolicyIdentity, case_id: str, state_key: str, response_sha256: str, seed: int) -> str:
    return sha256_json([ident.base_model, ident.policy_sha256, ident.protocol_sha256, case_id, state_key, response_sha256, seed])


class ContinuationCache:
    """Append-only JSONL. The key includes the adapter hash, so a policy change is always a miss."""

    def __init__(self, path: Optional[Path]):
        self.path = Path(path) if path else None; self._lock = threading.Lock(); self.records: dict[str, ContinuationRecord] = {}
        self.hits = self.misses = 0
        if self.path and self.path.exists():
            for line in self.path.read_text().splitlines():
                if line.strip(): self._add(from_dict(ContinuationRecord, json.loads(line)))

    def _add(self, rec: ContinuationRecord) -> None:
        ident = PolicyIdentity(rec.base_model, rec.policy_sha256, rec.protocol_sha256)
        if continuation_key(ident, rec.case_id, rec.state_key, rec.response_sha256, rec.seed) != rec.key:
            raise RuntimeError(f'corrupted continuation cache record {rec.key}')
        self.records[rec.key] = rec

    def get(self, key: str, policy_sha256: str) -> Optional[ContinuationRecord]:
        rec = self.records.get(key)
        if rec is not None and rec.policy_sha256 != policy_sha256:
            raise RuntimeError(f'continuation {key} belongs to policy {rec.policy_sha256[:12]}, not {policy_sha256[:12]}')
        with self._lock:
            if rec is None: self.misses += 1
            else: self.hits += 1
        return rec

    def put(self, rec: ContinuationRecord) -> None:
        with self._lock:
            self._add(rec)
            if self.path:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open('a') as f: f.write(json.dumps(to_dict(rec), ensure_ascii=False) + '\n')


def continue_k(policy, cache: ContinuationCache, ident: PolicyIdentity, case: dict, state_key: str, prefix: list[dict], turn: int,
               action: Optional[str], seeds: list[int], natural_seeds: Optional[list[int]], outcome_fn: Callable) -> list[ContinuationRecord]:
    """K continuations from (h, action), or from the policy's own reply when action is None (V(h) arm).
    Continuation i uses seeds[i] for every response and for NATURAL, so arms are paired by index."""
    if (action is None) != (natural_seeds is not None): raise ValueError('natural_seeds are required exactly for the NATURAL arm')
    rsha = NATURAL if action is None else sha256_text(action); natural = natural_seeds or []; out = []
    for i, seed in enumerate(seeds):
        key = continuation_key(ident, case['case_id'], state_key, rsha, seed)
        rec = cache.get(key, ident.policy_sha256)
        if rec is None:
            reply = action if action is not None else policy.natural(case, prefix, turn, natural[i])
            c = policy.continue_from(case, prefix, turn, reply, seed)
            rec = ContinuationRecord(key, ident.base_model, ident.policy_sha256, ident.protocol_sha256, case['case_id'], state_key, rsha, seed,
                                     c.get('final_option'), outcome_fn(case, c.get('final_option')), c.get('termination_reason'),
                                     reply if action is None else None, c.get('dialogue') or [])
            cache.put(rec)
        out.append(rec)
    return out
