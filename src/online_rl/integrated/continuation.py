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
    from src.online_rl.tree.rollout import ROLLOUT_BUDGETS
    ro = cfg['rollout']
    budget = ro.get('budget', 'initial_fanout')
    if budget not in ROLLOUT_BUDGETS: raise ValueError(f'unknown rollout.budget {budget!r}')
    from src.online_rl.tree.group_learning import enabled as group_enabled, identity as group_identity
    if group_enabled(cfg):
        from src.online_rl.doctor import OpenAIDoctor
        if cfg.get('rollout_protocol_sha256_override') is not None:
            raise ValueError('Group SHERPA cannot import a legacy rollout protocol override')
        return sha256_json({'base_model': cfg['policy']['base_model'],
            'rollout': {key: ro.get(key) for key in ('rounds', 'budget', 'branching', 'max_tokens_doctor',
                'max_completion_length', 'temperature', 'top_p', 'top_k', 'agent_protocol', 'agent_sampling_protocol', 'halting')},
            'agent_prompt': sha256_text((root / ro.get('agent_system_prompt_file', 'prompts/agent.txt')).read_text()),
            'partner': OpenAIDoctor(ro['doctor_backend'], ro['halting'],
                (root / ro['doctor_system_prompt_file']).read_text(), ro['max_tokens_doctor']).protocol,
            'group_learning': group_identity(cfg)})
    from src.online_rl.tree.effectiveness import RERUN_OUTCOME_POLICY
    out = {'base_model': cfg['policy']['base_model'],
           'rollout': {k: ro[k] for k in ('rounds', 'doctor_model', 'max_tokens_doctor', 'max_completion_length', 'temperature', 'top_p', 'top_k')},
           'agent_prompt': sha256_text((root / ro.get('agent_system_prompt_file', 'prompts/agent.txt')).read_text()),
           'doctor_prompt': sha256_text(doctor_system := (root / ro.get('doctor_system_prompt_file', 'prompts/user_simulator.txt')).read_text())}
    out['rollout_budget'] = {'protocol': 'initial_fanout_v1', 'branching': ro['branching']}
    if cfg.get('rubric', {}).get('enabled', True):
        if cfg.get('rubric', {}).get('cumulative_bank') is not None:
            from src.online_rl.tree.cumulative_run import config_identity
            out['cumulative_bank'] = config_identity(cfg, root)
        out['rerun_outcome_policy'] = RERUN_OUTCOME_POLICY
        from src.online_rl.tree.applicability import APPLICABILITY_CONTRACT
        from src.online_rl.tree.scoring import SCORE_RECORD_CONTRACT
        out['applicability_contract'] = APPLICABILITY_CONTRACT
        out['score_record_contract'] = SCORE_RECORD_CONTRACT
        if cfg.get('judges', {}).get('gate_protocol') == 'evidence_gate_v2':
            from scripts.online_rl.run_tree import evidence_gate_identity
            out['evidence_gate'] = evidence_gate_identity(cfg, root)
        if cfg.get('judges', {}).get('intervention_audit_protocol'):
            from scripts.online_rl.run_tree import intervention_audit_identity
            out['intervention_audit'] = intervention_audit_identity(cfg, root)
        if cfg.get('rubric', {}).get('generation', {}).get('output_contract') == 'trajectory_analysis_candidate_v2_with_intervention':
            from src.online_rl.tree.member_intervention import MEMBER_INTERVENTION_VERSION
            out['member_intervention_protocol'] = MEMBER_INTERVENTION_VERSION
        intervention_file = (cfg.get('intervention') or {}).get('instruction_template_file')
        if intervention_file:
            out['rerun_intervention_instruction'] = {'protocol': 'joint_generated_instruction_v1',
                'template_sha256': sha256_text((root / intervention_file).read_text())}
    if ro.get('agent_protocol'):                       # Agent action protocol (src/online_rl/agent_protocol.py)
        from src.online_rl.agent_protocol import IDENTITY
        if ro['agent_protocol'] != IDENTITY['version']: raise ValueError(f"unknown rollout.agent_protocol {ro['agent_protocol']!r}")
        out['agent_protocol'] = IDENTITY
    if ro.get('agent_sampling_protocol'):
        from src.online_rl.agent_batch import PROTOCOL
        if ro['agent_sampling_protocol']!=PROTOCOL:raise ValueError('Unknown Agent batch sampling protocol')
        out['agent_sampling_protocol']=PROTOCOL
    if ro.get('doctor_backend'):                       # local Doctor protocol v2 (src/online_rl/doctor.py): backend, prompts, seeds, halting
        from src.online_rl.doctor import OpenAIDoctor
        out['doctor'] = OpenAIDoctor(ro['doctor_backend'], ro.get('halting'), doctor_system, ro['max_tokens_doctor']).protocol
    method = {k: v for k, v in (('verification', cfg.get('verification')),
                                ('r_a_weighting', (cfg.get('agent_reward') or {}).get('r_a_weighting')),
                                ('feedback_behavioral_locus', (cfg.get('feedback') or {}).get('behavioral_locus')),
                                ('feedback_protocol', (cfg.get('feedback') or {}).get('protocol')),
                                ('rubric_source_consistency', (cfg.get('rubric') or {}).get('source_consistency'))) if v}
    if method:                                         # training-method protocol (bounded witness verification, equal-weight R_A,
        out['method'] = method                         # feedback locus): a run under it never shares keys with the earlier protocol
    if cfg.get('rubric', {}).get('schema'):
        from src.online_rl.tree.agent_state import SCHEMA, STATE_PROTOCOL
        if cfg['rubric']['schema'] != SCHEMA: raise ValueError('unknown rubric.schema')
        out['agent_state_rubrics'] = {'schema': SCHEMA, 'state_protocol': STATE_PROTOCOL,
                                     'addendum': sha256_text((root / cfg['rubric']['state_addendum_file']).read_text())}
    if cfg.get('prompt_update', {}).get('candidate_seed'):
        from src.online_rl.tree.pipelined_prompt import SEED_MODE
        if cfg['prompt_update']['candidate_seed'] != SEED_MODE: raise ValueError('unknown prompt candidate seed')
        out['prompt_candidate_seed'] = SEED_MODE
    if cfg.get('rubric', {}).get('match_batch_size',1)!=1:
        from src.online_rl.tree.rubric_bank import BATCH_VERSION
        if cfg['rubric']['match_batch_size']!=8:raise ValueError('unsupported rubric match batch size')
        out['rubric_batch_matching']={'version':BATCH_VERSION,'batch_size':8}
    repeat_after=(ro.get('halting') or {}).get('repeat_agent_reply_after_turns')
    if repeat_after is not None:
        if isinstance(repeat_after,bool) or not isinstance(repeat_after,int) or repeat_after<1:raise ValueError('repeat_agent_reply_after_turns must be a positive integer')
        out['agent_repeated_reply_exit']={'protocol':'normalized_exact_prior_reply_v1','after_agent_turns':repeat_after}
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
