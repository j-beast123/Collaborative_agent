"""Case-tree trainer. Per case: branching rollout -> FINAL/outcome per leaf -> Doctor feedback -> ONE rubric per trajectory
-> match on the source trajectory -> Score-0/Score-2 intervention, K reruns per target, s(tau) -> KEEP/REMOVE -> kept rubrics
score every response of the tree (No Match kept) -> Agent targets/update (injected).
run_state.json is written last per case; every external call goes through the immutable call cache, so a resumed case
repeats no completed API or model work."""
from __future__ import annotations
import dataclasses, shutil, time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional
from src.online_rl.guards import assert_not_protected
from src.online_rl.integrated.episode import schedule
from src.online_rl.integrated.prompt_search import PromptStore, SearchBackend
from src.online_rl.integrated.records import atomic_write_json, dir_sha256, from_dict, read_json, sha256_json, to_dict
from src.online_rl.tree.cache import CallCache
from src.online_rl.tree.decisions import Strategies
from src.online_rl.tree import rubric_bank as RB
from src.online_rl.tree.conditions import active_unresolved
from src.online_rl.tree.reward import (exclude_reflection_failure_trajectories, exclude_trajectories, scoring_execution_failure_ids, interaction_diagnostics,
                                       reflection_failure_ids, sibling_group_diagnostics)
from src.online_rl.tree.effectiveness import RERUN_CONTINUATION_PROTOCOL, RERUN_OUTCOME_POLICY, evaluate_rubric, select
from src.online_rl.tree.posthoc import doctor_feedback, freeze_trajectory_input, generate_rubric
from src.online_rl.tree.prompt_search import EvidenceUnit, ScoringDeps, ready, run_event
from src.online_rl.tree.records import (DoctorFeedback, RubricAssignment, SelectionDecision,
                                        TrajectoryRubric, Tree, TrajectoryRubricInput)
from src.online_rl.tree.reward import agent_targets
from src.online_rl.tree.rollout import build_tree, tree_counts
from src.online_rl.tree.scoring import score_nodes
from src.online_rl.tree import witness as W
from src.online_rl.tree.team import RUBRIC_ONLY_NA_POLICY

VERSION = 'sherpa_tree_v1'
REFLECTION_FAILURE_POLICY = 'exclude_failed_post_final_reflection_from_rubric_and_agent_reward_v1'


def feedback_failure_policy(cfg: dict) -> Optional[str]:
    return (REFLECTION_FAILURE_POLICY
            if (cfg.get('feedback') or {}).get('protocol') == 'physician_post_final_reflection_v1'
            else None)


def model_roles(cfg: dict) -> dict:
    """Provider model id of every pipeline role (provenance)."""
    r, j, pu = cfg.get('rubric', {}), cfg.get('judges', {}), cfg.get('prompt_update', {})
    b = cfg.get('rollout', {}).get('doctor_backend')
    doctor = cfg.get('rollout', {}).get('doctor_model') if not b else f"{b['model']} ({b['engine']} {b['ollama_version']}, {b['quantization']}, digest {b['model_digest']}; source {b.get('source_model')})"
    roles = {'doctor': doctor, 'agent': f"{cfg.get('policy', {}).get('base_model')} + LoRA (local)",
            'feedback_generator': (cfg.get('feedback') or {}).get('model'), 'rubric_generator': r.get('generator_model'),
            'rubric_gate': j.get('gate_model'), 'rubric_scorer': j.get('scorer_model'),
            'prompt_reflection': pu.get('reflection_model'), 'prompt_refiner': pu.get('refiner_model')}
    if j.get('intervention_audit_protocol'):
        roles['intervention_audit'] = j.get('intervention_audit_model')
    return roles


@dataclass(frozen=True)
class TreeRunState:
    config_sha256: str
    run_seed: int
    case_cursor: int
    prompt_id: str
    prompt_sha256: str
    policy_sha256: str
    checkpoint: Optional[str]
    lineage_cursor: int
    last_event_cursor: int
    history: tuple[dict, ...] = field(default_factory=tuple)
    eval_cursor: int = 0                  # last milestone whose held-out evaluation is complete


@dataclass(frozen=True)
class TreeEnv:
    """External components: agent.sample, doctor.turn/final, stop_rule, final_option, outcome_fn (gold, rollout side only),
    feedback_fn, generate_fn, scorer (two-stage, No Match), writer (Score-0/Score-2 reply), continuation (rerun to FINAL),
    gate_fn (post-hoc rubric gate) and behavior_logprobs (pi_old for the update)."""
    agent: object
    doctor: object
    stop_rule: Callable
    final_option: Callable
    outcome_fn: Callable
    feedback_fn: Callable
    generate_fn: Callable
    scorer: Callable
    writer: Callable
    continuation: Callable
    gate_fn: Optional[Callable] = None
    behavior_logprobs: Optional[Callable] = None
    assign_fn: Optional[Callable] = None        # MATCH-vs-NEW decision (rubric.reuse)
    opening: Optional[Callable] = None          # frozen shared D1 table (rollout.doctor_opening; src/online_rl/doctor_openings.py)
    intervention_audit_fn: Optional[Callable] = None  # pre-rerun written-guide validity, separate from outcome KEEP
    bank_matcher: Optional[Callable] = None       # immutable cross-case KEEP-bank reuse; no new intervention


class TreeTrainer:
    def __init__(self, cfg: dict, *, out_dir: Path, cases: list[dict], env: TreeEnv, strategies: Strategies, search_backend: SearchBackend,
                 initial_prompt: str, policy_sha: Callable[[], str], protocol_sha: str, feedback_prompt_sha256: str, updater=None,
                 evaluator: Optional[Callable] = None, execution: Optional[dict] = None, before_case: Optional[Callable] = None):
        if active_unresolved(cfg): raise ValueError(f'unresolved settings: {active_unresolved(cfg)}')
        if cfg['rubric']['gold_answer_visible']: raise NotImplementedError('a gold-visible Rubric Generator contract is not defined')
        assert_not_protected(cases, 'tree training cases')
        self.cfg, self.out, self.cases, self.env, self.st = cfg, Path(out_dir), {c['case_id']: c for c in cases}, env, strategies
        self.backend, self.policy_sha, self.protocol_sha, self.fb_sha, self.updater = search_backend, policy_sha, protocol_sha, feedback_prompt_sha256, updater
        self.every = (cfg.get('evaluation') or {}).get('checkpoint_every_cases'); self.evaluator = evaluator
        self.rubric_on = cfg['rubric'].get('enabled', True); self.fb_model = (cfg.get('feedback') or {}).get('model')
        self.assignment_protocol = cfg['rubric'].get('assignment_protocol')
        if self.assignment_protocol not in (None, RB.EXACT_SOURCE_CASE_PROTOCOL):
            raise ValueError(f'unknown rubric.assignment_protocol {self.assignment_protocol!r}')
        self.exact_source_case = self.assignment_protocol == RB.EXACT_SOURCE_CASE_PROTOCOL
        self.reuse = cfg['rubric'].get('reuse') == RB.ASSIGNMENT_VERSION or self.exact_source_case
        self.cumulative_bank = cfg['rubric'].get('cumulative_bank') or None
        from src.online_rl.tree.member_intervention import MEMBER_INTERVENTION_VERSION, EXACT_SOURCE_CASE_VERSION
        self.member_intervention_version = ((EXACT_SOURCE_CASE_VERSION if self.exact_source_case else MEMBER_INTERVENTION_VERSION) if self.rubric_on and
            cfg['rubric'].get('generation', {}).get('output_contract') == 'trajectory_analysis_candidate_v2_with_intervention' else None)
        if self.member_intervention_version and self.reuse and not self.exact_source_case and getattr(env.assign_fn, 'member_intervention_version', None) != self.member_intervention_version:
            raise ValueError('joint rubric reuse requires a matcher that produces per-trajectory intervention instructions')
        ex = cfg.get('execution') or {}                                # execution only: same values, more threads
        self.doctor_workers, self.rerun_workers = int(ex.get('doctor_workers', 1)), int(ex.get('rerun_workers', 1))
        rx = execution or {}                                           # runtime-only concurrency (not in the config; same values)
        self.doctor_workers = int(rx.get('doctor_workers', self.doctor_workers))
        self.agent_workers, self.api_workers, self.member_workers = (int(rx.get(k, 1)) for k in ('agent_workers', 'api_workers', 'member_workers'))
        self.before_case = before_case                                 # e.g. Agent replica sync + hash barrier
        if cfg['rubric'].get('reuse') not in (None, RB.ASSIGNMENT_VERSION): raise ValueError(f"unknown rubric.reuse {cfg['rubric']['reuse']!r}")
        if self.reuse and not self.exact_source_case and self.rubric_on and getattr(env, 'assign_fn', None) is None: raise ValueError('rubric.reuse needs env.assign_fn')
        if self.exact_source_case and self.rubric_on and (not self.member_intervention_version or not callable(env.gate_fn)):
            raise ValueError('exact source-case assignment requires joint generated guides and Gate')
        if not self.rubric_on and cfg['prompt_update']['enabled']: raise ValueError('prompt evolution needs the rubric pipeline (rubric.enabled)')
        self.verification = (cfg.get('verification') or {}).get('mode', 'exhaustive_members_v1')     # training-time rubric verification
        from src.online_rl.tree.instruction_audit import PROTOCOL as INSTRUCTION_AUDIT_PROTOCOL
        audit_protocol = cfg.get('judges', {}).get('intervention_audit_protocol')
        if audit_protocol is not None:
            if audit_protocol != INSTRUCTION_AUDIT_PROTOCOL:
                raise ValueError('Unknown intervention audit protocol')
            if (not self.rubric_on or not self.member_intervention_version or self.verification not in W.PROTOCOLS
                    or not callable(env.intervention_audit_fn)
                    or getattr(env.intervention_audit_fn, 'settings', {}).get('protocol') != audit_protocol):
                raise ValueError('Instruction audit requires a bound versioned auditor and joint bounded witness verification')
        elif env.intervention_audit_fn is not None:
            raise ValueError('An intervention auditor is bound without a pinned config protocol')
        if self.verification not in ('exhaustive_members_v1', *W.PROTOCOLS): raise ValueError(f'unknown verification.mode {self.verification!r}')
        if self.exact_source_case and self.rubric_on and self.verification not in W.PROTOCOLS:
            raise ValueError('exact source-case assignment requires bounded witness verification')
        if self.verification == W.ALL_MEMBERS_PROTOCOL and cfg['verification'].get('max_valid_supports') is not None:
            raise ValueError('all_members_witness_v1 has no max_valid_supports cap')
        if self.verification == W.RANDOM10_PROTOCOL and (cfg['verification'].get('max_valid_supports') != 10 or
                cfg['verification'].get('support_sampling') != W.RANDOM10_SAMPLING):
            raise ValueError('random10_witness_v1 requires a uniform random sample of 10 same-ID trajectories')
        if self.verification == W.RANDOM5_PROTOCOL and (cfg['verification'].get('max_valid_supports') != 5 or
                cfg['verification'].get('support_sampling') != W.RANDOM5_SAMPLING):
            raise ValueError('random5_witness_v1 requires a uniform random sample of 5 same-ID trajectories')
        if self.verification in W.PROTOCOLS and self.rubric_on:
            if not self.reuse: raise ValueError(f'{W.PROTOCOL} verifies consolidated rubrics: it needs rubric.reuse')
            if (cfg.get('agent_reward') or {}).get('r_a_weighting') != 'equal': raise ValueError(f'{W.PROTOCOL} needs agent_reward.r_a_weighting = equal')
            if not (cfg.get('feedback') or {}).get('behavioral_locus'): raise ValueError(f'{W.PROTOCOL} needs feedback.behavioral_locus (target_reply_id)')
            if getattr(env.scorer, 'applicability', None) is None: raise ValueError(f'{W.PROTOCOL} needs a scorer with a Stage-1 applicability call')
        if self.verification not in W.PROTOCOLS and (cfg.get('agent_reward') or {}).get('r_a_weighting', 'effectiveness') != 'effectiveness':
            raise ValueError('equal R_A weighting is defined only with bounded witness verification')
        if self.cumulative_bank:
            from src.online_rl.tree import cumulative_bank as CB
            if (self.cumulative_bank.get('protocol') != CB.VERSION or not self.rubric_on or not self.reuse
                    or self.verification not in W.PROTOCOLS or cfg['prompt_update']['enabled']
                    or not self.member_intervention_version or (not self.exact_source_case and not callable(env.bank_matcher))):
                raise ValueError('cumulative bank requires fixed P0, consolidated witness verification and a bank matcher')
        if (cfg.get('feedback') or {}).get('protocol') in ('behavior_observer_v4','behavior_observer_v5') and self.rubric_on and not self.reuse:
            raise ValueError('behavior-observer feedback (invalid feedback excluded from rubric mapping) needs rubric.reuse')
        sch = cfg['schedule']; self.calibration_only = set(sch.get('calibration_only_cursors') or [])   # e.g. pool positions 401-410
        if sch.get('positions') is not None and not set(sch['positions']) <= self.calibration_only:
            raise ValueError('schedule.positions may only replay the calibration-only positions (schedule.calibration_only_cursors)')
        self.source_consistency = (cfg['rubric'].get('source_consistency') == 'source_locus_v1')   # v2.1
        from src.online_rl.tree.agent_state import SCHEMA
        if cfg['rubric'].get('schema') not in (None, SCHEMA): raise ValueError('unknown rubric.schema')
        if cfg['rubric'].get('source_consistency') not in (None, 'source_locus_v1'): raise ValueError(f"unknown rubric.source_consistency {cfg['rubric']['source_consistency']!r}")
        if self.source_consistency and self.rubric_on and (self.verification not in W.PROTOCOLS or not (cfg.get('feedback') or {}).get('behavioral_locus')
                                                           or getattr(env.generate_fn, 'repair', None) is None):
            raise ValueError('rubric.source_consistency needs bounded witness verification, the feedback locus and a generator with repair')
        if self.exact_source_case and self.rubric_on and not self.source_consistency:
            raise ValueError('exact source-case assignment requires source_locus_v1 checking for every valid candidate')
        import os; self.parallel_strata = os.environ.get('SHERPA_PARALLEL_STRATA') == '1'     # execution only: identical banks
        self.opening_spec = (cfg.get('rollout') or {}).get('doctor_opening')
        if self.opening_spec:                                          # never a silent fallback to generating D1
            op = getattr(env, 'opening', None)
            if op is None: raise ValueError('rollout.doctor_opening is set but no frozen opening table is bound (env.opening)')
            if getattr(op, 'sha256', None) != self.opening_spec['train_table_sha256']: raise ValueError('env.opening is not the pinned training table')
        elif getattr(env, 'opening', None) is not None: raise ValueError('env.opening is bound but rollout.doctor_opening is not set')
        if self.every is not None and (not isinstance(self.every, int) or self.every < 1 or evaluator is None or updater is None):
            raise ValueError('evaluation.checkpoint_every_cases needs a positive integer, an evaluator and an updater')
        self.out.mkdir(parents=True, exist_ok=True)
        self.store = PromptStore(self.out / 'prompts'); self.cache = CallCache(self.out / 'calls.jsonl')
        self.config_sha = sha256_json(cfg); self.state_path = self.out / 'run_state.json'
        self.state = self._resume() if self.state_path.exists() else self._start(initial_prompt)

    def _case_path(self, i: int) -> Path: return self.out / 'cases' / f'case_{i:06d}.json'

    def _event_path(self, e: int) -> Path: return self.out / 'prompt_events' / f'event_{e:04d}.json'

    def _write(self, s: TreeRunState) -> None: atomic_write_json(self.state_path, to_dict(s)); self.state = s

    def _start(self, prompt: str) -> TreeRunState:
        if any((self.out / d).exists() for d in ('cases', 'prompt_events')): raise RuntimeError(f'{self.out} has run artifacts but no run_state.json')
        sha = self.store.put(prompt)
        if self.cfg['rubric']['prompt_sha256'] != sha: raise RuntimeError('initial Rubric-Generator prompt does not match rubric.prompt_sha256')
        ckpt = self._save_checkpoint(0, milestone=True) if self.updater is not None else None
        pol = self.cfg.get('policy') or {}
        atomic_write_json(self.out / 'run_manifest.json', {'version': VERSION, 'config': self.cfg, 'config_sha256': self.config_sha,
                                                           'rubric_only_na_policy': RUBRIC_ONLY_NA_POLICY if self._rubric_only_reward() else None,
                                                           'rerun_outcome_policy': RERUN_OUTCOME_POLICY if self.rubric_on else None,
                                                           'rerun_continuation_protocol': RERUN_CONTINUATION_PROTOCOL if self.rubric_on else None,
                                                           'protocol_sha256': self.protocol_sha, 'initial_prompt_sha256': sha, 'initial_policy_sha256': self.policy_sha(),
                                                           'feedback_failure_policy': feedback_failure_policy(self.cfg),
                                                           'lora_init_seed': (pol.get('lora') or {}).get('init_seed'), 'init_adapter': pol.get('init_adapter'),
                                                           'initial_checkpoint': ckpt, 'initial_checkpoint_sha256': dir_sha256(Path(ckpt)) if ckpt else None,
                                                           'feedback_model': self.fb_model, 'doctor_model': self.cfg['rollout'].get('doctor_model'),
                                                           'model_roles': model_roles(self.cfg),
                                                           'doctor_backend': self.cfg['rollout'].get('doctor_backend') or 'openrouter',
                                                           'doctor_protocol_sha256': getattr(self.env.doctor, 'protocol', None),
                                                           'halting': self.cfg['rollout'].get('halting'),
                                                           'doctor_runtime_endpoints': getattr(getattr(self.env.doctor, 'doctor', None), 'endpoints', None),   # provenance only
                                                           'missing_strategies': self.st.missing(), 'doctor_api_seeded': bool(self.cfg['rollout'].get('doctor_backend')),
                                                           'doctor_opening': self._opening_identity()})
        s = TreeRunState(self.config_sha, self.cfg['schedule']['seed'], 0, self.cfg['rubric']['prompt_id'], sha, self.policy_sha(), ckpt, 0, 0,
                         ({'prompt_id': self.cfg['rubric']['prompt_id'], 'sha256': sha, 'event': None, 'case_cursor': 0},))
        self._write(s); return s

    def _opening_identity(self) -> Optional[dict]:
        if not self.opening_spec: return None
        from src.online_rl.doctor_openings import protocol_identity
        return {**protocol_identity(self.opening_spec), 'train_table': self.opening_spec['train_table'], 'eval_table': self.opening_spec['eval_table']}

    def _resume(self) -> TreeRunState:
        s = from_dict(TreeRunState, read_json(self.state_path))
        if s.config_sha256 != self.config_sha: raise RuntimeError('config changed since the run started; refusing to resume')
        man = read_json(self.out / 'run_manifest.json') if (self.out / 'run_manifest.json').exists() else {}
        if man.get('feedback_failure_policy') != feedback_failure_policy(self.cfg):
            raise RuntimeError('Doctor-reflection failure policy changed since the run started; use a fresh output directory')
        if self._rubric_only_reward() and man.get('rubric_only_na_policy') != RUBRIC_ONLY_NA_POLICY:
            raise RuntimeError('rubric-only NA reward policy differs from this run; use a fresh output directory')
        if self.rubric_on and man.get('rerun_outcome_policy') != RERUN_OUTCOME_POLICY:
            raise RuntimeError('rerun outcome policy differs from this run; use a fresh output directory')
        if self.rubric_on and man.get('rerun_continuation_protocol') != RERUN_CONTINUATION_PROTOCOL:
            raise RuntimeError('rerun continuation protocol differs from this run; use a fresh output directory')
        if man.get('protocol_sha256') != self.protocol_sha:
            raise RuntimeError('protocol changed since the run started; refusing to resume')
        if man.get('initial_checkpoint_sha256') and dir_sha256(Path(man['initial_checkpoint'])) != man['initial_checkpoint_sha256']:
            raise RuntimeError('checkpoint_000000 changed since the run started; refusing to resume')
        if s.checkpoint and self.updater is not None: self.updater.load(s.checkpoint)
        if self.policy_sha() != s.policy_sha256: raise RuntimeError('policy hash differs from the committed one')
        if man.get('doctor_opening') != self._opening_identity():
            raise RuntimeError('Doctor-opening protocol (mode / table SHA256) differs from the run manifest; refusing to resume')
        self.store.get(s.prompt_sha256)
        self._prune(s)
        stale = [p for p in sorted((self.out / 'cases').glob('case_*.json')) if int(p.stem[5:]) >= s.case_cursor]
        stale += [p for p in sorted((self.out / 'prompt_events').glob('event_*.json')) if int(p.stem[6:]) > s.lineage_cursor]
        if stale:
            dest = self.out / 'recovered' / f'resume_{len(list((self.out / "recovered").glob("resume_*")))}'; dest.mkdir(parents=True)
            for p in stale: p.rename(dest / f'{p.parent.name}__{p.name}')
        return s

    def _rubric_only_reward(self) -> bool:
        ar = self.cfg.get('agent_reward') or {}
        return ar.get('w_outcome') == 0.0 and ar.get('w_rubric', 0.0) > 0.0

    def total_cases(self) -> int:
        pos = self.cfg['schedule'].get('positions')
        return len(pos) if pos else self.cfg['schedule']['cases']

    def _position(self, cursor: int) -> int:
        """Pool position of the cursor-th case: the cursor itself, or schedule.positions[cursor] (calibration runs replay fixed
        positions of the main pool permutation, so position p is exactly the case a main run sees at cursor p)."""
        pos = self.cfg['schedule'].get('positions')
        return pos[cursor] if pos else cursor

    def run(self, max_cases: Optional[int] = None) -> None:
        done = 0
        while True:
            self._maybe_eval()                                       # a pending milestone is evaluated before anything else
            self._maybe_prompt_event()
            if self.state.case_cursor >= self.total_cases() or (max_cases is not None and done >= max_cases): break
            self.step(); done += 1

    def step(self, *, frozen_record=None, prompt_text=None, prompt_id=None, score_responses=True, commit=True,
             apply_update=True, defer_verification=False, cumulative_snapshot=None, online_working_bank=False, rollout_only=None, stream_bank=None, rollout_ready=None) -> dict:
        # rollout_only: 'with_feedback' (tree + Doctor reflection for the offline bank build) or 'no_feedback' (tree only, frozen bank)
        cumulative_reuse = []; chunk_summary = None
        if getattr(self, 'cumulative_bank', None):
            from src.online_rl.tree import cumulative_bank as CB
            if (cumulative_snapshot is None or commit or apply_update or score_responses or
                    (not defer_verification and not online_working_bank and not rollout_only)):
                raise ValueError('cumulative bank preparation requires its snapshot and an explicit verification mode')
            CB.validate_snapshot(cumulative_snapshot)
        elif cumulative_snapshot is not None:
            raise ValueError('cumulative snapshot supplied without cumulative bank configuration')
        if commit and (prompt_text is not None or frozen_record is not None or not score_responses or not apply_update):
            raise ValueError('Batch assessment requires commit=False; the batch coordinator owns inheritance')
        if online_working_bank and (defer_verification or self.exact_source_case or not self.rubric_on):
            raise ValueError('online working bank requires semantic assignment and immediate verification')
        if stream_bank is not None and (not online_working_bank or frozen_record is not None
                or rollout_only or not self.cumulative_bank.get('trajectory_streaming')
                or (self.cfg.get('five_gpu') or {}).get('execution_mode') != 'shared_ready_tasks_v1'):
            raise ValueError('Streaming requires a fresh online chunk rollout with its configured shared pool')
        if defer_verification and (not self.rubric_on or self.verification not in W.PROTOCOLS):
            raise ValueError('deferred verification requires a rubric-enabled witness protocol')
        t_case = time.time(); rollout_seconds = 0.0
        s, cfg = self.state, self.cfg; ro, iv = cfg['rollout'], cfg['intervention']
        if not self.cfg['schedule'].get('positions') and s.case_cursor in self.calibration_only:
            raise RuntimeError(f'pool position {s.case_cursor + 1} is calibration-only (data/splits/calibration_only_v1.json); a training run must not reach it')
        cid = schedule(sorted(self.cases), s.run_seed, self._position(s.case_cursor), 1)[0]; case = self.cases[cid]; e = self.env
        if self.before_case is not None: self.before_case(s.policy_sha256)
        if prompt_text is not None:
            s = dataclasses.replace(s, prompt_sha256=self.store.put(prompt_text), prompt_id=prompt_id or s.prompt_id)
        early_feedback = None
        streamed_built = None
        if frozen_record is not None:
            tree = from_dict(Tree, frozen_record['tree'])
            if tree.case_id != cid or tree.policy_sha256 != s.policy_sha256:
                raise RuntimeError('Frozen tree case/policy differs from the current case/policy')
        else:
            rollout_started = time.perf_counter()
            def rollout(on_leaf=None,on_leaf_nodes=None):
                return build_tree(case, max_agent_turns=ro['rounds'], branching=ro['branching'], run_seed=s.run_seed, policy_sha256=s.policy_sha256,
                          protocol_sha256=self.protocol_sha, agent=e.agent, doctor=e.doctor, stop_rule=e.stop_rule, final_option=e.final_option,
                          outcome_fn=e.outcome_fn, cache=self.cache, behavior_logprobs=e.behavior_logprobs, workers=self.doctor_workers,
                          opening=getattr(e, 'opening', None), agent_workers=self.agent_workers,on_leaf=on_leaf,on_leaf_nodes=on_leaf_nodes,
                          repeat_agent_reply_after_turns=ro.get('halting',{}).get('repeat_agent_reply_after_turns'), budget=ro.get('budget', 'full_tree'))
            if self.rubric_on and rollout_only != 'no_feedback' and (cfg.get('five_gpu') or {}).get('execution_mode') == 'shared_ready_tasks_v1':
                from concurrent.futures import ThreadPoolExecutor
                from threading import Lock
                futures = {}; lock = Lock()
                from contextlib import nullcontext
                from src.online_rl.tree.trajectory_stream import StreamingCase
                stream_context=(StreamingCase(self,case,s,self.store.get(s.prompt_sha256),cumulative_snapshot,stream_bank)
                                if stream_bank is not None else nullcontext(None))
                with stream_context as stream:
                    with ThreadPoolExecutor(max_workers=self.api_workers,thread_name_prefix='leaf-feedback') as pool:
                        def feedback_ready(leaf,path_nodes):
                            feedback=doctor_feedback(case,leaf,e.feedback_fn,self.fb_sha,self.cache,self.fb_model)
                            if stream is not None and feedback.valid:
                                stream.push(leaf,feedback,path_nodes)
                            return feedback
                        def on_leaf(leaf,path_nodes=None):
                            if leaf.outcome is None:return
                            with lock:
                                if leaf.leaf_id in futures:raise RuntimeError('Duplicate completed leaf')
                                futures[leaf.leaf_id]=pool.submit(feedback_ready,leaf,path_nodes)
                        tree = rollout(on_leaf if stream is None else None,on_leaf if stream is not None else None)
                        rollout_seconds = time.perf_counter() - rollout_started
                        early_feedback = [futures[l.leaf_id].result() for l in tree.leaves if l.outcome is not None]
                    if stream is not None:
                        if rollout_ready is not None: rollout_ready()
                        score_tree = exclude_reflection_failure_trajectories(tree, early_feedback)
                        eligible_nodes = {node_id for leaf in score_tree.leaves for node_id in leaf.path}
                        score_tree = dataclasses.replace(score_tree, nodes=tuple(
                            node for node in score_tree.nodes if node.node_id in eligible_nodes))
                        streamed_built=stream.finish(score_tree)
            else:
                tree = rollout()
                rollout_seconds = time.perf_counter() - rollout_started
        decided = [l for l in tree.leaves if l.outcome is not None]
        feedback, tins, rubrics, source_scores, effs, selection, scores, weights, bank = [], [], [], [], [], [], [], {}, None
        invalid_fb = []
        training_tree = tree
        scoring_nodes = list(tree.nodes)
        if rollout_only == 'no_feedback':                            # frozen bank: the tree only; scoring uses the immutable bank
            bank = {'assignments': [], 'witness': [], 'telemetry': {}, 'member_intervention_version': self.member_intervention_version,
                    'verification_pending': False, 'rollout_only_record': rollout_only}
        if self.rubric_on and rollout_only != 'no_feedback':         # Outcome-only arm: no feedback, rubric, verification or scoring
            fb = lambda l: doctor_feedback(case, l, e.feedback_fn, self.fb_sha, self.cache, self.fb_model)
            if frozen_record is not None:
                feedback = [from_dict(DoctorFeedback, f) for f in frozen_record['feedback']]
                if [f.leaf_id for f in feedback] != [l.leaf_id for l in decided]:
                    raise RuntimeError('Frozen feedback does not match the decided tree leaves')
            elif early_feedback is not None:feedback = early_feedback
            elif self.api_workers > 1:                                 # independent per trajectory; collected in leaf order
                from concurrent.futures import ThreadPoolExecutor
                with ThreadPoolExecutor(self.api_workers) as pool: feedback = list(pool.map(fb, decided))
            else: feedback = [fb(l) for l in decided]
            invalid_fb = [f.leaf_id for f in feedback if not getattr(f, 'valid', True)]    # leakage or reflection validation failed
            training_tree = exclude_reflection_failure_trajectories(tree, feedback)
            excluded_reflection_ids = reflection_failure_ids(feedback)
            eligible_node_ids = {node_id for leaf in tree.leaves
                                 if leaf.leaf_id not in excluded_reflection_ids for node_id in leaf.path}
            scoring_nodes = [node for node in tree.nodes if node.node_id in eligible_node_ids]
            ruled = [l for l, f in zip(decided, feedback) if getattr(f, 'valid', True)]       # trajectories that get a rubric mapping
            tins = [freeze_trajectory_input(case, l, f, include_locus=self.source_consistency) for l, f in zip(decided, feedback) if getattr(f, 'valid', True)]
            prompt = self.store.get(s.prompt_sha256)
            nodes = {n.node_id: n for n in tree.nodes}
            if rollout_only:                                         # offline bank build: rollout + reflection now, chunk transactions later
                bank = {'assignments': [], 'witness': [], 'telemetry': {}, 'member_intervention_version': self.member_intervention_version,
                        'verification_pending': False, 'rollout_only_record': rollout_only}
            elif self.reuse and self.verification in W.PROTOCOLS:     # consolidation + bounded causal-witness verification (witness.py)
                tin_of = {t.leaf_id: t for t in tins}
                online_decisions = []
                if online_working_bank:
                    working = cumulative_snapshot; all_canon = {}; all_assignments = []; all_witness = []; all_wit_objs = {}
                    all_selection = []
                    from src.online_rl.tree import offline_chunk_bank as OCB
                    if self.cumulative_bank.get('working_bank_protocol') == OCB.VERSION:   # one generator call per chunk of <= 8
                        built = streamed_built if streamed_built is not None else OCB.process_case(self, case=case, state=s, tree=tree, feedback=feedback, ruled=ruled, tin_of=tin_of,
                                                 prompt_text=prompt, snapshot=cumulative_snapshot)
                        working, all_canon, all_assignments = built['working'], built['canon'], built['assignments']
                        all_witness, all_wit_objs, all_selection = built['witness'], built['witness_objects'], built['selection']
                        cumulative_reuse, online_decisions, chunk_summary = built['cumulative_reuse'], built['decisions'], built['summary']
                    for leaf in ([] if chunk_summary is not None else sorted(ruled, key=lambda x: x.leaf_id)):
                        tin = tin_of[leaf.leaf_id]; input_sha = working['sha256']
                        match = CB.match_existing(tin, working, e.bank_matcher, self.cache,
                                                  top_k=int(self.cumulative_bank.get('retrieval_top_k', 8)))
                        if match is not None:
                            cumulative_reuse.append({'leaf_id': tin.leaf_id, **match})
                            online_decisions.append({'case_cursor': s.case_cursor, 'case_id': cid, 'leaf_id': tin.leaf_id,
                                'action': 'REUSED_EXISTING', 'input_bank_sha256': input_sha,
                                'output_bank_sha256': input_sha, 'bank_id': match['bank_id'], 'rubric_id': None})
                            continue
                        ssf_one = (lambda target: W.source_stage1_for(case, nodes, target,
                            tin_of[target.leaf_id].target_reply_id, e.scorer.applicability, self.cache)) if self.source_consistency else None
                        canon, assignments = RB.build_banks(case, [leaf], tin_of, prompt, s.prompt_sha256,
                            e.generate_fn, e.assign_fn, e.gate_fn, self.cache, source_stage1_for=ssf_one)
                        wit = self._verify_bounded(case, tree, feedback, [leaf], canon, assignments)
                        chosen = self._witness_selection(canon, wit)
                        all_canon.update(canon); all_assignments.extend(assignments)
                        all_witness.extend(to_dict(w) for w in wit.values()); all_wit_objs.update(wit); all_selection.extend(chosen)
                        kept = [d for d in chosen if d.retained]
                        if kept:
                            partial = {'case_id': cid, 'case_cursor': s.case_cursor, 'policy_sha256': s.policy_sha256,
                                'prompt_sha256': s.prompt_sha256, 'trajectory_rubrics': [to_dict(r) for r in canon.values()],
                                'selection': [to_dict(d) for d in chosen], 'rubric_bank': {'witness': [to_dict(w) for w in wit.values()]}}
                            working = CB.extend_snapshot(working, [partial])
                            action = 'GENERATED_KEEP'
                        else:
                            action = 'GENERATED_REMOVE'
                        rid = next(iter(canon)) if canon else None
                        online_decisions.append({'case_cursor': s.case_cursor, 'case_id': cid, 'leaf_id': tin.leaf_id,
                            'action': action, 'input_bank_sha256': input_sha, 'output_bank_sha256': working['sha256'],
                            'bank_id': ('bank_' + CB.content_hash(canon[rid].rubric)) if kept else None, 'rubric_id': rid})
                    rubrics = list(all_canon.values()); assignments_raw = [to_dict(a) for a in all_assignments]
                    selection = all_selection
                    bank = {'assignments': assignments_raw, 'witness': all_witness,
                            'telemetry': {**RB.assignment_telemetry(all_canon, all_assignments),
                                **W.witness_telemetry(all_wit_objs)},
                            'member_intervention_version': self.member_intervention_version, 'verification_pending': False}
                    scores = []; weights = {d.rubric_id: 1.0 for d in selection if d.retained}
                elif cumulative_snapshot is not None and not self.exact_source_case:
                    def match_existing(tin):
                        return CB.match_existing(tin, cumulative_snapshot, e.bank_matcher, self.cache,
                                                 top_k=int(self.cumulative_bank.get('retrieval_top_k', 8)))
                    if self.api_workers > 1 and len(tins) > 1:
                        from concurrent.futures import ThreadPoolExecutor
                        from contextvars import copy_context
                        with ThreadPoolExecutor(self.api_workers) as pool:
                            futures = [pool.submit(copy_context().run, match_existing, tin) for tin in tins]
                            matches = [future.result() for future in futures]
                    else:
                        matches = [match_existing(tin) for tin in tins]
                    cumulative_reuse = [{'leaf_id': tin.leaf_id, **match} for tin, match in zip(tins, matches) if match is not None]
                    reused_ids = {item['leaf_id'] for item in cumulative_reuse}
                    ruled = [leaf for leaf in ruled if leaf.leaf_id not in reused_ids]
                if online_working_bank:
                    canon = all_canon; assignments = all_assignments; generated_candidates = None
                ssf = (lambda leaf: W.source_stage1_for(case, nodes, leaf, tin_of[leaf.leaf_id].target_reply_id, e.scorer.applicability, self.cache)
                       ) if self.source_consistency else None             # v2.1: source-self-applicability before the quality gate
                if online_working_bank:
                    pass
                elif self.exact_source_case:
                    canon, assignments, generated_candidates = RB.build_exact_generated_banks(
                        case, ruled, tin_of, prompt, s.prompt_sha256, e.generate_fn, e.gate_fn,
                        self.cache, source_stage1_for=ssf, parallel_strata=self.parallel_strata)
                else:
                    canon, assignments = RB.build_banks(case, ruled, tin_of, prompt, s.prompt_sha256, e.generate_fn,
                                                        e.assign_fn, e.gate_fn, self.cache, parallel_strata=self.parallel_strata, source_stage1_for=ssf)
                    generated_candidates = None
                rubrics = list(canon.values()); assignments_raw = [to_dict(a) for a in assignments]
                telemetry = {**RB.assignment_telemetry(canon, assignments, candidates=generated_candidates),
                             'protocol': self.verification,
                             'locus': dict(__import__('collections').Counter((f.locus or {}).get('status') for f in feedback)),
                             'feedback_invalid': invalid_fb,
                             'feedback_invalid_by_stratum': {st: sum(1 for f in feedback if not getattr(f, 'valid', True) and f.outcome == o)
                                                             for st, o in (('success', 1.0), ('failure', 0.0))},
                             'feedback_by_stratum': {st: sum(1 for f in feedback if f.outcome == o) for st, o in (('success', 1.0), ('failure', 0.0))},
                             'feedback_leakage_first_try': sum(bool(((f.locus or {}).get('leakage') or {}).get('first')) for f in feedback),
                             'source_check': dict(__import__('collections').Counter((r.source_check or {}).get('status') for r in (generated_candidates if generated_candidates is not None else canon.values())
                                                                                     if r.source_check is not None))}
                if not online_working_bank:
                    bank = {'assignments': assignments_raw, 'witness': [], 'telemetry': telemetry,
                            'member_intervention_version': self.member_intervention_version,
                            **({'assignment_protocol': RB.EXACT_SOURCE_CASE_PROTOCOL,
                                'generated_candidates': [to_dict(r) for r in generated_candidates]} if generated_candidates is not None else {}),
                            'verification_pending': bool(defer_verification)}
                if online_working_bank:
                    pass
                elif not defer_verification:
                    wit = self._verify_bounded(case, tree, feedback, ruled, canon, assignments)
                    kept = {rid for rid, w in wit.items() if w.decision == 'KEEP'}
                    selection = self._witness_selection(canon, wit)
                    bank['witness'] = [to_dict(w) for w in wit.values()]
                    bank['telemetry'].update(W.witness_telemetry(wit))
                    scores = score_nodes(case, scoring_nodes, [canon[c] for c in sorted(kept)], e.scorer, self.cache, workers=self.api_workers) if score_responses else []
                    weights = {rid: 1.0 for rid in sorted(kept)}                 # verification is a gate only
                else:
                    selection, scores, weights = [], [], {}
            elif self.reuse:                                         # same-reason REUSE vs NEW (src/online_rl/tree/rubric_bank.py)
                canon, assignments = RB.build_banks(case, ruled, {t.leaf_id: t for t in tins}, prompt, s.prompt_sha256, e.generate_fn,
                                                    e.assign_fn, e.gate_fn, self.cache)
                source_scores, member = RB.verify_members(case, nodes, {l.leaf_id: l for l in decided}, canon, assignments, scorer=e.scorer,
                                                          writer=e.writer, continuation=e.continuation, reruns=iv['reruns_per_intervention'],
                                                          run_seed=s.run_seed, policy_sha256=s.policy_sha256, strategies=self.st, cache=self.cache,
                                                          workers=self.rerun_workers, member_workers=self.member_workers)
                support = RB.canonical_support(canon, assignments, member, cfg['selection']['min_R_O'])
                rubrics, effs = list(canon.values()), [x for x in member.values() if x is not None]
                selection = [SelectionDecision(cid, canon[cid].leaf_id, sp.retained, 'existential_member_support',
                                               f'max member s = {sp.max_s} vs {cfg["selection"]["min_R_O"]} ({sp.supported_member_count}/{sp.member_count} members)')
                             for cid, sp in support.items()]
                kept = {cid for cid, sp in support.items() if sp.retained}
                scores = score_nodes(case, scoring_nodes, [canon[c] for c in sorted(kept)], e.scorer, self.cache, workers=self.api_workers) if score_responses else []
                weights = {cid: support[cid].weight for cid in sorted(kept)}             # R_O(R) = max_j s_j, once per rubric
                bank = {'assignments': [to_dict(a) for a in assignments], 'rubric_support': [to_dict(v) for v in support.values()],
                        'member_intervention_version': self.member_intervention_version,
                        'telemetry': RB.bank_telemetry(canon, assignments, support)}
            else:
                rubrics = [generate_rubric(prompt, s.prompt_sha256, t, e.generate_fn, self.cache, e.gate_fn) for t in tins]
                for leaf, rub in zip(decided, rubrics):
                    ss, eff = evaluate_rubric(case, nodes, leaf, rub, scorer=e.scorer, writer=e.writer, continuation=e.continuation,
                                              reruns=iv['reruns_per_intervention'], run_seed=s.run_seed, policy_sha256=s.policy_sha256,
                                              strategies=self.st, cache=self.cache)
                    source_scores += ss; effs += [eff] if eff is not None else []
                selection = select(effs, cfg['selection'])
                kept = {d.rubric_id for d in selection if d.retained}
                scores = score_nodes(case, scoring_nodes, [r for r in rubrics if r.rubric_id in kept], e.scorer, self.cache) if score_responses else []
                weights = {e.rubric_id: e.R_O for e in effs if e.rubric_id in kept}
        execution_failure_ids = scoring_execution_failure_ids(tree, scores)
        training_tree = exclude_trajectories(training_tree, execution_failure_ids)
        targets = agent_targets(training_tree, scores, weights, self.st)
        update = (self.st.need('agent_update')(training_tree, targets) if score_responses and apply_update
                  else {'optimizer_step': False, 'assessment_only': True})
        record = {'case_cursor': s.case_cursor, 'case_id': cid, 'prompt_id': s.prompt_id, 'prompt_sha256': s.prompt_sha256,
                  'policy_sha256': s.policy_sha256, 'counts': tree_counts(tree), 'tree': to_dict(tree), 'feedback': [to_dict(f) for f in feedback],
                  'feedback_invalid_trajectory_ids': invalid_fb,
                  'judge_error_trajectory_ids': sorted(execution_failure_ids),
                  'trajectory_inputs': [to_dict(t) for t in tins], 'trajectory_rubrics': [to_dict(r) for r in rubrics],
                  'source_path_scores': [to_dict(x) for x in source_scores], 'effectiveness': [to_dict(x) for x in effs],
                  'selection': [to_dict(d) for d in selection],
                  'response_scores': [to_dict(x) for x in scores], 'rubric_weights': weights, 'agent_targets': targets, 'update': update,
                  **({'rubric_bank': bank} if bank is not None else {}),
                  **({'cumulative_bank_input': cumulative_snapshot, 'cumulative_reuse': cumulative_reuse,
                      'cumulative_reused_trajectories': len(cumulative_reuse)} if cumulative_snapshot is not None else {}),
                  **({'cumulative_bank_output': working, 'online_working_bank_decisions': online_decisions}
                     if online_working_bank else {}),
                  **({'offline_chunk_bank': chunk_summary} if chunk_summary is not None else {}),
                  **({'early_cumulative_scoring': chunk_summary['early_cumulative_scoring']}
                     if chunk_summary is not None and chunk_summary.get('early_cumulative_scoring') else {}),
                  'group_diagnostics': sibling_group_diagnostics(tree, targets, update if isinstance(update, dict) else None),
                  'interaction_diagnostics': interaction_diagnostics(tree, targets if isinstance(targets, dict) else None),
                  'seconds': time.time() - t_case,
                  'timings': {'rollout_wall_seconds': rollout_seconds}}
        if commit:
            atomic_write_json(self._case_path(s.case_cursor), record)
            self._commit()
        return record

    def _verify_bounded(self, case, tree, feedback, ruled, canon, assignments):
        nodes = {n.node_id: n for n in tree.nodes}
        loci = {f.leaf_id: f.target_reply_id for f in feedback if getattr(f, 'valid', True)}
        vcfg = self.cfg['verification']
        audit = None
        if self.env.intervention_audit_fn is not None:
            from src.online_rl.tree.instruction_audit import cached_instruction_auditor
            audit = cached_instruction_auditor(self.env.intervention_audit_fn, self.cache, tree.policy_sha256)
            if (self.cfg.get('judges') or {}).get('execution_retry_attempts'):
                # The auditor already repeated an execution-invalid response locally. If its budget is exhausted the member has
                # no audit verdict (EXECUTION_UNRESOLVED): no evidence for or against; the witness rule itself is unchanged.
                from src.online_rl.tree.instruction_audit import execution_unresolved_auditor
                from src.online_rl.tree.applicability import log_execution_unresolved
                budget, cache, case_id = self.cfg['judges']['execution_retry_attempts'], self.cache, case['case_id']
                audit = execution_unresolved_auditor(audit, budget, lambda rubric, member_spec, exc: log_execution_unresolved(
                    cache, stage='intervention_audit', case_id=case_id, rubric_id=(rubric or {}).get('rubric_id'),
                    node_id=(member_spec or {}).get('leaf_id') if isinstance(member_spec, dict) else None,
                    attempts=budget, reasons=[str(exc)[:200]]))
        return W.verify_witness(case, nodes, {l.leaf_id: l for l in ruled}, canon, assignments, loci,
                                applicability=self.env.scorer.applicability, writer=self.env.writer,
                                continuation=self.env.continuation,
                                reruns=self.cfg['intervention']['reruns_per_intervention'],
                                run_seed=self.state.run_seed, policy_sha256=tree.policy_sha256,
                                cache=self.cache, max_valid_supports=vcfg['max_valid_supports'],
                                workers=self.member_workers,
                                support_sampling=vcfg.get('support_sampling', 'topology'),
                                intervention_audit=audit)

    @staticmethod
    def _witness_selection(canon, wit):
        return [SelectionDecision(cid, canon[cid].leaf_id, w.decision == 'KEEP', w.protocol,
                                  f'{w.decision} ({w.stop_reason}); validated_by={w.validated_by_trajectory_id}; '
                                  f'valid_supports={w.num_valid_supports}; reruns={w.num_reruns_executed}')
                for cid, w in wit.items()]

    def validate_member_interventions(self, record: dict) -> None:
        """Never resume a joint bank/phase with creator-only or missing member instructions."""
        if not self.member_intervention_version or not self.reuse: return
        from src.online_rl.tree.member_intervention import resolve_member_intervention
        bank = record.get('rubric_bank') or {}
        if bank.get('rollout_only_record'): return                    # tree (+ reflection) only: no assignment exists to validate
        if bank.get('member_intervention_version') != self.member_intervention_version:
            raise RuntimeError('saved rubric bank predates per-trajectory intervention instructions; regenerate the bank from the frozen tree')
        tree = from_dict(Tree, record['tree'])
        leaves = {leaf.leaf_id: leaf for leaf in tree.leaves}
        nodes = {node.node_id: node for node in tree.nodes}
        canon = {raw['rubric_id']: from_dict(TrajectoryRubric, raw) for raw in record['trajectory_rubrics']}
        decided = {leaf.leaf_id for leaf in tree.leaves if leaf.outcome is not None}
        feedback = record.get('feedback', [])
        feedback_ids = [f['leaf_id'] for f in feedback]
        if len(feedback_ids) != len(set(feedback_ids)) or set(feedback_ids) != decided:
            raise RuntimeError('saved joint bank feedback does not cover every decided trajectory exactly once')
        eligible = {f['leaf_id'] for f in feedback if f.get('valid', True)}
        if getattr(self, 'cumulative_bank', None):
            from src.online_rl.tree import cumulative_bank as CB
            snapshot = record.get('cumulative_bank_input')
            CB.validate_snapshot(snapshot)
            inputs = {raw['leaf_id']: from_dict(TrajectoryRubricInput, raw) for raw in record['trajectory_inputs']}
            if len(inputs) != len(record['trajectory_inputs']) or set(inputs) != eligible:
                raise RuntimeError('saved cumulative bank inputs do not cover eligible trajectories exactly once')
            reused = record.get('cumulative_reuse')
            if not isinstance(reused, list):
                raise RuntimeError('saved cumulative bank lacks reuse decisions')
            ids = [row['leaf_id'] for row in reused]
            if len(ids) != len(set(ids)) or not set(ids) <= eligible:
                raise RuntimeError('saved cumulative reuse has duplicate or ineligible trajectories')
            for row in reused:
                if row.get('chunk_protocol') and row.get('snapshot_sha256') != snapshot['sha256']:   # a later chunk saw an updated bank
                    if row['snapshot_sha256'] not in {c['bank_before_sha256'] for c in (record.get('offline_chunk_bank') or {}).get('chunks', [])}:
                        raise RuntimeError('saved chunk reuse names a bank that no chunk of this case consumed')
                    continue
                CB.validate_reuse(inputs[row['leaf_id']], snapshot, row)
            if record.get('cumulative_reused_trajectories') != len(ids):
                raise RuntimeError('saved cumulative reuse count differs from validated decisions')
            eligible -= set(ids)
            eligible -= set((record.get('offline_chunk_bank') or {}).get('uncovered_ids', []))   # beyond the 48 cap / unattached peers
        elif record.get('cumulative_bank_input') is not None or record.get('cumulative_reuse'):
            raise RuntimeError('saved cumulative bank cannot be used without its configured protocol')
        member_ids = [a['leaf_id'] for a in bank['assignments']]
        if len(member_ids) != len(set(member_ids)) or set(member_ids) != eligible:
            raise RuntimeError('saved joint bank assignments do not cover every eligible trajectory exactly once')
        if (len(canon) != len(record['trajectory_rubrics']) or
                {a['canonical_rubric_id'] for a in bank['assignments']} != set(canon)):
            raise RuntimeError('saved joint bank canonical rubrics and assignments differ')
        exact_candidates = None
        if getattr(self, 'exact_source_case', False):
            from src.online_rl.tree.cumulative_bank import content_hash
            if bank.get('assignment_protocol') != RB.EXACT_SOURCE_CASE_PROTOCOL:
                raise RuntimeError('saved bank is not the exact generated source-case protocol')
            raw_candidates = bank.get('generated_candidates')
            if not isinstance(raw_candidates, list):
                raise RuntimeError('saved exact source-case bank lacks generated candidates')
            exact_candidates = {raw['leaf_id']: from_dict(TrajectoryRubric, raw) for raw in raw_candidates}
            if (len(exact_candidates) != len(raw_candidates) or set(exact_candidates) != eligible or
                    len({raw['rubric_id'] for raw in raw_candidates}) != len(raw_candidates)):
                raise RuntimeError('saved generated candidates do not cover eligible trajectories exactly once')
            for candidate in exact_candidates.values():
                if candidate.valid:
                    source = candidate.source_check or {}
                    stage = source.get('second') if source.get('status') == 'REPAIRED_PASS' else source.get('first')
                    if ((candidate.gate or {}).get('passed') is not True or
                            source.get('status') not in ('PASS', 'REPAIRED_PASS') or
                            (stage or {}).get('applicable') is not True):
                        raise RuntimeError('saved generated candidate lacks its Gate/source proof')
        for raw in bank['assignments']:
            assignment = from_dict(RubricAssignment, raw)
            rubric = canon[assignment.canonical_rubric_id]
            leaf = leaves[assignment.leaf_id]
            if assignment.case_id != tree.case_id or assignment.stratum != ('success' if leaf.outcome == 1.0 else 'failure'):
                raise RuntimeError('saved joint assignment differs from its member case or outcome stratum')
            if exact_candidates is not None:
                candidate = exact_candidates[assignment.leaf_id]
                decision = assignment.decision or {}
                if decision.get('protocol') != RB.EXACT_SOURCE_CASE_PROTOCOL:
                    raise RuntimeError('saved assignment has another matching protocol')
                if assignment.assignment in ('CREATED_NEW', 'CREATE_REJECTED'):
                    if (candidate.rubric_id != assignment.canonical_rubric_id or
                            to_dict(candidate) != to_dict(rubric) or
                            decision.get('candidate_sha256') != sha256_json(to_dict(candidate))):
                        raise RuntimeError('saved generated source candidate differs from its canonical record')
                    if assignment.assignment == 'CREATED_NEW' and (not candidate.valid or
                            decision.get('content_sha256') != content_hash(candidate.rubric)):
                        raise RuntimeError('saved generated source candidate is not an exact valid criterion')
                    if assignment.assignment == 'CREATE_REJECTED' and (candidate.valid or assignment.intervention_spec is not None):
                        raise RuntimeError('saved rejected candidate has a valid score or member guide')
                elif assignment.assignment == RB.EXACT_DUPLICATE_ASSIGNMENT:
                    proof = decision.get('generated_candidate_proof') or {}
                    source_turn = (((candidate.rubric or {}).get('source_provenance') or {}).get('analysis') or {}).get('selected_agent_turn_id')
                    if (not candidate.valid or not rubric.valid or
                            content_hash(candidate.rubric) != content_hash(rubric.rubric) or
                            proof.get('candidate_rubric_id') != candidate.rubric_id or
                            proof.get('candidate_sha256') != sha256_json(to_dict(candidate)) or
                            proof.get('content_sha256') != content_hash(candidate.rubric) or
                            proof.get('selected_agent_turn_id') != source_turn or
                            proof.get('intervention_instruction_sha256') != sha256_json(candidate.intervention_instruction) or
                            decision.get('intervention_instruction') != candidate.intervention_instruction or
                            (assignment.intervention_spec or {}).get('origin') != 'generated_exact_duplicate'):
                        raise RuntimeError('saved exact duplicate guide or criterion differs from its generated source')
                else:
                    raise RuntimeError('saved exact source-case bank contains a semantic matching assignment')
            if assignment.assignment == 'CREATE_REJECTED' or not rubric.valid: continue
            if not rubric.intervention_instruction:
                raise RuntimeError('joint rubric record is missing its generated instruction')
            spec = resolve_member_intervention(rubric, leaf, member_spec=assignment.intervention_spec)
            resolve_member_intervention(rubric, leaf, member_spec=spec, node=nodes[leaf.path[spec['selected_agent_turn_id'] - 1]])

    def finish_deferred_verification(self, record: dict) -> dict:
        """Run a prepared case's witness checks after every case has generated its rubric bank."""
        bank = record.get('rubric_bank') or {}
        self.validate_member_interventions(record)
        if not bank.get('verification_pending'):
            if bank.get('witness') is not None and 'witness' in bank:
                return record
            raise ValueError('record has no deferred witness-verification payload')
        if self.verification not in W.PROTOCOLS:
            raise ValueError('deferred witness verification requires a supported witness protocol')
        tree = from_dict(Tree, record['tree'])
        if tree.policy_sha256 != self.policy_sha() or tree.case_id != record['case_id']:
            raise RuntimeError('prepared tree policy or case identity changed before verification')
        case = self.cases[tree.case_id]
        feedback = [from_dict(DoctorFeedback, f) for f in record['feedback']]
        ruled = [leaf for leaf in tree.leaves if leaf.outcome is not None and
                 any(f.leaf_id == leaf.leaf_id and getattr(f, 'valid', True) for f in feedback)]
        reused_ids = {row['leaf_id'] for row in record.get('cumulative_reuse', [])}
        ruled = [leaf for leaf in ruled if leaf.leaf_id not in reused_ids]
        canon = {r['rubric_id']: from_dict(TrajectoryRubric, r)
                 for r in record['trajectory_rubrics']}
        assignments = [from_dict(RubricAssignment, a)
                       for a in bank['assignments']]
        wit = self._verify_bounded(case, tree, feedback, ruled, canon, assignments)
        kept = {rid for rid, w in wit.items() if w.decision == 'KEEP'}
        bank['witness'] = [to_dict(w) for w in wit.values()]
        bank['verification_pending'] = False
        bank['telemetry'].update(W.witness_telemetry(wit))
        record['selection'] = [to_dict(x) for x in self._witness_selection(canon, wit)]
        record['rubric_weights'] = {rid: 1.0 for rid in sorted(kept)}
        record['response_scores'] = []
        record['rubric_bank'] = bank
        return record

    def score_cumulative(self, record: dict, snapshot: dict) -> dict:
        """Score one case against the batch coordinator's frozen cumulative KEEP set."""
        if not getattr(self, 'cumulative_bank', None):
            raise ValueError('cumulative scoring is not enabled')
        from src.online_rl.tree import cumulative_bank as CB
        from src.online_rl.tree.scoring import score_cumulative_nodes
        CB.validate_snapshot(snapshot)
        self.validate_member_interventions(record)
        if record['rubric_bank'].get('verification_pending'):
            raise RuntimeError('cumulative scoring cannot precede local verification')
        entries = {entry['bank_id']: entry for entry in snapshot['entries']}
        for previous in record['cumulative_bank_input']['entries']:
            current = entries.get(previous['bank_id'])
            if (current is None or current['rubric'] != previous['rubric'] or
                    any(proof not in current['sources'] for proof in previous['sources'])):
                raise RuntimeError('frozen scoring bank lost or changed prior validated criteria')
        local = {raw['rubric_id']: raw for raw in record['trajectory_rubrics']}
        for selection in record['selection']:
            if selection.get('retained') is True:
                rid = 'bank_' + CB.content_hash(local[selection['rubric_id']]['rubric'])
                if rid not in entries:
                    content_sha=CB.content_hash(local[selection['rubric_id']]['rubric'])
                    summary=record.get('offline_chunk_bank') or {}
                    alias=summary.get('semantic_aliases',{}).get(content_sha)
                    from src.online_rl.tree.bank_delta import validate_alias, VERSION as PARALLEL_VERSION
                    if not alias or summary.get('parallel_bank_protocol')!=PARALLEL_VERSION:
                        raise RuntimeError('frozen scoring bank omits a newly validated local criterion')
                    validate_alias(alias,content_sha,entries)
        tree = from_dict(Tree, record['tree'])
        if tree.policy_sha256 != self.policy_sha() or tree.case_id != record['case_id']:
            raise RuntimeError('cumulative scoring tree policy or case changed')
        case = self.cases[tree.case_id]
        scoring_scope = self.cumulative_bank.get('scoring_scope', 'top_k')
        prefetched_scores=None
        early=record.get('early_cumulative_scoring')
        if early:
            from src.online_rl.tree.trajectory_stream import EARLY_SCORE_VERSION, read as read_stream_json
            if (early.get('protocol')!=EARLY_SCORE_VERSION or Path(early.get('file','')).name!=early.get('file')):
                raise RuntimeError('Cumulative early score identity is invalid')
            early_path=Path(self.out)/early['file']
            payload=read_stream_json(early_path)
            if (sha256_json(payload)!=early.get('sha256') or payload.get('protocol')!=EARLY_SCORE_VERSION or
                    payload.get('case_id')!=record['case_id'] or payload.get('policy_sha256')!=tree.policy_sha256 or
                    payload.get('tree_sha256')!=sha256_json(record['tree']) or
                    payload.get('snapshot',{}).get('sha256')!=early.get('snapshot_sha256')):
                raise RuntimeError('Cumulative early score file failed identity validation')
            early_snapshot=CB.validate_snapshot(payload['snapshot'])
            final_entries={entry['bank_id']:entry for entry in snapshot['entries']}
            for entry in early_snapshot['entries']:
                current=final_entries.get(entry['bank_id'])
                if (current is None or current['rubric']!=entry['rubric'] or
                        any(proof not in current['sources'] for proof in entry['sources'])):
                    raise RuntimeError('Cumulative early score bank is not a preserved prefix of final scoring bank')
            if self.cumulative_bank.get('scoring_scope','top_k')!='all':
                raise RuntimeError('Cumulative early score reuse requires scoring_scope=all')
            prefetched_scores=payload['scores']
        scores, retrieval = score_cumulative_nodes(case, list(tree.nodes), snapshot, self.env.scorer, self.cache,
            top_k=int(self.cumulative_bank.get('retrieval_top_k', 8)), workers=self.api_workers,
            scoring_scope=scoring_scope,prefetched_scores=prefetched_scores)
        weights = {rubric.rubric_id: 1.0 for rubric in CB.scoring_rubrics(snapshot)}
        execution_failure_ids = scoring_execution_failure_ids(tree, scores)
        training_tree = exclude_trajectories(exclude_reflection_failure_trajectories(
            tree, [from_dict(DoctorFeedback, raw) for raw in record.get('feedback', [])]), execution_failure_ids)
        targets = agent_targets(training_tree, scores, weights, self.st)
        result = {**record, 'response_scores': [to_dict(row) for row in scores], 'rubric_weights': weights,
                  'judge_error_trajectory_ids': sorted(set(record.get('judge_error_trajectory_ids', ())) | execution_failure_ids),
                  'agent_targets': targets, 'update': {'optimizer_step': False, 'deferred_to_gpu4': True},
                  'cumulative_bank_scoring': {'snapshot': snapshot, 'snapshot_sha256': snapshot['sha256'],
                                              'retrieval': retrieval}, 'cumulative_retrieval': retrieval,
                  'group_diagnostics': sibling_group_diagnostics(tree, targets),
                  'interaction_diagnostics': interaction_diagnostics(tree, targets)}
        if scoring_scope == 'all':
            result['cumulative_bank_scoring']['scoring_scope'] = 'all'
        return result

    def _commit(self) -> None:
        """Save the checkpoint of the finished case, write run_state, then drop rolling checkpoints it no longer points to.
        Every `evaluation.checkpoint_every_cases` cases the checkpoint is a permanent milestone."""
        s = self.state; sha = self.policy_sha(); ckpt = s.checkpoint; n = s.case_cursor + 1
        milestone = bool(self.every) and n % self.every == 0
        if sha != s.policy_sha256 or milestone:
            if self.updater is None: raise RuntimeError('policy changed but no updater is available to checkpoint it')
            ckpt = self._save_checkpoint(n, milestone)
        self._write(dataclasses.replace(s, case_cursor=n, policy_sha256=sha, checkpoint=ckpt))
        self._prune(self.state)

    def _checkpoint_dir(self, n: int, milestone: bool) -> Path:
        return self.out / 'checkpoints' / f"{'checkpoint' if milestone else 'rolling'}_{n:06d}"

    def _save_checkpoint(self, n: int, milestone: bool) -> str:
        p = self._checkpoint_dir(n, milestone); tmp = p.with_name(p.name + '.tmp')
        for d in (p, tmp):
            if d.exists(): shutil.rmtree(d)                          # leftover of an uncommitted attempt at this case
        self.updater.save(tmp); tmp.rename(p); return str(p)

    def _prune(self, s: TreeRunState) -> None:
        """Remove rolling checkpoints other than the committed one, and anything written after the committed case."""
        for p in sorted((self.out / 'checkpoints').glob('*')) if (self.out / 'checkpoints').exists() else []:
            kind, _, idx = p.name.removesuffix('.tmp').partition('_')
            if p.name.endswith('.tmp') or int(idx) > s.case_cursor or (kind == 'rolling' and str(p) != s.checkpoint): shutil.rmtree(p)

    def _maybe_eval(self) -> None:
        """Held-out evaluation of the milestone checkpoint before the next case; read-only for training. A failure stops the
        run, and a resumed run retries it first."""
        s = self.state
        if not self.every or s.eval_cursor + self.every > s.case_cursor: return
        m = s.eval_cursor + self.every; ckpt = self._checkpoint_dir(m, True)
        if s.case_cursor != m or s.checkpoint != str(ckpt): raise RuntimeError(f'milestone {m} must be evaluated on its checkpoint before training continues')
        before = (self.policy_sha(), s.prompt_sha256)
        if self.before_case is not None: self.before_case(s.policy_sha256)          # replicas evaluate the milestone policy
        self.evaluator(milestone=m, checkpoint=ckpt, out_dir=self.out / 'eval' / ckpt.name,
                       provenance={'milestone': m, 'checkpoint': str(ckpt), 'checkpoint_sha256': dir_sha256(ckpt), 'policy_sha256': s.policy_sha256,
                                   'prompt_id': s.prompt_id, 'prompt_sha256': s.prompt_sha256, 'config_sha256': s.config_sha256})
        if (self.policy_sha(), self.state.prompt_sha256) != before: raise RuntimeError('evaluation changed the policy or the prompt')
        self._write(dataclasses.replace(s, eval_cursor=m))

    def _units(self) -> list[EvidenceUnit]:
        out = []
        for i in range(self.state.last_event_cursor, self.state.case_cursor):
            r = read_json(self._case_path(i)); tree = from_dict(Tree, r['tree']); nodes = {n.node_id: n for n in tree.nodes}
            for t in r['trajectory_inputs']:
                tin = from_dict(TrajectoryRubricInput, t); leaf = next(l for l in tree.leaves if l.leaf_id == tin.leaf_id)
                out.append(EvidenceUnit(r['case_id'], leaf, tuple(nodes[n] for n in leaf.path), tin, r['policy_sha256']))
        return out

    def _maybe_prompt_event(self) -> None:
        pu, s = self.cfg['prompt_update'], self.state
        if not pu['enabled'] or s.case_cursor >= self.total_cases() or s.case_cursor - s.last_event_cursor < pu['every_n_cases']: return
        units = self._units()
        if len(units) < pu['minimum_trajectories'] or not ready(units, s.run_seed, pu['acceptance_fraction']): return   # wait for both strata
        event = s.lineage_cursor + 1
        history = [{'event': d['event'], 'accepted': d['decision']['accepted'], 'reason': d['decision']['reason']}
                   for d in (read_json(self._event_path(e)) for e in range(1, event))]
        deps = ScoringDeps(pu['reruns_per_intervention'], s.run_seed, self.env.generate_fn,
                           self.env.gate_fn, self.env.scorer, self.env.writer, self.env.continuation, s.policy_sha256, self.st, self.cache,
                           self.env.assign_fn if self.reuse else None, self.source_consistency)
        res = run_event(event, s.prompt_id, s.prompt_sha256, units, self.cases, self.store, self.backend, deps, self.cfg, s.run_seed, s.case_cursor, history)
        atomic_write_json(self._event_path(event), {**res, 'versions': [to_dict(v) for v in res['versions']],
                                                    'rerun_policy_sha256': s.policy_sha256, 'evidence_policy_sha256': sorted({u.policy_sha256 for u in units}),
                                                    'policy_note': 'event reruns use the current policy, not the policy that produced the stored trajectories'})
        d = res['decision']; new = dataclasses.replace(s, lineage_cursor=event, last_event_cursor=s.case_cursor)
        if d['accepted']:
            new = dataclasses.replace(new, prompt_id=d['candidate_prompt_id'], prompt_sha256=d['candidate_sha256'],
                                      history=s.history + ({'prompt_id': d['candidate_prompt_id'], 'sha256': d['candidate_sha256'], 'event': event,
                                                            'case_cursor': s.case_cursor},))
        self._write(new)
