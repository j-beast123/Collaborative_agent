"""Case-tree trainer. Per case: branching rollout -> FINAL/outcome per leaf -> Doctor feedback -> ONE rubric per trajectory
-> match on the source trajectory -> Score-0/Score-2 intervention, K reruns per target, s(tau) -> KEEP/REMOVE -> kept rubrics
score every response of the tree (No Match kept) -> Agent targets/update (injected).
run_state.json is written last per case; every external call goes through the immutable call cache, so a resumed case
repeats no completed API or model work."""
from __future__ import annotations
import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional
from src.online_rl.guards import assert_not_protected
from src.online_rl.integrated.conditions import unresolved
from src.online_rl.integrated.episode import schedule
from src.online_rl.integrated.prompt_search import PromptStore, SearchBackend
from src.online_rl.integrated.records import atomic_write_json, from_dict, read_json, sha256_json, to_dict
from src.online_rl.tree.cache import CallCache
from src.online_rl.tree.decisions import Strategies
from src.online_rl.tree.effectiveness import evaluate_rubric, select
from src.online_rl.tree.posthoc import doctor_feedback, freeze_trajectory_input, generate_rubric
from src.online_rl.tree.prompt_search import EvidenceUnit, ScoringDeps, run_event
from src.online_rl.tree.records import Tree, TrajectoryRubricInput
from src.online_rl.tree.reward import agent_targets
from src.online_rl.tree.rollout import build_tree, tree_counts
from src.online_rl.tree.scoring import score_nodes

VERSION = 'sherpa_tree_v1'


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


class TreeTrainer:
    def __init__(self, cfg: dict, *, out_dir: Path, cases: list[dict], env: TreeEnv, strategies: Strategies, search_backend: SearchBackend,
                 initial_prompt: str, policy_sha: Callable[[], str], protocol_sha: str, feedback_prompt_sha256: str, updater=None):
        if unresolved(cfg): raise ValueError(f'unresolved settings: {unresolved(cfg)}')
        if cfg['rubric']['gold_answer_visible']: raise NotImplementedError('a gold-visible Rubric Generator contract is not defined')
        assert_not_protected(cases, 'tree training cases')
        self.cfg, self.out, self.cases, self.env, self.st = cfg, Path(out_dir), {c['case_id']: c for c in cases}, env, strategies
        self.backend, self.policy_sha, self.protocol_sha, self.fb_sha, self.updater = search_backend, policy_sha, protocol_sha, feedback_prompt_sha256, updater
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
        atomic_write_json(self.out / 'run_manifest.json', {'version': VERSION, 'config': self.cfg, 'config_sha256': self.config_sha,
                                                           'protocol_sha256': self.protocol_sha, 'initial_prompt_sha256': sha, 'initial_policy_sha256': self.policy_sha(),
                                                           'missing_strategies': self.st.missing(), 'doctor_api_seeded': False})
        ckpt = None
        if self.updater is not None:
            p = self.out / 'checkpoints' / 'case_000000'; tmp = p.with_name(p.name + '.tmp'); self.updater.save(tmp); tmp.rename(p); ckpt = str(p)
        s = TreeRunState(self.config_sha, self.cfg['schedule']['seed'], 0, self.cfg['rubric']['prompt_id'], sha, self.policy_sha(), ckpt, 0, 0,
                         ({'prompt_id': self.cfg['rubric']['prompt_id'], 'sha256': sha, 'event': None, 'case_cursor': 0},))
        self._write(s); return s

    def _resume(self) -> TreeRunState:
        s = from_dict(TreeRunState, read_json(self.state_path))
        if s.config_sha256 != self.config_sha: raise RuntimeError('config changed since the run started; refusing to resume')
        if s.checkpoint and self.updater is not None: self.updater.load(s.checkpoint)
        if self.policy_sha() != s.policy_sha256: raise RuntimeError('policy hash differs from the committed one')
        self.store.get(s.prompt_sha256)
        stale = [p for p in sorted((self.out / 'cases').glob('case_*.json')) if int(p.stem[5:]) >= s.case_cursor]
        stale += [p for p in sorted((self.out / 'prompt_events').glob('event_*.json')) if int(p.stem[6:]) > s.lineage_cursor]
        if stale:
            dest = self.out / 'recovered' / f'resume_{len(list((self.out / "recovered").glob("resume_*")))}'; dest.mkdir(parents=True)
            for p in stale: p.rename(dest / f'{p.parent.name}__{p.name}')
        return s

    def total_cases(self) -> int: return self.cfg['schedule']['cases']

    def run(self, max_cases: Optional[int] = None) -> None:
        done = 0
        while True:
            self._maybe_prompt_event()
            if self.state.case_cursor >= self.total_cases() or (max_cases is not None and done >= max_cases): break
            self.step(); done += 1

    def step(self) -> dict:
        s, cfg = self.state, self.cfg; ro, iv = cfg['rollout'], cfg['intervention']
        cid = schedule(sorted(self.cases), s.run_seed, s.case_cursor, 1)[0]; case = self.cases[cid]; e = self.env
        tree = build_tree(case, max_agent_turns=ro['rounds'], branching=ro['branching'], run_seed=s.run_seed, policy_sha256=s.policy_sha256,
                          protocol_sha256=self.protocol_sha, agent=e.agent, doctor=e.doctor, stop_rule=e.stop_rule, final_option=e.final_option,
                          outcome_fn=e.outcome_fn, cache=self.cache, behavior_logprobs=e.behavior_logprobs)
        decided = [l for l in tree.leaves if l.outcome is not None]
        feedback = [doctor_feedback(case, l, e.feedback_fn, self.fb_sha, self.cache) for l in decided]
        tins = [freeze_trajectory_input(case, l, f) for l, f in zip(decided, feedback)]
        prompt = self.store.get(s.prompt_sha256)
        rubrics = [generate_rubric(prompt, s.prompt_sha256, t, e.generate_fn, self.cache, e.gate_fn) for t in tins]
        nodes = {n.node_id: n for n in tree.nodes}; source_scores, effs = [], []
        for leaf, rub in zip(decided, rubrics):
            ss, eff = evaluate_rubric(case, nodes, leaf, rub, scorer=e.scorer, writer=e.writer, continuation=e.continuation,
                                      reruns=iv['reruns_per_intervention'], run_seed=s.run_seed, policy_sha256=s.policy_sha256,
                                      strategies=self.st, cache=self.cache)
            source_scores += ss; effs += [eff] if eff is not None else []
        selection = select(effs, cfg['selection'])
        kept = {d.rubric_id for d in selection if d.retained}
        scores = score_nodes(case, list(tree.nodes), [r for r in rubrics if r.rubric_id in kept], e.scorer, self.cache)
        weights = {e.rubric_id: e.R_O for e in effs if e.rubric_id in kept}
        targets = agent_targets(tree, scores, weights, self.st)
        update = self.st.need('agent_update')(tree, targets)
        record = {'case_cursor': s.case_cursor, 'case_id': cid, 'prompt_id': s.prompt_id, 'prompt_sha256': s.prompt_sha256,
                  'policy_sha256': s.policy_sha256, 'counts': tree_counts(tree), 'tree': to_dict(tree), 'feedback': [to_dict(f) for f in feedback],
                  'trajectory_inputs': [to_dict(t) for t in tins], 'trajectory_rubrics': [to_dict(r) for r in rubrics],
                  'source_path_scores': [to_dict(x) for x in source_scores], 'effectiveness': [to_dict(x) for x in effs],
                  'selection': [to_dict(d) for d in selection],
                  'response_scores': [to_dict(x) for x in scores], 'rubric_weights': weights, 'agent_targets': targets, 'update': update}
        atomic_write_json(self._case_path(s.case_cursor), record)
        self._commit(); return record

    def _commit(self) -> None:
        s = self.state; sha = self.policy_sha(); ckpt = s.checkpoint
        if sha != s.policy_sha256:
            if self.updater is None: raise RuntimeError('policy changed but no updater is available to checkpoint it')
            p = self.out / 'checkpoints' / f'case_{s.case_cursor + 1:06d}'; tmp = p.with_name(p.name + '.tmp'); self.updater.save(tmp); tmp.rename(p); ckpt = str(p)
        self._write(dataclasses.replace(s, case_cursor=s.case_cursor + 1, policy_sha256=sha, checkpoint=ckpt))

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
        if len(units) < pu['minimum_trajectories']: return
        event = s.lineage_cursor + 1
        history = [{'event': d['event'], 'accepted': d['decision']['accepted'], 'reason': d['decision']['reason']}
                   for d in (read_json(self._event_path(e)) for e in range(1, event))]
        deps = ScoringDeps(self.cfg['intervention']['reruns_per_intervention'], s.run_seed, self.env.generate_fn,
                           self.env.gate_fn, self.env.scorer, self.env.writer, self.env.continuation, s.policy_sha256, self.st, self.cache)
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
