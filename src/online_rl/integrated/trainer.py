"""Integrated trainer: one realised state per step, G same-prompt replies, K continuations, one ephemeral rubric,
one GRPO update, atomic commit; periodic bounded prompt search on the committed evidence since the last event.

Commit contract: run_state.json is written last, so step == evidence cursor == committed checkpoint/adapter hash ==
optimizer steps == prompt id/hash == lineage cursor. Anything newer is moved to recovered/ on resume, never used."""
from __future__ import annotations
import dataclasses, json, random, time
from pathlib import Path
from typing import Callable, Optional
from src.online_rl.cases import history_key
from src.online_rl.guards import assert_not_protected
from src.online_rl.integrated import metrics as M
from src.online_rl.integrated.conditions import prompt_mode, unresolved
from src.online_rl.integrated.episode import assert_single_prompt, pre_action_prefix, prompt_hash, sample_turn, schedule
from src.online_rl.integrated.continuation import ContinuationCache, PolicyIdentity, continue_k, derive_seed, in_baseline
from src.online_rl.integrated.prompt_search import PromptStore, SearchBackend, run_event, split_of
from src.online_rl.integrated.records import (BranchRecord, RunState, StateEvidence, atomic_write_json, dir_sha256, from_dict, read_json,
                                              sha256_json, sha256_text, to_dict)
from src.online_rl.integrated.rubric import (NOT_ELICITED, STATE_SCHEMA, STATE_SCHEMA_SHA256, freeze_generator_input, generate_rubric,
                                             score_responses)

VERSION = 'online_rl_integrated_v1'


class IntegratedTrainer:
    def __init__(self, cfg: dict, *, out_dir: Path, cases: list[dict], policy, updater, policy_sha: Callable[[], str], protocol_sha: str,
                 outcome_fn: Callable, backend: SearchBackend, initial_prompt: str, elicit_state: Callable):
        if unresolved(cfg): raise ValueError(f'unresolved settings: {unresolved(cfg)}')
        assert_not_protected(cases, 'integrated training cases')
        if not cfg['rubric']['enabled'] and (cfg['prompt_update']['enabled'] or cfg['reward']['w_A'] != 0):
            raise ValueError('a rubric-disabled condition needs prompt_update.enabled=false and w_A=0')
        self.cfg, self.out, self.cases = cfg, Path(out_dir), {c['case_id']: c for c in cases}
        self.policy, self.updater, self.policy_sha, self.protocol_sha = policy, updater, policy_sha, protocol_sha
        self.outcome_fn, self.backend, self.elicit_state = outcome_fn, backend, elicit_state
        self.out.mkdir(parents=True, exist_ok=True)
        self.store = PromptStore(self.out / 'prompts'); self.cache = ContinuationCache(self.out / 'continuations.jsonl')
        self.config_sha = sha256_json(cfg); self.state_path = self.out / 'run_state.json'
        self.state = self._resume() if self.state_path.exists() else self._start(initial_prompt)
        self.window = [self._evidence(i) for i in range(self.state.last_event_step, self.state.evidence_cursor)]

    def _evidence_path(self, step: int) -> Path: return self.out / 'evidence' / f'step_{step:06d}.json'

    def _event_path(self, event: int) -> Path: return self.out / 'prompt_events' / f'event_{event:04d}.json'

    def _evidence(self, step: int) -> StateEvidence:
        e = from_dict(StateEvidence, read_json(self._evidence_path(step)))
        if e.step != step: raise RuntimeError(f'evidence file for step {step} holds step {e.step}')
        return e

    def _write_state(self, state: RunState) -> None:
        atomic_write_json(self.state_path, to_dict(state)); self.state = state

    def _start(self, initial_prompt: str) -> RunState:
        if any((self.out / d).exists() for d in ('evidence', 'prompt_events', 'checkpoints')):
            raise RuntimeError(f'{self.out} has run artifacts but no run_state.json; refusing to start over them')
        rc = self.cfg['rubric']; sha = self.store.put(initial_prompt)
        if rc['prompt_sha256'] != sha: raise RuntimeError('initial generator prompt does not match rubric.prompt_sha256')
        atomic_write_json(self.out / 'run_manifest.json', {
            'version': VERSION, 'condition': self.cfg['condition'], 'prompt_mode': prompt_mode(self.cfg), 'config': self.cfg,
            'config_sha256': self.config_sha, 'advantage_mode': self.cfg['advantage_normalization']['mode'],
            'state_schema': {**STATE_SCHEMA, 'sha256': STATE_SCHEMA_SHA256},
            'policy': self.policy.identity(), 'updater': self.updater.identity(), 'protocol_sha256': self.protocol_sha,
            'train_cases': sorted(self.cases),
            'initial_prompt': {'prompt_id': rc['prompt_id'], 'sha256': sha, 'role': 'initial Rubric-Generator prompt (P0)'},
            'seed_rule': 'sha256(run_seed|key|role|index)[:8]', 'doctor_api_seeded': False})
        ckpt = self.out / 'checkpoints' / 'step_000000'                 # initial LoRA init is not seeded across processes
        tmp = ckpt.with_name(ckpt.name + '.tmp'); self.updater.save(tmp); tmp.rename(ckpt)
        state = RunState(self.config_sha[:16], self.config_sha, self.cfg['schedule']['seed'], 0, self.updater.steps, str(ckpt), dir_sha256(ckpt), self.policy_sha(),
                         rc['prompt_id'], sha, 0, 0, 0, False, ({'prompt_id': rc['prompt_id'], 'sha256': sha, 'event': None, 'step': 0},))
        self._write_state(state); return state

    def _resume(self) -> RunState:
        s = from_dict(RunState, read_json(self.state_path))
        if s.config_sha256 != self.config_sha: raise RuntimeError('config changed since the run started; refusing to resume')
        if read_json(self.out / 'run_manifest.json')['state_schema']['sha256'] != STATE_SCHEMA_SHA256: raise RuntimeError('state schema changed since the run started')
        if dir_sha256(Path(s.checkpoint)) != s.checkpoint_sha256: raise RuntimeError(f'checkpoint files of {s.checkpoint} changed since commit')
        self.updater.load(s.checkpoint)
        if self.policy_sha() != s.adapter_sha256:
            raise RuntimeError(f'adapter hash after loading {s.checkpoint} != committed {s.adapter_sha256[:12]}; checkpoint and run_state disagree')
        if self.updater.steps != s.optimizer_steps: raise RuntimeError(f'optimizer steps {self.updater.steps} != committed {s.optimizer_steps}')
        if s.evidence_cursor != s.step: raise RuntimeError('run_state evidence cursor differs from its step')
        last = s.history[-1]
        if (last['prompt_id'], last['sha256']) != (s.prompt_id, s.prompt_sha256): raise RuntimeError('current prompt is not the last accepted prompt')
        self.store.get(s.prompt_sha256)
        for i in range(s.evidence_cursor): self._evidence(i)
        for e in range(1, s.lineage_cursor + 1):
            if not self._event_path(e).exists(): raise RuntimeError(f'committed prompt event {e} is missing')
        self._recover(s)
        return s

    def _recover(self, s: RunState) -> None:
        """Move artifacts newer than the commit point to recovered/<n>/ (never reused, never deleted)."""
        stale = [p for p in sorted((self.out / 'evidence').glob('step_*.json')) if int(p.stem[5:]) >= s.evidence_cursor]
        stale += [p for p in sorted((self.out / 'prompt_events').glob('event_*.json')) if int(p.stem[6:]) > s.lineage_cursor]
        ck = Path(s.checkpoint).name
        stale += [p for p in sorted((self.out / 'checkpoints').glob('step_*')) if p.name != ck and int(p.name[5:11]) > s.step]
        if not stale: return
        dest = self.out / 'recovered' / f'resume_{len(list((self.out / "recovered").glob("resume_*")))}'
        dest.mkdir(parents=True)
        for p in stale: p.rename(dest / f'{p.parent.name}__{p.name}')
        self._log('recovery_log.jsonl', {'resume_at_step': s.step, 'moved': [p.name for p in stale], 'to': str(dest)})

    def _log(self, name: str, row: dict) -> None:
        with (self.out / name).open('a') as f: f.write(json.dumps(row, ensure_ascii=False, default=str) + '\n')

    def total_steps(self) -> int: return self.cfg['schedule']['stages'] * self.cfg['schedule']['steps_per_stage']

    def run(self, max_steps: Optional[int] = None) -> None:
        done = 0
        while True:
            self._maybe_prompt_event()
            if self.state.step >= self.total_steps() or (max_steps is not None and done >= max_steps): break
            self.step(); done += 1
        if self.state.step >= self.total_steps() and not self.state.finished: self._write_state(dataclasses.replace(self.state, finished=True))

    def step(self) -> StateEvidence:
        cfg, s, g, t0 = self.cfg, self.state, self.state.step, time.monotonic()
        ro, rc, mc = cfg['rollout'], cfg['rubric'], cfg['metrics']
        cid = schedule(sorted(self.cases), s.run_seed, g, 1)[0]; case = self.cases[cid]; rkey = f'{cid}@{g}'
        rollout_seed = derive_seed(s.run_seed, rkey, 'realise', 0)
        dialogue = self.policy.generate(case, seed=rollout_seed)['dialogue']
        agent_idx = [i for i, d in enumerate(dialogue) if d['speaker'] == 'agent' and not d.get('final')]
        t = sample_turn(len(agent_idx), ro['turn_sampling'], random.Random(derive_seed(s.run_seed, rkey, 'turn', 0)))
        prefix = pre_action_prefix(dialogue, agent_idx[t - 1])
        view = {k: case[k] for k in ('case_id', 'question', 'options')} | {'scenario': case.get('scenario') or ''}
        state = self.elicit_state(view, [dict(d) for d in prefix], t) if rc['enabled'] else NOT_ELICITED   # before any reply; never shown to the policy
        gin = freeze_generator_input(case, prefix, t, state)    # frozen before any reply or outcome exists
        state_key = history_key(cid, prefix)

        G, K = ro['group_size'], ro['outcome_continuations_K']
        rseeds = [derive_seed(s.run_seed, state_key, 'response', j) for j in range(G)]
        rollout_sha = self.policy_sha()
        if rollout_sha != s.adapter_sha256: raise RuntimeError('policy adapter differs from the committed adapter before sampling')
        traces = self.policy.sample_completions(case, prefix, t, rseeds)
        if len(traces) != G: raise RuntimeError(f'expected {G} completions, got {len(traces)}')
        for tr, lp in zip(traces, self.updater.behavior_logprobs(traces)): tr['behavior_logprobs'] = lp   # pi_old, before any update
        records = [{'trajectory_id': f'step{g}:{state_key[:12]}:a{j}', 'case_id': state_key, 'rollout_policy_sha256': rollout_sha, 'agent_traces': [tr]}
                   for j, tr in enumerate(traces)]
        assert_single_prompt(records)

        ident = PolicyIdentity(cfg['policy']['base_model'], s.adapter_sha256, self.protocol_sha)
        cseeds = [derive_seed(s.run_seed, state_key, 'continuation', i) for i in range(K)]
        conts = [continue_k(self.policy, self.cache, ident, case, state_key, prefix, t, tr['completion_text'], cseeds, None, self.outcome_fn) for tr in traces]
        R_O = [M.r_o([c.success for c in cs]) for cs in conts]
        V_h, base = None, []
        eff = cfg['effectiveness']
        if in_baseline(eff['baseline_mode'], eff['diagnostic_fraction'], s.run_seed, state_key):
            base = continue_k(self.policy, self.cache, ident, case, state_key, prefix, t, None, cseeds,
                              [derive_seed(s.run_seed, state_key, 'natural', i) for i in range(K)], self.outcome_fn)
            V_h = M.r_o([c.success for c in base])

        rubric = generate_rubric(self.store.get(s.prompt_sha256), s.prompt_id, s.prompt_sha256, gin, state_key, case,
                                 self.backend.generate, self.backend.gate, rc['max_attempts'], g)
        scores = score_responses(rubric, gin, [tr['completion_text'] if o is not None else None for tr, o in zip(traces, R_O)], self.backend.score)
        R_A = [None if x is None else x['R_A'] for x in scores]; hard = [None if x is None else x['R_A_hard'] for x in scores]
        w = cfg['reward']
        rewards = [M.reward(o, a, w['w_O'], w['w_A'], rubric.valid) for o, a in zip(R_O, R_A)]
        informative = M.informative(R_O, mc['epsilon_O'])
        diag = {**M.state_diagnostics(R_A, hard, R_O, mc['epsilon_score']), 'rubric_masked': not rubric.valid, 'rubric_failure': rubric.failure}
        if V_h is not None:
            dc = cfg['diagnostics']; diag.update(M.effect_diagnostics(R_A, [M.r_e(o, V_h) for o in R_O], dc['high_score'], dc['severe_effect']))

        for r, rw in zip(records, rewards): r['reward'] = rw
        mode = cfg['advantage_normalization']['mode']; usable = sum(r is not None for r in rewards)
        if usable < ro['min_usable_responses']:
            upd = {'optimizer_step': False, 'skip_reason': f'{usable} usable responses < min_usable_responses {ro["min_usable_responses"]}'}
        else:
            if self.policy_sha() != rollout_sha: raise RuntimeError('adapter changed between sampling and update')
            upd = self.updater.update(records)
        upd = {**upd, 'advantage_mode': mode, 'advantages': M.advantages(rewards, mode)}

        branches = tuple(BranchRecord(f'{state_key[:16]}:a{j}', f'step{g}:{cid}:A{t}:a{j}', f'step{g}:{cid}:rollout', state_key, t, rseeds[j], prompt_hash(tr),
                                      tr['completion_text'], sha256_text(tr['completion_text']), tuple(tr.get('completion_ids') or ()),
                                      tuple(tr.get('old_policy_logprobs') or ()), tuple(c.key for c in cs), tuple(c.success for c in cs),
                                      R_O[j], R_A[j], hard[j], scores[j]['probabilities'] if scores[j] else None, M.r_e(R_O[j], V_h), rewards[j],
                                      tuple(tr['behavior_logprobs']))
                         for j, (tr, cs) in enumerate(zip(traces, conts)))
        ev = StateEvidence(g, state_key, cid, t, rollout_seed, gin, rubric, branches, s.prompt_id, s.prompt_sha256, s.adapter_sha256, informative,
                           M.q_cov(R_A, R_O, rubric.valid, mc['epsilon_score']) if informative else None, V_h, tuple(c.key for c in base), diag, upd)
        atomic_write_json(self._evidence_path(g), to_dict(ev))
        self._commit(ev, time.monotonic() - t0); return ev

    def _commit(self, ev: StateEvidence, seconds: float) -> None:
        s = self.state; sha = self.policy_sha(); ckpt, ckpt_sha = s.checkpoint, s.checkpoint_sha256
        if sha != s.adapter_sha256 or self.updater.steps != s.optimizer_steps:
            final = self.out / 'checkpoints' / f'step_{s.step + 1:06d}'; tmp = final.with_name(final.name + '.tmp')
            if final.exists() or tmp.exists(): raise RuntimeError(f'{final} already exists; resume should have recovered it')
            self.updater.save(tmp); tmp.rename(final); ckpt, ckpt_sha = str(final), dir_sha256(final)
        self._write_state(dataclasses.replace(s, step=s.step + 1, evidence_cursor=s.step + 1, optimizer_steps=self.updater.steps, checkpoint=ckpt,
                                              checkpoint_sha256=ckpt_sha, adapter_sha256=sha))
        self.window.append(ev)
        self._log('train_log.jsonl', {'step': ev.step, 'case_id': ev.case_id, 'turn': f'A{ev.turn}', 'state_key': ev.state_key, 'prompt_id': ev.prompt_id,
                                      'informative': ev.informative, 'Q_cov': ev.Q_cov, 'R_O': [b.R_O for b in ev.branches], 'R_A': [b.R_A for b in ev.branches],
                                      'reward': [b.reward for b in ev.branches], 'rubric_failure': ev.rubric.failure, 'V_h': ev.V_h,
                                      'optimizer_step': ev.update.get('optimizer_step'), 'skip_reason': ev.update.get('skip_reason'), 'checkpoint': ckpt,
                                      'continuation_cache': {'hits': self.cache.hits, 'misses': self.cache.misses}, 'seconds': round(seconds, 2)})

    def _due(self) -> bool:
        pu = self.cfg['prompt_update']
        if not pu['enabled'] or self.state.step >= self.total_steps() or self.state.step - self.state.last_event_step < pu['every_n_steps']: return False
        if len(self.window) < pu['minimum_raw_states']: return False
        split = [split_of(self.state.run_seed, e.state_key, pu['acceptance_fraction']) for e in self.window if e.informative]
        return min(split.count('search'), split.count('acceptance')) >= pu['minimum_informative_states']

    def _maybe_prompt_event(self) -> None:
        if not self._due(): return
        s = self.state; event = s.lineage_cursor + 1
        history = [{'event': d['event'], 'accepted': d['decision']['accepted'], 'reason': d['decision']['reason'],
                    'mean_paired_Q_gain': (d['decision'].get('acceptance') or {}).get('mean_paired_Q_gain')}
                   for d in (read_json(self._event_path(e)) for e in range(1, event))]
        res = run_event(event, s.prompt_id, s.prompt_sha256, self.window, self.cases, self.store, self.backend, self.cfg, s.run_seed, s.step, history)
        atomic_write_json(self._event_path(event), {**res, 'versions': [to_dict(v) for v in res['versions']]})
        d = res['decision']; new = dataclasses.replace(s, lineage_cursor=event, last_event_step=s.step)
        if d['accepted']:
            new = dataclasses.replace(new, prompt_id=d['candidate_prompt_id'], prompt_sha256=d['candidate_sha256'],
                                      history=s.history + ({'prompt_id': d['candidate_prompt_id'], 'sha256': d['candidate_sha256'], 'event': event, 'step': s.step},))
        self._write_state(new); self.window = []
        self._log('prompt_log.jsonl', {'event': event, 'step': s.step, 'parent': s.prompt_id, 'accepted': d['accepted'], 'reason': d['reason'],
                                       'candidate': d.get('candidate_prompt_id'), 'search_states': len(res['search_evidence_ids']),
                                       'acceptance_states': len(res['acceptance_evidence_ids'])})

