"""Private Doctor reflection, Gemini generation and one blind call per trajectory."""
from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from src.common import usage
from src.online_rl.integrated.continuation import derive_seed
from src.online_rl.integrated.records import sha256_json, sha256_text
from src.online_rl.tree.backends import _generator_json, with_role
from src.online_rl.sherpa.rubrics import (
    SET_SCHEMA, LEAKAGE_INSTRUCTION, checklist_schema, filter_leakage, parse_rubrics,
)


def case_view(case):
    return {key: case.get(key) for key in ('scenario', 'question', 'options')}


def judge_dialogue(trajectory):
    # FINAL is the outcome-bearing decision, not collaborator behavior.
    return [{key: row.get(key) for key in ('speaker', 'round', 'text')}
            for row in trajectory['dialogue'] if row.get('stage') != 'FINAL']


class ChecklistBackend:
    def __init__(self, cfg, policy, root):
        from scripts.online_rl.run_tree import load_key
        self.cfg, self.policy, self.root = cfg, policy, Path(root)
        self.generator = cfg['rubric']['generator_model']
        generation = cfg['rubric']['generation']
        self.settings = {key: generation.get(key) for key in ('max_tokens', 'temperature', 'reasoning_effort')}
        self.settings.update(api_provider=generation.get('api_provider', 'openrouter'),
                             response_format='json_schema_strict')
        judges = cfg['judges']
        self.judge_model = judges['scorer_model']
        self.judge_settings = {'max_tokens': judges['scorer_max_tokens'], 'temperature': 0.0,
                              'reasoning_effort': judges.get('reasoning_effort', 'low'),
                              'api_provider': judges.get('api_provider', 'openrouter'),
                              'response_format': 'json_schema_strict'}
        self.key = load_key({'api_provider': self.settings['api_provider']})
        self.judge_key = load_key({'api_provider': self.judge_settings['api_provider']})
        self.judge_prompt = (self.root / cfg['rubric_pipeline']['judge_prompt_file']).read_text()
        self.reflection_prompt = (self.root / cfg['rubric_pipeline']['reflection_prompt_file']).read_text()

    def identity(self):
        return {'generator': self.generator, 'generation': self.settings,
                'judge': self.judge_model, 'judge_settings': self.judge_settings,
                'judge_prompt': sha256_text(self.judge_prompt),
                'reflection_prompt': sha256_text(self.reflection_prompt),
                'doctor': self.policy.doctor.protocol}

    def reflect(self, case, trajectory, seed):
        started = time.perf_counter()
        request = {'case': case_view(case), 'dialogue': trajectory['dialogue'],
                   'outcome': trajectory['outcome']}
        with usage.case(case['case_id']):
            raw = with_role('doctor_reflection', lambda: self.policy.doctor._chat(
                json.dumps(request, ensure_ascii=False), seed,
                self.cfg['rubric_pipeline']['reflection_max_tokens'],
                self.policy.doctor.system + '\n\n' + self.reflection_prompt))
        return {'trajectory_id': trajectory['trajectory_id'], 'outcome': trajectory['outcome'],
                'text': raw, 'seed': seed, 'seconds': time.perf_counter() - started}

    def generate(self, case, examples, prompt):
        started = time.perf_counter()
        request = {'case': case_view(case), 'examples': examples}
        attempts, format_retries, leakage_retries = [], 0, 0
        generated_count = removed_count = 0
        system = prompt if self.cfg.get('method') == 'rar' else prompt + '\n\n' + LEAKAGE_INSTRUCTION
        while True:
            effective = dict(request)
            if attempts:
                effective['repair'] = {'instruction': 'Regenerate the complete set of five or six safe rubrics.',
                                       'error': attempts[-1].get('error'),
                                       'removed_rubrics': attempts[-1].get('removed', [])}
            raw, meta = _generator_json(self.key, self.generator, system,
                                       json.dumps(effective, ensure_ascii=False), self.settings,
                                       SET_SCHEMA, 'trajectory_checklists', 'rubric_generator')
            attempt = {'raw': raw, 'meta': meta}
            attempts.append(attempt)
            try:
                rubrics = parse_rubrics(raw)
            except (ValueError, TypeError) as exc:
                attempt.update(valid=False, error=str(exc), error_type='format')
                if format_retries == 0:
                    format_retries += 1
                    continue
                break
            generated_count += len(rubrics)
            safe, removed = filter_leakage(case, rubrics, method=self.cfg.get('method', 'sherpa'))
            removed_count += len(removed)
            attempt.update(valid=not removed, removed=removed,
                           error='Answer leakage' if removed else None, error_type='leakage' if removed else None)
            if removed and leakage_retries == 0:
                leakage_retries += 1
                continue
            if len(safe) >= 5:
                return {'valid': True, 'rubrics': safe, 'attempts': attempts,
                        'removed_count': removed_count, 'generated_count': generated_count,
                        'removed_fraction': removed_count / generated_count if generated_count else 0.0,
                        'seconds': time.perf_counter() - started}
            break
        return {'valid': False, 'rubrics': [], 'attempts': attempts,
                'removed_count': removed_count, 'generated_count': generated_count,
                'removed_fraction': removed_count / generated_count if generated_count else 0.0,
                'seconds': time.perf_counter() - started}

    def score(self, case, trajectory, rubrics):
        started = time.perf_counter()
        # Explicit allowlist excludes outcomes, FINAL, reflections and source provenance.
        request = {'case': case_view(case), 'dialogue': judge_dialogue(trajectory),
                   'rubrics': [{key: row[key] for key in ('name', 'situation', 'behavior')}
                               for row in rubrics]}
        raw, meta = _generator_json(self.judge_key, self.judge_model, self.judge_prompt,
                                   json.dumps(request, ensure_ascii=False), self.judge_settings,
                                   checklist_schema(len(rubrics)), 'trajectory_scores', 'rubric_judge')
        result = json.loads(raw)
        scores = result.get('scores') if isinstance(result, dict) and set(result) == {'scores'} else None
        if (not isinstance(scores, list) or len(scores) != len(rubrics)
                or any(type(score) is not int or score not in (0, 1) for score in scores)):
            error = ValueError('Judge must return one binary score for every rubric')
            error.audit = {'raw': raw, 'meta': meta, 'request': request}
            raise error
        return {'trajectory_id': trajectory['trajectory_id'], 'scores': scores,
                'reward': sum(scores) / len(scores), 'raw': raw, 'meta': meta,
                'seconds': time.perf_counter() - started}

    def propose(self, parent, evidence, iteration, index):
        request = {'parent_prompt': parent, 'low_auc_cases': evidence,
                   'iteration': iteration, 'candidate_index': index}
        system = (self.root / self.cfg['prompt_optimization']['proposal_prompt_file']).read_text()
        schema = {'type': 'object', 'properties': {'prompt': {'type': 'string'}},
                  'required': ['prompt'], 'additionalProperties': False}
        raw, meta = _generator_json(self.key, self.generator, system + '\n' + LEAKAGE_INSTRUCTION,
                                   json.dumps(request, ensure_ascii=False), self.settings,
                                   schema, 'generator_prompt_candidate', 'prompt_optimizer')
        value = json.loads(raw)
        if set(value) != {'prompt'} or not isinstance(value['prompt'], str) or not value['prompt'].strip():
            raise ValueError('Prompt proposer returned no prompt')
        return {'text': value['prompt'].strip(), 'raw': raw, 'meta': meta}

    def score_all(self, case, trajectories, rubrics, cache, identity):
        def one(trajectory):
            ident = {**identity, 'trajectory': trajectory['trajectory_id'],
                     'dialogue': sha256_json(judge_dialogue(trajectory)), 'rubrics': sha256_json(rubrics),
                     'backend': self.identity()}
            with usage.case(case['case_id']):
                return cache.call('trajectory_score', ident, lambda: self.score(case, trajectory, rubrics))
        with ThreadPoolExecutor(self.cfg['execution']['judge_workers']) as pool:
            return list(pool.map(one, trajectories))
