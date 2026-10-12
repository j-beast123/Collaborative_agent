"""Turn-local reflection/checklists; outcome-blind, suffix-blind action scoring."""
import json
from pathlib import Path
import time

from src.common import usage
from src.online_rl.integrated.records import read_json, sha256_text
from src.online_rl.sherpa.backend import ChecklistBackend, case_view
from src.online_rl.sherpa.rubrics import checklist_schema, obj
from src.online_rl.tree.backends import _generator_json, with_role
from src.online_rl.sherpa.plan_c_rollouts import PROTOCOL


def generator_prompt(cfg, root, *, policy=None):
    """Transfer the optimized trajectory prompt with an explicit new turn contract.

    This is an adapted prompt, NOT an AUC-optimized Plan C artifact. Preserve
    both hashes so reports cannot confuse the two evaluation scopes.
    """
    root = Path(root)
    source = read_json(root / cfg['prompt_optimization']['final_artifact'])
    if source.get('schema') != 'sherpa_auc_prompt_v1' or source.get('test_used') is not False:
        raise RuntimeError('Expected a train-only optimized source prompt')
    base = (root / source['prompt_file']).read_text()
    if sha256_text(base) != source['prompt_sha256']:
        raise RuntimeError('Optimized source prompt hash changed')
    if source['base_identity']['base_model'] != cfg['policy']['base_model']:
        raise RuntimeError('Source optimized prompt used a different assistant model')
    if policy is not None and source['base_identity']['doctor'] != policy.doctor.protocol:
        raise RuntimeError('Source optimized prompt used a different Doctor protocol')
    base_file = cfg['plan_c'].get('generator_base_prompt_file')
    optimized_source_hash = source['prompt_sha256']
    if base_file:
        base = (root / base_file).read_text()
        if not base.strip():
            raise RuntimeError('Plan C generator base prompt is empty')
    contract = (root / cfg['plan_c']['generator_instruction_file']).read_text()
    text = base + '\n\n' + contract
    return text, {'schema': 'plan_c_transferred_prompt_v2', 'source_artifact':
                  cfg['prompt_optimization']['final_artifact'],
                  'source_optimized_prompt_sha256': optimized_source_hash,
                  'source_prompt_sha256': sha256_text(base),
                  'base_prompt_file': str((root / base_file).resolve()) if base_file else source['prompt_file'],
                  'base_prompt_source': 'explicit_user_revision' if base_file else 'auc_optimized',
                  'turn_contract_sha256': sha256_text(contract),
                  'prompt_sha256': sha256_text(text), 'optimized_for_plan_c': False}


def scoring_request(case, candidate, rubrics):
    # Explicit allowlist: absolutely no outcomes, future messages, FINAL,
    # reflections, branch IDs, private status, sampling traces or source labels.
    return {'case': case_view(case), 'target_turn': candidate['target_turn'],
            'shared_history': [{k: row.get(k) for k in ('speaker', 'round', 'text')}
                               for row in candidate['shared_history']],
            'target_response': candidate['target_response'],
            'rubrics': [{k: r[k] for k in ('name', 'situation', 'behavior')} for r in rubrics]}


def generation_examples(examples):
    return [{'shared_history': e['trajectory']['shared_history'],
             'target_turn': e['trajectory']['target_turn'],
             'target_response': e['trajectory']['target_response'],
             'continuation': e['trajectory']['dialogue'][
                 len(e['trajectory']['shared_history']) + 1:],
             'outcome': e['trajectory']['outcome'],
             'doctor_reflection': e['doctor_reflection']['text']} for e in examples]


class TurnChecklistBackend(ChecklistBackend):
    def identity(self):
        return {**super().identity(), 'turn_protocol': PROTOCOL,
                'generator_base': sha256_text((self.root / self.cfg['plan_c'][
                    'generator_base_prompt_file']).read_text())
                    if self.cfg['plan_c'].get('generator_base_prompt_file') else None,
                'turn_contract': sha256_text((self.root / self.cfg['plan_c'][
                    'generator_instruction_file']).read_text()),
                'scoring_visibility': 'shared_history_and_target_only_v1',
                'reflection_evidence': 'target_whole_turn_or_none_v1'}

    def generate(self, case, examples, prompt):
        # Parent generator retains strict schema, immutable assistant-only and
        # no-answer-leakage constraints, bounded repairs and complete raw audit.
        return super().generate(case, generation_examples(examples), prompt)

    def reflect(self, case, candidate, seed):
        started = time.perf_counter()
        request = {'case': case_view(case), 'target_turn': candidate['target_turn'],
                   'shared_history': candidate['shared_history'],
                   'target_response': candidate['target_response'],
                   'continuation': candidate['dialogue'][len(candidate['shared_history']) + 1:],
                   'outcome': candidate['outcome']}
        item = obj({'evidence_id': {'type': 'string', 'enum': ['target', 'none']},
                    'reflection': {'type': 'string', 'minLength': 1, 'maxLength': 900}})
        schema = obj({key: item for key in ('helpful', 'hindering', 'wished_for')})
        with usage.case(case['case_id']):
            raw = with_role('doctor_reflection', lambda: self.policy.doctor._chat(
                json.dumps(request, ensure_ascii=False), seed,
                self.cfg['rubric_pipeline']['reflection_max_tokens'],
                self.reflection_prompt, format_schema=schema))
        value = json.loads(raw)
        if not isinstance(value, dict) or set(value) != {'helpful', 'hindering', 'wished_for'}:
            raise ValueError('Invalid target-turn private reflection')
        lines, evidence = [], {}
        for key, row in value.items():
            if (not isinstance(row, dict) or set(row) != {'evidence_id', 'reflection'}
                    or row['evidence_id'] not in ('target', 'none')
                    or not isinstance(row['reflection'], str) or not row['reflection'].strip()):
                raise ValueError('Invalid grounded target-turn reflection field')
            if row['evidence_id'] == 'none' and not row['reflection'].startswith('None identified'):
                raise ValueError('Missing reflection evidence must be explicit')
            evidence[key] = candidate['target_response'] if row['evidence_id'] == 'target' else None
            lines.append(key + ': ' + row['reflection'])
        return {'trajectory_id': candidate['trajectory_id'], 'outcome': candidate['outcome'],
                'text': '\n'.join(lines), 'structured': value, 'evidence': evidence,
                'seed': seed, 'seconds': time.perf_counter() - started}

    def score(self, case, candidate, rubrics):
        started = time.perf_counter()
        request = scoring_request(case, candidate, rubrics)
        raw, meta = _generator_json(self.judge_key, self.judge_model, self.judge_prompt,
                                   json.dumps(request, ensure_ascii=False), self.judge_settings,
                                   checklist_schema(len(rubrics)), 'turn_scores', 'rubric_judge')
        value = json.loads(raw)
        scores = value.get('scores') if isinstance(value, dict) and set(value) == {'scores'} else None
        if (not isinstance(scores, list) or len(scores) != len(rubrics)
                or any(type(s) is not int or s not in (0, 1) for s in scores)):
            error = ValueError('Expected one binary score per target-turn rubric')
            error.audit = {'raw': raw, 'meta': meta, 'request': request}
            raise error
        return {'trajectory_id': candidate['trajectory_id'], 'scores': scores,
                'reward': sum(scores) / len(scores), 'raw': raw, 'meta': meta,
                'seconds': time.perf_counter() - started}
