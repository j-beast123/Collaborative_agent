"""Semantic dedup of verified candidates against ONLY newly registered criteria.

No bank lock is held during the Gemini request. An uncertain or execution-invalid
comparison keeps the independently verified candidate distinct, never merges it.
"""
import json
import re
from src.online_rl.integrated.records import sha256_json, to_dict
from src.online_rl.tree import cumulative_bank as CB

VERSION = 'parallel_chunk_delta_commit_v1'
CHECKS = ('same_applicability','same_required_behavior','same_minimum_and_completion','same_score_boundaries')
SYSTEM = '''Compare public behavioral rubrics for semantic identity, not mere similarity.
The candidates already passed their own Gate and witness verification. Compare each candidate only with the supplied newly added bank entries. Never change any criterion or consult medical correctness, outcomes, future dialogue or hidden labels.
Merge only when applicability scopes (Doctor/Agent state, objects, relation and qualifiers), required behavior, minimum/completion definitions, and all 0/1/2 score boundaries are mutually equivalent. Broader/narrower conditions, differing exclusions, different entities/relations or thresholds are distinct. Wording differences alone may be equivalent. Missing M/C fields must not be invented.
For every candidate return its candidate_id, either the matching bank_id or null, all four checks and a short rationale. A non-null bank_id requires all four checks true. If uncertain, use null and keep both criteria. If multiple bank entries qualify choose the lexicographically smallest bank_id. Return only the required JSON.'''


def candidates(built):
    retained={s.rubric_id for s in built['selection'] if s.retained}
    return {CB.content_hash(r.rubric): r for rid,r in built['canon'].items() if rid in retained}


def compare(trainer, pending, delta):
    if not pending or not delta: return {'aliases':{}, 'status':'no_comparison_needed'}
    from src.online_rl.tree import backends as B
    from jsonschema import validate, ValidationError
    context=trainer.env.generate_fn.chunk_context
    entries={e['bank_id']:e for e in delta}
    packet={'candidates':{key:CB.criteria(r.rubric) for key,r in sorted(pending.items())},
            'new_bank_entries':{key:CB.criteria(e['rubric']['rubric']) for key,e in sorted(entries.items())}}
    schema={'type':'object','additionalProperties':False,'required':['decisions'], 'properties':{'decisions':{
        'type':'array','items':{'type':'object','additionalProperties':False,
        'required':['candidate_id','bank_id','checks','rationale'],'properties':{
            'candidate_id':{'type':'string','enum':sorted(pending)},
            'bank_id':{'type':['string','null'],'enum':sorted(entries)+[None]},
            'checks':{'type':'object','additionalProperties':False,'required':list(CHECKS),
                      'properties':{key:{'type':'boolean'} for key in CHECKS}},
            'rationale':{'type':'string'}}}}}}
    settings={**context['settings'],'max_tokens':4096,'response_format':None,'bank_delta_protocol':VERSION}
    identity={'protocol':VERSION,'prompt_sha256':sha256_json(SYSTEM),'packet':packet,
              'model':context['model'],'settings':settings}
    def request():
        failures=[]
        budget=int(trainer.cfg.get('judges',{}).get('execution_retry_attempts',3))
        for attempt in range(budget):
            user=json.dumps(packet,ensure_ascii=False)
            if failures: user+='\nPrevious execution error: '+failures[-1]+'. Return every candidate exactly once in the original schema.'
            try:
                raw,meta=B._generator_json(context['key'],context['model'],SYSTEM,user,settings,schema,'bank_delta_dedup','rubric_matcher')
                parsed=json.loads(re.sub(r'^```(?:json)?\s*|\s*```$', '', (raw or '').strip()));validate(parsed,schema)
                rows=parsed['decisions']
                if len(rows)!=len(pending) or {r['candidate_id'] for r in rows}!=set(pending):
                    raise ValueError('Delta comparison candidate coverage differs')
                aliases={}
                for row in rows:
                    if not row['rationale'].strip():raise ValueError('Delta comparison needs rationale')
                    if row['bank_id'] is not None:
                        if not all(row['checks'][k] is True for k in CHECKS):
                            raise ValueError('Semantic duplicate requires all equivalence checks')
                        aliases[row['candidate_id']]={'protocol':VERSION,'candidate_content_sha256':row['candidate_id'],
                            'bank_id':row['bank_id'],'bank_content_sha256':entries[row['bank_id']]['content_sha256'],
                            'checks':row['checks'],'rationale':row['rationale'],'comparison_sha256':sha256_json(identity)}
                return {'status':'compared','aliases':aliases,'decisions':rows,'attempts':attempt+1,
                        'prior_errors':failures,'audit':{'packet':packet,'raw':raw,'meta':meta}}
            except (ValueError,TypeError,ValidationError,B.RubricOutputIncomplete) as exc:
                failures.append(str(exc)[:240])
        return {'status':'execution_unresolved','aliases':{},'attempts':budget,'errors':failures,
                'policy':'keep_verified_candidates_distinct'}
    return trainer.cache.call('bank_delta_dedup',identity,request)


def merge_records(built, case, state, aliases):
    return [{'case_id':case['case_id'],'case_cursor':state.case_cursor,
             'policy_sha256':state.policy_sha256,'prompt_sha256':state.prompt_sha256,
             'trajectory_rubrics':[to_dict(r) for r in built['canon'].values()],
             'selection':[to_dict(s) for s in built['selection']
                          if s.retained and CB.content_hash(built['canon'][s.rubric_id].rubric) not in aliases],
             'rubric_bank':{'witness':built['witness']}}]


def validate_alias(alias, content_sha, entries):
    target=entries.get(alias.get('bank_id'))
    if (alias.get('protocol')!=VERSION or alias.get('candidate_content_sha256')!=content_sha
            or target is None or alias.get('bank_content_sha256')!=target['content_sha256']
            or set(alias.get('checks',{}))!=set(CHECKS) or not all(alias['checks'][k] is True for k in CHECKS)
            or not alias.get('rationale') or not alias.get('comparison_sha256')):
        raise RuntimeError('Saved delta semantic alias lacks its bound equivalence evidence')
    return target
