"""Bounded per-case reward phases after the original five-case rollout barrier."""
import copy
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from src.judges.llm import ChatRequest
from src.online_rl.integrated.records import sha256_json
from src.online_rl.tree.cache import CallCache
from src.online_rl.tree.team import node_credit
from baselines.common.rubrics import RubricEngine, PROTOCOL, sibling_groups, state_input
from baselines.parallel_api.io import append_json
from baselines.parallel_api.json_response import decode_json_response


class ParallelRubricEngine(RubricEngine):
    def __init__(self, *args, api_workers=20, grader_workers=None, **kwargs):
        if isinstance(api_workers, bool) or not isinstance(api_workers, int) or not 1 <= api_workers <= 20:
            raise ValueError('api_workers must be an integer from 1 to 20 per method')
        grader_workers = api_workers if grader_workers is None else grader_workers
        if isinstance(grader_workers, bool) or not isinstance(grader_workers, int) or not 1 <= grader_workers <= 100:
            raise ValueError('grader_workers must be an integer from 1 to 100 per method')
        super().__init__(*args, **kwargs)
        if self.source and not self.registry:
            raise ValueError('Parallel cross-method rubric reuse requires the shared atomic registry')
        self.api_workers = api_workers
        self.grader_workers = grader_workers
        # One case coordinator per engine. The original hook processes cases in
        # their fixed order. Reference swaps and seeds also remain in group order.
        self._score_lock = threading.Lock()

    def call(self,kind,prompt,payload,validate,*,generator=False,initial=False,_locked=False):
        backend=self.generator if generator else self.grader
        params=self.gen_params if generator else self.grade_params
        identity={'backend':backend.identity(),'system':prompt,'payload':payload,'params':params,'protocol':PROTOCOL}
        cache=self.initial_cache if initial else self.cache
        if initial and self.registry and not _locked:
            import fcntl
            from src.online_rl.integrated.records import atomic_write_json
            key=sha256_json([kind,identity]);self.registry.mkdir(parents=True,exist_ok=True)
            path=self.registry/(key+'.json')
            with (self.registry/(key+'.lock')).open('a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX)
                if path.exists():
                    row=json.loads(path.read_text())
                    if row['key']!=key or row['identity']!=identity or row['value_sha256']!=sha256_json(row['value']):raise RuntimeError('Initial registry artifact changed')
                    validate(row['value'])
                    if key in cache.store and cache.store[key]['value']!=row['value']:raise RuntimeError('Initial rubric registry/cache conflict')
                    imported=key not in cache.store
                    value=cache.call(kind,identity,lambda:row['value'])
                    if imported:
                        append_json(self.out/'initial_rubric_imports.jsonl', {'source':str(path),'producer':row['producer'],'key':key,'value_sha256':row['value_sha256']})
                    return value
                value=self.call(kind,prompt,payload,validate,generator=generator,initial=initial,_locked=True)
                atomic_write_json(path,{'key':key,'identity':identity,'value':value,'value_sha256':sha256_json(value),'producer':str(self.out)})
                return value
        if initial and not self.registry and self.source and self.source.exists():
            source=CallCache(self.source);key=sha256_json([kind,identity])
            if key in source.store:
                value=source.store[key]['value']
                validate(value)  # validate imported artifacts; no cross-method API/cache writes
                if key in cache.store and cache.store[key]['value']!=value:raise RuntimeError('Same-input initial rubric conflict')
                if key not in cache.store:
                    cache.call(kind,identity,lambda:value)
                    self.out.mkdir(parents=True,exist_ok=True)
                    append_json(self.out/'initial_rubric_imports.jsonl', {'source':str(self.source),'key':key,'value_sha256':sha256_json(value)})
        def fetch():
            # Keep the exact request/model/parameters. Only malformed response
            # syntax triggers up to two fresh calls; invalid schema still fails.
            request=ChatRequest(prompt,json.dumps(payload,ensure_ascii=False),f'{self.method}.{kind}',params)
            for format_attempt in range(1,4):
                result=backend.complete(request)
                self.out.mkdir(parents=True,exist_ok=True)
                row={'kind':kind,'input_sha256':sha256_json(identity),'result':result.to_dict(),
                     'format_attempt':format_attempt}
                if result.status!='ok':
                    row['validation_status']='request_failed'
                    append_json(self.out/'judge_results.jsonl',row)
                    raise RuntimeError(f'{kind} failed: {result.error}')
                try:
                    value,normalization=decode_json_response(result.text)
                except ValueError:
                    row['validation_status']='invalid_json'
                    append_json(self.out/'judge_results.jsonl',row)
                    if format_attempt==3:
                        raise RuntimeError(f'{kind}: invalid JSON after 3 responses; see {self.out}/judge_results.jsonl; no invalid result cached') from None
                    continue
                row['json_normalization']=normalization
                try:validate(value)
                except (ValueError,TypeError,KeyError):
                    row['validation_status']='invalid_schema'
                    append_json(self.out/'judge_results.jsonl',row)
                    raise
                row['validation_status']='valid'
                append_json(self.out/'judge_results.jsonl',row)
                return value
        value=cache.call(kind,identity,fetch)
        validate(value)
        return value

    def score(self, tree, case):
        with self._score_lock:
            groups = list(sibling_groups(tree, case).items())
            with ThreadPoolExecutor(max_workers=self.api_workers, thread_name_prefix='rubric-api') as pool:
                # Each group generates its initial rubric once, before scoring
                # any candidate from that group. All threads finish before update.
                initial = list(pool.map(lambda group: self.initial(case, group[1][0]), groups))
                if self.reference:
                    # Reference generation stays on the coordinator, one at a
                    # time in original group order. Its cached responses are then
                    # read by online() in API threads without another GPU call.
                    from src.online_rl.integrated.continuation import derive_seed
                    from baselines.common.rubrics import public_case
                    references = {}
                    for _, nodes in groups:
                        state = state_input(case, nodes[0]); key = sha256_json(state)
                        seed = derive_seed(self.seed, key, 'online_initial_reference', 0)
                        references[(key, seed)] = self.reference.response(public_case(case), nodes[0].prefix, nodes[0].depth, seed)
                    original_reference = self.reference
                    class PreparedReference:
                        identity = original_reference.identity
                        def response(_, case_value, prefix, turn, seed):
                            # Same public state identity as state_input().
                            from baselines.common.rubrics import public_history
                            key = sha256_json({'case':public_case(case_value), 'history':public_history(prefix), 'turn':turn})
                            return references[(key, seed)]
                    online_engine = copy.copy(self)
                    online_engine.reference = PreparedReference()
                    final = list(pool.map(lambda item: online_engine.online(case, item[0][1], item[1]), zip(groups, initial)))
                else:
                    final = [(base, None) for base in initial]
                jobs = [(node, final[i][0]) for i, (_, nodes) in enumerate(groups) for node in nodes]
                with ThreadPoolExecutor(max_workers=self.grader_workers, thread_name_prefix='rubric-judge') as judges:
                    scores = list(judges.map(lambda item: self.grade(case, item[0], item[1]), jobs))
            direct = {node.node_id: value for (node, _), value in zip(jobs, scores)}
            reports = {}
            for i, (parent, nodes) in enumerate(groups):
                rubric, online = final[i]
                reports[parent] = {'state_sha256':sha256_json(state_input(case, nodes[0])),
                    'node_ids':[n.node_id for n in nodes], 'initial_rubric':initial[i],
                    'final_rubric':rubric, 'online':online,
                    'direct_rubric_rewards':{n.node_id:direct[n.node_id] for n in nodes}}
            return node_credit(**self.weights)(tree, {}, direct), {
                'engine':self.identity, 'groups':reports, 'reward_weights':self.weights,
                'rubric_credit':'direct_candidate_response_not_descendant_average'}
