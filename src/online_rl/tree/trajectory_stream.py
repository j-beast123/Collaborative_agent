"""Ready-trajectory chunk processing with a batch-wide bank journal.

The default serializes chunk transactions. The opt-in parallel protocol performs
generation and verification outside the lock, then checks newly added entries
and briefly locks the latest-bank merge. GPU requests use bounded shared actors.
"""
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from copy import copy
from dataclasses import replace
from pathlib import Path
from queue import Queue
from threading import Lock
import json
import time

from src.online_rl.integrated.records import atomic_write_json, to_dict, from_dict, sha256_json
from src.online_rl.tree import cumulative_bank as CB, offline_chunk_bank as OCB
from src.online_rl.tree.records import Tree, TrajectoryRubric, RubricAssignment, SelectionDecision
from src.online_rl.tree.witness import WitnessVerification
from src.online_rl.tree.posthoc import freeze_trajectory_input

VERSION = 'ready_trajectory_chunk8_bank_v1'
EARLY_SCORE_VERSION = 'stream_committed_bank_score_prefetch_v2'
_END = object()
_ABORT = object()


def read(path):
    return json.loads(Path(path).read_text())


def pack(built):
    result = {k: v for k, v in built.items() if k not in ('canon', 'assignments', 'selection', 'witness_objects')}
    result.update(canon={k: to_dict(v) for k, v in built['canon'].items()},
                  assignments=[to_dict(v) for v in built['assignments']],
                  selection=[to_dict(v) for v in built['selection']])
    return result


def unpack(built):
    result = dict(built)
    result.update(canon={k: from_dict(TrajectoryRubric, v) for k, v in built['canon'].items()},
                  assignments=[from_dict(RubricAssignment, v) for v in built['assignments']],
                  selection=[from_dict(SelectionDecision, v) for v in built['selection']],
                  witness_objects={v['rubric_id']: from_dict(WitnessVerification, v) for v in built['witness']})
    result['witness'] = [to_dict(w) for w in result['witness_objects'].values()]
    return result


def initialize(root, snapshot, identity, protocol=VERSION):
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    path = root / 'state.json'
    header = {'protocol': protocol, 'initial_bank_sha256': snapshot['sha256'], 'identity': identity}
    if path.exists():
        state = read(path)
        if any(state.get(k) != v for k, v in header.items()):
            raise RuntimeError('Pending trajectory stream bank/policy identity changed')
        CB.validate_snapshot(state['snapshot'])
    else:
        atomic_write_json(path, {**header, 'snapshot': snapshot, 'transactions': []})
    return str(root)


class StreamingCase:
    def __init__(self, trainer, case, state, prompt, snapshot, spec):
        if trainer.cumulative_bank.get('discovery_trajectory_cap', 48) is not None:
            raise ValueError('Trajectory streaming requires all eligible trajectories')
        self.parallel=bool(trainer.cumulative_bank.get('parallel_bank_processing'))
        self.chunk_workers=int(trainer.cumulative_bank.get('stream_chunk_workers',1))
        self.early_scoring=bool(trainer.cumulative_bank.get('stream_early_scoring'))
        self.score_workers=int(trainer.cumulative_bank.get('stream_score_workers',4))
        if self.chunk_workers < 1 or (self.chunk_workers > 1 and not self.parallel):
            raise ValueError('Concurrent chunks require parallel bank processing')
        if self.early_scoring and (not self.parallel or trainer.cumulative_bank.get('scoring_scope','top_k')!='all'):
            raise ValueError('Streaming early scoring requires parallel bank processing and scoring_scope=all')
        if self.score_workers < 1 or self.score_workers > 32:
            raise ValueError('stream_score_workers must be between 1 and 32')
        from src.online_rl.tree.bank_parallel_stream import VERSION as PARALLEL_VERSION
        self.protocol=PARALLEL_VERSION if self.parallel else VERSION
        self.trainer, self.case, self.prompt, self.initial = trainer, case, prompt, snapshot
        if self.early_scoring:
            from src.online_rl.tree.cumulative_bank import scoring_rubrics
            self.score_prefetch_rubrics=scoring_rubrics(snapshot)
            self.score_prefetch_rubric_ids=tuple(r.rubric_id for r in self.score_prefetch_rubrics)
        else:
            self.score_prefetch_rubrics=[];self.score_prefetch_rubric_ids=()
        self.state = replace(state, case_cursor=spec['case_cursor'])
        self.root, self.lock = Path(spec['root']), spec['lock']
        self.plan_path = trainer.out / 'stream_plan.json'
        self.progress_path = trainer.out / 'stream_progress.json'
        self.plan_identity = {'protocol': self.protocol, 'case_id': case['case_id'], 'policy_sha256': state.policy_sha256,
                              'prompt_sha256': state.prompt_sha256, 'initial_bank_sha256': snapshot['sha256'],
                              **({'stream_chunk_workers':self.chunk_workers} if self.chunk_workers > 1 else {})}
        if self.plan_path.exists():
            saved = read(self.plan_path)
            if saved['identity'] != self.plan_identity: raise RuntimeError('Stream case identity changed')
            self.plans = saved['chunks']
        else:
            self.plans = []
        self.queue = Queue(); self.guard = Lock(); self.seen = set(); self.results = []
        self.score_prefetch_executor = (ThreadPoolExecutor(self.score_workers,thread_name_prefix='bank-score-prefetch')
                                        if self.early_scoring else None)
        self.score_prefetch_futures=[]; self.score_prefetch_keys=set()
        self.committed_score_executor = (ThreadPoolExecutor(self.score_workers,thread_name_prefix='bank-score-committed')
                                         if self.early_scoring else None)
        self.committed_score_futures=[]
        self.orders = {'success': 0, 'failure': 0}
        self.progress = {'protocol': self.protocol, 'ready_trajectories': 0, 'rollout_done': False,
                         'chunks_committed': 0, 'stage': 'Rollout + reflection', 'stage_started': time.time()}
        self.chunk_executor = ThreadPoolExecutor(self.chunk_workers, thread_name_prefix='bank-chunk') if self.chunk_workers > 1 else None
        self.executor = ThreadPoolExecutor(1, thread_name_prefix='trajectory-bank')
        self.future = self.executor.submit(self._consume)
        self.finished = False

    def __enter__(self): return self

    def __exit__(self, exc_type, exc, tb):
        if not self.finished: self.queue.put(_ABORT)
        self.executor.shutdown(wait=True)
        if self.chunk_executor: self.chunk_executor.shutdown(wait=True)
        if self.score_prefetch_executor: self.score_prefetch_executor.shutdown(wait=True)
        if self.committed_score_executor: self.committed_score_executor.shutdown(wait=True)
        if exc_type is None and not self.finished: self.future.result()

    def _progress(self, **fields):
        with self.guard:
            self.progress.update(fields)
            atomic_write_json(self.progress_path, self.progress)

    def push(self, leaf, feedback, nodes):
        if not feedback.valid or leaf.outcome is None: return
        if nodes is None or tuple(n.node_id for n in nodes) != leaf.path:
            raise RuntimeError('Streaming leaf path nodes are incomplete')
        if self.future.done(): self.future.result()
        with self.guard:
            if leaf.leaf_id in self.seen: raise RuntimeError('Duplicate streaming trajectory')
            self.seen.add(leaf.leaf_id)
            self.progress['ready_trajectories'] = len(self.seen)
            atomic_write_json(self.progress_path, self.progress)
            if self.score_prefetch_executor and self.score_prefetch_rubrics:
                from src.online_rl.tree.scoring import _state_key
                from src.online_rl.tree.posthoc import case_view
                fresh=[]
                for node in nodes:
                    key=(_state_key(case_view(self.case),node),node.policy_sha256,self.score_prefetch_rubric_ids)
                    if key not in self.score_prefetch_keys:
                        self.score_prefetch_keys.add(key);fresh.append(node)
                if fresh:
                    from contextvars import copy_context
                    self.score_prefetch_futures.append(self.score_prefetch_executor.submit(
                        copy_context().run,self._prefetch_applicability,fresh,self.score_prefetch_rubrics))
        self.queue.put((leaf, feedback, nodes))

    def _prefetch_applicability(self,nodes,rubrics):
        from src.online_rl.tree.applicability import state_applicability
        for node in nodes:
            for rubric in rubrics:
                state_applicability(self.case,node,rubric,self.trainer.env.scorer,self.trainer.cache)

    def score_committed_delta(self, tree, before, after):
        """Score newly committed KEEP criteria on their source chunk while later chunks continue."""
        if not self.early_scoring or not after['entries'] or not tree.nodes:
            return
        previous={entry['bank_id'] for entry in before['entries']}
        new_ids={entry['bank_id'] for entry in after['entries']} - previous
        if not new_ids:
            return
        rubrics=[rubric for rubric in CB.scoring_rubrics(after) if rubric.rubric_id in new_ids]
        if not rubrics:
            return
        from contextvars import copy_context
        self._progress(stage='새 bank rubric 병렬 채점',stage_started=time.time(),
                       committed_score_rubrics=len(new_ids),committed_score_responses=len(tree.nodes))
        future=self.committed_score_executor.submit(copy_context().run,self._score_delta,tree,rubrics)
        with self.guard:
            self.committed_score_futures.append((frozenset(new_ids),future))

    def _score_delta(self,tree,rubrics):
        from src.common.live_gpu import activity
        from src.online_rl.tree.scoring import score_nodes
        with activity('bank_score.new_committed_delta'):
            return score_nodes(self.case,list(tree.nodes),rubrics,self.trainer.env.scorer,
                               self.trainer.cache,workers=1)

    def _score_committed_bank(self,tree):
        from src.online_rl.tree import cumulative_bank as CB
        from src.online_rl.tree.bank_parallel_stream import preserved
        from src.online_rl.tree.scoring import score_cumulative_nodes
        from src.online_rl.integrated.records import to_dict
        path=self.trainer.out/'stream_early_score.json'
        tree_sha=sha256_json(to_dict(tree))
        with self.lock:
            current=CB.validate_snapshot(read(self.root/'state.json')['snapshot'])
            current_ids={entry['bank_id'] for entry in current['entries']}
            with self.guard:
                prefix_futures=[future for ids,future in self.committed_score_futures if ids <= current_ids]
        prefix_rows=[row for future in prefix_futures for row in future.result()]
        if not current['entries']:
            return None
        if path.exists():
            payload=read(path)
            if (payload.get('protocol')!=EARLY_SCORE_VERSION or payload.get('case_id')!=self.case['case_id'] or
                    payload.get('policy_sha256')!=tree.policy_sha256 or payload.get('tree_sha256')!=tree_sha):
                raise RuntimeError('Saved early score belongs to another case/tree/policy')
            saved=CB.validate_snapshot(payload['snapshot']);preserved(saved,current)
        else:
            self._progress(stage='기존 rubric 사전 채점',stage_started=time.time(),
                           early_score_snapshot_rubrics=len(current['entries']))
            from src.common.live_gpu import activity
            with activity('bank_score.early_committed_prefix'):
                rows,retrieval=score_cumulative_nodes(self.case,list(tree.nodes),current,self.trainer.env.scorer,
                    self.trainer.cache,top_k=int(self.trainer.cumulative_bank.get('retrieval_top_k',8)),
                    workers=self.trainer.api_workers,scoring_scope='all',prefetched_scores=prefix_rows)
            payload={'protocol':EARLY_SCORE_VERSION,'case_id':self.case['case_id'],
                     'policy_sha256':tree.policy_sha256,'tree_sha256':tree_sha,
                     'snapshot':current,'scores':[to_dict(row) for row in rows],'retrieval':retrieval}
            atomic_write_json(path,payload)
        return {'protocol':EARLY_SCORE_VERSION,'file':path.name,'sha256':sha256_json(payload),
                'snapshot_sha256':payload['snapshot']['sha256'],'rubrics':len(payload['snapshot']['entries']),
                'responses':len(tree.nodes)}

    def finish(self,tree=None):
        self._progress(rollout_done=True)
        self.queue.put(_END)
        score_pool=None;score_future=None
        prefetch_stopped=False
        if self.early_scoring:
            if tree is None:raise RuntimeError('Early cumulative scoring needs the completed rollout tree')
            score_pool=ThreadPoolExecutor(1,thread_name_prefix='early-cumulative-score')
            from contextvars import copy_context
            score_future=score_pool.submit(copy_context().run,self._score_committed_bank,tree)
            self.score_prefetch_executor.shutdown(wait=True)
            prefetch_stopped=True
        try:
            for future in self.score_prefetch_futures:future.result()
            self.future.result()
            if score_future and not score_future.done():
                self._progress(stage='기존 rubric 사전 채점',stage_started=time.time())
            early_score=score_future.result() if score_future else None
            committed_rows=[]
            for _ids,future in self.committed_score_futures:
                committed_rows.extend(future.result())
            if committed_rows:
                from src.online_rl.integrated.records import to_dict
                path=self.trainer.out/'stream_early_score.json'
                payload=read(path) if path.exists() else {
                    'protocol':EARLY_SCORE_VERSION,'case_id':self.case['case_id'],
                    'policy_sha256':tree.policy_sha256,'tree_sha256':sha256_json(to_dict(tree)),
                    'snapshot':self.initial,'scores':[],'retrieval':[]}
                if (payload.get('protocol')!=EARLY_SCORE_VERSION or payload.get('case_id')!=self.case['case_id'] or
                        payload.get('policy_sha256')!=tree.policy_sha256 or
                        payload.get('tree_sha256')!=sha256_json(to_dict(tree)) or
                        payload.get('snapshot',{}).get('sha256')!=self.initial['sha256']):
                    raise RuntimeError('Saved early score belongs to another case/tree/policy/bank prefix')
                by_pair={(row['response_node_id'],row['rubric_id']):row for row in payload.get('scores',[])}
                for row in committed_rows:
                    raw=to_dict(row); key=(row.response_node_id,row.rubric_id)
                    if key in by_pair and by_pair[key]!=raw:
                        raise RuntimeError(f'Conflicting prefetched score for {key[0]}/{key[1]}')
                    by_pair[key]=raw
                payload['scores']=[by_pair[key] for key in sorted(by_pair)]
                payload['committed_delta_score_pairs']=len(committed_rows)
                atomic_write_json(path,payload)
                early_score=early_score or {'protocol':EARLY_SCORE_VERSION,'file':path.name,
                    'snapshot_sha256':self.initial['sha256'],'rubrics':len(self.initial['entries']),
                    'responses':len(tree.nodes)}
                early_score={**early_score,'sha256':sha256_json(payload),
                             'committed_delta_score_pairs':len(committed_rows)}
            if early_score:self._progress(stage='Bank + 사전 채점 완료',stage_started=time.time())
        finally:
            if self.score_prefetch_executor and not prefetch_stopped:
                self.score_prefetch_executor.shutdown(wait=True)
            if self.committed_score_executor:self.committed_score_executor.shutdown(wait=True)
            if score_pool:score_pool.shutdown(wait=True)
        if early_score:self.early_score=early_score
        self.finished = True
        return self._aggregate()

    def _consume(self):
        if self.chunk_executor: return self._consume_parallel()
        pending = {}; index = 0; done = False
        while True:
            item = self.queue.get()
            if item is _ABORT: return
            if item is _END: done = True
            else: pending[item[0].leaf_id] = item
            while pending:
                if index < len(self.plans):
                    ids = self.plans[index]
                    if not all(i in pending for i in ids): break
                elif len(pending) >= 8 or done:
                    ids = list(pending)[:8]
                    self.plans.append(ids)
                    atomic_write_json(self.plan_path, {'identity': self.plan_identity, 'chunks': self.plans})
                else: break
                self.results.append(self._transaction(index, [pending.pop(i) for i in ids]))
                self.orders = self.results[-1]['order_next']; index += 1
                self._progress(chunks_committed=index, stage='Rollout + reflection' if not done else 'Bank 완료',
                               stage_started=time.time())
            if done:
                if pending or index != len(self.plans): raise RuntimeError('Resumed stream chunk coverage changed')
                return

    def _consume_parallel(self):
        pending = {}; index = 0; futures = {}; results = {}
        def collect(block=False):
            if not futures: return
            ready = wait(futures, return_when=FIRST_COMPLETED).done if block else {f for f in futures if f.done()}
            for future in ready:
                chunk = futures.pop(future)
                results[chunk] = future.result()
                with self.guard:
                    self.progress.setdefault('active_chunks', {}).pop(str(chunk + 1), None)
                    self.progress['chunks_committed'] = len(results)
                    atomic_write_json(self.progress_path, self.progress)
        done = False
        while not done:
            item = self.queue.get()
            if item is _ABORT: return
            if item is _END: done = True
            else: pending[item[0].leaf_id] = item
            collect()
            while pending:
                if index < len(self.plans):
                    ids = self.plans[index]
                    if not all(i in pending for i in ids): break
                elif len(pending) >= 8 or done:
                    ids = list(pending)[:8]; self.plans.append(ids)
                    atomic_write_json(self.plan_path, {'identity':self.plan_identity,'chunks':self.plans})
                else: break
                while len(futures) >= self.chunk_workers: collect(True)
                owner = copy(self)
                # Reserve disjoint assignment ranges before submission; completion order cannot change identities.
                owner.orders = {'success':index * 8,'failure':index * 8}
                def progress(chunk=index, **fields):
                    with self.guard:
                        active = self.progress.setdefault('active_chunks', {})
                        active.setdefault(str(chunk + 1), {}).update(fields)
                        self.progress.update(fields)
                        atomic_write_json(self.progress_path,self.progress)
                owner._progress = progress
                progress(stage='Chunk 처리 준비',stage_started=time.time(),current_chunk=index+1)
                futures[self.chunk_executor.submit(owner._transaction,index,[pending.pop(i) for i in ids])] = index
                index += 1
            if done and (pending or index != len(self.plans)):
                raise RuntimeError('Resumed stream chunk coverage changed')
        while futures: collect(True)
        self.results = [results[i] for i in range(index)]
        self.orders = {layer:max((r['order_next'][layer] for r in self.results),default=0) for layer in ('success','failure')}
        # Working snapshot must reflect commit order rather than chunk index order.
        with self.lock: self.final_snapshot = CB.validate_snapshot(read(self.root/'state.json')['snapshot'])
        self._progress(stage='Bank 완료',stage_started=time.time())

    def _transaction(self, index, items):
        receipt_id = f'case_{self.state.case_cursor:06d}_chunk_{index:04d}'
        receipt_path = self.root / 'receipts' / (receipt_id + '.json')
        leaves = tuple(i[0] for i in items); feedback = [i[1] for i in items]
        nodes = {n.node_id: n for i in items for n in i[2]}
        ro = self.trainer.cfg['rollout']
        tree = Tree(self.case['case_id'], self.state.policy_sha256, ro['rounds'], ro['branching'], tuple(nodes.values()), leaves,
                    ro.get('budget', 'initial_fanout'))
        tins = [freeze_trajectory_input(self.case, l, f, include_locus=self.trainer.source_consistency)
                for l, f in zip(leaves, feedback)]
        input_identity = sha256_json({'case': self.plan_identity, 'tins': [to_dict(t) for t in tins],
                                     'nodes': [to_dict(n) for n in sorted(nodes.values(), key=lambda n: n.node_id)],
                                     'orders': self.orders, 'chunk': index})
        if self.parallel:
            from src.online_rl.tree.bank_parallel_stream import transaction
            return transaction(self,index,input_identity,tree,feedback,tins)
        self._progress(stage='Bank lock 대기', stage_started=time.time(), current_chunk=index + 1)
        with self.lock:
            state = read(self.root / 'state.json'); before = CB.validate_snapshot(state['snapshot'])
            receipt = read(receipt_path) if receipt_path.exists() else None
            if receipt and receipt['input_identity'] != input_identity:
                raise RuntimeError('Saved stream chunk inputs changed')
            if receipt_id in state['transactions']:
                if receipt is None: raise RuntimeError('Committed stream receipt is missing')
                return unpack(receipt['built'])
            self._progress(stage='Rubric 생성·Gate·verification', stage_started=time.time())
            if receipt is None or receipt['input_bank_sha256'] != before['sha256']:
                built = OCB.process_case(self.trainer, case=self.case, state=self.state, tree=tree, feedback=feedback,
                    ruled=list(leaves), tin_of={t.leaf_id: t for t in tins}, prompt_text=self.prompt,
                    snapshot=before, chunk_offset=index, order_offsets=self.orders)
                receipt = {'protocol': VERSION, 'id': receipt_id, 'input_identity': input_identity,
                           'input_bank_sha256': before['sha256'], 'output_bank_sha256': built['working']['sha256'],
                           'built': pack(built)}
                atomic_write_json(receipt_path, receipt)
            else:
                built = unpack(receipt['built'])
            state['snapshot'] = built['working']; state['transactions'].append(receipt_id)
            atomic_write_json(self.root / 'state.json', state)
            self.score_committed_delta(tree,before,built['working'])
            return built

    def _aggregate(self):
        result = {'working': self.initial, 'canon': {}, 'assignments': [], 'witness': [], 'witness_objects': {},
                  'selection': [], 'cumulative_reuse': [], 'decisions': [], 'order_next': self.orders}
        reports = []; uncovered = []
        for built in self.results:
            result['working'] = built['working']
            for key in ('canon', 'witness_objects'): result[key].update(built[key])
            for key in ('assignments', 'witness', 'selection', 'cumulative_reuse', 'decisions'): result[key].extend(built[key])
            reports.extend(built['summary']['chunks']); uncovered.extend(built['summary']['uncovered_ids'])
        if hasattr(self,'final_snapshot'): result['working']=self.final_snapshot
        selected = [i for group in self.plans for i in group]
        if len(selected) != len(set(selected)) or set(selected) != self.seen:
            raise RuntimeError('Streaming must process every eligible trajectory exactly once')
        result['summary'] = {'protocol': OCB.VERSION, 'trajectory_streaming': VERSION, 'eligible': len(selected),
            'cap': None, 'sampled': False, 'sampling_seed': None, 'selected_ids': selected, 'not_selected_ids': [],
            'uncovered_ids': sorted(uncovered), 'chunks': reports, 'generator_calls': len(reports),
            **{k: sum(r[k] for r in reports) for k in ('generator_api_requests', 'generator_correction_requests', 'peer_matching_calls')},
            'input_bank_sha256': self.initial['sha256'], 'output_bank_sha256': result['working']['sha256']}
        if self.parallel:
            result['summary']['parallel_bank_protocol']=self.protocol
            result['summary']['semantic_aliases']={key:alias for b in self.results for key,alias in b['summary'].get('semantic_aliases',{}).items()}
        if getattr(self,'early_score',None):result['summary']['early_cumulative_scoring']=self.early_score
        return result


def audit(root, before):
    state = read(Path(root) / 'state.json')
    if state['protocol'] != VERSION:
        from src.online_rl.tree.bank_parallel_stream import audit as parallel_audit, VERSION as PARALLEL_VERSION
        if state['protocol'] != PARALLEL_VERSION:raise RuntimeError('Unknown stream journal protocol')
        return parallel_audit(root,before)
    tail = before['sha256']; decisions = []
    if len(state['transactions']) != len(set(state['transactions'])): raise RuntimeError('Duplicate chunk commit')
    for receipt_id in state['transactions']:
        receipt = read(Path(root) / 'receipts' / (receipt_id + '.json'))
        if receipt['input_bank_sha256'] != tail: raise RuntimeError('Stream bank receipt chain is broken')
        for row in receipt['built']['decisions']:
            if row['input_bank_sha256'] != tail: raise RuntimeError('Stream decision chain is broken')
            if row['action'] != 'GENERATED_KEEP' and row['output_bank_sha256'] != tail:
                raise RuntimeError('Only KEEP may change the bank')
            tail = row['output_bank_sha256']; decisions.append(row)
        if tail != receipt['output_bank_sha256']: raise RuntimeError('Stream receipt output differs')
        CB.validate_snapshot(receipt['built']['working'])
    after = CB.validate_snapshot(state['snapshot'])
    if tail != after['sha256']: raise RuntimeError('Stream bank journal does not reach final bank')
    return after, decisions
