"""Generate/verify outside the bank lock, compare append-only delta, then merge.

Drafts pin the discovery snapshot and verified result across restarts. Receipts
record discovery and commit separately; a commit never replaces the latest bank
with the candidate's older speculative output snapshot.
"""
from pathlib import Path
import time
from src.online_rl.integrated.records import atomic_write_json, sha256_json
from src.online_rl.tree import cumulative_bank as CB, offline_chunk_bank as OCB, bank_delta as D

VERSION = D.VERSION


def preserved(before, after):
    entries={e['bank_id']:e for e in after['entries']}
    for old in before['entries']:
        new=entries.get(old['bank_id'])
        if new is None or sha256_json(new['rubric'])!=sha256_json(old['rubric']) or any(sha256_json(p) not in {sha256_json(q) for q in new['sources']} for p in old['sources']):
            raise RuntimeError('Parallel bank must preserve all prior criteria and proofs')


def transaction(owner, index, identity, tree, feedback, tins):
    from src.online_rl.tree.trajectory_stream import read, pack, unpack
    receipt_id=f'case_{owner.state.case_cursor:06d}_chunk_{index:04d}'
    receipt_path=owner.root/'receipts'/(receipt_id+'.json')
    input_path=owner.root/'drafts'/(receipt_id+'.input.json')
    built_path=owner.root/'drafts'/(receipt_id+'.built.json')
    owner._progress(stage='Snapshot lock 대기',stage_started=time.time(),current_chunk=index+1)
    with owner.lock:
        journal=read(owner.root/'state.json')
        current=CB.validate_snapshot(journal['snapshot'])
        receipt=read(receipt_path) if receipt_path.exists() else None
        if receipt and receipt['input_identity']!=identity:raise RuntimeError('Parallel chunk input changed')
        if receipt_id in journal['transactions']:
            if receipt is None:raise RuntimeError('Committed parallel receipt is missing')
            built=unpack(receipt['built'])
            owner.score_committed_delta(tree,receipt['commit_input_snapshot'],receipt['commit_snapshot'])
            return built
        if receipt and receipt['commit_input_bank_sha256']==current['sha256']:
            # Receipt was durable before the state write; apply it exactly once.
            after=CB.validate_snapshot(receipt['commit_snapshot']);preserved(current,after)
            journal['snapshot']=after;journal['transactions'].append(receipt_id)
            atomic_write_json(owner.root/'state.json',journal)
            built=unpack(receipt['built'])
            owner.score_committed_delta(tree,current,after)
            return built
        discovery=current
    if input_path.exists():
        saved=read(input_path)
        if saved['input_identity']!=identity:raise RuntimeError('Saved parallel discovery input changed')
        discovery=CB.validate_snapshot(saved['snapshot'])
    else:
        atomic_write_json(input_path,{'protocol':VERSION,'input_identity':identity,'snapshot':discovery})
    owner._progress(stage='Rubric 생성·Gate·verification (병렬)',stage_started=time.time())
    if built_path.exists():
        saved=read(built_path)
        if saved['input_identity']!=identity or saved['discovery_bank_sha256']!=discovery['sha256']:
            raise RuntimeError('Saved parallel verification input changed')
        built=unpack(saved['built'])
    else:
        built=OCB.process_case(owner.trainer,case=owner.case,state=owner.state,tree=tree,feedback=feedback,
            ruled=list(tree.leaves),tin_of={t.leaf_id:t for t in tins},prompt_text=owner.prompt,
            snapshot=discovery,chunk_offset=index,order_offsets=owner.orders)
        atomic_write_json(built_path,{'input_identity':identity,'discovery_bank_sha256':discovery['sha256'],'built':pack(built)})
    verified=D.candidates(built);aliases={};compared={e['bank_id'] for e in discovery['entries']};comparisons=[]
    generated_sha=built['working']['sha256']
    commit_started=time.monotonic()
    while True:
        with owner.lock:
            current=CB.validate_snapshot(read(owner.root/'state.json')['snapshot'])
        preserved(discovery,current)
        current_by_hash={e['content_sha256']:e for e in current['entries']}
        pending={key:rub for key,rub in verified.items() if key not in aliases and key not in current_by_hash}
        delta=[e for e in current['entries'] if e['bank_id'] not in compared]
        if pending and delta:
            owner._progress(stage='추가 rubric 중복 비교 (Gemini)',stage_started=time.time())
            result=D.compare(owner.trainer,pending,delta)
            aliases.update(result['aliases'])
            comparisons.append({'delta_bank_ids':[e['bank_id'] for e in delta],
                                'candidate_content_sha256s':sorted(pending),'result':result})
        compared.update(e['bank_id'] for e in current['entries'])
        owner._progress(stage='Merge lock 대기',stage_started=time.time())
        with owner.lock:
            journal=read(owner.root/'state.json');latest=CB.validate_snapshot(journal['snapshot'])
            if receipt_id in journal['transactions']:
                return unpack(read(receipt_path)['built'])
            # A concurrent commit during the API call is checked outside the lock.
            if any(e['bank_id'] not in compared for e in latest['entries']):continue
            preserved(discovery,latest)
            entries={e['bank_id']:e for e in latest['entries']}
            for key,alias in aliases.items():D.validate_alias(alias,key,entries)
            records=D.merge_records(built,owner.case,owner.state,aliases)
            after=CB.extend_snapshot(latest,records) if records[0]['selection'] else latest
            added=len(after['entries'])-len(latest['entries'])
            built['working']=after
            built['summary']['semantic_aliases']=aliases
            built['summary']['parallel_bank_protocol']=VERSION
            report=built['summary']['chunks'][0]
            report.update(commit_input_bank_sha256=latest['sha256'],commit_output_bank_sha256=after['sha256'],
                          commit_added_unique=added,semantic_duplicates=len(aliases),delta_comparisons=len(comparisons),
                          commit_wall_seconds=round(time.monotonic()-commit_started,1))
            receipt={'protocol':VERSION,'id':receipt_id,'input_identity':identity,
                     'discovery_snapshot':discovery,'input_bank_sha256':discovery['sha256'],
                     'generated_bank_sha256':generated_sha,'commit_input_snapshot':latest,
                     'commit_input_bank_sha256':latest['sha256'],'output_bank_sha256':after['sha256'],
                     'commit_snapshot':after,'merge_identity':{'case_id':owner.case['case_id'],'case_cursor':owner.state.case_cursor,
                         'policy_sha256':owner.state.policy_sha256,'prompt_sha256':owner.state.prompt_sha256},'semantic_aliases':aliases,'delta_comparisons':comparisons,'built':pack(built)}
            atomic_write_json(receipt_path,receipt)
            journal['snapshot']=after;journal['transactions'].append(receipt_id)
            atomic_write_json(owner.root/'state.json',journal)
            owner.score_committed_delta(tree,latest,after)
            return built


def audit(root, before):
    from src.online_rl.tree.trajectory_stream import read, unpack
    root=Path(root);journal=read(root/'state.json');tail=before;decisions=[]
    if len(journal['transactions'])!=len(set(journal['transactions'])):raise RuntimeError('Duplicate parallel commit')
    for receipt_id in journal['transactions']:
        receipt=read(root/'receipts'/(receipt_id+'.json'))
        if receipt['protocol']!=VERSION or receipt['commit_input_snapshot']!=tail:
            raise RuntimeError('Parallel commit journal chain is broken')
        discovery=CB.validate_snapshot(receipt['discovery_snapshot']);preserved(discovery,tail)
        built=unpack(receipt['built']);local_tail=discovery['sha256']
        for row in built['decisions']:
            if row['input_bank_sha256']!=local_tail:raise RuntimeError('Parallel discovery decision chain is broken')
            if row['action']!='GENERATED_KEEP' and row['output_bank_sha256']!=local_tail:
                raise RuntimeError('Only a verified candidate can advance speculative bank')
            local_tail=row['output_bank_sha256'];decisions.append(row)
        if local_tail!=receipt['generated_bank_sha256']:raise RuntimeError('Parallel speculative output differs')
        aliases=receipt['semantic_aliases'];entries={e['bank_id']:e for e in tail['entries']}
        candidates=D.candidates(built)
        for key,alias in aliases.items():
            if key not in candidates:raise RuntimeError('Alias must belong to a verified KEEP candidate')
            D.validate_alias(alias,key,entries)
        from types import SimpleNamespace
        # Identity is carried by the generated record in the immutable draft.
        draft=read(root/'drafts'/(receipt_id+'.input.json'))
        generated=read(root/'drafts'/(receipt_id+'.built.json'))
        if draft['snapshot']!=discovery or generated['input_identity']!=receipt['input_identity']:
            raise RuntimeError('Parallel receipt differs from its pinned draft')
        original=generated['built']
        if generated['discovery_bank_sha256']!=discovery['sha256'] or original['working']['sha256']!=receipt['generated_bank_sha256']:
            raise RuntimeError('Parallel receipt changed its verified speculative bank')
        for key in ('canon','assignments','selection','witness','cumulative_reuse','decisions','order_next'):
            if sha256_json(original[key])!=sha256_json(receipt['built'][key]):
                raise RuntimeError('Parallel commit changed immutable verified evidence')
        # Reconstruct the original merge identity from the receipt's case proof fields.
        info=receipt.get('merge_identity')
        if info is None:raise RuntimeError('Parallel receipt lacks merge identity')
        records=D.merge_records(built,{'case_id':info['case_id']},SimpleNamespace(**info),aliases)
        expected=CB.extend_snapshot(tail,records) if records[0]['selection'] else tail
        after=CB.validate_snapshot(receipt['commit_snapshot'])
        if sha256_json(expected)!=sha256_json(after) or sha256_json(built['working'])!=sha256_json(after) or receipt['output_bank_sha256']!=after['sha256']:
            raise RuntimeError('Parallel commit does not equal the append-only verified merge')
        tail=after
    if tail!=journal['snapshot']:raise RuntimeError('Parallel journal does not reach final bank')
    return tail,decisions
