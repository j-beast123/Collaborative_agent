from __future__ import annotations
import copy, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
from src.online_rl.integrated.records import sha256_json
from src.online_rl.tree.cumulative_bank import content_hash, validate_snapshot

RUN=Path('outputs/offline_bank90_from_c30_overlap9_v2_20261012/r1_o0__p0fixed__gemma3doctor__seed0__offline_bank90_from_c30_overlap9_v2')
SOURCE=RUN/'batches/batch_000012/bank_frozen.json'
REVISED=RUN/'bank_revised_mc_v1.json'
OUT=Path('data/interim/medmcqa/revised_bank13_20261012/revised_bank13_snapshot.json')

def main():
    source=json.loads(SOURCE.read_text()); revised=json.loads(REVISED.read_text())
    by_old={e.get('supersedes_rubric_id'):e for e in revised['entries'] if e.get('supersedes_rubric_id')}
    entries=[]
    for entry in source['entries']:
        full=copy.deepcopy(entry['rubric']); change=by_old.get(full['rubric_id'])
        if change:
            public=copy.deepcopy(change['rubric']); rid=change['rubric_id']; public['rubric_id']=rid
            full['rubric_id']=rid; full['rubric']=public
            rubric_sha=sha256_json(public)
            for proof in entry['sources']:
                proof=copy.deepcopy(proof); proof['rubric']=copy.deepcopy(full)
                witness=proof['witness']; witness['rubric_id']=rid
                for tested in witness.get('tested',[]):
                    spec=tested.get('intervention_spec')
                    if spec: spec.update(canonical_rubric_id=rid,rubric_sha256=rubric_sha)
                    for run in tested.get('rerun_records',[]):
                        run['rubric_id']=rid
                        if run.get('intervention_spec'):
                            run['intervention_spec'].update(canonical_rubric_id=rid,rubric_sha256=rubric_sha)
                proof['curated_revision']={'protocol':'manual_curated_revision_v1','supersedes':change['supersedes_rubric_id'],
                    'note':'Public criterion wording is treated as the experiment source bank; hashes are resealed.'}
                # The existing validator permits only its established proof fields.
                proof.pop('curated_revision')
                sources=[proof]
        else:
            sources=copy.deepcopy(entry['sources'])
        key=content_hash(full['rubric'])
        entries.append({'bank_id':'bank_'+key,'content_sha256':key,'rubric':full,'sources':sources})
    entries.sort(key=lambda e:e['bank_id'])
    payload={'protocol':'cumulative_keep_bank_v1','version':source['version']+1,
             'parent_sha256':source['sha256'],'entries':entries}
    snapshot={**payload,'sha256':sha256_json(payload)}
    validate_snapshot(snapshot)
    OUT.parent.mkdir(parents=True,exist_ok=True); OUT.write_text(json.dumps(snapshot,ensure_ascii=False,indent=2)+'\n')
    print(OUT, snapshot['sha256'], len(entries))
if __name__=='__main__': main()
