"""Batch barrier and case-local prompt optimization for training tree rollouts."""
from dataclasses import dataclass
from pathlib import Path
from src.online_rl.integrated.records import atomic_write_json, from_dict, to_dict
from src.online_rl.tree.records import Tree
from src.online_rl.tree.prompt_search import EvidenceUnit, ScoringDeps, run_event, score_prompt, objective, unit_values
from src.online_rl.tree.effectiveness import select
from src.online_rl.tree.scoring import score_nodes
from src.online_rl.tree.reward import agent_targets


@dataclass(frozen=True)
class CaseSlot:
    index: int
    worker: int
    lane: int


def batch_slots(start, remaining, workers=4, cases_per_worker=3):
    if workers < 1 or cases_per_worker < 1:
        raise ValueError('Positive worker and lane counts required')
    return [CaseSlot(start+i, i//cases_per_worker, i%cases_per_worker)
            for i in range(min(remaining, workers*cases_per_worker))]


def sibling_groups(tree, targets):
    """Keep each case/state distinct even when batches contain root siblings."""
    nodes={n.node_id:n for n in tree.nodes};groups=[]
    for parent,ids in sorted(targets['groups'].items()):
        records=[{'trajectory_id':nid, 'case_id':f'{tree.case_id}|{parent}',
                  'reward':targets['nodes'][nid]['reward'], 'loss_segments':targets.get('loss_segments'),
                  'rollout_policy_sha256':nodes[nid].policy_sha256,
                  'agent_traces':[nodes[nid].trace]} for nid in ids]
        from src.online_rl.integrated.episode import assert_single_prompt
        assert_single_prompt(records);groups.append(records)
    return groups


def update_batch(updater, records, expected_policy, actual_policy):
    """One optimizer call only after all cases and prompt comparisons finish."""
    if actual_policy()!=expected_policy:
        raise RuntimeError('Policy changed before the batch barrier')
    groups=[]
    for r in records:
        tree=from_dict(Tree,r['tree'])
        if tree.policy_sha256!=expected_policy:
            raise RuntimeError('Batch contains a different source policy')
        groups.extend(sibling_groups(tree,r['agent_targets']))
    return updater.update_groups(groups)


def optimize_case(*, case, tree, inputs, prompt_id, prompt_text, store, backend, deps,
                  cfg, event, case_index, history, out_dir):
    """Compare prompts on one immutable tree; score each prompt's retained rubrics.
    Acceptance decides inheritance. No outcome from another case is mixed here.
    """
    nodes={n.node_id:n for n in tree.nodes}; tins={t.leaf_id:t for t in inputs}
    units=[EvidenceUnit(case['case_id'],leaf,tuple(nodes[n] for n in leaf.path),
                        tins[leaf.leaf_id],tree.policy_sha256)
           for leaf in tree.leaves if leaf.leaf_id in tins]
    sha=store.put(prompt_text)
    if cfg['prompt_update']['enabled']:
        result=run_event(event,prompt_id,sha,units,{case['case_id']:case},store,backend,
                         deps,cfg,cfg['schedule']['seed'],case_index,history)
    else:
        result={'event':event,'versions':[], 'decision':{'accepted':False,'reason':'disabled'}}
    versions={sha:prompt_id}
    for version in result['versions']:
        if version.sha256 and store.dir.joinpath(version.sha256+'.txt').exists():
            versions[version.sha256]=version.prompt_id
    evaluations=[]
    for psha,pid in versions.items():
        res=score_prompt(store.get(psha),psha,units,{case['case_id']:case},deps)
        rubrics=[r['rubric'] for r in res.values()]
        effects=[r['effectiveness'] for r in res.values() if r['effectiveness'] is not None]
        if any(e.status=='incomplete' for e in effects):
            raise RuntimeError('Technical rerun incomplete; keep evidence and retry before updating')
        selection=select(effects,cfg['selection']);kept={s.rubric_id for s in selection if s.retained}
        scores=score_nodes(case,list(tree.nodes),[r for r in rubrics if r.rubric_id in kept],deps.scorer,deps.cache)
        weights={e.rubric_id:e.R_O for e in effects if e.rubric_id in kept}
        targets=agent_targets(tree,scores,weights,deps.strategies)
        evaluation={'prompt_id':pid,'prompt_sha256':psha,
                    'objective':objective(unit_values(res,deps.strategies),{k:r['source_outcome'] for k,r in res.items()}),
                    'trajectory_rubrics':[to_dict(r) for r in rubrics],
                    'effectiveness':[to_dict(e) for e in effects], 'selection':[to_dict(s) for s in selection],
                    'response_scores':[to_dict(s) for s in scores], 'rubric_weights':weights,'agent_targets':targets}
        atomic_write_json(Path(out_dir)/'prompt_evaluations'/f'{psha}.json',evaluation)
        evaluations.append(evaluation)
    d=result['decision'];winner_sha=d['candidate_sha256'] if d.get('accepted') else sha
    winner=next(e for e in evaluations if e['prompt_sha256']==winner_sha)
    event_record={**result,'versions':[to_dict(v) for v in result['versions']]}
    atomic_write_json(Path(out_dir)/'prompt_event.json',event_record)
    return {'case_cursor':case_index,'case_id':case['case_id'],'policy_sha256':tree.policy_sha256,
            'tree':to_dict(tree),'trajectory_inputs':[to_dict(t) for t in inputs],
            **winner,'prompt_event':event_record,'next_prompt_text':store.get(winner_sha),
            'candidate_summary':[{'prompt_id':e['prompt_id'],'sha256':e['prompt_sha256'],
                                  'objective':e['objective'],'retained':sum(s['retained'] for s in e['selection'])}
                                 for e in evaluations]}
