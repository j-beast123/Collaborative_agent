"""One P0/P1 comparison per batch, before the single Agent update."""
import dataclasses
from src.online_rl.integrated.records import from_dict, sha256_json, to_dict
from src.online_rl.tree import rubric_bank as RB
from src.online_rl.tree.prompt_search import EvidenceUnit, directional_scores, reflection_view
from src.online_rl.tree.records import Tree, TrajectoryRubric, TrajectoryRubricInput, RubricAssignment, ResponseRubricScore
from src.online_rl.tree.reward import agent_targets, interaction_diagnostics, sibling_group_diagnostics
from src.online_rl.tree.scoring import score_nodes

MODE = 'four_case_p0_p1_v1'
FIVE_MODE = 'five_case_p0_p1_v1'


def evaluate_prompt_record(trainer, record):
    """Score prompt effectiveness from the witness reruns already performed.

    Do not launch a second all-member rerun pass. Members reached before the witness
    stop (or the random-10 cap) contribute their observed target-hit fraction;
    known invalid/no-match rubrics contribute zero. Unexamined members are omitted.
    """
    tree = from_dict(Tree, record['tree'])
    nodes = {n.node_id:n for n in tree.nodes}
    leaves = {l.leaf_id:l for l in tree.leaves}
    tins = {t['leaf_id']:from_dict(TrajectoryRubricInput,t) for t in record['trajectory_inputs']}
    canon = {r['rubric_id']:from_dict(TrajectoryRubric,r) for r in record['trajectory_rubrics']}
    assignments = [from_dict(RubricAssignment,a) for a in record['rubric_bank']['assignments']]
    witness = record['rubric_bank']['witness']
    tested_by_leaf={}
    for item in witness:
        for test in item.get('tested',[]):
            if test.get('reruns',0):tested_by_leaf[test['leaf_id']]=(item,test)
            elif test.get('result') in ('no_match','no_locus'):
                tested_by_leaf[test['leaf_id']]=(item,test)
    members={a.canonical_rubric_id:[x.leaf_id for x in assignments if x.canonical_rubric_id==a.canonical_rubric_id]
             for a in assignments}
    units=[]; results={}; sampled_values={}; sampled_outcomes={}; effects=[]
    for a in assignments:
        raw=canon[a.canonical_rubric_id]
        tested=tested_by_leaf.get(a.leaf_id)
        if raw.valid and tested is None:continue
        leaf=leaves[a.leaf_id]
        unit=EvidenceUnit(tree.case_id,leaf,tuple(nodes[n] for n in leaf.path),tins[a.leaf_id],tree.policy_sha256)
        units.append(unit)
        eff=None
        if raw.valid:
            witness_row,test=tested
            rerun_count=test.get('reruns',0)
            if rerun_count:
                score=test['positive']/rerun_count
                run_rows=[type('ObservedRun',(),{'node_id':test['node_id'],'turn':test['target_reply_id'],
                                                   'level':test['level'],'intervention_text':test.get('intervention_reply',''),
                                                   'outcome':outcome})() for outcome in test.get('outcomes',[])]
                eff=type('ObservedEffect',(),{'status':'single_target','level':test['level'],
                    'target_node_ids':(test['node_id'],),'runs':tuple(run_rows),
                    'per_node_R_O':{test['node_id']:score},'R_O':score})()
                effects.append({'rubric_id':raw.rubric_id,'leaf_id':a.leaf_id,'R_O':score,
                                'reruns':rerun_count,'result':test.get('result')})
            else:
                eff=type('ObservedEffect',(),{'status':'no_match','level':None,'target_node_ids':(),
                    'runs':(),'per_node_R_O':{},'R_O':None})()
        rub=dataclasses.replace(raw,leaf_id=a.leaf_id)
        results[a.leaf_id]={'rubric':rub,'effectiveness':eff,'assignment':a,
                            'members':members.get(a.canonical_rubric_id,[])}
        sampled_values[a.leaf_id]=trainer.st.need('prompt_unit_value')({'rubric':rub,'effectiveness':eff})
        sampled_outcomes[a.leaf_id]=leaf.outcome
    scores={}
    for name,outcome in (('success',1.0),('failure',0.0)):
        side=[v for k,v in sampled_values.items() if sampled_outcomes[k]==outcome]
        scores['n_'+name]=len(side);scores['S_'+name]=sum(side)/len(side) if side else None
    scores['J']=None if scores['S_success'] is None or scores['S_failure'] is None else scores['S_success']*scores['S_failure']
    scores.update(invalid=sum(not canon[a.canonical_rubric_id].valid for a in assignments),
                  no_match=sum(results[k]['effectiveness'] is not None and results[k]['effectiveness'].status=='no_match' for k in results),
                  per_trajectory=sampled_values,source_outcome=sampled_outcomes)
    return {**scores, 'reflection':reflection_view(units,results),
            'effectiveness':effects,
            'rerun_policy_sha256':tree.policy_sha256,
            'aggregation':'mean_sampled_witness_effect_per_direction_product_v1',
            'effectiveness_source':'reuse_witness_reruns_no_additional_member_reruns_v1'}


def aggregate(records):
    values={}; outcomes={}
    for r in records:
        ev=r['prompt_evaluation']
        if ev['rerun_policy_sha256'] != r['policy_sha256']:
            raise RuntimeError('Prompt evaluation used a different rerun policy')
        for leaf,value in ev['per_trajectory'].items():
            if leaf in values: raise RuntimeError('Duplicate prompt-evaluation trajectory')
            values[leaf]=value; outcomes[leaf]=ev['source_outcome'][leaf]
    out={'per_trajectory':values,'source_outcome':outcomes}
    for name,source in (('success',1.0),('failure',0.0)):
        side=[values[k] for k in values if outcomes[k]==source]
        out['n_'+name]=len(side)
        out['S_'+name]=sum(side)/len(side) if side else None
    out['J']=None if out['S_success'] is None or out['S_failure'] is None else out['S_success']*out['S_failure']
    return out


def choose(parent_records, candidate_records):
    if len(parent_records)!=len(candidate_records): raise RuntimeError('P0/P1 case masks differ')
    for p,c in zip(parent_records,candidate_records):
        if p['case_id']!=c['case_id'] or p['policy_sha256']!=c['policy_sha256'] or sha256_json(p['tree'])!=sha256_json(c['tree']):
            raise RuntimeError('P0/P1 must share the exact tree and frozen Agent policy')
    p,c=aggregate(parent_records),aggregate(candidate_records)
    # P0 and P1 can stop on different witnesses. Compare their paired intersection
    # so neither side receives credit for a trajectory the other side did not test.
    common=set(p['per_trajectory']) & set(c['per_trajectory'])
    if any(p['source_outcome'][k]!=c['source_outcome'].get(k) for k in common):
        raise RuntimeError('P0/P1 sampled trajectory outcomes differ')
    for summary in (p,c):
        summary['per_trajectory']={k:v for k,v in summary['per_trajectory'].items() if k in common}
        summary['source_outcome']={k:v for k,v in summary['source_outcome'].items() if k in common}
        for name,outcome in (('success',1.0),('failure',0.0)):
            side=[v for k,v in summary['per_trajectory'].items() if summary['source_outcome'][k]==outcome]
            summary['n_'+name]=len(side);summary['S_'+name]=sum(side)/len(side) if side else None
        summary['J']=None if summary['S_success'] is None or summary['S_failure'] is None else summary['S_success']*summary['S_failure']
    p['paired_sample_size']=len(common);c['paired_sample_size']=len(common)
    accepted=p['J'] is not None and c['J'] is not None and c['J']>p['J']
    reason=('missing_direction' if p['J'] is None or c['J'] is None else
            'higher_product' if accepted else 'tie_keep_parent' if p['J']==c['J'] else 'parent_higher_product')
    return {'accepted':accepted,'winner':'P1' if accepted else 'P0','reason':reason,'parent':p,'candidate':c}


def content_key(rubric):
    """Ignore generated IDs/provenance and whitespace when identifying exact duplicates."""
    def normalize(v):
        if isinstance(v,str):return ' '.join(v.split()).casefold()
        if isinstance(v,dict):return {k:normalize(x) for k,x in v.items()}
        if isinstance(v,list):return [normalize(x) for x in v]
        return v
    from src.online_rl.tree.agent_state import visible_rubric
    return sha256_json(normalize({k:v for k,v in visible_rubric(rubric).items() if k != 'rubric_id'}))


def union_kept(trainer, parent, candidate, candidate_prompt):
    """Retain validated rubrics from both prompts, independently of which prompt won.

    Exact duplicates collapse; otherwise use the existing generator's same-reason
    matcher and all six membership checks. An uncertain match remains distinct.
    """
    pool=[]; keys={}; lineage=[]
    case=trainer.cases[parent['case_id']]
    tins={t['leaf_id']:from_dict(TrajectoryRubricInput,t) for t in candidate['trajectory_inputs']}
    for side,record in (('P0',parent),('P1',candidate)):
        for raw in record['trajectory_rubrics']:
            if raw['rubric_id'] not in record['rubric_weights']:continue
            rub=from_dict(TrajectoryRubric,raw)
            if not rub.valid:raise RuntimeError('Rejected rubric in KEEP pool')
            key=content_key(rub.rubric); duplicate=keys.get(key)
            if duplicate is None and side=='P1' and pool:
                identity={'mode':trainer.cfg['prompt_update'].get('mode',MODE),'candidate':rub.rubric_id,'content':key,
                          'prompt':sha256_json(candidate_prompt),'bank':sha256_json(RB.bank_view(pool)),
                          'model':getattr(trainer.env.assign_fn,'model',None),
                          'settings':getattr(trainer.env.assign_fn,'settings',None)}
                decision=trainer.cache.call('batch_union_match',identity,
                    lambda:trainer.env.assign_fn(candidate_prompt,RB.assign_message(tins[rub.leaf_id],tins[rub.leaf_id].outcome,pool)))
                if not RB.match_problem(decision,pool):duplicate=decision['rubric_id']
            if duplicate is None:
                if any(r.rubric_id==rub.rubric_id for r in pool):raise RuntimeError('Rubric ID collision with different content')
                pool.append(rub);duplicate=rub.rubric_id
            keys[key]=duplicate
            lineage.append({'side':side,'rubric_id':rub.rubric_id,'used_as':duplicate})
    return pool,lineage


def score_union(trainer,parent,candidate,candidate_prompt):
    if sha256_json(parent['tree'])!=sha256_json(candidate['tree']):raise RuntimeError('Cannot merge different trees')
    pool,lineage=union_kept(trainer,parent,candidate,candidate_prompt)
    tree=from_dict(Tree,parent['tree']); weights={r.rubric_id:1.0 for r in pool}
    # P0/P1 score each phase immediately after its KEEP set is finalized. Reuse
    # those scores for the final union instead of making a third scoring pass.
    sources={}
    for item in lineage:
        rid=item['rubric_id']; target=item['used_as']; side=item['side']
        if target not in sources or (side=='P0' and sources[target][0]!='P0'):
            sources[target]=(side,rid)
    expected={(node.node_id,r.rubric_id) for node in tree.nodes for r in pool}
    merged=[]; seen=set()
    records={'P0':parent,'P1':candidate}
    for target,(side,rid) in sources.items():
        for raw in records[side].get('response_scores',[]):
            if raw.get('rubric_id')!=rid:continue
            row=dict(raw);row['rubric_id']=target
            pair=(row.get('response_node_id'),target)
            if pair in seen:raise RuntimeError('Duplicate precomputed union response score')
            seen.add(pair);merged.append(from_dict(ResponseRubricScore,row))
    if expected==seen:
        scores=merged
    else:
        # Supports resumed/legacy phase records written before eager P0/P1 scoring.
        scores=score_nodes(trainer.cases[tree.case_id],list(tree.nodes),pool,trainer.env.scorer,trainer.cache,workers=trainer.api_workers)
    targets=agent_targets(tree,scores,weights,trainer.st)
    return {**parent,'response_scores':[to_dict(s) for s in scores],'rubric_weights':weights,'agent_targets':targets,
            'update':{'optimizer_step':False,'deferred_to_gpu4':True},
            'training_rubrics':[to_dict(r) for r in pool],
            'prompt_comparison':{'P0':parent['prompt_sha256'],'P1':candidate['prompt_sha256'],
                                 'P0_evaluation':parent['prompt_evaluation'],'P1_evaluation':candidate['prompt_evaluation'],
                                 'P0_selection':parent['selection'],'P1_selection':candidate['selection'],
                                 'P0_witness':parent['rubric_bank']['witness'],'P1_witness':candidate['rubric_bank']['witness'],
                                 'union_lineage':lineage,'deduplication':'exact_content_then_same_reason_match_v1'},
            'group_diagnostics':sibling_group_diagnostics(tree,targets,None),
            'interaction_diagnostics':interaction_diagnostics(tree,targets)}


def batch_reflection_view(records,max_examples=24):
    """All-unit summaries plus bounded examples; full trees can contain 4096 leaves.

    Only reflection context is bounded, never the scoring mask or rerun evidence.
    Round-robin across case/outcome buckets, alternating low/high effects.
    """
    if max_examples<1:raise ValueError('Reflection example budget must be positive')
    summary=aggregate(records)
    buckets={}; total=0
    for r in records:
        for row in r['prompt_evaluation']['reflection']:
            buckets.setdefault((r['case_id'],row['outcome']),[]).append(row)
            total+=1
    ordered={}
    for key,rows in sorted(buckets.items()):
        rows=sorted(rows,key=lambda row:((row['R_O'] or 0),row['trajectory']))
        spread=[];lo=0;hi=len(rows)-1
        while lo<=hi:
            spread.append(rows[lo]);lo+=1
            if lo<=hi:spread.append(rows[hi]);hi-=1
        ordered[key]=spread
    examples=[];depth=0
    while len(examples)<min(total,max_examples):
        for key,rows in ordered.items():
            if depth<len(rows) and len(examples)<max_examples:examples.append(rows[depth])
        depth+=1
    return [{'batch_summary':{k:v for k,v in summary.items() if k not in ('per_trajectory','source_outcome')},
             'case_summaries':[{'case_id':r['case_id'],
                 **{k:r['prompt_evaluation'][k] for k in ('S_success','S_failure','J','n_success','n_failure','invalid','no_match')},
                 'training_keep_telemetry':r['rubric_bank']['telemetry']} for r in records],
             'reflection_examples_shown':len(examples),'total_trajectories':total,
             'example_selection':'case_outcome_round_robin_low_high_effect_v1'}]+examples


def _candidate_section_problem(text, parent_text, direction):
    if '"user_side_context"' in parent_text and '"agent_side_state"' in parent_text:
        from src.prompt_opt.reflect import structural_problems
        schema_problems = structural_problems(text, require_observed_context=True)
        if schema_problems:
            return 'candidate must preserve the observed-context rubric schema: ' + '; '.join(schema_problems)
    marker=RB.MARKER
    if marker not in text:return 'missing [Assign or create] section'
    parent_suffix=parent_text.split(marker,1)[1] if marker in parent_text else None
    candidate_suffix=text.split(marker,1)[1]
    if parent_suffix is None:return 'parent prompt is missing [Assign or create] section'
    # Model output commonly drops the final newline. Treat trailing whitespace
    # as formatting, not as an edit to the protected reuse contract.
    if candidate_suffix.rstrip()==parent_suffix.rstrip():return None
    # Reuse-section edits are allowed only when the diagnosis explicitly targets
    # reuse/assignment. Preserve every existing check and the decision contract.
    d=(direction or '').lower()
    targets_reuse=(any(term in d for term in ('reuse','same-reason','assignment')) or
                   __import__('re').search(r'\bassign(?:ing|ed)?\b',d) is not None)
    if not targets_reuse:return 'changed [Assign or create] section outside a reuse-focused direction'
    required=('same_construct:', 'applicability_visible:', 'same_required_behavior:', 'same_direction:',
              'score_semantics_apply:', 'no_new_premise:', 'REUSE_EXISTING', 'CREATE_NEW',
              '"decision"', '"rubric_id"', '"checks"', '"rationale"')
    missing=[item for item in required if item not in candidate_suffix]
    if missing:
        return 'reuse edit removed required contract items: '+', '.join(missing)
    return None


def propose_candidate(parent_records,parent_text,store,backend,cache,cases,history):
    from src.online_rl.integrated.prompt_search import _propose
    from src.online_rl.tree.prompt_search import cached_backend
    view=batch_reflection_view(parent_records)
    proposal=_propose(parent_text,view,1,history,cached_backend(backend,cache),store.texts(),cases)
    valid=[]
    for item in proposal['proposals']:
        text=item.get('text')
        if text is None:
            continue
        problem=_candidate_section_problem(text,parent_text,item.get('direction'))
        if problem:
            item['guard_violation']=problem
            continue
        valid.append(item)
    text=valid[0]['text'] if valid else parent_text
    sha=store.put(text)
    return {'text':text,'sha256':sha,'proposal':proposal,'valid_candidate':bool(valid) and text!=parent_text}
