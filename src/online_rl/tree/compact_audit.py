"""Isolated audit adapter. No transport, training launch, or legacy cache writes.

Public definitions live in input once; returned refs are resolved, not paraphrased.
Contract-only and semantic-revision arms share a wire schema but different identities.
"""
import copy
import hashlib
import json
from pathlib import Path
from jsonschema import validate

PROTOCOL = 'compact_audit_v6'
PROMPTS = Path(__file__).resolve().parents[3] / 'prompts' / 'sherpa'

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()

def obj(fields):
    return {'type':'object','additionalProperties':False,'properties':fields,'required':list(fields)}

def arr(item):return {'type':'array','items':item}
def enum(values):return {'type':'string','enum':list(values)}
def reply_text(response):
    keys=[k for k in ('target_agent_reply','reply') if k in response]
    if len(keys)!=1 or not isinstance(response[keys[0]],str):raise ValueError('Exactly one response text required')
    return response[keys[0]]
TEXT={'type':'string'}
SPAN=obj({'turn_id':TEXT,'quote':TEXT})

# The v7.4 relation audit is part of the public Stage-1 output contract.  It
# checks what the Doctor actually asked about before the Agent reply exists.
RELATION_TYPES=('occurrence_risk','diagnostic_evidence','prevention_effect',
                'mimic_or_differential','other','none','uncertain')
RELATION_STATUSES=('same','different','uncertain','not_required')
LIFECYCLES=('open','resolved','absent','uncertain','not_required')
RELATION=obj({'relation_type':enum(RELATION_TYPES),'subject':TEXT,
              'object':TEXT,'temporal_target':TEXT})
RELATION_AUDIT=obj({'condition_relation':RELATION,'visible_relation':RELATION,
                    'relation_status':enum(RELATION_STATUSES),
                    'relation_evidence':arr(SPAN),'active_request_required':{'type':'boolean'},
                    'request_lifecycle':enum(LIFECYCLES),'explanation':TEXT})

def app_schema(items, *, quality=False, relation=False):
    if relation and not quality:raise ValueError('Relation audit requires v7.3 request-state evidence')
    component=obj({'clause':TEXT,'status':enum(['supported','absent','not_required','unmet','uncertain']),
        'evidence':arr(SPAN),'inspected':arr(TEXT),'note':TEXT})
    side={**arr(component),'minItems':1}
    fields = {'doctor_condition':side,'agent_condition':side}
    if quality:
        fields.update(request_state=enum(['not_required','absent','unresolved','resolved','partial','uncertain']),
                      request_evidence=arr(SPAN), remaining_request=TEXT)
    if relation:fields['relation_audit']=RELATION_AUDIT
    return obj({'answers':obj({rid:obj(copy.deepcopy(fields)) for rid in items})})

def score_schema(packet):
    refs=['score_anchors.0','score_anchors.1','score_anchors.2']
    fields={}
    for rid,rubric in packet['rubric_items'].items():
        allowed=refs+[k for k in ('minimum_performance','completion_condition') if k in rubric]
        check=obj({'ref':enum(allowed),'met':{'type':['boolean','null']},
            'cites':arr({'type':'integer','minimum':0}),'why':TEXT})
        fields[rid]=obj({'level':enum(['0','1','2','UNSCORABLE']),
            'quotes':arr(TEXT),'checks':arr(check),'gap':TEXT,'defect':TEXT})
    return obj({'answers':obj({r['response_id']:obj(copy.deepcopy(fields)) for r in packet['responses']})})

def prepare(payload, *, semantic=False, quality=False, relation=False):
    """Preserve exact criterion/input/model/settings. Replace output contract only,
    then optionally add the separately frozen semantic supplement.
    """
    q=copy.deepcopy(payload);packet=json.loads(q['messages'][1]['content'])
    if 'target_agent_reply' in packet.get('state',{}):
        state=packet['state'];reply=state.pop('target_agent_reply')
        packet={'stage':PROTOCOL+'_behavior','shared_pre_reply_state':state,
            'rubric_items':packet['items'],'responses':[{'response_id':'source_reply','reply':reply}]}
    kind='behavior' if 'responses' in packet else 'applicability'
    if kind=='applicability':
        if 'candidate_rubric' in packet.get('state',{}):raise ValueError('Gate is a separate experiment')
        schema=app_schema(packet['items'], quality=quality, relation=relation)
    else:schema=score_schema(packet)
    contract=(PROMPTS/f'compact_{kind}_contract_v1.txt').read_text()
    supplement=contract
    if semantic:supplement+='\n\n'+(PROMPTS/f'compact_{kind}_semantics_v1.txt').read_text()
    q['messages'][0]['content']+='\n\n'+supplement
    for item in packet['rubric_items' if kind=='behavior' else 'items'].values():
        item['instructions']=item.get('instructions','')+'\n\n'+supplement
    q['messages'][1]['content']=json.dumps(packet,ensure_ascii=False)
    q['response_format']={'type':'json_schema','json_schema':{'name':PROTOCOL+'_'+kind+('_quality' if quality else ''),'strict':True,'schema':schema}}
    # No input evidence, old rubric, sampling parameter, or token limit is changed.
    return q

def cache_identity(payload, *, semantic=False):
    # Exact requests are safe cache keys, but aliases should also share judgments.
    # Caller should use semantic_cache_identity for corpus/bank ID aliases.
    return digest({'contract':PROTOCOL,'semantic_revision':semantic,'payload':payload})

def semantic_cache_identity(payload, *, semantic=False, criterion_contents=None):
    q=copy.deepcopy(payload);packet=json.loads(q['messages'][1]['content'])
    key='rubric_items' if 'rubric_items' in packet else 'items'
    canonical={};mapping={}
    for rid,item in packet[key].items():
        content={k:v for k,v in item.items() if k not in ('rubric_id','id','name','rubric')}
        cid=digest(content);mapping[rid]=cid;canonical[cid]=content
    packet[key]=dict(sorted(canonical.items()))
    # An response ID is bookkeeping. Actual reply and context remain in identity.
    if 'responses' in packet:
        packet['responses']=[{'reply':reply_text(r)} for r in packet['responses']]
    q['messages'][1]['content']=json.dumps(packet,sort_keys=True,ensure_ascii=False)
    q.pop('response_format',None)  # schema contains aliases; contract version is explicit
    if key=='items' and criterion_contents is None:
        raise ValueError('Applicability cache requires full public criterion identity (not transmitted to judge)')
    public_identity=None if criterion_contents is None else sorted(digest({k:v for k,v in item.items()
        if k not in ('rubric_id','id','name','rubric','instructions')}) for item in criterion_contents.values())
    return digest({'contract':PROTOCOL,'semantic_revision':semantic,'payload':q,'full_public_criterion_identity':public_identity})

def _catalog(state):
    rows=state.get('case_evidence',[])+state.get('dialogue_before_target_reply',[])
    return {r['turn_id']:r for r in rows}

def _validate_relation(audit, state, judgment, answer):
    """Port the isolated v7.4 relation check without its frozen-harness imports."""
    if not audit['explanation'].strip():raise ValueError('Relation audit needs an explanation')
    required=audit['condition_relation']['relation_type']
    visible=audit['visible_relation']['relation_type']
    status=audit['relation_status']
    doctor_turns={row['turn_id']:row for row in state['dialogue_before_target_reply']
                  if row['speaker'].lower()=='doctor'}
    evidence=audit['relation_evidence']
    if len({(e['turn_id'],e['quote']) for e in evidence})!=len(evidence):
        raise ValueError('Duplicate relation evidence')
    for span in evidence:
        turn=doctor_turns.get(span['turn_id'])
        if turn is None or not span['quote'].strip() or span['quote'] not in turn['text']:
            raise ValueError('Relation evidence is not an exact visible Doctor quote')
    if status=='not_required':
        if required!='none' or evidence:raise ValueError('A required relation cannot be not_required')
    elif status=='same':
        if required in ('none','uncertain') or required!=visible or not evidence:
            raise ValueError('Same relation needs matching expressed relation')
    elif status=='different':
        if required in ('none','uncertain') or visible=='uncertain':
            raise ValueError('Different relation needs determinate kinds')
        if audit['condition_relation']==audit['visible_relation']:
            raise ValueError('Identical relation tuples cannot be different')
        if visible!='none' and not evidence:
            raise ValueError('Expressed different relation needs Doctor evidence')
        if visible=='none' and evidence:
            raise ValueError('Absent relation cannot cite expressed relation')
    else:
        if required=='none':raise ValueError('Uncertain relation requires written relation')
        if visible!='none' and not evidence:
            raise ValueError('Uncertain expressed relation needs Doctor evidence')
        if visible=='none' and evidence:
            raise ValueError('Absent relation cannot cite invented evidence')
    doctor_statuses=[part['status'] for part in answer['doctor_condition']]
    if status=='different' and ('unmet' not in doctor_statuses or judgment['applicable'] is not False):
        raise ValueError('Different relation must make Doctor condition unmet')
    if status=='uncertain' and ('uncertain' not in doctor_statuses or judgment['applicable'] is not None):
        raise ValueError('Material relation uncertainty must remain uncertain')
    expected={'unresolved':'open','partial':'open','resolved':'resolved',
              'absent':'absent','uncertain':'uncertain','not_required':'not_required'}[judgment['request_state']]
    if audit['request_lifecycle']!=expected:
        raise ValueError('Relation lifecycle contradicts request state')
    if (audit['active_request_required'] and judgment['request_state']=='resolved'
            and ('unmet' not in doctor_statuses or judgment['applicable'] is not False)):
        raise ValueError('Resolved request cannot satisfy required active request')
    return {**audit,'semantic_accuracy':'independent_review_pending'}

def _app(raw,packet):
    state=packet['state'];catalog=_catalog(state)
    prefix_ids=[r['turn_id'] for r in state['dialogue_before_target_reply']]
    answers={}
    for rid,assessment in raw['answers'].items():
        statuses=[];bound={}
        for side in ('doctor_condition', 'agent_condition'):
            components = assessment[side]
            definition=packet['items'][rid][side];bound[side]=[]
            for part in components:
                clause=part['clause'];status=part['status'];statuses.append(status)
                if not (status == 'not_required' and definition == clause == '') and (not clause.strip() or clause not in definition):raise ValueError('Condition clause is not exact')
                if not part['note'].strip():raise ValueError('Missing condition explanation')
                if len(set(part['inspected']))!=len(part['inspected']) or any(t not in catalog for t in part['inspected']):
                    raise ValueError('Invalid inspection scope')
                evidence=[]
                for quote in part['evidence']:
                    source=catalog.get(quote['turn_id'])
                    if not source or not quote['quote'].strip() or quote['quote'] not in source['text']:
                        raise ValueError('Exact prefix quote failed')
                    evidence.append({**quote,'speaker':source['speaker']})
                if status=='supported' and not evidence:raise ValueError('Positive support requires a quote')
                if status=='absent' and (evidence or part['inspected']!=prefix_ids):
                    raise ValueError('Absence requires whole prefix and no invented speech')
                if status=='not_required' and evidence:raise ValueError('No required state must not invent speech')
                bound[side].append({**part,'evidence':evidence})
        request_state = assessment.get('request_state')
        if request_state is not None:
            if request_state not in ('not_required','absent','unresolved','resolved','partial','uncertain'):
                raise ValueError('Invalid request state')
            request_evidence = assessment['request_evidence']
            if request_state in ('unresolved','resolved','partial','uncertain') and not request_evidence:
                raise ValueError('Request state needs exact prefix evidence')
            if request_state in ('absent','not_required') and request_evidence:
                raise ValueError('Absent/non-required request cannot cite invented evidence')
            for quote in request_evidence:
                source = catalog.get(quote['turn_id'])
                if (quote['turn_id'] not in prefix_ids or not source or
                        not quote['quote'].strip() or quote['quote'] not in source['text']):
                    raise ValueError('Request-state quote is not exact prefix evidence')
            remaining = assessment['remaining_request'].strip()
            if (request_state in ('unresolved','partial','uncertain')) != bool(remaining):
                raise ValueError('Request residual contradicts request state')
            if request_state == 'uncertain' and 'uncertain' not in statuses:
                raise ValueError('Materially uncertain request must remain uncertain')
            if request_state in ('resolved','partial') and len({q['turn_id'] for q in request_evidence}) < 2:
                raise ValueError('Resolution/partial handling must cite request and later handling')
            if any(part['inspected'] != prefix_ids for part in assessment['doctor_condition']):
                raise ValueError('Doctor condition must inspect the full visible prefix')
        # Necessary clauses are a conjunction: a definite unmet clause stays
        # false even when another clause is uncertain (isolated v7.4b fix).
        applicable = False if 'unmet' in statuses else None if 'uncertain' in statuses else True
        answers[rid]={'applicable':applicable,
            'status':('uncertain' if request_state is not None else 'unresolved') if applicable is None
                     else 'applicable' if applicable else 'not_applicable',
            'conditions':bound,'coverage_requires_semantic_review':True}
        if request_state is not None:
            answers[rid]['request_state'] = request_state
            answers[rid]['request_evidence'] = [{**q, 'speaker': catalog[q['turn_id']]['speaker']} for q in request_evidence]
            answers[rid]['remaining_request'] = assessment['remaining_request']
        if 'relation_audit' in assessment:
            answers[rid]['relation_audit']=_validate_relation(assessment['relation_audit'],state,
                                                               answers[rid],assessment)
    return {'status':'valid_execution_contract','applicability':answers}

def _behavior(raw,packet):
    replies={r['response_id']:reply_text(r) for r in packet['responses']};answers={}
    for nid,rows in raw['answers'].items():
        answers[nid]={}
        for rid,row in rows.items():
            rubric=packet['rubric_items'][rid];quotes=row['quotes']
            if any(not q.strip() or q not in replies[nid] for q in quotes):raise ValueError('Exact response quote failed')
            refs=['score_anchors.0','score_anchors.1','score_anchors.2']
            refs += [k for k in ('minimum_performance','completion_condition') if k in rubric]
            checks={c['ref']:c for c in row['checks']}
            if len(checks)!=len(row['checks']) or set(checks)!=set(refs):raise ValueError('Each public definition needs one check')
            for ref,c in checks.items():
                if not c['why'].strip():raise ValueError('Missing anchor connection')
                if len(set(c['cites']))!=len(c['cites']) or any(i>=len(quotes) for i in c['cites']):
                    raise ValueError('Invalid citation index')
                if c['met'] is True and (ref in ('minimum_performance','completion_condition','score_anchors.1','score_anchors.2')) and not c['cites']:
                    raise ValueError('Performed work needs literal response evidence')
            matched=[str(i) for i in range(3) if checks[f'score_anchors.{i}']['met'] is True]
            uncertain=any(checks[f'score_anchors.{i}']['met'] is None for i in range(3))
            level=row['level']
            if level=='UNSCORABLE':
                mc_conflict = False
                if 'minimum_performance' in checks:
                    m = checks['minimum_performance']['met']; c = checks['completion_condition']['met']
                    if m is None or c is None:
                        mc_conflict = True
                    elif c and not m:
                        mc_conflict = True
                    elif len(matched) == 1 and matched[0] != str(2 if m and c else 1 if m else 0):
                        mc_conflict = True
                if not row['defect'].strip() or (len(matched)==1 and not uncertain and not mc_conflict):
                    raise ValueError('UNSCORABLE without an actual declared gap/overlap/uncertainty')
                score=None;status='rubric_unscorable'
            else:
                if matched!=[level] or uncertain or row['defect']:raise ValueError('Numeric score contradicts anchor matches')
                if (level=='2' and row['gap']) or (level!='2' and not row['gap'].strip()):raise ValueError('Gap contradicts level')
                if 'minimum_performance' in checks:
                    m=checks['minimum_performance']['met'];c=checks['completion_condition']['met']
                    if m is None or c is None:raise ValueError('Numeric M/C unresolved')
                    if c and not m:raise ValueError('C cannot be complete without M')
                    if str(2 if m and c else 1 if m else 0)!=level:raise ValueError('Declared M/C and anchors conflict; do not override score')
                score=int(level);status='scored'
            answers[nid][rid]={'score':score,'scoring_status':status,'evidence':row,
                'definition_refs_preserved':True,'semantic_accuracy':'unreviewed'}
    return {'status':'valid_execution_contract','answers':answers}

def validate_response(payload,body):
    """API failure never becomes a rubric judgment. No automatic repair/retry."""
    try:
        choice=body['choices'][0]
        if choice.get('finish_reason')!='stop':return {'status':'execution_incomplete','kind':'finish_reason','score':None}
        raw=json.loads(choice['message']['content'])
    except (ValueError,KeyError,IndexError,TypeError) as exc:
        return {'status':'execution_error','kind':'json_or_response','score':None,'error':str(exc)}
    try:validate(raw,payload['response_format']['json_schema']['schema'])
    except Exception as exc:return {'status':'execution_error','kind':'schema','score':None,'error':str(exc)}
    try:
        packet=json.loads(payload['messages'][1]['content'])
        return _behavior(raw,packet) if 'responses' in packet else _app(raw,packet)
    except (ValueError,KeyError,TypeError) as exc:
        return {'status':'execution_error','kind':'evidence_contract','score':None,'error':str(exc),'raw_parsed':raw}

def share_applicability(result, response_ids):
    if result['status']!='valid_execution_contract':raise ValueError('Applicability execution incomplete')
    if any(v['applicable'] is None for v in result['applicability'].values()):raise ValueError('Unresolved applicability blocks scoring')
    return {nid:copy.deepcopy(result['applicability']) for nid in response_ids}

def evaluate_group(app_payload, behavior_payload, invoke, *, app_cache=None, semantic=False):
    """Dependency-injected audit path, default no transport. Never updates a model.
    One app decision per state/criterion; error blocks downstream rather than NA.
    Cache supplied by caller is isolated and versioned; bank IDs are aliases.
    """
    ap=json.loads(app_payload['messages'][1]['content']);bp=json.loads(behavior_payload['messages'][1]['content'])
    if ap['state']!=bp['shared_pre_reply_state']:raise ValueError('App and behavior prefixes differ')
    if set(ap['items'])!=set(bp['rubric_items']):raise ValueError('App/score criterion IDs differ')
    q=prepare(app_payload,semantic=semantic);key=semantic_cache_identity(q,semantic=semantic,criterion_contents=bp['rubric_items'])
    def cid(item):return digest({k:v for k,v in item.items() if k not in ('rubric','rubric_id','name','id')})
    mapping={rid:cid(item) for rid,item in ap['items'].items()}
    cache=app_cache if app_cache is not None else {}
    if key in cache:
        app={'status':'valid_execution_contract','applicability':{rid:copy.deepcopy(cache[key][c]) for rid,c in mapping.items()}}
    else:
        app=validate_response(q,invoke(q))
        if app['status']=='valid_execution_contract' and all(v['applicable'] is not None for v in app['applicability'].values()):
            cache[key]={mapping[rid]:copy.deepcopy(v) for rid,v in app['applicability'].items()}
    if app['status']!='valid_execution_contract' or any(v['applicable'] is None for v in app['applicability'].values()):
        return {'status':'blocked','applicability':app,'answers':{},'reason':'uncompleted_or_unresolved_applicability'}
    eligible=[rid for rid,v in app['applicability'].items() if v['applicable']]
    answers={};executions=[]
    for response in bp['responses']:
        nid=response['response_id'];answers[nid]={rid:{'score':None,'scoring_status':'not_applicable'} for rid in ap['items'] if rid not in eligible}
        if not eligible:continue
        one=copy.deepcopy(behavior_payload);packet=copy.deepcopy(bp)
        packet['responses']=[response];packet['rubric_items']={rid:packet['rubric_items'][rid] for rid in eligible}
        one['messages'][1]['content']=json.dumps(packet,ensure_ascii=False)
        score_request=prepare(one,semantic=semantic)
        result=validate_response(score_request,invoke(score_request));executions.append(result)
        if result['status']!='valid_execution_contract':
            return {'status':'blocked','applicability':app,'answers':answers,'executions':executions,'reason':'incomplete_behavior_vector'}
        answers[nid].update(result['answers'][nid])
    return {'status':'valid_execution_contract','applicability':app,'answers':answers,'executions':executions,
            'shared_applicability':True,'optimizer_updated':False}

def diagnostic_rows(result, group):
    """Bridge only a complete group to the frozen offline reward diagnostic."""
    if result['status']!='valid_execution_contract':raise ValueError('Incomplete group has no training diagnostic')
    if set(result['answers'])!=set(group['response_ids']):raise ValueError('Missing sibling')
    rid=group['target_criterion_id'];rows=[]
    for nid in group['response_ids']:
        v=result['answers'][nid][rid];status=v['scoring_status']
        if status not in ('scored','rubric_unscorable','not_applicable'):raise ValueError('Error is not a reward')
        rows.append({'fixture_id':group['fixture_by_response'][nid],'applicable':status!='not_applicable',
            'scoring_status':status,'score':v['score']})
    return rows
