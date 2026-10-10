"""One preselected creator per rubric; anchored precommit action + linear dialogue.

No verification dialogue is sent to a generator/scorer. V7 optionally audits the frozen instruction before any continuation. Only parsed outcomes count as hits/misses.
Unparsed FINALs are cached evidence, not wrong answers; unresolved majority is
INCOMPLETE and excluded. Generation/transport errors abort the batch. Resume
reuses each completed model call and trial, with the same plan and seeds.
"""
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from pathlib import Path
import json
import threading
import time

from src.online_rl.integrated.continuation import derive_seed
from src.online_rl.integrated.records import sha256_json
from src.online_rl.tree.records import to_dialogue
from src.online_rl.tree.witness import FIXED_SOURCE_PROTOCOL, WitnessVerification
from src.online_rl.tree.member_intervention import resolve_member_intervention

_audit_lock = threading.Lock()
SOURCE_RULE = 'canonical_creator_no_search_v1'


def audit(cache, event):
    if cache.path is None:
        return
    path = cache.path.with_name('verification_attempts.jsonl')
    with _audit_lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('a') as stream:
            stream.write(json.dumps({'unix_time': time.time(), **event}) + '\n')


def majority(outcomes, source_outcome):
    """None is neither a hit nor a miss, including when the source succeeded."""
    if source_outcome not in (0, 1):
        raise ValueError('Source must have a parsed binary outcome')
    hits = sum(o in (0, 1) and o != source_outcome for o in outcomes)
    misses = sum(o in (0, 1) and o == source_outcome for o in outcomes)
    return 'KEEP' if hits >= 2 else 'DROP' if misses >= 2 else None


class LinearInterventions:
    def __init__(self, policy, cfg):
        self.policy, self.cfg = policy, cfg

    def write(self, *args, **kwargs):
        raise RuntimeError('Linear verification must execute through its frozen trial plan')

    def continue_(self, *args, **kwargs):
        raise RuntimeError('Immediate-FINAL continuation is disabled for linear verification')

    def trial(self, case, node, plan, index, cache):
        from src.collaboration.generator import stop_reason, final_answer_option
        from src.online_rl.doctor import minimum_agent_turns
        from src.online_rl.environment import final_outcome
        from src.online_rl.tree.rollout import repeated_agent_reply
        from src.online_rl.tree.backends import with_role
        policy, doctor = self.policy, self.policy.doctor
        seeds = plan['trials'][index]
        dialogue = to_dialogue(node.prefix)
        anchor = node.depth
        if sum(d['speaker'] == 'agent' for d in dialogue) != anchor - 1:
            raise ValueError('Anchor does not agree with the frozen prefix')
        previous_status = node.prefix_doctor_status
        traces, statuses = [], []
        started = time.monotonic()
        max_turns = self.cfg['rollout']['rounds']
        repeat_after = self.cfg['rollout']['halting'].get('repeat_agent_reply_after_turns')
        identity = {'plan_sha256': sha256_json(plan), 'trial_index': index}
        for turn in range(anchor, max_turns + 1):
            guidance = plan['guidance'] if turn == anchor else None
            args = {'verification_guidance': guidance} if guidance is not None else {}
            trace = cache.call('linear_verification_agent', {**identity, 'turn': turn, 'seed': seeds['agent'][str(turn)]},
                lambda: with_role('verification_agent', lambda: policy.agent_turn(case, dialogue, turn, seeds['agent'][str(turn)], **args)))
            reply = trace['completion_text']
            traces.append({'turn': turn, 'guided': guidance is not None, 'trace': trace})
            why = stop_reason(turn, dialogue[-1]['text'], reply) if turn >= minimum_agent_turns(doctor) else None
            if why is None and repeat_after is not None and turn >= repeat_after and repeated_agent_reply(dialogue, reply):
                why = 'agent_repeated_reply'
            why = why or ('max_agent_turns' if turn == max_turns else None)
            dialogue = dialogue + [{'speaker': 'agent', 'stage': f'A{turn}', 'round': turn, 'text': reply}]
            if why is None:
                ds = seeds['doctor'][str(turn + 1)]
                d = cache.call('linear_verification_doctor', {**identity, 'turn': turn + 1, 'seed': ds},
                    lambda: with_role('verification_doctor', lambda: doctor.turn(case, dialogue, turn + 1, max_turns, ds)))
                dialogue.append({'speaker': 'doctor', 'stage': f'D{turn + 1}', 'round': turn + 1, 'text': d['visible']})
                statuses.append({'turn': turn + 1, **d})
                why = doctor.halt(agent_turns=turn, status=d['status'], prev_status=previous_status,
                                  last_agent=reply, options=case['options'])
                previous_status = d['status']
            if why is not None:
                break
        fs = seeds['final']
        final = cache.call('linear_verification_final', {**identity, 'seed': fs},
            lambda: with_role('verification_final', lambda: doctor.final(case, dialogue, fs)))
        option = final_answer_option(final)
        outcome = final_outcome(case, option)
        return {'trial_index': index, 'seeds': seeds, 'outcome': outcome, 'final_option': option,
                'final_text': final, 'termination_reason': why, 'agent_turns_total': turn,
                'generated_agent_turns': len(traces), 'generated_doctor_discussion_turns': len(statuses),
                'dialogue': dialogue + [{'speaker': 'doctor', 'stage': 'FINAL', 'round': 0, 'text': final}],
                'agent_traces': traces, 'doctor_statuses': statuses,
                'seconds': time.monotonic() - started, 'final_parse_error': outcome is None}


def make_plan(case, node, leaf, rubric, spec, run_seed, policy_sha256, cfg):
    """All seeds, guidance and source are committed before the first generation."""
    root = Path(__file__).resolve().parents[3]
    level = '0' if leaf.outcome == 1 else '2'
    template = (root / cfg['intervention'][f'score{level}_template_file']).read_text()
    guidance = template.format(intervention_instruction=spec['intervention_instruction'])
    key = f"{case['case_id']}|{rubric.rubric_id}|{leaf.leaf_id}|{node.node_id}|{FIXED_SOURCE_PROTOCOL}"
    seeds = [derive_seed(run_seed, key, 'trial', i) for i in range(3)]
    if len(set(seeds)) != 3:
        raise RuntimeError('Trial seed collision')
    rounds = cfg['rollout']['rounds']
    return {'protocol': FIXED_SOURCE_PROTOCOL, 'source_rule': SOURCE_RULE,
            'source_selection_seed': None, 'source_selection_is_random': False, 'run_seed': run_seed,
            'source_leaf_id': leaf.leaf_id, 'source_outcome': leaf.outcome,
            'rubric_id': rubric.rubric_id, 'node_id': node.node_id, 'anchored_turn': node.depth,
            'policy_sha256': policy_sha256, 'case_sha256': sha256_json(case),
            'rubric_sha256': sha256_json(rubric.rubric), 'config_sha256': sha256_json(cfg),
            'prefix': to_dialogue(node.prefix), 'prefix_doctor_status': node.prefix_doctor_status,
            'intervention_spec': spec, 'level': level, 'guidance': guidance,
            'trials': [{'index': i, 'seed': seed,
                'agent': {str(t): derive_seed(seed, key, 'agent', t) for t in range(node.depth, rounds + 1)},
                'doctor': {str(t): derive_seed(seed, key, 'doctor', t) for t in range(node.depth + 1, rounds + 1)},
                'final': derive_seed(seed, key, 'final', 0)} for i, seed in enumerate(seeds)]}


def verify_fixed_source(case, nodes, leaves, canon, assignments, *, backend, run_seed, policy_sha256, cache):
    out = {}
    for cid in sorted(canon):
        rubric = canon[cid]
        if not rubric.valid:
            continue
        members = [a for a in assignments if a.canonical_rubric_id == cid and a.assignment != 'CREATE_REJECTED']
        creators = [a for a in members if a.assignment == 'CREATED_NEW']
        if len(creators) != 1 or creators[0].leaf_id != rubric.leaf_id:
            raise ValueError('Exactly one canonical creator is required; never search for another source')
        creator = creators[0]
        leaf = leaves[creator.leaf_id]
        spec = resolve_member_intervention(rubric, leaf, member_spec=creator.intervention_spec)
        if spec is None:
            raise ValueError('Fixed source needs its anchored intervention specification')
        node = nodes[leaf.path[spec['selected_agent_turn_id'] - 1]]
        resolve_member_intervention(rubric, leaf, member_spec=spec, node=node)
        proposed = make_plan(case, node, leaf, rubric, spec, run_seed, policy_sha256, backend.cfg)
        plan = cache.call('fixed_verification_plan', {'identity': proposed}, lambda: proposed)

        instruction_review = None
        if getattr(backend, 'instruction_auditor', None) is not None:
            from src.online_rl.tree.instruction_audit import cached_instruction_auditor
            reviewer = cached_instruction_auditor(backend.instruction_auditor, cache, policy_sha256)
            instruction_review = reviewer(case, node.prefix, rubric.rubric, spec)
            if not instruction_review['passed']:
                row = {'leaf_id': leaf.leaf_id, 'node_id': node.node_id, 'target_reply_id': node.depth,
                       'plan': plan, 'instruction_audit': instruction_review, 'result': 'INSTRUCTION_REJECTED',
                       'reruns': 0, 'rerun_records': [], 'outcomes': [], 'valid_outcomes': 0, 'unparsed': 0,
                       'errors': [], 'error_count': 0, 'early_decision': False, 'planned_trials': 3,
                       'executed_trials': 0, 'third_trial_executed': False}
                out[cid] = WitnessVerification(cid, creator.stratum, len(members), tuple(a.leaf_id for a in members),
                    (leaf.leaf_id,), (row,), 0, 0, None, 'DROP', 'instruction_rejected_before_trials', FIXED_SOURCE_PROTOCOL)
                continue

        def one(index):
            ident = {'plan_sha256': sha256_json(plan), 'trial_index': index}
            def generate():
                audit(cache, {'event': 'trial_started', **ident, 'rubric_id': cid, 'seed': plan['trials'][index]['seed']})
                try:
                    result = backend.trial(case, node, plan, index, cache)
                except Exception as exc:
                    audit(cache, {'event': 'trial_error', **ident, 'exception_type': type(exc).__name__})
                    raise
                audit(cache, {'event': 'trial_completed', **ident, 'outcome': result['outcome']})
                return result
            return cache.call('fixed_verification_trial', ident, generate)

        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(copy_context().run, one, i) for i in (0, 1)]
            runs = [future.result() for future in futures]
        decision = majority([r['outcome'] for r in runs], leaf.outcome)
        early = decision is not None
        if not early:
            runs.append(one(2))
            decision = majority([r['outcome'] for r in runs], leaf.outcome) or 'INCOMPLETE'
        errors = [{'trial_index': r['trial_index'], 'status': 'unparsed_final'} for r in runs if r['outcome'] is None]
        row = {'leaf_id': leaf.leaf_id, 'target_reply_id': node.depth, 'node_id': node.node_id,
               'plan': plan, 'instruction_audit': instruction_review, 'stage1_applicable': True, 'stage1_source': 'creator_source_gate_prechecked',
               'reruns': len(runs), 'rerun_records': runs, 'outcomes': [r['outcome'] for r in runs],
               'valid_outcomes': len(runs) - len(errors), 'unparsed': len(errors), 'errors': errors,
               'error_count': len(errors), 'early_decision': early, 'planned_trials': 3,
               'executed_trials': len(runs), 'third_trial_executed': len(runs) == 3,
               'result': 'PASS' if decision == 'KEEP' else 'FAIL' if decision == 'DROP' else 'INCOMPLETE'}
        out[cid] = WitnessVerification(cid, creator.stratum, len(members), tuple(a.leaf_id for a in members),
            (leaf.leaf_id,), (row,), 1, len(runs), leaf.leaf_id if decision == 'KEEP' else None,
            decision, 'early_majority' if early else 'three_trials_complete', FIXED_SOURCE_PROTOCOL)
    return out
