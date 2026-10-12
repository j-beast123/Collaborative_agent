"""Anchored SHERPA: comparative reflection, three counterfactuals, blind group rewards.

Only the reference decision points create training groups. Every member's
outcome comes from its own natural continuation, including the reference member.
Rubrics never transfer to another case or decision point.
"""
from __future__ import annotations

import json
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

from src.online_rl.doctor import doctor_seed, minimum_agent_turns
from src.online_rl.integrated.continuation import derive_seed
from src.online_rl.integrated.metrics import advantages
from src.online_rl.integrated.records import atomic_write_json, read_json, sha256_json, to_dict
from src.online_rl.tree.cache import CallCache
from src.online_rl.tree.records import Tree, to_dialogue
from src.online_rl.tree.rollout import build_tree, repeated_agent_reply, tree_counts

PROTOCOL = 'anchored_group_sherpa_v1'


def quote_in_reply(quote, reply):
    """Allow typography differences, never fuzzy attribution to a different reply."""
    import unicodedata
    def words(text):
        return ' '.join(re.findall(r'\w+', unicodedata.normalize('NFKC', text).casefold()))
    if not isinstance(quote, str) or not quote.strip() or not isinstance(reply, str):
        return False
    if quote in reply:
        return True
    normalized = words(quote)
    return bool(normalized) and f' {normalized} ' in f' {words(reply)} '


def enabled(cfg):
    return (cfg.get('group_learning') or {}).get('protocol') == PROTOCOL


def config_problems(cfg, root):
    spec = cfg.get('group_learning') or {}
    problems = []
    if spec.get('protocol') != PROTOCOL: problems.append('Unknown group_learning.protocol')
    ro = cfg.get('rollout') or {}
    if ro.get('budget') != 'reference_spine' or ro.get('branching') != 4 or ro.get('rounds') != 4:
        problems.append('Group SHERPA requires reference_spine, branching=4 and rounds=4')
    if (ro.get('halting') or {}).get('min_agent_turns') != 2:
        problems.append('Group SHERPA requires at least two Agent turns')
    for flag in ('allow_mutual_conclusion', 'require_agent_doctor_agreement'):
        if type((ro.get('halting') or {}).get(flag, flag == 'allow_mutual_conclusion')) is not bool:
            problems.append(f'rollout.halting.{flag} must be boolean')
    if ro.get('agent_protocol') is not None:
        problems.append('Group SHERPA trains one visible reply conditioned on the shared prefix')
    if (ro.get('doctor_backend') or {}).get('private_state_protocol') != 'hypothesis_confidence_v1':
        problems.append('Group SHERPA requires partner hypothesis/confidence records')
    if cfg.get('agent_reward', {}).get('w_outcome') != 0 or cfg.get('agent_reward', {}).get('w_rubric') != 1:
        problems.append('Group SHERPA requires rubric-only reward')
    if cfg.get('rubric', {}).get('cumulative_bank') is not None or cfg.get('prompt_update', {}).get('enabled'):
        problems.append('Group SHERPA uses anchored rubrics and fixed generation prompts')
    if cfg.get('advantage_normalization', {}).get('mode') != 'group_zscore':
        problems.append('Group SHERPA requires group_zscore advantage')
    if (type(spec.get('max_rubrics_per_group')) is not int or spec.get('max_rubrics_per_group') not in (1, 2)
            or spec.get('verification_reruns') != 3 or spec.get('min_successes') != 2):
        problems.append('Group SHERPA requires one or two rubrics and two of three outcome flips')
    if spec.get('pipeline') not in (None, 'ready_groups_v1'):
        problems.append('Unknown group pipeline')
    if spec.get('reward_mode') not in (None, 'rubric_rank_v1'):
        problems.append('Unknown group reward mode')
    if (spec.get('reward_mode') == 'rubric_rank_v1'
            and cfg.get('judges', {}).get('scorer_protocol') != 'anchored_group_ranking_v1'):
        problems.append('Ranking rewards require the anchored_group_ranking_v1 judge protocol')
    if spec.get('candidate_direction') not in (None, 'source_outcome_v1'):
        problems.append('Unknown candidate direction protocol')
    if spec.get('candidate_direction') == 'source_outcome_v1' and spec.get('max_rubrics_per_group') != 1:
        problems.append('Decisive response selection requires one candidate per group')
    if type(spec.get('generation_format_retries', 0)) is not int or spec.get('generation_format_retries', 0) not in (0, 1):
        problems.append('generation_format_retries must be 0 or 1')
    for name in ('group_rollout_workers', 'group_pipeline_workers'):
        value = cfg.get('execution', {}).get(name, 4)
        if type(value) is not int or value < 1:
            problems.append(f'execution.{name} must be a positive integer')
    if type(spec.get('show_gold_to_reflection')) is not bool:
        problems.append('show_gold_to_reflection must be boolean')
    if type(spec.get('reflection_max_tokens')) is not int or spec.get('reflection_max_tokens', 0) < 1:
        problems.append('reflection_max_tokens must be a positive integer')
    if spec.get('reflection_format') not in (None, 'prose_json_schema_v2'):
        problems.append('Unknown group reflection format')
    if type(spec.get('reflection_format_retries', 0)) is not int or spec.get('reflection_format_retries', 0) not in (0, 1):
        problems.append('reflection_format_retries must be 0 or 1')
    rerun_workers = cfg.get('execution', {}).get('rerun_workers', 1)
    if type(rerun_workers) is not int or rerun_workers < 1:
        problems.append('execution.rerun_workers must be a positive integer')
    generation_provider = cfg.get('rubric', {}).get('generation', {}).get('api_provider', cfg.get('api_provider', 'openrouter'))
    judge_provider = cfg.get('judges', {}).get('api_provider', cfg.get('api_provider', 'openrouter'))
    for provider, model in ((generation_provider, cfg.get('rubric', {}).get('generator_model')),
                            (judge_provider, cfg.get('judges', {}).get('scorer_model'))):
        if provider not in ('openrouter', 'openai_direct'): problems.append('Unknown group API provider')
        if provider == 'openai_direct' and model != 'gpt-6-luna': problems.append('Direct group API requires gpt-6-luna')
    for role in ('reflection', 'generator', 'judge'):
        path, digest = spec.get(role + '_prompt_file'), spec.get(role + '_prompt_sha256')
        try:
            import hashlib
            if hashlib.sha256((Path(root) / path).read_bytes()).hexdigest() != digest:
                problems.append(f'Group {role} prompt hash differs')
        except (OSError, TypeError): problems.append(f'Group {role} prompt is missing')
    if cfg.get('rubric', {}).get('prompt_sha256') != spec.get('generator_prompt_sha256'):
        problems.append('Group generator prompt must match the fixed experiment prompt')
    return problems


def identity(cfg):
    return {'protocol': PROTOCOL, 'settings': cfg['group_learning'],
            'generator_model': cfg['rubric']['generator_model'], 'judge_model': cfg['judges']['scorer_model'],
            'generation': cfg['rubric']['generation'], 'judge_provider': cfg['judges'].get('api_provider'),
            'judge_max_tokens': cfg['judges']['scorer_max_tokens'],
            'judge_reasoning_effort': cfg['judges'].get('reasoning_effort', 'low'),
            'text_stop_rule': ('same_choice_mutual_agreement_without_mutual_conclusion_v2'
                if cfg['rollout'].get('halting', {}).get('allow_mutual_conclusion', True) is False else
                'mutual_conclusion_without_agent_final_answer_v1'),
            'reflection_routing': 'stateless_partner_call_to_generator_only_v2',
            'evidence_quote_matching': 'nfkc_casefold_word_sequence_v2',
            'reward': ('normalized_rank_with_average_occupied_ranks_for_ties'
                       if cfg['group_learning'].get('reward_mode') == 'rubric_rank_v1'
                       else 'mean_anchor_rubric_score_divided_by_two'), 'loss': 'anchor_reply_tokens_only',
            'verification': 'one_fixed_contrast_reply_three_independent_continuations'}


@dataclass(frozen=True)
class GroupEnv:
    agent: object
    doctor: object
    stop_rule: Callable
    final_option: Callable
    outcome_fn: Callable
    reflect: Callable
    generate: Callable
    judge: Callable
    write: Callable
    behavior_logprobs: Callable | None = None


def reference_groups(tree, anchors=None):
    if tree.rollout_budget != 'reference_spine' or not tree.leaves:
        raise ValueError('Anchored groups require a reference rollout')
    lookup = {node.node_id: node for node in tree.nodes}
    reference = tree.leaves[0]
    groups = []
    for ref_id in reference.path:
        ref = lookup[ref_id]
        if anchors is not None and ref.depth not in anchors:
            continue
        siblings = [node for node in tree.nodes if node.parent_id == ref.parent_id]
        if len(siblings) != 4 or any(node.prefix != ref.prefix for node in siblings):
            raise ValueError('Each reference turn needs four same-prefix replies')
        members = []
        for index, node in enumerate(siblings):
            continuations = [leaf for leaf in tree.leaves if node.node_id in leaf.path]
            if node.node_id == ref_id: leaf = reference
            elif len(continuations) == 1: leaf = continuations[0]
            else: raise ValueError('An alternative must have exactly one continuation')
            members.append({'response_id': f'S{index}', 'node_id': node.node_id, 'reply': node.response,
                            'leaf_id': leaf.leaf_id, 'outcome': leaf.outcome, 'final_option': leaf.final_option})
        groups.append({'group_id': f'{tree.case_id}/turn_{ref.depth}', 'anchor': ref.depth,
                       'parent_id': ref.parent_id or 'root', 'prefix': to_dialogue(ref.prefix),
                       'partner_state': ref.prefix_partner_state, 'partner_status': ref.prefix_doctor_status,
                       'members': members})
    return groups


def candidate_from_source_outcome(draft, group):
    """Choose endpoints in code; the generator never supplies a direction label."""
    source = next((m for m in group['members'] if m['response_id'] == draft['source_response_id']), None)
    if source is None or source['outcome'] not in (0.0, 1.0):
        raise ValueError('Selected source needs a resolved binary outcome')
    levels = draft['criteria']
    if set(levels) != {'observed', 'middle', 'contrast'}:
        raise ValueError('Decisive candidate requires observed/middle/contrast criteria')
    observed_endpoint = 2 if source['outcome'] == 1.0 else 0
    return {**draft, 'kind': 'helpful' if observed_endpoint == 2 else 'harmful',
            'criteria': {str(observed_endpoint): levels['observed'], '1': levels['middle'],
                         str(2 - observed_endpoint): levels['contrast']}}


def validate_candidate(candidate, group, reflection, *, direction_from_outcome=False):
    required = {'name', 'kind', 'source_response_id', 'state', 'criteria', 'b', 'b_star', 'evidence_quote'}
    if not isinstance(candidate, dict) or set(candidate) != required: raise ValueError('Incomplete group rubric')
    if candidate['kind'] not in ('helpful', 'harmful'): raise ValueError('Unknown rubric direction')
    for field in ('name', 'b', 'b_star', 'evidence_quote'):
        if not isinstance(candidate[field], str) or not candidate[field].strip(): raise ValueError(f'Empty {field}')
    if candidate['b'].strip() == candidate['b_star'].strip(): raise ValueError('Contrast behavior equals observed behavior')
    if not isinstance(candidate['state'], dict) or set(candidate['state']) != {'partner', 'assistant'}:
        raise ValueError('Rubric state requires partner and assistant contexts')
    if not isinstance(candidate['criteria'], dict) or set(candidate['criteria']) != {'0', '1', '2'}:
        raise ValueError('Rubric requires criteria for 0, 1 and 2')
    if any(not isinstance(text, str) or not text.strip() for section in ('state', 'criteria') for text in candidate[section].values()):
        raise ValueError('Rubric state and criteria must be substantive')
    source = next((m for m in group['members'] if m['response_id'] == candidate['source_response_id']), None)
    expected_outcome = 1.0 if candidate['kind'] == 'helpful' else 0.0
    if source is None or source['outcome'] != expected_outcome:
        raise ValueError('Helpful sources require success; harmful sources require failure')
    if not quote_in_reply(candidate['evidence_quote'], source['reply']):
        raise ValueError('Observed behavior quote is not in source reply')
    if not direction_from_outcome:
        observations = reflection[candidate['kind']]
        if not any(item['response_id'] == source['response_id'] for item in observations):
            raise ValueError('Rubric source is absent from partner comparative reflection')
    public = public_rubric(candidate)
    serialized = json.dumps(public, ensure_ascii=False)
    if re.search(r'\bS[0-3]\b', serialized) or any(m['node_id'] in serialized or m['leaf_id'] in serialized for m in group['members']):
        raise ValueError('Public rubric contains source identifiers')
    return source


def public_rubric(candidate):
    return {key: candidate[key] for key in ('name', 'state', 'criteria')}


def reward_request(case, group, rubrics, run_seed):
    order = list(group['members'])
    random.Random(derive_seed(run_seed, group['group_id'], 'blind_reward_order', 0)).shuffle(order)
    mapping = {f'A{i}': member['node_id'] for i, member in enumerate(order)}
    request = {'case': {key: case.get(key) for key in ('scenario', 'question', 'options')},
               'shared_prefix': group['prefix'],
               'rubrics': [{'rubric_id': f'R{i}', **public_rubric(rubric)} for i, rubric in enumerate(rubrics)],
               'responses': [{'response_id': f'A{i}', 'text': member['reply']} for i, member in enumerate(order)]}
    return request, mapping


def ranking_reward_values(result, mapping):
    """Best-first tie groups; ties receive the mean reward of occupied ranks."""
    if not isinstance(result, dict) or set(result) != {'ranking'} or not isinstance(result['ranking'], list):
        raise ValueError('Judge must return ranking array')
    if len(mapping) != 4 or not 1 <= len(result['ranking']) <= 4:
        raise ValueError('Judge must rank all four replies')
    rows, seen, position = {}, set(), 1
    for group in result['ranking']:
        if not isinstance(group, dict) or set(group) != {'response_ids', 'reason'}:
            raise ValueError('Incomplete ranking group')
        ids, reason = group['response_ids'], group['reason']
        if not isinstance(ids, list) or not 1 <= len(ids) <= 4 or not isinstance(reason, str) or not reason.strip():
            raise ValueError('Ranking group requires response IDs and a rubric-grounded reason')
        if any(not isinstance(key, str) or key not in mapping for key in ids) or len(set(ids)) != len(ids) or seen.intersection(ids):
            raise ValueError('Unknown or duplicate ranked response')
        mean_rank = position + (len(ids) - 1) / 2
        for key in ids:
            rows[mapping[key]] = {'rank': mean_rank, 'tie_size': len(ids), 'reason': reason,
                                  'reward': (4 - mean_rank) / 3}
        seen.update(ids)
        position += len(ids)
    if seen != set(mapping):
        raise ValueError('Ranking must include every response exactly once')
    return rows


def reward_values(result, mapping, rubric_count, *, mode=None):
    if mode == 'rubric_rank_v1':
        return ranking_reward_values(result, mapping)
    if mode is not None:
        raise ValueError('Unknown group reward mode')
    if not isinstance(result, dict) or set(result) != {'scores'} or not isinstance(result['scores'], list):
        raise ValueError('Judge must return scores array')
    rows = result['scores']
    if len(rows) != 4: raise ValueError('Judge must score all four replies')
    scores = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != {'response_id', 'rubric_scores'}:
            raise ValueError('Incomplete judge row')
        key = row['response_id']
        values = row['rubric_scores']
        if key not in mapping or key in scores or not isinstance(values, list) or len(values) != rubric_count:
            raise ValueError('Missing, duplicate or misaligned judge row')
        if any(type(value) is not int or value not in (0, 1, 2) for value in values):
            raise ValueError('Scores must be integer 0/1/2')
        scores[key] = values
    return {mapping[key]: {'scores': values, 'reward': sum(values) / (2 * rubric_count)} for key, values in scores.items()}


def continue_contrast(case, group, text, *, env, cfg, seed, cache, candidate_id):
    """Resume from the exact source prefix/status; the instruction never enters the dialogue."""
    ro = cfg['rollout']
    dialogue = list(group['prefix'])
    status = group['partner_status']
    private_states = []
    depth = group['anchor']
    reply = text
    ident = {'case': case['case_id'], 'candidate': candidate_id, 'policy': cfg['_source_policy'],
             'protocol': cfg['_protocol'], 'seed': seed}
    while True:
        prefix = list(dialogue)
        dialogue.append({'speaker': 'agent', 'stage': f'A{depth}', 'round': depth, 'text': reply})
        why = env.stop_rule(depth, prefix[-1]['text'], reply) if depth >= minimum_agent_turns(env.doctor) else None
        repeat_after = ro.get('halting', {}).get('repeat_agent_reply_after_turns')
        if why is None and repeat_after is not None and depth >= repeat_after and repeated_agent_reply(prefix, reply):
            why = 'agent_repeated_reply'
        why = why or ('max_agent_turns' if depth == ro['rounds'] else None)
        if why is not None: break
        ds = doctor_seed(seed, candidate_id, 'verification_doctor', depth + 1)
        nd = cache.call('group_verification_doctor', {**ident, 'turn': depth + 1},
                        lambda: env.doctor.turn(case, dialogue, depth + 1, ro['rounds'], ds))
        dialogue.append({'speaker': 'doctor', 'stage': f'D{depth + 1}', 'round': depth + 1, 'text': nd['visible']})
        private_states.append({'turn': depth + 1, 'state': nd.get('private_state')})
        why = env.doctor.halt(agent_turns=depth, status=nd['status'], prev_status=status,
                             last_agent=reply, options=case['options'])
        if why is not None: break
        status = nd['status']
        depth += 1
        agent_seed = derive_seed(seed, candidate_id, 'verification_agent', depth)
        trace = cache.call('group_verification_agent', {**ident, 'turn': depth},
                           lambda: env.agent.sample(case, dialogue, depth, agent_seed))
        reply = trace['completion_text']
    fs = doctor_seed(seed, candidate_id, 'verification_final', 0)
    final = cache.call('group_verification_final', ident, lambda: env.doctor.final(case, dialogue, fs))
    option = env.final_option(final)
    dialogue.append({'speaker': 'doctor', 'stage': 'FINAL', 'round': 0, 'text': final})
    return {'seed': seed, 'final_option': option, 'outcome': env.outcome_fn(case, option),
            'stop_reason': why, 'dialogue': dialogue, 'partner_states': private_states}


def learn_case(case, tree, *, env, cfg, cache, run_seed, protocol_sha, progress=None, groups=None):
    groups = reference_groups(tree) if groups is None else groups
    node_lookup = {node.node_id: node for node in tree.nodes}
    targets = {'nodes': {}, 'groups': {}, 'reward_weights': {'w_outcome': 0.0, 'w_rubric': 1.0}}
    book, reports = [], []
    cfg = {**cfg, '_source_policy': tree.policy_sha256, '_protocol': protocol_sha}
    for group in groups:
        report = {**group, 'status': 'reflection', 'candidates': [], 'kept_rubric_ids': [], 'reward': None, 'timings': {}}
        ident = {'case': case['case_id'], 'group': group['group_id'], 'policy': tree.policy_sha256,
                 'protocol': protocol_sha, 'group_sha256': sha256_json(group)}
        if progress: progress(report)
        stage_started = time.perf_counter()
        reflection = cache.call('group_reflection', ident, lambda: env.reflect(case, group))
        report['timings']['reflection_seconds'] = time.perf_counter() - stage_started
        report['reflection'] = reflection
        if not reflection['valid']:
            report['status'] = 'reflection_invalid'; reports.append(report)
            if progress: progress(report)
            continue
        report['status'] = 'generation'
        if progress: progress(report)
        stage_started = time.perf_counter()
        generation = cache.call('group_generation', {**ident, 'reflection_sha256': sha256_json(reflection)},
                                lambda: env.generate(case, group, reflection['parsed']))
        report['timings']['generation_seconds'] = time.perf_counter() - stage_started
        report['generation'] = generation
        if not generation['valid']:
            report['status'] = 'generation_invalid'; reports.append(report)
            if progress: progress(report)
            continue
        candidates = generation['parsed']['candidates']
        if not candidates:
            report.update(status='no_candidate_rubric',
                          no_candidate_reason=generation.get('no_candidate_reason', 'generator_returned_empty'))
            reports.append(report)
            if progress: progress(report)
            continue
        # The generator structures the partner's prose in the same call that
        # creates candidates. Keep it beside the raw private reflection, never
        # append either representation to the visible dialogue or Agent traces.
        direction_from_outcome = cfg['group_learning'].get('candidate_direction') == 'source_outcome_v1'
        comparison = generation['parsed'].get('comparative_reflection', reflection['parsed'])
        if not direction_from_outcome:
            report['structured_reflection'] = comparison
        kinds = [candidate.get('kind') for candidate in candidates]
        cap = cfg['group_learning']['max_rubrics_per_group']
        if len(candidates) > cap or len(set(kinds)) != len(kinds):
            report['status'] = 'generation_invalid'; report['error'] = f'At most {cap} candidates, with no repeated direction'
            reports.append(report)
            if progress: progress(report)
            continue
        kept = []
        report['status'] = 'verification'
        if progress: progress(report)
        stage_started = time.perf_counter()
        for index, candidate in enumerate(candidates):
            candidate_id = f"{group['group_id']}/rubric_{index}_{sha256_json(candidate)[:12]}"
            item = {'rubric_id': candidate_id, 'group_id': group['group_id'], 'anchor': group['anchor'],
                    'candidate': candidate, 's': None, 'retained': False, 'reruns': []}
            report['candidates'].append(item)
            try: source = validate_candidate(candidate, group, comparison, direction_from_outcome=direction_from_outcome)
            except ValueError as exc:
                item.update(status='candidate_invalid', error=str(exc)); continue
            item.update(source_node_id=source['node_id'], source_leaf_id=source['leaf_id'], source_outcome=source['outcome'])
            if direction_from_outcome:
                item['observed_score_endpoint'] = 2 if source['outcome'] == 1.0 else 0
            write_seed = derive_seed(run_seed, candidate_id, 'contrast_reply', 0)
            write_ident = {**ident, 'candidate_sha256': sha256_json(candidate), 'seed': write_seed}
            text = cache.call('group_contrast_reply', write_ident, lambda: env.write(case, group, candidate, write_seed))
            if not isinstance(text, str) or not text.strip(): raise ValueError('Empty contrast reply')
            item['contrast_reply'] = text
            item['write_seed'] = write_seed
            def rerun_once(rerun):
                seed = derive_seed(run_seed, candidate_id, 'contrast_continuation', rerun)
                return continue_contrast(case, group, text, env=env, cfg=cfg, seed=seed, cache=cache, candidate_id=candidate_id)
            workers = min(3, cfg.get('execution', {}).get('rerun_workers', 1))
            if workers > 1:
                from concurrent.futures import ThreadPoolExecutor
                with ThreadPoolExecutor(max_workers=workers, thread_name_prefix='group-verification') as pool:
                    item['reruns'] = list(pool.map(rerun_once, range(3)))
            else:
                item['reruns'] = [rerun_once(rerun) for rerun in range(3)]
            if any(result['outcome'] not in (0.0, 1.0) for result in item['reruns']):
                item['status'] = 'verification_unresolved'; continue
            flips = sum(result['outcome'] != source['outcome'] for result in item['reruns'])
            item.update(s=flips / 3, successes=flips, retained=flips >= 2, status='KEEP' if flips >= 2 else 'REMOVE')
            if item['retained']:
                entry = {**candidate, 'rubric_id': candidate_id, 'group_id': group['group_id'],
                         'case_id': case['case_id'], 'anchor': group['anchor'], 's': item['s']}
                kept.append(entry); book.append(entry)
        report['kept_rubric_ids'] = [entry['rubric_id'] for entry in kept]
        report['timings']['verification_seconds'] = time.perf_counter() - stage_started
        if not kept:
            report['status'] = 'no_verified_rubric'; reports.append(report)
            if progress: progress(report)
            continue
        request, mapping = reward_request(case, group, kept, run_seed)
        report['status'] = 'reward'
        if progress: progress(report)
        stage_started = time.perf_counter()
        judged = cache.call('group_reward', {**ident, 'request_sha256': sha256_json(request)}, lambda: env.judge(request))
        report['timings']['reward_seconds'] = time.perf_counter() - stage_started
        report['reward'] = {'request': request, 'response_mapping': mapping, 'judge': judged,
                            'mode': cfg['group_learning'].get('reward_mode') or 'rubric_score_v1'}
        try:
            if not judged['valid']: raise ValueError(judged.get('error', 'Judge execution invalid'))
            rows = reward_values(judged['parsed'], mapping, len(kept),
                                 mode=cfg['group_learning'].get('reward_mode'))
        except ValueError as exc:
            report.update(status='reward_unresolved', error=str(exc)); reports.append(report)
            if progress: progress(report)
            continue
        ids = [member['node_id'] for member in group['members']]
        for node_id in ids:
            trace = node_lookup[node_id].trace
            if len(trace.get('segments') or [trace]) != 1:
                raise ValueError('Group loss requires one visible reply segment per response')
            detail = ({'rank': rows[node_id]['rank'], 'tie_size': rows[node_id]['tie_size']}
                      if cfg['group_learning'].get('reward_mode') == 'rubric_rank_v1'
                      else {'rubric_scores': rows[node_id]['scores']})
            targets['nodes'][node_id] = {'outcome_component': None, 'R_A': rows[node_id]['reward'],
                                         'reward': rows[node_id]['reward'], **detail}
        targets['groups'][group['parent_id']] = ids
        values = [rows[node_id]['reward'] for node_id in ids]
        if cfg['group_learning'].get('reward_mode') == 'rubric_rank_v1':
            report['ranks'] = [rows[node_id]['rank'] for node_id in ids]
            report['reward']['ranking_values'] = rows
        report.update(status='scored', rewards=values, advantages=advantages(values, 'group_zscore'))
        reports.append(report)
        if progress: progress(report)
    return summarize_case(case, tree, env, groups, reports, book, targets)


def summarize_case(case, tree, env, groups, reports, book, targets):
    """Deterministic final aggregation of independent anchored group results."""
    node_lookup = {node.node_id: node for node in tree.nodes}
    assessed = [report for report in reports if report['status'] == 'scored']
    verified = [item for report in reports for item in report['candidates'] if item['s'] is not None]
    all_candidates = [item for report in reports for item in report['candidates']]
    zero = sum(all(value == 0 for value in report['advantages']) for report in assessed)
    lengths = [{'node_id': member['node_id'], 'anchor': group['anchor'],
                'tokens': len(node_lookup[member['node_id']].trace.get('completion_ids', [])),
                'characters': len(member['reply'])} for group in groups for member in group['members']]
    initial = groups[0]['partner_state'] if groups else None
    initial_correct = env.outcome_fn(case, initial['current_option']) if initial and initial['current_option'] != 'none' else None
    metrics = {'groups_total': len(groups), 'groups_with_rubric': sum(bool(report['kept_rubric_ids']) for report in reports),
               'ranked_groups': sum('ranks' in report for report in assessed),
               'ranking_tied_groups': sum('ranks' in report and len(set(report['ranks'])) < 4 for report in assessed),
               'reference_agent_turns': len(tree.leaves[0].path),
               'reflection_invalid_groups': sum(report['status'] == 'reflection_invalid' for report in reports),
               'reflection_invalid_fraction': sum(report['status'] == 'reflection_invalid' for report in reports) / len(groups) if groups else None,
               'groups_scored': len(assessed), 'zero_advantage_groups': zero,
               'rubric_group_fraction': sum(bool(report['kept_rubric_ids']) for report in reports) / len(groups) if groups else None,
               'zero_advantage_fraction': zero / len(assessed) if assessed else None,
               'candidates_total': len(all_candidates), 'verification_resolved': len(verified),
               'verification_passed': len(book), 'verification_pass_rate': len(book) / len(verified) if verified else None,
               'verification_s': [item['s'] for item in verified],
               'verification_rows': [{'rubric_id': item['rubric_id'], 'group_id': report['group_id'],
                   'kind': item['candidate']['kind'], 's': item['s'], 'retained': item['retained'], 'status': item.get('status')}
                   for report in reports for item in report['candidates']],
               'response_lengths': lengths, 'mean_response_tokens': sum(item['tokens'] for item in lengths) / len(lengths) if lengths else None,
               'partner_initial_state': initial, 'partner_initial_correct': initial_correct,
               'reference_outcome': tree.leaves[0].outcome,
               'rescue': initial_correct == 0.0 and tree.leaves[0].outcome == 1.0 if initial_correct is not None else None,
               'break': initial_correct == 1.0 and tree.leaves[0].outcome == 0.0 if initial_correct is not None else None}
    return {'groups': reports, 'rubric_book': book, 'agent_targets': targets, 'group_metrics': metrics}


class GroupTrainer:
    """The case coordinator interface; the parent runner owns the batch update."""
    def __init__(self, cfg, case, env, policy_sha, protocol_sha, execution, updater=None):
        self.cfg, self.case, self.env = cfg, case, env
        self.out = Path(cfg['output_dir']); self.out.mkdir(parents=True, exist_ok=True)
        self.policy_sha, self.protocol_sha, self.execution, self.updater = policy_sha, protocol_sha, execution, updater
        self.cache = CallCache(self.out / 'calls.jsonl')
        self.config_sha = sha256_json(cfg)
        committed = read_json(self.out / 'run_state.json') if (self.out / 'run_state.json').exists() else {}
        if committed and (committed['protocol_sha256'] != protocol_sha or committed['config_sha256'] != self.config_sha):
            raise RuntimeError('Group case protocol/config changed; use a fresh output directory')
        if committed and committed['source_policy_sha256'] != policy_sha():
            raise RuntimeError('Group case source policy changed')
        self.state = SimpleNamespace(policy_sha256=policy_sha(), case_cursor=committed.get('case_cursor', 0))

    def validate_member_interventions(self, record):
        if record.get('group_protocol') != PROTOCOL or record['policy_sha256'] != self.state.policy_sha256:
            raise RuntimeError('Group case protocol/policy mismatch')
        if record['protocol_sha256'] != self.protocol_sha or record['case_id'] != self.case['case_id']:
            raise RuntimeError('Group case identity mismatch')
        book = {entry['rubric_id']: entry for entry in record['rubric_book']}
        for group in record['groups']:
            for rubric_id in group['kept_rubric_ids']:
                rubric = book[rubric_id]
                if (rubric['case_id'] != self.case['case_id'] or rubric['group_id'] != group['group_id']
                        or rubric['anchor'] != group['anchor']):
                    raise RuntimeError('Saved rubric is attached to another group')

    def step(self):
        if self.state.case_cursor:
            record = read_json(self.out / 'cases' / 'case_000000.json')
            self.validate_member_interventions(record)
            return record
        from concurrent.futures import ThreadPoolExecutor
        from threading import RLock
        from copy import deepcopy
        started = time.perf_counter()
        cfg, env = self.cfg, self.env
        ro = cfg['rollout']
        streaming = cfg['group_learning'].get('pipeline') == 'ready_groups_v1'
        reports, futures, pipeline_times = {}, {}, {}
        progress_lock = RLock()
        def progress(report):
            with progress_lock:
                snapshot = deepcopy(report)
                snapshot['pipeline'] = dict(pipeline_times.get(report['anchor'], {}))
                reports[report['group_id']] = snapshot
                ordered = sorted(reports.values(), key=lambda row: row['anchor'])
                atomic_write_json(self.out / 'group_progress.json', {'protocol': PROTOCOL, 'groups': ordered})

        def process_group(snapshot, group):
            anchor = group['anchor']
            with progress_lock:
                pipeline_times[anchor]['started_at_seconds'] = time.perf_counter() - started
            result = learn_case(self.case, snapshot, env=env, cfg=cfg, cache=self.cache,
                                run_seed=cfg['schedule']['seed'], protocol_sha=self.protocol_sha,
                                progress=progress, groups=[group])
            with progress_lock:
                pipeline_times[anchor]['completed_at_seconds'] = time.perf_counter() - started
                result['groups'][0]['pipeline'] = dict(pipeline_times[anchor])
            progress(result['groups'][0])
            return result

        pipeline_workers = cfg.get('execution', {}).get('group_pipeline_workers', 4)
        with ThreadPoolExecutor(max_workers=pipeline_workers, thread_name_prefix='ready-group') as pool:
            def submit_group(snapshot, anchor):
                group = reference_groups(snapshot, anchors={anchor})[0]
                with progress_lock:
                    if anchor in futures:
                        raise RuntimeError('Duplicate ready group')
                    pipeline_times[anchor] = {'ready_at_seconds': time.perf_counter() - started}
                    futures[anchor] = pool.submit(process_group, snapshot, group)

            tree = build_tree(self.case, max_agent_turns=ro['rounds'], branching=ro['branching'],
                run_seed=cfg['schedule']['seed'], policy_sha256=self.state.policy_sha256, protocol_sha256=self.protocol_sha,
                agent=env.agent, doctor=env.doctor, stop_rule=env.stop_rule, final_option=env.final_option,
                outcome_fn=env.outcome_fn, cache=self.cache, budget=ro['budget'], behavior_logprobs=env.behavior_logprobs,
                workers=self.execution.get('doctor_workers', 1), agent_workers=self.execution.get('agent_workers', 1),
                repeat_agent_reply_after_turns=ro.get('halting', {}).get('repeat_agent_reply_after_turns'),
                on_group_ready=submit_group if streaming else None,
                reference_group_workers=cfg.get('execution', {}).get('group_rollout_workers', 4) if streaming else 1)
            rollout_seconds = time.perf_counter() - started
            atomic_write_json(self.out / 'group_rollout.json', {'tree': to_dict(tree), 'counts': tree_counts(tree)})
            if streaming:
                groups = reference_groups(tree)
                if set(futures) != {group['anchor'] for group in groups}:
                    raise RuntimeError('Incomplete ready-group pipeline')
                results = [futures[group['anchor']].result() for group in groups]
                book = [entry for result in results for entry in result['rubric_book']]
                all_reports = [row for result in results for row in result['groups']]
                targets = {'nodes': {}, 'groups': {}, 'reward_weights': {'w_outcome': 0.0, 'w_rubric': 1.0}}
                for result in results:
                    targets['nodes'].update(result['agent_targets']['nodes'])
                    targets['groups'].update(result['agent_targets']['groups'])
                learned = summarize_case(self.case, tree, env, groups, all_reports, book, targets)
            else:
                learned = learn_case(self.case, tree, env=env, cfg=cfg, cache=self.cache,
                                     run_seed=cfg['schedule']['seed'], protocol_sha=self.protocol_sha, progress=progress)
        if self.policy_sha() != self.state.policy_sha256: raise RuntimeError('Policy changed inside group case')
        case_seconds = time.perf_counter() - started
        record = {'case_id': tree.case_id, 'case_cursor': 0, 'policy_sha256': self.state.policy_sha256,
                  'protocol_sha256': self.protocol_sha, 'group_protocol': PROTOCOL, 'tree': to_dict(tree),
                  'counts': tree_counts(tree), 'prompt_sha256': cfg['rubric']['prompt_sha256'], **learned,
                  'timings': {'case_wall_seconds': case_seconds, 'rollout_wall_seconds': rollout_seconds,
                              'post_rollout_wait_seconds': case_seconds - rollout_seconds,
                              'group_pipeline': pipeline_times}}
        if self.updater is not None:
            from src.online_rl.tree.five_gpu import update_batch
            record['batch_update'] = update_batch(self.updater, [record], self.state.policy_sha256, self.policy_sha)
        atomic_write_json(self.out / 'cases' / 'case_000000.json', record)
        atomic_write_json(self.out / 'group_metrics.json', learned['group_metrics'])
        atomic_write_json(self.out / 'run_state.json', {'config_sha256': self.config_sha, 'protocol_sha256': self.protocol_sha,
                          'source_policy_sha256': self.state.policy_sha256, 'case_cursor': 1})
        self.state.case_cursor = 1
        return record
