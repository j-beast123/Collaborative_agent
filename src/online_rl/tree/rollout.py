"""Case-level branching rollout: Doctor opens; at every Agent decision point each live branch samples `branching`
Agent replies from one identical prompt; each non-terminal reply gets one Doctor response; up to max_agent_turns.
A branch ends on the stop rule or the turn limit, then the Doctor FINAL decides the outcome (no feedback, no rubric)."""
from __future__ import annotations
from typing import Callable, Optional
import re, unicodedata
from src.online_rl.integrated.continuation import derive_seed
from src.online_rl.tree.cache import CallCache
from src.online_rl.tree.records import AgentNode, Leaf, Tree, to_turns


def repeated_agent_reply(prefix, reply):
    """Exact normalized repetition of an earlier visible Agent reply. No model/API judgment."""
    norm=lambda value: ' '.join(re.findall(r'\w+',unicodedata.normalize('NFKC',value).casefold()))
    current=norm(reply)
    return bool(current) and any(d.get('speaker')=='agent' and norm(d.get('text',''))==current for d in prefix)


def build_tree(case: dict, *, max_agent_turns: int, branching: int, run_seed: int, policy_sha256: str, protocol_sha256: str,
               agent, doctor, stop_rule: Callable, final_option: Callable, outcome_fn: Callable, cache: CallCache,
               behavior_logprobs: Optional[Callable] = None, workers: int = 1, opening: Optional[Callable] = None, agent_workers: int = 1,
               on_leaf: Optional[Callable] = None, repeat_agent_reply_after_turns: Optional[int] = None,
               sampling_unit: str = 'response_tree') -> Tree:
    """agent.sample(case, prefix, turn, seed) -> trace with 'completion_text'; doctor.turn(case, dialogue, round, rounds) and
    doctor.final(case, dialogue) -> text. Ancestor replies depend on the policy, so every key carries policy_sha256."""
    if max_agent_turns < 1 or branching < 1: raise ValueError('max_agent_turns and branching must be >= 1')
    if sampling_unit not in ('response_tree', 'independent_trajectory'):
        raise ValueError(f'unknown rollout sampling_unit {sampling_unit!r}')
    if opening is not None and not getattr(doctor, 'status_protocol', False): raise ValueError('frozen openings need Doctor protocol v2')
    if getattr(doctor, 'status_protocol', False):
        return _build_tree_v2(case, max_agent_turns=max_agent_turns, branching=branching, run_seed=run_seed, policy_sha256=policy_sha256, opening=opening,
                              protocol_sha256=protocol_sha256, agent=agent, doctor=doctor, stop_rule=stop_rule, final_option=final_option,
                              outcome_fn=outcome_fn, cache=cache, behavior_logprobs=behavior_logprobs, workers=workers, agent_workers=agent_workers,on_leaf=on_leaf,
                              repeat_agent_reply_after_turns=repeat_agent_reply_after_turns, sampling_unit=sampling_unit)
    cid = case['case_id']; ident = {'case': cid, 'policy': policy_sha256, 'protocol': protocol_sha256}
    d1 = cache.call('doctor_turn', {**ident, 'path': '', 'round': 1}, lambda: doctor.turn(case, [], 1, max_agent_turns))
    frontier = [('', None, [{'speaker': 'doctor', 'stage': 'D1', 'round': 1, 'text': d1}])]
    nodes, leaves = [], []
    for depth in range(1, max_agent_turns + 1):
        nxt = []
        for path, parent, prefix in frontier:
            for j in range(branching if depth == 1 or sampling_unit == 'response_tree' else 1):
                cpath = f'{path}.{j}' if path else str(j); nid = f'{cid}/{cpath}'; seed = derive_seed(run_seed, nid, 'agent', 0)

                def sample(prefix=prefix, seed=seed, depth=depth):
                    tr = agent.sample(case, prefix, depth, seed)
                    if behavior_logprobs is not None: tr = {**tr, 'behavior_logprobs': behavior_logprobs([tr])[0]}   # pi_old, before any update
                    return tr
                trace = cache.call('agent', {**ident, 'path': cpath, 'seed': seed}, sample)
                reply = trace['completion_text']
                dialogue = prefix + [{'speaker': 'agent', 'stage': f'A{depth}', 'round': depth, 'text': reply}]
                why = stop_rule(depth, prefix[-1]['text'], reply)
                if (why is None and repeat_agent_reply_after_turns is not None and depth >= repeat_agent_reply_after_turns
                        and repeated_agent_reply(prefix,reply)): why='agent_repeated_reply'
                why = why or ('max_agent_turns' if depth == max_agent_turns else None)
                nd = None
                if why is None:
                    nd = cache.call('doctor_turn', {**ident, 'path': cpath, 'round': depth + 1}, lambda d=dialogue, r=depth + 1: doctor.turn(case, d, r, max_agent_turns))
                    nxt.append((cpath, nid, dialogue + [{'speaker': 'doctor', 'stage': f'D{depth + 1}', 'round': depth + 1, 'text': nd}]))
                nodes.append(AgentNode(nid, parent, cid, depth, to_turns(prefix), reply, trace, policy_sha256, seed, why, why is not None, nd))
                if why is not None:
                    final = cache.call('final', {**ident, 'path': cpath}, lambda d=dialogue: doctor.final(case, d))
                    opt = final_option(final)
                    full = dialogue + [{'speaker': 'doctor', 'stage': 'FINAL', 'round': 0, 'text': final}]
                    path_ids = tuple(f'{cid}/{".".join(cpath.split(".")[:k])}' for k in range(1, depth + 1))
                    leaves.append(Leaf(nid, cid, path_ids, to_turns(full), final, opt, outcome_fn(case, opt), why))
                    if on_leaf is not None:on_leaf(leaves[-1])
        frontier = nxt
    return Tree(cid, policy_sha256, max_agent_turns, branching, tuple(nodes), tuple(leaves))


def _build_tree_v2(case: dict, *, max_agent_turns: int, branching: int, run_seed: int, policy_sha256: str, protocol_sha256: str,
                   agent, doctor, stop_rule: Callable, final_option: Callable, outcome_fn: Callable, cache: CallCache,
                   behavior_logprobs: Optional[Callable] = None, workers: int = 1, opening: Optional[Callable] = None,
                   agent_workers: int = 1, on_leaf: Optional[Callable] = None,
                   repeat_agent_reply_after_turns: Optional[int] = None,
                   sampling_unit: str = 'response_tree') -> Tree:
    """build_tree under Doctor protocol v2 (src/online_rl/doctor.py): every Doctor call is seeded from its node identity
    (doctor_seed), its hidden status is cut from the visible reply and kept on the node, and after each Doctor turn the
    shared halting rule may end the branch ('doctor_halt'; that Doctor message stays in the dialogue and FINAL follows).
    Agent sampling, the explicit stop rules, the turn limit and FINAL are as in build_tree.
    Execution: per depth, all Agent replies are sampled first in the fixed node order (one frozen policy, a seed per reply);
    the depth's Doctor turns, then its FINALs, are independent calls (own seed and cache key each) and run on `workers`
    threads. Node / leaf order and every value are the same as with workers = 1.
    opening(case, seed) -> D1: the frozen shared opening (src/online_rl/doctor_openings.py), read, never generated; only D1.
    agent_workers > 1 (execution only, Agent replicas): the depth's Agent actions are independent given their seeds and are
    sampled concurrently; node order, seeds, cache keys and values are those of the sequential loop."""
    from concurrent.futures import ThreadPoolExecutor
    from src.online_rl.doctor import doctor_seed, minimum_agent_turns
    cid = case['case_id']; ident = {'case': cid, 'policy': policy_sha256, 'protocol': protocol_sha256}
    pmap = (lambda f, xs: list(ThreadPoolExecutor(workers).map(f, xs))) if workers > 1 else (lambda f, xs: [f(x) for x in xs])
    amap = (lambda f, xs: list(ThreadPoolExecutor(agent_workers).map(f, xs))) if agent_workers > 1 else (lambda f, xs: [f(x) for x in xs])
    s1 = doctor_seed(run_seed, cid, 'doctor', 1)
    d1 = (opening(case, s1) if opening is not None else
          cache.call('doctor_turn', {**ident, 'path': '', 'round': 1, 'seed': s1}, lambda: doctor.turn(case, [], 1, max_agent_turns, s1)))
    frontier = [('', None, [{'speaker': 'doctor', 'stage': 'D1', 'round': 1, 'text': d1['visible']}], d1['status'])]
    nodes, leaves = [], []
    for depth in range(1, max_agent_turns + 1):
        entries, specs = [], []
        for path, parent, prefix, prev_status in frontier:
            for j in range(branching if depth == 1 or sampling_unit == 'response_tree' else 1):
                cpath = f'{path}.{j}' if path else str(j); nid = f'{cid}/{cpath}'; seed = derive_seed(run_seed, nid, 'agent', 0)
                specs.append((cpath, nid, parent, prefix, prev_status, seed))

        def traced(sp):
            cpath, _, _, prefix, _, seed = sp

            def sample():
                tr = agent.sample(case, prefix, depth, seed)
                if behavior_logprobs is not None: tr = {**tr, 'behavior_logprobs': behavior_logprobs([tr])[0]}   # pi_old, before any update
                return tr
            return cache.call('agent', {**ident, 'path': cpath, 'seed': seed}, sample)
        for (cpath, nid, parent, prefix, prev_status, seed), trace in zip(specs, amap(traced, specs)):
            reply = trace['completion_text']
            dialogue = prefix + [{'speaker': 'agent', 'stage': f'A{depth}', 'round': depth, 'text': reply}]
            early = stop_rule(depth, prefix[-1]['text'], reply) if depth >= minimum_agent_turns(doctor) else None
            if (early is None and repeat_agent_reply_after_turns is not None and depth >= repeat_agent_reply_after_turns
                    and repeated_agent_reply(prefix,reply)): early='agent_repeated_reply'
            why = early or ('max_agent_turns' if depth == max_agent_turns else None)
            entries.append({'cpath': cpath, 'nid': nid, 'parent': parent, 'prefix': prefix, 'prev_status': prev_status, 'seed': seed, 'trace': trace,
                            'reply': reply, 'dialogue': dialogue, 'why': why, 'nd': None})

        def doctor_turn(e):
            ds = doctor_seed(run_seed, e['nid'], 'doctor', depth + 1)
            return cache.call('doctor_turn', {**ident, 'path': e['cpath'], 'round': depth + 1, 'seed': ds},
                              lambda: doctor.turn(case, e['dialogue'], depth + 1, max_agent_turns, ds))
        open_ = [e for e in entries if e['why'] is None]
        for e, nd in zip(open_, pmap(doctor_turn, open_)):
            e['nd'] = nd
            e['dialogue'] = e['dialogue'] + [{'speaker': 'doctor', 'stage': f'D{depth + 1}', 'round': depth + 1, 'text': nd['visible']}]
            e['why'] = doctor.halt(agent_turns=depth, status=nd['status'], prev_status=e['prev_status'], last_agent=e['reply'], options=case['options'])

        def final(e):
            fs = doctor_seed(run_seed, e['nid'], 'final', 0)
            text = cache.call('final', {**ident, 'path': e['cpath'], 'seed': fs}, lambda: doctor.final(case, e['dialogue'], fs))
            if on_leaf is not None:
                opt=final_option(text)
                full=e['dialogue']+[{'speaker':'doctor','stage':'FINAL','round':0,'text':text}]
                path_ids=tuple(f'{cid}/{".".join(e["cpath"].split(".")[:k])}' for k in range(1,depth+1))
                on_leaf(Leaf(e['nid'],cid,path_ids,to_turns(full),text,opt,outcome_fn(case,opt),e['why']))
            return text
        done = [e for e in entries if e['why'] is not None]
        finals = dict(zip((e['nid'] for e in done), pmap(final, done)))
        nxt = []
        for e in entries:
            nd = e['nd']
            nodes.append(AgentNode(e['nid'], e['parent'], cid, depth, to_turns(e['prefix']), e['reply'], e['trace'], policy_sha256, e['seed'], e['why'],
                                   e['why'] is not None, nd and nd['visible'], nd and {k: nd[k] for k in ('status', 'status_error', 'seed')}, e['prev_status']))
            if e['why'] is None:
                nxt.append((e['cpath'], e['nid'], e['dialogue'], nd['status'])); continue
            final_text = finals[e['nid']]; opt = final_option(final_text)
            full = e['dialogue'] + [{'speaker': 'doctor', 'stage': 'FINAL', 'round': 0, 'text': final_text}]
            path_ids = tuple(f'{cid}/{".".join(e["cpath"].split(".")[:k])}' for k in range(1, depth + 1))
            leaves.append(Leaf(e['nid'], cid, path_ids, to_turns(full), final_text, opt, outcome_fn(case, opt), e['why']))
        frontier = nxt
    return Tree(cid, policy_sha256, max_agent_turns, branching, tuple(nodes), tuple(leaves))

def tree_counts(tree: Tree) -> dict:
    return {'agent_nodes': len(tree.nodes), 'leaves': len(tree.leaves), 'doctor_turns': 1 + sum(n.next_doctor is not None for n in tree.nodes),
            'finals': len(tree.leaves), 'by_depth': {d: sum(n.depth == d for n in tree.nodes) for d in range(1, tree.max_agent_turns + 1)}}
