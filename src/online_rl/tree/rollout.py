"""Case-level rollout: fork after the shared Doctor opening, then continue without further forks.
A branch ends on the stop rule or the turn limit, then the Doctor FINAL decides the outcome (no feedback, no rubric)."""
from __future__ import annotations
from typing import Callable, Optional
import re, unicodedata
from src.online_rl.integrated.continuation import derive_seed
from src.online_rl.tree.cache import CallCache
from src.online_rl.tree.records import AgentNode, Leaf, Tree, to_turns

ROLLOUT_BUDGETS = ('initial_fanout',)


def repeated_agent_reply(prefix, reply):
    """Exact normalized repetition of an earlier visible Agent reply. No model/API judgment."""
    norm=lambda value: ' '.join(re.findall(r'\w+',unicodedata.normalize('NFKC',value).casefold()))
    current=norm(reply)
    return bool(current) and any(d.get('speaker')=='agent' and norm(d.get('text',''))==current for d in prefix)


def build_tree(case: dict, *, max_agent_turns: int, branching: int, run_seed: int, policy_sha256: str, protocol_sha256: str,
               agent, doctor, stop_rule: Callable, final_option: Callable, outcome_fn: Callable, cache: CallCache,
               behavior_logprobs: Optional[Callable] = None, workers: int = 1, opening: Optional[Callable] = None, agent_workers: int = 1,
               on_leaf: Optional[Callable] = None, on_leaf_nodes: Optional[Callable] = None, repeat_agent_reply_after_turns: Optional[int] = None,
               budget: str = 'initial_fanout', on_group_ready: Optional[Callable] = None) -> Tree:
    """agent.sample(case, prefix, turn, seed) -> trace with 'completion_text'; doctor.turn(case, dialogue, round, rounds) and
    doctor.final(case, dialogue) -> text. Ancestor replies depend on the policy, so every key carries policy_sha256."""
    if max_agent_turns < 1 or branching < 1: raise ValueError('max_agent_turns and branching must be >= 1')
    if budget not in ROLLOUT_BUDGETS: raise ValueError(f'unknown rollout.budget {budget!r}; expected {ROLLOUT_BUDGETS}')
    if opening is not None and not getattr(doctor, 'status_protocol', False): raise ValueError('frozen openings need Doctor protocol v2')
    if not getattr(doctor, 'status_protocol', False): raise ValueError('rollout requires the Doctor status protocol')
    builder = _build_initial_fanout
    kwargs = dict(max_agent_turns=max_agent_turns, run_seed=run_seed, policy_sha256=policy_sha256, opening=opening,
                  protocol_sha256=protocol_sha256, agent=agent, doctor=doctor, stop_rule=stop_rule, final_option=final_option,
                  outcome_fn=outcome_fn, cache=cache, behavior_logprobs=behavior_logprobs, workers=workers, agent_workers=agent_workers,
                  on_leaf=on_leaf, on_leaf_nodes=on_leaf_nodes, repeat_agent_reply_after_turns=repeat_agent_reply_after_turns, budget=budget)
    tree = builder(case, branching=branching, **kwargs)
    if on_group_ready is not None:
        on_group_ready(tree, 1)
    return tree


def _build_initial_fanout(case: dict, *, max_agent_turns: int, branching: int, run_seed: int, policy_sha256: str, protocol_sha256: str,
                   agent, doctor, stop_rule: Callable, final_option: Callable, outcome_fn: Callable, cache: CallCache,
                   behavior_logprobs: Optional[Callable] = None, workers: int = 1, opening: Optional[Callable] = None,
                   agent_workers: int = 1, on_leaf: Optional[Callable] = None, on_leaf_nodes: Optional[Callable] = None,
                   repeat_agent_reply_after_turns: Optional[int] = None, budget='initial_fanout',
                   frozen_state=None, start_depth=1) -> Tree:
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
    # Plan C may start at a frozen pre-action state at an absolute later turn.
    # Only newly sampled suffix nodes belong to leaf.path; shared-prefix actions
    # remain in the visible dialogue and never become suffix training targets.
    from concurrent.futures import ThreadPoolExecutor
    from src.online_rl.doctor import doctor_seed, minimum_agent_turns
    cid = case['case_id']; ident = {'case': cid, 'policy': policy_sha256, 'protocol': protocol_sha256}
    ident['rollout_budget'] = budget
    def mapped(fn, items, count):
        if count <= 1: return [fn(item) for item in items]
        with ThreadPoolExecutor(count) as pool:
            return list(pool.map(fn, items))
    pmap = lambda fn, items: mapped(fn, items, workers)
    amap = lambda fn, items: mapped(fn, items, agent_workers)
    if frozen_state is None:
        if start_depth != 1:
            raise ValueError('A later starting turn requires a frozen Doctor/history state')
        s1 = doctor_seed(run_seed, cid, 'doctor', 1)
        d1 = (opening(case, s1) if opening is not None else
              cache.call('doctor_turn', {**ident, 'path': '', 'round': 1, 'seed': s1}, lambda: doctor.turn(case, [], 1, max_agent_turns, s1)))
        prefix, status, private = [{'speaker': 'doctor', 'stage': 'D1', 'round': 1, 'text': d1['visible']}], d1['status'], d1.get('private_state')
    else:
        prefix, status, private = frozen_state
        if not 1 <= start_depth <= max_agent_turns or not prefix or prefix[-1]['speaker'] != 'doctor':
            raise ValueError('Invalid frozen continuation state')
        if (prefix[-1].get('round') != start_depth
                or sum(row['speaker'] == 'agent' for row in prefix) != start_depth - 1
                or any(row.get('stage') == 'FINAL' for row in prefix)):
            raise ValueError('Frozen history must end immediately before its target turn')
    frontier = [('', None, prefix, status, private)]
    nodes, leaves = [], []
    for depth in range(start_depth, max_agent_turns + 1):
        if not frontier: break
        entries, specs = [], []
        for path, parent, prefix, prev_status, prev_private in frontier:
            for j in (range(branching) if depth == start_depth else range(1)):
                cpath = f'{path}.{j}' if path else str(j); nid = f'{cid}/{cpath}'; seed = derive_seed(run_seed, nid, 'agent', 0)
                specs.append((cpath, nid, parent, prefix, prev_status, seed, prev_private))

        def traced(sp):
            cpath, _, _, prefix, _, seed, _ = sp

            def sample():
                tr = agent.sample(case, prefix, depth, seed)
                if behavior_logprobs is not None: tr = {**tr, 'behavior_logprobs': behavior_logprobs([tr])[0]}   # pi_old, before any update
                return tr
            return cache.call('agent', {**ident, 'path': cpath, 'seed': seed}, sample)
        for (cpath, nid, parent, prefix, prev_status, seed, prev_private), trace in zip(specs, amap(traced, specs)):
            reply = trace['completion_text']
            dialogue = prefix + [{'speaker': 'agent', 'stage': f'A{depth}', 'round': depth, 'text': reply}]
            early = stop_rule(depth, prefix[-1]['text'], reply) if depth >= minimum_agent_turns(doctor) else None
            if (early is None and repeat_agent_reply_after_turns is not None and depth >= repeat_agent_reply_after_turns
                    and repeated_agent_reply(prefix,reply)): early='agent_repeated_reply'
            why = early or ('max_agent_turns' if depth == max_agent_turns else None)
            entries.append({'cpath': cpath, 'nid': nid, 'parent': parent, 'prefix': prefix, 'prev_status': prev_status, 'prev_private': prev_private, 'seed': seed, 'trace': trace,
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

        def completed_node(e):
            nd=e['nd']
            return AgentNode(e['nid'],e['parent'],cid,depth,to_turns(e['prefix']),e['reply'],e['trace'],policy_sha256,e['seed'],e['why'],
                             e['why'] is not None,nd and nd['visible'],nd and {k:nd[k] for k in ('status','status_error','seed')},e['prev_status'],
                             prefix_partner_state=e['prev_private'], next_partner_state=nd and nd.get('private_state'))
        ancestor_lookup={node.node_id:node for node in nodes}
        def final(e):
            fs = doctor_seed(run_seed, e['nid'], 'final', 0)
            text = cache.call('final', {**ident, 'path': e['cpath'], 'seed': fs}, lambda: doctor.final(case, e['dialogue'], fs))
            if on_leaf is not None or on_leaf_nodes is not None:
                opt=final_option(text)
                full=e['dialogue']+[{'speaker':'doctor','stage':'FINAL','round':0,'text':text}]
                path_ids=tuple(f'{cid}/{".".join(e["cpath"].split(".")[:k])}' for k in range(1,len(e['cpath'].split('.'))+1))
                leaf=Leaf(e['nid'],cid,path_ids,to_turns(full),text,opt,outcome_fn(case,opt),e['why'])
                if on_leaf is not None:on_leaf(leaf)
                if on_leaf_nodes is not None:
                    on_leaf_nodes(leaf,tuple(ancestor_lookup[key] for key in path_ids[:-1])+(completed_node(e),))
            return text
        done = [e for e in entries if e['why'] is not None]
        finals = dict(zip((e['nid'] for e in done), pmap(final, done)))
        nxt = []
        for e in entries:
            nd = e['nd']
            nodes.append(AgentNode(e['nid'], e['parent'], cid, depth, to_turns(e['prefix']), e['reply'], e['trace'], policy_sha256, e['seed'], e['why'],
                                   e['why'] is not None, nd and nd['visible'], nd and {k: nd[k] for k in ('status', 'status_error', 'seed')}, e['prev_status'],
                                   prefix_partner_state=e['prev_private'], next_partner_state=nd and nd.get('private_state')))
            if e['why'] is None:
                nxt.append((e['cpath'], e['nid'], e['dialogue'], nd['status'], nd.get('private_state'))); continue
            final_text = finals[e['nid']]; opt = final_option(final_text)
            full = e['dialogue'] + [{'speaker': 'doctor', 'stage': 'FINAL', 'round': 0, 'text': final_text}]
            path_ids = tuple(f'{cid}/{".".join(e["cpath"].split(".")[:k])}' for k in range(1, len(e['cpath'].split('.')) + 1))
            leaves.append(Leaf(e['nid'], cid, path_ids, to_turns(full), final_text, opt, outcome_fn(case, opt), e['why']))
        frontier = nxt
    return Tree(cid, policy_sha256, max_agent_turns, branching, tuple(nodes), tuple(leaves), budget)

def tree_counts(tree: Tree) -> dict:
    return {'agent_nodes': len(tree.nodes), 'leaves': len(tree.leaves), 'doctor_turns': 1 + sum(n.next_doctor is not None for n in tree.nodes),
            'finals': len(tree.leaves), 'by_depth': {d: sum(n.depth == d for n in tree.nodes) for d in range(1, tree.max_agent_turns + 1)}}
