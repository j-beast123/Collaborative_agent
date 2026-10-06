"""Case-level branching rollout: Doctor opens; at every Agent decision point each live branch samples `branching`
Agent replies from one identical prompt; each non-terminal reply gets one Doctor response; up to max_agent_turns.
A branch ends on the stop rule or the turn limit, then the Doctor FINAL decides the outcome (no feedback, no rubric)."""
from __future__ import annotations
from typing import Callable, Optional
from src.online_rl.integrated.continuation import derive_seed
from src.online_rl.tree.cache import CallCache
from src.online_rl.tree.records import AgentNode, Leaf, Tree, to_turns


def build_tree(case: dict, *, max_agent_turns: int, branching: int, run_seed: int, policy_sha256: str, protocol_sha256: str,
               agent, doctor, stop_rule: Callable, final_option: Callable, outcome_fn: Callable, cache: CallCache,
               behavior_logprobs: Optional[Callable] = None) -> Tree:
    """agent.sample(case, prefix, turn, seed) -> trace with 'completion_text'; doctor.turn(case, dialogue, round, rounds) and
    doctor.final(case, dialogue) -> text. Ancestor replies depend on the policy, so every key carries policy_sha256."""
    if max_agent_turns < 1 or branching < 1: raise ValueError('max_agent_turns and branching must be >= 1')
    cid = case['case_id']; ident = {'case': cid, 'policy': policy_sha256, 'protocol': protocol_sha256}
    d1 = cache.call('doctor_turn', {**ident, 'path': '', 'round': 1}, lambda: doctor.turn(case, [], 1, max_agent_turns))
    frontier = [('', None, [{'speaker': 'doctor', 'stage': 'D1', 'round': 1, 'text': d1}])]
    nodes, leaves = [], []
    for depth in range(1, max_agent_turns + 1):
        nxt = []
        for path, parent, prefix in frontier:
            for j in range(branching):
                cpath = f'{path}.{j}' if path else str(j); nid = f'{cid}/{cpath}'; seed = derive_seed(run_seed, nid, 'agent', 0)

                def sample(prefix=prefix, seed=seed, depth=depth):
                    tr = agent.sample(case, prefix, depth, seed)
                    if behavior_logprobs is not None: tr = {**tr, 'behavior_logprobs': behavior_logprobs([tr])[0]}   # pi_old, before any update
                    return tr
                trace = cache.call('agent', {**ident, 'path': cpath, 'seed': seed}, sample)
                reply = trace['completion_text']
                dialogue = prefix + [{'speaker': 'agent', 'stage': f'A{depth}', 'round': depth, 'text': reply}]
                why = stop_rule(depth, prefix[-1]['text'], reply) or ('max_agent_turns' if depth == max_agent_turns else None)
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
        frontier = nxt
    return Tree(cid, policy_sha256, max_agent_turns, branching, tuple(nodes), tuple(leaves))


def tree_counts(tree: Tree) -> dict:
    return {'agent_nodes': len(tree.nodes), 'leaves': len(tree.leaves), 'doctor_turns': 1 + sum(n.next_doctor is not None for n in tree.nodes),
            'finals': len(tree.leaves), 'by_depth': {d: sum(n.depth == d for n in tree.nodes) for d in range(1, tree.max_agent_turns + 1)}}
