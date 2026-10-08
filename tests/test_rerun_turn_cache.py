"""Per-turn cache inside intervention continuation reruns (effectiveness.rerun_turn_cache + harness Env.continue_from):
a rerun that failed at Doctor turn N resumes at turn N with the earlier turns reused exactly; nothing failed is stored;
any change of model, prompt, input, rerun or policy misses; the outer 'rerun' record is unchanged. Scripted Doctor only."""
import hashlib, json
from argparse import Namespace
from pathlib import Path
import pytest
from test_tree_v1 import CASES, Log, tree_for
from src.common.retry import RetryExhausted
from src.online_rl.tree import effectiveness as EF
from src.online_rl.tree.backends import PolicyInterventions
from src.online_rl.tree.cache import CallCache
from src.online_rl.integrated.records import to_dict

ROOT = Path(__file__).resolve().parents[1]
RUB = type('Rub', (), {'rubric_id': 'rub1', 'rubric': {'name': 'n', 'applicability_condition': 'c', 'required_behavior': 'b', 'scores': {}}})()
h = lambda s: hashlib.sha256(s.encode()).hexdigest()


class Doctor:
    """Scripted Doctor: reply depends only on the exact input; fail_at = the n-th call (1-based) raises an exhausted 429."""
    def __init__(self, fail_at=None): self.fail_at, self.inputs, self.models, self.systems = fail_at, [], [], []

    def __call__(self, key, model, system, user, max_tokens, temperature=0.4):
        if self.fail_at is not None and len(self.inputs) + 1 == self.fail_at:
            self.fail_at = None; raise RetryExhausted('OpenRouter google/gemma-3-12b-it: FAILED (test)')
        self.inputs.append(user); self.models.append(model); self.systems.append(system)
        if 'final decision' in user.lower(): return f'Final answer: option {"AB"[int(h(user), 16) % 2]}'
        return f'Which finding matters most here? [{h(user)[:6]}]'


class Policy:
    """The SharedPolicy surface of the continuation (harness Env.continue_from), with a local deterministic Agent."""
    def __init__(self, doctor_model='google/gemma-3-12b-it', doctor_system=None):
        import torch
        self.torch, self.key = torch, 'k'; self.agent_calls = []
        self.args = Namespace(rounds=5, doctor_model=doctor_model, max_tokens_doctor=300, max_completion_length=400, generation_temperature=0.7,
                              top_p=0.9, top_k=20)
        self.doctor_system = doctor_system or (ROOT / 'prompts/user_simulator.txt').read_text(); self.agent_system = (ROOT / 'prompts/agent.txt').read_text()

    def agent_user_prompt(self, case, prefix, t):
        from src.prompt_opt import harness as H
        return H.Env.agent_user_prompt(self, case, prefix, t)

    def agent(self, user, seed):
        self.agent_calls.append((seed, h(user))); return f'I would weigh the history first ({seed % 97}).'

    def continue_from(self, case, prefix, t, fixed, seed, turn_cache=None):
        from src.prompt_opt import harness as H
        return H.Env.continue_from(self, case, prefix, t, fixed, seed, turn_cache=turn_cache)


def setup():
    tree, _ = tree_for(Log(), rounds=2, branching=2); nodes = {n.node_id: n for n in tree.nodes}
    leaf = next(l for l in tree.leaves if l.outcome in (0.0, 1.0)); return leaf, nodes[leaf.path[0]]


def interventions(monkeypatch, cache, doctor, policy=None, policy_sha='p0', reruns=2):
    from src.prompt_opt import harness as H
    monkeypatch.setattr(H, 'doctor_call', doctor)
    policy = policy or Policy(); leaf, n0 = setup(); iv = PolicyInterventions(policy, '{rubric_json}', '{rubric_json}')
    writer = lambda case, prefix, turn, rubric, level, seed: 'Intervention reply: check the timeline before deciding.'
    return EF.run_interventions(CASES[0], leaf, n0, RUB, reruns=reruns, run_seed=0, policy_sha256=policy_sha, writer=writer,
                                continuation=iv.continue_, cache=cache), policy


def kinds(path): return [json.loads(l)['kind'] for l in Path(path).read_text().splitlines()]


def test_failure_at_doctor_turn_n_resumes_at_turn_n(tmp_path, monkeypatch):
    path = tmp_path / 'calls.jsonl'; first = Doctor(fail_at=3)
    with pytest.raises(RetryExhausted): interventions(monkeypatch, CallCache(path), first)
    assert len(first.inputs) == 2 and kinds(path).count('rerun_doctor_turn') == 2 and 'rerun' not in kinds(path)   # turn 3 failed: not stored
    resumed = Doctor(); runs, pol = interventions(monkeypatch, CallCache(path), resumed)                             # new process: cache from disk
    clean = Doctor(); clean_runs, _ = interventions(monkeypatch, CallCache(tmp_path / 'clean.jsonl'), clean)
    assert clean.inputs[:2] == first.inputs and resumed.inputs == clean.inputs[2:]       # turns 1-2 reused, the rerun continues at turn 3
    assert [to_dict(r) for r in runs] == [to_dict(r) for r in clean_runs]                 # same outputs as an uninterrupted run
    rec = lambda p: {json.loads(l)['key']: json.loads(l)['value'] for l in Path(p).read_text().splitlines()}
    assert rec(path) == rec(tmp_path / 'clean.jsonl')                                     # identical evidence, outer 'rerun' records included


def test_resumed_agent_turns_are_replayed_from_the_cache(tmp_path, monkeypatch):
    path = tmp_path / 'calls.jsonl'
    with pytest.raises(RetryExhausted): interventions(monkeypatch, CallCache(path), Doctor(fail_at=4))
    done = kinds(path).count('rerun_agent_turn'); assert done >= 1
    _, pol = interventions(monkeypatch, CallCache(path), Doctor())
    clean_pol = interventions(monkeypatch, CallCache(None), Doctor())[1]
    assert len(pol.agent_calls) == len(clean_pol.agent_calls) - done                     # cached Agent turns are not regenerated


def test_failed_calls_are_never_cached(tmp_path, monkeypatch):
    path = tmp_path / 'calls.jsonl'; d = Doctor(fail_at=1)
    with pytest.raises(RetryExhausted): interventions(monkeypatch, CallCache(path), d)
    assert kinds(path) == ['behavior_text']                                               # nothing of the failed rerun


def test_complete_rerun_still_uses_the_outer_cache(tmp_path, monkeypatch):
    path = tmp_path / 'calls.jsonl'; runs, _ = interventions(monkeypatch, CallCache(path), Doctor())
    k = kinds(path); assert k.count('rerun') == 2 and k.count('rerun_doctor_turn') > 0 and k.count('rerun_agent_turn') > 0
    again = Doctor(); runs2, pol = interventions(monkeypatch, CallCache(path), again)
    assert again.inputs == [] and pol.agent_calls == [] and [to_dict(r) for r in runs2] == [to_dict(r) for r in runs]
    outer = [json.loads(l) for l in path.read_text().splitlines() if json.loads(l)['kind'] == 'rerun']
    assert set(outer[0]['identity']) == {'rubric', 'node', 'level', 'policy', 'text', 'seed'}           # outer identity unchanged


@pytest.mark.parametrize('change', ['model', 'system', 'policy', 'none'])
def test_changes_invalidate_the_inner_cache(tmp_path, monkeypatch, change):
    path = tmp_path / 'calls.jsonl'; base = Doctor(fail_at=2)
    with pytest.raises(RetryExhausted): interventions(monkeypatch, CallCache(path), base)          # stores Doctor turn 1 only
    kw = {'model': {'policy': Policy(doctor_model='google/gemma-3-27b-it')}, 'system': {'policy': Policy(doctor_system='another Doctor prompt')},
          'policy': {'policy_sha': 'p1'}, 'none': {}}[change]
    d = Doctor(fail_at=2)
    with pytest.raises(RetryExhausted): interventions(monkeypatch, CallCache(path), d, **kw)
    if change == 'none': assert d.inputs[0] != base.inputs[0]                             # stored turn 1 reused; the next turn is requested
    elif change == 'system': assert d.systems[0] == 'another Doctor prompt' and len(d.inputs) == 1        # turn 1 requested again
    else: assert d.inputs[0] == base.inputs[0]                                            # same input, other model / policy: requested again


def test_reruns_do_not_share_turns(tmp_path, monkeypatch):
    path = tmp_path / 'calls.jsonl'; interventions(monkeypatch, CallCache(path), Doctor(), reruns=2)
    recs = [json.loads(l) for l in path.read_text().splitlines() if json.loads(l)['kind'] == 'rerun_doctor_turn']
    owners = {(r['identity']['rerun']['rerun_index'], r['identity']['rerun']['seed']) for r in recs}
    assert len(owners) == 2 and all(r['identity']['rerun']['case_id'] == CASES[0]['case_id'] for r in recs)
    assert all({'model', 'system_sha256', 'stage', 'input_sha256'} <= set(r['identity']['turn']) for r in recs)


def test_a_different_dialogue_prefix_misses_even_for_the_same_rerun(tmp_path, monkeypatch):
    from src.prompt_opt import harness as H
    from src.online_rl.tree.records import to_dialogue
    _, node = setup(); prefix = to_dialogue(node.prefix); cache = CallCache(tmp_path / 'calls.jsonl'); pol = Policy()
    tc = EF.rerun_turn_cache(cache, CASES[0], {'rubric': 'rub1', 'node': node.node_id, 'level': '2', 'policy': 'p0', 'text': 'x', 'seed': 1}, 0)
    d1 = Doctor(); monkeypatch.setattr(H, 'doctor_call', d1); pol.continue_from(CASES[0], prefix, node.depth, 'x', 1, turn_cache=tc)
    d2 = Doctor(); monkeypatch.setattr(H, 'doctor_call', d2); pol.continue_from(CASES[0], prefix, node.depth, 'x', 1, turn_cache=tc)
    assert d1.inputs and d2.inputs == []                                                  # same rerun, same prefix: all reused
    other = [dict(m) for m in prefix]; other[-1]['text'] += ' (edited)'
    d3 = Doctor(); monkeypatch.setattr(H, 'doctor_call', d3); pol.continue_from(CASES[0], other, node.depth, 'x', 1, turn_cache=tc)
    assert len(d3.inputs) == len(d1.inputs) and not set(d3.inputs) & set(d1.inputs)       # different prefix: nothing reused
