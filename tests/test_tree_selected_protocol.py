"""Regressions for nested rubric scoring and current-policy prompt optimization."""
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import pytest
from src.online_rl.integrated.prompt_search import SearchBackend, _propose
from src.online_rl.integrated.records import sha256_text
from src.online_rl.tree.cache import CallCache
from src.online_rl.tree.decisions import Strategies
from src.online_rl.tree.effectiveness import evaluate_rubric, run_interventions
from src.online_rl.tree.posthoc import freeze_trajectory_input, generate_rubric, doctor_feedback
from src.online_rl.tree.prompt_contract import structural_problems, prompt_violations
from src.online_rl.tree.prompt_search import EvidenceUnit, ScoringDeps, score_prompt, reflection_view, evidence_id
from src.online_rl.tree.records import AgentNode, Leaf, TrajectoryRubric
from src.online_rl.tree.scoring import score_nodes
from src.online_rl.tree.trainer import TreeEnv, TreeTrainer
from src.online_rl.tree.team import team_strategies
from src.online_rl.tree.backends import two_stage_scorer

ROOT = Path(__file__).resolve().parents[1]
PROMPT = (ROOT / 'prompts/sherpa/trajectory_rubric_generator_P0.txt').read_text()
CASE = {'case_id': 'synthetic', 'scenario': '', 'question': 'Choose a result',
        'options': {'A': 'first result', 'B': 'second result'}, 'ground_truth': {'correct_option': 'A'}}


def payload(selected='synthetic/0'):
    app = {'user_side_context': 'The physician requests support', 'agent_side_state': 'No extra condition'}
    return {'analysis': {'selected_agent_turn_id': selected}, 'candidate_rubric': {
        'name': 'explain support', 'applicability': app,
        'scores': {k: {**app, 'behavior': f'provide support level {k}'} for k in '012'}}}


def source():
    prefix = (('doctor', 'D1', 1, 'Please explain'),)
    n = AgentNode('synthetic/0', None, 'synthetic', 1, prefix, 'Original explanation', {}, 'old', 0, 'limit', True, None)
    l = Leaf(n.node_id, 'synthetic', (n.node_id,), prefix + (
        ('agent', 'A1', 1, n.response), ('doctor', 'FINAL', 0, 'Final answer: A')), 'Final answer: A', 'A', 1.0, 'limit')
    tin = freeze_trajectory_input(CASE, l, doctor_feedback(CASE, l, lambda *a: 'Useful explanation', 'fb', CallCache(None)))
    r = generate_rubric(PROMPT, sha256_text(PROMPT), tin, lambda *a: {'rubric': payload(), 'raw': '{}'}, CallCache(None))
    assert r.valid
    return n, l, tin, r


def score(view, prefix, rubrics, reply):
    assert 'ground_truth' not in view
    assert all('applicability_condition' in r and 'analysis' not in r for r in rubrics)
    return {r['rubric_id']: {'applicable': True, 'probabilities': {'0': 0, '1': 0, '2': 1}, 'hard': 2} for r in rubrics}


def test_nested_scoring_and_real_judge_boundary(monkeypatch):
    import src.judges.rubric_scoring as J
    n, _, _, r = source()
    seen = []
    def judge(client, **kwargs):
        visible = kwargs['rubrics'][0]; seen.append(visible)
        assert 'analysis' not in visible and 'source_outcome' not in visible
        assert visible['scores']['2'] == 'provide support level 2'
        return {'applicability_labels': {r.rubric_id: 'Y'}, 'behavior_probabilities': {r.rubric_id: {'0': 0, '1': 0, '2': 1}},
                'behavior_scores': {r.rubric_id: 2}}
    monkeypatch.setattr(J, 'two_stage_jev_score', judge)
    result = score_nodes(CASE, [n], [r], two_stage_scorer(None), CallCache(None))
    assert result[0].score == 1 and seen


def test_source_policy_guard_remains_strict():
    n, l, _, r = source()
    with pytest.raises(ValueError, match='source generation policy'):
        run_interventions(CASE, l, n, r, reruns=3, run_seed=0, policy_sha256='current',
                          writer=None, continuation=None, cache=CallCache(None))


@pytest.mark.parametrize('matched', [True, False])
def test_current_policy_prompt_scoring_and_feedback(matched):
    n, l, tin, r = source(); calls = []
    def writer(case, prefix, turn, rub, level, seed):
        assert prefix[-1]['text'] == 'Please explain'; calls.append(seed); return f'Fresh reply {seed}'
    def scorer(*args):
        return score(*args) if matched else {r.rubric_id: {'applicable': False}}
    cache = CallCache(None)
    deps = ScoringDeps(3, 0, lambda *a: {'rubric': payload(), 'raw': '{}'}, None, scorer, writer,
                       lambda *a: {'final_option': 'B', 'outcome': 0.0}, 'current', team_strategies(None, 1, 1), cache)
    units = [EvidenceUnit('synthetic', l, (n,), tin, 'old')]
    result = score_prompt(PROMPT, sha256_text(PROMPT), units, {'synthetic': CASE}, deps)
    rows = reflection_view(units, result)
    assert rows[0]['rubric']['name'] == 'explain support'
    assert rows[0]['rubric']['scores']['2']['behavior'] == 'provide support level 2'
    assert rows[0]['selected_agent_turn_id'] == n.node_id
    if matched:
        assert len(set(calls)) == 3 and rows[0]['R_O'] == 1
        assert all(run.continuation_policy_sha256 == 'current' for run in result[evidence_id(units[0])]['effectiveness'].runs)
    else:
        assert not calls and rows[0]['verification'] == 'no_match'
        assert rows[0]['interventions'][0]['agent_turn'] is None
        assert rows[0]['interventions'][0]['per_node_R_O'] is None
    assert n.policy_sha256 == 'old'


def test_nested_prompt_guard_and_refiner_retries():
    assert structural_problems(PROMPT) == []
    assert prompt_violations(PROMPT, [PROMPT], [CASE]) == []
    legacy = 'Return {"name":"x","applicability_condition":"x","required_behavior":"x","scores":{}}'
    attempts = []
    def refine(*args):
        attempts.append(args[-1]); return legacy if len(attempts) == 1 else PROMPT
    backend = SearchBackend(lambda *a: {'diagnosis': 'd', 'directions': ['preserve evidence']}, refine, None, None, None)
    res = _propose(PROMPT, [], 1, [], backend, [PROMPT], [CASE], prompt_guard=prompt_violations)
    assert res['proposals'][0]['text'] == PROMPT and len(attempts) == 2
    assert 'nested JSON' in attempts[1]
    assert structural_problems(PROMPT.replace('"selected_agent_turn_id":', '"removed_turn":'))


@pytest.mark.parametrize('count,valid', [(2, False), (3, True), (4, False)])
def test_smoke_requires_three_reruns(count, valid):
    spec = importlib.util.spec_from_file_location('selected_run_tree', ROOT / 'scripts/online_rl/run_tree.py')
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    cfg = {'smoke': {}, 'schedule': {'cases': 2}, 'rollout': {'branching': 2, 'rounds': 2},
           'intervention': {'reruns_per_intervention': count},
           'prompt_update': {'candidate_count': 1, 'retain_top_k': 1, 'refinement_depth': 1}, 'output_dir': mod.SMOKE_DIR}
    assert (mod.smoke_problems(cfg) == []) == valid


def test_policy_update_then_prompt_event_end_to_end(tmp_path):
    state = {'v': 0}; writes = []
    class Agent:
        def sample(self, case, prefix, turn, seed):
            self.count = getattr(self, 'count', 0) + 1
            return {'completion_text': f'reply {self.count}', 'prompt_text': 'same', 'prompt_input_ids': [1], 'completion_ids': [2]}
    class Doctor:
        def turn(self, *a): return 'Please explain'
        def final(self, case, dialogue):
            index = int(dialogue[-1]['text'].split()[-1])
            return 'Final answer: ' + ('A' if index % 2 else 'B')
    def generate(prompt, tin): return {'raw': '{}', 'rubric': payload(tin.agent_node_ids[0])}
    def writer(*a): writes.append(state['v']); return 'Fresh response'
    st = Strategies(r_a_aggregator=lambda *a: 1, credit_assignment=lambda *a: {},
                    agent_update=lambda *a: state.update(v=state['v'] + 1) or {'optimizer_step': True},
                    prompt_unit_value=lambda r: r['effectiveness'].R_O if r['effectiveness'] and r['effectiveness'].R_O is not None else 0)
    cfg = {'schedule': {'cases': 2, 'seed': 0}, 'rollout': {'rounds': 1, 'branching': 4},
           'rubric': {'gold_answer_visible': False, 'prompt_sha256': sha256_text(PROMPT), 'prompt_id': 'P0'},
           'intervention': {'reruns_per_intervention': 3}, 'selection': {'mode': 'threshold', 'min_R_O': 2 / 3},
           'prompt_update': {'enabled': True, 'every_n_cases': 1, 'minimum_trajectories': 1, 'candidate_count': 1,
                             'retain_top_k': 1, 'refinement_depth': 1, 'acceptance_fraction': 0.5},
           'acceptance': {'minimum_trajectories': 1, 'delta': 0, 'p_min': 0.5, 'bootstrap_n': 20, 'bootstrap_seed': 0}}
    class Updater:
        def save(self, path): path.mkdir(parents=True); (path / 'v').write_text(str(state['v']))
    backend = SearchBackend(lambda *a: {'diagnosis': 'support', 'directions': ['clarify evidence']},
                            lambda parent, *a: parent + '\nPreserve grounded contributions.', None, None, None)
    env = TreeEnv(Agent(), Doctor(), lambda *a: None, lambda text, *a: text[-1], lambda case, option: float(option == 'A'),
                  lambda *a: 'Helpful support', generate, score, writer, lambda *a: {'final_option': 'B', 'outcome': 0})
    trainer = TreeTrainer(cfg, out_dir=tmp_path, cases=[CASE], env=env, strategies=st, search_backend=backend,
                          initial_prompt=PROMPT, policy_sha=lambda: f'p{state["v"]}', protocol_sha='protocol',
                          feedback_prompt_sha256='feedback', updater=Updater())
    trainer.run(max_cases=1)
    assert state['v'] == 1 and trainer.state.lineage_cursor == 1
    assert 0 in writes and 1 in writes
    event = json.loads((tmp_path / 'prompt_events/event_0001.json').read_text())
    assert event['rerun_policy_sha256'] == 'p1' and event['evidence_policy_sha256'] == ['p0']


def test_generator_abstains_without_gate_or_intervention():
    from src.online_rl.tree.posthoc import trajectory_message
    _, _, tin, _ = source()
    assert 'Judge Agent behavior independently' in trajectory_message(tin)
    def forbidden(*args):
        raise AssertionError('No gate should run for generator abstention')
    r = generate_rubric(PROMPT, sha256_text(PROMPT), tin,
                        lambda *args: {'rubric': {'eligible': False, 'reason': 'No observed useful Agent behavior', 'candidate_rubric': None}, 'raw': '{}'},
                        CallCache(None), gate_fn=forbidden)
    assert not r.valid and r.problems == () and r.selected_agent_turn_id is None
    assert r.gate['status'] == 'ineligible'
    assert r.rubric['reason'] == 'No observed useful Agent behavior'
