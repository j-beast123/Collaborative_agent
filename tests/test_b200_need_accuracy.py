import copy
import json
from types import SimpleNamespace
from dataclasses import replace, asdict
import pytest
from src.online_rl.tree.doctor_need import attach, validate_need, NeedFormatError, criterion_need_problem
from src.online_rl.tree.b200_runner import load_config
from src.online_rl.tree.cache import CallCache
from src.online_rl.tree.rollout import build_tree
from src.online_rl.tree.fixed_accuracy import ID, criterion, scoring_book, always_applicable
from src.online_rl.tree.scoring import score_nodes, ScoringContractError
from src.online_rl.tree.team import effectiveness_weighted_r_a, node_credit
from src.online_rl.tree.records import to_turns
from src.online_rl.integrated.records import sha256_json
from test_b200_partner import CASE, fixture, Doctor


def need(status='present', turn='D1', quote='Which evidence distinguishes the alternatives?'):
    return {'status': status, 'current_uncertainty': 'How to distinguish alternatives',
            'information_needed': 'Evidence supporting or challenging each alternative',
            'public_evidence': [{'turn_id': turn, 'quote': quote}] if status == 'present' else []}


class JointDoctor(Doctor):
    status_protocol = True
    def __init__(self, cfg, replies=None):
        super().__init__()
        self.system = 'ordinary doctor'; self.identity = {}; self.model = 'doctor'
        self.requests = []; self.replies = iter(replies) if replies else None
        self._chat_metadata = SimpleNamespace(last={})
        attach(self, cfg)
    def _chat(self, user, seed, max_tokens=None, system_override=None):
        self.requests.append((user, seed, max_tokens, system_override))
        assert 'ground_truth' not in user and 'private_need' not in user
        assert 'information_needed' not in user
        import re
        turn = int(re.search(r'utterance ID is D(\d+)', user).group(1))
        self._chat_metadata.last = {'input_tokens': 30, 'output_tokens': 90, 'transport_retries': 0}
        if self.replies: return next(self.replies)
        public = 'Which evidence distinguishes the alternatives?'
        status = {'current_option': 'A', 'open_issue': True, 'collaborator_objection': True, 'more_info_could_change_decision': True}
        return json.dumps({'public_reply': public + ('\nSTATUS: ' + json.dumps(status) if turn > 1 else ''),
                           'private_need': need(turn=f'D{turn}')})


def small_tree(doctor, cache):
    class Agent:
        def sample(self, case, prefix, turn, seed):
            assert doctor.requests  # need generation completed before the first Agent.
            assert all(set(d) <= {'speaker', 'stage', 'round', 'text'} for d in prefix)
            assert 'information_needed' not in json.dumps(prefix)
            return {'completion_text': f'Reasoning {seed}.'}
    return build_tree(CASE, max_agent_turns=2, branching=4, run_seed=0, policy_sha256='policy', protocol_sha256='need',
        agent=Agent(), doctor=doctor, stop_rule=lambda *a: None, final_option=lambda x: 'B', outcome_fn=lambda *a: 1,
        cache=cache, workers=2, agent_workers=4)


def test_need_precedes_siblings_and_is_shared_private_and_durable(tmp_path):
    doctor = JointDoctor(load_config()); path = tmp_path / 'calls.jsonl'
    tree = small_tree(doctor, CallCache(path))
    assert len(doctor.requests) == 5  # D1 plus four D2, not one need per candidate.
    roots = [n for n in tree.nodes if n.depth == 1]
    assert len(roots) == 4 and len({n.prefix_doctor_need['need_id'] for n in roots}) == 1
    for leaf in tree.leaves:
        assert len(leaf.private_needs) == 2
        assert 'information_needed' not in json.dumps(leaf.dialogue)
    before = sha256_json(asdict(tree))
    again = small_tree(doctor, CallCache(path))
    assert sha256_json(asdict(again)) == before and len(doctor.requests) == 5
    assert all('STATUS:' not in n.prefix[-1][3] for n in tree.nodes)


@pytest.mark.parametrize('bad', [need(turn='A1'), need(quote='uncited hidden requirement'), {**need(), 'public_evidence': []}])
def test_need_rejects_future_or_fabricated_evidence(bad):
    with pytest.raises(ValueError): validate_need(bad, {'D1': 'Which evidence distinguishes the alternatives?'})


def test_need_format_repair_is_bounded_and_not_silently_empty():
    doctor = JointDoctor(load_config(), ['not json', '{}'])
    with pytest.raises(NeedFormatError): doctor.turn(CASE, [], 1, 4, 3)
    assert len(doctor.requests) == 2


def test_unpublic_need_excludes_partner_criterion():
    _, leaf, rubric, _ = fixture(anchor=2)
    n = {'doctor_turn_id': 'D2', 'before_agent_turn': 2, 'need_id': 'N', **need('insufficient_public_evidence')}
    tin = SimpleNamespace(private_needs=(n,), dialogue=leaf.dialogue)
    assert 'insufficient' in criterion_need_problem(tin, rubric.rubric)
    n.update(need(turn='D2', quote='Consider evidence 2.'))
    assert criterion_need_problem(tin, rubric.rubric) is None
    n['public_evidence'][0]['turn_id'] = 'D3'
    assert 'insufficient' in criterion_need_problem(tin, rubric.rubric)


def test_reflection_links_do_not_rewrite_needs_or_change_attribution():
    from src.online_rl.tree.doctor_reflection import validate_need_links
    needs = [{'need_id': 'n1', 'before_agent_turn': 1, **need()}]
    links = [{'need_id': 'n1', 'agent_reply_id': 1, 'effect': 'helped', 'reason': 'Provided the comparison',
              'public_evidence': [{'turn_id': 'A1', 'quote': 'comparison'}]}]
    dialogue = [{'speaker': 'agent', 'stage': 'A1', 'round': 1, 'text': 'Here is the comparison'}]
    frozen = copy.deepcopy(needs)
    assert validate_need_links(links, needs, dialogue, 1) == links and needs == frozen
    links[0]['need_id'] = 'retrospective-new-need'
    with pytest.raises(ValueError): validate_need_links(links, needs, dialogue, 1)


class Scorer:
    model = 'fake'; protocol = 'public-only'; response_batch_size = 4
    def __init__(self, partner_match=True, bad=False, unscorable=False):
        self.reviews = 0; self.grades = 0; self.partner_match = partner_match; self.bad = bad; self.unscorable = unscorable
    def review_state(self, view, prefix, rubrics):
        self.reviews += 1
        assert 'private_need' not in json.dumps([view, prefix, rubrics])
        return {r['rubric_id']: {'applicable': self.partner_match} for r in rubrics}
    def score_behavior(self, *a): raise AssertionError('use batch')
    def score_behavior_batch(self, view, prefix, rubrics, responses):
        self.grades += 1
        assert 'ground_truth' not in view and 'information_needed' not in json.dumps([view, prefix])
        return {r['response_id']: {rub['rubric_id']: ({'hard': None, 'scoring_status': 'rubric_unscorable'}
            if self.unscorable and i == 0 else {'hard': i % 3}) for rub in rubrics}
            for i, r in enumerate(responses) if not (self.bad and i == 0)}


def siblings():
    nodes, _, _, _ = fixture(anchor=1)
    root = nodes['b200_case/1']
    return [replace(root, node_id=f'n{i}', response=f'Visible behavior {i}') for i in range(4)]


def test_fixed_accuracy_is_once_always_applicable_and_in_mean():
    cfg = load_config(); acc = criterion(cfg)
    partner = replace(acc, rubric_id='partner', rubric={**acc.rubric, 'rubric_id': 'partner'})
    book = scoring_book([partner], cfg)
    assert [r.rubric_id for r in book].count(ID) == 1
    with pytest.raises(ValueError): scoring_book([acc], cfg)
    scorer = always_applicable(Scorer(partner_match=False))
    scores = score_nodes(CASE, siblings(), book, scorer, CallCache(None))
    assert scorer.reviews == 1 and scorer.grades == 1
    for i, node in enumerate(siblings()):
        subset = [s for s in scores if s.response_node_id == node.node_id]
        assert next(s for s in subset if s.rubric_id == 'partner').score is None
        assert effectiveness_weighted_r_a(subset, {'partner': 1, ID: 1}) == (i % 3) / 2


def test_matching_situation_scores_absent_behavior_zero_not_na():
    cfg = load_config(); acc = criterion(cfg)
    partner = replace(acc, rubric_id='partner', rubric={**acc.rubric, 'rubric_id': 'partner'})
    scorer = always_applicable(Scorer())
    scores = score_nodes(CASE, siblings(), scoring_book([partner], cfg), scorer, CallCache(None))
    zero = [s for s in scores if s.response_node_id == 'n0']
    assert all(s.match and s.score == 0 and s.scoring_status == 'scored' for s in zero)


def test_partial_scoring_failure_aborts_and_unscorable_masks_whole_group():
    cfg = load_config()
    with pytest.raises(ScoringContractError):
        score_nodes(CASE, siblings(), scoring_book([], cfg), always_applicable(Scorer(bad=True)), CallCache(None))
    scores = score_nodes(CASE, siblings(), scoring_book([], cfg), always_applicable(Scorer(unscorable=True)), CallCache(None))
    assert all(s.scoring_status == 'group_unscorable' and s.score is None for s in scores)
    assert effectiveness_weighted_r_a(scores, {ID: 1}) is None


def test_accuracy_is_not_in_partner_verification_book():
    from test_b200_partner import Outcomes, verify
    backend = Outcomes([1, 1, 0])
    result = verify(backend)
    assert result.rubric_id != ID and len(backend.calls) == 2
    assert len(scoring_book([], load_config())) == 1


def test_private_assessment_cannot_split_shared_applicability():
    nodes = [replace(n, trace={'pre_response_belief': {'private': i}}) for i, n in enumerate(siblings())]
    scorer = always_applicable(Scorer())
    score_nodes(CASE, nodes, scoring_book([], load_config()), scorer, CallCache(None))
    assert scorer.grades == 1


def test_linear_verification_records_followup_needs_without_exposing_them(tmp_path):
    from test_b200_partner import backend
    from src.online_rl.tree.partner_verification import make_plan
    doctor = JointDoctor(load_config())
    b = backend(doctor)
    nodes, leaf, rubric, assignment = fixture(anchor=2)
    node = nodes[leaf.path[1]]
    plan = make_plan(CASE, node, leaf, rubric, assignment.intervention_spec, 0, 'policy', b.cfg)
    cache = CallCache(tmp_path / 'verification.jsonl')
    result = b.trial(CASE, node, plan, 0, cache)
    assert len(result['doctor_statuses']) == 2
    assert all(d['private_need']['before_agent_turn'] == d['turn'] for d in result['doctor_statuses'])
    assert 'information_needed' not in json.dumps(result['dialogue'])
    assert all('information_needed' not in call[0] for call in b.policy.calls)
    requests = len(doctor.requests)
    restored = b.trial(CASE, node, plan, 0, CallCache(cache.path))
    assert restored['doctor_statuses'] == result['doctor_statuses'] and len(doctor.requests) == requests


def test_trainer_connects_fixed_accuracy_to_rubric_only_targets(tmp_path, monkeypatch):
    from test_witness_trainer import make
    from src.online_rl.tree.team import node_credit
    from dataclasses import replace
    tr, _, _ = make(tmp_path, 'rubric_only')
    tr.cfg['fixed_accuracy'] = load_config()['fixed_accuracy']
    scorer = always_applicable(Scorer(partner_match=False))
    tr.env = replace(tr.env, scorer=scorer)
    tr.st = replace(tr.st, credit_assignment=node_credit(0, 1))
    def verify(case, tree, feedback, ruled, canon, assignments):
        assert ID not in canon
        return {}  # no verified partner criteria: fixed accuracy must still train.
    monkeypatch.setattr(tr, '_verify_bounded', verify)
    record = tr.step(apply_update=False, commit=False)
    assert len(record['fixed_criteria']) == 1
    assert record['rubric_weights'] == {ID: 1.0}
    assert all(s['rubric_id'] == ID for s in record['response_scores'])
    assert all(r['reward'] == r['R_A'] for r in record['agent_targets']['nodes'].values())
    assert any(r['reward'] is not None for r in record['agent_targets']['nodes'].values())
