"""Scheduling strata concurrently preserves bank, member-guide, and cache identities."""
import json
import threading
from collections import Counter

from src.online_rl.tree import posthoc, rubric_bank
from src.online_rl.tree.cache import CallCache
from src.online_rl.tree.records import DoctorFeedback, Leaf


CASE = {'case_id': 'parallel-case', 'scenario': '', 'question': 'Which option?',
        'options': {'A': 'First', 'B': 'Second'}, 'ground_truth': {'correct_option': 'A'}}
PROMPT = 'Generate one criterion and instruction.\n[Assign or create]\nReuse the same criterion.'


def sources():
    leaves, tins = [], {}
    # Deliberately not sorted: concurrency must not replace deterministic stratum/leaf order.
    for lid, outcome in [('S1', 1.0), ('F1', 0.0), ('S0', 1.0), ('F0', 0.0)]:
        dialogue = (('doctor', 'D1', 1, 'Explain the criterion.'),
                    ('agent', 'A1', 1, f'Initial reply {lid}.'),
                    ('doctor', 'D2', 2, f'Clarify the missing relation for {lid}.'),
                    ('agent', 'A2', 2, f'Second reply {lid}.'),
                    ('doctor', 'FINAL', 0, 'A' if outcome else 'B'))
        leaf = Leaf(lid, CASE['case_id'], (f'{lid}/first', f'{lid}/second'), dialogue,
                    dialogue[-1][-1], dialogue[-1][-1], outcome, 'max_turns')
        leaves.append(leaf)
        tins[lid] = posthoc.freeze_trajectory_input(CASE, leaf,
            DoctorFeedback(lid, outcome, f'Observed behavior {lid}.', 'feedback', target_reply_id=1), True)
    return leaves, tins


class Fakes:
    def __init__(self, overlap=False):
        self.barrier = threading.Barrier(2) if overlap else None
        self.lock = threading.Lock()
        self.calls = []
        self.generator_threads = set()

    def record(self, kind, leaf):
        with self.lock:
            self.calls.append((kind, leaf))

    def generate(self, prompt, tin):
        self.record('generate', tin.leaf_id)
        if self.barrier is not None:
            with self.lock:
                self.generator_threads.add(threading.get_ident())
            # A sequential implementation fails here; no wall-clock speed assertion is needed.
            self.barrier.wait(timeout=5)
        rubric = {
            'name': 'Explain the requested relation',
            'applicability_condition': 'The Doctor asks for the criterion.',
            'required_behavior': 'Explain the relation.',
            'scores': {'0': 'No relation', '1': 'Partial relation', '2': 'Clear relation'},
            'source_provenance': {'analysis': {'selected_agent_turn_id': 1},
                                  'selected_turn_policy': 'generator_v1',
                                  'source_trajectory_id': tin.leaf_id,
                                  'source_case_id': tin.case_id, 'source_outcome': tin.outcome}}
        return {'rubric': rubric, 'raw': json.dumps(rubric),
                'intervention_instruction': f'Handle the initial criterion for {tin.leaf_id}.'}

    def match(self, package, user):
        lid = json.loads(user.split('Source trajectory ID (JSON value; preserve type): ', 1)[1].splitlines()[0])
        bank = json.loads(user.rsplit('canonical rubrics):\n', 1)[1])
        self.record('match', lid)
        return {'decision': 'REUSE_EXISTING', 'rubric_id': bank[0]['rubric_id'],
                'checks': {key: True for key in rubric_bank.CHECKS}, 'rationale': 'Same relation.',
                'selected_agent_turn_id': 2,
                'intervention_instruction': f'Handle the follow-up relation for {lid}.'}

    def gate(self, tin, rubric):
        self.record('gate', tin.leaf_id)
        return {'passed': True, 'reasons': []}


def build(fake, cache, parallel):
    leaves, tins = sources()
    return rubric_bank.build_banks(CASE, leaves, tins, PROMPT, 'prompt',
        fake.generate, fake.match, fake.gate, cache, parallel_strata=parallel)


def test_parallel_strata_overlap_and_preserve_joint_assignments_and_durable_cache(tmp_path):
    serial_fake, parallel_fake = Fakes(), Fakes(overlap=True)
    serial_cache = CallCache(tmp_path / 'serial.jsonl')
    parallel_cache = CallCache(tmp_path / 'parallel.jsonl')
    serial = build(serial_fake, serial_cache, False)
    parallel = build(parallel_fake, parallel_cache, True)

    assert len(parallel_fake.generator_threads) == 2
    assert serial == parallel
    assert list(serial[0]) == list(parallel[0])
    assert [a.leaf_id for a in parallel[1]] == ['F0', 'F1', 'S0', 'S1']
    assert Counter(serial_fake.calls) == Counter(parallel_fake.calls)
    assert Counter(kind for kind, _ in parallel_fake.calls) == {'generate': 2, 'gate': 2, 'match': 2}

    by_leaf = {a.leaf_id: a for a in parallel[1]}
    for prefix, level in [('F', '2'), ('S', '0')]:
        source, reused = by_leaf[prefix + '0'], by_leaf[prefix + '1']
        assert source.canonical_rubric_id == reused.canonical_rubric_id
        assert source.intervention_spec['selected_agent_turn_id'] == 1
        assert reused.intervention_spec['selected_agent_turn_id'] == 2
        assert reused.intervention_spec['requested_level'] == level
        assert reused.intervention_spec['intervention_instruction'] == f'Handle the follow-up relation for {prefix}1.'
    assert by_leaf['F0'].canonical_rubric_id != by_leaf['S0'].canonical_rubric_id

    # JSONL order may differ; full keyed records (keys, identities and values) must not.
    assert serial_cache.store == parallel_cache.store
    for path, mode in [(serial_cache.path, True), (parallel_cache.path, False)]:
        resumed_fake = Fakes()
        resumed_cache = CallCache(path)
        assert build(resumed_fake, resumed_cache, mode) == serial
        assert resumed_fake.calls == []
        assert resumed_cache.store == serial_cache.store
