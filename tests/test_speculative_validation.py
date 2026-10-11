"""Speculative validation only pre-fills the exact-identity call cache: the unchanged posthoc.generate_rubric then returns
exactly the record it returns without speculation, including when the source check fails (the Gate answer is never read)."""
from types import SimpleNamespace as NS
import pytest
from src.online_rl.integrated.records import sha256_json, to_dict
from src.online_rl.tree import offline_chunk_bank as O
from src.online_rl.tree import posthoc


class Cache:
    def __init__(self): self.store = {}; self.calls = []
    def call(self, kind, identity, fn):
        key = sha256_json([kind, identity])
        if key not in self.store: self.calls.append(kind); self.store[key] = fn()
        return self.store[key]


RUBRIC = {'name': 'n', 'applicability_condition': 'c', 'required_behavior': 'b', 'scores': {0: 'x', 1: 'y', 2: 'z'},
          'source_provenance': {'selected_turn_policy': 'generator_v1'}}
TIN = NS(sha256='t' * 64, leaf_id='c/0', target_reply_id=1)


def build(applicable, speculative_first, monkeypatch):
    monkeypatch.setattr(posthoc, 'rubric_problems', lambda rubric: [])
    cache = Cache(); log = []
    gen = lambda prompt, tin: {'raw': 'raw', 'rubric': dict(RUBRIC), 'intervention_instruction': 'do'}
    gen.model = 'g'; gen.settings = {'s': 1}
    def gate(tin, rubric): log.append('gate'); return {'passed': True, 'reasons': []}
    gate.model = 'luna'; gate.protocol = 'p'
    def source(rubric): log.append('source'); return {'applicable': applicable, 'target_reply_id': 1, 'node_id': 'n'}
    if speculative_first:
        rid = 'p' * 12 + '-' + 't' * 12
        rubric = {**RUBRIC, 'rubric_id': rid, 'scores': {str(a): b for a, b in RUBRIC['scores'].items()}}
        identity = {'tin': TIN.sha256, 'rubric': sha256_json(rubric), 'model': 'luna', 'protocol': 'p'}
        O.speculate([lambda: source(rubric), lambda: cache.call('posthoc_gate', identity, lambda: gate(TIN, rubric))], 3)
    rub = posthoc.generate_rubric('prompt', 'p' * 64, TIN, gen, cache, gate, source_stage1=source)
    return to_dict(rub), log, cache


@pytest.mark.parametrize('applicable', [True, False, None])
def test_speculation_reproduces_the_sequential_record(applicable, monkeypatch):
    base, base_log, _ = build(applicable, False, monkeypatch)
    spec, spec_log, cache = build(applicable, True, monkeypatch)
    assert spec == base                                        # same validity, problems, gate, source check
    if applicable is True:
        assert spec['valid'] and spec['gate']['passed'] and spec_log.count('gate') == 1      # the speculative Gate answer is the one consumed
    else:
        assert not spec['valid'] and spec['gate'] is None       # the Gate answer exists in the cache but was never read
        assert base_log.count('gate') == 0 and spec_log.count('gate') == 1                   # the only difference: one extra request


def test_a_failing_speculative_request_is_ignored():
    def boom(): raise RuntimeError('provider error')
    seen = []
    O.speculate([boom, None, lambda: seen.append(1)], 3)
    assert seen == [1]
