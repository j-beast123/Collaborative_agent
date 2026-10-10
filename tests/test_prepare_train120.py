"""Offline reconstruction must preserve the selected records, their order and byte hashes."""
import copy
import hashlib
import json
from pathlib import Path

import pytest

from scripts.online_rl import prepare_train120 as P


def sha(payload):
    return hashlib.sha256(payload).hexdigest()


@pytest.fixture
def inputs(tmp_path):
    rows = [{'case_id': cid, 'question': f'Synthetic question {cid} café',
             'options': {'A': 'a', 'B': 'b'}, 'ground_truth': {'correct_option': 'A'}}
            for cid in ('a', 'b', 'c')]
    source_bytes = ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows).encode()
    selected = [rows[2], rows[0]]
    dataset = ''.join(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n'
                      for row in selected).encode()
    order = (json.dumps(['c', 'a'], indent=2) + '\n').encode()
    manifest = {'schema': 'train_subset_manifest_v1', 'count': 2, 'case_ids': ['c', 'a'],
                'source_train_sha256': sha(source_bytes), 'train_jsonl_sha256': sha(dataset),
                'order_sha256': sha(order), 'train_jsonl_file': 'data/interim/train120.jsonl',
                'order_file': 'data/interim/order.json',
                'question_sha256': {row['case_id']: P.question_sha256(row['question']) for row in selected}}
    source = tmp_path / 'source.jsonl'
    source.write_bytes(source_bytes)
    manifest_path = tmp_path / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest))
    return source, manifest_path, tmp_path / 'out', manifest, rows, dataset, order


def test_restores_order_and_exact_bytes_from_larger_source_and_existing_subset(inputs):
    source, manifest_path, out, manifest, rows, dataset, order = inputs
    result = P.prepare(source, out, manifest_path)
    assert result['count'] == 2
    assert (out / 'train120.jsonl').read_bytes() == dataset
    assert (out / 'order.json').read_bytes() == order
    subset_out = out.parent / 'subset_out'
    P.prepare(out / 'train120.jsonl', subset_out, manifest_path)
    assert (subset_out / 'train120.jsonl').read_bytes() == dataset
    # Identical existing files are safe to reuse, including the source itself.
    assert P.prepare(out / 'train120.jsonl', out, manifest_path) == result


@pytest.mark.parametrize('problem,match', [
    ('missing', 'missing 1 pinned'), ('duplicate', 'Duplicate source'),
    ('changed_content', 'Source hash differs'), ('changed_order', 'Reconstructed hash differs'),
    ('question_hash', 'Question hash differs'), ('dataset_hash', 'Reconstructed hash differs'),
])
def test_rejects_invalid_inputs_before_writing(inputs, problem, match):
    source, manifest_path, out, manifest, rows, dataset, order = inputs
    if problem == 'missing':
        rows.pop()
    elif problem == 'duplicate':
        rows.append(copy.deepcopy(rows[0]))
    elif problem == 'changed_content':
        rows[0]['ground_truth']['correct_option'] = 'B'
    elif problem == 'changed_order':
        manifest['case_ids'].reverse()
    elif problem == 'question_hash':
        manifest['question_sha256']['a'] = '0' * 64
    elif problem == 'dataset_hash':
        manifest['train_jsonl_sha256'] = '0' * 64
    source.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows), encoding='utf-8')
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=match):
        P.prepare(source, out, manifest_path)
    assert not out.exists()


def test_does_not_partially_replace_existing_output(inputs):
    source, manifest_path, out, manifest, rows, dataset, order = inputs
    out.mkdir()
    (out / 'order.json').write_bytes(b'local output\n')
    with pytest.raises(ValueError, match='Refusing to replace'):
        P.prepare(source, out, manifest_path)
    assert (out / 'order.json').read_bytes() == b'local output\n'
    assert not (out / 'train120.jsonl').exists()


def test_protected_data_guard_runs_before_output(inputs, monkeypatch):
    source, manifest_path, out, manifest, rows, dataset, order = inputs
    def reject(cases, purpose):
        raise RuntimeError('evaluation-only')
    monkeypatch.setattr(P, 'assert_not_protected', reject)
    with pytest.raises(RuntimeError, match='evaluation-only'):
        P.prepare(source, out, manifest_path)
    assert not out.exists()


def test_published_manifest_has_exact_pinned_order_and_no_question_text():
    manifest = json.loads(P.MANIFEST.read_text())
    order = manifest['case_ids']
    assert len(order) == len(set(order)) == manifest['count'] == 120
    assert set(order) == set(manifest['question_sha256'])
    assert sha((json.dumps(order, indent=2) + '\n').encode()) == manifest['order_sha256']
    assert manifest['train_jsonl_sha256'] == '3d34929094925c822062d03839b49de8f6fb43afba71fb573de409471bb5b8ab'
    assert manifest['order_sha256'] == 'edf23262be5430d10a0d02dde3da80710bf9a91bd941d83234e9c98372752c41'
    for value in manifest['question_sha256'].values():
        assert len(value) == 64 and set(value) <= set('0123456789abcdef')
    assert not {'question', 'options', 'ground_truth'} & set(manifest)
