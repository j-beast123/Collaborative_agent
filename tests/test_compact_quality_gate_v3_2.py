"""Offline v3.2 request-lifecycle regression fixtures, not semantic gold."""
from __future__ import annotations

import copy
import pytest

from src.online_rl.tree.compact_quality_gate_v3_2 import (
    GateExecutionError, PROTOCOL, assemble_v3, validate_v3_bundle,
)
from tests.test_compact_quality_gate_v3 import inputs


def payload(*, request='resolved', active_request_required=False):
    source = inputs(request=request)
    source['applicability']['relation_audit'] = {
        'active_request_required': active_request_required,
        'relation_status': 'same',
    }
    return source


def test_resolved_prior_request_does_not_global_veto_grounded_rubric():
    bundle = assemble_v3(**payload(request='resolved'))
    assert bundle['protocol'] == PROTOCOL
    assert bundle['compact_result']['decision'] == 'pass'
    assert 'REQUEST_NOT_ACTIVE' not in bundle['compact_result']['failure_codes']
    assert bundle['audit_checks']['applicability']['request_state'] == 'resolved'
    assert bundle['audit_checks']['applicability']['active_request_required'] is False
    checks = bundle['audit_checks']
    validate_v3_bundle(bundle, expected_candidate_sha=checks['candidate_content_sha256'],
                       expected_input_sha=checks['input_sha256'],
                       namespace=checks['candidate_namespace'],
                       generator_model=checks['candidate_generator_model'])


def test_only_written_clause_support_can_reject_resolved_request_case():
    p = payload(request='resolved', active_request_required=True)
    app = p['applicability']
    app['status'] = 'not_applicable'
    app['applicable'] = False
    app['conditions']['doctor_condition'][0]['status'] = 'unmet'
    bundle = assemble_v3(**p)
    assert bundle['compact_result']['decision'] == 'reject'
    assert bundle['compact_result']['failure_codes'] == ['APPLICABILITY_UNSUPPORTED']
    assert bundle['audit_checks']['applicability']['active_request_required'] is True


def test_active_required_flag_is_audit_only_no_blanket_veto():
    bundle = assemble_v3(**payload(request='resolved', active_request_required=True))
    assert bundle['compact_result']['decision'] == 'pass'
    assert bundle['audit_checks']['applicability']['raw_request_state'] == 'resolved'


@pytest.mark.parametrize('flag', [None, 'yes', 1])
def test_missing_or_nonboolean_relation_flag_is_execution_error(flag):
    p = payload()
    p['applicability']['relation_audit']['active_request_required'] = flag
    with pytest.raises(GateExecutionError):
        assemble_v3(**p)


def test_v3_1_bundle_is_not_accepted_as_v3_2():
    from src.online_rl.tree.compact_quality_gate_v3 import assemble_v3 as old_assemble
    old = payload()
    old['applicability'].pop('relation_audit')
    bundle = old_assemble(**old)
    checks = bundle['audit_checks']
    with pytest.raises(GateExecutionError):
        validate_v3_bundle(bundle, expected_candidate_sha=checks['candidate_content_sha256'],
                           expected_input_sha=checks['input_sha256'],
                           namespace=checks['candidate_namespace'])


def test_forged_global_request_state_code_is_rejected():
    bundle = assemble_v3(**payload())
    forged = copy.deepcopy(bundle)
    forged['compact_result']['decision'] = 'reject'
    forged['compact_result']['failure_codes'].append('REQUEST_NOT_ACTIVE')
    forged['passed'] = False
    checks = forged['audit_checks']
    with pytest.raises(GateExecutionError):
        validate_v3_bundle(forged, expected_candidate_sha=checks['candidate_content_sha256'],
                           expected_input_sha=checks['input_sha256'],
                           namespace=checks['candidate_namespace'])
