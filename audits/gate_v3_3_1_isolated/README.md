# Gate v3.3.1 isolated audit

This directory contains the isolated Gate v3.3.1 implementation, frozen offline results, regression fixtures, C3 blind-source request preview, and rerun audit.

It is intentionally not connected to the active trainer, bank, cache, scores, or checkpoints. The replay utilities point to the original read-only audit records under `/home/kai/audits/compact_gate_contract_v1`; the committed results make review possible without rerunning those inputs.

Run the self-contained contract tests from this directory:

```bash
pytest -q tests/test_contracts.py
```

See [reports/implementation_report.md](reports/implementation_report.md) for scope, results, hashes, remaining limitations, and promotion status. The C3 request in `api_preview/` is prepared but was not sent.
