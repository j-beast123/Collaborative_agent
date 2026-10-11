# Gate v3.3.1 isolated implementation report

## Scope

This work created and exercised only `/home/kai/audits/compact_gate_contract_v1/gate_v3_3_1_isolated`. It did not call an external API, resume training/evaluation/reruns, run the full 29-candidate Gate, or modify the active bank, cache, scores, checkpoints, v3.2 outputs, or v3.3 outputs.

## Isolated implementation

- `compact_quality_gate_v3_3_1.py`: independent v3.3.1 validator and assembler. It does not import the v3.2/v3.3 assembler.
- `legacy_v33_stage_adapter_v1.py`: read-only, provenance-producing legacy adapter.
- `contracts/public_evidence_contract_v4.schema.json`: separates requested operation, target, observed operations, target coverage, operation performance, literal evidence, and missing component.
- `contracts/applicability_contract_v3_3_1.schema.json`: explicit clause polarity, status, evidence scope, inspected turns, and satisfaction.
- `prompts/blind_source_v3_3_1.txt` and `prompts/applicability_v3_3_1.txt`: new isolated wire contracts.
- `migration_reviews.json`: hash-bound human polarity review used only for the frozen regression fixture. It does not grant KEEP inheritance.
- `replay_offline.py`: read-only replay and C3 request-preview builder.
- `tests/test_contracts.py`: general synthetic fixtures.
- `audit_rerun.py`: read-only rerun wire-field audit.
- `check_no_case_exceptions.py`: runtime case-literal scan.

The final assembler is `gate_v3_3_1_isolated_assembler_v1`, SHA-256 `31b9f64510fd31ab0ba86ba3b6c1ec4decc2e0c7203a427ed933275bc30e8972`.

## Contract behavior

### Behavior contract

The closed operation taxonomy is:

- `mention`
- `comparison`
- `compatibility_evaluation`
- `causal_explanation`
- `exception_analysis`
- `decision_linkage`

The validator does not use string membership to infer semantic completion. It checks only structural consistency and literal evidence:

- target coverage is derived independently from operation performance;
- performance requires a typed observation for the requested target and an exact target-reply quote;
- nonperformance requires a concrete missing component;
- target mismatch and operation mismatch remain distinct;
- an exact quote does not automatically establish semantic support;
- an unscorable source stays `UNSCORABLE`.

### Applicability contract

- `presence` is satisfied only by `supported` with literal prefix evidence.
- `absence` is satisfied only by `absent`, `complete_visible_prefix`, and the exact complete prefix turn list.
- `not_required` is satisfied only by `not_required` and carries no evidence quote.
- `unknown` is not converted to `absent`.
- target or future-turn evidence raises `EXECUTION_ERROR`.
- aggregate applicability is computed locally from clause satisfaction and must agree with the model aggregate.

### State separation

`PASS`, `REJECT`, `UNSCORABLE`, `EXECUTION_ERROR`, `N/A`, and actual score zero are represented separately and covered by tests. No execution error is converted into a rubric decision or score.

## Synthetic validation

`pytest -q tests/test_contracts.py` completed with **14 passed**. The twelve required semantic-structure fixtures all passed, plus typed-operation consistency and state-separation tests. `py_compile` also passed for all new Python files.

These are structural contract tests. They do not establish LLM semantic accuracy.

## Six-case offline replay

All six records carry the same assembler version and hash.

| Candidate | v3.3.1 offline status | Source level | Failure codes |
|---|---:|---:|---|
| C5/0.0 | PASS | 2 | none |
| C4/0.0 | PASS | 2 | none |
| C1/1.0 | REJECT | 1 | `INSTRUCTION_ALIGNMENT`, `SOURCE_ANCHOR_MISMATCH` |
| C5/1.2 | REJECT | 2 | `MC_ANCHOR_ALIGNMENT`, `BOUNDARY_COVERAGE`, `CONCLUSION_NEUTRALITY` |
| C3/0.0 | EXECUTION_ERROR | — | none; the failed blind-source stage remains unresolved |
| C4/0.3 | PASS | 2 | none |

C3 was not converted to a decision. Its new API request is frozen but unexecuted.

## C4/0.3 migration provenance

The legacy clause requiring that an earlier comparison had not yet occurred was explicitly reviewed as an `absence` requirement and bound to clause SHA-256 `89d443f76db7cccbc6424d49e68c33e3829b1e0ae44cc623837db93dbdc1509c`.

The adapter additionally verified:

- saved parser validation preserves the stored answer;
- `status=absent` has no positive evidence;
- inspected turn IDs equal the complete visible prefix;
- no target/future turn was used;
- aggregate applicability is true;
- the migration review does not inherit KEEP.

The transformation provenance is stored in `results/candidates/C4_0.3/legacy_conversion_provenance.json`. Without the explicit hash-bound review, the adapter returns migration review required rather than guessing polarity from prose.

## C3/0.0 API preview

The frozen request is `api_preview/C3_0.0_blind_source_request.json`.

- model/provider: `gpt-6-luna`, OpenAI direct
- request payload SHA-256: `ce4fadd65f63c53cc3c130b30bb2cf25912b8aacc2ccff22f51328286491c55e`
- max output tokens: 4096
- attempts: one initial call; at most one additional technical retry
- retryable: transport, provider-schema, or local-schema failure only
- semantic disagreement is not retryable
- previous first attempt usage: 5,094 prompt tokens and 2,497 completion tokens
- expected new prompt size: approximately 5,100 tokens; maximum completion envelope: 4,096 tokens
- monetary estimate: unavailable because the frozen direct-API records mark this model unpriced and contain no pinned price table

The preview's JSON Schema validates offline. Its user payload contains no `ground_truth`, `gold_answer`, `source_outcome`, `expected_level`, or future dialogue. No API request was sent.

## Rerun wire audit

The existing `intervention_text` field represents the generated Agent reply. The new audit report preserves that legacy fact and separates:

- `intervention_instruction`
- `agent_request`
- `agent_reply`
- `legacy_agent_reply_field: intervention_text`
- `doctor_request`
- `doctor_final`

C5/0.0 remains:

- normal completion: 3/3
- success to failure: 0/3
- KEEP: false
- adherence: diagnostic only
- causal ineffectiveness: not established

`The correct answer is C` appears in a generated Agent reply, not in the intervention instruction. The Agent request contains visible question/options/prefix but no hidden GT, outcome, or future dialogue. All three requests use the exact frozen source prefix.

Policy and Doctor identity:

- Agent policy SHA-256: `9018ce3bbacd319049576e8ace88afb4777621a8769919bf46676fca8a85b17c`
- LoRA adapter SHA-256: `0a08536a3879c3df375e1745e4f6bd166c8e040c16af5bcd7d97528c73e7ce06`
- Doctor: `gemma3:12b-it-q8_0`
- Doctor digest: `997a7c2c09753e9011ec18a3fcb498e27c0ba330599dfb26a2e05b58cfbdc1bc`
- protocol: one fresh Agent reply followed immediately by one Doctor FINAL

The common Agent role gives the selected behavior instruction precedence, so no direct prompt conflict was found. All three generated replies nevertheless reintroduced at least part of the omitted mechanism or comparison. This remains a diagnostic adherence finding and does not change KEEP.

## No case-specific runtime logic

The runtime modules, prompts, and schemas were searched for the regression IDs, case UUID form, disease names, and clinical phrases. No matches were found. Case IDs occur only in replay/test selection data.

## Original and active files

The following original hashes were identical before and after replay:

- v3.2 all-29 manifest: `87399d78d34b6453b7ee845c6505bd629dbf1f7a3311e1818a947a5b2be2cde9`
- v3.3 runner: `d9b24ef02624ee16b948aefdd4d88a78bc8d65a66f8902972ef8823daab5acd5`
- v3.3 results manifest: `4aa5f461b2fc3ffbfc985a119b22dd3324f2cedd7dcc1d2f4483085b97b3327d`
- v3.3 offline regression: `2a73d51074cfb7d04960846054b7efef167c00a689424abb8f1e0a7f8b341d7c`

No write was made outside the isolated v3.3.1 directory. The active worktree was already dirty; its existing unrelated changes were not modified or cleaned.

## Remaining work and promotion status

Not yet verified:

- an actual v4 blind-source response for C3/0.0;
- semantic correctness of v4 model judgments;
- six fully completed decisions, because C3 remains an execution error;
- prespecified unseen-case holdout behavior;
- full 29-candidate replay;
- active integration and cache behavior.

The full 29-candidate run is **not ready** because C3 is incomplete and no unseen holdout has been run. Active promotion is also **not ready**. The next authorized step would be the single frozen C3 blind-source API call, followed by one-assembler replay of all six and then a separately frozen holdout plan.
