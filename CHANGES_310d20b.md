# Changes in 310d20b

Commit: `310d20b Harden value dataset splits and pair evaluation`

## Changed Files

- `src/effectiveness/dataset.py`
  - Case split selection now uses `case_id`, not `state_id`.
  - Verifies that each written split preserves its expected case set.
  - Added `write_split_rows()` as the reloadable raw `{state, target}` format.

- `src/effectiveness/evaluate.py`
  - Validates prediction/example lengths independently.
  - Rejects duplicate pair IDs.
  - Requires `before_agent` and `after_agent` state types.
  - Keeps before/after effectiveness evaluation pair-aware.

- `src/effectiveness/model.py`
  - Restored `use_lora=True` when loading a LoRA checkpoint.

- `src/effectiveness/train.py`
  - Uses explicit `--train-data`, `--dev-data`, and `--test-data` files.
  - Rejects overlapping case sets.
  - Evaluates dev and test data in batches.
  - Uses `--training-mode lora|head-only`.

## Verification

The split test produced:

```text
train/dev/test: 8 / 1 / 1 cases
overlap: 0 / 0 / 0
```

## Inspect the Exact Diff

```bash
git show 310d20b
git show --stat 310d20b
```
