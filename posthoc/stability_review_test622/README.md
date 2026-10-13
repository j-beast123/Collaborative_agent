# Stability-calibrated stance with a conflict-only source-blind review (CasiMedicos622)

**Status: POST-HOC / EXPLORATORY.** The rule below was designed after CasiMedicos622 results had been inspected, and each
configuration was run once. Treat the numbers as a lead to confirm, not as a pre-registered result.

## Result

All runs: CasiMedicos622 (622 cases, one episode per case), Qwen3-8B Agent with Thinking OFF, at most 4 rounds,
temperature 0.7 / top-p 0.9, Gemma3-12B Doctor, one frozen Doctor-opening bank, same seeds, same host (A6000).

| Policy weights | Inference policy | Correct | Accuracy |
|---|---|---|---|
| theta0 | canonical | 425 | 0.683 |
| Offline (case 120) | canonical | 434 | 0.698 |
| Online (case 120) | canonical | 430 | 0.691 |
| theta0 | stability stance v1 | 437 | 0.703 |
| Offline (case 120) | stability stance v1 | 438 | 0.704 |
| Online (case 120) | stability stance v1 | 435 | 0.699 |
| **Offline (case 120)** | **stability stance v1 + blind review** | **448** | **0.720** |

Paired comparisons of the last row (exact McNemar test, same 622 cases):

| Compared with | Incorrect → correct | Correct → incorrect | p |
|---|---|---|---|
| canonical theta0 | 53 | 30 | 0.015 |
| canonical Offline | 38 | 24 | 0.098 |
| Offline + stability stance v1 | 26 | 16 | 0.164 |

Only the comparison with theta0 is significant at 0.05. The step from stability v1 to the blind review (+10 cases) is not.
A probe run before this evaluation predicted no gain from the review, so the +10 may be run-to-run variation.

## What the inference policy does

The private assessment (prompt, seed, output format) is unchanged. Nothing is trained; only the Agent's visible reply is steered.

1. **Stability.** The same case is answered with five alternative short private prompts (`VIEWS` in the scripts). If all
   five give the same option as the private assessment, the Agent is *stable*.
2. **Stance** (appended to the hidden context of the visible reply):
   - stable → state the option plainly, say why the physician's option does not fit, keep it unless new evidence appears;
   - not stable → present the option as a possibility and support the physician's option when it is given a case-grounded reason.
3. **Conflict-only source-blind review** (`test_eval_review.py`). When the Agent's option and the physician's current option
   differ, the two options and their stated reasons are compared as "Assessment 1 / Assessment 2" without saying who wrote
   which, in both orders, by reading the option logits (no generation). The result is passed to the physician in one
   sentence only when it points the same way as the stance. The physician makes the final choice; nothing is decided
   automatically.

## Files

- `test_eval_canonical.py`: the canonical TEST622 evaluation.
- `test_eval_stability_v1.py`: canonical + steps 1–2.
- `test_eval_review.py`: canonical + steps 1–3 (the 0.720 run).
- `orch.py`: shard orchestrator (8 shards, one GPU each with its own Doctor server).
- `TEST_PROTOCOL_FROZEN.json`, `targets/`, `openings_test622.jsonl`, `TEST_OPENING_BANK_MANIFEST.json`: the frozen TEST
  protocol, the evaluated targets (checkpoint case and policy hash) and the Doctor-opening bank.
- `results/SUMMARY.json`: accuracies and paired tests. `results/*.per_case.jsonl`: per-case outcomes of every row above.
  `results/offline_stability_v1_blind_review_full_rows/`: full dialogues of the 0.720 run.

## Reproduction

The scripts run against this branch (evaluation config `configs/experiments/fixed_t2/main120/e3_private_5_5_t4_off.json`).
The policy weights are not in the repository:

- Offline (case 120): trained on branch `sj-max4-speed-dev` at commit `9bd7cc8`
  (`configs/experiments/train120_max4_offline_frozen_bank_stage_b_final120.json`), policy hash `275839b82037931b…`.
- theta0: LoRA initialisation seed 20261006, policy hash `9018ce3bbacd3190…`.

```
PH_ROOT=<output dir> PH_ORDER=B_offline PH_WAIT=<file whose last line contains "all done"> python orch.py
```

`orch.py` expects `targets/<arm>.json` under `PH_ROOT` with `local_checkpoint` pointing to the adapter directory, and the
opening bank next to the canonical TEST directory; adjust the paths at the top of `orch.py` for a new machine.

## Caveats

- The agent prompt and the precommit protocol are byte-identical in the training code line and in this evaluation code
  line, but training (temperature 1.0) and evaluation (0.7) use different sampling.
- The stability rule, the stance texts and the review were written while looking at these 622 cases.
- One run per configuration; no repeated seeds. Stability v1 gives about +2 points on theta0 and under +1 on the trained
  checkpoints, none significant on its own.
- The review has only been run on the Offline checkpoint.
