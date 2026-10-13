# Evidence sheet for the reported checkpoint appendix

This file separates verified implementation facts from paper wording and from
values that still require an external baseline artifact.

## Verified result

| Item | Value | Evidence |
|---|---:|---|
| Evaluation cases | 622 | `artifacts/reported_checkpoint/data/all_en_622.jsonl` |
| Correct | 442 | `artifacts/reported_checkpoint/saved_evaluation/summary.json` |
| Incorrect | 180 | same |
| Accuracy | 71.0609% | recomputed as 442/622 |
| Parsed FINAL | 622/622 | per-case result validation |

## Models and roles

| Role | Frozen identity |
|---|---|
| Agent | `Qwen/Qwen3-8B`, HF revision `b968826d9c46dd6066d109eabc6255188de91218` |
| Doctor | `gemma3:12b-it-q8_0`, digest `997a7c2c09753e9011ec18a3fcb498e27c0ba330599dfb26a2e05b58cfbdc1bc`, Ollama 0.12.5 |
| Partner-side reflection | `anthropic/claude-4.5-haiku-20251001` |
| Rubric generator | `google/gemini-3.8-flash-20260902` |
| Gate / scorer | `openai/gpt-5.6-luna` |
| Prompt reflection / refinement | `anthropic/claude-sonnet-5.5-20260928` |

## Training

| Item | Value |
|---|---|
| Method | response-level tree GRPO |
| Branching | 4 |
| Maximum Agent turns | 4 |
| Earliest adaptive stop | after 2 Agent turns |
| Batch | 5 cases |
| GPUs | five A100 80GB GPUs |
| LoRA | rank 16, alpha 32, RS-LoRA, dropout 0 |
| LoRA targets | q/k/v/o projections |
| Learning rate | 1e-5 |
| KL beta | 0.04 |
| Clip epsilon | 0.2 |
| Gradient clip | 1.0 |
| Advantage | same-state sibling mean-centering; no standard-deviation division |
| Reward | rubric only: outcome weight 0, rubric weight 1 |
| Missing rubric assessment | excluded, never coerced to zero |

## Rubric pipeline

1. Full trajectory is completed.
2. Answer-neutral partner-side reflection selects one Agent reply.
3. Gemini creates one rubric or assigns an existing same-behavior rubric.
4. Applicability uses only the strict prefix before the target reply.
5. Gate checks schema/leakage, answer presupposition, unsupported domain premise,
   and source-locus consistency.
6. Failure sources request score 2; success sources request selective score 0.
7. One intervened Agent reply is followed by one Doctor FINAL.
8. KEEP requires the desired result in at least 2/3 reruns on one valid sampled
   witness; at most ten same-rubric witnesses are sampled.
9. Only applicable KEEP rubrics score response nodes.

## Terminology to preserve

- Call the feedback **partner-side reflection** or **post-trajectory behavioral
  reflection**.
- State that the separate observer analyzes Doctor behavior. Do not claim that
  the local Gemma Doctor itself authored the reflection text.
- Do not call rubric score medical correctness or effectiveness.
- Do not treat Gate PASS as equivalent to rerun KEEP.
- Do not treat NA/no-match as behavioral score 0.
- reported checkpoint means 30 committed cases, not 30 optimizer steps.

## Baseline information still required

The user has previously referenced a vanilla accuracy near 67.52%, but the
matching per-case baseline artifact is not contained in the final reported checkpoint package.
Before placing that value in the paper, locate and freeze:

- the exact baseline policy identity;
- identical 622 case IDs and ordering;
- each baseline prediction and correctness value;
- the baseline collaboration/turn protocol;
- parsing errors, if any.

Then compute paired counts by `case_id`:

```text
Fix   = baseline incorrect AND reported checkpoint correct
Break = baseline correct   AND reported checkpoint incorrect
Both correct / both incorrect are reported separately if needed.
```

Without the paired baseline file, Fix/Break cannot be recovered from the two
aggregate accuracies alone.

## Primary source files

- `configs/online_rl/paper_reproduction.json`
- `artifacts/reported_checkpoint/provenance/run_manifest.json`
- `artifacts/reported_checkpoint/provenance/checkpoint_COMPLETE.json`
- `src/online_rl/integrated/grpo.py`
- `src/online_rl/integrated/metrics.py`
- `src/online_rl/tree/reward.py`
- `src/online_rl/tree/team.py`
- `prompts/sherpa/behavior_feedback_v5.txt`
- `prompts/sherpa/trajectory_rubric_generator_P0_aoc_v3.txt`
- `prompts/sherpa/luna_gate_v1.txt`
- `prompts/sherpa/luna_scorer_v2.txt`

