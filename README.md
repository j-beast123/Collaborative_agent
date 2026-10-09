# SHERPA

SHERPA trains a clinical collaborator Agent for multi-turn Doctor–Agent discussions of multiple-choice medical
questions. Training is online GRPO over branching interaction trees. The reward combines the final outcome with
verified behavioral rubrics, and the Rubric-Generator prompt can evolve during training.

## Overview

```
policy θ_k (frozen for the batch)
 → 5 distinct training cases, each rolled out as a tree
 → outcome + rubric reward per Agent reply
 → sibling-group GRPO over all 5 trees
 → at most one optimizer update → θ_{k+1}
 → next 5 cases rolled out fresh with θ_{k+1}
```

Per case:

1. **Tree rollout.** The Doctor opens the case. At every Agent turn, each branch samples G = 4 Agent replies. A branch
   ends at the turn limit, when the Doctor signals that the discussion is settled, or when the Agent repeats an earlier
   reply. The Doctor then gives a FINAL answer, and the outcome is success or failure.
2. **Feedback.** A Feedback Generator describes the Agent's behavior from the dialogue alone. It sees no question, gold
   answer or outcome, and it names one Agent reply as the behavioral locus.
3. **Rubrics.** Each completed trajectory maps to one rubric, either new or reused for the same behavioral reason within
   its case and outcome stratum. New rubrics pass a validity gate.
4. **Verification.** A rubric is kept if rewriting the matched reply at the rubric's target level flips the outcome in
   2 of 3 continuations, on at least one of up to 10 sampled member trajectories.
5. **Scoring.** Each kept rubric scores every Agent reply in the tree: applicability first, then level 0/1/2.

Each Agent action has two stages: a private assessment formed without the dialogue, then the visible reply. GRPO
treats both stages as one action.

## Training protocol

| Setting | Value |
|---|---|
| Training cases | 500 distinct MedMCQA training cases in a fixed order |
| Batch | 5 cases per batch; policy frozen within the batch |
| Update | At most one optimizer step per batch over all sibling groups |
| Tree | G = 4 replies per Agent turn, at most 4 Agent turns |
| Halting | Doctor-signalled stop after ≥ 2 Agent turns; a repeated Agent reply stops the branch from turn 4 |
| GRPO | lr 1e-5, β 0.04, clip 0.2, grad clip 1.0, mean-centred advantages within same-state sibling groups |
| Skipped groups | Fewer than two scored replies, or zero reward variance |
| Agent | `Qwen/Qwen3-8B` + LoRA (r 16, α 32, rsLoRA, seeded initial adapter θ0) |
| Checkpoints | Initial adapter, every 30 cases, and case 500; a recovery checkpoint after every batch |

## Reward configurations

```
final_reward = w_outcome · R_outcome + w_rubric · R_rubric
```

- `R_outcome` is the mean outcome of the decided leaves below the reply.
- `R_rubric` is the mean level/2 over the kept rubrics that apply to the reply.
- If no kept rubric applies, `R_rubric` is NA. With `w_outcome = 0`, an NA reply gets no reward and is excluded from
  GRPO. Otherwise the NA reply keeps its outcome term.

| Arm | Rubric | Outcome | Prompt |
|---|---:|---:|---|
| Outcome-only | 0.0 | 1.0 | N/A |
| R03/O07 | 0.3 | 0.7 | Evolving |
| R05/O05 | 0.5 | 0.5 | Evolving |
| R07/O03 | 0.7 | 0.3 | Evolving |
| Rubric-reward-only | 1.0 | 0.0 | Evolving |

Outcome-only disables the rubric pipeline: there are no feedback, rubric, verification or prompt calls. In the
Rubric-reward-only arm the training reward has no outcome term, but prompt evolution and verification still use
outcomes.

## Prompt modes

- `evolving`: start from P0 and run one prompt comparison per batch.
  1. The first case to finish under the current prompt seeds one candidate prompt.
  2. Both prompts are evaluated on the same frozen trees.
  3. J = (mean success-direction effect) × (mean failure-direction effect).
  4. The candidate is adopted only if its J is strictly higher.
  5. Training uses the deduplicated union of both prompts' kept rubrics.
- `p0_fixed`: P0 for the whole run.
- `pfinal_fixed`: the final adopted prompt of a completed `evolving` run, fixed from case 0. The exported prompt must
  match its source run's reward, feedback model, data, seed, θ0, P0 and protocol. Otherwise loading fails.

## Experiment plan

| Priority | Experiment | Runs |
|---|---|---|
| 1 | Reward ablation | The five arms above |
| 2 | Prompt ablation at R05/O05 | `p0_fixed`; the `evolving` arm is the R05/O05 reward-ablation run |
| 3 | Prompt ablation at R05/O05 | `pfinal_fixed`, after the R05/O05 `evolving` run completes |

R05/O05 is a fixed balanced anchor. It is not chosen from evaluation results.

## Model roles

| Role | Model |
|---|---|
| Agent (trained) | `Qwen/Qwen3-8B` + LoRA |
| Doctor | `gemma3:12b-it-q8_0` (pinned digest) on local Ollama 0.12.5 |
| Feedback Generator | `anthropic/claude-4.6-sonnet-20260217` |
| Rubric Generator and matcher | `google/gemini-3.8-flash-20260902` |
| Rubric gate and scorer | `openai/gpt-5.6-luna` |
| Prompt reflection and refinement | `anthropic/claude-sonnet-5.5-20260928` |

API models are called through OpenRouter. Prompts and templates are pinned by SHA256 in the config.

## Evaluation

- Dataset: CasiMedicos, all 622 English cases. Training and evaluation questions must not overlap.
- Cadence: every 30-case checkpoint and the final checkpoint at case 500. Training pauses until the evaluation of
  a checkpoint completes.
- Protocol: full Doctor–Agent episodes with the checkpoint's frozen adapter and the training Doctor. There is no
  training signal.
- Seeds: case `i` uses seed `eval_seed + i`, with `eval_seed = 0`. Every checkpoint and run uses the same seeds.
- Output: `evaluation/checkpoint_<cases>/` with per-case results and `summary.json`.

## Configuration

Experiment configs are in `configs/experiments/reward_prompt_matrix/`:

| Path | Content |
|---|---|
| `matrix.json` | Reward ablation and prompt ablation definitions |
| `base_batch5.json` | Shared protocol; extends `configs/online_rl/tree_v21_five_gpu_shared5_random10_promptv7.json` |
| `reward_sweep/`, `prompt_ablation/` | Generated run configs |
| `manifest.json` | Generated list of runs with family, priority, identity and status |

Each run config sets these `experiment` fields:

- `reward_weight_rubric`, `reward_weight_outcome`
- `prompt_mode`, `pfinal_artifact`
- `feedback_model`
- `seed`
- `train_order_sha256`
- `output_dir`

A run id has the form `<reward>__<prompt mode>__<feedback model>__seed<n>`. Outputs go to
`outputs/reward_prompt_matrix/<run id>/`.

## Running

```bash
pip install -e ".[train,test]"
export OPENROUTER_API_KEY=...
```

The training and evaluation data are local files. Their paths and SHA256 values are in the base protocol config.
Prepare them with `scripts/online_rl/prepare_five_gpu_train500.py` and `scripts/online_rl/prepare_casimedicos622.py`.

Check configs (no GPU or API calls):

```bash
python scripts/online_rl/experiment_matrix.py --check
python scripts/online_rl/run_five_gpu.py --config configs/experiments/reward_prompt_matrix/reward_sweep/r05_o05_evolving.json
```

Launch a run (starts five local Doctor servers, then trains). Re-running the same command resumes:

```bash
SHERPA_FIVE_GPU_CONFIG=configs/experiments/reward_prompt_matrix/reward_sweep/r05_o05_evolving.json \
  bash scripts/online_rl/launch_five_gpu.sh
```

A run uses five logical GPUs. To place them on other devices, set `CUDA_VISIBLE_DEVICES` to exactly five ids, e.g.
`CUDA_VISIBLE_DEVICES=0,1,2,3,4`. A run resumes only with the same config, reward, prompt, feedback model, data, seed
and θ0. The physical devices may change.

Export Pfinal after the R05/O05 `evolving` run completes, then set `pfinal_artifact` in `matrix.json` and regenerate:

```bash
python scripts/online_rl/export_pfinal.py --run outputs/reward_prompt_matrix/r05_o05__evolving__sonnet46__seed0
python scripts/online_rl/experiment_matrix.py
```

## Outputs

| Path in the run directory | Content |
|---|---|
| `run_state.json`, `run_manifest.json` | Committed cursor, hashes, full config |
| `checkpoints/` | Adapter, optimizer and RNG state with a receipt |
| `cases/` | Tree, feedback, rubrics, scores and training targets per case |
| `prompt_events/`, `prompts/` | Per-batch prompt decisions and prompt texts (`evolving`) |
| `reward_components/` | Per reply: `R_outcome`, `R_rubric`, weights, `final_reward` |
| `evaluation/` | Checkpoint evaluation results |

## Tests

```bash
python -m pytest -q tests/test_reward_prompt_matrix.py
python -m pytest -q tests
```

Tests run on CPU with mocks and need no API key.

## Repository layout

```
configs/          protocol configs (online_rl/) and experiment matrices (experiments/)
prompts/          Agent, Doctor, feedback, rubric and prompt-evolution prompts
scripts/online_rl/  launchers, config validation, Pfinal export
src/online_rl/    rollout, rewards, rubrics, GRPO, prompt evolution, experiment configs
tests/            unit tests and mocked end-to-end runs
```
