# SHERPA

SHERPA trains a clinical collaborator Agent for short Doctor–Agent discussions of multiple-choice medical questions.
Training is online GRPO over branching interaction trees, with rewards for the final outcome and for verified
behavioral rubrics.

## Overview

The project studies how to train a collaborative clinical agent, with separate interest in three things:

- **medical decision quality**: is the Agent's own (private) answer correct;
- **collaboration quality**: is the Agent's visible reply appropriate, as scored by verified rubrics (`R_A`);
- **final team correctness**: is the answer the Doctor gives after the discussion correct. This is the primary endpoint.

One training batch:

```
policy θ_k (frozen for the batch)
 → 5 training cases, each rolled out as a tree
 → rewards per Agent action
 → sibling-group GRPO over all 5 trees
 → at most one optimizer update → θ_{k+1}
```

Per case:

1. **Tree rollout.** The Doctor opens the case. At every Agent turn each branch samples G = 4 Agent actions. An action
   has two segments: a *private* assessment formed from the case alone (`OPTION` + `KEY_REASON`), then the *visible*
   reply, which sees the dialogue and the private answer as hidden context. After the last turn the Doctor gives the
   FINAL answer, which decides the outcome.
2. **Feedback and rubrics.** A Feedback Generator describes the Agent's behavior from the dialogue alone. Each
   trajectory maps to one rubric; new rubrics pass a validity gate and are kept only if an intervention on the matched
   reply changes the outcome in reruns.
3. **Scoring.** Each kept rubric scores every Agent reply it applies to. `R_A` is the mean score of the applicable
   rubrics, or NA when none applies.

## Current Canonical Protocol

| Setting | Value |
|---|---|
| Agent (trained) | `Qwen/Qwen3-8B` + LoRA (r 16, α 32, rsLoRA, seeded initial adapter θ0) |
| Doctor | `gemma3:12b-it-q8_0` (pinned digest) on local Ollama 0.12.5 |
| Rubric prompt | `p0_fixed`: the initial Rubric-Generator prompt for the whole run, no prompt evolution |
| Thinking | Qwen3 native thinking ON for every Agent call |
| Turns | `max_turn = 2` Agent turns per branch |
| Sibling sampling | G = 4 actions per Agent turn |
| Agent generation batch | 1 |
| Training batch | 5 cases |
| Optimizer | AdamW, lr 1e-5, no scheduler; KL β 0.04, clip 0.2, grad clip 1.0 |
| Advantages | mean-centred within each same-state sibling group |
| Agent sampling | temperature 0.7, top-p 0.9, top-k from the model's generation config |
| Training cases | first 120 cases of a fixed order of MedMCQA training cases |
| Checkpoints | θ0, then 30 / 60 / 90 / 120 cases |

Other model roles: Feedback Generator `anthropic/claude-4.6-sonnet-20260217`, Rubric Generator
`google/gemini-3.8-flash-20260902`, rubric gate and scorer `openai/gpt-5.6-luna`. API models are called through
OpenRouter. Prompts are pinned by SHA256 in the config, and identical scorer requests are answered from a request
cache.

## Important Batch Terminology

**Training batch = 5 cases.** The policy is frozen while the 5 trees are rolled out. Each training batch makes one call
to the update path and has at most one optimizer update; a batch without any usable sibling group is skipped. So
60 cases are 12 attempted training batches and 120 cases are 24.

**Generation batch = 1.** This is how many Agent generations the GPU processes at the same time. With thinking ON the
batch size changes the sampled text, so the canonical protocol fixes it to 1 on every host.

**G4** is the number of sibling actions per Agent turn. It is independent of the generation batch: with generation
batch 1 the four siblings are still sampled, each with its own seed.

## Experimental Conditions

All conditions use the canonical protocol above with thinking ON. They differ only in the credit rule and its
coefficients. Configs are in `configs/experiments/fixed_t2/main120/` (120 cases) and
`configs/experiments/fixed_t2/pilot/` (60-case pilots and 5-case smoke runs).

### E1 — Shared 5:5

The original SHERPA objective. Each action gets one reward, `0.5 · Outcome + 0.5 · R_A`, and its sibling advantage is
applied to both segments of the action.

### E2 — Decision-only diagnostic

`R_D = 1[private OPTION == gold]`. Only the private segment enters the loss; there is no collaboration loss and no
rubric pipeline. E2 is a mechanism / learnability diagnostic for the private decision, not a collaborative method.

### E3-outcome — Separated 5:5

- Decision branch: final team outcome → private segment
- Collaboration branch: `R_A` → visible reply
- Objective: `L = 0.5 · L_D + 0.5 · L_C`

### E3-private — Separated 5:5

- Decision branch: private OPTION correctness (`R_D`) → private segment
- Collaboration branch: `R_A` → visible reply
- Objective: `L = 0.5 · L_D + 0.5 · L_C`

This is the pre-specified main candidate. It differs from E3-outcome only in the decision reward. 5:5 is the balanced
default, not a tuned or theoretically optimal ratio.

### E3-private 7:3 / 3:7

Sensitivity conditions with the same structure as E3-private:

- 7:3: `L = 0.7 · L_D + 0.3 · L_C`
- 3:7: `L = 0.3 · L_D + 0.7 · L_C`

## Thinking Semantics

- Every Agent call is rendered with Qwen3 native thinking enabled.
- The reasoning tokens are part of the sampled completion: they are stored with their log-probabilities and receive
  the training loss of their segment.
- Only the text after `</think>` is used downstream: as the private answer in the hidden context, and as the visible
  reply in the dialogue. The Doctor, the feedback model and the rubric scorer never see the reasoning.
- Reasoning has its own token budget (2048) on top of the answer budget. If the reasoning does not close within the
  budget the answer is empty, and an empty private answer counts as wrong for `R_D`.

## Credit Assignment

- **Shared** (`agent_reward.credit` absent): one reward per action; the same sibling advantage weights every token of
  both segments.
- **Decision-only** (`decision_only`): `R_D` on the private segment.
- **Separated** (`separated`): the decision reward and `R_A` are centred independently within the same sibling groups.
  The decision advantage applies to the private segment and the collaboration advantage to the visible reply. A reply
  without a rubric score is left out of the collaboration branch only. The two branch gradients are computed
  separately, logged, and combined once with the branch coefficients (`reward_weight_outcome` for the decision branch,
  `reward_weight_rubric` for the collaboration branch). `agent_reward.decision_target` selects `private_option`
  (default) or `outcome`.

Separation is by token segment, not by parameters: both branches update the same LoRA adapter. The coefficients weight
the two objectives; they do not equalize gradient norms.

## Checkpoints and Evaluation

- A checkpoint (adapter, optimizer and RNG state, with a receipt) is written at 30, 60, 90 and 120 cases, and a recovery
  checkpoint after every batch.
- Training always continues to case 120; intermediate evaluation results do not stop or change a run.
- **DEV** is 100 held-out MedMCQA cases (positions 400–499 of the training order, never trained on). It is used for
  development and checkpoint analysis.
- **CasiMedicos622 is TEST.** It must not be used for method selection, coefficient selection, learning-rate tuning,
  prompt selection, stopping, or checkpoint selection.

## Reproducibility

Arms that are compared keep these fixed: θ0 initialization, case order, 5-case batch boundaries, `max_turn`, G4,
thinking protocol, generation batch, optimizer and learning rate. A run resumes only from its own output directory
and only with an unchanged config; a changed config needs a new output directory.

## Running

```bash
pip install -e ".[train,test]"
export OPENROUTER_API_KEY=...
```

The data are local files whose paths and SHA256 values are in the config. Prepare them with
`scripts/online_rl/prepare_five_gpu_train500.py`, `scripts/dev_opt/make_dev_split.py` and
`scripts/online_rl/prepare_casimedicos622.py`.

A run uses five GPUs and five local Doctor servers. `supervise_arm.sh` launches an arm and resumes it after a crash:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4
export SHERPA_OLLAMA_BINARY=/path/to/ollama SHERPA_OLLAMA_MODELS=/path/to/ollama/models
bash scripts/online_rl/supervise_arm.sh t2m_e3_private_5_5
```

| Arm | Condition |
|---|---|
| `t2m_e1_shared_5_5` | E1 |
| `t2m_e3_outcome_5_5` | E3-outcome |
| `t2m_e3_private_5_5` | E3-private |
| `t2m_e3_private_7_3`, `t2m_e3_private_3_7` | E3-private sensitivity |
| `t2p_e2_think_decision` | E2 (60 cases). Its config leaves the generation batch at the default; set it to 1 with a host runtime overlay (`SHERPA_RUNTIME_OVERLAY`, `five_gpu.agent_batch_size`) |

`scripts/online_rl/launch_production.sh <arm> --dry-run` checks an arm's config and environment without starting it.

## Outputs

| Path in the run directory | Content |
|---|---|
| `run_state.json`, `run_manifest.json` | Committed cursor, hashes, full config |
| `checkpoints/` | Adapter, optimizer and RNG state with a receipt |
| `cases/` | Per case: tree, sampled segments with log-probabilities, rubrics, scores, training targets and the batch's update statistics |
| `reward_components/` | Per reply: outcome, rubric score, weights, reward |

## Tests

```bash
python -m pytest -q tests/test_sherpa_credit.py tests/test_fixed_t2.py
python -m pytest -q tests
```

Tests run on CPU with mocks and need no API key.

## Repository layout

```
configs/            protocol configs (online_rl/) and experiment configs (experiments/)
prompts/            Agent, Doctor, feedback and rubric prompts
scripts/online_rl/  launchers, supervisor, preflight, evaluation
scripts/dev_opt/    DEV split, run verification
src/online_rl/      rollout, rewards, rubrics, GRPO, experiment configs
tests/              unit tests and mocked end-to-end runs
```

`configs/experiments/` also holds the configs of earlier protocols (for example `reward_prompt_matrix/`, with prompt
evolution and four Agent turns). They are kept for provenance and are not part of the canonical protocol.
