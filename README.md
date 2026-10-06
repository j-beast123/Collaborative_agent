# SHERPA

SHERPA trains a clinical collaborator Agent with online GRPO in multi-turn Doctor–Agent discussions of
multiple-choice medical questions. For every training case the current policy plays a branching interaction tree. Each
completed trajectory gets one behavioral rubric, written after the outcome is known and verified by intervention
reruns. The Agent is rewarded for downstream success and for satisfying verified rubrics. In parallel, the
Rubric-Generator prompt evolves toward prompts whose rubrics actually change outcomes when acted on.

## Pipeline

```
case
 └─ interaction tree: Doctor opens; at each Agent turn every live branch samples G replies; up to T Agent turns
     └─ completed trajectories → Doctor FINAL → outcome (success / failure)
         └─ Doctor feedback on the Agent's behavior (no gold answer)
             └─ ONE rubric per trajectory (Rubric Generator, prompt P_t) → validity gate
                 └─ verification: rewrite each matched Agent reply at the rubric's
                    Score-2 level (failure) or Score-0 level (success), then K continuations
                     └─ s(τ) = share of reruns that flip the outcome → keep / remove the rubric
                         └─ kept rubrics score every Agent reply of the tree → R_A
                             └─ reward per reply = w_outcome · mean(descendant outcomes) + w_rubric · R_A
                                 └─ GRPO with same-state sibling replies as one group → updated Agent

prompt evolution (every N cases):
  stored trajectories → ONE rubric each under P → s(τ) → S(P) = Σ s(τ)
   → reflection / refinement → candidate prompts → paired acceptance → P_t+1
```

- **G vs K.** G is the branching factor of the tree; K is the number of continuations per intervention. K is a rerun
  count, never a number of rubrics: each trajectory has exactly one rubric.
- **s(τ)** (stored as `R_O`) only keeps or removes a rubric, weights it inside R_A and scores Rubric-Generator prompts.
  It is never added to the Agent reward on its own.
- **R_A** of a reply is the s(τ)-weighted mean of the scores of the kept rubrics that apply to it (two-stage judge:
  applicability on the dialogue before the reply, then the 0/1/2 level).

## Model roles

| Role | Model |
|---|---|
| Doctor (simulated physician, feedback) | `google/gemma-3-12b-it` (OpenRouter) |
| Agent (trained policy; also writes intervention replies) | `Qwen/Qwen3-8B` + LoRA |
| Rubric Generator, prompt reflection / refinement | `openai/gpt-6.1-sol` (OpenRouter) |
| Rubric gate and scoring | `typesafe/jev-1.13` (OpenRouter decisions) |

## Installation

```bash
pip install -e ".[train,test]"
export OPENROUTER_API_KEY=...        # or a .env file at the repository root, or SHERPA_ENV_FILE=<path to it>
```

Training cases are a JSONL file of MedMCQA-style items (`case_id`, `question`, `options`, `correct_option`) set in
`data.train`. Question text is not redistributed; evaluation-only splits are protected through
`data/splits/protected_eval_v1.json` (ids and question hashes only).

## Running

```bash
python scripts/online_rl/run_tree.py --config configs/online_rl/tree_v1.json            # prints blockers only
python scripts/online_rl/run_tree.py --config configs/online_rl/tree_v1.json --launch
```

`configs/online_rl/tree_v1.json` is the research configuration (G = 4, T = 5). The launcher refuses to start while any
value is `CALIBRATION_REQUIRED` or a prompt file differs from its pinned sha256, and resumes from the last committed
case. A config with a `smoke` block is a size-capped wiring check run one phase at a time (`--phase A`: one training
case; `--phase B`: the following prompt event).

## Tests

```bash
python -m pytest -q tests
```

The tests use mocks and a tiny CPU LoRA model; no API access or GPU is needed. Tests that need the local training split
are skipped without it.

## Repository structure

```
src/online_rl/tree/         case tree: rollout, post-hoc feedback and rubrics, verification, scoring, reward, trainer,
                            prompt search, real backends
src/online_rl/integrated/   GRPO update, prompt store, records and statistics shared with the tree (and the earlier
                            single-state pipeline, run by scripts/online_rl/run_integrated.py)
src/online_rl/              policy runtime, environment, case loading, protected-data guards
src/prompt_opt/             Agent/Doctor harness, rubric parsing and gates
src/judges/                 Jev client and two-stage rubric scoring
src/rq_eval/                evaluation harnesses
scripts/online_rl/          launchers
configs/online_rl/          configurations
prompts/                    Agent, Doctor, Rubric-Generator, feedback, intervention and prompt-search prompts
tests/                      unit tests and mocked end-to-end runs
```

## Reproducibility

Every external call (Doctor, Agent, judge, rubric generation, reruns) goes through an immutable call cache, so a resumed
run repeats no completed work. Each case commits its record, the adapter checkpoint and its hash atomically. Seeds
derive from `sha256(run_seed | key | role | index)`; the Doctor API is not seedable, so cached results are treated as
recorded samples. Prompt files are pinned by sha256 and prompt versions are content-addressed.

## Status

The pipeline is implemented and has passed a real end-to-end wiring smoke. Research hyperparameters (number of cases,
K, keep threshold, reward weights, learning rate, prompt-update cadence and acceptance) are still to be calibrated, and
the fidelity of the intervention replies and the Score-0 path are to be validated in the pilot.
