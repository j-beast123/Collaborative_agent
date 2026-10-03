# Collaborative_agent

**SERL: Reinforcement Learning via Self-Evolving Rubrics for Collaborative Clinical Agents**

A clinical agent that is right on its own can still collaborate badly: it may yield to an unsupported claim, withhold
a finding the other side needs, or keep a position that better evidence has overturned. SERL trains the agent on its
collaboration behavior. Each iteration turns the agent's failed collaborations into rubric candidates, keeps the
candidates whose ideal behavior repairs the failure and whose behavior predicts downstream success, and trains the
agent with GRPO on a reward that combines rubric appropriateness, a learned effectiveness estimate and the final outcome.

## Pipeline

```
policy θ_i ─ TRAIN collaborations ─ failures
   └─ one rubric candidate per failure (failure analyzer)
      └─ Rubric Validation: rerun the failed turn under the candidate's ideal behavior (3 reruns, keep at ≥ 2 correct)
         └─ consolidation into categories ─ effectiveness gap (keep gap > 0, weight = gap, at most 8)
            └─ active rubric set R_i
               └─ GRPO with R_t = 0.4 R_O + 0.3 R_A,t + 0.3 R_E,t ─ θ_(i+1) ─ next iteration (3 iterations)
```

- **Environment (AgentClinic):** the Agent (Qwen3-4B + LoRA) consults a Doctor model that gives the final diagnosis;
  up to 7 turns with early stopping. The outcome is judged by a diagnosis-equivalence judge with a frozen
  majority-vote mapping.
- **R_O:** the Doctor's final diagnosis is correct. **R_A:** the active rubrics scored on the response (Jev).
  **R_E:** P(success | history, response) − P(success | history) from the Effectiveness predictor
  (Qwen3-8B + LoRA + scalar head, served over gRPC).
- **Conditions:** Baseline, Outcome, Effectiveness + Outcome, Static rubric, No-filtering, Ours, and the component
  ablations (`configs/experiments/*/matrix.json`).

## Repository structure

```
configs/
  serl_agentclinic_qwen4b_max7_train30_clean.json   main configuration (TRAIN30 / DEV31 / TEST60)
  serl_agentmediated_frontierdoctor_v1.json         agent-first, agent-mediated Doctor protocol
  effectiveness/                                    environments (models, prompts, protocol, judge)
  experiments/train30_4b_clean/                     condition configs and the experiment matrix
  judges/                                           rubric model roles and defaults
prompts/                                            Doctor, Agent, judge and rubric prompts
src/
  agentclinic/    protocol, cases, trajectories, outcome judge, runtime
  judges/         failure analysis, Rubric Validation, consolidation, effectiveness gap, rubric scoring
  effectiveness/  labels, predictor, calibration, gRPC service
  grpo/           reward, trainer, rollouts
  evaluation/     RQ2 / RQ3 analyses
scripts/
  run_serl.sh            one SERL iteration (base, rubrics, train, report, generate)
  run_grpo.py            run creation, GRPO training, trajectory generation
  build_rubrics.py       the rubric step of an iteration
  serve_effectiveness.py Effectiveness predictor service
  main_experiments.py    condition registry, DEV evaluation, tables
  rq3_fixed_probes.py    RQ3 fixed-probe evaluation
  diagnostics/           pilots, audits and analyses behind design decisions; not part of the main pipeline
                         (run as `python -m scripts.diagnostics.<name>`)
tests/
```

## Setup

Python 3.10 with PyTorch, `transformers`, `peft`, TRL and `grpcio`. API models (Doctor, judge, rubric models) are
called through OpenRouter; put the key in a `.env` file (never committed):

```
OPENROUTER_API_KEY=...
```

The AgentClinic cases are read from `data/raw/agentclinic/agentclinic_medqa_extended.jsonl` (from the AgentClinic
repository; not tracked); the case-level split manifests are in `data/splits/`.

Local model directories are written as `${MODEL_ROOT}/<model>` in the configs (Qwen3-4B, Qwen3-8B); set
`MODEL_ROOT` to the directory that holds them. `MODEL_PATH_MAP=<from>=<to>` remaps a path prefix without editing a config.

## Running

Effectiveness predictor service (needed when w_E > 0 or for the effectiveness gap):

```
python -m scripts.serve_effectiveness --config configs/serl_agentclinic_qwen4b_max7_train30_clean.json --device cuda:0 --port 50558
```

One condition, three SERL iterations:

```
CONFIG=configs/experiments/train30_4b_clean/ours.json OUTPUT_ROOT=outputs/serl_clean ITERATION=0 bash scripts/run_serl.sh
CONFIG=configs/experiments/train30_4b_clean/ours.json OUTPUT_ROOT=outputs/serl_clean ITERATION=1 RUN_ID=<run id> bash scripts/run_serl.sh
CONFIG=configs/experiments/train30_4b_clean/ours.json OUTPUT_ROOT=outputs/serl_clean ITERATION=2 RUN_ID=<run id> bash scripts/run_serl.sh
```

DEV evaluation of a checkpoint and the tables:

```
python -m scripts.main_experiments register --matrix configs/experiments/train30_4b_clean/matrix.json --condition ours --seed 0 --run-id <run id>
python -m scripts.main_experiments evaluate --matrix configs/experiments/train30_4b_clean/matrix.json --condition ours --seed 0 --table-iteration 1
python -m scripts.main_experiments tables --matrix configs/experiments/train30_4b_clean/matrix.json
```

The TEST split is never read unless `--allow-test` is given.

Tests:

```
python -m pytest tests
```

## Outputs

Runs, labels, predictor checkpoints, caches and evaluation results are written under `outputs/` and are not tracked
(`.gitignore`), nor are model weights (`*.safetensors`, `*.pt`, `*.bin`).
