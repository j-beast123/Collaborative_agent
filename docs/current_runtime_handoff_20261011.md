# Current max4 training runtime handoff

This branch captures the code, prompts, configs, tests, and small input datasets
used by the current max4 cumulative-rubric training runtime.

## Models and judging path

- Agent: local Qwen3-8B
- Doctor: local Gemma 3 12B
- Rubric generator: Gemini, as selected by the experiment config
- Gate: Luna through the configured OpenRouter path
- Behavior scorer: the C10 config selects direct OpenAI `gpt-6-luna`
- Maximum dialogue length: four Agent turns, with the current early-stop policy

## Included configs

- `configs/experiments/train120_max4_group4_parallel_gate_guardian_v1.json`:
  fresh guardian run configuration.
- `configs/experiments/train120_max4_group4_parallel_gate_guardian_from_c5_v1.json`:
  recorded C5 handoff configuration.
- `configs/experiments/train120_max4_guardian_c10_openai_direct_luna6_scorer_v1.json`:
  recorded C10 handoff with the direct OpenAI Luna behavior scorer.

The C5 and C10 configs intentionally preserve the original handoff hashes and
paths. Their LoRA recovery checkpoints and rubric-bank snapshots live under
`outputs/` and are not stored in Git. Copy those artifacts separately to resume
the exact policy state. Use the fresh guardian config when starting without
those artifacts.

## Environment

Create the environment file outside Git and provide the API credentials expected
by the selected config. Do not commit it. The launchers read it through
`SHERPA_ENV_FILE`.

The local Agent/Doctor model paths and Ollama version must match the config.
Input files required by the 120-case run and the 622-case idle evaluation are
included under `data/interim/` so they do not depend on machine-specific
symbolic links.

## Launch

From the repository root, validate first:

```bash
SHERPA_ENV_FILE=/absolute/path/to/.env \
python scripts/online_rl/run_five_gpu.py \
  --config configs/experiments/train120_max4_group4_parallel_gate_guardian_v1.json
```

Then launch:

```bash
nohup env SHERPA_ENV_FILE=/absolute/path/to/.env \
  python scripts/online_rl/run_five_gpu.py \
  --config configs/experiments/train120_max4_group4_parallel_gate_guardian_v1.json \
  --launch > logs/current_max4_runtime.log 2>&1 < /dev/null &
```

The saved C10 config cannot be resumed from Git alone because its 176 MB LoRA
recovery checkpoint and 1.5 MB live bank snapshot are runtime state rather than
source files.
