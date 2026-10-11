# Train120 max-4 active model roles

This branch separates the active training configuration from the isolated
all-Luna C3 audit harness.

| Role | Active model/runtime |
|---|---|
| Trainable Agent | Local `Qwen/Qwen3-8B` with a fresh LoRA adapter |
| Rollout Doctor | Local Ollama `gemma3:12b-it-q8_0`, pinned digest `997a7c2c0975...` |
| Rubric generator | OpenAI direct `gpt-6-luna` |
| Gate | OpenAI direct `gpt-6-luna` |
| Behavior scorer | OpenAI direct `gpt-6-luna` |
| Intervention audit | OpenAI direct `gpt-6-luna` |
| Prompt reflection/refinement | Disabled |

The active run uses 120 pinned training cases, at most four Agent turns, five
cases per optimizer update, checkpoints every 30 committed cases, and the
pinned 622-case evaluation set at each checkpoint.

Use:

```bash
SHERPA_FIVE_GPU_CONFIG=configs/experiments/train120_max4_qwen8b_gemma12b_luna_judges_v1.json \
scripts/online_rl/launch_five_gpu.sh
```

The scripts under `audits/gate_v3_3_1_isolated/` used Luna for Agent, Doctor,
and scorer only for one isolated C3 causal diagnostic. They are not part of
the training import or launch path and must not be cited as the training model
identity.
