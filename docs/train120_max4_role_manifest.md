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
cases per optimizer update, and checkpoints every 30 committed cases. The
built-in blocking checkpoint evaluation is disabled. A separate cooperative
idle evaluator handles all 622 pinned cases at C30, C60, C90, and C120. It uses
the same four-turn rollout and two-turn minimum before early termination.

Use the combined launcher so training and evaluation share the same GPU
priority coordinator:

```bash
nohup scripts/online_rl/launch_train120_max4_idle_eval.sh \
  > logs/train120_max4_idle_eval.log 2>&1 < /dev/null &
```

The scripts under `audits/gate_v3_3_1_isolated/` used Luna for Agent, Doctor,
and scorer only for one isolated C3 causal diagnostic. They are not part of
the training import or launch path and must not be cited as the training model
identity.
