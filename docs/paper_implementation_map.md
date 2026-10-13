# Paper-to-code implementation map

This document maps the submitted SHERPA method to the frozen implementation.
It distinguishes the exact reported checkpoint from a fresh 120-case run.

## Entry points

| Purpose | Entry point | Configuration |
|---|---|---|
| Verify the package without GPU/API calls | `bash scripts/verify_package.sh` | frozen artifact hashes and focused tests |
| Reproduce the reported checkpoint training protocol | `bash run_paper.sh [--launch]` | `configs/online_rl/paper_reproduction.json` |
| Run the same method over 120 cases | `bash run_paper.sh --full [--launch]` | `configs/online_rl/paper_full_120.json` |
| Re-evaluate the saved LoRA on 622 cases | `python scripts/online_rl/eval_checkpoint.py [--launch ...]` | frozen artifact manifest |

No command performs paid inference or loads a GPU without `--launch`.

## Algorithm 1 and Algorithm 2

| Paper step | Implementation |
|---|---|
| Tree rollout with branching factor 4 and at most 4 Agent turns | `src/online_rl/tree/rollout.py`, `src/online_rl/tree/team.py` |
| Partner-side post-trajectory reflection | `prompts/sherpa/behavior_feedback_v5.txt`; model configured as Claude Haiku 4.5 |
| Rubric generation and conservative same-reason reuse | `src/online_rl/tree/rubric_bank.py`, `prompts/sherpa/trajectory_rubric_generator_P0_aoc_v3.txt`; Gemini 3.8 Flash |
| Schema, leakage, source-locus, and LLM Gate validation | `src/online_rl/tree/posthoc.py`, `prompts/sherpa/luna_gate_v1.txt`; GPT-5.6 Luna |
| Counterfactual score-0/score-2 intervention and 2-of-3 verification | `src/online_rl/tree/effectiveness.py`, `src/online_rl/tree/witness.py`, guided-response templates |
| Prefix-only applicability and 0/1/2 response scoring | `src/online_rl/tree/scoring.py`, `src/online_rl/tree/luna.py`, `prompts/sherpa/luna_scorer_v2.txt`; GPT-5.6 Luna |
| Rubric-only reward, with NA excluded rather than converted to zero | `src/online_rl/tree/reward.py` |
| Same-state sibling mean-centered advantages and clipped GRPO | `src/online_rl/integrated/grpo.py`, `src/online_rl/integrated/metrics.py` |
| Five-case online prompt refinement and product fitness | `src/online_rl/tree/prompt_search.py`; Claude Sonnet 5.5 |
| Five-GPU scheduling and atomic checkpoint/recovery receipts | `scripts/online_rl/run_five_gpu.py`, `src/online_rl/tree/gpu_pool.py` |

## Frozen experimental contract

- Agent: `Qwen/Qwen3-8B`, pinned Hugging Face revision, rank-16 RS-LoRA,
  alpha 32, dropout 0, q/k/v/o projections.
- Physician simulator: Gemma 3 12B IT, Ollama 0.12.5, pinned Q8_0 digest.
- Four maximum Agent turns; adaptive termination only after two Agent turns.
- Agent sampling: temperature 0.7, top-p 0.9, maximum 400 completion tokens.
- Five A100 80GB GPUs; five cases per update batch; GPU 4 performs the update.
- Learning rate 1e-5, gradient clipping 1.0, GRPO epsilon 0.2, KL beta 0.04.
- Reward is rubric-only (`w_outcome=0`, `w_rubric=1`). Applicable levels map
  to 0, 0.5, and 1 and are averaged. Missing/NA assessments are excluded.
- Advantages are mean-centered among responses sharing the same dialogue state.
- A Gate PASS is not a KEEP. KEEP requires a target outcome in at least two of
  three trials under the bounded witness protocol. Each trial generates one intervention Agent reply followed by Doctor FINAL; a third trial is needed only when the first two disagree.
- Evaluation uses the physician's final decision on all 622 English
  CasiMedicos-Arg cases.

## Prompt files

The exact run uses content-addressed files under `prompts/`. The primary files
are the Agent prompt, physician simulator prompt, partner reflection prompt,
rubric generator, Gate, scorer, intervention templates, and prompt-refinement
prompts. Their hashes are recorded in the reproduction config and run manifest.
`prompts/sherpa/trajectory_rubric_generator_readable.txt` is a readable paper
view; it does not replace the exact hashed generator prompt.

## Reported artifact and fresh training

`artifacts/reported_checkpoint/` contains the LoRA adapter, data, prompt copies,
run provenance, and all 622 saved predictions for the reported result. Its
integrity is checked independently by `scripts/verify_package.sh`.

The exact reported checkpoint was committed after 30 training cases. The paper
describes a 120-case training pool. `paper_full_120.json` changes only the fresh
run name/output and scheduled case count to 120; it does not claim to reproduce
the already saved checkpoint bit-for-bit and cannot inherit its cache or state.

## External dependencies

Fresh training requires five matching local Gemma Doctor workers, the pinned
Qwen base weights, five CUDA GPUs, and private OpenRouter credentials for the
Claude, Gemini, Luna, and Sonnet roles. Secrets are read from `SHERPA_ENV_FILE`
or a local `.env` and must never be committed.

The base model can be downloaded from Hugging Face at the pinned revision; no
machine-specific `../vllm/qwen3` path is required by the paper configuration.
