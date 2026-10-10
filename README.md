# SHERPA — B200 two-GPU improvement

The new `sherpa-jh` experiment uses train120, four Agent turns, branching four,
five-case frozen-policy batches and rubric-only GRPO. It adds prospective private
Doctor needs, need-linked reflection, fixed-source linear continuation verification,
and one always-applicable medical-accuracy criterion per case.

Generator, consolidate, gate and scorer now use direct OpenAI `gpt-6-luna`
with `OPENAI_API_KEY`. Generator/consolidate explicitly request reasoning `none`;
gate/scorer retain provider-default reasoning.
Their unsupported temperature parameter is omitted; Agent/Doctor decoding is unchanged.
Independent candidates run in configurable chunks (default8); one case-level
consolidation fixes the bank and source mappings before gate/verification.
All API roles and retries share a global50 in-flight limit per run (two servers:100).
Doctor, rubric-only reward and the five-case update barrier remain unchanged.
OpenRouter credentials are not required for this B200 configuration.
[Existing-server environment correction](docs/B200_CLONE_RUN.md#현재-서버-preflight-환경-수정).

**현재 Doctor: 기존 로컬 HF Gemma, BF16 (비양자화).** `SHERPA_DOCTOR_BACKEND=hf_local`로 사용하며 Ollama는 필요 없습니다. [모델 pin·실행 명령](docs/B200_CLONE_RUN.md#로컬-hf-gemma-3-12b-비양자화-doctor-사용-현재-선택).

**공유 파일시스템에서 두 B200 실험:** `--run-name NAME`으로 로컬 `outputs/runs/NAME/`과 HF `runs/NAME/`을 함께 분리합니다. [이름 입력·LR 비교·재개 명령](docs/B200_CLONE_RUN.md#공유-파일시스템의-b200-두-서버-실험-이름-하나로-분리).

**새 서버 clone → 환경 준비 → 학습/재개/평가:** [한국어 실행 명령](docs/B200_CLONE_RUN.md).
HF dataset `zzu819/SHERPA` is pinned and byte-verified against train120/test622.
Set `SHERPA_HF_RUN_ID` and `HF_TOKEN` to back up complete committed batches and
final evaluation outputs to the model repo `zzu819/SHERPA` (LoRA, optimizer,
state, diagnostics and caches). [HF preparation/backup/restore commands](docs/B200_CLONE_RUN.md#9-hf-checkpointoutput-저장과-불러오기).
Learning rate: `SHERPA_LEARNING_RATE=5e-6` 또는 `--learning-rate 5e-6` (기본 `1e-5`).
다른 LR은 별도 출력 경로를 사용하며 재개·평가는 학습 당시 값을 유지합니다.

**Setup, complete commands, resume rules and changes:** [B200 execution guide](docs/B200_SHERPA.md).
Configuration: [train120_max4_rubric_only_b200.json](configs/experiments/train120_max4_rubric_only_b200.json).
Launcher: [launch_b200.sh](scripts/online_rl/launch_b200.sh).
Outputs after Doctor preparation: `outputs/sherpa_b200_max4_parallel_bank_v2/doctor_<digest12>/`.
The execution guide includes an account-local Ollama0.12.5 installer and Doctor
download/pinning commands; `.local/b200_doctor.json` automatically binds runtime
paths and the actual downloaded digest. Existing system services are left intact.
CPU regressions: `SHERPA_PYTHON=.venv/bin/python bash scripts/online_rl/test_b200.sh`.

Unlike the earlier rule, all partner criteria being N/A does **not** exclude the
group: a valid fixed accuracy score still contributes to learning. Outcome never
fills a missing rubric reward. The fixed criterion bypasses verification and is
inserted once per case, with the same weight as each verified partner criterion.

The following section documents the preserved **older two-turn configuration**;
its reward, GPU layout and immediate-FINAL verification do not describe the new run.

## Earlier 120-case, two-turn training

This branch packages the current fixed-P0 training setup. Start with
[`configs/experiments/train120_max2_fixed_p0.json`](configs/experiments/train120_max2_fixed_p0.json)
and [`scripts/online_rl/launch_train120_max2.sh`](scripts/online_rl/launch_train120_max2.sh).
The configuration is standalone: it does not extend older experiment files.

## Active protocol

| Setting | Value |
|---|---|
| Data | The same pinned 120 MedMCQA cases, in the same order |
| Prompt | Fixed P0 v2.2 with Doctor-reflection grounding checks; no P1 optimization |
| Rollout | At most **2 Agent turns**, 4 sibling replies per state; at most 16 leaves per case |
| Batch | 5 cases; one frozen Agent policy throughout the batch |
| GPU work | GPU0–4 share ready Agent, Doctor, FINAL and rerun tasks |
| Concurrency | Agent batch 4; Doctor requests 1 per actor |
| Update | GPU4 performs at most one LoRA optimizer step after the five cases finish |
| Reward | 0.5 × outcome + 0.5 × rubric score |
| GRPO | Same-state siblings, mean-centred advantage; learning rate 1e-5, KL coefficient 0.04, clip 0.2 |
| Sampling | Temperature 1, top-p 1, top-k 0; raw-policy probability checks |
| Checkpoints | Every 30 cases and at completion; recovery state after each batch |
| Evaluation | Disabled in this configuration |
| End | Stop after the selected 120 cases |

The Agent uses a private pre-assessment followed by a visible reply. Their joint
probability is used for the training action. A fresh run starts from the base
model with a seeded LoRA adapter; an existing run resumes its saved state.

### Order of work

1. Roll out the conversation trees and obtain each Doctor FINAL.
2. The same local Doctor writes its post-FINAL reflection. The generator checks
   the reflection's claims against actual speakers, turn IDs and quotations;
   a later need cannot become a request that existed before the selected reply.
3. Generate or reuse a rubric within each case and success/failure stratum.
   Rubrics retain user context, Agent state, and distinct 0/1/2 behavior criteria.
   A reused rubric gets a separate intervention instruction and target turn for
   each member trajectory. Success and failure strata can run concurrently.
4. Check source applicability and the evidence-based quality gate, including
   an independent score of the original response.
5. Verify a rubric on at most **5 sampled member trajectories**. Each rerun is
   one fresh guided Agent reply followed by one Doctor FINAL. Within a member,
   two agreeing results can end the planned three reruns early. A rubric is
   kept when a member produces the requested outcome flip in at least 2/3.
6. Freeze the case's KEEP set, then score its Agent responses. Same-state
   siblings share applicability; behavior scoring batches four responses.
7. Form rewards and sibling advantages, update, save, and proceed to the next
   five cases with the updated policy.

NA is distinct from behavior score 0. If no rubric applies, the mixed reward
retains the outcome contribution. Unscorable behavior records are separately
represented and masked consistently within sibling comparisons. Groups with
insufficient valid responses or no reward variation do not produce an update.

## Model roles and files

| Role | Model / asset |
|---|---|
| Agent | `Qwen/Qwen3-8B`, LoRA r16/alpha32, rsLoRA, q/k/v/o projections |
| Doctor and its reflection | Local `gemma3:12b-it-q8_0`, Ollama **0.12.5** |
| Rubric generation and matching | `google/gemini-3.8-flash-20260902` through OpenRouter |
| Quality gate | `openai/gpt-5.6-luna` through OpenRouter |
| Behavior scorer | `openai/gpt-6-luna` through OpenRouter |

Active prompt assets:

- [Main Agent](prompts/agent.txt) and [Doctor discussion/FINAL](prompts/user_simulator_v4_final_label.txt)
- [Generator v2.2](prompts/sherpa/trajectory_rubric_generator_v2_2_reflection_audit.txt)
  and [local output schema](prompts/sherpa/trajectory_rubric_generator_v2_2_reflection_audit.schema.json)
- [Doctor reflection](prompts/sherpa/doctor_post_final_reflection_v1.txt)
- [Quality gate](prompts/sherpa/luna_gate_v2_1.txt)
- [Scorer](prompts/sherpa/pilot_joint_v2_1/scorer.txt)
- [Rerun Agent role](prompts/sherpa/pilot_joint_v2_1/agent_system.txt),
  [score-0 intervention](prompts/sherpa/pilot_joint_v2_1/score0.txt),
  [score-2 intervention](prompts/sherpa/pilot_joint_v2_1/score2.txt)

The `pilot_joint_v2_1` filenames are retained because these exact files are used
by the active training path. Prompt/schema hashes are checked before execution.
Compatibility files referenced by runtime defaults, validation or retained tests remain.
This branch's entry point is the standalone config above. Inactive prompt-search
fields and historical notes do not enable P1 or evaluation.

## Execution map and repository cleanup

| Stage | Files to edit or inspect |
|---|---|
| Training settings | `configs/experiments/train120_max2_fixed_p0.json` |
| Shell entry / environment | `scripts/online_rl/launch_train120_max2.sh` → `launch_five_gpu.sh` |
| Preflight / Doctor startup | `scripts/online_rl/run_five_gpu.py`, `start_five_gpu_doctors.py` |
| Five-case rollout barrier / checkpoints | `src/online_rl/tree/five_gpu.py`, `five_gpu_worker.py`, `gpu_pool.py` |
| Tree and method wiring | `scripts/online_rl/run_tree.py`, `src/online_rl/tree/trainer.py`, `backends.py` |
| Doctor reflection / grounding | `src/online_rl/tree/doctor_reflection.py`, `feedback_grounding.py` |
| Rubric reuse / quality gate / witness | `src/online_rl/tree/rubric_bank.py`, `quality_gate.py`, `witness.py` |
| Behavior scoring / reward / update | `src/online_rl/tree/scoring.py`, `team.py`, `src/online_rl/integrated/grpo.py` |
| Monitoring / review export | `scripts/online_rl/watch_five_gpu.py`, `export_training_snapshot.py` |

The `sherpa-jh` cleanup removed 129 unreferenced legacy files: 5 prompt assets,
22 old configurations, and 102 diagnostic/one-off launch/recovery scripts.
The [cleanup manifest](maintenance/cleanup_train120_max2.json) records every
removed path and its original SHA-256 against base commit `de756fc`.
All `src/` modules and tests remain. Prompt/configuration files referenced by
those tests, default arguments or shared helpers are retained even if inactive
in the standalone training setup. In particular, `pilot_joint_v2_1` is active;
`observed_context_behavior_v2` and the v3 prompt-search files remain pinned
compatibility assets. They must not be removed just because their flags are off.

No active prompt, algorithm, model role, reward weight, seed, or training setting
was changed. The effective configuration hash below remains unchanged. This
cleanup is local to the `sherpa-jh` worktree; existing baseline run directories
and the parent `method-comparisons` checkout are separate.

## Prepare another server

Use Python 3.11 and five suitable CUDA GPUs. Install dependencies in an isolated
environment, selecting a CUDA-compatible Torch build for the server:

```bash
python -m pip install -r requirements-active.txt
python -m pip install -e . --no-deps
```

The tested environment used Torch 2.6.0+cu124 and Transformers 4.53.3.
Before launching, cache `Qwen/Qwen3-8B` locally and install Ollama 0.12.5 with
`gemma3:12b-it-q8_0`. The required Doctor digest is:

```text
997a7c2c09753e9011ec18a3fcb498e27c0ba330599dfb26a2e05b58cfbdc1bc
```

The launcher uses offline Hugging Face loading and checks the Doctor version and
digest. It creates five Doctor endpoints on ports 11685–11689. Set
`SHERPA_OLLAMA_BINARY` to the installed 0.12.5 executable; the dedicated launcher
uses this environment variable rather than the original server's binary path.
Doctor model storage defaults to `~/.ollama/models`; set `SHERPA_OLLAMA_MODELS`
if your installation keeps the pinned model elsewhere.

### Exact training data

Question text, answers, model weights, run outputs and credentials are not
included in this branch. Supply the canonical training JSONL from the original
server, then reconstruct and verify the exact 120-case selection:

```bash
python scripts/online_rl/prepare_train120.py \
  --source /path/to/train120_excluding_calibration_seed20261008.jsonl
```

The helper also accepts the canonical 500-case source JSONL. It checks source
hashes, ordered IDs, question hashes, protected/calibration exclusions and final
file hashes. It refuses to replace conflicting existing files. The portable
[selection manifest](data/splits/medmcqa_train120_seed20261008.manifest.json)
contains IDs and hashes without question text.

For overlap/identity checks, the current preflight also requires
`data/interim/casimedicos/all_en_622.jsonl`, even though evaluation is disabled.
Copy the canonical file or prepare it from the original English split files:

```bash
python scripts/online_rl/prepare_casimedicos622.py \
  --source /path/to/casimedicos-exp/source/data/en
```

Required hashes:

| File | SHA-256 |
|---|---|
| Training JSONL | `3d34929094925c822062d03839b49de8f6fb43afba71fb573de409471bb5b8ab` |
| Training order JSON | `edf23262be5430d10a0d02dde3da80710bf9a91bd941d83234e9c98372752c41` |
| CasiMedicos 622 JSONL | `e42077d006c4e8227564f99b06050b66e7df9989f9b2b17f5fc5d9dd083feb56` |

## Run and monitor

From the repository root, first run the read-only preflight:

```bash
python scripts/online_rl/run_five_gpu.py \
  --config configs/experiments/train120_max2_fixed_p0.json
```

This checks configuration, prompt/schema identities and data. It does not launch
training, load GPU models or call the APIs.

Provide `OPENROUTER_API_KEY` in the environment or in the untracked `.env` file.
Set executable paths for your server, then launch detached from the terminal:

```bash
export SHERPA_PYTHON="$(command -v python)"
export SHERPA_OLLAMA_BINARY=/path/to/ollama-0.12.5/bin/ollama
export SHERPA_ENV_FILE="$PWD/.env"
mkdir -p logs
nohup setsid bash scripts/online_rl/launch_train120_max2.sh \
  > logs/train120_max2.log 2>&1 < /dev/null &
echo "PID=$!"
```

A successful launch persists after terminal logout. Confirm startup in
`logs/train120_max2.log`; a printed PID alone does not establish successful
startup. Avoid launching a second job into an output directory already in use.
Re-running the same command with the same saved run directory resumes committed
progress. Moving to another server requires transferring the saved run and its
checkpoints separately; cloning this branch alone starts a fresh run.
Saved checkpoint references use absolute paths: preserve the original repository
and run locations when resuming on another server. Resuming under a different
absolute path requires a separate, validated migration.

```bash
RUN_DIR=outputs/reward_prompt_matrix/p0_fixed_v2_2_reflection_audit_max2_train120/r05_o05__p0fixed__gemma3doctor__seed0
python scripts/online_rl/watch_five_gpu.py \
  --run-dir "$RUN_DIR" --compact --interval 2
```

Export originals, rubrics, scoring records, a browsable HTML report and a ZIP:

```bash
python scripts/online_rl/export_training_snapshot.py \
  --run-dir "$RUN_DIR" --through-batch 1
```

Remove `--through-batch 1` to export all available cases at snapshot time. Exports
are written under ignored `outputs/exports/` and do not change training.

## Saved records and checks

The run directory contains `run_manifest.json`, `run_state.json`, checkpoints,
per-case trees, Doctor reflections, generated/reused rubrics, gate evidence,
witness results, final scores, reward components and update diagnostics. Cache
error evidence is recorded separately from valid results.

For regression checks of the active training path, without downloading models
or calling APIs:

```bash
bash scripts/online_rl/test_active_training.sh
```

Some worker-pool tests need permission to create local Unix sockets. The
inherited full `tests/` suite includes older entry points that currently fail
collection; the dedicated command selects the current training checks. The
standalone configuration resolves to the same effective config hash as the
active max2 run:

```text
b6475b72d7f8bd5892008bcf2af1d6b8f7402fbb26b0beef9eb0ebf28a248d6a
```
