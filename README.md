# SHERPA — B200 V7 rubric-only

현재 추가 실험: **V7 생성·적용 판단·채점 + rubric-only + case별 병렬 + OpenAI Luna 통일**.
고정 medical-accuracy criterion과 case 간 누적 rubric bank는 사용하지 않습니다.
Case 내부 후보 생성/중복 제거, 고정 source의 선형 continuation verification은 유지합니다.

- API 역할: direct OpenAI `gpt-6-luna`, generator/consolidate reasoning `none`.
- 유효 KEEP rubric 점수만 동일 가중 평균. 모두 N/A/유효 rubric 없음 → 학습 제외. Outcome fallback 없음.
- Train120, max4/branching4, 5-case frozen-policy update, 기존 GRPO/LoRA 유지.
- Doctor: 로컬 HF Gemma BF16, 사전 need 없음, terminal reflection 유지. Assistant: 로컬 Qwen3-8B.
- GPU0 Doctor / GPU1 Assistant, case workers5, 후보 생성 묶음8, 전체 API 동시 상한50/run.
- 학습 중 test 없음. 완료 후 최종 checkpoint로 test622 한 번 평가.

**[새 V7 설정·두 서버 tmux 실행·재개·평가 명령](docs/B200_V7_RUBRIC_ONLY.md)**

설정: [train120_max4_v7_rubric_only_b200.json](configs/experiments/train120_max4_v7_rubric_only_b200.json)
런처: [launch_b200_v7.sh](scripts/online_rl/launch_b200_v7.sh)
새 이름 사용: 로컬 `outputs/runs/NAME/`, HF `zzu819/SHERPA`의 `runs/NAME/`.
이전 run의 checkpoint/cache를 새 보상에 이어 붙이지 않습니다.

V7 upstream commit: `ca0ad31f78347b23cad20b638bffeb65dd888a62`.
성능 향상 자체를 확인한 것은 아니며 구현/CPU 검증만 수행합니다.
Compact scorer는 criterion×응답별 호출이므로 기존 묶음 채점보다 API 비용이 늘 수 있습니다.

이전 accuracy 포함 버전의 설정/런처는 보존합니다:
[기존 B200 설명](docs/B200_SHERPA.md), [clone·HF 준비](docs/B200_CLONE_RUN.md).
새 변형 실행에는 위의 V7 안내를 사용하세요.

CPU regressions: `SHERPA_PYTHON=.venv/bin/python bash scripts/online_rl/test_b200.sh`.

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
