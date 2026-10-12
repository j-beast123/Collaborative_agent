# SHERPA online — anchored group learning

The default execution uses `configs/group_online.json`. Five cases run in
parallel through the shared two-GPU pool with one frozen policy per batch;
the batch barrier precedes one GRPO update. LoRA, clipping and KL settings are
inherited from the existing online configuration.

```bash
bash run.sh                          # preparation checks only
bash run_group.sh --launch           # start the group SHERPA run
bash run_group.sh --launch --max-batches 1
bash monitor.sh                      # monitor the group run
runtime/python scripts/online_rl/inspect_group_signal.py --batch 1
```

Results are stored under `outputs/train120_group_sherpa_min2_v6/`. The rubric-only
entrypoint `run_rubric_only.sh` selects the same configuration. The specialized
historical entrypoints select their explicit older configurations.
The earlier first-batch pilot remains under `outputs/train120_group_sherpa_min2/`.
The changed reflection/halting protocol starts a separate run; it does not reuse
that pilot's case records, rewards or checkpoint as if they used the new method.
Interrupted v2/v3 attempts remain under their respective output roots.
The completed v4 pilot remains under `outputs/train120_group_sherpa_min2_v4/`.

## Current learning loop

1. Finish a reference dialogue using the same sampling settings as its sibling
   replies. At each reference turn, reuse the prefix through the partner message
   and add three alternative replies, each with one natural continuation.
   Every trajectory requires at least two Agent turns, and allows at most four.
   A reference of length k=2/3/4 yields 7/10/13 trajectories and k groups of four.
2. The partner privately records a hypothesis, favored option and confidence
   in [0,1] at every discussion turn, including the opening. Discussion uses an
   Ollama schema with separate `reply`, `status` and `private_state` fields, with
   dynamic option labels from each case. Confidence is constrained by a numeric
   enum to 0–1 in increments of 0.01, because numeric min/max alone did not
   constrain Ollama decoding. Only `reply` enters the dialogue. A
   malformed structured discussion gets one seeded format retry; missing states
   remain an error and are never fabricated. The visible reply is requested to
   stay within 60 words so the hidden fields fit the existing 256-token budget.
   These fields never
   enter the assistant's visible transcript. For every group the partner compares
   the shared prefix, its recorded state, four replies and their own natural
   outcomes. Reflection still runs when all outcomes are the same. Optional gold
   is controlled by `group_learning.show_gold_to_reflection` (default false).
   The partner writes natural first-person prose in one `reflection` string.
   Ollama's per-call `format` schema enforces this single-field JSON object;
   the partner does not have to match source IDs or extract exact quotations.
   Replies are labeled Reply 1–4, separately from medical answer choices.
   A format/empty-output failure gets at most one retry with a different seed;
   raw attempts and errors are saved. No valid reflection is rewritten.
3. One generator call per group selects ONE sibling reply whose observed
   behavior was most decisive for the partner's final decision and describes
   that behavior as b. It does not classify it as helpful/harmful. A shared
   reinforcement can be represented by one reply when the partner reports no
   unique difference. The prompt requires reflection/outcome consistency:
   helpful behaviors must come from successful conversations, and hindering
   behaviors from failed ones. If none qualifies, return `{"candidates": []}`;
   this is a valid no-candidate result, not a format failure or verification
   rejection. It does not trigger regeneration, verification, or judging, and
   the group is logged as `no_candidate_rubric`. Unresolved outcomes cannot
   anchor a candidate. The generator supplies state,
   mutually exclusive observed/middle/contrast criteria, opposite behavior
   b_star, and a quote. CODE sets b's endpoint from that source trajectory:
   success -> 2, failure -> 0. The contrast gets the opposite endpoint and the
   middle gets 1. Numeric criteria and the internal direction label are therefore
   assigned by code. Interpretation of the prose judgment remains with the
   generator; code checks the source, quote, and outcome-based endpoints.
   A schema/field/ID/quote validation failure gets at most ONE regeneration,
   with the previous output and error supplied only to the generator. All raw
   attempts are saved. Verification failures do not trigger regeneration.
   Quotes must appear in the specified reply;
   validation allows Unicode normalization, case, whitespace and punctuation
   differences, without fuzzy word-overlap attribution or source-ID repair.
4. Generate one contrast reply per candidate at the original shared prefix;
   reuse that fixed reply in exactly three independently seeded continuations.
   `execution.rerun_workers=3` runs these three continuations concurrently;
   their results are stored in rerun-index order. GPU queues bound actual concurrency.
   Subsequent replies use the ordinary policy without the contrast instruction.
   Keep the rubric only when at least two of the three outcomes flip in the
   required direction. Unknown outcomes are unresolved and receive no numeric s.
5. For each group use only its verified anchored rubrics. One blind judge call
   ranks all four replies against these rubrics from best to worst. Response order is
   shuffled with a reproducible seed and anonymous IDs. Public rubrics include
   only name, state and criteria; source IDs, outcomes, b and b_star are omitted.
   There is no applicability call. The judge explains rubric-grounded differences
   and permits ties only when it cannot meaningfully distinguish the responses.
   With multiple rubrics it gives each equal weight. `r(a)=(4-rank(a))/3`:
   ranks 1/2/3/4 yield rewards 1, 2/3, 1/3, 0. Ties receive the average of
   their occupied ranks (a tie for 2nd and 3rd gives each rank 2.5 and reward 0.5).
   Four indistinguishable responses all receive 0.5 and zero advantage.
   Ranking groups must cover all four shuffled IDs exactly once, and have a
   rubric-grounded reason. Invalid rankings exclude that group from learning.
   Groups without a verified rubric, or without a complete valid judge result,
   are excluded from the update.
6. Collect the batch's scored groups and compute `(r-mean)/(std+1e-8)` within each
   four-response group. Equal rewards yield zero advantage. The loss includes
   only each anchor reply's tokens, with its shared prefix as input. The group
   configuration samples one visible reply per action. Continuation and contrast
   replies never enter the loss. A batch with no scored groups skips the optimizer.

An assistant's `Final answer:` text alone does not stop the dialogue.
`mutual_conclusion` is disabled: conclusion phrases without a matching choice
are insufficient. Explicit `mutual_agreement` requires matching parsed choices
and conclusion signals from both participants. STATUS-based `doctor_halt` now
also requires the latest assistant's parsed option to equal the current Doctor
option, in addition to stable valid STATUS and all unresolved flags being false.
If the assistant's option is absent or different, this path continues. The same
rules apply to natural rollout and verification. The repeated-reply rule remains
at turn four, alongside the four-turn limit. These changes can extend dialogues;
matching, resolved discussions can still finish after two turns.
The update batch remains five cases and the learning rate remains `1e-5`.

### Private reflection boundary

Reflection uses a separate stateless partner `_chat` request with its own system
prompt and response schema. Its result goes to the rubric generator and private
case logs. Neither raw reflection nor structured observations are appended to
the visible dialogue, assistant system/user prompts, continuations, or GRPO
input traces. The schema is scoped to that call and does not persist into later
Doctor discussion or FINAL. Verification supplies only the candidate's `b_star`
behavior instruction to the one replacement reply. The blind judge receives
public rubric state/criteria and visible replies, without reflection or source
provenance. JSON schema constrains output structure; routing enforces this boundary.

For k=4, the case has 13 natural trajectories, four partner reflections, four
generator calls, up to four injected replies and 12 verification continuations,
and up to four reward judge calls. Calls persist in `calls.jsonl` with policy and
protocol identities, so completed work is reused after an interruption.

### Group readiness and concurrency

`group_learning.pipeline=ready_groups_v1` first completes the reference dialogue,
then submits its turn-specific forks concurrently. Each group's three alternative
trajectories complete before that group starts reflection, generation, verification
and judging. The group does not wait for other forks or group pipelines. Final
tree/group serialization follows reference-turn order, independent of completion
order. The source policy stays frozen until all groups from all five cases finish.
`execution.group_rollout_workers=4` and `group_pipeline_workers=4` cover the maximum
four reference groups per case. Existing GPU and API slots bound model concurrency.
`five_gpu.overlap_rollout_with_bank=false` controls the historical cumulative-bank
path, and does not disable this group pipeline. There is no shared mutable bank.

Core code: `src/online_rl/tree/group_learning.py`,
`src/online_rl/tree/group_backends.py`, and `src/online_rl/tree/rollout.py`.
Reflection and generation use `prompts/sherpa/group_reflection_v2.txt` and
`group_generator_v3.txt`; judging uses `group_judge_v1.txt`. Their hashes are
pinned in `configs/group_online.json`. The default partner ports are 12025–12028.

## Group logs

Each case saves `group_rollout.json`, incremental `group_progress.json`,
`group_metrics.json` and its final case record. The latter contains every group's
source outcomes, raw partner reflection and its attempts, selected behavior,
code-assigned direction/endpoint, generation attempts, fixed contrast reply, three continuations,
s, KEEP/REMOVE result, anonymized judge request, scores and advantages.

The parent saves `group_metrics/batch_*.json` and `rubric_books/batch_*.json`, each
bound to its frozen source policy. Metrics include rubric group fraction,
zero advantage fraction among scored groups, verification pass rate among
resolved candidates, raw s rows and failure statuses, response token/character
lengths, initial partner state/correctness, reference outcome, and rescue/break
flags. Raw s rows can be used directly for the verification analysis figure.
Partner gold correctness is logged without adding outcome to the training reward.
Group metrics also include reference turn count and reflection failure fraction.
Each group stores stage durations and readiness/start/completion times, so overlap
can be audited. Group work durations overlap and must not be summed as batch wall
time. `batches/batch_*/timing.json` stores elapsed batch time through update/commit,
plus `projected_24_batches_seconds`. The inspection command prints that estimate.

## Earlier execution configurations

The following notes describe the earlier trajectory-bank execution configs.
To select `configs/online.json` explicitly, use
`SHERPA_FIVE_GPU_CONFIG=configs/online.json bash run.sh`.

This folder contains copied execution code, selected prompt/schema resources,
training data and an explicit online config. Source: `sj-max4-speed-dev` at
`0b3572b`, plus the local two-GPU, structured-Doctor and halting changes.
No training, Doctor server startup or remote API call was performed to prepare it.

Run from this folder:

```bash
bash run.sh                  # validate only
bash run.sh --launch         # start online training when ready
```

For a one-batch pilot use `bash run.sh --launch --max-batches 1`. The same command
without `--max-batches` resumes that output directory. CUDA device selection can
be overridden with `CUDA_VISIBLE_DEVICES=0,1` (two distinct physical devices).

## Files to edit

- `configs/online.json`: one flattened config, no inherited config chain.
- `prompts/doctor_structured_v1.txt`: the user's Doctor prompt, unchanged.
- `src/online_rl/clean/doctor_prompt.py`: per-turn input and hidden STATUS instructions.
- `src/online_rl/clean/halting.py`: at least two Agent turns, two valid STATUS
  objects with the same option, and all three current unresolved flags false.
  The Agent's last option does not veto termination.
- `src/online_rl/tree/offline_chunk_bank.py`: discovery selection (`CAP=48`,
  `select()`), chunks of eight, candidate creation/reuse and validation. Its name
  is historical; this config uses it online. It is copied unchanged.

Five cases are processed per update with GPU actors 0 and 1; GPU1 updates after
the batch barrier. Rubric bank transactions remain in case/chunk order. The
whole updated bank is frozen for that batch's scoring, then carried forward.
There are 120 cases, 24 batches and checkpoints every 30 cases. Reward remains
0.5 outcome + 0.5 rubric. Prompt evolution and evaluation are disabled.
Outputs are local under `outputs/train120_online_reference13/`.

`configs/online.json` now uses `rollout.budget="reference_spine"`, inherited by
the online variant configs. With `branching=4` and `rounds=4`, the rollout first
finishes one reference dialogue, then samples three alternative replies at each
visited reference turn and continues each alternative with one reply per turn.
Earlier reference replies and Doctor messages are shared. Each completed path
gets its own FINAL and reflection. Alternatives do not branch again.

Each online trajectory must complete at least two Agent turns before an early
stop is allowed (`rollout.halting.min_agent_turns=2`). For a reference with T
Agent turns, the case has `1 + 3*T` trajectories: 7 / 10 / 13 for T=2 / 3 / 4.
Each path independently follows the stop rules and four-turn limit.
If the reference ends early, alternatives may
still continue through turn four, but later reference turns are not created.
With no early stopping there are 34 unique Agent nodes, four sibling groups of
four replies at the reference turns, and single-response continuation groups.
The existing GRPO updater skips groups with fewer than two rewarded responses;
the outcome component still averages valid descendant outcomes for each node.

Set `rollout.budget="full_tree"` to use full branching. Omitted budget also
means full branching for legacy configs. Rollout budget is recorded on the
saved tree and in protocol/cache identities. Existing full-tree runs cannot
resume under the new budget. All online configs use a new output root ending
in `_reference13`. Their existing launch scripts select the new budget on the
next launch. To locate a variant's current output root, read its config instead
of the historical output paths in the revision notes below.
The existing rubric discovery cap of 48 cannot truncate these 13 trajectories.

## Shared runtime assets

Code, prompts and data are real copies; changing the source checkout will not
change them. Large model files, credentials and installed dependencies are
shared through `runtime/` links:

- `venv/` and `python`: existing Python environment in `Collaborative_agent-sherpa`.
- `ollama`: existing Ollama 0.12.5 executable.
- `huggingface/` and `ollama_models/`: existing Qwen3-8B/Gemma3-12B model caches.
- `credentials.env`: the existing private workspace `.env`; no secret is copied.

Keep those runtime assets available. This package is runnable in this workspace;
copying it to another machine requires replacing the runtime links or setting
`SHERPA_PYTHON`, `SHERPA_ENV_FILE`, `SHERPA_OLLAMA_BINARY`,
`SHERPA_OLLAMA_MODELS`, `HF_HOME` and `HF_HUB_CACHE`.
`requirements.txt` records the principal installed package versions; Torch needs
a CUDA build compatible with B200 (the reused environment has 2.7.1+cu128).

## Validation and scope

`run.sh` checks config/data/prompt hashes, model files, local CUDA/credentials,
and generator/chunk/scorer/Gate/protocol construction. It does not load model
weights or send API requests unless `--launch` is supplied.

```bash
runtime/python -m pytest -q tests
```

`PACKAGE_MANIFEST.json` records provenance and packaged file hashes.
`configs/baseline_online.json` is only used by tests to verify that the Doctor
revision leaves other scientific settings unchanged.

The code set is the conservative local import dependency closure of the runner,
not a rewritten trainer: some copied helper modules also contain legacy or
inactive paths used by the original multi-purpose modules. Historical run
configs, unused prompt resources and unrelated tools are excluded. Removing
inactive functions themselves is a separate refactor.

## Live monitoring

Open another terminal in this folder:

```bash
bash monitor.sh                 # refresh every 2 seconds; Ctrl+C exits only the monitor
bash monitor.sh --once          # one snapshot
bash monitor.sh --interval 5    # refresh every 5 seconds
bash monitor.sh --json          # one machine-readable snapshot
```

The monitor shows current batch / 24, committed cases / 120, the current
rollout/reflection, rubric-bank, scoring or GRPO stage, elapsed stage and batch
time, completed case files / 5, and per-GPU current case/activity durations.
An active case/operation timer does not mean every case in that stage is done.
The GPU pools can execute requests originating from other case coordinators.

It works with the currently running process using existing `live/` and batch
files. Old runs lack a global stage timestamp, so the inferred duration is
explicitly labeled as an estimate. On subsequent launches the parent runner
writes `progress.json` and `progress_events.jsonl`, with stage start time and
batch boundaries; the monitor uses these recorded timestamps. No restart is
needed just to use the monitor. Logs are read without modifying the training
state. The monitor does not call APIs or start GPU processes.

The monitor also reads each case's `calls.jsonl` incrementally. It reports
completed FINAL trajectories, generated Agent response nodes, Doctor discussion
calls, completed reflections and terminal depth counts per case. These are
original tree calls, excluding intervention reruns. Cache hits/duplicate records
do not increment the same completed call twice. A case is counted as rollout
complete only when its `bank_rollout.json` exists; partial cases remain visible.
Total trajectories are adaptive, so the monitor does not assume 48 is the total.

### Optional Doctor concurrency comparison

Default capacity is unchanged: Agent4 / Doctor1 per GPU. A separate fresh config
allows Doctor2 per GPU with Agent4 and the same five-case training batch:

```bash
SHERPA_FIVE_GPU_CONFIG=configs/online_doctor2.json bash run.sh             # checks only
SHERPA_FIVE_GPU_CONFIG=configs/online_doctor2.json bash run.sh --launch --max-batches 1
```

This was prepared but not launched. It uses a separate output directory and
must not be launched alongside the current run on the same GPUs. More Ollama
concurrency can improve Doctor queue waits but can also increase KV memory and
per-request latency; a measured batch comparison is needed before using it as
the default. GPU/model execution changes need not reproduce bitwise-identical
outputs even with the same request seeds. No model, prompt, reward, case order,
trajectory cap or update batch size is changed by this config.

### Independent Doctor replicas on each GPU

`configs/online_doctor_replicas2.json` places two single-request Ollama servers
on each GPU (four copies total), using ports 11955–11958. This differs from
Doctor2 on one server: each replica has its own model residency and request queue.
The local Doctor measurement during preparation was about 15.4 GiB per loaded
copy; KV/cache requirements can vary. Replica count consumes additional memory
and does not guarantee a proportional speedup.

```bash
SHERPA_FIVE_GPU_CONFIG=configs/online_doctor_replicas2.json bash run.sh
# After the current run has stopped, start a fresh one-batch comparison:
SHERPA_FIVE_GPU_CONFIG=configs/online_doctor_replicas2.json bash run.sh --launch --max-batches 1
bash monitor.sh --config configs/online_doctor_replicas2.json
```

This config has not been launched. The default and running process are unchanged.
All replicas on the updater GPU are unloaded before GRPO backward. Requests
carry the same per-call seeds and Doctor protocol; endpoint placement is runtime
metadata. Concurrent/GPU execution can still change numerical results. Case
order, five-case update batch, prompts, reward and random48 selection are unchanged.

### 케이스 완료 즉시 bank 처리

온라인 chunk8 실행은 모든 5개 tree가 끝날 때까지 기다리지 않습니다. 각 GPU coordinator에 한 케이스씩 rollout을 제출하고, 먼저 끝난 케이스부터 reflection이 포함된 rollout 결과를 이용해 rubric 생성·Gate·verification을 시작합니다. 다른 coordinator의 rollout은 계속 진행됩니다. 독립적인 rollout 스케줄러가 완료 즉시 다음 케이스를 제출하므로 bank API 대기 중에도 다른 케이스의 rollout이 진행됩니다. Bank 작업과 rollout은 같은 coordinator의 단일 worker 큐에서 순차 실행됩니다. 이미 제출된 다음 rollout 때문에 해당 coordinator의 bank 작업 시작이 늦어질 수 있습니다. 같은 coordinator의 mutable policy/client context를 공유하는 케이스는 동시에 실행하지 않습니다.

Bank 갱신은 케이스별로 직렬 처리하며 각 케이스는 직전 KEEP까지 반영한 최신 snapshot을 사용합니다. 실제 선택 순서는 `batches/batch_*/bank_case_order.json`에 먼저 저장되어 중단 후 재개 시 그대로 재현됩니다. 완료 순서는 실행마다 달라질 수 있어 기존 고정 케이스 순서와 다른 bank 내용이 나올 수 있습니다. 모든 5개 케이스의 bank 처리가 끝나면 전체 bank를 고정해 채점하고 GRPO를 한 번 업데이트합니다. 최대 48개 선택, chunk당 생성기 1회, chunk 내부 판정 로직은 유지합니다. Offline bank build와 frozen-bank 경로는 기존 스케줄을 사용합니다.

`monitor.sh`의 `Rollout + 완료 케이스 rubric 처리` 단계에서는 GPU별 진행 상태와 케이스별 rollout/reflection/bank 완료 수를 함께 확인할 수 있습니다. 이 단계 경과 시간은 겹쳐 실행된 전체 구간이며 rollout 시간과 bank 시간을 더한 값이 아닙니다. 변경은 다음 프로세스 실행부터 적용됩니다.

### Rubric-only 학습 (별도 실행)

`bash run_rubric_only.sh`는 실행 준비 확인만 하고, `bash run_rubric_only.sh --launch`는 rubric=1.0 / outcome=0.0으로 학습합니다. Doctor는 GPU당 2개이며 나머지 online/48개/chunk8/5-case update 설정은 `configs/online_doctor_replicas2.json`과 같습니다. 결과는 `outputs/train120_online_rubric_only`에 별도로 저장됩니다. `--launch --max-batches 1`로 한 배치만 실행할 수도 있습니다.

모니터는 `bash monitor.sh --config configs/online_rubric_only.json`을 사용합니다. Outcome은 최종 reward에 더하지 않지만 기존 성공/실패 trajectory 분류, reflection과 rubric 검증에서 정답 정보를 사용하는 로직은 유지합니다.

다른 서버로 옮길 때 `runtime/` 심볼릭 링크 대상(venv, HF cache, Ollama 바이너리/모델, credentials)이 그 서버에도 있어야 합니다. 대상이 다르면 서버의 로컬 경로로 링크를 연결하거나 기존 실행기가 지원하는 `SHERPA_PYTHON`, `HF_HOME`, `HF_HUB_CACHE`, `SHERPA_OLLAMA_BINARY`, `SHERPA_OLLAMA_MODELS`, `SHERPA_ENV_FILE` 환경 변수를 지정합니다. 동일하게 GPU 2장 환경을 전제로 합니다.

### Doctor prompt v2

`prompts/doctor_structured_v2.txt`는 선택을 가르는 쟁점을 현재 답변에서 해결하도록 하고, 다음 교환이 필요한 구체적 쟁점과 유용한 clarification이 있을 때만 질문하도록 합니다. STATUS 지시와 종료 조건, Agent 최소 2턴, 최대 4턴, branching, trajectory cap 48은 유지합니다. 턴 감소와 정답률 영향은 측정하지 않았습니다.

0.5/0.5 학습: `SHERPA_FIVE_GPU_CONFIG=configs/online_doctor_v2.json bash run.sh --launch`.
Rubric-only: `bash run_rubric_only.sh --launch` (이제 v2를 상속).
모니터링은 각 실행 config를 `monitor.sh --config`에 지정합니다. V2는 별도 출력 폴더와 Doctor/opening prompt 해시를 사용하며 기존 v1 설정과 결과는 보존합니다.

### Judge execution retry budget

모든 online 설정이 `judges.execution_retry_attempts=3`을 상속합니다. 최초 호출 포함 총 3회이며, 형식/근거 검증 실패 시 최대 2회 추가로 같은 요청을 보냅니다. Behavior 채점이 끝내 유효하지 않으면 해당 response×rubric 판정을 `execution_unresolved`로 남기고 숫자 점수를 부여하지 않습니다. Source applicability 및 witness audit은 기존의 검증 불가 제외 규칙을 따릅니다. 이 공유 설정은 chunk generator와 peer 재확인에도 사용됩니다. HTTP client 내부의 통신 재시도 및 Gate의 별도 근거 교정 루프와는 별개입니다. 실행 중인 프로세스의 이미 로드된 설정은 바뀌지 않습니다.

### 케이스별 bank 모니터링

`monitor.sh`는 기존 rollout 표 아래에 케이스별 chunk 생성 응답 완료 수, CREATE/REUSE/PEER 제안 수, Gate 통과/거절/미판정 수, applicability·peer·intervention 검사 완료 캐시 수, verification rerun 및 응답 채점 완료 호출 수를 표시합니다. 현재 coordinator의 cache operation으로 작업 이름과 경과 시간을 표시하고, `bank_online.json`이 완성되면 최종 KEEP/제외 및 bank rubric 증가량을 표시합니다. Chunk 생성 응답 완료는 bank commit 완료가 아니며, 캐시 집계는 실제 HTTP 요청 수나 재시도 수가 아닙니다. Applicability 수는 source·witness·채점의 공유 검사 기록을 포함합니다. 완료 기록을 읽기만 하므로 진행 중인 학습은 재시작할 필요가 없고 모니터만 다시 실행하면 됩니다.

### Gemini10 / OpenAI Luna20 실행

`bash run_openai20.sh --launch`는 기존 0.5/0.5와 Doctor v2를 유지하며 Gemini generation/matching은 OpenRouter, Luna Gate/applicability/behavior/intervention audit은 OpenAI 직접 Chat Completions를 사용합니다. `bash run_rubric_only_openai20.sh --launch`는 같은 구성의 rubric-only 버전입니다. 인자 없는 두 스크립트는 준비 검사만 합니다. 모니터 config는 각각 `configs/online_openai20.json`, `configs/online_rubric_only_openai20.json`입니다. Doctor 포트는 11965~11968이며 GPU당 2개입니다. 별도 output 폴더를 사용하므로 기존 실험의 설정/캐시와 섞이지 않습니다.

API worker는 20개, 독립 CREATE 후보의 chunk worker는 8개입니다. Chunk snapshot과 생성 호출/반영 순서는 유지하며 후보별 source/Gate/peer 검사 및 witness가 병렬화됩니다. 공유 슬롯은 run 전체 Gemini 최대 10개 / Luna 최대 20개로 제한하고 빈 슬롯이 없으면 기다립니다. Luna의 HTTP 통신 재시도는 슬롯을 반환한 뒤 대기합니다. Gemini는 생성 transaction 전체 동안 슬롯을 보유할 수 있습니다. 슬롯 한도는 실제 사용량 목표가 아니며 독립 작업이 적으면 실제 동시 요청 수는 더 적습니다. Doctor reflection은 로컬 Ollama와 기존 GPU 큐를 사용합니다. API 동시 슬롯 대기/실행은 live 기록의 `api.gemini.slot_wait`/`api.luna.request` 등으로 확인합니다.

OpenAI 키는 `OPENAI_API_KEY` 환경 변수, 명시한 `SHERPA_OPENAI_ENV_FILE`, 기존 비공개 `~/.config/sherpa/openai.env` 또는 launcher의 `SHERPA_ENV_FILE`에서 읽습니다. 기본 `runtime/credentials.env`에 `OPENAI_API_KEY`와 `OPENROUTER_API_KEY`가 있으면 별도 키 파일이 필요 없습니다. Preflight는 두 공급자의 키 존재를 확인하며 값은 출력하지 않습니다.

### Gate 검증 불가 처리와 전체 trajectory discovery

Blind source behavior judge가 총 3회 실패하여 `stage2=None` / `scoring_status=execution_unresolved`를 반환하면 Gate는 이를 숫자 판정이나 UNSCORABLE로 바꾸지 않고 `source_execution_unresolved` 사유와 원래 실패 기록으로 후보를 admission에서 제외합니다. `source_anchor.observed_level`과 `matches`는 null이며 학습 프로세스는 계속 진행합니다. 정상 source 0/1/2/UNSCORABLE의 기존 Gate 규칙은 유지합니다.

새 `online_openai20.json` 및 `online_rubric_only_openai20.json`은 `rubric.cumulative_bank.discovery_trajectory_cap=null`을 사용합니다. 유효한 reflection과 outcome이 있는 모든 trajectory를 정렬하여 8개씩 처리하고 random-48 selection을 하지 않습니다. 예를 들어 73개이면 10개 chunk입니다. 기존 v1/v2 설정은 기본 cap48을 유지합니다. Protocol identity와 새 output root의 `_all` 접미사로 조건을 구분하며 모니터도 전체 대상 수를 표시합니다. 이 두 설정은 case 완료 후 bank 처리를 유지합니다. Trajectory streaming은 아래 별도 설정으로 실행합니다. 실행 명령은 동일하게 `bash run_openai20.sh --launch`, 모니터는 `bash monitor.sh --config configs/online_openai20.json`입니다.


### Trajectory 준비 즉시 chunk8 처리

`bash run_trajectory_stream.sh --launch`는 outcome/rubric 0.5/0.5, `bash run_rubric_only_trajectory_stream.sh --launch`는 rubric=1 / outcome=0으로 실행합니다. 인자 없이는 준비 검사만 합니다. 모니터는 각각 `bash monitor.sh --config configs/online_trajectory_stream.json`, `bash monitor.sh --config configs/online_rubric_only_trajectory_stream.json`입니다. 새 출력 폴더와 Doctor 포트 11975~11978을 사용합니다. 기존 실행은 자동 변경되지 않습니다.

Terminal FINAL 이후 유효 reflection과 경로 노드가 준비된 trajectory를 케이스별 큐에 넣습니다. 8개가 준비되면 해당 케이스의 나머지 rollout을 기다리지 않고 생성 → source/Gate/peer 검사 → witness verification을 시작합니다. 케이스 끝의 1~7개도 처리하며 random-48 제한은 없습니다. 최대 Agent 4턴, 최소 2턴 종료 조건과 Doctor v2 prompt, GPU당 Doctor 2개, Gemini10/Luna20 공유 API 한도는 유지합니다.

두 coordinator는 각자 한 케이스씩 처리하며 같은 coordinator의 서로 다른 케이스는 순차 실행합니다. Bank chunk 작업은 별도 스레드에서 rollout/reflection과 겹쳐 진행합니다. Bank chunk transaction은 두 케이스 사이에서도 공유 lock으로 직렬화되어 최신 bank를 읽고 KEEP 결과를 한 번 반영합니다. 독립 CREATE 후보의 검사와 witness는 기존 chunk_workers=8 범위에서 병렬 처리하고 GPU 요청은 기존 공유 슬롯이 비면 실행하고, 없으면 기다립니다. 반영 순서는 준비 시점에 따라 달라질 수 있으므로 이전 case 완료 방식과 bank 결과가 동일하다고 가정하지 않습니다.

`stream_plan.json`은 최초의 8개 묶음을 저장합니다. 배치의 `trajectory_stream/receipts/`에는 chunk 결과를 먼저 저장하고 `state.json`에 반영합니다. 재개 시 완료된 chunk는 영수증에서 복원하며 중복 반영하지 않습니다. 이 bank는 배치 내 임시 상태이고, 모든 5개 케이스의 rollout/reflection/chunk verification이 완료된 뒤 전체 bank를 고정해 채점하고 기존처럼 GRPO를 한 번 업데이트합니다. 오류로 미완료된 배치는 policy/checkpoint에 commit하지 않습니다.

모니터의 `Reflection 준비`, `bank chunk 반영`, 현재 bank 작업 및 경과 시간으로 rollout 도중 시작된 bank 처리를 확인할 수 있습니다. 전체 대상 수는 rollout/reflection 완료 후 확정됩니다. 케이스별 bank 증가량은 해당 케이스 chunk가 추가한 unique rubric 수입니다. GPU/API를 호출하지 않는 테스트로 threshold8, tail 처리, 경로 노드 무결성, 두 케이스 transaction 순서, 재개/중복 방지, 전체 bank 채점 순서와 모니터 출력을 검증했습니다. 실제 학습 실행과 처리량 측정은 수행하지 않았습니다.

Trajectory streaming 모니터는 케이스별 `준비 누적 / 묶음 대기 / bank 처리 중 / bank 완료 trajectory·chunk` 표와 최근 6개 chunk 상태를 표시합니다. Bank 완료 수는 KEEP된 rubric 수가 아니라 재사용·생성·검증 처리가 완료된 입력 trajectory 수입니다. 공유 journal의 commit을 우선 읽어 progress 파일의 짧은 지연에도 완료 chunk를 처리 중으로 표시하지 않습니다. 학습을 재시작할 필요 없이 모니터만 재실행하면 적용됩니다.


### 다음 case rollout과 bank 처리 겹치기

`bash run_stream_overlap.sh --launch`는 0.5/0.5, `bash run_rubric_only_stream_overlap.sh --launch`는 rubric-only입니다. 새 출력 폴더와 Doctor 포트 11985~11988을 사용합니다. 모니터 config는 각각 `configs/online_stream_overlap.json`, `configs/online_rubric_only_stream_overlap.json`입니다. 현재 실행에는 자동 적용되지 않으며 인자 없는 스크립트는 환경 검사만 합니다.

Coordinator마다 원래 rollout/reflection 슬롯은 하나입니다. 완료 후 바로 슬롯을 반환하고 이전 케이스의 bank 처리는 뒤에서 계속 진행하므로 다음 케이스가 시작될 수 있습니다. 케이스별 RPC context와 Doctor proxy를 분리하고 동일한 frozen model, GPU actor와 용량 슬롯을 공유합니다. GPU model을 추가로 올리지 않습니다. 동시 case context는 현재 배치의 5개로 제한하며 모두 bank 처리까지 완료한 후 전체 bank 채점과 policy 업데이트를 진행합니다. Bank chunk transaction lock은 생성/Gate/verification/commit 전 구간에 유지되므로 chunk 사이의 직렬 대기는 남습니다.

`judges.execution_contract_repair=true`는 compact behavior 및 applicability 재시도에 validation 오류 원인을 전달합니다. 원래 public criterion, 입력과 schema는 유지하며 최대 3회입니다. Boolean·인용을 코드에서 임의 보정하거나 evidence 기준을 완화하지 않습니다. 실제 M/C 충돌은 기존 UNSCORABLE 규칙을 사용하고, 3회 기술 실패는 EXECUTION_UNRESOLVED입니다. Opt-in 요청 및 cache protocol이 달라져 새 run에서 사용해야 합니다. 실제 API 오류율과 처리 시간 개선은 아직 측정하지 않았습니다.


### Bank chunk 병렬 검증과 짧은 병합 lock

새 설정 `online_parallel_bank.json`과 `online_rubric_only_parallel_bank.json`은 `rubric.cumulative_bank.parallel_bank_processing=true`를 사용합니다. 각 케이스의 chunk는 순서대로 처리하지만 서로 다른 케이스의 생성·Gate·verification은 동시에 진행합니다. 시작할 때 bank snapshot을 고정하고, 모든 API 및 GPU 호출은 bank lock 밖에서 수행합니다.

검증이 끝나면 시작 snapshot 이후 추가된 rubric만 Gemini로 의미 중복 비교합니다. 비교 중 추가된 항목이 있으면 새로 추가된 부분만 다시 확인합니다. 반영 순간에는 짧은 lock으로 최신 bank에 병합하므로 다른 케이스의 추가 내용을 덮어쓰지 않습니다. 정확한 criterion hash가 같으면 기존 rubric에 source proof를 합칩니다. 표현이 다른 의미 중복은 먼저 등록된 rubric을 유지하고, 새 후보의 검증 기록은 chunk receipt에 보존합니다. 의미 중복 판정이 형식 검증에 끝내 실패하면 독립 검증을 통과한 후보는 별개로 유지합니다.

발견 snapshot과 검증 결과는 `trajectory_stream/drafts/`, 병합 내역은 `receipts/`에 저장합니다. 재개 시 이미 검증한 결과를 재사용하고 반영을 중복 수행하지 않습니다. 전체 bank 채점은 5개 케이스의 처리가 모두 끝난 뒤 진행하며 업데이트 주기는 그대로입니다. 다음 케이스 rollout과 이전 bank 처리의 겹치기, 모든 유효 trajectory의 chunk8 처리, Gemini 10개·Luna 20개 한도도 유지합니다. 모니터에는 추가 rubric 중복 비교 단계와 완료 수가 표시됩니다.

서버 1 (rubric/outcome 0.5/0.5):

```bash
cd /NHNHOME/shkim/SHERPA/sherpa-online
bash run_parallel_bank.sh --launch
# 별도 터미널
bash monitor.sh --config configs/online_parallel_bank.json
```

서버 2 (rubric-only 1/0):

```bash
cd /NHNHOME/shkim/SHERPA/sherpa-online
bash run_rubric_only_parallel_bank.sh --launch
# 별도 터미널
bash monitor.sh --config configs/online_rubric_only_parallel_bank.json
```

각 서버에 최신 코드가 있어야 합니다. 새 결과 폴더 `outputs/train120_online_parallel_bank` / `outputs/train120_online_rubric_only_parallel_bank`와 Doctor 포트 11995~11998을 사용합니다. 인자 없이는 준비 검사만 합니다. 기존 실행 중인 학습에는 적용되지 않으며 다음 새 실행부터 사용합니다. 동시 처리·중복 병합·재개·채점·모니터 회귀 테스트와 두 설정의 준비 검사를 통과했습니다. 실제 학습 실행과 처리량 측정은 수행하지 않았습니다.


### 확정 rubric의 bank 채점 미리 진행

`online_parallel_stream_score.json`과 `online_rubric_only_parallel_stream_score.json`은 기존 rubric 채점과 trajectory rollout을 겹칩니다. 각 trajectory가 완성될 때 기존 bank의 applicability(API Stage 1)를 캐시에 미리 채우고, rollout이 끝나면 현재까지 확정된 bank snapshot을 전체 응답 tree에 채점합니다. 이 채점은 남은 chunk의 rubric 생성·검증과 동시에 진행됩니다. 최종 bank가 만들어지면 미리 채점한 rubric 점수는 재사용하고, 이후 추가된 rubric만 채점해 최종 점수와 Agent target을 조립합니다. GRPO 업데이트는 최종 bank와 전체 점수의 확인 뒤 기존대로 한 번 수행합니다.

이 최적화는 모든 bank rubric을 평가하는 `scoring_scope=all`에서만 허용하며, rubric이 하나도 없는 첫 batch에는 사전 채점할 대상이 없습니다. Score snapshot과 점수 파일은 case 출력 폴더의 `stream_early_score.json`에 저장하고 policy/tree/bank hash를 검증해 재개 시 재사용합니다. 모니터의 케이스별 단계에서 `기존 rubric 사전 채점`을 볼 수 있습니다.

서버 1 (0.5/0.5, chunk worker 8):

```bash
cd /NHNHOME/shkim/SHERPA/sherpa-online
bash run_parallel_stream_score.sh --launch
# 별도 터미널
bash monitor.sh --config configs/online_parallel_stream_score.json
```

서버 2 (rubric-only 1/0):

```bash
cd /NHNHOME/shkim/SHERPA/sherpa-online
bash run_rubric_only_parallel_stream_score.sh --launch
# 별도 터미널
bash monitor.sh --config configs/online_rubric_only_parallel_stream_score.json
```

두 설정은 새 출력 폴더와 Doctor 포트 12015~12018을 사용하며 기존 프로세스에는 적용되지 않습니다. 현재 학습은 변경하지 않았고 실제 학습 처리량은 측정하지 않았습니다.

### Readable group monitor and rank rewards

The group monitor shows batch progress/time, completed-group learning signals, aligned case rows, Korean turn statuses, compact GPU activities, and the latest saved GRPO update. `bash monitor.sh --details` adds ranks, verification results, and internal operation names. Completed-batch time is fixed at its saved duration; the displayed total estimate is that measured batch duration times 24. Ratios use explicitly labeled completed denominators; groups still being prepared are not counted as finished.

The default v6 run uses `group_learning.reward_mode="rubric_rank_v1"` and `prompts/sherpa/group_judge_rank_v1.txt`. Previous v5 score-based artifacts remain separate. Ranking changes apply on the next launch; restarting the monitor refreshes its display. The rank judge sees only the shuffled replies, public rubric fields and visible context, with no source outcomes or private reflection.

## Shared-folder rollout: two servers with two GPUs each

Start on **server 2** (worker only):

```bash
cd /NHNHOME/shkim/SHERPA/sherpa-online
CUDA_VISIBLE_DEVICES=0,1 bash run_shared_worker.sh
```

Start on **server 1** (coordinator and only optimizer owner):

```bash
cd /NHNHOME/shkim/SHERPA/sherpa-online
CUDA_VISIBLE_DEVICES=0,1 bash run_shared.sh --launch
```

Monitor from either shared-folder terminal:

```bash
bash monitor.sh --config configs/group_shared.json
```

Both commands use exactly two GPUs local to their own server. They coordinate using atomic request/result files and a worker heartbeat under the shared run's `distributed/` directory. Each eight-case batch assigns four complete cases to each server. Each server has an independent local RPC/GPU pool and partner server PID namespace; no socket or multiprocessing manager is shared across machines. The worker cannot perform optimizer updates. It reads the current frozen checkpoint, verifies policy/protocol/case identities, finishes all four cases and publishes result-file hashes only after its actors become idle. The coordinator joins both sets of results before one GRPO update. Missing, failed, stale or mismatched remote work stops the batch without updating. Worker locks distinguish local process ownership; heartbeat records the server boot ID. Startup performs the launcher's normal asset/identity checks, but no new two-server training benchmark was run while implementing this path.

The run uses rank rewards, 120 cases, 15 batches and recovery checkpoints every batch. Regular milestone checkpoints occur at 32/64/96/120 cases. `run_b200.sh` is now an alias for the two-GPU coordinator; the earlier single-machine four-GPU command was incorrect for this topology.

Full checkpoint/resume snapshots upload asynchronously to the public repo `zzu819/SHERPA`, under `sherpa-ckpt/sherpa_main_shared_batch8/<run_id>/batch_XXXXXX/`. Snapshots contain the entire checkpoint directory (adapter, optimizer, CPU/CUDA RNG and newly saved Python/NumPy RNG states), committed run state, saved config/manifest, relevant prompts and available progress/timing files. Base model and dataset remain runtime assets. Detached uploader processes do not hold GPU/runner locks and preserve local pinned files until upload completes. Only explicitly allowed resume files are uploaded; .env/credentials/key files, symlinks, secret-bearing JSON fields, token/private-key patterns and actual credential values are blocked before each upload.

Retry failed uploads:

```bash
runtime/python scripts/online_rl/checkpoint_hub.py --config configs/group_shared.json --retry
```

Restore a complete snapshot on a fresh matching SHERPA checkout:

```bash
runtime/python scripts/online_rl/checkpoint_hub.py --restore sherpa-ckpt/sherpa_main_shared_batch8/RUN_ID/batch_000001
```

The restore command verifies file hashes, refuses to overwrite an existing run or different prompt artifacts, relocates the checkpoint pointer and prints the resume command. Both servers need the restored run's config and shared runtime assets. Server placement and upload destination are excluded from the central training config identity; reward/batch/sampling identities remain checked.
