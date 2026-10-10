# Base / Outcome-only GRPO / RaR / OnlineRubrics / SHERPA 비교

현재 RaR/OnlineRubrics의 실행·재개는 [OpenAI 병렬 채점 안내](parallel_api/README.md)를 따른다. Gemini 생성은 OpenRouter 최대20개, GPT-5.6 Luna 채점은 OpenAI 최대100개/방법이다. 기존 launcher와 과거 점검 JSON은 재현 및 변경 이력 확인용으로 보존한다.

`runtime.local.json`, 데이터 원문, API 키, outputs/checkpoint/cache는 Git에 포함하지 않는다. 새 checkout에서는 `runtime.example.json`을 `runtime.local.json`으로 복사하고 로컬 경로를 맞춘 뒤, hash가 고정된 train120과 test622 파일을 별도로 준비한다.

계정 전용 Ollama 설치, 고정 Doctor 모델 복사/다운로드 차이, 디스크 용량과 전체 digest 검증 명령은 [Doctor 준비 안내](DOCTOR_SETUP.md)에 있다. 현재 registry의 새 digest `88c2e99b4a44…` 사용이 승인되었으며 새 run/cache로 분리했다.
사용자가 제공한 초기 adapter parameter SHA256 `9018ce3bbacd319049576e8ace88afb4777621a8769919bf46676fca8a85b17c`를 `runtime.local.json`과 예시에 기대값으로 연결했다. 실제 시작 시 이 값과 다르면 rollout 전에 중단한다. [대화·평가 조건 대조 기록](common/user_identity_audit.json)에 일치 항목과 아직 미검증인 base revision·평가 protocol hash·artifact 파일 해시를 구분했다.
현재 경로 확인 상태와 키 숨김 입력, 실제 설정 파일을 명시한 단계별 명령은 [실행 안내](RUN_COMMANDS.md)를 따른다. 현재 사용 중인 Ollama0.12.5와 Doctor 저장소 경로는 `runtime.example.json`에 기록했다. 초기 발견 기록은 당시 상태를 나타내며 현재 경로 존재 여부는 preflight로 확인한다.
RaR와 OnlineRubrics는 **rubric reward만 사용**한다. 두 방법 모두 `w_outcome=0`, `w_rubric=1`이며 `r05_o05_p0_fixed`의 0.5/0.5는 공통 조건이 아니다. Outcome-only는 기존 `node_credit(1,0)`를 유지한다. CollabLLM은 제외하고 기존 코드를 보존한다. SHERPA는 다른 서버에서 실행한다.

`src/`, `scripts/`, `configs/`는 수정하지 않았다. GPU 학습과 평가의 실행은 사용자가 담당한다. 최신 CPU 검증은 54개가 통과했으며 [초기 검증 기록](common/validation_v5.json)은 과거 24개 검사 결과다. 아래 명령은 사용자가 실행한다. [4-GPU 배치·동기화·reference 구현과 전체 실행 절차](four_gpu/README.md)에 자세한 설명이 있다.

## 공통 설정과 로컬 파일

공통 값은 기존 `configs/online_rl/tree_v21_five_gpu_shared5_random10_promptv7.json`, outcome-only 상속 설정과 실제 코드에서 가져온다. 외부 SHERPA manifest 추출은 필수 조건이 아니다. [해석한 공통 값](common/resolved_common.json)과 방법별 [RaR 설정](rar/resolved_config.json), [OnlineRubrics 설정](online_rubrics/resolved_config.json), [Outcome-only 설정](outcome_only/resolved_config.json)을 보관한다. Launcher는 실행할 때 원래 설정에서 다시 해석하고 코드·공통 값·방법 설정을 run에 기록한다.

| 항목 | 설정 |
|---|---|
| Base | 로컬 Qwen3-8B, snapshot `b968826d9c46dd6066d109eabc6255188de91218`, BF16 |
| LoRA | 원래 `seeded_lora`, init seed=20261006, r16, alpha32, rsLoRA, dropout0, q/k/v/o projections; 별도 init adapter 없음 |
| Train | 제공된 train120 원본 행 순서, 1 epoch, batch5 × 24; 배치마다 최대1회 update |
| Seed | schedule0, 기존 case/action seed 산출 방식 |
| Agent | 원래 `precommit_v1`, private assessment+visible reply를 같은 GRPO action에 포함; visible max400, temperature.7, top_p.9, 모델 generation config의 top_k20 |
| Doctor | Gemma3-12B Q8_0, Ollama0.12.5, 고정 digest·prompt·decoding·hidden status·FINAL 처리 |
| Tree | G4, Agent 최대4턴, 원래 최소2턴 및 halt/repetition 종료 조건 |
| GRPO | lr1e-5, beta.04, clip.2, grad clip1, mean-centering, 기존 pi_old/pi_ref·그룹 제외 규칙 |
| 저장·평가 | 30case마다 저장; 학습 중 test 평가 없음; 최종120 checkpoint로 test622 1회, seed0+행 인덱스, Doctor FINAL 정확도 |

로컬 모델 경로는 다음과 같다. `runtime.local.json`과 예시 파일에 연결했으며 다시 다운로드하지 않는다.

```text
/convei_nas2/huggingface_cache/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218
```

JSON 해시와 5개 shard의 HF blob 이름·크기·mtime를 기록했다. 16GB 전체 shard를 새로 SHA256 계산한 것은 아니다. 각 방법은 같은 로컬 snapshot을 읽는다. 첫 실제 시작에서 초기 LoRA tensor hash를 측정하여 `outputs/common_initialization_v4/`의 공통 기록과 비교한다. 각 run의 `initial_policy.json`에도 저장하고 재개 시 확인한다. 초기 seed/LoRA/model 설정이 같고도 tensor hash가 다르면 rollout 전에 중단한다. 다른 서버의 실제 tensor hash를 검증했다고 주장하지 않는다.

현재 설치된 Transformers4.51.1은 원래 코드의 `dtype=bfloat16` 대신 `torch_dtype=bfloat16`을 받으므로 비교 launcher 내부에서 키워드만 호환 처리한다. 동일 BF16·원래 SharedPolicy·초기화 함수를 사용하며 원본 코드를 수정하거나 패키지를 임의 업그레이드하지 않는다. 설치 패키지 버전은 공통 기록과 재개 identity에 포함한다.

## 데이터

사용자 원본 `data/train120_excluding_calibration_seed20261008.jsonl`은 보존하고 `data/splits/train120_excluding_calibration_seed20261008.jsonl`에 바이트 그대로 복사했다. 별도 order 파일은 원본 행의 ID 순서다. 재샘플링·재정렬하지 않았다. [데이터 감사](common/data_audit.json)에서 행 수·순서·calibration 제외·해시가 검증됐다.

| 파일 | SHA256 |
|---|---|
| train120 | `3d34929094925c822062d03839b49de8f6fb43afba71fb573de409471bb5b8ab` |
| `baselines/common/train120_order.json` | `edf23262be5430d10a0d02dde3da80710bf9a91bd941d83234e9c98372752c41` |
| `data/interim/casimedicos/all_en_622.jsonl` | `e42077d006c4e8227564f99b06050b66e7df9989f9b2b17f5fc5d9dd083feb56` |

CasiMedicos 영어 원본 train434 + validation63 + test125 전체가 외부 test622다. [출처와 다운로드 검증](common/test_download_audit.json)을 보관한다. 평가 분모는 622이며 FINAL 파싱 실패도 분모에 포함한다. Base의 `BASE_raw_no_precommit` 결과는 현재 precommit 조건과 다르므로 그대로 합치지 않는다.

## 방법별 보상

RaR-Explicit (adapted)는 case·현재까지 공개 history·train reference answer로 초기 rubric을 생성한다. Current 후보·partner reflection·미래 대화 결과를 제공하지 않는다. 같은 history의 후보4개에 같은 rubric을 적용하고 항목별 binary score의 중요도 가중 평균을 후보 응답 자체에 직접 부여한다. 초기 항목에는 별도 최소 개수 제한이 없으며, 빈 rubric은 보상 분모가 없어 거부한다. 최대20개이고 Essential1, Important.7, Optional.3, Pitfall.9다. Pitfall은 오류 회피를 긍정형으로 표현한다. 근거는 [RaR v2 §4.4](https://arxiv.org/html/2507.17746v2)다.

OnlineRubrics (adapted)는 동일 입력에서 RaR와 같은 초기 rubric을 사용한다. 고정 seed로 current sibling4개 중1개를 선택하고, 같은 history에 대해 frozen initial policy의 reference1개를 별도로 생성한다. **Current–initial-reference 1쌍**에서 새 기준을 추출하고 초기 기준과 중복을 제거한 뒤 모든 후보4개를 동일한 최종 rubric으로 채점한다. Current sibling끼리만 비교하지 않는다. 근거는 [OnlineRubrics v1 Algorithm1, §4.1](https://arxiv.org/html/2510.07284v1)이다.

OnlineRubrics의 새 기준은 signed integer weight(-10…10, 0 제외)를 사용한다. 초기 RaR 항목은 보존한다. 점수는 `sum(weight × binary)/sum(positive weights)`이므로 음수가 될 수 있다. 공통 초기 rubric, 비교1쌍, physician–assistant 다중 턴, 기존 tree/GRPO 연결은 adaptation이며 원 논문 전체 학습 재현이라고 표기하지 않는다. Clipping/z-score/outcome 혼합은 추가하지 않는다.

```text
RaR:            R(n) = Q_initial(n)
OnlineRubrics:  R(n) = Q_initial_and_deduplicated_pair_criteria(n)
A(n) = R(n) - mean(valid sibling rewards)  # 원래 GRPO
```

**원래 그룹 유효성 규칙은 유지한다.** 원 코드 `node_credit(0,1)`도 유효 FINAL descendant가 하나도 없는 node는 reward=None으로 제외한다. FINAL 정오 값(0/1)은 두 방법의 보상 수치에 기여하지 않는다. Outcome은 tree 진단값으로만 남는다. Rubric은 descendant leaf 평균으로 바꾸지 않는다. Judge 실패는 0점이나 조용한 node 제외로 처리하지 않고 update 전에 중단한다.

Rubric judge에는 task를 포함한 공개 history·평가 대상 응답·rubric만 제공한다. Private assessment·미래 대화·최종 outcome·train answer 필드·다른 후보/reference는 제외한다. Train reference answer는 초기 rubric 생성에만 사용하므로 rubric 문구 자체가 정답에서 유도될 수 있다. Partner reflection·verification·prompt optimization은 모두 꺼져 있다.

기존 OpenRouter launcher의 generator는 `google/gemini-3.8-flash`, grader는 `openai/gpt-5.6-luna`다. 새 병렬 launcher는 grader를 동일 계열 `gpt-5.6-luna`의 OpenAI 직접 호출로 전환하며, 요청 파라미터와 전환 checkpoint 이력은 [별도 안내](parallel_api/README.md)에 기록한다. 다음 설명은 기존 OpenRouter backend에 해당한다. 기존 metadata의 scorer canonical ID도 응답 ID로 허용한다. Generation temperature/max_tokens/reasoning은 기존 설정을 재사용하며 criterion JSON schema로만 바꾼다. Binary grader는 temperature0/max_tokens4096/JSON이다. Provider fallback을 금지하고 request ID·response model·usage를 기록한다. API alias가 불변 revision을 보장하는 것은 아니며, 모델 접근 가능 여부는 실제 요청 전에는 검증하지 않았다. 자동 모델 대체는 없다.

## GPU·cache·재개

| 방법 | 물리 GPU | Doctor ports | 출력·checkpoint·cache root |
|---|---|---|---|
| Outcome-only | 기존 공유5-GPU 구성 0–4 | 11631–11635 | `outputs/outcome_only_train120_d88c2e99b4a44_v5` |
| RaR | Doctor0,1 / Assistant2,3, updater3 | 11731–11732 | `outputs/rar_train120_d88c2e99b4a44_v5` |
| OnlineRubrics | Doctor4,5 / Assistant6,7, updater7 | 11831–11832 | `outputs/online_rubrics_train120_d88c2e99b4a44_v5` |

RaR/OnlineRubrics는 동시에 실행 가능하고 학습·최종 평가 모두 각자의4장 안에서 수행한다. 두 Assistant worker가 같은 policy의 case5개를 3+2로 처리한 뒤 barrier에서 원래 GRPO update를 최대1회 한다. Outcome-only는 GPU가 겹치므로 별도 시간에 실행한다.

OnlineRubrics reference는 idle barrier에서 GPU6의 current LoRA를 CPU에 백업하고 보존한 초기 LoRA로 잠시 교체한다. 같은 base에서 원래 precommit 응답을 생성한 뒤 current LoRA·gradient flag를 복원하고 hash를 확인한다. 초기 reference는 업데이트하지 않으며 추가 base 복사본·추가 GPU·reference 뒤 Doctor 대화가 없다. Reference cache는 실제 초기 adapter hash·입력·모델·생성 설정·seed를 포함한다.

각 Assistant GPU에는 전체 BF16 Qwen 한 개가 올라간다. KV cache·activation·optimizer가 추가되므로 24GB에서 생성 batch4와 backward까지 가능한지는 실제 점검 전에는 보장하지 않는다. 메모리 부족 시 자동 양자화·batch 축소를 하지 않는다. Worker 수 감소로 dynamic batch·padding·요청 순서가 달라지므로 같은 seed라도 bitwise 동일한 대화·가중치는 보장하지 않는다.

초기 rubric만 `outputs/shared_initial_rubrics_d88c2e99b4a44_v5/`의 파일 잠금이 있는 공통 저장소에서 재사용하고 각 방법 cache로 출처와 함께 복사한다. Judge·추출·reference cache는 방법별로 분리한다. 생성 비용은 실제 호출한 방법에 기록하고 공유 artifact import 수를 별도 집계한다. 공통 저장소를 실행 중 지우지 않는다.

재개 시 코드·설정·모델·패키지·데이터/order·보상·judge 설정·초기 adapter를 대조한다. 이전 혼합 보상 v3 checkpoint를 새 Doctor rubric-only v5 run으로 이어 쓰지 않는다. 원래 pending batch·optimizer·recovery checkpoint 복구를 사용한다. 평가 재개는 저장된 case 결과를 재생성하지 않는다.

## 필요한 환경과 실행 명령

현재 준비할 항목은 세 가지다. `baselines/runtime.local.json`의 `ollama_binary`를 Ollama0.12.5 실행 파일로, `ollama_models_dir`를 고정 Gemma3 모델 저장소로 설정한다. Doctor tag는 `gemma3:12b-it-q8_0`, manifest SHA256은 `88c2e99b4a44ef8a4c855315927ae99f6ea77a06f4ec24ad2d08929f818f5f00`다. `OPENROUTER_API_KEY`는 각 실행 터미널의 환경 변수에만 설정한다. 비밀값을 JSON/명령 로그에 넣지 않는다. 데이터·Qwen 다운로드·외부 manifest·사전 초기 adapter hash 입력은 필요 없다.

```bash
cd /convei_nas2/zzuhyun/Collaborative_agent
python -m baselines.common.data --out baselines/common/data_audit.json
CUDA_VISIBLE_DEVICES='' python -m unittest baselines.common.test_methods baselines.common.test_four_gpu -v
python -m baselines.rar.launch --action preflight --cpu-only
python -m baselines.online_rubrics.launch --action preflight --cpu-only
```

아래는 사용자 실행용 명령이다. GPU 사전 점검은 모델을 생성하지 않는다. 작은 점검은 본 학습 첫5case를 실제 학습하므로 다음 train 명령이 남은115개를 이어서 진행한다. `--max-batches 1`은 호출할 때마다 최대 추가1배치다.

```bash
# 터미널 A: RaR, GPU0–3
python -m baselines.rar.launch --action preflight
python -m baselines.rar.launch --action train --max-batches 1
python -m baselines.common.check_run --run outputs/rar_train120_d88c2e99b4a44_v5 --expected-cases 5
python -m baselines.rar.launch --action train
python -m baselines.rar.launch --action evaluate
```

```bash
# 터미널 B: OnlineRubrics, GPU4–7
python -m baselines.online_rubrics.launch --action preflight
python -m baselines.online_rubrics.launch --action train --max-batches 1
python -m baselines.common.check_run --run outputs/online_rubrics_train120_d88c2e99b4a44_v5 --expected-cases 5
python -m baselines.online_rubrics.launch --action train
python -m baselines.online_rubrics.launch --action evaluate
```

Outcome-only의 동일 조건 결과가 있다면 재학습할 필요 없다. 기존 결과의 read-only 대조 명령은 `python -m baselines.outcome_only.launch --action audit-existing --existing-run /실제/기존/run`이다. 새로 학습할 때는 두 방법과 동시에 실행하지 않는다.

```bash
python -m baselines.outcome_only.launch --action preflight --cpu-only
python -m baselines.outcome_only.launch --action preflight
python -m baselines.outcome_only.launch --action train --max-batches 1
python -m baselines.common.check_run --run outputs/outcome_only_train120_d88c2e99b4a44_v5 --expected-cases 5
python -m baselines.outcome_only.launch --action train
python -m baselines.outcome_only.launch --action evaluate
```

학습 중단 후에는 해당 방법의 같은 `--action train` 명령으로 재개한다. 평가 중단 후에는 같은 `--action evaluate`를 사용한다. 120개 완료 이전에는 평가가 거부되며 완료 후 마지막 checkpoint만 평가한다. Test 결과로 가중치·checkpoint를 선택하지 않는다. 결과는 방법별 `final_accuracy.json`, `evaluation/checkpoint_000120/summary.json`이다.

`resolved_launch_config.json`, `method_binding.json`, `initial_policy.json`, `method_reports/`와 case의 `agent_targets`에 실제 설정·보상을 보관한다. Run manifest의 `agent_reward=(1,0)`는 원래 rollout 진단용이다. **Optimizer 입력은 배치 hook이 교체한 `agent_targets`; 실제 가중치는 `comparison_method.actual_reward_weights=(0,1)`**이며 CPU 통합 테스트로 연결을 검증한다.

```bash
python -m baselines.common.costs --runs outputs/rar_train120_d88c2e99b4a44_v5 --out outputs/rar_train120_d88c2e99b4a44_v5/costs.json
python -m baselines.common.costs --runs outputs/online_rubrics_train120_d88c2e99b4a44_v5 --out outputs/online_rubrics_train120_d88c2e99b4a44_v5/costs.json
# 강제 종료 후 해당 방법이 소유한 Doctor만 정리할 때
python -m baselines.rar.launch --action stop-doctors
python -m baselines.online_rubrics.launch --action stop-doctors
python -m baselines.outcome_only.launch --action stop-doctors
```

Reward API 시도·실패·usage·비용과 reference private/visible 토큰·generation/swap 시간은 원래 rollout/train/eval 비용과 분리한다. 미제공 API 비용은 0원으로 채우지 않는다. GPU-hours·peak VRAM은 실제 실행에서 별도 측정해야 한다. 실패 요청의 API 중복 시도는 가능하지만 성공 cache는 재사용한다.
