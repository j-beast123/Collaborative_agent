# RaR / OnlineRubrics 동시 실행: Doctor2 + Assistant2

계정 전용 Ollama 설치, 고정 Doctor 모델 복사/다운로드 차이, 디스크 용량과 전체 digest 검증 명령은 [Doctor 준비 안내](../DOCTOR_SETUP.md)에 있다. 현재 registry의 새 digest `88c2e99b4a44…` 사용이 승인되었으며 새 run/cache로 분리했다.
현재 경로 확인 상태와 키 숨김 입력, 실제 설정 파일을 명시한 단계별 명령은 [실행 안내](../RUN_COMMANDS.md)를 따른다. 현재 설치 경로는 `runtime.example.json`에 기록했다. 현재 사용하는 OpenAI 병렬 채점 launcher는 [최신 실행 안내](../parallel_api/README.md)를 따른다.
실제 GPU 학습·추론·test 평가는 실행하지 않았다. CPU 합성 테스트만 수행했다. 아래 명령은 사용자가 준비 완료 후 실행한다. 공통 값은 기존 config와 코드에서 재사용하며 외부 SHERPA manifest는 필수 조건이 아니다.

## 물리 GPU와 실행 경로

| 방법 | Doctor | Assistant rollout replicas | GRPO updater | Initial reference | Doctor ports | 출력 root |
|---|---|---|---|---|---|---|
| RaR | 0,1 | 2,3 | 3 | 없음 | 11731,11732 | `outputs/rar_train120_d88c2e99b4a44_v5` |
| OnlineRubrics | 4,5 | 6,7 | 7 | 6에서 초기 adapter 교체 | 11831,11832 | `outputs/online_rubrics_train120_d88c2e99b4a44_v5` |

각 Assistant GPU에는 전체 Qwen 모델 복제본이 하나씩 있다. 두 GPU에 모델을 쪼개거나 DDP로 optimizer를 두 개 만드는 구성은 아니다. Python parent는 Assistant 두 장만 볼 수 있게 `CUDA_VISIBLE_DEVICES`를 각각 `2,3` / `6,7`로 설정하고 updater를 논리 GPU1에 올린다. 다른 Assistant worker는 논리 GPU0(물리2/6)를 사용한다. Ollama 프로세스는 각각 물리 Doctor GPU 하나만 볼 수 있다. 학습과 최종 평가는 같은 배치를 사용한다.

방법별 `method.json`이 GPU 역할·포트·출력 경로를 고정한다. `runtime.local.json.training_physical_gpus`의 예전 5장 값은 RaR/OnlineRubrics에서는 사용하지 않는다. 두 방법은 서로 다른 네 장에만 launcher lock을 잡으므로 동시에 실행할 수 있다. Outcome-only의 기존 5-GPU 실행은 이들과 겹치므로 함께 실행하지 않는다.

이전 outcome 혼합 실험과 섞지 않도록 새 Doctor rubric-only v5 경로를 사용한다. 완료한 실험이나 5-GPU checkpoint를 새 topology로 조용히 재개하지 않는다. 출력 root 아래 checkpoints, pending batch, tree cache, reward cache, reference cache, 비용 로그가 분리된다.

## 5-case 배치와 동기화

이전 runner에서 `batch_slots`의 batch 크기는 `workers × lanes`에 묶여 있었다. 5→2 worker만 바꾸면 2-case 배치가 되므로 그대로 사용하지 않았다. 새 `four_gpu.batch_slots`는 **항상 원래 순서의 연속 5 case**를 선택한다. 첫 worker가 3개, 둘째 worker가 2개를 순차 처리한다. 두 worker는 원래 ready-work pool을 공유하여 실제 Agent/Doctor 요청을 두 actor에 배분한다.

1. 두 Assistant에 같은 committed checkpoint를 로드하고 hash를 대조한다.
2. 같은 frozen current policy로 case5개의 tree를 생성한다. 각 worker는 lane1이다.
3. 두 coordinator future가 모두 완료되고 pool lease가 0일 때만 다음 단계로 간다. Case cursor 순서로 records를 정렬한다.
4. 후보별 direct rubric reward만 부여한다. Outcome 보상 가중치는 0이다. OnlineRubrics reference는 이 대기 구간에서만 허용한다.
5. 원래 `update_batch`/GRPO로 모든 기존 sibling group을 모아 **최대 한 번** update한다. CPU에 보존한 reference 초기 adapter는 업데이트하지 않는다.
6. 원래 checkpoint/recovery receipt와 optimizer state를 저장한 뒤 다음 5개로 진행한다.

Train120 순서·case/action seed·G=4·max4턴·Doctor halt·private+visible action·배치5·24배치·저장 간격30case·GRPO 설정을 유지한다. Batch의 invalid/zero-advantage group 제외 규칙도 그대로다. 따라서 24는 최대 optimizer step 수이며 실제 step 수는 생성 결과에 따라 적을 수 있다. Assistant worker 수를 batch 크기나 epoch 수로 바꾸지 않는다.

공통 값은 기존 protocol config와 실제 코드에서 기록한다. `batch_cases`, seed, 데이터/order hash, 학습 횟수, rollout·optimizer는 보존한다. 보상은 두 방법 모두 outcome/rubric=0/1이다. Topology·코드·공통 값·보상 설정은 재개 identity에 고정한다. 원래 코드의 유효 descendant가 없는 node 제외 및 zero-advantage group 제외 규칙은 유지한다.

## Frozen initial reference를 Assistant GPU 안에서 처리

OnlineRubrics 물리 GPU6은 optimizer가 없는 rollout replica다. 초기화 직후 **현재 checkpoint를 로드하기 전에** 초기 adapter 텐서를 CPU에 clone하고 같은 실행의 parent에서 측정한 초기 hash와 대조한다. Parent는 `outputs/common_initialization_v4/`의 방법 간 공통 기록과도 비교하며 불일치하면 첫 rollout 전에 중단한다. 전체 base 가중치는 LoRA 학습에서 바뀌지 않는다.

기존 case5 rollout이 모두 끝난 뒤 reference cache miss가 있으면:

- Actor를 unavailable로 표시하고 current adapter를 CPU에 백업한다.
- 동일 base에 보존한 초기 adapter를 복사하고 모든 parameter의 gradient를 끈다.
- 고정된 입력·생성 설정·seed로 원래 precommit private+visible 응답 1개를 만든다. Actor의 native Agent 함수만 호출하며 Doctor/FINAL/미래 대화는 호출하지 않는다.
- 성공·실패에 관계없이 current adapter, gradient flags, model mode와 generation 함수를 복원한다. Current hash를 검증한 뒤 actor를 복구한다.

이 구현은 **별도 GPU 또는 두 번째 BF16 base 복사본 없이** frozen initial policy를 사용한다. Current sibling끼리만 비교하거나 현재 policy를 reference로 사용하는 대체가 아니다. 초기 adapter 복사본은 CPU에 불변으로 보관한다. 동일 입력의 reference는 method 전용 cache에서 재사용한다. Swap/복원 시간이 reference generation wall time에 포함된다.

GPU6은 reference 생성 중 current rollout을 처리하지 않는다. 이는 batch barrier 안에서 이루어지므로 원래 5-case frozen policy 의미와 optimizer state를 바꾸지 않는다. 실패 시 해당 batch의 update 전에 중단한다. 재개 시 성공 cache를 재사용한다.

## 메모리와 결과 차이

Qwen3-8B BF16은 가중치만 대략 16 GB 규모이고, KV cache·생성 logits·LoRA·activation·optimizer state가 추가된다. Assistant GPU마다 전체 모델 하나가 필요하다. Reference는 두 번째 base를 추가하지 않으므로 메모리 요구가 크게 늘지는 않지만, **24 GB에서 rollout batch4와 GRPO backward까지 충분한지는 GPU 점검 전에는 보장할 수 없다.** CPU 메모리에는 초기/current LoRA 백업이 추가된다. Doctor는 별도 GPU에 있으므로 Assistant와 VRAM을 경쟁하지 않는다.

메모리가 부족하면 오류로 중단한다. 양자화, context/응답 길이 축소, Agent batch 크기 축소, 다른 모델, offload를 자동으로 적용하지 않는다. 그런 변경이 필요하면 SHERPA 공통 조건과 비교 의미를 다시 확인해야 한다. 원래 gradient checkpointing은 유지한다. 아래 1배치 점검으로 생성·reference·backward·checkpoint의 peak memory를 실제 서버에서 확인한다. GPU 메모리는 두 방법을 동시에 점검하는 동안 `nvidia-smi` 등으로 별도 기록한다.

Worker 수의 영향은 다음과 같다.

| 항목 | 이전 → 변경 | 의미 |
|---|---|---|
| Assistant actor / Doctor endpoint | 각5 → 각2 | 같은 모델 호출의 총 동시 처리량이 줄어듦 |
| Agent 요청 capacity | 전체20 → 전체8, GPU당4 유지 | Dynamic batch 조합·padding이 달라질 수 있음 |
| Doctor 동시 처리 | 전체5 → 전체2, 서버당1 유지 | 요청 도착 순서·서버 배분·cache hit 패턴이 달라질 수 있음 |
| Case coordinator | 5 → 2, 5개를 3+2 처리 | Batch 크기는 그대로, 느린 case의 대기 시간이 달라짐 |
| 평가 in-flight case | 25 → 10 | 같은 test 행과 seed, 같은 최종 정확도 분모; 스케줄은 달라짐 |
| Reference | 별도 모델/GPU → idle replica adapter 교체 | 같은 초기 policy, swap 비용 추가, 추가 GPU 없음 |

Agent는 원래 per-row RNG sampling을 유지하고 Doctor 요청에도 원래 seed를 넣는다. 하지만 batch 조합, CUDA 수치 연산, Ollama 실행 스케줄, 외부 generator/judge의 비결정성으로 **대화·reward·최종 가중치의 bitwise 일치는 보장하지 않는다**. 동일한 실험 조건/알고리즘이라는 의미이며 결과 값까지 같다는 의미가 아니다. 추가 API 동시 요청으로 rate limit/retry 비용과 wall time도 달라질 수 있다.

## 동시에 같은 초기 rubric을 요청하는 경우

방법별 cache는 유지하면서 초기 rubric만 `outputs/shared_initial_rubrics_d88c2e99b4a44_v5`의 공통 immutable artifact registry를 사용한다. 입력·generator·revision·prompt·params가 같은 key에는 파일 잠금을 잡고 한 프로세스만 생성한다. 다른 프로세스는 그 결과를 자기 cache로 복사하고 provenance를 기록한다. 서로 다른 history는 서로 다른 key다. 채점 cache, OnlineRubrics extraction/dedup/reference cache는 공유하지 않는다.

생성 API 비용은 실제 호출한 방법에 기록한다. 따라서 어떤 방법이 먼저 key를 생성했는지에 따라 방법별 관측 생성 비용 배분은 달라질 수 있다. 비교 보고서에는 import 수와 공통 artifact 생성 총비용을 함께 제시하고, 공유분을 두 방법에 중복 청구하지 않는다. 저장 중 실패한 외부 요청의 exactly-once는 보장하지 않으며 성공 artifact를 재사용한다.

## 준비와 명령

먼저 [공통 환경 설정](../README.md)을 확인한다. `baselines/runtime.local.json`에는 이미 다운로드된 Qwen3-8B의 로컬 snapshot과 train120 경로를 연결했다. 모델 재다운로드나 외부 SHERPA manifest 추출은 필요 없다. 추가로 준비할 항목은 **Ollama 0.12.5 실행 파일 경로, 고정 Doctor 모델 저장소, OPENROUTER_API_KEY**다. Generator/scorer 모델 ID는 기존 config에서 가져온다.

Train120 원본을 실행 경로에 바이트 그대로 복사했고 고정 SHA256·행 순서가 일치한다. Test622와 순서 파일도 검증됐다. 원본은 `data/train120_excluding_calibration_seed20261008.jsonl`에 보존했다.

```bash
cd /convei_nas2/zzuhyun/Collaborative_agent
# baselines/runtime.local.json의 ollama_binary, ollama_models_dir를 설치 경로로 설정
# OPENROUTER_API_KEY는 RUN_COMMANDS.md의 숨김 입력으로 각 터미널에서 설정
python -m baselines.common.data --out baselines/common/data_audit.json
CUDA_VISIBLE_DEVICES='' python -m unittest baselines.common.test_methods baselines.common.test_four_gpu -v
```

사전 점검(모델 생성·학습 없음):

```bash
python -m baselines.rar.launch --action preflight --cpu-only
python -m baselines.online_rubrics.launch --action preflight --cpu-only
python -m baselines.rar.launch --action preflight
python -m baselines.online_rubrics.launch --action preflight
```

작은 학습 점검은 **본 실행의 첫 5개 case, 1배치**다. 실제 update가 최대1회 일어나며 본 학습에서 이 batch를 다시 학습하지 않는다. 데이터/tree 크기나 보상 설정을 축소한 별도 pilot이 아니다. 아래 두 명령은 서로 다른 터미널에서 동시에 실행해도 된다. 사용자가 실행한 이후 메모리·로그를 확인한다.

```bash
# 터미널 A: 물리 GPU 0–3
python -m baselines.rar.launch --action train --max-batches 1
python -m baselines.common.check_run --run outputs/rar_train120_d88c2e99b4a44_v5 --expected-cases 5

# 터미널 B: 물리 GPU 4–7
python -m baselines.online_rubrics.launch --action train --max-batches 1
python -m baselines.common.check_run --run outputs/online_rubrics_train120_d88c2e99b4a44_v5 --expected-cases 5
```

`--max-batches 1`은 **호출당 최대 추가1배치**다. 이미 완료한 상태에서 같은 작은 점검 명령을 반복하면 다음 5개를 진행한다. 중단 재개를 포함해 현재 위치는 `run_state.json`으로 확인한다. `check_run`은 5-case 배치·순서·policy hash·rubric-only 보상·update 횟수와 test 미실행을 검증한다. GPU memory와 reference 비용은 각각 runtime 로그/`costs.json`에서 확인한다.

전체 train120 학습 또는 중단 후 재개(두 터미널에서 동시 실행 가능):

```bash
# 터미널 A
python -m baselines.rar.launch --action train

# 터미널 B
python -m baselines.online_rubrics.launch --action train
```

각 방법이 120개를 완료한 뒤, 마지막 checkpoint의 test622 평가(두 터미널에서 동시 실행 가능):

```bash
# 터미널 A
python -m baselines.rar.launch --action evaluate

# 터미널 B
python -m baselines.online_rubrics.launch --action evaluate
```

평가 중단 후 같은 `evaluate` 명령으로 재개한다. 기존 case 결과를 재생성하지 않는다. 두 방법 모두 자기 네 장만 사용하고 reference 생성·rubric API 없이 원래 Doctor FINAL 평가를 수행한다. Test 결과로 가중치나 checkpoint를 선택하지 않는다. 결과는 각각 `final_accuracy.json`, `evaluation/checkpoint_000120/summary.json`이다.

Launcher 정상 종료 시 해당 방법의 Doctor 서버만 종료한다. 강제 종료 후 정리가 필요하면 해당 명령을 실행한다.

```bash
python -m baselines.rar.launch --action stop-doctors
python -m baselines.online_rubrics.launch --action stop-doctors
```

학습 전후 명세·코드·모델·보상 설정·경로를 바꾸면 재개가 거부된다. 새 topology를 위해 이전 output을 삭제하거나 checkpoint를 덮어쓰지 않는다. CollabLLM 코드는 보존하며 실행하지 않는다. SHERPA도 이 서버에서 실행하지 않는다.

## 2026-10-09: Generator ID correction and preserved first batch

사용자 확인에 따라 두 방법의 generator 요청 ID를
`google/gemini-3.8-flash-20260902`에서 `google/gemini-3.8-flash`로
변경했다. 반환 ID도 provider prefix를 포함하여 정확히 검증하며 자동
fallback은 사용하지 않는다. 이는 요청 모델의 명시적 변경이다.
이 routing ID만으로 provider 내부 가중치 revision의 불변성을 입증하지는 않는다.
변경 기록: `baselines/common/generator_change.json`.

두 출력의 첫 배치는 모두 5개 tree 생성 이후 첫 rubric 요청의 model-ID
검사에서 실패했으며 optimizer update는 없었다. 해당 tree, behavior logprobs,
초기 adapter/optimizer, 실패 요청 비용을 보존하고 모델 설정 및 재개 해시만
갱신했다. 원본 메타데이터와 변경 전후 기록은 각 출력의
`repairs/generator_id_20261009/journal.json`에 있다. 38개 파일씩 변경되지
않았음을 해시로 검사했다. 모델 ID가 cache key에 들어가므로 이전 generator의
성공한 rubric을 새 generator의 결과로 가져오지 않는다.

수정 후 CPU 테스트 28개와 두 run의 parent/checkpoint/worker 재개 설정 검사는
통과했다. 실제 provider 요청, GPU 학습 및 test 평가는 실행하지 않았다.
각 터미널에서 API 키가 설정된 기존 Python 환경을 사용해 첫 배치를 이어간다.
수동으로 tree나 checkpoint를 지우지 않는다.

```bash
cd /convei_nas2/zzuhyun/Collaborative_agent
# 터미널 1: RaR, Doctor 0/1 + Assistant 2/3, 포트 11731/11732
python -m baselines.rar.launch --config baselines/rar/method.json --runtime baselines/runtime.local.json --action train --max-batches 1
python -m baselines.common.check_run --run outputs/rar_train120_d88c2e99b4a44_v5 --expected-cases 5
```

```bash
cd /convei_nas2/zzuhyun/Collaborative_agent
# 터미널 2: OnlineRubrics, Doctor 4/5 + Assistant 6/7, 포트 11831/11832
python -m baselines.online_rubrics.launch --config baselines/online_rubrics/method.json --runtime baselines/runtime.local.json --action train --max-batches 1
python -m baselines.common.check_run --run outputs/online_rubrics_train120_d88c2e99b4a44_v5 --expected-cases 5
```

두 점검이 통과하면 각 train 명령에서 `--max-batches 1`을 빼고 실행한다.
완료된 첫 배치 이후부터 같은 출력으로 이어간다. 최종 평가는 위 문서의
`--action evaluate` 명령을 사용한다.

## 2026-10-09: 초기 rubric 최소 개수 조건 제거

사용자 요청에 따라 초기 rubric prompt의 `7 to 20`을 `up to 20`으로 바꾸고,
validator의 최소 7개 조건을 제거했다. 1–6개도 동일한 중요도 가중 평균으로
채점한다. 최대 20개, 항목 구조·category·binary 채점 검증은 유지한다.
빈 초기 rubric은 보상 분모가 없으므로 명시적으로 거부한다. OnlineRubrics의
추가 기준은 기존처럼 빈 목록이 가능하다.

이번 변경은 generator/grade 모델, GRPO, 데이터 순서, Doctor, 초기 adapter,
대화 조건을 바꾸지 않는다. 두 run 모두 optimizer update 이전이었고,
첫 배치 5개 tree씩과 성공한 grade/reference cache를 보존했다. 기존에 성공한
initial rubric은 완화된 조건에서도 유효하므로 재사용한다. 기존 cache 항목을
유지하고, 새 prompt identity에 연결한 항목에 `reuse_provenance`로 이전 key와
prompt hash를 기록했다. 새 prompt로 다시 생성한 결과로 간주하지 않는다.
RaR 5건, OnlineRubrics 10건의 기존 성공 initial rubric을 재사용할 수 있다.
직전 6개 응답은 실패 당시 성공 cache에 저장되지 않았으므로 해당 요청은
재개 시 다시 수행된다. 이전 호출 비용은 그대로 남아 있다.

원본 메타데이터 및 cache backup, 변경 내역과 보존 hash:
`outputs/repairs/remove_rubric_minimum_20261009/journal.json`.
적용 코드: `baselines/repairs/remove_minimum.py` (이미 적용했으므로 재실행 불필요).
103개 기존 파일의 byte hash를 보존했다. CPU 테스트 32개 및 두 run의
parent/checkpoint/worker 재개 검사 통과. 실제 API/GPU 실행은 하지 않았다.
위와 동일한 `--action train --max-batches 1` 명령으로 재개한다.
