# GPT-5.6 Luna on OpenAI, parallel rubric APIs

채점 모델은 **gpt-5.6-luna**로 유지하며, 호출 경로를 OpenRouter에서 OpenAI로
전환한다. Gemini 루브릭 생성·추출·중복 제거는 기존 OpenRouter를 사용한다.
실행 중인 프로세스와 저장된 실험 파일에는 아직 적용하지 않았다.

| 역할 | 제공자 / 모델 | 방법당 동시 요청 상한 |
|---|---|---|
| Generator | OpenRouter / `google/gemini-3.8-flash` | 20 |
| Rubric judge | OpenAI / `gpt-5.6-luna` | 100 |

두 방법을 함께 실행하면 judge 요청은 최대 200개다. 실제 요청 수는 case의 후보
수와 cache hit에 따라 줄어든다. 계정의 RPM·TPM은 별도로 적용된다. 429는 기존
bounded retry(최초 요청 + 최대 2회 재시도)를 사용하고 실패하면 업데이트하지 않는다.
실제 OpenAI 인증·유료 채점·100개 부하 검증은 실행하지 않았다.

## 유지하는 학습 절차

기존 5-case frozen-policy rollout barrier가 완료된 후 보상을 계산한다. Case는
원래 순서로 처리한다. Case 안에서는 sibling group별 초기 루브릭을 병렬 생성한다.
OnlineRubrics의 frozen reference 생성은 기존 GPU에서 원래 group 순서로 직렬
수행한다. Adapter swap은 겹치지 않는다. 이후 그룹별 추출→중복 제거 의존성을
유지하면서 API 요청을 병렬화하고, 최종 루브릭으로 후보를 병렬 채점한다.
모든 case 채점이 성공한 후 기존 GRPO가 배치당 최대 한 번 업데이트한다.

데이터·seed·배치·Doctor·프롬프트·직접 candidate 보상·루브릭 가중치·종료 조건과
최종 test 평가는 유지한다. 동일 API 응답에 대해서 reward와 advantage가 동일하다.
원격 생성의 응답은 요청 순서에 따라 달라질 수 있으므로 재현성을 보장하지 않는다.

Cross-method initial rubric 공유는 per-key lock과 atomic registry를 사용한다.
JSONL audit 쓰기는 동기화한다. Generator/reference 입력이 같으면 기존 cache를
재사용한다. Judge의 provider와 params는 cache key에 포함되므로 **기존 OpenRouter
채점은 OpenAI 채점으로 재사용하지 않는다**. 기존 cache 항목과 비용 로그는 보존한다.

## OpenAI 요청과 제공자 변경 기록

- Endpoint: `https://api.openai.com/v1/chat/completions`
- Model/request/revision 선언: `gpt-5.6-luna`; 자동 모델 대체 없음.
- 인증: 실행 터미널의 `OPENAI_API_KEY`만 읽음. 파일에 키 저장 안 함.
- Params: `max_completion_tokens=4096`, `reasoning_effort=medium`,
  `response_format={"type":"json_object"}`. Temperature는 지정하지 않는다.
- 기존 GPT-5.6 Luna 로그에 reasoning 토큰이 사용됐으므로 reasoning을 끄지 않았다.
  GPT-5.6 기본 effort인 medium을 명시했다. 이전 OpenRouter의 실제 effort와
  temperature 처리는 독립적으로 확인되지 않았다. 이전 `max_tokens`는 OpenAI의
  `max_completion_tokens`로 바뀐다. 이 차이를 profile과 전환 journal에 기록한다.
- 동일 모델 계열이어도 제공자 내부 weight snapshot/기본값이 정확히 같다고
  입증할 수는 없다. 현재 허용한 응답 model ID는 `gpt-5.6-luna`뿐이다.
  다른 ID가 오면 중단하고 audit의 실제 ID를 확인한다.
- 요청 ID, 반환 model, usage, 시도 수, latency, 오류를 기록한다. OpenAI가 비용을
  반환하지 않으면 달러 비용은 `null`이며 0으로 간주하지 않는다.

확인한 공식 문서:
[GPT-5.6 Luna](https://developers.openai.com/api/docs/models/gpt-5.6-luna),
[GPT-5.6 migration](https://developers.openai.com/api/docs/guides/gpt-5.6).

## 기존 학습에서 전환

**먼저 기존 launcher를 정상 종료하고 worker/Doctor 정리가 끝날 때까지 기다린다.**
가능하면 `batch_committed`를 확인한 직후 종료한다. 이 준비 작업에서는 프로세스를
중단하지 않았다. 새 코드는 이미 실행 중인 Python에 자동 반영되지 않는다.

`transition --apply`는 해당 방법의 GPU 4장과 runner lock을 모두 확보해야 동작한다.
학습이 진행 중이면 거부한다. 완료된 batch와 optimizer/adapter 파일을 보존하고
현재 checkpoint에서 이어가도록 provider 관련 메타데이터만 변경한다. 미완료
batch의 동일 frozen-policy tree는 재사용하며 그 batch의 채점은 OpenAI로 계산한다.
이미 업데이트한 batch는 다시 학습하지 않는다.

전환 전에 모든 변경 파일의 원문과 보존 파일 hash를
`<output>/repairs/openai_gpt56_luna_v1/journal.json`에 저장한다. 전환 중 중단되면
같은 명령을 다시 실행한다. 변경 전후 버전 외의 수정이 발견되면 거부한다.
완료된 이전 batch는 OpenRouter로 채점됐다는 사실과 전환 case/batch 지점을 기록한다.
따라서 전체 run이 처음부터 OpenAI였다고 보고하면 안 된다.

Optimizer 업데이트 직후 저장/commit 사이에 중단된 경우, uncommitted checkpoint나
`.pending` checkpoint가 있으면 전환을 거부한다. 먼저 **원래 launcher**로 그 배치의
recovery를 완료해야 한다. Checkpoint나 binding 파일을 직접 지워 우회하지 않는다.
데이터·Doctor·학습 횟수 등 다른 설정이 동시에 바뀌어도 전환을 거부한다.

Extension source hash는 profile에 고정되고 `api_execution_history.jsonl`에도 기록한다.
전환 후 extension 코드를 수정하면 재개 시 불일치로 중단한다.

## 실행 환경과 안전한 키 입력 (실행할 터미널마다)

기존 학습과 같은 Python 환경을 활성화한다. 이미 설정된 OPENROUTER 키는 유지한다.

```bash
cd /convei_nas2/zzuhyun/Collaborative_agent
set +x
read -r -s -p 'OPENAI_API_KEY: ' OPENAI_API_KEY
printf '\n'
export OPENAI_API_KEY
# OPENROUTER_API_KEY가 없는 터미널에서만 실행:
if [ -z "${OPENROUTER_API_KEY:-}" ]; then
  read -r -s -p 'OPENROUTER_API_KEY: ' OPENROUTER_API_KEY
  printf '\n'
  export OPENROUTER_API_KEY
fi
# 인증/모델 조회만 수행. Completion이나 동시 요청 부하 검증은 하지 않음.
python -m baselines.parallel_api.probe
```

## 터미널 1: RaR

Config: `baselines/parallel_api/rar_openai.json`.
Output/checkpoints/cache: `outputs/rar_train120_d88c2e99b4a44_v5/`.
Doctor GPU 0/1, ports 11731/11732; Assistant GPU 2/3.

```bash
# 기존 RaR 프로세스 종료 후, 최초 전환 한 번
python -m baselines.parallel_api.transition --method rar --runtime baselines/runtime.local.json --apply
# Preflight
python -m baselines.parallel_api.launch --method rar --openai-grader --api-workers 20 --grader-workers 100 --config baselines/parallel_api/rar_openai.json --runtime baselines/runtime.local.json --action preflight
# 현재 checkpoint에서 추가 1배치 점검 (처음부터 재학습하지 않음)
python -m baselines.parallel_api.launch --method rar --openai-grader --api-workers 20 --grader-workers 100 --config baselines/parallel_api/rar_openai.json --runtime baselines/runtime.local.json --action train --max-batches 1
python -m baselines.common.check_run --run outputs/rar_train120_d88c2e99b4a44_v5
# 나머지 전체 학습 / 이후 중단 후 재개: 같은 명령
python -m baselines.parallel_api.launch --method rar --openai-grader --api-workers 20 --grader-workers 100 --config baselines/parallel_api/rar_openai.json --runtime baselines/runtime.local.json --action train
# train120 완료 후 최종 checkpoint의 test622 평가 1회
python -m baselines.parallel_api.launch --method rar --openai-grader --api-workers 20 --grader-workers 100 --config baselines/parallel_api/rar_openai.json --runtime baselines/runtime.local.json --action evaluate
```

## 터미널 2: OnlineRubrics

Config: `baselines/parallel_api/online_rubrics_openai.json`.
Output/checkpoints/cache: `outputs/online_rubrics_train120_d88c2e99b4a44_v5/`.
Doctor GPU 4/5, ports 11831/11832; Assistant GPU 6/7, frozen reference도 GPU 6.

```bash
# 기존 OnlineRubrics 프로세스 종료 후, 최초 전환 한 번
python -m baselines.parallel_api.transition --method online_rubrics --runtime baselines/runtime.local.json --apply
# Preflight
python -m baselines.parallel_api.launch --method online_rubrics --openai-grader --api-workers 20 --grader-workers 100 --config baselines/parallel_api/online_rubrics_openai.json --runtime baselines/runtime.local.json --action preflight
# 추가 1배치 점검
python -m baselines.parallel_api.launch --method online_rubrics --openai-grader --api-workers 20 --grader-workers 100 --config baselines/parallel_api/online_rubrics_openai.json --runtime baselines/runtime.local.json --action train --max-batches 1
python -m baselines.common.check_run --run outputs/online_rubrics_train120_d88c2e99b4a44_v5
# 전체 학습 / 중단 후 재개
python -m baselines.parallel_api.launch --method online_rubrics --openai-grader --api-workers 20 --grader-workers 100 --config baselines/parallel_api/online_rubrics_openai.json --runtime baselines/runtime.local.json --action train
# 최종 test622 평가 1회
python -m baselines.parallel_api.launch --method online_rubrics --openai-grader --api-workers 20 --grader-workers 100 --config baselines/parallel_api/online_rubrics_openai.json --runtime baselines/runtime.local.json --action evaluate
```

각 명령은 앞 단계가 성공한 뒤 실행한다. Preflight 실패 상태에서 train을 실행하지 않는다.
학습 명령은 각 터미널에서 동시에 실행할 수 있다. 최종 정확도는 기존 Doctor FINAL
기준이며 rubric judge가 test 정답률을 판정하는 방식으로 바뀌지 않는다.

## 제공자별 비용 확인

```bash
python -m baselines.parallel_api.costs --run outputs/rar_train120_d88c2e99b4a44_v5
python -m baselines.parallel_api.costs --run outputs/online_rubrics_train120_d88c2e99b4a44_v5
```

`reward_api_attempts.jsonl`에는 기존 OpenRouter와 새 OpenAI 이력을 모두 보존하고
위 명령이 제공자별로 분리한다. 재시도도 실제 요청 수에 포함한다. Provider billing과
기록된 token usage를 함께 확인해야 최종 비용을 확정할 수 있다.

## CPU 검증

병렬/직렬 reward·report·advantage 및 기존 GRPO 연결, reference 순서/seed,
20/100 concurrency 상한, JSONL 동시 쓰기, 공유 rubric 단일 생성, OpenAI 키/URL/요청
형식, 429 retry, 실패 시 batch update 방지, checkpoint·optimizer·cache 보존,
전환 중 중단/재개, live lock 차단, unrelated config 및 미완료 checkpoint 거부를 검사한다.
실제 두 설정에 대해서도 provider 이외 설정과 dialogue protocol이 동일함을 확인했다.

```bash
CUDA_VISIBLE_DEVICES='' python -m unittest baselines.common.test_methods baselines.common.test_four_gpu baselines.common.test_doctor_assets baselines.common.test_generator_repair baselines.common.test_rubric_count baselines.parallel_api.test_parallel baselines.parallel_api.test_transition -v
```

## 2026-10-10: Gemini JSON 포장 응답 수정

기존 RaR 4번째 배치에서 Gemini가 실제 JSON을 `cat << 'EOF' ... EOF`로 감싸
반환해 원래 `json.loads`가 실패했다. 새 launcher는 응답 전체가 정확한 JSON
코드 펜스 또는 literal heredoc 형식인 경우 바깥 포장만 제거한다. Shell 명령은
실행하지 않고 JSON 내용도 수정하지 않는다. 기존 criteria/grade schema 검증은
그대로 적용한다. Schema 오류는 통과시키거나 재시도하지 않는다.

포장을 제거해도 JSON 문법이 잘못되면 동일한 model/prompt/params로 최대 3개
응답까지 시도한다. 각 응답의 원문, format_attempt, validation_status,
json_normalization을 judge_results.jsonl에 기록한다. HTTP 재시도도 겹치면 하나의
논리 요청에서 최대 9회의 실제 요청이 발생할 수 있고, 각각 기존 API audit에
기록된다. 실패 결과는 cache하지 않으며 소진 시 batch 업데이트 없이 중단한다.

OpenAI 전환을 이 수정 전에 완료한 RaR는 다음 재개 전에 아래 명령으로 코드 hash
변경만 별도 기록해야 한다. 원래 제공자 전환 명령을 다시 실행하지 않는다.
이미 수정된 hash로 시작한 OnlineRubrics에는 이 명령이 변경 없이 종료한다.
GPU/runner lock을 획득해야 하므로 해당 학습 종료 후 실행한다.

```bash
python -m baselines.repairs.json_response --method rar --apply
# 이후 위 새 launcher의 preflight / train 명령으로 재개
```

Repair journal은 `<output>/repairs/json_response_v1/journal.json`에 저장한다.
모델·params·프롬프트·학습 진도·optimizer·유효 cache는 유지한다.
