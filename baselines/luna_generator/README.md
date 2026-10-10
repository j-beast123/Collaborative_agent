# Luna generator로 RaR / OnlineRubrics 새 실험

기존 `_v5` 학습·평가가 끝난 뒤 사용합니다. 기존 launcher, runtime.local.json,
method config, source hash, 결과와 checkpoint는 변경하지 않았습니다.
새 launcher를 추가했으며 실제 재학습·평가·API 요청은 실행하지 않았습니다.

- Generator: OpenRouter `openai/gpt-6-luna`, reasoning `none`.
- 초기 rubric 생성과 OnlineRubrics의 비교 기준 추출·중복 제거에 모두 적용합니다.
- Grader: 기존 direct OpenAI `gpt-5.6-luna`, reasoning medium 그대로.
- 기존 generator token budget/JSON format과 역할별 prompts는 유지합니다.
  Generator `temperature=0.7`은 Luna endpoint의 지원 목록에 없어 제거했습니다.
  실제 temperature는 provider 기본 동작이며 0.7과 같다고 가정하지 않습니다.
  이는 generator 전용 변경이며 Agent/Doctor의 sampling 설정은 바꾸지 않습니다.
- 모델 ID·reasoning·backend·새 코드 identity를 binding/manifest/cache key에 기록합니다.
- Provider parameter 지원을 요구하며 fallback하지 않습니다. 실제 provider 지원은
  아직 호출 검증하지 않았습니다. 다른 모델 반환/미지원/불완전 응답은 기존 bounded
  실패·재시도 방식으로 처리하며 0점으로 대체하지 않습니다.
- RaR는 rubric-only, OnlineRubrics는 frozen initial reference와 비교1쌍 그대로.
  SHERPA의 private need, verification, accuracy criterion이나 LR 옵션을 baseline에 추가하지 않습니다.
- 원래 train120 순서·seed·LoRA 초기화·5-case 업데이트·대화 조건·GRPO와 평가를 유지합니다.
  이전 학습의 final checkpoint에서 이어서 미세조정하지 않고 초기 adapter부터 시작합니다.
- Generator 동시성20, grader 동시성100/방법. 기존 두 phase 방식입니다.
  두 방법을 합친 전역100 제한으로 바꾼 실험이 아닙니다.

| 방법 | GPU | Doctor 포트 | 새 output/checkpoint/cache root |
|---|---|---|---|
| RaR | Doctor0,1 / Assistant2,3 |11741,11742|`outputs/rar_train120_d88c2e99b4a44_gen_gpt6luna_none_v1`|
| OnlineRubrics | Doctor4,5 / Assistant6,7 |11841,11842|`outputs/online_rubrics_train120_d88c2e99b4a44_gen_gpt6luna_none_v1`|

두 방법의 동일 입력 초기 rubric만 새 registry
`outputs/shared_initial_rubrics_d88c2e99b4a44_gen_gpt6luna_none_v1`에서 공유합니다.
기존 Gemini 초기 rubric registry나 기존 reward/reference/tree cache는 가져오지 않습니다.
초기 adapter identity registry는 같은 초기화를 비교하기 위해 기존 것을 유지합니다.

## 현재 서버 실행 명령

### NAS가 없는 새 서버

```bash
git clone --branch method-comparisons --single-branch https://github.com/j-beast123/Collaborative_agent.git
cd Collaborative_agent
cp baselines/runtime.example.json baselines/runtime.local.json
```

`runtime.local.json`의 `base_model_path`, `ollama_binary`, `ollama_models_dir`를
새 서버의 실제 로컬 경로로 바꿉니다. 아래 기존 서버용 명령의 Python 경로도
새 서버에서 의존성을 설치한 Python으로 대체합니다. API 키는 새 터미널에서 입력합니다.

Git에는 다음 파일이 포함되지 않으므로 별도로 이전하거나 준비해야 합니다.

- `data/splits/train120_excluding_calibration_seed20261008.jsonl`
- `data/interim/casimedicos/all_en_622.jsonl`
- Qwen3-8B 로컬 모델, Ollama 0.12.5와 고정 digest의 Doctor 모델
- 기존 실행을 보존할 경우 해당 `outputs/` 전체와 공유 rubric/초기화 registry

Train order는 `baselines/common/train120_order.json`으로 Git에 포함됩니다.
데이터를 새로 샘플링하지 말고 기존 파일을 복사해 preflight에서 해시를 검증합니다.
코드 push는 checkpoint·optimizer·rollout·cache를 이전하지 않습니다.
기존 run에는 절대 경로·모델 파일 metadata·패키지 버전이 기록되므로, 새 서버에
outputs를 복사하는 것만으로 즉시 재개된다고 보장할 수 없습니다. 이어서 학습하려면
새 서버 경로와 기존 binding을 대조하는 별도 이전 검증이 필요합니다.
이 README의 일반 train 명령은 새 서버에서 초기 adapter부터 시작하는 경우에 사용합니다.

### 기존 서버

현재 진행 중인 v5 평가에는 기존 명령을 계속 사용하세요. **새 명령은 두 기존
방법의 평가까지 마친 뒤** 실행합니다. 기존 GPU 소유 lock도 유지되어 진행 중인
동일 GPU 작업과 겹쳐 학습할 수 없습니다. 새 실험을 자동으로 시작하는 작업은 없습니다.

두 터미널/tmux 각각에서:

```bash
cd /convei_nas2/zzuhyun/Collaborative_agent
set +x
read -rsp 'OPENROUTER_API_KEY: ' OPENROUTER_API_KEY
printf '\n'
export OPENROUTER_API_KEY
read -rsp 'OPENAI_API_KEY: ' OPENAI_API_KEY
printf '\n'
export OPENAI_API_KEY
```

키는 파일이나 CLI 인수에 넣지 않습니다. 기존 `baselines/runtime.local.json`의
로컬 모델·Doctor·데이터 경로를 그대로 읽습니다.

RaR 터미널:

```bash
/home/common/miniconda3/bin/python -m baselines.luna_generator.launch --method rar --action preflight
/home/common/miniconda3/bin/python -m baselines.luna_generator.launch --method rar --action train --max-batches 1
/home/common/miniconda3/bin/python -m baselines.common.check_run --run outputs/rar_train120_d88c2e99b4a44_gen_gpt6luna_none_v1 --expected-cases 5
/home/common/miniconda3/bin/python -m baselines.luna_generator.launch --method rar --action train
```

OnlineRubrics 터미널:

```bash
/home/common/miniconda3/bin/python -m baselines.luna_generator.launch --method online_rubrics --action preflight
/home/common/miniconda3/bin/python -m baselines.luna_generator.launch --method online_rubrics --action train --max-batches 1
/home/common/miniconda3/bin/python -m baselines.common.check_run --run outputs/online_rubrics_train120_d88c2e99b4a44_gen_gpt6luna_none_v1 --expected-cases 5
/home/common/miniconda3/bin/python -m baselines.luna_generator.launch --method online_rubrics --action train
```

첫1배치는 정규 학습에 포함됩니다. 완료 후 마지막 train 명령으로 이어가면 됩니다.
중단 후에도 각각 같은 train 명령으로 새 실험의 checkpoint에서 재개합니다.
`--openai-grader` 옵션은 이 launcher에 필요하지 않습니다(이미 고정 적용).
예전 `repairs` 또는 provider `transition`을 새 run에 적용하지 마세요.

각 train120 완료 후 최종 checkpoint에서 test622를 한 번 평가:

```bash
/home/common/miniconda3/bin/python -m baselines.luna_generator.launch --method rar --action evaluate
/home/common/miniconda3/bin/python -m baselines.luna_generator.launch --method online_rubrics --action evaluate
```

결과는 각각 새 root의 `final_accuracy.json`, 비용은 `reward_api_attempts.jsonl`,
`rewards/judge_results.jsonl`, `costs.json`에 저장됩니다. 입력·모델·reasoning을 포함한
binding이 다르면 재개를 거부합니다. 이전 결과를 보고 LR/다른 설정을 조정하는
작업은 하지 않았으며 이번 차이는 generator 모델·reasoning과 실행 경로/포트입니다.

## 첫 배치 HTTP 404 복구

오류 본문의 `No endpoints found that can handle the requested parameters`와
`failed_routing_step: Filter by Parameters`를 확인했습니다. [해당 모델의 endpoint 목록](https://openrouter.ai/api/v1/models/openai/gpt-6-luna/endpoints)은
`temperature`를 지원 파라미터로 등록하지 않습니다. `require_parameters=true`를
유지하고 generator 요청에서 temperature만 제거합니다. 모델·reasoning none은 유지합니다.

이전 코드로 404가 난 실행에는 아래 복구를 한 번 적용한 뒤 재개합니다.
기본 명령은 읽기 전용 계획이며 `--apply`가 있어야 저장 기록을 갱신합니다.
수정 전 설정 백업과 변경 기록을 해당 run의 `repairs/luna_temperature_v1/`에 남깁니다.
이 복구는 완료된 업데이트 0회 및 성공한 reward cache가 없는 경우에만 허용됩니다.
GPU와 runner lock을 확보하고 초기 checkpoint·optimizer·저장된 rollout/cache의
바이트 보존을 확인합니다. 부분 복구가 중단되어도 같은 명령으로 완료할 수 있습니다.

```bash
cd /convei_nas2/zzuhyun/Collaborative_agent
/home/common/miniconda3/bin/python -m baselines.repairs.luna_temperature --method rar --apply &&
/home/common/miniconda3/bin/python -m baselines.luna_generator.launch --method rar --action train
```

OnlineRubrics도 이전 코드로 이미 시작했다면 같은 절차를 사용합니다.
아직 시작하지 않았거나 이미 수정된 설정이면 복구 명령은 변경 없이 종료합니다.

```bash
/home/common/miniconda3/bin/python -m baselines.repairs.luna_temperature --method online_rubrics --apply &&
/home/common/miniconda3/bin/python -m baselines.luna_generator.launch --method online_rubrics --action train
```

## 구현 검증

CPU fake API 및 기존 baseline 회귀 테스트 **46개 통과**. 모델·reasoning 요청,
grader 유지, 역할별 생성 경로, 성공 cache 재사용, 잘못된 모델·미지원 파라미터
거부, bounded retry, 키 비기록, 기존 GRPO 연결과 재개 binding을 확인했습니다.
기존 baseline의 scientific source hash와 parallel execution 파일 해시는 변경 전과 같습니다.
현재 runtime으로 두 방법의 setup을 읽기 전용 실행해 오류가 없고 train/order/test
검증이 통과함을 확인했습니다. 실제 GPU resource preflight와 provider 요청은 실행하지 않았습니다.

OpenRouter 요청 형식은 공식 [reasoning 설정](https://github.com/OpenRouterTeam/docs/blob/main/guides/best-practices/reasoning-tokens.mdx)과
[provider parameter 지원 요구](https://github.com/OpenRouterTeam/docs/blob/main/guides/routing/provider-selection.mdx)를 기준으로 합니다.
`require_parameters`는 요청 라우팅 조건이며 특정 provider의 실제 동작을 독립적으로
증명하지는 않습니다. 반환 model ID를 검사하고, 보고된 reasoning token이 양수면
중단합니다. 실제 `openai/gpt-6-luna` 제공 여부와 해당 모델의 `none` 지원은 첫 사용자 실행에서 확인됩니다.
