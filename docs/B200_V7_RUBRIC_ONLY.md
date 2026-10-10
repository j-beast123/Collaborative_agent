# B200: V7 + rubric-only + case 병렬 + Luna

이 버전은 `train120-max2-fixed-p0-v22`의 V7 구현
`ca0ad31f78347b23cad20b638bffeb65dd888a62`에서 generator/schema,
공개 history 기반 applicability, M/C gate와 compact scorer, 개입 지시 audit를 가져옵니다.
성능 향상은 이 서버에서 검증하지 않았습니다. 테스트는 CPU fake API 기반 계약 검증입니다.

설정: `configs/experiments/train120_max4_v7_rubric_only_b200.json`
런처: `scripts/online_rl/launch_b200_v7.sh`
이전 설정/런처는 보존합니다. 이전 checkpoint/cache를 새 실험에 이어 붙이지 않습니다.

## 적용 범위

- **고정 medical-accuracy criterion 없음**. KEEP partner rubric의 유효 0/1/2 점수를
  0/0.5/1로 변환하여 동일 가중 평균합니다. Outcome weight=0, rubric weight=1.
  유효 KEEP rubric이 없거나 모두 N/A면 학습에서 제외하며 outcome fallback은 없습니다.
  생성된 partner criterion 자체의 의학적 내용까지 제거한다는 뜻은 아닙니다.
- **Case 간 누적 rubric bank/retrieval 없음**. 기존 case 내부의 독립 후보 생성 →
  중복 제거(consolidate) → canonical source 고정만 유지합니다. 다른 case의 rubric을
  검색·상속하지 않습니다. 후보는 ID순으로 정렬하고 원문 representative를 보존합니다.
  M(최소 수행)과 C(완료 조건)가 모두 동등할 때만 합치도록 지시합니다.
- 후보 생성 묶음 크기 `rubric.candidate_batch_size=8`, case workers=5.
  5개 case가 동일한 frozen policy를 사용한 후 기존 group 전체에 최대 한 번 update합니다.
  API 최대 in-flight는 **전체 역할/재시도 합산 프로세스당 50**입니다.
- Generator/consolidate/gate/applicability/scorer/instruction auditor 모두
  **OpenAI 직접 API `gpt-6-luna`**입니다. Generator/consolidate reasoning=`none`;
  나머지는 기존 judge 방식대로 provider 기본 reasoning을 사용합니다.
  지원되지 않는 요청이나 다른 모델 응답은 중단하며 자동 대체하지 않습니다.
  로컬 Assistant=Qwen3-8B, Doctor/terminal reflection=Gemma BF16는 그대로입니다.
- Train120 파일/순서/seed, LoRA 초기화/hash, max4/branching4,
  5-case 경계/GRPO/학습 횟수/최종 test622 규칙은 이전 B200 설정과 같습니다.
  Prompt optimization 및 학습 중 test 평가는 끕니다.
- 사전 private need는 계속 **비활성화**합니다. 일반 Doctor 공개 발화 + 숨겨진 STATUS,
  terminal별 helpful/unresolved reflection(512 tokens)을 사용합니다.
- 검증은 기존 **고정 creator source / anchored precommit / 선형 continuation /
  최대 3회 중 2회 뒤집힘**을 유지합니다. 첫 2회가 같은 판정이면 세 번째는 실행하지 않습니다.
  V7의 개입 지시 audit가 prefix/rubric/고정 guidance를 먼저 점검합니다.
  거부되면 continuation은 0회이며 `instruction_rejected_before_trials`로 별도 기록합니다.
  검증 대화 자체를 API scorer/generator에 보내지 않습니다.

## 프롬프트와 점수

| 단계 | 파일 |
| --- | --- |
| 후보 생성 | `prompts/sherpa/trajectory_rubric_generator_v2_8_app_alignment.txt` 및 동일 이름 `.schema.json` |
| Case 내부 중복 제거 | `prompts/sherpa/consolidate_partner_v7.txt` |
| Quality gate | `prompts/sherpa/gate_public_mc_v5.txt` |
| 적용 판단 | `prompts/sherpa/applicability_public_mc_v3.txt` |
| 행동 채점 | `prompts/sherpa/scorer_public_mc_v3.txt` |
| Compact 요청 계약 | `prompts/sherpa/compact_{applicability,behavior}_{contract_v1,semantics_v2}.txt` |
| 개입 지시 audit | `prompts/sherpa/intervention_instruction_audit_v1.txt` |
| Doctor reflection | `prompts/sherpa/doctor_post_final_reflection_v1.txt` |

M 미수행=0, M 수행/C 미완료=1, M+C 수행=2입니다. 후보 응답과 무관하게 공개 prefix만으로
sibling 공통 applicability를 판단합니다. 필요 행동을 하지 않았다는 이유로 N/A로 바꾸지 않습니다.
Scorer 입력에서 gold/outcome/future/private assessment/reflection/guidance를 제외합니다.
Source gate는 성공 source=2 / 실패 source=0 정합성을 독립적으로 검사합니다.
Generator가 source의 부분 수행(1)을 관찰해도 0으로 강제하지 않으며, source gate에서 불일치를 거부합니다.

Compact scorer는 **응답 하나 × criterion 하나**씩 요청합니다. 기존 4응답 묶음 요청보다
호출 수/비용이 증가할 수 있습니다. 최대 50은 상한이며 항상 50개를 채우지는 않습니다.
`api_usage.jsonl`, `http_attempts.jsonl`, `retry_audit.jsonl`, `diagnostics/`에 비용·토큰·시간을 기록합니다.
API transport는 기존 bounded retry/Retry-After를 사용하며 대기 중 limiter 슬롯을 놓습니다.
Compact 출력이 malformed/unresolved이면 즉시 중단하고 오류를 저장합니다(0/N/A로 보정하지 않음).
유효한 `UNSCORABLE` 판정은 같은 history의 해당 criterion을 모든 sibling에서 함께 제외합니다.
남은 유효 criterion이 있으면 그 평균을 쓰고, 없으면 group을 학습에서 제외합니다.

## 공유 파일시스템: 코드 갱신

두 서버가 같은 checkout을 사용하므로 실행 중인 두 실험을 사용자가 중단한 뒤 한 곳에서만 pull합니다.
이미 실행 중인 Python은 이전 코드를 메모리에 보유할 수 있어 실행 도중 pull로 버전을 섞지 않습니다.
새 과학적 설정은 **새 run name**으로 시작합니다. 기존 run의 재개는 이전 코드/설정에서 해야 합니다.

```bash
cd /NHNHOME/shkim/SHERPA/Collaborative_agent-sherpa
git pull --ff-only origin sherpa-jh
```

## 각 서버의 tmux 터미널에서 환경 설정

공유 `.local/b200.env`를 읽어도 환경변수는 터미널별로 다시 설정해야 합니다.
아래는 이미 모델/데이터/venv/HF Doctor receipt 준비가 끝난 서버용입니다.
Ollama 설치/서버 시작은 필요 없습니다. HF Doctor는 train/evaluate 런처가 GPU0에서 로드합니다.

```bash
cd /NHNHOME/shkim/SHERPA/Collaborative_agent-sherpa
source .local/b200.env
export SHERPA_PROJECT="$PWD"
export SHERPA_PYTHON="$SHERPA_PROJECT/.venv/bin/python"
export SHERPA_BASE_MODEL=/NHNHOME/shkim/SHERPA/vllm/qwen3
export SHERPA_DOCTOR_BACKEND=hf_local
export SHERPA_DOCTOR_MODEL=/NHNHOME/shkim/SHERPA/vllm/gemma3
export SHERPA_HF_DOCTOR_RECEIPT="$SHERPA_PROJECT/.local/b200_hf_doctor.json"
export SHERPA_HF_MODEL_REPO=zzu819/SHERPA
export CUDA_VISIBLE_DEVICES=0,1
unset SHERPA_HF_RUN_ID
set +x
read -rsp 'OpenAI API key: ' OPENAI_API_KEY; echo
export OPENAI_API_KEY
read -rsp 'Hugging Face write token: ' HF_TOKEN; echo
export HF_TOKEN
```

이미 해당 터미널에 키가 있으면 `read`는 생략합니다. 키를 파일에 저장하거나 출력하지 않습니다.
데이터 경로는 기존 `SHERPA_TRAIN_FILE`, `SHERPA_TRAIN_ORDER`, `SHERPA_TEST_FILE`을 유지합니다.
확인된 모델 파일을 바꾸지 않았다면 Doctor receipt를 다시 pin할 필요 없습니다.

**서버 A (기존 LR):**

```bash
export SHERPA_RUN_NAME=gemma-bf16-v7-rubric-lr1e5-run01
export SHERPA_LEARNING_RATE=1e-5
```

**서버 B (변경 LR):**

```bash
export SHERPA_RUN_NAME=gemma-bf16-v7-rubric-lr5e6-run01
export SHERPA_LEARNING_RATE=5e-6
```

이름을 직접 입력해도 되며 두 서버의 이름은 달라야 합니다.
로컬 `outputs/runs/$SHERPA_RUN_NAME/`과 HF model repo의 `runs/$SHERPA_RUN_NAME/`에
각 실험의 output/checkpoint/cache를 함께 저장합니다. GPU 0/1은 **각 호스트의** GPU입니다.

## 사전 점검 → 첫 배치 → 전체 학습 → 재개 → 평가

각 서버에서 위의 서로 다른 이름/LR을 설정하고 같은 명령을 실행합니다.

```bash
# 읽기 전용 환경/데이터/prompt/hash 검사; 모델 로드/API 호출 없음
bash scripts/online_rl/launch_b200_v7.sh --action preflight

# 선택 사항: 첫 5-case 배치 점검 (실제 실행은 사용자)
bash scripts/online_rl/launch_b200_v7.sh --action train --max-batches 1
"$SHERPA_PYTHON" scripts/online_rl/report_b200.py --output "outputs/runs/$SHERPA_RUN_NAME"

# 전체 train120 / 총 24개 배치. 첫 배치가 끝났으면 다음 배치부터 자동 재개
bash scripts/online_rl/launch_b200_v7.sh --action train

# 중단 후: 같은 환경/키/run name/LR로 같은 명령
bash scripts/online_rl/launch_b200_v7.sh --action train

# train120 완료 후 최종 checkpoint로 test622 한 번 평가
bash scripts/online_rl/launch_b200_v7.sh --action evaluate
"$SHERPA_PYTHON" scripts/online_rl/report_b200.py --output "outputs/runs/$SHERPA_RUN_NAME"
```

이미 완료된 배치/verification/API cache는 재사용하고 중간 실패 작업은 필요한 부분만 재실행합니다.
LR/config/code/model hash가 달라진 재개는 거부합니다. Test 결과로 LR/가중치/checkpoint를 고르지 않습니다.
최종 Doctor FINAL 정답률은 각 run의 `final_accuracy.json`에 저장합니다.

## CPU 검증

```bash
SHERPA_PYTHON=.venv/bin/python bash scripts/online_rl/test_b200.sh
SHERPA_PYTHON=.venv/bin/python bash scripts/online_rl/launch_b200_v7.sh --action preflight --cpu-only
```

기존 학습/재개/5-case barrier/early-majority 테스트에 V7 fake API 검증을 추가합니다.
실제 모델의 의미적 판단 품질, GPU 메모리/속도, 원격 API 지원 여부는 CPU 테스트로 입증하지 않습니다.


## Consolidate partition 오류 후 재개 (44ded5a에서 시작한 run)

`Consolidate cannot merge success and failure strata`는 모델이 성공/실패 후보를
한 cluster에 넣어 validator가 거부한 것입니다. 해당 batch의 update 전 실패이며,
이미 commit된 checkpoint와 성공 cache는 유지됩니다.

수정 버전은 첫 요청·모델·prompt·정상 partition/cache identity를 유지하고, 잘못된
JSON/partition에 한해 같은 원본 후보와 검증 오류를 넣어 최대 2회 수정 요청합니다.
세 번의 응답 모두 유효하지 않으면 다시 중단합니다. 후보를 코드로 임의 분할/삭제하거나
검증 실패를 0점으로 바꾸지 않습니다. 추가 요청 비용은 기존 API usage에 포함되며
`diagnostics/consolidate_validation.jsonl`에 각 응답·오류·수정 시행 수를 기록합니다.

기존 run을 새로 만들 필요는 없습니다. `maintenance/consolidate_repair_compatibility.json`에
명시한 원본 커밋의 전체 코드 hash와 이번 수정 파일 hash가 정확히 맞고 config가 같은
경우에만 재개합니다. 다른 코드/LR/config 변경은 계속 거부합니다.
원래 checkpoint의 source identity는 유지하며 실제 실행 코드는 별도의
`execution_source_sha256` 및 `diagnostics/code_compatibility_<hash>.json`에 기록합니다.
기존 valid cache/완료 case는 그대로 쓰고 미완료 consolidate부터 필요한 작업을 다시 합니다.

공유 checkout을 사용하는 **두 서버의 프로세스를 먼저 중단**한 뒤 한 곳에서 pull:

```bash
cd /NHNHOME/shkim/SHERPA/Collaborative_agent-sherpa
git pull --ff-only origin sherpa-jh
```

각 터미널의 환경/키는 그대로 사용합니다. 기존 run name/LR을 반드시 유지하세요.

```bash
# 서버 A (아래 이름으로 시작한 경우)
bash scripts/online_rl/launch_b200_v7.sh --action train \
  --run-name gemma-bf16-v7-rubric-lr1e5-run01 --learning-rate 1e-5

# 서버 B (아래 이름으로 시작한 경우)
bash scripts/online_rl/launch_b200_v7.sh --action train \
  --run-name gemma-bf16-v7-rubric-lr5e6-run01 --learning-rate 5e-6
```

기존에 다른 이름을 사용했다면 위 이름 대신 실제 기존 이름을 사용합니다.
호환 재개에서는 `stage: compatible_code_resume`과 완료 case 수가 출력됩니다.
모델/데이터 재다운로드, cache 삭제, 초기 adapter 재설정은 필요 없습니다.
CPU 회귀 검증: `python -m pytest -q tests/test_consolidate_repair.py tests/test_parallel_bank.py tests/test_b200_v7.py tests/test_b200_runner.py`.
