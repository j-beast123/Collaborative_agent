# 승인된 새 Doctor 모델 준비

사용자가 기존 Doctor digest `997a7c2c…` 대신 현재 다운로드 가능한 `gemma3:12b-it-q8_0` 사용을 승인했다. 현재 새 고정값은 다음과 같다.

```text
88c2e99b4a44ef8a4c855315927ae99f6ea77a06f4ec24ad2d08929f818f5f00
```

[공식 registry manifest](https://registry.ollama.ai/v2/library/gemma3/manifests/12b-it-q8_0)의 실제 응답 바이트에서 SHA256을 재확인했다. 변경은 [doctor_change.json](common/doctor_change.json)에 기록했다. **기존 digest와 다르다는 이유는 더 이상 blocker가 아니다.** 앞으로는 이 새 digest가 기대값이며, 태그가 다시 바뀌면 자동 수용하지 않고 다운로드·preflight·재개 시 거부한다. 원본 서버 복사는 생략한다.

RaR와 OnlineRubrics의 학습/평가, Doctor D1 cache identity에 같은 새 digest를 적용한다. 원래 프롬프트·생성 옵션·Ollama0.12.5·데이터·GRPO·방법별 보상은 유지한다. 새 모델 artifact에는 template/params 등 모델 자체 파일의 변경이 포함될 수 있으며 이전 Doctor와 동일한 모델 환경이라고 표기하지 않는다. 기존 Doctor로 수행한 외부 SHERPA/Base/Outcome-only 결과와는 Doctor 조건이 다르다는 변경 이력을 함께 보고한다.

## 경로 및 cache 분리

이미 `baselines/runtime.local.json`과 예시에 연결했다. 진행 중인 Ollama 다운로드/설치 스크립트와 실행 파일 경로는 변경하지 않았다.

```json
{
  "ollama_binary": "/convei_nas2/zzuhyun/ollama_baselines/opt/ollama-0.12.5/bin/ollama",
  "ollama_models_dir": "/convei_nas2/zzuhyun/ollama_baselines/models_gemma3_88c2e99b4a44",
  "doctor_model_digest": "88c2e99b4a44ef8a4c855315927ae99f6ea77a06f4ec24ad2d08929f818f5f00"
}
```

| 방법 | 출력/checkpoint/cache root | Doctor GPU / port | Assistant |
|---|---|---|---|
| RaR | `outputs/rar_train120_d88c2e99b4a44_v5` | 0→11731, 1→11732 | GPU2,3 |
| OnlineRubrics | `outputs/online_rubrics_train120_d88c2e99b4a44_v5` | 4→11831, 5→11832 | GPU6,7; reference6 |

D1·tree·evaluation·rubric judge·OnlineRubrics reference cache는 새 run root에 저장한다. 두 방법의 공통 초기 rubric 저장소도 `outputs/shared_initial_rubrics_d88c2e99b4a44_v5`로 분리했다. 이전 v4 root/cache를 복사하거나 이어 쓰지 않는다. Doctor와 무관한 초기 Assistant adapter 비교 저장소는 유지한다. 준비된 Outcome-only launcher도 새 공통 Doctor 및 별도 `outputs/outcome_only_train120_d88c2e99b4a44_v5`를 사용한다. 기존 결과 파일을 변경하지 않는다.

## 1. Ollama 설치 완료 대기

이미 다운로드 중이면 완료를 기다리고 중복 실행하지 않는다. 설치본은 계속 아래 경로다.

```bash
cd /convei_nas2/zzuhyun/Collaborative_agent
OLLAMA_HOST=http://127.0.0.1:11931 \
  /convei_nas2/zzuhyun/ollama_baselines/opt/ollama-0.12.5/bin/ollama --version
```

서버 미실행 경고와 함께 client version0.12.5가 표시될 수 있다. 이 명령은 서버를 시작하지 않는다. 설치를 처음 시작하는 경우에만 `bash baselines/setup/install_ollama_user.sh`를 사용한다. 새 경로에만 설치하며 기존 프로세스/시스템 설치를 변경하지 않는다.

## 2. 새 Doctor 모델 다운로드

```bash
cd /convei_nas2/zzuhyun/Collaborative_agent
bash baselines/setup/download_doctor_if_matching.sh
```

스크립트는 작은 manifest를 먼저 새 고정값과 대조하고, 일치하면 model/projector/template/params/license/config blob을 내려받는다. 모든 파일의 크기와 전체 SHA256을 확인한 뒤 새 저장소에 tag를 등록한다. **Ollama daemon이나 GPU 모델 로딩은 필요 없다.** 중단 후 같은 명령으로 재개한다. 같은 다운로드 명령을 동시에 중복 실행하지 않는다.

모델 blob 총합은13,373,415,378bytes(약13.37GB/12.45GiB)다. Ollama 압축본 약1.69GB, 해제 공간과 전송 임시 파일을 고려해 **35GB 이상**을 준비 예산으로 잡는다. 학습 checkpoint/tree/log는 별도다. RaR/OnlineRubrics의 Doctor는 같은 새 저장소를 읽으므로 디스크 모델 복사본은1개다.

## 3. 모델 전체 검증

```bash
python -m baselines.common.doctor_assets \
  --models /convei_nas2/zzuhyun/ollama_baselines/models_gemma3_88c2e99b4a44
```

`manifest_sha256`이 위 새 digest이고 `all_blob_hashes_verified`가 true여야 한다. 이 명령은 디스크 읽기만 수행하며 추론/학습/평가를 하지 않는다. Preflight는 별도로 같은 manifest digest와 실제 Ollama 버전을 확인한다. Launcher가 시작하는 Doctor 서버도 API에서 반환하는 모델 digest를 검사한다.

## 4. 두 방법의 preflight

키를 설정하지 않았다면 각 실행 터미널에서 숨김 입력한다. 파일에 저장하거나 출력하지 않는다.

```bash
cd /convei_nas2/zzuhyun/Collaborative_agent
set +x
IFS= read -r -s -p 'OPENROUTER_API_KEY: ' OPENROUTER_API_KEY
printf '\n'
export OPENROUTER_API_KEY
```

```bash
# 터미널 A
python -m baselines.rar.launch --config baselines/rar/method.json --runtime baselines/runtime.local.json --action preflight
```

```bash
# 터미널 B
python -m baselines.online_rubrics.launch --config baselines/online_rubrics/method.json --runtime baselines/runtime.local.json --action preflight
```

Preflight에 출력되는 `doctor.model_digest`와 `opening_doctor_digest`가 모두 새 고정값인지 확인한다. `blockers: []`인 경우에만 사용자가 학습을 시작한다. 실제 GPU 서버에서는 RTX3090 24GB 8장이 확인됐다. Sandbox 안의 NVIDIA 오류는 실제 서버 장애와 구분한다. Docker 내부는 계정 소켓 권한 부족으로 미확인이지만 이 구성은 Docker를 요구하지 않는다.

학습 및 최종 평가 명령은 새 출력 경로로 갱신한 [RUN_COMMANDS.md](RUN_COMMANDS.md)를 따른다. 재개 시 run config·method binding·Doctor digest·D1 identity·코드 hash를 대조하므로 다른 Doctor를 기존 run에 적용하면 거부한다. 설정·검증 도구만 준비했으며 모델 다운로드·학습·평가는 사용자가 실행한다.
