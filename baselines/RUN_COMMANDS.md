# RaR / OnlineRubrics 실행 명령

이 문서는 기존 OpenRouter launcher의 준비·실행 기록이다. 현재 GPT-5.6 Luna/OpenAI 병렬 채점과 기존 checkpoint 재개 명령은 [최신 실행 안내](parallel_api/README.md)를 사용한다.

사용자가 제공한 초기 adapter parameter SHA256 `9018ce3bbacd319049576e8ace88afb4777621a8769919bf46676fca8a85b17c`를 `runtime.local.json`과 예시에 기대값으로 연결했다. 실제 시작 시 이 값과 다르면 rollout 전에 중단한다. [대화·평가 조건 대조 기록](common/user_identity_audit.json)에 일치 항목과 아직 미검증인 base revision·평가 protocol hash·artifact 파일 해시를 구분했다.
현재 Ollama0.12.5 실행 파일과 Doctor 저장소 경로는 `runtime.example.json`에 기록되어 있다. 초기 `doctor_discovery.json`은 설치 전 발견 상태이며, 현재 설치 확인은 preflight로 수행한다.

[Doctor 준비 안내](DOCTOR_SETUP.md)에 따라 새 모델을 다운로드·검증한다. 원본 서버 복사는 생략한다. Ollama 실행 파일 경로는 그대로이며 모델 저장소는 `/convei_nas2/zzuhyun/ollama_baselines/models_gemma3_88c2e99b4a44`다. 승인된 새 digest `88c2e99b4a44…`를 고정했다. 이전 모델과의 불일치는 승인 기록으로 남기며 실행 blocker로 삼지 않는다. 앞으로 새 pin과 다르면 중단한다.

| 방법 | 설정 파일 | Doctor GPU / port | Assistant GPU | 출력 root |
|---|---|---|---|---|
| RaR | `baselines/rar/method.json` | 0→11731, 1→11732 | 2,3; updater3 | `outputs/rar_train120_d88c2e99b4a44_v5` |
| OnlineRubrics | `baselines/online_rubrics/method.json` | 4→11831, 5→11832 | 6,7; updater7, reference6 | `outputs/online_rubrics_train120_d88c2e99b4a44_v5` |

공통 로컬 경로 파일은 `baselines/runtime.local.json`이다. 출력 root 아래 checkpoint·cache·Doctor 로그·평가 결과가 방법별로 분리된다. GPU 배분은 launcher가 설정하므로 전체4장을 `CUDA_VISIBLE_DEVICES`로 따로 지정하지 않는다. Reference는 GPU6의 초기 LoRA 교체로 생성하며 추가 GPU를 사용하지 않는다. 기존 프로세스가 지정 포트에 있거나 run의 PID 파일이 살아 있는 프로세스를 가리키면 중단·재시작·채택 없이 오류로 멈춘다. 다른 포트의 기존 Ollama에는 접근하지 않는다. 포트 가용성은 실행 시 다시 확인한다.

## 1. 두 터미널에서 각각 폴더 이동과 키 입력

아래 블록을 **터미널 A와 B에서 각각** 실행한다. 키는 화면에 표시하거나 파일에 저장하지 않고 해당 셸의 환경 변수로만 전달한다. 값은 `read`가 기다릴 때 입력하므로 셸 명령 기록에도 키 문자열이 들어가지 않는다. `set +x`는 확장된 키가 디버그 출력에 표시되지 않게 한다. 키를 명령줄에 직접 붙여 넣지 않는다.

```bash
cd /convei_nas2/zzuhyun/Collaborative_agent
set +x
IFS= read -r -s -p 'OPENROUTER_API_KEY: ' OPENROUTER_API_KEY
printf '\n'
export OPENROUTER_API_KEY
```

`runtime.local.json`의 `env_file`은 null이다. `.env`를 만들 필요가 없다. 새 터미널에서 재개할 때도 위 블록을 실행한다. 실험 완료 후에는 `unset OPENROUTER_API_KEY`로 해당 터미널의 변수를 지울 수 있다.

## 2. 각 방법의 사전 점검

Ollama 설치와 새 Doctor 다운로드·전체 검증이 완료된 후 GPU가 보이는 터미널에서 preflight를 통과해야 다음 단계로 진행한다. Preflight는 모델 생성·학습·평가를 시작하지 않는다.

터미널 A:

```bash
python -m baselines.rar.launch --config baselines/rar/method.json --runtime baselines/runtime.local.json --action preflight
```

터미널 B:

```bash
python -m baselines.online_rubrics.launch --config baselines/online_rubrics/method.json --runtime baselines/runtime.local.json --action preflight
```

GPU 접근 없이 파일·키 설정·포트만 확인하려면 같은 명령 뒤에 `--cpu-only`를 붙인다. API 키 존재 확인은 키 유효성·provider 모델 접근 권한에 대한 실제 API 검증은 아니다.

## 3. 첫 1배치 학습 점검과 결과 확인

처음 비어 있는 각 output에서 수행한다. 본 학습 첫5case를 실제 학습하며 최대1회 update 후 멈춘다. 아래 `&&`는 학습 명령이 성공한 경우에만 검증한다. 두 방법을 각 터미널에서 동시에 실행할 수 있다.

터미널 A:

```bash
python -m baselines.rar.launch --config baselines/rar/method.json --runtime baselines/runtime.local.json --action train --max-batches 1 &&
python -m baselines.common.check_run --run outputs/rar_train120_d88c2e99b4a44_v5 --expected-cases 5
```

터미널 B:

```bash
python -m baselines.online_rubrics.launch --config baselines/online_rubrics/method.json --runtime baselines/runtime.local.json --action train --max-batches 1 &&
python -m baselines.common.check_run --run outputs/online_rubrics_train120_d88c2e99b4a44_v5 --expected-cases 5
```

`check: passed`, `committed_cases: 5`, `five_case_batches: 1`, `reward_weights: {w_outcome: 0, w_rubric: 1}`, `test_evaluated: false`를 확인한다. 전부 zero-advantage/invalid인 배치는 원래 규칙에 따라 update가 생략될 수 있다. `--max-batches 1`은 **호출당 추가1배치**이므로 점검이 완료된 뒤 반복하지 않는다. GPU 메모리 사용량은 별도 `nvidia-smi`로 확인한다. 실제 peak VRAM·모델 로드는 아직 검증하지 않았다.

두 output의 `initial_policy.json`에 있는 `initial_policy_sha256`이 같아야 한다. Launcher가 공통 초기화 저장소와 비교하여 불일치 시 첫 rollout 전에 중단한다. 초기/reference adapter·보상 설정·judge 설정은 재개 시에도 검증한다.

## 4. 두 터미널에서 전체 학습을 동시에 이어서 실행

터미널 A:

```bash
python -m baselines.rar.launch --config baselines/rar/method.json --runtime baselines/runtime.local.json --action train
```

터미널 B:

```bash
python -m baselines.online_rubrics.launch --config baselines/online_rubrics/method.json --runtime baselines/runtime.local.json --action train
```

첫5case를 다시 학습하지 않고 남은115case를 진행한다. 전체는 원본 순서120case, batch5 × 24, 최대24 optimizer update다. RaR/OnlineRubrics 모두 rubric-only이며 학습 중 test 평가를 하지 않는다.

## 5. 중단 후 재개

새 터미널이라면 1번 블록으로 폴더와 키를 설정한 뒤 **4번의 같은 train 명령**을 각각 실행한다. 기존 output·checkpoint·cache와 설정을 유지한다. `run_state.json`과 recovery receipt에 따라 마지막 committed batch부터 복구한다.

강제 종료로 baseline Doctor가 남아 있으면 preflight가 해당 PID/포트를 표시하고 거부한다. 자동으로 종료하거나 재시작하지 않는다. 기존 프로세스의 소유자와 실행 상태를 확인한 뒤 해당 실행이 정리되고 포트가 비어 있을 때 재개한다. 다른 Ollama를 종료하는 명령은 이 안내에 포함하지 않는다.

## 6. 완료 후 최종 checkpoint의 test622 1회 평가

각 방법의 `run_state.json`에서120case가 완료되어야 실행된다. Test 결과로 checkpoint를 선택하지 않고 각각 `checkpoints/checkpoint_000120`만 사용한다. 두 평가도 각자의4장 안에서 동시에 실행할 수 있다.

터미널 A:

```bash
python -m baselines.rar.launch --config baselines/rar/method.json --runtime baselines/runtime.local.json --action evaluate
python -m json.tool outputs/rar_train120_d88c2e99b4a44_v5/final_accuracy.json
```

터미널 B:

```bash
python -m baselines.online_rubrics.launch --config baselines/online_rubrics/method.json --runtime baselines/runtime.local.json --action evaluate
python -m json.tool outputs/online_rubrics_train120_d88c2e99b4a44_v5/final_accuracy.json
```

평가 중단 시 같은 evaluate 명령으로 이어서 실행하며 완료된 case는 재생성하지 않는다. 정확도는 원래 evaluator의 Doctor FINAL 정답률, 분모622이며 unparsed도 분모에 포함한다. 상세 결과는 각 output의 `evaluation/checkpoint_000120/`에 저장한다. Rubric API/reference 생성은 평가에서 사용하지 않는다.

보상 추가 비용은 각 output의 `costs.json`, `reward_api_attempts.jsonl`, reference 로그에 기록한다. 원래 rollout/train/eval 비용과 분리하고, provider가 제공하지 않은 비용은0원으로 채우지 않는다.
