# B200 새 서버: clone부터 학습·재개·평가까지

이 문서는 `sherpa-jh` 브랜치의 B200 2장 실행용입니다. Doctor는 GPU0와
`127.0.0.1:11985`, Assistant rollout/학습은 GPU1을 사용합니다.
이 서버에서 다른 작업이 해당 GPU/포트를 사용한다면 먼저 별도 자원을 정해야 합니다.
아래 명령은 사용자가 실행합니다. 구현 과정에서는 설치·다운로드·API·GPU 실행을 하지 않았습니다.

## 공유 파일시스템의 B200 두 서버: 실험 이름 하나로 분리

이 절은 `/NHNHOME/shkim/SHERPA/Collaborative_agent-sherpa`를 두 서버가 공유할 때의
실행 절차입니다. 코드·venv·원본 데이터·Qwen·Ollama 설치/모델·Doctor digest receipt는
공유하고, **서로 다른 실험 이름**을 입력하여 checkpoint·tree·rubric/verification cache·
진행 상태·API 사용량·평가 결과를 모두 분리합니다. 데이터 준비/설치는 한 번만 합니다.
공유 코드/venv는 두 실험을 시작하기 전에 갱신하고 실행 중에는 변경하지 않습니다.

| 서버 | 입력 예시 | Learning rate | 로컬 출력 | HF 모델 저장소 경로 |
| --- | --- | --- | --- | --- |
| A | `lr1e5-run01` | `1e-5` | `outputs/runs/lr1e5-run01/` | `zzu819/SHERPA`의 `runs/lr1e5-run01/` |
| B | `lr5e6-run01` | `5e-6` | `outputs/runs/lr5e6-run01/` | `zzu819/SHERPA`의 `runs/lr5e6-run01/` |

실험 이름은 영문/숫자로 시작하는 1~96자의 영문·숫자·`_`·`-`·`.` 조합입니다.
`--run-name NAME` 또는 `SHERPA_RUN_NAME`을 사용합니다. CLI가 우선하며 새 이름은
기존 `SHERPA_HF_RUN_ID`보다 우선합니다. 기존 이름 없는 실행은 옛 경로를 그대로 사용합니다.
**기존 run을 이름만 바꿔 재개하는 기능은 아닙니다.** 새 이름은 새 실험을 만듭니다.
같은 이름에 LR·Doctor·코드 등 다른 설정을 넣으면 재개 시 거부합니다.

HF는 실행 파일시스템이 아닙니다. 계산은 로컬 폴더에서 진행하고 완료된 5-case 배치마다
`runs/<name>/artifacts/` 및 `snapshot.json`을 함께 백업합니다. HF 폴더는 첫 성공한
백업 때 생성되므로 첫 배치 전에는 보이지 않을 수 있습니다. 동시에 다른 실험이 HF에
commit하면 최신 head를 다시 확인해 최대 5회 시도하며, 다른 실험 폴더는 수정하지 않습니다.
인증 오류는 이 재시도의 대상이 아닙니다. 백업 실패로 중단해도 완료된 로컬 업데이트는 보존됩니다.

**시작 전 공유 checkout에서 한 번만:**

```bash
cd /NHNHOME/shkim/SHERPA/Collaborative_agent-sherpa
git pull --ff-only
```

아래는 이전에 준비한 `.local/b200.env`와 모델·데이터를 재사용합니다. 아직 준비하지
않았다면 아래 1~5절의 준비 절차부터 수행하고 모든 경로를 공유 프로젝트 아래로 지정하세요.
각 서버의 Doctor는 별도 tmux에서 `source .local/b200.env` 후 다음 명령으로 시작합니다.
이미 이 서버에서 정상 실행 중이면 다시 시작하지 않습니다.

```bash
"$SHERPA_PYTHON" -m src.online_rl.tree.b200_doctor_setup serve
```

최초 모델 pull/pin은 **서버 A에서만** 기존 `pull` 명령으로 완료한 후 서버 B를 시작합니다.
두 서버는 같은 binary/models/receipt를 읽으며 `server.json`만
`$SHERPA_OLLAMA_HOME/servers/<hostname>-<hash>/`에 각각 저장합니다.
서로 다른 실제 서버이면 각각 GPU0/1과 localhost:11985를 사용해도 충돌하지 않습니다.
공유 receipt를 삭제하거나 양쪽에서 별도로 새 모델을 pin하지 마세요.

**각 서버의 학습 tmux에서 입력:**

```bash
cd /NHNHOME/shkim/SHERPA/Collaborative_agent-sherpa
source .local/b200.env
read -rp 'Experiment name (A: lr1e5-run01 / B: lr5e6-run01): ' SHERPA_RUN_NAME
read -rp 'Learning rate (A: 1e-5 / B: 5e-6): ' SHERPA_LEARNING_RATE
export SHERPA_RUN_NAME SHERPA_LEARNING_RATE
export SHERPA_HF_MODEL_REPO=zzu819/SHERPA
export CUDA_VISIBLE_DEVICES=0,1
set +x
read -rsp 'OPENROUTER_API_KEY: ' OPENROUTER_API_KEY; printf '\n'
read -rsp 'HF_TOKEN (model repo write access): ' HF_TOKEN; printf '\n'
export OPENROUTER_API_KEY HF_TOKEN
"$SHERPA_PYTHON" -m src.online_rl.tree.b200_doctor_setup verify
bash scripts/online_rl/launch_b200.sh --action preflight
```

환경변수는 각 터미널에만 설정합니다. 실험마다 공유 `.local/b200.env`를 덮어쓰지 않습니다.
Preflight에서 `blockers: []`, LR, `output`, `hf_storage.run_id`를 확인합니다.
각 run의 API 상한은 100이므로 두 서버 합계는 최대 200 in-flight일 수 있습니다.
각각 같은 train120 전체를 학습하며 5-case 업데이트/최대24배치와 나머지 과학적 조건은 유지됩니다.

**각 서버에서 첫 배치 → 확인 → 전체 학습:**

```bash
bash scripts/online_rl/launch_b200.sh --action train --max-batches 1
"$SHERPA_PYTHON" scripts/online_rl/report_b200.py
bash scripts/online_rl/launch_b200.sh --action train
```

동일 터미널에서 실험 이름/LR을 유지합니다. 중단 후 새 터미널에서는 위 입력 블록을
**같은 이름·LR**로 실행한 뒤 `--action train`만 다시 실행합니다. `--max-batches 1`을
다시 주면 추가 1배치를 수행합니다. 완료 checkpoint·optimizer·진행 상태·cache를 복원합니다.
공유 filesystem의 동일 run 이름은 `runner.lock`으로 동시 실행을 막습니다.

**각 실험 train120 완료 후 최종 checkpoint의 test622 1회 평가:**

```bash
bash scripts/online_rl/launch_b200.sh --action evaluate
"$SHERPA_PYTHON" scripts/online_rl/report_b200.py
```

결과는 `outputs/runs/$SHERPA_RUN_NAME/final_accuracy.json`과 같은 HF run에 저장합니다.
학습 중 test는 실행하지 않으며 test 결과로 LR이나 checkpoint를 고르지 않습니다.
모든 train/evaluate 명령에 `--run-name lr1e5-run01 --learning-rate 1e-5`처럼
명시적으로 전달하는 방식도 가능합니다. 이름에는 LR이 자동 해석되지 않으므로 둘 다 맞춰주세요.

## 1. 새 폴더에 clone하고 Python 환경 준비

```bash
git clone --branch sherpa-jh --single-branch https://github.com/j-beast123/Collaborative_agent.git "$HOME/Collaborative_agent-sherpa"
cd "$HOME/Collaborative_agent-sherpa"
python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
.venv/bin/python -m pip install -r requirements-b200.txt
.venv/bin/python -m pip install -e . --no-deps
.venv/bin/python -m pip check
```

Python3.11, git, curl, tmux 및 B200에 맞는 NVIDIA 드라이버가 서버에 필요합니다.
이미 준비된 B200용 Python 환경이 있다면 위 venv/설치를 생략하고 아래
`SHERPA_PYTHON`만 그 환경의 Python 절대경로로 지정해도 됩니다.
기존 시스템 환경이나 드라이버를 이 명령으로 변경하지 않습니다.

PyTorch2.7부터 Blackwell과 CUDA12.8 wheel 지원이 추가되었습니다.
위 Torch2.7.1 cu128은 [공식 설치 명령](https://pytorch.org/get-started/previous-versions/)을
사용합니다. [Blackwell 지원 설명](https://pytorch.org/blog/pytorch-2-7/).
`requirements-active.txt`는 이전 서버의 Torch2.6 설정이므로 B200 설치에 사용하지 마세요.
`requirements-b200.txt`는 Torch를 제외한 기존 주요 의존성 버전을 유지합니다.
B200에서 이 조합의 실제 학습은 아직 검증하지 않았습니다. Torch/CUDA 버전은 run manifest에 기록됩니다.

## 2. Hugging Face에서 동일 데이터·Qwen 준비

Dataset 저장소는 `zzu819/SHERPA`이며 아래 두 파일을 읽기 전용 확인했습니다.

- `train/medmcqa_case120.jsonl`: 기존 train120과 SHA256·행 순서 동일.
- `test/casimedicos-arg.jsonl`: 기존 test622와 SHA256·행 순서 동일.

현재 고정 revision은 `7f222446084ed59885986a2a35fb6e467110da96`입니다.
`main`의 추후 변경을 자동으로 따라가지 않습니다. Train order는 Git의 기존
selection manifest로 재구성하고 원래 SHA256까지 검증합니다. 새로 샘플링하지 않습니다.

```bash
cd "$HOME/Collaborative_agent-sherpa"
export SHERPA_PYTHON="$PWD/.venv/bin/python"
"$SHERPA_PYTHON" -m src.online_rl.tree.b200_hub data
```

Qwen 원본은 upstream의 고정 revision에서 준비합니다. `zzu819/SHERPA` 모델 저장소에는
학습한 LoRA·optimizer·출력·cache를 저장하며, 원본 Qwen 전체를 매번 복제하지 않습니다.
이미 동일 revision의 로컬 모델이 있다면 다운로드를 생략하고 그 경로를 설정하세요.

```bash
export SHERPA_BASE_MODEL="$HOME/models/Qwen3-8B-b968826d9c46dd6066d109eabc6255188de91218"
"$SHERPA_PYTHON" - <<'PY_MODEL'
import os
from huggingface_hub import snapshot_download
snapshot_download("Qwen/Qwen3-8B", revision="b968826d9c46dd6066d109eabc6255188de91218",
                  local_dir=os.environ["SHERPA_BASE_MODEL"])
PY_MODEL
```

Ollama용 Gemma GGUF 모델은 아래 4절의 기존 installer/Doctor pin 절차를 유지합니다.
HF에 저장한 LoRA만으로는 학습 재개에 충분하지 않으며 optimizer·상태·cache를 함께
복원해야 합니다. 아래 9절이 전체 백업/복원 경로입니다.

## 3. 키를 제외한 실행 환경 저장

아래 `SHERPA_BASE_MODEL`만 실제 B200의 Qwen snapshot 경로로 바꾸세요.
다른 위치의 데이터/기존 Python 환경을 쓸 때는 해당 export도 바꾸세요.
이 파일은 gitignored이며 API 키는 넣지 않습니다.

```bash
cd "$HOME/Collaborative_agent-sherpa"
mkdir -p .local
cat > .local/b200.env <<'ENV'
export SHERPA_PYTHON="$HOME/Collaborative_agent-sherpa/.venv/bin/python"
export SHERPA_BASE_MODEL="$HOME/models/Qwen3-8B-b968826d9c46dd6066d109eabc6255188de91218"
export SHERPA_TRAIN_FILE="$HOME/Collaborative_agent-sherpa/data/interim/medmcqa/five_gpu_train120_seed20261008/train120_excluding_calibration_seed20261008.jsonl"
export SHERPA_TRAIN_ORDER="$HOME/Collaborative_agent-sherpa/data/interim/medmcqa/five_gpu_train120_seed20261008/train120_order_seed20261008.json"
export SHERPA_TEST_FILE="$HOME/Collaborative_agent-sherpa/data/interim/casimedicos/all_en_622.jsonl"
export SHERPA_LEARNING_RATE=1e-5
export SHERPA_HF_MODEL_REPO=zzu819/SHERPA
export SHERPA_HF_RUN_ID=b200-need-bank-lr1e5-run01
export CUDA_VISIBLE_DEVICES=0,1
export SHERPA_OLLAMA_HOME="$HOME/.local/share/sherpa-b200"
export SHERPA_DOCTOR_RECEIPT="$HOME/Collaborative_agent-sherpa/.local/b200_doctor.json"
unset SHERPA_OLLAMA_BINARY SHERPA_OLLAMA_MODELS
ENV
source .local/b200.env
```

### Learning rate 지정 방식

* 우선순위: `--learning-rate` > `SHERPA_LEARNING_RATE` > config 기본값 `1e-5`.
* 예: `.local/b200.env`에 `export SHERPA_LEARNING_RATE=5e-6`.
* 또는 `bash scripts/online_rl/launch_b200.sh --action train --learning-rate 5e-6`.
* CLI로 지정했다면 report/evaluate에도 같은 `--learning-rate`를 전달하세요.
* 0·음수·NaN·무한대는 모델 로딩 전에 거부합니다.
* 실제 값은 optimizer, manifest의 config 및 resume config hash에 들어갑니다.
* 기본값 `1e-5`의 경로는 기존 그대로 `outputs/sherpa_b200_max4_parallel_bank_v2/doctor_<digest12>/`입니다.
* `5e-6`의 경로는 `outputs/sherpa_b200_max4_parallel_bank_v2/lr_5e-06/doctor_<digest12>/`입니다.
  다른 LR은 별도 학습이며 이전 checkpoint/cache를 자동으로 이어 쓰지 않습니다.
* 재개와 최종 평가는 학습 당시 LR로 실행하세요. 기존 출력에 다른 LR을 강제로 연결하면
  config identity 검증에서 중단합니다. 학습 중 LR 스케줄이나 자동 튜닝을 추가한 변경은 아닙니다.

Learning rate는 업데이트 크기에 영향을 주지만, 큰 값이나 작은 값 어느 쪽도 성능 개선을
보장하지 않습니다. 비교한다면 train에서 사전에 정한 점검 기준으로 선택하고, test622를
보고 LR을 선택하지 마세요. 이번 변경은 LR 입력만 열며 beta·clip·reward·배치·턴 수는 유지합니다.

## 4. Ollama 설치, Doctor 전용 tmux

먼저 학습용 터미널에서 전용 설치 경로에 Ollama0.12.5만 준비합니다.
설치·모델 다운로드 준비에 약30GiB 여유를 두고 checkpoint 공간은 별도로 확보하세요.

```bash
cd "$HOME/Collaborative_agent-sherpa"
source .local/b200.env
"$SHERPA_PYTHON" -m src.online_rl.tree.b200_doctor_setup install
tmux new -s sherpa-doctor
```

생성된 **Doctor tmux 안에서**:

```bash
cd "$HOME/Collaborative_agent-sherpa"
source .local/b200.env
"$SHERPA_PYTHON" -m src.online_rl.tree.b200_doctor_setup serve
```

서비스가 실행되면 `Ctrl+b` 다음 `d`로 detach합니다. `Ctrl+c`는 서비스를 종료하므로
detach 용도로 사용하지 마세요. 기존 프로세스를 kill하거나 시스템 서비스를 교체하지 않습니다.

## 5. 학습용 tmux에서 Doctor 다운로드·검증, 키 입력

```bash
tmux new -s sherpa-train
```

**학습 tmux 안에서**:

```bash
cd "$HOME/Collaborative_agent-sherpa"
source .local/b200.env
"$SHERPA_PYTHON" -m src.online_rl.tree.b200_doctor_setup pull
"$SHERPA_PYTHON" -m src.online_rl.tree.b200_doctor_setup verify
set +x
read -rsp 'OPENROUTER_API_KEY: ' OPENROUTER_API_KEY
printf '\n'
export OPENROUTER_API_KEY
read -rsp 'HF_TOKEN (model repo write access): ' HF_TOKEN
printf '\n'
export HF_TOKEN
CFG=configs/experiments/train120_max4_rubric_only_b200.json
bash scripts/online_rl/launch_b200.sh --config "$CFG" --action preflight
```

최초 pull은 현재 Doctor 모델을 받고 실제 digest를 `.local/b200_doctor.json`에 고정합니다.
재실행하면 기존 pin을 검증하고 mutable tag를 다시 pull하지 않습니다. 모델과 receipt를
삭제하거나 교체하지 않아야 같은 실험을 재개할 수 있습니다.
Preflight의 `blockers: []`, 선택한 `learning_rate`와 `output`을 확인하세요.
Provider의 실제 모델 접근 권한/지원 여부는 CPU preflight가 보증하지 않습니다.
Generator/consolidate는 OpenRouter `openai/gpt-6-luna`, reasoning `none`을 실제 요청하며
미지원 응답은 중단합니다. Generator/consolidate의 temperature는 지원되지 않아 요청에서
제외하며 provider 기본 동작을 사용합니다(기존 0.7과 같다고 가정하지 않습니다).
Agent/Doctor/gate/scorer 설정은 유지합니다.

## 6. 첫 1배치 점검 → 전체 학습

```bash
bash scripts/online_rl/launch_b200.sh --config "$CFG" --action train --max-batches 1
"$SHERPA_PYTHON" scripts/online_rl/report_b200.py --config "$CFG"
```

`completed_cases: 5`, `completed_batches: 1`과 출력 디렉터리의
`diagnostics/batch_000001.json`을 확인하세요. 이 5개는 정규 학습의 첫 배치입니다.
`--max-batches 1`은 이미 진행된 run에서는 **추가 1배치**를 뜻합니다.

```bash
bash scripts/online_rl/launch_b200.sh --config "$CFG" --action train
```

이어지는 배치부터 train120까지 진행합니다. tmux는 `Ctrl+b`, `d`로 detach합니다.
SSH를 끊었다가 `tmux attach -t sherpa-train`으로 다시 볼 수 있습니다.
첫 배치 후 변경할 LR을 정했다면 기존 run을 수정하지 말고 새 LR의 별도 run으로 시작하세요.

## 7. 실제 중단 후 재개

같은 코드·모델·Doctor receipt·LR로 재실행합니다. 재개 중인 코드 폴더에 `git pull`로
새 코드를 덮어쓰면 source identity가 달라져 중단될 수 있으므로 학습 버전을 유지하세요.

```bash
cd "$HOME/Collaborative_agent-sherpa"
source .local/b200.env
set +x
read -rsp 'OPENROUTER_API_KEY: ' OPENROUTER_API_KEY
printf '\n'
export OPENROUTER_API_KEY
read -rsp 'HF_TOKEN (model repo write access): ' HF_TOKEN
printf '\n'
export HF_TOKEN
CFG=configs/experiments/train120_max4_rubric_only_b200.json
bash scripts/online_rl/launch_b200.sh --config "$CFG" --action preflight
bash scripts/online_rl/launch_b200.sh --config "$CFG" --action train
```

Doctor tmux가 살아 있다면 serve를 다시 실행하지 않습니다. 서버 재부팅 등으로 종료됐다면
4절의 serve만 다시 시작하세요. 완료 checkpoint·optimizer·진행 상태·검증 cache를 복원합니다.
별도의 resume flag는 없습니다.

## 8. train120 완료 후 최종 checkpoint의 test622 1회 평가

같은 학습 tmux와 환경에서:

```bash
"$SHERPA_PYTHON" scripts/online_rl/report_b200.py --config "$CFG"
bash scripts/online_rl/launch_b200.sh --config "$CFG" --action evaluate
"$SHERPA_PYTHON" scripts/online_rl/report_b200.py --config "$CFG"
```

학습 완료는120 case/24 batch이며 최종 checkpoint는 해당 출력 경로의
`checkpoints/checkpoint_000120`입니다. 학습 미완료 시 evaluate는 거부합니다.
Doctor FINAL 정확도는 `final_accuracy.json`에 저장됩니다. 같은 평가 명령 재실행은
미완료 case만 재개하며 완료된622개를 다시 평가하지 않습니다. 평가 결과로 LR이나
checkpoint를 선택하지 않습니다. 상세 실험 정의와 진단 항목은 [B200 실행 가이드](B200_SHERPA.md)를 참고하세요.


## 9. HF checkpoint·output 저장과 불러오기

`SHERPA_HF_RUN_ID`를 설정하면 매 5-case 업데이트와 로컬 checkpoint 저장을 완료한
barrier에서 `zzu819/SHERPA` **모델 저장소**의 `runs/<run-id>/`로 동기 백업합니다.
첫 배치 완료 후 HF 저장까지 확인하고 다음 배치를 진행합니다. 최종 test 평가 완료
후에도 평가 결과를 백업합니다. API key 파일·runtime 환경 파일·Doctor/Qwen 원본은
포함하지 않습니다. 저장소는 현재 공개이므로 저장한 trajectory·reflection·진단도 공개됩니다.

```text
zzu819/SHERPA (model)
└── runs/b200-need-bank-lr1e5-run01/
    ├── snapshot.json              # 파일별 SHA256, 코드/설정/정책 identity
    └── artifacts/
        ├── checkpoints/           # LoRA, optimizer, CUDA RNG, COMPLETE.json
        ├── run_state.json / run_manifest.json
        ├── batches/               # tree, need, rubric, verification/cache
        ├── cases/ / diagnostics/ / openings/
        ├── evaluation/ / final_accuracy.json
        └── API 사용량·재시도 로그
```

기존과 같은 local-first 저장입니다. HF를 GPU에서 직접 파일시스템처럼 사용하지
않습니다. HF 백업은 추가 네트워크·저장 시간이며 학습/보상 API 비용과 별도입니다.
실행별 `hf_sync.json`에 HF commit SHA와 마지막 전송 시간이 남습니다. HF commit 하나에
manifest와 파일들을 같이 게시하며, 이전 버전은 해당 HF commit으로 조회합니다.
서로 다른 실험/LR에는 다른 `SHERPA_HF_RUN_ID`를 사용하세요. 다른 config·code·경로의
run을 같은 ID에 덮어쓰거나 더 오래된 진도로 덮어쓰는 것은 거부합니다.

HF 저장 실패 시 해당 로컬 배치는 이미 완료된 상태로 중단됩니다. 다시 실행하면
그 checkpoint를 복원하고 백업을 재시도한 뒤 다음 배치로 진행합니다. 업데이트를
다시 적용하지 않습니다. 갑작스런 프로세스 종료·서버 손실에서는 마지막 성공한
HF commit 이후의 미백업 작업은 복원할 수 없습니다.

학습을 중지한 상태에서 추가 백업/파일 규모 확인:

```bash
SHERPA_RUN_DIR=$("$SHERPA_PYTHON" -c 'from src.online_rl.tree.b200_runner import load_config; print(load_config()["output_dir"])')
"$SHERPA_PYTHON" -m src.online_rl.tree.b200_hub plan --run-dir "$SHERPA_RUN_DIR"
"$SHERPA_PYTHON" -m src.online_rl.tree.b200_hub upload --run-dir "$SHERPA_RUN_DIR" --run-id "$SHERPA_HF_RUN_ID"
```

위 명령은 수동 업로드 중 active run의 `runner.lock`을 요구하므로 실행 중인 학습과
동시에 복사하지 않습니다. 자동 백업은 원래 runner가 lock을 가진 barrier에서 실행합니다.

HF에서 다운로드할 때는 `hf_sync.json`의 **40자리 revision SHA**를 지정합니다.
아래 예시는 검사용 새 폴더로 다운로드하며, 기존 폴더를 덮어쓰지 않습니다.

```bash
HF_RUN_REVISION='hf_sync.json에_기록된_40자리_revision'
"$SHERPA_PYTHON" -m src.online_rl.tree.b200_hub download \
  --run-id "$SHERPA_HF_RUN_ID" --revision "$HF_RUN_REVISION" \
  --dest "$HOME/sherpa-restored-run01"
```

Adapter만 사용하려면 복원한 `checkpoints/checkpoint_000120`의 PEFT adapter를 같은
Qwen base model에 로드합니다. **학습 재개**는 전체 artifact와 기존 코드·LR·Doctor
receipt·모델·데이터 경로가 필요합니다. 현재 trainer는 절대 경로/config/source hash를
검사하므로 다른 경로로 다운로드한 폴더를 바로 train에 연결할 수는 없습니다.
같은 원래 output 절대경로가 비어 있는 환경에서는 `--dest`를 그 경로로 지정하고
동일 환경에서 7절 명령으로 재개합니다. 다른 서버/경로로의 identity 재작성은
이 도구에서 자동 수행하지 않습니다. Doctor receipt(`.local/b200_doctor.json`)와
원본 모델·설치 경로도 별도로 보존하세요.

이름 없는 기존 실행에서 HF 동기화를 사용하지 않으려면 시작 전 `unset SHERPA_HF_RUN_ID`를 실행합니다.
새 `SHERPA_RUN_NAME`/`--run-name` 방식은 로컬/HF 이름을 함께 지정하며 HF 백업을 활성화합니다.
스토리지 옵션은 GRPO 학습 설정을 바꾸지 않습니다. Local checkpoint가 먼저 저장되므로
HF 저장을 비활성화하고도 같은 로컬 run을 재개할 수 있습니다.
