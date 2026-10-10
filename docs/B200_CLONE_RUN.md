# B200 새 서버: clone부터 학습·재개·평가까지

이 문서는 `sherpa-jh` 브랜치의 B200 2장 실행용입니다. Doctor는 GPU0와
`127.0.0.1:11985`, Assistant rollout/학습은 GPU1을 사용합니다.
이 서버에서 다른 작업이 해당 GPU/포트를 사용한다면 먼저 별도 자원을 정해야 합니다.
아래 명령은 사용자가 실행합니다. 구현 과정에서는 설치·다운로드·API·GPU 실행을 하지 않았습니다.

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

## 2. 원본 데이터 3개 준비

데이터는 git에 없으므로 clone만으로 준비되지 않습니다. B200에 이미 있다면
아래 복사는 생략하고 3절의 데이터 환경변수에 그 경로를 지정하세요.
그렇지 않으면 B200 터미널에서 현재 편집 서버의 원본을 가져옵니다.
`EDITING_SERVER`에는 실제 접속 가능한 SSH 주소나 SSH config alias를 넣으세요.

```bash
cd "$HOME/Collaborative_agent-sherpa"
EDITING_SERVER='zzuhyun@실제_편집서버_SSH주소'
TRAIN_SOURCE=/convei_nas2/zzuhyun/Collaborative_agent/.worktrees/sherpa-jh/data/interim/medmcqa/five_gpu_train120_seed20261008
TEST_SOURCE=/convei_nas2/zzuhyun/Collaborative_agent/.worktrees/sherpa-jh/data/interim/casimedicos
mkdir -p data/interim/medmcqa/five_gpu_train120_seed20261008 data/interim/casimedicos
scp "${EDITING_SERVER}:${TRAIN_SOURCE}/train120_excluding_calibration_seed20261008.jsonl" data/interim/medmcqa/five_gpu_train120_seed20261008/
scp "${EDITING_SERVER}:${TRAIN_SOURCE}/train120_order_seed20261008.json" data/interim/medmcqa/five_gpu_train120_seed20261008/
scp "${EDITING_SERVER}:${TEST_SOURCE}/all_en_622.jsonl" data/interim/casimedicos/
```

원본 파일과 순서를 그대로 사용하며 preflight가 세 파일의 SHA256과 개수를 검사합니다.
Qwen은 **B200에서 접근 가능한 기존 로컬 Qwen3-8B snapshot**을 사용합니다.
편집 서버의 경로가 B200에도 있다는 가정은 하지 않습니다. 기존 snapshot이 없다면
동일한 파일을 먼저 복사해야 하며, 다른 revision을 자동 다운로드하지 않습니다.

## 3. 키를 제외한 실행 환경 저장

아래 `SHERPA_BASE_MODEL`만 실제 B200의 Qwen snapshot 경로로 바꾸세요.
다른 위치의 데이터/기존 Python 환경을 쓸 때는 해당 export도 바꾸세요.
이 파일은 gitignored이며 API 키는 넣지 않습니다.

```bash
cd "$HOME/Collaborative_agent-sherpa"
mkdir -p .local
cat > .local/b200.env <<'ENV'
export SHERPA_PYTHON="$HOME/Collaborative_agent-sherpa/.venv/bin/python"
export SHERPA_BASE_MODEL="/실제/B200/로컬/Qwen3-8B/snapshot"
export SHERPA_TRAIN_FILE="$HOME/Collaborative_agent-sherpa/data/interim/medmcqa/five_gpu_train120_seed20261008/train120_excluding_calibration_seed20261008.jsonl"
export SHERPA_TRAIN_ORDER="$HOME/Collaborative_agent-sherpa/data/interim/medmcqa/five_gpu_train120_seed20261008/train120_order_seed20261008.json"
export SHERPA_TEST_FILE="$HOME/Collaborative_agent-sherpa/data/interim/casimedicos/all_en_622.jsonl"
export SHERPA_LEARNING_RATE=1e-5
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
CFG=configs/experiments/train120_max4_rubric_only_b200.json
bash scripts/online_rl/launch_b200.sh --config "$CFG" --action preflight
```

최초 pull은 현재 Doctor 모델을 받고 실제 digest를 `.local/b200_doctor.json`에 고정합니다.
재실행하면 기존 pin을 검증하고 mutable tag를 다시 pull하지 않습니다. 모델과 receipt를
삭제하거나 교체하지 않아야 같은 실험을 재개할 수 있습니다.
Preflight의 `blockers: []`, 선택한 `learning_rate`와 `output`을 확인하세요.
Provider의 실제 모델 접근 권한/지원 여부는 CPU preflight가 보증하지 않습니다.
Generator/consolidate는 OpenRouter `openai/gpt-6-luna`, reasoning `none`을 실제 요청하며
미지원 응답은 중단합니다. Gate/scorer는 기존 설정입니다.

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
