# B200 Doctor 동시 요청

`launch_b200_v7_doctor_batch.sh`는 GPU0의 frozen Gemma BF16 **한 복사본**으로
여러 Doctor 요청을 함께 생성합니다. Assistant rollout/GRPO는 GPU1을 그대로 사용합니다.
모델은 실행 시작 시 로드되고 모든 batch 동안 상주합니다. 프로세스를 종료하면 내려갑니다.

이번 변경은 Doctor 실행에만 적용합니다. Rubric 생성/반영 순서, generator prompt/API 모델,
reflection 내용, verification 판정, rollout의 턴/종료 조건, rubric-only 보상,
train120 순서, frozen-policy 5-case 업데이트, Assistant 설정은 유지합니다.
Rollout 도중 rubric을 먼저 생성하는 기능은 추가하지 않았습니다.

## 요청 처리와 설정

새 config: `configs/experiments/train120_max4_v7_doctor_batch_b200.json`.

- `doctor_batching.max_batch_size: 8`: 준비된 요청을 최대 8개까지 묶습니다.
- `doctor_batching.wait_ms: 15`: 첫 요청 이후 묶음을 모으는 짧은 대기 시간입니다.
- `doctor_batching.max_padded_tokens: 32768`: 묶음의 `행 수 × (최장 입력 + 최대 출력 예산)` 상한.
- `execution.doctor_workers`, `five_gpu.doctor_concurrency`, `two_gpu.doctor_concurrency`: 모두 8.
- API 전역 동시성은 기존 50. Doctor 배치 크기와 별개입니다.

긴 입력은 묶음을 더 작게 나눕니다. 입력을 잘라 넣지 않으며 각 요청의 기존 8192 context
제한을 검사합니다. 새로운 요청을 만들어 슬롯을 채우지 않습니다. 기본 max8은 실제
192GB GPU에서 메모리 사용량을 측정한 최댓값이 아닙니다. 실제 GPU 검증은 실행자가 수행합니다.
상한을 바꿀 때는 위 동시성 값들을 함께 수정하고, 실행 중 설정을 바꾸지 마세요.

이 구현은 묶음 단위 추론입니다. 진행 중인 묶음에 새 요청을 끼워 넣는 continuous batching은
아닙니다. Rollout·FINAL·reflection·verification·최종 평가가 같은 모델/queue를 사용합니다.
반사실 검증의 첫 2회/필요 시 3회 판정 로직은 그대로입니다.

HF Gemma의 `generate`와 KV-cache/mask 처리를 사용하고, 각 행의 독립 `torch.Generator`
(seed는 기존 요청 seed)를 이용해 temperature=0.4, top_p=1, top_k=0의 토큰을 선택합니다.
이 토큰만 남긴 logits를 greedy outer loop에 전달합니다. 따라서 내부 `do_sample=False`는
실험을 greedy decoding으로 바꾼다는 뜻이 아닙니다. GPU 전역 난수 상태를 재설정하지 않습니다.
요청마다 출력 길이/EOS를 따로 처리하고, padding·길이 종료용 filler는 응답/토큰 기록에서 제외합니다.
GPU OOM/빈 응답 등은 오류로 전파하며, 자동 backend 변경이나 0점 대체는 하지 않습니다.

## 기존 run과 구분

Sampling protocol은 `hf_doctor_batched_per_request_rng_v1`입니다. 같은 seed라도
배치 수치연산으로 기존 직렬 HF 실행과 완전히 같은 텍스트가 나온다고 보장하지 않습니다.
Doctor protocol 및 opening/cache identity에 배치 설정을 포함하고 **새 run 이름**으로 실행합니다.
기존 `gemma-bf16-v7-rubric-lr5e6-run01`에 새 config를 덮어 재개하면 config 검사가 거부합니다.
기존 출력/캐시/체크포인트는 보존됩니다. 원래 launcher/config로 기존 run을 재개하는 경로도
정확한 코드 hash를 대조한 경우에만 유지합니다.

## 실행

공유 checkout이면 양쪽 학습 프로세스를 중단한 뒤 한 곳에서 pull합니다. 두 서버가 같은
스토리지를 쓰므로 run 이름은 반드시 다르게 사용하세요. 두 서버는 각자의 물리 GPU0/1을 씁니다.
키는 각 터미널 환경에만 설정하고 파일에 넣지 않습니다.

```bash
cd /NHNHOME/shkim/SHERPA/Collaborative_agent-sherpa
git pull --ff-only origin sherpa-jh
source .local/b200.env
export SHERPA_BASE_MODEL=/NHNHOME/shkim/SHERPA/vllm/qwen3
export SHERPA_DOCTOR_BACKEND=hf_local
export SHERPA_DOCTOR_MODEL=/NHNHOME/shkim/SHERPA/vllm/gemma3
export SHERPA_HF_DOCTOR_RECEIPT="$PWD/.local/b200_hf_doctor.json"
export CUDA_VISIBLE_DEVICES=0,1
```

새 터미널에서 키가 없을 때만:

```bash
read -rsp 'OpenAI key: ' OPENAI_API_KEY; echo
export OPENAI_API_KEY
read -rsp 'HF token: ' HF_TOKEN; echo
export HF_TOKEN
```

서버 B (`5e-6`):

```bash
export SHERPA_RUN_NAME=gemma-bf16-v7-doctorbatch8-lr5e6-run01
export SHERPA_LEARNING_RATE=5e-6
bash scripts/online_rl/launch_b200_v7_doctor_batch.sh --action preflight
bash scripts/online_rl/launch_b200_v7_doctor_batch.sh --action train --max-batches 1
```

서버 A (`1e-5`)에서는 위 두 export만 다음처럼 바꾸고 같은 preflight/첫 배치 명령을 실행합니다.

```bash
export SHERPA_RUN_NAME=gemma-bf16-v7-doctorbatch8-lr1e5-run01
export SHERPA_LEARNING_RATE=1e-5
```

첫 배치 확인 후 전체 학습 / 중단 후 재개(동일 명령, 같은 환경과 run 이름 유지):

```bash
bash scripts/online_rl/launch_b200_v7_doctor_batch.sh --action train
```

Train120 완료 후 최종 checkpoint로 test622 1회 평가:

```bash
bash scripts/online_rl/launch_b200_v7_doctor_batch.sh --action evaluate
```

HF 저장소는 기존 환경의 `zzu819/SHERPA`, 경로는 `runs/$SHERPA_RUN_NAME`입니다.
로컬 출력/체크포인트/캐시는 `outputs/runs/$SHERPA_RUN_NAME/` 아래에 분리됩니다.

## 실제 배치 확인

`hf_doctor_calls.jsonl`의 `batch_id`, `batch_size`, `input_tokens`, `output_tokens`,
`request_seconds`, `queue_wait_seconds`, `seed`, `messages_sha256`, `sampling_protocol`을 확인합니다.
같은 batch_id의 request_seconds는 같은 GPU 작업 시간입니다. 행별로 합산해 GPU 시간을
중복 계산하지 마세요. API 비용과 Doctor GPU 시간은 별도입니다.

```bash
"$SHERPA_PYTHON" - <<'PY'
import json, os
from collections import Counter
from pathlib import Path
p = Path('outputs/runs') / os.environ['SHERPA_RUN_NAME'] / 'hf_doctor_calls.jsonl'
rows = [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
batches = {r['batch_id']: r for r in rows if r.get('batch_id')}
print('requests:', len(rows), 'batches:', len(batches))
print('batch_size_counts:', dict(Counter(r['batch_size'] for r in batches.values())))
print('unique_batch_seconds:', sum(r['request_seconds'] for r in batches.values()))
print('errors:', sum(bool(r.get('error')) for r in rows))
PY
```

CPU 검증(실제 모델 다운로드/API/GPU 실행 없음):

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  "$SHERPA_PYTHON" -m pytest -q tests/test_doctor_batch.py tests/test_hf_doctor.py \
  tests/test_batched_generation.py tests/test_b200_v7.py tests/test_b200_runner.py \
  tests/test_consolidate_repair.py tests/test_b200_partner.py tests/test_b200_posthoc_only.py \
  tests/test_parallel_bank.py tests/test_rubric_quote_repair.py
```

Doctor 동시성과 별도로 generator JSON 문법 오류의 제한된 복구 기능도 제공됩니다.
[JSON 오류 복구 및 동일 run 재개](B200_V7_RUBRIC_ONLY.md#generator-json-문법-오류-복구)를
참고하세요. 복구할 수 없는 출력은 오류 기록을 남기고 중단합니다.
