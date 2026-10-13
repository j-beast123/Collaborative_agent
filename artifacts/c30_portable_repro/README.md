# Portable LoRA inference bundle

This bundle pins the inputs that can change cross-server behavior: the exact Qwen3-8B Hub revision, complete LoRA adapter, RS-LoRA configuration, package versions, chat-template hash, thinking mode, BF16 precision, decoding settings, prompts, and Doctor model digest.

## Install and verify

Use Python 3.11 and CUDA 12.4-compatible PyTorch. Install `requirements.lock.txt`, then cache and verify the exact base revision:

```bash
huggingface-cli download Qwen/Qwen3-8B --revision b968826d9c46dd6066d109eabc6255188de91218
python artifacts/c30_portable_repro/verify_environment.py
CUDA_VISIBLE_DEVICES=0 python artifacts/c30_portable_repro/verify_environment.py --load-model
```

The first verifier checks files, adapter configuration, and package versions. `--load-model` also resolves the pinned Hub commit, verifies the tokenizer chat template, loads the LoRA with PEFT, enforces BF16, and renders a prompt with `enable_thinking=False`.

## Required inference contract

- Base: `Qwen/Qwen3-8B` revision `b968826d9c46dd6066d109eabc6255188de91218`
- Adapter: `adapter/`, with `use_rslora=true`
- Agent precision: BF16 without quantization
- Chat rendering: `apply_chat_template(..., add_generation_prompt=True, enable_thinking=False)`
- Agent decoding: temperature 0.7, top-p 0.9, model-default top-k, maximum 400 new tokens
- Doctor: the exact Ollama digest and options in `model_identity.json`
- Expected original-runtime trainable-policy hash: `bbbfc75889508d0438094559c6c3066759c1d4a35548f75f86125c56b68e8299`

`trainer_state.pt` and CUDA RNG state are intentionally excluded because they are needed to resume training, not inference; trainer state also exceeds GitHub's single-file limit. The complete 61 MB adapter is included.

The copied config, evaluation manifest, and runtime sources preserve the evaluated call path. Absolute paths in the historical config are provenance and should be replaced with local paths on the destination server.
