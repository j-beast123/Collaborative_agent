# C30 evaluation handoff

This bundle contains the artifacts needed to reproduce the saved C30 evaluation protocol.

## Frozen identities

- Repository anchor: `j-beast123/Collaborative_agent`, commit `d7b9556`
- Base Agent: `Qwen/Qwen3-8B`, Hugging Face revision `b968826d9c46dd6066d109eabc6255188de91218`
- LoRA checkpoint: C30, policy SHA-256 `bbbfc75889508d0438094559c6c3066759c1d4a35548f75f86125c56b68e8299`
- Doctor: `gemma3:12b-it-q8_0`, Ollama digest `997a7c2c09753e9011ec18a3fcb498e27c0ba330599dfb26a2e05b58cfbdc1bc`
- Ollama: `0.12.5`
- Evaluation set: CasiMedicos `all_en_622`, SHA-256 `e42077d006c4e8227564f99b06050b66e7df9989f9b2b17f5fc5d9dd083feb56`
- Protocol: maximum four Agent turns; early stopping only after at least two Agent turns
- Saved result: 442/622 = 71.0609%

The original run did not record a Git commit. `d7b9556` is the closest repository anchor on the active branch after this run and contains the matching portable five-GPU pipeline. Artifact and prompt hashes in `SHA256SUMS` are the authoritative identities.

## Transfer

Copy this bundle or its archive to the other server. Do not put API keys or `.env` files in the bundle.

For GitHub, keep source/config/prompt files in Git and upload `adapter/adapter_model.safetensors` as a GitHub Release asset or through Git LFS. The evaluation dataset may need private transfer depending on its redistribution terms.

The base model is not included. Download the pinned Hugging Face revision on the destination server. The Doctor model is also not included; install Ollama 0.12.5 and verify the exact model digest before evaluating.

`trainer_state.pt` is intentionally omitted because it is required only to resume training, not to evaluate the LoRA adapter.
