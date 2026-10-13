#!/usr/bin/env python3
import argparse, hashlib, importlib, json
from pathlib import Path

HERE = Path(__file__).resolve().parent
IDENTITY = json.loads((HERE / "model_identity.json").read_text())
EXPECTED = {
    "adapter/README.md": "c8f4653f627908dbfe46e562bce92b1fcb49d11cebdd4c338d790a33a274070d",
    "adapter/adapter_config.json": "edc7d9c18b5e6bcd96394d3cee88da1bd78d6d9185eb7efa4a761105eecb4eed",
    "adapter/adapter_model.safetensors": "e81c3975dab9f9cc08965563aacd1b1cfce5dae90cdbb22e5a83849e3acae178",
    "prompts/agent.txt": "5ef5bb764ee955f067e66d1cb8e51950e40c3e7e021994fc8b561cfeaa22942f",
    "prompts/user_simulator_v4_final_label.txt": "026f3771659f97b16f7d8985169478bd0b9eb5c141316a982eb49f3f60cff3a2",
}
VERSIONS = {"torch":"2.6.0+cu124", "transformers":"4.53.3", "peft":"0.15.2", "safetensors":"0.4.4", "accelerate":"1.6.0", "tokenizers":"0.21.4", "numpy":"1.26.4"}

def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--load-model", action="store_true")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    errors = []
    for rel, expected in EXPECTED.items():
        got = sha(HERE / rel)
        if got != expected:
            errors.append(f"{rel}: {got} != {expected}")
    for name, expected in VERSIONS.items():
        got = importlib.import_module(name).__version__
        if got != expected:
            errors.append(f"{name}: {got} != {expected}")
    cfg = json.loads((HERE / "adapter/adapter_config.json").read_text())
    for key, expected in {"base_model_name_or_path":"Qwen/Qwen3-8B", "r":16, "lora_alpha":32, "use_rslora":True}.items():
        if cfg.get(key) != expected:
            errors.append(f"adapter_config.{key}: {cfg.get(key)!r} != {expected!r}")
    from safetensors import safe_open
    with safe_open(HERE / "adapter/adapter_model.safetensors", framework="pt", device="cpu") as weights:
        keys = list(weights.keys())
    if len(keys) != 288 or any("lora_" not in key for key in keys):
        errors.append(f"adapter tensor layout differs: {len(keys)} tensors")
    if args.load_model:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import PeftModel
        base_id = IDENTITY["base_model"]
        tok = AutoTokenizer.from_pretrained(base_id["repo_id"], revision=base_id["revision"], trust_remote_code=True)
        template_hash = hashlib.sha256((tok.chat_template or "").encode()).hexdigest()
        if template_hash != base_id["chat_template_sha256"]:
            errors.append(f"chat_template: {template_hash} != {base_id['chat_template_sha256']}")
        base = AutoModelForCausalLM.from_pretrained(base_id["repo_id"], revision=base_id["revision"], torch_dtype=torch.bfloat16, trust_remote_code=True)
        model = PeftModel.from_pretrained(base, HERE / "adapter", is_trainable=False).to(args.device)
        messages = [{"role":"system", "content":(HERE / "prompts/agent.txt").read_text()}, {"role":"user", "content":"Reply with option A."}]
        rendered = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        if "<think>" in rendered:
            errors.append("thinking mode unexpectedly enabled")
        print(json.dumps({"loaded":True, "device":args.device, "dtype":str(next(model.parameters()).dtype), "chat_template_sha256":template_hash}, indent=2))
    if errors:
        raise SystemExit("REPRODUCIBILITY CHECK FAILED\n- " + "\n- ".join(errors))
    print("REPRODUCIBILITY CHECK PASSED")

if __name__ == "__main__":
    main()
