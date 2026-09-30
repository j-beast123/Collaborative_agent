"""Run one zero-shot standalone answer per Medicine case with Qwen3."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/home/kai/models/Qwen3-8B")
    parser.add_argument("--input", default="data/interim/medmcqa/medicine_sample_500.jsonl")
    parser.add_argument("--output", default="outputs/medicine_standalone_qwen3_8b.jsonl")
    parser.add_argument("--max-cases", type=int, default=500)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def extract_option(text: str, option_ids: set[str]) -> str:
    text = text.strip().upper()
    if text in option_ids:
        return text
    parsed = re.search(r'"(?:selected_option|answer|current_judgment)"\s*:\s*"?([A-D])', text, re.I)
    if parsed and parsed.group(1).upper() in option_ids:
        return parsed.group(1).upper()
    matches = re.findall(
        r"(?:final answer|correct answer|answer|option|choice)"
        r"\s*(?:is\s*)?[:\-]?\s*(?:\*\*)?([A-D])\b",
        text,
        re.I,
    )
    for match in reversed(matches):
        if match.upper() in option_ids:
            return match.upper()
    return ""


def build_prompt(case: dict) -> str:
    options = "\n".join(f"{key}. {value}" for key, value in case["options"].items())
    return (
        f"Question: {case['question']}\n"
        f"{options}\n"
        "Output exactly one uppercase letter.\n"
        "Valid outputs are only: A, B, C, or D.\n"
        "Do not output any explanation.\n"
        "Answer:"
    )


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    all_cases = load_jsonl(root / args.input)[: args.max_cases]
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index must be in [0, num-shards)")
    cases = all_cases[args.shard_index :: args.num_shards]
    output_path = root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    with output_path.open("w", encoding="utf-8") as handle:
        for index, case in enumerate(cases):
            messages = [{"role": "user", "content": build_prompt(case)}]
            try:
                prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
                )
            except TypeError:
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt, return_tensors="pt")
            inputs = {key: value.to(model.device) for key, value in inputs.items()}
            with torch.inference_mode():
                output = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            generated = output[0][inputs["input_ids"].shape[-1] :]
            raw = tokenizer.decode(generated, skip_special_tokens=True).strip()
            selected = extract_option(raw, set(case["options"]))
            correct = selected == str(case["ground_truth"]["correct_option"]).upper()
            record = {
                "case_id": case["case_id"],
                "subject": case.get("subject", ""),
                "topic": case.get("topic", ""),
                "selected_option": selected,
                "ground_truth": case["ground_truth"]["correct_option"],
                "is_correct": correct,
                "raw_response": raw,
                "model": args.model,
                "decoding": {"mode": "zero_shot", "do_sample": False, "max_new_tokens": args.max_new_tokens},
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"{index + 1}/{len(cases)} option={selected or '?'} correct={correct}", flush=True)
    print(f"saved {output_path}")


if __name__ == "__main__":
    main()
