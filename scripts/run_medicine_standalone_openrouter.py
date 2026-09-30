"""Run zero-shot standalone MedMCQA cases with an OpenRouter chat model."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import time
import urllib.request
import urllib.error
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/gemma-3-4b-it")
    parser.add_argument("--input", default="data/interim/medmcqa/medicine_sample_500.jsonl")
    parser.add_argument("--output", default="outputs/medicine_standalone_gemma_api.jsonl")
    parser.add_argument("--max-cases", type=int, default=500)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--retries", type=int, default=4)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


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


def extract_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {}
    try:
        value = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def extract_option(text: str) -> str:
    text = text.strip().upper()
    if text in {"A", "B", "C", "D"}:
        return text
    match = re.search(r"(?:answer|option|choice)\s*[:\-]?\s*([A-D])\b", text, re.I)
    if match:
        return match.group(1).upper()
    matches = re.findall(r"\b([A-D])\b", text.upper())
    return matches[-1] if matches else ""


def call_openrouter(api_key: str, model: str, prompt: str, max_tokens: int, retries: int) -> str:
    payload = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/j-beast123/clinicalqa-en",
            "X-Title": "clinicalqa-en standalone evaluation",
        },
        method="POST",
    )
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                body = json.loads(response.read().decode("utf-8"))
            return body["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as error:
            if error.code not in {429, 500, 502, 503, 504} or attempt >= retries:
                raise
            retry_after = error.headers.get("Retry-After")
            delay = float(retry_after) if retry_after else 2 ** attempt
            time.sleep(min(delay, 120))
        except Exception:
            if attempt >= retries:
                raise
            time.sleep(2 ** attempt)


def process_case(case: dict, api_key: str, model: str, max_tokens: int, retries: int) -> dict:
    raw = call_openrouter(api_key, model, build_prompt(case), max_tokens, retries)
    parsed = extract_json(raw)
    selected = str(parsed.get("selected_option", "")).strip().upper() or extract_option(raw)
    correct = selected == str(case["ground_truth"]["correct_option"]).upper()
    return {
        "case_id": case["case_id"],
        "subject": case.get("subject", ""),
        "topic": case.get("topic", ""),
        "selected_option": selected if selected in case["options"] else "",
        "ground_truth": case["ground_truth"]["correct_option"],
        "is_correct": correct,
        "reason": parsed.get("reason", ""),
        "raw_response": raw,
        "model": model,
        "decoding": {"mode": "zero_shot", "temperature": 0, "max_tokens": max_tokens},
    }


def main() -> None:
    args = parse_args()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY is not set")

    root = Path(__file__).resolve().parents[1]
    cases = load_jsonl(root / args.input)[: args.max_cases]
    output_path = root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    existing = {}
    if output_path.exists():
        for line in output_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                existing[record["case_id"]] = record
    cases = [case for case in cases if case["case_id"] not in existing]
    mode = "a" if output_path.exists() else "w"

    with output_path.open(mode, encoding="utf-8") as handle:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(process_case, case, api_key, args.model, args.max_tokens, args.retries): index
                for index, case in enumerate(cases, start=1)
            }
            completed = 0
            for future in concurrent.futures.as_completed(futures):
                index = futures[future]
                try:
                    record = future.result()
                except Exception as error:
                    record = {
                        "case_id": cases[index - 1]["case_id"],
                        "model": args.model,
                        "error": f"{type(error).__name__}: {error}",
                    }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                completed += 1
                print(
                    f"{completed}/{len(cases)} case={index} "
                    f"option={record.get('selected_option', '?')} "
                    f"correct={record.get('is_correct', '?')} "
                    f"error={record.get('error', '')}",
                    flush=True,
                )

    print(f"saved {output_path}")


if __name__ == "__main__":
    main()
