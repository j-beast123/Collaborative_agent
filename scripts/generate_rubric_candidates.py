"""Extract reusable interaction-failure rubric candidates from failed trajectories."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="outputs/collaboration_gemma12b_500.jsonl")
    p.add_argument("--prompt", default="prompts/rubric_evaluator.txt")
    p.add_argument("--output", default="outputs/rubric_candidates/rubric_candidates_gpt4o.jsonl")
    p.add_argument("--model", default="openai/gpt-4o")
    p.add_argument("--max-failures", type=int, default=20)
    p.add_argument("--max-tokens", type=int, default=900)
    return p.parse_args()


def call_openrouter(key: str, model: str, system: str, user: str, max_tokens: int) -> str:
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "temperature": 0,
        "max_tokens": max_tokens,
    }).encode()
    request = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=payload,
        headers={"Authorization": f"Bearer {key.strip()}", "Content-Type": "application/json"},
        method="POST",
    )
    for attempt in range(6):
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                body = json.loads(response.read().decode())
            return body["choices"][0]["message"]["content"].strip()
        except urllib.error.HTTPError as error:
            if error.code not in {429, 500, 502, 503, 504} or attempt == 5:
                raise
            retry_after = error.headers.get("Retry-After")
            delay = float(retry_after) if retry_after else min(60, 2**attempt)
            print(f"HTTP {error.code}; retrying in {delay:.0f}s", flush=True)
            time.sleep(delay)
    raise RuntimeError("OpenRouter request failed")


def parse_json(text: str) -> dict:
    try:
        value = json.loads(text.strip())
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return {"parse_status": "invalid", "raw_response": text}
        try:
            value = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {"parse_status": "invalid", "raw_response": text}
    if not isinstance(value, dict):
        return {"parse_status": "invalid", "raw_response": text}
    value["parse_status"] = "valid"
    value["raw_response"] = text
    return value


def main() -> None:
    args = parse_args()
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise SystemExit("OPENROUTER_API_KEY is not set")
    root = Path(__file__).resolve().parents[1]
    system = (root / args.prompt).read_text(encoding="utf-8")
    rows = [json.loads(line) for line in (root / args.input).read_text(encoding="utf-8").splitlines() if line.strip()]
    failures = [row for row in rows if not row.get("is_correct", False)][: args.max_failures]
    output = root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for index, trajectory in enumerate(failures, 1):
            user = (
                "Analyze this failed collaboration trajectory and extract exactly one reusable interaction failure. "
                "Do not define failure as final incorrectness alone.\n\n"
                + json.dumps(trajectory, ensure_ascii=False)
            )
            candidate = parse_json(call_openrouter(key, args.model, system, user, args.max_tokens))
            candidate.update({
                "source_case_id": trajectory.get("case_id"),
                "source_trajectory_id": trajectory.get("trajectory_id"),
                "evaluator_model": args.model,
            })
            handle.write(json.dumps(candidate, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"{index}/{len(failures)} {trajectory.get('case_id')} parse={candidate['parse_status']}", flush=True)
    print(f"saved {output}")


if __name__ == "__main__":
    main()
