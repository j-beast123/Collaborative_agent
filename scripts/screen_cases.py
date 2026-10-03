"""Screen TRAIN/DEV cases for effectiveness-label signal before any rollouts."""

from __future__ import annotations

import argparse
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from src.collaboration.generator import case_text, doctor_call, doctor_turn_prompt, option
from src.common.io import read_jsonl, write_jsonl
from src.judges.option_extraction import openrouter_extractor

ROOT = Path(__file__).resolve().parents[1]
BUCKETS = ("challenge", "accept", "unstable", "doctor_undecided", "both_wrong", "saturated")
AGENT_PROMPT = "{case}\n\nAnswer the question. Reply with the letter of the single best option only."


def extract_options(cases, doctor: dict, *, model: str, workers: int, cache_path: Path) -> dict:
    """{case_id: [option each doctor sample favors, "" when none]}, read by the option extractor."""
    extractor = openrouter_extractor(model, cache_path, ROOT)

    def run(case):
        return case["case_id"], [extractor.stated_option(case, text) or ""
                                 for text in doctor[case["case_id"]]["doctor_texts"]]

    with ThreadPoolExecutor(workers) as pool:
        return dict(pool.map(run, cases))


def bucket(p_doctor_correct: float | None, agent_correct: bool) -> str:
    if p_doctor_correct is None:
        return "doctor_undecided"
    if p_doctor_correct == 1.0:
        return "saturated" if agent_correct else "accept"
    if agent_correct:
        return "challenge"
    return "unstable" if p_doctor_correct > 0 else "both_wrong"


def select(rows: list[dict], saturated_share: float, buckets: tuple[str, ...] = ()) -> list[dict]:
    """Cases of `buckets` (default: every non-saturated bucket) round-robin over buckets, so
    any prefix mixes them, with saturated cases spread in at `saturated_share` of the selection."""
    queues = {b: [r for r in rows if r["bucket"] == b] for b in (buckets or BUCKETS) if b != "saturated"}
    mixed = []
    while any(queues.values()):
        for rows_in_bucket in queues.values():
            if rows_in_bucket:
                mixed.append(rows_in_bucket.pop(0))
    saturated = [r for r in rows if r["bucket"] == "saturated"]
    keep = min(len(saturated), round(len(mixed) * saturated_share / (1 - saturated_share))) if saturated_share < 1 else 0
    if not keep:
        return mixed
    step = max(1, len(mixed) // keep)
    out, used = [], 0
    for i, row in enumerate(mixed):
        out.append(row)
        if used < keep and (i + 1) % step == 0:
            out.append(saturated[used])
            used += 1
    return out


def screen_doctor(cases, *, model: str, samples: int, rounds: int, workers: int, cache_path: Path) -> dict:
    """First doctor turn, `samples` times per case. Cached per case in cache_path."""
    cache = {r["case_id"]: r for r in read_jsonl(cache_path, missing_ok=True)}
    todo = [c for c in cases if c["case_id"] not in cache]
    system = (ROOT / "prompts/user_simulator.txt").read_text(encoding="utf-8")
    key = os.environ.get("OPENROUTER_API_KEY")
    if todo and not key:
        raise SystemExit("OPENROUTER_API_KEY is not set")

    def run(case):
        prompt = doctor_turn_prompt(case, [], 1, rounds)
        texts = [doctor_call(key, model, system, prompt, 512) for _ in range(samples)]
        return {"case_id": case["case_id"], "doctor_texts": texts}

    with cache_path.open("a", encoding="utf-8") as handle, ThreadPoolExecutor(workers) as pool:
        for n, row in enumerate(pool.map(run, todo), 1):
            cache[row["case_id"]] = row
            handle.write(json.dumps(row) + "\n")
            handle.flush()
            if n % 50 == 0:
                print(f"doctor {n}/{len(todo)}", flush=True)
    return cache


def screen_agent(cases, *, model_path: str, batch_size: int, cache_path: Path) -> dict:
    """The agent's own greedy answer (letter only), batched. Cached per case."""
    cache = {r["case_id"]: r for r in read_jsonl(cache_path, missing_ok=True)}
    todo = [c for c in cases if c["case_id"] not in cache]
    if not todo:
        return cache
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float16, device_map="auto",
                                                 trust_remote_code=True).eval()
    with cache_path.open("a", encoding="utf-8") as handle:
        for start in range(0, len(todo), batch_size):
            batch = todo[start : start + batch_size]
            prompts = [tokenizer.apply_chat_template([{"role": "user", "content": AGENT_PROMPT.format(case=case_text(c))}],
                                                     tokenize=False, add_generation_prompt=True, enable_thinking=False)
                       for c in batch]
            inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
            with torch.inference_mode():
                output = model.generate(**inputs, max_new_tokens=64, do_sample=False,
                                        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
            for case, ids in zip(batch, output[:, inputs["input_ids"].shape[-1] :]):
                text = tokenizer.decode(ids, skip_special_tokens=True).strip()
                bare = re.fullmatch(r"\W*([A-D])\W*", text)
                row = {"case_id": case["case_id"], "agent_option": bare.group(1) if bare else option(text),
                       "agent_raw": text}
                cache[row["case_id"]] = row
                handle.write(json.dumps(row) + "\n")
            handle.flush()
    del model
    torch.cuda.empty_cache()
    return cache


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cases", required=True)
    p.add_argument("--split", choices=("train", "dev"), required=True, help="EVAL cases are never screened")
    p.add_argument("--doctor-model", required=True)
    p.add_argument("--agent-model", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--doctor-samples", type=int, default=3)
    p.add_argument("--rounds", type=int, default=3, help="rounds of the later collaboration (shown in the prompt)")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-cases", type=int)
    p.add_argument("--saturated-share", type=float, default=0.15,
                   help="share of saturated cases kept in the selection (the predictor must still see them)")
    p.add_argument("--select-buckets", nargs="+", choices=[b for b in BUCKETS if b != "saturated"], default=[],
                   help="buckets that enter the selection (default: all non-saturated)")
    a = p.parse_args()

    cases = read_jsonl(a.cases)[: a.max_cases]
    if any("eval" in str(c.get("split", "")) for c in cases) or "eval" in Path(a.cases).name:
        raise SystemExit("EVAL cases are never screened")
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    doctor = screen_doctor(cases, model=a.doctor_model, samples=a.doctor_samples, rounds=a.rounds,
                           workers=a.workers, cache_path=out.with_name(out.stem + "_doctor.jsonl"))
    extracted = extract_options(cases, doctor, model=a.doctor_model, workers=a.workers,
                                cache_path=out.with_name(out.stem + "_option_cache.jsonl"))
    agent = screen_agent(cases, model_path=a.agent_model, batch_size=a.batch_size,
                         cache_path=out.with_name(out.stem + "_agent.jsonl"))

    rows = []
    for case in cases:
        gold = case["ground_truth"]["correct_option"]
        options = extracted[case["case_id"]]
        stated = [o for o in options if o]  # a turn that states no option is not counted as wrong
        p_doctor = sum(o == gold for o in stated) / len(stated) if stated else None
        agent_option = agent[case["case_id"]]["agent_option"]
        rows.append({"case_id": case["case_id"], "split": a.split, "gold": gold, "doctor_options": options,
                     "p_doctor_correct": p_doctor, "agent_option": agent_option,
                     "agent_correct": agent_option == gold, "bucket": bucket(p_doctor, agent_option == gold)})
    write_jsonl(out, rows)

    by_id = {c["case_id"]: c for c in cases}
    selected = select(rows, a.saturated_share, tuple(a.select_buckets))
    selected_path = out.with_name(out.stem + "_selected_cases.jsonl")
    write_jsonl(selected_path, (by_id[r["case_id"]] for r in selected))
    counts = {b: sum(r["bucket"] == b for r in rows) for b in BUCKETS}
    print(json.dumps({"cases": len(rows), "buckets": counts, "selected": len(selected),
                      "selected_cases": str(selected_path)}, indent=2))


if __name__ == "__main__":
    main()
