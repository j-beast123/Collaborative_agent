"""Prompts must not depend on the process (string hash randomization changes set order)."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = r"""
import hashlib, json, sys
sys.path.insert(0, "tests")
import src.collaboration.generator as g
from test_serl_steps import fake_runtime, rich_trajectory
from src.collaboration.rerun import continue_collaboration

prompts = []
real_doctor_call = g.doctor_call
valid = json.dumps({f: ([] if f.endswith(("so_far", "evidence", "points", "details")) else "x")
                    for f in g.RECONSTRUCTION_FIELD_ORDER})
g.doctor_call = lambda key, model, system, user, max_tokens, temperature=0.4: (prompts.append(user), valid)[1]
case = {"question": "q", "options": {"A": "a", "B": "b"}, "scenario": ""}
g.reconstruct_call("k", "m", case, [{"speaker": "doctor", "text": "t", "round": 1}], 10)

runtime, _ = fake_runtime(agent_text="reply")
agent = runtime.agent
runtime.agent = lambda system, user, tokens: (prompts.append(user), agent(system, user, tokens))[1]
doctor = runtime.doctor
runtime.doctor = lambda system, user: (prompts.append(user), doctor(system, user))[1]
continue_collaboration(rich_trajectory("t1"), 3, runtime)
print(len(prompts), hashlib.sha256("\x00".join(prompts).encode()).hexdigest())
"""


def run(seed: str) -> str:
    return subprocess.run([sys.executable, "-c", SCRIPT], cwd=ROOT, capture_output=True, text=True, check=True,
                          env={"PYTHONHASHSEED": seed, "PATH": "/usr/bin:/bin"}).stdout.strip()


def test_prompts_are_identical_across_hash_seeds():
    outputs = {run(seed) for seed in ("1", "2", "3")}
    assert len(outputs) == 1, outputs
    assert int(outputs.pop().split()[0]) >= 4  # reconstruction, judgment, agent and doctor prompts were built
