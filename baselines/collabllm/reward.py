"""Score existing leaves once and preserve the original descendant-leaf mean.

No model rollout or optimizer is called here. Additional conversations = zero.
CallCache must have one writer process; its thread-level single-flight and durable
results let repeated leaves/ancestors/resumed scoring reuse the same judgment.
"""
from __future__ import annotations

import json
import math
import hashlib
from dataclasses import asdict
from pathlib import Path

from src.judges.llm import ChatRequest
from src.online_rl.integrated.records import sha256_json, sha256_text
from src.online_rl.tree.team import node_credit

METHOD = "CollabLLM (adapted, GRPO)"
PROTOCOL = "collabllm_existing_terminal_tree_v2"
PROMPT = Path(__file__).with_name("interactivity.txt").read_text()
PARAMS = {
    "temperature": 0.0,
    "max_tokens": 512,
    "response_format": {"type": "json_schema", "json_schema": {
        "name": "collabllm_interactivity", "strict": True,
        "schema": {"type": "object", "additionalProperties": False,
                   "required": ["thought", "interactivity"], "properties": {
                       "thought": {"type": "string"},
                       "interactivity": {"type": "number", "minimum": 0, "maximum": 1}}}}},
}


def number(value, label, low=0.0, high=math.inf):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or not low <= value <= high):
        raise ValueError(f"{label} must be finite and in [{low}, {high}]")
    return float(value)


def visible_discussion(leaf):
    """Read the original visible dialogue only, never the trace or case/gold."""
    messages = []
    for speaker, stage, round_, text in leaf.dialogue:
        if stage == "FINAL":
            continue
        if (speaker not in ("doctor", "agent") or round_ < 1
                or stage != ("D" if speaker == "doctor" else "A") + str(round_)):
            raise ValueError(f"Unexpected visible dialogue entry: {speaker}/{stage}")
        if not isinstance(text, str):
            raise ValueError("Dialogue text must be a string")
        messages.append({"role": "user" if speaker == "doctor" else "assistant", "content": text})
    if not messages or not any(m["role"] == "assistant" for m in messages):
        raise ValueError("A terminal conversation needs an assistant reply")
    return messages


class Cl100kCounter:
    def __init__(self):
        import tiktoken
        from importlib.metadata import version
        spec = json.loads(Path(__file__).with_name("adaptation_spec.json").read_text())
        pin = spec["reward"]["tokenizer_pin"]
        if version("tiktoken") != pin["tiktoken_version"]:
            raise ValueError("tiktoken version differs from the frozen tokenizer definition")
        self.encoding = tiktoken.get_encoding("cl100k_base")
        definition = {"name": self.encoding.name, "pattern": self.encoding._pat_str,
                      "mergeable_ranks": sorted((k.hex(), v) for k, v in self.encoding._mergeable_ranks.items()),
                      "special_tokens": self.encoding._special_tokens}
        digest = hashlib.sha256(json.dumps(definition, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        if digest != pin["definition_sha256"]:
            raise ValueError("cl100k_base vocabulary/pattern/special tokens changed")
        self.identity = {"encoding": "cl100k_base", "tiktoken_version": version("tiktoken"),
                         "definition_sha256": digest,
                         "text": "space_joined_visible_assistant_messages"}

    def __call__(self, text):
        return len(self.encoding.encode(text, disallowed_special=()))


class InteractivityJudge:
    """Use the existing backend's bounded transport retries; invalid JSON aborts.

    Persist every returned attempt before validation for usage/error accounting.
    The backend's revision field is a provenance declaration; returned model ID
    is also recorded. Provider routing/revision availability needs preflight.
    """
    def __init__(self, backend, audit_path):
        self.backend = backend
        backend_id = backend.identity()
        if not backend_id.get("revision"):
            raise ValueError("Pin the judge revision before scoring")
        self.audit_path = Path(audit_path)
        self.identity = {"backend": backend_id, "prompt_sha256": sha256_text(PROMPT),
                         "params": PARAMS, "protocol": PROTOCOL}

    def __call__(self, messages):
        user = json.dumps({"conversation": messages}, ensure_ascii=False)
        result = self.backend.complete(ChatRequest(
            system=PROMPT, user=user, prompt_version=PROTOCOL, params=PARAMS))
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        with self.audit_path.open("a") as stream:
            stream.write(json.dumps({"judge": self.identity, "input_sha256": sha256_text(user),
                                     "result": result.to_dict()}, ensure_ascii=False) + "\n")
        if result.status != "ok":
            raise RuntimeError(f"Interactivity judge failed: {result.error}")
        obj = json.loads(result.text)
        if not isinstance(obj, dict) or set(obj) != {"thought", "interactivity"}:
            raise ValueError("Invalid interactivity JSON fields")
        if not isinstance(obj["thought"], str):
            raise ValueError("Interactivity thought must be a string")
        number(obj["interactivity"], "interactivity", high=1)
        return {**obj, "actual_model": result.model, "usage": result.usage,
                "attempts": result.attempts, "latency_s": result.latency_s}


def validate_tree(tree):
    nodes = {n.node_id: n for n in tree.nodes}
    if len(nodes) != len(tree.nodes) or len({l.leaf_id for l in tree.leaves}) != len(tree.leaves):
        raise ValueError("Duplicate node or leaf IDs")
    for n in tree.nodes:
        if n.case_id != tree.case_id or n.policy_sha256 != tree.policy_sha256:
            raise ValueError("Mixed case or policy in tree")
    for leaf in tree.leaves:
        if leaf.case_id != tree.case_id or not leaf.path or leaf.path[-1] != leaf.leaf_id:
            raise ValueError("Leaf identity differs from its path/case")
        parent = None
        for nid in leaf.path:
            if nid not in nodes or nodes[nid].parent_id != parent:
                raise ValueError("Invalid leaf ancestry")
            parent = nid
        if leaf.outcome is not None and (isinstance(leaf.outcome, bool) or leaf.outcome not in (0., 1.)):
            raise ValueError("Expected binary outcome or None")


def score_tree(tree, *, judge, cache, token_counter):
    """One cache lookup per valid leaf, never per ancestor. Keep leaf multiplicity."""
    validate_tree(tree)
    rows = {}
    hits_before = cache.hits.get(PROTOCOL, 0)
    calls_before = cache.calls.get(PROTOCOL, 0)
    for leaf in tree.leaves:
        if leaf.outcome is None:
            rows[leaf.leaf_id] = {"status": "unparsed_final", "success": None}
            continue
        messages = visible_discussion(leaf)
        identity = {"judge": judge.identity, "messages": messages}
        value = cache.call(PROTOCOL, identity, lambda: _judge(judge, messages))
        # Also validate values read from a persisted cache.
        interactivity = number(value["interactivity"], "interactivity", high=1)
        text = " ".join(m["content"] for m in messages if m["role"] == "assistant")
        tokens = token_counter(text)
        if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens < 0:
            raise ValueError("Token counter must return a nonnegative integer")
        rows[leaf.leaf_id] = {
            "status": "scored", "success": leaf.outcome, "interactivity": interactivity,
            "agent_visible_tokens": tokens, "agent_ktokens": tokens / 1000,
            "terminal_sha256": sha256_json(leaf.dialogue),
            "judge_cache_key": sha256_json([PROTOCOL, identity]),
        }
    return {"schema": PROTOCOL, "method_label": METHOD, "case_id": tree.case_id,
            "policy_sha256": tree.policy_sha256, "tree_sha256": sha256_json(asdict(tree)),
            "judge": judge.identity, "tokenizer": token_counter.identity, "leaves": rows,
            "cost": {"additional_conversations": 0, "additional_agent_generations": 0,
                     "additional_doctor_generations": 0,
                     "new_successful_judgments": cache.calls.get(PROTOCOL, 0) - calls_before,
                     "judgment_cache_hits": cache.hits.get(PROTOCOL, 0) - hits_before}}


def _judge(judge, messages):
    value = judge(messages)
    number(value["interactivity"], "interactivity", high=1)
    return value


def component_targets(tree, report):
    """Use the existing success calculation/groups and the exact same leaf set."""
    validate_tree(tree)
    if report["schema"] != PROTOCOL or report["tree_sha256"] != sha256_json(asdict(tree)):
        raise ValueError("Reward report belongs to another tree/protocol")
    if set(report["leaves"]) != {l.leaf_id for l in tree.leaves}:
        raise ValueError("Reward report leaf set differs")
    targets = node_credit(1.0, 0.0)(tree, {}, {})
    descendant = {n.node_id: [] for n in tree.nodes}
    for leaf in tree.leaves:
        row = report["leaves"][leaf.leaf_id]
        if row["success"] != leaf.outcome:
            raise ValueError("Stored success differs from original outcome")
        if leaf.outcome is None:
            continue
        number(row["interactivity"], "interactivity", high=1)
        number(row["agent_ktokens"], "agent_ktokens")
        for nid in leaf.path:
            descendant[nid].append(row)
    for nid, node in targets["nodes"].items():
        rows = descendant[nid]
        for name in ("interactivity", "agent_ktokens"):
            node[name] = sum(r[name] for r in rows) / len(rows) if rows else None
    return targets


def make_targets(tree, report, *, length_weight, interactivity_weight=1.0):
    """Same targets shape consumed by the unmodified five_gpu.update_batch.

    This pure helper accepts explicit weights for pilot diagnostics and tests.
    Production callers must require a frozen train-pilot weight artifact first.
    """
    lam = number(length_weight, "length_weight")
    wi = number(interactivity_weight, "interactivity_weight")
    targets = component_targets(tree, report)
    for node in targets["nodes"].values():
        success = node["outcome_component"]
        node["reward"] = (None if success is None else
                          success + wi * node["interactivity"] - lam * node["agent_ktokens"])
    targets["reward_weights"] = {"task_success": 1.0, "interactivity": wi, "agent_ktokens": -lam}
    targets["method_label"] = METHOD
    targets["estimator"] = "existing_decided_descendant_leaves_uniform_mean"
    return targets
