import ast
import json
from pathlib import Path

from execute_c3_blind_once_v4_1 import target_reply_from_user_message

ROOT = Path(__file__).resolve().parents[1]


def test_raw_provider_write_precedes_any_local_validation():
    source = (ROOT / "execute_c3_blind_once_v4_1.py").read_text()
    tree = ast.parse(source)
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    calls = []
    for node in ast.walk(main):
        if isinstance(node, ast.Call):
            name = ast.unparse(node.func)
            if name in {"write_atomic", "json.loads", "Draft202012Validator(schema).validate", "validate_wire_behavior"}:
                calls.append((node.lineno, name))
    raw_write = min(line for line, name in calls if name == "write_atomic" and line < 100)
    validations = [line for line, name in calls if name in {"json.loads", "Draft202012Validator(schema).validate", "validate_wire_behavior"} and line > raw_write]
    assert validations
    assert raw_write < min(validations)


def test_runner_has_no_automatic_retry():
    source = (ROOT / "execute_c3_blind_once_v4_1.py").read_text()
    assert "max_retries=0" in source
    assert source.count("chat.completions.create") == 1


def test_target_reply_comes_from_frozen_structured_field():
    content = json.loads((ROOT / "api_preview/C3_0.0_blind_source_request_v4_1.json").read_text())["messages"][1]["content"]
    expected = json.loads(content)["state"]["target_agent_reply"]
    assert target_reply_from_user_message(content) == expected
