"""Which option a round-1 message favors, read by an LLM (temperature 0, cached)."""

from __future__ import annotations

import json
import re
from typing import Optional

from .llm import CachedChat, ChatRequest, openrouter_chat


class OptionExtractor:
    def __init__(self, chat: CachedChat, *, system_prompt: str, prompt_version: str, params: dict) -> None:
        self.chat = chat
        self.system_prompt = system_prompt
        self.prompt_version = prompt_version
        self.params = params

    def stated_option(self, trajectory: dict, message: str) -> Optional[str]:
        """An option key, "" when the message favors none, None when the call or its output failed."""
        options = trajectory["options"]
        user = json.dumps({"question": trajectory["question"], "options": options, "message": message},
                          ensure_ascii=False)
        result = self.chat.complete(ChatRequest(self.system_prompt, user, self.prompt_version, self.params))
        if result.status != "ok":
            return None
        reply = re.sub(r"[^A-Za-z]", " ", result.text).split()
        if len(reply) != 1:
            return None
        return "" if reply[0].upper() == "NONE" else reply[0].upper() if reply[0].upper() in options else None


def openrouter_extractor(model: str, cache_path, root) -> OptionExtractor:
    """Extractor on an OpenRouter model, with the repository prompt."""
    return OptionExtractor(openrouter_chat(model, cache_path),
                           system_prompt=(root / "prompts/option_extractor.txt").read_text(encoding="utf-8"),
                           prompt_version="option-extractor-v1", params={"temperature": 0, "max_tokens": 5})


def with_initial_options(trajectory: dict, extractor: OptionExtractor) -> dict:
    """The trajectory with both initial options re-read from its round-1 messages."""
    first = {}
    for turn in trajectory.get("turns", []):
        if int(turn.get("round", 0)) == 1 and not turn.get("final"):
            first.setdefault(turn["speaker"], turn["text"])
    read = {speaker: extractor.stated_option(trajectory, text) for speaker, text in first.items()}
    return dict(trajectory, agent_initial_option=read.get("agent") or "", doctor_initial_option=read.get("doctor") or "",
                initial_option_source=extractor.prompt_version)
