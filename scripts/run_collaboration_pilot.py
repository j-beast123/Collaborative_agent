"""Backward-compatible entry point for collaboration trajectory generation."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.collaboration.generator import (  # noqa: F401
    agent_call,
    args,
    case_text,
    dialogue_text,
    doctor_call,
    initial_option,
    main,
    option,
    parse_reconstruction,
    reconstruct_call,
    signals_conclusion,
)


if __name__ == "__main__":
    main()
