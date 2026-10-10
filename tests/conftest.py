import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# Existing fixtures use both ``tests.test_*`` and ``test_*`` imports.
# Keep those local imports available now that tests is an explicit package.
TESTS = ROOT / 'tests'
if str(TESTS) not in sys.path:
    sys.path.insert(0, str(TESTS))

AGENTCLINIC = "data/raw/agentclinic/agentclinic_medqa_extended.jsonl"


def require_local(*paths: str, why: str) -> None:
    """Skip when a local file that is not part of the repository (dataset, experiment artifact) is missing."""
    import pytest

    missing = [p for p in paths if not (ROOT / p).exists()]
    if missing:
        pytest.skip(f"{why}; missing local file(s): {', '.join(missing)}")


def require_env_histories(*config_paths: str) -> None:
    """Skip when a main config's frozen D1 histories (an environment run output under outputs/) are missing."""
    from src.common.config import read_config

    require_local(*(read_config(ROOT / c)["environment"]["histories"] for c in config_paths),
                  why="needs the frozen D1 histories of the environment run (outputs/, not tracked)")
