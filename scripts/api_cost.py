"""Cost of the API calls recorded in usage ledgers (src/common/usage.py)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.common.usage import summarize

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("ledgers", nargs="+")
    parser.add_argument("--pricing", default=str(ROOT / "configs/pricing.json"))
    args = parser.parse_args()
    prices = json.loads(Path(args.pricing).read_text(encoding="utf-8"))["prices"]
    print(json.dumps(summarize(args.ledgers, prices), indent=2))


if __name__ == "__main__":
    main()
