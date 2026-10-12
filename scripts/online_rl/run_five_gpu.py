"""Compatibility entrypoint for the trajectory-level SHERPA runner."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.online_rl.run_sherpa import main

if __name__ == '__main__':
    main()
