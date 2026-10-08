"""Regenerate review artifacts without GPU access or API calls."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from src.online_rl.tree.artifacts import export_run
if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--run-dir', required=True)
    args=parser.parse_args()
    print(json.dumps(export_run(args.run_dir),indent=2))
