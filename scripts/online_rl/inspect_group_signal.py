"""Read-only trajectory reward, AUC and answer-filter telemetry."""
import argparse
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    from src.online_rl.tree.experiment import load_config
    from src.online_rl.sherpa.monitor import snapshot, render
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/sherpa.json')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--json', action='store_true')
    parser.add_argument('--batch', '--step', type=int)
    args = parser.parse_args()
    cfg = load_config(args.config)
    home = args.output or ROOT / cfg['output_dir']
    data = snapshot(home, selected_step=args.batch)
    print(json.dumps(data, ensure_ascii=False, indent=2) if args.json else render(data))


if __name__ == '__main__':
    main()
