"""Explicit schema/config verification; never called automatically by run.sh."""
import argparse
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    import jsonschema
    from src.online_rl.tree.experiment import load_config
    from src.online_rl.sherpa.config import config_problems
    from src.online_rl.sherpa.rubrics import SET_SCHEMA, checklist_schema
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/sherpa.json')
    args = parser.parse_args()
    cfg = load_config(args.config)
    problems = config_problems(cfg, ROOT)
    if problems:
        raise ValueError('; '.join(problems))
    jsonschema.Draft202012Validator.check_schema(SET_SCHEMA)
    for count in (5, 6):
        jsonschema.Draft202012Validator.check_schema(checklist_schema(count))
    print('Trajectory/checklist configuration and schemas ready; no model or API calls.')


if __name__ == '__main__':
    main()
