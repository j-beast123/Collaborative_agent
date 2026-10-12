# General bank13 candidates v2 handoff

This branch is based on `train120-traj-g6-fused-bank13-v1` and adds 13
general-purpose rubric candidates. The existing evidence-backed bank13 remains
unchanged.

## Checkout

```bash
git clone --branch general-bank13-candidates-v2-20261012 \
  https://github.com/j-beast123/Collaborative_agent.git \
  Collaborative_agent_general_bank13_v2
cd Collaborative_agent_general_bank13_v2
```

If the repository already exists and has local changes, use a separate clone or
worktree. Do not reset the existing checkout.

## Reproduce and validate the candidate artifact

Use the project's Python environment, then run:

```bash
python scripts/online_rl/build_general_bank13_candidates.py
python - <<'PY'
import json
from pathlib import Path
from src.judges.rubric_schema import validate_rubric
from src.online_rl.integrated.records import sha256_json

p = Path('data/interim/medmcqa/general_bank13_candidates_v2_20261012/general_bank13_candidates.json')
a = json.loads(p.read_text())
assert a['sha256'] == sha256_json({k: v for k, v in a.items() if k != 'sha256'})
assert len(a['candidates']) == 13
assert len({x['content_sha256'] for x in a['candidates']}) == 13
assert all(not validate_rubric(x['rubric']) for x in a['candidates'])
print(a['status'], a['sha256'])
PY
```

Expected artifact SHA-256:

```text
e7d6b2f878dfd4287aa533d0e73802192af0456d8aa0c41bf099980c9e95cc6b
```

## Admission before training

The JSON is a candidate artifact, not an executable cumulative KEEP snapshot.
For each candidate, run the existing source-applicability check,
`public_mc_gate_v3`, and independent rerun witness validation. Admit only KEEP
items with their actual proof records. Do not point a training config at the
candidate JSON or synthesize missing proof fields.

After admission, set the training config's
`rubric.cumulative_bank.initial_snapshot_file` and
`initial_snapshot_sha256` to the newly sealed snapshot. The inherited runtime
and GPU orchestration then run in the same way as
`train120-traj-g6-fused-bank13-v1`.
