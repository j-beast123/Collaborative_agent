"""Frozen shared Doctor openings (doctor_opening_mode = frozen_shared_v1).

D1 depends only on the case, the Doctor (model, prompts, decoding), `rounds` and its seed, never on the Agent policy or the
reward arm. To remove runtime noise in this policy-independent environment opening, D1 is generated ONCE per case in one
controlled pass (scripts/online_rl/build_doctor_openings.py) into an immutable table per split, and every arm and every
checkpoint reads the same record. Only D1 is shared: Agent actions, D2+, halting, FINAL, rewards and updates stay per run.

A table is a JSON file {schema, mode, split, doctor_identity, doctor_protocol_sha256, rounds, created, records: [...]}; its
file SHA256 is pinned in rollout.doctor_opening and enters the rollout / evaluation protocol hash. Lookup recomputes the
record identity from the live case, Doctor and seed and fails loudly on a missing, corrupt or mismatched record: a frozen
opening is never regenerated at run time."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
from src.online_rl.integrated.records import sha256_json, sha256_text

MODE = 'frozen_shared_v1'
SCHEMA = 'doctor_opening_table_v1'


class OpeningError(RuntimeError):
    pass


def case_sha256(case: dict) -> str:
    return sha256_json(case)


def opening_input(doctor, case: dict, rounds: int) -> str:
    """The exact single user message the Doctor receives for D1 (system + blank line + D1 turn prompt)."""
    from src.online_rl.doctor import turn_prompt
    return f'{doctor.system}\n\n{turn_prompt(case, [], 1, rounds, with_status=True)}'


def identity(split: str, case: dict, doctor, rounds: int, seed: int) -> dict:
    """Every semantic input of D1, and nothing else (no arm, policy, rubric setting or output directory)."""
    d = doctor.identity
    return {'schema': SCHEMA, 'mode': MODE, 'split': split, 'case_id': case['case_id'], 'case_sha256': case_sha256(case),
            'input_sha256': sha256_text(opening_input(doctor, case, rounds)), 'doctor_protocol_sha256': doctor.protocol,
            'model': d['model'], 'engine': d['engine'],
            'system_sha256': d['system_sha256'], 'turn_prompt_sha256': d['turn_prompt_sha256'], 'options_sha256': sha256_json(d['options']),
            'rounds': rounds, 'seed': int(seed)}


def train_seed(cfg: dict, case_id: str) -> int:
    """D1 seed of a training tree (src/online_rl/tree/rollout._build_initial_fanout): doctor_seed(run_seed, case, 'doctor', 1)."""
    from src.online_rl.doctor import doctor_seed
    return doctor_seed(cfg['schedule']['seed'], case_id, 'doctor', 1)


def eval_seed(cfg: dict, case_id: str) -> int:
    """D1 seed of an evaluation episode: episode seed derive_seed(eval seed, case, 'eval', 0) (tree/evaluation.py), then
    doctor_seed(episode seed, case, 'doctor', 1) (environment.generate_trajectory_v2)."""
    from src.online_rl.doctor import doctor_seed
    from src.online_rl.integrated.continuation import derive_seed
    return doctor_seed(derive_seed(cfg['evaluation']['seed'], case_id, 'eval', 0), case_id, 'doctor', 1)


def make_record(ident: dict, out: dict, provenance: dict) -> dict:
    if out['seed'] != ident['seed']: raise OpeningError('Doctor returned a different seed')
    return {'key': sha256_json(ident), 'identity': ident, 'raw': out['raw'], 'visible': out['visible'],
            'visible_sha256': sha256_text(out['visible']), 'raw_sha256': sha256_text(out['raw']), 'provenance': provenance}


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_table(path: Path, split: str, doctor, rounds: int, records: list[dict], created: dict) -> str:
    """Write the table once (refuses to overwrite) read-only; returns its SHA256."""
    path = Path(path)
    if path.exists(): raise OpeningError(f'{path} exists; frozen tables are never overwritten')
    if len({r['identity']['case_id'] for r in records}) != len(records): raise OpeningError('duplicate case ids')
    body = {'schema': SCHEMA, 'mode': MODE, 'split': split, 'doctor_identity': doctor.identity, 'doctor_protocol_sha256': doctor.protocol,
            'rounds': rounds, 'created': created, 'n_records': len(records), 'records': records}
    path.parent.mkdir(parents=True, exist_ok=True); tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(body, ensure_ascii=False, indent=1, sort_keys=True) + '\n'); tmp.chmod(0o444); tmp.rename(path)
    return file_sha256(path)


class OpeningTable:
    """Read-only lookup of one frozen split table, verified against its pinned SHA256 at load."""

    def __init__(self, path: Path, sha256: str, split: str, doctor, rounds: int):
        path = Path(path)
        if not path.exists(): raise OpeningError(f'frozen opening table missing: {path}')
        got = file_sha256(path)
        if got != sha256: raise OpeningError(f'{path}: SHA256 {got[:12]} differs from the pinned {str(sha256)[:12]}')
        t = json.loads(path.read_text())
        if (t.get('schema'), t.get('mode'), t.get('split'), t.get('rounds')) != (SCHEMA, MODE, split, rounds):
            raise OpeningError(f'{path}: schema/mode/split/rounds {t.get("schema"), t.get("mode"), t.get("split"), t.get("rounds")} '
                               f'!= {SCHEMA, MODE, split, rounds}')
        if t.get('doctor_protocol_sha256') != doctor.protocol: raise OpeningError(f'{path}: built for another Doctor protocol')
        self.records = {}
        for r in t['records']:
            if sha256_json(r['identity']) != r['key'] or sha256_text(r['visible']) != r['visible_sha256'] or sha256_text(r['raw']) != r['raw_sha256']:
                raise OpeningError(f"{path}: corrupt record {r['identity'].get('case_id')}")
            self.records[r['identity']['case_id']] = r
        self.path, self.sha256, self.split, self.doctor, self.rounds = path, sha256, split, doctor, rounds

    def __call__(self, case: dict, seed: int) -> dict:
        """D1 for (case, seed) in the doctor.turn result shape; never generates."""
        r = self.records.get(case['case_id'])
        if r is None: raise OpeningError(f"{self.split}: no frozen D1 for case {case['case_id']}")
        want = identity(self.split, case, self.doctor, self.rounds, seed)
        bad = sorted(k for k in set(want) | set(r['identity']) if want.get(k) != r['identity'].get(k))
        if bad: raise OpeningError(f"{self.split}: frozen D1 of case {case['case_id']} does not match the live inputs at {bad}")
        return {'raw': r['raw'], 'visible': r['visible'], 'status': None, 'status_error': None, 'seed': int(seed),
                'opening': {'mode': MODE, 'table_sha256': self.sha256, 'key': r['key']}}


def protocol_identity(spec: dict) -> dict:
    """The part of rollout.doctor_opening that enters the rollout / evaluation protocol hash (no file paths)."""
    if spec.get('mode') != MODE or spec.get('schema') != SCHEMA: raise ValueError(f'rollout.doctor_opening must be mode {MODE}, schema {SCHEMA}')
    for k in ('train_table_sha256', 'eval_table_sha256'):
        if not isinstance(spec.get(k), str) or len(spec[k]) != 64: raise ValueError(f'rollout.doctor_opening.{k} must be a SHA256')
    return {'doctor_opening_mode': MODE, 'schema': SCHEMA, 'train_table_sha256': spec['train_table_sha256'], 'eval_table_sha256': spec['eval_table_sha256']}
