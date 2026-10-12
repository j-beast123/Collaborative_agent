"""Bind generator/scorer/gate contracts without model loading or external requests."""
import argparse
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.online_rl.tree.experiment import load_config
from scripts.online_rl.run_tree import judge_client, make_scorer, make_gate, evidence_gate_identity, intervention_audit_identity
from src.online_rl.tree.backends import api_rubric_generator, generation_settings
from src.online_rl.tree.offline_chunk_bank import api_chunk_generator
from src.online_rl.cases import load_cases
from src.online_rl.integrated.continuation import protocol_sha256
parser=argparse.ArgumentParser()
parser.add_argument('--config',default='configs/online.json')
args=parser.parse_args()
cfg=load_config(args.config);jm=cfg['judges']
from src.online_rl.tree.group_learning import enabled, config_problems
if enabled(cfg):
    from src.online_rl.tree.group_backends import REFLECTION_SCHEMA, GENERATION_SCHEMA, JUDGE_SCHEMA
    import jsonschema
    problems=config_problems(cfg,ROOT)
    if problems:raise ValueError('; '.join(problems))
    for schema in (REFLECTION_SCHEMA,GENERATION_SCHEMA,JUDGE_SCHEMA):
        jsonschema.Draft202012Validator.check_schema(schema)
    protocol_sha256(cfg,ROOT)
    print('Anchored reflection, generation, verification, blind group rewards and protocol: ready (no API requests).')
    sys.exit(0)
text=lambda p:(ROOT/p).read_text()
clients={role:judge_client(jm[role+'_model'],'preflight-only',text(jm[role+'_prompt_file']),max_tokens=jm[role+'_max_tokens'],provider=jm.get('api_provider',cfg.get('api_provider','openrouter'))) for role in ('gate','scorer')}
make_scorer(cfg,clients['scorer'])
make_gate(cfg,clients['gate'],load_cases(ROOT/cfg['data']['train']))
generator=api_rubric_generator('preflight-only',cfg['rubric']['generator_model'],generation_settings(cfg['rubric']))
api_chunk_generator(generator.chunk_context,text(cfg['rubric']['cumulative_bank']['matcher_prompt_file']))
evidence_gate_identity(cfg,ROOT)
intervention_audit_identity(cfg,ROOT)
protocol_sha256(cfg,ROOT)
print('Generator, chunk bank, scorer, Gate and protocol wiring: ready (no API requests).')
