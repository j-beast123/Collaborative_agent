"""Rejudge the stopped pilot with Gemini, preserving its frozen source generation policy."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.online_rl import run_tree as RT, run_prompt_pilot as pilot
from src.online_rl.integrated import continuation
from src.online_rl.integrated.records import sha256_json,sha256_text
from src.online_rl.tree.ollama_policy import OllamaPolicy
from src.online_rl.tree import rollout

CONFIG='configs/online_rl/tree_pilot5_gpu_gemini38_resume.json'
cfg=json.loads((ROOT/CONFIG).read_text());out=ROOT/cfg['output_dir']
provenance=json.loads((out/'resume_provenance.json').read_text())
source=provenance['source_manifest'];frozen=out/'frozen_inputs/prompts'
ro=cfg['rollout']
payload={'base_model':cfg['policy']['base_model'],
         'rollout':{k:ro[k] for k in ('rounds','doctor_model','max_tokens_doctor','max_completion_length','temperature','top_p','top_k')},
         'agent_prompt':sha256_text((frozen/'agent.txt').read_text()),
         'doctor_prompt':sha256_text((frozen/'user_simulator.txt').read_text())}
assert sha256_json(payload)==source['protocol_sha256'], 'Frozen source protocol does not match'

class SourcePolicy(OllamaPolicy):
    def __init__(self,cfg,key):
        super().__init__(cfg,key)
        self.agent_system=(frozen/'agent.txt').read_text()
        self.doctor_system=(frozen/'user_simulator.txt').read_text()
        self.judgment_system=(frozen/'local_epistemic_judgment.txt').read_text()
        if self.policy_sha()!=source['initial_policy_sha256']:
            raise RuntimeError('Frozen Agent policy differs from source generation policy')

RT.make_policy=lambda cfg,key:SourcePolicy(cfg,key)
continuation.protocol_sha256=lambda cfg,root:sha256_json(payload)
# Existing source trajectories were generated before repetition stopping was added.
rollout.repeated_agent_response=lambda history,response:False
if __name__=='__main__':
    sys.argv=['run_prompt_pilot.py','--config',CONFIG,'--launch']
    pilot.main()
