"""Summarize repeated request timing and answer-format checks, without judging clinical correctness."""
import json
from pathlib import Path
import statistics
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.collaboration.generator import option,final_answer_option
from src.online_rl.doctor import split_status
from src.online_rl.cases import load_cases


def summarize(directory):
    out=Path(directory);data=json.loads((out/'requests.json').read_text());results=json.loads((out/'results.json').read_text())
    cases={c['case_id']:c['options'] for c in load_cases(ROOT/data['config']['data']['train'])}
    arms=[]
    for parallel in sorted({r['parallel'] for r in results}):
        records=[r for r in results if r['parallel']==parallel];errors=unparsed=limit=0;private_matches=total=0
        for r in records:
            repeat=r['repeat'];doctor=json.loads((out/f'responses_r{repeat}_p{parallel}.json').read_text())
            for d in doctor:
                limit+=d['tokens']>=data['config']['rollout']['max_tokens_doctor']
                if d['kind']=='final':unparsed+=not bool(final_answer_option(d['response']))
                else:errors+=split_status(d['response'],cases[d['case_id']])[2] is not None
            path=out/f'agent_responses_r{repeat}_p1.json'
            if path.exists():
                base=json.loads(path.read_text());agent=json.loads((out/f'agent_responses_r{repeat}_p{parallel}.json').read_text())
                total+=len(base);private_matches+=sum(option(a['private_assessment'])==option(b['private_assessment']) for a,b in zip(base,agent))
        arms.append({'parallel':parallel,'repeats':len(records),'doctor_seconds':statistics.median(r['wall_seconds'] for r in records),
                     'agent_seconds':statistics.median(r['agent_wall_seconds'] for r in records),
                     'doctor_tokens_per_second':statistics.median(r['tokens_per_second'] for r in records),
                     'agent_tokens_per_second':statistics.median(r['agent_tokens_per_second'] for r in records),
                     'peak_vram_mb':max(r['peak_vram_mb'] for r in records),'doctor_status_errors':errors,
                     'doctor_unparsed_finals':unparsed,'doctor_limit_hits':limit,
                     'agent_private_option_agreement':private_matches/total if total else None})
    baseline=next(r for r in arms if r['parallel']==1)
    for r in arms:r.update(agent_speedup=baseline['agent_seconds']/r['agent_seconds'],doctor_speedup=baseline['doctor_seconds']/r['doctor_seconds'])
    report={'scope':'same 8 actual Agent actions and 8 Doctor requests; model load and warmup excluded; not a full training batch',
            'ollama_version':data['config']['rollout']['doctor_backend']['ollama_version'],'arms':arms,'training_auto_resume':False}
    (out/'summary.json').write_text(json.dumps(report,indent=2));return report

if __name__=='__main__':print(json.dumps(summarize(ROOT/sys.argv[1]),ensure_ascii=False,indent=2))
