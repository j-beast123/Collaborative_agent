"""Local-only Qwen doctor smoke, including structured turns and final decision."""
import fcntl
import json
import os
from pathlib import Path
import sys
import tempfile
from contextlib import ExitStack

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    os.chdir(ROOT)
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0,1')
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    os.environ.setdefault('OMP_NUM_THREADS', '4')
    from src.online_rl.tree.experiment import load_config
    from src.online_rl.doctor import make_doctor
    from src.online_rl.local_doctor import start_server
    cfg = load_config('configs/ours.json')
    ro = cfg['rollout']
    doctor = make_doctor(ro, (ROOT / ro['doctor_system_prompt_file']).read_text())
    gpu = os.environ['CUDA_VISIBLE_DEVICES'].split(',')[1]
    home = Path(tempfile.mkdtemp(prefix='sherpa-qwen-doctor-'))
    with ExitStack() as stack:
        lock = stack.enter_context(Path(f'/tmp/sherpa_tree_physical_gpu{gpu}.lock').open('a'))
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        server = start_server(cfg, ROOT, home, doctor, gpu)
        stack.callback(server.close)
        case = {'case_id':'synthetic','scenario':'','question':'What is two plus two?',
                'options':{'A':'4','B':'5'}, 'ground_truth':{'correct_option':'A'}}
        first = doctor.turn(case, [], 1, 4, 100)
        prefix = [{'speaker':'doctor','stage':'D1','round':1,'text':first['visible']},
                  {'speaker':'agent','stage':'A1','round':1,'text':'Two plus two is four, option A.'}]
        second = doctor.turn(case, prefix, 2, 4, 101)
        prefix.append({'speaker':'doctor','stage':'D2','round':2,'text':second['visible']})
        final = doctor.final(case, prefix, 102)
        reflection = doctor._chat('Briefly explain your choice in one sentence.', 103, 128)
        from src.collaboration.generator import final_answer_option
        assert first['private_state'] and second['status']
        assert final_answer_option(final) == 'A', repr(final)
        assert '<think>' not in first['visible'] and reflection
        print(json.dumps({'model':doctor.model,'opening':True,'structured_second_turn':True,
                          'final':final,'reflection':True,'logs':str(home)}),flush=True)


if __name__ == '__main__':
    main()
