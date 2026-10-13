"""Final TEST622 orchestrator (Server 3 only): C Online, A theta0 (no checkpoint), B Offline, D E3-private T4/OFF, each exactly once, same script / config / seeds /
opening bank (Thinking OFF, 4 rounds, temperature 0.7, top_p 0.9). A target is evaluated only after its target file is frozen:
  C: Online case-120 (OES did not trigger)                       -> targets/C_online.json (written before start)
  B: Offline Stage B: OES clean-stop checkpoint, else case-120   -> written when the Stage B supervisor reports COMPLETE
  D: the DEV-selected checkpoint                                 -> written when SELECTION_d_e3_private_5_5_t4_off.json exists
GPU rule: candidate D's DEV evaluation owns GPUs 5-7 whenever it has work; TEST never starts a shard there while a D checkpoint awaits
DEV or checkpoint 120 is close (training cursor >= 115) and the selection is not frozen. GPUs 0-4 are used once D training has released them."""
import glob, hashlib, json, os, shutil, subprocess, time
from pathlib import Path
ROOT = Path(os.environ['PH_ROOT']); CANON = Path('/home/sj/col/final_test622'); OUT = ROOT / 'runs'; TG = ROOT / 'targets'; LOG = ROOT / 'status'; REPO = '/home/sj/col/d_dev_eval/repo'
PY = '/home/sj/col/.venv-qwen3/bin/python'; OB = '/home/sj/opt/ollama-0.12.5/bin/ollama'; S = 8; RETRIES = 2; ORDER = os.environ['PH_ORDER'].split(',')
ENV = dict(os.environ, SHERPA_ENV_FILE='/home/sj/col/.env', HF_HUB_OFFLINE='1', TOKENIZERS_PARALLELISM='false', SHERPA_AGENT_REPLICAS='1', PYTHONDONTWRITEBYTECODE='1', SHERPA_AGENT_BATCH_SIZE='1')
BANK = CANON / 'openings_test622.jsonl'; BANK_SHA = json.loads((CANON / 'TEST_OPENING_BANK_MANIFEST.json').read_text())['sha256']
DE = Path('/home/sj/col/d_dev_eval'); DSEL = DE / 'SELECTION_d_e3_private_5_5_t4_off.json'; DRUN = Path(glob.glob('/home/sj/col/e3priv_t4_off/outputs/e3priv_t4_off_120/r05*')[0])
K = os.path.expanduser('~/.ssh/id_ed25519_r07'); S0 = 'sj@165.132.143.27'; BSUP = '/data/sj/max4/logs/offline_stage_b_120.log.supervisor'
BRUN = '/data/sj/max4/Collaborative_agent/outputs/train120_max4_offline_frozen_bank_stageB_final120_20261012/r05_o05__p0fixed__gemma3doctor__seed0__max4_offline_frozen_bank_stage_b_final120'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
_checked = {'b': 0}


def log(m):
    with open(LOG, 'a') as f: f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} [posthoc] {m}\n")


def sh(c): return subprocess.run(c, shell=True, capture_output=True, text=True)
def mem(g): return int(sh(f'nvidia-smi -i {g} --query-gpu=memory.used --format=csv,noheader,nounits').stdout.strip() or 99999)
def ssh(c): return sh(f"timeout 90 ssh -o BatchMode=yes -o ConnectTimeout=15 -i {K} {S0} '{c}'")
def idx(d): return {json.loads(l)['index'] for f in d.glob('test_shard*.jsonl') for l in open(f)}


def freeze(arm, rec):
    f = TG / f'{arm}.json'; TG.mkdir(exist_ok=True)
    try: fd = os.open(f, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    except FileExistsError: return
    with os.fdopen(fd, 'w') as h: h.write(json.dumps({'frozen_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'targets': {arm: rec}}, indent=1) + '\n')
    os.chmod(f, 0o444); log(f"TARGET FROZEN {arm}: case {rec['case']} policy {rec['policy_sha256'][:16]} ({rec['rule']})")


def target_d():
    if (TG / 'D_e3private.json').exists() or not DSEL.exists(): return
    s = json.loads(DSEL.read_text()); ck = s['selected_checkpoint']; local = DE / 'runs' / 'checkpoints' / f'checkpoint_{ck:06d}'
    freeze('D_e3private', {'case': ck, 'policy_sha256': json.loads((local / 'COMPLETE.json').read_text())['policy_sha256'], 'local_checkpoint': str(local),
                           'source_checkpoint': s['selected_checkpoint_path'], 'rule': 'highest DEV accuracy, earlier checkpoint on a tie', 'dev_correct_of_400': s['dev_correct_of_400']})


def target_b():
    if (TG / 'B_offline.json').exists() or time.time() - _checked['b'] < 180: return
    _checked['b'] = time.time()
    if 'COMPLETE' not in ssh(f'tail -n 1 {BSUP}').stdout: return
    r = ssh(f'cat {BRUN}/OES_STOP_RECORD.json 2>/dev/null; echo; echo ====; cat {BRUN}/run_state.json').stdout.split('====')
    if len(r) != 2 or not r[1].strip(): return
    oes = json.loads(r[0]) if r[0].strip() else None; state = json.loads(r[1])
    src = oes['checkpoint'] if oes else f'{BRUN}/checkpoints/checkpoint_000120'
    if not oes and state['case_cursor'] != 120: log(f"B: supervisor COMPLETE but cursor {state['case_cursor']} without an OES record; not frozen (needs attention)"); return
    local = ROOT / 'checkpoints' / 'B_offline_selected'; (local / 'adapter').mkdir(parents=True, exist_ok=True)
    c = sh(f"rsync -a -e 'ssh -o BatchMode=yes -i {K}' {S0}:{src}/adapter/ {local}/adapter/ && rsync -a -e 'ssh -o BatchMode=yes -i {K}' {S0}:{src}/COMPLETE.json {local}/COMPLETE.json")
    if c.returncode != 0: log('B: checkpoint copy failed; will retry'); return
    rc = json.loads((local / 'COMPLETE.json').read_text()); want = oes['policy_sha256'] if oes else state['policy_sha256']
    if rc['policy_sha256'] != want: log('B: copied checkpoint policy hash differs from the selected policy; not frozen (needs attention)'); return
    freeze('B_offline', {'case': rc['completed_cases'], 'policy_sha256': rc['policy_sha256'], 'local_checkpoint': str(local), 'source_checkpoint': f'{S0}:{src}',
                         'rule': 'OES clean-stop checkpoint' if oes else 'case-120 (OES did not trigger)', 'oes_triggered': bool(oes), 'rubric_bank_sha256': state.get('rubric_bank_sha256')})


def d_dev_needs_gpus():
    if DSEL.exists(): return False
    cur = json.loads((DRUN / 'run_state.json').read_text())['case_cursor']
    if cur >= 115: return True
    return any((DRUN / 'checkpoints' / f'checkpoint_{c:06d}' / 'COMPLETE.json').exists() and not (DE / 'runs' / 'd_e3_private_5_5_t4_off' / f'ckpt{c:03d}' / 'DONE_A0').exists() for c in (30, 60, 90, 120))


def start(arm, i, g):
    t = json.loads((TG / f'{arm}.json').read_text())['targets'][arm]; d = OUT / arm; d.mkdir(parents=True, exist_ok=True); port = 11900 + g; f = d / f'openings_shard{i}.jsonl'
    if not f.exists(): shutil.copy(BANK, f)
    doc = subprocess.Popen([OB, 'serve'], env=dict(ENV, CUDA_VISIBLE_DEVICES=str(g), OLLAMA_HOST=f'127.0.0.1:{port}', OLLAMA_MODELS='/home/sj/sj_ollama/models', OLLAMA_NUM_PARALLEL='1',
                           OLLAMA_MAX_LOADED_MODELS='1', OLLAMA_KEEP_ALIVE='60m'), stdout=open(d / f'doctor{i}.log', 'a'), stderr=subprocess.STDOUT, start_new_session=True)
    for _ in range(60):
        if sh(f'curl -s http://127.0.0.1:{port}/api/version').returncode == 0: break
        time.sleep(1)
    cmd = [PY, '/home/sj/col/blind_review/test_eval_review.py', '--arm', arm, '--ck', str(t['case']), '--selection', str(TG / f'{arm}.json'), '--repo', REPO, '--shard', str(i), '--shards', str(S),
           '--doctor-url', f'http://127.0.0.1:{port}', '--out', str(d)] + (['--checkpoint', t['local_checkpoint']] if t.get('local_checkpoint') else [])
    proc = subprocess.Popen(cmd, env=dict(ENV, CUDA_VISIBLE_DEVICES=str(g)), stdout=open(d / f'test{i}.log', 'a'), stderr=subprocess.STDOUT)
    log(f'start {arm} shard {i}/{S} on GPU {g}'); return {'arm': arm, 'shard': i, 'gpu': g, 'doc': doc, 'proc': proc, 'dir': d}


def finish(u):
    rc = u['proc'].returncode; d = u['dir']; u['doc'].terminate()
    try: u['doc'].wait(60)
    except subprocess.TimeoutExpired: pass
    bank_ok = sha(d / f"openings_shard{u['shard']}.jsonl") == BANK_SHA
    if rc != 0 or not bank_ok:
        f = d / f"UNIT_FAILS_{u['shard']}"; n = int(f.read_text()) + 1 if f.exists() else 1; f.write_text(str(n)); log(f"UNIT FAILED {u['arm']} shard {u['shard']} rc={rc} bank_ok={bank_ok} (failure {n})")
        if n > RETRIES or not bank_ok: (d / 'FAILED_TEST').write_text(json.dumps({'shard': u['shard'], 'rc': rc, 'bank_ok': bank_ok})); log(f"JOB FAILED {u['arm']}: needs attention")
        return
    (d / f"UNIT_DONE_{u['shard']}").write_text('0')
    if all((d / f'UNIT_DONE_{i}').exists() for i in range(S)):
        ok = idx(d) == set(range(622)); (d / ('DONE_TEST' if ok else 'FAILED_TEST')).write_text(json.dumps({'units': S, 'generation_batch': 1})); log(f"{'done' if ok else 'FAILED (row check)'} {u['arm']}")


def main():
    assert sha(BANK) == BANK_SHA
    while not (Path(os.environ['PH_WAIT']).exists() and 'all done' in Path(os.environ['PH_WAIT']).read_text().splitlines()[-1]): time.sleep(30)        # never before the canonical A/B/C/D TEST has completed
    log('orchestrator start: ' + ', '.join(ORDER)); running = []
    while True:
        for u in [u for u in running if u['proc'].poll() is not None]: running.remove(u); finish(u)
        closed = lambda a: (OUT / a / 'DONE_TEST').exists() or (OUT / a / 'FAILED_TEST').exists()
        if all(closed(a) for a in ORDER) and not running: log('all done'); return
        gpus = list(range(8))
        free = [g for g in gpus if g not in {u['gpu'] for u in running} and mem(g) < 1000]
        for arm in [a for a in ORDER if (TG / f'{a}.json').exists() and not closed(a)]:
            busy = {u['shard'] for u in running if u['arm'] == arm}
            for i in [i for i in range(S) if not (OUT / arm / f'UNIT_DONE_{i}').exists() and i not in busy]:
                if not free: break
                running.append(start(arm, i, free.pop(0)))
            if not free: break
        time.sleep(20)


if __name__ == '__main__':
    main()
