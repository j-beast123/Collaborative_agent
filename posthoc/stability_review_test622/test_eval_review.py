"""POST-HOC / EXPLORATORY: stability stance v1 UNCHANGED + conflict-only source-blind review (see agent_turn_stab): when the Agent's option and the physician's current option differ, the two options and their stated reasons are compared without saying who wrote which (option logits, both orders); the result is reported to the physician, who decides. No automatic deference and no vote. Otherwise final_test622/test_eval_final.py.
FINAL TEST622 evaluation of ONE frozen target (A theta0, B Offline, C Online, D E3-private T4/OFF): Thinking OFF, 4 rounds, temperature 0.7, top_p 0.9.
main120_queue/test_eval_think.py with the selection check replaced by the frozen-target check; nothing else differs.

Original description (thinking ON, DEV-selected checkpoint):

Same inference path, config and generation settings as the DEV evaluation (eval_ckpt_think.py --mode g4 --variants A0 = CURRENT inference),
with the repo's TEST episode protocol (scripts/online_rl/async_eval.py): one episode per case, seed = five_gpu.eval_seed + case index,
Doctor opening cached per case (fixed seed; independent of the Agent, shared by theta0 and every arm).
Frozen policy (FREEZE_CHECKPOINT_DEV_TEST_POLICY.json): a checkpoint is evaluated here only after its DEV selection has been recorded and
frozen; the script refuses anything else. No result of this script feeds any selection.

    CUDA_VISIBLE_DEVICES=<gpu> python test_eval_think.py --repo <repo> --arm <arm> --ck <N> --selection <SELECTION_<arm>.json> \
        --checkpoint <dir> --shard k --shards n --doctor-url URL --out DIR
    theta0: --arm theta0 --ck 0 --selection <SELECTION of the main arm> (the TEST stage has begun), no --checkpoint
    --smoke N: N TRAIN-pool cases instead of the TEST file (no TEST access, no selection needed); checks the code path only"""
from __future__ import annotations
import argparse, copy, json, os, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

CONFIG = 'configs/experiments/fixed_t2/main120/e3_private_5_5_t4_off.json'                            # Thinking OFF, 4 rounds, temperature 0.7, top_p 0.9
TRAIN = 'data/interim/medmcqa/five_gpu_train500/train500_excluding_calibration_seed20261008.jsonl'
THETA0 = '9018ce3bbacd319049576e8ace88afb4777621a8769919bf46676fca8a85b17c'
THREADS = 4


def main():
    a = argparse.ArgumentParser()
    a.add_argument('--repo', required=True); a.add_argument('--arm', required=True); a.add_argument('--ck', type=int, required=True); a.add_argument('--selection')
    a.add_argument('--shard', type=int, default=0); a.add_argument('--shards', type=int, default=1); a.add_argument('--doctor-url', required=True); a.add_argument('--out', required=True)
    a.add_argument('--checkpoint'); a.add_argument('--smoke', type=int, default=0)
    x = a.parse_args(); repo = Path(x.repo); sys.path.insert(0, str(repo)); os.chdir(repo)
    if os.environ.get('SHERPA_AGENT_BATCH_SIZE') != '1': raise SystemExit('canonical evaluation requires Agent generation batch 1 (SHERPA_AGENT_BATCH_SIZE=1)')
    if not x.smoke:
        tgt = json.loads(Path(x.selection).read_text())['targets'][x.arm]                                  # must exist: the four targets are frozen
        if tgt['case'] != x.ck or bool(x.checkpoint) != (x.arm != 'A_theta0'): raise SystemExit('not the frozen target of this arm')
    from scripts.online_rl.run_tree import load_key
    from src.online_rl.cases import file_sha256, load_cases
    from src.online_rl.environment import final_outcome
    from src.online_rl.integrated.continuation import adapter_sha256, derive_seed
    from src.online_rl.runtime import SharedPolicy
    from src.online_rl.tree import experiment as X
    from src.online_rl.tree.five_gpu import cache_openings
    cfg = X.load_config(CONFIG); fg = cfg['five_gpu']; assert cfg['rollout']['rounds'] == 4 and not cfg['rollout'].get('agent_thinking') and (cfg['rollout']['temperature'], cfg['rollout']['top_p']) == (0.7, 0.9)
    if x.smoke: cases = load_cases(repo / TRAIN)[:x.smoke]
    else:
        if file_sha256(repo / fg['eval_file']) != fg['eval_sha256']: raise SystemExit('TEST data hash differs')
        cases = load_cases(repo / fg['eval_file'])
        if len(cases) != fg['eval_cases'] or len(cases) != 622: raise SystemExit('TEST size differs')
    local = copy.deepcopy(cfg); local['rollout']['doctor_backend']['endpoint'] = x.doctor_url
    policy = SharedPolicy(local, load_key(), device='cuda:0')
    if x.checkpoint:
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        set_peft_model_state_dict(policy.model, load_file(str(Path(x.checkpoint) / 'adapter' / 'adapter_model.safetensors'))); getattr(policy, 'mark_dirty', lambda: None)()
        receipt = json.loads((Path(x.checkpoint) / 'COMPLETE.json').read_text())
        if adapter_sha256(policy.model) != receipt['policy_sha256']: raise SystemExit('loaded adapter differs from the checkpoint receipt')
        if receipt['completed_cases'] != x.ck: raise SystemExit('checkpoint receipt is not the requested checkpoint')
        if not x.smoke and receipt['policy_sha256'] != tgt['policy_sha256']: raise SystemExit('checkpoint is not the frozen target policy')
    elif adapter_sha256(policy.model) != THETA0: raise SystemExit('policy is not theta0')
    policy.seed_all = lambda seed: None
    # POST-HOC / EXPLORATORY: stability-calibrated stance. The private assessment (prompt, seed, output) is unchanged. Five alternative
    # short private prompts are answered for the same case; if all five give the same option as the private assessment the Agent is
    # "stable" and takes the INSIST stance in its visible replies, otherwise the DEFER stance. Views, rule and texts fixed before the run.
    import re as _re
    from src.online_rl import agent_protocol as AP
    _FMT = "\n\nReply in exactly this format:\nOPTION: <letter>\nKEY_REASON: <one or two sentences>"
    VIEWS = {
     'RF': ("Answer the multiple-choice case independently. Briefly state the key reason, then give the option you believe is best supported by the case. Do not assume another person's answer or reasoning.\n\nReply in exactly this format:\nKEY_REASON: <one or two sentences>\nOPTION: <letter>", 120),
     'KF': ("Answer the multiple-choice case independently. Name the decisive finding in the case, choose the option you believe is best supported by it, and briefly state the key reason. Do not assume another person's answer or reasoning.\n\nReply in exactly this format:\nKEY_FINDING: <one short phrase from the case>\nOPTION: <letter>\nKEY_REASON: <one or two sentences>", 150),
     'PE': ("You are a senior attending physician who writes and grades national medical residency (MIR) examination questions. Answer the multiple-choice case independently, as the exam's official answer key would: choose the single best option according to standard clinical guidelines and briefly state the key reason. Read every detail of the vignette; the decisive finding is often a single phrase. Do not assume another person's answer or reasoning." + _FMT, 120),
     'DX': ("Answer the multiple-choice case independently. First state the answer in your own words, as you would without seeing the options; then give the option that matches it and briefly state the key reason. Do not assume another person's answer or reasoning.\n\nReply in exactly this format:\nANSWER_IN_OWN_WORDS: <a few words>\nOPTION: <letter>\nKEY_REASON: <one or two sentences>", 150),
     'CARD': ("Answer the multiple-choice case independently. Fill in the short case card, then choose the option you believe is best supported by the case and briefly state the key reason. Do not assume another person's answer or reasoning.\n\nReply in exactly this format:\nPATIENT: <age, sex and main problem in one phrase>\nKEY_FINDINGS: <up to three findings, comma-separated>\nOPTION: <letter>\nKEY_REASON: <one or two sentences>", 180)}
    INSIST = ("\n\nStance for this reply: you hold the independent assessment above firmly. In your first sentence say plainly which option you consider correct; do not "
              "open by agreeing with the physician if they favour a different option. Give the decisive finding for your option and, if the physician favours a different "
              "option, say specifically why that option does not fit this case. Keep your option unless the physician gives new, specific evidence that contradicts it.")
    DEFER = ("\n\nStance for this reply: your independent assessment above is uncertain. Mention it only as a possibility. Do not argue against the physician's option. "
             "If the physician favours a different option and gives a case-grounded reason, support the physician's option.")
    _OPT = _re.compile(r'(?m)^\**OPTION:\**\s*\(?([A-E])\b'); stab = {}; stab_lock = threading.Lock()
    import torch as _torch
    _KEY = _re.compile(r'KEY_REASON:\**\s*(.+)', _re.S); _L = {x: sorted({i[0] for s_ in (x, ' ' + x) for i in [policy.tok.encode(s_, add_special_tokens=False)] if len(i) == 1}) for x in 'ABCDE'}
    EXTRACT = "Read the physician's message about a multiple-choice case and identify the option the physician currently favours. Reply in exactly this format:\nOPTION: <letter>"
    REVIEW = ("Two independent assessments of a multiple-choice clinical case disagree. You are not told who wrote them. Judge only by the case: decide which option the "
              "findings in the case support better. Reply in exactly this format:\nOPTION: <letter>")
    NOTE = ("\n\nA source-blind comparison of the two disagreeing options against the case favoured option {pick}. Tell the physician this result in one sentence; "
            "the physician makes the final choice.")
    def _letters(system, user, allowed):
        with policy.gen_lock, _torch.no_grad():
            policy.model.eval(); p = _torch.softmax(policy.model(**policy.tok(policy.ST.chat_prompt(policy.tok, system, user) + 'OPTION:', return_tensors='pt').to(policy.model.device)).logits[0, -1].float(), -1)
        raw = {x: float(p[_L[x]].sum()) for x in allowed}; zz = sum(raw.values()) or 1.0; return {x: v / zz for x, v in raw.items()}
    def _opt(t):
        m = _OPT.findall(t or ''); return m[-1] if m else None
    def agent_turn_stab(case, prefix, turn, seed):
        user = policy.agent_user_prompt(case, prefix, turn); base = policy.ST.chat_prompt(policy.tok, policy.agent_system, user)
        sa, sb = AP.stage_seeds(seed, case['case_id']); pa = policy.ST.chat_prompt(policy.tok, AP.ASSESS_SYSTEM, AP.blind_case_prompt(case))
        z = policy._generate(pa, AP.ASSESS_MAX_TOKENS, sa); cid = case['case_id']
        with stab_lock: known = stab.get(cid)
        if known is None:                                                 # the views see the case only, so one evaluation per case (first turn) is reused
            o = _opt(z['completion_text']); votes = [_opt(policy._generate(policy.ST.chat_prompt(policy.tok, sys_, AP.blind_case_prompt(case)), bud, sa)['completion_text']) for sys_, bud in VIEWS.values()]
            known = {'private': o, 'views': votes, 'stable': o is not None and all(v == o for v in votes)}
            with stab_lock: stab[cid] = known
        # Conflict-only source-blind review (logits only, both orders); the stability stance itself is unchanged. The review result is passed on only when
        # it points the same way as the stance (stable + review for the Agent's option, or unstable + review for the physician's option).
        docs = [t for t in prefix if (t.get('speaker') if isinstance(t, dict) else t[0]) == 'doctor']; last = ((docs[-1].get('text') if isinstance(docs[-1], dict) else docs[-1][3]) if docs else '') or ''
        own = known['private'] if turn == 1 else _opt(z['completion_text']); note = ''; rec = {'turn': turn, 'own': own}
        if last and own and own in case['options']:
            opts_ = '\n'.join(f'{k_}. {v_}' for k_, v_ in case['options'].items()); ex = _letters(EXTRACT, f"OPTIONS:\n{opts_}\n\nPHYSICIAN'S MESSAGE:\n{last}", list(case['options'])); doc = max(ex, key=ex.get); rec['doctor'] = doc
            if doc != own:
                kr = _KEY.search(z['completion_text'] or ''); mine = (own, kr.group(1).strip() if kr else (z['completion_text'] or '').strip()); theirs = (doc, last); f_ = []
                for first, second in ((mine, theirs), (theirs, mine)):
                    f_.append(_letters(REVIEW, AP.blind_case_prompt(case) + f"\n\nAssessment 1 favours option {first[0]}:\n{first[1]}\n\nAssessment 2 favours option {second[0]}:\n{second[1]}", [own, doc])[own])
                p_own = sum(f_) / 2; pick = own if p_own >= 0.5 else doc; rec.update(review_own=p_own, pick=pick)
                if (known['stable'] and pick == own) or (not known['stable'] and pick == doc): note = NOTE.format(pick=pick); rec['note'] = True
        with stab_lock: known.setdefault('reviews', []).append(rec)
        pb = policy.ST.chat_prompt(policy.tok, policy.agent_system, user + AP.HIDDEN_BLOCK.format(assessment=z['completion_text'].strip()) + (INSIST if known['stable'] else DEFER) + note)
        r = policy._generate(pb, policy.args.max_completion_length, sb)
        return {'stage': f'A{turn}', 'prompt_text': pb, 'seed': seed, **r, 'private_assessment': z['completion_text'], 'state_prompt_hash': AP.state_hash(z['prompt_input_ids'], base), 'agent_protocol': AP.VERSION}
    policy.agent_turn = agent_turn_stab
    out = Path(x.out); out.mkdir(parents=True, exist_ok=True)
    problems = policy.doctor.verify()['problems']
    if problems: raise SystemExit('Doctor identity: ' + '; '.join(problems))
    cache_openings(policy.doctor, out / f'openings_shard{x.shard}.jsonl', fg['opening_identity'])
    private = {}; lock = threading.Lock(); original_turn = policy.agent_turn

    def agent_text(case, prefix, turn, seed):                                                              # CURRENT inference; records the private assessment
        r = original_turn(case, prefix, turn, seed)
        with lock: private[(case['case_id'], seed, turn)] = r['private_assessment']
        return r['completion_text']
    policy.agent_text = agent_text
    path = out / f'test_shard{x.shard}.jsonl'
    done = {json.loads(l)['index'] for l in open(path)} if path.exists() else set()
    jobs = [(i, c) for i, c in enumerate(cases) if i % x.shards == x.shard and i not in done]

    def one(job):
        i, case = job; cid = case['case_id']; seed = fg['eval_seed'] + i; ep = policy.generate(case, seed)
        keys = [(cid, derive_seed(seed, cid, 'agent', int(tr['stage'][1:])), int(tr['stage'][1:])) for tr in ep['agent_traces']]
        with lock: priv = [private.get(key) for key in keys]
        return {'index': i, 'case_id': cid, 'seed': seed, 'final_option': ep['final_option'], 'outcome': final_outcome(case, ep['final_option']),
                'dialogue': ep['dialogue'], 'num_rounds': ep['num_discussion_rounds'], 'private_assessments': priv, 'stability': stab.get(cid)}
    started = time.time()
    with ThreadPoolExecutor(THREADS) as pool, open(path, 'a') as f:
        for row in pool.map(one, jobs):
            f.write(json.dumps(row) + '\n'); f.flush()
    print(json.dumps({'mode': 'smoke' if x.smoke else 'test', 'arm': x.arm, 'ck': x.ck, 'shard': x.shard, 'episodes': len(jobs), 'seconds': round(time.time() - started)}), flush=True)


if __name__ == '__main__':
    main()
