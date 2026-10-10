import copy
import json
from types import SimpleNamespace

from scripts.online_rl import run_tree as RT
from src.online_rl.tree.experiment import load_config
from src.online_rl.tree.cache import CallCache
from src.online_rl.tree.posthoc import doctor_feedback, freeze_trajectory_input, trajectory_message
from src.online_rl.tree.records import Leaf


CONFIG = 'configs/experiments/train120_max2_fixed_p0.json'


def test_training_wiring_uses_actual_doctor_and_preserves_full_reflection_evidence():
    cfg = load_config(CONFIG)
    sent = []
    class Doctor:
        model = cfg['feedback']['model']
        protocol = 'pinned-doctor-protocol'
        system = 'You are the participating clinician.'
        def _chat(self, user, seed, max_tokens, system_override=None):
            sent.append({'user': user, 'system': system_override, 'seed': seed})
            return json.dumps({'helpful': [{'agent_reply_id': 1, 'experience': 'I could compare the stated reasons.'}],
                               'unresolved': []})
    feedback = RT.make_feedback(cfg, SimpleNamespace(doctor=Doctor()), 'must-not-be-used')
    case = {'case_id': 'c', 'scenario': '', 'question': 'Which option?', 'options': {'A': 'first', 'B': 'second'},
            'correct_option': 'ANSWER_KEY_SENTINEL'}
    leaf = Leaf('c/0', 'c', ('c/0',), (
        ('doctor', 'D1', 1, 'Please compare the stated reasons.'),
        ('agent', 'A1', 1, 'The first reason addresses your request; the second leaves a gap.'),
        ('doctor', 'FINAL', 1, 'A'),
    ), 'A', 'A', 1.0, 'max_turns')
    cache = CallCache(None)
    result = doctor_feedback(case, leaf, feedback, cfg['feedback']['prompt_sha256'], cache, cfg['feedback']['model'])
    assert result.kind == 'physician_post_final_reflection_v1'
    assert result.locus['reflection']['helpful'][0]['agent_reply_id'] == 1
    assert result.locus['evidence'][0]['messages']
    assert 'ANSWER_KEY_SENTINEL' not in json.dumps(sent)
    assert 'success' not in sent[0]['user'] and 'failure' not in sent[0]['user']
    assert 'final decision' in sent[0]['user'] and 'A' in sent[0]['user']
    tin = freeze_trajectory_input(case, leaf, result, True)
    generator_input = trajectory_message(tin)
    assert "Doctor's own post-FINAL reflection" in generator_input
    assert 'reported experience' in generator_input
    assert 'ANSWER_KEY_SENTINEL' not in generator_input
    assert doctor_feedback(case, leaf, feedback, cfg['feedback']['prompt_sha256'], cache) == result
    assert len(sent) == 1
    # A change in Doctor semantic identity invalidates reflection cache entries.
    changed = Doctor(); changed.protocol = 'different-doctor-protocol'
    changed_feedback = RT.make_feedback(cfg, SimpleNamespace(doctor=changed), 'unused')
    doctor_feedback(case, leaf, changed_feedback, cfg['feedback']['prompt_sha256'], cache)
    assert len(sent) == 2


def test_reflection_config_keeps_training_settings_and_rejects_an_observer_model():
    cfg = load_config(CONFIG)
    assert RT.blockers(cfg) == []
    assert cfg['schedule']['cases'] == 120 and cfg['rollout']['rounds'] == 2
    assert not cfg['prompt_update']['enabled']
    assert cfg['agent_reward']['w_rubric'] == cfg['agent_reward']['w_outcome'] == 0.5
    assert cfg['five_gpu']['checkpoint_every_cases'] == 30
    assert cfg['experiment']['feedback_model'] == cfg['feedback']['model']
    bad = copy.deepcopy(cfg)
    bad['feedback']['model'] = 'anthropic/claude-4.6-sonnet-20260217'
    assert any('same pinned local Doctor' in x for x in RT.blockers(bad))
