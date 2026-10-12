"""Validate the new message path with fake transport; no model or API calls."""
import copy
import json
from pathlib import Path
from src.online_rl.clean import doctor_prompt as P
from src.online_rl.doctor import OllamaDoctor, STATUS_KEYS
from src.online_rl.tree.experiment import load_config

ROOT = Path(__file__).resolve().parents[1]
CONFIG = 'configs/online.json'
CASE = {'question': 'Which option?', 'options': {'A': 'alpha', 'B': 'beta'},
        'ground_truth': 'SECRET_GOLD', 'rubric': 'SECRET_RUBRIC', 'reward': 'SECRET_REWARD'}


def test_opening_does_not_force_a_followup_or_expose_private_fields():
    text = P.turn_prompt(CASE, [], 1, 4, True)
    assert 'Start the discussion.' in text
    assert 'STATUS:' not in text
    assert 'SECRET_' not in text
    assert 'weigh in on' not in text
    assert 'Case/vignette' not in text


def test_actual_transport_uses_profile_and_status_stays_hidden():
    cfg = load_config(CONFIG)
    requests = []
    status = dict(zip(STATUS_KEYS, ['A', False, False, False]))
    def http(url, payload, timeout):
        requests.append(payload)
        return {'message': {'content': 'Option A is better supported.\nSTATUS: '+json.dumps(status)},
                'prompt_eval_count': 100, 'eval_count': 35}
    ro = cfg['rollout']
    doctor = OllamaDoctor(ro['doctor_backend'], ro['halting'],
                          (ROOT/ro['doctor_system_prompt_file']).read_text(), ro['max_tokens_doctor'], http=http)
    result = doctor.turn(CASE, [{'speaker': 'agent', 'text': 'Consider option B.'}], 2, 4, 17)
    message = requests[0]['messages'][0]['content']
    assert "Respond to the colleague's latest argument." in message
    assert 'Consider option B.' in message
    assert 'SECRET_' not in message
    assert result['status'] == status and 'STATUS' not in result['visible']
    legacy = copy.deepcopy(ro['doctor_backend'])
    legacy.pop('discussion_prompt_profile')
    previous = OllamaDoctor(legacy, ro['halting'], doctor.system, ro['max_tokens_doctor'], http=http)
    assert doctor.protocol != previous.protocol
    assert doctor.identity['turn_prompt_sha256'] == P.prompt_sha256()


def test_online_config_keeps_other_settings_with_reference_rollout_and_two_turn_floor():
    old = load_config('configs/baseline_online.json')
    new = load_config(CONFIG)
    for section in ('policy', 'data', 'rubric', 'feedback', 'intervention', 'verification',
                    'optimizer', 'schedule', 'agent_reward', 'judges'):
        assert old[section] == new[section]
    assert new['rollout']['halting'] == {**old['rollout']['halting'], 'min_agent_turns': 2}
    assert new['rollout']['budget'] == 'reference_spine'
    assert new['five_gpu']['batch_cases'] == 5
    assert not new['rubric']['cumulative_bank'].get('offline_build')
    assert new['output_dir'] != old['output_dir']
