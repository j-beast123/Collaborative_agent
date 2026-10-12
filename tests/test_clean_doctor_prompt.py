"""Validate the new message path with fake transport; no model or API calls."""
import copy
import json
from pathlib import Path
from src.online_rl.clean import doctor_prompt as P
from src.online_rl.doctor import OpenAIDoctor, STATUS_KEYS
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


