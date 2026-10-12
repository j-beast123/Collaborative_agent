"""Online Doctor stop rule: stable valid choices and no remaining substantive issue."""
from pathlib import Path
from src.online_rl.integrated.records import sha256_text

FLAGS = ('open_issue', 'collaborator_objection', 'more_info_could_change_decision')
KEYS = {'current_option', *FLAGS}


def valid_status(status, options):
    return (isinstance(status, dict) and set(status) == KEYS
            and isinstance(status['current_option'], str)
            and status['current_option'] in {str(label).upper() for label in options}
            and all(type(status[key]) is bool for key in FLAGS))


def halt_reason(*, agent_turns, status, prev_status, options, min_agent_turns=2):
    """Agent option mentions do not veto a resolved Doctor judgment; all three flags must be false."""
    if agent_turns < min_agent_turns:
        return None
    if not valid_status(status, options) or not valid_status(prev_status, options):
        return None
    if status['current_option'] != prev_status['current_option']:
        return None
    return 'doctor_halt' if not any(status[key] for key in FLAGS) else None


def protocol_sha256():
    return sha256_text(Path(__file__).read_text())
