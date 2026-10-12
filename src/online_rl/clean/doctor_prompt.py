"""Doctor user-message construction only; system behavior lives in doctor_structured_v1.txt.

No model calls, sampling, outcome access, stopping decisions or reward logic.
Reuse the existing transport and STATUS parser while reviewing those separately.
"""
from pathlib import Path
from src.online_rl.integrated.records import sha256_text

PROFILE = 'structured_clinician_v1'
STATUS = '''Append one final line using exactly these fields:
STATUS: {{"current_option": "<{labels} or none>", "open_issue": <true|false>, "collaborator_objection": <true|false>, "more_info_could_change_decision": <true|false>}}
All fields describe what remains AFTER your reply:
- current_option: your leading option label, or none.
- open_issue: a substantive unresolved issue another exchange could meaningfully advance. Already-discussed ambiguity alone does not count.
- collaborator_objection: a substantive colleague argument still unaddressed or unreconciled. An argument addressed in your reply does not count even if disagreement remains.
- more_info_could_change_decision: another exchange could realistically provide reasoning or information that changes your choice. Repeated uncertainty and hypothetical unavailable evidence do not count.
If unsure about a boolean field, use true. The STATUS line is hidden from the colleague.'''


def turn_prompt(case, dialogue, turn, rounds, with_status):
    """Whitelist question/options/scenario and visible conversation; never serialize the case object."""
    sections = [f"Question:\n{case['question']}",
                'Options:\n' + '\n'.join(f'{label}. {text}' for label, text in case['options'].items())]
    if case.get('scenario'):
        sections.insert(0, f"Supplied scenario:\n{case['scenario']}")
    conversation = '\n\n'.join(f"{row['speaker'].title()}: {row['text']}" for row in dialogue)
    sections.append('Dialogue:\n' + (conversation or '(No previous dialogue.)'))
    sections.append('Start the discussion.' if turn == 1 else "Respond to the colleague's latest argument.")
    if with_status and turn >= 2:
        sections.append(STATUS.format(labels=', '.join(case['options'])))
    return '\n\n'.join(sections)


def prompt_sha256():
    """Fingerprint formatting and instructions for cache/run identity, without GPU or endpoint data."""
    return sha256_text(Path(__file__).read_text())
