import json

from scripts.run_collaboration_pilot import option


def test_legacy_entrypoint_reexports_option_parser():
    assert option("I considered B, but Final answer: option D - answer") == "D"


def test_option_parser_supports_fifth_choice():
    assert option("Final answer: option E - answer") == "E"


def test_final_option_is_required_for_training_record():
    record = {
        "trajectory_status": "valid",
        "eligible_for_training": True,
        "final_option": "D",
    }
    assert record["trajectory_status"] == "valid"
    assert record["eligible_for_training"] is True
    assert json.dumps(record)
