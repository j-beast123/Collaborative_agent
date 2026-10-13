"""RQ2 row contracts, without scoring, aggregation or figure rendering."""
import csv
import io
from pathlib import Path

COMMON = "schema_version run_id source_dataset dataset_version split_manifest_hash split original_case_id case_id history_id pair_id model_family model_id checkpoint iteration training_seed response_seed system reference_protocol_version judge_version expected_behavior_class predicted_behavior_class appropriateness_match evaluation_status exclusion_reason synthetic".split()
EXTRA = {"state": "collaboration_state agent_initial_correctness collaborator_correctness".split(),
         "controlled": "condition_id initial_state_stratum claim_correctness evidence_strength controlled_input_hash validation_status".split()}
ACTIONS = {"ACCEPT_REVISE", "MAINTAIN_CHALLENGE", "CONSIDER_REQUEST_EVIDENCE", "OTHER"}
CONDITIONS = {"correct_strong":("incorrect",True,"strong"), "correct_weak":("incorrect",True,"weak"),
              "incorrect_strong_looking":("correct",False,"strong_looking"), "incorrect_none":("correct",False,"none")}

def validate_rows(rows, kind, *, synthetic=False):
    if kind not in EXTRA or type(synthetic) is not bool:
        raise ValueError("Unknown figure kind/mode")
    rows = list(rows)
    if not rows:
        raise ValueError("No result rows; refusing fabricated output")
    keys = set()
    pairs = {}
    for row in rows:
        if set(row) != set(COMMON + EXTRA[kind]):
            raise ValueError("Figure schema fields do not match contract")
        if type(row["synthetic"]) is not bool or row["synthetic"] != synthetic:
            raise ValueError("Synthetic/mixed rows forbidden in this output mode")
        if row["schema_version"] != "1" or row["split"] not in ("train","dev","eval") or row["system"] not in ("base","serl"):
            raise ValueError("Invalid schema version/split/system")
        for k in ("iteration","training_seed","response_seed"):
            if type(row[k]) is not int or row[k] < 0:
                raise ValueError("Expected non-negative integer: " + k)
        nullable = {"expected_behavior_class","predicted_behavior_class","appropriateness_match","exclusion_reason"}
        boolean = {"synthetic","agent_initial_correctness","collaborator_correctness","claim_correctness"}
        for k in set(row) - nullable - boolean - {"iteration","training_seed","response_seed"}:
            if not isinstance(row[k],str) or not row[k].strip():
                raise ValueError("Expected nonempty string: " + k)
        for k in boolean & set(row):
            if type(row[k]) is not bool:
                raise ValueError("Expected boolean: " + k)
        for k in ("expected_behavior_class","predicted_behavior_class"):
            if row[k] is not None and row[k] not in ACTIONS:
                raise ValueError("Invalid action label")
        if row["evaluation_status"] == "scored":
            if row["expected_behavior_class"] is None or row["predicted_behavior_class"] is None or type(row["appropriateness_match"]) is not int or row["appropriateness_match"] != int(row["expected_behavior_class"] == row["predicted_behavior_class"]) or row["exclusion_reason"] is not None:
                raise ValueError("Inconsistent scored action/match")
        elif row["evaluation_status"] in ("failed","excluded","pending"):
            if row["appropriateness_match"] is not None or not isinstance(row["exclusion_reason"],str) or not row["exclusion_reason"].strip():
                raise ValueError("Unscored rows need null match and explicit reason")
        else:
            raise ValueError("Invalid evaluation_status")
        if kind == "state":
            state = {(True,True):"Preserve",(False,True):"Accept",(True,False):"Challenge",(False,False):"Recover"}[(row["agent_initial_correctness"],row["collaborator_correctness"])]
            if row["collaboration_state"] != state:
                raise ValueError("State/correctness mismatch")
            condition = state
        else:
            condition = row["condition_id"]
            if CONDITIONS.get(condition) != (row["initial_state_stratum"],row["claim_correctness"],row["evidence_strength"]):
                raise ValueError("Controlled condition mismatch")
            if row["validation_status"] not in ("accepted","rejected","pending") or (row["evaluation_status"] == "scored" and row["validation_status"] != "accepted"):
                raise ValueError("Scored controlled rows must be validated")
        key = tuple(row[k] for k in ("run_id","source_dataset","case_id","history_id","pair_id","model_family","model_id","checkpoint","iteration","training_seed","response_seed","system")) + (condition,)
        if key in keys:
            raise ValueError("Duplicate figure observation")
        keys.add(key)
        if kind == "controlled":
            # pair_id identifies a comparison across runs/checkpoints, not just a history.
            pair = tuple(row[k] for k in ("source_dataset","dataset_version","split_manifest_hash","pair_id","model_family","training_seed","response_seed")) + (condition,)
            signature = tuple(row[k] for k in ("original_case_id","case_id","history_id","split","controlled_input_hash","expected_behavior_class","reference_protocol_version"))
            if pair in pairs and pairs[pair] != signature:
                raise ValueError("Paired inputs/reference differ between systems")
            pairs[pair] = signature
    return rows

def export_rows(rows, kind, path, *, synthetic=False):
    rows = validate_rows(rows, kind, synthetic=synthetic)
    path = Path(path).resolve()
    repo = Path(__file__).resolve().parents[2]
    if synthetic:
        allowed = [(repo / "artifacts/smoke").resolve(), (repo / "tests/fixtures/synthetic").resolve()]
        if not any(p in path.parents for p in allowed):
            raise ValueError("Synthetic output must stay in fixture/smoke directories")
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=COMMON + EXTRA[kind])
    writer.writeheader()
    for row in rows:
        writer.writerow({k: ("true" if v else "false") if type(v) is bool else v for k, v in row.items()})
    text = output.getvalue()
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive create: never silently replace results from another run.
    with path.open("x", newline="") as f:
        f.write(text)
