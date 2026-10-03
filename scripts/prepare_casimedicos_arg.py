"""Download and convert all official English CasiMedicos-Arg splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARG_REPO = "HiTZ/casimedicos-arg"
ARG_REVISION = "54a07ccf2993afc42a368e04682125b6f7b13be4"
EXP_REPO = "HiTZ/casimedicos-exp"
EXP_REVISION = "771c0d7ab77881fda4c30631183ef242a8425445"
SPLIT_FILES = {
    "train": {
        "arg": "en/train_en_ordered.json",
        "companion": "data/en/en_train_casimedicos.jsonl",
    },
    "validation": {
        "arg": "en/validation_en_ordered.json",
        "companion": "data/en/en_dev_casimedicos.jsonl",
    },
    "test": {
        "arg": "en/test_en_ordered.json",
        "companion": "data/en/en_test_casimedicos.jsonl",
    },
}
EXPECTED_ROWS = {"train": 434, "validation": 63, "test": 125}
LETTERS = "ABCDE"


def download(repo: str, revision: str, filename: str, destination: Path) -> None:
    url = f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{filename}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=180) as response:
        destination.write_bytes(response.read())


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def normalize(companion: dict, arg_record: dict, split: str) -> dict:
    numbered_options = [
        (int(number), str(text).strip())
        for number, text in companion["options"].items()
        if text is not None and str(text).strip().lower() not in {"", "nan", "none", "null"}
    ]
    numbered_options.sort()
    if not numbered_options or len(numbered_options) > len(LETTERS):
        raise ValueError(f"unsupported options for CasiMedicos document {arg_record['id']}")
    number_to_letter = {
        number: LETTERS[index]
        for index, (number, _) in enumerate(numbered_options)
    }
    correct_number = int(companion["correct_option"])
    if correct_number not in number_to_letter:
        raise ValueError(f"correct option is missing for CasiMedicos document {arg_record['id']}")
    options = {number_to_letter[number]: text for number, text in numbered_options}
    correct_option = number_to_letter[correct_number]
    return {
        "case_id": f"casimedicos_arg_en_{split}_{arg_record['id']}",
        "source_dataset": ARG_REPO,
        "source_revision": ARG_REVISION,
        "source_split": split,
        "source_question_id": arg_record["id"],
        "language": "en",
        "question": companion["full_question"],
        "scenario": "",
        "options": options,
        "ground_truth": {
            "correct_option": correct_option,
            "correct_text": options[correct_option],
        },
        "reference_explanation": companion.get("full_answer"),
        "subject": companion.get("type"),
        "argument_annotations": {
            "text": arg_record["text"],
            "labels": arg_record["labels"],
        },
        "metadata": {
            "year": companion.get("year"),
            "question_id_specific": companion.get("question_id_specific"),
            "companion_dataset": EXP_REPO,
            "companion_revision": EXP_REVISION,
        },
    }


def load_split(split: str, raw_dir: Path, reuse_downloads: bool) -> tuple[list[dict], dict]:
    arg_path = raw_dir / f"casimedicos_arg_en_{split}_ordered.json"
    companion_path = raw_dir / f"casimedicos_exp_en_{split}.jsonl"
    source_files = SPLIT_FILES[split]
    if not reuse_downloads or not arg_path.exists():
        download(ARG_REPO, ARG_REVISION, source_files["arg"], arg_path)
    if not reuse_downloads or not companion_path.exists():
        download(EXP_REPO, EXP_REVISION, source_files["companion"], companion_path)

    arg_rows = json.loads(arg_path.read_text(encoding="utf-8"))
    arg_by_id = {str(row["id"]): row for row in arg_rows}
    companion_rows = read_jsonl(companion_path)
    companion_by_id = {
        f"{row['id']}_{row['question_id_specific']}": row
        for row in companion_rows
    }
    missing_companions = sorted(set(arg_by_id) - set(companion_by_id))
    missing_annotations = sorted(set(companion_by_id) - set(arg_by_id))
    if missing_companions:
        raise ValueError(f"{split}: missing companion records: {missing_companions[:5]}")
    if missing_annotations:
        raise ValueError(f"{split}: missing argument records: {missing_annotations[:5]}")

    rows = [
        normalize(companion_by_id[source_id], arg_by_id[source_id], split)
        for source_id in companion_by_id
    ]
    expected = EXPECTED_ROWS[split]
    if len(rows) != expected:
        raise ValueError(f"{split}: expected {expected} rows, found {len(rows)}")
    if len(rows) != len({row["case_id"] for row in rows}):
        raise ValueError(f"{split}: duplicate CasiMedicos case_id")

    metadata = {
        "source_rows": len(arg_rows),
        "companion_rows": len(companion_rows),
        "output_rows": len(rows),
        "argument_annotated_rows": sum(bool(row.get("text")) for row in arg_rows),
        "rows_without_argument_annotations": sum(not bool(row.get("text")) for row in arg_rows),
        "raw_files": {
            str(arg_path): sha256(arg_path),
            str(companion_path): sha256(companion_path),
        },
    }
    return rows, metadata


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data/raw/casimedicos_arg")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data/interim/casimedicos_arg",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=tuple(SPLIT_FILES),
        default=list(SPLIT_FILES),
    )
    parser.add_argument("--reuse-downloads", action="store_true")
    args = parser.parse_args()

    all_rows: list[dict] = []
    split_metadata: dict[str, dict] = {}
    for split in args.splits:
        rows, metadata = load_split(split, args.raw_dir, args.reuse_downloads)
        output_path = args.output_dir / f"en_{split}.jsonl"
        write_jsonl(output_path, rows)
        metadata["output"] = str(output_path)
        metadata["output_sha256"] = sha256(output_path)
        split_manifest_path = args.output_dir / f"en_{split}.manifest.json"
        split_manifest = {
            "source_dataset": ARG_REPO,
            "source_revision": ARG_REVISION,
            "companion_dataset": EXP_REPO,
            "companion_revision": EXP_REVISION,
            "source_split": split,
            "language": "en",
            **metadata,
            "license": "CC BY 4.0",
        }
        split_manifest_path.write_text(
            json.dumps(split_manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        split_metadata[split] = metadata
        all_rows.extend(rows)

    if len(all_rows) != len({row["case_id"] for row in all_rows}):
        raise ValueError("duplicate CasiMedicos case_id across splits")
    combined_path = args.output_dir / "en_all.jsonl"
    write_jsonl(combined_path, all_rows)
    manifest = {
        "source_dataset": ARG_REPO,
        "source_revision": ARG_REVISION,
        "companion_dataset": EXP_REPO,
        "companion_revision": EXP_REVISION,
        "language": "en",
        "splits": split_metadata,
        "total_output_rows": len(all_rows),
        "combined_output": str(combined_path),
        "combined_output_sha256": sha256(combined_path),
        "license": "CC BY 4.0",
    }
    manifest_path = args.output_dir / "en_all.manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
