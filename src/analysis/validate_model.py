"""Validate existing ParkVision validation artifacts without inference or MLflow."""

import argparse
import csv
import hashlib
import json
import math
import re
import sys
import zlib
from pathlib import Path
from zipfile import BadZipFile, ZipFile

import yaml

KEYS = ("min_map50_95", "max_mae_empty", "max_mae_occupied", "min_exact_count_accuracy")
COLUMNS = (
    "image",
    "libres_annotees",
    "libres_predites",
    "occupees_annotees",
    "occupees_predites",
)
MAP_KEY = "best_val/metrics/mAP50-95_B_"
MAX_METADATA_BYTES = 2 * 1024 * 1024


def number(value, label, minimum=0, maximum=None):
    if isinstance(value, bool):
        raise TypeError(f"{label}: boolean is not a number")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label}: invalid number {value!r}") from error
    if (
        not math.isfinite(result)
        or result < minimum
        or (maximum is not None and result > maximum)
    ):
        raise ValueError(f"{label}: value outside valid range: {value!r}")
    return result


def integer(value, label):
    result = number(value, label)
    if result != int(result):
        raise ValueError(f"{label}: expected a nonnegative integer")
    return int(result)


def digest(value, label):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", value):
        raise ValueError(f"{label}: expected SHA256 hex digest")
    return value.lower()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def parse_json_object(content, label):
    data = json.loads(
        content,
        parse_constant=lambda x: (_ for _ in ()).throw(
            ValueError(f"Invalid JSON number: {x}")
        ),
    )
    if not isinstance(data, dict):
        raise TypeError(f"{label}: expected JSON object")
    return data


def read_json(path):
    return parse_json_object(Path(path).read_text(encoding="utf-8"), path)


def read_metrics(path):
    if Path(path).suffix.lower() == ".zip":
        try:
            with ZipFile(path) as archive:
                info = archive.getinfo("metadata.json")
                if info.file_size > MAX_METADATA_BYTES:
                    raise ValueError("metadata.json exceeds 2 MiB size limit")
                with archive.open(info) as stream:
                    content = stream.read(MAX_METADATA_BYTES + 1)
                if len(content) > MAX_METADATA_BYTES:
                    raise ValueError("metadata.json exceeds 2 MiB size limit")
        except (BadZipFile, EOFError, RuntimeError, zlib.error) as error:
            raise ValueError(f"invalid ZIP metadata: {error}") from error
        return parse_json_object(content.decode("utf-8"), "metadata.json")
    return read_json(path)


def read_config(path):
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(config, dict) or not isinstance(config.get("quality_gate"), dict):
        raise TypeError("quality_gate: expected mapping")
    gate = config["quality_gate"]
    if set(gate) != set(KEYS):
        raise ValueError(f"quality_gate: expected exactly {', '.join(KEYS)}")
    return {
        key: number(gate[key], key, maximum=None if key.startswith("max_mae") else 1)
        for key in KEYS
    }


def validate(
    config,
    counting_csv,
    summary_path,
    detection_path,
    expected_dataset=None,
    expected_model=None,
    checkpoint=None,
    dataset_manifest=None,
):
    """Return a report; raise ValueError for missing or contradictory evidence."""
    gate = read_config(config)
    with Path(counting_csv).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or not set(COLUMNS).issubset(reader.fieldnames):
            raise ValueError("comptage.csv: missing required columns")
        rows = list(reader)
    if not rows:
        raise ValueError("comptage.csv: no evaluated images")
    gaps_empty = gaps_occupied = exact = 0
    totals = {key: 0 for key in COLUMNS[1:]}
    images = set()
    for i, row in enumerate(rows, 2):
        image = row["image"]
        if not image or image in images:
            raise ValueError(f"comptage.csv row {i}: missing or duplicate image")
        images.add(image)
        counts = {key: integer(row[key], f"row {i} {key}") for key in COLUMNS[1:]}
        for key, value in counts.items():
            totals[key] += value
        e = abs(counts["libres_predites"] - counts["libres_annotees"])
        o = abs(counts["occupees_predites"] - counts["occupees_annotees"])
        gaps_empty += e
        gaps_occupied += o
        exact += e == 0 and o == 0
    summary = read_json(summary_path)
    n = len(rows)
    if integer(summary.get("images_evaluees"), "images_evaluees") != n:
        raise ValueError("inconsistent image counts between CSV and summary")
    for key, value in (
        ("mae_libres", gaps_empty / n),
        ("mae_occupees", gaps_occupied / n),
    ):
        if not math.isclose(number(summary.get(key), key), value, abs_tol=1e-9):
            raise ValueError(f"inconsistent {key} between CSV and summary")
    for column, total in totals.items():
        key = "total_" + column
        if key in summary and integer(summary[key], key) != total:
            raise ValueError(f"inconsistent {key} between CSV and summary")
    metadata = read_metrics(detection_path)
    if (
        metadata.get("format") != "parkvision-run-v1"
        or metadata.get("status") != "FINISHED"
    ):
        raise ValueError(
            "detection evidence must be a finished parkvision-run-v1 export"
        )
    history = metadata.get("metrics")
    if not isinstance(history, list) or any(not isinstance(m, dict) for m in history):
        raise TypeError("metrics: expected list of objects")
    matches = [m["value"] for m in history if m.get("key") == MAP_KEY]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one best checkpoint metric {MAP_KEY}")
    model = digest(summary.get("poids_sha256"), "summary checkpoint SHA256")
    recorded_model = digest(
        metadata.get("artifact_hashes", {}).get("training/weights/best.pt"),
        "export checkpoint SHA256",
    )
    if model != recorded_model:
        raise ValueError(
            "checkpoint SHA256 differs between counting and detection evidence"
        )
    dataset = digest(
        metadata.get("tags", {}).get("manifest_sha256"), "recorded manifest SHA256"
    )
    for expected, actual, label in (
        (expected_model, model, "model"),
        (expected_dataset, dataset, "dataset"),
    ):
        if (
            expected is not None
            and digest(expected, f"expected {label} SHA256") != actual
        ):
            raise ValueError(f"{label} SHA256 mismatch")
    for path, recorded, label in (
        (checkpoint, model, "checkpoint"),
        (dataset_manifest, dataset, "dataset manifest"),
    ):
        if path is not None and file_hash(path) != recorded:
            raise ValueError(f"{label} file SHA256 mismatch")
    metrics = {
        "map50_95": number(matches[0], "map50_95", maximum=1),
        "mae_empty": gaps_empty / n,
        "mae_occupied": gaps_occupied / n,
        "exact_count_accuracy": exact / n,
    }
    checks = {
        "map50_95": metrics["map50_95"] >= gate["min_map50_95"],
        "mae_empty": metrics["mae_empty"] <= gate["max_mae_empty"],
        "mae_occupied": metrics["mae_occupied"] <= gate["max_mae_occupied"],
        "exact_count_accuracy": metrics["exact_count_accuracy"]
        >= gate["min_exact_count_accuracy"],
    }
    return {
        "status": "passed" if all(checks.values()) else "quality_failed",
        "checks": checks,
        "metrics": metrics,
        "images_evaluated": n,
        "provenance": {
            "checkpoint": {
                "sha256": model,
                "independently_verified": checkpoint is not None,
            },
            "dataset_manifest": {
                "sha256": dataset,
                "independently_verified": dataset_manifest is not None,
            },
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/model_quality.yaml")
    )
    parser.add_argument("--counting-csv", type=Path)
    parser.add_argument("--counting-summary", type=Path)
    parser.add_argument("--detection-metrics", type=Path)
    parser.add_argument("--expected-dataset-sha256")
    parser.add_argument("--expected-model-sha256")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Optional checkpoint file for independent hash verification",
    )
    parser.add_argument(
        "--dataset-manifest",
        type=Path,
        help="Optional manifest file for independent hash verification",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check-config", action="store_true")
    args = parser.parse_args(argv)
    if not args.check_config and any(
        getattr(args, key) is None
        for key in ("counting_csv", "counting_summary", "detection_metrics", "output")
    ):
        parser.error(
            "counting CSV, summary, detection metrics, and output are required"
        )
    try:
        report = (
            {"status": "passed", "quality_gate": read_config(args.config)}
            if args.check_config
            else validate(
                args.config,
                args.counting_csv,
                args.counting_summary,
                args.detection_metrics,
                args.expected_dataset_sha256,
                args.expected_model_sha256,
                args.checkpoint,
                args.dataset_manifest,
            )
        )
        code = 0 if report["status"] == "passed" else 1
    except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError) as error:
        report, code = {"status": "invalid_evidence", "error": str(error)}, 2
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
    print(
        (
            "Invalid evidence: " + report["error"]
            if code == 2
            else "Quality thresholds failed: "
            + ", ".join(k for k, v in report["checks"].items() if not v)
            if code == 1
            else "Quality gate passed"
        ),
        file=sys.stderr if code else sys.stdout,
    )
    return code


if __name__ == "__main__":
    sys.exit(main())
