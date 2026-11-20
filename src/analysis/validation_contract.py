"""Validation-only selection contract; no inference and no test image reads."""

import csv
import hashlib
import math
from pathlib import Path, PurePosixPath

import yaml


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def valid_basenames(manifest):
    """CSV uses unique filenames, as emitted by the historical analysis script."""
    groups, days, paths, names = set(), {}, set(), set()
    with Path(manifest).open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            split, day, image = row["split"], row["date"], row["image"]
            path = PurePosixPath(image)
            if (split not in {"train", "valid", "test"} or not day
                    or path.is_absolute() or ".." in path.parts or "\\" in image
                    or not image.startswith("data/raw/") or ":" in image):
                raise ValueError("Invalid split manifest row")
            if image in paths or days.setdefault(day, split) != split:
                raise ValueError("Duplicate image or shared day in manifest")
            paths.add(image)
            groups.add(split)
            if split == "valid":
                if path.name in names:
                    raise ValueError("Ambiguous validation filenames")
                names.add(path.name)
    if groups != {"train", "valid", "test"} or not names:
        raise ValueError("Manifest must contain three nonempty splits")
    return names


def verify_contract(metadata, summary, counting_csv, manifest, protocol):
    spec = yaml.safe_load(Path(protocol).read_text(encoding="utf-8-sig"))
    required = {"split", "manifest_sha256", "images", "confidence", "nms_iou", "matching_iou", "imgsz"}
    if not isinstance(spec, dict) or set(spec) != required or spec["split"] != "valid":
        raise ValueError("Invalid evaluation protocol")
    for key in ("images", "imgsz"):
        if type(spec[key]) is not int or spec[key] <= 0:
            raise ValueError("Invalid protocol " + key)
    for key in ("confidence", "nms_iou", "matching_iou"):
        value = spec[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("Invalid protocol " + key)
    tags = metadata.get("tags", {})
    if tags.get("type") != "training" or tags.get("test_evaluated") != "false":
        raise ValueError("Selection requires a training run with test_evaluated=false")
    manifest_hash = sha256(manifest)
    if manifest_hash != spec["manifest_sha256"] or manifest_hash != tags.get("manifest_sha256"):
        raise ValueError("Manifest differs from fixed protocol or run")
    names = valid_basenames(manifest)
    with Path(counting_csv).open(encoding="utf-8-sig", newline="") as stream:
        evaluated = [row["image"] for row in csv.DictReader(stream)]
    if len(names) != spec["images"] or len(evaluated) != len(names) or set(evaluated) != names:
        raise ValueError("Counting CSV must cover the complete validation split exactly")
    for summary_key, protocol_key in (
        ("seuil_confiance", "confidence"), ("seuil_nms_iou", "nms_iou"),
        ("seuil_appariement_iou", "matching_iou"), ("imgsz", "imgsz"),
    ):
        actual = summary.get(summary_key)
        if (isinstance(actual, bool) or not isinstance(actual, (int, float))
                or not math.isfinite(actual) or not math.isclose(actual, spec[protocol_key], rel_tol=0, abs_tol=1e-12)):
            raise ValueError("Counting protocol mismatch: " + summary_key)
    if summary.get("split", "valid") != "valid" or summary.get("validation_complete", True) is not True:
        raise ValueError("Partial or non-validation analysis")
    for key, actual in (("manifest_sha256", manifest_hash), ("comptage_sha256", sha256(counting_csv))):
        if key in summary and summary[key] != actual:
            raise ValueError("Counting summary hash mismatch: " + key)
    return {
        "complete_validation_set": True,
        "protocol_matches": True,
        "training_run": True,
        "test_not_evaluated_tag": True,
        "protocol_sha256": sha256(protocol),
        "counting_csv_sha256": sha256(counting_csv),
        "summary_manifest_hash_recorded": "manifest_sha256" in summary,
        "historical_git_dirty": tags.get("git_dirty", "unknown"),
        "limitation": "Recorded provenance and file consistency do not prove historical image/label bytes or deployment readiness.",
    }
