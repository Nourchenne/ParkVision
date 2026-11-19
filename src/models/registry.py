"""Register existing ParkVision checkpoints and manage the champion alias.

No command trains a model. Remote writes occur only in register, promote-auto,
override, or rollback, and only when those commands are explicitly invoked.
"""

import argparse
import csv
import hashlib
import json
import math
import statistics
import subprocess
import sys
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZipFile

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.analysis.validate_model import file_hash, read_json, validate
from src.analysis.validation_contract import verify_contract
from src.models.registry_lock import registry_lock
from src.models.registry_policy import compare, load_policy

ENDPOINT = "https://dagshub.com/Nourchenne/ParkVision.mlflow"
MODEL = "ParkVision-Detector"
EVENT_EXPERIMENT = "ParkVision_Registry"


def digest_bytes(value):
    return hashlib.sha256(value).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def is_not_found(error):
    code = str(getattr(error, "error_code", ""))
    return code in ("RESOURCE_DOES_NOT_EXIST", "NOT_FOUND")


def registered_model(client):
    page_token = None
    while True:
        models = (client.search_registered_models(page_token=page_token) if page_token else
                  client.search_registered_models())
        for model in models:
            if model.name == MODEL:
                return model
        page_token = getattr(models, "token", None)
        if not page_token:
            return None


def expected_alias(client):
    model = registered_model(client)
    if model is None or "champion" not in model.aliases:
        return None
    return str(client.get_model_version_by_alias(MODEL, "champion").version)


def read_latency(path, protocol):
    """Summarize supplied benchmark samples; no latency number is invented."""
    data = read_json(path)
    required = {"samples_ms", "hardware", "runtime", "device", "input_shape",
                "batch_size", "warmup_runs", "method", "confidence", "nms_iou", "image_size"}
    if set(data) != required:
        raise ValueError("latency evidence has missing or unknown fields")
    samples = data["samples_ms"]
    if not isinstance(samples, list) or not samples or any(
        isinstance(value, bool) or not isinstance(value, (int, float)) or
        not math.isfinite(value) or value <= 0 for value in samples
    ):
        raise ValueError("latency samples must be positive finite numbers")
    for key in ("hardware", "runtime", "device", "input_shape", "method"):
        if not isinstance(data[key], str) or not data[key].strip():
            raise ValueError(f"latency {key} is missing")
    for key in ("batch_size", "warmup_runs"):
        if isinstance(data[key], bool) or not isinstance(data[key], int) or data[key] < 1:
            raise ValueError(f"latency {key} must be a positive integer")
    for key in ("confidence", "nms_iou", "image_size"):
        if data[key] != protocol[key]:
            raise ValueError(f"latency {key} differs from evaluation")
    conditions = {key: value for key, value in data.items() if key != "samples_ms"}
    return {"milliseconds": statistics.median(samples), "sample_count": len(samples),
            "warmup_runs": data["warmup_runs"],
            "protocol_id": digest_bytes(canonical(conditions).encode()),
            "evidence_sha256": file_hash(path)}


def read_evaluation_environment(path, commit, code_hash, image_hash, raw_dvc_hash,
                                equivalent_dvc_hashes=()):
    data = read_json(path)
    expected = {
        "git_commit": (commit, "counting summary"),
        "analysis_code_sha256": (code_hash, "evaluator code at the recorded Git commit"),
        "validation_image_names_sha256": (image_hash, "counting CSV image names"),
        "raw_dvc_sha256": (raw_dvc_hash, "training export DVC metadata"),
    }
    required = {*expected, "ultralytics_version"}
    missing, unexpected = required - set(data), set(data) - required
    if missing:
        raise ValueError(f"evaluation environment missing fields: {', '.join(sorted(missing))}")
    if unexpected:
        raise ValueError(f"evaluation environment unexpected fields: {', '.join(sorted(unexpected))}")
    mismatched = [f"{key} differs from {source}" for key, (value, source) in expected.items()
                  if data[key] != value and not (key == "raw_dvc_sha256" and
                                                data[key] in equivalent_dvc_hashes)]
    if mismatched:
        raise ValueError("evaluation environment mismatch: " + "; ".join(mismatched))
    if not isinstance(data["ultralytics_version"], str) or not data["ultralytics_version"]:
        raise ValueError("evaluation environment ultralytics_version is missing or invalid")
    return data


def raw_dvc_identity(content):
    """Dataset identity comes from the DVC tree hash, not platform line endings."""
    import re

    data = yaml.safe_load(content.decode("utf-8-sig"))
    outs = data.get("outs") if isinstance(data, dict) else None
    if not isinstance(outs, list) or len(outs) != 1 or not isinstance(outs[0], dict):
        raise ValueError("expected one raw dataset DVC output")
    out = outs[0]
    if out.get("path") != "raw" or not re.fullmatch(r"[0-9a-f]{32}\.dir", str(out.get("md5"))):
        raise ValueError("invalid raw DVC dataset identity")
    for key in ("size", "nfiles"):
        if type(out.get(key)) is not int or out[key] <= 0:
            raise ValueError("invalid raw DVC " + key)
    return {key: out[key] for key in ("md5", "size", "nfiles")}


def prepare(archive, checkpoint, manifest, counting_csv, counting_summary,
            imported_run_id, quality_config=ROOT / "configs/model_quality.yaml",
            latency_evidence=None, evaluation_environment=None,
            evaluation_protocol=ROOT / "configs/evaluation_protocol.yaml"):
    """Build verified local evidence; no MLflow connection or remote writes."""
    archive, checkpoint, manifest = map(Path, (archive, checkpoint, manifest))
    if len(imported_run_id) != 32 or any(c not in "0123456789abcdef" for c in imported_run_id):
        raise ValueError("imported run ID must be 32 lowercase hexadecimal characters")
    with ZipFile(archive) as stream:
        if len(stream.namelist()) != len(set(stream.namelist())):
            raise ValueError("duplicate ZIP entries")
        metadata = json.loads(stream.read("metadata.json"))
        def artifact(name):
            content = stream.read("artifacts/" + name)
            if digest_bytes(content) != metadata["artifact_hashes"][name]:
                raise ValueError(f"export artifact hash mismatch: {name}")
            return content
        artifact("training/weights/best.pt")
        environment = json.loads(artifact("provenance/environment.json"))
        dataset_dvc = artifact("provenance/raw.dvc")
        config_artifacts = [name for name in metadata["artifact_hashes"]
                            if name.startswith("provenance/train_") and name.endswith(".yaml")]
        if len(config_artifacts) != 1:
            raise ValueError("expected one training configuration in export")
        training_config = yaml.safe_load(artifact(config_artifacts[0]))
    report = validate(quality_config, counting_csv, counting_summary, archive,
                      checkpoint=checkpoint, dataset_manifest=manifest)
    if report["status"] != "passed":
        raise ValueError(f"Phase 1 gate did not pass: {report['status']}")
    if not all(value["independently_verified"] for value in report["provenance"].values()):
        raise ValueError("independent checkpoint and manifest verification required")
    if metadata["tags"].get("type") != "training" or metadata["tags"].get("test_evaluated") != "false":
        raise ValueError("a finished, non-test training run is required")
    summary = read_json(counting_summary)
    validation = verify_contract(metadata, summary, counting_csv, manifest, evaluation_protocol)
    commit = metadata["tags"].get("git_commit")
    evaluation_commit = summary.get("commit_git")
    if not commit or not evaluation_commit:
        raise ValueError("training or evaluation Git commit is missing")
    try:
        evaluation_code = subprocess.check_output(
            ["git", "show", f"{evaluation_commit}:src/analysis/analyze_errors.py"], cwd=ROOT
        )
    except subprocess.CalledProcessError as error:
        raise ValueError("evaluation code commit is unavailable") from error
    with Path(counting_csv).open(encoding="utf-8-sig", newline="") as stream:
        images = sorted(row["image"] for row in csv.DictReader(stream))
    image_hash = digest_bytes(canonical(images).encode())
    code_hash = digest_bytes(evaluation_code)
    raw_dvc_hash = digest_bytes(dataset_dvc)
    raw_identity = raw_dvc_identity(dataset_dvc)
    normalised_dvc = dataset_dvc.replace(b"\r\n", b"\n")
    equivalent_dvc_hashes = {digest_bytes(normalised_dvc),
                             digest_bytes(normalised_dvc.replace(b"\n", b"\r\n"))}
    evaluation_runtime = (read_evaluation_environment(
        evaluation_environment, evaluation_commit, code_hash, image_hash, raw_dvc_hash,
        equivalent_dvc_hashes
    ) if evaluation_environment else None)
    protocol = {
        "manifest_sha256": report["provenance"]["dataset_manifest"]["sha256"],
        "raw_dvc_identity": raw_identity,
        "schema_version": 2,
        "validation_image_names_sha256": image_hash,
        "analysis_code_sha256": code_hash,
        "evaluation_ultralytics_version": (evaluation_runtime["ultralytics_version"]
                                             if evaluation_runtime else "unverified"),
        "detection_metric": "best_val/metrics/mAP50-95_B_",
        "class_names": ["space-empty", "space-occupied"],
        "confidence": summary.get("seuil_confiance"),
        "nms_iou": summary.get("seuil_nms_iou"),
        "matching_iou": summary.get("seuil_appariement_iou"),
        "image_size": summary.get("imgsz"),
    }
    if any(value is None for value in protocol.values()):
        raise ValueError("incomplete evaluation protocol")
    return {
        "original_run_id": metadata["source_run_id"],
        "imported_run_id": imported_run_id,
        "architecture": Path(training_config["model"]).stem,
        "training_git_commit": commit,
        "evaluation_git_commit": evaluation_commit,
        "checkpoint_sha256": report["provenance"]["checkpoint"]["sha256"],
        "manifest_sha256": protocol["manifest_sha256"],
        "training_config_sha256": metadata["artifact_hashes"][config_artifacts[0]],
        "training_script_sha256": metadata["tags"].get("train_script_sha256"),
        "evaluation_csv_sha256": file_hash(counting_csv),
        "evaluation_summary_sha256": file_hash(counting_summary),
        "quality_config_sha256": file_hash(quality_config),
        "export_sha256": file_hash(archive),
        "training_ultralytics_version": environment.get("ultralytics"),
        "evaluation_environment_sha256": (file_hash(evaluation_environment)
                                           if evaluation_environment else None),
        "quality_status": "passed", "verified": True,
        "validation_contract": validation,
        "training_raw_dvc_file_sha256": raw_dvc_hash,
        "comparison_verified": evaluation_runtime is not None,
        "metrics": report["metrics"],
        "protocol": protocol,
        "protocol_id": digest_bytes(canonical(protocol).encode()),
        "latency": read_latency(latency_evidence, protocol) if latency_evidence else None,
    }


def package_model(record, checkpoint, tracking_uri):
    """Create a separate preparation run; original imported runs stay untouched."""
    import mlflow
    from mlflow.models.signature import ModelSignature
    from mlflow.types.schema import ColSpec, Schema

    sys.path.insert(0, str(ROOT / "src"))
    from models.registry_adapter import ParkVisionDetector

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EVENT_EXPERIMENT)
    protocol = record["protocol"]
    with mlflow.start_run(run_name=f"package-{record['original_run_id'][:8]}") as run:
        with tempfile.TemporaryDirectory() as temporary:
            model_dir = Path(temporary) / "model"
            mlflow.pyfunc.save_model(
                path=str(model_dir),
                python_model=ParkVisionDetector(protocol["confidence"],
                                               protocol["nms_iou"], protocol["image_size"]),
                artifacts={"checkpoint": str(checkpoint)},
                code_paths=[str(ROOT / "src")],
                signature=ModelSignature(inputs=Schema([ColSpec("string", "image_path")])),
                pip_requirements=["mlflow==3.17.2", "ultralytics==8.4.201", "pandas"],
            )
            mlflow.log_artifacts(str(model_dir), "model")
        mlflow.set_tags({"parkvision.original_run_id": record["original_run_id"],
                         "parkvision.imported_run_id": record["imported_run_id"],
                         "parkvision.checkpoint_sha256": record["checkpoint_sha256"]})
        return f"runs:/{run.info.run_id}/model", run.info.run_id


def verify_inference(version, sample_image, tracking_uri):
    import mlflow
    import pandas as pd

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)
    model = mlflow.pyfunc.load_model(f"models:/{MODEL}/{version}")
    output = model.predict(pd.DataFrame({"image_path": [str(sample_image)]}))
    if len(output) != 1 or not {"empty_count", "occupied_count", "detections_json"}.issubset(output):
        raise ValueError("registered model returned an invalid inference result")
    if any(int(output.iloc[0][key]) < 0 for key in ("empty_count", "occupied_count")):
        raise ValueError("registered model returned invalid counts")
    json.loads(output.iloc[0]["detections_json"])


class Registry:
    def __init__(self, client, tracking_uri=ENDPOINT, lock_file=None,
                 packager=package_model, verifier=verify_inference):
        self.client = client
        self.tracking_uri = tracking_uri
        self.lock_file = lock_file
        self.packager = packager
        self.verifier = verifier

    def versions(self):
        if registered_model(self.client) is None:
            return []
        versions, token, seen = [], None, set()
        while True:
            page = self.client.search_model_versions(
                f"name = '{MODEL}'", **({"page_token": token} if token else {})
            )
            versions.extend(page)
            token = getattr(page, "token", None)
            if not token:
                return versions
            if token in seen:
                raise ValueError("repeated model-version page token")
            seen.add(token)

    def version_record(self, version):
        try:
            mv = self.client.get_model_version(MODEL, str(version))
        except Exception as error:
            if isinstance(error, KeyError) or is_not_found(error):
                raise ValueError(f"registry version {version} does not exist") from error
            raise
        if mv.tags.get("parkvision.inference_verified") != "true":
            raise ValueError("version has not passed exact-version inference verification")
        if "parkvision.candidate" not in mv.tags:
            raise ValueError("version lacks ParkVision candidate evidence")
        record = json.loads(mv.tags["parkvision.candidate"])
        if record.get("quality_status") != "passed" or not record.get("verified"):
            raise ValueError("version has invalid quality or provenance")
        if (mv.tags.get("parkvision.checkpoint_sha256") != record.get("checkpoint_sha256") or
            mv.tags.get("parkvision.imported_run_id") != record.get("imported_run_id")):
            raise ValueError("version provenance tags contradict candidate evidence")
        return record

    def register(self, record, checkpoint, sample_image):
        if record.get("quality_status") != "passed" or not record.get("verified"):
            raise ValueError("candidate is not eligible for registration")
        if file_hash(checkpoint) != record["checkpoint_sha256"]:
            raise ValueError("local checkpoint hash mismatch")
        with registry_lock(self.lock_file):
            run = self.client.get_run(record["imported_run_id"])
            tags = run.data.tags
            if (run.info.status != "FINISHED" or
                tags.get("parkvision.import_complete") != "true" or
                tags.get("parkvision.source_run_id") != record["original_run_id"] or
                tags.get("manifest_sha256") != record["manifest_sha256"] or
                tags.get("git_commit") != record["training_git_commit"] or
                tags.get("parkvision.archive_sha256") != record["export_sha256"]):
                raise ValueError("imported run lineage is incomplete or contradictory")
            remote = self.client.download_artifacts(
                record["imported_run_id"], "training/weights/best.pt")
            if file_hash(remote) != record["checkpoint_sha256"]:
                raise ValueError("remote checkpoint hash mismatch")
            for mv in self.versions():
                old = mv.tags
                if (old.get("parkvision.imported_run_id") == record["imported_run_id"] and
                    old.get("parkvision.checkpoint_sha256") == record["checkpoint_sha256"]):
                    if json.loads(old.get("parkvision.candidate", "{}")) != record:
                        raise ValueError("existing version has conflicting candidate evidence")
                    self.verifier(mv.version, sample_image, self.tracking_uri)
                    self.client.set_model_version_tag(MODEL, mv.version,
                                                      "parkvision.inference_verified", "true")
                    return str(mv.version)
            uri, package_run = self.packager(record, checkpoint, self.tracking_uri)
            try:
                self.client.get_registered_model(MODEL)
            except Exception as error:
                if not is_not_found(error):
                    raise
                self.client.create_registered_model(MODEL)
            version_tags = {
                "parkvision.candidate": canonical(record),
                "parkvision.imported_run_id": record["imported_run_id"],
                "parkvision.original_run_id": record["original_run_id"],
                "parkvision.checkpoint_sha256": record["checkpoint_sha256"],
                "parkvision.architecture": record["architecture"],
                "parkvision.training_git_commit": record["training_git_commit"],
                "parkvision.evaluation_git_commit": record["evaluation_git_commit"],
                "parkvision.manifest_sha256": record["manifest_sha256"],
                "parkvision.training_config_sha256": record["training_config_sha256"],
                "parkvision.quality_config_sha256": record["quality_config_sha256"],
                "parkvision.evaluation_protocol_id": record["protocol_id"],
                "parkvision.quality_status": record["quality_status"],
                "parkvision.packaging_code_sha256": file_hash(ROOT / "src/models/registry_adapter.py"),
                "parkvision.inference_verified": "false",
            }
            mv = self.client.create_model_version(MODEL, uri, run_id=package_run,
                                                  tags=version_tags)
            self.verifier(mv.version, sample_image, self.tracking_uri)
            self.client.set_model_version_tag(MODEL, mv.version,
                                              "parkvision.inference_verified", "true")
            return str(mv.version)

    def _event(self, action, candidate, previous, decision, reason):
        experiment = self.client.get_experiment_by_name(EVENT_EXPERIMENT)
        experiment_id = (experiment.experiment_id if experiment else
                         self.client.create_experiment(EVENT_EXPERIMENT))
        event_id = uuid.uuid4().hex
        run = self.client.create_run(experiment_id, tags={
            "parkvision.registry_event": "true", "parkvision.event_id": event_id,
            "parkvision.action": action,
        })
        event = {"id": event_id, "at_utc": datetime.now(timezone.utc).isoformat(),
                 "action": action, "candidate": candidate,
                 "previous_champion": previous, "decision": decision, "reason": reason}
        self.client.log_dict(run.info.run_id, event, "decision.json")
        return run.info.run_id, event

    def _move(self, action, version, expected, decision, reason):
        current = expected_alias(self.client)
        if current != expected:
            raise ValueError(f"stale champion: expected {expected}, found {current}")
        self.version_record(version)
        if current == str(version):
            return {"changed": False, "reason": "already champion"}
        event_run, event = self._event(action, str(version), current, decision, reason)
        try:
            self.client.set_registered_model_alias(MODEL, "champion", str(version))
            observed = expected_alias(self.client)
            if observed != str(version):
                raise RuntimeError(f"alias readback mismatch: {observed}")
            self.client.log_dict(event_run, {"status": "verified", "target": observed},
                                 "outcome.json")
            self.client.set_terminated(event_run, "FINISHED")
        except Exception as error:
            try:
                self.client.log_dict(event_run, {"status": "uncertain", "error": str(error)},
                                     "outcome.json")
                self.client.set_terminated(event_run, "FAILED")
            except Exception as log_error:
                raise RuntimeError(
                    "alias update outcome uncertain; audit write also failed; inspect alias and event"
                ) from log_error
            raise RuntimeError("alias update outcome uncertain; inspect alias and event before retry") from error
        return {"changed": True, "event": event}

    def promote_auto(self, version, expected, policy):
        if not policy["enabled"]:
            raise ValueError("automatic promotion is disabled")
        with registry_lock(policy["lock_file"]):
            current = expected_alias(self.client)
            if current != expected:
                raise ValueError(f"stale champion: expected {expected}, found {current}")
            candidate = self.version_record(version)
            champion = self.version_record(current) if current else None
            decision = compare(candidate, champion, policy)
            decision["policy_sha256"] = digest_bytes(canonical(policy).encode())
            decision["candidate_metrics"] = candidate["metrics"]
            decision["champion_metrics"] = champion["metrics"] if champion else None
            decision["protocol_id"] = candidate["protocol_id"]
            if not decision["promote"]:
                event_run, event = self._event("promote-auto", str(version), current,
                                               decision, "; ".join(decision["reasons"]))
                self.client.log_dict(event_run, {"status": "rejected"}, "outcome.json")
                self.client.set_terminated(event_run, "FINISHED")
                return {"changed": False, "decision": decision, "event": event}
            return self._move("promote-auto", version, expected, decision,
                              "; ".join(decision["reasons"]))

    def process(self, record, checkpoint, sample_image, policy):
        """Register, then compare/promote when explicitly enabled by policy."""
        if policy["enabled"] and Path(self.lock_file).resolve() != Path(policy["lock_file"]).resolve():
            raise ValueError("registration and promotion must use the same shared lock file")
        initial_champion = expected_alias(self.client)
        version = self.register(record, checkpoint, sample_image)
        result = {"version": version, "initial_champion": initial_champion}
        if policy["enabled"]:
            result["promotion"] = self.promote_auto(version, initial_champion, policy)
        else:
            result["promotion"] = {"changed": False, "reason": "automatic promotion disabled"}
        return result

    def manual(self, action, version, expected, reason):
        if action not in ("override", "rollback") or not reason.strip():
            raise ValueError("manual action and reason are required")
        with registry_lock(self.lock_file):
            if action == "rollback" and str(version) not in self.rollback_targets():
                raise ValueError("rollback target was not a previous champion")
            return self._move(action, version, expected, {"manual": True}, reason)

    def history(self):
        experiment = self.client.get_experiment_by_name(EVENT_EXPERIMENT)
        if experiment is None:
            return []
        runs = self.client.search_runs([experiment.experiment_id],
                                       filter_string="tags.`parkvision.registry_event` = 'true'")
        events = []
        for run in runs:
            decision = read_json(self.client.download_artifacts(run.info.run_id,
                                                                "decision.json"))
            try:
                outcome = read_json(self.client.download_artifacts(run.info.run_id,
                                                                   "outcome.json"))
            except Exception as error:
                if not is_not_found(error):
                    raise
                outcome = {"status": "pending"}
            events.append({**decision, "outcome": outcome})
        return events

    def rollback_targets(self):
        return {event["previous_champion"] for event in self.history()
                if event["outcome"].get("status") == "verified" and
                event["previous_champion"] is not None}


def main(argv=None):
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env", override=False)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracking-uri", default=ENDPOINT)
    sub = parser.add_subparsers(dest="action", required=True)
    prep = sub.add_parser("prepare")
    for key in ("archive", "checkpoint", "manifest", "counting-csv",
                "counting-summary", "imported-run-id", "output"):
        prep.add_argument("--" + key, required=True)
    prep.add_argument("--latency-evidence")
    prep.add_argument("--evaluation-environment")
    for action in ("inspect", "history", "register", "process", "compare", "promote-auto",
                   "override", "rollback"):
        cmd = sub.add_parser(action)
        if action in ("register", "process"):
            for key in ("candidate", "archive", "checkpoint", "manifest",
                        "counting-csv", "counting-summary", "sample-image", "lock-file"):
                cmd.add_argument("--" + key, required=True)
            cmd.add_argument("--latency-evidence")
            cmd.add_argument("--evaluation-environment")
        if action == "process":
            cmd.add_argument("--policy", default="configs/registry_policy.yaml")
        if action in ("compare", "promote-auto", "override", "rollback"):
            cmd.add_argument("--version", required=True)
            cmd.add_argument("--expected-champion", required=True,
                             help="Version number or 'none'")
        if action in ("compare", "promote-auto"):
            cmd.add_argument("--policy", default="configs/registry_policy.yaml")
        if action in ("override", "rollback"):
            cmd.add_argument("--reason", required=True)
            cmd.add_argument("--lock-file", required=True)
    args = parser.parse_args(argv)
    if args.action == "prepare":
        record = prepare(args.archive, args.checkpoint, args.manifest,
                         args.counting_csv, args.counting_summary, args.imported_run_id,
                         latency_evidence=args.latency_evidence,
                         evaluation_environment=args.evaluation_environment)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        print(args.output)
        return
    from mlflow import MlflowClient

    client = MlflowClient(tracking_uri=args.tracking_uri, registry_uri=args.tracking_uri)
    lock_file = getattr(args, "lock_file", None)
    service = Registry(client, args.tracking_uri, lock_file)
    expected = getattr(args, "expected_champion", None)
    if expected == "none":
        expected = None
    if args.action == "inspect":
        result = {"champion": expected_alias(client),
                  "versions": [{"version": mv.version, "tags": mv.tags}
                               for mv in service.versions()]}
    elif args.action == "history":
        result = service.history()
    elif args.action in ("register", "process"):
        record = read_json(args.candidate)
        verified = prepare(args.archive, args.checkpoint, args.manifest,
                           args.counting_csv, args.counting_summary,
                           record["imported_run_id"], latency_evidence=args.latency_evidence,
                           evaluation_environment=args.evaluation_environment)
        if record != verified:
            raise ValueError("candidate record differs from independently recomputed evidence")
        result = (service.process(verified, args.checkpoint, args.sample_image,
                                  load_policy(args.policy)) if args.action == "process" else
                  {"version": service.register(verified, args.checkpoint,
                                               args.sample_image)})
    elif args.action == "compare":
        policy = load_policy(args.policy)
        current = expected_alias(client)
        if current != expected:
            raise ValueError(f"stale champion: expected {expected}, found {current}")
        result = compare(service.version_record(args.version),
                         service.version_record(current) if current else None, policy)
    elif args.action == "promote-auto":
        result = service.promote_auto(args.version, expected, load_policy(args.policy))
    else:
        result = service.manual(args.action, args.version, expected, args.reason)
    print(json.dumps(result, indent=2, default=str))


def exit_on_mlflow_error(error):
    """Report remote failures without echoing exception text or credentials."""
    status = error.get_http_status_code()
    if status in (401, 403):
        raise SystemExit(f"DagsHub authentication or permission denied (HTTP {status})")
    raise SystemExit(f"MLflow request failed (HTTP {status}); inspect remote state before retrying")


if __name__ == "__main__":
    from mlflow.exceptions import MlflowException

    try:
        main()
    except MlflowException as error:
        exit_on_mlflow_error(error)
