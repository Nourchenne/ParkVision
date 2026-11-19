"""Registry and promotion tests; no GPU, network, or DagsHub credentials."""

import csv
import hashlib
import importlib.util
import io
import json
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from zipfile import ZipFile

from src.analysis import analyze_errors
from src.models.registry import (
    Registry,
    canonical,
    expected_alias,
    main,
    prepare,
    read_evaluation_environment,
    read_latency,
    verify_inference,
)
from src.models.registry_policy import compare, load_policy


def candidate(exact=0.8, protocol="same", **changes):
    record = {
        "quality_status": "passed", "verified": True,
        "comparison_verified": True, "protocol_id": protocol,
        "metrics": {"exact_count_accuracy": exact, "map50_95": 0.9,
                    "mae_empty": 0.2, "mae_occupied": 0.1},
        "latency": None,
    }
    record.update(changes)
    return record


def policy(lock):
    return {"enabled": True, "lock_file": str(lock), "min_exact_gain": 0.01,
            "max_map_drop": 0.02, "max_mae_empty_increase": 0.05,
            "max_mae_occupied_increase": 0.05,
            "bootstrap": "manual",
            "latency": {"mode": "require_if_recorded", "max_relative_increase": 0.1,
                        "min_samples": 3, "min_warmup_runs": 2}}


class Missing(Exception):
    error_code = "RESOURCE_DOES_NOT_EXIST"


class InvalidParameter(Exception):
    error_code = "INVALID_PARAMETER_VALUE"


class FakeClient:
    def __init__(self, root, checkpoint):
        self.root = root
        self.checkpoint = checkpoint
        self.models = {}
        self.model_exists = False
        self.alias = None
        self.alias_reads = 0
        self.version_searches = 0
        self.experiment = None
        self.runs = {}
        self.fail_alias = False
        self.bad_readback = False
        self.events = {}
        self.imported = SimpleNamespace(
            info=SimpleNamespace(status="FINISHED"),
            data=SimpleNamespace(tags={"parkvision.import_complete": "true",
                                       "parkvision.source_run_id": "a" * 32,
                                       "manifest_sha256": "c" * 64,
                                       "git_commit": "d" * 40,
                                       "parkvision.archive_sha256": "e" * 64}),
        )

    def get_run(self, run_id):
        return self.imported

    def download_artifacts(self, run_id, artifact_path):
        if artifact_path == "training/weights/best.pt":
            return str(self.checkpoint)
        try:
            return str(self.events[run_id][artifact_path])
        except KeyError as error:
            raise Missing() from error

    def search_model_versions(self, filter_string):
        self.version_searches += 1
        if not self.model_exists and not self.models:
            raise AssertionError("version search must skip an absent model")
        return list(self.models.values())

    def search_registered_models(self, page_token=None):
        if self.model_exists or self.models:
            return [SimpleNamespace(name="ParkVision-Detector",
                                    aliases={"champion": self.alias} if self.alias else {})]
        return []

    def get_registered_model(self, name):
        if not self.models and not self.model_exists:
            raise Missing()
        return SimpleNamespace(name=name)

    def create_registered_model(self, name):
        self.model_exists = True
        return SimpleNamespace(name=name)

    def create_model_version(self, name, source, run_id, tags):
        version = str(len(self.models) + 1)
        mv = SimpleNamespace(version=version, tags=dict(tags), source=source, run_id=run_id)
        self.models[version] = mv
        return mv

    def set_model_version_tag(self, name, version, key, value):
        self.models[str(version)].tags[key] = value

    def get_model_version(self, name, version):
        return self.models[str(version)]

    def get_model_version_by_alias(self, name, alias):
        self.alias_reads += 1
        if self.alias is None:
            raise InvalidParameter(f"Registered model alias {alias} not found.")
        version = self.alias
        if self.bad_readback:
            self.bad_readback = False
            version = "999"
        return SimpleNamespace(version=version)

    def set_registered_model_alias(self, name, alias, version):
        if self.fail_alias:
            raise RuntimeError("alias write failed")
        self.alias = str(version)

    def get_experiment_by_name(self, name):
        return self.experiment

    def create_experiment(self, name):
        self.experiment = SimpleNamespace(experiment_id="1")
        return "1"

    def create_run(self, experiment_id, tags):
        run_id = str(len(self.runs) + 1)
        self.runs[run_id] = SimpleNamespace(info=SimpleNamespace(run_id=run_id),
                                            data=SimpleNamespace(tags=tags))
        self.events[run_id] = {}
        return self.runs[run_id]

    def log_dict(self, run_id, value, artifact_path):
        path = self.root / f"{run_id}-{artifact_path}"
        path.write_text(json.dumps(value), encoding="utf-8")
        self.events[run_id][artifact_path] = path

    def set_terminated(self, run_id, status):
        self.runs[run_id].info.status = status

    def search_runs(self, experiment_ids, filter_string):
        return list(self.runs.values())


class RegistryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.checkpoint = self.root / "best.pt"
        self.checkpoint.write_bytes(b"synthetic checkpoint")
        self.client = FakeClient(self.root, self.checkpoint)
        self.lock = self.root / "registry.lock"
        self.loaded = []
        self.service = Registry(
            self.client, "file:///fake", self.lock,
            packager=lambda record, checkpoint, uri: ("runs:/package/model", "package"),
            verifier=lambda version, sample, uri: self.loaded.append(str(version)),
        )
        import hashlib
        self.record = candidate(
            original_run_id="a" * 32, imported_run_id="b" * 32,
            checkpoint_sha256=hashlib.sha256(self.checkpoint.read_bytes()).hexdigest(),
            manifest_sha256="c" * 64,
            training_git_commit="d" * 40, export_sha256="e" * 64,
            evaluation_git_commit="d" * 40,
            architecture="yolo26n", training_config_sha256="f" * 64,
            quality_config_sha256="0" * 64, protocol_id="same",
        )

    def add_version(self, number, record):
        self.client.models[str(number)] = SimpleNamespace(
            version=str(number), tags={"parkvision.candidate": canonical(record),
                                       "parkvision.inference_verified": "true"})

    def test_inspect_empty_registry_and_model_without_alias(self):
        fake_mlflow = types.ModuleType("mlflow")
        fake_mlflow.MlflowClient = lambda **kwargs: self.client
        for model_exists in (False, True):
            with self.subTest(model_exists=model_exists):
                self.client.model_exists = model_exists
                output = io.StringIO()
                with patch.dict(sys.modules, {"mlflow": fake_mlflow}), redirect_stdout(output):
                    main(["inspect"])
                self.assertEqual(json.loads(output.getvalue()),
                                 {"champion": None, "versions": []})
                self.assertEqual(self.client.alias_reads, 0)
                if not model_exists:
                    self.assertEqual(self.client.version_searches, 0)

    def test_existing_model_with_champion_reads_alias(self):
        self.client.model_exists = True
        self.client.alias = "7"
        self.assertEqual(expected_alias(self.client), "7")
        self.assertEqual(self.client.alias_reads, 1)

    def test_existing_model_without_champion_skips_alias_endpoint(self):
        self.client.model_exists = True
        self.assertIsNone(expected_alias(self.client))
        self.assertEqual(self.client.alias_reads, 0)

    def test_registry_search_error_is_not_treated_as_missing_model(self):
        with (patch.object(self.client, "search_registered_models",
                           side_effect=RuntimeError("registry API unavailable")),
              self.assertRaisesRegex(RuntimeError, "registry API unavailable")):
            expected_alias(self.client)
        self.assertEqual(self.client.alias_reads, 0)

    def test_registry_search_checks_later_pages(self):
        class Page(list):
            pass

        first = Page([SimpleNamespace(name="another-model", aliases={})])
        first.token = "next"
        second = Page([SimpleNamespace(name="ParkVision-Detector",
                                       aliases={"champion": "7"})])
        second.token = None
        self.client.alias = "7"
        with patch.object(self.client, "search_registered_models",
                          side_effect=[first, second]) as search:
            self.assertEqual(expected_alias(self.client), "7")
        self.assertEqual(search.call_count, 2)
        self.assertEqual(self.client.alias_reads, 1)

    def test_missing_alias_errors_do_not_hide_other_failures(self):
        self.client.model_exists = True
        self.client.alias = "7"
        for error in (InvalidParameter("unsupported alias API"),
                      InvalidParameter("invalid model name"),
                      RuntimeError("permission denied"),
                      ConnectionError("network unavailable")):
            with (self.subTest(error=str(error)),
                  patch.object(self.client, "get_model_version_by_alias", side_effect=error),
                  self.assertRaises(type(error))):
                expected_alias(self.client)

    def test_missing_alias_paths_keep_manual_bootstrap(self):
        self.add_version(1, candidate())
        self.assertIsNone(expected_alias(self.client))
        self.assertFalse(self.service.promote_auto("1", None, policy(self.lock))["changed"])
        self.assertIsNone(self.client.alias)

        fake_mlflow = types.ModuleType("mlflow")
        fake_mlflow.MlflowClient = lambda **kwargs: self.client
        output = io.StringIO()
        with patch.dict(sys.modules, {"mlflow": fake_mlflow}), redirect_stdout(output):
            main(["compare", "--version", "1", "--expected-champion", "none"])
        self.assertFalse(json.loads(output.getvalue())["promote"])

    def test_rollback_can_read_missing_alias_after_recorded_champion(self):
        self.add_version(1, candidate())
        self.add_version(2, candidate(0.82))
        self.client.alias = "1"
        self.service.manual("override", "2", "1", "test override")
        self.client.alias = None
        result = self.service.manual("rollback", "1", None, "restore previous champion")
        self.assertTrue(result["changed"])
        self.assertEqual(self.client.alias, "1")

    def test_register_and_duplicate(self):
        version = self.service.register(self.record, self.checkpoint, "sample.jpg")
        self.assertEqual(version, "1")
        self.assertEqual(self.client.models[version].run_id, "package")
        self.assertEqual(self.loaded, ["1"])
        self.assertEqual(self.service.register(self.record, self.checkpoint, "sample.jpg"), "1")
        self.assertEqual(len(self.client.models), 1)

    def test_process_respects_disabled_and_enabled_policy(self):
        disabled = load_policy(Path(__file__).resolve().parents[1] /
                               "configs/registry_policy.yaml")
        result = self.service.process(self.record, self.checkpoint, "sample.jpg", disabled)
        self.assertFalse(result["promotion"]["changed"])
        self.assertIsNone(self.client.alias)
        enabled = policy(self.lock)
        enabled["bootstrap"] = "allow_eligible_candidate"
        result = self.service.process(self.record, self.checkpoint, "sample.jpg", enabled)
        self.assertTrue(result["promotion"]["changed"])
        self.assertEqual(self.client.alias, "1")

    def test_process_blocks_auto_without_evaluator_environment(self):
        p = policy(self.lock)
        p["bootstrap"] = "allow_eligible_candidate"
        record = {**self.record, "comparison_verified": False}
        result = self.service.process(record, self.checkpoint, "sample.jpg", p)
        self.assertFalse(result["promotion"]["changed"])
        self.assertIsNone(self.client.alias)

    def test_failed_gate_or_wrong_remote_hash_blocks_registration(self):
        with self.assertRaisesRegex(ValueError, "not eligible"):
            self.service.register({**self.record, "quality_status": "quality_failed"},
                                  self.checkpoint, "sample.jpg")
        self.client.checkpoint = self.root / "other.pt"
        self.client.checkpoint.write_bytes(b"different")
        with self.assertRaisesRegex(ValueError, "remote checkpoint hash mismatch"):
            self.service.register(self.record, self.checkpoint, "sample.jpg")

    def test_duplicate_with_conflicting_evidence_blocks(self):
        self.service.register(self.record, self.checkpoint, "sample.jpg")
        changed = {**self.record, "evaluation_csv_sha256": "different"}
        with self.assertRaisesRegex(ValueError, "conflicting"):
            self.service.register(changed, self.checkpoint, "sample.jpg")

    def test_policy_better_worse_missing_and_incomparable(self):
        p = policy(self.lock)
        champion = candidate(0.8)
        self.assertTrue(compare(candidate(0.82), champion, p)["promote"])
        self.assertFalse(compare(candidate(0.79), champion, p)["promote"])
        self.assertFalse(compare(candidate(0.82, protocol="other"), champion, p)["promote"])
        self.assertFalse(compare(candidate(0.82, metrics={}), champion, p)["promote"])
        self.assertFalse(compare(candidate(0.82, comparison_verified=False),
                                 champion, p)["promote"])
        self.assertFalse(compare(candidate(0.82, latency={"protocol_id": "cpu", "milliseconds": 8}),
                                 champion, p)["promote"])
        self.assertFalse(compare(candidate(0.82, metrics={**candidate()["metrics"],
                                                          "mae_empty": 0.3}), champion, p)["promote"])
        self.assertFalse(compare(candidate(0.82, latency={"protocol_id": "cpu", "milliseconds": 12,
                                                           "sample_count": 3, "warmup_runs": 2}),
                                 candidate(0.8, latency={"protocol_id": "cpu", "milliseconds": 10,
                                                          "sample_count": 3, "warmup_runs": 2}),
                                 p)["promote"])

    def test_no_champion_requires_explicit_bootstrap(self):
        p = policy(self.lock)
        self.assertFalse(compare(candidate(), None, p)["promote"])
        p["bootstrap"] = "allow_eligible_candidate"
        self.assertTrue(compare(candidate(), None, p)["promote"])

    def test_auto_promotion_and_stale_attempt(self):
        self.add_version(1, candidate(0.8))
        self.add_version(2, candidate(0.82))
        self.client.alias = "1"
        result = self.service.promote_auto("2", "1", policy(self.lock))
        self.assertTrue(result["changed"])
        self.assertEqual(self.client.alias, "2")
        self.assertEqual(self.service.history()[0]["previous_champion"], "1")
        with self.assertRaisesRegex(ValueError, "stale champion"):
            self.service.promote_auto("1", "1", policy(self.lock))

    def test_worse_candidate_keeps_champion(self):
        self.add_version(1, candidate(0.8))
        self.add_version(2, candidate(0.7))
        self.client.alias = "1"
        result = self.service.promote_auto("2", "1", policy(self.lock))
        self.assertFalse(result["changed"])
        self.assertEqual(self.client.alias, "1")
        self.assertEqual(self.service.history()[0]["outcome"]["status"], "rejected")

    def test_alias_write_and_readback_failures(self):
        self.add_version(1, candidate(0.8))
        self.add_version(2, candidate(0.82))
        self.client.alias = "1"
        self.client.fail_alias = True
        with self.assertRaisesRegex(RuntimeError, "outcome uncertain"):
            self.service.promote_auto("2", "1", policy(self.lock))
        self.assertEqual(self.client.alias, "1")
        self.client.fail_alias = False
        # The initial champion read succeeds; the readback returns the wrong version.
        original = self.client.set_registered_model_alias
        def bad_alias(name, alias, version):
            original(name, alias, version)
            self.client.bad_readback = True
        self.client.set_registered_model_alias = bad_alias
        with self.assertRaisesRegex(RuntimeError, "outcome uncertain"):
            self.service.promote_auto("2", "1", policy(self.lock))
        self.assertEqual(self.client.alias, "2")

    def test_override_and_rollback(self):
        self.add_version(1, candidate(0.8))
        self.add_version(2, candidate(0.82))
        self.client.alias = "1"
        self.service.manual("override", "2", "1", "reviewed by owner")
        self.assertEqual(self.client.alias, "2")
        self.service.manual("rollback", "1", "2", "regression found")
        self.assertEqual(self.client.alias, "1")
        with self.assertRaisesRegex(ValueError, "stale champion"):
            self.service.manual("override", "2", "2", "old request")

    def test_rollback_rejects_unknown_target(self):
        self.add_version(1, candidate())
        with self.assertRaisesRegex(ValueError, "not a previous champion"):
            self.service.manual("rollback", "1", None, "try")

    def test_override_rejects_invalid_version(self):
        with self.assertRaisesRegex(ValueError, "does not exist"):
            self.service.manual("override", "99", None, "reviewed")
        self.assertIsNone(self.client.alias)

    def test_disabled_policy_config(self):
        config = load_policy(Path(__file__).resolve().parents[1] /
                             "configs/registry_policy.yaml")
        self.assertFalse(config["enabled"])

    def test_latency_evidence_and_protocol(self):
        path = self.root / "latency.json"
        data = {"samples_ms": [9.0, 11.0, 10.0], "hardware": "fixture-cpu",
                "runtime": "fixture-runtime", "device": "cpu", "input_shape": "640x640",
                "batch_size": 1, "warmup_runs": 2, "method": "wall-clock",
                "confidence": 0.25, "nms_iou": 0.7, "image_size": 640}
        path.write_text(json.dumps(data), encoding="utf-8")
        result = read_latency(path, {"confidence": 0.25, "nms_iou": 0.7,
                                     "image_size": 640})
        self.assertEqual(result["milliseconds"], 10.0)
        self.assertEqual(result["sample_count"], 3)
        data["image_size"] = 320
        path.write_text(json.dumps(data), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "differs from evaluation"):
            read_latency(path, {"confidence": 0.25, "nms_iou": 0.7,
                                "image_size": 640})

    def test_evaluation_environment_requires_matching_evidence(self):
        path = self.root / "evaluation_environment.json"
        data = {"git_commit": "a" * 40, "analysis_code_sha256": "b" * 64,
                "validation_image_names_sha256": "c" * 64,
                "raw_dvc_sha256": "d" * 64, "ultralytics_version": "8.4.201"}
        path.write_text(json.dumps(data), encoding="utf-8")
        self.assertEqual(read_evaluation_environment(path, "a" * 40, "b" * 64,
                                                      "c" * 64, "d" * 64), data)
        data["git_commit"] = "e" * 40
        path.write_text(json.dumps(data), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "git_commit differs from counting summary"):
            read_evaluation_environment(path, "a" * 40, "b" * 64,
                                        "c" * 64, "d" * 64)

    def test_prepare_identifies_uncommitted_evaluator_code(self):
        checkpoint = self.root / "best.pt"
        checkpoint.write_bytes(b"checkpoint")
        checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        manifest = self.root / "split_manifest.csv"
        manifest.write_text("image,date,split\n"
                            "data/raw/train/images/a.jpg,2012-01-01,train\n"
                            "data/raw/valid/images/img.jpg,2012-01-02,valid\n"
                            "data/raw/test/images/z.jpg,2012-01-03,test\n", encoding="utf-8")
        manifest_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
        evaluation_protocol = self.root / "protocol.yaml"
        evaluation_protocol.write_text(json.dumps({"split": "valid", "images": 1,
            "manifest_sha256": manifest_hash, "confidence": 0.25, "nms_iou": 0.7,
            "matching_iou": 0.5, "imgsz": 640}), encoding="utf-8")
        counting_csv = self.root / "comptage.csv"
        with counting_csv.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("image", "libres_annotees", "libres_predites",
                             "occupees_annotees", "occupees_predites"))
            writer.writerow(("img.jpg", 1, 1, 1, 1))
        training_commit, evaluation_commit = "a" * 40, "b" * 40
        summary = self.root / "comptage_resume.json"
        summary.write_text(json.dumps({"images_evaluees": 1, "mae_libres": 0,
                                       "mae_occupees": 0, "poids_sha256": checkpoint_hash,
                                       "commit_git": evaluation_commit,
                                       "seuil_confiance": 0.25, "seuil_nms_iou": 0.7,
                                       "seuil_appariement_iou": 0.5, "imgsz": 640}),
                           encoding="utf-8")
        artifacts = {
            "training/weights/best.pt": checkpoint.read_bytes(),
            "provenance/environment.json": b'{"ultralytics":"8.4.201"}',
            "provenance/raw.dvc": (b"outs:\n- md5: " + b"a" * 32 +
                                   b".dir\n  path: raw\n  size: 20\n  nfiles: 2\n"),
            "provenance/train_c.yaml": b"model: yolo26n.pt\n",
        }
        archive = self.root / "export.zip"
        metadata = {
            "format": "parkvision-run-v1", "status": "FINISHED",
            "source_run_id": "c" * 32,
            "tags": {"type": "training", "test_evaluated": "false",
                     "git_commit": training_commit, "manifest_sha256": manifest_hash},
            "artifact_hashes": {name: hashlib.sha256(content).hexdigest()
                                for name, content in artifacts.items()},
            "metrics": [{"key": "best_val/metrics/mAP50-95_B_", "value": 0.9}],
        }
        with ZipFile(archive, "w") as stream:
            stream.writestr("metadata.json", json.dumps(metadata))
            for name, content in artifacts.items():
                stream.writestr("artifacts/" + name, content)
        committed_code = b"committed evaluator source"
        environment = self.root / "evaluation_environment.json"
        sidecar = {
            "git_commit": evaluation_commit,
            "analysis_code_sha256": hashlib.sha256(b"uncommitted evaluator source").hexdigest(),
            "validation_image_names_sha256": hashlib.sha256(
                canonical(["img.jpg"]).encode()
            ).hexdigest(),
            "raw_dvc_sha256": hashlib.sha256(artifacts["provenance/raw.dvc"]).hexdigest(),
            "ultralytics_version": "8.4.201",
        }
        environment.write_text(json.dumps(sidecar), encoding="utf-8")
        with patch("src.models.registry.subprocess.check_output", return_value=committed_code):
            with self.assertRaises(ValueError) as mismatch:
                prepare(archive, checkpoint, manifest, counting_csv, summary,
                        "d" * 32, evaluation_environment=environment,
                        evaluation_protocol=evaluation_protocol)
            self.assertIn("analysis_code_sha256 differs from evaluator code at the recorded Git commit",
                          str(mismatch.exception))
            self.assertNotIn(sidecar["analysis_code_sha256"], str(mismatch.exception))
            # A matching evaluator snapshot passes even though the training commit differs.
            sidecar["analysis_code_sha256"] = hashlib.sha256(committed_code).hexdigest()
            environment.write_text(json.dumps(sidecar), encoding="utf-8")
            record = prepare(archive, checkpoint, manifest, counting_csv, summary,
                             "d" * 32, evaluation_environment=environment,
                        evaluation_protocol=evaluation_protocol)
        self.assertEqual(record["training_git_commit"], training_commit)
        self.assertEqual(record["evaluation_git_commit"], evaluation_commit)
        self.assertTrue(record["comparison_verified"])

    def test_evaluator_emits_verifiable_environment_fields(self):
        (self.root / "data").mkdir()
        (self.root / "data/raw.dvc").write_text("fixture", encoding="utf-8")
        with patch.object(analyze_errors, "ROOT", self.root):
            record = analyze_errors.evaluation_environment(
                [{"image": "b.jpg"}, {"image": "a.jpg"}], "a" * 40, "8.4.201"
            )
        self.assertEqual(record["ultralytics_version"], "8.4.201")
        self.assertEqual(record["validation_image_names_sha256"],
                         hashlib.sha256(canonical(["a.jpg", "b.jpg"]).encode()).hexdigest())


class ModelPackagingTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("mlflow"), "MLflow not installed")
    def test_packaged_adapter_loads_and_predicts(self):
        import mlflow.pyfunc
        import pandas as pd
        from mlflow.models.signature import ModelSignature
        from mlflow.types.schema import ColSpec, Schema

        root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(root / "src"))
        from models.registry_adapter import ParkVisionDetector

        class Values:
            def __init__(self, values):
                self.values = values

            def tolist(self):
                return self.values

        class FakeYOLO:
            def __init__(self, path):
                self.path = path
                self.names = {0: "space-empty", 1: "space-occupied"}

            def predict(self, **kwargs):
                boxes = SimpleNamespace(cls=Values([0, 1]), conf=Values([0.8, 0.9]),
                                        xyxy=Values([[1, 2, 3, 4], [5, 6, 7, 8]]))
                return [SimpleNamespace(boxes=boxes)]

        fake_ultralytics = types.ModuleType("ultralytics")
        fake_ultralytics.YOLO = FakeYOLO
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "best.pt"
            checkpoint.write_bytes(b"fixture")
            path = Path(temporary) / "model"
            with patch.dict(sys.modules, {"ultralytics": fake_ultralytics}):
                mlflow.pyfunc.save_model(
                    path=str(path), python_model=ParkVisionDetector(0.25, 0.7, 640),
                    artifacts={"checkpoint": str(checkpoint)},
                    code_paths=[str(root / "src")],
                    signature=ModelSignature(inputs=Schema([ColSpec("string", "image_path")])),
                    pip_requirements=["mlflow==3.17.2", "ultralytics==8.4.201", "pandas"],
                )
                self.assertTrue((path / "MLmodel").is_file())
                loaded = mlflow.pyfunc.load_model(str(path))
                output = loaded.predict(pd.DataFrame({"image_path": ["fixture.jpg"]}))
                with patch("mlflow.pyfunc.load_model", return_value=loaded) as loader:
                    verify_inference("7", "fixture.jpg", str(path))
                    loader.assert_called_once_with("models:/ParkVision-Detector/7")
            self.assertEqual(int(output.iloc[0]["empty_count"]), 1)
            self.assertEqual(int(output.iloc[0]["occupied_count"]), 1)


if __name__ == "__main__":
    unittest.main()
