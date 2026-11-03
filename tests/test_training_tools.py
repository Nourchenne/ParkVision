"""Tests logiciels sans GPU, sans telechargement de modele et sans PKLot."""
import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


train = load_module("parkvision_train", ROOT / "src/models/train.py")
transfer = load_module("parkvision_transfer", ROOT / "src/utils/transfer_run.py")


class DataChecks(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.previous_root = train.ROOT
        train.ROOT = self.root
        self.processed = self.root / "data/processed/yolo"
        self.processed.mkdir(parents=True)
        rows = []
        for index, split in enumerate(("train", "valid", "test"), start=1):
            image = self.root / "data/raw" / split / "images" / f"image_{index}.jpg"
            image.parent.mkdir(parents=True)
            image.write_bytes(b"placeholder-image-not-used-for-training")
            label = image.parent.parent / "labels" / f"image_{index}.txt"
            label.parent.mkdir()
            label.write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")
            rows.append([image.relative_to(self.root).as_posix(), f"2012-01-0{index}", split])
            (self.processed / f"{split}.txt").write_text(image.as_posix() + "\n", encoding="utf-8")
        self.rows = rows
        self.write_manifest()
        (self.processed / "data.yaml").write_text(yaml.safe_dump({
            "train": "train.txt", "val": "valid.txt", "test": "test.txt",
            "nc": 2, "names": ["space-empty", "space-occupied"],
        }), encoding="utf-8")
        self.config = self.root / "config.yaml"
        self.config.write_text(yaml.safe_dump({
            "model": "yolov8n.pt", "data": "data/processed/yolo/data.yaml",
            "epochs": 50, "batch": 16, "imgsz": 640, "seed": 42, "name": "A",
        }), encoding="utf-8")

    def write_manifest(self):
        with (self.root / "data/processed/split_manifest.csv").open(
            "w", encoding="utf-8", newline=""
        ) as stream:
            writer = csv.writer(stream)
            writer.writerow(["image", "date", "split"])
            writer.writerows(self.rows)

    def tearDown(self):
        train.ROOT = self.previous_root
        self.temporary.cleanup()

    def test_runtime_copy_preserves_raw_and_excludes_test(self):
        config, groups, _, names = train.read_inputs(self.config)
        path, selection = train.prepare_runtime_data(
            groups, names, self.root / "runtime", True, config["seed"])
        self.assertEqual(set(selection), {"train", "valid"})
        self.assertNotIn("test", yaml.safe_load(path.read_text()))
        copy = path.parent / "train/images/image_1.jpg"
        copy.write_bytes(b"modified copy")
        self.assertNotEqual(copy.read_bytes(), groups["train"][0].read_bytes())

    def test_day_leakage_rejected(self):
        self.rows[1][1] = self.rows[0][1]
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "Journee partagee"):
            train.read_inputs(self.config)

    def test_manifest_list_disagreement_rejected(self):
        (self.processed / "train.txt").write_text("different.jpg\n")
        with self.assertRaisesRegex(ValueError, "Liste YOLO differente"):
            train.read_inputs(self.config)


class RunTransfer(unittest.TestCase):
    def test_roundtrip_and_corruption_detection(self):
        from mlflow import MlflowClient
        from mlflow.entities import Metric, Param

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_uri = f"sqlite:///{(root / 'source.db').as_posix()}"
            destination_uri = f"sqlite:///{(root / 'destination.db').as_posix()}"
            source = MlflowClient(tracking_uri=source_uri)
            destination = MlflowClient(tracking_uri=destination_uri)
            experiment = source.create_experiment("TransferTest", (root / "source_artifacts").as_uri())
            destination.create_experiment("TransferTest", (root / "dest_artifacts").as_uri())
            run_id = source.create_run(experiment, start_time=1000, tags={"mlflow.runName": "check"}).info.run_id
            source.log_batch(run_id, params=[Param("batch", "16")], metrics=[
                Metric("val/map", 0.2, 2000, 1), Metric("val/map", 0.4, 3000, 2),
            ])
            sample = root / "sample.txt"
            sample.write_text("artifact content", encoding="utf-8")
            source.log_artifact(run_id, str(sample), "nested")
            source.set_terminated(run_id, "FINISHED", 4000)
            archive = transfer.export_run(source_uri, run_id, root / "run.zip")
            imported = transfer.import_run(archive, destination_uri)
            restored = destination.get_run(imported)
            self.assertEqual(restored.data.params, {"batch": "16"})
            self.assertEqual((restored.info.start_time, restored.info.end_time), (1000, 4000))
            self.assertEqual(restored.info.status, "FINISHED")
            self.assertEqual(
                [(m.step, m.value, m.timestamp) for m in destination.get_metric_history(imported, "val/map")],
                [(1, 0.2, 2000), (2, 0.4, 3000)],
            )
            downloaded = Path(destination.download_artifacts(imported, "nested/sample.txt"))
            self.assertEqual(downloaded.read_text(), "artifact content")
            self.assertEqual(transfer.import_run(archive, destination_uri), imported)
            self.assertEqual(len(destination.search_runs([restored.info.experiment_id])), 1)
            tampered = root / "tampered.zip"
            with ZipFile(archive) as original, ZipFile(tampered, "w", ZIP_DEFLATED) as edited:
                for entry in original.namelist():
                    content = original.read(entry)
                    if entry == "artifacts/nested/sample.txt":
                        content = b"tampered content"
                    edited.writestr(entry, content)
            with self.assertRaisesRegex(ValueError, "empreintes"):
                transfer.import_run(tampered, destination_uri)
            # Liberer les fichiers SQLite avant suppression du dossier sous Windows.
            source._tracking_client.store.engine.dispose()
            destination._tracking_client.store.engine.dispose()


if __name__ == "__main__":
    unittest.main()
