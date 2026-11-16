"""Synthetic quality-gate tests; no dataset, weights, or MLflow installation."""

import csv
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "src/analysis/validate_model.py"


class QualityGate(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=ROOT)
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.config = self.root / "quality.yaml"
        self.config.write_text(
            "quality_gate:\n  min_map50_95: 0.9\n  max_mae_empty: 0.35\n  max_mae_occupied: 0.25\n  min_exact_count_accuracy: 0.75\n"
        )
        self.csv = self.root / "comptage.csv"
        self.summary = self.root / "comptage_resume.json"
        self.metrics = self.root / "metadata.json"
        self.output = self.root / "report.json"
        self.rows = [
            {
                "image": str(i),
                "libres_annotees": 1,
                "libres_predites": 1,
                "occupees_annotees": 1,
                "occupees_predites": 1,
            }
            for i in range(4)
        ]
        self.write_rows()
        self.write_summary()
        self.metadata = {
            "format": "parkvision-run-v1",
            "status": "FINISHED",
            "tags": {"manifest_sha256": "a" * 64},
            "artifact_hashes": {"training/weights/best.pt": "b" * 64},
            "metrics": [{"key": "best_val/metrics/mAP50-95_B_", "value": 0.92}],
        }
        self.write_metadata()

    def write_rows(self):
        with self.csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=list(self.rows[0])
                if self.rows
                else [
                    "image",
                    "libres_annotees",
                    "libres_predites",
                    "occupees_annotees",
                    "occupees_predites",
                ],
            )
            writer.writeheader()
            writer.writerows(self.rows)

    def write_summary(self):
        self.summary.write_text(
            json.dumps(
                {
                    "images_evaluees": len(self.rows),
                    "mae_libres": sum(
                        abs(r["libres_predites"] - r["libres_annotees"])
                        for r in self.rows
                    )
                    / len(self.rows),
                    "mae_occupees": sum(
                        abs(r["occupees_predites"] - r["occupees_annotees"])
                        for r in self.rows
                    )
                    / len(self.rows),
                    "poids_sha256": "b" * 64,
                }
            ),
            encoding="utf-8",
        )

    def write_metadata(self):
        self.metrics.write_text(json.dumps(self.metadata), encoding="utf-8")

    def run_gate(self, *extra):
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--config",
                str(self.config),
                "--counting-csv",
                str(self.csv),
                "--counting-summary",
                str(self.summary),
                "--detection-metrics",
                str(self.metrics),
                "--output",
                str(self.output),
                *extra,
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        return result, json.loads(self.output.read_text())

    def test_pass_and_recorded_provenance(self):
        proc, report = self.run_gate()
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["metrics"]["exact_count_accuracy"], 1)
        self.assertFalse(report["provenance"]["checkpoint"]["independently_verified"])

    def test_thresholds_fail_independently(self):
        for key, value, check in [
            ("min_map50_95", "0.95", "map50_95"),
            ("max_mae_empty", "0", "mae_empty"),
            ("max_mae_occupied", "0", "mae_occupied"),
            ("min_exact_count_accuracy", "1", "exact_count_accuracy"),
        ]:
            with self.subTest(key=key):
                if key == "min_map50_95":
                    self.metadata["metrics"][0]["value"] = 0.92
                    self.write_metadata()
                else:
                    self.rows[0]["libres_predites"] = 2
                    self.rows[0]["occupees_predites"] = 2
                    self.write_rows()
                    self.write_summary()
                self.config.write_text(
                    self.config.read_text().replace(
                        {
                            "min_map50_95": "0.9",
                            "max_mae_empty": "0.35",
                            "max_mae_occupied": "0.25",
                            "min_exact_count_accuracy": "0.75",
                        }[key],
                        value,
                    )
                )
                proc, report = self.run_gate()
                self.assertEqual(proc.returncode, 1, proc.stderr)
                self.assertEqual(report["status"], "quality_failed")
                self.assertFalse(report["checks"][check])
                self.config.write_text(
                    "quality_gate:\n  min_map50_95: 0.9\n  max_mae_empty: 0.35\n  max_mae_occupied: 0.25\n  min_exact_count_accuracy: 0.75\n"
                )

    def test_invalid_evidence(self):
        cases = [
            lambda: self.csv.write_text(""),
            lambda: self.csv.write_text("image,libres_annotees\nx,1\n"),
            lambda: self.csv.write_text(
                "image,libres_annotees,libres_predites,occupees_annotees,occupees_predites\nx,-1,0,0,0\n"
            ),
            lambda: self.csv.write_text(
                "image,libres_annotees,libres_predites,occupees_annotees,occupees_predites\nx,NaN,0,0,0\n"
            ),
            lambda: self.csv.write_text(
                "image,libres_annotees,libres_predites,occupees_annotees,occupees_predites\nx,inf,0,0,0\n"
            ),
            lambda: self.csv.write_text(
                "image,libres_annotees,libres_predites,occupees_annotees,occupees_predites\nx,no,0,0,0\n"
            ),
            lambda: self.summary.write_text("{}"),
            lambda: self.metadata["metrics"][0].update(value=float("nan")),
            lambda: self.metrics.unlink(),
        ]
        for change in cases:
            with self.subTest(change=cases.index(change)):
                self.rows = [
                    {
                        "image": str(i),
                        "libres_annotees": 1,
                        "libres_predites": 1,
                        "occupees_annotees": 1,
                        "occupees_predites": 1,
                    }
                    for i in range(4)
                ]
                self.write_rows()
                self.write_summary()
                self.metadata["metrics"][0]["value"] = 0.92
                self.write_metadata()
                change()
                if self.metrics.exists():
                    self.write_metadata()
                proc, report = self.run_gate()
                self.assertEqual(proc.returncode, 2, proc.stderr)
                self.assertEqual(report["status"], "invalid_evidence")
                self.assertIn("Invalid evidence", proc.stderr)

    def test_provenance_mismatch(self):
        for args in [
            ("--expected-model-sha256", "c" * 64),
            ("--expected-dataset-sha256", "c" * 64),
        ]:
            with self.subTest(args=args):
                proc, report = self.run_gate(*args)
                self.assertEqual(proc.returncode, 2)
                self.assertEqual(report["status"], "invalid_evidence")

    def test_summary_disagreement(self):
        data = json.loads(self.summary.read_text())
        data["mae_libres"] = 0.5
        self.summary.write_text(json.dumps(data))
        proc, report = self.run_gate()
        self.assertEqual(proc.returncode, 2)
        self.assertEqual(report["status"], "invalid_evidence")

    def test_image_count_disagreement(self):
        data = json.loads(self.summary.read_text())
        data["images_evaluees"] = 5
        self.summary.write_text(json.dumps(data))
        proc, report = self.run_gate()
        self.assertEqual(proc.returncode, 2)
        self.assertIn("image counts", report["error"])

    def test_actual_file_hash_is_independently_checked(self):
        checkpoint = self.root / "best.pt"
        checkpoint.write_bytes(b"synthetic checkpoint")
        proc, report = self.run_gate("--checkpoint", str(checkpoint))
        self.assertEqual(proc.returncode, 2)
        self.assertIn("checkpoint file SHA256 mismatch", report["error"])

    def write_zip(self, content=None, name="metadata.json"):
        archive = self.root / "export.zip"
        with ZipFile(archive, "w") as stream:
            stream.writestr(
                name, json.dumps(self.metadata) if content is None else content
            )
        return archive

    def test_valid_zip_metadata(self):
        proc, report = self.run_gate("--detection-metrics", str(self.write_zip()))
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(report["status"], "passed")

    def test_corrupt_zip_is_invalid_evidence(self):
        archive = self.root / "broken.zip"
        archive.write_bytes(b"not a zip")
        proc, report = self.run_gate("--detection-metrics", str(archive))
        self.assertEqual(proc.returncode, 2, proc.stderr)
        self.assertEqual(report["status"], "invalid_evidence")

    def test_zip_without_metadata_is_invalid_evidence(self):
        proc, report = self.run_gate(
            "--detection-metrics", str(self.write_zip(name="other.json"))
        )
        self.assertEqual(proc.returncode, 2, proc.stderr)
        self.assertEqual(report["status"], "invalid_evidence")

    def test_oversized_zip_metadata_is_invalid_evidence(self):
        proc, report = self.run_gate(
            "--detection-metrics", str(self.write_zip(" " * (2 * 1024 * 1024 + 1)))
        )
        self.assertEqual(proc.returncode, 2, proc.stderr)
        self.assertIn("size", report["error"].lower())

    def test_zip_nonfinite_metrics_are_invalid(self):
        for value in ("NaN", "Infinity"):
            with self.subTest(value=value):
                content = json.dumps(self.metadata).replace("0.92", value)
                proc, report = self.run_gate(
                    "--detection-metrics", str(self.write_zip(content))
                )
                self.assertEqual(proc.returncode, 2, proc.stderr)
                self.assertEqual(report["status"], "invalid_evidence")

    def test_duplicate_counting_images_are_invalid(self):
        self.rows[1]["image"] = self.rows[0]["image"]
        self.write_rows()
        proc, report = self.run_gate()
        self.assertEqual(proc.returncode, 2)
        self.assertIn("duplicate image", report["error"])

    def test_malformed_metrics_structure_is_invalid(self):
        for malformed in (None, [None], [{"key": "other"}]):
            with self.subTest(malformed=malformed):
                self.metadata["metrics"] = malformed
                self.write_metadata()
                proc, report = self.run_gate()
                self.assertEqual(proc.returncode, 2, proc.stderr)
                self.assertEqual(report["status"], "invalid_evidence")

    def test_contradictory_summary_total_is_invalid(self):
        summary = json.loads(self.summary.read_text())
        summary["total_libres_annotees"] = 999
        self.summary.write_text(json.dumps(summary))
        proc, report = self.run_gate()
        self.assertEqual(proc.returncode, 2)
        self.assertIn("total_libres_annotees", report["error"])

    def test_matching_actual_hashes_are_verified(self):
        checkpoint = self.root / "best.pt"
        manifest = self.root / "split_manifest.csv"
        checkpoint.write_bytes(b"synthetic checkpoint")
        manifest.write_bytes(b"synthetic manifest")
        model_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        dataset_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
        summary = json.loads(self.summary.read_text())
        summary["poids_sha256"] = model_hash
        self.summary.write_text(json.dumps(summary))
        self.metadata["artifact_hashes"]["training/weights/best.pt"] = model_hash
        self.metadata["tags"]["manifest_sha256"] = dataset_hash
        self.write_metadata()
        proc, report = self.run_gate(
            "--checkpoint", str(checkpoint), "--dataset-manifest", str(manifest)
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertTrue(report["provenance"]["checkpoint"]["independently_verified"])
        self.assertTrue(
            report["provenance"]["dataset_manifest"]["independently_verified"]
        )


if __name__ == "__main__":
    unittest.main()
