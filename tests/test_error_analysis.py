"""Tests de la logique d'analyse des erreurs (sans GPU, sans modele, sans PKLot)."""
import importlib.util
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "parkvision_analysis", ROOT / "src/analysis/analyze_errors.py")
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)

A = (0, 0, 10, 10)
B = (100, 100, 110, 110)


def kinds(events):
    return sorted(event["type"] for event in events)


class Matching(unittest.TestCase):
    def test_perfect_match_has_no_event(self):
        self.assertEqual(analysis.match_detections(
            [(0, A), (1, B)], [(0, A, 0.9), (1, B, 0.8)]), [])

    def test_missed_place(self):
        self.assertEqual(kinds(analysis.match_detections([(0, A), (1, B)], [(0, A, 0.9)])),
                         ["oubli"])

    def test_false_detection(self):
        self.assertEqual(kinds(analysis.match_detections([(0, A)], [(0, A, 0.9), (1, B, 0.7)])),
                         ["fausse_detection"])

    def test_double_detection(self):
        events = analysis.match_detections([(0, A)], [(0, A, 0.9), (0, (1, 1, 11, 11), 0.6)])
        self.assertEqual(kinds(events), ["double_detection"])

    def test_class_confusion_counts_once(self):
        events = analysis.match_detections([(0, A)], [(1, A, 0.9)])
        self.assertEqual(kinds(events), ["confusion"])
        self.assertEqual((events[0]["classe_verite"], events[0]["classe_predite"]), (0, 1))

    def test_low_overlap_is_not_matched(self):
        events = analysis.match_detections([(0, A)], [(0, (8, 8, 18, 18), 0.9)])
        self.assertEqual(kinds(events), ["fausse_detection", "oubli"])

    def test_iou_values(self):
        self.assertAlmostEqual(analysis.iou(A, A), 1.0)
        self.assertEqual(analysis.iou(A, B), 0.0)
        self.assertAlmostEqual(analysis.iou(A, (5, 0, 15, 10)), 50 / 150)


class Counting(unittest.TestCase):
    def test_mae_is_separate_per_class(self):
        rows = [
            analysis.count_row("a", [(0, A), (0, B), (1, A)], [(0, A, 1), (1, A, 1)]),
            analysis.count_row("b", [(1, A)], [(1, A, 1), (1, B, 1), (1, B, 1)]),
        ]
        summary = analysis.summarize_counts(rows)
        self.assertEqual(summary["images_evaluees"], 2)
        self.assertAlmostEqual(summary["mae_libres"], (1 + 0) / 2)
        self.assertAlmostEqual(summary["mae_occupees"], (0 + 2) / 2)

    def test_empty_input_rejected(self):
        with self.assertRaises(ValueError):
            analysis.summarize_counts([])


class Labels(unittest.TestCase):
    def test_read_labels_converts_to_pixels(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "a.txt"
            path.write_text("1 0.5 0.5 0.2 0.4\n\n", encoding="utf-8")
            (cls, box), = analysis.read_labels(path, 100, 200)
            self.assertEqual(cls, 1)
            self.assertEqual(box, (40.0, 60.0, 60.0, 140.0))

    def test_invalid_label_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "a.txt"
            path.write_text("3 0.5 0.5 0.2 0.4\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                analysis.read_labels(path, 100, 100)

    def test_examples_cover_error_types(self):
        events = {"x": [{"type": "oubli"}] * 3, "y": [{"type": "double_detection"}],
                  "z": []}
        chosen = analysis.choose_examples(events, {"x": 3, "y": 1, "z": 0})
        self.assertIn(("oubli", "x"), chosen)
        self.assertIn(("double_detection", "y"), chosen)
        self.assertNotIn("z", [name for _, name in chosen])


if __name__ == "__main__":
    unittest.main()
