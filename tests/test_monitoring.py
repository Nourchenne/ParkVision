"""Monitoring : journal JSONL borné et opt-in, intégration API/vidéo, rapport et règles d'alerte."""

import io
import json
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from src.analysis import monitor_report
from src.api.config import Settings, get_settings
from src.api.main import create_app
from src.api.model_service import ModelInfo, ModelService
from src.api.monitoring import ROUTE_PREDICT, ROUTE_VIDEO, PredictionMonitor
from src.api.schemas import Counts, Detection, ModelSummary, Prediction
from src.api.video_service import analyze_video


def prediction(confidences=(0.9, 0.7), empty=1, inference_ms=12.5):
    detections = [Detection(class_id=0, class_name="space-empty", confidence=c, bbox=[1, 2, 10, 12])
                  for c in confidences]
    return Prediction(model=ModelSummary(name="ParkVision-Detector", version="7", architecture="yolov8s"),
                      counts=Counts(empty=empty, occupied=len(detections) - empty, total=len(detections)),
                      detections=detections, inference_time_ms=inference_ms)


def lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


# --- journal ---------------------------------------------------------------------------------

def test_record_writes_one_json_line_without_image_data(tmp_path):
    log = tmp_path / "logs" / "predictions.jsonl"
    PredictionMonitor(log, 10_000).record(ROUTE_PREDICT, prediction())
    (entry,) = lines(log)
    assert set(entry) == {"ts", "route", "model_version", "error", "empty", "occupied", "total",
                          "conf_mean", "conf_min", "inference_ms"}
    assert entry["route"] == "/predict" and entry["model_version"] == "7" and entry["error"] is False
    assert (entry["empty"], entry["occupied"], entry["total"]) == (1, 1, 2)
    assert entry["conf_mean"] == 0.8 and entry["conf_min"] == 0.7 and entry["inference_ms"] == 12.5


def test_zero_detections_store_null_confidence(tmp_path):
    log = tmp_path / "p.jsonl"
    PredictionMonitor(log, 10_000).record(ROUTE_VIDEO, prediction(confidences=(), empty=0))
    (entry,) = lines(log)
    assert entry["total"] == 0 and entry["conf_mean"] is None and entry["conf_min"] is None


def test_record_error_marks_the_failure(tmp_path):
    log = tmp_path / "p.jsonl"
    PredictionMonitor(log, 10_000).record_error(ROUTE_PREDICT, "7")
    (entry,) = lines(log)
    assert entry["error"] is True and entry["total"] is None and entry["model_version"] == "7"


def test_log_is_bounded_by_rotation(tmp_path):
    log = tmp_path / "p.jsonl"
    monitor = PredictionMonitor(log, 1024)
    for _ in range(200):
        monitor.record(ROUTE_PREDICT, prediction())
    rotated = log.with_name("p.jsonl.1")
    assert rotated.exists()
    assert log.stat().st_size <= 1024 and rotated.stat().st_size <= 1024
    assert len(lines(log)) > 0
    assert sorted(p.name for p in tmp_path.iterdir()) == ["p.jsonl", "p.jsonl.1"]


def test_write_failure_never_raises(tmp_path):
    blocked = tmp_path / "blocked"
    blocked.mkdir()  # un dossier à la place du fichier : l'écriture échoue
    PredictionMonitor(blocked, 10_000).record(ROUTE_PREDICT, prediction())


# --- configuration -----------------------------------------------------------------------------

def test_monitoring_is_off_by_default(monkeypatch):
    for name in ("PARKVISION_MONITORING_ENABLED", "PARKVISION_MONITORING_LOG", "PARKVISION_MONITORING_MAX_BYTES"):
        monkeypatch.delenv(name, raising=False)
    value = get_settings()
    assert value.monitoring_enabled is False
    assert value.monitoring_max_bytes == 5 * 1024 * 1024
    assert value.monitoring_log is not None and value.monitoring_log.name == "predictions.jsonl"
    assert PredictionMonitor.from_settings(value) is None


def test_monitoring_environment_overrides(monkeypatch, tmp_path):
    monkeypatch.setenv("PARKVISION_MONITORING_ENABLED", "true")
    monkeypatch.setenv("PARKVISION_MONITORING_LOG", str(tmp_path / "run.jsonl"))
    monkeypatch.setenv("PARKVISION_MONITORING_MAX_BYTES", "2048")
    value = get_settings()
    monitor = PredictionMonitor.from_settings(value)
    assert monitor is not None and monitor.path == (tmp_path / "run.jsonl").resolve() and monitor.max_bytes == 2048


@pytest.mark.parametrize(("name", "value"), [
    ("PARKVISION_MONITORING_ENABLED", "yes"),
    ("PARKVISION_MONITORING_MAX_BYTES", "100"),
    ("PARKVISION_MONITORING_MAX_BYTES", "abc"),
])
def test_invalid_monitoring_settings_are_rejected(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match="PARKVISION_MONITORING"):
        get_settings()


# --- API et vidéo --------------------------------------------------------------------------------

class FakeModel:
    def __init__(self, fail=False):
        self.fail = fail

    def predict(self, frame):
        if self.fail:
            raise RuntimeError("private backend details")
        path = frame.iloc[0]["image_path"]
        detections = [{"class_id": 0, "confidence": 0.94, "xyxy": [1.0, 2.0, 11.0, 12.0]},
                      {"class_id": 1, "confidence": 0.8, "xyxy": [3.0, 4.0, 13.0, 14.0]}]
        return pd.DataFrame([{"image_path": str(path), "empty_count": 1, "occupied_count": 1,
                              "detections_json": json.dumps(detections)}])


def api_client(tmp_path, enabled, fail=False):
    log = tmp_path / "monitor.jsonl"
    settings = Settings("https://example.invalid/mlflow", "ParkVision-Detector", "champion",
                        10 * 1024 * 1024, tmp_path, "WARNING",
                        monitoring_enabled=enabled, monitoring_log=log)
    service = ModelService(FakeModel(fail), ModelInfo("ParkVision-Detector", "7", "yolov8s", None,
                                                      "models:/ParkVision-Detector/7", "champion"))
    return TestClient(create_app(settings, lambda _: service)), log


def jpeg():
    stream = io.BytesIO()
    Image.new("RGB", (32, 24)).save(stream, format="JPEG")
    return stream.getvalue()


def post(client, route):
    return client.post(route, files={"file": ("lot.jpg", jpeg(), "image/jpeg")})


def test_disabled_monitoring_writes_nothing_and_changes_nothing(tmp_path):
    client, log = api_client(tmp_path, enabled=False)
    with client:
        response = post(client, "/predict")
    assert response.status_code == 200 and response.json()["counts"] == {"empty": 1, "occupied": 1, "total": 2}
    assert not log.exists()


def test_enabled_monitoring_records_each_route(tmp_path):
    client, log = api_client(tmp_path, enabled=True)
    with client:
        assert post(client, "/predict").status_code == 200
        assert post(client, "/predict/visualize").status_code == 200
    first, second = lines(log)
    assert (first["route"], second["route"]) == ("/predict", "/predict/visualize")
    assert first["total"] == 2 and first["conf_mean"] == 0.87 and first["conf_min"] == 0.8


def test_failed_inference_is_recorded_as_an_error(tmp_path):
    client, log = api_client(tmp_path, enabled=True, fail=True)
    with client:
        assert post(client, "/predict").status_code == 500
    (entry,) = lines(log)
    assert entry["error"] is True and entry["route"] == "/predict" and entry["model_version"] == "7"


class FakeCapture:
    def __init__(self, count):
        self.count = count
        self.index = 0

    def isOpened(self):
        return True

    def read(self):
        if self.index >= self.count:
            return False, None
        self.index += 1
        return True, np.zeros((24, 32, 3), dtype=np.uint8)

    def get(self, prop):
        return {cv2.CAP_PROP_FPS: 1.0, cv2.CAP_PROP_FRAME_COUNT: self.count,
                cv2.CAP_PROP_POS_MSEC: (self.index - 1) * 1000.0}.get(prop, 0)

    def release(self):
        pass


class FakeVideoService:
    def __init__(self, monitor):
        self.monitor = monitor
        self.info = SimpleNamespace(version="7")

    def predict(self, _path):
        return prediction()


def video_settings(tmp_path):
    return SimpleNamespace(temp_dir=tmp_path, video_max_width=64, video_max_height=64,
                           max_video_duration_seconds=30, max_video_frames=10,
                           video_sample_fps=1.0, max_video_bytes=1024)


def test_video_frames_are_recorded_with_the_video_route(tmp_path):
    video = tmp_path / "v.mp4"
    video.write_bytes(b"\x00\x00\x00\x18ftypisom" + b"0" * 64)
    log = tmp_path / "video.jsonl"
    analyze_video(video, FakeVideoService(PredictionMonitor(log, 10_000)), video_settings(tmp_path),
                  threading.Event(), time.monotonic() + 10, lambda *_: None,
                  capture_factory=lambda _: FakeCapture(3))
    assert [entry["route"] for entry in lines(log)] == ["video_frame"] * 3


def test_video_without_monitor_is_unchanged(tmp_path):
    video = tmp_path / "v.mp4"
    video.write_bytes(b"\x00\x00\x00\x18ftypisom" + b"0" * 64)
    published = []
    analyze_video(video, FakeVideoService(None), video_settings(tmp_path),
                  threading.Event(), time.monotonic() + 10, lambda result, _: published.append(result),
                  capture_factory=lambda _: FakeCapture(2))
    assert len(published) == 2


# --- agrégation et règles d'alerte -----------------------------------------------------------------

def entry(total=40, conf=0.9, ms=20.0, error=False):
    if error:
        return {"error": True, "total": None, "conf_mean": None, "conf_min": None, "inference_ms": None}
    return {"error": False, "total": total, "conf_mean": conf if total else None,
            "conf_min": conf - 0.1 if total else None, "inference_ms": ms}


IN_DOMAIN = [entry(total=40, conf=0.9) for _ in range(30)]
OUT_OF_DOMAIN = [entry(total=0) for _ in range(30)]
LOW_CONFIDENCE = [entry(total=12, conf=0.4) for _ in range(30)]


def test_summary_counts_errors_latency_and_empty_share():
    records = [entry(ms=10.0), entry(ms=20.0), entry(total=0), entry(error=True)]
    s = monitor_report.summarize(records)
    assert s["requetes"] == 4 and s["erreurs"] == 1 and s["taux_erreur"] == 0.25
    assert s["latence_mediane_ms"] == 20.0 and s["confiance_moyenne"] == 0.9
    assert s["part_sans_detection"] == pytest.approx(1 / 3)


def test_percentile_interpolates():
    assert monitor_report.percentile([10, 20, 30, 40], 0.5) == 25
    assert monitor_report.percentile([5], 0.95) == 5
    assert monitor_report.percentile(list(range(101)), 0.95) == 95


def test_in_domain_run_raises_no_alert():
    assert monitor_report.rolling_alerts(IN_DOMAIN) == []


def test_out_of_domain_run_fires_the_empty_results_rule():
    alerts = monitor_report.rolling_alerts(OUT_OF_DOMAIN)
    assert alerts[0] == (20, [monitor_report.RULE_EMPTY])


def test_low_confidence_run_fires_the_confidence_rule():
    alerts = monitor_report.rolling_alerts(LOW_CONFIDENCE)
    assert alerts[0] == (20, [monitor_report.RULE_CONFIDENCE])


def test_alert_needs_a_full_window():
    assert monitor_report.rolling_alerts(OUT_OF_DOMAIN[:19]) == []
    text, alert = monitor_report.format_report("court", OUT_OF_DOMAIN[:19], 20)
    assert alert is False and "non évaluables" in text


def test_errors_are_ignored_by_the_rules_but_counted_in_the_rate():
    records = IN_DOMAIN + [entry(error=True)] * 10
    assert monitor_report.rolling_alerts(records) == []
    assert monitor_report.summarize(records)["taux_erreur"] == 0.25


def test_load_reads_rotation_then_log_and_skips_bad_lines(tmp_path):
    log = tmp_path / "p.jsonl"
    log.with_name("p.jsonl.1").write_text('{"total": 1}\n', encoding="utf-8")
    log.write_text('{"total": 2}\nnot json\n[1]\n{"total": 3}\n', encoding="utf-8")
    assert [r["total"] for r in monitor_report.load(log)] == [1, 2, 3]


def write_log(path, records):
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")


def run_cli(monkeypatch, capsys, *args):
    monkeypatch.setattr(sys, "argv", ["monitor_report.py", *map(str, args)])
    with pytest.raises(SystemExit) as stop:
        monitor_report.main()
    return stop.value.code, capsys.readouterr().out


def test_cli_exit_code_distinguishes_the_two_runs(tmp_path, monkeypatch, capsys):
    good, bad = tmp_path / "demo.jsonl", tmp_path / "ood.jsonl"
    write_log(good, IN_DOMAIN)
    write_log(bad, OUT_OF_DOMAIN)
    code, out = run_cli(monkeypatch, capsys, good)
    assert code == 0 and "aucune alerte" in out
    code, out = run_cli(monkeypatch, capsys, bad)
    assert code == 2 and "ALERTE (resultats_vides)" in out


def test_cli_rejects_missing_log(tmp_path, monkeypatch, capsys):
    with pytest.raises(SystemExit, match="journal absent ou vide"):
        monkeypatch.setattr(sys, "argv", ["monitor_report.py", str(tmp_path / "none.jsonl")])
        monitor_report.main()


def test_plot_writes_a_png(tmp_path):
    pytest.importorskip("matplotlib")
    out = tmp_path / "chart.png"
    monitor_report.plot([("démo", IN_DOMAIN), ("hors domaine", OUT_OF_DOMAIN)], out, 20)
    assert out.stat().st_size > 5000 and Path(out).read_bytes()[:4] == b"\x89PNG"
