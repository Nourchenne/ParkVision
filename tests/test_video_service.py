"""Deterministic tests for video configuration and frame processing."""

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pandas as pd
import pytest

from src.api.config import get_settings
from src.api.model_service import ModelInfo, ModelService
from src.api.schemas import Counts, ModelSummary, Prediction
from src.api.video_service import (
    VideoProcessingError,
    analyze_video,
    annotate_frame,
    validate_mp4,
)

VIDEO_ENV = (
    "PARKVISION_VIDEO_SAMPLE_FPS", "PARKVISION_MAX_VIDEO_BYTES",
    "PARKVISION_MAX_VIDEO_DURATION_SECONDS", "PARKVISION_MAX_VIDEO_FRAMES",
    "PARKVISION_MAX_ACTIVE_VIDEO_SESSIONS", "PARKVISION_VIDEO_SESSION_TTL_SECONDS",
    "PARKVISION_VIDEO_MAX_WIDTH", "PARKVISION_VIDEO_MAX_HEIGHT",
    "PARKVISION_YOUTUBE_ENABLED", "PARKVISION_YOUTUBE_SOCKET_TIMEOUT_SECONDS",
    "PARKVISION_VIDEO_PROCESSING_TIMEOUT_SECONDS",
    "PARKVISION_VIDEO_STORAGE_BUDGET_BYTES", "PARKVISION_VIDEO_ORPHAN_GRACE_SECONDS",
    "PARKVISION_LOCALHOST_ONLY", "PARKVISION_FFMPEG_LOCATION",
)


def clear_video_environment(monkeypatch):
    for name in VIDEO_ENV:
        monkeypatch.delenv(name, raising=False)


def test_video_settings_defaults(monkeypatch):
    clear_video_environment(monkeypatch)
    value = get_settings()
    assert value.video_sample_fps == 1.0
    assert value.max_video_bytes == 104_857_600
    assert value.max_video_duration_seconds == 420
    assert value.max_video_frames == 450
    assert value.video_storage_budget_bytes == 209_715_200
    assert value.video_processing_timeout_seconds == 600
    assert value.localhost_only is True


@pytest.mark.parametrize(("name", "value"), [
    ("PARKVISION_VIDEO_SAMPLE_FPS", "0"),
    ("PARKVISION_VIDEO_SAMPLE_FPS", "nan"),
    ("PARKVISION_VIDEO_SAMPLE_FPS", "6"),
    ("PARKVISION_MAX_VIDEO_BYTES", "-1"),
    ("PARKVISION_MAX_ACTIVE_VIDEO_SESSIONS", "5"),
    ("PARKVISION_YOUTUBE_ENABLED", "yes"),
    ("PARKVISION_LOCALHOST_ONLY", "1"),
])
def test_invalid_video_setting_rejected(monkeypatch, name, value):
    clear_video_environment(monkeypatch)
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError):
        get_settings()


def test_storage_budget_cannot_be_smaller_than_one_video(monkeypatch):
    clear_video_environment(monkeypatch)
    monkeypatch.setenv("PARKVISION_MAX_VIDEO_BYTES", "100")
    monkeypatch.setenv("PARKVISION_VIDEO_STORAGE_BUDGET_BYTES", "99")
    with pytest.raises(ValueError, match="storage budget"):
        get_settings()


def test_processing_timeout_must_exceed_socket_timeout(monkeypatch):
    clear_video_environment(monkeypatch)
    monkeypatch.setenv("PARKVISION_YOUTUBE_SOCKET_TIMEOUT_SECONDS", "20")
    monkeypatch.setenv("PARKVISION_VIDEO_PROCESSING_TIMEOUT_SECONDS", "20")
    with pytest.raises(ValueError, match="processing timeout"):
        get_settings()


def test_model_service_serializes_predictions(tmp_path):
    from PIL import Image

    class BlockingModel:
        def __init__(self):
            self.active = 0
            self.maximum = 0
            self.lock = threading.Lock()

        def predict(self, frame):
            with self.lock:
                self.active += 1
                self.maximum = max(self.maximum, self.active)
            time.sleep(0.03)
            with self.lock:
                self.active -= 1
            path = frame.iloc[0]["image_path"]
            return pd.DataFrame([{
                "image_path": path, "empty_count": 0, "occupied_count": 0,
                "detections_json": json.dumps([]),
            }])

    model = BlockingModel()
    info = ModelInfo("ParkVision-Detector", "2", "yolov8s", None,
                     "models:/ParkVision-Detector/2", "champion")
    service = ModelService(model, info)
    images = [tmp_path / f"{index}.jpg" for index in range(2)]
    for path in images:
        Image.new("RGB", (16, 16)).save(path)
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(service.predict, images))
    assert model.maximum == 1
    assert [result.counts.total for result in results] == [0, 0]


class FakeCapture:
    def __init__(self, timestamps, fps=10.0):
        self.timestamps = timestamps
        self.fps = fps
        self.index = 0
        self.released = False

    def isOpened(self):
        return True

    def read(self):
        if self.index >= len(self.timestamps):
            return False, None
        self.index += 1
        return True, np.zeros((24, 32, 3), dtype=np.uint8)

    def get(self, prop):
        if prop == cv2.CAP_PROP_FPS:
            return self.fps
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return len(self.timestamps)
        if prop == cv2.CAP_PROP_POS_MSEC:
            return self.timestamps[max(0, self.index - 1)]
        return 0

    def release(self):
        self.released = True


class FakeVideoModel:
    def predict(self, path):
        return Prediction(
            model=ModelSummary(name="ParkVision-Detector", version="2", architecture="yolov8s"),
            counts=Counts(empty=1, occupied=0, total=1),
            detections=[{"class_id": 0, "class_name": "space-empty", "confidence": 0.9,
                         "bbox": [1.0, 2.0, 10.0, 12.0]}],
            inference_time_ms=4.5,
        )


def video_file(tmp_path):
    path = tmp_path / "sample.mp4"
    path.write_bytes(b"\x00\x00\x00\x18ftypisom" + b"0" * 64)
    return path


def video_settings(tmp_path, **changes):
    defaults = {"temp_dir": tmp_path, "video_max_width": 64, "video_max_height": 64,
                "max_video_duration_seconds": 30, "max_video_frames": 10,
                "video_sample_fps": 1.0}
    defaults.update(changes)
    return SimpleNamespace(**defaults, max_video_bytes=1024)


def test_six_minute_mp4_is_within_new_default_duration(tmp_path, monkeypatch):
    clear_video_environment(monkeypatch)
    settings = get_settings()
    capture = FakeCapture([0] * 361, fps=1.0)
    metadata = validate_mp4(video_file(tmp_path), settings, capture_factory=lambda _: capture)
    assert metadata.duration_seconds == 361.0
    assert capture.released


def test_samples_by_source_timestamp_and_includes_last_interval(tmp_path):
    timestamps = [0, 300, 1000, 1500, 2000, 2990]
    captures = []

    def factory(_):
        capture = FakeCapture(timestamps)
        captures.append(capture)
        return capture

    published = []
    analyze_video(video_file(tmp_path), FakeVideoModel(), video_settings(tmp_path),
                  threading.Event(), time.monotonic() + 10,
                  lambda result, preview: published.append((result, preview)),
                  capture_factory=factory)
    assert [item[0].timestamp_seconds for item in published] == [0.0, 1.0, 2.0]
    assert all(item[0].inference_time_ms == 4.5 for item in published)
    assert all(item[0].analysis_time_ms >= 0 for item in published)
    assert all(item[1].startswith(b"\xff\xd8") for item in published)
    assert all(capture.released for capture in captures)


def test_non_monotonic_timestamps_fall_back_to_fps(tmp_path):
    timestamps = [0, 0, 0]
    factory = lambda _: FakeCapture(timestamps, fps=2.0)
    published = []
    analyze_video(video_file(tmp_path), FakeVideoModel(),
                  video_settings(tmp_path, video_sample_fps=2.0), threading.Event(),
                  time.monotonic() + 10, lambda result, _: published.append(result),
                  capture_factory=factory)
    assert [item.timestamp_seconds for item in published] == [0.0, 0.5, 1.0]


def test_missing_timestamps_and_fps_fail_closed(tmp_path):
    factory = lambda _: FakeCapture([0, 0], fps=0.0)
    with pytest.raises(VideoProcessingError, match="invalid_timestamps"):
        analyze_video(video_file(tmp_path), FakeVideoModel(), video_settings(tmp_path),
                      threading.Event(), time.monotonic() + 10, lambda *_: None,
                      capture_factory=factory)


def test_processing_deadline_stops_before_decode(tmp_path):
    with pytest.raises(VideoProcessingError, match="processing_timeout"):
        analyze_video(video_file(tmp_path), FakeVideoModel(), video_settings(tmp_path),
                      threading.Event(), 5.0, lambda *_: None,
                      capture_factory=lambda _: FakeCapture([0]), clock=lambda: 5.0)


def test_annotation_uses_green_for_empty_space():
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    prediction = FakeVideoModel().predict(Path("unused"))
    prediction.detections[0].bbox = [5.0, 35.0, 30.0, 55.0]
    annotated = annotate_frame(frame, prediction, 1.0)
    assert annotated[35, 5].tolist() == [0, 180, 0]
