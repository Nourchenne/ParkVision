"""Deterministic tests for ONNX export and latency benchmarking; no real YOLO/ONNX Runtime needed."""

import json
import time
from pathlib import Path

from PIL import Image

from src.models.export_onnx import (
    benchmark_onnxruntime,
    benchmark_pytorch,
    export_to_onnx,
    main,
)


def make_image(tmp_path, size=(32, 24)):
    path = tmp_path / "sample.jpg"
    Image.new("RGB", size).save(path, format="JPEG")
    return path


class FakeYOLO:
    def __init__(self, checkpoint):
        self.checkpoint = checkpoint
        self.predict_calls = 0

    def export(self, **kwargs):
        destination = Path(self.checkpoint).with_suffix(".onnx")
        destination.write_bytes(b"fake-onnx-bytes")
        return str(destination)

    def predict(self, source, verbose=False):
        self.predict_calls += 1
        time.sleep(0.001)


def test_export_to_onnx_moves_result_to_requested_output(tmp_path):
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"fake-checkpoint")
    output = tmp_path / "exports" / "parkvision.onnx"
    factory_calls = []

    def factory(path):
        factory_calls.append(path)
        return FakeYOLO(path)

    result = export_to_onnx(checkpoint, output=output, yolo_factory=factory)
    assert result == output
    assert output.is_file()
    assert factory_calls == [str(checkpoint)]


def test_export_to_onnx_keeps_default_location_when_no_output_given(tmp_path):
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"fake-checkpoint")
    result = export_to_onnx(checkpoint, yolo_factory=FakeYOLO)
    assert result == checkpoint.with_suffix(".onnx")
    assert result.is_file()


def test_benchmark_pytorch_warms_up_then_times_requested_runs(tmp_path):
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"fake-checkpoint")
    image = make_image(tmp_path)
    instances = []

    def factory(path):
        instance = FakeYOLO(path)
        instances.append(instance)
        return instance

    latency = benchmark_pytorch(checkpoint, image, runs=5, yolo_factory=factory)
    assert latency >= 0
    assert instances[0].predict_calls == 6  # 1 warm-up + 5 timed runs


class FakeInput:
    name = "images"


class FakeSession:
    def __init__(self, path):
        self.path = path
        self.run_calls = 0

    def get_inputs(self):
        return [FakeInput()]

    def run(self, output_names, feeds):
        self.run_calls += 1
        assert list(feeds) == ["images"]
        tensor = feeds["images"]
        assert tensor.shape == (1, 3, 640, 640)
        return [tensor]


def test_benchmark_onnxruntime_warms_up_then_times_requested_runs(tmp_path):
    onnx_path = tmp_path / "parkvision.onnx"
    onnx_path.write_bytes(b"fake-onnx-bytes")
    image = make_image(tmp_path)
    sessions = []

    def factory(path):
        session = FakeSession(path)
        sessions.append(session)
        return session

    latency = benchmark_onnxruntime(onnx_path, image, runs=4, session_factory=factory)
    assert latency >= 0
    assert sessions[0].run_calls == 5  # 1 warm-up + 4 timed runs


def test_main_reports_export_path_and_both_latencies(tmp_path, capsys, monkeypatch):
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"fake-checkpoint")
    image = make_image(tmp_path)

    monkeypatch.setattr("src.models.export_onnx.export_to_onnx",
                        lambda ckpt, output=None, yolo_factory=None: tmp_path / "best.onnx")
    monkeypatch.setattr("src.models.export_onnx.benchmark_pytorch", lambda *a, **k: 10.0)
    monkeypatch.setattr("src.models.export_onnx.benchmark_onnxruntime", lambda *a, **k: 2.5)

    report = main(["--checkpoint", str(checkpoint), "--image", str(image), "--runs", "3"])
    assert report == {
        "onnx_path": str(tmp_path / "best.onnx"), "runs": 3,
        "pytorch_inference_ms": 10.0, "onnxruntime_inference_ms": 2.5, "speedup": 4.0,
    }
    assert json.loads(capsys.readouterr().out) == report