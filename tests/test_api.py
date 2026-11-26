"""Serving tests use a local fake PyFunc; no network, checkpoint, or GPU."""

import io
import json
import shutil
import uuid
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from src.api.config import Settings
from src.api.main import create_app
from src.api.model_service import (
    ModelInfo,
    ModelService,
    load_champion,
    parse_prediction,
)


@pytest.fixture
def tmp_path():
    """Avoid pytest's inaccessible mode-0700 basetemp on this Windows runner."""
    root = Path(__file__).resolve().parents[1] / "phase3_tmp_test"
    root.mkdir(exist_ok=True)
    path = root / f"case_{uuid.uuid4().hex}"
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path)
        try:
            root.rmdir()
        except OSError:
            pass  # Another test process may still be using the directory.


def image_bytes(fmt):
    stream = io.BytesIO()
    Image.new("RGB", (32, 24)).save(stream, format=fmt)
    return stream.getvalue()


def settings(tmp_path, maximum=10 * 1024 * 1024):
    return Settings("https://example.invalid/mlflow", "ParkVision-Detector", "champion",
                    maximum, tmp_path, "WARNING")


def output(path, detections=None):
    detections = detections if detections is not None else [
        {"class_id": 0, "confidence": 0.94, "xyxy": [1.0, 2.0, 11.0, 12.0]},
        {"class_id": 1, "confidence": 0.8, "xyxy": [3.0, 4.0, 13.0, 14.0]},
    ]
    return pd.DataFrame([{"image_path": str(path), "empty_count": 1, "occupied_count": 1,
                          "detections_json": json.dumps(detections)}])


class FakeModel:
    def __init__(self, fail=False, invalid=False):
        self.calls = []
        self.fail = fail
        self.invalid = invalid

    def predict(self, frame):
        path = Path(frame.iloc[0]["image_path"])
        assert path.is_file()
        self.calls.append(path)
        if self.fail:
            raise RuntimeError("private backend details")
        result = output(path)
        if self.invalid:
            result.loc[0, "empty_count"] = 99
        return result


def service(model):
    return ModelService(model, ModelInfo("ParkVision-Detector", "7", "yolov8s",
                                         "a" * 32, "models:/ParkVision-Detector/7", "champion"))


@pytest.fixture
def loaded(tmp_path):
    model = FakeModel()
    instance = service(model)
    app = create_app(settings(tmp_path), lambda _: instance)
    with TestClient(app) as client:
        yield client, model, tmp_path


def upload(client, fmt="JPEG", data=None):
    media = "image/jpeg" if fmt == "JPEG" else "image/png"
    return client.post("/predict", files={"file": ("parking.jpg", image_bytes(fmt) if data is None else data, media)})


def test_health_ready_info_and_reuse(loaded):
    client, model, directory = loaded
    assert client.get("/health").json() == {"status": "ok", "gpu_available": False, "model_version": "7"}
    assert client.get("/ready").json() == {"status": "ready"}
    info = client.get("/model-info").json()
    assert info == {"name": "ParkVision-Detector", "version": "7", "architecture": "yolov8s",
                    "source_run_id": "a" * 32, "model_uri": "models:/ParkVision-Detector/7",
                    "alias": "champion", "status": "loaded"}
    for fmt in ("JPEG", "PNG", "JPEG"):
        response = upload(client, fmt)
        assert response.status_code == 200
        body = response.json()
        assert body["model"] == {"name": "ParkVision-Detector", "version": "7",
                                 "architecture": "yolov8s"}
        assert body["counts"] == {"empty": 1, "occupied": 1, "total": 2}
        assert body["detections"] == [
            {"class_id": 0, "class_name": "space-empty", "confidence": 0.94,
             "bbox": [1.0, 2.0, 11.0, 12.0]},
            {"class_id": 1, "class_name": "space-occupied", "confidence": 0.8,
             "bbox": [3.0, 4.0, 13.0, 14.0]},
        ]
        assert body["inference_time_ms"] >= 0
        assert list(directory.iterdir()) == []
    assert len(model.calls) == 3


def test_unavailable(tmp_path):
    app = create_app(settings(tmp_path), lambda _: (_ for _ in ()).throw(RuntimeError("secret")))
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/ready").status_code == 503
        assert client.get("/model-info").json()["status"] == "unavailable"
        response = upload(client)
        assert response.status_code == 503
        assert "secret" not in response.text


@pytest.mark.parametrize("data", [b"", b"not an image", b"\x89PNG\r\n\x1a\ntruncated"])
def test_invalid_image(loaded, data):
    client, model, directory = loaded
    assert upload(client, data=data).status_code == 400
    assert model.calls == []
    assert list(directory.iterdir()) == []


def test_size_missing_file_and_media_type(tmp_path):
    app = create_app(settings(tmp_path, maximum=20), lambda _: service(FakeModel()))
    with TestClient(app) as client:
        assert upload(client).status_code == 413
        assert client.post("/predict").status_code == 422
        assert client.post("/predict", files={"file": ("x.gif", b"GIF89a", "image/gif")}).status_code == 400
        assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("fail,invalid", [(True, False), (False, True)])
def test_inference_failure_cleans_file(tmp_path, fail, invalid):
    model = FakeModel(fail=fail, invalid=invalid)
    app = create_app(settings(tmp_path), lambda _: service(model))
    with TestClient(app) as client:
        response = upload(client)
        assert response.status_code == 500
        assert response.json()["detail"]["code"] == "inference_failed"
        assert "private backend details" not in response.text
        assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("mutation", [
    lambda r: r.assign(empty_count=-1),
    lambda r: r.assign(detections_json='[{"class_id":2,"confidence":0.5,"xyxy":[1,2,3,4]}]'),
    lambda r: r.assign(detections_json='[{"class_id":0,"confidence":1.2,"xyxy":[1,2,3,4]}]'),
    lambda r: r.assign(detections_json='[{"class_id":0,"confidence":0.5,"xyxy":[3,2,1,4]}]'),
    lambda r: r.drop(columns=["detections_json"]),
])
def test_invalid_model_output_is_rejected(mutation):
    path = Path("image.jpg")
    with pytest.raises(ValueError):
        parse_prediction(mutation(output(path)), path, service(FakeModel()).info, 1.0)


def test_loader_pins_alias_version_and_only_reads(monkeypatch, tmp_path):
    import mlflow

    calls = []
    version = SimpleNamespace(version="7", run_id="a" * 32,
                              tags={"parkvision.inference_verified": "true",
                                    "parkvision.architecture": "yolov8s"})

    class Client:
        def __init__(self, **kwargs):
            calls.append(("client", kwargs))

        def get_model_version_by_alias(self, name, alias):
            calls.append(("alias", name, alias))
            return version

    class PyFunc(FakeModel):
        metadata = SimpleNamespace(signature=SimpleNamespace(inputs=SimpleNamespace(
            inputs=[SimpleNamespace(name="image_path", type="DataType.string")])))

    monkeypatch.setattr(mlflow, "set_tracking_uri", lambda uri: calls.append(("tracking", uri)))
    monkeypatch.setattr(mlflow, "set_registry_uri", lambda uri: calls.append(("registry", uri)))
    monkeypatch.setattr("src.api.model_service._obtain_package",
                        lambda *_: tmp_path)
    bundled = tmp_path / "code" / "src" / "models"
    bundled.mkdir(parents=True)
    (bundled / "registry_adapter.py").write_text("", encoding="utf-8")
    instance = load_champion(
        settings(tmp_path), client_factory=Client,
        artifact_downloader=lambda **_: pytest.fail("package should be supplied by cache"),
        model_loader=lambda uri: calls.append(("load", uri)) or PyFunc(),
    )
    assert instance.info.version == "7"
    assert ("load", str(tmp_path)) in calls
    assert len([call for call in calls if call[0] == "alias"]) == 1
    assert not any(call[0] in ("set_alias", "create_model_version", "set_tag") for call in calls)
    assert {item.name for item in tmp_path.iterdir()} == {"code"}



def test_model_metadata_reports_classes_and_metrics(loaded):
    client, _model, _directory = loaded
    body = client.get("/model/metadata").json()
    assert body == {"name": "ParkVision-Detector", "version": "7", "architecture": "yolov8s",
                    "classes": ["space-empty", "space-occupied"], "validation_metrics": {}}


def test_model_metadata_unavailable(tmp_path):
    app = create_app(settings(tmp_path), lambda _: (_ for _ in ()).throw(RuntimeError("secret")))
    with TestClient(app) as client:
        response = client.get("/model/metadata")
        assert response.status_code == 503
        assert response.json()["detail"]["code"] == "model_unavailable"


def test_predict_visualize_returns_annotated_image(loaded):
    client, model, directory = loaded
    response = client.post("/predict/visualize",
                           files={"file": ("parking.jpg", image_bytes("JPEG"), "image/jpeg")})
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    with Image.open(io.BytesIO(response.content)) as annotated:
        annotated.load()
        assert annotated.size == (32, 24)
    assert len(model.calls) == 1
    assert list(directory.iterdir()) == []


def test_predict_visualize_rejects_invalid_image(loaded):
    client, model, directory = loaded
    response = client.post("/predict/visualize",
                           files={"file": ("x.jpg", b"not an image", "image/jpeg")})
    assert response.status_code == 400
    assert model.calls == []
    assert list(directory.iterdir()) == []


def test_predict_visualize_unavailable(tmp_path):
    app = create_app(settings(tmp_path), lambda _: (_ for _ in ()).throw(RuntimeError("secret")))
    with TestClient(app) as client:
        response = client.post("/predict/visualize",
                               files={"file": ("x.jpg", image_bytes("JPEG"), "image/jpeg")})
        assert response.status_code == 503


def test_loader_reads_validation_metrics_from_the_run(monkeypatch, tmp_path):
    import mlflow

    version = SimpleNamespace(version="7", run_id="a" * 32,
                              tags={"parkvision.inference_verified": "true",
                                    "parkvision.architecture": "yolov8s"})
    run = SimpleNamespace(data=SimpleNamespace(metrics={
        "best_val/metrics/mAP50(B)": 0.91, "best_val/metrics/precision(B)": 0.88,
        "train/box_loss": 0.4,
    }))

    class Client:
        def __init__(self, **kwargs):
            pass

        def get_model_version_by_alias(self, name, alias):
            return version

        def get_run(self, run_id):
            assert run_id == version.run_id
            return run

    class PyFunc(FakeModel):
        metadata = SimpleNamespace(signature=SimpleNamespace(inputs=SimpleNamespace(
            inputs=[SimpleNamespace(name="image_path", type="DataType.string")])))

    monkeypatch.setattr(mlflow, "set_tracking_uri", lambda uri: None)
    monkeypatch.setattr(mlflow, "set_registry_uri", lambda uri: None)
    monkeypatch.setattr("src.api.model_service._obtain_package", lambda *_: tmp_path)
    bundled = tmp_path / "code" / "src" / "models"
    bundled.mkdir(parents=True)
    (bundled / "registry_adapter.py").write_text("", encoding="utf-8")
    instance = load_champion(
        settings(tmp_path), client_factory=Client,
        artifact_downloader=lambda **_: pytest.fail("package should be supplied by cache"),
        model_loader=lambda uri: PyFunc(),
    )
    assert instance.info.validation_metrics == {
        "metrics/mAP50(B)": 0.91, "metrics/precision(B)": 0.88,
    }