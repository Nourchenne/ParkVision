"""The browser UI is served separately from the FastAPI inference API."""

from fastapi.testclient import TestClient

from src.api.config import Settings
from src.api.main import create_app
from src.api.model_service import ModelInfo, ModelService


class UnusedModel:
    def predict(self, _):
        raise AssertionError("dashboard must not run inference")


def test_obsolete_dashboard_routes_are_not_served(tmp_path):
    service = ModelService(UnusedModel(), ModelInfo(
        "ParkVision-Detector", "2", "yolov8s", None,
        "models:/ParkVision-Detector/2", "champion"))
    settings = Settings("unused", "ParkVision-Detector", "champion", 1024, tmp_path, "WARNING")
    app = create_app(settings, loader=lambda _: service)
    with TestClient(app, base_url="http://localhost", client=("127.0.0.1", 50000)) as client:
        assert client.get("/dashboard").status_code == 404
        assert client.get("/static/dashboard.js").status_code == 404
        assert client.get("/health").status_code == 200
