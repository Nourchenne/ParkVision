"""Local-only video REST, byte-range, WebSocket, and regression tests."""

import json
import time
from dataclasses import replace

import cv2
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from src.api.config import Settings
from src.api.main import create_app
from src.api.model_service import ModelInfo, ModelService


class FakePyFunc:
    def predict(self, frame):
        path = frame.iloc[0]["image_path"]
        return pd.DataFrame([{
            "image_path": path,
            "empty_count": 1,
            "occupied_count": 0,
            "detections_json": json.dumps([
                {"class_id": 0, "confidence": 0.9, "xyxy": [1, 2, 10, 12]},
            ]),
        }])


def model_service():
    return ModelService(FakePyFunc(), ModelInfo(
        "ParkVision-Detector", "2", "yolov8s", None,
        "models:/ParkVision-Detector/2", "champion"))


def settings(tmp_path, **changes):
    value = Settings(
        "unused", "ParkVision-Detector", "champion", 1024 * 1024, tmp_path, "WARNING",
        max_video_bytes=1024 * 1024, video_storage_budget_bytes=2 * 1024 * 1024,
        video_max_width=64, video_max_height=64, max_video_duration_seconds=10,
        max_video_frames=10, video_processing_timeout_seconds=10,
        youtube_socket_timeout_seconds=1,
    )
    return replace(value, **changes)


def mp4_bytes(tmp_path):
    path = tmp_path / "fixture.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 2.0, (32, 24))
    assert writer.isOpened()
    for value in (0, 80, 160, 240):
        writer.write(np.full((24, 32, 3), value, dtype=np.uint8))
    writer.release()
    return path.read_bytes()


def local_client(tmp_path):
    app = create_app(settings(tmp_path), loader=lambda _: model_service())
    return TestClient(app, base_url="http://localhost", client=("127.0.0.1", 50000))


def wait_terminal(client, session_id):
    for _ in range(200):
        response = client.get(f"/video/sessions/{session_id}")
        assert response.status_code == 200
        if response.json()["status"] in {"completed", "failed", "cancelled"}:
            return response.json()
        time.sleep(0.01)
    raise AssertionError("session did not terminate")


def test_mp4_upload_lifecycle_preview_frames_and_range(tmp_path):
    with local_client(tmp_path) as client:
        response = client.post(
            "/video/upload",
            files={"file": ("parking.mp4", mp4_bytes(tmp_path), "video/mp4")},
            headers={"Origin": "http://localhost"},
        )
        assert response.status_code == 202
        session_id = response.json()["id"]
        started = client.post(f"/video/sessions/{session_id}/playback",
                              json={"current_time_seconds": 1.0, "playing": True},
                              headers={"Origin": "http://localhost"})
        assert started.status_code == 200
        snapshot = wait_terminal(client, session_id)
        assert snapshot["status"] == "completed"
        assert snapshot["frames_processed"] >= 1
        assert snapshot["latest"]["counts"] == {"empty": 1, "occupied": 0, "total": 1}

        preview = client.get(f"/video/sessions/{session_id}/preview")
        assert preview.status_code == 200
        assert preview.headers["cache-control"] == "no-store"
        assert preview.content.startswith(b"\xff\xd8")

        frames = client.get(f"/video/sessions/{session_id}/frames?limit=1").json()
        assert frames["total"] >= 1
        assert len(frames["items"]) == 1

        source = client.get(
            f"/video/sessions/{session_id}/source", headers={"Range": "bytes=0-9"})
        assert source.status_code == 206
        assert source.headers["content-range"].startswith("bytes 0-9/")
        assert len(source.content) == 10

        cancelled = client.delete(
            f"/video/sessions/{session_id}", headers={"Origin": "http://localhost"})
        assert cancelled.status_code == 202


def test_playback_pause_and_end_gate_predictions(tmp_path):
    with local_client(tmp_path) as client:
        response = client.post(
            "/video/upload",
            files={"file": ("parking.mp4", mp4_bytes(tmp_path), "video/mp4")},
            headers={"Origin": "http://localhost"},
        )
        session_id = response.json()["id"]
        endpoint = f"/video/sessions/{session_id}/playback"
        headers = {"Origin": "http://localhost"}
        time.sleep(0.1)
        assert client.get(f"/video/sessions/{session_id}").json()["frames_processed"] == 0
        assert client.post(endpoint, json={"current_time_seconds": 0, "playing": True},
                           headers=headers).status_code == 200
        for _ in range(100):
            if client.get(f"/video/sessions/{session_id}").json()["frames_processed"]:
                break
            time.sleep(0.01)
        else:
            raise AssertionError("first playback frame was not analyzed")
        assert client.post(endpoint, json={"current_time_seconds": 0.2, "playing": False},
                           headers=headers).status_code == 200
        time.sleep(0.1)
        assert client.get(f"/video/sessions/{session_id}").json()["frames_processed"] == 1
        assert client.post(endpoint, json={"current_time_seconds": 0.2,
                                           "playing": False, "ended": True},
                           headers=headers).status_code == 200
        snapshot = wait_terminal(client, session_id)
        assert snapshot["status"] == "completed"
        assert snapshot["frames_processed"] == 1


def test_playback_resume_processes_next_second(tmp_path):
    with local_client(tmp_path) as client:
        response = client.post(
            "/video/upload",
            files={"file": ("parking.mp4", mp4_bytes(tmp_path), "video/mp4")},
            headers={"Origin": "http://localhost"},
        )
        session_id = response.json()["id"]
        endpoint = f"/video/sessions/{session_id}/playback"
        headers = {"Origin": "http://localhost"}
        client.post(endpoint, json={"current_time_seconds": 0, "playing": True}, headers=headers)
        for _ in range(100):
            if client.get(f"/video/sessions/{session_id}").json()["frames_processed"] == 1:
                break
            time.sleep(0.01)
        else:
            raise AssertionError("first playback frame was not analyzed")
        client.post(endpoint, json={"current_time_seconds": 0.2, "playing": False}, headers=headers)
        time.sleep(0.1)
        assert client.get(f"/video/sessions/{session_id}").json()["frames_processed"] == 1
        client.post(endpoint, json={"current_time_seconds": 1.0, "playing": True}, headers=headers)
        snapshot = wait_terminal(client, session_id)
        assert snapshot["status"] == "completed"
        assert [item["timestamp_seconds"] for item in client.get(
            f"/video/sessions/{session_id}/frames").json()["items"]] == [0.0, 1.0]


def test_invalid_and_oversized_uploads_are_cleaned(tmp_path):
    with local_client(tmp_path) as client:
        invalid = client.post(
            "/video/upload", files={"file": ("bad.mp4", b"not video", "video/mp4")},
            headers={"Origin": "http://localhost"})
        assert invalid.status_code == 400
    tiny = settings(tmp_path, max_video_bytes=10, video_storage_budget_bytes=20)
    app = create_app(tiny, loader=lambda _: model_service())
    with TestClient(app, base_url="http://localhost", client=("127.0.0.1", 50000)) as client:
        oversized = client.post(
            "/video/upload", files={"file": ("big.mp4", b"x" * 11, "video/mp4")},
            headers={"Origin": "http://localhost"})
        assert oversized.status_code == 413


def test_capacity_response_has_code_for_dashboard(tmp_path):
    with local_client(tmp_path) as client:
        reserved, _ = client.app.state.video_manager.allocate_upload()
        try:
            response = client.post(
                "/video/upload",
                files={"file": ("parking.mp4", b"video", "video/mp4")},
                headers={"Origin": "http://localhost"},
            )
            assert response.status_code == 409
            assert response.json()["detail"]["code"] == "session_capacity"
        finally:
            client.app.state.video_manager.cancel(reserved.id)


def test_remote_peer_and_bad_origin_cannot_use_video_routes(tmp_path):
    app = create_app(settings(tmp_path), loader=lambda _: model_service())
    with TestClient(app, base_url="http://localhost", client=("203.0.113.9", 50000)) as remote:
        assert remote.get("/video/sessions/opaque-id").status_code == 403
    with TestClient(app, base_url="http://localhost", client=("127.0.0.1", 50000)) as local:
        response = local.post(
            "/video/upload", files={"file": ("x.mp4", b"x", "video/mp4")},
            headers={"Origin": "https://attacker.example"})
        assert response.status_code == 403


def test_forwarded_header_does_not_override_direct_peer(tmp_path):
    app = create_app(settings(tmp_path), loader=lambda _: model_service())
    with TestClient(app, base_url="http://localhost", client=("203.0.113.9", 50000)) as client:
        response = client.get("/video/sessions/opaque-id",
                              headers={"X-Forwarded-For": "127.0.0.1"})
        assert response.status_code == 403


def test_loopback_vite_proxy_origin_is_accepted_without_remote_access(tmp_path):
    app = create_app(settings(tmp_path), loader=lambda _: model_service())
    with TestClient(app, base_url="http://127.0.0.1:8000",
                    client=("127.0.0.1", 50000)) as local:
        response = local.post(
            "/video/upload", files={"file": ("bad.mp4", b"invalid", "video/mp4")},
            headers={"Origin": "http://127.0.0.1:5173"},
        )
        assert response.status_code == 400  # The request reached MP4 validation.
        rejected = local.post(
            "/video/upload", files={"file": ("bad.mp4", b"invalid", "video/mp4")},
            headers={"Origin": "http://127.0.0.1:9999"},
        )
        assert rejected.status_code == 403


def test_websocket_requires_local_origin_and_emits_session(tmp_path):
    with local_client(tmp_path) as client:
        created = client.post(
            "/video/upload",
            files={"file": ("parking.mp4", mp4_bytes(tmp_path), "video/mp4")},
            headers={"Origin": "http://localhost"},
        ).json()
        with client.websocket_connect(
            f"ws://localhost/video/sessions/{created['id']}/events",
            headers={"Origin": "http://localhost"},
        ) as websocket:
            event = websocket.receive_json()
            assert event["type"] in {"session", "terminal"}
            assert event["session"]["id"] == created["id"]


def test_websocket_rejects_missing_or_remote_origin(tmp_path):
    with local_client(tmp_path) as client:
        for headers in ({}, {"Origin": "https://attacker.example"}):
            with (
                pytest.raises(WebSocketDisconnect) as error,
                client.websocket_connect(
                    "ws://localhost/video/sessions/unknown/events", headers=headers,
                ),
            ):
                pass
            assert error.value.code == 1008


def test_websocket_accepts_fixed_vite_proxy_origin(tmp_path):
    app = create_app(settings(tmp_path), loader=lambda _: model_service())
    with (
        TestClient(app, base_url="http://127.0.0.1:8000",
                   client=("127.0.0.1", 50000)) as client,
        client.websocket_connect(
            "ws://127.0.0.1:8000/video/sessions/unknown/events",
            headers={"Origin": "http://127.0.0.1:5173"},
        ) as websocket,
    ):
        assert websocket.receive_json() == {"type": "error", "code": "session_not_found"}


def test_uvicorn_has_a_real_websocket_transport():
    from uvicorn.protocols.websockets.auto import AutoWebSocketsProtocol

    assert AutoWebSocketsProtocol is not None


def test_existing_health_endpoint_is_not_localhost_restricted(tmp_path):
    app = create_app(settings(tmp_path), loader=lambda _: model_service())
    with TestClient(app, base_url="http://example.test", client=("203.0.113.9", 50000)) as client:
        assert client.get("/health").status_code == 200
