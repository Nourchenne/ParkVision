"""Deterministic, offline tests for the verified champion cache.

No test contacts DagsHub or downloads model artifacts. Fake clients, fake
PyFuncs and synthetic model packages are used throughout.
"""

import hashlib
import json
import os
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from src.api import model_cache
from src.api.config import Settings
from src.api.main import create_app
from src.api.model_cache import (
    IDENTITY_FILE,
    MANIFEST_FILE,
    CacheVerificationError,
    ModelIdentity,
    cache_entry,
    identity_from_version,
    import_candidate,
    verify_model_package,
)
from src.api.model_service import ModelInfo, ModelService, load_champion
from src.api.video_service import VideoMetadata
from src.api.video_sessions import VideoSessionManager

RUN_ID = "bf74a68ba942400caab0c9acbefabfa4"
CHECKPOINT = b"synthetic-checkpoint-bytes"
ADAPTER = b"synthetic-adapter-bytes"
MLMODEL = """flavors:
  python_function:
    artifacts:
      checkpoint:
        path: artifacts/best.pt
        uri: local-checkpoint
    code: code
    env:
      conda: conda.yaml
      virtualenv: python_env.yaml
    loader_module: mlflow.pyfunc.model
    python_model: python_model.pkl
    python_version: '3.10.11'
    streamable: false
mlflow_version: 3.17.2
model_uuid: test-model-uuid
signature:
  inputs: '[{"type": "string", "name": "image_path", "required": true}]'
  outputs: null
utc_time_created: '2026-11-20 09:30:03.693100'
"""


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def identity(**changes) -> ModelIdentity:
    values = {
        "name": "ParkVision-Detector", "version": "2", "run_id": RUN_ID,
        "checkpoint_sha256": sha(CHECKPOINT),
        "packaging_code_sha256": sha(ADAPTER),
    }
    values.update(changes)
    return ModelIdentity(**values)


def make_package(root: Path, version: str = "2", checkpoint: bytes = CHECKPOINT,
                 adapter: bytes = ADAPTER, complete: bool = True) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "MLmodel").write_text(MLMODEL, encoding="utf-8")
    (root / "python_model.pkl").write_bytes(b"pickle")
    (root / "conda.yaml").write_text("dependencies: []\n", encoding="utf-8")
    (root / "python_env.yaml").write_text("python: '3.10.11'\n", encoding="utf-8")
    (root / "requirements.txt").write_text("mlflow\n", encoding="utf-8")
    (root / "registered_model_meta").write_text(
        f"model_name: ParkVision-Detector\nmodel_version: '{version}'\n", encoding="utf-8")
    adapter_path = root / "code" / "src" / "models" / "registry_adapter.py"
    adapter_path.parent.mkdir(parents=True)
    adapter_path.write_bytes(adapter)
    checkpoint_path = root / "artifacts" / "best.pt"
    checkpoint_path.parent.mkdir(parents=True)
    if complete:
        checkpoint_path.write_bytes(checkpoint)
    return root


def version_stub(**changes):
    tags = {
        "parkvision.inference_verified": "true",
        "parkvision.checkpoint_sha256": sha(CHECKPOINT),
        "parkvision.packaging_code_sha256": sha(ADAPTER),
    }
    tags.update(changes.pop("tags", {}))
    values = {"version": "2", "run_id": RUN_ID, "tags": tags}
    values.update(changes)
    return SimpleNamespace(**values)


class Client:
    def __init__(self, version):
        self.version = version

    def __call__(self, **kwargs):
        return self

    def get_model_version_by_alias(self, name, alias):
        return self.version


class FakePyFunc:
    metadata = SimpleNamespace(signature=SimpleNamespace(inputs=SimpleNamespace(
        inputs=[SimpleNamespace(name="image_path", type="DataType.string")])))

    def predict(self, frame):
        return pd.DataFrame([{
            "image_path": frame.iloc[0]["image_path"], "empty_count": 0,
            "occupied_count": 0, "detections_json": "[]",
        }])


def settings_for(tmp_path: Path, version: str = "2", **changes) -> Settings:
    values = {
        "tracking_uri": "https://example.invalid/mlflow",
        "model_name": "ParkVision-Detector",
        "model_alias": "champion",
        "max_upload_bytes": 10 * 1024 * 1024,
        "temp_dir": tmp_path,
        "log_level": "WARNING",
        "model_cache_dir": tmp_path / "cache",
        "localhost_only": False,
        "model_reload_enabled": True,
        "model_reload_min_interval_seconds": 0,
    }
    values.update(changes)
    return Settings(**values)


def downloader_from(source: Path, calls: list, delay: float = 0.0):
    def download(**kwargs):
        calls.append(kwargs)
        if delay:
            time.sleep(delay)
        destination = Path(kwargs["dst_path"])
        shutil.copytree(source, destination, dirs_exist_ok=True)
        return destination
    return download


def load_with(tmp_path: Path, package: Path, version=None, **changes):
    calls = []
    service = load_champion(
        settings_for(tmp_path, **changes),
        client_factory=Client(version or version_stub()),
        artifact_downloader=downloader_from(package, calls),
        model_loader=lambda _: FakePyFunc(),
    )
    return service, calls


def test_identity_is_deterministic_and_binds_provenance():
    base = identity()
    assert base.key() == identity().key()
    assert base.key() != identity(version="3").key()
    assert base.key() != identity(run_id="other-run").key()
    assert base.key() != identity(checkpoint_sha256="a" * 64).key()
    assert base.key() != identity(packaging_code_sha256="b" * 64).key()
    assert identity_from_version("ParkVision-Detector", version_stub()) == base


def test_identity_requires_verified_provenance_tags():
    version = SimpleNamespace(version="2", run_id=RUN_ID,
                              tags={"parkvision.inference_verified": "true"})
    assert identity_from_version("ParkVision-Detector", version) is None
    with pytest.raises(CacheVerificationError):
        identity_from_version("ParkVision-Detector", SimpleNamespace(
            version="2", run_id=RUN_ID, tags={"parkvision.checkpoint_sha256": "xyz"}))


def test_missing_provenance_never_downloads_unverified_model(tmp_path):
    def forbidden(**_):
        raise AssertionError("download must not start")

    version = SimpleNamespace(version="2", run_id=RUN_ID,
                              tags={"parkvision.inference_verified": "true"})
    with pytest.raises(CacheVerificationError) as error:
        load_champion(settings_for(tmp_path), client_factory=Client(version),
                      artifact_downloader=forbidden, model_loader=lambda _: FakePyFunc())
    assert error.value.code == "missing_provenance"


def test_complete_package_verifies(tmp_path):
    package = make_package(tmp_path / "package")
    verify_model_package(package, identity())


def test_corrupted_checkpoint_is_rejected(tmp_path):
    package = make_package(tmp_path / "package")
    (package / "artifacts" / "best.pt").write_bytes(b"truncated")
    with pytest.raises(CacheVerificationError) as error:
        verify_model_package(package, identity())
    assert error.value.code == "checkpoint_hash_mismatch"


def test_missing_model_file_is_rejected(tmp_path):
    package = make_package(tmp_path / "package")
    (package / "python_model.pkl").unlink()
    with pytest.raises(CacheVerificationError) as error:
        verify_model_package(package, identity())
    assert error.value.code == "missing_model_file"


def test_wrong_version_identity_is_rejected(tmp_path):
    package = make_package(tmp_path / "package", version="3")
    with pytest.raises(CacheVerificationError) as error:
        verify_model_package(package, identity())
    assert error.value.code == "wrong_version"


def test_wrong_packaging_run_is_rejected(tmp_path):
    package = make_package(tmp_path / "package")
    record = identity(run_id="another-run").as_dict()
    (package / IDENTITY_FILE).write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(CacheVerificationError) as error:
        verify_model_package(package, identity())
    assert error.value.code == "wrong_identity"


def test_checkpoint_hash_mismatch_in_identity_is_rejected(tmp_path):
    package = make_package(tmp_path / "package")
    with pytest.raises(CacheVerificationError):
        verify_model_package(package, identity(checkpoint_sha256="c" * 64))


def test_adapter_hash_mismatch_is_rejected(tmp_path):
    package = make_package(tmp_path / "package", adapter=b"tampered")
    with pytest.raises(CacheVerificationError) as error:
        verify_model_package(package, identity())
    assert error.value.code == "adapter_hash_mismatch"


@pytest.mark.skipif(os.name == "nt",
                    reason="creating symlinks requires elevated privileges on Windows")
def test_symlinked_checkpoint_is_rejected(tmp_path):
    package = make_package(tmp_path / "package")
    outside = tmp_path / "outside.pt"
    outside.write_bytes(CHECKPOINT)
    checkpoint = package / "artifacts" / "best.pt"
    checkpoint.unlink()
    checkpoint.symlink_to(outside)
    with pytest.raises(CacheVerificationError):
        verify_model_package(package, identity())


def test_import_candidate_copies_without_touching_source(tmp_path):
    package = make_package(tmp_path / "candidate")
    cache = tmp_path / "cache"
    entry = import_candidate(package, identity(), cache)
    assert entry == cache_entry(cache, identity())
    verify_model_package(entry, identity())
    assert (package / MANIFEST_FILE).exists() is False
    verify_model_package(package, identity())
    assert (cache / "staging").exists()


def test_bytecode_cache_is_not_part_of_the_manifest(tmp_path):
    package = make_package(tmp_path / "candidate")
    cache = tmp_path / "cache"
    entry = import_candidate(package, identity(), cache)
    pycache = entry / "code" / "src" / "models" / "__pycache__"
    pycache.mkdir()
    (pycache / "registry_adapter.cpython-310.pyc").write_bytes(b"bytecode")
    verify_model_package(entry, identity())
    manifest = json.loads((entry / MANIFEST_FILE).read_text(encoding="utf-8"))
    assert not any("__pycache__" in name for name in manifest["files"])

def test_cache_miss_downloads_verifies_and_publishes_atomically(tmp_path):
    package = make_package(tmp_path / "remote")
    service, calls = load_with(tmp_path, package)
    assert service.info.version == "2"
    assert calls[0]["artifact_uri"] == "models:/ParkVision-Detector/2"
    assert "dst_path" in calls[0]
    entry = cache_entry(tmp_path / "cache", identity())
    verify_model_package(entry, identity())
    assert (entry / MANIFEST_FILE).is_file()
    assert (entry / IDENTITY_FILE).is_file()
    assert not list((tmp_path / "cache" / "staging").glob("stage-*"))


def test_verified_cache_hit_reuses_entry_without_download(tmp_path):
    package = make_package(tmp_path / "remote")
    load_with(tmp_path, package)
    calls = []

    def forbidden(**kwargs):
        calls.append(kwargs)
        raise AssertionError("cache hit must not download")

    service = load_champion(
        settings_for(tmp_path), client_factory=Client(version_stub()),
        artifact_downloader=forbidden, model_loader=lambda _: FakePyFunc(),
    )
    assert service.info.version == "2"
    assert calls == []


def test_corrupt_cache_entry_is_quarantined_and_redownloaded(tmp_path):
    package = make_package(tmp_path / "remote")
    load_with(tmp_path, package)
    entry = cache_entry(tmp_path / "cache", identity())
    (entry / "artifacts" / "best.pt").write_bytes(b"corrupt")
    service, calls = load_with(tmp_path, package)
    assert service.info.version == "2"
    assert len(calls) == 1
    assert any(item.name.startswith("entry-") and ".invalid-" in item.name
               for item in (tmp_path / "cache" / "entries").iterdir()
               if item != entry)


def test_partial_staged_download_is_never_published(tmp_path):
    package = make_package(tmp_path / "remote", complete=False)
    with pytest.raises(CacheVerificationError) as error:
        load_with(tmp_path, package)
    assert error.value.code == "missing_model_file"
    assert not (tmp_path / "cache" / "entries").exists()
    assert not list((tmp_path / "cache" / "staging").glob("stage-*"))


def test_remote_download_failure_leaves_no_cache_entry(tmp_path):
    def failing(**kwargs):
        raise RuntimeError("network unavailable")

    with pytest.raises(RuntimeError):
        load_champion(
            settings_for(tmp_path), client_factory=Client(version_stub()),
            artifact_downloader=failing, model_loader=lambda _: FakePyFunc(),
        )
    assert not (tmp_path / "cache" / "entries").exists()
    assert not list((tmp_path / "cache" / "staging").glob("stage-*"))


def test_concurrent_initialization_downloads_once(tmp_path):
    package = make_package(tmp_path / "remote")
    calls = []
    downloader = downloader_from(package, calls, delay=0.05)

    def run():
        return load_champion(
            settings_for(tmp_path), client_factory=Client(version_stub()),
            artifact_downloader=downloader, model_loader=lambda _: FakePyFunc(),
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: run(), range(2)))
    assert [item.info.version for item in results] == ["2", "2"]
    assert len(calls) == 1


def test_cache_lock_is_exclusive(tmp_path):
    cache = tmp_path / "cache"
    entered = threading.Event()
    release = threading.Event()

    def holder():
        with model_cache.cache_lock(cache, identity()):
            entered.set()
            release.wait(5)

    thread = threading.Thread(target=holder)
    thread.start()
    try:
        assert entered.wait(5)
        started = time.monotonic()
        with (pytest.raises(CacheVerificationError) as error,
              model_cache.cache_lock(cache, identity(), timeout=0.3)):
            pass
        assert error.value.code == "cache_busy"
        assert time.monotonic() - started >= 0.2
    finally:
        release.set()
        thread.join(5)


def _write_lock(cache: Path, content: str, age_seconds: float = 0.0) -> Path:
    locks = cache / "locks"
    locks.mkdir(parents=True, exist_ok=True)
    lock = locks / f"{identity().key()}.lock"
    lock.write_text(content, encoding="ascii")
    if age_seconds:
        old = time.time() - age_seconds
        os.utime(lock, (old, old))
    return lock


def test_dead_owner_lock_is_reclaimed(tmp_path):
    cache = tmp_path / "cache"
    lock = _write_lock(cache, "2147483647")
    with model_cache.cache_lock(cache, identity(), timeout=0.5):
        assert lock.exists()
    assert not lock.exists()


def test_live_owner_lock_is_never_reclaimed_even_when_old(tmp_path):
    cache = tmp_path / "cache"
    lock = _write_lock(cache, str(os.getpid()), age_seconds=100_000)
    with (pytest.raises(CacheVerificationError) as error,
          model_cache.cache_lock(cache, identity(), timeout=0.2)):
        pass
    assert error.value.code == "cache_busy"
    assert lock.read_text(encoding="ascii") == str(os.getpid())


def test_unreadable_lock_owner_is_not_reclaimed(tmp_path):
    cache = tmp_path / "cache"
    lock = _write_lock(cache, "not-a-pid", age_seconds=100_000)
    with (pytest.raises(CacheVerificationError),
          model_cache.cache_lock(cache, identity(), timeout=0.2)):
        pass
    assert lock.exists()


def test_mlflow_network_settings_are_bounded_and_restored(tmp_path):
    settings = settings_for(tmp_path, mlflow_http_request_timeout=300,
                            mlflow_http_request_max_retries=2,
                            mlflow_http_request_backoff_factor=0)
    import os
    assert "MLFLOW_HTTP_REQUEST_TIMEOUT" not in os.environ
    with model_cache.mlflow_network_settings(settings):
        assert os.environ["MLFLOW_HTTP_REQUEST_TIMEOUT"] == "300"
        assert os.environ["MLFLOW_HTTP_REQUEST_BACKOFF_FACTOR"] == "0"
    assert "MLFLOW_HTTP_REQUEST_TIMEOUT" not in os.environ


def _fake_service(tmp_path):
    return ModelService(FakePyFunc(), ModelInfo(
        "ParkVision-Detector", "2", "yolov8s", RUN_ID,
        "models:/ParkVision-Detector/2", "champion"))


def test_readiness_failure_and_recovery_preserve_contracts(tmp_path):
    attempts = []

    def loader(settings):
        attempts.append(1)
        if len(attempts) <= 2:
            raise RuntimeError("private remote failure")
        return _fake_service(tmp_path)

    app = create_app(settings_for(tmp_path, model_reload_min_interval_seconds=0),
                     loader, local_check=lambda _: True)
    with TestClient(app) as client:
        assert client.get("/health").json() == {"status": "ok", "gpu_available": False, "model_version": None}
        assert client.get("/ready").status_code == 503
        assert client.get("/model-info").json()["status"] == "unavailable"
        assert client.post("/internal/model/reload").status_code == 503
        recovered = client.post("/internal/model/reload")
        assert recovered.status_code == 200
        assert recovered.json() == {"status": "ready", "version": "2"}
        assert client.get("/ready").json() == {"status": "ready"}
        assert client.get("/model-info").json()["version"] == "2"


def test_reload_is_throttled(tmp_path):
    app = create_app(settings_for(tmp_path, model_reload_min_interval_seconds=3600),
                     lambda _: _fake_service(tmp_path), local_check=lambda _: True)
    with TestClient(app) as client:
        assert client.post("/internal/model/reload").status_code == 200
        throttled = client.post("/internal/model/reload")
        assert throttled.status_code == 429
        assert throttled.json()["detail"]["code"] == "reload_throttled"


def test_video_session_keeps_one_model_across_reload(tmp_path):
    used = []
    started = threading.Event()
    release = threading.Event()
    original = _fake_service(tmp_path)

    def analyzer(path, service, settings, cancel_event, deadline, publish, *, sample_gate=None):
        used.append(service)
        started.set()
        release.wait(5)
        return VideoMetadata(None, 640, 480, None)

    manager = VideoSessionManager(
        original, settings_for(tmp_path), analyzer=analyzer,
        storage_root=tmp_path / "video")
    record = manager._reserve("upload")
    (record.source_path).write_bytes(b"\x00\x00\x00\x18ftypmp42")
    try:
        manager.start_upload(record.id, VideoMetadata(None, 640, 480, None))
        assert started.wait(5)
        manager.set_model_service(_fake_service(tmp_path))
    finally:
        release.set()
        manager.shutdown()
    assert used == [original]


def test_reload_is_disabled_unless_explicitly_enabled(tmp_path):
    app = create_app(settings_for(tmp_path, model_reload_enabled=False),
                     lambda _: _fake_service(tmp_path), local_check=lambda _: True)
    with TestClient(app) as client:
        assert client.post("/internal/model/reload").status_code == 404


def test_reload_requires_loopback_even_in_public_mode(tmp_path):
    app = create_app(settings_for(tmp_path, localhost_only=False),
                     lambda _: _fake_service(tmp_path), local_check=lambda _: False)
    with TestClient(app) as client:
        response = client.post("/internal/model/reload")
        assert response.status_code == 403
        assert response.json()["detail"]["code"] == "local_access_required"


def test_image_api_still_works_with_cache_settings(tmp_path):
    app = create_app(settings_for(tmp_path), lambda _: _fake_service(tmp_path))
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/ready").status_code == 200
        info = client.get("/model-info").json()
        assert info["version"] == "2" and info["alias"] == "champion"
        stream = BytesIO()
        Image.new("RGB", (24, 24)).save(stream, format="JPEG")
        response = client.post("/predict", files={
            "file": ("parking.jpg", stream.getvalue(), "image/jpeg")})
        assert response.status_code == 200
        assert response.json()["model"]["version"] == "2"
