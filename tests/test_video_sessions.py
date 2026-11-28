"""Bounded session lifecycle, storage, lease, and cleanup tests."""

import time
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from src.api.config import Settings
from src.api.schemas import Counts, FrameResult, ModelSummary
from src.api.video_service import VideoMetadata, VideoProcessingError
from src.api.video_sessions import SessionConflict, SessionNotFound, VideoSessionManager


def settings(tmp_path, **changes):
    value = Settings("unused", "ParkVision-Detector", "champion", 1024, tmp_path, "WARNING",
                     max_video_bytes=1000, video_storage_budget_bytes=2000,
                     max_video_frames=3, video_processing_timeout_seconds=10,
                     youtube_socket_timeout_seconds=1, video_orphan_grace_seconds=1)
    return replace(value, **changes)


def result(timestamp=0.0):
    return FrameResult(
        timestamp_seconds=timestamp, source_width=32, source_height=24,
        model=ModelSummary(name="ParkVision-Detector", version="2", architecture="yolov8s"),
        counts=Counts(empty=1, occupied=0, total=1), detections=[],
        decode_time_ms=1, inference_time_ms=2, analysis_time_ms=3, session_elapsed_ms=4,
    )


def wait_terminal(manager, session_id):
    for _ in range(200):
        snapshot = manager.get(session_id)
        if snapshot.status in {"completed", "failed", "cancelled"}:
            return snapshot
        time.sleep(0.005)
    raise AssertionError("session did not terminate")


def successful_analyzer(path, model, settings, cancel, deadline, publish, *, sample_gate=None):
    publish(result(), b"jpeg")
    return VideoMetadata(1.0, 32, 24, 1.0)


def allocate(manager):
    snapshot, path = manager.allocate_upload()
    path.write_bytes(b"video")
    manager.start_upload(snapshot.id, VideoMetadata(1.0, 32, 24, 1.0))
    return snapshot.id, path


def test_session_completes_and_publishes_bounded_state(tmp_path):
    manager = VideoSessionManager(object(), settings(tmp_path), analyzer=successful_analyzer,
                                  storage_root=tmp_path / "store")
    try:
        session_id, path = allocate(manager)
        snapshot = wait_terminal(manager, session_id)
        assert snapshot.status == "completed"
        assert snapshot.frames_processed == 1
        assert manager.preview(session_id) == b"jpeg"
        assert path.exists()  # retained for uploaded playback until expiry/cancel
    finally:
        manager.shutdown()


def test_late_inference_result_is_not_discarded(tmp_path):
    manager = VideoSessionManager(object(), settings(tmp_path), analyzer=successful_analyzer,
                                  storage_root=tmp_path / "store")
    try:
        record = manager._reserve("upload")
        manager.update_playback(record.id, 2.0, True)
        assert manager._await_playback(record.id, 0.0, 1.0, True)[0] == "skip"
        assert manager._await_playback(record.id, 0.0, 1.0, False)[0] == "process"
    finally:
        manager.shutdown()


def test_capacity_is_reserved_before_worker_starts(tmp_path):
    gate = False

    def slow(*args, **kwargs):
        while not args[3].is_set() and not gate:
            time.sleep(0.005)
        raise VideoProcessingError("cancelled", "Analysis was cancelled")

    manager = VideoSessionManager(object(), settings(tmp_path), analyzer=slow,
                                  storage_root=tmp_path / "store")
    try:
        first, _ = manager.allocate_upload()
        with pytest.raises(SessionConflict, match="capacity"):
            manager.allocate_upload()
        manager.cancel(first.id)
    finally:
        manager.shutdown()


def test_active_worker_observes_cancellation(tmp_path):
    started = False

    def cancellable(*args, **kwargs):
        nonlocal started
        started = True
        cancel = args[3]
        for _ in range(200):
            if cancel.is_set():
                raise VideoProcessingError("cancelled", "Analysis was cancelled")
            time.sleep(0.002)
        raise AssertionError("cancellation was not delivered")

    manager = VideoSessionManager(object(), settings(tmp_path), analyzer=cancellable,
                                  storage_root=tmp_path / "store")
    try:
        session_id, path = allocate(manager)
        for _ in range(200):
            if started:
                break
            time.sleep(0.002)
        snapshot = manager.cancel(session_id)
        assert snapshot.status == "cancelled"
        assert wait_terminal(manager, session_id).status == "cancelled"
        for _ in range(200):
            if not path.exists():
                break
            time.sleep(0.002)
        assert not path.exists()
    finally:
        manager.shutdown()


def test_processing_timeout_is_reported_with_safe_error(tmp_path):
    def timeout(*_, **kwargs):
        raise VideoProcessingError(
            "processing_timeout", "Analysis exceeded its processing time limit")

    manager = VideoSessionManager(object(), settings(tmp_path), analyzer=timeout,
                                  storage_root=tmp_path / "store")
    try:
        session_id, _ = allocate(manager)
        snapshot = wait_terminal(manager, session_id)
        assert snapshot.status == "failed"
        assert snapshot.error.code == "processing_timeout"
    finally:
        manager.shutdown()


def test_cancelled_source_deletes_after_last_lease(tmp_path):
    manager = VideoSessionManager(object(), settings(tmp_path), analyzer=successful_analyzer,
                                  storage_root=tmp_path / "store")
    try:
        session_id, path = allocate(manager)
        wait_terminal(manager, session_id)
        with manager.lease_source(session_id):
            manager.cancel(session_id)
            assert path.exists()
        assert not path.exists()
    finally:
        manager.shutdown()


def test_worker_error_is_sanitized_and_cleans_source(tmp_path):
    def fail(*_, **kwargs):
        raise RuntimeError("token=private")

    manager = VideoSessionManager(object(), settings(tmp_path), analyzer=fail,
                                  storage_root=tmp_path / "store")
    try:
        session_id, path = allocate(manager)
        snapshot = wait_terminal(manager, session_id)
        assert snapshot.status == "failed"
        assert snapshot.error.code == "video_processing_failed"
        assert "private" not in snapshot.error.message
        assert not path.exists()
    finally:
        manager.shutdown()


def test_expired_session_is_no_longer_reconnectable(tmp_path):
    current = [datetime.now(timezone.utc)]
    manager = VideoSessionManager(
        object(), settings(tmp_path, video_session_ttl_seconds=1),
        analyzer=successful_analyzer, storage_root=tmp_path / "store",
        now=lambda: current[0],
    )
    try:
        session_id, path = allocate(manager)
        wait_terminal(manager, session_id)
        current[0] += timedelta(seconds=2)
        assert manager.expire() == 1
        assert not path.exists()
        with pytest.raises(SessionNotFound):
            manager.get(session_id)
    finally:
        manager.shutdown()


def test_storage_budget_rejects_second_reservation(tmp_path):
    manager = VideoSessionManager(
        object(), settings(tmp_path, max_active_video_sessions=2,
                           video_storage_budget_bytes=1000),
        analyzer=successful_analyzer, storage_root=tmp_path / "store")
    try:
        manager.allocate_upload()
        with pytest.raises(SessionConflict, match="storage budget"):
            manager.allocate_upload()
    finally:
        manager.shutdown()


def test_orphan_cleanup_only_removes_old_marked_directory(tmp_path):
    root = tmp_path / "store"
    root.mkdir()
    old = root / "old"
    old.mkdir()
    (old / ".parkvision-video-session").write_text("parkvision-video-v1\n")
    unknown = root / "unknown"
    unknown.mkdir()
    old_time = time.time() - 10
    import os
    os.utime(old, (old_time, old_time))
    manager = VideoSessionManager(object(), settings(tmp_path), analyzer=successful_analyzer,
                                  storage_root=root)
    try:
        assert not old.exists()
        assert unknown.exists()
    finally:
        manager.shutdown()


def test_cancel_leaves_cleanup_to_the_running_worker(tmp_path):
    """The worker still holds the source open: deleting it during cancel failed on Windows."""
    import threading

    release = threading.Event()
    started = threading.Event()

    def holding(*args, **kwargs):
        started.set()
        cancel = args[3]
        assert cancel.wait(2), "cancellation was not delivered"
        assert release.wait(2), "test did not release the worker"
        raise VideoProcessingError("cancelled", "Analysis was cancelled")

    manager = VideoSessionManager(object(), settings(tmp_path), analyzer=holding,
                                  storage_root=tmp_path / "store")
    try:
        session_id, path = allocate(manager)
        assert started.wait(2)
        assert manager.cancel(session_id).status == "cancelled"
        assert path.exists()  # not deleted under the worker's feet
        release.set()
        for _ in range(400):
            if not path.exists():
                break
            time.sleep(0.005)
        assert not path.exists()
    finally:
        release.set()
        manager.shutdown()


def test_cancel_without_worker_deletes_immediately(tmp_path):
    manager = VideoSessionManager(object(), settings(tmp_path), analyzer=successful_analyzer,
                                  storage_root=tmp_path / "store")
    try:
        snapshot, path = manager.allocate_upload()
        path.write_bytes(b"video")
        assert manager.cancel(snapshot.id).status == "cancelled"
        assert not path.exists()
    finally:
        manager.shutdown()


def test_failed_deletion_is_deferred_and_retried_on_expiry(tmp_path, monkeypatch):
    import src.api.video_sessions as sessions

    clock = {"now": datetime.now(timezone.utc)}
    manager = VideoSessionManager(object(), settings(tmp_path), analyzer=successful_analyzer,
                                  storage_root=tmp_path / "store", now=lambda: clock["now"])
    try:
        session_id, path = allocate(manager)
        wait_terminal(manager, session_id)
        real_rmtree = sessions.shutil.rmtree

        def locked(_path, *args, **kwargs):
            raise PermissionError("file is in use")

        monkeypatch.setattr(sessions.shutil, "rmtree", locked)
        manager.cancel(session_id)  # used to raise PermissionError, hence HTTP 500
        assert path.exists() and manager._sessions[session_id].pending_cleanup
        monkeypatch.setattr(sessions.shutil, "rmtree", real_rmtree)
        clock["now"] += timedelta(seconds=manager.settings.video_session_ttl_seconds + 5)
        assert manager.expire() == 1
        assert not path.exists()
    finally:
        manager.shutdown()
