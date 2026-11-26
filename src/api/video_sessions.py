"""Single-process bounded video sessions, owned storage, and file leases."""

import shutil
import tempfile
import threading
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from src.api.config import Settings
from src.api.model_service import ModelService
from src.api.schemas import FrameResult, SessionError, VideoSessionSnapshot
from src.api.video_service import VideoMetadata, VideoProcessingError, analyze_video

ACTIVE = {"queued", "acquiring", "processing"}
TERMINAL = {"completed", "failed", "cancelled"}
MARKER = ".parkvision-video-session"


class SessionNotFound(KeyError):
    pass


class SessionConflict(RuntimeError):
    pass


@dataclass
class _Session:
    id: str
    source_type: str
    directory: Path
    source_path: Path
    reserved_bytes: int
    status: str = "queued"
    revision: int = 0
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    metadata: VideoMetadata | None = None
    results: list[FrameResult] = field(default_factory=list)
    preview: bytes | None = None
    error: SessionError | None = None
    youtube_video_id: str | None = None
    youtube_url: str | None = None
    cancel_event: threading.Event = field(default_factory=threading.Event)
    leases: int = 0
    pending_cleanup: bool = False
    playback_time: float = 0.0
    playback_playing: bool = False
    playback_ended: bool = False
    last_playback_update: float = field(default_factory=time.monotonic)


class SourceLease:
    def __init__(self, manager, session_id: str, path: Path, size: int):
        self.manager = manager
        self.session_id = session_id
        self.path = path
        self.size = size
        self._closed = False

    def __enter__(self):
        return self

    def __exit__(self, *_):
        if not self._closed:
            self._closed = True
            self.manager._release_lease(self.session_id)


class VideoSessionManager:
    def __init__(
        self,
        model_service: ModelService,
        settings: Settings,
        *,
        analyzer: Callable = analyze_video,
        youtube_source=None,
        storage_root: Path | None = None,
        monotonic: Callable[[], float] = time.monotonic,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ):
        self.model_service = model_service
        self.settings = settings
        self.analyzer = analyzer
        self.youtube_source = youtube_source
        self.monotonic = monotonic
        self.now = now
        base = storage_root or Path(settings.temp_dir or tempfile.gettempdir()) / "parkvision-video"
        self.root = base.resolve()
        self._lock = threading.RLock()
        self._playback_condition = threading.Condition(self._lock)
        self._sessions: dict[str, _Session] = {}
        self._reserved_bytes = 0
        self._orphan_bytes = self._cleanup_orphans()
        self._executor = ThreadPoolExecutor(
            max_workers=settings.max_active_video_sessions,
            thread_name_prefix="parkvision-video",
        )
        self._closed = False

    def _owned_directory(self, path: Path) -> bool:
        try:
            resolved = path.resolve()
            resolved.relative_to(self.root)
        except (OSError, ValueError):
            return False
        return (resolved.parent == self.root and resolved.is_dir() and not resolved.is_symlink()
                and (resolved / MARKER).is_file())

    @staticmethod
    def _directory_size(path: Path) -> int:
        return sum(item.stat().st_size for item in path.rglob("*")
                   if item.is_file() and not item.is_symlink())

    def _cleanup_orphans(self) -> int:
        if not self.root.exists():
            return 0
        remaining = 0
        cutoff = time.time() - self.settings.video_orphan_grace_seconds
        for child in self.root.iterdir():
            if not self._owned_directory(child):
                continue
            try:
                if child.stat().st_mtime < cutoff:
                    shutil.rmtree(child)
                else:
                    remaining += self._directory_size(child)
            except OSError:
                remaining += self._directory_size(child) if child.exists() else 0
        return remaining

    def _touch(self, record: _Session):
        record.updated_at = self.now()
        record.revision += 1

    def _snapshot(self, record: _Session) -> VideoSessionSnapshot:
        latest = record.results[-1] if record.results else None
        return VideoSessionSnapshot(
            id=record.id,
            source_type=record.source_type,
            status=record.status,
            revision=record.revision,
            created_at=record.created_at.isoformat(),
            updated_at=record.updated_at.isoformat(),
            duration_seconds=record.metadata.duration_seconds if record.metadata else None,
            source_width=record.metadata.width if record.metadata else None,
            source_height=record.metadata.height if record.metadata else None,
            current_timestamp_seconds=latest.timestamp_seconds if latest else None,
            analyzed_frontier_seconds=latest.timestamp_seconds if latest else None,
            frames_processed=len(record.results),
            sample_fps=self.settings.video_sample_fps,
            latest=latest,
            error=record.error,
            youtube_video_id=record.youtube_video_id,
        )

    def _record(self, session_id: str) -> _Session:
        try:
            return self._sessions[session_id]
        except KeyError as error:
            raise SessionNotFound(session_id) from error

    def _reserve(self, source_type: str, youtube_video_id=None, youtube_url=None) -> _Session:
        with self._lock:
            if self._closed:
                raise SessionConflict("video service is shutting down")
            if sum(item.status in ACTIVE for item in self._sessions.values()) >= self.settings.max_active_video_sessions:
                raise SessionConflict("active video-session capacity is full")
            reservation = self.settings.max_video_bytes
            if self._reserved_bytes + self._orphan_bytes + reservation > self.settings.video_storage_budget_bytes:
                raise SessionConflict("temporary video storage budget is full")
            session_id = str(uuid.uuid4())
            self.root.mkdir(parents=True, exist_ok=True)
            directory = (self.root / session_id).resolve()
            directory.relative_to(self.root)
            directory.mkdir()
            (directory / MARKER).write_text("parkvision-video-v1\n", encoding="ascii")
            record = _Session(
                id=session_id,
                source_type=source_type,
                directory=directory,
                source_path=directory / "source.mp4",
                reserved_bytes=reservation,
                youtube_video_id=youtube_video_id,
                youtube_url=youtube_url,
                last_playback_update=self.monotonic(),
            )
            self._sessions[session_id] = record
            self._reserved_bytes += reservation
            return record

    def allocate_upload(self) -> tuple[VideoSessionSnapshot, Path]:
        record = self._reserve("upload")
        return self._snapshot(record), record.source_path

    def create_youtube(self, video_id: str, canonical_url: str) -> VideoSessionSnapshot:
        if self.youtube_source is None:
            raise SessionConflict("YouTube acquisition is unavailable")
        record = self._reserve("youtube", video_id, canonical_url)
        self._executor.submit(self._run, record.id)
        return self._snapshot(record)

    def start_upload(self, session_id: str, metadata: VideoMetadata) -> VideoSessionSnapshot:
        with self._lock:
            record = self._record(session_id)
            if record.source_type != "upload" or record.status != "queued":
                raise SessionConflict("upload session cannot be started")
            record.metadata = metadata
            self._touch(record)
            self._executor.submit(self._run, record.id)
            return self._snapshot(record)

    def reject_upload(self, session_id: str) -> None:
        with self._lock:
            record = self._record(session_id)
            record.status = "failed"
            self._touch(record)
            self._reserved_bytes = max(0, self._reserved_bytes - record.reserved_bytes)
            record.reserved_bytes = 0
            self._schedule_cleanup(record)

    def _publish(self, session_id: str, result: FrameResult, preview: bytes):
        with self._lock:
            record = self._record(session_id)
            if record.status not in ACTIVE:
                return
            record.results.append(result)
            if len(record.results) > self.settings.max_video_frames:
                del record.results[:-self.settings.max_video_frames]
            record.preview = preview
            self._touch(record)

    def update_playback(self, session_id: str, current_time: float,
                        playing: bool, ended: bool = False) -> VideoSessionSnapshot:
        with self._playback_condition:
            record = self._record(session_id)
            if record.status not in TERMINAL and not record.playback_ended:
                next_playing = playing and not ended
                changed = (record.playback_playing != next_playing or record.playback_ended != ended)
                record.playback_time = current_time
                record.playback_playing = next_playing
                record.playback_ended = ended
                record.last_playback_update = self.monotonic()
                if changed:
                    self._touch(record)
                self._playback_condition.notify_all()
            return self._snapshot(record)

    def _await_playback(self, session_id: str, timestamp: float,
                        interval: float, skip_stale: bool) -> tuple[str, float]:
        started = self.monotonic()
        with self._playback_condition:
            while True:
                record = self._record(session_id)
                now = self.monotonic()
                if record.cancel_event.is_set():
                    raise VideoProcessingError("cancelled", "Analysis was cancelled")
                if record.playback_ended:
                    return "stop", now - started
                if now - record.last_playback_update >= self.settings.video_session_ttl_seconds:
                    raise VideoProcessingError("playback_timeout", "Playback was inactive for too long")
                if record.playback_playing and now - record.last_playback_update < 3:
                    if skip_stale and timestamp < record.playback_time - interval * 1.5:
                        return "skip", now - started
                    if timestamp <= record.playback_time + 0.1:
                        return "process", now - started
                self._playback_condition.wait(timeout=0.25)

    def _run(self, session_id: str):
        deadline = self.monotonic() + self.settings.video_processing_timeout_seconds
        try:
            with self._lock:
                record = self._record(session_id)
                if record.status == "cancelled":
                    return
                # Bind one model for the whole session so a reload cannot mix
                # versions inside a single analysis result.
                service = self.model_service
                if record.source_type == "youtube":
                    record.status = "acquiring"
                    self._touch(record)
            if record.source_type == "youtube":
                path = self.youtube_source.acquire(
                    record.youtube_url, record.source_path, self.settings,
                    record.cancel_event, deadline,
                )
                if Path(path).resolve() != record.source_path.resolve():
                    raise VideoProcessingError("unsafe_download", "Downloaded video path is invalid")
            with self._lock:
                if record.status == "cancelled":
                    return
                record.status = "processing"
                self._touch(record)
            metadata = self.analyzer(
                record.source_path, service, self.settings,
                record.cancel_event, deadline,
                lambda result, preview: self._publish(session_id, result, preview),
                sample_gate=lambda timestamp, interval, skip_stale: self._await_playback(
                    session_id, timestamp, interval, skip_stale),
            )
            with self._lock:
                if record.status not in TERMINAL:
                    record.metadata = metadata
                    record.status = "completed"
                    self._touch(record)
        except VideoProcessingError as error:
            with self._lock:
                if record.status != "cancelled":
                    record.status = "failed"
                    record.error = SessionError(code=error.code, message=error.public_message)
                    self._touch(record)
        except Exception:  # noqa: BLE001 - sanitize worker failures
            with self._lock:
                if record.status != "cancelled":
                    record.status = "failed"
                    record.error = SessionError(
                        code="video_processing_failed", message="Video analysis failed")
                    self._touch(record)
        finally:
            with self._lock:
                if record.source_type == "youtube" or record.status in {"failed", "cancelled"}:
                    self._schedule_cleanup(record)
                self._reserved_bytes = max(0, self._reserved_bytes - record.reserved_bytes)
                record.reserved_bytes = 0

    def set_model_service(self, service: ModelService) -> None:
        """Bind a reloaded model for sessions started after the reload."""
        with self._lock:
            self.model_service = service

    def get(self, session_id: str) -> VideoSessionSnapshot:
        with self._lock:
            return self._snapshot(self._record(session_id))

    def frames(self, session_id: str, offset: int = 0, limit: int = 100) -> tuple[list[FrameResult], int]:
        with self._lock:
            record = self._record(session_id)
            total = len(record.results)
            return list(record.results[offset:offset + min(limit, 100)]), total

    def preview(self, session_id: str) -> bytes | None:
        with self._lock:
            return self._record(session_id).preview

    def cancel(self, session_id: str) -> VideoSessionSnapshot:
        with self._playback_condition:
            record = self._record(session_id)
            record.cancel_event.set()
            self._playback_condition.notify_all()
            if record.status in ACTIVE:
                if record.status == "queued":
                    self._reserved_bytes = max(0, self._reserved_bytes - record.reserved_bytes)
                    record.reserved_bytes = 0
                record.status = "cancelled"
                self._touch(record)
            self._schedule_cleanup(record)
            return self._snapshot(record)

    @contextmanager
    def lease_source(self, session_id: str):
        with self._lock:
            record = self._record(session_id)
            if record.source_type != "upload" or not record.source_path.is_file():
                raise SessionNotFound(session_id)
            record.leases += 1
            lease = SourceLease(self, session_id, record.source_path, record.source_path.stat().st_size)
        try:
            yield lease
        finally:
            lease.__exit__(None, None, None)

    def _release_lease(self, session_id: str):
        with self._lock:
            record = self._sessions.get(session_id)
            if record is None:
                return
            record.leases = max(0, record.leases - 1)
            if record.pending_cleanup and record.leases == 0:
                self._delete_directory(record)

    def _schedule_cleanup(self, record: _Session):
        record.pending_cleanup = True
        if record.leases == 0:
            self._delete_directory(record)

    def _delete_directory(self, record: _Session):
        if record.directory.exists() and self._owned_directory(record.directory):
            shutil.rmtree(record.directory)
        record.pending_cleanup = False

    def expire(self) -> int:
        expired = 0
        cutoff = self.now().timestamp() - self.settings.video_session_ttl_seconds
        with self._lock:
            for session_id, record in list(self._sessions.items()):
                if record.status in TERMINAL and record.updated_at.timestamp() < cutoff:
                    self._schedule_cleanup(record)
                    if record.leases == 0 and not record.pending_cleanup:
                        del self._sessions[session_id]
                        expired += 1
        return expired

    def shutdown(self):
        with self._playback_condition:
            self._closed = True
            for record in self._sessions.values():
                record.cancel_event.set()
            self._playback_condition.notify_all()
        self._executor.shutdown(wait=True, cancel_futures=True)
        with self._lock:
            for record in self._sessions.values():
                self._schedule_cleanup(record)
