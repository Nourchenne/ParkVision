"""Serving configuration. Process environment takes precedence over .env."""

import math
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    tracking_uri: str
    model_name: str
    model_alias: str
    max_upload_bytes: int
    temp_dir: Path | None
    log_level: str
    video_sample_fps: float = 1.0
    max_video_bytes: int = 100 * 1024 * 1024
    max_video_duration_seconds: int = 420
    max_video_frames: int = 450
    max_active_video_sessions: int = 1
    video_session_ttl_seconds: int = 1800
    video_max_width: int = 1280
    video_max_height: int = 720
    youtube_enabled: bool = True
    youtube_socket_timeout_seconds: int = 20
    video_processing_timeout_seconds: int = 600
    video_storage_budget_bytes: int = 200 * 1024 * 1024
    video_orphan_grace_seconds: int = 3600
    localhost_only: bool = True
    ffmpeg_location: Path | None = None
    model_cache_dir: Path | None = None
    model_cache_lock_timeout_seconds: int = 120
    mlflow_http_request_timeout: int | None = None
    mlflow_http_request_max_retries: int | None = None
    mlflow_http_request_backoff_factor: int | None = None
    model_reload_enabled: bool = False
    model_reload_min_interval_seconds: int = 60
    model_reload_max_failures: int = 3
    monitoring_enabled: bool = False
    monitoring_log: Path | None = None
    monitoring_max_bytes: int = 5 * 1024 * 1024


def _positive_int(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError as error:
        raise ValueError(f"{name} must be a positive integer") from error
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _positive_float(name: str, default: float) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except ValueError as error:
        raise ValueError(f"{name} must be a positive number") from error
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return value


def _strict_bool(name: str, default: bool) -> bool:
    value = os.getenv(name, str(default)).strip().lower()
    if value not in {"true", "false"}:
        raise ValueError(f"{name} must be true or false")
    return value == "true"


def _int_or_none(name: str, minimum: int) -> int | None:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return None
    try:
        value = int(raw)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer") from error
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def get_settings() -> Settings:
    load_dotenv(ROOT / ".env", override=False)
    size = _positive_int("PARKVISION_MAX_UPLOAD_BYTES", 10 * 1024 * 1024)
    directory = os.getenv("PARKVISION_TEMP_DIR")
    sample_fps = _positive_float("PARKVISION_VIDEO_SAMPLE_FPS", 1.0)
    if sample_fps > 5:
        raise ValueError("PARKVISION_VIDEO_SAMPLE_FPS must not exceed 5")
    max_video_bytes = _positive_int("PARKVISION_MAX_VIDEO_BYTES", 100 * 1024 * 1024)
    max_sessions = _positive_int("PARKVISION_MAX_ACTIVE_VIDEO_SESSIONS", 1)
    if max_sessions > 4:
        raise ValueError("PARKVISION_MAX_ACTIVE_VIDEO_SESSIONS must not exceed 4")
    socket_timeout = _positive_int("PARKVISION_YOUTUBE_SOCKET_TIMEOUT_SECONDS", 20)
    processing_timeout = _positive_int("PARKVISION_VIDEO_PROCESSING_TIMEOUT_SECONDS", 600)
    if processing_timeout <= socket_timeout:
        raise ValueError("video processing timeout must exceed YouTube socket timeout")
    storage_budget = _positive_int("PARKVISION_VIDEO_STORAGE_BUDGET_BYTES", 200 * 1024 * 1024)
    if storage_budget < max_video_bytes:
        raise ValueError("video storage budget must cover at least one maximum-size video")
    ffmpeg = os.getenv("PARKVISION_FFMPEG_LOCATION")
    ffmpeg_path = Path(ffmpeg).resolve() if ffmpeg else None
    if ffmpeg_path is not None and not ffmpeg_path.exists():
        raise ValueError("PARKVISION_FFMPEG_LOCATION must exist")
    cache = os.getenv("PARKVISION_MODEL_CACHE_DIR")
    cache_dir = (Path(cache).expanduser().resolve() if cache and cache.strip()
                 else Path.home() / ".parkvision" / "model-cache")
    monitoring_max_bytes = _positive_int("PARKVISION_MONITORING_MAX_BYTES", 5 * 1024 * 1024)
    if monitoring_max_bytes < 1024:
        raise ValueError("PARKVISION_MONITORING_MAX_BYTES must be at least 1024")
    monitoring_log = os.getenv("PARKVISION_MONITORING_LOG")
    monitoring_path = (Path(monitoring_log).expanduser().resolve() if monitoring_log and monitoring_log.strip()
                       else Path.home() / ".parkvision" / "monitoring" / "predictions.jsonl")
    return Settings(
        tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "https://dagshub.com/Nourchenne/ParkVision.mlflow"),
        model_name=os.getenv("PARKVISION_MODEL_NAME", "ParkVision-Detector"),
        model_alias=os.getenv("PARKVISION_MODEL_ALIAS", "champion"),
        max_upload_bytes=size,
        temp_dir=Path(directory) if directory else None,
        log_level=os.getenv("PARKVISION_LOG_LEVEL", "INFO"),
        video_sample_fps=sample_fps,
        max_video_bytes=max_video_bytes,
        max_video_duration_seconds=_positive_int(
            "PARKVISION_MAX_VIDEO_DURATION_SECONDS", 420),
        max_video_frames=_positive_int("PARKVISION_MAX_VIDEO_FRAMES", 450),
        max_active_video_sessions=max_sessions,
        video_session_ttl_seconds=_positive_int(
            "PARKVISION_VIDEO_SESSION_TTL_SECONDS", 1800),
        video_max_width=_positive_int("PARKVISION_VIDEO_MAX_WIDTH", 1280),
        video_max_height=_positive_int("PARKVISION_VIDEO_MAX_HEIGHT", 720),
        youtube_enabled=_strict_bool("PARKVISION_YOUTUBE_ENABLED", True),
        youtube_socket_timeout_seconds=socket_timeout,
        video_processing_timeout_seconds=processing_timeout,
        video_storage_budget_bytes=storage_budget,
        video_orphan_grace_seconds=_positive_int(
            "PARKVISION_VIDEO_ORPHAN_GRACE_SECONDS", 3600),
        localhost_only=_strict_bool("PARKVISION_LOCALHOST_ONLY", True),
        ffmpeg_location=ffmpeg_path,
        model_cache_dir=cache_dir,
        model_cache_lock_timeout_seconds=_positive_int(
            "PARKVISION_MODEL_CACHE_LOCK_TIMEOUT_SECONDS", 120),
        mlflow_http_request_timeout=_int_or_none("MLFLOW_HTTP_REQUEST_TIMEOUT", 1),
        mlflow_http_request_max_retries=_int_or_none("MLFLOW_HTTP_REQUEST_MAX_RETRIES", 0),
        mlflow_http_request_backoff_factor=_int_or_none("MLFLOW_HTTP_REQUEST_BACKOFF_FACTOR", 0),
        model_reload_enabled=_strict_bool("PARKVISION_MODEL_RELOAD_ENABLED", False),
        model_reload_min_interval_seconds=_positive_int(
            "PARKVISION_MODEL_RELOAD_MIN_INTERVAL_SECONDS", 60),
        model_reload_max_failures=_positive_int(
            "PARKVISION_MODEL_RELOAD_MAX_FAILURES", 3),
        monitoring_enabled=_strict_bool("PARKVISION_MONITORING_ENABLED", False),
        monitoring_log=monitoring_path,
        monitoring_max_bytes=monitoring_max_bytes,
    )
