"""Bounded MP4 validation, timestamp sampling, inference, and annotation."""

import math
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from threading import Event

import cv2

from src.api.config import Settings
from src.api.model_service import ModelService
from src.api.monitoring import ROUTE_VIDEO
from src.api.schemas import FrameResult, Prediction


class VideoProcessingError(RuntimeError):
    """Internal video failure with a safe public code and message."""

    def __init__(self, code: str, message: str):
        self.code = code
        self.public_message = message
        super().__init__(code)


@dataclass(frozen=True)
class VideoMetadata:
    duration_seconds: float | None
    width: int
    height: int
    source_fps: float | None


def check_deadline(cancel_event: Event, deadline: float,
                   clock: Callable[[], float] = time.monotonic) -> None:
    if cancel_event.is_set():
        raise VideoProcessingError("cancelled", "Analysis was cancelled")
    if clock() >= deadline:
        raise VideoProcessingError(
            "processing_timeout", "Analysis exceeded its processing time limit")


def _is_finite_positive(value) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value) and value > 0


def _has_mp4_signature(path: Path) -> bool:
    with path.open("rb") as stream:
        header = stream.read(64)
    return len(header) >= 12 and header[4:8] == b"ftyp"


def validate_mp4(path: Path, settings: Settings,
                 capture_factory=cv2.VideoCapture) -> VideoMetadata:
    if not path.is_file() or path.stat().st_size == 0:
        raise VideoProcessingError("invalid_video", "Video is empty or missing")
    if path.stat().st_size > settings.max_video_bytes:
        raise VideoProcessingError("video_too_large", "Video exceeds the size limit")
    if not _has_mp4_signature(path):
        raise VideoProcessingError("invalid_video", "File is not an MP4 video")
    capture = capture_factory(str(path))
    try:
        if not capture.isOpened():
            raise VideoProcessingError("invalid_video", "Video cannot be decoded")
        ok, frame = capture.read()
        if not ok or frame is None or getattr(frame, "ndim", 0) != 3:
            raise VideoProcessingError("invalid_video", "Video has no decodable frame")
        height, width = frame.shape[:2]
        if width <= 0 or height <= 0:
            raise VideoProcessingError("invalid_video", "Video dimensions are invalid")
        if width > settings.video_max_width or height > settings.video_max_height:
            raise VideoProcessingError("video_resolution_exceeded", "Video resolution exceeds the limit")
        fps_value = capture.get(cv2.CAP_PROP_FPS)
        fps = float(fps_value) if _is_finite_positive(fps_value) else None
        count_value = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        duration = (float(count_value) / fps
                    if fps and _is_finite_positive(count_value) else None)
        if duration is not None and duration > settings.max_video_duration_seconds:
            raise VideoProcessingError("video_duration_exceeded", "Video duration exceeds the limit")
        return VideoMetadata(duration, int(width), int(height), fps)
    finally:
        capture.release()


def annotate_frame(frame, prediction: Prediction, timestamp_seconds: float):
    annotated = frame.copy()
    colors = {0: (0, 180, 0), 1: (0, 0, 220)}
    for detection in prediction.detections:
        x1, y1, x2, y2 = (round(value) for value in detection.bbox)
        color = colors[detection.class_id]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = f"{detection.class_name} {detection.confidence:.2f}"
        cv2.putText(annotated, label, (x1, max(15, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    summary = (f"empty {prediction.counts.empty} | occupied {prediction.counts.occupied} | "
               f"total {prediction.counts.total} | {timestamp_seconds:.2f}s")
    cv2.rectangle(annotated, (0, 0), (min(annotated.shape[1], 520), 28), (20, 20, 20), -1)
    cv2.putText(annotated, summary, (8, 19), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return annotated


def _timestamp_seconds(capture, frame_index: int, previous: float | None,
                       source_fps: float | None) -> float:
    milliseconds = capture.get(cv2.CAP_PROP_POS_MSEC)
    timestamp = (float(milliseconds) / 1000.0
                 if isinstance(milliseconds, (int, float)) and math.isfinite(milliseconds)
                 and milliseconds >= 0 else None)
    if timestamp is not None and (previous is None or timestamp > previous):
        return timestamp
    if source_fps:
        fallback = frame_index / source_fps
        if previous is None or fallback > previous:
            return fallback
    raise VideoProcessingError("invalid_timestamps", "Video has no usable timestamp sequence")


def analyze_video(
    path: Path,
    model_service: ModelService,
    settings: Settings,
    cancel_event: Event,
    deadline: float,
    publish: Callable[[FrameResult, bytes], None],
    *,
    capture_factory=cv2.VideoCapture,
    clock: Callable[[], float] = time.monotonic,
    performance_clock: Callable[[], float] = time.perf_counter,
    sample_gate: Callable[[float, float, bool], tuple[str, float]] | None = None,
) -> VideoMetadata:
    metadata = validate_mp4(path, settings, capture_factory)
    capture = capture_factory(str(path))
    if not capture.isOpened():
        capture.release()
        raise VideoProcessingError("invalid_video", "Video cannot be decoded")
    interval = 1.0 / settings.video_sample_fps
    next_sample = 0.0
    previous_timestamp = None
    frame_index = 0
    processed = 0
    active_deadline = deadline
    session_started = performance_clock()
    try:
        while True:
            check_deadline(cancel_event, active_deadline, clock)
            decode_started = performance_clock()
            ok, frame = capture.read()
            decode_ms = (performance_clock() - decode_started) * 1000
            if not ok:
                break
            timestamp = _timestamp_seconds(
                capture, frame_index, previous_timestamp, metadata.source_fps)
            previous_timestamp = timestamp
            frame_index += 1
            if timestamp > settings.max_video_duration_seconds:
                raise VideoProcessingError(
                    "video_duration_exceeded", "Video duration exceeds the limit")
            if timestamp + 1e-9 < next_sample:
                continue
            if sample_gate is not None:
                action, waited = sample_gate(timestamp, interval, True)
                active_deadline += waited
                if action == "stop":
                    return metadata
                if action == "skip":
                    while next_sample <= timestamp + 1e-9:
                        next_sample += interval
                    continue
            check_deadline(cancel_event, active_deadline, clock)
            if processed >= settings.max_video_frames:
                raise VideoProcessingError("frame_limit_exceeded", "Video frame limit was reached")
            analysis_started = performance_clock()
            with tempfile.NamedTemporaryFile(
                dir=settings.temp_dir, suffix=".jpg", delete=False,
            ) as stream:
                frame_path = Path(stream.name)
            try:
                if not cv2.imwrite(str(frame_path), frame):
                    raise VideoProcessingError("frame_encode_failed", "Video frame could not be encoded")
                monitor = getattr(model_service, "monitor", None)
                try:
                    prediction = model_service.predict(frame_path)
                except Exception:
                    if monitor is not None:
                        version = getattr(getattr(model_service, "info", None), "version", None)
                        monitor.record_error(ROUTE_VIDEO, version)
                    raise
                if monitor is not None:
                    monitor.record(ROUTE_VIDEO, prediction)
            finally:
                frame_path.unlink(missing_ok=True)
            annotated = annotate_frame(frame, prediction, timestamp)
            encoded, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not encoded:
                raise VideoProcessingError("preview_encode_failed", "Annotated preview could not be encoded")
            if sample_gate is not None:
                action, waited = sample_gate(timestamp, interval, False)
                active_deadline += waited
                if action == "stop":
                    return metadata
                if action == "skip":
                    while next_sample <= timestamp + 1e-9:
                        next_sample += interval
                    continue
            analysis_ms = (performance_clock() - analysis_started) * 1000
            result = FrameResult(
                timestamp_seconds=round(timestamp, 6),
                source_width=metadata.width,
                source_height=metadata.height,
                model=prediction.model,
                counts=prediction.counts,
                detections=prediction.detections,
                decode_time_ms=round(decode_ms, 3),
                inference_time_ms=prediction.inference_time_ms,
                analysis_time_ms=round(analysis_ms, 3),
                session_elapsed_ms=round((performance_clock() - session_started) * 1000, 3),
            )
            publish(result, bytes(buffer))
            processed += 1
            while next_sample <= timestamp + 1e-9:
                next_sample += interval
        return metadata
    finally:
        capture.release()
