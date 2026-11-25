"""Public API response schemas."""

from pydantic import BaseModel, Field


class ModelSummary(BaseModel):
    name: str
    version: str
    architecture: str | None = None


class Counts(BaseModel):
    empty: int
    occupied: int
    total: int


class Detection(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    bbox: list[float]


class Prediction(BaseModel):
    model: ModelSummary
    counts: Counts
    detections: list[Detection]
    inference_time_ms: float


class FrameResult(BaseModel):
    timestamp_seconds: float
    source_width: int
    source_height: int
    model: ModelSummary
    counts: Counts
    detections: list[Detection]
    decode_time_ms: float
    inference_time_ms: float
    analysis_time_ms: float
    session_elapsed_ms: float


class SessionError(BaseModel):
    code: str
    message: str


class VideoSessionSnapshot(BaseModel):
    id: str
    source_type: str
    status: str
    revision: int
    created_at: str
    updated_at: str
    duration_seconds: float | None = None
    source_width: int | None = None
    source_height: int | None = None
    current_timestamp_seconds: float | None = None
    analyzed_frontier_seconds: float | None = None
    frames_processed: int = 0
    sample_fps: float
    latest: FrameResult | None = None
    error: SessionError | None = None
    youtube_video_id: str | None = None


class YouTubeRequest(BaseModel):
    url: str


class PlaybackUpdate(BaseModel):
    current_time_seconds: float = Field(ge=0, allow_inf_nan=False)
    playing: bool
    ended: bool = False
