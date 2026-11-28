"""Opt-in, file-based prediction monitoring.

One JSON object per prediction is appended to a local JSONL file: route, model version,
counts, confidence summary and latency. The image itself, its name and its path are never
stored. The file is bounded: when it reaches the size limit it is rotated to "<name>.1"
(replacing any previous rotation), so the total never exceeds twice the limit.

Monitoring is off by default (PARKVISION_MONITORING_ENABLED=false) and a failure to write
never affects an inference.
"""

import json
import logging
import os
import statistics
import threading
from datetime import datetime, timezone
from pathlib import Path

from src.api.config import Settings
from src.api.schemas import Prediction

logger = logging.getLogger(__name__)

ROUTE_PREDICT = "/predict"
ROUTE_VISUALIZE = "/predict/visualize"
ROUTE_VIDEO = "video_frame"


class PredictionMonitor:
    def __init__(self, path: Path, max_bytes: int):
        self.path = path
        self.max_bytes = max_bytes
        self._lock = threading.Lock()
        self._warned = False

    @classmethod
    def from_settings(cls, settings: Settings) -> "PredictionMonitor | None":
        if not settings.monitoring_enabled or settings.monitoring_log is None:
            return None
        return cls(settings.monitoring_log, settings.monitoring_max_bytes)

    def record(self, route: str, prediction: Prediction) -> None:
        confidences = [d.confidence for d in prediction.detections]
        self._append({
            "ts": _now(), "route": route, "model_version": prediction.model.version,
            "error": False,
            "empty": prediction.counts.empty, "occupied": prediction.counts.occupied,
            "total": prediction.counts.total,
            "conf_mean": round(statistics.fmean(confidences), 4) if confidences else None,
            "conf_min": round(min(confidences), 4) if confidences else None,
            "inference_ms": prediction.inference_time_ms,
        })

    def record_error(self, route: str, model_version: str | None) -> None:
        self._append({
            "ts": _now(), "route": route, "model_version": model_version, "error": True,
            "empty": None, "occupied": None, "total": None,
            "conf_mean": None, "conf_min": None, "inference_ms": None,
        })

    def _append(self, entry: dict) -> None:
        try:
            line = json.dumps(entry, separators=(",", ":")) + "\n"
            with self._lock:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                size = self.path.stat().st_size if self.path.exists() else 0
                if size > 0 and size + len(line.encode("utf-8")) > self.max_bytes:
                    os.replace(self.path, self.path.with_name(self.path.name + ".1"))
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(line)
        except Exception as exc:  # noqa: BLE001 - monitoring must never break inference
            if not self._warned:
                self._warned = True
                logger.warning("monitoring_write_failed exc=%s", type(exc).__name__)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")
