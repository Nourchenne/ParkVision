"""Read-only registry loading and PyFunc output validation."""

import json
import logging
import math
import sys
import tempfile
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from src.api import model_cache
from src.api.config import Settings
from src.api.monitoring import PredictionMonitor
from src.api.schemas import Counts, Detection, ModelSummary, Prediction

logger = logging.getLogger(__name__)
CLASS_NAMES = ("space-empty", "space-occupied")
OUTPUT_COLUMNS = {"image_path", "empty_count", "occupied_count", "detections_json"}


@contextmanager
def _stage(name: str, version: str | None = None) -> Iterator[None]:
    """Log the failing stage without logging exception text or credentials."""
    started = time.perf_counter()
    try:
        yield
    except Exception as exc:
        logger.error("champion_load_failed stage=%s version=%s exc=%s elapsed_ms=%.1f",
                     name, version, type(exc).__name__, (time.perf_counter() - started) * 1000)
        raise


@dataclass(frozen=True)
class ModelInfo:
    name: str
    version: str
    architecture: str | None
    source_run_id: str | None
    model_uri: str
    alias: str
    status: str = "loaded"
    validation_metrics: dict = field(default_factory=dict)


class ModelService:
    def __init__(self, model, info: ModelInfo):
        self.model = model
        self.info = info
        self._predict_lock = threading.Lock()
        self.monitor: PredictionMonitor | None = None  # set by create_app when monitoring is on

    def predict(self, image_path: Path) -> Prediction:
        start = time.perf_counter()
        with self._predict_lock:
            result = self.model.predict(pd.DataFrame({"image_path": [str(image_path)]}))
        duration = (time.perf_counter() - start) * 1000
        with Image.open(image_path) as image:
            image_size = image.size
        return parse_prediction(result, image_path, self.info, duration, image_size)


def parse_prediction(result, image_path: Path, info: ModelInfo, duration: float,
                     image_size: tuple[int, int] | None = None) -> Prediction:
    if not isinstance(result, pd.DataFrame) or len(result) != 1 or set(result.columns) != OUTPUT_COLUMNS:
        raise ValueError("invalid model output columns or row count")
    row = result.iloc[0]
    if row["image_path"] != str(image_path):
        raise ValueError("model output path mismatch")
    counts = []
    for key in ("empty_count", "occupied_count"):
        value = row[key]
        if isinstance(value, (bool, float)) or not isinstance(value, (int, np.integer)) or value < 0:
            raise ValueError("invalid model output count")
        counts.append(int(value))
    try:
        raw = json.loads(row["detections_json"])
    except (TypeError, ValueError) as error:
        raise ValueError("invalid detections JSON") from error
    if not isinstance(raw, list):
        raise ValueError("detections must be a list")  # noqa: TRY004 - invalid model output
    detections = []
    class_counts = [0, 0]
    for item in raw:
        if not isinstance(item, dict) or set(item) != {"class_id", "confidence", "xyxy"}:
            raise ValueError("invalid detection fields")
        class_id = item["class_id"]
        confidence = item["confidence"]
        box = item["xyxy"]
        if type(class_id) is not int or class_id not in (0, 1):
            raise ValueError("invalid detection class")
        if (type(confidence) not in (int, float) or not math.isfinite(confidence)
                or not 0 <= confidence <= 1):
            raise ValueError("invalid detection confidence")
        if (not isinstance(box, list) or len(box) != 4 or
                any(type(v) not in (int, float) or not math.isfinite(v) for v in box) or
                box[0] < 0 or box[1] < 0 or box[2] <= box[0] or box[3] <= box[1] or
                (image_size is not None and (box[2] > image_size[0] or box[3] > image_size[1]))):
            raise ValueError("invalid xyxy bounding box")
        class_counts[class_id] += 1
        detections.append(Detection(class_id=class_id, class_name=CLASS_NAMES[class_id],
                                    confidence=float(confidence), bbox=[float(v) for v in box]))
    if class_counts != counts:
        raise ValueError("model counts disagree with detections")
    if not math.isfinite(duration) or duration < 0:
        raise ValueError("invalid inference duration")
    return Prediction(model=ModelSummary(name=info.name, version=info.version,
                                         architecture=info.architecture),
                      counts=Counts(empty=counts[0], occupied=counts[1], total=sum(counts)),
                      detections=detections, inference_time_ms=round(duration, 3))


def resolve_champion(settings: Settings, client_factory=None):
    """Resolve the champion alias once and return its exact registered version.

    This performs metadata-only registry reads. No alias, tag, or version is written.
    """
    import mlflow
    from mlflow import MlflowClient

    mlflow.set_tracking_uri(settings.tracking_uri)
    mlflow.set_registry_uri(settings.tracking_uri)
    client = (client_factory or MlflowClient)(tracking_uri=settings.tracking_uri,
                                              registry_uri=settings.tracking_uri)
    with _stage("alias_resolve"):
        version = client.get_model_version_by_alias(settings.model_name, settings.model_alias)
    tags = version.tags or {}
    if tags.get("parkvision.inference_verified") != "true":
        raise ValueError("alias target lacks verified inference tag")
    return version


def _obtain_package(settings: Settings, version, identity, uri, artifact_downloader) -> Path:
    """Return a verified local package: cached, or freshly downloaded and published."""
    import mlflow

    downloader = artifact_downloader or mlflow.artifacts.download_artifacts
    if identity is None:
        raise model_cache.CacheVerificationError(
            "missing_provenance", "champion lacks verified cache provenance")
    if settings.model_cache_dir is None:
        raise model_cache.CacheVerificationError(
            "cache_unavailable", "verified model cache is not configured")

    cache_dir = model_cache.cache_root(settings)
    staging = None
    try:
        with model_cache.cache_lock(cache_dir, identity,
                                    settings.model_cache_lock_timeout_seconds):
            entry = model_cache.cache_entry(cache_dir, identity)
            if entry.is_dir():
                try:
                    with _stage("cache_verification", identity.version):
                        model_cache.verify_model_package(entry, identity)
                    logger.info("model_cache_hit version=%s key=%s", identity.version, identity.key())
                    return entry
                except model_cache.CacheVerificationError as error:
                    logger.warning("model_cache_rejected version=%s code=%s",
                                   identity.version, error.code)
                    entry.rename(entry.with_name(
                        f"{entry.name}.invalid-{time.time_ns()}"))
            staging = model_cache.cache_staging(cache_dir)
            with model_cache.mlflow_network_settings(settings), _stage(
                    "artifact_download", identity.version):
                downloaded = Path(downloader(
                    artifact_uri=uri, tracking_uri=settings.tracking_uri,
                    registry_uri=settings.tracking_uri, dst_path=str(staging),
                ))
            root = staging.resolve()
            if downloaded.resolve() != root and root not in downloaded.resolve().parents:
                raise model_cache.CacheVerificationError(
                    "invalid_package", "downloaded package escaped the staging directory")
            with _stage("cache_verification", identity.version):
                model_cache.verify_model_package(downloaded, identity)
            with _stage("cache_publication", identity.version):
                published = model_cache.publish(downloaded, entry, identity)
            logger.info("model_cache_published version=%s key=%s", identity.version, identity.key())
            return published
    finally:
        model_cache.discard_staging(staging)


def load_champion(settings: Settings, client_factory=None, model_loader=None,
                  artifact_downloader=None) -> ModelService:
    """Resolve the champion, obtain a verified package, and load its exact PyFunc."""
    import mlflow

    version = resolve_champion(settings, client_factory)
    tags = version.tags or {}
    version_id = str(version.version)
    uri = f"models:/{settings.model_name}/{version_id}"
    identity = model_cache.identity_from_version(settings.model_name, version)
    local_model = _obtain_package(settings, version, identity, uri, artifact_downloader)

    # Phase 2 packaged `src` inside `code/`, while the pickled class imports
    # `models.registry_adapter`. Add that bundled directory before unpickling.
    bundled_source = local_model / "code" / "src"
    if not (bundled_source / "models" / "registry_adapter.py").is_file():
        raise ValueError("registered model is missing its bundled adapter")
    sys.path.insert(0, str(bundled_source))
    try:
        with _stage("pyfunc_load", version_id):
            model = (model_loader or mlflow.pyfunc.load_model)(str(local_model))
    finally:
        sys.path.remove(str(bundled_source))
    signature = model.metadata.signature
    if (signature is None or signature.inputs is None or len(signature.inputs.inputs) != 1 or
            signature.inputs.inputs[0].name != "image_path" or
            str(signature.inputs.inputs[0].type) != "DataType.string"):
        raise ValueError("registered model has an incompatible input signature")
    validation_metrics: dict[str, float] = {}
    try:
        from mlflow import MlflowClient
        metrics_client = (client_factory or MlflowClient)(tracking_uri=settings.tracking_uri,
                                                           registry_uri=settings.tracking_uri)
        with _stage("validation_metrics_fetch", version_id):
            run_metrics = metrics_client.get_run(version.run_id).data.metrics
        validation_metrics = {
            key[len("best_val/"):]: value
            for key, value in run_metrics.items() if key.startswith("best_val/")
        }
    except Exception:  # noqa: BLE001 - metrics are informational, never block serving
        logger.warning("validation_metrics_unavailable version=%s", version_id)

    info = ModelInfo(name=settings.model_name, version=version_id,
                     architecture=tags.get("parkvision.architecture"),
                     source_run_id=version.run_id, model_uri=uri, alias=settings.model_alias,
                     validation_metrics=validation_metrics)
    service = ModelService(model, info)
    # The registered PyFunc signature declares inputs only; probe its output locally.
    with _stage("inference_probe", version_id):
        with tempfile.NamedTemporaryFile(dir=settings.temp_dir, suffix=".png", delete=False) as stream:
            image_path = Path(stream.name)
        try:
            Image.new("RGB", (64, 64)).save(image_path)
            service.predict(image_path)
        finally:
            image_path.unlink(missing_ok=True)
    return service