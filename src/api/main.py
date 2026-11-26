"""FastAPI entry point: uvicorn src.api.main:app."""

import asyncio
import contextlib
import io
import logging
import tempfile
import time
import warnings
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from PIL import Image, ImageDraw, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool
from starlette.responses import JSONResponse, Response

from src.api.config import Settings, get_settings
from src.api.model_service import CLASS_NAMES, ModelService, load_champion
from src.api.schemas import Prediction
from src.api.video_routes import create_video_router, local_http_allowed
from src.api.video_sessions import VideoSessionManager
from src.api.youtube_source import YouTubeSource

logger = logging.getLogger(__name__)
CHUNK_SIZE = 1024 * 1024


def error(status: int, code: str, message: str) -> HTTPException:
    return HTTPException(status_code=status, detail={"code": code, "message": message})


async def _receive_validated_image(file: UploadFile, settings: Settings) -> Path:
    """Stream an upload to disk and strictly validate it; caller owns the returned path."""
    if file.content_type not in ("image/jpeg", "image/png"):
        raise error(400, "invalid_image", "Only JPEG and PNG images are accepted")
    path = None
    try:
        with tempfile.NamedTemporaryFile(dir=settings.temp_dir, suffix=".upload", delete=False) as temp:
            path = Path(temp.name)
            size = 0
            while chunk := await file.read(CHUNK_SIZE):
                size += len(chunk)
                if size > settings.max_upload_bytes:
                    raise error(413, "upload_too_large", "Image exceeds the upload limit")
                temp.write(chunk)
        if size == 0:
            raise error(400, "invalid_image", "Image is empty")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(path) as image:
                    image.verify()
                    image_format = image.format
                with Image.open(path) as image:
                    image.load()
        except (UnidentifiedImageError, OSError, ValueError,
                Image.DecompressionBombWarning, Image.DecompressionBombError):
            raise error(400, "invalid_image", "Image is corrupt or unsupported") from None
        if image_format not in ("JPEG", "PNG"):
            raise error(400, "invalid_image", "Only JPEG and PNG images are accepted")
        expected = "JPEG" if file.content_type == "image/jpeg" else "PNG"
        if image_format != expected:
            raise error(400, "invalid_image", "Image content does not match its media type")
        new_path = path.with_suffix(".jpg" if image_format == "JPEG" else ".png")
        path.rename(new_path)
        return new_path
    except Exception:
        if path is not None:
            path.unlink(missing_ok=True)
        raise


_DETECTION_COLORS = {"space-empty": (34, 197, 94), "space-occupied": (239, 68, 68)}


def _draw_detections(path: Path, prediction: Prediction) -> bytes:
    """Render bounding boxes and labels over the source image; returns encoded bytes."""
    with Image.open(path) as source:
        image = source.convert("RGB")
    draw = ImageDraw.Draw(image)
    for detection in prediction.detections:
        color = _DETECTION_COLORS.get(detection.class_name, (59, 130, 246))
        x0, y0, x1, y1 = detection.bbox
        draw.rectangle([x0, y0, x1, y1], outline=color, width=3)
        label = f"{detection.class_name} {detection.confidence:.2f}"
        text_y = max(y0 - 14, 0)
        draw.rectangle([x0, text_y, x0 + 7 * len(label), text_y + 14], fill=color)
        draw.text((x0 + 2, text_y + 1), label, fill=(255, 255, 255))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG" if path.suffix == ".jpg" else "PNG")
    return buffer.getvalue()


def create_app(settings: Settings | None = None, loader=load_champion,
               video_manager_factory=VideoSessionManager,
               youtube_source_factory=YouTubeSource,
               local_check=local_http_allowed) -> FastAPI:
    settings = settings or get_settings()
    logging.basicConfig(level=settings.log_level.upper())

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.model_service = None
        app.state.video_manager = None
        cleanup_task = None
        try:
            app.state.model_service = await run_in_threadpool(loader, settings)
            youtube_source = youtube_source_factory(settings) if youtube_source_factory else None
            app.state.video_manager = video_manager_factory(
                app.state.model_service, settings, youtube_source=youtube_source)

            async def expire_sessions():
                while True:
                    await asyncio.sleep(30)
                    await asyncio.to_thread(app.state.video_manager.expire)

            cleanup_task = asyncio.create_task(expire_sessions())
        except Exception as exc:  # noqa: BLE001 - failed startup leaves liveness available
            # Remote errors may include credentials or secret paths; log only the type.
            logger.error("Model startup failed (%s)", type(exc).__name__)
        yield
        if cleanup_task:
            cleanup_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await cleanup_task
        if app.state.video_manager is not None:
            await asyncio.to_thread(app.state.video_manager.shutdown)

    app = FastAPI(title="ParkVision Model Serving", lifespan=lifespan)
    app.state.model_service = None
    app.state.video_manager = None

    protected = ("/video",)

    @app.middleware("http")
    async def enforce_local_video_access(request: Request, call_next):
        if (settings.localhost_only and request.url.path.startswith(protected)
                and not local_check(request)):
            return JSONResponse(status_code=403, content={
                "detail": {"code": "local_access_required",
                           "message": "Video monitoring is available only from localhost"},
            })
        return await call_next(request)

    @app.get("/health")
    def health():
        service = app.state.model_service
        try:
            import torch
            gpu_available = bool(torch.cuda.is_available())
        except Exception:  # noqa: BLE001 - GPU detection must never break liveness
            gpu_available = False
        return {
            "status": "ok",
            "gpu_available": gpu_available,
            "model_version": service.info.version if service is not None else None,
        }

    @app.get("/ready")
    def ready():
        if app.state.model_service is None:
            raise error(503, "model_unavailable", "Model is unavailable")
        return {"status": "ready"}

    @app.get("/model-info")
    def model_info():
        service = app.state.model_service
        if service is None:
            return {"name": settings.model_name, "version": None, "architecture": None,
                    "source_run_id": None, "model_uri": None, "alias": settings.model_alias,
                    "status": "unavailable"}
        info = service.info
        return {"name": info.name, "version": info.version, "architecture": info.architecture,
                "source_run_id": info.source_run_id, "model_uri": info.model_uri,
                "alias": info.alias, "status": info.status}

    @app.get("/model/metadata")
    def model_metadata():
        service = app.state.model_service
        if service is None:
            raise error(503, "model_unavailable", "Model is unavailable")
        info = service.info
        return {
            "name": info.name,
            "version": info.version,
            "architecture": info.architecture,
            "classes": list(CLASS_NAMES),
            "validation_metrics": info.validation_metrics,
        }

    @app.post("/predict", response_model=Prediction)
    async def predict(file: Annotated[UploadFile, File()]):
        service: ModelService | None = app.state.model_service
        if service is None:
            raise error(503, "model_unavailable", "Model is unavailable")
        path = None
        try:
            path = await _receive_validated_image(file, settings)
            try:
                return await run_in_threadpool(service.predict, path)
            except Exception:  # noqa: BLE001 - sanitize all model failures
                logger.error("Model inference failed")
                raise error(500, "inference_failed", "Model inference failed") from None
        finally:
            await file.close()
            if path is not None:
                path.unlink(missing_ok=True)

    @app.post("/predict/visualize")
    async def predict_visualize(file: Annotated[UploadFile, File()]):
        service: ModelService | None = app.state.model_service
        if service is None:
            raise error(503, "model_unavailable", "Model is unavailable")
        path = None
        try:
            path = await _receive_validated_image(file, settings)
            try:
                prediction = await run_in_threadpool(service.predict, path)
            except Exception:  # noqa: BLE001 - sanitize all model failures
                logger.error("Model inference failed")
                raise error(500, "inference_failed", "Model inference failed") from None
            annotated = await run_in_threadpool(_draw_detections, path, prediction)
            media_type = "image/jpeg" if path.suffix == ".jpg" else "image/png"
            return Response(content=annotated, media_type=media_type)
        finally:
            await file.close()
            if path is not None:
                path.unlink(missing_ok=True)

    app.include_router(create_video_router(lambda: app.state.video_manager, settings))

    reload_lock = asyncio.Lock()
    reload_state = {"last": None, "failures": 0}

    @app.post("/internal/model/reload")
    async def reload_model(request: Request):
        """Opt-in, loopback-only bounded retry; never called by inference requests."""
        if not settings.model_reload_enabled:
            raise error(404, "not_found", "Not found")
        if not local_check(request):
            raise error(403, "local_access_required", "Model reload is available only from localhost")
        async with reload_lock:
            now = time.monotonic()
            interval = settings.model_reload_min_interval_seconds
            last = reload_state["last"]
            if ((last is not None and now - last < interval) or
                    reload_state["failures"] >= settings.model_reload_max_failures):
                raise error(429, "reload_throttled", "Model reload is throttled")
            reload_state["last"] = now
            try:
                service = await run_in_threadpool(loader, settings)
            except Exception as exc:  # noqa: BLE001 - never expose remote details
                reload_state["failures"] += 1
                logger.error("Model reload failed (%s)", type(exc).__name__)
                raise error(503, "model_unavailable", "Model is unavailable")
            reload_state["failures"] = 0
            app.state.model_service = service
            manager = app.state.video_manager
            if manager is not None and hasattr(manager, "set_model_service"):
                manager.set_model_service(service)
            return {"status": "ready", "version": service.info.version}

    return app


app = create_app()