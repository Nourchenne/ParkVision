"""FastAPI entry point: uvicorn src.api.main:app."""

import asyncio
import contextlib
import logging
import tempfile
import time
import warnings
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from PIL import Image, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool
from starlette.responses import JSONResponse

from src.api.config import Settings, get_settings
from src.api.model_service import ModelService, load_champion
from src.api.schemas import Prediction
from src.api.video_routes import create_video_router, local_http_allowed
from src.api.video_sessions import VideoSessionManager
from src.api.youtube_source import YouTubeSource

logger = logging.getLogger(__name__)
CHUNK_SIZE = 1024 * 1024


def error(status: int, code: str, message: str) -> HTTPException:
    return HTTPException(status_code=status, detail={"code": code, "message": message})


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
        return {"status": "ok"}

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
        return vars(service.info)

    @app.post("/predict", response_model=Prediction)
    async def predict(file: Annotated[UploadFile, File()]):
        service: ModelService | None = app.state.model_service
        if service is None:
            raise error(503, "model_unavailable", "Model is unavailable")
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
            path = new_path
            try:
                return await run_in_threadpool(service.predict, path)
            except Exception:  # noqa: BLE001 - sanitize all model failures
                logger.error("Model inference failed")
                raise error(500, "inference_failed", "Model inference failed") from None
        finally:
            await file.close()
            if path is not None:
                path.unlink(missing_ok=True)

    app.include_router(create_video_router(lambda: app.state.video_manager, settings))

    reload_lock = asyncio.Lock()
    reload_state = {"last": 0.0, "failures": 0}

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
            if (now - reload_state["last"] < interval or
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
