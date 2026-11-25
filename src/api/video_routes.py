"""Local-only REST and WebSocket routes for bounded video sessions."""

import asyncio
import ipaddress
from typing import Annotated
from urllib.parse import urlsplit

from fastapi import (
    APIRouter,
    File,
    HTTPException,
    Query,
    Request,
    UploadFile,
    WebSocket,
)
from fastapi.responses import Response, StreamingResponse
from starlette.websockets import WebSocketDisconnect

from src.api.config import Settings
from src.api.schemas import PlaybackUpdate, YouTubeRequest
from src.api.video_service import VideoProcessingError, validate_mp4
from src.api.video_sessions import SessionConflict, SessionNotFound

CHUNK_SIZE = 1024 * 1024
TERMINAL = {"completed", "failed", "cancelled"}
DEV_ORIGIN = "http://127.0.0.1:5173"


def api_error(status: int, code: str, message: str) -> HTTPException:
    return HTTPException(status_code=status, detail={"code": code, "message": message})


def _host_name(value: str) -> str | None:
    try:
        return urlsplit(f"//{value}").hostname
    except ValueError:
        return None


def _loopback(value: str | None) -> bool:
    if not value:
        return False
    if value.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(value).is_loopback
    except ValueError:
        return False


def _allowed_origin(origin: str, host_header: str, scheme: str) -> bool:
    """Allow exact same-origin requests and the fixed local Vite proxy origin."""
    try:
        parsed = urlsplit(origin)
        target = urlsplit(f"{scheme}://{host_header}")
        if (parsed.scheme not in {"http", "https"} or parsed.username or parsed.password
                or parsed.path not in {"", "/"} or parsed.query or parsed.fragment):
            return False
        host = parsed.hostname
        if not _loopback(host) or host != target.hostname:
            return False
        default = 443 if parsed.scheme == "https" else 80
        target_default = 443 if target.scheme == "https" else 80
        same_origin = (parsed.scheme == target.scheme
                       and (parsed.port or default) == (target.port or target_default))
        return same_origin or (origin == DEV_ORIGIN and host == "127.0.0.1")
    except ValueError:
        return False
def local_http_allowed(request: Request) -> bool:
    peer = request.client.host if request.client else None
    host_header = request.headers.get("host", "")
    host = _host_name(host_header)
    if not (_loopback(peer) and _loopback(host)):
        return False
    origin = request.headers.get("origin")
    if not origin or request.method in {"GET", "HEAD", "OPTIONS"}:
        return True
    return _allowed_origin(origin, host_header, request.url.scheme)


def local_websocket_allowed(websocket: WebSocket) -> bool:
    peer = websocket.client.host if websocket.client else None
    host_header = websocket.headers.get("host", "")
    host = _host_name(host_header)
    origin = websocket.headers.get("origin")
    scheme = "https" if websocket.url.scheme == "wss" else "http"
    return bool(_loopback(peer) and _loopback(host) and origin
                and _allowed_origin(origin, host_header, scheme))


def _manager(get_manager):
    manager = get_manager()
    if manager is None:
        raise api_error(503, "video_unavailable", "Video analysis is unavailable")
    return manager


def _parse_range(header: str | None, size: int) -> tuple[int, int, bool]:
    if not header:
        return 0, size - 1, False
    if not header.startswith("bytes=") or "," in header:
        raise api_error(416, "invalid_range", "Requested byte range is invalid")
    value = header[6:]
    try:
        start_text, end_text = value.split("-", 1)
        if start_text:
            start = int(start_text)
            end = int(end_text) if end_text else size - 1
        else:
            suffix = int(end_text)
            if suffix <= 0:
                raise ValueError
            start, end = max(0, size - suffix), size - 1
    except (ValueError, TypeError):
        raise api_error(416, "invalid_range", "Requested byte range is invalid") from None
    if start < 0 or end < start or start >= size or end >= size:
        raise api_error(416, "invalid_range", "Requested byte range is unsatisfiable")
    return start, end, True


def create_video_router(get_manager, settings: Settings) -> APIRouter:
    router = APIRouter()

    @router.post("/video/upload", status_code=202)
    async def upload_video(file: Annotated[UploadFile, File()]):
        manager = _manager(get_manager)
        if file.content_type not in {"video/mp4", "application/mp4"}:
            raise api_error(400, "invalid_video", "Only MP4 videos are accepted")
        try:
            snapshot, path = manager.allocate_upload()
        except SessionConflict as error:
            raise api_error(409, "session_capacity", str(error)) from None
        try:
            size = 0
            with path.open("wb") as stream:
                while chunk := await file.read(CHUNK_SIZE):
                    size += len(chunk)
                    if size > settings.max_video_bytes:
                        raise api_error(413, "video_too_large", "Video exceeds the size limit")
                    stream.write(chunk)
            try:
                metadata = await asyncio.to_thread(validate_mp4, path, settings)
            except VideoProcessingError as error:
                status = 413 if error.code == "video_too_large" else 400
                raise api_error(status, error.code, error.public_message) from None
            return manager.start_upload(snapshot.id, metadata)
        except Exception:
            manager.reject_upload(snapshot.id)
            raise
        finally:
            await file.close()

    @router.post("/video/youtube", status_code=202)
    def youtube_video(request: YouTubeRequest):
        manager = _manager(get_manager)
        if not settings.youtube_enabled:
            raise api_error(503, "youtube_disabled", "YouTube analysis is disabled")
        source = getattr(manager, "youtube_source", None)
        if source is None:
            raise api_error(503, "youtube_unavailable", "YouTube analysis is unavailable")
        try:
            reference = source.parse(request.url)
            return manager.create_youtube(reference.video_id, reference.canonical_url)
        except VideoProcessingError as error:
            raise api_error(400, error.code, error.public_message) from None
        except SessionConflict as error:
            raise api_error(409, "session_capacity", str(error)) from None

    @router.get("/video/sessions/{session_id}")
    def session(session_id: str):
        try:
            return _manager(get_manager).get(session_id)
        except SessionNotFound:
            raise api_error(404, "session_not_found", "Video session was not found") from None

    @router.post("/video/sessions/{session_id}/playback")
    def playback(session_id: str, update: PlaybackUpdate):
        try:
            return _manager(get_manager).update_playback(
                session_id, update.current_time_seconds, update.playing, update.ended)
        except SessionNotFound:
            raise api_error(404, "session_not_found", "Video session was not found") from None

    @router.get("/video/sessions/{session_id}/frames")
    def frames(session_id: str, offset: int = Query(0, ge=0),
               limit: int = Query(100, ge=1, le=100)):
        try:
            items, total = _manager(get_manager).frames(session_id, offset, limit)
            return {"items": items, "offset": offset, "limit": limit, "total": total}
        except SessionNotFound:
            raise api_error(404, "session_not_found", "Video session was not found") from None

    @router.get("/video/sessions/{session_id}/preview")
    def preview(session_id: str):
        try:
            content = _manager(get_manager).preview(session_id)
        except SessionNotFound:
            raise api_error(404, "session_not_found", "Video session was not found") from None
        if content is None:
            raise api_error(404, "preview_unavailable", "Annotated preview is not available yet")
        return Response(content, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @router.get("/video/sessions/{session_id}/source")
    def source(session_id: str, request: Request):
        manager = _manager(get_manager)
        try:
            lease_context = manager.lease_source(session_id)
            lease = lease_context.__enter__()
        except SessionNotFound:
            raise api_error(404, "source_unavailable", "Video source is unavailable") from None
        try:
            start, end, partial = _parse_range(request.headers.get("range"), lease.size)
        except Exception:
            lease_context.__exit__(None, None, None)
            raise

        def chunks():
            try:
                with lease.path.open("rb") as stream:
                    stream.seek(start)
                    remaining = end - start + 1
                    while remaining:
                        chunk = stream.read(min(CHUNK_SIZE, remaining))
                        if not chunk:
                            break
                        remaining -= len(chunk)
                        yield chunk
            finally:
                lease_context.__exit__(None, None, None)

        headers = {"Accept-Ranges": "bytes", "Content-Length": str(end - start + 1),
                   "Cache-Control": "no-store"}
        if partial:
            headers["Content-Range"] = f"bytes {start}-{end}/{lease.size}"
        return StreamingResponse(chunks(), status_code=206 if partial else 200,
                                 media_type="video/mp4", headers=headers)

    @router.delete("/video/sessions/{session_id}", status_code=202)
    def cancel(session_id: str):
        try:
            return _manager(get_manager).cancel(session_id)
        except SessionNotFound:
            raise api_error(404, "session_not_found", "Video session was not found") from None

    @router.websocket("/video/sessions/{session_id}/events")
    async def events(websocket: WebSocket, session_id: str):
        if settings.localhost_only and not local_websocket_allowed(websocket):
            await websocket.close(code=1008, reason="Local access required")
            return
        manager = get_manager()
        if manager is None:
            await websocket.close(code=1013, reason="Video analysis unavailable")
            return
        await websocket.accept()
        revision = -1
        try:
            while True:
                try:
                    snapshot = manager.get(session_id)
                except SessionNotFound:
                    await websocket.send_json({"type": "error", "code": "session_not_found"})
                    await websocket.close(code=1008)
                    return
                if snapshot.revision != revision:
                    event_type = "terminal" if snapshot.status in TERMINAL else "session"
                    await websocket.send_json({"type": event_type,
                                               "session": snapshot.model_dump(mode="json")})
                    revision = snapshot.revision
                if snapshot.status in TERMINAL:
                    await websocket.close(code=1000)
                    return
                await asyncio.sleep(0.2)
        except WebSocketDisconnect:
            return

    return router
