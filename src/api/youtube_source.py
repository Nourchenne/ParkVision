"""Fail-closed validation and bounded acquisition of ordinary YouTube videos."""

import ipaddress
import math
import re
import socket
from dataclasses import dataclass
from pathlib import Path
from threading import Event
from urllib.parse import parse_qs, urlsplit

from src.api.config import Settings
from src.api.video_service import VideoProcessingError, check_deadline

VIDEO_ID = re.compile(r"[A-Za-z0-9_-]{11}\Z")
PAGE_HOSTS = {"youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be"}
MEDIA_SUFFIXES = (".googlevideo.com", ".youtube.com")


class _YDLLogger:
    """Avoid printing third-party errors that may contain signed media URLs."""

    def debug(self, _message):
        pass

    def warning(self, _message):
        pass

    def error(self, _message):
        pass


@dataclass(frozen=True)
class YouTubeVideoRef:
    video_id: str
    canonical_url: str


def parse_youtube_url(url: str) -> YouTubeVideoRef:
    if not isinstance(url, str) or len(url) > 2048:
        raise VideoProcessingError("invalid_youtube_url", "YouTube URL is invalid")
    try:
        parsed = urlsplit(url)
        port = parsed.port
    except ValueError:
        raise VideoProcessingError("invalid_youtube_url", "YouTube URL is invalid") from None
    host = (parsed.hostname or "").lower().rstrip(".")
    if (parsed.scheme != "https" or host not in PAGE_HOSTS or parsed.username
            or parsed.password or port not in (None, 443) or parsed.fragment):
        raise VideoProcessingError("invalid_youtube_url", "Only ordinary HTTPS YouTube video URLs are accepted")
    query = parse_qs(parsed.query, keep_blank_values=True)
    if "list" in query:
        raise VideoProcessingError("youtube_playlist_unsupported", "YouTube playlists are not supported")
    if host == "youtu.be":
        parts = [part for part in parsed.path.split("/") if part]
        video_id = parts[0] if len(parts) == 1 else ""
    else:
        values = query.get("v", [])
        video_id = values[0] if parsed.path == "/watch" and len(values) == 1 else ""
    if not VIDEO_ID.fullmatch(video_id):
        raise VideoProcessingError("invalid_youtube_url", "YouTube video ID is invalid")
    return YouTubeVideoRef(video_id, f"https://www.youtube.com/watch?v={video_id}")


def _safe_ip(address: str) -> bool:
    value = ipaddress.ip_address(address)
    return not (value.is_private or value.is_loopback or value.is_link_local
                or value.is_multicast or value.is_reserved or value.is_unspecified)


class YouTubeSource:
    def __init__(self, settings: Settings, *, ydl_factory=None, resolver=socket.getaddrinfo):
        self.settings = settings
        self.ydl_factory = ydl_factory
        self.resolver = resolver

    @staticmethod
    def parse(url: str) -> YouTubeVideoRef:
        return parse_youtube_url(url)

    def _destination_addresses(self, url: str) -> frozenset[str]:
        try:
            parsed = urlsplit(url)
            port = parsed.port
        except ValueError:
            raise VideoProcessingError("unsafe_destination", "YouTube media destination is unsafe") from None
        host = (parsed.hostname or "").lower().rstrip(".")
        allowed_host = host in {"youtube.com", "googlevideo.com"} or any(
            host.endswith(suffix) for suffix in MEDIA_SUFFIXES)
        if (parsed.scheme != "https" or not allowed_host or parsed.username
                or parsed.password or port not in (None, 443)):
            raise VideoProcessingError("unsafe_destination", "YouTube media destination is unsafe")
        try:
            answers = self.resolver(host, 443, type=socket.SOCK_STREAM)
            addresses = frozenset(answer[4][0] for answer in answers)
        except (OSError, IndexError, TypeError):
            raise VideoProcessingError("unsafe_destination", "YouTube media destination cannot be verified") from None
        if not addresses or any(not _safe_ip(address) for address in addresses):
            raise VideoProcessingError("unsafe_destination", "YouTube media destination is unsafe")
        return addresses

    def _validate_info(self, info: dict, reference: YouTubeVideoRef) -> dict:
        if not isinstance(info, dict) or info.get("_type") in {"playlist", "multi_video"}:
            raise VideoProcessingError("youtube_playlist_unsupported", "YouTube playlists are not supported")
        if str(info.get("extractor_key", "")).lower() != "youtube" or info.get("id") != reference.video_id:
            raise VideoProcessingError("youtube_identity_mismatch", "YouTube video identity could not be verified")
        try:
            webpage = parse_youtube_url(info["webpage_url"])
        except (KeyError, VideoProcessingError):
            raise VideoProcessingError(
                "youtube_identity_mismatch", "YouTube webpage identity could not be verified") from None
        if webpage.video_id != reference.video_id:
            raise VideoProcessingError("youtube_identity_mismatch", "YouTube webpage identity changed")
        if info.get("is_live") or info.get("live_status") in {"is_live", "is_upcoming", "post_live"}:
            raise VideoProcessingError("youtube_live_unsupported", "Live YouTube videos are not supported")
        availability = info.get("availability")
        if availability not in {"public", "unlisted"} or info.get("has_drm"):
            raise VideoProcessingError("youtube_restricted", "YouTube video requires unsupported access")
        duration = info.get("duration")
        if not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration <= 0:
            raise VideoProcessingError("youtube_metadata_invalid", "YouTube video duration is unavailable")
        if duration > self.settings.max_video_duration_seconds:
            raise VideoProcessingError("video_duration_exceeded", "Video duration exceeds the limit")
        candidates = []
        for item in info.get("formats") or []:
            if (item.get("ext") != "mp4" or item.get("protocol") != "https"
                    or item.get("vcodec") in (None, "none")
                    or item.get("has_drm") or not item.get("url")):
                continue
            width, height = item.get("width"), item.get("height")
            if (_positive_dimension(width) and width > self.settings.video_max_width) or (
                    _positive_dimension(height) and height > self.settings.video_max_height):
                continue
            size = item.get("filesize") or item.get("filesize_approx")
            if _positive_dimension(size) and size > self.settings.max_video_bytes:
                continue
            candidates.append(item)
        if not candidates:
            raise VideoProcessingError("youtube_format_unavailable", "No supported bounded MP4 format is available")
        return max(candidates, key=lambda item: (item.get("height") or 0, item.get("tbr") or 0))

    def acquire(self, canonical_url: str, destination: Path, settings: Settings,
                cancel_event: Event, deadline: float) -> Path:
        if not settings.youtube_enabled:
            raise VideoProcessingError("youtube_disabled", "YouTube analysis is disabled")
        reference = parse_youtube_url(canonical_url)
        check_deadline(cancel_event, deadline)
        factory = self.ydl_factory
        if factory is None:
            from yt_dlp import YoutubeDL
            factory = YoutubeDL
        common = {
            "quiet": True,
            "no_warnings": True,
            "logger": _YDLLogger(),
            "ignoreconfig": True,
            "noplaylist": True,
            "socket_timeout": settings.youtube_socket_timeout_seconds,
            "max_filesize": settings.max_video_bytes,
            "cookiefile": None,
            "cookiesfrombrowser": None,
            "usenetrc": False,
        }
        if settings.ffmpeg_location:
            common["ffmpeg_location"] = str(settings.ffmpeg_location)
        try:
            with factory(common) as extractor:
                info = extractor.extract_info(reference.canonical_url, download=False)
            selected = self._validate_info(info, reference)
            first_addresses = self._destination_addresses(selected["url"])
            check_deadline(cancel_event, deadline)
            if self._destination_addresses(selected["url"]) != first_addresses:
                raise VideoProcessingError("unsafe_destination", "YouTube media destination changed")

            def progress(update):
                check_deadline(cancel_event, deadline)
                downloaded = update.get("downloaded_bytes") or 0
                if downloaded > settings.max_video_bytes:
                    raise VideoProcessingError("video_too_large", "Video exceeds the size limit")

            options = dict(common)
            options.update({
                "format": selected["format_id"],
                "outtmpl": str(destination),
                "overwrites": False,
                "continuedl": False,
                "nopart": False,
                "progress_hooks": [progress],
            })
            with factory(options) as downloader:
                downloaded = downloader.process_ie_result(info, download=True)
            if downloaded.get("id") != reference.video_id:
                raise VideoProcessingError("youtube_identity_mismatch", "Downloaded video identity changed")
        except VideoProcessingError:
            raise
        except Exception as error:  # noqa: BLE001 - third-party downloader errors are sanitized
            if "HTTP Error 403" in str(error) or "403 Forbidden" in str(error):
                raise VideoProcessingError(
                    "youtube_access_denied",
                    "YouTube denied the media download (HTTP 403). Try another authorized video or upload an MP4",
                ) from None
            raise VideoProcessingError("youtube_unavailable", "YouTube video is unavailable or restricted") from None
        if not destination.is_file() or destination.stat().st_size == 0:
            raise VideoProcessingError("youtube_download_failed", "YouTube video download did not complete")
        if destination.stat().st_size > settings.max_video_bytes:
            destination.unlink(missing_ok=True)
            raise VideoProcessingError("video_too_large", "Video exceeds the size limit")
        return destination


def _positive_dimension(value) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value) and value > 0
