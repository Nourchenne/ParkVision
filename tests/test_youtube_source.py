"""Network-free URL, metadata, DNS, and acquisition tests."""

import socket
import threading
import time
from dataclasses import replace

import pytest

from src.api.config import Settings
from src.api.video_service import VideoProcessingError
from src.api.youtube_source import YouTubeSource, parse_youtube_url

VIDEO_ID = "dQw4w9WgXcQ"


def settings(tmp_path, **changes):
    value = Settings(
        "unused", "ParkVision-Detector", "champion", 1024, tmp_path, "WARNING",
        max_video_bytes=1000, video_storage_budget_bytes=2000,
        max_video_duration_seconds=300, video_max_width=1280, video_max_height=720,
        video_processing_timeout_seconds=10, youtube_socket_timeout_seconds=1,
    )
    return replace(value, **changes)


@pytest.mark.parametrize("url", [
    f"https://www.youtube.com/watch?v={VIDEO_ID}",
    f"https://youtube.com/watch?v={VIDEO_ID}&t=4",
    f"https://youtu.be/{VIDEO_ID}",
])
def test_ordinary_youtube_urls_are_canonicalized(url):
    reference = parse_youtube_url(url)
    assert reference.video_id == VIDEO_ID
    assert reference.canonical_url == f"https://www.youtube.com/watch?v={VIDEO_ID}"


@pytest.mark.parametrize("url", [
    f"http://youtube.com/watch?v={VIDEO_ID}",
    f"https://youtube.com/watch?v={VIDEO_ID}&list=PL123",
    f"https://youtube.com/shorts/{VIDEO_ID}",
    f"https://youtube.com/embed/{VIDEO_ID}",
    f"https://attacker.example/watch?v={VIDEO_ID}",
    f"https://youtube.com:444/watch?v={VIDEO_ID}",
    f"https://user:pass@youtube.com/watch?v={VIDEO_ID}",
    f"https://youtube.com/watch?v={VIDEO_ID}#playlist",
])
def test_unsupported_or_malicious_urls_are_rejected(url):
    with pytest.raises(VideoProcessingError):
        parse_youtube_url(url)


def safe_info(**changes):
    value = {
        "id": VIDEO_ID,
        "extractor_key": "Youtube",
        "webpage_url": f"https://www.youtube.com/watch?v={VIDEO_ID}",
        "duration": 60,
        "availability": "public",
        "is_live": False,
        "formats": [{
            "format_id": "18", "ext": "mp4", "vcodec": "avc1", "acodec": "mp4a",
            "protocol": "https",
            "width": 640, "height": 360, "filesize": 500,
            "url": "https://rr1.googlevideo.com/videoplayback",
        }],
    }
    value.update(changes)
    return value


def safe_dns(host, port, type=0):
    return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("142.250.1.1", port))]


class FakeYDL:
    def __init__(self, options, info, destination):
        self.options = options
        self.info = info
        self.destination = destination

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def extract_info(self, url, download=False):
        assert download is False
        return self.info

    def process_ie_result(self, info, download=True):
        assert download is True
        self.destination.write_bytes(b"x" * 500)
        for hook in self.options.get("progress_hooks", []):
            hook({"downloaded_bytes": 500})
        return info


def test_youtube_media_403_is_reported_without_exposing_download_url(tmp_path):
    destination = tmp_path / "source.mp4"
    info = safe_info()

    class DeniedYDL(FakeYDL):
        def process_ie_result(self, info, download=True):
            raise RuntimeError("HTTP Error 403: Forbidden https://signed.example/token=private")

    source = YouTubeSource(settings(tmp_path),
                           ydl_factory=lambda options: DeniedYDL(options, info, destination),
                           resolver=safe_dns)
    with pytest.raises(VideoProcessingError) as error:
        source.acquire(f"https://www.youtube.com/watch?v={VIDEO_ID}", destination,
                       settings(tmp_path), threading.Event(), time.monotonic() + 10)
    assert error.value.code == "youtube_access_denied"
    assert "private" not in error.value.public_message


def test_bounded_acquisition_uses_validated_format(tmp_path):
    destination = tmp_path / "source.mp4"
    info = safe_info()
    calls = []

    def factory(options):
        calls.append(options)
        return FakeYDL(options, info, destination)

    source = YouTubeSource(settings(tmp_path), ydl_factory=factory, resolver=safe_dns)
    result = source.acquire(f"https://www.youtube.com/watch?v={VIDEO_ID}", destination,
                            settings(tmp_path), threading.Event(), time.monotonic() + 10)
    assert result == destination
    assert calls[-1]["format"] == "18"
    assert calls[-1]["noplaylist"] is True
    assert calls[-1]["cookiesfrombrowser"] is None


def test_direct_mp4_is_selected_over_same_resolution_hls(tmp_path):
    direct = safe_info()["formats"][0]
    hls = {**direct, "format_id": "232", "protocol": "m3u8_native",
           "filesize": None}
    info = safe_info(formats=[direct, hls])
    source = YouTubeSource(settings(tmp_path), resolver=safe_dns)
    selected = source._validate_info(info, parse_youtube_url(
        f"https://www.youtube.com/watch?v={VIDEO_ID}"))
    assert selected["format_id"] == "18"


def test_hls_only_metadata_is_rejected_for_mp4_processing(tmp_path):
    hls = {**safe_info()["formats"][0], "protocol": "m3u8_native"}
    source = YouTubeSource(settings(tmp_path), resolver=safe_dns)
    with pytest.raises(VideoProcessingError) as error:
        source._validate_info(safe_info(formats=[hls]), parse_youtube_url(
            f"https://www.youtube.com/watch?v={VIDEO_ID}"))
    assert error.value.code == "youtube_format_unavailable"


@pytest.mark.parametrize("address", ["127.0.0.1", "10.0.0.2", "169.254.1.2", "::1"])
def test_private_destinations_fail_closed(tmp_path, address):
    resolver = lambda host, port, type=0: [
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, port))]
    source = YouTubeSource(settings(tmp_path), ydl_factory=lambda _: None, resolver=resolver)
    with pytest.raises(VideoProcessingError, match="unsafe_destination"):
        source._destination_addresses("https://rr1.googlevideo.com/videoplayback")


def test_dns_answer_change_fails_closed(tmp_path):
    answers = iter(["142.250.1.1", "142.250.1.2"])

    def resolver(host, port, type=0):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (next(answers), port))]

    destination = tmp_path / "source.mp4"
    info = safe_info()
    source = YouTubeSource(
        settings(tmp_path),
        ydl_factory=lambda options: FakeYDL(options, info, destination), resolver=resolver)
    with pytest.raises(VideoProcessingError, match="unsafe_destination"):
        source.acquire(f"https://youtube.com/watch?v={VIDEO_ID}", destination,
                       settings(tmp_path), threading.Event(), time.monotonic() + 10)


@pytest.mark.parametrize("changes", [
    {"extractor_key": "Generic"},
    {"id": "aaaaaaaaaaa"},
    {"is_live": True},
    {"availability": "private"},
    {"availability": "needs_auth"},
    {"webpage_url": "https://www.youtube.com/watch?v=aaaaaaaaaaa"},
    {"duration": 301},
    {"formats": []},
])
def test_restricted_or_unsupported_metadata_is_rejected(tmp_path, changes):
    source = YouTubeSource(settings(tmp_path), resolver=safe_dns)
    with pytest.raises(VideoProcessingError):
        source._validate_info(safe_info(**changes), parse_youtube_url(
            f"https://youtube.com/watch?v={VIDEO_ID}"))
