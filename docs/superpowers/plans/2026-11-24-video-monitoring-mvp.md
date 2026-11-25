# ParkVision Video Monitoring MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add bounded MP4 and authorized YouTube video analysis with timestamped detections, secure localhost sessions, and a synchronized browser dashboard while preserving the existing image API and MLflow champion path.

**Architecture:** OpenCV decodes timestamped samples into the existing synchronized `ModelService`; an in-process fixed worker owns bounded session state, temporary storage, cancellation, deadlines, and previews. FastAPI exposes localhost-protected REST/WebSocket routes, while native browser players and a canvas overlay render only validated timestamp matches and fall back to a separately timestamped annotated preview.

**Tech Stack:** Python 3.10+, FastAPI, Pydantic, MLflow PyFunc, OpenCV, yt-dlp, browser-native HTML/CSS/JavaScript/Canvas/WebSocket/YouTube IFrame API, pytest, Ruff.

**Spec:** `docs/superpowers/specs/2026-11-24-video-monitoring-mvp-design.md`

## Global Constraints

- Preserve `/health`, `/ready`, `/model-info`, and `/predict` behavior.
- Use the already loaded `ModelService`; never load a model per frame or add Ultralytics inference to the API layer.
- Keep raw per-frame counts; no tracking, transitions, or smoothing.
- Default to one active session, one Uvicorn worker, and trusted localhost only.
- Enforce 100 MiB per video, 200 MiB storage budget, 300-second source duration, 300 analyzed frames, 1280x720 source resolution, 1 sample/second, and a 600-second wall-clock deadline unless environment settings override valid values.
- Reject cameras, RTSP, MJPEG, webcams, arbitrary remote URLs, playlists, live YouTube streams, access-control bypasses, and unsafe destinations.
- No Docker implementation, production monitoring, external queues, DagsHub writes, registration, alias changes, model training, held-out test access, commits, pushes, or merges.
- Automated tests use fake model/YouTube services and no network, DagsHub credentials, GPU, or real YouTube download.
- Each task ends with a local verification checkpoint instead of a commit because the user explicitly prohibited commits.

---

## File Map

- Modify `src/api/config.py`: parse and validate all video, storage, deadline, localhost, and optional FFmpeg settings.
- Modify `src/api/model_service.py`: serialize access to the shared PyFunc model.
- Modify `src/api/schemas.py`: public video request/result/session/event schemas.
- Create `src/api/video_service.py`: MP4 metadata validation, timestamp sampling, inference, annotation, and timings.
- Create `src/api/video_sessions.py`: storage ownership, reservations, leases, worker lifecycle, session state, cancellation, expiration, and cleanup.
- Create `src/api/youtube_source.py`: URL validation, metadata/destination validation, DNS safety checks, and bounded yt-dlp acquisition.
- Create `src/api/video_routes.py`: localhost/Origin checks plus video REST, range-response, and WebSocket handlers.
- Modify `src/api/main.py`: lifespan composition, manager cleanup task, dashboard/static mounting, and route inclusion.
- Create `src/api/static/dashboard.html`, `dashboard.css`, `dashboard.js`: source controls, player, overlay state machine, fallback preview, and metrics.
- Create `tests/test_video_service.py`, `test_video_sessions.py`, `test_youtube_source.py`, `test_video_api.py`, `test_dashboard.py`: focused deterministic coverage.
- Modify `requirements-serving.txt`, `.env.example`, `.github/workflows/ci.yml`, `README.md`.
- Create `docs/mlops_phase3_video_mvp.md`.

---

### Task 1: Video Configuration and Shared Model Synchronization

**Files:**
- Modify: `src/api/config.py`
- Modify: `src/api/model_service.py`
- Modify: `requirements-serving.txt`
- Modify: `.env.example`
- Test: `tests/test_video_service.py`

**Interfaces:**
- Produces: `Settings.video_sample_fps`, `max_video_bytes`, `max_video_duration_seconds`, `max_video_frames`, `max_active_video_sessions`, `video_session_ttl_seconds`, `video_max_width`, `video_max_height`, `youtube_enabled`, `youtube_socket_timeout_seconds`, `video_processing_timeout_seconds`, `video_storage_budget_bytes`, `video_orphan_grace_seconds`, `localhost_only`, and `ffmpeg_location`.
- Produces: thread-safe `ModelService.predict(image_path: Path) -> Prediction`.

- [ ] **Step 1: Add failing settings tests**

Create `tests/test_video_service.py` with environment-isolated tests proving defaults and rejecting zero/negative/non-numeric values, FPS above 5, active sessions above 4, storage below per-video size, processing timeout at or below the YouTube socket timeout, invalid booleans, and nonexistent FFmpeg paths.

```python
def test_video_settings_defaults(monkeypatch):
    clear_video_environment(monkeypatch)
    value = get_settings()
    assert value.video_sample_fps == 1.0
    assert value.max_video_bytes == 104_857_600
    assert value.video_storage_budget_bytes == 209_715_200
    assert value.video_processing_timeout_seconds == 600
    assert value.localhost_only is True


def test_storage_budget_cannot_be_smaller_than_one_video(monkeypatch):
    monkeypatch.setenv("PARKVISION_MAX_VIDEO_BYTES", "100")
    monkeypatch.setenv("PARKVISION_VIDEO_STORAGE_BUDGET_BYTES", "99")
    with pytest.raises(ValueError, match="storage budget"):
        get_settings()
```

- [ ] **Step 2: Run the settings tests and confirm failure**

Run: `venv\Scripts\python.exe -m pytest tests/test_video_service.py -q`

Expected: failures because video settings do not exist.

- [ ] **Step 3: Implement strict parsing and add dependencies**

Use small helpers that reject booleans disguised as integers and non-finite floats:

```python
def positive_int(name: str, default: int) -> int:
    raw = os.getenv(name, str(default))
    value = int(raw)
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def strict_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name, str(default)).strip().lower()
    if raw not in {"true", "false"}:
        raise ValueError(f"{name} must be true or false")
    return raw == "true"
```

Pin the Ultralytics-compatible `opencv-python` distribution and `yt-dlp` in `requirements-serving.txt`; document every non-secret setting in `.env.example`. Document that a future Docker image may deliberately substitute the headless wheel only when it also prevents Ultralytics from installing the GUI wheel, because both distributions own `cv2`.

- [ ] **Step 4: Serialize model prediction and prove it**

Add `self._predict_lock = threading.Lock()` to `ModelService` and wrap the existing predict/parse flow. Test two callers with a blocking fake model and assert its maximum concurrent call count is one while both receive valid predictions.

- [ ] **Step 5: Verify Task 1**

Run:

```powershell
venv\Scripts\python.exe -m pytest tests/test_api.py tests/test_video_service.py -q
ruff check --target-version py310 src/api/config.py src/api/model_service.py tests/test_video_service.py
```

Expected: existing API tests and new configuration/concurrency tests pass.

---

### Task 2: MP4 Validation, Timestamp Sampling, Annotation, and Timings

**Files:**
- Create: `src/api/video_service.py`
- Extend: `src/api/schemas.py`
- Extend test: `tests/test_video_service.py`

**Interfaces:**
- Produces: `VideoMetadata(duration_seconds: float, width: int, height: int, source_fps: float | None)`.
- Produces: `FrameResult(timestamp_seconds, source_width, source_height, counts, detections, decode_time_ms, inference_time_ms, analysis_time_ms, session_elapsed_ms)`.
- Produces: `validate_mp4(path: Path, settings: Settings, capture_factory=cv2.VideoCapture) -> VideoMetadata`.
- Produces: `analyze_video(path, model_service, settings, cancel_event, deadline, publish, capture_factory=cv2.VideoCapture) -> VideoMetadata`.
- Consumes: Task 1 settings and synchronized `ModelService.predict`.

- [ ] **Step 1: Write failing media and sampler tests**

Cover MP4 `ftyp` signature, readable first frame, dimensions, duration, missing time base, frame limit, cancellation, deadline, variable timestamp sequences, non-monotonic/missing timestamps with valid FPS fallback, and final sampling interval. Use an injected fake capture for timestamp cases and a tiny synthetic MP4 only for one integration test.

```python
def test_samples_by_timestamp_and_includes_last_interval(fake_capture, fake_model, settings):
    fake_capture.timestamps_ms = [0, 300, 1000, 1500, 2000, 2990]
    results = collect_analysis(fake_capture, fake_model, settings(video_sample_fps=1))
    assert [item.timestamp_seconds for item in results] == [0.0, 1.0, 2.0]


def test_processing_deadline_fails(monotonic_clock, fake_capture, fake_model, settings):
    monotonic_clock.values = [0.0, 0.1, 601.0]
    with pytest.raises(VideoProcessingError, match="processing_timeout"):
        collect_analysis(fake_capture, fake_model, settings(), clock=monotonic_clock)
```

- [ ] **Step 2: Run focused tests and confirm failure**

Run: `venv\Scripts\python.exe -m pytest tests/test_video_service.py -q`

Expected: import failure for `src.api.video_service`.

- [ ] **Step 3: Implement validation and bounded analysis**

Define typed errors with safe public codes. Validate path containment at callers, MP4 header, byte size, first-frame decode, finite positive dimensions, duration, and limits. Decode sequentially and choose a sample when `timestamp >= next_sample_timestamp`; never retain whole-video frames.

```python
class VideoProcessingError(RuntimeError):
    def __init__(self, code: str, message: str):
        self.code = code
        self.public_message = message
        super().__init__(code)


def deadline_check(cancel_event, deadline, clock=time.monotonic):
    if cancel_event.is_set():
        raise VideoProcessingError("cancelled", "Analysis was cancelled")
    if clock() >= deadline:
        raise VideoProcessingError("processing_timeout", "Analysis exceeded its time limit")
```

Measure decode, PyFunc inference, total frame analysis, and total session elapsed independently with `perf_counter`. Keep counts unchanged.

- [ ] **Step 4: Implement annotation and preview encoding**

Draw green empty boxes, red occupied boxes, class/confidence labels, raw counts, and source timestamp with OpenCV. Validate JPEG encoding succeeds and cap quality at a fixed value. Test decoded pixels near representative box borders and verify result schemas retain source-pixel coordinates.

- [ ] **Step 5: Verify Task 2**

Run:

```powershell
venv\Scripts\python.exe -m pytest tests/test_video_service.py -q
ruff check --target-version py310 src/api/video_service.py src/api/schemas.py tests/test_video_service.py
```

Expected: MP4, timestamp, deadline, cancellation, annotation, and timing tests pass.

---

### Task 3: Bounded Session Manager, Storage, Leases, and Lifecycle

**Files:**
- Create: `src/api/video_sessions.py`
- Extend: `src/api/schemas.py`
- Create: `tests/test_video_sessions.py`

**Interfaces:**
- Produces: `SessionState` enum and immutable public `VideoSessionSnapshot`.
- Produces: `VideoSessionManager.create_upload(path, metadata)`, `create_youtube(video_id, url)`, `get`, `frames`, `preview`, `cancel`, `expire`, `shutdown`, and `lease_source`.
- Produces: context-managed `SourceLease(path, size)` that defers deletion until readers close.
- Consumes: `analyze_video` and `VideoProcessingError` from Task 2.

- [ ] **Step 1: Write failing lifecycle tests**

Test monotonic states, one-job capacity, bounded frame history, revision changes, no duplicate processing per viewer, cancel before/during work, processing timeout mapping, inference failure recovery, terminal TTL, graceful shutdown, idempotent cleanup, and safe error text.

```python
def test_capacity_is_reserved_before_worker_starts(manager, owned_upload):
    first = manager.create_upload(owned_upload())
    with pytest.raises(SessionConflict, match="capacity"):
        manager.create_upload(owned_upload())
    assert manager.get(first.id).status in {"queued", "processing"}


def test_cancelled_session_deletes_after_last_lease(manager, completed_upload):
    session = completed_upload()
    with manager.lease_source(session.id) as lease:
        manager.cancel(session.id)
        assert lease.path.exists()
    assert not lease.path.exists()
```

- [ ] **Step 2: Write failing storage/orphan tests**

Use a permitted test root. Verify reservations reject over-budget sources, actual writes cannot exceed reservations, session paths resolve beneath the dedicated root, symlinks/unknown directories are ignored, old marked orphans are removed, recent marked directories remain charged, and traversal targets are never removed.

- [ ] **Step 3: Implement manager and owned storage**

Use a lock-protected dictionary and fixed `ThreadPoolExecutor`. Session directories use UUID names plus a fixed `.parkvision-video-session` marker. Store internal paths only on private records and convert to Pydantic snapshots at the boundary.

```python
def _transition(record, target):
    if target not in ALLOWED_TRANSITIONS[record.status]:
        raise RuntimeError("invalid session transition")
    record.status = target
    record.revision += 1
    record.updated_at = utc_now()
```

Use a reservation counter under the same lock. Cleanup first marks pending deletion, then removes only resolved owned directories with no leases. Do not follow directory symlinks.

- [ ] **Step 4: Implement worker, expiration, and shutdown**

Set `deadline = monotonic() + processing_timeout` before acquisition. The worker maps typed errors to safe terminal states, publishes bounded results, and releases capacity/reservation in `finally`. `shutdown()` signals every cancellation event and waits for the fixed executor after bounded operations have their own timeouts.

- [ ] **Step 5: Verify Task 3**

Run:

```powershell
venv\Scripts\python.exe -m pytest tests/test_video_sessions.py tests/test_video_service.py -q
ruff check --target-version py310 src/api/video_sessions.py tests/test_video_sessions.py
```

Expected: lifecycle, deadline, lease, storage, orphan, cleanup, and shutdown tests pass.

---

### Task 4: Localhost-Protected Video REST, Range Playback, and WebSockets

**Files:**
- Create: `src/api/video_routes.py`
- Modify: `src/api/main.py`
- Extend: `src/api/schemas.py`
- Create: `tests/test_video_api.py`
- Modify: `tests/test_api.py`

**Interfaces:**
- Produces: `create_video_router(manager, settings) -> APIRouter`.
- Produces: `enforce_local_request(request)` and `allow_local_websocket(websocket)`.
- Consumes: Task 3 manager and source leases.

- [ ] **Step 1: Write failing localhost security tests**

Test direct loopback IPv4/IPv6, allowed `localhost`/loopback Host values, remote peers, spoofed `X-Forwarded-For`, non-loopback Host, mismatched/non-loopback Origin on mutations, absent Origin for loopback CLI requests, and WebSocket peer/Origin rejection. Apply checks to dashboard, assets, session state, frames, preview, source, deletion, and events.

```python
@pytest.mark.parametrize("path", [
    "/dashboard", "/video/sessions/id", "/video/sessions/id/preview",
    "/video/sessions/id/source",
])
def test_remote_peer_cannot_use_session_id(path, remote_client):
    assert remote_client.get(path).status_code == 403
```

- [ ] **Step 2: Write failing upload/API tests**

Cover valid MP4 202, missing/corrupt/mismatched/oversized upload, full capacity, state/frames pagination, no-store preview, cancellation, missing sessions, unavailable model, sanitized errors, and all existing image endpoint tests.

- [ ] **Step 3: Implement router and lifespan composition**

Keep `create_app` dependency injection by accepting optional `video_manager_factory` and `youtube_acquirer_factory`. Create the manager only after the model loads, start an async 30-second expiry loop, and cancel/await it before manager shutdown. If model startup fails, video creation returns 503 while current health/readiness behavior remains unchanged.

- [ ] **Step 4: Implement safe MP4 byte ranges with leases**

Parse only `bytes=start-end`, `bytes=start-`, or `bytes=-suffix`; reject commas and unsatisfiable ranges with 416. Acquire the source lease before stat/open and release it in the generator `finally`.

```python
def stream_range(lease, start, end, chunk_size=1024 * 1024):
    with lease, lease.path.open("rb") as stream:
        stream.seek(start)
        remaining = end - start + 1
        while remaining:
            chunk = stream.read(min(chunk_size, remaining))
            if not chunk:
                break
            remaining -= len(chunk)
            yield chunk
```

- [ ] **Step 5: Implement revision-based WebSocket events**

Validate local peer and Origin before acceptance. Send the initial snapshot, then poll revision with an async bounded delay and emit only changes. Close after a terminal event; treat `WebSocketDisconnect` as normal.

- [ ] **Step 6: Verify Task 4**

Run:

```powershell
venv\Scripts\python.exe -m pytest tests/test_api.py tests/test_video_api.py tests/test_video_sessions.py -q
ruff check --target-version py310 src/api/main.py src/api/video_routes.py tests/test_video_api.py
```

Expected: existing image behavior and all protected video lifecycle routes pass.

---

### Task 5: Fail-Closed YouTube Acquisition

**Files:**
- Create: `src/api/youtube_source.py`
- Modify: `src/api/video_sessions.py`
- Extend: `src/api/video_routes.py`
- Create: `tests/test_youtube_source.py`
- Extend: `tests/test_video_api.py`

**Interfaces:**
- Produces: `parse_youtube_url(url: str) -> YouTubeVideoRef(video_id, canonical_url)`.
- Produces: `YouTubeSource.acquire(ref, destination, limits, cancel_event, deadline) -> Path`.
- Consumes: Task 3 reservations/deadline and Task 4 POST route.

- [ ] **Step 1: Write failing URL tests**

Accept only canonical watch URLs and `youtu.be` links with one valid 11-character video ID. Reject HTTP, credentials, ports, playlist/list parameters, fragments controlling playlists, shorts/live/embed paths, localhost/numeric/alternate domains, duplicate IDs, and malformed encodings.

- [ ] **Step 2: Write failing extraction and destination tests**

Inject fake yt-dlp metadata and DNS resolution. Cover wrong extractor/video identity, playlist/live/private/auth/DRM/oversize/overduration/overresolution metadata, no progressive MP4, unsafe scheme/credentials/port/host, private/link-local/reserved IPv4 and IPv6, DNS answer change between validation passes, unsafe redirected webpage identity, download byte overflow, cancellation, deadline, and sanitized exceptions.

```python
def test_dns_change_fails_closed(fake_dns, source, safe_info):
    fake_dns.answers = [["142.250.1.1"], ["127.0.0.1"]]
    with pytest.raises(YouTubeSourceError, match="unsafe_destination"):
        source.acquire_info(safe_info)
```

- [ ] **Step 3: Implement validator and yt-dlp adapter**

Use `urllib.parse`, `ipaddress`, and `socket.getaddrinfo`. Invoke `yt_dlp.YoutubeDL` directly with fixed options: `noplaylist`, `socket_timeout`, `max_filesize`, progressive MP4 format bounded by height, restricted filename in the owned directory, no cookies, and optional validated `ffmpeg_location`. Validate metadata before calling download and revalidate selected URL/DNS immediately before it.

Progress hooks compare `downloaded_bytes` with the reservation and call the cancellation/deadline guard. Map known yt-dlp failures to safe codes without returning extracted URLs or exception text.

- [ ] **Step 4: Connect YouTube sessions**

`POST /video/youtube` validates synchronously, then reserves capacity/storage and queues acquisition. The worker transitions `queued -> acquiring -> processing`, validates the downloaded MP4 through Task 2, and always deletes it after terminal processing.

- [ ] **Step 5: Verify Task 5**

Run:

```powershell
venv\Scripts\python.exe -m pytest tests/test_youtube_source.py tests/test_video_api.py tests/test_video_sessions.py -q
ruff check --target-version py310 src/api/youtube_source.py tests/test_youtube_source.py
```

Expected: all tests pass without network access or a real YouTube download.

---

### Task 6: Synchronized Dashboard with Preview Fallback

**Files:**
- Create: `src/api/static/dashboard.html`
- Create: `src/api/static/dashboard.css`
- Create: `src/api/static/dashboard.js`
- Modify: `src/api/main.py`
- Create: `tests/test_dashboard.py`

**Interfaces:**
- Consumes: Task 4 REST/WebSocket endpoints and Task 5 YouTube source identity.
- Produces: `GET /dashboard`, fixed static assets, and browser functions `selectFrame`, `contentRect`, `syncOverlay`, `invalidateAlignment`, and `applySessionEvent`.

- [ ] **Step 1: Write failing dashboard contract tests**

Assert the dashboard and assets are local-only, forms have labels, status uses `aria-live`, Stop is a button, preview includes an explicit timestamp, canvas is noninteractive, and JavaScript contains no external framework/CDN except the official YouTube IFrame API URL.

- [ ] **Step 2: Extract and test pure synchronization functions**

Keep timestamp/geometry functions pure and executable under a tiny JavaScript test harness or expose deterministic fixtures checked by Python. Cover letterboxing for wider/taller containers, device-pixel ratio, exact/nearest sample selection, 1.5-interval expiry, seeks, paused time, missing/non-monotonic content time, player/content-time offset, aspect/rendition change, fullscreen resize, and last interval.

```javascript
export function selectFrame(frames, time, interval) {
  const match = frames.findLast(frame => frame.timestamp_seconds <= time);
  return match && time - match.timestamp_seconds <= 1.5 * interval ? match : null;
}
```

- [ ] **Step 3: Build the source/session controls**

Use `fetch` with same-origin URLs. Starting one source resets old player state and opens one WebSocket. Stop sends DELETE, closes the socket, pauses playback, clears the canvas, and retains the terminal status. Reconnection performs one REST snapshot before reopening events.

- [ ] **Step 4: Build players, overlay state machine, and fallback**

For uploads, point the HTML5 player at the protected source endpoint. For YouTube, instantiate the official IFrame API with the validated video ID and official controls. Maintain `unverified`, `aligned`, and `preview-only`; invalidate on seek, identity mismatch, missing time, resize/rendition change, or stale result. Always show the separate annotated preview and its timestamp; hide canvas in preview-only mode.

Pause at one interval from the frontier and resume only after two intervals when `userWantsPlayback` remains true. Never auto-resume a manual pause.

- [ ] **Step 5: Verify Task 6**

Run:

```powershell
venv\Scripts\python.exe -m pytest tests/test_dashboard.py tests/test_video_api.py -q
ruff check --target-version py310 src/api tests/test_dashboard.py
```

Expected: dashboard contract, synchronization fixtures, fallback states, and API integration pass.

---

### Task 7: Documentation, CI, and Full Verification

**Files:**
- Create: `docs/mlops_phase3_video_mvp.md`
- Modify: `README.md`
- Modify: `.github/workflows/ci.yml`
- Review: all files above

**Interfaces:**
- Consumes: completed Tasks 1 through 6.
- Produces: reproducible local setup and CI coverage.

- [ ] **Step 1: Update CI dependencies and lint targets**

Install pinned OpenCV headless and yt-dlp in CI. Add all new API/test paths to the existing Ruff command; keep the full `python -m pytest tests/ -q` command and Python 3.10/3.12 matrix.

- [ ] **Step 2: Write operational documentation**

Document supported sources, localhost binding/enforcement, Origin checks,
single-worker constraint, dashboard use, MP4/YouTube commands, Stop, settings,
deadlines, storage leases/orphan grace, timing semantics, preview fallback,
timestamp/rendition limitations, YouTube/SSRF limits, FFmpeg/OpenCV codec
configuration, graceful shutdown, and later Docker/auth requirements.

- [ ] **Step 3: Run focused and full checks**

Run:

```powershell
venv\Scripts\python.exe -m pytest tests/test_video_service.py tests/test_video_sessions.py tests/test_youtube_source.py tests/test_video_api.py tests/test_dashboard.py -q
venv\Scripts\python.exe -m pytest tests/ -q
ruff check --target-version py310 src/api tests/
python -m compileall -q src tests
git diff --check
git status -sb
```

Expected: all commands pass; Git status shows only local Phase 3 work and no temporary media.

- [ ] **Step 4: Perform an optional local MP4 smoke test**

Only if an appropriate local non-held-out MP4 is already available, start one worker on `127.0.0.1`, submit it, observe processing and preview, seek/play/pause, stop it, and confirm cleanup. Record exact observed behavior. Do not download a YouTube video or claim a YouTube smoke test.

- [ ] **Step 5: Report evidence and limitations**

Report created/modified files, endpoints, sources, synchronization/fallback behavior, exact test counts, whether an MP4 smoke test ran, and remaining limits: sampled overlays, CPU throughput, YouTube platform drift, incomplete application-level DNS pinning, localhost-only access, single process, and no production Docker/auth/monitoring.
