# Phase 3 video monitoring MVP

## What it supports

ParkVision can analyze one bounded video session at a time from either:

- an uploaded MP4 file; or
- an ordinary authorized public or unlisted YouTube video URL.

The same MLflow champion loaded by the image API analyzes every sampled frame.
It is resolved from the registry alias at startup and may come from the verified
local model cache described in the [serving guide](mlops_phase3_fastapi.md), so
video analysis shares one loaded model with the image endpoints. The video
layer does not call Ultralytics directly, download a model per frame, change the
champion alias, or create MLflow runs. A model-loading failure leaves the video
manager unavailable as well; `/ready` distinguishes that case from a video-only
problem. An opt-in local model reload rebinds new sessions while an in-progress
session keeps the model version it started with (see the serving guide).
The champion must carry verified provenance tags; without them, startup refuses
an unverified package instead of downloading it.

Open `http://127.0.0.1:5173/` after starting the React frontend to submit a
source, watch playback, see timestamped boxes and counts, and stop, reset, or
reconnect the session. The frontend remembers the session ID in browser session storage, so refreshing
the same tab reconnects while the server process and session remain available.
WebSocket updates fall back to polling when the connection fails. Empty spaces are green and
occupied spaces are red. Results are per-frame estimates sampled at 1 FPS by
default. They are not persistent parking-slot IDs, vehicle tracking, or
occupancy transitions.

## Installation and startup

The registered model was packaged with Python 3.10.11. Use the project virtual
environment or create a Python 3.10 environment:

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements-serving.txt
.\venv\Scripts\python.exe -m uvicorn src.api.main:app `
  --host 127.0.0.1 --port 8000 --workers 1
```

In a second PowerShell terminal:

```powershell
cd frontend
npm install
npm run dev
```

Start FastAPI first and wait for `http://127.0.0.1:8000/health` to respond before
opening the frontend. If Vite starts while port 8000 is closed, its proxy prints
`ECONNREFUSED`. The frontend keeps a saved session ID and checks health every
15 seconds; it restores the session after the API becomes reachable.

Vite listens on `127.0.0.1:5173` and proxies `/health`, `/ready`, `/model-info`,
`/predict`, and `/video` (including WebSockets and MP4 byte ranges) to
`127.0.0.1:8000`. The backend sees a loopback peer and Host; the browser Origin
is also loopback, so the existing local access boundary remains enabled. Use
the Vite URL for the frontend; FastAPI no longer serves `/dashboard` or `/static`.

One worker is required. Sessions, leases, and model synchronization are
in-process. Multiple workers would create unrelated session managers and model
instances.

`opencv-python` is pinned because Ultralytics declares that distribution as a
dependency. Do not install `opencv-python-headless` beside it: both own the
`cv2` namespace. A future Docker image may substitute the headless wheel only
if it deliberately controls Ultralytics dependency installation. Codec support
depends on the OpenCV wheel and source encoding. Progressive MP4 is preferred;
set `PARKVISION_FFMPEG_LOCATION` only when a compatible local FFmpeg is needed.
`wsproto` is required for Uvicorn's WebSocket upgrade. After changing serving
dependencies, reinstall them in the same `venv` used to start FastAPI and
restart the server; otherwise Vite will report WebSocket proxy errors and the
frontend will use HTTP polling.

## Local access boundary

Every `/video` route is localhost-only. The server checks
the direct peer address and Host header. Browser mutations also require a
matching loopback Origin, and WebSockets require a loopback peer, Host, and
Origin. Forwarded headers are ignored. Session IDs identify records; they are
not authentication credentials.

Keep Uvicorn bound to `127.0.0.1`. Before LAN, public, reverse-proxied, or
multi-user deployment, add authenticated users, per-session authorization,
CSRF protection, TLS, rate limits, strict WebSocket origins, and trusted-proxy
configuration. The MVP deliberately has no public mode.

## API

| Endpoint | Purpose |
| --- | --- |
| `POST /video/upload` | Create a session from multipart field `file` |
| `POST /video/youtube` | Create a session from JSON field `url` |
| `GET /video/sessions/{id}` | Current state and latest estimate |
| `GET /video/sessions/{id}/frames` | Bounded timestamped results (`offset`, `limit<=100`) |
| `GET /video/sessions/{id}/preview` | Latest annotated JPEG (`no-store`) |
| `GET /video/sessions/{id}/source` | Uploaded MP4 playback with one byte range |
| `POST /video/sessions/{id}/playback` | Report player time and play/pause/end state |
| `WS /video/sessions/{id}/events` | Revision-based status and prediction updates |
| `DELETE /video/sessions/{id}` | Cancel and release resources |

Upload an MP4:

```powershell
$response = curl.exe -sS -X POST http://127.0.0.1:8000/video/upload `
  -F "file=@C:\path\parking.mp4;type=video/mp4" | ConvertFrom-Json
$response.id
```

Video sessions wait for a player signal before inference. The React dashboard
sends these automatically. For an API-only check, send a playback position:

```powershell
$playback = @{ current_time_seconds = 1.0; playing = $true } | ConvertTo-Json
Invoke-RestMethod -Method Post `
  -Uri "http://127.0.0.1:8000/video/sessions/$($response.id)/playback" `
  -ContentType application/json -Body $playback
```

Submit an authorized YouTube video:

```powershell
$body = @{ url = "https://www.youtube.com/watch?v=VIDEO_ID" } | ConvertTo-Json
$response = Invoke-RestMethod -Method Post `
  -Uri http://127.0.0.1:8000/video/youtube `
  -ContentType application/json -Body $body
$response.id
```

Read or stop it:

```powershell
Invoke-RestMethod "http://127.0.0.1:8000/video/sessions/$($response.id)"
Invoke-RestMethod -Method Delete `
  "http://127.0.0.1:8000/video/sessions/$($response.id)"
```

States are `queued`, `acquiring`, `processing`, `completed`, `failed`, and
`cancelled`. A failed snapshot contains a sanitized error code/message.
The playback endpoint accepts JSON fields `current_time_seconds` (nonnegative
number), `playing` (boolean), and optional `ended` (boolean).

## Playback and timestamp semantics

Uploaded MP4 uses the browser video element. YouTube uses a privacy-enhanced
embed controlled through the YouTube IFrame Player API. Processing waits for
playback, samples one source frame per video second by default, pauses new
inference while playback is paused, and completes when playback ends. A frame
already in inference may finish after pause, but its result is held until
playback resumes. The browser reports playback position every half second
while playing. If playback updates stop, inference pauses after three seconds;
an inactive session eventually fails after the session TTL. Processing time
spent waiting for playback does not count toward the processing deadline.

For uploaded MP4, a transparent canvas scales source-pixel `xyxy`
boxes over the contained video rectangle. YouTube displays the separately
timestamped annotated preview because downloaded media and browser playback
may have different clocks or renditions.

The UI clears the MP4 overlay when the player is seeking, video dimensions are
unknown or differ from analysis, or no timestamped sample is close enough. The
separately timestamped annotated preview remains visible. The overlay does not
claim frame-accurate tracking. YouTube embedding restrictions may prevent
playback even when acquisition succeeds.

Each frame result keeps raw counts plus four separate measurements:

- frame decode time;
- MLflow inference time;
- total frame analysis time, including image/annotation work; and
- elapsed session processing time.

No smoothing is calculated.

## Limits and temporary storage

| Variable | Default |
| --- | ---: |
| `PARKVISION_VIDEO_SAMPLE_FPS` | `1` |
| `PARKVISION_MAX_VIDEO_BYTES` | `104857600` |
| `PARKVISION_MAX_VIDEO_DURATION_SECONDS` | `420` |
| `PARKVISION_MAX_VIDEO_FRAMES` | `450` |
| `PARKVISION_MAX_ACTIVE_VIDEO_SESSIONS` | `1` |
| `PARKVISION_VIDEO_SESSION_TTL_SECONDS` | `1800` |
| `PARKVISION_VIDEO_MAX_WIDTH` | `1280` |
| `PARKVISION_VIDEO_MAX_HEIGHT` | `720` |
| `PARKVISION_YOUTUBE_ENABLED` | `true` |
| `PARKVISION_YOUTUBE_SOCKET_TIMEOUT_SECONDS` | `20` |
| `PARKVISION_VIDEO_PROCESSING_TIMEOUT_SECONDS` | `600` |
| `PARKVISION_VIDEO_STORAGE_BUDGET_BYTES` | `209715200` |
| `PARKVISION_VIDEO_ORPHAN_GRACE_SECONDS` | `3600` |
| `PARKVISION_LOCALHOST_ONLY` | `true` |

Video assets live below a dedicated `parkvision-video` directory inside the
configured/system temporary directory. Each active session reserves the
maximum source size against the total budget. Uploaded playback obtains a file
lease, so cancellation or expiration does not delete a file while a response is
reading it. Cleanup is idempotent and limited to marked session directories.
Old marked orphan directories are removed after the grace period; recent ones
remain charged against the budget.

## YouTube safety and limitations

Only ordinary HTTPS `youtube.com` watch URLs and `youtu.be` URLs are accepted.
Playlists, live streams, private/authentication-restricted videos, DRM formats,
cookies, browser sessions, credentials, unsupported hosts, and arbitrary
extractors are rejected. Extractor identity, video ID, webpage identity,
duration, dimensions, size, MP4 format, media host, and resolved public IPs are
validated before download and DNS is checked again immediately before it.
Only direct HTTPS MP4 video formats are selected. HLS manifests labeled `mp4`
are excluded because they are not direct MP4 files; the supplied test video
advertises both at 720p.

yt-dlp does not expose reliable application control over every internal redirect
and connection target. Application checks cannot fully pin DNS across redirect
races. A production environment must enforce egress firewall/proxy rules that
block private, link-local, and cloud metadata networks. YouTube ingestion is
best-effort and may need maintenance as the platform changes. Use MP4 upload
when acquisition or embedding is unavailable. No automated or manual real
YouTube download is performed by the test suite.
An HTTP 403 from YouTube's media server is reported as
`youtube_access_denied`. It is an upstream refusal, not an API or WebSocket
failure. YouTube may require a proof-of-origin token for some formats (see
[yt-dlp's PO Token Guide](https://github.com/yt-dlp/yt-dlp/wiki/PO-Token-Guide)); this
MVP does not use cookies, browser sessions, or token-provider plugins. Try
another authorized public video or upload an MP4. Do not assume every public
YouTube video is downloadable.

## Tests

```powershell
.\venv\Scripts\python.exe -m pytest tests/ -q
.\venv\Scripts\python.exe -m ruff check src/api tests/
.\venv\Scripts\python.exe -m pip check
git diff --check
cd frontend
npm run build
npm run lint
npm run test
```

Tests create tiny synthetic MP4s and fake model/YouTube services. They require
no DagsHub credentials, network, GPU, or real checkpoint download.

## Manual browser check

1. With both servers running, open `http://127.0.0.1:5173/` and check API/model
   status, Model Information, and Diagnostics.
2. Upload an authorized local MP4 and press Play. Check that one new prediction
   appears per video second, pause and resume the video, then let it end. Check
   progress, boxes, preview, counts, and timestamps. Refresh during analysis;
   the same ID should return and wait for playback to resume.
3. Stop and reset, then reconnect using a known session ID. Try an invalid MP4
   and confirm the backend error is shown.
4. Submit an authorized public or unlisted YouTube URL manually. Check
   acquiring/processing/completed or the reported failure, and compare the
   separate annotated preview with source playback. Automated tests do not
   perform real YouTube acquisition.

If the UI says the API is offline, check that FastAPI is listening on port 8000
and Vite on 5173. If the model is unavailable, check FastAPI startup logs and
the verified local cache; the frontend does not trigger model downloads. If
source playback fails, try an MP4 codec supported by the browser and OpenCV.
If FastAPI logs `No supported WebSocket library detected`, run
`.\venv\Scripts\python.exe -m pip install -r requirements-serving.txt` and
restart Uvicorn. The `wsproto` dependency supplies the missing transport.

## Docker and production follow-up

This implementation is container-compatible in structure: one worker, a
configurable writable temp directory, bounded bytes/time/frames, graceful
shutdown, and startup orphan cleanup. It does not add a Dockerfile. A later
Docker phase must choose codecs/FFmpeg explicitly, mount a persistent verified
model-cache volume, configure a writable temp volume, enforce request limits at
the proxy, and replace in-memory sessions and localhost access with production
storage, workers, authentication, and network egress controls. Keep the model
reload endpoint disabled unless an authenticated admin channel is added.
