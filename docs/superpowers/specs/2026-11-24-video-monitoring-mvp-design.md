# ParkVision Video Monitoring MVP Design

Date: 2026-11-24

## Goal

Extend the existing FastAPI service so one user can analyze an authorized,
ordinary YouTube video or an uploaded MP4 and watch it with timestamp-matched
parking-space detections drawn over the player. The implementation continues to
use the loaded MLflow champion through `ModelService`; it does not add another
YOLO inference path.

The displayed values remain per-frame estimates. ParkVision does not assign
persistent parking-slot identities, track vehicles, or infer occupancy
transitions between frames.

## Scope

The MVP supports:

- one active video-analysis session by default;
- ordinary public YouTube video URLs and local MP4 uploads;
- timestamp-based sampling at approximately one analyzed frame per second;
- a browser video player with a transparent annotation canvas;
- bounded REST session state, preview images, frame history, and WebSocket
  updates;
- trusted-localhost access enforcement for every video and dashboard route;
- bounded temporary storage and a monotonic wall-clock processing deadline;
- cancellation, expiry, cleanup, and graceful application shutdown.

It excludes IP cameras, RTSP, MJPEG, webcams, arbitrary remote URLs, model
training or registration, alias changes, Docker, production monitoring,
external job queues, persistent session storage, object tracking, and temporal
smoothing.

## User Experience

The dashboard offers a YouTube URL field, an MP4 picker, Start and Stop controls,
the video player, status and progress, model metadata, timestamp, inference
latency, and empty, occupied, and total counts.

The original video plays in the browser. A canvas with `pointer-events: none`
sits over the visible video region and draws the detections for the closest
valid analyzed timestamp. Green means `space-empty`; red means
`space-occupied`. Each box includes its class and confidence.

Playback begins only after a small analysis buffer exists. If the player reaches
the analyzed frontier, JavaScript pauses it and displays "Analysis catching
up." Playback resumes only when the user previously intended it to play and a
buffer is available. A manual pause is never overridden. Failed or cancelled
analysis leaves playback paused and shows the sanitized session error.

The overlay updates at the sampling rate, not at the source frame rate. It is
therefore visually discrete at the default 1 FPS. The dashboard never presents
an old detection as current: it clears the overlay when no analyzed result is
close enough to the playback timestamp. Counts and latency always describe the
timestamp printed beside them.

An overlay is enabled only when the dashboard can establish all of the
following: the player reports the expected source identity and content time,
the analyzed source dimensions are known, the visible content rectangle can be
calculated after letterboxing, and playback has not crossed a seek or rendition
change without receiving a matching analyzed result. Otherwise the canvas is
hidden and the latest annotated-frame preview is shown in a separate panel with
its own source timestamp. Approximate alignment is never labelled or presented
as frame-accurate.

For YouTube, the official IFrame Player API supplies playback and content time.
The canvas is positioned over the player while preserving official controls.
The dashboard uses a fullscreen wrapper containing both iframe and canvas so
annotations remain visible in dashboard-initiated fullscreen. Native iframe
fullscreen is not used for annotated viewing.

Seeking invalidates the current overlay immediately. The dashboard waits for a
result close to the new content time before drawing again and pauses at the
analyzed frontier as usual. Paused playback keeps the last matching overlay but
does not advance counts. A resize, fullscreen transition, device-pixel-ratio
change, or rendition/aspect-ratio change recalculates the content rectangle;
failure to validate it switches to the separate preview.

For an upload, an HTML5 `<video>` element reads the temporary MP4 through a
session-scoped byte-range endpoint. The source is never addressed by a caller
supplied filesystem path.

## Architecture

```text
YouTube URL                 MP4 upload
    |                           |
validated yt-dlp acquisition   streamed size limit + OpenCV validation
    |                           |
temporary bounded video file---+
                |
       in-process session worker
                |
    OpenCV timestamp sampling (~1 FPS)
                |
 temporary frame -> shared ModelService -> delete frame
                |
 validated Prediction + OpenCV annotation
                |
 bounded session state + latest preview JPEG
                |
 REST/WebSocket -> video player + canvas overlay
```

The FastAPI lifespan loads the champion once, creates a `VideoSessionManager`,
and starts one periodic expiry task. On shutdown it cancels sessions, stops the
expiry task, waits for the bounded worker, and removes all owned temporary
files.

The manager uses a fixed `ThreadPoolExecutor`. Its default capacity is one,
matching the default maximum active sessions. Video acquisition, decoding,
annotation, and CPU inference run in this executor and never on the FastAPI
event loop. Every viewer observes the same session state; opening another
dashboard or WebSocket never starts duplicate processing.

Every worker records a monotonic deadline at start. Acquisition, decoding,
inference, and annotation check it between bounded operations. Crossing the
deadline transitions the session to `failed` with error code
`processing_timeout`, signals cancellation, and enters the same cleanup path as
other failures. Network and decoder calls also have finite operation-level
timeouts so the deadline is not defeated by an indefinitely blocked call.

The existing `ModelService` instance is shared. A lock serializes calls because
the loaded PyFunc/Ultralytics object is not assumed to be thread-safe. Image
prediction and video prediction use the same validated `Prediction` schema and
output checks.

## Components

### Existing modules

- `src/api/main.py` retains the image endpoints and wires the video routes,
  static dashboard, manager startup, expiry, and shutdown.
- `src/api/model_service.py` remains the only API-level MLflow inference entry
  point. It adds a small synchronization boundary around `predict`.
- `src/api/config.py` validates video limits in addition to current settings.
- `src/api/schemas.py` adds request, session, frame-result, and event schemas.

### New modules

- `video_service.py` validates media metadata, decodes by timestamp, samples,
  invokes `ModelService`, annotates frames, and reports progress.
- `video_sessions.py` owns session records, the bounded executor, cancellation,
  expiration, snapshots, bounded frame history, and cleanup.
- `youtube_source.py` validates YouTube URLs and performs bounded, best-effort
  acquisition through the `yt-dlp` Python API.
- `static/dashboard.html`, `dashboard.css`, and `dashboard.js` implement the
  dependency-free browser UI and canvas synchronization.

Annotation stays in `video_service.py` unless it makes that module difficult to
read. A separate abstraction or frontend framework is not needed for this MVP.

## Session Model

States are `queued`, `acquiring`, `processing`, `completed`, `failed`, and
`cancelled`. State transitions are monotonic:

```text
queued -> acquiring -> processing -> completed
   |          |             |
   +----------+-------------+-> cancelled
              +-------------+-> failed
```

Uploaded sessions may move directly from `queued` to `processing` because the
HTTP upload has already acquired and validated the file. Terminal sessions do
not return to an active state.

Each record contains:

- opaque UUID session ID and source type;
- source video ID for YouTube, never an extracted media URL;
- state, creation time, last update, and expiry time;
- video duration and source dimensions when known;
- current analyzed timestamp and analyzed frontier;
- frames processed, sampling FPS, and configured frame limit;
- latest counts, inference latency, and model metadata;
- latest frame decode time, MLflow inference time, total per-frame analysis
  time, and total session elapsed time as separate measurements;
- a bounded list of timestamped detection results;
- the latest annotated JPEG only;
- a cancellation event and an internal owned-file path;
- a sanitized error code and user-facing message.

The default maximum of 300 analyzed frames matches five minutes at 1 FPS.
Frame history is capped at the configured maximum and API responses impose a
smaller page limit. The preview replaces the previous JPEG atomically, so image
memory cannot grow with video length.

Each state mutation increments a session revision. The WebSocket checks that
revision at a short bounded interval and sends a new snapshot only when it
changes. This avoids per-subscriber background jobs and unbounded event queues.
Disconnecting a WebSocket has no effect on processing.

Expiry applies to terminal sessions after their last update. Active sessions
are constrained by duration, frame, download, processing-deadline, timeout, and
cancellation limits.
Deleting a session signals cancellation immediately; the worker observes it
between acquisition, decode, and inference operations. A network operation may
take up to its explicit socket timeout before cancellation completes.

## Video Processing

OpenCV opens the temporary video and validates that it can decode a real first
frame. Width and height must be positive and within configured limits. Duration
is derived from available container metadata and checked against the maximum;
processing also enforces timestamps and frame count so inaccurate metadata
cannot bypass the limit.

Frames are decoded sequentially without loading the video into memory. Sampling
uses `CAP_PROP_POS_MSEC` timestamps: a frame is selected when its timestamp
reaches the next target (`0`, `1/sample_fps`, and so on). Non-increasing or
invalid decoder timestamps fall back to a frame-rate-derived timestamp only
when a finite positive source FPS is available. A source with no usable time
base is rejected.

For each sample:

1. Check cancellation and resource limits.
2. Measure decoding time from the previous selected-frame request through the
   decoded sampled frame becoming available.
3. Write one random temporary JPEG in the configured temporary directory.
4. Call the shared `ModelService.predict` under its lock and retain the
   `inference_time_ms` measured around the MLflow call.
5. Delete the temporary JPEG in `finally`.
6. Draw validated boxes, labels, confidence, counts, and timestamp on the frame.
7. Encode one bounded JPEG preview and publish decode, inference, total
   per-frame analysis, and session elapsed times.

Coordinates are retained in source pixels. The event includes source width and
height, allowing the browser to scale boxes into the visible video rectangle
while accounting for `object-fit: contain` letterboxing.

Completing a video removes its temporary source immediately after the worker no
longer needs it, except an uploaded source remains until playback ends, the
session is deleted, or the terminal-session TTL expires. A YouTube download is
never served and is removed when processing terminates.

Raw empty, occupied, and total counts are stored exactly as returned for each
sample. Timing fields and any future smoothed display series are separate
fields. This MVP does not calculate or expose smoothing.

## Temporary Storage and File Leases

All video assets live below a dedicated resolved directory such as
`<temp>/parkvision-video`, never in the repository. Each session owns a random
subdirectory containing only its upload/download, transient frame, preview,
and a non-secret ownership marker. Creation reserves the declared maximum file
size against a process-wide storage budget; actual bytes are checked while
writing. A session is rejected before acquisition when the reservation would
exceed `PARKVISION_VIDEO_STORAGE_BUDGET_BYTES`.

The manager tracks open playback leases. The MP4 source response obtains a
lease before opening the trusted path and releases it in the streaming
generator's `finally` block. Cancellation or expiry marks a leased source for
deletion; physical deletion occurs only after its final reader releases the
lease. This prevents Windows file-sharing failures and truncated active
responses.

Startup scans only direct children of the dedicated root that contain the
ParkVision ownership marker, resolve beneath that root, and are older than the
configured orphan grace period. It removes those orphan directories without
following symlinks. Recent directories are counted against the storage budget
and left for a later cleanup cycle, which avoids deleting files that may belong
to another accidentally overlapping process. The documented MVP supports one
process; deployments must provide a shared lease/store design before enabling
multiple workers or replicas.

Every failure path releases its storage reservation and schedules owned-file
cleanup. Cleanup is idempotent, path-contained, and logs only session IDs and
error categories. It never scans or deletes arbitrary system temporary files.

## YouTube Acquisition and SSRF Boundary

Only `https` ordinary video URLs on `youtube.com`, `www.youtube.com`,
`m.youtube.com`, and `youtu.be` are accepted. The validator extracts exactly one
standard video ID, rejects credentials, non-default ports, fragments used as
playlist controls, playlist/query identifiers, malformed IDs, alternate
services, and local or numeric hosts.

`yt-dlp` is invoked as a library with no shell, cookies, browser sessions,
credentials, or arbitrary user options. It uses `noplaylist`, an explicit
socket timeout, a bounded output template in a manager-owned temporary
directory, a maximum file size, and a format constrained to MP4-compatible
video at or below the configured resolution. Metadata is extracted first and
must identify the YouTube extractor, the requested video ID, an allowed
duration, and a non-live video.

Before download, selected media URLs must use HTTPS and resolve to approved
YouTube/Google video host suffixes. Resolved IP addresses must not be loopback,
private, link-local, multicast, reserved, or unspecified. Redirected extractor
identity and final webpage identity are checked again. These checks reduce the
SSRF surface but do not turn `yt-dlp` into a stable platform API; YouTube
acquisition remains best-effort and may require maintenance.

The acquirer validates every selected format URL before download and rejects
unknown schemes, credentials, ports, host suffixes, and unsafe DNS answers. It
revalidates the selected URL immediately before acquisition and checks the
post-extraction webpage/video identity. `yt-dlp` redirect handling cannot expose
each connection target to application code reliably, so the MVP fails closed
when a destination cannot be prevalidated and documents that DNS pinning across
all internal downloader redirects is not guaranteed. A production deployment
must add an egress proxy/firewall that blocks private and metadata networks;
application checks alone cannot fully prevent DNS rebinding or redirect races.

Progress hooks check cancellation and downloaded byte counts. Unavailable,
private, age/authentication restricted, DRM-protected, oversized, live, and
unsupported videos fail with sanitized errors. The service never attempts to
bypass those restrictions and directs the user to MP4 upload as the fallback.

## HTTP and WebSocket API

### `GET /dashboard`

Returns the static dashboard. Static assets are served from a fixed package
directory.

## Localhost Security Boundary

Video routes, dashboard assets, source playback, previews, session mutation,
frame history, and WebSockets are trusted-localhost-only by default. HTTP
middleware rejects requests unless the direct peer IP is loopback and the Host
header is a loopback literal or `localhost` with the configured port. Forwarded
headers are ignored. State-changing browser requests with an `Origin` header
must have a loopback origin matching the request host. WebSocket handshakes
require an allowed loopback Origin and direct loopback peer before acceptance.

This is an access boundary for a single-user development MVP, not user
authentication. Opaque session IDs are identifiers only and grant no authority.
Binding Uvicorn to `127.0.0.1` remains required and is documented alongside the
server-side checks. A CLI client without an Origin header is allowed only from a
direct loopback peer.

Before any remote, LAN, reverse-proxied, container-published, or multi-user
deployment, the localhost policy must be replaced with authenticated users,
authorization per session, CSRF protection, strict WebSocket origin checks,
TLS, rate limits, and trusted-proxy configuration. Setting
`PARKVISION_LOCALHOST_ONLY=false` is rejected unless an application access
controller is explicitly supplied; the MVP does not provide a public mode.

### `POST /video/youtube`

Accepts JSON `{ "url": "https://..." }`. It validates the URL synchronously,
reserves capacity, creates a queued session, and returns HTTP 202 with its
snapshot. Acquisition and processing continue in the worker.

### `POST /video/upload`

Accepts multipart field `file`. The body is copied in bounded chunks to a
random `.mp4` file while enforcing `PARKVISION_MAX_VIDEO_BYTES`. Content type,
MP4 container signature, decodable first frame, duration, and dimensions are
validated before the session is accepted. It returns HTTP 202. Every rejection
or capacity race deletes the temporary upload.

### `GET /video/sessions/{session_id}`

Returns a session snapshot or 404. It never returns an owned path, extracted
media URL, token, raw exception, or credential.

### `GET /video/sessions/{session_id}/frames`

Accepts `offset` and `limit`; `limit` is capped at 100. It returns timestamped
raw predictions and pagination metadata. It does not return image bytes.

### `GET /video/sessions/{session_id}/preview`

Returns the latest annotated JPEG with `Cache-Control: no-store`, or 404 until
the first result exists.

### `GET /video/sessions/{session_id}/source`

Available only for uploaded MP4 sessions. It supports a single valid byte range
and returns 206 with `Content-Range`, or the complete file with 200. Invalid or
multiple ranges return 416. The route resolves the source exclusively through
the manager's trusted record and stops serving after cleanup.

### `WS /video/sessions/{session_id}/events`

After acceptance, sends the current snapshot and subsequent revisions. Events
contain `session`, `prediction`, `terminal`, or `error` types. The server closes
normally after a terminal event and handles client disconnect without logging
an application error.

### `DELETE /video/sessions/{session_id}`

Signals cancellation and returns HTTP 202 with the cancelled/cancelling
snapshot. Repeating deletion is idempotent for a known terminal session.

The API returns 400 for malformed sources, 404 for unknown sessions or missing
preview/source, 409 when session capacity is full, 413 for oversized uploads,
416 for invalid ranges, 422 for request-schema errors, and 503 when the model or
video subsystem is unavailable.

## Playback Synchronization

The dashboard stores results ordered by source timestamp. On each player time
update or WebSocket prediction:

1. Read the content playback timestamp.
2. Find the latest result at or before that timestamp.
3. Use it only while its age is at most 1.5 sampling intervals.
4. Scale source-pixel boxes into the video's contained display rectangle.
5. Draw boxes and update counts using that same result.

The synchronization state machine has explicit `unverified`, `aligned`, and
`preview-only` modes. Initial load, seek, missing/non-monotonic player time,
unexpected video identity, unknown dimensions, rendition/aspect-ratio change,
or content-rectangle validation failure enters `preview-only`. A new matching
result plus stable player identity/time can return to `aligned`. The separate
preview remains visible in all modes and states its timestamp, so the user can
always distinguish analyzed evidence from the current playback frame.

Before user-initiated playback, the dashboard requires an analyzed frontier of
at least two sampling intervals. While processing is active, playback pauses
when it comes within one sampling interval of the frontier. It resumes after
the two-interval buffer is restored, unless the user manually paused or stopped
the session. When processing completes normally, playback may traverse the
final sampling interval using the last valid result and then clears the overlay.

This mechanism synchronizes by timestamps; it does not promise frame-perfect
tracking between sampled frames. YouTube ads, unavailable embedding, platform
player behavior, and rendition differences may prevent exact visual alignment.
The dashboard reports that limitation rather than implying otherwise.

## Configuration

The existing frozen `Settings` dataclass gains validated fields:

| Variable | Default | Validation |
| --- | ---: | --- |
| `PARKVISION_VIDEO_SAMPLE_FPS` | `1` | finite, `0 < value <= 5` |
| `PARKVISION_MAX_VIDEO_BYTES` | `104857600` | positive integer |
| `PARKVISION_MAX_VIDEO_DURATION_SECONDS` | `300` | positive integer |
| `PARKVISION_MAX_VIDEO_FRAMES` | `300` | positive integer |
| `PARKVISION_MAX_ACTIVE_VIDEO_SESSIONS` | `1` | integer from 1 through 4 |
| `PARKVISION_VIDEO_SESSION_TTL_SECONDS` | `1800` | positive integer |
| `PARKVISION_VIDEO_MAX_WIDTH` | `1280` | positive integer |
| `PARKVISION_VIDEO_MAX_HEIGHT` | `720` | positive integer |
| `PARKVISION_YOUTUBE_ENABLED` | `true` | strict boolean |
| `PARKVISION_YOUTUBE_SOCKET_TIMEOUT_SECONDS` | `20` | positive integer |
| `PARKVISION_VIDEO_PROCESSING_TIMEOUT_SECONDS` | `600` | positive integer greater than the socket timeout |
| `PARKVISION_VIDEO_STORAGE_BUDGET_BYTES` | `209715200` | integer at least as large as maximum video bytes |
| `PARKVISION_VIDEO_ORPHAN_GRACE_SECONDS` | `3600` | positive integer |
| `PARKVISION_LOCALHOST_ONLY` | `true` | must remain true without an injected access controller |
| `PARKVISION_FFMPEG_LOCATION` | unset | optional existing executable or directory passed to yt-dlp |

The system temporary directory remains the default. `.env.example` documents
these non-secret values and contains placeholder MLflow credentials only.

## Dependencies

Add only:

- `opencv-python` for decoding, validation, JPEG encoding, and drawing. The
  existing Ultralytics package declares this exact distribution as a runtime
  dependency, so installing `opencv-python-headless` beside it would create two
  packages owning the same `cv2` namespace;
- `yt-dlp` for best-effort YouTube metadata extraction and bounded download.

The dashboard uses browser-native HTML, CSS, JavaScript, Canvas, WebSocket,
HTML5 video, and the official YouTube IFrame API. No frontend framework,
database, Redis, Celery, Kafka, tracking library, or separate video server is
introduced.

## Testing

All automated tests remain network-free, GPU-free, and independent of DagsHub.
They inject the existing fake `ModelService` and a fake YouTube acquirer.
OpenCV creates tiny deterministic MP4 fixtures when the local codec is
available; decoder boundaries are injected for timestamp edge cases so codec
behavior is not the only test oracle.

Focused coverage includes:

- all existing image endpoints unchanged;
- valid, corrupt, empty, mismatched, and oversized MP4 uploads;
- duration, dimensions, frame count, and timestamp sampling limits;
- output reuse, class colors, coordinate scaling data, and annotated JPEGs;
- separate decode, MLflow inference, per-frame total, and session elapsed timing;
- YouTube host/video-ID acceptance and malicious URL, playlist, redirect,
  DNS-rebinding simulation, private-IP, unsafe extracted-destination,
  authentication-restricted, unavailable, and disabled-source rejection;
- session state transitions, capacity, idempotent cancellation, expiration,
  wall-clock timeout, cancellation during acquisition/inference, file cleanup,
  orphan cleanup containment, storage reservations, playback leases, failure
  recovery, and manager shutdown;
- bounded frame pagination, preview cache headers, and MP4 range responses;
- initial and incremental WebSocket events plus clean disconnection;
- loopback peer, Host, and Origin enforcement across HTTP, media, preview,
  control, dashboard, and WebSocket routes, proving session IDs alone grant no
  access;
- one processing job regardless of viewer count;
- absence of MLflow registry writes and alias mutations;
- dashboard assets and essential accessible controls;
- variable-FPS sampling, inaccurate or non-monotonic metadata timestamps,
  missing timestamps, seeks, paused playback, player/content-time differences,
  letterboxing, resizing, fullscreen, rendition changes, invalidated overlays,
  preview fallback, and the final sampling interval.

Milestones are verified independently: MP4 processing, session management,
YouTube acquisition, then dashboard synchronization. A local MP4 smoke test is
run only when a suitable file is available. A real YouTube test remains an
explicit manual action using a video the user is authorized to process.

## Documentation and Operational Limits

`docs/mlops_phase3_video_mvp.md` will document setup, endpoints, PowerShell
examples, dashboard use, resource settings, timestamps, cleanup, security,
YouTube maintenance risk, single-worker behavior, and future Docker needs. The
README will link to it.

The in-memory manager is intentionally single-process. Multiple Uvicorn workers
would have separate sessions and model instances, so the documented launch
command uses one worker. Restarting the process loses all session state and
cleans files owned by that process when shutdown is graceful. A production
container will need persistent/shared job state, controlled media storage,
request-size enforcement at the proxy, and a deliberate worker model; those
belong to a later phase.

Container compatibility is preserved through a configurable writable temporary
root, byte and wall-clock budgets, one Uvicorn
worker, signal-driven cancellation, and idempotent startup orphan cleanup. The
runtime documents the OpenCV codec support it needs. A later Docker build may
substitute `opencv-python-headless` only if it deliberately installs
Ultralytics without its declared GUI OpenCV dependency; both OpenCV wheel
variants must never coexist. yt-dlp uses progressive
MP4 where available so FFmpeg is optional; when a compatible FFmpeg binary is
needed, its location is configured rather than hardcoded. Docker packaging and
image-level codec installation remain outside this phase.

## Implementation Order

1. MP4 validation, timestamp sampling, shared inference, annotation, and tests.
2. Bounded session manager, REST lifecycle, WebSocket events, cancellation,
   expiry, range playback, and tests.
3. YouTube validation/acquisition and fully mocked tests.
4. Dashboard player/canvas synchronization, regression tests, documentation,
   and final verification.

No implementation step commits, pushes, merges, downloads a real YouTube video,
changes the champion alias, registers a model, or accesses held-out test data.
