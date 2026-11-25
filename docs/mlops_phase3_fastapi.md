# Phase 3: FastAPI serving and verified model cache

This is the local MVP for image inference, bounded video analysis, and the
React frontend. It performs **no registry writes**: the `champion` alias is
read, resolved to an exact registered version, and that version alone is loaded.

## Architecture

- `src/api/main.py` — HTTP endpoints, lifespan, localhost access boundary, reload.
- `src/api/config.py` — all settings; process environment takes precedence over `.env`.
- `src/api/model_service.py` — champion resolution, cache/download orchestration,
  PyFunc and output validation.
- `src/api/model_cache.py` — provenance-bound identity, fail-closed verification,
  locking, atomic publication, and the import helper.
- `src/api/video_routes.py`, `video_sessions.py`, `video_service.py`,
  `youtube_source.py` — bounded MP4/YouTube sessions (see the
  [video guide](mlops_phase3_video_mvp.md)).
- `src/api/schemas.py` — response contracts.
- `frontend/` — React/TypeScript application, typed API client, Vite proxy,
  Tailwind styling, and session reconnection.
- `src/models/registry_adapter.py` — the registered PyFunc adapter; inference
  goes through MLflow, never directly through Ultralytics in the API layer.

The registered Phase 2 package stores its adapter under `code/src/models`. The
loader temporarily puts `code/src` on `sys.path` while unpickling, then checks
the `image_path` string signature and probes output with a generated local
image. The probe creates no MLflow run. The model has no output signature, so
counts, detections, confidences, and box ordering are validated explicitly.

## Windows setup and startup

The registered model was packaged with Python 3.10.11. On this checkout the
existing `venv` already has the serving dependencies. **Use its interpreter
explicitly**: unqualified `python` points to a separate Conda environment with
MLflow 2.22.5 and no Ultralytics.

```powershell
# From the repository root (existing venv on this checkout)
.\venv\Scripts\python.exe -m pip install -r requirements-serving.txt
.\venv\Scripts\python.exe -m uvicorn src.api.main:app `
  --host 127.0.0.1 --port 8000 --workers 1

# Fresh Python 3.10 environment (alternative)
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-serving.txt
.\.venv\Scripts\python.exe -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --workers 1
```

One worker is required: sessions, file leases, and model state are in-process.

## Credentials

Put personal DagsHub credentials in the ignored project-root `.env` (copy
`.env.example`), or in the process environment. Environment variables take
precedence because `load_dotenv(..., override=False)` is used.

```powershell
Copy-Item .env.example .env
# Edit .env with your own MLFLOW_TRACKING_USERNAME and MLFLOW_TRACKING_PASSWORD
```

Never commit `.env`, or place tokens/URLs in commands, logs, or screenshots.
The configured tracking URI is also used as the registry URI.

| Variable | Default |
| --- | --- |
| `MLFLOW_TRACKING_URI` | `https://dagshub.com/Nourchenne/ParkVision.mlflow` |
| `PARKVISION_MODEL_NAME` | `ParkVision-Detector` |
| `PARKVISION_MODEL_ALIAS` | `champion` |
| `PARKVISION_MAX_UPLOAD_BYTES` | `10485760` (10 MiB) |
| `PARKVISION_TEMP_DIR` | system temporary directory |
| `PARKVISION_LOG_LEVEL` | `INFO` |

## Champion resolution

At startup `load_champion` reads the alias, requires the
`parkvision.inference_verified` tag, pins `models:/ParkVision-Detector/<version>`,
and never mutates the registry. The last verified champion is Version 2
(YOLOv8s, packaging run `bf74a68ba942400caab0c9acbefabfa4`); the running version
is always taken from the live registry, never hard-coded. An alias change is
picked up on the next startup (or an explicit reload), and produces a different
cache key.

## Verified persistent cache

The cache avoids repeated large downloads on unstable links.

- **Key:** SHA-256 of registered model name, exact version, packaging run ID,
  checkpoint SHA-256, packaging-code SHA-256, and evaluation protocol ID.
  Directory names and `registered_model_meta` alone are never trusted.
- **Verification:** valid `MLmodel` with a `python_function` flavor; all required
  files present, regular, and non-symlinked; `registered_model_meta` matches the
  resolved name/version; checkpoint SHA-256 equals `parkvision.checkpoint_sha256`;
  bundled adapter SHA-256 equals `parkvision.packaging_code_sha256`; hidden
  `cache-identity.json` and `cache-manifest.json` are checked on every reuse.
- **Publication:** MLflow downloads into a unique `staging/stage-*` directory;
  after verification, `os.replace` publishes atomically under a lock file. A
  corrupt entry is renamed to `entry-*.invalid-*` and re-downloaded, never
  silently reused.
- **Locking:** the lock records the owning PID. It is reclaimed **only** when
  that PID is confirmed dead; elapsed time alone never invalidates a lock. If
  ownership cannot be determined, startup fails closed with `cache_busy` rather
  than risking concurrent publication. A lock left by a crashed process is
  reclaimed on the next attempt.
- **Location:** default `~/.parkvision/model-cache` (Windows:
  `C:\Users\<user>\.parkvision\model-cache`), outside the Git repository.
  Configure with `PARKVISION_MODEL_CACHE_DIR`; a future Docker image can mount a
  persistent volume there.

When no trusted entry exists, the exact model is downloaded once, verified, and
published. A cache hit performs **no artifact download**.

### Importing an existing complete local package

An administrator can import a complete, previously downloaded package without
a new download. The candidate is fully verified and **copied**, never moved or
deleted; partial or unverified copies are rejected.

```powershell
.\venv\Scripts\python.exe -m src.api.model_cache `
  --candidate "<complete-package-dir>" `
  --image "<one-existing-validation-image.jpg>"
```

`--image` is optional and runs a local read-only inference probe. The helper
prints the resolved version and cache key only — never tokens or URLs. Do not
point it at arbitrary checkpoints or incomplete `%TEMP%` directories.

### Failed downloads

Supported MLflow HTTP settings are applied only for the duration of an uncached
download and then restored; no inference request changes globals.

| Variable | Suggested value | Scope |
| --- | --- | --- |
| `MLFLOW_HTTP_REQUEST_TIMEOUT` | `600` | Per HTTP request, including proxied artifact files |
| `MLFLOW_HTTP_REQUEST_MAX_RETRIES` | `3` | Bounded retries; the file download restarts rather than resumes |
| `MLFLOW_HTTP_REQUEST_BACKOFF_FACTOR` | `2` | Exponential backoff |

A partial download leaves no trusted cache entry and is never reused. Import a
verified complete copy as above, then restart the process or use the bounded
reload operation once connectivity improves.

## Local model reload (opt-in)

The operator endpoint is **disabled by default** and loopback-only even when
`PARKVISION_LOCALHOST_ONLY=false`. Enable it explicitly for local development:

```powershell
$env:PARKVISION_MODEL_RELOAD_ENABLED = "true"
Invoke-RestMethod -Method Post http://127.0.0.1:8000/internal/model/reload
```

It is serialized, rate-limited (`PARKVISION_MODEL_RELOAD_MIN_INTERVAL_SECONDS`,
default 60) and capped by `PARKVISION_MODEL_RELOAD_MAX_FAILURES` (default 3).
It re-runs the same resolution/verification path, never downloads on ordinary
inference requests, and never changes the alias or version. A video session
started before a reload keeps the model it started with; new sessions use the
reloaded model. Internal logs record the failing stage (`alias_resolve`,
`artifact_download`, `cache_verification`, `cache_publication`, `pyfunc_load`,
`inference_probe`), exception type, and elapsed time — never credentials,
signed URLs, or remote exception text.

## Endpoints and quick checks

| Endpoint | Meaning |
| --- | --- |
| `GET /health` | `{"status":"ok"}` if the process is running (no DagsHub call) |
| `GET /ready` | `{"status":"ready"}` when the model loaded; otherwise 503 |
| `GET /model-info` | Name, exact version, architecture, source run ID, pinned URI, alias, status |
| `POST /predict` | Multipart field `file` with a JPEG or PNG image |
| `POST /video/upload`, `POST /video/youtube` | Bounded video sessions |
| `POST /internal/model/reload` | Opt-in, localhost-only bounded reload |

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
Invoke-RestMethod http://127.0.0.1:8000/ready
Invoke-RestMethod http://127.0.0.1:8000/model-info
curl.exe -X POST "http://127.0.0.1:8000/predict" `
  -F "file=@<path-to-validation-image.jpg>;type=image/jpeg"
```

`POST /predict` returns `model`, `counts` (`empty`, `occupied`, `total`),
`detections`, and `inference_time_ms`. `bbox` is `[x_min,y_min,x_max,y_max]` in
source pixels; class 0 is `space-empty`, class 1 is `space-occupied`. Invalid or
corrupt images return 400, oversized images 413, unavailable models 503, and
model/output failures 500. A missing `file` field returns FastAPI's 422.

For an explicit local run against the real champion, choose one existing image
from the validation split:

```powershell
.\venv\Scripts\python.exe -m src.api.smoke_test --image "<validation-image.jpg>"
```

This is a manual check, not part of the automated suite. If credentials or
network are unavailable, leave it pending.

## Tests

All tests are local or mocked: no DagsHub download, GPU, dataset, or real YouTube
acquisition.

```powershell
.\venv\Scripts\python.exe -m pytest tests/ -q
.\venv\Scripts\python.exe -m ruff check src/api tests/
.\venv\Scripts\python.exe -m pip check
git diff --check
```

## Security and synchronization limits

- No authentication, per-user authorization, CSRF protection, or TLS. The
  `/video` routes are loopback-only; forwarded headers are ignored. The local
  React development server proxies requests from `127.0.0.1:5173`.
  Reload is opt-in and loopback-only, but a reverse proxy on the same host would
  still satisfy a loopback peer check — do not expose this MVP publicly.
- One worker only. Sessions, model state, and reload state are in-memory; a
  restart loses active sessions.
- `python_model.pkl` execution implies trusted artifacts; only the registered
  champion with verified hashes is accepted.
- Multipart bodies can spool before endpoint validation; a production proxy
  must enforce request-size limits.
- Registry reads still require connectivity for alias resolution even on a
  cache hit, so a DagsHub outage yields 503 rather than an unverified fallback.

## Before Dockerization

No Dockerfile is included yet. The Docker stage should:

- use a Python 3.10 CPU base image matching the packaged model, with the
  pinned `requirements-serving.txt`;
- mount `PARKVISION_MODEL_CACHE_DIR` as a persistent volume and pre-populate it
  with the verified champion during image build or an init step;
- keep one worker, set a writable `PARKVISION_TEMP_DIR`, and size the video
  storage budget against the mounted volume;
- keep reload disabled unless a separate authenticated admin channel exists;
- add proxy request limits and real access control before any non-local use;
- serve `frontend/dist` using a separate static web server with same-origin
  API/WebSocket proxying. FastAPI does not serve frontend files.

This phase does not add Airflow, production monitoring, retraining, registration,
or automatic promotion.
