# Handoff — remaining work

This guide covers the work remaining on ParkVision after the frontend redesign
was merged into `main`. It is written to be followed without further context.
Run every command from a Windows PowerShell session at the repository root
unless a section says otherwise.

Owner of each task below: member B. Repository ownership, the DVC remote, the
model registry, and the champion alias stay with member A. Nothing in this
guide requires a registry write.

## 0. State of the project

What is already in place on `main`:

| Area | State |
|---|---|
| Data pipeline | `dvc.yaml` with `prepare_split` and `prepare_yolo`, DVC remote on DagsHub |
| Training | Experiments A (YOLOv8n), B (YOLOv8s), C (YOLO26n), reports in `reports/` |
| Registry | `ParkVision-Detector`, version 2 (YOLOv8s, experiment B) holds the `champion` alias |
| Serving | FastAPI reads the champion from a verified local cache, no registry writes |
| Frontend | React console: Overview, Video Analysis, Image Analysis, Model Information, API Diagnostics |
| Orchestration | `orchestration/parkvision_flow.py`, a Prefect flow wrapping the two DVC stages |
| CI | Lint, tests, frontend build, mypy, Trivy image scan, on pull requests to `main` |

Four tasks remain, in priority order. Tasks 3 and 4 are an investigation and
are expected to lower the reported metrics on purpose; read section 3 before
starting either.

## 1. Set up and verify the current state

```powershell
git fetch origin
git switch main
git pull --ff-only origin main
python -m venv .venv
.\.venv\Scripts\Activate.ps1
.\.venv\Scripts\python.exe -m pip install -r requirements-serving.txt
```

Copy `.env.example` to `.env` and fill in **your own** DagsHub username and
access token. Never commit `.env`; it is already ignored.

Start the backend in one terminal:

```powershell
.\.venv\Scripts\python.exe -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --workers 1
```

Start the frontend in a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open `http://127.0.0.1:5173/`. Both badges in the top right must turn green
("API online", "Model ready") before you consider the environment working.
The first start downloads the champion package from DagsHub and caches it in
`~/.parkvision/model-cache`; later starts are fast.

### Repository conventions

- Branch from an up-to-date `main`, named `feat/<short-topic>` or `fix/<short-topic>`.
- One pull request per task. Do not push to `main` directly.
- CI must be green before merge. Merge with **Squash and merge**.
- Before opening a pull request, run from `frontend/`: `npm run lint`,
  `npm run test -- --run`, `npm run build`. From the root: `python -m pytest tests/ -q`.

## 2. Prefect orchestration — finish and document

**Current state.** `orchestration/parkvision_flow.py` defines a
`parkvision-pipeline` flow with two tasks, `prepare_split` and `prepare_yolo`,
each calling `dvc repro <stage>` with one retry. The flow runs, but there is no
recorded execution and no deployment.

**What to do.**

1. Pull the dataset so the stages have inputs: `dvc pull -r dagshub`.
2. Start the Prefect server in one terminal: `prefect server start`.
3. Run the flow in a second terminal: `python orchestration/parkvision_flow.py`.
4. Open the Prefect UI, confirm both tasks succeeded, and capture screenshots
   of the flow run and of the task timeline.
5. Create a deployment so the pipeline can be triggered or scheduled rather
   than run by hand, and record the command used.
6. Write `docs/orchestration.md`: what the flow does, how to run it, how the
   deployment is created, what a successful run looks like, and the known
   limits (the flow orchestrates data preparation only; training stays on
   Colab and is not part of the flow).

**Acceptance criteria.** A reader who has never run Prefect can reproduce a
successful flow run from `docs/orchestration.md` alone. The document states
explicitly that DVC skips stages that are already up to date, so a second run
showing "skipped" stages is a correct result, not a failure.

## 3. Camera-based split — investigation

**Why this matters.** The champion reports a recall of 0.999 and a mAP50-95 of
0.933 on validation, but detects almost nothing on a parking lot it has never
seen. The cause is in `src/data/prepare_split.py`:

```python
days = sorted({day for _, day in images})
random.Random(SEED).shuffle(days)
```

The split groups images **by capture day only**. PKLot was recorded by a small
number of fixed cameras over many days, so every camera appears in train,
validation and test alike. The reported scores therefore measure performance on
camera viewpoints the model has already memorised, not generalisation. The
experiment B report already lists this under its limitations.

**Read this before planning the work.** The camera identifier is **not
available in the data**. The Roboflow export names files by timestamp only,
for example `2012-10-27_06_35_35_jpg.rf.99e9d59f32a59.jpg`, and
`prepare_split.py` parses nothing but the first ten characters as a date. There
is no lot or camera column anywhere in `data/raw`. Grouping by camera requires
recovering that grouping first.

**Proposed approach.** The cameras are fixed, so all images from one camera
share a near-identical static background. Cluster on that background:

1. Load every image from `data/raw/*/images/`, convert to greyscale and
   downscale hard, for example to 64x36.
2. Flatten each to a vector and run k-means. Start at k=3, the number of PKLot
   lots (PUCPR, UFPR04, UFPR05).
3. Sample 5 to 10 images per cluster and view them. Clusters must be visually
   distinct locations. If one lot was recorded from two slightly different
   camera positions over the period, raise k and merge clusters by hand.
4. Write the mapping to `data/processed/camera_map.csv` with columns
   `image,cluster`, and commit the script that produced it under `src/data/`.

Then add a camera-aware split, as a **new** function or script rather than a
rewrite of `prepare_split.py`, so the existing day-based manifest and all
published results stay reproducible. Hold out one full cluster for test and one
for validation.

**Acceptance criteria.** `data/processed/camera_map.csv` exists, the clustering
script is committed and rerunnable, and a short note in
`docs/camera_split.md` shows one sample image per cluster and states how many
images each cluster holds. If clustering cannot separate the lots cleanly,
write that down and stop — a documented negative result closes this task.

## 4. Ablation retraining — only if time allows

This task depends on task 3 and is optional. Do not start it if task 1 or 2 is
unfinished.

Retrain YOLOv8s on the camera-based split, using the same configuration as
`configs/train_b.yaml` so the only variable is the split, and evaluate with the
protocol already fixed in `configs/evaluation_protocol.yaml` (confidence 0.25,
NMS IoU 0.7, matching IoU 0.5, image size 640). Expect substantially lower
numbers than experiment B. That drop **is** the result: it measures the real
generalisation gap. Report it as an ablation in the final document, next to the
experiment B figures, not as a replacement for them.

**Acceptance criteria.** A report at `reports/ablation_camera_split/` holding
the training configuration, the metrics, and a short comparison table against
experiment B, plus a paragraph explaining what the difference means.

**Do not** register this model, and **do not** move the `champion` alias.
Version 2 stays the champion for the demonstration.

## 5. Gallery picker and webcam capture

**Integration point.** `frontend/src/pages/ImagePage.tsx`. The page already
holds a complete single-image flow: a drop zone and file input that set a
`File` into state, then an `analyze()` function calling `api.predict(file)` and
`api.predictVisualize(file)` in parallel and rendering counts, the annotated
image, and a detections table.

**What to add.** Two further ways to produce that same `File`:

- A gallery picker, by adding `capture`-free multiple selection or a simple
  recently-used list, whichever you judge more useful.
- A webcam capture, using `navigator.mediaDevices.getUserMedia`, drawing the
  current video frame to a canvas, and converting it with `canvas.toBlob()`
  into a `File` passed to the existing `selectFile()`.

Route both through `selectFile()` so preview, cleanup and analysis stay
unchanged. Remove the placeholder hint at the bottom of the upload panel once
the features exist.

**Required handling.** Request camera permission only when the user activates
the webcam, never on page load. Stop every track with `track.stop()` when the
user leaves the page or switches back to upload, otherwise the camera light
stays on. Show a clear message when permission is denied or no camera is
present, reusing the existing `inline-error` class.

**Acceptance criteria.** All three input paths produce the same analysis
result for the same image. `npm run lint`, `npm run test -- --run` and
`npm run build` pass. Camera tracks are released on unmount, verified by the
camera indicator switching off when navigating away.

## 6. Out of scope

Do not change any of the following without talking to member A first:

- The `champion` alias or any registered model version.
- `configs/evaluation_protocol.yaml`, which fixes the comparison contract
  between experiments A, B and C.
- The existing day-based `split_manifest.csv`, which every published result
  depends on.
- The held-out PKLot test split, 1029 images listed in
  `data/processed/yolo/test.txt`, reserved for the final evaluation.
