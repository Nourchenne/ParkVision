# Phase 2: registry candidates and champion promotion

Phase 2 uses `https://dagshub.com/Nourchenne/ParkVision.mlflow` and the
registered name `ParkVision-Detector`. It does not train, deploy, or modify
imported experiment runs. `configs/registry_policy.yaml` disables automatic
promotion by default. As verified on 2026-11-20, version 2 (YOLOv8s, B)
is the initial champion, selected manually with an audit record. Version 3
contains C's new evaluation; historical version 1 remains unchanged.
This selection does not deploy an application. See the
[teammate guide](model_registry_teammate_guide.md) for the current workflow.

## Candidate evidence

`prepare` calls the existing Phase 1 gate with an export ZIP, counting CSV and
summary, actual checkpoint, and split manifest. It verifies the checkpoint in
the ZIP and creates a protocol fingerprint from split-manifest and raw DVC
metadata hashes, validation image names, committed analysis code, evaluator
Ultralytics version, class names, metric name, and inference settings. Training and
evaluation Git commits are recorded separately. These checks do not
prove the bytes of every image and label or which code actually executed.
The evaluator runtime must be supplied through `--evaluation-environment PATH`
for **both** preparation and registration. This JSON has exactly
`git_commit`, `analysis_code_sha256`, `validation_image_names_sha256`,
`raw_dvc_sha256`, and `ultralytics_version`. The four hashes/IDs are checked
against the export, counting CSV, and Git; the runtime version is an evaluator
attestation. Future `analyze_errors.py` runs write this file alongside the
counting artifacts. Without this record the candidate can be registered and manually
selected, but automatic promotion is blocked. Existing Experiment C results
remain unchanged; its evaluator runtime is not reconstructed from training.
Run evaluation from a clean, committed checkout: the recorded Git commit must
contain the exact evaluator source whose SHA-256 appears in the sidecar. If the
working copy was dirty, keep the old reports intact, commit reviewed evaluator
code later, and rerun validation into a new output directory. Do not edit the
old sidecar to make its hashes agree.

Use the **DagsHub imported** run ID, not the original training run ID:

```powershell
New-Item -ItemType Directory -Force runs/registry
python src/models/registry.py prepare `
  --archive exports/C_yolo26n_a43e67dc22424bb6842532de9ee5bcb7.zip `
  --checkpoint runs/detect/C_yolo26n_a43e67dc/weights/best.pt `
  --manifest data/processed/split_manifest.csv `
  --counting-csv reports/experience_c/validation/comptage.csv `
  --counting-summary reports/experience_c/validation/comptage_resume.json `
  --imported-run-id <DAGSHUB_RUN_ID> `
  --output runs/registry/candidate_c.json
```

`register` and `process` recompute this evidence and reject altered candidate
JSON. They check imported-run lineage, download its `training/weights/best.pt`,
and compare its hash. Optional `--latency-evidence PATH` must be passed to both
preparation and registration. Its JSON fields are `samples_ms`, `hardware`,
`runtime`, `device`, `input_shape`, `batch_size`, `warmup_runs`, `method`,
`confidence`, `nms_iou`, and `image_size`. The record stores the sample median
and a fingerprint of benchmark conditions. One-sided or incomparable latency
evidence blocks automatic promotion. With no latency evidence for either
version, no latency non-regression claim is made.

## Registration and inference

Registration creates a separate preparation run containing an MLflow `pyfunc`
Model and the existing checkpoint. The imported training run remains unchanged.
Version tags preserve original and imported run IDs, hashes, quality result,
evaluation protocol, and packaging code hash. The adapter accepts a pandas
table with one `image_path` column and returns class counts and JSON detections.
It runs CPU inference for a bounded validation sample. This is a loadability
contract, not a serving API. An exact registry version is loaded and invoked
before it becomes promotion-eligible. Versions reflect registration order.

`register` only registers. `process` registers and then attempts automatic
promotion if the policy is enabled. Both take the evidence arguments above,
plus `--candidate`, `--sample-image`, and `--lock-file`. `process` also accepts
`--policy`. Example:

```powershell
python src/models/registry.py process `
  --candidate runs/registry/candidate_c.json `
  --archive exports/C_yolo26n_a43e67dc22424bb6842532de9ee5bcb7.zip `
  --checkpoint runs/detect/C_yolo26n_a43e67dc/weights/best.pt `
  --manifest data/processed/split_manifest.csv `
  --counting-csv reports/experience_c/validation/comptage.csv `
  --counting-summary reports/experience_c/validation/comptage_resume.json `
  --sample-image <VALIDATION_IMAGE_PATH> `
  --lock-file <SHARED_LOCK_PATH> `
  --policy configs/registry_policy.yaml
```

Registration requires MLflow, Ultralytics, compatible PyTorch, remote write
permission, and a shared lock path. DagsHub credentials belong in a runtime
secret store or temporary environment, never Git or command arguments.
Loading `.pt` or custom MLflow model code requires trusted artifacts.

## Promotion and rollback

The Phase 1 gate decides minimum acceptance. The configured comparison policy
requires a strict joint exact-count accuracy improvement meeting
`min_exact_gain`, plus mAP50-95 and both class MAEs within configured
non-regression limits. No margin or latency threshold is preselected. When
comparable reliable latency benchmarks exist, the configured relative limit
applies. The policy requires project-configured minimum sample and warm-up
counts. Missing metrics or mismatched evaluation protocols block automation.

`bootstrap: manual` leaves a registry without a champion. The explicit
`allow_eligible_candidate` alternative assigns the first fully verified
candidate only when no champion exists. Normal comparisons never promote ties.

All writers must use the same lock file on a shared filesystem supporting
advisory locks. A local-only path cannot coordinate hosts, and DagsHub UI
edits bypass this lock. **Keep automatic promotion disabled if any alias writer
can bypass it.** MLflow alias writes have no compare-and-swap. Under the lock,
promotion rechecks candidate, champion, policy, and expected alias; writes an
event; changes the alias; and reads it back. A stale or uncertain outcome stops
without blindly restoring an older alias.

`inspect`, `history`, and `compare` are read-only. `promote-auto`, `override`,
and `rollback` require `--version` and `--expected-champion VERSION|none`.
The manual commands also require `--reason` and `--lock-file`. Rollback accepts
only a recorded previous champion. Override can bypass ranking but not model
integrity or inference checks. Each decision has a separate MLflow event run
with decision and outcome artifacts. This is operational history, not a
tamper-proof audit log or an atomic transaction with the alias update. An
uncertain result needs alias/event inspection before retry.

The fake-client tests run without GPU or DagsHub access. The local MLflow
packaging test skips when MLflow is not installed. The existing Phase 1 CI
and quality thresholds remain intact.
