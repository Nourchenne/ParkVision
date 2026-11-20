# ParkVision MLOps Phase 2 handoff

Last verified: 2026-11-17 (Africa/Tunis). This document separates repository
evidence from user-provided project context and remaining assumptions.

## Repository state

Verified branch: `feature/mlops-phase2-registry`; current HEAD before the next
commit is `6f38befd898cc3e4fecafb7dbd97d577bbac3334`.

Verified `git status --short --branch`:

```text
## feature/mlops-phase2-registry
M  .github/workflows/ci.yml
M  README.md
A  configs/registry_policy.yaml
A  docs/mlops_phase2.md
M  src/analysis/analyze_errors.py
A  src/models/registry.py
A  src/models/registry_adapter.py
A  src/models/registry_lock.py
A  src/models/registry_policy.py
A  tests/test_registry.py
?? docs/PHASE2_HANDOFF.md
?? reports/experience_c/registry_validation/
?? reports/experience_c/validation/quality_gate_phase2_check.json
```

The implementation files were staged outside this handoff edit; this session
did not run `git add`. `docs/PHASE2_HANDOFF.md` and the report paths remain
untracked. The report files are user-generated evidence; do not silently stage,
replace, or delete them. No commit or push has been made in this session.

## Phase 1: verified implementation

- `src/analysis/validate_model.py` validates existing detection/counting
  evidence without training or inference. It checks the provisional thresholds
  in `configs/model_quality.yaml`, consistency of the counting CSV/summary,
  finished export metadata, checkpoint SHA-256, and split-manifest SHA-256.
- Supplying `--checkpoint` and `--dataset-manifest` independently recomputes
  their hashes. The manifest hash identifies the split manifest, not every
  dataset byte.
- Exit codes remain: `0` passed, `1` quality failure, `2` invalid evidence.
- `.github/workflows/ci.yml` runs syntax, Ruff, configuration validation,
  synthetic Phase 1/Phase 2 tests, and whitespace checks without a GPU,
  dataset, checkpoint, DagsHub credentials, or remote writes.
- Verified locally after the latest provenance fix: all **51 tests passed**;
  the configured Ruff command passed; `git diff --check` passed. MLflow emitted
  dependency/deprecation warnings during tests but no failures.
- Project context supplied by the user says Phase 1 and its GitHub Actions CI
  were completed before Phase 2. The current uncommitted Phase 2 branch has not
  been verified by a remote GitHub Actions run.

## Phase 2 architecture

- `src/models/registry.py`: main CLI and orchestration. It prepares candidate
  evidence, reuses the Phase 1 gate, fingerprints the evaluation protocol,
  packages and registers candidates, verifies exact-version inference, avoids
  duplicate versions, compares candidates, manages promotion history, and
  implements inspect/compare/promote/override/rollback commands.
- `src/models/registry_adapter.py`: MLflow `pyfunc` wrapper for an existing YOLO
  `best.pt`. It checks the two ParkVision classes and returns empty/occupied
  counts plus JSON detections. It is a loadability contract, not serving code.
- `src/models/registry_policy.py`: pure fail-closed champion policy. Quality-gate
  acceptance is separate from champion comparison. Exact-count accuracy is the
  primary objective; configured mAP, class-MAE, and comparable-latency limits
  prevent regressions. Missing/incompatible evidence blocks automation.
- `src/models/registry_lock.py`: cross-process advisory file lock used by every
  registry writer. It is safe only when all writers use the same shared path;
  DagsHub UI changes or writers on another unshared host bypass it.
- `configs/registry_policy.yaml`: intentionally disabled. Improvement margins,
  non-regression limits, latency requirements, absolute shared lock path, and
  bootstrap behavior remain unset. `bootstrap: manual` is the default.
- `src/analysis/analyze_errors.py`: now emits
  `evaluation_environment.json`, recording the evaluation commit, evaluator
  source hash, validation-image-name hash, raw DVC metadata hash, and
  Ultralytics version. It still evaluates validation only.
- `tests/test_registry.py`: fake-client tests for registration, duplicates,
  invalid evidence, protocol comparison, missing metrics, bootstrap, stale
  promotion, alias failures/readback, override, rollback, latency evidence,
  evaluator provenance, and local MLflow packaging/load behavior.
- `docs/mlops_phase2.md`: operator workflow, evidence limitations, commands,
  concurrency constraints, and recovery guidance.

## DagsHub configuration and limitations

Verified configuration in code/documentation:

- Tracking/registry URI:
  `https://dagshub.com/Nourchenne/ParkVision.mlflow`
- Registered model name: `ParkVision-Detector`
- Experiment C original local run ID:
  `a43e67dc22424bb6842532de9ee5bcb7`
- Experiment C imported DagsHub run ID supplied by the user:
  `ae1ab363823f48a78622d56959e0cdbc`
- Imported checkpoints are expected at `training/weights/best.pt`.

User-provided fact: experiments A, B, and C have already been imported. This
environment had no DagsHub credentials, so their live run metadata, artifacts,
registry versions, permissions, alias APIs, and current `champion` were not
verified. No remote registration or alias mutation has occurred. MLflow alias
writes provide no compare-and-swap; the shared lock plus expected-champion
check reduces races only when every writer obeys it. Decision event runs and
alias updates are not one atomic transaction.

## Experiment C evaluator-provenance mismatch

Verified values from the fresh files in
`reports/experience_c/registry_validation/`:

| Field | Expected by `prepare()` | Fresh sidecar | Result |
|---|---|---|---|
| Evaluation Git commit | `6f38befd...3334` from the counting summary | same | match |
| Evaluator code SHA-256 | `3594107b...463ba1` from `git show 6f38befd...:src/analysis/analyze_errors.py` | `26b5375b...624b7cc` | **mismatch** |
| Validation image-name SHA-256 | `7813c7f3...c68f9` from fresh CSV | same | match |
| Raw DVC metadata SHA-256 | `87ebcc48...bb963` from the original export | same | match |
| Ultralytics version | nonempty evaluator attestation | `8.4.201` | valid |

Training and evaluation provenance are intentionally separate:

- training Git commit from the original export: `4c9d8c7fa696b7a99b3aece911d4276f868c2995`
- evaluation Git commit from the fresh summary: `6f38befd898cc3e4fecafb7dbd97d577bbac3334`

Root cause: the fresh evaluation ran with the current uncommitted
`analyze_errors.py`; its sidecar correctly hashed the file that executed, but
the recorded evaluation commit points to an older committed copy. This is a
genuine reproducibility mismatch, not a training-vs-evaluation-commit bug.
Phase 1 still passed and independently verified checkpoint
`efde5092...c69a` and split manifest `b51c7072...630a`.

The fix does not bypass provenance. `read_evaluation_environment()` now reports
missing, unexpected, and mismatched fields individually without printing hash
values. A regression test reproduces uncommitted evaluator code and confirms
that different training/evaluation commits are accepted when the evaluator
hash matches its recorded evaluation commit.

Safe resolution: preserve the current reports as evidence of the failed
provenance attempt. Commit the reviewed implementation, confirm the evaluator
file has no tracked modifications, then rerun validation into a **new** output
directory. Do not edit or regenerate the old sidecar in place.

## Implemented versus unverified

Implemented and locally verified:

- Phase 1 reuse and independent checkpoint/manifest checks.
- Local candidate preparation and protocol fingerprinting.
- Loadable MLflow model packaging contract.
- Idempotent registration logic and conflicting-duplicate rejection.
- Exact-version load/inference verification logic.
- Configurable fail-closed comparison, optional latency evidence, manual
  bootstrap default, decision history, stale-state checks, override, rollback,
  and alias readback handling.
- Field-specific evaluator provenance errors and regression coverage.

Unverified or intentionally incomplete:

- Live DagsHub reads/writes, artifact downloads, registry compatibility, and
  alias behavior for this repository.
- Actual A/B/C imported run IDs other than the supplied C ID and their complete
  comparable evaluation evidence.
- Any model registration, `champion` assignment, override, or rollback.
- Policy values, reliable comparable latency benchmarks, shared lock location,
  and explicit bootstrap authorization.
- Remote CI for the current uncommitted Phase 2 branch.

## Next exact commands

Run these from `C:\Users\user\Desktop\ParkVision`. First review all tracked and
new implementation files; do not stage the two report paths listed in status.

```powershell
git status --short --branch
git diff --check
git diff -- .github/workflows/ci.yml README.md src/analysis/analyze_errors.py
Get-Content configs/registry_policy.yaml
Get-Content docs/mlops_phase2.md
Get-Content docs/PHASE2_HANDOFF.md
Get-Content src/models/registry.py
Get-Content src/models/registry_adapter.py
Get-Content src/models/registry_lock.py
Get-Content src/models/registry_policy.py
Get-Content tests/test_registry.py
python -m unittest discover -s tests -p 'test_*.py' -q
python -m ruff check src/analysis/analyze_errors.py src/analysis/validate_model.py src/models/registry.py src/models/registry_policy.py src/models/registry_adapter.py src/models/registry_lock.py tests/test_error_analysis.py tests/test_validate_model.py tests/test_registry.py
```

After review, add the handoff and confirm that only implementation, tests,
configuration, and docs are staged. The explicit `git add` is idempotent for
the files already in the index:

```powershell
git add .github/workflows/ci.yml README.md configs/registry_policy.yaml docs/mlops_phase2.md docs/PHASE2_HANDOFF.md src/analysis/analyze_errors.py src/models/registry.py src/models/registry_adapter.py src/models/registry_lock.py src/models/registry_policy.py tests/test_registry.py
git diff --cached --check
git diff --cached --stat
git commit -m "feat: add Phase 2 model registry workflow"
git status --short --branch
git diff --exit-code -- src/analysis/analyze_errors.py
```

Rerun Experiment C evaluation on the **validation split only** into a new
directory. This performs inference but does not train:

```powershell
python src/analysis/analyze_errors.py `
  --weights runs/detect/C_yolo26n_a43e67dc/weights/best.pt `
  --manifest data/processed/split_manifest.csv `
  --output reports/experience_c/registry_validation_committed `
  --conf 0.25 `
  --nms-iou 0.7 `
  --imgsz 640 `
  --batch 16 `
  --device 0
```

Then prepare a local candidate. `prepare` is read-only with respect to DagsHub;
it writes only the requested local candidate JSON:

```powershell
New-Item -ItemType Directory -Force runs/registry
python src/models/registry.py prepare `
  --archive exports/C_yolo26n_a43e67dc22424bb6842532de9ee5bcb7.zip `
  --checkpoint runs/detect/C_yolo26n_a43e67dc/weights/best.pt `
  --manifest data/processed/split_manifest.csv `
  --counting-csv reports/experience_c/registry_validation_committed/comptage.csv `
  --counting-summary reports/experience_c/registry_validation_committed/comptage_resume.json `
  --evaluation-environment reports/experience_c/registry_validation_committed/evaluation_environment.json `
  --imported-run-id ae1ab363823f48a78622d56959e0cdbc `
  --output runs/registry/candidate_c.json
```

Review the resulting JSON locally. Stop before `register`, `process`,
`promote-auto`, `override`, or `rollback` until DagsHub read access, policy
values, shared locking, and explicit remote-write approval are established.

## Safety constraints

- Do not retrain A, B, or C.
- Use validation only; do not open or evaluate the held-out test split.
- Preserve DVC data, original exports/checkpoints, Experiment C results, and
  the failed provenance reports.
- Keep DagsHub/MLflow tokens in runtime secrets or temporary environment
  variables; never write credentials to Git, reports, command arguments, or
  logs.
- Do not register remote models, change aliases, deploy, commit, or push unless
  the user explicitly authorizes that specific next action.
