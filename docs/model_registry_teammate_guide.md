# Model Registry — teammate workflow (Phase 2)

Use this guide from a Windows PowerShell session at the ParkVision repository
root. Experiment A uses YOLOv8n; Experiment B uses YOLOv8s. Replace every
`<PLACEHOLDER>` below with a value from your own training and validation
outputs. Never guess a run ID, hash, metric, or file path.

## 1. Understand the three stages

An MLflow **experiment** holds training and evaluation runs. A **registered
model version** is a separately packaged, loadable checkpoint with evidence
tags; version numbers reflect registration order. The **champion alias** is a
separate selection decision. Registration does not deploy a model or assign
champion. Automatic promotion is disabled in `configs/registry_policy.yaml`.
Coordinate **one registry writer at a time** across teammates. The required
`--lock-file` is local unless all writers actually share a filesystem with
working advisory locks; it cannot coordinate separate laptops.

## 2. Prepare your checkout and local environment

You need Git access, a DagsHub account with the needed MLflow read and (when
registering) write permissions, Python 3.12, a compatible PyTorch runtime,
and the project dependencies. Obtain the latest reviewed Phase 2 branch or
the merged `main` before preparing a candidate. Do not merge unreviewed local
changes into someone else's evidence. You also need the original exported ZIP,
its exact `training/weights/best.pt` checkpoint, the DVC controlled dataset
and `data/processed/split_manifest.csv`, and complete validation evidence.
The manifest and data are not in a fresh Git checkout; obtain them through the
team's DVC storage workflow. Do not substitute the held-out test split.

```powershell
git fetch origin
git switch main
git pull --ff-only origin main
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-training.txt
python -m pytest tests/ -q
```

If Phase 2 has not been merged, switch to the reviewed
`feature/mlops-phase2-registry` branch instead and pull it fast-forward.
Training and inference also require the appropriate PyTorch installation for
your CPU/GPU; check it separately. The team must provide DVC remote access or
an approved local dataset copy. Do not infer that `dvc pull` is configured.

Copy the tracked template and set **your own** DagsHub username and personal
access token. `.env` is Git-ignored; `.env.example` contains placeholders only.
Both CLI scripts load the root `.env` with `python-dotenv`, even when launched
from another working directory. Existing process environment variables win.
Never paste tokens into command arguments, logs, screenshots, or reports.

```powershell
Copy-Item .env.example .env
notepad .env
git check-ignore -v .env
```

Verify credentials with a read-only call. `registry.py inspect` lists versions
and the champion alias, if any; it makes no remote changes. An authentication
or permission error is not an empty registry. Stop and fix access before
attempting any import or registration.

```powershell
python src/models/registry.py inspect
```

## 3. Identify the exact training run and imported run

Use the original run ID recorded in the ZIP's `metadata.json`; it is **not**
the DagsHub imported run ID. The ZIP's SHA256 and original run ID together
identify an import. Use the importer inventory to find the exact imported run:

```powershell
$Archive = 'exports\<EXPERIMENT_A_OR_B_EXPORT>.zip'
python src/utils/transfer_run.py inspect $Archive
```

`inspect` is read-only, but it may download remote artifacts to verify their
hashes and can take time. Read `source_run_id`, `archive_sha256`, `matches`,
and `reusable_run_id`. A reusable run must have the exact archive SHA, a
FINISHED status, an import-complete tag, and matching params, metrics, tags,
timestamps, and artifacts. Verify the run in the DagsHub `ParkVision`
experiment as well. Experiment C's reviewed example is original run
`a43e67dc22424bb6842532de9ee5bcb7` and verified imported run
`47c671a7bccd4389a287ee3f09b28ca9`; these IDs do not apply to A or B.

If `reusable_run_id` is present, use it in `prepare` and **do not import
again**. Running `import` without `--write` is another read-only inventory:

```powershell
python src/utils/transfer_run.py import $Archive
```

Only a designated writer should run the following after an inventory confirms
there is **no** exact complete or partial matching run. The CLI searches again
before creation; a failed or uncertain search blocks writes. Two writers on
different machines can still race, so coordinate externally.

```powershell
python src/utils/transfer_run.py import $Archive --write
python src/utils/transfer_run.py inspect $Archive
```

A RUNNING or FAILED exact match is **partial**, not reusable. Do not start a
fresh import. Investigate its run ID and artifacts, then explicitly resume
one matching partial run only after team review:

```powershell
python src/utils/transfer_run.py inspect $Archive
python src/utils/transfer_run.py import $Archive --write --resume-run-id <PARTIAL_DAGSHUB_RUN_ID>
python src/utils/transfer_run.py inspect $Archive
```

The resume command checks existing content, uploads only missing content,
then verifies and finalizes. Conflicting bytes or metadata stop recovery.
If a create or upload times out, inspect again before retrying; the remote
operation may already have landed. Never delete or retag old runs to resolve
an ambiguity. A different archive SHA, even for a similarly named run, is
different lineage and needs investigation.

## 4. Produce validation evidence and pass the gate

Run the evaluator from a clean, committed checkout so its code hash matches
the recorded evaluation commit. Use the **validation** split from the
manifest; `analyze_errors.py` selects `split == valid` and does not read the
held-out test split. A and B need independent output directories.

```powershell
$Checkpoint = 'runs\<EXPERIMENT_A_OR_B>\weights\best.pt'
$Evidence = 'reports\<EXPERIMENT_A_OR_B>\registry_validation'
python src/analysis/analyze_errors.py `
  --weights $Checkpoint `
  --manifest data/processed/split_manifest.csv `
  --output $Evidence
```

The evaluator writes `comptage.csv`, `comptage_resume.json`,
`evaluation_environment.json`, error details, and examples. Do not use
`--limit` for registration evidence. Run the Phase 1 gate using actual
checkpoint and manifest files so their hashes are independently checked:

```powershell
python src/analysis/validate_model.py `
  --config configs/model_quality.yaml `
  --counting-csv "$Evidence\comptage.csv" `
  --counting-summary "$Evidence\comptage_resume.json" `
  --detection-metrics $Archive `
  --checkpoint $Checkpoint `
  --dataset-manifest data/processed/split_manifest.csv `
  --output "runs\quality\<EXPERIMENT_A_OR_B>.json"
$LASTEXITCODE
```

Exit 0 means the gate passed; 1 means a threshold failed; 2 means missing,
malformed, or inconsistent evidence. A pass makes the candidate eligible for
registration, not champion selection. Keep the gate JSON for your review.

## 5. Prepare and verify a candidate

Set `$ImportedRunId` to the **verified exact DagsHub import** from step 3.
Prepare recomputes the gate and provenance locally; it does not register.

```powershell
$ImportedRunId = '<VERIFIED_DAGSHUB_RUN_ID>'
$Candidate = 'runs\registry\candidate_<A_OR_B>.json'
New-Item -ItemType Directory -Force runs\registry | Out-Null
python src/models/registry.py prepare `
  --archive $Archive `
  --checkpoint $Checkpoint `
  --manifest data/processed/split_manifest.csv `
  --counting-csv "$Evidence\comptage.csv" `
  --counting-summary "$Evidence\comptage_resume.json" `
  --evaluation-environment "$Evidence\evaluation_environment.json" `
  --imported-run-id $ImportedRunId `
  --output $Candidate
```

Inspect the JSON locally. Confirm `quality_status` is `passed`, `verified` is
true, the architecture is A's YOLOv8n or B's YOLOv8s as appropriate, and
`original_run_id`, `imported_run_id`, `export_sha256`, checkpoint, manifest,
evaluation, and code hashes match your own source files. If any mismatch
appears, fix the source evidence and regenerate; never edit candidate or
historical evidence to force a match. `register` independently recomputes
the candidate and checks the imported run lineage and remote checkpoint.
If you used `--latency-evidence PATH` during prepare, pass the same file to
register; otherwise omit it from both.

## 6. Register one model version and verify it

Get a real validation image from the manifest. This uses its `image` column
and checks the file exists. Do not pick a test image or a guessed path.

```powershell
$SampleImage = Import-Csv data/processed/split_manifest.csv |
  Where-Object { $_.split -eq 'valid' -and (Test-Path $_.image) } |
  Select-Object -First 1 -ExpandProperty image
if (-not $SampleImage) { throw 'No local validation image found' }
```

Coordinate a single writer before `register`. This command makes a remote
packaging run and model version; it never assigns the champion alias. Supply
a local lock path for this invocation, understanding that it does not lock
other machines.

```powershell
python src/models/registry.py register `
  --candidate $Candidate `
  --archive $Archive `
  --checkpoint $Checkpoint `
  --manifest data/processed/split_manifest.csv `
  --counting-csv "$Evidence\comptage.csv" `
  --counting-summary "$Evidence\comptage_resume.json" `
  --evaluation-environment "$Evidence\evaluation_environment.json" `
  --sample-image $SampleImage `
  --lock-file runs\registry\registration.lock
```

The result includes the exact `version`. Registration reloads that version
and runs inference on the validation sample before setting
`parkvision.inference_verified=true`. If an error occurs after a remote
write, inspect the registry before retrying; registration can find an
equivalent existing version, but network uncertainty still needs review.

```powershell
python src/models/registry.py inspect
```

In DagsHub's [ParkVision Model Registry](https://dagshub.com/Nourchenne/ParkVision.mlflow),
open `ParkVision-Detector` and check the exact version, architecture,
original/imported IDs, checkpoint hash, quality status, protocol, and
inference verification tags against your candidate. The `inspect` output
should show the same version and tags. Registration alone does not promote
or deploy a model.

State verified on 2026-11-20: version 2 (YOLOv8s, B) is the initial champion;
version 3 contains C's new verified evaluation, and version 1 preserves C's
historical evaluation. A remains an MLflow baseline because it fails the
counting MAE thresholds. Future version numbers must be read from the Registry.

B and C's current packaging runs contain six verified files under
`validation_evidence/`. Automatic promotion remains disabled.
The initial champion decision is recorded in MLflow run
`f6bd457accf54b5291f31395811f3e7b`.

## Troubleshooting and review checklist

| Symptom | Safe action |
| --- | --- |
| 401/403 or permission error | Check your own `.env`, DagsHub role, and token; do not infer an empty search or create a run. |
| Missing ZIP, changed ZIP SHA, or lineage mismatch | Match `metadata.json`, source ID, archive SHA, and imported tags; do not relabel a different run. |
| FAILED/RUNNING import, partial artifacts, timeout | Run read-only `inspect`; coordinate and resume one exact partial ID only if metadata and existing artifacts agree. |
| Checkpoint mismatch | Compare local, ZIP, and remote `training/weights/best.pt` hashes; obtain the correct checkpoint. |
| Missing evaluation files or failed gate | Rerun validation on `valid`, then the gate; preserve old evidence. |
| Evaluator code or environment mismatch | Use committed evaluator code and regenerate evidence in a new directory. |
| Existing equivalent/conflicting registry version | Inspect its tags and package run; equivalent evidence can be reused, conflicting evidence must be resolved before any retry. |

Keep source ZIPs and weights out of Git. Review small reproducibility files
before staging. The `comptage.csv`, `comptage_resume.json`, and
`evaluation_environment.json` files used to prepare a candidate must remain
available together and unchanged. Put large example images and checkpoints
in DVC or an approved artifact store, not ordinary Git. The local
`runs/registry/candidate_*.json` is generated and Git-ignored; each teammate
can regenerate it from the reviewed evidence. Do not replace a historical
report with a newer run's output. See `docs/mlops_phase2.md` for protocol
details and the promotion policy; leave automatic promotion disabled during
team onboarding.
