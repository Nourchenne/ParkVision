# Phase 1: model quality gate and CI

The quality gate reads existing validation artifacts and checks four provisional
thresholds in `configs/model_quality.yaml`. mAP and exact-count accuracy use
decimal values in `[0, 1]`; MAE is the mean absolute count error per image.
These thresholds are engineering starting points, not established production
requirements. A passing gate does not approve a model for deployment.

## Inputs and result

Supply `comptage.csv` and `comptage_resume.json` from `analyze_errors.py`, plus
the `metadata.json` (or complete ZIP) from `transfer_run.py export`. The
detector metric is the single `best_val/metrics/mAP50-95_B_` entry recorded
after validation of the best checkpoint. Epoch history and the last row of
`results.csv` are not accepted as substitutes. The counting CSV supplies the
image count, class MAEs, and joint exact-count accuracy. Summary totals and
MAEs are checked against it. The report records four independent checks.
For ZIP input, only `metadata.json` is read; nothing is extracted. Its
uncompressed size is limited to 2 MiB using the declared entry size and the
bytes actually read.

The export and counting summary record a checkpoint SHA256; the export also
records a manifest SHA256. Expected hashes compare against these recorded
values. If actual files are supplied via `--checkpoint` and
`--dataset-manifest`, their SHA256 values are recomputed and the report marks
them independently verified. Otherwise the report marks them as recorded only.
The manifest hash identifies the recorded split manifest, not every dataset
image or annotation. Inputs should represent the same validation run and
checkpoint; a matching recorded hash cannot independently prove that the
counting CSV was generated from that checkpoint.

Exit 0 means all checks passed. Exit 1 and `quality_failed` mean at least one
threshold failed. Exit 2 and `invalid_evidence` mean an input was missing,
malformed, oversized, corrupt, or inconsistent. The CLI prints the reason.
Both outcomes write a JSON report when `--output`
is given. A failed check calls for investigation or improvement; it is not a
request to evaluate the held-out test set.

## Local commands

For existing Experiment C results, when the local export ZIP is available:

```powershell
python src/analysis/validate_model.py `
  --config configs/model_quality.yaml `
  --counting-csv reports/experience_c/validation/comptage.csv `
  --counting-summary reports/experience_c/validation/comptage_resume.json `
  --detection-metrics exports/C_yolo26n_a43e67dc22424bb6842532de9ee5bcb7.zip `
  --expected-model-sha256 efde509277dc0ea69f3077356c81b86839bd3caf19343ce58ca562fbe097c69a `
  --expected-dataset-sha256 b51c7072a9276eff8c69d9023849979efaecf1aadd458f31c2e81651ee76630a `
  --output runs/quality/experiment_c.json
```

Run synthetic tests and validate the config:

```powershell
python -m unittest discover -s tests -p test_validate_model.py -v
python src/analysis/validate_model.py --check-config --config configs/model_quality.yaml
```

GitHub Actions runs on pushes
to `main` and pull requests targeting `main`. CI installs only PyYAML and Ruff,
runs syntax, lint, unit, schema, synthetic gate, and whitespace checks. It has
no dataset, GPU, checkpoint, or MLflow requirement. A later phase can register
reviewed models with the MLflow Model Registry and add a separate deployment
approval process.
