# ParkVision

ParkVision detects parking spaces and classifies them as `space-empty` or
`space-occupied`. The number of detections in each class provides a per-image
parking count. The repository contains YOLO training and validation tools,
day-separated PKLot data preparation with DVC, MLflow run exports, and a
lightweight model-quality gate.

Registry packaging, policy-based champion selection, and rollback are covered
in [the Phase 2 workflow](docs/mlops_phase2.md). Automatic promotion is disabled
by default and requires explicit policy values and a shared writer lock.

## Model Registry — Phase 2

The [teammate registration guide](docs/model_registry_teammate_guide.md) walks
through read-only import inspection, validation evidence, candidate preparation,
and explicit registration in the [ParkVision DagsHub Model Registry](https://dagshub.com/Nourchenne/ParkVision.mlflow).
YOLO26n is registered as `ParkVision-Detector` Version 1. Registration creates
a candidate version; it does not assign the `champion` alias or deploy it.
No champion is currently assigned. Coordinate a single registry writer until
all machines share a verified lock strategy.

The recorded Experiment C results are **validation** results. The held-out test
split is reserved for final evaluation after model selection.

## Requirements

Run commands from the repository root. For the quality gate and its tests,
install only PyYAML (Python 3.12 is used in CI):

```powershell
python -m pip install PyYAML==6.0.3
```

For lint checks, also install `ruff==0.17.3`. Training and inference require
the heavier dependencies in `requirements-training.txt`, a compatible PyTorch
runtime, and locally available data and model weights. They are not needed for
the quality-gate tests. See [the training workflow](docs/training_workflow.md)
for setup, GPU, and MLflow export details.

## Data and training

`dvc.yaml` defines `prepare_split` and `prepare_yolo`. These stages need the
PKLot data in `data/raw`; a fresh checkout does not contain the dataset. After
obtaining the data through the project's DVC workflow, run:

```powershell
dvc repro
python src/models/train.py --config configs/train_c.yaml --check
```

`--check` validates inputs without training. For training and run transfer,
follow [the training workflow](docs/training_workflow.md). Experiment reports
are available for [B](docs/experience_b.md) and [C](docs/experience_c.md).

## Model-quality validation

The gate reads existing counting results and a portable MLflow export ZIP (or
its `metadata.json`). It uses the recorded best-checkpoint mAP50-95, recomputes
counting errors from `comptage.csv`, and evaluates the **provisional** thresholds
in [model_quality.yaml](configs/model_quality.yaml). It performs no training or
inference. An approved quality check is not deployment approval.

Validate an existing Experiment C export, if the local ZIP is available:

```powershell
python src/analysis/validate_model.py `
  --config configs/model_quality.yaml `
  --counting-csv reports/experience_c/validation/comptage.csv `
  --counting-summary reports/experience_c/validation/comptage_resume.json `
  --detection-metrics exports/C_yolo26n_a43e67dc22424bb6842532de9ee5bcb7.zip `
  --output runs/quality/experiment_c.json
$LASTEXITCODE
```

Exit `0` means all thresholds passed; `1` means a quality threshold failed;
`2` means evidence is missing, malformed, or inconsistent. The output JSON
records the status, metrics, checks, and provenance. Hashes in the export are
recorded provenance; add `--checkpoint PATH` and `--dataset-manifest PATH` to
recompute and verify hashes from actual files. See the
[Phase 1 guide](docs/mlops_phase1.md) for the full input contract, optional
expected-hash arguments, ZIP limit, and limitations.

## Tests and CI

Run the GPU-free tests and checks:

```powershell
python -m unittest discover -s tests -p test_validate_model.py -v
python -m unittest discover -s tests -p test_error_analysis.py -v
python -m unittest discover -s tests -p test_training_tools.py -v -k DataChecks
python src/analysis/validate_model.py --check-config --config configs/model_quality.yaml
ruff check src/analysis/validate_model.py tests/test_validate_model.py
ruff format --check src/analysis/validate_model.py tests/test_validate_model.py
git diff --check
```

The full local and mocked test suite, including `test_transfer_run.py`, runs
with `python -m pytest tests/ -q` after installing the pinned CI dependencies
in [GitHub Actions](.github/workflows/ci.yml). CI uses Python 3.12 and MLflow
3.17.2. It needs no GPU, dataset, checkpoint, or remote tracking server.
The commands above remain useful for a lighter quality-gate check.
Current local verification used Python 3.13.12 and MLflow 2.22.5; this does
not substitute for a passing CI run with the pinned Python 3.12/MLflow 3.17.2
environment before merge.
