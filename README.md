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
Registry state verified on 2026-11-20:
- Version 2: YOLOv8s (B), selected as the initial `champion`.
- Version 3: YOLO26n (C), with a new verified evaluation.
- Version 1: historical YOLO26n evaluation, preserved unchanged.

Use `models:/ParkVision-Detector@champion` to resolve the selected model.
B and C have verified inference and archived validation evidence.
A remains an MLflow baseline because its counting MAEs exceed the quality thresholds.
The initial champion was selected manually with an MLflow audit record.
Automatic promotion remains disabled; no application deployment has been performed.
Coordinate a single registry writer until all machines share a verified lock strategy.

The recorded Experiment C results are **validation** results. The held-out test
split is reserved for final evaluation after model selection.

## Requirements

Run commands from the repository root. For the quality gate and its tests,
install only PyYAML (CI tests Python 3.10 and 3.12):

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

The full local and mocked test suite runs with: python -m pytest tests/ -q.

Install the pinned test dependencies listed in .github/workflows/ci.yml.
CI runs on pull requests targeting main and pushes to main,
using Python 3.10 and 3.12 with MLflow 3.17.2.
Tests require no GPU, dataset, trained checkpoint, or remote tracking server.
The transfer test uses temporary local MLflow storage.
These checks do not deploy models or change the champion alias.

## Jalon de mi-parcours

Les livrables de mi-parcours sont dans `docs/mi_parcours/` : rapport (PDF), pitch (PDF et PPTX) et sources LaTeX du rapport.

### Reproduire les résultats

```powershell
git clone https://github.com/Nourchenne/ParkVision.git
cd ParkVision
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-training.txt "dvc[s3]"
dvc remote modify dagshub --local access_key_id VOTRE_JETON_DAGSHUB
dvc remote modify dagshub --local secret_access_key VOTRE_JETON_DAGSHUB
dvc pull -r dagshub
dvc repro
python src/models/train.py --config configs/train_b.yaml --check
```

Le jeton DagsHub reste personnel : il est écrit dans `.dvc/config.local`, qui n'est pas suivi par Git.

Suivi des expériences : https://dagshub.com/Nourchenne/ParkVision.mlflow (runs `A_yolov8n_baseline`, `B_yolov8s_baseline`, `C_yolo26n`).
