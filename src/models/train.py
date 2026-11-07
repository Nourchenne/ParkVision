"""Entrainement YOLO, suivi MLflow local et export portable pour ParkVision."""
import argparse
import csv
import hashlib
import math
import os
import platform
import random
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resolve_project_path(value):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def read_inputs(config_path):
    config = yaml.safe_load(config_path.read_text(encoding="utf-8-sig"))
    required = {"model", "data", "epochs", "batch", "imgsz", "seed", "name"}
    if not isinstance(config, dict) or not required.issubset(config):
        raise ValueError("Configuration incomplete.")
    data_path = resolve_project_path(config["data"])
    data = yaml.safe_load(data_path.read_text(encoding="utf-8-sig"))
    expected = ["space-empty", "space-occupied"]
    names = data["names"]
    if isinstance(names, dict):
        names = [names[index] for index in range(len(names))]
    if names != expected:
        raise ValueError("Les classes ne correspondent pas a PKLot.")
    manifest = ROOT / "data/processed/split_manifest.csv"
    groups = {"train": [], "valid": [], "test": []}
    day_groups, seen = {}, set()
    with manifest.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            split, day = row["split"], row["date"]
            image = resolve_project_path(row["image"])
            if split not in groups or image in seen:
                raise ValueError("Split inconnu ou image dupliquee dans le manifeste.")
            if day_groups.setdefault(day, split) != split:
                raise ValueError(f"Journee partagee entre plusieurs splits : {day}")
            if not image.is_relative_to((ROOT / "data/raw").resolve()):
                raise ValueError("Une image est situee hors de data/raw.")
            seen.add(image)
            groups[split].append(image)
            if split != "test":
                label = image.parent.parent / "labels" / f"{image.stem}.txt"
                if not image.is_file() or not label.is_file():
                    raise FileNotFoundError(f"Image ou annotation manquante : {image}")
    if any(not images for images in groups.values()):
        raise ValueError("Un split est vide.")
    # Comparer les listes produites par prepare_yolo au manifeste DVC.
    base = resolve_project_path(data["path"]) if data.get("path") else data_path.parent
    for split, key in (("train", "train"), ("valid", "val"), ("test", "test")):
        listing = (base / data[key]).resolve()
        paths = [(listing.parent / line.strip()).resolve()
                 for line in listing.read_text(encoding="utf-8-sig").splitlines()
                 if line.strip()]
        if len(paths) != len(set(paths)) or set(paths) != set(groups[split]):
            raise ValueError(f"Liste YOLO differente du manifeste : {split}")
        print(f"{split} : {len(groups[split])} images")
    return config, groups, manifest, expected


def prepare_runtime_data(groups, names, destination, smoke, seed):
    """Copies de travail pour que YOLO n'ecrive pas de caches dans data/raw."""
    destination.mkdir(parents=True, exist_ok=False)
    selected = {}
    rng = random.Random(seed)
    for split, limit in (("train", 128), ("valid", 32)):
        images = sorted(groups[split])
        if smoke:
            images = sorted(rng.sample(images, min(limit, len(images))))
        selected[split] = [path.relative_to(ROOT).as_posix() for path in images]
        image_dir, label_dir = destination / split / "images", destination / split / "labels"
        image_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)
        for image in images:
            target = image_dir / image.name
            if target.exists():
                raise ValueError(f"Collision de noms : {image.name}")
            label = image.parent.parent / "labels" / f"{image.stem}.txt"
            shutil.copyfile(image, target)
            shutil.copyfile(label, label_dir / label.name)
    dataset = {"path": destination.as_posix(), "train": "train/images",
               "val": "valid/images", "nc": len(names), "names": names}
    path = destination / "data.yaml"
    path.write_text(yaml.safe_dump(dataset, sort_keys=False), encoding="utf-8", newline="\n")
    return path, selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train_a.yaml")
    parser.add_argument("--check", action="store_true", help="Verifier sans GPU ni entrainement")
    parser.add_argument("--smoke", action="store_true", help="1 epoque, 128 images train, 32 validation")
    parser.add_argument("--device", help="Exemple : 0 ou cpu (CPU autorise uniquement en mode smoke)")
    parser.add_argument("--tracking-dir", help="Dossier local pour SQLite et les fichiers MLflow")
    args = parser.parse_args()
    config_path = resolve_project_path(args.config)
    config, groups, manifest, names = read_inputs(config_path)
    if args.check:
        print("Verification OK : configuration, fichiers et separation des jours.")
        print("Aucun entrainement lance. Images de test non ouvertes.")
        return

    # Imports differes : --check fonctionne sans PyTorch ni Ultralytics.
    import mlflow
    import torch
    import ultralytics
    from ultralytics import YOLO, settings

    sys.path.insert(0, str(ROOT / "src/utils"))
    from transfer_run import export_run

    os.chdir(ROOT)
    device = args.device if args.device is not None else config.get("device", 0)
    if str(device) == "cpu" and not args.smoke:
        raise ValueError("Utiliser --smoke pour un essai CPU ; entrainement complet prevu sur GPU.")
    if str(device) != "cpu" and not torch.cuda.is_available():
        raise RuntimeError("GPU indisponible. Utiliser --check ou attendre une session GPU.")
    # Desactiver les autres integrations automatiques disponibles dans cette version.
    settings.update({key: False for key in ("mlflow", "wandb", "comet", "clearml", "dvc")
                     if key in settings})
    tracking = (Path(args.tracking_dir).resolve() if args.tracking_dir else
                (Path("/content/parkvision_tracking") if Path("/content").is_dir()
                 else ROOT / "tracking"))
    tracking.mkdir(parents=True, exist_ok=True)
    artifacts = tracking / "artifacts"
    artifacts.mkdir(exist_ok=True)
    uri = f"sqlite:///{(tracking / 'mlflow.db').as_posix()}"
    mlflow.set_tracking_uri(uri)
    experiment = "ParkVision_Verifications" if args.smoke else "ParkVision"
    if mlflow.get_experiment_by_name(experiment) is None:
        mlflow.create_experiment(experiment, artifact_location=artifacts.as_uri())
    mlflow.set_experiment(experiment)
    run_name = f"{config['name']}_smoke" if args.smoke else config["name"]
    run_id = None
    started = time.monotonic()
    try:
        with mlflow.start_run(run_name=run_name) as run:
            run_id = run.info.run_id
            mlflow.set_tags({"type": "smoke" if args.smoke else "training",
                             "test_evaluated": "false",
                             "member": os.environ.get("PARKVISION_MEMBER", "Nourchenne")})
            commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
            mlflow.set_tag("git_commit", commit)
            mlflow.set_tag("manifest_sha256", fingerprint(manifest))
            mlflow.set_tag("train_script_sha256", fingerprint(__file__))
            mlflow.log_artifact(str(config_path), "provenance")
            for path in (ROOT / "dvc.lock", ROOT / "data/raw.dvc",
                         ROOT / "requirements-training.txt", manifest, Path(__file__),
                         ROOT / "src/utils/transfer_run.py"):
                if path.is_file():
                    mlflow.log_artifact(str(path), "provenance")
            environment = {"python": platform.python_version(), "pytorch": torch.__version__,
                           "ultralytics": ultralytics.__version__, "mlflow": mlflow.__version__,
                           "cuda": torch.version.cuda, "device": str(device),
                           "gpu": torch.cuda.get_device_name(0) if str(device) != "cpu" else None}
            mlflow.log_dict(environment, "provenance/environment.json")
            mlflow.log_text(subprocess.check_output(
                [sys.executable, "-m", "pip", "freeze"], text=True), "provenance/pip_freeze.txt")
            runtime_data = ROOT / "runs/datasets" / run_id
            dataset_yaml, selected = prepare_runtime_data(
                groups, names, runtime_data, args.smoke, config["seed"])
            mlflow.log_dict(selected, "provenance/selected_images.json")
            train_args = dict(config)
            weights = train_args.pop("model")
            train_args.update(data=str(dataset_yaml), device=device,
                              project=str(resolve_project_path(config.get("project", "runs/detect"))),
                              name=f"{run_name}_{run_id[:8]}", exist_ok=False)
            if args.smoke:
                train_args.update(epochs=1, patience=0, close_mosaic=0, warmup_epochs=0)
            model = YOLO(weights)
            logged_epochs = set()

            def record_parameters(trainer):
                mlflow.log_params({key: str(value) for key, value in vars(trainer.args).items()})

            def record_epoch(trainer):
                # final_eval peut rappeler ce callback pour le meilleur modele.
                step = int(trainer.epoch) + 1
                if step in logged_epochs:
                    return
                values = {**trainer.label_loss_items(trainer.tloss, prefix="train"),
                          **trainer.metrics, **trainer.lr}
                metrics = {re.sub(r"[^\w/ .-]", "_", key): float(value)
                           for key, value in values.items() if math.isfinite(float(value))}
                mlflow.log_metrics(metrics, step=step, synchronous=True)
                logged_epochs.add(step)

            model.add_callback("on_pretrain_routine_end", record_parameters)
            model.add_callback("on_fit_epoch_end", record_epoch)
            try:
                result = model.train(**train_args)
                if result is not None:
                    mlflow.log_metrics({
                        "best_val/" + re.sub(r"[^\w/ .-]", "_", key): float(value)
                        for key, value in result.results_dict.items()
                        if math.isfinite(float(value))
                    }, synchronous=True)
            finally:
                trainer = getattr(model, "trainer", None)
                output = getattr(trainer, "save_dir", None)
                if output is not None and Path(output).is_dir():
                    mlflow.log_artifacts(str(output), "training")
                mlflow.log_metric("duration_seconds", time.monotonic() - started)
    finally:
        if run_id is not None:
            training_error = sys.exc_info()[0] is not None
            try:
                archive = export_run(uri, run_id, ROOT / "exports" / f"{run_name}_{run_id}.zip")
                print("\nEXPORT A TELECHARGER :", archive, flush=True)
            except Exception as error:
                print(f"Export non termine : {error}. Le suivi est dans {tracking}.", file=sys.stderr)
                if not training_error:
                    raise RuntimeError("Entrainement termine mais export a refaire avant fermeture.") from error


if __name__ == "__main__":
    main()
