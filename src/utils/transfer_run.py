"""Transferer un run MLflow (parametres, historique, fichiers) entre machines."""
import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path, PurePosixPath
from zipfile import ZipFile, ZIP_DEFLATED


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def export_run(tracking_uri, run_id, destination):
    from mlflow import MlflowClient

    client = MlflowClient(tracking_uri=tracking_uri)
    run = client.get_run(run_id)
    if run.info.status not in {"FINISHED", "FAILED", "KILLED"}:
        raise ValueError("Terminer le run avant de l'exporter.")
    destination = Path(destination).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        artifacts = root / "artifacts"
        artifacts.mkdir()
        if client.list_artifacts(run_id):
            download = root / "download"
            download.mkdir()
            source = Path(client.download_artifacts(run_id, "", str(download)))
            shutil.copytree(source, artifacts, dirs_exist_ok=True)
        metadata = {
            "format": "parkvision-run-v1",
            "source_run_id": run_id,
            "experiment": client.get_experiment(run.info.experiment_id).name,
            "start_time": run.info.start_time,
            "end_time": run.info.end_time,
            "status": run.info.status,
            "params": run.data.params,
            "tags": run.data.tags,
            "metrics": [
                {"key": key, "value": item.value,
                 "timestamp": item.timestamp, "step": item.step}
                for key in sorted(run.data.metrics)
                for item in client.get_metric_history(run_id, key)
            ],
            "artifact_hashes": {
                path.relative_to(artifacts).as_posix(): sha256(path)
                for path in sorted(artifacts.rglob("*")) if path.is_file()
            },
        }
        (root / "metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8", newline="\n",
        )
        pending = destination.with_suffix(".zip.tmp")
        with ZipFile(pending, "w", compression=ZIP_DEFLATED) as archive:
            archive.write(root / "metadata.json", "metadata.json")
            for path in sorted(artifacts.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(root).as_posix())
        pending.replace(destination)
    return destination


def import_run(archive_path, tracking_uri, experiment_name=None):
    from mlflow import MlflowClient
    from mlflow.entities import Metric, Param

    client = MlflowClient(tracking_uri=tracking_uri)
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        with ZipFile(archive_path) as archive:
            names = archive.namelist()
            if len(names) != len(set(names)):
                raise ValueError("Archive contenant des noms dupliques.")
            for name in names:
                path = PurePosixPath(name)
                if path.is_absolute() or ".." in path.parts or "\\" in name or ":" in name:
                    raise ValueError("Chemin non autorise dans l'archive.")
                if name != "metadata.json" and not name.startswith("artifacts/"):
                    raise ValueError("Contenu inattendu dans l'archive.")
            archive.extractall(root)
        metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
        if metadata.get("format") != "parkvision-run-v1":
            raise ValueError("Format d'export inconnu.")
        artifacts = root / "artifacts"
        actual = {
            path.relative_to(artifacts).as_posix(): sha256(path)
            for path in artifacts.rglob("*") if path.is_file()
        }
        if actual != metadata["artifact_hashes"]:
            raise ValueError("Les fichiers ne correspondent pas a leurs empreintes SHA256.")
        name = experiment_name or metadata["experiment"]
        experiment = client.get_experiment_by_name(name)
        experiment_id = experiment.experiment_id if experiment else client.create_experiment(name)
        source_id = metadata["source_run_id"]
        if len(source_id) != 32 or any(c not in "0123456789abcdef" for c in source_id):
            raise ValueError("Identifiant source invalide.")
        existing = client.search_runs(
            [experiment_id],
            filter_string=(f"tags.`parkvision.source_run_id` = '{source_id}' "
                           "AND tags.`parkvision.import_complete` = 'true'"),
            max_results=1,
        )
        if existing:
            print("Run deja importe :", existing[0].info.run_id)
            return existing[0].info.run_id
        tags = {key: value for key, value in metadata["tags"].items()
                if not key.startswith("mlflow.")}
        tags.update({
            "mlflow.runName": metadata["tags"].get("mlflow.runName", source_id),
            "parkvision.source_run_id": source_id,
            "parkvision.import_complete": "false",
            "parkvision.archive_sha256": sha256(archive_path),
        })
        run_id = client.create_run(
            experiment_id, start_time=metadata["start_time"], tags=tags
        ).info.run_id
        try:
            params = [Param(key, str(value)) for key, value in metadata["params"].items()]
            for offset in range(0, len(params), 100):
                client.log_batch(run_id, params=params[offset:offset + 100])
            metrics = [Metric(**item) for item in metadata["metrics"]]
            for offset in range(0, len(metrics), 500):
                client.log_batch(run_id, metrics=metrics[offset:offset + 500])
            if actual:
                client.log_artifacts(run_id, str(artifacts))
            client.set_terminated(run_id, metadata["status"], metadata["end_time"])
            client.set_tag(run_id, "parkvision.import_complete", "true")
        except BaseException:
            client.set_terminated(run_id, "FAILED")
            raise
        print("Run importe :", run_id)
        return run_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    export = sub.add_parser("export")
    export.add_argument("--tracking-uri", required=True)
    export.add_argument("--run-id", required=True)
    export.add_argument("--output", required=True)
    restore = sub.add_parser("import")
    restore.add_argument("archive", type=Path)
    restore.add_argument("--tracking-uri", default="http://127.0.0.1:5000")
    restore.add_argument("--experiment")
    args = parser.parse_args()
    if args.action == "export":
        print(export_run(args.tracking_uri, args.run_id, args.output))
    else:
        import_run(args.archive, args.tracking_uri, args.experiment)


if __name__ == "__main__":
    main()
