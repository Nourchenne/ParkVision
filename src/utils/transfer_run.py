"""Transferer un run MLflow (parametres, historique, fichiers) entre machines."""
import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path, PurePosixPath
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[2]
PAGE_SIZE = 100
READ_ATTEMPTS = 3


class RemoteImportError(RuntimeError):
    """A remote operation failed; messages contain no server response or credentials."""


class ImportConflict(ValueError):
    """An existing run cannot safely be reused or resumed."""


def _configure_network():
    # MLflow retries writes internally by default. A failed write must be inspected first.
    os.environ["MLFLOW_HTTP_REQUEST_MAX_RETRIES"] = "0"
    os.environ["MLFLOW_HTTP_REQUEST_TIMEOUT"] = "30"
    os.environ["MLFLOW_ARTIFACT_UPLOAD_DOWNLOAD_TIMEOUT"] = "120"


def _remote(operation, action, *, read=False):
    from requests.exceptions import ConnectionError as RequestConnectionError
    from requests.exceptions import Timeout as RequestTimeout

    for attempt in range(READ_ATTEMPTS if read else 1):
        try:
            return action()
        except Exception as error:  # noqa: BLE001 - sanitize all SDK/network failures
            status_method = getattr(error, "get_http_status_code", None)
            status = status_method() if callable(status_method) else None
            code = str(getattr(error, "error_code", ""))
            auth = status in (401, 403) or code in ("UNAUTHENTICATED", "PERMISSION_DENIED")
            temporary = (status in (408, 429, 500, 502, 503, 504) or
                         isinstance(error, (TimeoutError, RequestTimeout, RequestConnectionError)))
            if read and temporary and not auth and attempt + 1 < READ_ATTEMPTS:
                time.sleep(2 ** attempt)
                continue
            reason = ("authentication/authorization failed (401/403)" if auth else
                      "temporary server/network failure" if temporary else
                      "remote API failure")
            raise RemoteImportError(f"{operation}: {reason}; inspect before retrying") from None


def _archive_contents(archive_path, root):
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
    source_id = metadata["source_run_id"]
    if len(source_id) != 32 or any(c not in "0123456789abcdef" for c in source_id):
        raise ValueError("Identifiant source invalide.")
    artifacts = root / "artifacts"
    actual = {path.relative_to(artifacts).as_posix(): sha256(path)
              for path in artifacts.rglob("*") if path.is_file()}
    if actual != metadata["artifact_hashes"]:
        raise ValueError("Les fichiers ne correspondent pas a leurs empreintes SHA256.")
    return metadata, artifacts


def _matching_runs(client, experiment_id, source_id):
    from mlflow.entities import ViewType

    token = None
    seen = set()
    runs = []
    while True:
        page = _remote("search existing imports", lambda token=token: client.search_runs(
            [experiment_id],
            filter_string=f"tags.`parkvision.source_run_id` = '{source_id}'",
            run_view_type=ViewType.ALL, max_results=PAGE_SIZE, page_token=token,
        ), read=True)
        if page is None:
            raise RemoteImportError("search existing imports: uncertain response")
        runs.extend(page)
        next_token = getattr(page, "token", None)
        if not next_token:
            return runs
        if next_token in seen:
            raise RemoteImportError("search existing imports: repeated page token")
        seen.add(next_token)
        token = next_token


def _remote_artifacts(client, run_id, destination, expected_paths):
    destination = destination / run_id
    destination.mkdir(parents=True, exist_ok=True)
    paths = []
    pending = [""]
    while pending:
        directory = pending.pop()
        entries = _remote("list remote artifacts",
                          lambda directory=directory: client.list_artifacts(run_id, directory),
                          read=True)
        if entries is None:
            raise RemoteImportError("list remote artifacts: uncertain response")
        for entry in entries:
            if entry.is_dir:
                if entry.path not in expected_paths and not any(
                    path.startswith(entry.path + "/") for path in expected_paths
                ):
                    raise ImportConflict(f"run {run_id} has an unexpected artifact directory")
                pending.append(entry.path)
            else:
                if entry.path not in expected_paths:
                    raise ImportConflict(f"run {run_id} has an unexpected artifact")
                paths.append(entry.path)
    hashes = {}
    for index, path in enumerate(sorted(paths), 1):
        print(f"Run {run_id}: verifying artifact {index}/{len(paths)}",
              file=sys.stderr, flush=True)
        downloaded = _remote("download remote artifact", lambda path=path: client.download_artifacts(
            run_id, path, str(destination)), read=True)
        hashes[path] = sha256(downloaded)
    return hashes


def _remote_state(client, run, metadata, destination):
    """Compare remote payload with the archive without writing to the run."""
    run_id = run.info.run_id
    tags = run.data.tags
    if (tags.get("parkvision.source_run_id") != metadata["source_run_id"] or
            tags.get("parkvision.archive_sha256") != metadata["archive_sha256"]):
        raise ImportConflict(f"run {run_id} has different source/archive lineage")
    if run.info.start_time != metadata["start_time"]:
        raise ImportConflict(f"run {run_id} has conflicting start time")
    import_tags = {"parkvision.source_run_id", "parkvision.archive_sha256",
                   "parkvision.import_complete"}
    expected_tags = {key: value for key, value in metadata["tags"].items()
                     if not key.startswith("mlflow.") and key not in import_tags}
    if any(tags.get(key) != value for key, value in expected_tags.items()):
        raise ImportConflict(f"run {run_id} has conflicting training tags")
    expected_params = {key: str(value) for key, value in metadata["params"].items()}
    if any(expected_params.get(key) != value for key, value in run.data.params.items()):
        raise ImportConflict(f"run {run_id} has conflicting parameters")
    missing_params = {key: value for key, value in expected_params.items()
                      if key not in run.data.params}
    expected_metrics = Counter((item["key"], item["value"], item["timestamp"], item["step"])
                               for item in metadata["metrics"])
    actual_metrics = Counter()
    for key in sorted(set(run.data.metrics) | {item["key"] for item in metadata["metrics"]}):
        history = _remote("read remote metrics", lambda key=key: client.get_metric_history(run_id, key),
                          read=True)
        actual_metrics.update((key, item.value, item.timestamp, item.step) for item in history)
    if actual_metrics - expected_metrics:
        raise ImportConflict(f"run {run_id} has conflicting metrics")
    missing_metrics = list((expected_metrics - actual_metrics).elements())
    remote_artifacts = _remote_artifacts(client, run_id, destination,
                                         set(metadata["artifact_hashes"]))
    if any(metadata["artifact_hashes"].get(path) != digest
           for path, digest in remote_artifacts.items()):
        raise ImportConflict(f"run {run_id} has conflicting artifacts")
    missing_artifacts = sorted(set(metadata["artifact_hashes"]) - set(remote_artifacts))
    return missing_params, missing_metrics, missing_artifacts


def inspect_import(archive_path, tracking_uri, experiment_name=None, *, client=None):
    """Read-only inventory; completed imports are reused only after payload verification."""
    from mlflow import MlflowClient

    _configure_network()
    archive_path = Path(archive_path)
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        metadata, _ = _archive_contents(archive_path, root)
        metadata["archive_sha256"] = sha256(archive_path)
        client = client or MlflowClient(tracking_uri=tracking_uri)
        name = experiment_name or metadata["experiment"]
        experiment = _remote("find experiment", lambda: client.get_experiment_by_name(name),
                             read=True)
        report = {"source_run_id": metadata["source_run_id"],
                  "archive_sha256": metadata["archive_sha256"], "experiment": name,
                  "matches": [], "reusable_run_id": None}
        if experiment is None:
            return report
        runs = _matching_runs(client, experiment.experiment_id, metadata["source_run_id"])
        for run in sorted(runs, key=lambda item: item.info.run_id):
            tags = run.data.tags
            exact = tags.get("parkvision.archive_sha256") == metadata["archive_sha256"]
            complete = tags.get("parkvision.import_complete") == "true"
            item = {"run_id": run.info.run_id, "status": run.info.status,
                    "archive_sha256": tags.get("parkvision.archive_sha256"),
                    "import_complete": complete, "exact_archive": exact}
            if exact and complete:
                current = _remote("read completed run",
                                  lambda run_id=run.info.run_id: client.get_run(run_id),
                                  read=True)
                missing = _remote_state(client, current, metadata, root / "verification")
                if (current.info.status != metadata["status"] or
                        current.info.start_time != metadata["start_time"] or
                        current.info.end_time != metadata["end_time"] or any(missing)):
                    raise ImportConflict(f"run {run.info.run_id} claims completion but payload differs")
                item["verified"] = True
                report["reusable_run_id"] = report["reusable_run_id"] or run.info.run_id
            else:
                item["verified"] = False
            report["matches"].append(item)
        return report


def _upload_missing(client, run_id, metadata, artifacts, root):
    from mlflow.entities import Metric, Param

    run = _remote("read partial run", lambda: client.get_run(run_id), read=True)
    missing_params, missing_metrics, missing_artifacts = _remote_state(
        client, run, metadata, root / "verification",
    )
    if run.data.tags.get("parkvision.import_complete") == "true":
        raise ImportConflict(f"run {run_id} is already marked complete")
    if missing_params:
        print(f"Run {run_id}: uploading {len(missing_params)} parameters", flush=True)
        params = [Param(key, value) for key, value in missing_params.items()]
        for offset in range(0, len(params), 100):
            _remote("upload parameters", lambda batch=params[offset:offset + 100]:
                    client.log_batch(run_id, params=batch))
    if missing_metrics:
        print(f"Run {run_id}: uploading {len(missing_metrics)} metrics", flush=True)
        metrics = [Metric(key, value, timestamp, step)
                   for key, value, timestamp, step in missing_metrics]
        for offset in range(0, len(metrics), 500):
            _remote("upload metrics", lambda batch=metrics[offset:offset + 500]:
                    client.log_batch(run_id, metrics=batch))
            print(f"Run {run_id}: metrics {min(offset + 500, len(metrics))}/{len(metrics)}",
                  flush=True)
    for index, path in enumerate(missing_artifacts, 1):
        print(f"Run {run_id}: uploading artifact {index}/{len(missing_artifacts)}", flush=True)
        _remote("upload artifact", lambda path=path: client.log_artifact(
            run_id, str(artifacts / path),
            str(PurePosixPath(path).parent) if "/" in path else None,
        ))
    current = _remote("verify uploaded run", lambda: client.get_run(run_id), read=True)
    if any(_remote_state(client, current, metadata, root / "final_verification")):
        raise ImportConflict(f"run {run_id} remains incomplete after upload")
    print(f"Run {run_id}: finalizing", flush=True)
    _remote("finalize run", lambda: client.set_terminated(
        run_id, metadata["status"], metadata["end_time"],
    ))
    _remote("mark import complete", lambda: client.set_tag(
        run_id, "parkvision.import_complete", "true",
    ))
    final = _remote("verify final status", lambda: client.get_run(run_id), read=True)
    if (final.info.status != metadata["status"] or
            final.data.tags.get("parkvision.import_complete") != "true"):
        raise ImportConflict(f"run {run_id} final status is uncertain")
    return run_id


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


def import_run(archive_path, tracking_uri, experiment_name=None, *, write=False,
               resume_run_id=None, client=None):
    """Inspect by default; explicit writes create or safely resume one matching run."""
    from mlflow import MlflowClient

    if resume_run_id and not write:
        raise ValueError("--resume-run-id requires --write")
    _configure_network()
    client = client or MlflowClient(tracking_uri=tracking_uri)
    report = inspect_import(archive_path, tracking_uri, experiment_name, client=client)
    if not write:
        return report
    if report["reusable_run_id"]:
        print("Verified completed import:", report["reusable_run_id"], flush=True)
        return report["reusable_run_id"]
    exact_partials = [item for item in report["matches"]
                      if item["exact_archive"] and not item["import_complete"]]
    if exact_partials and not resume_run_id:
        ids = ", ".join(item["run_id"] for item in exact_partials)
        raise ImportConflict(f"partial matching import(s): {ids}; inspect and select --resume-run-id")
    if resume_run_id and resume_run_id not in {item["run_id"] for item in exact_partials}:
        raise ImportConflict("resume run ID is not an incomplete exact source/archive match")
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        metadata, artifacts = _archive_contents(archive_path, root)
        metadata["archive_sha256"] = report["archive_sha256"]
        name = experiment_name or metadata["experiment"]
        if resume_run_id:
            print(f"Resuming verified partial run {resume_run_id}", flush=True)
            return _upload_missing(client, resume_run_id, metadata, artifacts, root)
        experiment = _remote("find experiment", lambda: client.get_experiment_by_name(name),
                             read=True)
        if experiment is None:
            experiment_id = _remote("create experiment", lambda: client.create_experiment(name))
        else:
            experiment_id = experiment.experiment_id
            # Repeat the search immediately before create. A failed search is never empty.
            if any(run.data.tags.get("parkvision.archive_sha256") == metadata["archive_sha256"]
                   for run in _matching_runs(client, experiment_id, metadata["source_run_id"])):
                raise ImportConflict("import state changed during preflight; inspect again")
        tags = {key: value for key, value in metadata["tags"].items()
                if not key.startswith("mlflow.")}
        tags.update({
            "mlflow.runName": metadata["tags"].get("mlflow.runName", metadata["source_run_id"]),
            "parkvision.source_run_id": metadata["source_run_id"],
            "parkvision.import_complete": "false",
            "parkvision.archive_sha256": metadata["archive_sha256"],
        })
        created = _remote("create run", lambda: client.create_run(
            experiment_id, start_time=metadata["start_time"], tags=tags,
        ))
        run_id = created.info.run_id
        print(f"Created remote run {run_id}; import_complete=false", flush=True)
        return _upload_missing(client, run_id, metadata, artifacts, root)


def main():
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env", override=False)
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    export = sub.add_parser("export")
    export.add_argument("--tracking-uri", required=True)
    export.add_argument("--run-id", required=True)
    export.add_argument("--output", required=True)
    restore = sub.add_parser("import")
    restore.add_argument("archive", type=Path)
    restore.add_argument("--tracking-uri")
    restore.add_argument("--experiment")
    restore.add_argument("--write", action="store_true",
                         help="Explicitly allow remote creation or resume")
    restore.add_argument("--resume-run-id", help="Explicitly resume one exact partial import")
    inspect = sub.add_parser("inspect", help="Read-only import inventory")
    inspect.add_argument("archive", type=Path)
    inspect.add_argument("--tracking-uri")
    inspect.add_argument("--experiment")
    args = parser.parse_args()
    try:
        if args.action == "export":
            print(export_run(args.tracking_uri, args.run_id, args.output))
        else:
            tracking_uri = args.tracking_uri or os.environ.get(
                "MLFLOW_TRACKING_URI", "http://127.0.0.1:5000",
            )
            if args.action == "inspect":
                result = inspect_import(args.archive, tracking_uri, args.experiment)
            else:
                result = import_run(args.archive, tracking_uri, args.experiment,
                                    write=args.write, resume_run_id=args.resume_run_id)
            print(json.dumps(result, indent=2) if isinstance(result, dict) else result)
    except (RemoteImportError, ImportConflict, ValueError) as error:
        parser.exit(1, f"Import stopped: {error}\n")


if __name__ == "__main__":
    main()
