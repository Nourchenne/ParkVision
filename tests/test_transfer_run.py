"""Deterministic import safety tests; no DagsHub calls or model downloads."""

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import pytest

from src.utils import transfer_run as transfer

SOURCE = "a" * 32


def archive_at(path, artifact=b"checkpoint"):
    metadata = {
        "format": "parkvision-run-v1", "source_run_id": SOURCE,
        "experiment": "ParkVision", "status": "FINISHED",
        "start_time": 1000, "end_time": 2000,
        "params": {"batch": "16"},
        "tags": {"mlflow.runName": "C_yolo26n", "git_commit": "abc123"},
        "metrics": [{"key": "val/map", "value": 0.5, "timestamp": 1500, "step": 1}],
        "artifact_hashes": {"best.pt": hashlib.sha256(artifact).hexdigest()},
    }
    with ZipFile(path, "w") as stream:
        stream.writestr("metadata.json", json.dumps(metadata))
        stream.writestr("artifacts/best.pt", artifact)
    return path


class Page(list):
    def __init__(self, items, token=None):
        super().__init__(items)
        self.token = token


class FakeClient:
    def __init__(self):
        self.runs = {}
        self.files = {}
        self.created = 0
        self.uploads = 0
        self.fail_upload_once = False
        self.fail_before_upload_once = False
        self.fail_create_after_write = False
        self.search_error = None
        self.search_failures = 0
        self.search_calls = 0

    def get_experiment_by_name(self, name):
        return SimpleNamespace(experiment_id="1")

    def search_runs(self, experiment_ids, filter_string, run_view_type, max_results, page_token):
        self.search_calls += 1
        if self.search_error:
            raise self.search_error
        if self.search_failures:
            self.search_failures -= 1
            raise FakeHttpError(503, "secret-password")
        matches = [run for run in self.runs.values()
                   if run.data.tags.get("parkvision.source_run_id") == SOURCE]
        offset = int(page_token or 0)
        selected = matches[offset:offset + 1]
        next_token = str(offset + 1) if offset + 1 < len(matches) else None
        return Page(selected, next_token)

    def create_run(self, experiment_id, start_time, tags):
        self.created += 1
        run_id = f"run-{self.created}"
        run = SimpleNamespace(
            info=SimpleNamespace(run_id=run_id, status="RUNNING", start_time=start_time,
                                 end_time=None),
            data=SimpleNamespace(tags=dict(tags), params={}, metrics={}),
        )
        self.runs[run_id] = run
        self.files[run_id] = {}
        if self.fail_create_after_write:
            self.fail_create_after_write = False
            raise TimeoutError("secret-password")
        return run

    def get_run(self, run_id):
        return self.runs[run_id]

    def get_metric_history(self, run_id, key):
        return self.runs[run_id].data.metrics.get(key, [])

    def log_batch(self, run_id, metrics=(), params=()):
        run = self.runs[run_id]
        run.data.params.update({item.key: item.value for item in params})
        for item in metrics:
            run.data.metrics.setdefault(item.key, []).append(item)

    def list_artifacts(self, run_id, directory=""):
        return [SimpleNamespace(path=path, is_dir=False)
                for path in self.files[run_id] if not directory]

    def download_artifacts(self, run_id, path, destination):
        target = Path(destination) / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(self.files[run_id][path])
        return str(target)

    def log_artifact(self, run_id, local_file, artifact_path=None):
        self.uploads += 1
        if self.fail_before_upload_once:
            self.fail_before_upload_once = False
            raise TimeoutError("secret-password")
        path = f"{artifact_path}/{Path(local_file).name}" if artifact_path else Path(local_file).name
        self.files[run_id][path] = Path(local_file).read_bytes()
        if self.fail_upload_once:
            self.fail_upload_once = False
            raise TimeoutError("secret-password")

    def set_terminated(self, run_id, status, end_time):
        self.runs[run_id].info.status = status
        self.runs[run_id].info.end_time = end_time

    def set_tag(self, run_id, key, value):
        self.runs[run_id].data.tags[key] = value


class FakeHttpError(Exception):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status

    def get_http_status_code(self):
        return self.status


def test_default_import_is_read_only_and_repeated_write_reuses_verified_run(tmp_path):
    archive = archive_at(tmp_path / "run.zip")
    client = FakeClient()
    report = transfer.import_run(archive, "unused", client=client)
    assert report["matches"] == []
    assert client.created == 0
    first = transfer.import_run(archive, "unused", write=True, client=client)
    assert transfer.import_run(archive, "unused", write=True, client=client) == first
    assert client.created == 1
    assert client.uploads == 1
    assert transfer.inspect_import(archive, "unused", client=client)["reusable_run_id"] == first


def test_interrupted_upload_is_reported_and_explicit_resume_verifies_it(tmp_path):
    archive = archive_at(tmp_path / "run.zip")
    client = FakeClient()
    client.fail_upload_once = True
    with pytest.raises(transfer.RemoteImportError, match="upload artifact") as error:
        transfer.import_run(archive, "unused", write=True, client=client)
    assert "secret-password" not in str(error.value)
    partial = transfer.inspect_import(archive, "unused", client=client)["matches"][0]
    assert not partial["import_complete"]
    with pytest.raises(transfer.ImportConflict, match="partial matching import"):
        transfer.import_run(archive, "unused", write=True, client=client)
    run_id = partial["run_id"]
    assert transfer.import_run(archive, "unused", write=True,
                               resume_run_id=run_id, client=client) == run_id
    assert client.created == 1
    assert client.uploads == 1  # The timed-out upload actually landed; resume skipped it.
    assert client.runs[run_id].data.tags["parkvision.import_complete"] == "true"


def test_uncertain_create_does_not_auto_create_another_run(tmp_path):
    archive = archive_at(tmp_path / "run.zip")
    client = FakeClient()
    client.fail_create_after_write = True
    with pytest.raises(transfer.RemoteImportError, match="create run"):
        transfer.import_run(archive, "unused", write=True, client=client)
    assert client.created == 1
    with pytest.raises(transfer.ImportConflict, match="partial matching import"):
        transfer.import_run(archive, "unused", write=True, client=client)
    assert client.created == 1


def test_resume_uploads_only_missing_artifact_from_failed_run(tmp_path):
    archive = archive_at(tmp_path / "run.zip")
    client = FakeClient()
    client.fail_before_upload_once = True
    with pytest.raises(transfer.RemoteImportError):
        transfer.import_run(archive, "unused", write=True, client=client)
    run_id = next(iter(client.runs))
    client.runs[run_id].info.status = "FAILED"
    assert transfer.import_run(archive, "unused", write=True,
                               resume_run_id=run_id, client=client) == run_id
    assert client.uploads == 2
    assert client.runs[run_id].info.status == "FINISHED"


def test_completed_run_with_wrong_artifact_is_never_reused(tmp_path):
    archive = archive_at(tmp_path / "run.zip")
    client = FakeClient()
    run_id = transfer.import_run(archive, "unused", write=True, client=client)
    client.files[run_id]["best.pt"] = b"different"
    with pytest.raises(transfer.ImportConflict, match="conflicting artifacts"):
        transfer.import_run(archive, "unused", write=True, client=client)
    assert client.created == 1


def test_completed_run_with_changed_training_tag_is_never_reused(tmp_path):
    archive = archive_at(tmp_path / "run.zip")
    client = FakeClient()
    run_id = transfer.import_run(archive, "unused", write=True, client=client)
    client.runs[run_id].data.tags["git_commit"] = "different"
    with pytest.raises(transfer.ImportConflict, match="conflicting training tags"):
        transfer.import_run(archive, "unused", write=True, client=client)
    assert client.created == 1


def test_search_paginates_past_partial_to_completed_run(tmp_path):
    archive = archive_at(tmp_path / "run.zip")
    client = FakeClient()
    partial = transfer.import_run(archive, "unused", write=True, client=client)
    client.runs[partial].data.tags["parkvision.import_complete"] = "false"
    second = client.create_run("1", 1000, dict(client.runs[partial].data.tags))
    second.data.params.update(client.runs[partial].data.params)
    second.data.metrics.update(client.runs[partial].data.metrics)
    second.info.status = "FINISHED"
    second.info.end_time = 2000
    second.data.tags["parkvision.import_complete"] = "true"
    client.files[second.info.run_id] = dict(client.files[partial])
    report = transfer.inspect_import(archive, "unused", client=client)
    assert len(report["matches"]) == 2
    assert report["reusable_run_id"] == second.info.run_id
    assert client.search_calls >= 3


@pytest.mark.parametrize("status", [401, 403, 503])
def test_search_errors_never_look_like_empty_results(tmp_path, status, monkeypatch):
    archive = archive_at(tmp_path / "run.zip")
    client = FakeClient()
    client.search_error = FakeHttpError(status, "secret-password")
    monkeypatch.setattr(transfer.time, "sleep", lambda _: None)
    with pytest.raises(transfer.RemoteImportError) as error:
        transfer.import_run(archive, "unused", write=True, client=client)
    assert client.created == 0
    assert "secret-password" not in str(error.value)
    assert ("authentication" in str(error.value)) == (status in (401, 403))


def test_temporary_search_error_has_bounded_read_retry(tmp_path, monkeypatch):
    archive = archive_at(tmp_path / "run.zip")
    client = FakeClient()
    client.search_failures = 2
    monkeypatch.setattr(transfer.time, "sleep", lambda _: None)
    assert transfer.inspect_import(archive, "unused", client=client)["matches"] == []
    assert client.search_calls == 3


def test_cli_import_defaults_to_read_only_and_uses_root_env_uri(monkeypatch, capsys):
    calls = []
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://example.invalid/mlflow")
    monkeypatch.setattr(sys, "argv", ["transfer_run.py", "import", "run.zip"])
    monkeypatch.setattr(transfer, "import_run", lambda archive, uri, experiment,
                        *, write, resume_run_id: calls.append((uri, write, resume_run_id))
                        or {"matches": []})
    transfer.main()
    assert calls == [("https://example.invalid/mlflow", False, None)]
    assert "example.invalid" not in capsys.readouterr().out
