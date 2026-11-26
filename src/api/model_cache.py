"""Verified, persistent cache for the exact registered MLflow champion.

The cache never selects a model by directory name alone. Every entry is bound
to the resolved registry version and its trusted provenance tags, and is fully
verified before publication and before reuse. No registry write is performed.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml
from mlflow.models import Model

from src.api.config import Settings

MANIFEST_FILE = "cache-manifest.json"
IDENTITY_FILE = "cache-identity.json"
REGISTRY_META_FILE = "registered_model_meta"
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_ENTRY_PREFIX = "entry-"
_STAGING_PREFIX = "stage-"


class CacheVerificationError(RuntimeError):
    """A cache candidate failed a closed integrity check."""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_hash(value: object, label: str) -> str:
    if not isinstance(value, str) or not _HEX64.fullmatch(value):
        raise CacheVerificationError("invalid_provenance", f"{label} is missing or malformed")
    return value


@dataclass(frozen=True)
class ModelIdentity:
    """Provenance-derived identity of one exact registered model version."""

    name: str
    version: str
    run_id: str
    checkpoint_sha256: str
    packaging_code_sha256: str
    protocol_id: str | None = None

    def key_fields(self) -> dict:
        return {
            "name": self.name,
            "version": self.version,
            "run_id": self.run_id,
            "checkpoint_sha256": self.checkpoint_sha256,
            "packaging_code_sha256": self.packaging_code_sha256,
            "protocol_id": self.protocol_id,
        }

    def as_dict(self) -> dict:
        return asdict(self)

    def key(self) -> str:
        return hashlib.sha256(_canonical(self.key_fields()).encode()).hexdigest()


def identity_from_version(name: str, version) -> ModelIdentity | None:
    """Build an identity from registry metadata, or return None if unavailable.

    Returning None means no verified cache can be selected; serving fails
    closed rather than downloading an unverified package.
    """
    tags = getattr(version, "tags", None) or {}
    checkpoint = tags.get("parkvision.checkpoint_sha256")
    adapter = tags.get("parkvision.packaging_code_sha256")
    if checkpoint is None and adapter is None:
        return None
    run_id = getattr(version, "run_id", None)
    number = getattr(version, "version", None)
    if not isinstance(run_id, str) or not run_id or number is None:
        raise CacheVerificationError("invalid_provenance", "registry version identity is incomplete")
    return ModelIdentity(
        name=name,
        version=str(number),
        run_id=run_id,
        checkpoint_sha256=_require_hash(checkpoint, "checkpoint SHA-256"),
        packaging_code_sha256=_require_hash(adapter, "packaging code SHA-256"),
        protocol_id=tags.get("parkvision.evaluation_protocol_id"),
    )


def cache_root(settings: Settings) -> Path:
    if not settings.model_cache_dir:
        raise CacheVerificationError("cache_disabled", "model cache directory is not configured")
    return settings.model_cache_dir.resolve()


def cache_entry(cache_dir: Path, identity: ModelIdentity) -> Path:
    return cache_dir.resolve() / "entries" / f"{_ENTRY_PREFIX}{identity.key()}"


def cache_staging(cache_dir: Path) -> Path:
    root = cache_dir.resolve() / "staging"
    root.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=_STAGING_PREFIX, dir=root))


def discard_staging(path: Path | None) -> None:
    """Remove only a staging directory created by this process."""
    if path is not None and path.name.startswith(_STAGING_PREFIX):
        shutil.rmtree(path, ignore_errors=True)


def _pid_alive(pid: int) -> bool | None:
    """Return True/False if liveness is known, or None if it cannot be determined.

    Never uses elapsed time as evidence: a long-running download may hold a
    lock for many minutes. On Windows, OpenProcess gives a reliable answer; on
    POSIX, signal 0 probes the process without sending a signal.
    """
    if not isinstance(pid, int) or pid <= 0:
        return None
    if os.name == "nt":
        import ctypes

        process_query_limited_information = 0x1000
        still_active = 259
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        handle = kernel32.OpenProcess(process_query_limited_information, False, pid)
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            if kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return code.value == still_active
            return None
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return None
    return True


def _lock_owner_dead(lock: Path) -> bool:
    """True only when the lock's recorded process is confirmed not running."""
    try:
        owner = int(lock.read_text(encoding="ascii").strip())
    except (OSError, UnicodeError, ValueError):
        return False  # unknown owner: fail closed, never reclaim
    return _pid_alive(owner) is False


@contextmanager
def cache_lock(cache_dir: Path, identity: ModelIdentity, timeout: float = 120.0) -> Iterator[None]:
    """Cross-platform exclusive lock file; prevents concurrent publication.

    A lock is reclaimed only if its owner PID is confirmed dead. Age alone
    never invalidates a lock.
    """
    locks = cache_dir.resolve() / "locks"
    locks.mkdir(parents=True, exist_ok=True)
    lock = locks / f"{identity.key()}.lock"
    deadline = time.monotonic() + max(0.0, timeout)
    acquired = False
    while not acquired:
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode("ascii"))
            os.close(fd)
            acquired = True
        except FileExistsError:
            try:
                if _lock_owner_dead(lock):
                    lock.unlink(missing_ok=True)
                    continue
            except FileNotFoundError:
                continue
            if time.monotonic() >= deadline:
                raise CacheVerificationError("cache_busy", "model cache is locked by another process")
            time.sleep(0.05)
    try:
        yield
    finally:
        lock.unlink(missing_ok=True)


@contextmanager
def mlflow_network_settings(settings: Settings) -> Iterator[None]:
    """Apply only supported MLflow HTTP settings for the duration of a download.

    Values are restored afterwards, so globals are never left modified and no
    request path changes them. Unset settings are not injected at all.
    """
    names = {
        "MLFLOW_HTTP_REQUEST_TIMEOUT": settings.mlflow_http_request_timeout,
        "MLFLOW_HTTP_REQUEST_MAX_RETRIES": settings.mlflow_http_request_max_retries,
        "MLFLOW_HTTP_REQUEST_BACKOFF_FACTOR": settings.mlflow_http_request_backoff_factor,
    }
    applied = {key: value for key, value in names.items() if value is not None}
    previous = {key: os.environ.get(key) for key in applied}
    for key, value in applied.items():
        os.environ[key] = str(value)
    try:
        yield
    finally:
        for key, old in previous.items():
            if old is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old


def _safe_file(package: Path, relative: str) -> Path:
    """Resolve a required relative path, rejecting traversal and symlinks."""
    if not relative or Path(relative).is_absolute():
        raise CacheVerificationError("invalid_package", "model package contains an unsafe path")
    path = (package / relative).resolve()
    try:
        path.relative_to(package.resolve())
    except ValueError as error:
        raise CacheVerificationError("invalid_package", "model package path escapes its root") from error
    if path.is_symlink() or not path.is_file():
        raise CacheVerificationError("missing_model_file", f"required model file is missing: {relative}")
    return path


def _read_yaml(path: Path, code: str) -> dict:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise CacheVerificationError(code, "model package metadata is unreadable") from error
    if not isinstance(data, dict):
        raise CacheVerificationError(code, "model package metadata is invalid")
    return data


def _load_mlmodel(package: Path):
    try:
        return Model.load(str(package))
    except Exception as error:
        raise CacheVerificationError("invalid_mlmodel", "MLmodel metadata is invalid") from error


def verify_model_package(package: Path, identity: ModelIdentity) -> None:
    """Fail-closed verification of a complete MLflow PythonModel package."""
    package = Path(package)
    if package.is_symlink() or not package.is_dir():
        raise CacheVerificationError("invalid_package", "model package is not a directory")
    meta = _load_mlmodel(package)
    flavor = (meta.flavors or {}).get("python_function")
    if not isinstance(flavor, dict):
        raise CacheVerificationError("invalid_mlmodel", "python_function flavor is missing")
    python_model = flavor.get("python_model") or "python_model.pkl"
    code_dir = flavor.get("code") or "code"
    artifacts = flavor.get("artifacts") or {}
    if not isinstance(artifacts, dict) or "checkpoint" not in artifacts:
        raise CacheVerificationError("invalid_mlmodel", "checkpoint artifact is missing")
    checkpoint_path = (artifacts["checkpoint"] or {}).get("path")
    if not isinstance(checkpoint_path, str):
        raise CacheVerificationError("invalid_mlmodel", "checkpoint artifact path is invalid")
    adapter = f"{code_dir}/src/models/registry_adapter.py".replace("//", "/")

    required = ["MLmodel", python_model, checkpoint_path, adapter, REGISTRY_META_FILE]
    env = flavor.get("env") or {}
    if isinstance(env, dict):
        required.extend(value for value in (env.get("conda"), env.get("virtualenv")) if isinstance(value, str))
    files = {name: _safe_file(package, name) for name in dict.fromkeys(required)}

    registry_meta = _read_yaml(files[REGISTRY_META_FILE], "wrong_version")
    if (str(registry_meta.get("model_name")) != identity.name or
            str(registry_meta.get("model_version")) != identity.version):
        raise CacheVerificationError("wrong_version", "cached package version does not match the champion")

    if _sha256(files[checkpoint_path]) != identity.checkpoint_sha256:
        raise CacheVerificationError("checkpoint_hash_mismatch", "checkpoint hash does not match registry provenance")
    if _sha256(files[adapter]) != identity.packaging_code_sha256:
        raise CacheVerificationError("adapter_hash_mismatch", "adapter hash does not match registry provenance")

    identity_path = package / IDENTITY_FILE
    if identity_path.is_file():
        expected = identity.as_dict()
        recorded = _read_yaml(identity_path, "wrong_identity") if identity_path.suffix == ".yaml" else None
        if recorded is None:
            try:
                recorded = json.loads(identity_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, ValueError) as error:
                raise CacheVerificationError("wrong_identity", "cache identity is unreadable") from error
        if not isinstance(recorded, dict) or any(recorded.get(key) != value for key, value in expected.items()):
            raise CacheVerificationError("wrong_identity", "cache identity does not match the resolved version")

    manifest_path = package / MANIFEST_FILE
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, ValueError) as error:
            raise CacheVerificationError("manifest_invalid", "cache manifest is unreadable") from error
        entries = manifest.get("files") if isinstance(manifest, dict) else None
        if not isinstance(entries, dict) or not entries:
            raise CacheVerificationError("manifest_invalid", "cache manifest is invalid")
        actual = {
            str(path.relative_to(package.resolve())).replace(os.sep, "/")
            for path in package.rglob("*")
            if path.is_file() and not path.is_symlink() and not _is_bytecode_cache(path)
        } - {MANIFEST_FILE, IDENTITY_FILE}
        if set(entries) != actual:
            raise CacheVerificationError("manifest_mismatch", "cache file set differs from its manifest")
        for name, digest in entries.items():
            if _sha256(_safe_file(package, name)) != digest:
                raise CacheVerificationError("manifest_mismatch", "cached file hash differs from its manifest")

def _is_bytecode_cache(path: Path) -> bool:
    """Python writes __pycache__/*.pyc next to loaded code; they are not part of the model package."""
    return "__pycache__" in path.parts or path.suffix == ".pyc"
def _write_metadata(package: Path, identity: ModelIdentity) -> None:
    (package / IDENTITY_FILE).write_text(
        json.dumps(identity.as_dict(), sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    files = {
        str(path.relative_to(package.resolve())).replace(os.sep, "/"): _sha256(path)
        for path in package.rglob("*") if path.is_file() and not path.is_symlink()
        and path.name not in {IDENTITY_FILE, MANIFEST_FILE} and not _is_bytecode_cache(path)
    }
    (package / MANIFEST_FILE).write_text(
        json.dumps({"files": files}, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")


def publish(staging: Path, entry: Path, identity: ModelIdentity) -> Path:
    """Publish a verified package atomically. Staging and entry share a filesystem."""
    staging = Path(staging)
    if staging.is_symlink() or not staging.is_dir():
        raise CacheVerificationError("invalid_package", "staged package is not a directory")
    verify_model_package(staging, identity)
    _write_metadata(staging, identity)
    entry.parent.mkdir(parents=True, exist_ok=True)
    if entry.exists():
        quarantine = entry.with_name(f"{entry.name}.invalid-{os.getpid()}-{time.time_ns()}")
        entry.rename(quarantine)
    os.replace(staging, entry)
    return entry


def import_candidate(candidate: Path, identity: ModelIdentity, cache_dir: Path) -> Path:
    """Verify and copy an existing local package into the persistent cache.

    The source directory is never moved, modified, or deleted.
    """
    candidate = Path(candidate).resolve()
    verify_model_package(candidate, identity)
    with cache_lock(cache_dir, identity):
        entry = cache_entry(cache_dir, identity)
        staging = cache_staging(cache_dir)
        try:
            shutil.copytree(candidate, staging, dirs_exist_ok=True, symlinks=False)
            return publish(staging, entry, identity)
        finally:
            discard_staging(staging)


def main(argv: list[str] | None = None) -> None:
    """Operator helper: verify a local copy and import it into the cache."""
    import argparse

    from src.api.config import get_settings
    from src.api.model_service import resolve_champion

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True, type=Path,
                        help="Complete local model package directory to verify and import")
    parser.add_argument("--image", type=Path,
                        help="Optional local image for a final read-only inference probe")
    args = parser.parse_args(argv)

    settings = get_settings()
    version = resolve_champion(settings)
    identity = identity_from_version(settings.model_name, version)
    if identity is None:
        raise SystemExit("registry version lacks verified provenance tags; refusing to import")
    entry = import_candidate(args.candidate, identity, cache_root(settings))
    print(json.dumps({"version": identity.version, "cache_key": identity.key(),
                      "entry": str(entry)}, indent=2))
    if args.image:
        from src.api.model_service import load_champion
        service = load_champion(settings)
        result = service.predict(args.image)
        print(json.dumps({"version": service.info.version, "counts": result.counts.model_dump()}, indent=2))


if __name__ == "__main__":
    main()
