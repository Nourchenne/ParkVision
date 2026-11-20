"""Advisory file lock shared by every ParkVision registry writer."""

import contextlib
import os
import time
from pathlib import Path


@contextlib.contextmanager
def registry_lock(path, timeout=30):
    if not path:
        raise ValueError("registry write requires a configured shared lock file")
    path = Path(path)
    if not path.parent.is_dir():
        raise ValueError("shared lock directory does not exist")
    with path.open("a+b") as stream:
        if stream.seek(0, os.SEEK_END) == 0:
            stream.write(b"\0")
            stream.flush()
        deadline = time.monotonic() + timeout
        while True:
            try:
                if os.name == "nt":
                    import msvcrt

                    stream.seek(0)
                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except (OSError, BlockingIOError) as error:
                if time.monotonic() >= deadline:
                    raise TimeoutError("registry writer lock is busy") from error
                time.sleep(0.2)
        try:
            yield
        finally:
            if os.name == "nt":
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
