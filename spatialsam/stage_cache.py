"""Atomic stage provenance, output validation, and recoverable cache replacement."""

import contextlib
import fcntl
import hashlib
import json
import logging
import os
import time
from pathlib import Path


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _inventory(paths):
    """Track exact file sets, sizes, and nanosecond modification times."""
    records = []
    for path in paths:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)
        files = sorted(path.rglob("*")) if path.is_dir() else [path]
        files = [file for file in files if file.is_file()]
        if not files:
            raise ValueError(f"Artifact is empty: {path}")
        for file in files:
            stat = file.stat()
            records.append((str(file.resolve()), stat.st_size, stat.st_mtime_ns))
    return records


def _atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


class StageCache:
    """Own only newly declared stage directories and retain stale generations."""

    def __init__(self, output_root: Path, source_root: Path):
        self.root = output_root
        self.state_root = output_root / "state"
        digest = hashlib.sha256()
        for package in (
            "spatialsam",
            "training",
            "predict",
            "sam2",
            "dataprocess",
            "utils",
            "config",
        ):
            for path in sorted((source_root / package).rglob("*")):
                if path.is_file() and path.suffix in (".py", ".yaml"):
                    digest.update(str(path.relative_to(source_root)).encode())
                    digest.update(path.read_bytes())
        self.source_digest = digest.hexdigest()

    @contextlib.contextmanager
    def lock(self):
        """Prevent two processes from writing the same run's stages."""
        self.state_root.mkdir(parents=True, exist_ok=True)
        with (self.state_root / "run.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def run(self, stage, executor, *, require_cached=False):
        """Execute missing stages; reject stale outputs when explicitly skipping."""
        record_path = self.state_root / (stage.name.replace("/", "__") + ".json")
        signature = _digest(
            {
                "inputs": _inventory(stage.inputs),
                "config": stage.config,
                "source": self.source_digest,
            }
        )
        previous = json.loads(record_path.read_text()) if record_path.exists() else None
        matching = previous is not None and previous.get("signature") == signature
        if matching and previous.get("status") == "completed":
            try:
                valid = previous["outputs"] == json.loads(
                    json.dumps(_inventory(stage.outputs.values()))
                )
            except (FileNotFoundError, ValueError):
                valid = False
            if valid:
                logging.info("Reusing stage %s", stage.name)
                return False
            matching = False
        if require_cached:
            raise ValueError(
                f"Stage {stage.name} has no verified cache; restart from this stage"
            )
        if stage.directory.exists() and not matching:
            if previous is None:
                raise ValueError(
                    f"Refusing to replace an untracked directory: {stage.directory}"
                )
            archive = (
                self.root
                / "stale"
                / f"{stage.name.replace('/', '__')}-{time.time_ns()}"
            )
            archive.parent.mkdir(parents=True, exist_ok=True)
            stage.directory.rename(archive)
        stage.directory.mkdir(parents=True, exist_ok=True)
        record = {
            "stage": stage.name,
            "signature": signature,
            "status": "running",
            "source": self.source_digest,
            "config": stage.config,
        }
        _atomic_json(record_path, record)
        try:
            logging.info("Running stage %s", stage.name)
            executor(stage)
            record["outputs"] = _inventory(stage.outputs.values())
            record["status"] = "completed"
            _atomic_json(record_path, record)
        except BaseException as error:
            record.update(status="failed", error=str(error))
            _atomic_json(record_path, record)
            raise
        return True
