"""Append-only run log: the project's audit trail.

Every experiment run appends exactly one JSON object to a JSON Lines file
(one object per line, by default ``logs/runs.jsonl``). The line records when
the run happened, which commit of the code produced it, the full config, the
random seed, key outputs, and whether it succeeded.

Why JSON Lines rather than a database or a CSV: it is human-readable, it
diffs sensibly in git, appending a line cannot corrupt earlier lines, and
`json` in the standard library parses it. Nothing here imports a third-party
package, so every line of this module can be explained without reference to
library internals.

Two properties make the log usable as evidence rather than decoration:

  * **Contemporaneous.** Timestamps are taken by the machine inside
    :class:`RunLog`. There is no argument that sets them, so an entry cannot
    be backdated through this API.
  * **Honest about the code.** Each entry records the git commit hash and
    whether the working tree had uncommitted changes. A result produced by
    uncommitted code is not reproducible from the recorded hash, and the
    ``git_dirty`` flag says so.

Standard procedure: commit the code, run the experiment, commit the new log
line. That keeps ``git_dirty`` false and makes the recorded hash the code
that actually ran.

Typical use::

    from datasnoop.runlog import RunLog

    CONFIG = {"n_series": 1000}
    SEED = 20260902

    with RunLog(experiment="noise_baseline", config=CONFIG, seed=SEED) as run:
        best = search(...)
        run.record(best_sharpe=best)
"""

from __future__ import annotations

import json
import platform
import subprocess
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

__all__ = [
    "SCHEMA_VERSION",
    "DEFAULT_LOG_RELPATH",
    "RunLog",
    "append_entry",
    "find_repo_root",
    "git_commit",
    "git_is_dirty",
    "read_log",
    "utc_now_iso",
]

# Bump this if the set of fields in an entry ever changes, so that old and
# new lines in the same file can still be told apart when reading the log.
SCHEMA_VERSION = 1

DEFAULT_LOG_RELPATH = Path("logs") / "runs.jsonl"

# Changes under this directory do not count towards `git_dirty`: the log file
# itself lives here, and the previous run's line must not mark the next run
# as dirty.
DIRTY_EXCLUDE = ("logs",)


# --- time -----------------------------------------------------------------


def utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string ending in ``+00:00``.

    UTC and an explicit offset, not local time, so that entries from
    different machines or either side of a daylight-saving change sort and
    compare correctly.
    """
    return datetime.now(timezone.utc).isoformat()


# --- git ------------------------------------------------------------------


def _git(repo_root: Path | str, *args: str) -> str | None:
    """Run a git command in `repo_root`; return stdout, or None on failure.

    Returns None rather than raising when git is missing, the directory is
    not a repository, or the repository has no commits. A missing git state
    is recorded as null in the log; it never stops an experiment running.
    """
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=str(repo_root),
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def find_repo_root(start: Path | str | None = None) -> Path | None:
    """Return the root of the git repository containing `start`, or None."""
    start = Path.cwd() if start is None else Path(start)
    top = _git(start, "rev-parse", "--show-toplevel")
    return Path(top) if top else None


def git_commit(repo_root: Path | str) -> str | None:
    """Return the full 40-character hash of HEAD, or None if unavailable."""
    return _git(repo_root, "rev-parse", "HEAD") or None


def git_is_dirty(
    repo_root: Path | str, exclude: tuple[str, ...] = DIRTY_EXCLUDE
) -> bool:
    """Return True if the working tree has changes outside `exclude`.

    Untracked files count as changes. That is deliberate: a result produced
    by code that is not committed cannot be regenerated from the commit hash
    recorded alongside it.

    `exclude` entries are git pathspecs, passed with git's ``:(exclude)``
    magic prefix. The default excludes ``logs`` so the log file does not
    mark subsequent runs as dirty.
    """
    pathspecs = [".", *[f":(exclude){p}" for p in exclude]]
    status = _git(repo_root, "status", "--porcelain", "--", *pathspecs)
    if status is None:
        return False
    return status != ""


# --- reading and writing the log -----------------------------------------


def _json_default(value: Any) -> Any:
    """Convert values `json` cannot serialise on its own.

    Handles two cases only, so nothing is silently coerced:

      * ``Path`` becomes its string form;
      * anything with ``.tolist()`` (numpy scalars and arrays) becomes the
        equivalent plain Python value. ``numpy.float64(0.5).tolist()`` returns
        a float and ``numpy.arange(3).tolist()`` returns a list, so one branch
        covers both. numpy is not imported here; the check is on the object.

    Anything else raises TypeError, which propagates out of
    :func:`append_entry`. A value that cannot be recorded is a bug to fix,
    not something to drop from the audit trail.
    """
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(
        f"run log cannot serialise a value of type {type(value).__name__}; "
        "convert it to a plain Python value before recording it"
    )


def append_entry(path: Path | str, entry: dict[str, Any]) -> None:
    """Append one entry to the JSON Lines file at `path`.

    Opened in append mode, so an existing file is never rewritten and earlier
    lines cannot be altered by writing a new one. Keys are sorted so that two
    runs with the same content produce byte-identical lines.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(entry, sort_keys=True, default=_json_default)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def read_log(path: Path | str) -> list[dict[str, Any]]:
    """Return every entry in the log at `path`, in file order.

    Returns an empty list if the file does not exist. Blank lines are
    skipped; any other unparseable line raises, because a corrupt audit
    trail should be noticed rather than quietly skipped.
    """
    path = Path(path)
    if not path.exists():
        return []
    entries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            entries.append(json.loads(line))
    return entries


# --- the context manager --------------------------------------------------


class RunLog:
    """Context manager that writes one log entry per experiment run.

    `config` and `seed` are required, so no experiment can be logged without
    declaring what it was configured with and what made its randomness
    reproducible.

    A run that raises is still logged, with ``status`` set to ``"failed"``
    and the traceback's final line in ``error``; the exception is then
    re-raised. Runs that broke are part of the record.
    """

    def __init__(
        self,
        experiment: str,
        config: dict[str, Any],
        seed: int,
        *,
        log_path: Path | str | None = None,
        repo_root: Path | str | None = None,
    ) -> None:
        self.experiment = experiment
        self.config = config
        self.seed = seed
        self.outputs: dict[str, Any] = {}

        self.repo_root = (
            Path(repo_root) if repo_root is not None else find_repo_root()
        )
        if log_path is not None:
            self.log_path = Path(log_path)
        elif self.repo_root is not None:
            self.log_path = self.repo_root / DEFAULT_LOG_RELPATH
        else:
            self.log_path = Path(DEFAULT_LOG_RELPATH)

        self._started_utc: str | None = None
        self._started_monotonic: float | None = None

    def record(self, **outputs: Any) -> None:
        """Add key outputs to the entry. Call as often as needed."""
        self.outputs.update(outputs)

    def __enter__(self) -> "RunLog":
        self._started_utc = utc_now_iso()
        # A monotonic clock for the duration, so that a system clock
        # adjustment during a long run cannot produce a negative runtime.
        self._started_monotonic = time.monotonic()
        return self

    def __exit__(self, exc_type, exc_value, exc_tb) -> bool:
        runtime = time.monotonic() - self._started_monotonic

        if exc_type is None:
            status, error = "ok", None
        else:
            status = "failed"
            error = traceback.format_exception_only(exc_type, exc_value)[-1].strip()

        # Deliberately absent: hostname, username, working directory, and any
        # absolute path. Each can carry a personal or machine name into the
        # repository, which the project's anonymity rule forbids.
        entry = {
            "schema_version": SCHEMA_VERSION,
            "experiment": self.experiment,
            "started_utc": self._started_utc,
            "finished_utc": utc_now_iso(),
            "runtime_seconds": round(runtime, 3),
            "git_commit": git_commit(self.repo_root) if self.repo_root else None,
            "git_dirty": git_is_dirty(self.repo_root) if self.repo_root else None,
            "python_version": platform.python_version(),
            "seed": self.seed,
            "config": self.config,
            "outputs": self.outputs,
            "status": status,
            "error": error,
        }
        append_entry(self.log_path, entry)

        # False, so an exception raised inside the block propagates.
        return False
