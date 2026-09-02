"""Tests for the run logger.

The run log is the project's audit trail, so these tests check the
properties that make it trustworthy as evidence:

  * entries are contemporaneous (timestamps come from the machine, and the
    public API offers no way to set them),
  * the file is append-only (writing a new entry cannot alter an old one),
  * failed runs are recorded rather than silently dropped,
  * the recorded git state honestly describes the code that ran,
  * no entry leaks an identifying detail, because the submission is
    blind-judged.
"""

import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

from datasnoop.runlog import (
    RunLog,
    append_entry,
    find_repo_root,
    git_commit,
    git_is_dirty,
    read_log,
    utc_now_iso,
)

# A run entry must always carry these keys, whether the run succeeded or not.
REQUIRED_KEYS = {
    "schema_version",
    "experiment",
    "started_utc",
    "finished_utc",
    "runtime_seconds",
    "git_commit",
    "git_dirty",
    "python_version",
    "seed",
    "config",
    "outputs",
    "status",
    "error",
}

# Keys that must never appear: each one can carry a personal name, a school
# machine name, or a home directory into the repository.
FORBIDDEN_KEYS = {
    "hostname",
    "host",
    "user",
    "username",
    "cwd",
    "home",
    "path",
    "log_path",
    "repo_root",
}

HEX40 = re.compile(r"^[0-9a-f]{40}$")


def run_git(repo, *args):
    """Run a git command in `repo` with a fixed identity and no signing.

    The developer's global git config may sign commits; test fixtures must
    not depend on a signing key being available.
    """
    return subprocess.run(
        [
            "git",
            "-c", "user.name=test",
            "-c", "user.email=test@example.invalid",
            "-c", "commit.gpgsign=false",
            *args,
        ],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


@pytest.fixture
def temp_repo(tmp_path):
    """A git repository with one commit, a tracked file, and a logs/ dir."""
    repo = tmp_path / "repo"
    (repo / "logs").mkdir(parents=True)
    (repo / "tracked.txt").write_text("original\n")
    (repo / "logs" / ".gitkeep").write_text("")
    run_git(repo, "init", "-q", "-b", "main")
    run_git(repo, "add", "-A")
    run_git(repo, "commit", "-q", "-m", "initial")
    return repo


@pytest.fixture
def log_path(tmp_path):
    return tmp_path / "runs.jsonl"


# --- timestamps -----------------------------------------------------------


def test_utc_now_iso_is_utc_and_parses():
    stamp = utc_now_iso()
    parsed = datetime.fromisoformat(stamp)
    assert parsed.tzinfo is not None
    assert parsed.utcoffset().total_seconds() == 0
    assert stamp.endswith("+00:00")


def test_timestamps_bracket_the_run(log_path):
    before = datetime.now(timezone.utc)
    with RunLog("t", config={}, seed=0, log_path=log_path):
        pass
    after = datetime.now(timezone.utc)

    entry = read_log(log_path)[0]
    started = datetime.fromisoformat(entry["started_utc"])
    finished = datetime.fromisoformat(entry["finished_utc"])
    assert before <= started <= finished <= after


def test_runtime_seconds_reflects_actual_elapsed_time(log_path):
    pause = 0.05
    with RunLog("t", config={}, seed=0, log_path=log_path):
        time.sleep(pause)

    entry = read_log(log_path)[0]
    assert entry["runtime_seconds"] >= pause


def test_runlog_has_no_way_to_set_timestamps(log_path):
    """Entries cannot be backdated through the public API."""
    with pytest.raises(TypeError):
        RunLog(
            "t",
            config={},
            seed=0,
            log_path=log_path,
            started_utc="1999-01-01T00:00:00+00:00",
        )


# --- entry shape ----------------------------------------------------------


def test_entry_has_exactly_the_required_keys(log_path):
    with RunLog("demo", config={"a": 1}, seed=7, log_path=log_path):
        pass

    entry = read_log(log_path)[0]
    assert set(entry) == REQUIRED_KEYS
    assert entry["schema_version"] == 1
    assert entry["experiment"] == "demo"


def test_entry_leaks_no_identifying_field(log_path):
    with RunLog("demo", config={"a": 1}, seed=7, log_path=log_path):
        pass

    entry = read_log(log_path)[0]
    assert FORBIDDEN_KEYS.isdisjoint(entry)


def test_python_version_is_recorded(log_path):
    with RunLog("t", config={}, seed=0, log_path=log_path):
        pass

    entry = read_log(log_path)[0]
    assert entry["python_version"].startswith(
        f"{sys.version_info.major}.{sys.version_info.minor}"
    )


def test_config_and_seed_round_trip(log_path):
    config = {"n_series": 500, "grid": {"fast": [5, 10], "slow": [50, 200]}}
    with RunLog("t", config=config, seed=20260902, log_path=log_path):
        pass

    entry = read_log(log_path)[0]
    assert entry["config"] == config
    assert entry["seed"] == 20260902


def test_recorded_outputs_appear_in_the_entry(log_path):
    with RunLog("t", config={}, seed=0, log_path=log_path) as run:
        run.record(best_sharpe=1.25)
        run.record(n_rules=4096)

    entry = read_log(log_path)[0]
    assert entry["outputs"] == {"best_sharpe": 1.25, "n_rules": 4096}


# --- append-only ----------------------------------------------------------


def test_second_run_appends_and_leaves_the_first_line_untouched(log_path):
    with RunLog("first", config={}, seed=1, log_path=log_path):
        pass
    first_line = log_path.read_bytes().splitlines()[0]

    with RunLog("second", config={}, seed=2, log_path=log_path):
        pass

    lines = log_path.read_bytes().splitlines()
    assert len(lines) == 2
    assert lines[0] == first_line

    entries = read_log(log_path)
    assert [e["experiment"] for e in entries] == ["first", "second"]


def test_append_entry_writes_one_line_per_entry(log_path):
    append_entry(log_path, {"a": 1})
    append_entry(log_path, {"b": 2})
    assert len(log_path.read_text().splitlines()) == 2


def test_append_entry_creates_the_parent_directory(tmp_path):
    path = tmp_path / "nested" / "logs" / "runs.jsonl"
    append_entry(path, {"a": 1})
    assert read_log(path) == [{"a": 1}]


# --- failures are recorded ------------------------------------------------


def test_failed_run_is_logged_and_the_exception_propagates(log_path):
    with pytest.raises(ValueError, match="deliberate"):
        with RunLog("t", config={}, seed=0, log_path=log_path):
            raise ValueError("deliberate failure")

    entry = read_log(log_path)[0]
    assert entry["status"] == "failed"
    assert "deliberate failure" in entry["error"]
    assert "ValueError" in entry["error"]


def test_successful_run_has_status_ok_and_no_error(log_path):
    with RunLog("t", config={}, seed=0, log_path=log_path):
        pass

    entry = read_log(log_path)[0]
    assert entry["status"] == "ok"
    assert entry["error"] is None


# --- git state ------------------------------------------------------------


def test_find_repo_root_locates_the_repository(temp_repo):
    nested = temp_repo / "a" / "b"
    nested.mkdir(parents=True)
    assert find_repo_root(nested) == temp_repo


def test_find_repo_root_returns_none_outside_a_repository(tmp_path):
    outside = tmp_path / "not_a_repo"
    outside.mkdir()
    if find_repo_root(outside) is not None:
        pytest.skip("temp directory is itself inside a git repository")
    assert find_repo_root(outside) is None


def test_git_commit_is_a_full_hash_matching_head(temp_repo):
    recorded = git_commit(temp_repo)
    assert HEX40.match(recorded)
    assert recorded == run_git(temp_repo, "rev-parse", "HEAD")


def test_git_commit_is_none_outside_a_repository(tmp_path):
    outside = tmp_path / "not_a_repo"
    outside.mkdir()
    if find_repo_root(outside) is not None:
        pytest.skip("temp directory is itself inside a git repository")
    assert git_commit(outside) is None


def test_git_is_dirty_false_on_a_clean_tree(temp_repo):
    assert git_is_dirty(temp_repo) is False


def test_git_is_dirty_true_when_a_tracked_file_changes(temp_repo):
    (temp_repo / "tracked.txt").write_text("modified\n")
    assert git_is_dirty(temp_repo) is True


def test_git_is_dirty_true_for_an_untracked_source_file(temp_repo):
    """Uncommitted code makes a result irreproducible from the recorded hash."""
    (temp_repo / "new_module.py").write_text("x = 1\n")
    assert git_is_dirty(temp_repo) is True


def test_git_is_dirty_ignores_changes_under_logs(temp_repo):
    """A previous run's log line must not mark the next run as dirty."""
    (temp_repo / "logs" / "runs.jsonl").write_text('{"a": 1}\n')
    assert git_is_dirty(temp_repo) is False


def test_run_inside_a_repository_records_its_commit(temp_repo):
    path = temp_repo / "logs" / "runs.jsonl"
    with RunLog("t", config={}, seed=0, log_path=path, repo_root=temp_repo):
        pass

    entry = read_log(path)[0]
    assert entry["git_commit"] == run_git(temp_repo, "rev-parse", "HEAD")
    assert entry["git_dirty"] is False


# --- serialisation --------------------------------------------------------


def test_numpy_values_are_serialised(log_path):
    np = pytest.importorskip("numpy")
    with RunLog("t", config={}, seed=0, log_path=log_path) as run:
        run.record(mean=np.float64(0.5), counts=np.arange(3))

    entry = read_log(log_path)[0]
    assert entry["outputs"]["mean"] == 0.5
    assert entry["outputs"]["counts"] == [0, 1, 2]


def test_path_values_are_serialised_as_strings(log_path):
    append_entry(log_path, {"where": Path("results/figure.png")})
    assert read_log(log_path)[0]["where"] == "results/figure.png"


def test_unserialisable_value_raises_rather_than_being_dropped(log_path):
    with pytest.raises(TypeError):
        append_entry(log_path, {"bad": object()})


def test_each_line_is_standalone_json(log_path):
    with RunLog("t", config={"a": 1}, seed=0, log_path=log_path):
        pass
    for line in log_path.read_text().splitlines():
        json.loads(line)


def test_git_is_dirty_ignores_changes_under_results(temp_repo):
    """A run must not mark itself dirty by writing its own output.

    `git_dirty` answers a narrower question than "is the tree clean": was the
    code that produced this result committed? Figures and tables written by
    the run itself do not bear on that, so `results/` is excluded for the
    same reason `logs/` is.
    """
    (temp_repo / "results").mkdir()
    (temp_repo / "results" / "summary.csv").write_text("generator,rate\ngbm,0.05\n")
    assert git_is_dirty(temp_repo) is False


def test_git_is_dirty_still_sees_source_changes_alongside_results(temp_repo):
    """Excluding results must not hide an uncommitted code change."""
    (temp_repo / "results").mkdir()
    (temp_repo / "results" / "summary.csv").write_text("x\n")
    (temp_repo / "analysis.py").write_text("x = 1\n")
    assert git_is_dirty(temp_repo) is True
