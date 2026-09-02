"""Smoke test for the run log — the template every experiment script follows.

Does no science. It exists to (a) prove the logger writes a genuine entry
end to end, and (b) fix the shape that every later experiment copies:

    config in a dict at the top, never as literals buried in the code;
    one seed, used to build one explicit random generator;
    all work inside the RunLog block;
    key outputs handed to run.record().

Standard procedure: commit the code, run it, then commit the new log line.
That way `git_dirty` is false and the commit hash in the log is the code
that actually produced the numbers.

Run from anywhere in the repository:

    python experiments/smoke_test_runlog.py
"""

import numpy as np

from datasnoop.runlog import RunLog

# Every number the run depends on lives here, and the whole dict is copied
# into the log entry.
CONFIG = {
    "n_samples": 1000,
    "distribution": "standard normal",
}

# One seed for the whole run. np.random.default_rng is the current NumPy
# generator API; it is seeded explicitly rather than relying on global state,
# so two runs with the same seed give identical numbers.
SEED = 20260902


def main() -> None:
    with RunLog(experiment="smoke_test_runlog", config=CONFIG, seed=SEED) as run:
        rng = np.random.default_rng(SEED)
        sample = rng.standard_normal(CONFIG["n_samples"])

        # The sample mean of n standard normals has standard deviation
        # 1/sqrt(n), so for n = 1000 this should land within about 0.1 of
        # zero. A wildly different value means the seeding is not doing
        # what it should.
        run.record(
            sample_mean=float(sample.mean()),
            sample_std=float(sample.std(ddof=1)),
        )


if __name__ == "__main__":
    main()
