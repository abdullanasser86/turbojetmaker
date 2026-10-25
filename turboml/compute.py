"""CPU parallelism settings for physics data generation.

Physics evaluations run on the CPU in parallel worker processes. The worker count comes
from (in order): the ``TURBOML_N_WORKERS`` environment variable, ``compute.n_workers`` in
the config, then the default of (physical cores - 1, at least 1), so a laptop stays usable
while a dataset is being generated.
"""

from __future__ import annotations

import os


def physical_cores() -> int:
    from joblib import cpu_count

    return max(1, int(cpu_count(only_physical_cores=True)))


def n_workers(configured: int | None = None) -> int:
    env = os.environ.get("TURBOML_N_WORKERS")
    if env:
        return max(1, int(env))
    if configured is not None:
        return max(1, int(configured))
    return max(1, physical_cores() - 1)
