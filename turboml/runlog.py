"""Minimal run logging: every run gets a directory with config, git hash and results.

    runs/<UTC timestamp>_<name>/
        config.yaml      resolved config that was used
        meta.json        git commit, dirty flag, python + package versions, seed
        results.*        whatever the run writes

Kept deliberately simple (no MLflow) until we need experiment comparison dashboards.
"""

from __future__ import annotations

import json
import platform
import random
import subprocess
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
import yaml

from turboml.config import TurboMLConfig

REPO_ROOT = Path(__file__).resolve().parent.parent


def git_info() -> dict[str, str | bool]:
    def _git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=False
        ).stdout.strip()

    return {
        "commit": _git("rev-parse", "HEAD") or "none",
        "dirty": bool(_git("status", "--porcelain")),
    }


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 -- legacy global RNG used by third-party code
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def start_run(cfg: TurboMLConfig, name: str, root: Path = REPO_ROOT / "runs") -> Path:
    """Create the run directory, write config + metadata, seed RNGs. Returns the dir."""
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = root / f"{stamp}_{name}"
    run_dir.mkdir(parents=True, exist_ok=False)
    seed_everything(cfg.seed)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg.model_dump(), sort_keys=False))
    meta = {
        "name": name,
        "utc": stamp,
        "seed": cfg.seed,
        "git": git_info(),
        "python": platform.python_version(),
        "packages": {p: version(p) for p in ("numpy", "openmdao", "om-pycycle", "torch")},
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return run_dir
