"""Resumable checkpoints for long training runs (laptop sleep, thermal throttling, crashes).

Usage pattern in a training loop::

    ckpt = Checkpointer(run_dir / "checkpoints", every_n_steps=200)
    state = ckpt.load_latest()            # None on a fresh run
    start = 0 if state is None else state["step"] + 1
    if state: model.load_state_dict(state["model"]); opt.load_state_dict(state["optimizer"])
    for step in range(start, n_steps):
        ...
        ckpt.maybe_save(step, model=model.state_dict(), optimizer=opt.state_dict())

Saves are atomic (write to a temp file, then rename), so a run killed mid-save never leaves a
corrupt "latest" file. Python, NumPy and torch RNG states are stored and restored, so a resumed
run continues the same random stream. Checkpoints live under ``runs/`` (git-ignored).
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np


class Checkpointer:
    def __init__(self, directory: str | Path, every_n_steps: int = 100, keep_last: int = 2):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.every, self.keep = every_n_steps, keep_last

    def _paths(self) -> list[Path]:
        return sorted(self.dir.glob("step_*.pt"), key=lambda p: int(p.stem.split("_")[1]))

    def save(self, step: int, **state) -> Path:
        import torch

        payload = {
            "step": step,
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),  # noqa: NPY002 -- legacy global RNG state
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
            **state,
        }
        path = self.dir / f"step_{step:09d}.pt"
        tmp = path.with_suffix(".tmp")
        torch.save(payload, tmp)
        os.replace(tmp, path)
        for old in self._paths()[: -self.keep]:
            old.unlink(missing_ok=True)
        return path

    def maybe_save(self, step: int, **state) -> Path | None:
        return self.save(step, **state) if (step + 1) % self.every == 0 else None

    def load_latest(self, map_location=None, restore_rng: bool = True) -> dict | None:
        import torch

        paths = self._paths()
        if not paths:
            return None
        state = torch.load(paths[-1], map_location=map_location, weights_only=False)
        if restore_rng:
            rng = state["rng"]
            random.setstate(rng["python"])
            np.random.set_state(rng["numpy"])  # noqa: NPY002
            torch.set_rng_state(rng["torch"])
            if rng["cuda"] is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(rng["cuda"])
        return state
