"""Tests for CPU worker settings and resumable checkpoints."""

import random

import numpy as np
import pytest

from turboml.checkpoint import Checkpointer
from turboml.compute import n_workers, physical_cores


def test_n_workers_precedence(monkeypatch):
    monkeypatch.delenv("TURBOML_N_WORKERS", raising=False)
    assert n_workers() == max(1, physical_cores() - 1)
    assert n_workers(3) == 3
    assert n_workers(0) == 1
    monkeypatch.setenv("TURBOML_N_WORKERS", "5")
    assert n_workers(3) == 5


def test_checkpoint_roundtrip_and_rng(tmp_path):
    torch = pytest.importorskip("torch")
    ck = Checkpointer(tmp_path, every_n_steps=2, keep_last=2)
    assert ck.load_latest() is None
    net = torch.nn.Linear(3, 2)
    for step in range(6):
        ck.maybe_save(step, model=net.state_dict())
    assert sorted(p.name for p in tmp_path.glob("step_*.pt")) == [
        "step_000000003.pt",
        "step_000000005.pt",
    ]
    assert not list(tmp_path.glob("*.tmp"))
    # RNG state is restored: draws after loading equal draws right after saving
    ck.save(10, model=net.state_dict())
    expected = (random.random(), np.random.rand(), torch.rand(1).item())  # noqa: NPY002
    random.random(), np.random.rand(), torch.rand(1)  # noqa: NPY002 -- advance the streams
    state = ck.load_latest()
    assert state["step"] == 10
    assert (random.random(), np.random.rand(), torch.rand(1).item()) == expected  # noqa: NPY002
    net2 = torch.nn.Linear(3, 2)
    net2.load_state_dict(state["model"])
    assert torch.equal(net2.weight, net.weight)
