import json

import numpy as np

from turboml.config import load_config
from turboml.runlog import start_run


def test_start_run_writes_metadata_and_seeds(tmp_path):
    cfg = load_config()
    run_dir = start_run(cfg, "unit_test", root=tmp_path)
    assert (run_dir / "config.yaml").exists()
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["seed"] == cfg.seed and "commit" in meta["git"]
    a = np.random.rand()  # noqa: NPY002
    start_run(cfg, "unit_test_2", root=tmp_path)
    assert np.random.rand() == a  # noqa: NPY002 -- same seed -> same stream
