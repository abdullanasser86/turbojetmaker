"""Phase 4 dataset generator: sampling, frozen split, failure records, resumability."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from turboml.data.generator import (
    DatasetSpec,
    done_ids,
    evaluate_design,
    make_test_split,
    sample_designs,
)

SPEC = DatasetSpec.load()


def _load_script():
    p = Path("scripts/phase4/generate_dataset.py")
    s = importlib.util.spec_from_file_location("gen", p)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def test_sampling_deterministic_and_in_bounds():
    a, b = sample_designs(SPEC), sample_designs(SPEC)
    assert a == b and len(a) == SPEC.cfg["n_designs"]
    for name, (lo, hi, _scale, key) in SPEC.bounds.items():
        v = np.array([d["inputs"][key] for d in a])
        assert v.min() >= lo and v.max() <= hi, name
    # log-scaled mass flow: about half the designs below the geometric mean of the bounds
    lo, hi = SPEC.bounds["air_mass_flow_kg_s"][:2]
    w = np.array([d["inputs"]["W_air_kg_s"] for d in a])
    assert 0.45 < (w < np.sqrt(lo * hi)).mean() < 0.55


def test_calibration_sampling_fraction_and_band():
    a = sample_designs(SPEC)
    s = [d for d in a if d["calibration_sampled"]]
    assert 0.17 < len(s) / len(a) < 0.23
    lo, hi = SPEC.calibration["delta_sigma"]["band_inflated"]
    assert all(lo <= d["delta_sigma"] <= hi for d in s)
    nominal = [d for d in a if not d["calibration_sampled"]]
    assert all(d["delta_sigma"] == SPEC.calibration["delta_sigma"]["value"] for d in nominal)


def test_split_is_frozen_and_matches_committed_file():
    m = make_test_split(2000, 0.15, SPEC.cfg["seed"])
    assert m.sum() == 300 and np.array_equal(m, make_test_split(2000, 0.15, SPEC.cfg["seed"]))
    p = Path("data/splits/phase4_v1_split.csv")
    if p.exists():
        import pandas as pd

        committed = pd.read_csv(p)
        assert np.array_equal(committed.split.eq("test").to_numpy(), m)


def test_impossible_design_is_recorded_not_dropped():
    from turboml.physics.engine import load_engine_config

    rec = {"design_id": -1, "inputs": {"pr_c": 2.0, "W_air_kg_s": 1.9, "n_rpm": 220000.0, "tit_K": 1100.0, "turb_phi": 0.5,
                                       "turb_psi": 1.2, "turb_reaction": 0.4, "alt_m": 0.0, "mach": 0.0, "dT_isa_K": 0.0},
           "calibration_sampled": False, "delta_sigma": -0.1041, "diffuser_cf": 0.0937}  # fmt: skip
    r = evaluate_design(rec, load_engine_config(), SPEC.cfg["points"])
    assert r["status"] == "design_failed" and r["failure_reason"]
    assert set(r["points"]) == {p["name"] for p in SPEC.cfg["points"]}
    assert all(
        not p["converged"] and p["failure_reason"] == "design_failed" for p in r["points"].values()
    )


@pytest.mark.slow
def test_baseline_like_design_all_points():
    from turboml.physics.engine import load_engine_config

    cfg = load_engine_config()
    inputs = {
        k: v["value"] if isinstance(v, dict) else v for k, v in cfg["baseline_design"].items()
    }
    rec = {
        "design_id": -2,
        "inputs": inputs,
        "calibration_sampled": False,
        "delta_sigma": -0.1041,
        "diffuser_cf": 0.0937,
    }
    r = evaluate_design(rec, cfg, SPEC.cfg["points"])
    assert r["status"] == "ok"
    assert all(p["converged"] for p in r["points"].values())
    # At/Ah = 1, SF 1.5, 100 h, PF 0.25 (NGV 1330 K, above the MAR-M 247 data): see D-037, D-044
    assert r["mechanical"]["active_constraints"] == ["AN2", "TIT_ngv_above_material_data"]
    json.dumps(r)  # serialisable


def test_resume_helpers(tmp_path):
    gen = _load_script()
    f = tmp_path / "designs.jsonl"
    f.write_text('{"design_id": 0}\n{"design_id": 1}\n{"design_id": 2, "trunc', encoding="utf-8")
    gen.repair_tail(f)
    assert f.read_text(encoding="utf-8").endswith("\n")
    assert done_ids(f) == {0, 1}
