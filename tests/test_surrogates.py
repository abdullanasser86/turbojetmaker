"""Phase 5 surrogate pipeline: input map, metrics, OOD flags, models on a small synthetic problem, data loading."""

from pathlib import Path

import numpy as np
import pytest

from turboml.surrogates import metrics
from turboml.surrogates.data import InputMap, Standardizer, load, load_surrogate_config
from turboml.surrogates.ood import OODDetector


def test_input_map_round_trip_and_out_of_box():
    im = InputMap(["a", "b", "c"], np.array([0.1, 350.0, 1000.0]), np.array([2.4, 500.0, 1350.0]),
                  ["log", "linear", "piecewise"], [None, None, [[0.85, 1150.0]]])  # fmt: skip
    rng = np.random.default_rng(0)
    U = rng.random((50, 3))
    assert np.allclose(im.to_unit(im.from_unit(U)), U)
    # 15 % of the unit interval above the knot maps to 1150-1350 K
    assert np.isclose(im.to_unit(np.array([[1.0, 400.0, 1150.0]]))[0, 2], 0.85)
    out = im.to_unit(np.array([[3.0, 520.0, 1400.0]]))
    assert (out > 1).all()


def test_calibration_metrics_on_a_calibrated_gaussian():
    rng = np.random.default_rng(1)
    mu, sd = np.zeros(20000), np.ones(20000)
    z = rng.normal(size=20000)
    c = metrics.calibration(z, mu, sd)
    assert (
        abs(c["coverage"]["0.95"] - 0.95) < 0.01
        and abs(c["std_of_standardized_residual"] - 1) < 0.02
    )
    over = metrics.calibration(z, mu, 0.5 * sd)  # overconfident model -> under-coverage
    assert over["coverage"]["0.95"] < 0.75


def test_classification_metrics():
    y = np.array([0, 0, 1, 1, 1, 0], bool)
    p = np.array([0.1, 0.2, 0.9, 0.8, 0.7, 0.3])
    m = metrics.classification(y, p)
    assert m["accuracy"] == 1.0 and m["roc_auc"] == 1.0 and m["brier"] < 0.1


def test_ood_flags():
    rng = np.random.default_rng(2)
    U = rng.random((400, 3))
    det = OODDetector(
        {
            "knn_k": 5,
            "knn_quantile": 0.99,
            "epistemic_quantile": 0.99,
            "extrapolated_tip_speed_m_s": 415.0,
        },
        3,
    )
    det.fit(U)
    q = np.array([[0.5, 0.5, 0.5], [1.4, 0.5, 0.5], [3.0, 3.0, 3.0]])
    f = det.flags(q, tip_speed_m_s=np.array([400.0, 420.0, 400.0]))
    assert list(f["outside_bounds"]) == [False, True, True]
    assert list(f["extrapolated_tip"]) == [False, True, False]
    assert f["sparse_region"][2] and not f["sparse_region"][0]


def test_ensemble_and_gp_fit_a_smooth_function():
    from turboml.surrogates.ensemble import EnsembleRegressor
    from turboml.surrogates.gp import GPSurrogate

    rng = np.random.default_rng(3)
    U = rng.random((200, 2))
    y = np.sin(3 * U[:, :1]) + U[:, 1:] ** 2
    st = Standardizer.fit(y)
    Z = st.forward(y)
    Uv = rng.random((50, 2))
    Zv = st.forward(np.sin(3 * Uv[:, :1]) + Uv[:, 1:] ** 2)
    gp = GPSurrogate(
        {"max_points_per_output": 1000, "training_iters": 80, "lr": 0.1, "jitter": 1e-6},
        device="cpu",
    ).fit(U, Z, log=lambda *_: None)
    mu, sd = gp.predict(Uv)
    assert np.sqrt(np.mean((mu - Zv) ** 2)) < 0.1
    ens = EnsembleRegressor(
        {"members": 2, "hidden": [32, 32], "epochs": 300, "lr": 0.01, "batch_size": 200},
        device="cpu",
    )
    ens.fit(U, Z, Uv, Zv, log=lambda *_: None)
    m2, s2, e2 = ens.predict(Uv)
    assert np.sqrt(np.mean((m2 - Zv) ** 2)) < 0.3 and (s2 >= e2).all()
    with pytest.raises(ValueError, match="exact-GP limit"):
        GPSurrogate(
            {"max_points_per_output": 10, "training_iters": 1, "lr": 0.1, "jitter": 1e-6},
            device="cpu",
        ).fit(U, Z)


@pytest.mark.skipif(
    not Path("datasets/phase4_v5/designs.csv").exists(), reason="dataset not generated here"
)
def test_load_v5_respects_frozen_test_split():
    import pandas as pd

    cfg = load_surrogate_config()
    d = load("phase4_v5", cfg)
    frozen = pd.read_csv("data/splits/phase4_v5_split.csv").set_index("design_id").split
    assert (frozen.loc[d.design_ids[d.split == "test"]] == "test").all()
    assert (frozen.loc[d.design_ids[d.split != "test"]] == "train_val").all()
    assert d.label == "provisional (pre-v6 calibration)"
    assert (
        np.nanmax(d.U[:, : d.n_design_vars]) <= 1 + 1e-9
        and np.nanmin(d.U[:, : d.n_design_vars]) >= -1e-9
    )
    # standardized targets on the training rows: mean ~0, std ~1
    Z = d.Z()[d.split == "train"]
    assert np.allclose(np.nanmean(Z, 0), 0, atol=1e-8) and np.allclose(
        np.nanstd(Z, 0), 1, atol=1e-8
    )
