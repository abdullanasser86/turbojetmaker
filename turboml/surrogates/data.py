"""Phase 5 data layer: turn a generated dataset (any version) into model-ready arrays.

What a surrogate sees
---------------------
* Inputs x: the dataset's SAMPLED design variables (read from the dataset config, so v5, v6, ... need no code
  change), mapped to [0, 1] with the same bounds and scale (linear / log / piecewise) the sampler used. Mapping back
  to the sampler's unit cube makes every input equally "wide" for the kernel or network, and a value outside [0, 1]
  means "outside the training box" (an out-of-distribution flag). Optionally the per-design compressor calibration
  factors (delta_sigma, diffuser_cf) and, from v6, the turbine loss multiplier (turbine_loss_multiplier), standardized,
  so model uncertainty can be propagated through the surrogate.
* Regression targets y: chosen quantities of DESIGNED engines at a converged operating point (configs/surrogates.yaml),
  log-transformed when strictly positive and multiplicative (thrust, SFC, mass), then standardized with the TRAINING
  mean/std. Non-converged points are not imputed: that row is simply missing for that target (failures are data,
  but a regression surrogate is fitted only where the physics produced a number).
* Classification label: feasible (designed AND mechanically feasible) for every sampled design; design failures are
  infeasible and keep their failure reason.

Splits: the frozen test ids (data/splits/<dataset>_split.csv, written before evaluation) are never used for fitting
or tuning. A seeded validation subset is drawn from the train_val ids for early stopping / model choice.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
SURROGATE_CONFIG = ROOT / "configs" / "surrogates.yaml"


def load_surrogate_config(path: str | Path = SURROGATE_CONFIG) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def dataset_config_path(dataset: str) -> Path:
    p = ROOT / "configs" / f"dataset_{dataset}.yaml"
    if not p.exists():
        raise FileNotFoundError(f"no dataset config {p}")
    return p


@dataclass
class InputMap:
    """Bijective map between physical design variables and the unit cube (the sampler's coordinates)."""

    names: list  # input keys as stored in the dataset (in_<key>)
    lo: np.ndarray
    hi: np.ndarray
    scale: list  # "linear" | "log" | "piecewise"
    knots: list  # per variable: None or [[u, value], ...]

    @classmethod
    def from_dataset(cls, dataset: str) -> InputMap:
        from turboml.data.generator import DatasetSpec

        spec = DatasetSpec.load(dataset_config_path(dataset))
        names, lo, hi, scale, knots = [], [], [], [], []
        for name, (lo_, hi_, sc, key) in spec.bounds.items():
            names.append(key)
            lo.append(lo_)
            hi.append(hi_)
            scale.append(sc)
            knots.append(
                spec.cfg["design_variables"][name].get("knots") if sc == "piecewise" else None
            )
        return cls(names, np.array(lo), np.array(hi), scale, knots)

    def to_unit(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, float)
        U = np.empty_like(X)
        for j in range(X.shape[1]):
            lo, hi, sc = self.lo[j], self.hi[j], self.scale[j]
            if sc == "log":
                U[:, j] = np.log(X[:, j] / lo) / np.log(hi / lo)
            elif sc == "piecewise":
                ks = self.knots[j]
                us, vs = [0.0, *[k[0] for k in ks], 1.0], [lo, *[k[1] for k in ks], hi]
                U[:, j] = np.interp(X[:, j], vs, us, left=np.nan, right=np.nan)
                # outside the box: extend the end segments linearly so out-of-box values map outside [0, 1]
                for side, (a, b) in (("lo", (0, 1)), ("hi", (-2, -1))):
                    m = X[:, j] < lo if side == "lo" else X[:, j] > hi
                    if m.any():
                        slope = (us[b] - us[a]) / (vs[b] - vs[a])
                        U[m, j] = us[a] + slope * (X[m, j] - vs[a])
            else:
                U[:, j] = (X[:, j] - lo) / (hi - lo)
        return U

    def from_unit(self, U: np.ndarray) -> np.ndarray:
        U = np.asarray(U, float)
        X = np.empty_like(U)
        for j in range(U.shape[1]):
            lo, hi, sc = self.lo[j], self.hi[j], self.scale[j]
            if sc == "log":
                X[:, j] = lo * (hi / lo) ** U[:, j]
            elif sc == "piecewise":
                ks = self.knots[j]
                X[:, j] = np.interp(
                    U[:, j], [0.0, *[k[0] for k in ks], 1.0], [lo, *[k[1] for k in ks], hi]
                )
            else:
                X[:, j] = lo + (hi - lo) * U[:, j]
        return X


@dataclass
class Standardizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, Y: np.ndarray) -> Standardizer:
        m = np.nanmean(Y, axis=0)
        s = np.nanstd(Y, axis=0)
        return cls(m, np.where(s > 0, s, 1.0))

    def forward(self, Y):
        return (Y - self.mean) / self.std

    def inverse(self, Z):
        return Z * self.std + self.mean

    def inverse_std(self, S):
        return S * self.std


@dataclass
class SurrogateData:
    """Model-ready arrays of one dataset (see module docstring)."""

    dataset: str
    label: str  # result label, e.g. "provisional (pre-v6 calibration)"
    design_ids: np.ndarray
    split: np.ndarray  # "train" | "val" | "test"
    X_raw: pd.DataFrame  # physical inputs (+ calibration factors)
    U: np.ndarray  # model inputs (unit-cube design variables + standardized calibration factors)
    input_names: list
    n_design_vars: int
    Y_raw: pd.DataFrame  # physical targets (NaN = not available)
    target_names: list
    transforms: list
    feasible: np.ndarray  # bool, all designs
    status: np.ndarray
    failure_reason: np.ndarray
    tip_speed_m_s: np.ndarray
    input_map: InputMap = field(repr=False, default=None)
    cal_std: Standardizer | None = field(repr=False, default=None)
    y_std: Standardizer | None = field(repr=False, default=None)

    def Z(self) -> np.ndarray:
        """Transformed + standardized targets (NaN where missing)."""
        return self.y_std.forward(self.transformed(self.Y_raw.to_numpy(float)))

    def transformed(self, Y: np.ndarray) -> np.ndarray:
        T = np.array(Y, float, copy=True)
        for j, t in enumerate(self.transforms):
            if t == "log":
                with np.errstate(divide="ignore", invalid="ignore"):
                    T[:, j] = np.where(T[:, j] > 0, np.log(T[:, j]), np.nan)
        return T

    def untransform(self, T: np.ndarray) -> np.ndarray:
        Y = np.array(T, float, copy=True)
        for j, t in enumerate(self.transforms):
            if t == "log":
                Y[:, j] = np.exp(Y[:, j])
        return Y

    def mask(self, which: str) -> np.ndarray:
        return self.split == which


def load(dataset: str, cfg: dict | None = None) -> SurrogateData:
    """Load datasets/<dataset>/designs.csv + points.csv and the frozen split; build inputs, targets and splits."""
    cfg = cfg or load_surrogate_config()
    dcfg = yaml.safe_load(dataset_config_path(dataset).read_text(encoding="utf-8"))
    out_dir = ROOT / dcfg["output_dir"]
    D = pd.read_csv(out_dir / "designs.csv").sort_values("design_id").reset_index(drop=True)
    P = pd.read_csv(out_dir / "points.csv")
    frozen = (
        pd.read_csv(ROOT / "data" / "splits" / f"{dataset}_split.csv").set_index("design_id").split
    )
    imap = InputMap.from_dataset(dataset)
    Xd = D[[f"in_{k}" for k in imap.names]].to_numpy(float)
    U = imap.to_unit(Xd)
    names = list(imap.names)
    X_raw = pd.DataFrame(Xd, columns=names)
    cal_std = None
    if cfg["inputs"].get("include_calibration_factors", False):
        cal_cols = [c for c in ("delta_sigma", "diffuser_cf", "turbine_loss_multiplier")
                    if c in D and D[c].notna().all()]  # fmt: skip
        C = D[cal_cols].to_numpy(float)
        X_raw[cal_cols] = C
    # split: frozen test; seeded validation subset of train_val
    rng = np.random.default_rng(cfg["seed"])
    split = np.where(D.design_id.map(frozen).to_numpy() == "test", "test", "train")
    tv = np.flatnonzero(split == "train")
    n_val = int(round(cfg["validation_fraction"] * len(tv)))
    split[rng.permutation(tv)[:n_val]] = "val"
    if cfg["inputs"].get("include_calibration_factors", False):
        cal_std = Standardizer.fit(C[split == "train"])
        U = np.hstack([U, cal_std.forward(C)])
        names += cal_cols
    # targets
    ok = (D.status == "ok").to_numpy()
    cols, transforms, tnames = {}, [], []
    for t in cfg["targets"]:
        if "design_column" in t:
            v = D[t["design_column"]].to_numpy(float)
        else:
            pt = P[P.point == t["point"]].set_index("design_id")
            v = D.design_id.map(pt[t["column"]].where(pt.converged.astype(bool))).to_numpy(float)
        v = np.where(ok, v, np.nan)
        cols[t["name"]] = v
        transforms.append(t.get("transform", "none"))
        tnames.append(t["name"])
    Y_raw = pd.DataFrame(cols)
    feas = ok & D.feasible.eq(True).to_numpy()  # NaN (design failed) -> False
    tip = (
        D["in_impeller_tip_speed_m_s"].to_numpy(float)
        if "in_impeller_tip_speed_m_s" in D
        else np.full(len(D), np.nan)
    )
    data = SurrogateData(dataset=dataset, label=cfg.get("result_labels", {}).get(dataset, ""),
                         design_ids=D.design_id.to_numpy(), split=split, X_raw=X_raw, U=U, input_names=names,
                         n_design_vars=len(imap.names), Y_raw=Y_raw, target_names=tnames, transforms=transforms,
                         feasible=feas, status=D.status.to_numpy(), failure_reason=D.failure_reason.fillna("").to_numpy(),
                         tip_speed_m_s=tip, input_map=imap, cal_std=cal_std)  # fmt: skip
    T = data.transformed(Y_raw.to_numpy(float))
    data.y_std = Standardizer.fit(T[split == "train"])
    return data
