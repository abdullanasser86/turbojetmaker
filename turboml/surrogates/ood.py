"""Out-of-distribution (OOD) flags: when NOT to trust a surrogate prediction without a physics re-check.

Each flag is a boolean per query point (configs/surrogates.yaml `ood`):
* outside_bounds     -- a design variable lies outside the sampled box (unit-cube coordinate < 0 or > 1).
* extrapolated_tip   -- impeller tip speed above the top measured compressor speed line (415 m/s, D-060): the
                        PHYSICS model itself is extrapolating there, however well the surrogate fits the data.
* sparse_region      -- mean distance to the k nearest training inputs above a high quantile of the same quantity
                        inside the training set (leave-one-out), i.e. a region with little data.
* high_epistemic     -- the deep ensemble's members disagree more than at a high quantile of the training points.
Any flag set -> the design must be re-evaluated with the physics model before it is used (final results are always
physics-verified, CLAUDE.md).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class OODDetector:
    cfg: dict
    n_design_vars: int
    U_train: np.ndarray = None
    knn_threshold: float = float("nan")
    epi_threshold: np.ndarray = None

    def fit(self, U_train: np.ndarray, epistemic_train: np.ndarray | None = None) -> OODDetector:
        from sklearn.neighbors import NearestNeighbors

        self.U_train = U_train[:, : self.n_design_vars]
        k = self.cfg["knn_k"]
        self._nn = NearestNeighbors(n_neighbors=k + 1).fit(self.U_train)
        dist, _ = self._nn.kneighbors(self.U_train)
        self.knn_threshold = float(
            np.quantile(dist[:, 1:].mean(1), self.cfg["knn_quantile"])
        )  # leave-one-out
        if epistemic_train is not None:
            self.epi_threshold = np.quantile(
                epistemic_train, self.cfg["epistemic_quantile"], axis=0
            )
        return self

    def flags(self, U: np.ndarray, tip_speed_m_s: np.ndarray | None = None,
              epistemic: np.ndarray | None = None) -> dict:  # fmt: skip
        Ud = U[:, : self.n_design_vars]
        out = {"outside_bounds": ((Ud < -1e-9) | (Ud > 1 + 1e-9)).any(1)}
        if tip_speed_m_s is not None:
            out["extrapolated_tip"] = (
                np.nan_to_num(tip_speed_m_s, nan=0.0) > self.cfg["extrapolated_tip_speed_m_s"]
            )
        dist, _ = self._nn.kneighbors(Ud, n_neighbors=self.cfg["knn_k"])
        out["sparse_region"] = dist.mean(1) > self.knn_threshold
        if epistemic is not None and self.epi_threshold is not None:
            out["high_epistemic"] = (epistemic > self.epi_threshold).any(1)
        out["any"] = np.any(np.stack(list(out.values())), axis=0)
        return out
