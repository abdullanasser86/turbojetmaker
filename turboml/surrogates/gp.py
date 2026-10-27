"""Exact Gaussian-process surrogates (one independent GP per target), gpytorch.

What a GP is (for this project): a probabilistic model that predicts, at a new design, a Gaussian distribution of the
output. Its mean interpolates the training data smoothly; its variance grows away from the data, so it says "I don't
know" where it has seen nothing. The kernel (Matern 5/2 with one length scale per input, "ARD") sets how fast the
output may vary along each input; the length scales and the noise level are fitted by maximising the marginal
likelihood (Rasmussen & Williams 2006, Gaussian Processes for Machine Learning, ch. 2 and 5).

Exact GPs cost O(n^3) time and O(n^2) memory; `max_points_per_output` (configs/surrogates.yaml, ~10k on a 12 GB GPU)
is enforced as an error, never by silent subsampling. Device-agnostic (turboml.device.get_device()).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _model_cls():
    import gpytorch

    class ExactGP(gpytorch.models.ExactGP):
        def __init__(self, x, y, likelihood, d):
            super().__init__(x, y, likelihood)
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=d)
            )

        def forward(self, x):
            return gpytorch.distributions.MultivariateNormal(
                self.mean_module(x), self.covar_module(x)
            )

    return ExactGP


class GPSurrogate:
    """Independent exact GPs for the standardized targets Z (NaN rows dropped per target)."""

    def __init__(self, cfg: dict, device=None, seed: int = 0):
        from turboml.device import get_device

        self.cfg, self.device, self.seed = cfg, device or get_device(), seed
        self.models, self.likelihoods = [], []

    def fit(self, U: np.ndarray, Z: np.ndarray, log=print) -> GPSurrogate:
        import gpytorch
        import torch

        torch.manual_seed(self.seed)
        cls = _model_cls()
        self.models, self.likelihoods, self.train_n = [], [], []
        for j in range(Z.shape[1]):
            m = np.isfinite(Z[:, j])
            n = int(m.sum())
            if n > self.cfg["max_points_per_output"]:
                raise ValueError(f"target {j}: {n} points > exact-GP limit {self.cfg['max_points_per_output']} "
                                 "(configs/surrogates.yaml gp.max_points_per_output); use the ensemble or a sparse GP")  # fmt: skip
            x = torch.as_tensor(U[m], dtype=torch.float64, device=self.device)
            y = torch.as_tensor(Z[m, j], dtype=torch.float64, device=self.device)
            lik = gpytorch.likelihoods.GaussianLikelihood(
                noise_constraint=gpytorch.constraints.GreaterThan(self.cfg["jitter"])
            ).to(self.device, torch.float64)
            model = cls(x, y, lik, U.shape[1]).to(self.device, torch.float64)
            model.train()
            lik.train()
            opt = torch.optim.Adam(model.parameters(), lr=self.cfg["lr"])
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(lik, model)
            for _ in range(self.cfg["training_iters"]):
                opt.zero_grad()
                loss = -mll(model(x), y)
                loss.backward()
                opt.step()
            model.eval()
            lik.eval()
            self.models.append(model)
            self.likelihoods.append(lik)
            self.train_n.append(n)
            log(f"  GP target {j}: n={n}, -mll/n={float(loss):.3f}")
        return self

    def predict(self, U: np.ndarray, include_noise: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """Mean and standard deviation of the standardized targets at U (n x d)."""
        import gpytorch
        import torch

        x = torch.as_tensor(U, dtype=torch.float64, device=self.device)
        mu, sd = [], []
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            for model, lik in zip(self.models, self.likelihoods, strict=True):
                f = model(x)
                p = lik(f) if include_noise else f
                mu.append(p.mean.cpu().numpy())
                sd.append(p.variance.clamp_min(0).sqrt().cpu().numpy())
        return np.stack(mu, 1), np.stack(sd, 1)

    def save(self, path: Path) -> None:
        import torch

        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state": [m.state_dict() for m in self.models], "train_n": self.train_n}, path)
