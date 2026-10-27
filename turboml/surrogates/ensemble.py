"""Deep ensembles: M independently initialised neural networks (Lakshminarayanan, Pritzel & Blundell 2017, "Simple and
scalable predictive uncertainty estimation using deep ensembles", NeurIPS).

Regression: each member outputs a mean and a variance per target and is trained with the Gaussian negative
log-likelihood (missing targets masked out). The ensemble prediction is the mixture: mean = average of member means;
variance = average member variance (ALEATORIC: noise the member thinks is in the data) + variance of the member means
(EPISTEMIC: disagreement between members, large where there was no training data).
Classification: each member outputs a logit; P(feasible) = average member probability; member disagreement is the
epistemic part.

Training: Adam, full or mini-batch, early stopping on the validation loss (best weights kept), checkpoint every
`checkpoint_every_epochs` so a long run on the laptop can resume. Device-agnostic.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _mlp(d_in: int, hidden: list, d_out: int):
    import torch.nn as nn

    layers, d = [], d_in
    for h in hidden:
        layers += [nn.Linear(d, h), nn.SiLU()]
        d = h
    layers.append(nn.Linear(d, d_out))
    return nn.Sequential(*layers)


class _Trainer:
    def __init__(self, cfg: dict, device=None, seed: int = 0, ckpt_dir: Path | None = None):
        from turboml.device import get_device

        self.cfg, self.device, self.seed, self.ckpt_dir = (
            cfg,
            device or get_device(),
            seed,
            ckpt_dir,
        )
        self.members = []

    def _fit_member(self, k, net, loss_fn, x, y, xv, yv, log):
        import torch

        opt = torch.optim.Adam(
            net.parameters(), lr=self.cfg["lr"], weight_decay=self.cfg.get("weight_decay", 0.0)
        )
        best, best_state, bs = float("inf"), None, self.cfg.get("batch_size", len(x))
        gen = torch.Generator(device="cpu").manual_seed(self.seed * 1000 + k)
        start = 0
        ck = self.ckpt_dir / f"member{k}.pt" if self.ckpt_dir else None
        if ck is not None and ck.exists():  # resume
            s = torch.load(ck, map_location=self.device)
            net.load_state_dict(s["net"])
            opt.load_state_dict(s["opt"])
            start, best, best_state = s["epoch"], s["best"], s["best_state"]
        for ep in range(start, self.cfg["epochs"]):
            net.train()
            perm = torch.randperm(len(x), generator=gen).to(self.device)
            for i in range(0, len(x), bs):
                idx = perm[i : i + bs]
                opt.zero_grad()
                loss = loss_fn(net(x[idx]), y[idx])
                loss.backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                v = float(loss_fn(net(xv), yv)) if len(xv) else float(loss)
            if v < best:
                best, best_state = v, {
                    kk: vv.detach().clone() for kk, vv in net.state_dict().items()
                }
            if ck is not None and (ep + 1) % self.cfg.get("checkpoint_every_epochs", 250) == 0:
                ck.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"net": net.state_dict(), "opt": opt.state_dict(), "epoch": ep + 1, "best": best,
                            "best_state": best_state}, ck)  # fmt: skip
        net.load_state_dict(best_state)
        net.eval()
        log(f"  member {k}: best validation loss {best:.4f}")
        return net


class EnsembleRegressor(_Trainer):
    """Deep ensemble with Gaussian heads for the standardized targets Z (NaN = missing, masked in the loss)."""

    def fit(self, U, Z, Uv, Zv, log=print):
        import torch

        torch.manual_seed(self.seed)
        d_out = Z.shape[1]
        x, xv = (torch.as_tensor(a, dtype=torch.float32, device=self.device) for a in (U, Uv))
        y, yv = (torch.as_tensor(a, dtype=torch.float32, device=self.device) for a in (Z, Zv))

        def nll(out, t):
            mu, logv = out[:, :d_out], out[:, d_out:].clamp(-10, 5)
            m = torch.isfinite(t)
            tt = torch.where(m, t, torch.zeros_like(t))
            per = 0.5 * (logv + (tt - mu) ** 2 / logv.exp())
            return (per * m).sum() / m.sum().clamp_min(1)

        self.members = []
        for k in range(self.cfg["members"]):
            torch.manual_seed(self.seed * 1000 + k)
            net = _mlp(U.shape[1], self.cfg["hidden"], 2 * d_out).to(self.device)
            self.members.append(self._fit_member(k, net, nll, x, y, xv, yv, log))
        self.d_out = d_out
        return self

    def predict(self, U):
        """(mean, total std, epistemic std) of the standardized targets."""
        import torch

        x = torch.as_tensor(U, dtype=torch.float32, device=self.device)
        mus, vs = [], []
        with torch.no_grad():
            for net in self.members:
                o = net(x)
                mus.append(o[:, : self.d_out].cpu().numpy())
                vs.append(o[:, self.d_out :].clamp(-10, 5).exp().cpu().numpy())
        mus, vs = np.stack(mus), np.stack(vs)
        mean = mus.mean(0)
        epi = mus.var(0)
        return mean, np.sqrt(vs.mean(0) + epi), np.sqrt(epi)

    def save(self, path: Path) -> None:
        import torch

        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save([m.state_dict() for m in self.members], path)


class EnsembleClassifier(_Trainer):
    """Deep-ensemble binary classifier (P(feasible))."""

    def fit(self, U, y, Uv, yv, log=print):
        import torch

        torch.manual_seed(self.seed)
        x, xv = (torch.as_tensor(a, dtype=torch.float32, device=self.device) for a in (U, Uv))
        t, tv = (
            torch.as_tensor(a.astype(np.float32), device=self.device).unsqueeze(1) for a in (y, yv)
        )
        bce = torch.nn.BCEWithLogitsLoss()
        self.members = []
        for k in range(self.cfg["ensemble_size"]):
            torch.manual_seed(self.seed * 1000 + k)
            net = _mlp(U.shape[1], self.cfg["hidden"], 1).to(self.device)
            self.members.append(self._fit_member(k, net, bce, x, t, xv, tv, log))
        return self

    def predict_proba(self, U):
        """(mean probability, std of member probabilities)."""
        import torch

        x = torch.as_tensor(U, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            p = np.stack([torch.sigmoid(n(x)).squeeze(1).cpu().numpy() for n in self.members])
        return p.mean(0), p.std(0)

    def save(self, path: Path) -> None:
        import torch

        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save([m.state_dict() for m in self.members], path)
