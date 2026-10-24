"""Verify the Python environment against uv.lock and check the GPU / ML stack.

Run this on any machine (e.g. the GPU workstation) after `uv sync`:

    uv run python scripts/check_env.py            # report + exit code 1 on any mismatch
    uv run python scripts/check_env.py --all      # compare every locked package, not just key ones

Checks:
1. Installed versions of the key packages (or all packages) equal the versions in uv.lock.
2. torch: CUDA build version, `torch.cuda.is_available()`, device names, a GPU matmul.
3. GPyTorch + BoTorch: fit a tiny SingleTaskGP on the auto-selected device (cuda if
   available, else cpu) to prove the training stack works end to end.
"""

from __future__ import annotations

import argparse
import platform
import sys
import tomllib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

LOCK = Path(__file__).resolve().parents[1] / "uv.lock"
KEY_PACKAGES = [
    "numpy", "scipy", "pandas", "openmdao", "om-pycycle", "radcompressor", "turboflow",
    "CoolProp", "torch", "gpytorch", "botorch", "linear-operator", "pymoo", "SALib",
    "scikit-learn", "pydantic",
]  # fmt: skip


def _norm(name: str) -> str:
    return name.lower().replace("_", "-")


def check_lock(all_packages: bool) -> list[str]:
    locked = {_norm(p["name"]): p["version"] for p in tomllib.loads(LOCK.read_text())["package"]}
    names = sorted(locked) if all_packages else [_norm(n) for n in KEY_PACKAGES]
    problems = []
    print(f"{'package':22s} {'locked':14s} {'installed':14s}")
    for n in names:
        want = locked.get(n)
        if want is None:
            continue  # e.g. platform-specific package not in this lock
        try:
            have = version(n)
        except PackageNotFoundError:
            have = "MISSING"
        flag = "" if have == want else "  <-- MISMATCH"
        if flag and (all_packages or n in {_norm(k) for k in KEY_PACKAGES}):
            if not (have == "MISSING" and all_packages):  # all-mode: other-platform wheels
                problems.append(f"{n}: locked {want}, installed {have}")
        if not all_packages or flag:
            print(f"{n:22s} {want:14s} {have:14s}{flag}")
    return problems


def check_torch_stack() -> list[str]:
    import torch

    problems = []
    cuda = torch.cuda.is_available()
    print(
        f"\ntorch {torch.__version__} built for CUDA {torch.version.cuda}; cuda available: {cuda}"
    )
    device = torch.device("cuda" if cuda else "cpu")
    if cuda:
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            print(
                f"  GPU {i}: {p.name}, {p.total_memory / 2**30:.1f} GiB, capability {p.major}.{p.minor}"
            )
        x = torch.randn(2048, 2048, device=device)
        torch.cuda.synchronize()
        print(f"  GPU matmul ok: {float((x @ x).abs().mean()):.3f}")
    else:
        print("  no CUDA device visible -> training code will run on CPU (device auto-select)")

    try:
        import botorch
        import gpytorch
        from botorch.fit import fit_gpytorch_mll
        from botorch.models import SingleTaskGP
        from gpytorch.mlls import ExactMarginalLogLikelihood

        torch.manual_seed(0)
        X = torch.rand(32, 3, dtype=torch.double, device=device)
        Y = X.sin().sum(-1, keepdim=True) + 0.01 * torch.randn(
            32, 1, dtype=torch.double, device=device
        )
        gp = SingleTaskGP(X, Y)
        fit_gpytorch_mll(ExactMarginalLogLikelihood(gp.likelihood, gp))
        with torch.no_grad():
            err = (gp.posterior(X).mean - Y).abs().max().item()
        print(f"gpytorch {gpytorch.__version__} / botorch {botorch.__version__}: "
              f"SingleTaskGP fit on {device} ok (max train residual {err:.3f})")  # fmt: skip
    except Exception as e:  # report, don't crash the whole check
        problems.append(f"GPyTorch/BoTorch smoke test failed: {e!r}")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--all", action="store_true", help="compare every package in uv.lock")
    args = ap.parse_args()
    print(f"python {sys.version.split()[0]} ({platform.platform()})\n")
    problems = check_lock(args.all) + check_torch_stack()
    print()
    if problems:
        print("PROBLEMS:\n  " + "\n  ".join(problems))
        print("Fix with: uv sync --frozen")
        return 1
    print("environment matches uv.lock; ML stack OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
