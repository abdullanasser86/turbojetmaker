"""Print versions of every key dependency and check GPU availability.

Usage:  uv run python scripts/check_env.py
"""

from __future__ import annotations

import platform
import sys
from importlib.metadata import PackageNotFoundError, version

PACKAGES = [
    "numpy",
    "scipy",
    "pandas",
    "openmdao",
    "om-pycycle",
    "radcompressor",
    "turboflow",
    "CoolProp",
    "torch",
    "gpytorch",
    "botorch",
    "pymoo",
    "SALib",
    "scikit-learn",
    "pydantic",
]


def main() -> int:
    print(f"python   {sys.version.split()[0]}  ({platform.platform()})")
    for pkg in PACKAGES:
        try:
            print(f"{pkg:14s} {version(pkg)}")
        except PackageNotFoundError:
            print(f"{pkg:14s} NOT INSTALLED")

    import torch

    cuda = torch.cuda.is_available()
    print(f"\ntorch built for CUDA {torch.version.cuda}; torch.cuda.is_available() = {cuda}")
    if cuda:
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        x = torch.randn(1024, 1024, device="cuda")
        print(f"  GPU matmul smoke test ok: {float((x @ x).sum()):.3e}")
    else:
        print("  No GPU visible: NN/BoTorch work will run on CPU (fine for development).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
