"""Verify the Python environment against uv.lock, report the GPU, and benchmark the ML stack.

Run on any machine (e.g. the RTX 4080 Laptop workstation) after `uv sync`:

    uv run python scripts/check_env.py            # report + benchmarks; exit code 1 on any problem
    uv run python scripts/check_env.py --all      # compare every locked package, not just key ones
    uv run python scripts/check_env.py --quick    # skip the benchmarks

Checks:
1. Installed versions of the key packages (or all packages) equal the versions in uv.lock.
2. GPU: name, VRAM, driver, CUDA runtime/driver version, current and max power limit
   (`nvidia-smi`), compute capability, torch's CUDA build.
3. Benchmarks on the auto-selected device (CUDA if available, else CPU):
   * exact GP (GPyTorch/BoTorch SingleTaskGP, float64, 3 inputs): fit time and peak memory,
   * MLP (float32, 20 -> 256 -> 256 -> 4, 20 epochs on 100k samples): time and peak memory.
   Peak memory = torch.cuda.max_memory_allocated on GPU, max RSS growth on CPU.
"""

from __future__ import annotations

import argparse
import platform
import resource
import shutil
import subprocess
import sys
import time
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
        if flag and not (have == "MISSING" and all_packages):  # all-mode: other-platform wheels
            problems.append(f"{n}: locked {want}, installed {have}")
        if not all_packages or flag:
            print(f"{n:22s} {want:14s} {have:14s}{flag}")
    return problems


def gpu_report() -> None:
    import torch

    print(f"\ntorch {torch.__version__}, built for CUDA {torch.version.cuda}, cuDNN "
          f"{torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else 'n/a'}")  # fmt: skip
    if shutil.which("nvidia-smi"):
        q = "name,memory.total,driver_version,power.limit,power.max_limit,temperature.gpu,clocks.max.sm"
        out = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader"],
                             capture_output=True, text=True, check=False).stdout.strip()  # fmt: skip
        cuda_drv = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, check=False
        ).stdout
        cuda_line = next((ln for ln in cuda_drv.splitlines() if "CUDA Version" in ln), "")
        print("nvidia-smi:", q)
        for line in out.splitlines():
            print("   ", line)
        if cuda_line:
            print("   ", cuda_line.strip(" |"))
    else:
        print("nvidia-smi not found (no NVIDIA driver visible)")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            print(f"  torch sees GPU {i}: {p.name}, {p.total_memory / 2**30:.1f} GiB, "
                  f"compute capability {p.major}.{p.minor}, {p.multi_processor_count} SMs")  # fmt: skip
    else:
        print("  torch.cuda.is_available() = False -> training code falls back to CPU")


def _peak_reset(device):
    import torch

    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        return None
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # KiB on Linux


def _peak_read(device, base) -> str:
    import torch

    if device.type == "cuda":
        torch.cuda.synchronize()
        return f"{torch.cuda.max_memory_allocated() / 2**20:.0f} MiB GPU"
    return f"+{(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - base) / 1024:.0f} MiB RSS"


def benchmarks() -> list[str]:
    import torch

    from turboml.device import get_device

    device = get_device()
    problems = []
    print(f"\nbenchmarks on {device}:")
    try:
        from botorch.fit import fit_gpytorch_mll
        from botorch.models import SingleTaskGP
        from gpytorch.mlls import ExactMarginalLogLikelihood

        for n in (500, 2000):
            torch.manual_seed(0)
            X = torch.rand(n, 3, dtype=torch.float64, device=device)
            Y = X.sin().sum(-1, keepdim=True) + 0.01 * torch.randn(
                n, 1, dtype=torch.float64, device=device
            )
            base = _peak_reset(device)
            t = time.perf_counter()
            gp = SingleTaskGP(X, Y)
            fit_gpytorch_mll(ExactMarginalLogLikelihood(gp.likelihood, gp))
            with torch.no_grad():
                gp.posterior(X[:256]).mean.sum().item()
            print(
                f"  exact GP float64, n={n:5d}: fit+predict {time.perf_counter() - t:6.2f} s, peak {_peak_read(device, base)}"
            )
    except Exception as e:  # noqa: BLE001
        problems.append(f"GP benchmark failed: {e!r}")
    try:
        torch.manual_seed(0)
        X = torch.randn(100_000, 20, device=device)
        Y = torch.randn(100_000, 4, device=device)
        net = torch.nn.Sequential(torch.nn.Linear(20, 256), torch.nn.SiLU(), torch.nn.Linear(256, 256),
                                  torch.nn.SiLU(), torch.nn.Linear(256, 4)).to(device)  # fmt: skip
        opt = torch.optim.Adam(net.parameters(), 1e-3)
        base = _peak_reset(device)
        t = time.perf_counter()
        for _ in range(20):
            for i in range(0, len(X), 1024):
                opt.zero_grad()
                loss = torch.nn.functional.mse_loss(net(X[i : i + 1024]), Y[i : i + 1024])
                loss.backward()
                opt.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        print(
            f"  MLP float32, 100k x 20 epochs: {time.perf_counter() - t:6.2f} s, peak {_peak_read(device, base)}"
        )
    except Exception as e:  # noqa: BLE001
        problems.append(f"MLP benchmark failed: {e!r}")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--all", action="store_true", help="compare every package in uv.lock")
    ap.add_argument("--quick", action="store_true", help="skip benchmarks")
    args = ap.parse_args()
    print(f"python {sys.version.split()[0]} ({platform.platform()})\n")
    problems = check_lock(args.all)
    gpu_report()
    if not args.quick:
        problems += benchmarks()
    print()
    if problems:
        print("PROBLEMS:\n  " + "\n  ".join(problems))
        print("Fix version mismatches with: uv sync --frozen")
        return 1
    print("environment matches uv.lock; ML stack OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
