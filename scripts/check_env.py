"""Verify the Python environment against uv.lock, report the GPU, and benchmark the ML stack.

Run on any machine (e.g. the RTX 4080 Laptop workstation) after `uv sync`:

    uv run python scripts/check_env.py            # report + benchmarks; exit code 1 on any problem
    uv run python scripts/check_env.py --all      # compare every locked package, not just key ones
    uv run python scripts/check_env.py --quick    # skip the benchmarks

Checks:
1. Installed versions of the key packages (or all packages) equal the versions in uv.lock.
2. GPU: name, VRAM, driver, CUDA runtime/driver version, current and max power limit
   (`nvidia-smi`), compute capability, torch's CUDA build.
3. On Windows: torch must be the CUDA 12.x build from the PyTorch index (version ends in +cu12x, torch.version.cuda
   12.x) and must see the GPU; the CPU-only PyPI wheel is reported as a problem.
4. Benchmarks on the auto-selected device (CUDA if available, else CPU):
   * exact GP (GPyTorch/BoTorch SingleTaskGP, float64, 3 inputs): fit time and peak memory,
   * MLP (float32, 20 -> 256 -> 256 -> 4, 20 epochs on 100k samples): time and peak memory.
   Peak memory = torch.cuda.max_memory_allocated on GPU, peak RSS growth on CPU (psutil; `resource` on POSIX).
Works on Linux and Windows (no POSIX-only imports at module level).
"""

from __future__ import annotations

import argparse
import platform
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


def locked_versions() -> dict[str, set[str]]:
    """All versions uv.lock holds per package. A package can appear more than once with platform markers (torch:
    PyPI build everywhere except Windows, +cu128 build from the PyTorch index on Windows)."""
    out: dict[str, set[str]] = {}
    for p in tomllib.loads(LOCK.read_text(encoding="utf-8"))["package"]:
        out.setdefault(_norm(p["name"]), set()).add(p["version"])
    return out


def expected_version(name: str, versions: set[str], platform_name: str = sys.platform) -> str:
    """The locked version for this platform: on Windows the CUDA (+cu) torch build, elsewhere the plain one."""
    if len(versions) == 1:
        return next(iter(versions))
    cuda = sorted(v for v in versions if "+cu" in v)
    plain = sorted(v for v in versions if "+" not in v)
    if name == "torch" and platform_name == "win32" and cuda:
        return cuda[-1]
    return (plain or sorted(versions))[-1]


def check_lock(all_packages: bool) -> list[str]:
    locked = {n: expected_version(n, v) for n, v in locked_versions().items()}
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


def check_torch_build(platform_name: str = sys.platform) -> list[str]:
    """Windows: the installed torch must be a CUDA 12.x build and see a GPU (the laptop has an RTX 4080)."""
    import torch

    problems = []
    if platform_name == "win32" and not sys.flags.utf8_mode:
        problems.append("Python UTF-8 mode is off: run `setx PYTHONUTF8 1` and open a new terminal (docs and notebooks "
                        "read/write UTF-8 text; Windows' default code page is cp1252)")  # fmt: skip
    if platform_name == "win32":
        if "+cu" not in torch.__version__ or not (torch.version.cuda or "").startswith("12"):
            problems.append(f"torch {torch.__version__} (CUDA {torch.version.cuda}) is not the CUDA 12.x build: "
                            "run `uv sync --frozen` (uv.lock takes torch from the PyTorch cu128 index on Windows)")  # fmt: skip
        elif not torch.cuda.is_available():
            problems.append("CUDA torch installed but torch.cuda.is_available() is False: check the NVIDIA driver "
                            "(nvidia-smi must show CUDA >= 12.8)")  # fmt: skip
    return problems


def _rss_bytes() -> int:
    try:
        import psutil

        return psutil.Process().memory_info().rss
    except ImportError:  # POSIX fallback
        import resource

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


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
    return _rss_bytes()


def _peak_read(device, base) -> str:
    import torch

    if device.type == "cuda":
        torch.cuda.synchronize()
        return f"{torch.cuda.max_memory_allocated() / 2**20:.0f} MiB GPU"
    return f"+{(_rss_bytes() - base) / 2**20:.0f} MiB RSS"


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
    problems += check_torch_build()
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
