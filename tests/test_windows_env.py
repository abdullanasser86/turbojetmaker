"""Windows support: platform-specific torch in uv.lock and the CUDA-build check of scripts/check_env.py.

These run on Linux too: the platform is passed in explicitly."""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("check_env", ROOT / "scripts" / "check_env.py")
check_env = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check_env)


def test_lock_has_cuda_torch_for_windows_and_pypi_torch_elsewhere():
    v = check_env.locked_versions()["torch"]
    assert v == {"2.10.0", "2.10.0+cu128"}
    assert check_env.expected_version("torch", v, "win32") == "2.10.0+cu128"
    assert check_env.expected_version("torch", v, "linux") == "2.10.0"
    lock = (ROOT / "uv.lock").read_text(encoding="utf-8")
    assert "download.pytorch.org/whl/cu128" in lock and "cp311-cp311-win_amd64" in lock


def test_cpu_torch_on_windows_is_a_problem(monkeypatch):
    import torch

    monkeypatch.setattr(torch, "__version__", "2.10.0+cpu")
    monkeypatch.setattr(torch.version, "cuda", None)
    probs = check_env.check_torch_build("win32")
    assert any("not the CUDA 12.x build" in p for p in probs)
    assert check_env.check_torch_build("linux") == []


def test_check_env_has_no_posix_only_imports_at_module_level():
    src = (ROOT / "scripts" / "check_env.py").read_text(encoding="utf-8")
    top = [ln for ln in src.splitlines() if ln.startswith(("import ", "from "))]
    assert not any("resource" in ln or "fcntl" in ln for ln in top)
