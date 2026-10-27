# TurboML

Machine-learning-assisted **preliminary design and multi-objective optimization of micro
turbojet engines** (50–1000 N thrust class: single spool, centrifugal compressor, annular
combustor, single-stage axial turbine, convergent nozzle).

Physics models (thermodynamic cycle + 1-D meanline components) generate data; fast ML
surrogates with uncertainty drive optimizers; every reported design is re-verified with the
physics models.

> **Scope / safety note.** This is a *preliminary / conceptual design* tool. Its outputs are
> **not manufacturing-ready**. Real micro turbines spin at 100 000+ rpm and need detailed 3-D
> aerodynamic, structural, rotordynamic, thermal and combustion analysis plus physical
> testing by qualified people before anything is built or run.

## Status
Phase 4 (training dataset) complete: `phase4_v5`, 2000 designs (compressor refit on Murray's measured speed lines,
designs above 415 m/s tip speed flagged "extrapolated", buildable-range turbine, 15 % high-TIT), 604 feasible, pyCycle
cross-check passed. A KJ-66 replica with the calibrated (not measured) turbine reproduces the idle EGT; the turbine
efficiency gap to Murray (+11–13 pts) is open. See `docs/PHASE4_REPORT.md` (round 5) and
`docs/dataset_cards/phase4_v5.md` (earlier: `docs/PHASE0_REPORT.md` … `docs/PHASE3_REPORT.md`). Phase 5 (surrogates)
starts after the user's go.

What exists now: a 0-D cycle (plus pyCycle-CEA cross-check), meanline compressor and axial-turbine models, a coupled
engine model with design mode (sizing) and off-design matching, turbine creep checks from NiDI No. 393 rupture data,
and a resumable dataset generator. Example: the ~150 N baseline engine's design card is in
`validation/phase3/baseline_design_card.md`. Every design card carries the scope/safety note above.
The full plan is in `PROJECT_BRIEF.md`.

Generate the dataset (resumable; `estimate_walltime.py` first measures how long it takes on your machine):
```bash
uv run python scripts/phase4/estimate_walltime.py
uv run python scripts/phase4/generate_dataset.py
uv run python scripts/phase4/crosscheck_dataset.py && uv run python scripts/phase4/make_dataset_card.py
```

## Quick start
```bash
uv sync                                   # creates .venv from uv.lock
uv run python scripts/check_env.py        # versions + GPU check
uv run pytest                             # tests (use -m "not slow" to skip solver runs)
uv run jupyter lab notebooks/             # phase notebooks (or open them in VS Code)
```

## Laptop setup (Windows 11, RTX 4080 Laptop GPU, 12 GB)

The lock file pins **torch 2.10.0 built for CUDA 12.8**: on Windows from the PyTorch package index
(`2.10.0+cu128`, configured in `pyproject.toml` → `[tool.uv.sources]`), on Linux from PyPI. GPyTorch 1.15.2 and
BoTorch 0.18.1 on both. Any recent NVIDIA driver (≥ 570, i.e. `nvidia-smi` shows CUDA ≥ 12.8) works.

**Windows 11 (PowerShell)**
1. Install git, then uv, and enable Python's UTF-8 mode once (Windows defaults to the cp1252 code page; our docs,
   cards and notebooks are UTF-8):
   ```powershell
   winget install --id Git.Git -e
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   setx PYTHONUTF8 1          # takes effect in NEW terminals
   ```
   Close and reopen PowerShell.
2. Clone and create the environment (uv installs Python 3.11 itself if needed):
   ```powershell
   git clone https://github.com/abdullanasser86/turbojetmaker.git
   cd turbojetmaker
   uv sync --frozen           # exact versions from uv.lock, incl. the CUDA torch build (~3 GB)
   ```
3. Check the environment. On Windows this also verifies that torch is the CUDA 12.x build, that it sees the GPU and
   that UTF-8 mode is on (exit code 1 and a PROBLEMS list otherwise):
   ```powershell
   uv run python scripts/check_env.py
   ```
4. Check that the dataset generator runs with parallel workers on Windows (spawn start method; ~1 min):
   ```powershell
   uv run python scripts/phase4/windows_smoke_test.py
   ```
5. Run the tests:
   ```powershell
   uv run pytest -m "not slow"
   uv run pytest
   ```
6. Long runs: plug into **AC power** and select the **performance / turbo power mode**. Choose the number of worker
   processes per run with an environment variable, e.g. `$env:TURBOML_N_WORKERS = "8"` (PowerShell) before the
   command. The dataset generator is resumable: after sleep, a crash or Ctrl-C, run the same command again.
   Training writes resumable checkpoints (`turboml/checkpoint.py`).

**Linux** (same steps with bash): `curl -LsSf https://astral.sh/uv/install.sh | sh`, `uv sync --frozen`,
`uv run python scripts/check_env.py`. No UTF-8 setting is needed.

**Parallel code and Windows.** Windows has no `fork`: worker processes start fresh and import the module again.
All parallel code uses joblib/loky (or a `spawn` pool) with top-level, picklable worker functions, and every script's
work sits behind `if __name__ == "__main__":`. Keep it that way in new scripts.

**Resource settings**
- Physics data generation runs on the CPU with `compute.n_workers` parallel processes
  (`configs/default.yaml`; default = physical cores − 1, so the laptop stays usable). Override per
  run with the `TURBOML_N_WORKERS` environment variable (PowerShell: `$env:TURBOML_N_WORKERS = "4"`).
- Training code picks CUDA automatically and falls back to CPU (`turboml/device.py`, override with
  `TURBOML_DEVICE=cpu`). GPs use float64, neural networks float32. Sized for 12 GB VRAM: exact GPs
  up to ~10k points per output, variational/sparse GPs or deep ensembles above that, ensemble
  members trained one after another if VRAM is tight.
- Generated datasets and model checkpoints are git-ignored. Seeds, configs and dataset cards are
  committed so everything can be regenerated.

## Documentation
- `docs/LEARNING.md` – glossary and "what we did and why" per phase
- `docs/DECISIONS.md` – architecture / dependency decisions
- `docs/VALIDATION.md` – what has been validated, against what, with what error
