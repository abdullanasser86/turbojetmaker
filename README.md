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
Phase 4 (training dataset), round 2: the option-C pilot (`phase4_v2`) was STOPPED by the agreed stop rule
(5.5 % feasible < 10 %); the full run waits for a sampling decision. See `docs/PHASE4_REPORT.md` and the dataset cards
`docs/dataset_cards/phase4_v1.md`, `phase4_v2.md` (earlier: `docs/PHASE0_REPORT.md` … `docs/PHASE3_REPORT.md`).

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

## Laptop setup (RTX 4080 Laptop GPU, 12 GB)

The lock file pins **torch 2.10.0 built for CUDA 12.8** (Linux x86-64 wheels from PyPI), with
GPyTorch 1.15.2 and BoTorch 0.18.1. Any recent NVIDIA driver (≥ 570, CUDA 12.8 capable) works.
*Windows:* PyPI only ships CPU-only torch wheels for Windows, so a CUDA build needs the PyTorch
package index. This is not yet in the lock (see docs/PHASE3_REPORT.md, open question about your OS).

1. Install [uv](https://docs.astral.sh/uv/) and git.
2. Clone and create the environment:
   ```bash
   git clone https://github.com/abdullanasser86/turbojetmaker.git
   cd turbojetmaker
   uv sync --frozen            # exact versions from uv.lock (~3 GB incl. CUDA libraries)
   ```
3. Check the environment, GPU and ML stack (versions vs lock, GPU name/VRAM/driver/power limit,
   short GP and MLP benchmark with peak memory):
   ```bash
   uv run python scripts/check_env.py
   ```
4. Run the tests (fast set first, then everything):
   ```bash
   uv run pytest -m "not slow"
   uv run pytest
   ```
5. For long runs: plug into **AC power** and select the laptop's **performance / turbo power mode**.
   Expect lower sustained throughput than a desktop RTX 4080, and possible thermal throttling.
   Training writes resumable checkpoints (`turboml/checkpoint.py`), so a run interrupted by sleep
   or throttling can continue from its last checkpoint.

**Resource settings**
- Physics data generation runs on the CPU with `compute.n_workers` parallel processes
  (`configs/default.yaml`; default = physical cores − 1, so the laptop stays usable). Override per
  run with `TURBOML_N_WORKERS=4`.
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
