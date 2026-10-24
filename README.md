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
Phase 2 (component meanline models) complete — see `docs/PHASE2_REPORT.md` (earlier: `docs/PHASE0_REPORT.md`, `docs/PHASE1_REPORT.md`).
The full plan is in `PROJECT_BRIEF.md`.

## Quick start
```bash
uv sync                                   # creates .venv from uv.lock
uv run python scripts/check_env.py        # versions + GPU check
uv run pytest                             # tests (use -m "not slow" to skip solver runs)
uv run jupyter lab notebooks/             # phase notebooks (or open them in VS Code)
```

## Documentation
- `docs/LEARNING.md` – glossary and "what we did and why" per phase
- `docs/DECISIONS.md` – architecture / dependency decisions
- `docs/VALIDATION.md` – what has been validated, against what, with what error
