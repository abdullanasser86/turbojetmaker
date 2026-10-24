# CLAUDE.md — conventions for Claude Code sessions on TurboML

Read `PROJECT_BRIEF.md` first: it is the source of truth for scope, phases and rules.

## Working agreement with the user
- The user is new to ML and propulsion: explain physics/ML in docstrings and keep
  `docs/LEARNING.md` (glossary + per-phase "what we did and why") up to date.
- **Stop at the end of every phase** with: what was built, test/validation results, open
  issues, decisions needed. Never start the next phase without an explicit "go".
- **Push to the GitHub remote at the end of every phase** (remote URL still to be provided by the user).
- Genuine design choices: give 2–3 options + recommendation, then wait.
- **Never invent** validation data, engine specs, material limits or paper results. Unknown
  values stay `null` in `configs/` until a source is found; ask the user for papers/data.

## Environment
- Python 3.11, `uv`. Install: `uv sync`. Run anything: `uv run <cmd>`.
- `uv.lock` is committed and must stay in sync (`uv lock --check`, also a pre-commit hook).
- Pins with reasons are in `pyproject.toml` comments and `docs/DECISIONS.md`
  (pyCycle git pin, numpy<2.4, TurboFlow dependency override). Re-check them before upgrading.
- This repo is developed partly in a CPU-only cloud container; the user's workstation has
  a CUDA GPU. Code must run on both (`device = "cuda" if torch.cuda.is_available() else "cpu"`).

## Commands
- Tests: `uv run pytest` (all) · `uv run pytest -m "not slow"` (fast, no external solvers)
- Lint/format: `uv run ruff check --fix . && uv run black .` (or `uv run pre-commit run -a`)
- Env report: `uv run python scripts/check_env.py`

## Code rules
- SI units internally; unit suffix in names when ambiguous (`n_rpm`, `omega_rad_s`,
  `tsfc_kg_per_N_s`). Convert at the boundary of external tools with `turboml.units`
  (pyCycle uses US customary units).
- Every physics function: docstring with equations, assumptions, source citation.
  Preliminary-design proxies (stress, mass, limits) are labelled as proxies.
- No hard-coded numbers that belong in config: bounds, limits, operating points live in
  `configs/*.yaml` (pydantic-validated in `turboml/config.py`; `require()` for nullable values).
- Randomness: pass `np.random.Generator` objects / seeds explicitly; runs call
  `turboml.runlog.start_run` (writes config + git hash + versions to `runs/`).
- Failed / non-converged simulations are data: record them with a failure reason; never
  silently clip, drop or "fix" them.
- Final results are always physics-verified, never surrogate-only.
- Training data come from the 0-D cycle (`cycle_simple`). Every generated dataset and every final
  Pareto set must pass `turboml.physics.crosscheck` (pyCycle-CEA on a random 2 % + all Pareto
  designs; flag > 1 % on Fn or TSFC; stop if > 5 % flagged). Report |ΔFn|/Fg only as a labelled
  secondary metric.
- Bounds live in `configs/bounds.yaml`, each with a source. Uploaded PDFs are never committed;
  commit extracted CSVs + parsers in `scripts/data/`. Digitized values are labelled as such.
- Training code is device-agnostic: use `turboml.device.get_device()`.
- `third_party/` holds verbatim upstream files — do not edit, wrap instead.
- Record every significant choice in `docs/DECISIONS.md`; validation evidence in
  `docs/VALIDATION.md`.

## Layout
`configs/` YAML · `turboml/` package (`physics/`, `data/`, `surrogates/`, `optimize/`,
`analysis/`) · `validation/` cases + reference outputs · `tests/` · `notebooks/` one per
phase · `docs/` · `scripts/` one-off/phase scripts.
