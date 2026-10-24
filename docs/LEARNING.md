# LEARNING.md — glossary and project log

A running notebook for learning the propulsion and ML behind TurboML. Each phase adds new
terms and a short "what we did and why".

---

## Glossary

### Engine and thermodynamics
- **Turbojet** – Air is compressed, heated by burning fuel, expanded through a turbine
  (which drives the compressor through a shared shaft) and finally accelerated through a
  nozzle. The jet's momentum gain produces thrust.
- **Station numbering** – Engineers label locations along the engine with numbers
  (e.g. 2 = compressor inlet, 3 = compressor exit, 4 = turbine inlet, 5 = turbine exit,
  9 = nozzle exit). Quantities are written like `Tt4` = total temperature at station 4.
- **Total vs static** – *Static* pressure/temperature is what a probe moving with the flow
  would feel. *Total* (stagnation) values are what you'd get by slowing the flow to rest
  without losses. Total = static + a contribution from flow speed. Cycle analysis mostly
  tracks totals.
- **Thrust (Fn, net thrust)** – Force produced by the engine [N]. *Gross* thrust (Fg) is
  jet momentum + pressure force at the nozzle exit. *Ram drag* (Fram) is the momentum of
  the incoming air at flight speed. Fn = Fg − Fram, so the two are equal for a static engine.
- **TSFC (thrust-specific fuel consumption)** – Fuel mass flow divided by thrust
  [kg/(N·s); micro-turbojet literature often uses g/(kN·s) or kg/(N·h)]. Lower is better.
  It is the jet-engine equivalent of "litres per 100 km".
- **Pressure ratio (PR, OPR)** – Compressor exit total pressure ÷ inlet total pressure.
  OPR = overall pressure ratio of the engine. Micro turbojets with one centrifugal stage
  have low PR (≈2 for the KJ-66 compressor at full speed, per the brief's reference) compared
  with large engines (13.5 in the pyCycle example).
- **TIT (turbine inlet temperature, Tt4)** – The hottest gas temperature the turbine sees.
  Higher TIT means more thrust per kg of air, but the turbine material limits it.
- **EGT (exhaust gas temperature)** – Temperature after the turbine. It's easy to measure,
  so hobby engines often use it as a limit.
- **FAR (fuel–air ratio)** – Fuel mass flow ÷ air mass flow. It follows from the combustor
  energy balance.
- **Isentropic efficiency (η_is)** – Ideal (loss-free) work ÷ actual work for a compressor
  (actual ÷ ideal for a turbine). It's a single number that sums up all aerodynamic losses.
- **Polytropic efficiency (η_p)** – The "small-stage" efficiency. It doesn't depend on
  pressure ratio, so it's better for comparing machines with different PR.
  pyCycle reports both, as `eta_a` (adiabatic = isentropic) and `eta_p`.
- **Total-to-total vs total-to-static efficiency (η_tt, η_ts)** – For turbines: η_ts counts
  the exit kinetic energy as lost, η_tt does not. TurboFlow reports both, in percent.
- **Corrected flow / corrected speed** – Mass flow and shaft speed rescaled by inlet
  pressure and temperature, so one compressor map works at any flight condition.
- **Compressor / turbine map** – Charts of PR and efficiency versus corrected flow, with one
  curve per corrected speed (a *speed line*). They describe component behaviour off-design.
- **Surge** – At too low flow for a given speed, the compressor flow breaks down and can
  violently reverse. The *surge line* bounds the usable left side of the map.
  **Surge margin** = how far the operating point sits from the surge line.
- **Choke** – At too high flow, some passage reaches sonic speed (Mach 1). Flow can't
  increase further, so this is the right-hand limit of the map.
- **Design point vs off-design** – The design point is the condition the engine is sized
  for (here: static, sea level). Off-design means the same fixed hardware at other speeds or
  flight conditions; the operating point then comes from *matching* the components.
- **Matching** – Finding the shaft speed, mass flow and pressures at which compressor,
  turbine and nozzle are all consistent: turbine power = compressor power / mechanical
  efficiency, and mass flows agree.
- **Meanline (1-D) model** – A model that analyses a turbomachine along one representative
  streamline at mid-span. Velocity triangles plus empirical *loss correlations* predict
  efficiency and PR from geometry. Far cheaper than CFD and standard for preliminary design.
- **Velocity triangle** – Vector diagram of absolute velocity, blade speed and relative
  velocity at a blade's inlet or exit. It sets the work done (Euler turbine equation).
- **Slip factor** – A centrifugal impeller doesn't turn the flow fully to the blade angle at
  its exit. The slip factor (≈0.8–0.9) measures that shortfall. It's correlation-based
  (e.g. Wiesner) and uncertain, so we treat it as an uncertain input.
- **Impeller tip speed (U2)** – Blade speed at the impeller outer radius = ω·r2. It drives
  pressure ratio and also centrifugal stress, so the material limits it.
- **AN²** – Annulus area × (shaft speed)². A standard proxy for turbine blade root stress.
- **ISA** – International Standard Atmosphere. At sea level it's 288.15 K and 101 325 Pa.
- **pyCycle / OpenMDAO** – NASA's cycle-analysis library (pyCycle), built on NASA's
  optimisation framework (OpenMDAO). It solves the engine's balance equations with Newton
  solvers and gives analytic derivatives.
- **CEA thermo** – NASA Chemical Equilibrium with Applications. pyCycle uses it for
  real-gas properties of air and combustion products.
- **CoolProp** – Open-source fluid-property library, used by RadComp and TurboFlow.

### ML and optimization (expanded from Phase 4 on)
- **Surrogate model** – A fast ML approximation of a slow physics model. Here it runs in
  microseconds instead of seconds.
- **Gaussian Process (GP)** – A surrogate that predicts a mean *and* an uncertainty for every
  input. The uncertainty grows away from training data. Works best with < ~5 000 samples.
- **Deep ensemble** – Several neural networks trained from different random starts. The
  spread of their predictions is an uncertainty estimate.
- **Calibration** – Whether stated uncertainty is honest: e.g. a "95 % interval" should
  contain the truth 95 % of the time.
- **Pareto front** – With several objectives (low TSFC *and* high thrust-to-weight), the
  set of designs where you can't improve one objective without worsening another.
- **Hypervolume** – The size of the objective-space region dominated by the current Pareto
  front. If it stops growing, the optimization has converged.
- **Bayesian optimization / qLogNEHVI** – Optimization that picks the next physics runs where
  the surrogate predicts the biggest expected hypervolume gain. "q" = a batch of points,
  "Log" = numerically stable version, "N" = handles noisy observations.
- **Active learning** – Choosing new training samples where the model is most uncertain or
  most useful, instead of sampling uniformly.
- **Sobol sequence** – A "quasi-random" sampling pattern that covers a design space more
  evenly than random sampling.
- **Sobol indices** – A sensitivity analysis: what fraction of output variance each input
  causes.

---

## Project log

### Phase 0 — Setup & verification (2026-10-24)
**What we did**
1. Checked current versions of every tool in the brief against PyPI/GitHub and built one
   reproducible environment (`uv.lock`).
2. Found and resolved three dependency conflicts (details in `docs/DECISIONS.md`):
   - The released pyCycle (4.4.0) crashes with current numpy, so we pin a newer git commit
     that has the fix.
   - numpy 2.4 breaks pyCycle's output code, so we use numpy 2.3.
   - TurboFlow claims to need numpy 1.x. We overrode that, then ran TurboFlow's own
     regression tests to prove the physics results are unchanged.
3. Ran three "hello world" checks: pyCycle's turbojet example, a RadComp compressor
   point (plus an air speed sweep), and a TurboFlow turbine point.
4. Scaffolded the repo: config system, units module, run logging, tests, lint hooks, docs.

**Why it matters:** every later phase stacks models on top of these tools. Proving the
tools run in a pinned environment, with regression tests, means a future upgrade can't
silently change our physics.

**How to read the pyCycle example output** (it's in US units):
the design point has W = 147.3 lbm/s air (66.8 kg/s), Fn = 11 800 lbf (52.5 kN), OPR = 13.5,
burner exit 2370 °R (1317 K) and TSFC = 0.7985 lbm/(lbf·h) (22.6 g/(kN·s)). That's a
full-size engine, roughly 100× the thrust we're designing for. Micro turbojets have much
lower pressure ratio and component efficiencies, so expect noticeably higher TSFC. Phase 1
adapts the model to that scale using sourced efficiencies.
