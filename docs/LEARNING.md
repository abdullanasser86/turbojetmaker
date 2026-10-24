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

### Added in Phase 1
- **Specific thrust** – Net thrust per unit air mass flow [N per kg/s]. High specific thrust
  means a small engine for a given thrust. In a turbojet it rises mainly with TIT.
- **LHV (lower heating value)** – Heat released by burning 1 kg of fuel completely, with the
  water left as vapour. For Jet A/A-1 it's about 43 MJ/kg (ASTM D1655 minimum 42.8).
- **Combustion efficiency η_b** – Fraction of the LHV actually released in the combustor.
  Small combustors burn less completely than large ones.
- **Mechanical efficiency η_m** – Fraction of turbine power that reaches the compressor after
  bearing and windage losses.
- **Intake pressure recovery** – Pt2/Pt0, the share of free-stream total pressure the intake keeps.
- **Nozzle velocity coefficient Cv** – Actual ÷ ideal jet velocity. It lumps nozzle friction
  losses into one number.
- **Choked nozzle** – When the nozzle pressure ratio Pt/Pambient exceeds ≈1.85, the throat
  reaches Mach 1. The exit pressure then stays above ambient, and the excess pushes as
  "pressure thrust" A·(Ps − Pamb).
- **Variable cp / NASA polynomials** – Gas specific heat rises with temperature (air: 1005
  J/(kg K) at 300 K, ~1140 at 1000 K). NASA publishes fitted polynomials for cp, h and s of each
  species; we use them for air and combustion products.
- **Absolute enthalpy** – Enthalpy measured including the chemical (formation) energy, so one
  energy balance covers combustion: fuel + air in, CO2 + H2O + leftover air out.
- **Equilibrium chemistry (CEA)** – NASA's CEA solves for the chemical equilibrium mixture,
  including dissociation (CO2 splitting into CO + O at high temperature). That's negligible below
  ~1500 K, which is why our simpler "complete combustion" model matches it.
- **Geometric vs geopotential altitude** – Geometric is true height. Geopotential corrects for
  gravity weakening with height and is what the standard atmosphere is defined in (they differ
  by 1.4 m at 3 km).
- **Verification vs validation** – Verification asks "are the equations solved right?" (e.g. two
  independent models agree). Validation asks "are they the right equations?" (agreement with
  measurements). See docs/VALIDATION.md.
- **Inverse problem** – Instead of predicting outputs from inputs, inferring unknown inputs
  (here TIT and turbine efficiency) from measured outputs (thrust, SFC).
- **Ill-conditioning** – When the answer is a small difference of large numbers (net thrust =
  jet momentum − ram drag in slow, weak engines), tiny input errors become large output errors.

### Added in Phase 2
- **Velocity triangle** – At each blade edge, absolute velocity C = relative velocity W + blade speed U. Its
  geometry decides the work done and the losses.
- **Euler turbine equation** – Specific work = U2·Cθ2 − U1·Cθ1 (Cθ = swirl velocity). The heart of every
  turbomachine.
- **Slip factor σ** – The air leaving a centrifugal impeller swirls less than the blades would suggest. σ ≈ 0.7–0.9;
  correlations (Wiesner, Stodola) are uncertain by roughly ±0.05 or more.
- **Backsweep** – Impeller blades bent backwards at the tip. Lowers the work and the exit speed, widens the stable
  operating range, and usually improves efficiency.
- **Incidence** – The angle between the incoming flow and the blade's leading edge. Off-design flow means large
  incidence, which means loss.
- **Loss correlation / loss set** – Empirical formulas for each loss mechanism (incidence, friction, tip clearance,
  mixing, disk friction…). A "set" is one consistent choice of them (e.g. Oh et al. 1997).
- **Internal vs parasitic loss** – Internal losses reduce the useful pressure rise. Parasitic losses (disk friction,
  recirculation, leakage) need extra shaft work that ends up as heat.
- **Vaneless diffuser** – A plain annular channel outside the impeller. The flow slows down as it moves outward,
  converting speed into pressure; wall friction costs total pressure.
- **Speed line / map** – PR and efficiency versus mass flow at one shaft speed; several speeds make a map.
- **Choke (compressor)** – Some passage (inducer, throat) reaches Mach 1, so the flow can't increase further.
- **Surge criterion** – Meanline models can't really predict surge. We flag a PR peak or diffuser stall (Senoo).
  Unvalidated here.
- **Blind prediction / calibration / held-out validation** – Run the model with only published inputs first (blind),
  tune a few parameters on part of the data (calibration), then judge only on data the tuning never saw (held-out).
  Only the held-out error says something about predictive skill.
- **Effective parameter** – A tuned value that lumps several unknown effects together. Valid for the case it was fitted
  on; transferring it elsewhere is an assumption.
- **NGV / stator, rotor, total-to-static efficiency** – Turbine stator vanes accelerate and swirl the gas; the rotor
  extracts the work. η_ts counts the exit kinetic energy as lost.

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

### Phase 1 — Cycle model (2026-10-24)
**What we did**
1. Wrote a gas model (`turboml/gas/thermo.py`): air and kerosene combustion products as
   ideal-gas mixtures with temperature-dependent cp from NASA polynomials.
2. Built two independent design-point cycle models with the same inputs and outputs:
   - `cycle_pycycle.py` adapts NASA pyCycle to a micro turbojet: convergent nozzle, shaft
     mechanical loss, real fuel heating value.
   - `cycle_simple.py` is our own 0-D model, about 200× faster.
3. Verified them:
   - The ideal-Brayton test: with no losses and a constant-cp gas, the code reproduces the
     textbook formula η = 1 − PR^(−(γ−1)/γ) to 1e-9.
   - Energy and mass balances close to 1e-10.
   - The two models agree to ~0.1 % over a 64-point sweep.
4. Compared against five JetCat engines (63–425 N) and inferred what TIT and turbine efficiency
   their datasheet numbers imply.

**Why it matters:** every later surrogate is trained on these numbers. Two independent models
that agree, plus conservation checks, make a hidden modelling bug unlikely. They do *not* prove
the efficiencies are right; that's what validation data is for.

**Things we learned the hard way**
- pyCycle's default fuel enthalpy silently uses a ~5 % too-high heating value (now fixed in
  our wrapper).
- pyCycle's fast "tabular" property option costs up to ~3 % accuracy here.
- Exhaust temperature is pinned by energy conservation: thrust × SFC / air flow fixes it almost
  regardless of component efficiencies. That makes it a powerful consistency check on
  datasheets.

### Phase 2 — Component meanline models (2026-10-24)
**What we did**
1. Ingested your documents into `validation/cases/` (the 109-engine review table, the JetCat catalog and datasheets,
   KJ-66 measured and CFD data, geometry), then set the active bounds.
2. Tried TurboFlow's centrifugal compressor and found three problems (a slip-angle units bug, a lossless diffuser, a
   crashing 1-D diffuser). So we wrote our own compressor meanline model, verified it against TurboFlow's impeller
   (0.01 %), and gave it a real friction diffuser plus explicit choke and surge checks.
3. Validated against the measured KJ-66 compressor. Blind it misses (inputs unknown); calibrated on one speed it hits
   the other speeds within ±4.6 % PR.
4. Validated TurboFlow's axial turbine against NASA's Kofskey turbine (about 1–2 %, no tuning) and quantified the error of
   using air for turbine gas (< 0.5 %).

**Why it matters:** Phase 3 couples these maps into the cycle. Off-design behaviour (idle, flight), surge margin and
choke now come from geometry instead of guessed efficiencies.
