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

### Added in Phase 3
- **Matching (engine)** – Finding the one operating point where all components agree: same shaft speed, same mass
  flow (plus fuel), turbine power = compressor power / η_mech, and the nozzle passes the flow. Off-design performance
  comes out of this, not out of assumed efficiencies.
- **Design mode vs off-design mode** – Design mode *sizes* the geometry for chosen targets (PR, mass flow, TIT, speed).
  Off-design mode keeps that geometry fixed and asks how the engine behaves at another speed or flight condition.
- **Flow coefficient φ = Cx/U, loading coefficient ψ = Δh0/U², reaction R** – The three dimensionless numbers that
  describe an axial turbine stage. ψ sets the work per blade speed², φ the axial speed, R the split of the expansion
  between stator and rotor. With them, the work and the speed fix the radius: U = √(Δh0/ψ), r = U/ω.
- **Zweifel coefficient** – A blade-loading number used to choose the blade spacing (pitch); about 0.8 is the classic
  optimum.
- **Gauging angle / throat** – The narrowest passage between two blades (the throat, width "opening"). cos(gauging
  angle) = throat area / exit area. It roughly sets the exit flow angle and the choking flow.
- **Deviation** – The exit flow doesn't follow the blade exactly. Aungier's model gives a small deviation at low Mach
  that goes to zero at the critical Mach number.
- **Critical Mach number** – With losses, a nozzle passes its maximum flow slightly below Mach 1. That's when a blade
  row chokes in the TurboFlow formulation.
- **Corrected flow W√T/P** – The mass flow normalised by inlet temperature and pressure. Components behave the same
  at the same corrected flow and corrected speed, and duct pressure losses scale with its square.
- **Operating line** – The path the engine's operating point follows across the compressor map as speed changes.
- **AN²** – Annulus area × speed². It is proportional to the centrifugal stress at the turbine blade root: the classic
  quick check on whether a turbine is too fast for its blade height.
- **Gas-to-metal temperature allowance** – How much cooler the blade metal is than the gas (cooling, radiation, pattern
  factor). Phase 3 used 0 K; since Phase 4 there is no fixed allowance (see *metal temperature* below).
- **Proxy** – A simple estimate standing in for a proper analysis (mass ~ k·D²·L, stress ~ AN²). Good for ranking
  designs and rejecting the impossible ones; not for sign-off.

### Added in Phase 4
- **Creep / creep rupture** – At high temperature a metal under constant load slowly stretches (creeps) and finally
  breaks. The hotter it is, the lower the stress it can carry for a given time. Turbine blades are designed for a
  *life*: the stress that breaks a test bar in 100 h (or 1000 h) at the blade's metal temperature is the **rupture
  strength**. We require rupture strength ≥ 1.5 × blade stress at 100 h (both are config settings).
- **Larson–Miller parameter** – P = T (C + log10 t), T in K, t in h, C ≈ 20. Different (temperature, life) pairs with
  the same P give about the same rupture strength, so one curve σ(P) trades temperature against life. We use it only
  for lives other than the tabulated 100 h / 1000 h.
- **Metal temperature** – The blade sees the gas temperature *relative to the blade*. A rotor blade moves away from the
  flow, so it feels the rotor-relative total temperature, which is lower than the TIT. A stator vane (NGV) is still and
  sees the full gas temperature, including hot streaks.
- **Pattern factor (PF)** – How non-uniform the combustor exit is: PF = (T_max − T_mean)/(T_mean − T3). The hottest vane
  sees TIT + PF (TIT − T3). We have no sourced PF for micro combustors yet, so each design stores PF_max, the largest
  PF its vanes would survive.
- **Gas bending stress** – The aerodynamic force that turns the flow also bends the blade like a cantilever; the root
  stress is the bending moment over the section modulus. Small compared with the centrifugal stress in our designs
  (median ~2 %).
- **Active constraint** – The check that makes a design infeasible (AN2, creep_rotor, creep_ngv, …above_material_data,
  tip_speed). An optimizer needs to know *which* wall it hits, not just that it hit one.
- **Hub-to-tip ratio** – Blade-root radius / blade-tip radius. A low value means long blades on a small hub: big
  annulus, high AN², high root stress. TurboFlow recommends a range for its loss correlations; designs outside it are
  kept with a warning.
- **Dataset card** – A short document that ships with a dataset: how it was made (code version, seed, config), what one
  record means, the counts of failures and their reasons, and the known limitations.
- **Held-out test set** – Examples set aside before any model is fitted and never used for training or tuning. The
  error on them is an honest estimate of the error on new designs. Ours is frozen in `data/splits/` before evaluation.
- **Low-discrepancy sequence** – A deterministic point set (e.g. scrambled Sobol) that fills a box evenly. "Scrambled"
  randomises it with a seed, so it can be used for statistics and is still reproducible.
- **Profiling** – Measuring where a program spends its time before optimising it. Here 90 % of the compressor time was
  in one inner loop (the diffuser continuity solve).
- **Warm start** – Starting an iterative solver from the previous answer. Neighbouring steps of a march have nearly the
  same solution, so Newton converges in 2–3 iterations instead of a bracketed search of ~20.

- **Data types (MEAS / CFD / MFR / ASSUM)** – Where a number comes from: measured, simulated (CFD), a manufacturer's
  claim, or a modelling assumption recommended by a source. When sources disagree we trust them in that order. Two
  engines of the same make and model can differ by up to 10 %, so a model can't be validated more tightly than that
  against catalogue data.
- **Specific speed** – N √Q / Δh^0.75: one number for the "shape" of a compressor. A high-specific-speed wheel has a big
  inlet compared with its outer diameter. For a given pressure ratio, the specific speed decides how much tip speed is
  needed. That is why PR and tip speed are tied together in one compressor family.
- **Thermocouple recovery factor** – A bare thermocouple in a gas stream reads between the static and the total
  temperature: T_s + r (T_t − T_s), with r ≈ 0.68 for a bare wire across the flow (Murray). The difference is small
  when the gas is slow (≈ 1 K at idle).
- **Stop rule** – A pass/fail criterion fixed *before* a run (here: ≥ 10 % feasible in the pilot), so the decision
  isn't bent by the result.

- **Transcribed vs digitized** – Numbers copied from a printed table are *transcribed*: exact, apart from typos.
  Numbers read off a plotted curve are *digitized*: they carry a reading error, which we measure by digitizing a plot
  whose numbers are also tabulated (Murray's Fig C.7 vs Appendix D: ±6 °C rms).
- **Energy-balance (derived) quantities** – A value computed from other measurements, not measured itself. Murray's
  turbine inlet temperature is the exhaust temperature plus the turbine work, and the turbine work is the compressor
  work / η_m. A "measured turbine efficiency" built that way inherits all their errors.
- **Exhaust diffuser vs propelling nozzle** – A turbojet ends in a nozzle that accelerates the gas into a jet (thrust).
  A cogeneration rig ends in a diffuser that slows the gas down to recover pressure, so the turbine can extract more
  work. The two give different operating lines, so data from one can't directly validate the other.
- **Thermal soak (hysteresis)** – After running fast, the engine casing and probes are hot, so readings taken while
  slowing down differ from readings taken while speeding up at the same speed.
- **fork vs spawn** – Two ways to start a worker process. Linux can *fork* (copy the running program). Windows can only
  *spawn* (start a fresh Python that imports the script again), so all work must sit behind
  `if __name__ == "__main__":`.
- **Code page / UTF-8 mode** – How text is turned into bytes. Windows' default (cp1252) can't write characters like
  "Δσ" or "²"; Python's UTF-8 mode (`PYTHONUTF8=1`) and explicit `encoding="utf-8"` avoid that.

- **Corrected speed / corrected mass flow** – N√(T_ref/T0) and m√(T0/T_ref)·p_ref/p0. They remove the effect of the day's
  inlet temperature and pressure, so a compressor map measured on a warm day applies on a cold one.
- **Speed line** – How pressure ratio and efficiency vary with mass flow at one corrected speed. A *flat* speed line
  means the pressure ratio hardly changes with flow, so a small pressure error corresponds to a large flow error.
  That is why we compare flows by the closest approach to the line, not by inverting it.
- **Buildable range** – The range of a design variable for which a physical geometry exists at all. Sampling only
  inside it (as for the compressor wheel and the turbine hub/tip ratio) stops the dataset from filling up with
  impossible engines.
- **Calibration vs extrapolation** – A model fitted on 70–110 krpm data is a calibration there and an extrapolation
  above it. The measured 120 krpm line showed the extrapolation is optimistic, a reason not to trust a model far
  beyond its data.

### Added in Phase 4, round 5
- **Refit / held-out speed line** – Re-estimating calibration factors on some measured speed lines and keeping the
  others aside. The error on the held-out lines tells how well the model predicts speeds it was not tuned on.
- **Extrapolation** – Using a model outside the range of the data it was checked against. Our compressor data stop at
  120 krpm (tip speed 415 m/s). Designs above that are labelled "extrapolated" and carry a wider uncertainty band, sized
  by how fast the error grew toward the last measured line.
- **Model-form error** – Error that no choice of parameter values can remove, because the equations miss some physics.
  Example: two calibration factors cannot follow an error that changes sign with speed.
- **Identifiability** – Whether the data can pin down a parameter. If two parameters change the prediction in the same
  way (hub/tip and NGV angle both set the turbine's throat area), only their combination is identified. Signs: a huge
  condition number, correlations near ±1, and very different answers from different starting points.
- **Profile (likelihood) scan** – Fix one parameter at a series of values, re-fit the others each time, and watch how
  the fit quality changes. The range where it stays within ~1 standard error is an honest uncertainty, even when the
  usual Jacobian formula fails.
- **Gauging angle** – For a blade row, cos(g) = throat opening / pitch. It sets the exit flow angle and, with the blade
  height, the throat area that limits (chokes) the flow.
- **Solidus** – The temperature at which an alloy starts to melt (IN-713C: 2300 °F, 1533 K). An uncooled vane in gas
  hotter than that melts, whatever its stress: no rupture data can help.
- **Specification minimum (AMS 5391)** – A purchase requirement every batch must meet (e.g. ≥ 30 h at 1800 °F and
  22 ksi). Handbook "typical" values are averages and sit above such minima.
- **Physics vs missing data** – An infeasible design fails on *physics* if some check that the data do cover fails
  (stress above strength at that temperature). It fails on *missing data* if the only problem is a temperature beyond
  the data, so new data could change the verdict.

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

- **Loss multiplier (k_loss)** – One number that scales a loss model's predicted losses so the model's efficiency
  matches measurements. It does not explain *why* losses are higher (small blades, low Reynolds number, roughness);
  it only makes the size right, like a calibration factor on a sensor.
- **Expected calibration error (ECE)** – For a classifier: group predictions by probability (0–0.1, 0.1–0.2, …) and
  compare the average predicted probability with how often the event really happened in each group. 0 = perfectly
  honest probabilities.
- **Aleatoric vs epistemic uncertainty** – Aleatoric: scatter the data themselves contain (it does not shrink with
  more data). Epistemic: the model not knowing, because it has seen little data nearby (it shrinks with more data). A
  deep ensemble measures epistemic uncertainty as the disagreement between its members.
- **Out-of-distribution (OOD)** – A query unlike the training data (outside the sampled box, in a sparse region, or
  where the physics model itself is extrapolating). Surrogate predictions there must be re-checked with physics.

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

### Phase 3 — Coupled engine model (2026-10-25)
**What we did**
1. Wrote our own **fast axial turbine** model: TurboFlow's equations (Benner losses, Aungier deviation,
   critical-Mach choking) solved row by row instead of all at once. It takes ~7.5 ms per point in a sweep, against
   ~3–20 s for TurboFlow. It matches TurboFlow to < 0.15 % on the NASA Kofskey turbine. We also verified it on 256
   randomly sized stages (512 points) across the design space. Along the way we found that TurboFlow's code never
   uses its intended rotor incidence sign, and that the version actually used fits the measurements better (open
   question).
2. **Turbine sizing** from φ, ψ, R (velocity triangles → angles, heights, pitch, throat), plus a solve that makes the
   stage pass exactly the design flow at exactly the design work.
3. **Compressor sizing**: a KJ-66-like wheel resized for each design (r2 for the target PR, inducer and exit width
   from the KJ-66 flow coefficients). It uses the KJ-66 calibration as provisional defaults, now with an uncertainty
   band.
4. **The coupled engine**: design mode, then off-design matching (Newton on mass flow, TIT and turbine PR) with
   continuation in speed. Failures are labelled (choke, surge, no nozzle pressure ratio, …).
5. Mechanical proxies: material temperature limit (IN-713C default, MAR-M 247 premium), blade root stress (AN²,
   unchecked until the allowable stress is sourced), tip speed, envelope and mass.
6. A ~150 N baseline engine: 148 N, 494 N·s/kg, TSFC 41.5 g/(kN·s), 130 krpm, Ø115 × 298 mm (proxy). All of these
   lie inside the database ranges for 104–193 N engines. The operating line runs 35–100 % speed, with idle at 6 %
   thrust.

**Why it matters:** The design variables now produce a full engine *with its off-design behaviour*, in seconds.
Phase 4 can sample this model to build the training data.

### Phase 4 — Training dataset, pilot (2026-10-25)
**What we did**
1. **Turbine creep checks** from real alloy data (NiDI No. 393 Tables 9/10, 100 h and 1000 h rupture strength of
   IN-713C and MAR-M 247): rotor at its relative total temperature (centrifugal + gas bending stress), NGVs at TIT plus
   a hot-streak margin. The failing check is reported as the *active constraint*. The Phase 3 baseline turned out
   infeasible: its blade root stress is too high for either alloy (AN²).
2. **Calibration band widened** for model-form error until all 34 held-out compressor measurements fall inside it.
   20 % of designs sample the calibration factors from that band.
3. **Profiling**: one inner loop dominated, so a warm-started Newton step replaced it. The engine model got 3–4× faster
   with identical results. No rewrite.
4. **Robust sizing and continuation**, so a design fails only when it is physically impossible, not because a solver
   started from a bad guess.
5. **Dataset generator**: Sobol sampling of 7 variables, frozen 15 % test split, 4 operating points per design,
   failures recorded with reasons, resumable after a crash, parallel workers.
6. **Pilot**: the first 256 designs. 92 engines could be designed and 3 are mechanically feasible. The binding walls are
   turbine blade root stress (AN²) and impeller tip speed. 61 of the 92 have long turbine blades (hub/tip outside
   TurboFlow's recommended range).

**Why it matters:** The surrogate in Phase 5 learns from this data. The pilot shows that sampling the box naïvely
wastes most runs on engines that can't exist, so the sampling has to be fixed before the full run (decision for the
user).

### Phase 4, round 2 — Reference data, option C pilot, idle EGT (2026-10-25)
**What we did**
1. Took in the user's extracts of the KJ-66, DePaola, Murray and JetCat documents, each value tagged by data type.
2. Checked the KJ-66 compressor against Xiang's geometry. It was already the same geometry, so the calibration stays.
   The model follows the *measured* efficiency, not the CFD one.
3. Sampled impeller tip speed and turbine hub/tip ratio instead of shaft speed and flow coefficient (option C). The
   pilot stopped at 5.5 % feasible: with this compressor family, pressure ratio and tip speed are tied, so most random
   (PR, tip speed) pairs can't be built at all.
4. Investigated the idle EGT. The model gets the KJ-66 full-power EGT about right, but its EGT barely drops towards
   idle. Efficiencies, thermocouple reading, combustion efficiency and Reynolds number can't explain that. The
   low-speed matching is the next suspect, and Murray's low-speed measurements would test it.

**Why it matters:** Sampling must respect the physics of the component family, or the dataset fills up with
impossible engines. Validating the *shape* of a trend (EGT vs speed) can reveal model problems that a single design
point hides.

### Phase 4, round 3 — Murray's measurements, option A pilot, Windows (2026-10-25)
**What we did**
1. Transcribed Murray's measured KJ-66 running data (15 tables) and digitized his run plots, including three runs of
   the real turbojet from 28 to 120 krpm. The digitizing error was measured against the tables.
2. Validated our compressor model on the new data: pressure ratio within about 2 % rms, including speeds below the
   calibration data.
3. Redid the idle-EGT question with real low-speed data. The real KJ-66 IS hot at idle (about as hot as at full
   speed); our model is ~40–80 K too hot there, not 150–250 K. The likely cause is the turbine/nozzle side of the
   matching, and an indicative turbine check suggests the meanline turbine is ~10 points too optimistic at this size.
4. Option A: the compressor is now sized directly from tip speed and specific speed, so every wheel is buildable and
   PR comes out (1.7–3.3). The pilot still stopped (8.5 % feasible), now because of the turbine: a prescribed hub/tip
   ratio is often impossible for low-work engines.
5. Windows: CUDA torch for Windows in the lock, a Windows-aware environment check, UTF-8 everywhere, spawn-safe scripts
   and a smoke test.

**Why it matters:** Measured data turned a vague worry ("idle too hot") into a quantified model error with a likely
cause. The same "make it buildable by construction" step that fixed the compressor is now needed for the turbine.

### Phase 4, round 4 — Measured maps, buildable turbines, the v4 dataset (2026-10-25)
**What we did**
1. Digitized Murray's measured KJ-66 maps and operating lines, checking the digitizing against his tables (TIT within
   ~1 K).
2. Compared our compressor model with the measured maps. It is good at 50–100 krpm (pressure ratio within ~2 %,
   inside the measurement uncertainty) but too optimistic at 110–120 krpm. We reported that instead of refitting.
3. Confirmed the idle question with measured turbine inlet temperatures: the model's low-speed TIT is 25–65 K too
   high, which explains its hot idle EGT.
4. Made the turbine buildable by construction: the hub/tip ratio is sampled only where the exit Mach number stays
   ≤ 0.5. High-TIT designs were limited to 15 % of the dataset.
5. The v4 pilot reached 32 % feasible, so the full 2,000-design run went ahead.

**Why it matters:** The dataset now mostly contains engines that could exist, and we know where the model is trustworthy
(KJ-66-like speeds) and where it is optimistic (high tip speed, low-speed turbine).


### Phase 4, round 5 — Measured speed lines, a rebuilt KJ-66 turbine, material data (2026-10-27)
**What we did**
1. Refitted the compressor's two calibration factors on four of Murray's measured speed lines and tested on the other
   five. The fit is good from 50 to 100 krpm, but the error still grows toward 120 krpm (+5 % pressure ratio): the
   model's form, not its numbers, limits it. Designs faster than the last measured line (415 m/s tip speed) are now
   labelled "extrapolated" and get a wider uncertainty band, sized by how fast that error grows.
2. Rebuilt the KJ-66 turbine from published blade counts and fitted its unknown hub/tip ratio and blade angles to
   Murray's turbine measurements. We learned what the data can and cannot tell: the flow map alone only fixes the
   nozzle throat area; the measured exhaust velocity fixes the hub/tip ratio. The efficiency stays 11–13 points too
   high whatever the geometry, so that gap is in the loss model (or in Murray's derived efficiency), not the shape.
3. Rebuilt a whole KJ-66 from these parts. With the turbine that matches the measured flow map, the idle exhaust
   temperature and Murray's idle TIT are reproduced (within ~15 K); the air flow is still ~13 % low at idle.
4. Read the Alloy 713C handbook (Nickel 337). It agrees with the earlier data and passes the purchase-spec check, but
   it has nothing above 2000 °F at 100 h. Most hot designs fail because the spinning blades are over-stressed at their
   temperature — physics that more data will not change.
5. Generated the v5 dataset: 30 % feasible, cross-checked against pyCycle.

**Why it matters:** We now know which numbers in the model are measured, which are calibrated, and where each is
trustworthy. A surrogate trained on v5 can be told which designs lie beyond the validated compressor range.

### Phase 4, round 6 + Phase 5 start — Loss calibration, two turbines, surrogates (2026-10-27)
**What we did**
1. Read the 1973 NASA compressor report the user pointed to (Galvas). It has no loss term for high inducer Mach
   numbers; it says so itself. Its choke check never triggers on Murray's measured points, so it cannot explain the
   high-speed error. Recommendation: do not add it.
2. Fitted one turbine loss multiplier on Murray's measured turbine efficiency. With the published correlations the
   turbine was 11–13 points too efficient; with losses ×~1.9 the held-out error drops to +2 (T-A) / +7 (T-B) points.
3. Carried two plausible KJ-66 turbine geometries side by side, so we can see how much the unknown geometry matters
   (idle exhaust temperature differs by ~90 K; thrust and fuel consumption by 4–8 %).
4. Built the surrogate pipeline: a Gaussian process and a deep ensemble predict thrust, fuel consumption,
   temperatures and mass in milliseconds, with honest uncertainty; a classifier predicts feasibility; flags mark
   predictions that need a physics re-check. On v5 it reaches ~0.3 % median error (provisional).

**Why it matters:** Calibrating the turbine changes engine fuel consumption by ~20 %. That is why the dataset must be
regenerated (v6) before the surrogates' numbers mean anything, and why the pipeline was built to retrain in one command.
