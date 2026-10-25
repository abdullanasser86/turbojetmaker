# Draft issue reports for TurboFlow (NOT posted)

These drafts are for the user to review and decide whether to post at https://github.com/turbo-sim/turboflow.
TurboML does not post them itself (user decision 2026-10-25). All findings were made with **turboflow 0.1.18**
(PyPI), Python 3.11, numpy 2.3.5, CoolProp 8.0.0, on Linux. Reproducers 1–3 were re-run on 2026-10-25 and print the
quoted output.

---

## 1. Benner incidence loss: rotor sign never applied (`cascade_type = ["cascade_type"]`)

**Where:** `turboflow/axial_turbine/loss_model_benner.py`, `get_incidence_loss`, line 450:
```python
cascade_type = ["cascade_type"]        # a list literal, not geometry["cascade_type"]
...
chi = get_incidence_parameter(le, s, We, theta_in, theta_out, beta_in, beta_des, cascade_type)
```
`get_incidence_parameter` uses `sign_factor = 2 * (cascade_type == "rotor") - 1`. A list never equals `"rotor"`, so
every row (stator and rotor) gets `sign_factor = -1`. The evident intent is `+1` for rotors.

**Minimal reproducer**
```python
from turboflow.axial_turbine import loss_model_benner as br
flow = {"beta_in": 40.0, "gamma_out": 1.4, "Ma_rel_out": 0.7}
geom = {"leading_edge_diameter": 2e-3, "pitch": 1.5e-2, "leading_edge_wedge_angle": 50.0,
        "leading_edge_angle": 30.0, "A_throat": 0.5, "A_out": 1.0}
for ct in ("stator", "rotor"):
    print(ct, br.get_incidence_loss(flow, {**geom, "cascade_type": ct}, beta_des=30.0))
# stator 0.0034409213530271435
# rotor  0.0034409213530271435   <- identical: cascade_type is ignored
```

**Effect on the Kofskey & Nusbaum (1972) one-stage validation case** (your `validation_cases/axial_turbine/Kofskey1972_one_stage`
geometry and digitized measurements). We applied the intended sign (`geometry["cascade_type"]`) in an
equation-identical re-implementation. Our port reproduces TurboFlow to < 0.1 % in mass flow and < 0.07 pts in η_ts.
| η_ts error vs measured, mean (pts) | 50 % speed | 70 % | 90 % | 100 % | 110 % | all (n = 106) |
|---|---|---|---|---|---|---|
| as released (sign −1 everywhere) | +0.13 | −1.33 | −1.89 | −1.73 | −1.11 | −1.20 (max 2.84) |
| intended rotor sign (+1) | −2.63 | −2.66 | −1.98 | −1.73 | −1.10 | −2.00 (max 3.72) |
Torque mean error changes from −0.53 % to −2.38 % (max 5.1 % → 9.4 %). The released behaviour fits the measurements
*better*. At 50 % speed the rotor sees about +27° incidence, and the data don't show the extra loss that the intended
sign predicts.

**Question for the maintainers:** which angle convention does Benner, Sjolander & Moustapha (1997, J. Turbomach.
119:193) use for χ in a rotor frame where the inlet relative angle is positive and the exit angle negative? If −1 is
correct for rotors too, the line should become `sign_factor = -1` explicitly. If +1 is correct, the published Kofskey
agreement will change, so this should be fixed together with a re-validation.

---

## 2. Wiesner slip: trailing-edge angle in degrees passed to `np.cos` (radians)

**Where:** `turboflow/centrifugal_compressor/slip_model.py`, `get_slip_weisner`:
```python
return u_out*np.sqrt(np.cos(theta_out))/(z**0.7)    # theta_out is in degrees everywhere else
```
**Reproducer**
```python
from turboflow.centrifugal_compressor import slip_model as sm
for th in (-20.0, -40.0):
    print(th, sm.get_slip_weisner({"u_out": 400.0}, {"trailing_edge_angle": th, "number_of_blades": 10}))
# -20.0 50.98...      (correct: 400*sqrt(cos 20 deg)/10**0.7 = 77.4)
# -40.0 nan           (cos(-40 rad) < 0 -> sqrt of a negative number)
```
**Expected:** `np.cos(np.radians(theta_out))` (or the package's `math.cosd`). With backsweep beyond about 33° the slip
velocity is NaN and the compressor solve fails; below that it's silently wrong.

---

## 3. `print_simulation_summary` crashes when a solver has `elapsed_time = None`

**Where:** `turboflow/axial_turbine/performance_analysis.py`, `print_simulation_summary`. Every solver that has the
attribute is appended to `times`, even when the value is `None`. `np.mean(times)` then raises, and the exception
propagates out of `compute_performance(..., stop_on_failure=False)`, losing the results of all the other points.
**Reproducer**
```python
import logging
from turboflow.axial_turbine import performance_analysis as pa
class S: elapsed_time = None; success = False
pa.print_simulation_summary([S()], logging.getLogger("x"))
# TypeError: unsupported operand type(s) for /: 'NoneType' and 'int'
```
We hit this in a 512-point batch (one failed initial-guess solve). **Expected:** skip `None` times (or only append
`elapsed_time` for successful solves).

---

## 4. Vaneless diffuser (algebraic model): losses don't reduce the exit total pressure

**Observed:** with the `oh` loss model, changing the vaneless-diffuser skin-friction factor from 0.02 to 0.5 gives
identical exit total pressure. The diffuser is effectively lossless, so the reported stage efficiency equals the
impeller efficiency. In `flow_model.py` (algebraic vaneless diffuser, around line 446) the friction factor enters only
the angular-momentum residual `phi_AM = exp(-Cf (r_out - r_in)/(b_in cos(alpha_in)))`. The loss coefficient returned
by `compute_vaneless_diffuser_losses` doesn't appear to feed back into the stagnation pressure.
**Reproducer:** the shipped centrifugal-compressor example with `vaneless_diffuser` friction factor 0.02 vs 0.5,
comparing `p0` at the diffuser exit. (To be tightened into a stand-alone script before posting: we observed this
through our wrapper `turboml/physics/compressor_turboflow.py`.)

---

## 5. 1-D vaneless diffuser model raises `TypeError: Expected numeric dtype, got object`

**Observed** with numpy 2.3 / scipy 1.17 when selecting the 1-D (`evaluate_vaneless_diffuser_1D`) diffuser model; the
algebraic model runs. Probably an object-dtype array built from a dict of mixed values before the ODE integration.
(Stand-alone reproducer to be extracted before posting.)
