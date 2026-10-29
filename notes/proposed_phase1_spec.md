# Proposed spec changes before the Phase-1 full sweep (2026-10-29)

Patch: `notes/proposed_phase1_spec.patch` (apply with `git apply notes/proposed_phase1_spec.patch`). Touches `plan.md` (§3.1 ranges,
§3.2, §3.3, §3.5, §3.6, new §3.7, §6, §7, §8, §13), `docs/data_spec.md`, and adds `docs/analytic_baselines.md`. No code is changed.

## Decision points inside the patch (owner to confirm)

### 1. t_win for the sustained peak-crush proxy
| Option | Effect in this simulator | |
|---|---|---|
| 50 ms (**recommended, in the patch**) | longer than contact-onset spikes (a few ms), shorter than the acceleration pulse of the fastest carry (≈ 0.1 s at s = 0.5 on the 2× faster path) → carry-induced normal-force excursions (≈ m·a_lat/2 per pad, up to ≈ 5 N) count | |
| 100 ms | borderline at the fastest s: the dynamic excursion is partly averaged out; kept as the second P5 value | |
| 200 ms | misses the dynamic excursion entirely → the proxy reduces to F > f_tol; not proposed | |

### 2. T_max
| Option | T_max | deadline binds at s | passing grid levels |
|---|---|---|---|
| 2 × nominal carry + 0.5 s settle (**recommended, in the patch**) | 4.0 s | 2.29 | 5 of 6 (0.50, 0.72, 1.02, 1.47, 2.10); s = 3.00 fails by rule and stays in the sweep as a physically informative cell |
| 1.5 × nominal + 0.5 s | 3.1 s | 1.79 | 4 of 6 |
| episode cycle budget, e.g. 6 s | 6.0 s | 3.43 | 6 of 6 — label vacuous on the grid |
The rule is declared before any sweep result; the fact that one grid level falls outside it is a consequence, not the motive.

### 3. CoM offset fraction (horizontal axis orthogonal to closing, w = 4 cm)
Wrench-aware static F_min [N] for (m, μ) = (0.5,0.2) / (0.5,0.4) / (1.0,0.2) / (1.0,0.4) / (1.5,0.2) / (1.5,0.4); centred: 12 / 6 / 25 / 12 / 37 / 18.
| Offset | F_min per combo [N] | statically feasible, fragile (f_tol 20) | rigid (f_tol 50) |
|---|---|---|---|
| 0.15 w = 6 mm | 17 / 9 / 34 / 17 / 52 / 26 | 3 / 6 | 5 / 6 |
| 0.25 w = 10 mm | 23 / 12 / 47 / 23 / 70 / 35 | 1 / 6 | 5 / 6 |
| 0.35 w = 14 mm | 31 / 15 / 61 / 31 / 92 / 46 | 1 / 6 | 4 / 6 |
**Recommend 0.15 w = 6 mm (in the patch):** with the wrench-aware estimate (r_eff = 6.1 mm) it keeps 8 of 12 offset combos statically feasible while raising F_min by ≈ 1.4× over centred, so torsional slip is observable without making most cells infeasible; 0.25 w leaves only 1 of 6 fragile offset combos feasible and puts the heaviest rigid one at the top of the F range; 0.35 w is worse. The estimate is uncertain by up to ≈ 1.8× because MuJoCo resolves the pad as ~5 discrete contact points (lever ≈ 11 mm at the corners), so **demonstration (b) is the check**: if the refined slip boundary at 0.15 w shifts by < 15 % of the F range relative to centred, move to 0.25 w before the sweep. Set through `body_ipos` at runtime (no new asset).

**Cylinder (optional):** `CylinderObject` exists in robosuite; cylinder–pad contact gives 1–2 points per pad (line contact), so torsional resistance would be much lower and MuJoCo's cylinder collision is less robust. Estimated cost 0.5 day for contact tuning; benefit is a second object family for the held-out-instance split. Recommendation: defer to data pipeline v1, not Phase 1.

### 4. κ for the shear-augmented proxy
L = N + κ|T|, |T| ≈ m(g + a)/2 per pad = 7–13 N at 1.5 kg between rest and the 7 m/s² ceiling. κ = 1 (**in the patch**) moves the damage boundary by that amount; κ ∈ {0.5, 1, 2} is the P5 sensitivity range. The plain resultant sqrt(N² + T²) was considered and rejected: it changes the boundary by < 20 % and hides the s dependence.

### 5. Real-robot damage observable
Pressure-indicating film on both pads (**recommended**: cheap, per-grasp peak pressure, no fixture) plus residual deformation of a deformable cup for the fragile proxy object. A calibrated breakaway link measures a single separation force per fixture and would need one fixture per threshold; not proposed. Transfer test: Kendall τ ≥ 0.8 between simulated damage score and measured peak pressure (and cup deformation) over the hardware F × s sweep.

### 6. Boundary refinement
Bisection in F between bracketing grid levels, R = 5 repeats per probe, stop at ≤ 2 N bracket or 4 probes (≤ 20 extra branches per boundary per s); s_rec by bisection in s for P3 combos. Boundary = 50 % crossing of a logistic fit over grid + probe outcomes; 95 % CI by bootstrap over repeats. Probes stored under `probes/`, excluded from training groups.

## Numbers that follow from the decisions
- F grid (8, log-spaced, Newtons, no rescaling): 5.0, 7.3, 10.6, 15.5, 22.6, 32.9, 48.0, 70.0 N
- s grid (6, log-spaced): 0.50, 0.72, 1.02, 1.47, 2.10, 3.00
- 24 property combinations; 5760 grid branches ≈ 1.3 h; refinement adds ≤ 20 branches per boundary per (combo, s)
- Coarse attribute vocabulary gains a balance bin (balanced / off-centre) so that B5b and the learned method see the same information

## Open items the patch does not settle
- The pad half-size h = 8 mm and r_eff = 0.765 h in B5b are estimates from the gripper geometry; B5b-c may calibrate r_eff on the training split as a third scalar if the owner agrees.
- Whether the appendix impulse definition uses the carry only or grasp + carry: proposed grasp + carry (what the object experiences).
