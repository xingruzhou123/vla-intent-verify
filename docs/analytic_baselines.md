# Analytic baselines (B5a, B5b, B6) — specification

Proposed 2026-10-29 for plan.md §6 (P4) and §8. Both baselines are physics rules that output a feasible interval in F for every
s and then select (F, s) with the **same selection rule as the teacher** (plan §3.4: maximise the normalized margin, with the
nominal-distance penalty γ), restricted to deadline-feasible s (T_carry(s) ≤ T_max). They never see simulator outcomes at test
time. "a" below is always the **realized** peak end-effector acceleration logged per branch (plan §3.2; 50 ms-smoothed, matching the
damage window), never the commanded profile. Per s level the rules use the median realized acceleration over the sweep's branches at that s.

## 1. B5a — translational Coulomb rule

Per pad normal force N = F (force-saturated actuator). Two pads share the load.

- Slip bound, checked per phase (the carry runs along the closing axis, plan §3.5, so the lateral inertial load is carried by the
  pad normal forces and unloads the leading pad): lift / lower `F ≥ m (g + a_vert) / (2 μ)`; lateral `μ (F − m a_lat / 2) ≥ m g / 2`,
  i.e. `F ≥ m g / (2 μ) + m a_lat / 2`. `F_min(s)` is the largest of the three. (The first draft used `m·sqrt(g² + a_lat²)/(2μ)` for the
  lateral phase, which is the rule for a carry orthogonal to the closing axis; corrected 2026-10-29.)
- Damage cap: `F_max = f_tol − m · a_lat(s) / 2` (primary proxy: the trailing pad sees F + m·a_lat/2 because the carry runs along
  the closing axis; plan §3.5).
- Feasible interval `[F_min(s), F_max(s)]`; empty if F_min > F_max.
- Ignores the CoM offset by construction (model mismatch is intended: this is the rule a practitioner would write).

## 2. B5b — wrench-aware rule

Adds the CoM-induced torque, the pad contact geometry and a friction limit surface.

- Contact geometry: each pad is a 16 × 16 mm square patch (half-size h = 8 mm) centred on the object's side face at the grasp
  height; estimated, not read from the simulator, from the known pad size and the commanded grasp pose. Effective friction
  radius for a uniformly loaded square patch: `r_eff = 0.765 h ≈ 6.1 mm` (mean distance from the patch centre).
- Load per pad: tangential force `f_t = m · (g + a_vert) / 2` in the lift/lower phases and `m g / 2` in the lateral phase; torque about the pad normal from the CoM offset d (horizontal,
  orthogonal to closing): `M = m · g · d / 2` per pad (the lateral inertial term does not add torque about the pad normal for an
  offset along that axis; the vertical term does: use `M = m · (g + a_vert) · d / 2`).
- Limit surface (ellipsoidal approximation, Howe & Cutkosky): `(f_t / (μ N))² + (M / (μ N r_eff))² ≤ 1`, giving
  `F_min(s) = sqrt( f_t² + (M / r_eff)² ) / μ`.
- Damage cap: the feasible set uses the primary proxy (docs/data_spec.md §1.3), so B5b's default cap is the same trailing-pad bound
  as B5a, `F_max = f_tol − m·a_lat/2`. The shear-augmented cap `N + κ·|T| ≤ f_tol` with `|T| = sqrt(f_t² + (M/r_eff)²)` is reported as
  a variant against the shear-proxy boundary only.
- Lateral phase with the CoM torque: `F ≥ sqrt((m g / 2)² + (M / r_eff)²) / μ + m a_lat / 2` with `M = m g d / 2`.
- Measured 2026-10-29 (results/phase1/demo_report): with r_eff = 0.765 h the rule overestimates the slip boundary of the offset box by
  up to 14 N (10 mm offset: 30.2 N predicted vs 16.3 N measured at s = 0.5). The MuJoCo box–box contact resolves each pad as 4–5 corner
  points and holds torsion far better than a uniformly loaded patch; the measured shift implies an equivalent r_eff ≈ 17 mm. r_eff is
  therefore a third calibrated scalar in the coarse-calibrated setting (fitted on the training split); the uncalibrated B5b is reported as is.
- Feasible interval `[F_min(s), F_max(s)]`.

## 3. Information settings (never pooled)

| Setting | m, μ, f_tol, d given as | Calibration | Reported as |
|---|---|---|---|
| Oracle | true values | none | B6 (reference only; one row per rule) |
| Coarse, uncalibrated | bin centres fixed a priori (plan §3.6 bands): mass light/medium/heavy → 0.5 / 1.0 / 1.5 kg; friction slippery/normal → 0.2 / 0.4; fragility fragile/rigid → 20 / 50 N; balance balanced/off-centre → d = 0 / 0.25 w; r_eff = 0.765 h | none; reference only | B5a-u, B5b-u |
| Coarse, calibrated | the noisy coarse bins the VLA sees (ε = 0.1) | per coarse bin, a point estimate of m, μ, f_tol, d fitted on the **training split only** (mean of the true values of training groups in that bin), plus one scalar safety multiplier on F_min and one on F_max (and r_eff for B5b) fitted to maximise deadline-constrained safe success on training groups | B5a-c, B5b-c (**the fair comparison**) |

Same information and the same data access as the learned method: the coarse text with the same noise realisation, and the same
training split. Nothing from validation or test is used. Both uncalibrated and calibrated versions are reported; P4 and the
headline comparison use the **stronger calibrated rule**.

## 4. Evaluation

- Boundary error: |F_min^rule(s) − F_min^sim(s)| and |F_max^rule(s) − F_max^sim(s)| against the refined simulated boundaries
  (plan §6, bisection with CIs), per (combination, s), on held-out combinations.
- Selection quality: infeasible-selection rate, realized margin and deadline-constrained safe success of the selected (F, s),
  evaluated by executing it in the simulator (R repeats), on held-out and extrapolation combinations.
- P4 passes if the calibrated stronger rule selects deadline-infeasible parameters on ≥ 10 % of held-out combinations, or
  misplaces a boundary by more than the median feasible-set width / 4.
- What the novelty claim may rest on: robustness to attribute uncertainty (noisy bins) and to model mismatch (contact
  geometry, limit-surface shape, controller saturation, onset transients). It may **not** rest on B5a omitting the CoM offset:
  B5b has it, and B5b-c is the comparison that counts.

## 5. Implementation notes (for after approval)

- `baselines/analytic.py`: `coulomb_interval(m, mu, f_tol, a_lat, a_vert)`, `wrench_interval(m, mu, f_tol, d, a_lat, a_vert, h=0.008, kappa=1.0)`,
  `select(intervals, s_grid, T_max, nominal, gamma)` shared with the teacher's selection code (single implementation, imported).
- Calibration fits run once per dataset version and are logged in `data/<version>/VERSION.md`.
