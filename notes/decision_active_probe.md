# Decision record — "active probing before commit" direction (three-gate check, 2026-11-02)

Owner request of 2026-11-02. Inputs: sweep v1 (48 objects, label version 3, tag `phase1-final`), the leave-one-object-out
rows of `results/phase1/coarse_bound_v2/`, and simulated probe observations. No new phase, no π0.5, no probe rollouts.
Script: `analysis/active_probe_gates.py`; report with every table: `results/phase1/active_probe/report.md`; numbers:
`gates.json`. All CIs are 95 % bootstrap over units.

## Step 1 — independent contexts (unit = object × fragility × urgency; 192 units)

| class | units | fragile / rigid | relaxed / urgent |
|---|---|---|---|
| common-safe (one grid (F, s) in 𝓕 of all four objects of the coarse bin) | 96 | 20 / 76 | 52 / 44 |
| ambiguous-but-feasible, strict (all four objects feasible, no common cell) | 4 | 0 / 4 | 0 / 4 |
| mixed (unit feasible, no common cell, at least one sibling infeasible) | 23 | 16 / 7 | 11 / 12 |
| intrinsically infeasible (own 𝓕 empty) | 69 | 60 / 9 | 33 / 36 |

Ambiguous in the broad sense (ambiguous + mixed) = **27 units**; strict = **4**. Under leave-one-object-out (held-out
unit feasible, its three training siblings without a common cell) 25 held-out units are ambiguous; 2 of the 4 strict and
25 of the 27 broad units stay ambiguous when held out.

**Fewer than 50.** Proposal, not run: at the measured rates (0.141 broad, 0.021 strict ambiguous units per unit) 50 broad
ambiguous units need ≈ 41 more continuously-sampled objects (≈ 10 000 grid + ≈ 10 000 probe branches, ≈ 46 min wall at
12 processes); 50 strict ambiguous units would need ≈ 550 objects (≈ 10 h), i.e. strict ambiguity is rare at this grid
resolution and these bands. Sampling only the bins that produce ambiguity (slippery and heavy bins) would cut both numbers
at the cost of a non-uniform bin prior; the strict count is also limited by the 8 × 6 grid, since a finer F grid would
split more "common" cells.

## Step 2 — Gate 1: value of exact information (LOO, rules calibrated per fold and executed)

Ambiguous subset, broad (27 units):

| method | safe success [CI] | infeasible-selection rate [CI] |
|---|---|---|
| per-object oracle | 1.000 | 0.000 |
| B5a calibrated | 0.704 [0.556, 0.889] | 0.296 [0.111, 0.444] |
| noise-free coarse lookup | 0.704 [0.519, 0.852] | 0.296 [0.148, 0.481] |

Oracle − B5a-c: +0.296 success [0.111, 0.444] (criterion ≥ 0.10); infeasible rate 0.296 → 0 (criterion ≥ 40 % relative and
≥ 0.10 absolute). **Gate 1: PASS** (also on the strict subset, +0.250 [0.000, 0.750] on 4 units; on all 192 units
+0.068 [0.026, 0.120], which fails the success criterion but passes the infeasible-rate criterion). Caveat: the oracle is
the best grid cell of the same outcomes used to score it (in-sample), so on feasible units it is 1.000 by construction;
the infeasible-rate criterion is met automatically whenever the baseline's rate exceeds 0.10.

## Step 3 — Gate 2: probe + formula

Sensor model (assumptions, simulated from the true m, μ, d): micro-lift mass from the Franka external-force estimate,
per-episode bias σ = 0.05 kg and noise σ = 0.02 kg; lateral pulse at 12 N (below every fragility tolerance) giving μ̂ with
10 % relative noise and 5 % relative bias; CoM offset from the micro-lift torque about the pad normal, σ = 0.02 N·m
(d̂ = 2 M̂ / m̂ g); fragility not probeable (calibrated coarse-bin estimate, as B5a-c); realized accelerations = sweep
medians; interval updater uses the ±2σ box (worst case for F_min and F_max). Updaters: Coulomb (B5a) and wrench (B5b)
rules on point or interval estimates, multipliers calibrated per fold on the training units with the same probe noise,
selections executed in the simulator with 5 repeats (seed-0 draws); 10 noise seeds for the interval-membership rate.

ρ_analytic = (J_probe − J_coarse) / (J_oracle − J_coarse), J = safe success, J_coarse = B5a-c (same rule family without
probes):

| subset | strongest updater | J_probe [CI] | J_coarse | J_oracle | ρ_analytic (success) [CI] | ρ_analytic (infeasible rate) |
|---|---|---|---|---|---|---|
| ambiguous, broad (27) | B5b point | 0.889 [0.741, 1.000] | 0.704 | 1.000 | **0.625 [−0.20, 1.00]** | 0.625 |
| ambiguous, strict (4) | B5a point | 1.000 | 0.750 | 1.000 | 1.000 | 1.000 |
| all 192 | B5a interval | 0.667 [0.599, 0.729] | 0.573 | 0.641 | 1.385 [0.28, 3.50] | 0.505 |

On the broad ambiguous subset the point estimate is below 0.8, but the CI spans [−0.20, 1.00]: 27 units cannot separate
ρ = 0.6 from ρ = 0.8 or 1.0. On all 192 units the executed probe rule exceeds the grid oracle (ρ > 1) because a
continuous F between grid levels can succeed on units whose grid 𝓕 is empty; the grid oracle is not an upper bound for
executed continuous selections, so that row is not a valid ρ. The B5b interval updater is worse than the point updaters
(the 2σ worst case over μ and d pushes F_min above F_max on slippery objects).

**Gate 2 as stated: ρ_analytic = 0.625 < 0.8 → does not stop the direction, but the estimate is not statistically
distinguishable from ≥ 0.8.**

## Step 4 — Gate 3: small learned model

MLP 2 × 64 (and a 5-member ensemble with pessimistic selection p̄ − σ) from the same probe observations plus coarse bins and
urgency to the 48-cell feasibility mask; grouped 4-fold splits by object index; 144 training units × 20 noise draws per
fold; selection = cell with the highest predicted feasibility (ties toward nominal s); evaluated on the held-out unit's
grid outcomes over 10 noise seeds.

| subset | model | J_small [CI] | infeasible rate | ρ_small (success) [CI] | ρ_small (infeasible) |
|---|---|---|---|---|---|
| ambiguous, broad (27) | mlp-mean | 0.540 [0.359, 0.720] | 0.463 | **−0.552 [−2.87, 0.21]** | −0.563 |
| ambiguous, broad (27) | ensemble-pessimistic | 0.515 [0.355, 0.689] | 0.485 | −0.638 [−2.80, 0.09] | −0.638 |
| all 192 | mlp-mean | 0.564 [0.493, 0.628] | 0.139 | −0.125 [−2.11, 0.47] | 0.593 |

The small model is below the coarse rule on the ambiguous subset and roughly at it overall (lower infeasible rate,
same success). With 144 training units this measures the data regime, not a ceiling on learning: ρ_small is "not close
to 1" here because the learned model has 36 training objects, not because learning cannot use the probe.

## Matrix row

- Gate 1: pass. ρ_analytic = 0.625 [−0.20, 1.00]. ρ_small = −0.55 [−2.87, 0.21].
- Mechanically: **both insufficient → π0.5 worth testing (requires multiple language-conditioned tasks).**
- Honest reading: the row is selected on 27 ambiguous units with CIs that include every other row's condition. Gate 2
  cannot exclude ρ_analytic ≥ 0.8, and Gate 3's negative ρ_small reflects the training-set size. Before acting on the row,
  the ambiguous count has to grow (≈ 41 more objects for 50 broad units, 46 min; owner decision, not run) and the gates
  re-run; the strict count (4) says the "every object has a safe action but no common one" regime barely exists at the
  current grid and bands.

Not done, by instruction: no new phase, no π0.5 fine-tuning, no probe rollouts, no new objects.


## Owner decisions of 2026-11-03

- **The matrix row above is not valid.** Gate 3 failed because of data size (36 training objects per fold, 144 training
  units), and a small model failing from data scarcity does not imply that a larger model (π0.5) would succeed. The row
  "both insufficient → π0.5 worth testing" is therefore withdrawn; no conclusion about π0.5 follows from the v1 check.
- P2: frozen rule kept; P2 fails; no override. Item closed.
- Audit: pending the owner's review of `results/phase1/audit/audit_sheet.csv` (`owner_verdict` to be filled by the owner).
- The 41-object extension is approved (continuous sampling inside the same bands, `data/objects_ext1.json`,
  sweep `data/sweep_v1_ext/`), and Gates 1–3 are to be re-run with: (1) an out-of-sample oracle — B5a/B5b with the true
  properties, executed in simulation; the in-sample best-grid-cell oracle only as a reference; (2) the unchanged sensor
  model, ρ_analytic with bootstrap CIs over units, and separately the absolute gap (oracle − probe + formula) in success
  points; (3) grouped splits and a learning curve (ρ_small at 25 / 50 / 75 / 100 % of the training objects) so that data
  scarcity can be separated from model capacity; (4) the fraction of all contexts that are ambiguous. No π0.5
  fine-tuning, no probe rollouts. Results: `results/phase1/active_probe_v2/` (section below once run).

## Three-gate check v2 on the extended object set (2026-11-03)

Inputs: sweep v1 (48 objects) + extension `data/sweep_v1_ext` (41 objects, `data/objects_ext1.json`, seed 1, 3–4 per bin
cell with the extra cells drawn at random; 9 840 grid + 10 105 probe branches, 43 min wall, 6.8 GB; label version 3;
G7/G8 on the extension: 100 % label agreement over 19 945 branches, feasible sets identical on 7 872 / 7 872 cells,
teacher targets 164 / 164 — `results/phase1/crosscheck_ext1/`). The extension is NOT part of dataset-v1 (separate sweep
root; the tagged 48-object labels are untouched). Script `analysis/active_probe_gates2.py`; report and numbers in
`results/phase1/active_probe_v2/` (report.md, gates.json, selections.json). The same script run on the 48 objects alone
(`results/phase1/active_probe_v2_smoke48/`) reproduces the v1 coarse baselines exactly (B5a-c 0.704 / 0.296, lookup
0.704 on the same 27 units), so the vectorised calibration matches the old code path. Folds: object index mod 4 (24 / 24 /
24 / 17 objects); every rule is calibrated on the fold's training units and executed with 5 paired repeats; 95 % bootstrap
CIs over units (units are correlated within an object).

### Item 4 — fraction of contexts that are ambiguous (356 units = 89 objects × fragility × urgency)

| class | units | fraction of all [CI] |
|---|---|---|
| common-safe | 157 | |
| ambiguous, strict (every bin object feasible, no common cell) | 0 | 0.000 |
| mixed (unit feasible, no common cell, some sibling infeasible) | 65 | |
| intrinsically infeasible | 134 | |
| **ambiguous, broad (strict + mixed)** | **65** | **0.183 [0.143, 0.222]** (0.293 [0.230, 0.356] of the 222 feasible units) |

All 65 broad units stay ambiguous under leave-one-object-out. The count depends on the bin size: with 7–8 objects per
bin a common cell is harder to find than with 4 (v1: 27 broad / 4 strict of 192, i.e. 0.141 / 0.021), and the 4 strict
units of v1 became "mixed" once a sibling with an empty feasible set joined their bin. The strict regime ("every object
has a safe action but no common one") does not occur at this grid and these bands with the larger bins. 50 broad
ambiguous units reached (65 ≥ 50).

### Gate 1 — out-of-sample oracle (65 broad ambiguous units)

| method | safe success [CI] | infeasible-selection rate [CI] |
|---|---|---|
| grid oracle (in-sample, reference only) | 1.000 | 0.000 |
| B5a on true properties, c = 1 | 0.985 [0.954, 1.000] | 0.015 [0.000, 0.046] |
| B5b on true properties, c = 1 | 0.631 [0.508, 0.754] | 0.369 [0.246, 0.492] |
| B5a / B5b on true properties, calibrated per fold | 0.985 / 0.985 | 0.015 / 0.015 |
| B5a-c / B5b-c (ε-noisy bins, calibrated) | 0.615 / 0.612 | 0.385 / 0.400 |
| noise-free coarse lookup | 0.720 [0.609, 0.828] | 0.292 [0.185, 0.415] |

Headline out-of-sample oracle = B5b-true-cal (higher of the two calibrated true-property rules on all 356 units, 0.702
vs 0.698; identical on the ambiguous subset). Oracle − strongest coarse baseline (lookup): **+0.265 [0.154, 0.385]**
success points; infeasible rate 0.292 → 0.015. **Gate 1: PASS** (all 356 units: +0.132 [0.097, 0.166]; feasible units:
+0.098 [0.059, 0.142]). The out-of-sample oracle is below the in-sample grid oracle on feasible units (0.985 vs 1.000)
and above it on all units (0.702 vs 0.624) because a continuous F between grid levels succeeds on some units whose grid
𝓕 is empty; the v1 problem (ρ > 1 against the grid oracle) is gone.

### Gate 2 — probe + formula (sensor model unchanged; 65 broad ambiguous units)

| updater | safe success [CI] | infeasible rate | gap = oracle − probe [CI] | ρ_analytic [CI] |
|---|---|---|---|---|
| B5a point | 0.892 [0.815, 0.969] | 0.108 | +0.092 [0.031, 0.169] | 0.750 [0.58, 0.92] |
| **B5b point** | **0.923 [0.862, 0.985]** | 0.077 | **+0.062 [0.015, 0.123]** | **0.833 [0.67, 0.96]** |
| B5a interval | 0.831 [0.738, 0.908] | 0.169 | +0.154 [0.062, 0.246] | 0.583 [0.37, 0.80] |
| B5b interval | 0.908 [0.831, 0.969] | 0.092 | +0.077 [0.000, 0.154] | 0.792 [0.59, 1.00] |

J_oracle = 0.985, J_coarse (B5a-c) = 0.615. The strongest updater (B5b on point estimates) gives **ρ_analytic = 0.833
[0.67, 0.96]** and leaves an absolute gap of **6.2 success points [1.5, 12.3]** to the true-property rule; the gap is
statistically non-zero (sensor noise costs something) but the CI on ρ includes both < 0.8 and ≈ 1. On all 356 units
ρ_analytic = 0.664 [0.52, 0.79] (gap +0.051 [0.030, 0.073]); on the 222 feasible units 0.784 [0.64, 0.91]. The strongest
updater is picked among four variants on the same subset (mild optimism); the second-best (B5a point) gives 0.750
[0.58, 0.92].

### Gate 3 — small learned model, learning curve (grouped folds; 65 broad ambiguous units; mlp-mean)

| training objects per fold | safe success [CI] | gap = oracle − model [CI] | ρ_small [CI] |
|---|---|---|---|
| 25 % (≈ 19) | 0.652 [0.570, 0.727] | +0.333 [0.254, 0.416] | 0.099 [−0.37, 0.37] |
| 50 % (≈ 31) | 0.655 [0.566, 0.738] | +0.329 [0.242, 0.423] | 0.108 [−0.35, 0.39] |
| 75 % (48) | 0.696 [0.597, 0.785] | +0.289 [0.199, 0.387] | 0.218 [−0.18, 0.49] |
| 100 % (≈ 67) | 0.740 [0.637, 0.832] | +0.245 [0.140, 0.357] | **0.337 [−0.07, 0.63]** |

(ensemble-pessimistic at 100 %: 0.262 [−0.16, 0.56]; all 356 units at 100 %: 0.162 [−0.10, 0.38].) The curve rises
monotonically from ≈ 0.10 at 19 objects to 0.34 at 67 and has not flattened; the v1 regime (36 objects) sits between the
50 % and 75 % points, which is where ρ_small ≈ 0.1–0.2. At every point the small model is well below the analytic
updater on the same observations (0.74 vs 0.92 success at 100 %).

### Reading (owner's call; no matrix row assigned)

- Gate 1 passes with an out-of-sample oracle: exact properties are worth +0.27 success points over the best coarse rule
  on the ambiguous contexts, which are 18 % of all contexts (29 % of the feasible ones) at this bin size.
- Gate 2: the probe + formula recovers ρ_analytic = 0.83 of that gap as a point estimate, with CI [0.67, 0.96]; the
  residual cost of the modelled sensor noise is 6 success points [1.5, 12.3].
- Gate 3: ρ_small is still rising with the number of training objects (0.10 → 0.34) and is data-limited at 67 objects;
  it is not evidence about model capacity in either direction.
- Under the owner's rule of 2026-11-03 the matrix row "π0.5 worth testing" is not derivable from Gate 3; the row that
  the point estimates mechanically select is "ρ_analytic ≥ 0.8 → active identification + analytic control", but the CI
  does not exclude ρ_analytic < 0.8, and the strongest-updater choice is in-sample. The owner reads the row.

Not done, by instruction: no π0.5 fine-tuning, no probe rollouts, no new phase. Open: audit sign-off (dataset-v1 tag).

## Final decision (owner, 2026-11-03) — direction closed

**Matrix row: active identification + analytic control.** Reasoning, as given by the owner:

- ρ_analytic = 0.833 [0.67, 0.96] (B5b on point estimates, 65 broad ambiguous units, out-of-sample oracle executed in
  simulation): the probe + formula recovers most of the value of exact information.
- The absolute gap to the oracle is 6 success points (+0.062 [0.015, 0.123]): what the modelled sensor noise costs is
  small in absolute terms.
- The small learned model stays below the analytic updater at every training size (ρ_small 0.10 / 0.11 / 0.22 / 0.34 at
  25 / 50 / 75 / 100 % of ≈ 67 training objects, against 0.83 for the analytic rule on the same observations).

Conclusion: the bottleneck is information, not control. Once the properties are identified, the analytic rule suffices;
there is no VLA paper and no learning paper in this direction. **The active-probing direction is closed.**

Consequences recorded the same day:

- Audit of the 50-branch sheet (`results/phase1/audit/audit_sheet.csv`): **cancelled for now** — no paper depends on
  these labels. It can be resumed if the negative result is written up later; the sheet, videos and frame strips stay in
  `results/phase1/audit/` and the labels under tag `phase1-final` are unchanged. dataset-v1 is therefore not tagged.
- Everything on branch `probing` committed, merged to `main`, tagged `active-probe-final`.
- One section on this check added to `notes/phase1_postmortem.md`.
- `status.md`: feasibility line closed (Phase 1 + active-probing check); no active project in this repo.

## Diagnostic pass before final closure (owner request of 2026-11-03; existing 89-object data only)

Script `analysis/active_probe_diag.py`; report and numbers `results/phase1/active_probe_diag/` (report.md, diag.json).
No new objects, no learned model, no π0.5. The only simulation was the execution of the inner-fold analytic selections
(3 284 distinct, 29 min). Timeboxed to one session; the line is closed regardless of the outcome (owner hard stop).

### 1. Updater-selection bias removed (65 broad ambiguous units; oracle B5b-true-cal, J_coarse B5a-c, as in v2)

| protocol | safe success [CI] | gap = oracle − analytic [CI] | ρ_analytic [CI] |
|---|---|---|---|
| nested: updater chosen per outer fold on inner held-out ambiguous units (inner calibrations executed) | 0.938 [0.877, 0.985] | +0.046 [0.000, 0.108] | **0.875 [0.72, 1.00]** |
| nested, chosen on all inner units | 0.877 [0.800, 0.954] | +0.108 [0.046, 0.185] | 0.708 [0.52, 0.89] |
| frozen a priori: B5a on point estimates (the plan's primary rule) | 0.892 [0.815, 0.969] | +0.092 [0.031, 0.169] | 0.750 [0.58, 0.92] |
| v2 in-sample choice (B5b point), for reference | 0.923 [0.862, 0.985] | +0.062 [0.015, 0.123] | 0.833 [0.67, 0.96] |

Chosen updaters per fold (ambiguous criterion): B5a-point, B5b-point, B5b-point, B5b-point; the inner scores of the two
point updaters differ by ≤ 0.04 in every fold. On all 356 units the nested ρ is 0.701 [0.55, 0.83] (gap +0.046
[0.024, 0.069]); on the 222 feasible units 0.811 [0.67, 0.94]. **Bounded reading:** the bias-free point estimate of
ρ_analytic on the ambiguous population lies between 0.71 and 0.88 depending on the selection protocol, the CIs span
[0.52, 1.00], and the absolute gap lies between 5 and 11 success points with every CI touching or including 0 at the
low end and ≈ 0.11–0.19 at the high end. The v2 value 0.833 is inside that range; the matrix threshold 0.8 is not
established either way by these data.

### 2. Decomposition of the 65 broad ambiguous units

| class | units |
|---|---|
| all members of the bin individually feasible (strict) | 0 |
| mixed (at least one sibling intrinsically infeasible) | 65 |
| feasibility disagreement only: the feasible siblings share a grid cell | 33 |
| action disagreement: the feasible siblings have no common grid cell | 32 — of which 21 have a common *continuous* F at some s (grid artefact) |
| removable by a finer coarse vocabulary (half-width sub-bins, all sub-bin siblings feasible with a common cell) | 45 (upper bound: 1–4 units per sub-bin group) |

Mutually exclusive: feasibility-only / sub-bin-removable 26; feasibility-only / persists 7; action + feasibility /
sub-bin-removable 19; action + feasibility / persists 13. So only 13 of the 65 units show an action disagreement among
feasible siblings that survives a finer vocabulary, and 11 of the 32 action disagreements have no common continuous F.

Where the Gate-1 gain comes from (oracle − lookup = +0.265, sum +17.2 over the 65 units): +18.2 from the 19 units where
the lookup cell lies outside the unit's feasible set (a different action was needed), −1.0 from the 46 units where the
lookup cell is feasible (one unit with lookup above the oracle; no stochastic partial successes). **Zero of the +0.265
comes from identifying infeasible instances** — every unit of this population is feasible by construction. On all 356
units (+0.132) the split is common-safe +4.6 (10 %), mixed +17.2 (37 %), infeasible +25.2 (54 %): the largest share of
the all-units gain is the oracle finding a continuous F that succeeds on units whose grid feasible set is empty, not
abstention. Abstention on the 134 infeasible units: oracle 92, B5b-point probe 78, B5a-point probe 74, B5a-c 23, lookup
0; damage rate when attempting: 0.07 / 0.17 / 0.17 / 0.28. **"Broad ambiguous" is therefore not a usable paper
population without this decomposition:** it is 100 % "mixed", half of it is feasibility disagreement with a shared safe
cell among the feasible siblings, and most of it disappears under a finer vocabulary.

### 3. Residual oracle − analytic gap, case by case (seed-0 observation, fixed attribution order)

| updater | cases | stochastic boundary labels | boundary resolution | probe noise (μ) | probe noise (m) | omitted CoM/torque physics | calibration | other |
|---|---|---|---|---|---|---|---|---|
| B5a point (frozen; 6 of 65, 0.092 points) | 6 | 1 | 0 | 1 | 1 | 0 | 3 | 0 |
| B5b point (4 of 65, 0.062 points) | 4 | 1 | 1 | 1 | 0 | 0 | 1 | 0 |

Each case is one unit with oracle 1.0 and analytic 0.0 (the per-unit table with miss direction, distance to the
boundary, standardised estimation errors and the counterfactual fixes is in the report). No case is attributed to omitted
CoM/torque physics: substituting all true properties fixes every B5a case through the multipliers, not through the
torque term. Three of the six B5a cases and one of the four B5b cases are calibration (the per-fold multipliers, fitted
on noisy training estimates, push the selection out of the interval although the noisy estimate alone would not).

**Systematic, observable pattern?** Over the 65 units × 10 noise seeds (label-based outcome): B5a point 53 misses of
634 (0.084), B5b point 42 of 646 (0.065). Miss direction is balanced (above the damage boundary 27 / 23, below the slip
boundary 26 / 19, abstained 16 / 4), so there is no sign bias to correct. A grouped out-of-fold ridge model on the
run-time observables (bin, fragility, urgency, m̂, μ̂, d̂, F, s) predicts the failure indicator at chance (AUC 0.41 /
0.59); its R² on the signed margin (0.36 / 0.31) reflects that the margin itself depends on bin and F, not that the
errors do. What is systematic is per-object: the misses sit in 13–14 objects and repeat across noise seeds
(heavy_normal_balanced_5 misses in 9 of 10 seeds with both updaters, heavy_slippery_balanced_1 in 8 of 10,
medium_normal_off-centre_7 in 10 of 10 with B5a), i.e. they are modelling errors of the analytic interval for particular
objects, not probe noise — but with one such object per bin they are not predictable out-of-fold from the observables at
this data size. **Verdict: no observable pattern that would support an analytic-plus-learned residual correction is
established by these data.**

Possible future project (recorded only; no work started, owner hard stop): characterise the object-persistent misses of
the analytic interval (which physical feature of those objects the Coulomb/wrench rule misses) on a larger object set;
a residual corrector would need many objects per bin with such persistent misses to be learnable out-of-fold.

### Bounded conclusions (2026-11-03)

- **P2 remains failed** (frozen rule: 0 of 48 objects reach a 9.75 N separated boundary shift; largest slip shift 7.5 N).
- **P6-frozen remains failed** (22 disagreements below 0.9 outside boundary uncertainty); the conditioned result is
  reported separately and never as a pass: every one of the 22 lies in a stochastic-flagged cell, and all remaining
  outside-boundary cells have repeat agreement 1.0.
- **The current data do not establish that learning is unnecessary in general.** The bias-free ρ_analytic is 0.71–0.88
  with CIs down to 0.52; the residual gap of 5–11 points is attributed to calibration, probe noise, label stochasticity
  and boundary resolution on 4–6 units; and the small learned model's curve was still rising at 67 objects.
- **The current evidence does establish that π0.5 fine-tuning is not justified for this environment and deadline:**
  exact information closes +0.265 of the gap on the ambiguous contexts (18 % of all), the analytic rule on probe
  estimates recovers most of it, the population where it matters is mostly a feasibility-disagreement and
  vocabulary artefact, and nothing in the residual is a learnable pattern at this data size.
- Direction closed (owner, 2026-11-03). The dataset is not tagged: the final dataset tag waits for the owner audit
  sign-off (audit still cancelled for now, resumable).
