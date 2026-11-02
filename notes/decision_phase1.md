# Phase 1 decision record

## Verdict (2026-11-02, full sweep `data/sweep_v1`, report `results/phase1/full_v1/report.md`)

The plan's decision rule (plan.md §6) applied mechanically to the 48-object sweep (11 520 grid + 12 015 probe branches,
every criterion exactly as frozen below; nothing changed after seeing results):

| check | result | measured |
|---|---|---|
| P1 | **pass** | measured pad force = 0.993–1.001 · F, R² ≥ 0.9999 on every object; fingers ≥ 20.1 mm from the joint limit |
| P2 | **fail** | 0 of 48 objects show a boundary shift ≥ 9.75 N between s = 0.5 and the slowest deadline-ok s (relaxed: s = 2.1; urgent: s = 1.02). Largest slip-boundary shift 7.5 N (medium_slippery_balanced_3, heavy_slippery_balanced_0; separated intervals), largest damage-boundary shift 4.4 N (rigid, heavy_normal_balanced_0). 28 of 47 slip shifts are statistically separated but all are below the 9.75 N bar |
| P3 | **pass** | 10 objects (criterion ≥ 2) have at least one context with no feasible F at s = 0.5 and a feasible F at a slower deadline-ok s; 17 recovery contexts, 69 of 192 contexts infeasible by design (excluded); every recovery window contains a grid F level; widths 1.6–19.0 N, median 6.8 N |
| P4 | **pass** | stronger calibrated rule B5a-c on the 12 held-out objects: executed infeasible-selection rate 9/36 = 0.25 (criterion ≥ 0.10); median boundary error 4.2 N vs W/4 = 3.9 N (W = 15.6 N median feasible width) |
| P5 | **pass** | damage boundaries monotone in f_tol (573/573) and t_win (1127/1127); P2 and P3 verdicts identical under all 12 damage settings; pooled Kendall τ(q_N, q_L) = 0.946 (per context: median 0.961, 112/123 ≥ 0.8) |
| P6 | **fail** | 27 (object, label, cell) entries with repeat agreement < 0.9 outside the refined boundaries' uncertainty, out of 10 700 (cells × labels); 23 of them are cells where every repeat slipped (the disagreement is where a dropped object lands or whether it is crushed between the fingers), 4 are one cell of heavy_slippery_off-centre_1 (F = 48.0 N, s = 1.02; slip 0.8) |
| P7 | **pass** | 80 of 136 feasible contexts (0.588) select a non-modal s on the refined feasible set (criterion ≥ 0.25); grid teacher 0.398 |

**Outcome under the rule: P2 fails → stop. No π0.5 training on F × s; the RSS Stage-1 submission is cancelled; re-plan for
T-RL (re-scope to a force-only study is the rule's prescription, not decided here).** P6 fails as well → the rule says fix
the simulation before anything else. The owner takes the re-scoping decisions; nothing is re-scoped in this record.

Reference numbers that do not enter the rule: best single fixed (F, s) = (32.9 N, s = 1.02) scores 0.378 mean
deadline-constrained safe success on the 144 training contexts (per-context oracle 0.633) and 0.375 on the 48 held-out
contexts (oracle 0.729). B5a oracle: median boundary error 0.6 N, infeasible-selection rate 0.00; B5b oracle 0.7 N / 0.11;
B5a-u 4.2 N / 0.22; B5b-u 6.0 N / 0.56; B5b-c 6.3 N / 0.31. Post-hoc diagnostics (not gates): slip-boundary shift over
47 objects, relaxed deadline: median 2.5 N (12.8 % of F_min at s = 0.5), max 7.5 N (25.9 %); 15 objects shift by ≥ 15 % of their
own F_min(0.5); urgent deadline: median 1.5 N, max 3.8 N. The s-dependence of the slip boundary is therefore real
and statistically separated on most objects, but smaller than the frozen absolute bar on every object.

Both independent-implementation gates (G7/G8) have not been run on this sweep; the verdict above rests on
implementation A only.

## Frozen decisions and definitional changes (unchanged, recorded before the sweep)

This section records the frozen decisions and definitional changes made before the sweep, each dated, so that nothing is
tuned after the fact.

## Frozen before the full sweep (owner decisions, 2026-10-29)

| Item | Decision | Reason / evidence |
|---|---|---|
| Controller gain | OSC kp = 600 (kd = 2√kp), position and orientation | kp = 150 could not execute the 2× faster nominal path: lift 9 mm short at s = 0.5, lateral acceleration capped at 4.7 m/s²; kp = 600 tracks to < 1 mm and reaches 7 m/s² (results/phase1/demo_report, gain table) |
| CoM offset | nuisance variable, off-centre band [0.20, 0.30] w; **no torsional-slip claim** | measured slip-boundary shift at 0.25 w: 1.3–3.4 N; torsion ≤ 1.4° before translational slip; the torsion label (pad-edge displacement ≥ 5 mm) never fired |
| B5b r_eff | fitted scalar on the training split; uncalibrated (0.765 h) reported as reference only | uncalibrated B5b overestimates the offset box's slip boundary by up to 14 N (equivalent r_eff ≈ 17 mm) |
| Urgent deadline | T_max = 1.0 × T_carry(s₀) + 0.5 s = 2.25 s (relaxed: 2.0 × + 0.5 s = 4.0 s) | operational: urgent = nominal-speed execution plus settling |
| P2 | absolute threshold: shift ≥ 15 % of (F_hi − F_lo) = 9.75 N | kept absolute; frozen before the kp = 600 rerun |
| P6 | cells whose F lies inside a refined boundary's uncertainty (95 % CI ∪ bisection bracket) are excluded from the pass rate and reported separately | with R = 5 a single disagreeing repeat at a boundary cell fails the cell |
| Property generation | continuous sampling inside coarse bands; 4 objects per bin cell (48), 1 test-combos object per cell | the coarse setting otherwise equals the oracle (plan §3.6) |

## Definitional change after the demonstrations (2026-10-29)

**P3 reference speed: nominal s → fastest grid s (0.5).** In the demonstrations no property combination was infeasible at
s = 1 under any controller gain (kp = 150, 300, 600): the realized acceleration at the nominal path is too small for the
slip need to exceed the damage cap for any object in the property grid. The slow-down recovery mechanism does appear
from the fast end (at kp ≥ 300 the heavy–slippery rigid box has no feasible F at s = 0.5 — 44 N slips, 48 N damages —
and recovers at s = 1). P3 is therefore evaluated from the fastest deadline-feasible grid level, and the width of the
recovered window (N) and whether a grid level falls inside it are reported alongside the pass/fail.

## Pre-sweep demonstrations

- kp = 150, six combinations: results/phase1/demo_report/report.md (2026-10-29).
- kp = 600, (a) 1.5 kg / μ 0.2 and (c) 1.5 kg / μ 0.4: results/phase1/demo600_report/report.md (2026-10-29).
