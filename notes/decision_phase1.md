# Phase 1 decision record

Verdict (P1–P7, RSS go/no-go): **not yet taken** — the full sweep has not been run. This file currently records the
frozen decisions and definitional changes made before the sweep, each dated, so that nothing is tuned after the fact.

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
