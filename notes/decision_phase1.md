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

## Status after the owner's three items (2026-11-02, second pass)

**P2 fails as frozen. P3 and P7, also predeclared, pass.** The owner will decide after items 1–3 below whether to keep
the rule or override it with a documented rationale. No decision is taken in this record.

1. **P6 label fix (label version 2, docs/data_spec.md §1.2).** `completed` is False after a drop regardless of where the
   object lands, and the damage proxies stop at the drop onset (an object crushed between the closing fingers after it
   fell is not grasp damage). All 23 535 branches were relabelled from the stored time series without re-simulation
   (`envs/feasibility_carry/relabel.py`; version-1 files kept as `summary.labels_v1.json`), the baseline selections were
   re-executed, and every check was re-evaluated (`results/phase1/full_v2/report.md`). **No verdict changes:** P1, P3, P4,
   P5, P7 pass and P2 fails with identical numbers; P6 still fails, with 22 instead of 27 entries below 0.9 outside the
   boundaries' uncertainty. The 5 removed entries were the drop-and-crush cells. Of the remaining 22, 18 are cells where
   every repeat slipped but the object was not (or not always) dropped: it slid in the grasp, was released, and whether
   it came to rest inside the 3 cm goal tolerance varied across repeats, so `completed` (geometric) disagrees. The other
   4 are one cell of heavy_slippery_off-centre_1 (F = 48.0 N, s = 1.02; slip rate 0.8) that the refined boundary's
   uncertainty does not cover. Whether `completed` should also be False after a slip that did not end in a drop is an
   owner decision (it would make `completed` redundant with `physical_safe ∧ completed_geometric` and is not what item 1
   specified); it is not applied.
2. **G7 / G8 (implementation B, `envs/feasibility_carry/crosscheck_b.py`; runner `analysis/crosscheck_g7g8.py`;
   `results/phase1/crosscheck_v1/report.md`).** Independent code path and numerics (hand-written quaternion algebra,
   min-filter sustained peak, run-length lost-contact test, group-by rates, vectorised margin), same author. On the full
   sweep: label agreement 100 % on all eight classes (23 535 branches and probes; 0 disagreements); feasible sets identical
   on 9 216 of 9 216 grid cells (192 contexts), margins identical (max difference 0), teacher targets identical on 192 of
   192 contexts. **G7 and G8 pass** against the frozen thresholds (≥ 95 %, ≥ 98 %, ≤ 1 grid step).
3. **Coarse-information upper bound (`analysis/coarse_bound.py`; `results/phase1/coarse_bound_v1/report.md`).** Per
   coarse cell (mass × friction × balance × fragility) × urgency, the single grid (F, s) maximising mean deadline-constrained
   safe success over the cell's three training objects, evaluated on the held-out test object (noise-free bins; no new
   simulation). On the 48 held-out contexts: infeasible-selection rate 5/35 = 0.143 on contexts with a non-empty grid
   feasible set (0.375 over all 48, where the empty-truth contexts count as infeasible); mean success 0.625 against the
   per-object oracle 0.729 and the best single fixed (F, s) 0.375. B5a-c / B5b-c executed infeasible-selection rates:
   0.250 / 0.306 (ε-noisy bins). 24 of the 48 (cell, fragility, urgency) groups have at least one grid (F, s) feasible for
   all four objects of the cell; 13 of the 24 fragile groups and 1 of the 24 rigid groups have none, and 11 groups have no
   feasible object at all. Reading: coarse information alone recovers most of the oracle success (0.625 of 0.729) when
   the per-cell lookup is fitted on the sweep; the room for a learned method over that lookup is the remaining 0.104 of
   success and the 0.143 infeasible-selection rate, plus robustness to the ε-noise that the lookup here does not see.

## Phase 1 final reporting (frozen 2026-11-02, tag `phase1-final`)

| check | reported as | number |
|---|---|---|
| P2 | **fail** (frozen rule) | 0 of 48 objects reach a 9.75 N separated boundary shift; largest slip shift 7.5 N |
| P6, frozen | **fail** | 22 (object, label, cell) disagreements below 0.9 outside boundary uncertainty (`results/phase1/full_v3/report.md`) |
| P6, conditioned (reported separately, never as a pass of the frozen check) | all remaining cells agree | excluding the 19 stochastic-flagged cells outside boundary uncertainty (18 held-slip `completed` cells + the heavy_slippery_off-centre_1 known-exception cell, docs/data_spec.md §1.2a), every other outside-boundary cell has repeat agreement 1.0 on all five labels. Every one of the 22 disagreements lies in a stochastic-flagged cell (verified against `cells[]`) |
| P1, P3, P4, P5, P7 | pass | see the verdict table above |

Label version 3, implementation A, cross-checked by implementation B (G7/G8 pass). Later label issues go into a known-issues
list, not a relabel (owner rule). The dataset tag `dataset-v1` still waits for the audit sign-off; `phase1-final` freezes
the Phase-1 reporting only.

## Third pass (2026-11-02): owner items 0–7

0. Previous pass committed as 2ef25ca (label v2, cross-check B, coarse bound). This pass is uncommitted.
1. **Label v3 (owner rule): `completed` = False iff the gripper loses contact (run-length test, both pads ≥ 50 ms) before
   the scheduled release; a slip whose grasp holds keeps the geometric result.** Relabelled from stored records
   (`summary.labels_v2.json` kept), per-cell rate tables with a `stochastic` flag stored in every `summary.json`
   (`cells[]`), baselines re-executed, everything re-evaluated: `results/phase1/full_v3/report.md`. **No verdict other
   than P6 changed; P6 as frozen still fails on the same 22 entries** (P1, P3, P4, P5, P7 pass; P2 fails; identical
   numbers). Of the 22 entries, **0 were cleared by the v3 rule** (in every one the grasp held until the release) and
   **18 are flagged stochastic** (`completed` after a held slip, listed in docs/data_spec.md §1.2a); the remaining 4 are the
   heavy_slippery_off-centre_1 boundary cell (F = 48.0 N, s = 1.02), documented as a known exception without widening the
   uncertainty. With the 18 stochastic cells and the known-exception cell set aside, no disagreement outside boundary
   uncertainty remains; whether that closes P6 is the owner's reading. Sweep-wide, 213 of 2 304 grid cells carry a
   stochastic flag on at least one label (most on `damage_shear` or `slip` at a boundary). G7/G8 re-run on v3
   (`results/phase1/crosscheck_v3/report.md`): 100 % agreement on all nine classes, feasible sets / margins / teacher
   identical on 192/192 contexts.
2. **P2: decision pending.** The owner's template was left unfilled; P2 is reported as failed under the frozen rule
   everywhere, and no override is recorded.
3. **Noisy coarse lookup** (`analysis/coarse_bound2.py`, `results/phase1/coarse_bound_v2/report.md`): the lookup table is
   keyed by the training objects' ε-noisy bins and queried with the held-out object's ε-noisy bins (spec draw, seed 0, the
   same realisation the rules see). Canonical split, 48 held-out contexts: infeasible-selection rate 0.314 (noise-free
   lookup 0.143; B5a-c 0.229; B5b-c 0.286; best single fixed 0.486), mean success 0.500 (noise-free 0.625; oracle 0.729;
   B5a-c 0.562; B5b-c 0.521; best fixed 0.375). Over 19 further noise seeds the noisy lookup scores 0.584 ± 0.051 success
   and 0.198 ± 0.070 infeasible, so the seed-0 draw is on the unlucky side.
4. **Leave-one-object-out** (4 folds, mean ± sd, all 48 groups): oracle success 0.641 ± 0.054; noise-free lookup
   0.569 ± 0.038 (infeasible 0.171 ± 0.055); noisy lookup 0.506 ± 0.016 (0.249 ± 0.045); B5a-c 0.573 ± 0.010
   (0.133 ± 0.072); B5b-c 0.556 ± 0.040 (0.163 ± 0.104); best single fixed 0.340 ± 0.022 (0.495 ± 0.064). The calibrated
   Coulomb rule is as good as the noise-free lookup and better than the noisy one under the deployment noise.
5. **Difficulty split** (24 groups with a grid cell feasible for all four objects vs 24 without; 11 of the latter have no
   feasible object at all). LOO, groups without a common cell, success / infeasible: oracle 0.281 / 0; noise-free lookup
   0.210 / 0.278; noisy lookup 0.165 / 0.407; B5a-c 0.240 / 0.295; B5b-c 0.217 / 0.402; best fixed 0.112 / 0.559. Excluding
   the 11 all-empty groups: oracle 0.519, B5a-c 0.442, lookup 0.388, noisy lookup 0.304. On the groups with a common cell
   every method except the fixed cell is near the oracle (lookup 0.927, B5a-c 0.906, noisy lookup 0.848 vs 1.000).
   Reading: the paper's target groups are where coarse bins do not determine a safe cell; there the gap between the best
   coarse-information method and the oracle is 0.08–0.2 of success and 0.3–0.4 infeasible-selection rate.
6. **Audit sample** (`results/phase1/audit/`, `analysis/audit_sample.py`): 50 branches, stratified over 8 classes with
   positives and negatives of each (seed 0), each replayed from its stored state and rendered as an mp4 and a frame strip
   with the recorded force / slip / contact traces; `audit_sheet.csv` and `audit_sheet.md` carry the predicted labels and
   a blank owner-verdict column. Replay reproduced every stored label on all 50 branches.
7. **Freeze: not done.** Waiting for the owner's audit sign-off and P2 decision; no tag, no next phase.

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
