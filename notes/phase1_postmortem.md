# Phase 1 post-mortem

Scope: the counterfactual-feasibility line of `plan.md` (Phase 1 physics go/no-go, tag `phase1-final`) and the
active-probing check that followed it (tag `active-probe-final`). The Phase-1 verdict itself and its three passes are in
`notes/decision_phase1.md` (P2 fail, P6-frozen fail, no override; the line stopped there under the frozen rule). This
file holds the post-mortem sections; the first one, below, was written when the owner closed the active-probing
direction on 2026-11-03.

## Active-probing check (2026-11-02 → 2026-11-03)

**Question.** After Phase 1 showed that coarse-attribute text leaves a real gap to the per-object oracle (coarse bound v2:
best coarse method 0.08–0.2 below the oracle on the bins without a common feasible cell), is there a paper in letting the
policy *probe* the object before committing to (F, s)? The owner's three-gate check asked, in order: (1) is exact
information worth anything on the contexts where it can matter, (2) how much of that value does a probe plus the analytic
rule recover under a fixed sensor model, (3) how much does a small learned model recover from the same observations.
Decision matrix: ρ_analytic ≥ 0.8 → active identification + analytic control; both ρ low → π0.5 worth testing.

**Set-up.** Unit = object × fragility × urgency. A unit is "ambiguous (broad)" when it is feasible but its coarse bin has
no grid (F, s) that is safe for every object in the bin. v1 used the 48 objects of `data/sweep_v1` (27 broad ambiguous
units of 192, below the required 50); the owner then approved 41 more continuously-sampled objects in the same bands
(`data/sweep_v1_ext/`, outside dataset-v1, G7/G8 pass) and a v2 with an out-of-sample oracle (B5a/B5b on true properties,
calibrated per grouped fold, executed in simulation), bootstrap CIs over units, the absolute gap in success points, a
learning curve for the small model, and the fraction of ambiguous contexts. Scripts `analysis/active_probe_gates.py`
(v1) and `analysis/active_probe_gates2.py` (v2); reports in `results/phase1/active_probe/` and
`results/phase1/active_probe_v2/`; full record in `notes/decision_active_probe.md`.

**Numbers (v2, 89 objects, 356 units, 65 broad ambiguous = 0.183 [0.143, 0.222] of all contexts, 0.293 of the feasible
ones; strict ambiguity 0 at 7–8 objects per bin).**

| gate | quantity | value [95 % CI] | verdict |
|---|---|---|---|
| 1 | oracle (B5b, true properties, calibrated) − best coarse rule (lookup), success points | +0.265 [0.154, 0.385] | pass |
| 1 | infeasible-selection rate, lookup → oracle | 0.292 → 0.015 | pass |
| 2 | ρ_analytic, B5b on point probe estimates | 0.833 [0.67, 0.96] | ≥ 0.8 (point) |
| 2 | gap oracle − probe + formula, success points | +0.062 [0.015, 0.123] | |
| 3 | ρ_small at 25 / 50 / 75 / 100 % of ≈ 67 training objects | 0.10 / 0.11 / 0.22 / 0.34; 100 %: [−0.07, 0.63] | below analytic at every size |

v1 on 48 objects had given ρ_analytic 0.625 [−0.20, 1.00] and ρ_small −0.55; the v1 row "π0.5 worth testing" was
withdrawn by the owner because Gate 3 had failed from data size (36 training objects) and a small model failing from
scarcity says nothing about a larger model. The learning curve in v2 confirms the data limitation (still rising at 67
objects) and at the same time shows the learned model far below the analytic updater on identical observations
(0.74 vs 0.92 success at 100 %).

**Conclusion (owner, 2026-11-03).** Matrix row = active identification + analytic control. The bottleneck is
information: exact properties are worth +0.27 success points on the ambiguous contexts, and once that information is
available the analytic rule recovers 0.83 of it and leaves a 6-point gap. Nothing in the check calls for a learned
controller, let alone a VLA; it is neither a VLA paper nor a learning paper. Direction closed; the 50-branch label audit
is cancelled for now (resumable if the negative result is written up).

**What held up and what did not.**

- Held up: the frozen-rule discipline (P2 and P6 were left as fails; the matrix row was read from the numbers, not
  adjusted to them); the two-implementation cross-check (G7/G8 at 100 % on every sweep including the extension); the
  executed-selection evaluation (every baseline selection is run in the simulator, so "success" is measured, not
  predicted); the out-of-sample oracle, which removed v1's ρ > 1 artefact.
- Did not hold up: the v1 Gate 3 verdict (data-size effect read as a capacity verdict until the owner caught it); strict
  ambiguity as a target (rare at the 8 × 6 grid and these bands, and it vanishes as bins grow); the in-sample choice of
  the strongest updater (mild optimism in ρ_analytic; the second-best gives 0.750 [0.58, 0.92]).
- Reusable: `envs/feasibility_carry/` (force-limited Panda carry, state branching, labels v3), the sweep / bisection
  tooling, `crosscheck_b.py` + `analysis/crosscheck_g7g8.py`, the analytic baselines B5a/B5b with per-fold calibration,
  and the gate workflow (frozen criteria, executed selections, bootstrap CIs over units).
