# Data specification — interaction groups (plan.md §7)

Written 2026-10-29 from plan.md §7. No earlier version of this file existed in the repo.
Phase 1 (physics gate) produces sweep data in the same group format but without observations for the VLA;
the observation fields become mandatory from data pipeline v1 onward.

## 1. Unit of data: the interaction group

An **interaction group** is one pre-contact state together with all of its (F, s) branches. Branches within a group
differ **only** in the commanded (F, s); everything else (initial pose, object properties, carry path, solver settings)
is restored from the saved MuJoCo state at the branch point.

### 1.1 Group-level fields

| Field | Type | Notes |
|---|---|---|
| `group_id` | str | unique; never reused across splits |
| `object_id` | str | object instance (mesh / geometry) |
| `properties_true` | {m, μ, f_tol (fragility, both sampled classes), com_offset (3-vector, m; along the horizontal axis orthogonal to closing)} — continuous values sampled inside the coarse bands (plan §3.6; `envs/feasibility_carry/objects.py`, `data/objects_<version>.json`) | **labels and oracle setting only** |
| `properties_coarse` | {mass_bin, friction_bin, fragility_bin, balance_bin} + `noise_eps` | binned text attributes as the VLA sees them (plan §3.6: light/medium/heavy, slippery/normal, fragile/rigid, balanced/off-centre); noise ε = 0.1 applied at generation time and logged |
| `instruction` | str | task language; states the urgency level |
| `urgency` | relaxed / urgent | deadline context (plan §3.7); groups sharing a branch state but differing in urgency share their branches |
| `obs_decision` | two camera views + proprioception at the decision chunk | the chunk that closes the gripper (plan §3.1) |
| `branch_state` | full MuJoCo state (qpos, qvel, act, ctrl, time, warmstart) | save/restore point |
| `nominal` | (F₀, s₀) | demonstration parameters |
| `T_max` | {urgency: float} | deadline on the s-scaled carry per urgency level (plan §3.7); frozen per dataset version |
| `path` | waypoints + nominal segment durations | the fixed carry path (plan §3.1) |
| `grid` | F levels × s levels, R repeats | Phase 1: 8 × 6 × 5 |
| `solver` | timestep, `noslip_iterations`, `impratio`, cone, integrator | frozen per dataset version; logged |
| `split` | train / val / test-objects / test-combos / test-extrapolation / test-layout | grouped; see §3 |

### 1.2 Branch-level fields (one row per (F, s, repeat))

| Field | Type | Notes |
|---|---|---|
| `F`, `s` | float | commanded; `F` = actuator `forcerange` upper bound, `s` = carry time scaling |
| `F_norm`, `s_norm` | float in [0,1] | F̃ linear, s̃ log-linear (plan §3.1) |
| `perturbation` | {pose_delta, friction_noise, seed} | small per-repeat perturbation (plan §3.2) |
| `force_normal` | time series N(t) per pad | summed normal contact force on the object, per pad |
| `force_tangential` | time series T(t) per pad (2-vector) | summed tangential contact force per pad |
| `n_contacts` | time series per pad | number of active contact points (torsional resistance diagnostic) |
| `kinematics` | time series of object and gripper pose | carry kinematics; slip displacement (translation and rotation about the pad normal) derived from it |
| `a_realized` | {lift, lateral, lower, resultant} peak, 50 ms-smoothed (20 ms peak also stored) | realized end-effector acceleration from the simulated kinematics; used by all analyses and baselines instead of the commanded profile |
| `T_carry` | float | s-scaled carry duration; `deadline_success` compares it with `T_max` |
| `outcome` | {slip, damage_primary, damage_shear, physical_safe, completed, deadline_success_relaxed, deadline_success_urgent} ∈ {0,1}⁷ | plan §3.2, §3.5; `damage` in the feasible-set definition means `damage_primary`; fragility-dependent labels are stored once per fragility level |
| `dropped`, `drop_idx` | bool, int | the object left the gripper during the carry (centre > 0.08 m from the pad midpoint, or both pads out of contact for ≥ 50 ms during lift/lateral/lower); `drop_idx` = first sample of the drop. **Label-definition fix, 2026-11-02 (label version 2):** after a drop the branch outcome is the drop: `completed` is False regardless of where the object lands (`completed_geometric` keeps the purely geometric test for reference), and the damage proxies (`damage_primary_score`, `damage_shear_score`, sensitivity grid, impulse) are evaluated only over the samples before `drop_idx`, so an object crushed between the closing fingers after it fell is not labelled as grasp damage (the slip label already records the drop). Reason: Phase-1 check P6 on the 48-object sweep failed on 27 cells, 23 of which were drop cells whose `completed` / `damage_primary` disagreed across repeats only through where the dropped object landed. Labels of version 1 are kept beside the recomputed files (`summary.labels_v1.json`); the fix is applied by recomputing from the stored time series (`envs/feasibility_carry/relabel.py`), without re-simulation. |
| `completed` (label version 3, 2026-11-02, owner rule) | bool | `completed_geometric` ∧ ¬`lost_contact`, where `lost_contact` = both pads out of contact for ≥ 50 ms (run-length test) at any time before the scheduled release (lift / lateral / lower / settle). A slip whose grasp holds until the planned release keeps the geometric result; "any slip → False" is explicitly **not** the rule. Damage proxies stop at the drop onset as in version 2. Version-2 files kept as `summary.labels_v2.json`; recomputed from stored records by `relabel.py` (no re-simulation); re-evaluation `results/phase1/full_v3/`. |
| `cells[]` (per sweep, version 3) | per grid cell: `rates` (empirical rate of every label over the R repeats, stored alongside the binary branch labels), `stochastic` (bool), `disagreeing_labels` | **After label v3, any cell whose repeats still disagree is genuine physical stochasticity, not a label error** (owner decision 2026-11-02). Sweep v1: 213 of 2 304 grid cells are flagged stochastic on at least one label (most on `damage_shear` or `slip` at a boundary); the 18 flagged cells that lie outside every refined boundary's uncertainty are listed below. |
| `damage_primary_score` | max over t of min_{τ∈[t,t+t_win]} N(τ), per pad (max of the two) | sustained-peak crush proxy; `damage_primary` = score > f_tol at t_win = 50 ms |
| `damage_shear_score` | same with L(τ) = N(τ) + κ·|T(τ)|, κ = 1 | shear-augmented proxy; `damage_shear` = score > f_tol |
| `damage_sensitivity` | labels for f_tol × {0.75, 1, 1.25} × t_win ∈ {50, 100} ms × κ ∈ {0.5, 1, 2} | Phase-1 check P5; recomputable from the force series |
| `damage_impulse` | D = ∫ max(0, N − f_tol) dt (appendix only) | stress test, never a primary label |
| `actions` | per-step (Δx, ΔR, g) trajectory | required for every executed episode (𝓛_flow) |
| `replay_error` | contact-force and kinematics error on replay | see gate G5 |

### 1.2a Stochastic cells outside boundary uncertainty and known exceptions (sweep v1, label version 3)

Phase-1 check P6 (plan §6) requires repeat agreement ≥ 0.9 per cell for every label on cells outside the refined
boundaries' uncertainty. After label version 3, 22 (object, label, cell) entries remain below 0.9
(`results/phase1/full_v3/report.md`); none was cleared by the v3 rule relative to version 2, because in every one of
them the grasp held until the scheduled release. They are classified as follows (owner decision 2026-11-02):

**Stochastic cells (18 entries, 18 cells):** every repeat slipped more than 5 mm but the object stayed in the gripper until
the release; whether it then came to rest inside the 3 cm goal tolerance varied across repeats. The `completed` label is
genuinely stochastic there (`stochastic = true` in `cells[]`); the per-cell rate is the stored quantity.

| object | F [N] | s | label | agreement | P(slip) |
|---|---|---|---|---|---|
| light_slippery_balanced_2 | 22.6 | 0.5 | completed | 0.8 | 1.0 |
| light_slippery_balanced_2 | 22.6 | 0.715 | completed | 0.8 | 1.0 |
| light_slippery_off-centre_1 | 10.6 | 1.465 | completed | 0.8 | 1.0 |
| light_slippery_off-centre_1 | 10.6 | 2.096 | completed | 0.6 | 1.0 |
| light_slippery_off-centre_1 | 10.6 | 3 | completed | 0.8 | 1.0 |
| light_slippery_off-centre_2 | 15.5 | 1.465 | completed | 0.6 | 1.0 |
| medium_slippery_off-centre_0 | 32.9 | 1.024 | completed | 0.6 | 1.0 |
| medium_slippery_off-centre_0 | 32.9 | 1.465 | completed | 0.6 | 1.0 |
| medium_slippery_off-centre_0 | 32.9 | 3 | completed | 0.8 | 1.0 |
| medium_slippery_off-centre_1 | 32.9 | 0.5 | completed | 0.8 | 1.0 |
| heavy_slippery_balanced_2 | 32.9 | 0.5 | completed | 0.8 | 1.0 |
| heavy_slippery_balanced_2 | 32.9 | 0.715 | completed | 0.6 | 1.0 |
| heavy_slippery_off-centre_0 | 48.0 | 2.096 | completed | 0.8 | 1.0 |
| heavy_normal_balanced_0 | 22.6 | 0.5 | completed | 0.6 | 1.0 |
| heavy_normal_off-centre_0 | 22.6 | 3 | completed | 0.8 | 1.0 |
| heavy_normal_off-centre_3 | 22.6 | 1.465 | completed | 0.8 | 1.0 |
| heavy_normal_off-centre_3 | 22.6 | 2.096 | completed | 0.6 | 1.0 |
| heavy_normal_off-centre_3 | 22.6 | 3 | completed | 0.6 | 1.0 |

**Known exception (4 entries, 1 cell, not a label error, uncertainty not widened):** heavy_slippery_off-centre_1 at
F = 48.0 N, s = 1.02 slips in 4 of 5 repeats; the refined slip boundary's 95 % CI ∪ bisection bracket at that s does not
cover 48.0 N, so the cell counts as outside boundary uncertainty although it is a boundary cell physically. Per the owner
(2026-11-02) it is recorded here as a known exception rather than by widening the uncertainty definition.

| object | F [N] | s | label | agreement | P(slip) |
|---|---|---|---|---|---|
| heavy_slippery_off-centre_1 | 48.0 | 1.024 | slip | 0.8 | 0.8 |
| heavy_slippery_off-centre_1 | 48.0 | 1.024 | physical_safe/rigid | 0.8 | 0.8 |
| heavy_slippery_off-centre_1 | 48.0 | 1.024 | deadline_success/rigid/relaxed | 0.8 | 0.8 |
| heavy_slippery_off-centre_1 | 48.0 | 1.024 | deadline_success/rigid/urgent | 0.8 | 0.8 |

P6 as frozen therefore still reads "fail" on 22 entries; with the 18 stochastic cells and the 1 known-exception cell
set aside, no label disagreement outside boundary uncertainty remains. The verdict text in notes/decision_phase1.md
reports both readings; the owner decides which one closes P6.

### 1.3 Derived per group (computed, never hand-edited)

- All derived quantities below are computed **per context** (fragility level × urgency level) of the group.
- `P_hat[k][F][s]` for k ∈ {deadline_success, physical_safe, completed, slip, damage}: empirical rate over the R repeats.
- `feasible[F][s]`: P_deadline_success ≥ τ_c ∧ P_slip ≤ τ_s ∧ P_damage ≤ τ_d (initial τ = 0.9 / 0.05 / 0.05, frozen before main experiments).
- `feasible_phys[F][s]`: the same with P_physical_safe∧completed; analysis only, never a training target.
- `margin[F][s]`: distance in (F̃, s̃) space to the nearest infeasible cell, with virtual cells one step beyond the grid edges counted as infeasible; negative outside 𝓕.
- `q[F][s]` = P_deadline_success − α·P_slip − β·P_damage (α = β = 1), ranking loss and diagnostics only.
- `teacher_target` = (F*, s*) from plan §3.4, or `null` with `infeasible_by_design = true` when 𝓕 is empty.
- All derived quantities are stored once per sensitivity setting (proxy × f_tol multiplier × t_win × κ, §1.2), so Phase-1 check P5 (damage-definition and threshold sensitivity, plan §6) can be recomputed from stored data. P5 is a physics check, not a data gate.

## 2. Validation gates

Every dataset version must pass all gates before it is used for training or for a results table. Gate results are written
to `data/<version>/gates.json` with the numbers; a failing gate blocks the version. Thresholds come from plan §7 and are not
relaxed after the fact.

| ID | Gate | Check | Pass |
|---|---|---|---|
| **G1** | Scene validity | object and goal region in both camera views at the decision chunk; no interpenetration at the branch state (MuJoCo contact penetration below tolerance) | 100 % of groups |
| **G2** | Branch purity | within a group, every field except (F, s, perturbation) is identical; branch state hash matches across branches | 100 % of groups |
| **G3** | Split integrity | no `group_id`, and no `object_id` or property combo assigned to a held-out split, appears in train/val; boundary probes (§4a) carry their combination's split and never appear in training groups | 0 leaks |
| **G4** | Trivial-feature control | a classifier using only trivial features (object_id, group_id, object count) predicts branch outcomes on held-out groups | held-out AUROC ≤ 0.6, or accuracy ≤ majority-class baseline + 5 points (owner-approved 2026-10-29) |
| **G5** | Replay fidelity | replaying stored (a_{1:T}, F, s) from the branch state reproduces logged contact force and carry kinematics within tolerance (tolerance set in Phase 1 and recorded here) | ≥ 99 % of branches (owner-approved 2026-10-29) |
| **G6** | Label spot check | video of a random sample of branches is checked by a human against the stored outcome labels (all six labels of §1.2) | error ≤ 5 % per outcome class |
| **G7** | Independent second labeler | a second, independent implementation of the outcome labeler (different code path, same raw time series) is run on all branches | agreement ≥ 95 % per class; disagreements listed, none silently resolved |

Additional gate for derived quantities (code correctness, **same damage definition, same inputs** as the first implementation):

| ID | Gate | Check | Pass |
|---|---|---|---|
| **G8** | Feasible-set cross-check | a second, independent implementation computes `feasible`, `margin` and `teacher_target` from the stored `P_hat` | feasible sets identical on ≥ 98 % of grid cells; margins differ by ≤ 1 grid step; disagreements listed |

G7 and G8 together form the two-implementation cross-check. They test code, not physics. The agreement between the
primary and secondary **damage definitions** (Kendall τ ≥ 0.8 between feasible-set rankings, plan §3.5) is Phase-1 check
P5 and is reported in `notes/decision_phase1.md`, not here.

## 2a. Boundary probes (Phase 1, plan §6)

Bisection probes that refine the slip and damage boundaries in F (and s_rec in s) are stored under `probes/`, keyed by (combination, s_idx, probe_idx), with the same branch fields as §1.2 and R repeats each. They are analysis data: they never enter an interaction group, are never used for training, and inherit the split of their combination (G3).

## 3. Splits (grouped, never by frame)

| Split | Held out | Purpose |
|---|---|---|
| test-objects | object instances | generalization across geometry/appearance |
| test-combos | sampled objects (continuous property values) seen in no training group; one per bin cell | feasible-set shift with context under attribute uncertainty |
| test-extrapolation | combos outside the demonstration coverage (m, μ, fragility beyond the demo range) | demo-anchored selection (B7) versus margin-aware selection |
| test-layout | seen combos, new initial layout | layout robustness |
| val | grouped like train, disjoint groups | λ tuning only |

The test splits are never used for threshold or loss-weight tuning.

## 4. Pilot and scaling

1. Generate 30 groups; run G1–G8; record storage per group and wall-clock per branch.
2. Only after all gates pass: estimate the full budget and scale up.
3. Each dataset version gets a `VERSION.md` under `data/<version>/` with: solver settings, τ values, F and s ranges and
   levels, object list, property grid, noise ε, gate results, and the commit hash of the generator.

## 5. Storage (not in git)

```
data/<version>/
├── VERSION.md
├── gates.json
├── groups/<group_id>/
│   ├── group.json          (§1.1 fields; properties_true kept in a separate labels.json)
│   ├── labels.json         (properties_true, teacher_target, P_hat, feasible, margin — both damage definitions)
│   ├── obs/                (images, proprioception at the decision chunk)
│   ├── branch_state.npz
│   └── branches/<F_idx>_<s_idx>_<rep>.npz   (§1.2 fields)
├── probes/<combo>/<s_idx>/<probe_idx>/   (§2a; analysis only)
└── splits.json
```

`properties_true` lives only in `labels.json` so that a loader for the policy can exclude it by construction.
