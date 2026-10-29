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
| `properties_true` | {m, μ, fragility (f_tol, D_max)} | **labels and oracle setting only** |
| `properties_coarse` | {mass_bin, friction_bin, fragility_bin} + `noise_eps` | binned text attributes as the VLA sees them (plan §3.6); noise applied at generation time and logged |
| `instruction` | str | task language |
| `obs_decision` | two camera views + proprioception at the decision chunk | the chunk that closes the gripper (plan §3.1) |
| `branch_state` | full MuJoCo state (qpos, qvel, act, ctrl, time, warmstart) | save/restore point |
| `nominal` | (F₀, s₀) | demonstration parameters |
| `grid` | F levels × s levels, R repeats | Phase 1: 8 × 6 × 5 |
| `solver` | timestep, `noslip_iterations`, `impratio`, cone, integrator | frozen per dataset version; logged |
| `split` | train / val / test-objects / test-combos / test-extrapolation / test-layout | grouped; see §3 |

### 1.2 Branch-level fields (one row per (F, s, repeat))

| Field | Type | Notes |
|---|---|---|
| `F`, `s` | float | commanded; `F` = actuator `forcerange` upper bound, `s` = carry time scaling |
| `F_norm`, `s_norm` | float in [0,1] | F̃ linear, s̃ log-linear (plan §3.1) |
| `perturbation` | {pose_delta, friction_noise, seed} | small per-repeat perturbation (plan §3.2) |
| `force_contact` | time series f_n(t) per finger | measured normal force |
| `kinematics` | time series of object and gripper pose | carry kinematics; slip displacement derived from it |
| `outcome` | {slip, damage, complete} ∈ {0,1}³ | plan §3.2 definitions |
| `damage_impulse` | D = ∫ max(0, f_n − f_tol) dt; `damage_primary` = D > D_max | primary definition (plan §3.5) |
| `damage_peak` | max f_n; `damage_secondary` = max f_n > f_tol | secondary definition |
| `actions` | per-step (Δx, ΔR, g) trajectory | required for every executed episode (𝓛_flow) |
| `replay_error` | contact-force and kinematics error on replay | see gate G5 |

### 1.3 Derived per group (computed, never hand-edited)

- `P_hat[k][F][s]` for k ∈ {complete, slip, damage}: empirical rate over the R repeats.
- `feasible[F][s]`: P_complete ≥ τ_c ∧ P_slip ≤ τ_s ∧ P_damage ≤ τ_d (initial τ = 0.9 / 0.05 / 0.05, frozen before main experiments).
- `margin[F][s]`: distance in (F̃, s̃) space to the nearest infeasible cell; negative outside 𝓕.
- `q[F][s]` = P_complete − α·P_slip − β·P_damage (α = β = 1), ranking loss and diagnostics only.
- `teacher_target` = (F*, s*) from plan §3.4, or `null` with `infeasible_by_design = true` when 𝓕 is empty.
- All derived quantities are stored twice, once per damage definition (§1.2), so the P5 comparison is reproducible.

## 2. Validation gates

Every dataset version must pass all gates before it is used for training or for a results table. Gate results are written
to `data/<version>/gates.json` with the numbers; a failing gate blocks the version. Thresholds come from plan §7 and are not
relaxed after the fact.

| ID | Gate | Check | Pass |
|---|---|---|---|
| **G1** | Scene validity | object and goal region in both camera views at the decision chunk; no interpenetration at the branch state (MuJoCo contact penetration below tolerance) | 100 % of groups |
| **G2** | Branch purity | within a group, every field except (F, s, perturbation) is identical; branch state hash matches across branches | 100 % of groups |
| **G3** | Split integrity | no `group_id`, and no `object_id` or property combo assigned to a held-out split, appears in train/val | 0 leaks |
| **G4** | Trivial-feature control | a classifier on (object_id, group_id, object count) alone predicts branch outcomes no better than the majority rate on val | AUROC ≤ 0.55 (or not above chance with CI) |
| **G5** | Replay fidelity | replaying stored (a_{1:T}, F, s) from the branch state reproduces logged contact force and carry kinematics within tolerance (tolerance set in Phase 1 and recorded here) | ≥ 99 % of branches |
| **G6** | Label spot check | video of a random sample of branches is checked by a human against the stored outcome labels | error ≤ 5 % per outcome class |
| **G7** | Independent second labeler | a second, independent implementation of the outcome labeler (different code path, same raw time series) is run on all branches | agreement ≥ 95 % per class; disagreements listed, none silently resolved |

Additional gate for derived quantities:

| ID | Gate | Check | Pass |
|---|---|---|---|
| **G8** | Feasible-set cross-check | a second implementation computes `feasible`, `margin` and `teacher_target` from the stored `P_hat`; feasible-set rankings across the two damage definitions compared | identical feasible masks; Kendall τ ≥ 0.8 between damage definitions (plan §3.5) |

## 3. Splits (grouped, never by frame)

| Split | Held out | Purpose |
|---|---|---|
| test-objects | object instances | generalization across geometry/appearance |
| test-combos | property combinations seen in no training group | feasible-set shift with context |
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
└── splits.json
```

`properties_true` lives only in `labels.json` so that a loader for the policy can exclude it by construction.
