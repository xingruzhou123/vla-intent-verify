# Research Plan — Counterfactual Feasibility Distillation for Physically Safe VLA Control

**Working title:** *Counterfactual Feasibility Distillation: Teaching VLAs the Context-Dependent Range of Safe Execution*
**Owner:** Grace (UT Austin)
**Plan date:** 2026-10-29
**Status:** Replaces the 2026-09-23 plan *"Is This Action Licensed? Verifying VLA Policy Intent Against Scene Evidence"* in full. The old plan is kept as `plan_v0.1.md` for reference and is not active.
**Targets:** RSS 2027 Stage 1 (2026-12-04, conditional on the Phase-1 gate) → IEEE T-RL.

---

## 1. Problem

**Research question**

> How can a VLA learn not only *which* action to execute, but also the context-dependent *range of execution parameters* that completes the task without slip or damage?

**Motivation (paper wording)**

> Current VLAs are trained primarily from single demonstrated executions and therefore receive limited supervision about how continuous execution parameters trade off task completion against slip and physical damage, or about how this trade-off shifts as object properties change.

**Scope of version 1:** pick-and-carry with a parallel-jaw gripper. Two execution parameters:

- **F** — maximum gripper force for the grasp;
- **s** — time scaling of the carry segment (fixed path; velocity ∝ 1/s, acceleration ∝ 1/s²).

Object properties that move the feasible set: mass *m*, friction coefficient *μ*, fragility (damage tolerance).

---

## 2. Method chain

```
same pre-contact state
  → counterfactual sweep over F × s   (outcomes: slip / damage / completion)
  → empirical feasible set + safety margin per interaction group
  → teacher selects a margin-maximizing (F*, s*) near the nominal parameters
  → distill into π0.5 with an extended action interface
      + auxiliary outcome head + ranking loss on all branches
  → deploy π0.5 alone: it outputs (F, s) directly; no external selector
```

---

## 3. Definitions (frozen)

### 3.1 Action interface

- Per-step actions keep the existing format: `a_t = (Δx_t, ΔR_t, g_t)`, as in the LIBERO π0.5 checkpoints.
- **F and s are primitive-level scalars.** They are predicted **once**, by a parameter head, at the pre-grasp decision chunk (the chunk that closes the gripper), and held fixed for that grasp-and-carry primitive.
- **F**, simulation: the gripper actuator's `forcerange` upper bound (force-limited closing). **F**, real robot: the `force` argument of the Franka Hand grasp command. The simulated range is mapped to the Franka Hand's 30–70 N continuous range.
- **s**, both: retiming of the carry segment's reference trajectory. The executor stretches the time base of the executed chunks by s.
- **Encoding:** `F̃ = (F − F_lo)/(F_hi − F_lo) ∈ [0,1]`; `s̃ = (log s − log s_lo)/(log s_hi − log s_lo) ∈ [0,1]`.
- **Ranges and nominal path (frozen 2026-10-29):** F_lo = 5 N, F_hi = 70 N, in Newtons with no affine rescaling (the Franka Hand's 30–70 N window is a subset; the calibration table in §10 decides the usable floor). s_lo = 0.5, s_hi = 3, nominal s₀ = 1. Nominal carry path: 0.12 m lift in 0.5 s, 0.30 m lateral **along the gripper's closing axis** in 0.75 s, lower in 0.5 s (T_carry(s₀) = 1.75 s); min-jerk segments, tracked by OSC in absolute mode. Grasp closing, settle and release segments are not scaled by s.
- This **extends π0.5's action space**. The paper states this explicitly; it is not plain fine-tuning of an unmodified model.
- **Acceptance test:** replaying stored `(a_{1:T}, F, s)` in simulation reproduces the logged contact force and carry kinematics within tolerance.

### 3.2 Outcomes

For each branch (one (F, s) execution from the shared branch state):

- `slip` — the object's displacement in the gripper frame relative to its post-grasp position exceeds δ_slip = 5 mm at any time during the carry (translation of the CoM, or rotation about the pad normal measured as pad-edge displacement: torsional slip), or the object leaves the gripper;
- `damage` — the Phase-1 primary damage proxy fires (§3.5);
- `physical_safe` — not `slip` and not `damage`;
- `completed` — the object rests in the goal region after release (xy within 3 cm of the goal, resting on the support);
- `deadline_success` — `completed` and `physical_safe` and the s-scaled carry duration T_carry(s) ≤ T_max(u), where u is the urgency level of the context (§3.7). One label per urgency level, computed post hoc from the same branch.

`deadline_success` is the primary deployment label. A **context** is (property combination, fragility, urgency level); fragility and urgency are label parameters, so every context of one physical sweep shares its branches. Physically-safe success (`physical_safe ∧ completed`) and the completion time are always reported separately, so a slow-but-safe policy is visible as such rather than hidden.

Each (F, s) cell is repeated *R* times with small perturbations (initial pose, friction noise), **paired across cells**: repeat r uses the same perturbed branch state for every (F, s). The outcome probability `P̂_k(F,s)` for k ∈ {deadline_success, physical_safe, completed, slip, damage} is the empirical rate over those repeats. Every branch also logs the realized peak end-effector acceleration (lateral, vertical, resultant) from the simulated kinematics; all analyses and both analytic baselines use realized, not commanded, acceleration.

### 3.3 Feasible set and margin

```
𝓕(x) = { (F,s) : P_deadline_success ≥ τ_c,  P_slip ≤ τ_s,  P_damage ≤ τ_d }
```

- Thresholds `τ_c, τ_s, τ_d` are fixed before the main experiments (initial values: 0.9 / 0.05 / 0.05).
- 𝓕(x), the margin and the teacher target are defined **per context** (property combination × fragility × urgency level); the deadline enters through `P_deadline_success` only.
- **Margin:** `m(F,s; x)` = distance, in normalized (F̃, s̃) space, from (F,s) to the nearest grid cell outside 𝓕(x). Virtual cells one grid step beyond every edge count as outside 𝓕 (𝓕 ⊂ grid), so margins are bounded and the maximiser sits inside the feasible band; cells that fail the deadline are outside 𝓕. It is computed from the empirical sweep and is negative outside 𝓕. Ties in the teacher's argmax are broken toward the nominal s.
- **Feasibility score:** `q(F,s) = P_deadline_success − α·P_slip − β·P_damage`, with initial values α = β = 1. This score is used only by the ranking loss and for diagnostics.
- **Physically-safe feasible set** 𝓕_phys(x): the same definition with `P_physical_safe ∧ completed` in place of `P_deadline_success`. It is reported alongside 𝓕(x) for analysis (how much of the physically safe region the deadline removes) and is never the training target.

### 3.4 Teacher selection

```
(F*, s*) = argmax_{(F,s) ∈ 𝓕(x)}  m(F,s; x) − γ·‖(F̃,s̃) − (F̃₀,s̃₀)‖²
```

- (F₀, s₀) are the nominal parameters of the demonstration.
- If 𝓕(x) is empty, the teacher returns no target, and the group is flagged as infeasible-by-design.
- **Round 1** uses the empirical sweep.
- **Round 2** (student states, which have no sweep) uses a learned teacher outcome model `T(o, c, F, s)`. Its inputs are the same deploy-time inputs as the VLA's. Its targets are kept only where its predicted margin exceeds a confidence threshold.

### 3.5 Damage model

Measured pad force in the simulator tracks the commanded bound F almost exactly (pilot 2026-10-29: 0.9998·F, R² = 1.000), so any damage rule evaluated on steady grasp force reduces to a threshold on F. The definitions below are therefore **Phase-1 damage proxies**, not realistic material models. What makes their boundaries non-trivial is the part of the pad load that F does not fix (the contextual variables below).

**Primary (Phase 1): sustained peak-crush proxy.** Per pad, N(t) is the summed normal contact force on the object. `damage_primary` fires if N(t) > f_tol on either pad **continuously for t_win = 50 ms**. Initial thresholds: fragile f_tol = 20 N, rigid f_tol = 50 N, subject to the sensitivity study (P5). t_win = 50 ms is shorter than the acceleration pulse of the fastest carry (≈ 0.1 s at s = 0.5 on the nominal path), so carry-induced normal-force excursions count, and longer than contact-onset spikes (a few ms). Longer windows (≥ 200 ms) would miss the dynamic excursion and reduce the proxy to F > f_tol.

**Second (Phase 1): shear-augmented load proxy.** Per pad, L(t) = N(t) + κ·|T(t)|, with T(t) the tangential contact force on that pad (summed over its contact points) and κ = 1. `damage_shear` fires if L(t) > f_tol continuously for t_win. The tangential load is ≈ m·(g + a_realized)/2 per pad plus the CoM-torque contribution, so this boundary depends on mass, carry dynamics and CoM offset even when N(t) = F exactly. Both proxies are computed from constraint forces, not penetrations, so they do not depend on the soft-contact parameters (solref / solimp) beyond those parameters' effect on the forces themselves. κ ∈ {0.5, 1, 2} is part of the sensitivity study.

**Carry direction (design choice).** The lateral carry runs along the gripper's closing axis on purpose: it is the direction in which the object's inertial load is carried by the pad normal forces rather than by friction, so it is the direction in which speed changes the pad load and hence the damage boundary. The real-robot sweep (§10) uses the same carry direction, so the simulated and measured rankings are compared on the same mechanism.

**Contextual variables that make the proxy boundaries non-trivial.** (i) The lateral carry runs along the gripper's closing axis, so the object's inertial load m·a_y is carried by a normal-force difference between the pads: the trailing pad sees ≈ F + m·a_y/2 (pilot: +0.5 N at 0.3 kg and 3 m/s²; ≈ +5 N at 1.5 kg and the controller's 7 m/s² ceiling). (ii) The tangential load m(g + a)/2 per pad (second proxy). (iii) The CoM offset: its gravity torque adds tangential load at the pad edges (second proxy) and, if the offset has a vertical component, edge-loads the pad. (iv) The contact-onset transient of the closing ramp (≈ 2 % overshoot here, independent of s). Variables that do **not** move the boundary in this gripper: object width and finger gain (the actuator is force-saturated).

**Appendix stress test only: accumulated excess impulse** D = ∫ max(0, N(t) − f_tol) dt, `damage_impulse` if D > D_max. Because F is held for the whole primitive, D grows with duration and penalises slow carries, which is the opposite of the slip mechanism; it is reported in the appendix as a robustness check and is never a primary label.

**Secondary option (used only if the two proxies disagree): contact-area pressure** p = N / A_contact, with A_contact the area of the convex hull of the pad's contact points. It remains a scaled threshold on F and is listed for completeness only.

**Real-robot validation of the damage ranking (§10).** Pressure-indicating film on both finger pads (ultra-low-range sheet, ≈ 0.05–0.6 MPa; 20–70 N over the 16 × 16 mm pad face is ≈ 0.08–0.27 MPa — verify the sheet grade before ordering) records the peak pressure of every hardware grasp. The fragile proxy object is additionally a deformable cup whose residual deformation after release is measured with a caliper. **Transfer test:** over the hardware F × s sweep, Kendall τ ≥ 0.8 between the simulated damage score (sustained-peak N for the primary proxy, sustained-peak L for the second) and the measured peak pressure, and for the cup between the simulated score and residual deformation. The film measures normal pressure only, so the shear term of the second proxy is validated through the ordering across s rather than directly.

### 3.6 Information settings (never pooled)

| Setting | VLA inputs besides images, proprioception and instruction | Role |
|---|---|---|
| **Oracle** | exact m, μ, fragility | upper bound |
| **Coarse (main claim)** | binned attributes in text (light/medium/heavy, slippery/normal, fragile/rigid, balanced/off-centre), with label noise ε = 0.1 (each bin independently replaced by a neighbouring bin with probability ε) | main results |
| **Vision-only** | none | generalization test; meaningful only where appearance correlates with properties |

**Main claim:** under observable but imprecise attribute information, a learned, margin-aware feasible set is more robust than analytic rules and single-point imitation. We do **not** claim to infer unobservable physical properties from vision.

---

### 3.7 Time budget (contextual deadline)

A fixed slow policy can be physically safe yet operationally unusable, so the deadline is a separate label (§3.2), not a change to the physics labels. With a single fixed T_max every physical mechanism (slip need, trailing-pad load, shear) favours slower carries, so the optimal s would be "slowest allowed minus margin" for every object and s would stop being context-dependent. The deadline is therefore a **context variable with two predeclared levels**; the level appears in the instruction text ("… quickly" / "… carefully", wording fixed in data pipeline v1), and deadline labels, feasible sets, margins and teacher targets are computed per (property combination, fragility, urgency level) from the same branches — no extra simulation.

- The s-scaled part of the primitive is the carry (lift, lateral, lower); T_carry(s₀) = 1.75 s. Grasp closing, settle and release are not scaled and are excluded from the budget.
- **relaxed:** T_max = 2 × T_carry(s₀) + 0.5 s = **4.0 s**, binds at s = 2.29 — five of the six grid levels (0.50, 0.72, 1.02, 1.47, 2.10) pass; s = 3.00 fails and is kept in the sweep as a physically informative cell.
- **urgent:** T_max = 1.5 × T_carry(s₀) + 0.5 s = **3.125 s**, binds at s = 1.79 — four levels pass; 2.10 and 3.00 fail.
- Alternative urgent level, evaluated post hoc in the pre-sweep demonstrations and decided before the full sweep: 1.0 × T_carry(s₀) + 0.5 s = 2.25 s (binds at s = 1.29, three levels pass), which separates the two contexts by two grid steps instead of one.
- Both levels are frozen before the full sweep. Rejected: an episode-level cycle budget such as 6 s (binds at s = 3.43; nothing in the grid fails and the label is vacuous).

---

## 4. Model and training objective

**Architecture:** π0.5 (LeRobot implementation), extended with two heads that read the shared latent `z`:

- **Parameter head** `h_θ(z) → (F̃, s̃)`;
- **Outcome head** `g_φ(z, F̃, s̃) → (P̂_complete, P̂_slip, P̂_damage)`.

For both heads, `z` is a pooled action-expert representation at the decision chunk. The choice of layer and flow step is fixed in Phase 2 and recorded.

**Loss:**

```
𝓛 = 𝓛_flow + λ_p·𝓛_{F,s} + λ_o·𝓛_outcome + λ_r·𝓛_rank
```

| Term | Data | Definition |
|---|---|---|
| `𝓛_flow` | per-step actions of feasible executions | original π0.5 flow-matching loss on (Δx, ΔR, g) |
| `𝓛_{F,s}` | decision-chunk states with a teacher target | MSE (or Gaussian NLL) between h_θ(z) and (F̃*, s̃*) |
| `𝓛_outcome` | **all** branches, including slip and damage | BCE between g_φ(z, F̃, s̃) and the empirical outcome rates |
| `𝓛_rank` | pairs within one interaction group: (feasible, too-loose) and (feasible, too-tight), sampled separately | pairwise logistic loss `−log σ(q̂(feasible) − q̂(infeasible))`, where q̂ is computed from g_φ |

- Gradients from `𝓛_outcome` and `𝓛_rank` flow into the shared latent (LoRA on π0.5). This is how the feasibility information reaches the policy.
- Loss weights λ are tuned on a validation split (grouped). The test split is never used for tuning.

**Deployment (frozen rule):** π0.5 outputs (F, s) directly from h_θ. **No external selector and no parameter search.** g_φ may be kept only for diagnostics and never influences the executed parameters. An "external selector" variant is reported separately as a reference, not as our method.

---

## 5. Related work and positioning

| Work | What it does | Our increment (claim exactly this, no more) |
|---|---|---|
| **VisualForce** (arXiv 2610.04741) | Visual force estimation from compliant-gripper deformation; an action–force proposal policy; at test time samples candidates and executes the one whose predicted force is closest to a target force from demonstrations | Its inference target is anchored to a demonstrated target force. It does not explicitly learn a feasible set jointly defined by slip, damage and completion outcomes, and does not explicitly optimize margin from failure boundaries. We do both, over F × s jointly. **Do not claim it cannot extrapolate.** |
| **FARM** | Diffusion policy jointly predicting pose, grip width and grip force; dynamic force adaptation with tactile input | We need no deploy-time tactile sensing, and we supervise failure boundaries with counterfactual branches |
| **V2F** (arXiv 2607.19804) | Predicts a safe grasp force from pre-contact vision | We learn the full success–slip–damage set and its shift with context, not one force value |
| **FAVLA** (arXiv 2602.23648) | Force-adaptive fast–slow VLA using F/T feedback | Same: no F/T feedback at deploy time; explicit boundary supervision |
| **Proactive slip control** (Nazari et al., arXiv 2209.06019) | Learned action-conditioned slip predictor + trajectory adaptation when force cannot be increased | Closest precedent for F–motion coupling. Ours: within a VLA, adds a damage upper bound, uses counterfactual branch data, and distills into the policy |
| **ManiGuard, OopsieVerse, SafeVLA-Bench, HazardArena** | Safety evaluation; success ≠ safety; damage simulation; safe SFT / DPO baselines | Outcome differences attributable to execution parameters via same-state branching; continuous margin rather than binary labels |
| **SafeBranch** (arXiv 2608.19729) | Decision-point safe/unsafe branch pairs over high-level symbolic actions | Continuous, low-level execution parameters; two-sided boundaries; margin |
| **CounterAlign** (arXiv 2608.21740) | Counterfactual relabeling + advantage-weighted flow matching on π0.5 | Method precedent for training flow VLAs without likelihoods; not about physical safety |

**Claims we must NOT make:** first force-aware VLA; first visual force prediction; first same-state safe/unsafe branching; safety guarantees.

**Contributions:**

1. **Same-state F × s counterfactual branch data**, with outcomes attributable to the execution parameters.
2. **Context-conditioned feasible-set learning inside the VLA**, through an auxiliary outcome head and a ranking loss over all branches.
3. **Margin-aware parameter selection distilled into π0.5**, deployed without an external selector.

Latent probing is analysis only (§9), not a contribution.

---

## 6. Phase 1 — Physical-mechanism go/no-go (2026-10-30 → 2026-11-09)

No π0.5 training until this gate passes.

**Environment:** `envs/feasibility_carry/` — pick-and-carry on the robosuite/MuJoCo Panda used by LIBERO, with a force-limited gripper (`forcerange` = F; the finger target is driven past the joint range so the actuator saturates at F), a parametric box (m, μ, fragility, CoM offset along the horizontal axis orthogonal to closing, set through the body's inertial position), the fixed min-jerk carry path of §3.1 retimed by s, and save/restore of the full MuJoCo integration state at the pre-contact branch point. Frozen environment settings (2026-10-29): canonical top-down grasp with vertical pads (the Panda home orientation tilts the pads 7° about the closing axis and makes the box rock on pad edges); object contact `solref` = (0.005 s, 1) with geom priority 1 (robosuite's 0.02 s default lets a held object creep ≈ 4 mm across the pads under the oscillating normal load of a fast carry; 0.005 s reduces this to < 0.8 mm with unchanged force tracking); slip is measured relative to the finger pads, not the hand, because the finger joints (damping 100 N·s/m, 1 N friction loss) let the finger–object assembly drift along the closing axis under lateral acceleration without any pad slip. The pilot (2026-10-29) verified force tracking (R² = 1.000), branch restore and 0.7 s wall-clock per branch.

**Grid (frozen; this is also the training-data grid):**

- F: 8 log-spaced levels in [5, 70] N — 5.0, 7.3, 10.6, 15.5, 22.6, 32.9, 48.0, 70.0
- s: 6 log-spaced levels in [0.5, 3] — 0.50, 0.72, 1.02, 1.47, 2.10, 3.00 (T_max = 4.0 s binds at s = 2.29, §3.7)
- R = 5 paired repeats per cell
- Properties: m ∈ {0.5, 1.0, 1.5} kg × μ ∈ {0.2, 0.4} × CoM ∈ {centred, offset} = **12 physical combinations**, each labelled under fragility ∈ {fragile f_tol = 20 N, rigid f_tol = 50 N} and urgency ∈ {relaxed, urgent} = **48 contexts** (fragility and the deadline are label parameters and do not enter the simulation)
- CoM offset level: 0.15 w = 6 mm proposed; the pre-sweep demonstration (b) measured the slip-boundary shift at 0.15 w, 0.20 w and 0.25 w (results/phase1/demo_report) — level to be fixed by the owner before the sweep
- Object: 4 × 4 × 6 cm box (w = 4 cm along the closing axis); δ_slip = 5 mm
- 8 × 6 × 5 × 12 = 2880 branches ≈ 35 min on the RTX 5080 host (MuJoCo is CPU-bound; 16 cores, combos run in parallel), plus ≈ 250 probe branches per combination for the boundary refinement

**Boundary refinement (not training data):** for every (combination, s), after the grid, the slip boundary and each damage boundary in F are refined by bisection between the bracketing grid levels, with R repeats per probe, until the bracket is ≤ 2 N or four probes have been spent. For P3, the recovery threshold s_rec is bisected in s between the nominal level and the slowest deadline-feasible grid level that is feasible, at the best grid F. A boundary estimate is the 50 % crossing of a logistic fit over the grid and probe outcomes at that s; its 95 % CI comes from a bootstrap over repeats, and when every repeat agrees (the simulation is near-deterministic) the reported uncertainty is the final bisection bracket (≤ 2 N). Probes are stored under `probes/` and never enter training groups (docs/data_spec.md §4).

**Realized acceleration:** every branch logs the peak realized end-effector acceleration (lift, lateral, lower, resultant), smoothed over 50 ms to match the damage window (the 20 ms peak is also stored; it carries the 20 Hz controller ripple). Boundary analyses and both analytic baselines use realized acceleration; the commanded profile is reported only to show controller saturation (measured 2026-10-29 with kp = 150: lift 3.0 / 2.3 / 1.6 / 0.95 / 0.55 / 0.31 m/s² and lateral 4.7 / 3.1 / 2.0 / 1.2 / 0.7 / 0.35 m/s² at s = 0.5 … 3.0, against commanded 11 m/s² at s = 0.5).

| ID | Check | Pass criterion |
|---|---|---|
| **P1** | Force control is real | Steady-state measured pad normal force tracks commanded `forcerange` monotonically (R² ≥ 0.9) over the 8 F levels and all combinations. Fingers are not at the position limit, and the actuator is saturated only by the force bound. |
| **P2** | s moves a boundary | For ≥ 2 property combinations, the refined slip or damage boundary in F shifts by ≥ 15 % of (F_hi − F_lo) = 9.75 N between the fastest and the slowest deadline-feasible s, with non-overlapping 95 % CIs. |
| **P3** | Slow-down recovery region exists | For ≥ 2 property combinations: at nominal s no grid F is deadline-successful (every F slips or triggers the primary damage proxy), and at some slower, deadline-feasible s (s ≤ 2.29) a deadline-successful F exists. Counted under the primary proxy at f_tol × 1.0. Combinations whose feasible set is empty at every s are "infeasible by design" and are excluded from the denominator. |
| **P4** | Analytic rules are not near-perfect under realistic information | With coarse attributes (uncalibrated and calibrated on the training split; docs/analytic_baselines.md) and realized acceleration, the stronger of the two analytic baselines (translational Coulomb; wrench-aware) selects deadline-infeasible parameters on ≥ 10 % of held-out combinations, or misplaces a boundary by more than the median feasible-set width / 4. |
| **P5** | Damage-definition and threshold sensitivity | Over f_tol × {0.75, 1.0, 1.25}, t_win ∈ {50, 100} ms and both proxies (κ ∈ {0.5, 1, 2} for the second): damage boundaries move monotonically with f_tol and with t_win; the P2 and P3 verdicts are unchanged for every setting in which the ground-truth feasible set is non-empty; combinations that become infeasible under a setting are reported, not counted as failures; Kendall τ ≥ 0.8 between feasible-set rankings under the two proxies. The impulse definition is reported in the appendix only. |
| **P6** | Repeatability | Outcome agreement across the 5 repeats ≥ 90 % per cell for all five labels (tune solver timestep, `noslip_iterations` and `impratio` if needed; log the final values). |
| **P7** | Context dependence of s | For each context (property combination × fragility × urgency level), take the margin-maximising (F, s) on the refined feasible set. Pass if ≥ 25 % of the contexts with a non-empty feasible set select an s different from the modal s (threshold declared 2026-10-29, before the sweep). Reported for the grid teacher as well. |

**Decision rule:**

- **P1, P2 or P3 fails →** stop. Re-scope to a force-only study, and do not train π0.5 on F × s. The RSS submission is cancelled; re-plan for T-RL.
- **P7 fails →** stop and report: the F × s claim would reduce to force selection plus a fixed slow-speed rule.
- **P4 fails →** continue, but narrow the claim to robustness under attribute uncertainty; the analytic baselines become headline comparisons.
- **P5 or P6 fails →** fix the simulation before anything else.
- **RSS decision is made on 2026-11-09:** proceed with the Stage-1 track only if P1–P3, P6 and P7 pass.

**Pre-sweep demonstrations (required before the 5760-branch run):** rerun the pilot on (a) one heavy–slippery–fragile combination and (b) one CoM-offset combination, and show (1) an interior P3 recovery region — increasing F violates the primary damage proxy while reducing speed restores deadline-constrained safe completion; (2) torsional slip caused by the CoM offset; (3) both analytic baselines overlaid on the simulated boundaries, in oracle and coarse settings; (4) P7 and the best-single-fixed reference on those combinations. Done 2026-10-29: results/phase1/demo_report/report.md (see status.md).

**Best-single-fixed reference (post hoc):** the single grid (F, s) that maximises mean deadline-constrained safe success over all training contexts (both urgency levels). Its success rate is reported next to the per-context oracle (best cell per context); it bounds how strong baseline B4 can get.

**Outputs:** `results/phase1/` (feasible-set heatmaps for deadline_success (per urgency level) and physical_safe per combination, refined F_min(s) and F_dmg(s) curves with CIs, P3 recovery examples, analytic overlays for both baselines, P5 sensitivity grid, repeatability table, P7 table, best-single-fixed reference) and `notes/decision_phase1.md`.

---

## 7. Data

**Interaction group** = one pre-contact state together with all of its (F, s) branches. Each group stores:

- observations at the decision chunk: two camera views, proprioception, instruction (which states the urgency level), the coarse-attribute text (including the balance bin), and true attributes m, μ, f_tol and CoM offset (labels only);
- per branch: commanded (F, s), per-pad normal and tangential contact force, contact-point count per pad, carry kinematics with realized peak acceleration, T_carry, the outcome labels of §3.2 (deadline_success once per urgency level), and the damage scores of both Phase-1 proxies (impulse damage in the appendix fields);
- for every executed episode: the full per-step action trajectory, used for 𝓛_flow.

**Splits (grouped, never by frame):**

- held-out object instances;
- held-out property combinations, including **extrapolation combos** outside the demonstration coverage (these directly test demo-anchored selection);
- seen combos with a new initial layout.

**Validation gates** (defined in `docs/data_spec.md`): objects in view and no interpenetration; branches within a group differ only in (F, s); no group crosses splits; trivial features (object ID, group ID, object count) cannot predict outcomes; replay matches execution; spot checks of video against labels, with error ≤ 5% per class; an independent second labeler with agreement ≥ 95%.

**Pilot:** 30 groups → pass the gates → estimate storage and time → scale up.

---

## 8. Baselines, ablations and metrics

### Baselines (matched data budget, steps and inputs)

| ID | Baseline | Question it answers |
|---|---|---|
| B0 | Raw π0.5 with default (F₀, s₀) | Where the starting point is |
| B1 | Standard demonstration SFT, fixed (F₀, s₀) | Is the gain just extra training? |
| B2 | **Same branch data, SFT on teacher targets only** (𝓛_flow + 𝓛_{F,s}; no outcome or rank loss) | **Key ablation: does the gain come from the data or from learning the feasible set?** |
| B3 | Binary pairs (SafeBranch-style): one feasible and one infeasible branch per group, pairwise loss only | Does continuous, two-sided structure beat binary preference? |
| B4 | Fixed conservative force and speed limits | Is learning better than conservatism? |
| B5a | Translational Coulomb rule m(g + a_realized)/(2μ) with the damage cap, coarse attributes, uncalibrated and calibrated on the training split (docs/analytic_baselines.md) | Does learning beat deployable physics? |
| B5b | Wrench-aware rule (CoM torque, pad contact geometry, friction limit surface, shear-augmented damage cap), coarse attributes, uncalibrated and calibrated | Does learning beat a physics model that knows about the CoM offset? |
| B6 | B5a and B5b with oracle attributes (reference only) | Upper reference |
| B7 | Demo-anchored selection (VisualForce-style): sample K (F,s) candidates from the policy, choose the one closest to the demonstration target force | Margin-aware feasible set versus a demo-anchored target, especially on extrapolation combos |
| R1 | Ours + external selector at deployment (reference only) | How much distillation loses |
| R2 | Best single fixed (F, s), chosen post hoc on the training contexts (reference only; §6) | Upper bound on B4; how much context dependence is worth |

### Ablations

Remove `𝓛_outcome`; remove `𝓛_rank`; remove the margin term (select any feasible point); vary γ; run each information setting (oracle / coarse / vision-only).

### Metrics

- **Primary:**
  - **deadline-constrained safe success rate** (primary) and physically-safe success rate on held-out and extrapolation combos, with completion time reported separately;
  - the four outcome classes (safe success, unsafe success, safe failure, unsafe failure);
  - **infeasible-selection rate** (chosen parameters fall outside 𝓕);
  - **realized margin** of the chosen parameters (empirical, from the sweep);
  - **context adaptation:** sign agreement between the change in the chosen (F, s) and the change in the ground-truth feasible-set centre when one property is varied.
- **Feasibility learning:** outcome-head AUROC and boundary-location error on held-out combos.
- **Secondary:** stability under small parameter perturbations; decision-point preference.
- **Statistics:** paired initial states; ≥ 3 seeds; bootstrap 95% CIs over interaction groups. Leave result cells empty until measured.

---

## 9. Analysis (not a contribution)

**Latent probing.** Linear probes for outcome, margin and object attributes on frozen representations from:

- raw π0.5;
- B1;
- B2;
- ours;
- a matched-compute random-label control.

Probes are tested on held-out objects and combos. The framing is *"does feasibility-aware training change what the representation encodes?"* No expected result is pre-specified.

**Failure analysis.** Separate attribute-information errors from feasible-set prediction errors from distillation errors.

---

## 10. Real robot (Franka Panda + Franka Hand)

1. **Calibration:** commanded versus measured grip force with a load-cell fixture, over 30–70 N (and 10–30 N attempted). The Franka Hand has no force sensor; this table is mandatory.
2. **Proxy objects (2):** chosen so that slip / feasible / damage regions all exist within 30–70 N (e.g., a weighted low-friction block, a deformable cup).
3. **Boundary check:** a small F × s sweep on hardware; report ranking agreement between the simulated and real boundaries.
4. **Closed-loop evaluation (journal version):** the distilled policy versus B1, B2 and B5 on the proxy objects, with CIs.

---

## 11. Timeline

| Dates | Work | Exit criterion |
|---|---|---|
| 10-30 → 11-09 | Phase 1 physics gate (§6); accept the PaliGemma licence in parallel | `decision_phase1.md`; RSS go / no-go |
| 11-10 → 11-20 | Data pipeline v1 + gates; extended action interface in π0.5; parameter and outcome heads; smoke training; B4–B6 | Pilot data passes the gates; one end-to-end training run |
| 11-21 → 11-30 | Train Ours, B1 and B2 at small scale (A100); first results table; Franka calibration | Table with CIs on held-out combos |
| 12-01 → 12-04 | Write and submit RSS Stage 1 (5 pages + references) | Submitted |
| Dec → early Feb | Scale data; all baselines and ablations; round-2 distillation; probing analysis; real-robot boundary check | Full results |
| After 2027-02-05 | RSS Stage 2 (due 04-16) or T-RL submission | — |

**Minimum RSS Stage-1 deliverable:** the Phase-1 figures (feasible set, F_min(s), recovery region), and Ours versus B1 / B2 / B5 / B7 on held-out combos in simulation.

**Journal additions:** full baselines and ablations, all three information settings, the real-robot closed loop, and the probe analysis. A semantic-safety extension (which object to act on, and whether it may be acted on) is a candidate for the T-RL version only.

---

## 12. Compute

| Workload | Where |
|---|---|
| MuJoCo sweeps, data generation, gates | RTX 5080 / CPU |
| π0.5 inference and hooks | RTX 5080 (bf16, ≈ 9.5 GiB peak, verified) |
| π0.5 LoRA training with the extra heads | A100 |
| Probes and the teacher outcome model | RTX 5080 |

---

## 13. Risks

| Risk | Mitigation |
|---|---|
| s barely moves the boundaries | Gate P2/P3 → re-scope to force-only before any training |
| s enters only through the realized acceleration, which the OSC controller saturates (≈ 3 m/s² lift at kp = 150) | Measured in the pre-sweep demonstrations; controller gain is an owner decision before the sweep; P2 is evaluated on realized acceleration |
| s is context-dependent only through the deadline | Two urgency levels (§3.7); gate P7; best-single-fixed reference R2 |
| The analytic rules are near-perfect | Gate P4; both baselines get the same coarse information and training-split calibration as the learned method; main claim restricted to robustness under uncertainty and model mismatch; extrapolation combos |
| The damage proxy reduces to F > f_tol | Carry along the closing axis; shear-augmented second proxy; P5 sensitivity; stated as a Phase-1 proxy, never as a material model; real-robot rank-transfer test (§3.5) |
| The feasible set is an artifact of the damage threshold | Two damage definitions; property grid; P5 |
| Simulated contact is unrealistic | Solver tuning (P6); real-robot ranking check |
| The outcome head does not shape the policy | Ablation B2 and removal of 𝓛_outcome / 𝓛_rank; report honestly |
| Extending π0.5's action space destabilizes training | Heads on a pooled latent; LoRA; smoke test before scaling |
| Overlap with VisualForce / FARM / V2F | Positioning table (§5); B7 as a direct comparison |
| RSS timeline | Hard gate on 11-09; journal-first fallback |

---

## 14. Repository layout

```
vla-intent-verify/               (existing repo; keep the environment and hooks)
├── plan.md                      (this file)
├── plan_v0.1.md                 (archived intent–evidence plan; not active)
├── status.md
├── CLAUDE.md                    (update to this plan)
├── docs/
│   └── data_spec.md             (rewrite for interaction groups, §7)
├── envs/feasibility_carry/      (task, force-limited gripper, time scaling, branching, damage)
├── data/                        (generation, gates, splits; not in git)
├── models/pi05_ext/             (action-interface extension, parameter head, outcome head)
├── training/                    (losses, teacher selection, round-2 distillation)
├── baselines/                   (B0–B7)
├── eval/                        (metrics, CIs)
├── analysis/                    (phase-1 figures, probes)
├── results/                     (never edited by hand)
└── notes/decision_phase1.md
```
