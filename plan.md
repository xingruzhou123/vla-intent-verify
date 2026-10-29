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
- This **extends π0.5's action space**. The paper states this explicitly; it is not plain fine-tuning of an unmodified model.
- **Acceptance test:** replaying stored `(a_{1:T}, F, s)` in simulation reproduces the logged contact force and carry kinematics within tolerance.

### 3.2 Outcomes

For each branch:

- `slip` — object displacement relative to the gripper exceeds δ_slip, or the object is dropped;
- `damage` — the damage functional exceeds the object's tolerance (§3.5);
- `complete` — the object rests in the goal region and is not damaged.

Each (F, s) cell is repeated *R* times with small perturbations (initial pose, friction noise). The outcome probability `P̂_k(F,s)` for k ∈ {complete, slip, damage} is the empirical rate over those repeats.

### 3.3 Feasible set and margin

```
𝓕(x) = { (F,s) : P_complete ≥ τ_c,  P_slip ≤ τ_s,  P_damage ≤ τ_d }
```

- Thresholds `τ_c, τ_s, τ_d` are fixed before the main experiments (initial values: 0.9 / 0.05 / 0.05).
- **Margin:** `m(F,s; x)` = distance, in normalized (F̃, s̃) space, from (F,s) to the nearest grid cell outside 𝓕(x). It is computed from the empirical sweep and is negative outside 𝓕.
- **Feasibility score:** `q(F,s) = P_complete − α·P_slip − β·P_damage`, with initial values α = β = 1. This score is used only by the ranking loss and for diagnostics.

### 3.4 Teacher selection

```
(F*, s*) = argmax_{(F,s) ∈ 𝓕(x)}  m(F,s; x) − γ·‖(F̃,s̃) − (F̃₀,s̃₀)‖²
```

- (F₀, s₀) are the nominal parameters of the demonstration.
- If 𝓕(x) is empty, the teacher returns no target, and the group is flagged as infeasible-by-design.
- **Round 1** uses the empirical sweep.
- **Round 2** (student states, which have no sweep) uses a learned teacher outcome model `T(o, c, F, s)`. Its inputs are the same deploy-time inputs as the VLA's. Its targets are kept only where its predicted margin exceeds a confidence threshold.

### 3.5 Damage model

- **Primary:** accumulated excess impulse `D = ∫ max(0, f_n(t) − f_tol) dt`; damage occurs if `D > D_max`.
- **Secondary (robustness check):** peak normal force > f_tol.
- **Requirement:** the conclusions must hold under both definitions. Feasible-set rankings must agree across the two definitions (Kendall τ ≥ 0.8). The feasible region must not be a mechanical consequence of one hand-set threshold.

### 3.6 Information settings (never pooled)

| Setting | VLA inputs besides images, proprioception and instruction | Role |
|---|---|---|
| **Oracle** | exact m, μ, fragility | upper bound |
| **Coarse (main claim)** | binned attributes in text (light/heavy, slippery/normal, fragile/rigid), with label noise ε | main results |
| **Vision-only** | none | generalization test; meaningful only where appearance correlates with properties |

**Main claim:** under observable but imprecise attribute information, a learned, margin-aware feasible set is more robust than analytic rules and single-point imitation. We do **not** claim to infer unobservable physical properties from vision.

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

**Environment:** a new pick-and-carry task on the robosuite/MuJoCo Panda used by LIBERO, with a force-limited gripper, a parametric object (m, μ, fragility), a fixed carry path with time scaling s, and save/restore of the full MuJoCo state at the pre-contact branch point.

**Sweep:** F 8 levels × s 6 levels × R = 5 repeats, over ≥ 12 property combinations (3 m × 2 μ × 2 fragility).

| ID | Check | Pass criterion |
|---|---|---|
| **P1** | Force control is real | Steady-state measured contact normal force tracks commanded `forcerange` monotonically (R² ≥ 0.9). Fingers are not at the position limit, and the actuator is not saturated for reasons other than the force bound. |
| **P2** | s moves a boundary | For ≥ 2 property combos, the slip or damage boundary shifts by ≥ 15% of the normalized F range between the fastest and slowest s, with non-overlapping 95% CIs |
| **P3** | Slow-down recovery region exists | For ≥ 2 property combos, no feasible F exists at nominal s (every F either slips or damages), but a feasible F exists at a slower s |
| **P4** | Analytic rule is not near-perfect under realistic information | With coarse or noisy attributes and non-ideal contact, the analytic baseline (m(g+a)/(2μ) plus a damage upper bound) selects infeasible parameters on ≥ 10% of held-out combos, or misplaces the boundary by more than the median feasible-set width / 4 |
| **P5** | Structure is robust, not a threshold artifact | P2–P3 hold across the property grid and under both damage definitions (Kendall τ ≥ 0.8) |
| **P6** | Repeatability | Outcome agreement across repeats ≥ 90% per cell (tune solver timestep, `noslip_iterations` and `impratio` if needed; log the final values) |

**Decision rule:**

- **P1, P2 or P3 fails →** stop. Re-scope to a force-only study, and do not train π0.5 on F × s. The RSS submission is cancelled; re-plan for T-RL.
- **P4 fails →** continue, but narrow the claim to robustness under attribute uncertainty; the analytic baseline becomes a headline comparison.
- **P5 or P6 fails →** fix the simulation before anything else.
- **RSS decision is made on 2026-11-09:** proceed with the Stage-1 track only if P1–P3 and P6 pass.

**Outputs:** `results/phase1/` (feasible-set heatmaps per combo, F_min(s) curves with CIs, analytic-versus-simulation overlays) and `notes/decision_phase1.md`.

---

## 7. Data

**Interaction group** = one pre-contact state together with all of its (F, s) branches. Each group stores:

- observations at the decision chunk: two camera views, proprioception, instruction, the coarse-attribute text, and true attributes (labels only);
- per branch: commanded (F, s), measured contact force and carry kinematics, outcome labels, and damage under both definitions;
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
| B5 | Analytic rule with coarse attributes (fair) | Does learning beat deployable physics? |
| B6 | Analytic rule with oracle attributes (reference only) | Upper reference |
| B7 | Demo-anchored selection (VisualForce-style): sample K (F,s) candidates from the policy, choose the one closest to the demonstration target force | Margin-aware feasible set versus a demo-anchored target, especially on extrapolation combos |
| R1 | Ours + external selector at deployment (reference only) | How much distillation loses |

### Ablations

Remove `𝓛_outcome`; remove `𝓛_rank`; remove the margin term (select any feasible point); vary γ; run each information setting (oracle / coarse / vision-only).

### Metrics

- **Primary:**
  - safe completion rate on held-out and extrapolation combos;
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
| The analytic rule is near-perfect | Gate P4; main claim restricted to coarse attributes; extrapolation combos |
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
