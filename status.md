# Status

**Active plan:** `plan.md` — Counterfactual Feasibility Distillation (2026-10-29)
**Archived plan:** `plan_v0.1.md` — intent–evidence verification (2026-09-23), not active
**Last updated:** 2026-11-02
**Current phase:** Phase 1 — physical-mechanism go/no-go (2026-10-30 → 2026-11-09)
**Next decision:** owner re-planning after the Phase-1 verdict of 2026-11-02 (P2 and P6 fail; `notes/decision_phase1.md`)

---

## 1. Carried over from v0.1 (still valid)

| Item | State | Use in the new plan |
|---|---|---|
| Dedicated venv: Python 3.12, torch 2.11 cu128, LeRobot 0.6.2 | ✅ verified on RTX 5080 (sm_120) | Unchanged |
| LIBERO with EGL headless rendering | ✅ working | Base for `envs/feasibility_carry/` (robosuite/MuJoCo Panda) |
| Two π0.5 LIBERO checkpoints load in bf16 | ✅ ≈ 9.5 GiB peak, ≈ 165 ms per inference | Base model to extend (§3.1, §4) |
| Hooks on action-expert layers and `pre_velocity` | ✅ fire correctly | Pooled latent `z` for the parameter and outcome heads; probe analysis (§9) |
| Data validation gates (`docs/data_spec.md`) | ⬜ not present in repo; written 2026-10-29 | Written fresh from plan §7 for interaction groups |
| Two-implementation cross-check workflow (`CLAUDE.md`) | ⬜ not present in repo; written 2026-10-29 | Applied to outcome labels (slip / damage / complete) and feasible-set computation |
| `.gitignore` for `data/`, `results/`, `third_party/`, large binaries | ✅ | Unchanged |

## 2. Blockers

- [ ] **PaliGemma tokenizer licence (gated on Hugging Face).** Accept it manually in the browser, then re-run the cold smoke test. Blocks π0.5 rollouts, not the Phase-1 physics sweeps.
- [x] ~~Full sweep awaits owner confirmation~~ approved and run 2026-11-02 (`data/sweep_v1`, 48 objects, 54 min wall).
- [ ] **Phase-1 verdict: P2 and P6 fail** (2026-11-02). Under plan §6 the project stops: no F × s training, RSS Stage-1 cancelled, re-plan for T-RL; P6 says fix the simulation first (23 of 27 failing cells are drop cells where `completed` / crush-between-fingers disagree across repeats). Owner decides the re-scoping.

## 3. Retired with v0.1 (do not continue)

| Item | Reason |
|---|---|
| Month-1 latent-intent go/no-go (LIBERO intent-divergence scenes, intent probes vs. chunk baselines) | Question no longer in the paper; latent work is now analysis only |
| Future-oriented intent labels, interaction-event instrumentation for intent | Not needed; labels are now branch outcomes (slip / damage / complete) |
| SimpleVLA-Guard reproduction (RedVLA appendix parameters) | Runtime monitors are no longer a baseline family |
| HazardArena SOL-L1/L2, SAFE/SAFECAST baselines | Same as above |
| Obligation router, evidence retriever, joint conformal calibration, REOBSERVE | Belongs to the archived plan; candidate for a later semantic-safety extension only |
| Real-robot dispatch gating | Replaced by the real-robot boundary check and closed-loop evaluation (§10) |
| Pending item “reproduce libero_spatial success and pick the Month-1 checkpoint” | Superseded; checkpoint choice is now tied to the new carry task |
| Pending item “check four perturbations with LIBERO assets” | Retired with the intent scenes |

Code written for these items stays in the repo but is not extended. Do not delete it without asking.

## 4. Phase 1 checklist (due 2026-11-09)

- [x] Commit the uncommitted Week-1 files from v0.1 (environment, hooks) before starting new work — done 2026-10-26 (7b96be5); v0.1 plan and notes archived 2026-10-29
- [x] Rewrite `CLAUDE.md` and write `docs/data_spec.md` for the new plan — done 2026-10-29 (no earlier copies found; data spec written fresh from plan §7)
- [x] `envs/feasibility_carry/`: pick-and-carry task on the robosuite/MuJoCo Panda — built 2026-10-29 (`env.py`, `labels.py`, `sweep.py`; pilot scale)
  - [x] force-limited gripper (`forcerange` = F; finger target ramped past the joint range so the actuator saturates at F)
  - [x] parametric object (m, μ via geom priority 1; fragility = (f_tol, D_max) tolerances, values pending owner decision)
  - [x] fixed carry path with time scaling s (min-jerk lift / lateral / lower, OSC_POSE absolute; realized acceleration saturates below s ≈ 0.4)
  - [x] save/restore of the full MuJoCo state at the pre-contact branch point (`mj_getState` / `mj_setState`, INTEGRATION spec)
  - [x] labels (implementation A, `labels.py`, 2026-10-29): slip (pad-relative translation, torsion, drop), sustained peak-crush proxy (primary), shear-augmented proxy, impulse (appendix), completed, physical_safe, deadline_success per urgency level, P5 sensitivity grid, realized acceleration; implementation B (G7/G8) not written yet
  - [x] CoM offset via `body_ipos` (+ `body_sameframe`/`body_simple` reset), canonical vertical grasp, object contact solref 0.005 s, 2× faster nominal path (T_carry(s₀) = 1.75 s)
  - [x] bisection boundary refinement (`refine.py`), feasible set / margin / teacher / P7 / best-single-fixed (`feasible.py`, impl. A), analytic baselines B5a/B5b (`baselines/analytic.py`), demo report (`analysis/phase1_demo.py`)
- [x] Pre-sweep demonstrations (a), (b) run 2026-10-29 on 6 physical combos at kp = 150 (`results/phase1/demo_report/`), then (a), (c) rerun at the frozen kp = 600 (`results/phase1/demo600_report/report.md`)
- [x] Owner decisions frozen 2026-10-29 (`notes/decision_phase1.md`): kp = 600; CoM offset nuisance variable at [0.20, 0.30] w, no torsional-slip claim; B5b r_eff fitted on the training split; urgent deadline 1.0 × nominal + 0.5 s; P2 absolute; P6 excludes boundary-uncertainty cells; P3 reference speed = fastest grid s (definitional change, dated); properties sampled continuously inside coarse bands (`objects.py`, 48 objects, `run_objects.py` driver)
- [x] Full sweep run 2026-11-02 after owner approval (4 objects per cell): `data/sweep_v1` (48 objects, 11 520 grid + 12 015 probe branches, 54 min wall at 12 processes, 1.45 s per branch under contention); analysis `analysis/phase1_full.py` → `results/phase1/full_v1/report.md`, `summary.json`, 48 boundary figures, heatmaps for the recovery and test objects, `diagnostics_shift_window.png`, `p4_boundary_errors.png`; baseline selections executed in the simulator (`<object>/p4_exec.json`)
- [x] P1–P7 on the full sweep (2026-11-02): P1 🟢 (R² ≥ 0.9999 on 48/48) · P2 🔴 (0/48 objects reach 9.75 N; max slip shift 7.5 N, max damage shift 4.4 N) · P3 🟢 (10 objects with a recovery, 17 contexts, windows 1.6–19.0 N, all contain a grid level) · P4 🟢 (B5a-c: infeasible selections 9/36 = 0.25, median boundary error 4.2 N > W/4 = 3.9 N) · P5 🟢 (monotone; verdicts stable; τ 0.946) · P6 🔴 (27 entries < 0.9 outside boundaries, 23 in drop cells) · P7 🟢 (0.588 non-modal of 136) · best single fixed (32.9 N, 1.02) 0.378 vs oracle 0.633 (train), 0.375 vs 0.729 (held-out)
- [x] Figures in `results/phase1/full_v1/`; verdict written in `notes/decision_phase1.md` (2026-11-02): rule applied mechanically → stop
- [ ] Implementation B of labels and feasible sets for G7/G8 (sweep exists now; not run — verdict rests on implementation A)
- [x] Flex-body research note `notes/flex_feasibility.md` (2026-11-02; no implementation)

## 5. After Phase 1 (only if P1–P3 and P6 pass)

- [ ] Data pipeline v1 + gates (30-group pilot)
- [ ] Extended action interface in π0.5; parameter head and outcome head
- [ ] Smoke training on A100; baselines B4–B6
- [ ] Ours, B1, B2 at small scale; first results table
- [ ] Franka Hand grip-force calibration (load cell, 30–70 N)

## 6. Log

| Date | Entry |
|---|---|
| 2026-09-23 | v0.1 plan (intent–evidence verification) written |
| Late Sep → Oct | Week-1 environment set up: venv, LIBERO, π0.5 checkpoints, hooks verified; tokenizer licence blocking |
| 2026-10-29 | Switched to `plan.md` (Counterfactual Feasibility Distillation); v0.1 archived as `plan_v0.1.md`; status refreshed |
| 2026-10-29 | Housekeeping done: v0.1 materials archived (e8145f8), plan switch committed (bcaa0f4), `CLAUDE.md` rewritten and `docs/data_spec.md` written for interaction groups. Tokenizer still gated (owner accepting licence). Next: Phase 1 `envs/feasibility_carry/` |
| 2026-11-02 | Owner approved 4 objects per cell. Full sweep run (`data/sweep_v1`, 48 objects, 23 535 branches, 54 min wall). `analysis/phase1_full.py` written (P1–P7, formal P4 with training-split calibration, ε = 0.1 noisy bins and simulator execution of every baseline selection, diagnostics). Results: P1, P3, P4, P5, P7 pass; **P2 fails** (0/48 objects ≥ 9.75 N; max 7.5 N) and **P6 fails** (27 entries, 23 in drop cells). Decision rule applied in `notes/decision_phase1.md`: stop, no F × s training, RSS cancelled, re-plan for T-RL; owner re-scopes. `feasible.refined_intervals` got an optional `n_boot` argument (default unchanged). Flex note `notes/flex_feasibility.md`: MuJoCo 3.8.1 full flexes stable only for E ≲ 10⁵ Pa (foam), 6–10× slower per branch, grasp outcome and strain flip with timestep/solver; trilinear flexes do not grasp; ≈ 1.5–2 weeks to a pilot after a MuJoCo upgrade. Nothing committed yet. |
| 2026-11-02 | Report on the kp = 600 reruns delivered to the owner (results/phase1/demo600_report/report.md): P2 🔴 as written (7.6 N < 9.75 N on (a)); P3 🟡 on 1 of 2 required combos ((a) rigid infeasible at s = 0.5, recovers at s = 1.02, window 10.5 N at s = 2.1 with grid level 48 inside; (c) feasible at the reference speed); P7 50 % non-modal on 4 feasible contexts; B5a oracle |F_min error| 0.3–4.2 N; best-single-fixed = oracle = 0.50 (degenerate on 2 combos). Sampling proposal: 48 objects, ≈ 30 min wall. Stopped before the full sweep; owner confirmation pending. No code changes. |
| 2026-10-29 | Owner decisions frozen (notes/decision_phase1.md): kp = 600, CoM nuisance at 0.25 w with no torsion claim, r_eff fitted, urgent 1.0×, P2 absolute, P6 boundary exclusion, P3 reference = fastest s, continuous property sampling (bands in plan §3.6; 48 objects proposed, ≈ 30 min wall). Demos (a), (c) rerun at kp = 600 (results/phase1/demo600_report): realized lateral acceleration 7.2 m/s² at s = 0.5; (a) rigid shows the P3 recovery from the fast end (window 0 → 10.5 N, s_rec ≈ 1.0); P2 shift still 7.6 N < 9.75 N. Full sweep not started. |
| 2026-10-29 | Spec approved with amendments (contextual deadline with two urgency levels, P7, best-single-fixed reference R2, κ ∈ {0.5,1,2}, carry-direction design note) and applied to plan.md, docs/data_spec.md, docs/analytic_baselines.md (B5a lateral-phase formula corrected to the closing-axis geometry). Code updated and pre-sweep demonstrations run on 6 combos (results/phase1/demo_report). Simulation fixes found on the way: `body_ipos` needs `body_sameframe = body_simple = 0` or the CoM offset is silently ignored; Panda home orientation tilts the pads 7° (fixed: canonical grasp); robosuite's object solref 0.02 s lets the held object creep ≈ 4 mm across the pads (fixed: 0.005 s); slip must be measured against the pads, not the hand (finger-joint drift). Findings for the owner: realized acceleration saturates at kp = 150 (lift ≈ 3 m/s², lateral ≈ 4.7 m/s² at s = 0.5; lift end-point 9 mm short), compressing every s-dependence; kp = 300–600 reach 6–7 m/s² lateral and make the fastest cell of (1.5 kg, μ 0.2, rigid) infeasible with recovery at s = 1. CoM offset shifts the slip boundary by only 0.2–1.2 N (0.15 w), 0.5–2.4 N (0.20 w), 1.3–3.4 N (0.25 w); torsion ≤ 1.4° before translational slip. Full sweep not started: owner decisions needed on controller gain, CoM offset level, urgent multiple (1.5× vs 1.0×), P2/P3 thresholds, P6 strictness, torsion lever arm. |
| 2026-10-29 | Phase 1 started. `envs/feasibility_carry/` built and smoke-tested; pilot 4 × 3 × 2 on one combo (m 0.3 kg, μ 0.4): force tracking R² = 1.000, 0.8 s wall per branch (≈ 39 min for the full 2880-branch sweep); no slip anywhere in the pilot, slip boundary confirmed reachable on a 0.8 kg / μ 0.25 probe (all F ≤ 12 N drop, analytic need ≈ 16–17 N). Options for parameter ranges, F→N mapping and damage tolerances sent to owner; full sweep not started. Code uncommitted pending review. |