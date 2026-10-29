# Status

**Active plan:** `plan.md` — Counterfactual Feasibility Distillation (2026-10-29)
**Archived plan:** `plan_v0.1.md` — intent–evidence verification (2026-09-23), not active
**Last updated:** 2026-10-29
**Current phase:** Phase 1 — physical-mechanism go/no-go (2026-10-30 → 2026-11-09)
**Next decision:** RSS Stage-1 go/no-go on 2026-11-09 (`notes/decision_phase1.md`)

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
  - [x] damage functionals: accumulated excess impulse (primary), peak force (secondary) — implementation A only; implementation B (G7/G8) not written yet
- [ ] Sweep: F 8 × s 6 × R 5 over ≥ 12 property combos — infrastructure ready; **pilot only so far** (1 combo, F 4 × s 3 × R 2, `results/phase1/pilot/`) plus a 12-branch slip-boundary probe; full sweep waits on owner decisions (parameter ranges, F→N mapping, damage tolerances)
- [ ] P1 force control (🟡 pilot: measured = 0.9998·F − 0.007 N, R² = 1.000 over 8–60 N; fingers at 19.0–19.7 mm vs 0 mm limit; formal check at full sweep) · P2 s moves a boundary · P3 slow-down recovery region · P4 analytic rule not near-perfect · P5 robustness · P6 repeatability
- [ ] Figures in `results/phase1/`; verdict in `notes/decision_phase1.md`

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
| 2026-10-29 | Phase 1 started. `envs/feasibility_carry/` built and smoke-tested; pilot 4 × 3 × 2 on one combo (m 0.3 kg, μ 0.4): force tracking R² = 1.000, 0.8 s wall per branch (≈ 39 min for the full 2880-branch sweep); no slip anywhere in the pilot, slip boundary confirmed reachable on a 0.8 kg / μ 0.25 probe (all F ≤ 12 N drop, analytic need ≈ 16–17 N). Options for parameter ranges, F→N mapping and damage tolerances sent to owner; full sweep not started. Code uncommitted pending review. |