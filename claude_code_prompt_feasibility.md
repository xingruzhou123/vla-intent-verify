# Task: switch the repo to the new plan and run the Phase-1 physics gate

## 0. Read first

1. `plan.md` — the **only active plan** (Counterfactual Feasibility Distillation). It is the source of truth for the method, loss, action interface, feasible-set definition and go/no-go criteria. If anything in this prompt conflicts with `plan.md`, `plan.md` wins; tell me about the conflict.
2. `status.md` — what carries over, what is retired, and the Phase-1 checklist.
3. `plan_v0.1.md` is archived. Do not implement anything from it.

## 1. Housekeeping (do first, then stop and report)

- Commit any uncommitted Week-1 work (environment, hooks) with a clear message. Do not delete code from v0.1.
- Rewrite `AGENTS.md` for the new plan. Keep the two-implementation cross-check idea, now applied to outcome labels (slip / damage / complete) and feasible-set computation.
- Rewrite `docs/data_spec.md` for interaction groups (plan §7), keeping the gate structure.
- Check whether the PaliGemma tokenizer loads. If not, tell me which Hugging Face repo needs licence acceptance; I will click it. Phase-1 sweeps do not depend on it, so continue with them.

## 2. Phase 1 — physical-mechanism go/no-go (plan §6, due 2026-11-09)

Build `envs/feasibility_carry/` on the robosuite/MuJoCo Panda used by LIBERO:

- force-limited gripper, with F set through the actuator's `forcerange`; F is fixed once the grasp closes;
- parametric object (m, μ, fragility);
- fixed carry path with time scaling s;
- save/restore of the full MuJoCo state at the pre-contact branch point;
- outcomes and both damage definitions exactly as in plan §3.2 and §3.5.

Run the sweep in plan §6 and evaluate **P1–P6 with the exact pass criteria in the plan**. Required outputs:

- `results/phase1/`:
  - feasible-set heatmaps per property combo;
  - F_min(s) and damage-boundary curves with 95% CIs;
  - P3 recovery-region examples;
  - analytic-rule overlays (coarse vs. oracle attributes);
  - Kendall τ between the two damage definitions;
  - repeatability table.
- `notes/decision_phase1.md`: pass/fail per check with numbers, the final solver parameters, and the decision per the plan's rule.

**Rules:**

- If P1, P2 or P3 fails, stop and report. Do not start π0.5 training or the data pipeline.
- Simulator ground truth is used only for labels, oracle settings and reference baselines.
- Never tune thresholds after looking at held-out combos.
- Do not report numbers you did not measure. Leave cells empty instead.

## 3. Do not start yet

Do not start data pipeline v1, the π0.5 action-interface extension, the heads or training (plan §4, §7, §8) until I confirm the Phase-1 decision.

## 4. Reporting

After housekeeping, and after Phase 1, give me:

- a one-sentence conclusion;
- key numbers;
- figure paths;
- an update to `status.md` (tick items, add a log line).

When a design choice is genuinely mine (e.g., object parameter ranges, how to map F to the 30–70 N scale, damage tolerances), list 2–3 options with your recommendation and stop.
