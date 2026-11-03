# CLAUDE.md
Branch `semantic-obligations`. Research direction under evaluation: can a VLA turn open-ended task semantics into
explicit safety obligations, judge whether scene evidence suffices to verify them, ask a targeted question when it does
not, and generate continuous actions that satisfy them. The owner's brief is `notes/new_direction_brief.md`; `plan.md` is
the only active plan; `status.md` holds the phase, checklist, blockers and log.
Previous project (counterfactual feasibility distillation + active probing) is closed: `notes/phase1_postmortem.md`,
`notes/decision_active_probe.md`, archived plans in `docs/archive/`. Do not implement anything from them.
Background for the old direction: docs/summary.md, docs/related_work.md (written for v0.1). Paper PDFs in docs/papers/.

Rules:
- Current phase: **Phase 0 — novelty and necessity check only** (plan.md). Deliverables are notes: novelty matrix,
  claim coverage, candidate tasks, modular baselines, Gate-0 design, recommendation. No environment building, no data
  collection, no π0.5 or other fine-tuning, no four-month plan until the owner confirms the Phase-0 decision.
- Gate 0 asks for a measured empirical gain, under fair baselines, from accessing the action representation or jointly
  generating continuous actions — not for a proof that only a VLA can do the task. If the modular pipeline (VLM +
  verifier / optimizer) matches on safety, success and generalisation, say so and recommend closing; do not construct a
  problem in order to use a VLA.
- Honesty: report only measured numbers or cited facts; leave cells empty otherwise; report failures with the output.
  When a claim about prior work is not verified against the paper, mark it as unverified.
- Thresholds, pass criteria and solver settings are frozen and logged before any experiment; never tune after looking at
  held-out data; the test split is never used for tuning. Pre-register the Gate-0 pass criterion before it is run.
- When a design choice belongs to the owner (task choice, baseline strength, Gate-0 criterion, deletions, anything that
  spends A100 time), give 2–3 options with a recommendation and stop.
- Two-implementation cross-check for any label or metric that a decision depends on (gate G8 pattern from
  docs/data_spec.md): independent second implementation on the same inputs; pass thresholds set before running.
- Simulator ground truth is used only for labels, oracle arms and reference baselines; a deployed policy sees only
  images, proprioception and language.
- Closed-project code (envs/feasibility_carry, analysis/, baselines/, v0.1 LIBERO eval wrapper) stays in the
  repo as reusable assets or evidence; do not delete or extend it without asking. Cleanup follows
  `notes/cleanup_proposal.md` only after owner confirmation.
- Large artifacts (sweeps, rollouts, latent dumps, checkpoints) go in data/ or results/ and are never committed, except
  the small evidence files the owner explicitly whitelists in .gitignore. results/ is never edited by hand.
- External repos (LeRobot, LIBERO, openpi) live in third_party/; do not edit them in place — wrap or patch.
- Use the project venv `.venv/` (Python 3.12, torch 2.11 cu128, LeRobot 0.6.2).
- After finishing a task, update status.md: tick checklist items, update blockers, add a dated log line.
- GPU: RTX 5080 (16 GB, sm_120). π0.5 inference fits in bf16 (≈ 9.5 GiB). Anything that fine-tunes goes to the A100 —
  ask before starting it.
