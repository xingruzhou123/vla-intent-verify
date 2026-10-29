# CLAUDE.md
Research project: Counterfactual Feasibility Distillation — teaching a VLA (π0.5) the context-dependent range of
gripper force F and carry time-scaling s that completes a pick-and-carry without slip or damage.
`plan.md` is the only active plan; `plan_v0.1.md` (intent–evidence verification) is archived — do not implement
anything from it. `status.md` holds the current phase, checklist, blockers and log.
Background: docs/summary.md, docs/related_work.md (written for v0.1; positioning for v1 is plan.md §5). Paper PDFs in docs/papers/.

Rules:
- Current phase: Phase 1 physics go/no-go (plan.md §6, due 2026-11-09). Build only `envs/feasibility_carry/`, the F × s sweep,
  checks P1–P6, `results/phase1/` figures and `notes/decision_phase1.md`. No π0.5 training, data pipeline v1, action-interface
  extension or heads (plan.md §4, §7, §8) until the Phase-1 decision is confirmed by the owner.
- If P1, P2 or P3 fails, stop and report; do not re-scope on your own.
- Simulator ground truth (true m, μ, fragility; contact forces; object pose) is used only for labels, oracle settings and
  reference baselines. The deployed policy sees only images, proprioception, instruction and the coarse-attribute text.
- Thresholds (τ_c, τ_s, τ_d, damage tolerances, solver settings) are frozen before the main experiments and logged.
  Never tune anything after looking at held-out combos; the test split is never used for tuning.
- Report only measured numbers; leave result cells empty otherwise. Report failures with the output.
- When a design choice belongs to the owner (parameter ranges, F-to-Newton mapping, damage tolerances), give 2–3 options
  with a recommendation and stop.
- Two-implementation cross-check (code correctness, data gate G8 in docs/data_spec.md): outcome labels and feasible-set
  computation each get an independent second implementation, run on the same inputs under the same damage definition.
  Pass: label agreement ≥ 95%, feasible sets identical on ≥ 98% of grid cells, margins differ by ≤ 1 grid step.
- Damage-definition robustness (physics, Phase-1 check P5, not a data gate): Kendall τ ≥ 0.8 between feasible-set rankings
  under the primary (excess impulse) and secondary (peak force) damage definitions.
- v0.1 code (LIBERO eval wrapper, intent-scene notes) stays in the repo but is not extended; do not delete it without asking.
- Large artifacts (sweeps, rollouts, latent dumps, checkpoints) go in data/ or results/, never committed. results/ is never edited by hand.
- External repos (LeRobot, LIBERO, openpi) live in third_party/; do not edit them in place — wrap or patch.
- Use the project venv `.venv/` (Python 3.12, torch 2.11 cu128, LeRobot 0.6.2), not physhall-safety's.
- After finishing a task, update status.md: tick checklist items, update blockers, add a dated log line.
- GPU: RTX 5080 (16 GB, Blackwell sm_120). MuJoCo sweeps and probes run here; π0.5 inference fits in bf16 (≈ 9.5 GiB);
  LoRA training with the extra heads goes to the A100 — ask before starting anything that needs fine-tuning.
