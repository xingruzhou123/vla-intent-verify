# Status

**Branch:** `semantic-obligations` · **Active plan:** `plan.md` (Phase 0 only) · **Brief:** `notes/new_direction_brief.md`
**Last updated:** 2026-11-03
**Current phase:** **Phase 0 — novelty and necessity check.** No implementation, no data collection, no fine-tuning.
**Next decision (owner):** after the Phase-0 deliverables — does a task with real action–semantic coupling pass the
design review for Gate 0, and is Gate 0 to be run? Until then nothing is built.

---

## 1. Previous project — closed 2026-11-03

Counterfactual feasibility distillation (Phase 1 physics gate, tag `phase1-final`) and the active-probing check that
followed (tag `active-probe-final`). Verdict and reasons: `notes/decision_phase1.md` (P2 fail, P6-frozen fail, no
override), `notes/decision_active_probe.md` (matrix row = active identification + analytic control; diagnostic pass of
2026-11-03 with bounded conclusions), `notes/phase1_postmortem.md` (what the result does and does not show).
Archived plans: `docs/archive/plan_feasibility_v1.md`, `docs/archive/plan_intent_evidence_v0.1.md`.
Not done and not planned: the 50-branch human audit (cancelled, resumable), the `dataset-v1` tag (audit not signed).

## 2. Reusable assets

| Asset | Where | State |
|---|---|---|
| Pick-and-carry MuJoCo/robosuite Panda env with force-limited grasp, time-scaled carry, CoM offset, full-state save/restore | `envs/feasibility_carry/env.py` | working, tested on 89 objects |
| Same-state counterfactual branching sweep + bisection boundary refinement | `envs/feasibility_carry/sweep.py`, `refine.py`, `run_objects.py` | working; 48 objects in 54 min at 12 processes |
| Outcome labels (slip / damage / complete), relabel without re-simulation | `envs/feasibility_carry/labels.py`, `relabel.py` | label v3 |
| Two-implementation cross-check (G7 labels, G8 feasible sets) | `envs/feasibility_carry/crosscheck_b.py`, `analysis/crosscheck_g7g8.py` | 100 % agreement on v1, v3, ext1 |
| Analytic baselines B5a (Coulomb), B5b (wrench-aware) | `baselines/analytic.py`, `docs/analytic_baselines.md` | calibrated per fold |
| Information-conditioned performance estimates, nested (grouped, inner-fold) evaluation, bootstrap CIs | `analysis/coarse_bound2.py`, `analysis/active_probe_gates2.py`, `analysis/active_probe_diag.py` | reusable as patterns |
| Probe primitives (trial lift, light squeeze) and branch/restore smoke test | moved out of this repo to `~/world-model` (owner commit cc1cb8a); leftover untracked `probing/results/`, `probing/__pycache__` on disk | not in this repo |
| Data gates G1–G8 and the gate-before-training workflow | `docs/data_spec.md`, `CLAUDE.md` | reuse the structure, rewrite the content |
| Human audit tooling (random branches, mp4 + frame strips, blank-verdict sheet, replay check) | `analysis/audit_sample.py` | 0 replay mismatches on 50 branches |
| venv (Python 3.12, torch 2.11 cu128, LeRobot 0.6.2), LIBERO with EGL, two π0.5 LIBERO checkpoints loading in bf16 (≈ 9.5 GiB, ≈ 165 ms/chunk), action-expert hooks | `.venv/`, `configs/pi05_libero.yaml`, `models/checkpoints.md`, `scripts/` | verified 2026-09-24; PaliGemma tokenizer licence still to accept for π0.5 rollouts |
| Raw sweeps (not in git): `data/sweep_v1` (48 objects, 8.1 GB), `data/sweep_v1_ext` (41 objects, 6.9 GB) | `data/` | regenerable, see `notes/cleanup_proposal.md` |

## 3. Blockers

- [ ] PaliGemma tokenizer licence (gated on Hugging Face) — only matters once π0.5 rollouts are needed; not a Phase-0 blocker.
- [x] Repo cleanup done 2026-11-03 per owner decisions (`notes/cleanup_proposal.md` §5); both raw sweeps kept.

## 4. Phase 0 checklist

- [ ] Novelty matrix over the nearest work (`notes/novelty_matrix.md`)
- [ ] Per-claim coverage: which claims are already made by existing work
- [ ] 2–3 concrete tasks with action–semantic coupling
- [ ] Strong modular baseline (VLM + verifier, VLM + motion optimizer) per task
- [ ] Gate-0 design with the five arms of the brief and a pre-registered pass criterion
- [ ] Recommendation: proceed to Gate 0, or close the direction
- [x] Branch set up: brief saved verbatim, old plans archived, plan/status/CLAUDE.md reset, cleanup proposal written (2026-11-03)

## 5. Log

| Date | Entry |
|---|---|
| 2026-11-03 | One-object regeneration check (`heavy_normal_balanced_0`, 8.1 min + relabel): all 495 npz files bit-identical to `data/sweep_v1`; `summary.json` and `probes.json` identical after dropping wall-clock fields, provided `relabel` is run after the sweep (fresh labels differ at 1e-11 because the stored ones were recomputed from float32 npz). Record in `notes/cleanup_proposal.md` §6. Sweeps regenerable exactly on this machine; both kept. |
| 2026-11-03 | Owner decisions applied. Evidence whitelist in `.gitignore`: `report.md`, `summary.json` and figures of `results/phase1/full_v3`, `demo600_report`, `pilot` only (80 files, 9.2 MB). Removed from the branch: `claude_code_prompt_feasibility.md`, `notes/proposed_phase1_spec.{md,patch}`, `notes/week1_pi0_options.md`, `notes/week2_perturbations_libero.md`. Deleted from disk: npz time series of the eight pre-sweep demo dirs (results/ 1.5 GB → 84 MB), untracked `probing/` leftovers (16 MB), project `__pycache__` dirs, pip cache (38 GB → 32 MB). `~/.cache/huggingface` untouched. Both sweeps kept; one-object regeneration check: see next line. |
| 2026-11-03 | Branch `semantic-obligations` set up for the new direction (owner's brief in `notes/new_direction_brief.md`). `plan.md` → `docs/archive/plan_feasibility_v1.md`, `plan_v0.1.md` → `docs/archive/plan_intent_evidence_v0.1.md`; new Phase-0 `plan.md`; `status.md` and `CLAUDE.md` rewritten; `notes/cleanup_proposal.md` written (data 15 GB, results 1.5 GB, third_party 174 MB, .venv 8.7 GB; nothing deleted). Tracking of `results/phase1` reports, summaries and figures waits on the owner: the full set is 46 MB, over the 20 MB limit. |
| 2026-11-03 | Repo published to GitHub (`origin`, owner-created). Feasibility line closed; history of that project is in the previous `status.md` at tag `active-probe-final` / commit 16e6442. |
