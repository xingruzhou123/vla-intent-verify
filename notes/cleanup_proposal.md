# Cleanup proposal — branch `semantic-obligations` (2026-11-03)

Nothing has been deleted. Every item below waits for the owner's confirmation. Sizes measured with `du -sh` on
2026-11-03.

## 1. Disk usage

| Path | Size | Content |
|---|---|---|
| `data/` | 15 GB | `sweep_v1` 8.1 GB (48 objects), `sweep_v1_ext` 6.9 GB (41 objects); objects JSONs 60 KB |
| `results/` | 1.5 GB | eight pre-sweep demo dirs `results/phase1/demo_*` 1.4 GB (4 012 `.npz` time series); `audit` 19 MB (50 mp4 8.1 MB + frame strips); `pilot` 12 MB; `full_v1/v2/v3` 8.2 MB each (figures); everything else < 4 MB per dir |
| `third_party/` | 174 MB | LeRobot, LIBERO, openpi checkouts |
| `.venv/` | 8.7 GB | project venv (torch 2.11 cu128 is most of it) |
| `~/.cache/huggingface` | 132 GB | **shared with other projects**; the π0.5 entries (`models--lerobot--pi05_libero_base`, `..._finetuned_v044`) are 14.5 GB fp32 + 7.5 GB bf16 per `models/checkpoints.md`; the other 100+ GB (Wan2.1, open_llama, Qwen, Llama) are not this project's — not proposed here |
| `~/.cache/pip` | 38 GB | pip wheel cache, shared; safe to purge any time (`pip cache purge`), only costs re-download |
| `~/.cache/torch` | 2.8 GB | shared |
| `__pycache__/` in analysis/, baselines/, envs/ | < 5 MB | regenerated on import; already gitignored |
| `probing/` leftovers on disk (`results/`, `__pycache__`) | 16 MB | untracked remains of the probing code the owner moved to `~/world-model` (commit cc1cb8a); the smoke results are in that repo's history — safe to delete from disk |
| checkpoints inside the repo | none | `find` for *.pt / *.pth / *.ckpt / *.safetensors returned nothing |

Tracked in git on this branch: 1.1 MB. The candidate evidence whitelist (`results/**/report.md`, `summary.json`,
`results/phase1/**/*.png|pdf`) would add 46 MB: 9.6 MB reports + summaries (7.5 MB of that is the eight 0.94 MB
`demo_*/summary.json`), 36.2 MB figures (`audit` 10.3 MB, `full_v1`/`v2`/`v3` 7.8 MB each with `v2` and `v3` byte-for-byte
the same size, `demo_report` 1.9 MB, `demo600_report` 0.6 MB, `pilot` 0.1 MB).

## 2. Files on this branch: specific to the closed ideas vs reusable

**Reusable (keep):**

| File(s) | Why |
|---|---|
| `envs/feasibility_carry/env.py`, `sweep.py`, `refine.py`, `run_objects.py`, `objects.py` | env, same-state branching, bisection refinement, object sampler |
| `envs/feasibility_carry/labels.py`, `relabel.py`, `feasible.py`, `crosscheck_b.py`; `analysis/crosscheck_g7g8.py` | labels, feasible sets, implementation B, G7/G8 cross-check |
| `baselines/analytic.py`, `docs/analytic_baselines.md` | analytic baselines (any Gate-0 "physics verifier" arm starts here) |
| `analysis/coarse_bound2.py`, `active_probe_gates2.py`, `active_probe_diag.py` | nested grouped evaluation, bootstrap CIs, information-conditioned estimates — patterns for Gate 0 |
| `analysis/audit_sample.py` | human-audit tooling (random branches, mp4, frame strips, blank verdict sheet, replay check) |
| `docs/data_spec.md` | gate workflow G1–G8 (structure reusable, thresholds to be rewritten) |
| `configs/pi05_libero.yaml`, `configs/constraints-torch.txt`, `models/checkpoints.md`, `scripts/check_gpu.py`, `scripts/eval_libero.py` | π0.5 loading, pinned revisions, venv constraints, LIBERO eval |
| `notes/decision_phase1.md`, `decision_active_probe.md`, `phase1_postmortem.md`, `docs/design_history.md` | the record of the closed project and its reasons |

**Specific to the closed ideas (evidence for reported numbers — keep unless the owner wants a lean branch):**

| File(s) | Produced |
|---|---|
| `analysis/phase1_pilot.py`, `phase1_demo.py`, `phase1_full.py` | pilot, pre-sweep demos, P1–P7 verdict (`results/phase1/full_v*`) |
| `analysis/coarse_bound.py` (v1), `active_probe_gates.py` (v1) | superseded by the v2 scripts; still the source of the v1 numbers in the decision notes |
| `notes/flex_feasibility.md` | flex-body research note for the feasibility line (no implementation) |

**Specific to the closed ideas and not evidence (candidates to remove from the branch):**

| File(s) | Size | Why removable |
|---|---|---|
| `claude_code_prompt_feasibility.md` | 11 KB | the task prompt that started the feasibility phase; fully superseded by `docs/archive/plan_feasibility_v1.md` and the decision notes |
| `notes/proposed_phase1_spec.md`, `notes/proposed_phase1_spec.patch` | 37 KB + 15 KB | draft spec and patch for Phase 1, folded into the archived plan and `docs/data_spec.md` |
| `notes/week1_pi0_options.md`, `notes/week2_perturbations_libero.md` | small | v0.1 intent-scene notes (LIBERO perturbations, π0 options); the π0.5 facts they hold are in `models/checkpoints.md` |

Everything stays in git history and at the tags `phase1-final` / `active-probe-final` whatever is removed here.

## 3. Can the raw sweeps be regenerated exactly from code + seeds?

**Inputs are fully seeded.** Object sets: `data/objects_v1.json` (seed 0, 4 per cell, 48 objects) and
`data/objects_ext1.json` (seed 1, k_offset 4, 41 objects). Sweep perturbations and the env RNG use `--seed 0` (default)
per object; the branch state is saved and restored with `mj_getState`/`mj_setState`. Versions: MuJoCo 3.8.1, robosuite
1.4.0, numpy 2.2.6, Python 3.12 in `.venv/`.

```
MUJOCO_GL=egl .venv/bin/python -m envs.feasibility_carry.objects --out data/objects_v1.json --per_cell 4 --seed 0
MUJOCO_GL=egl .venv/bin/python -m envs.feasibility_carry.run_objects data/objects_v1.json  --out_root data/sweep_v1     --parallel 12   # 54 min wall
MUJOCO_GL=egl .venv/bin/python -m envs.feasibility_carry.run_objects data/objects_ext1.json --out_root data/sweep_v1_ext --parallel 12   # 43 min wall
MUJOCO_GL=egl .venv/bin/python -m envs.feasibility_carry.relabel data/sweep_v1      # label v3 (also v1/v2 files)
MUJOCO_GL=egl .venv/bin/python -m envs.feasibility_carry.relabel data/sweep_v1_ext
```
(`objects_ext1.json` was produced with `--seed 1 --k_offset 4 --n_objects 41`; `objects_v1_ext.json` is the union of
the two files. The `p4_exec.json` in each sweep dir is written by `analysis/phase1_full.py`, not by the sweep.)

**Exactness: expected but not verified end-to-end.** MuJoCo is deterministic for the same model, state, binary and
thread count, and the audit replay reproduced all 50 stored labels from the saved branch states (0 mismatches), which
supports determinism on this machine. Cross-version or cross-machine bit-identity is not guaranteed by MuJoCo. Before
deleting a sweep, the check is: re-run one object (`--only heavy_normal_balanced_0`, ≈ 10–18 min) into a scratch dir and
`diff` its `summary.json` against the stored one. I have not run this check.

## 4. Proposal

**Delete from the branch (git rm; ≈ 70 KB):** the four "not evidence" files in §2 — `claude_code_prompt_feasibility.md`,
`notes/proposed_phase1_spec.md`, `notes/proposed_phase1_spec.patch`, `notes/week1_pi0_options.md`,
`notes/week2_perturbations_libero.md`. Keep every script that produced a reported number.

**Delete from disk (keeping all `report.md`, `summary.json`, `*.json`, figures and mp4):**

| Target | Size freed | Reason | Risk |
|---|---|---|---|
| `results/phase1/demo_*/branches/*.npz` and `probes/*.npz` (eight dirs) | ≈ 1.4 GB | pre-sweep demonstrations at kp 150 / 600, superseded by the full sweep; summaries and reports stay | low — not referenced by any gate; demo figures already rendered |
| `data/sweep_v1_ext/` | 6.9 GB | outside dataset-v1; its numbers are in `results/phase1/active_probe_v2*`, `active_probe_diag`, `crosscheck_ext1`; regenerable in 43 min | medium — any re-analysis of the 89-object set needs it back |
| `data/sweep_v1/` | 8.1 GB | dataset-v1 is unsigned and untagged; regenerable in 54 min | **not recommended now** — the audit is "cancelled, resumable", and resuming it or writing up the negative result needs the stored branches |
| `~/.cache/pip` | 38 GB | wheel cache only | none beyond re-download |

Recommended: do the first row and the pip cache now (≈ 39 GB freed, no loss); keep both sweeps until the owner decides
whether the negative result will be written up; if it will not, delete `sweep_v1_ext` first and `sweep_v1` only after
the one-object regeneration check above passes. `third_party/` (174 MB) and `.venv/` (8.7 GB) are needed for anything
that loads π0.5 and stay.
