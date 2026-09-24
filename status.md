# Status — VLA Intent–Evidence Verification (Paper 2)

**Last updated:** 2026-09-24
**Phase:** Month 1 — Go/No-Go (Week 1 in progress)
**Current gate verdict:** ⚪ pending
**Target venue:** IEEE T-RL (fallback RA-L)

---

## Snapshot

| Item | State |
|---|---|
| Literature positioning | ✅ done (see plan.md §2) |
| Core framing (policy intent vs. commanded intent vs. outcome) | ✅ decided |
| Obligation taxonomy (reference / precondition / effect) | ✅ decided |
| Label definition (future-oriented next-interaction target) | ✅ decided |
| Month-1 go/no-go criteria | ✅ drafted — freeze before held-out evaluation |
| Environment on RTX 5080 | 🟡 venv + PyTorch cu128 verified on sm_120 (2026-09-24); π0/LIBERO not yet installed |
| π0 + LIBERO benign reproduction | ⬜ |
| Activation hooks (layers × flow steps) | ⬜ |
| RedVLA / HazardArena code availability check | 🟡 RedVLA: no code/assets released as of 2026-09-24 (page has dead Code button); HazardArena not yet checked |
| Intent-divergence scenes | ⬜ |
| Episode collection (target 300–500) | ⬜ |
| Intent readers (B2, B3, B4, latent, oracle) | ⬜ |
| SimpleVLA-Guard reproduction | ⬜ |
| Month-1 decision memo | ⬜ |

---

## Decisions log

| Date | Decision | Reason |
|---|---|---|
| 2026-09-23 | Drop "first latent guard / first pre-action guard" claims | SAFE and SimpleVLA-Guard already cover them |
| 2026-09-23 | Central claim = intent licensed by scene evidence, not "latent-only fails on twins" | HazardArena twins change instruction and actor–target binding, so latent-only may separate them well |
| 2026-09-23 | Separate \(I^{\mathrm{cmd}}\), \(I^{\mathrm{policy}}\), \(Y^{\mathrm{outcome}}\) | Outcome ≠ intent (grasp failures, collisions, replanning) |
| 2026-09-23 | Intent label = next realized interaction event (future-oriented) | Current-chunk rollout labels would make Chunk-Rollout baseline the label by construction |
| 2026-09-23 | Chunk-Rollout baseline uses perceived, not privileged, poses | Fair comparison with the monitor |
| 2026-09-23 | Non-hazardous reference deviations → REOBSERVE, excluded from hazard metrics | Otherwise the paper collapses into task-failure detection |
| 2026-09-23 | Joint conformal calibration on max obligation risk | Three independent 5% heads ≈ 14.3% FPR |
| 2026-09-23 | Rename "forbidden transition" → "effect compatibility"; "target grounding" → "reference fidelity" | Evidence-type taxonomy, orthogonal to RedVLA and HazardArena |
| 2026-09-23 | Month 1 uses one environment only (prefer π0 on LIBERO / RedVLA scenes) | Avoid maintaining two stacks before the gate |
| 2026-09-23 | π0 primary backbone; OpenVLA secondary (cross-architecture) | Main competitors are π0-based; flow-step analysis needs a flow model |
| 2026-09-23 | Month 1 on RTX 5080 (inference + probes); fine-tuning on A100 | 16 GB fits π0 inference, not fine-tuning |

---

## Next actions (Week 1)

1. ~~Set up CUDA ≥ 12.8 + PyTorch with Blackwell support; confirm `torch.cuda` works on the 5080.~~ ✅ 2026-09-24 (see Week 1 log)
2. Install a PyTorch π0 implementation (openpi PyTorch or LeRobot) and LIBERO; headless MuJoCo rendering. → **decided: LeRobot** (see `notes/week1_pi0_options.md`); install pending go-ahead
3. Identify and download a LIBERO-fine-tuned π0 checkpoint; record source and licence in `models/checkpoints.md`. → candidates identified: `lerobot/pi05_libero_finetuned_v044`, `lerobot/pi0_libero_finetuned_v044`; download pending
4. Run 2–3 benign LIBERO tasks; compare success with published numbers.
5. Implement activation hooks; measure per-call latency and dump size with pooling.
6. Check RedVLA and HazardArena repositories (code, assets, checkpoints, licence); pick the Month-1 environment.

---

## Open questions

- ~~Which π0 LIBERO checkpoints are actually public, and do they match those used by RedVLA?~~ Answered 2026-09-24: PI hosts only `pi05_libero`; LeRobot hosts π0 and π0.5 LIBERO conversions/fine-tunes. RedVLA cites `huggingface.co/physical-intelligence/` for one unified checkpoint per model; exact ids not named.
- ~~Is SimpleVLA-Guard code released?~~ Not as of 2026-09-24 → re-implement from appendix E (details captured in `notes/week1_pi0_options.md`).
- Can LIBERO scenes express "hazard stacked on target" and "target absent" without new assets?
- Where do verb labels become reliable enough to add (pour vs. place)?
- Which evidence retriever for Month 2: open-vocabulary detector, targeted VLM query, or both?
- Real-robot platform and π0 fine-tuning data for hardware experiments (Month 3).

---

## Blockers

_None blocking._ Watch item: system `nvcc` is CUDA 12.6, which cannot target sm_120. Anything that builds CUDA extensions from source (flash-attn, custom kernels) will need a CUDA ≥ 12.8 toolkit installed or prebuilt wheels; the torch wheel itself bundles its own 12.8 runtime and is unaffected.

---

## Weekly log

### Week 0 (2026-09-23)
- Literature survey completed; closest works: SimpleVLA-Guard (RedVLA), SAFE, HazardArena, SAFECAST.
- Plan and gate criteria written.

### Week 1 (2026-09-24 → )
- **Item 1 done — dedicated venv + PyTorch on RTX 5080.**
  - Venv: `.venv/` in project root (uv, CPython 3.11.14; chosen because openpi requires 3.11). Not shared with physhall-safety. Added to `.gitignore`.
  - Installed from `https://download.pytorch.org/whl/cu128`: `torch==2.11.0+cu128`, `torchvision==0.26.0+cu128`, `triton==3.6.0`, cuDNN 9.19.0. Nothing else installed yet.
  - Host: driver 595.91.07, `nvidia-smi` CUDA 13.2; system `nvcc` is 12.6 (see Blockers watch item).
  - Verification (`scripts/check_gpu.py`): device `NVIDIA GeForce RTX 5080`, capability (12, 0) = sm_120, 84 SMs, 15.45 GiB; `sm_120` present in torch arch list (no PTX JIT fallback); bf16 supported.
  - bf16 matmul 2048² : max rel. error vs fp32 CPU reference 0.0021 (expected for bf16); ~0.20 ms/call ≈ 86 TFLOP/s. cuDNN bf16 conv2d OK.
  - Note: ~0.7 GB of the 16 GB is held by the desktop (Xorg/GNOME); budget ~15 GB usable for π0 inference.
- Next: item 2 (π0 PyTorch implementation + LIBERO + headless MuJoCo). Stopped here for review before installing anything further.
- **Items 2–3 research done (no install yet)** — full comparison in `notes/week1_pi0_options.md`.
  - Cloned `third_party/openpi` (215abfb) and `third_party/lerobot` (e624f3f) for inspection only.
  - openpi PyTorch port: hard-pins `torch==2.7.1` (no sm_120 wheel), hard JAX/flax/orbax deps, transformers 4.53.2 monkey-patch, no hosted fine-tuned π0 LIBERO checkpoint (only π0.5, JAX format). Rejected.
  - LeRobot: torch `>=2.7,<2.12` from the cu128 index (lock already at 2.11.0+cu128), no JAX, `[pi,libero]` extras, in-process LIBERO via `hf-libero`, π0 and π0.5 LIBERO checkpoints on HF, same `action_out_proj` pre-velocity tensor and hookable 18-layer expert. **Recommended.** Cost: needs Python ≥ 3.12 → recreate `.venv` (uv 3.12.12 available).
  - RedVLA: Table 11 → one unified checkpoint per π model from PI; chunk 50, replan 10; SimpleVLA-Guard appendix details recorded. No code/assets released yet (CC BY-NC 4.0 promised).
  - Decision: primary checkpoint `lerobot/pi05_libero_finetuned_v044` (documented 97.5% avg), secondary `lerobot/pi0_libero_finetuned_v044`. Stopped before installing.
