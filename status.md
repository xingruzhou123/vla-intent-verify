# Status — VLA Intent–Evidence Verification (Paper 2)

**Last updated:** 2026-09-23
**Phase:** Month 1 — Go/No-Go (not started)
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
| Environment on RTX 5080 | ⬜ not started |
| π0 + LIBERO benign reproduction | ⬜ |
| Activation hooks (layers × flow steps) | ⬜ |
| RedVLA / HazardArena code availability check | ⬜ |
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

1. Set up CUDA ≥ 12.8 + PyTorch with Blackwell support; confirm `torch.cuda` works on the 5080.
2. Install a PyTorch π0 implementation (openpi PyTorch or LeRobot) and LIBERO; headless MuJoCo rendering.
3. Identify and download a LIBERO-fine-tuned π0 checkpoint; record source and licence in `models/checkpoints.md`.
4. Run 2–3 benign LIBERO tasks; compare success with published numbers.
5. Implement activation hooks; measure per-call latency and dump size with pooling.
6. Check RedVLA and HazardArena repositories (code, assets, checkpoints, licence); pick the Month-1 environment.

---

## Open questions

- Which π0 LIBERO checkpoints are actually public, and do they match those used by RedVLA?
- Is SimpleVLA-Guard code released, or must it be re-implemented from the appendix?
- Can LIBERO scenes express "hazard stacked on target" and "target absent" without new assets?
- Where do verb labels become reliable enough to add (pour vs. place)?
- Which evidence retriever for Month 2: open-vocabulary detector, targeted VLM query, or both?
- Real-robot platform and π0 fine-tuning data for hardware experiments (Month 3).

---

## Blockers

_None yet._

---

## Weekly log

### Week 0 (2026-09-23)
- Literature survey completed; closest works: SimpleVLA-Guard (RedVLA), SAFE, HazardArena, SAFECAST.
- Plan and gate criteria written.
