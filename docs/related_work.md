# Related Work Notes

Status as of 2026-09-23. Most entries are arXiv preprints from 2026; re-search before submission.

## Closest competitors (must reproduce / compare)

### RedVLA + SimpleVLA-Guard — arXiv 2604.22591 (PKU, Apr 2026)
- Red teaming: injects risk objects into LIBERO scenes; elicits unsafe behavior on OpenVLA, OpenVLA-OFT, VLA-Adapter(-Pro), π0, π0.5.
- Taxonomy: state / cumulative / conditional safety cost × resource damage / dangerous item misuse / robot damage / environmental harm.
- **SimpleVLA-Guard:** π0 `pre_velocity` feature, diffusion idx 0, horizon idx 0 (1024-d) → LSTM (hidden 256, dropout 0.2) → step-wise risk; ramp-weighted BCE; FCP threshold (α = 0.05, 100 time bins, smoothing window 3).
- Data: LIBERO-Long, 40 safe + 40 unsafe rollouts per task (800 total); safe = 200 benign + 200 with risk factor but no violation; tasks 4 and 8 held out.
- Results: PRC-AUC 0.94 seen / 0.89 unseen; online ASR 90.9 → 31.4 (seen), 98.3 → 46.7 (unseen); benign SR drop 4.08 / 10.00.
- Real robot: Franka + π0, red teaming only (guard not deployed on hardware).
- Weaknesses to exploit: single scalar; only flow step 0; unsafe rollouts collapse into one latent region regardless of task (possible off-nominal shortcut); no evidence grounding.

### SAFE — arXiv 2506.09937 (NeurIPS 2025)
- Final-layer VLA hidden states (before token logits / velocity field) → MLP or LSTM failure score; functional conformal prediction.
- Task failure, not semantic hazard. Key positioning paper.

### HazardArena — arXiv 2604.12447 (Fudan, Apr 2026)
- Safe/unsafe twins sharing objects, layout, action requirements; differ in risk-critical semantics (actor–target binding + instruction).
- 2,000+ assets, 40 tasks, 7 hazard categories (ISO 13482 + AutoRT).
- Stage events: attempt → commit (pre-IPE) → success.
- Finding: safe-only fine-tuning raises hazard completion on unsafe twins as capability grows.
- SOL gate before commit: L1 attribute rules (perfect by construction), L2 VLM judge (Qwen3-VL-32B; category-dependent, blind to property hazards).
- Note: intent = instruction by construction → weak test of latent intent value; use for permission-divergence axis.

## Failure-detection family
- **SAFECAST** — 2608.04246 (USC): contrast-set perturbations for hidden-state probe training and calibration.
- **VLA-FAIL** — 2606.21386: last-layer Mahalanobis distance + action-chunk consistency; no failure data.
- **FailureSpot** — 2609.04277: timestamp-level failure detection.
- **Foresight** — 2606.23085: action-conditioned world-model latents for long-horizon failure detection.
- **Tri-Info** — 2606.19998: interpretable, transferable failure prediction via information theory.
- **UNISafe** (Seo et al., 2025): uncertainty-aware latent safety filters.

## Latent / concept-level safety
- **SAFE-Dict** — 2602.01834: concept dictionaries over hidden activations; jailbreak instructions.
- **PRISM** — 2607.15218: hidden-state probe for physical vs. textual jailbreak (instruction-level, no vision).

## Runtime gates and verifiers
- **Pre-VLA** — 2605.22446: verifier over candidate action chunks before execution / world-model rollout.
- **ActFovea** — 2607.29169: spatiotemporal visual-action consistency safeguard for π0.
- **VLSA / SafeLIBERO** (Hu et al., 2025): plug-and-play CBF safety layer.
- **SafeVLA** (NeurIPS 2025): constrained RL safety alignment.

## VLA interpretability and probing
- **VLA commonsense retention** — 2606.19297: linear probes on every layer and every flow iteration (VLM + action expert).
- **SAEs for VLAs** — 2603.19183: sparse autoencoders; π0.5 action-expert hooks at final denoising step.
- **Observing and controlling features in VLAs** — 2603.05487: linear observers + minimal linear steering.
- **VLA-Trace** — 2605.30117; **Embodied Interpretability** — 2605.00321.

## Surveys
- Security of foundation-model-powered embodied agents — 2608.16843.
- VLA Safety: threats, challenges, evaluations, mechanisms — 2604.23775; list: github.com/LiQiiiii/Awesome-VLA-Safety.
