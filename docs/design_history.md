# Design History

How the framing evolved (2026-09-23). Keep this so later decisions can be traced back to their reasons.

## v0 — "VLA Shield" proposal
- Pre-decoder latent safety gating; obligation-conditioned detection (target grounding / affordance-precondition / forbidden transition); latent + visual evidence; onset analysis; real execution gate.
- Original proposal text: `docs/history/00_original_proposal.md`.

## Round 1 — literature check
- Pre-action latent gating already done: SAFE (hidden-state failure detection), SimpleVLA-Guard (π0 `pre_velocity`, LSTM, FCP halt).
- Matched safe/unsafe twins already done: HazardArena.
- Still open: obligation decomposition; latent + scene-evidence grounding; safety-specific layer × flow-step onset; real-robot latent gate.
- Pivot: grounding becomes the core; latent gating becomes mechanism, not contribution.

## Round 2 — correction of the central argument
- Overclaim removed: "latent-only guards theoretically fail on semantic twins." HazardArena twins change instruction and actor–target binding, so latents likely separate them.
- New claim: a single latent risk score cannot verify whether an action intent is licensed by current scene evidence.
- Structure: intent reader → obligation router → evidence retriever → compatibility check (not feature concatenation).
- Taxonomy defined by required evidence type; "forbidden transition" → "effect compatibility".

## Round 3 — the critical question
- Why read intent from latents at all? Instruction parsing and decoded action chunks are cheaper intent sources.
- On HazardArena, intent = instruction → latent adds nothing there.
- Latent value only when policy intent diverges from instruction and the chunk geometry does not reveal it (or reveals it later).
- Added baselines: instruction-parsed intent, decoded-chunk intent.
- Added: joint calibration (three 5% heads ≈ 14.3% FPR); oracle vs. perceived evidence reporting.

## Round 4 — labels and gate
- Separate \(I^{\mathrm{cmd}}\), \(I^{\mathrm{policy}}\), \(Y^{\mathrm{outcome}}\).
- Two data axes: intent divergence × permission divergence (four-quadrant verdict table).
- Four latent-value cases: earlier intent, partial-trajectory ambiguity, target switching, geometric overlap.
- Obligations renamed: reference fidelity / execution precondition / effect compatibility.
- Month-1 green / yellow / red gate.

## Round 5 — fixes to Round 4
- **Label circularity:** labelling intent by rolling out only the current chunk makes Chunk-Rollout equal to the label. Fix: label = next realized interaction event in the full closed-loop rollout.
- **Fair chunk baseline:** Chunk-Rollout must use perceived poses; privileged variant reported separately.
- **Non-hazardous reference deviation** → REOBSERVE, excluded from hazard metrics (else the paper becomes task-failure detection).
- Lead time measured in chunks / control steps, not ms (latent and chunk come from the same forward pass).

## Venue reasoning
- Green → T-RL; yellow → T-RL or RA-L; red → RA-L / ICRA / IROS.
- π0 primary; OpenVLA secondary for cross-architecture (originally reserved for this paper).
