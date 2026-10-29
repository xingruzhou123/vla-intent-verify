# Research Plan — Evidence-Grounded Verification of VLA Policy Intent

**Working title:** *Is This Action Licensed? Verifying VLA Policy Intent Against Scene Evidence Before Execution*
**Project alias:** Paper 2 (RA-L follow-up → upgraded target: IEEE T-RL)
**Owner:** Grace (UT Austin)
**Plan date:** 2026-09-23
**Horizon:** 4 months (≈ 2026-09-24 → 2027-01-22), with a hard go/no-go gate at the end of Month 1

---

## 1. One-sentence thesis

Existing latent guards predict whether a rollout *resembles* previously observed failures. We instead ask whether the VLA's **actual policy intent** — read from its internal action representation — is **licensed by the scene evidence** required for that intent to be executed safely, and we block the action chunk before dispatch when it is not.

**Central research question**

> When does a VLA's internal representation reveal its actual interaction intent beyond what can be inferred from the instruction or the decoded action chunk, and can that incremental information support evidence-grounded safety decisions before execution?

This question is deliberately falsifiable. Month 1 exists to answer its first half.

---

## 2. Positioning against prior work

| Work | What it does | What it does *not* do (our opening) |
|---|---|---|
| **SAFE** (Gu et al., NeurIPS 2025) | Final-layer VLA hidden states → task-failure score, functional conformal prediction (FCP) | Task failure, not semantic hazard; single scalar; no evidence grounding |
| **RedVLA / SimpleVLA-Guard** (PKU, Apr 2026) | π0 action-expert `pre_velocity` feature (flow step 0, horizon index 0) → LSTM → step-wise risk → FCP halt; online ASR −59.5% in sim | Single binary score; only flow step 0; no intent/evidence decomposition; no real-robot guard; unsafe rollouts collapse into one latent region (possible "off-nominal" shortcut) |
| **HazardArena** (Fudan, Apr 2026) | Safe/unsafe twins; SOL gate before commit via attribute rules (L1) or external VLM judge (L2) | Gate uses instruction/observation/planned action, not policy internals; intent = instruction by construction |
| **SAFECAST, VLA-FAIL, FailureSpot, Foresight** | Hidden-state / world-model failure detection, calibration | Failure detection, not permission verification |
| **SAFE-Dict** | Concept dictionaries over hidden activations against jailbreak instructions | Instruction-level harm, not scene-state permission |
| **Tri-Info** | Interpretable, architecture-transferable failure prediction | Failure-mode diagnosis, not obligation-level scene verification |
| **Pre-VLA, ActFovea, VLSA** | Chunk-level verifier / consistency monitor / CBF layer | Geometric or consistency constraints; no semantic intent–evidence contract |
| **Layer × flow-step probing** (VLA commonsense-retention paper, 2026) | Linear probes on every layer and every flow iteration | Probes knowledge, not safety-relevant intent |

**Claims we must NOT make:** first latent VLA safety monitor; first pre-action guard; first hidden-state VLA failure detection; first matched safe/unsafe VLA evaluation.

**Taxonomy orthogonality (important for reviewers):**
- RedVLA classifies risk by **temporal structure** (state / cumulative / conditional).
- HazardArena classifies risk by **harm type** (food, property, chemical, privacy, fire, personal, electrical).
- We classify by **the evidence required to license an action** (reference / precondition / effect). These axes are orthogonal, not competing.

---

## 3. Core idea

Separate three quantities that prior guards conflate:

| Symbol | Meaning | Source |
|---|---|---|
| \(I_t^{\mathrm{cmd}}\) | Commanded intent — what the instruction asks for | Instruction |
| \(I_t^{\mathrm{policy}}\) | Policy intent — the interaction the policy is currently forming | VLA internals (what we read) |
| \(Y_t^{\mathrm{outcome}}\) | Realized outcome — what physically happens | Execution |

Then verify \(I_t^{\mathrm{policy}}\) against scene evidence \(E_t\) through three **obligations**:

1. **Reference fidelity** — does the policy intend to act on the object/region the instruction refers to? (\(I^{\mathrm{policy}} \overset{?}{=} I^{\mathrm{cmd}}\))
2. **Execution precondition** — does the current scene state permit this operation? (lid open, drawer open, gripper free, object graspable)
3. **Effect compatibility** — is the predicted effect of this intent permissible in this context? (liquid over electronics, powerbank into outlet, flammable on active stove)

Decision:

$$
\mathrm{ALLOW} \iff C_{\mathrm{ref}} \land C_{\mathrm{pre}} \land C_{\mathrm{eff}}
$$

with **joint** calibration (Section 5.5), plus a `REOBSERVE` outcome for insufficient evidence.

**Why the combination matters:**
- Evidence-only guards (rules / VLM) see that a hazard *exists* but not whether the policy is *about to act on it*.
- Latent-only guards know what the policy is doing but not *why* that is forbidden here.
- We verify the intersection, and output a typed reason.

Lineage: inherits CLEAR's (Paper 1, ICRA 2027) traceable, obligation-based verification, moving the object of verification from a generated text plan to a forming action representation.

---

## 4. Contributions (conditional on Month-1 outcome)

1. **Policy-intent extraction.** Read *policy* intent (distinct from commanded intent and realized outcome) from VLA internal action representations, and characterize *when* (layer × flow step × chunk) it becomes decodable.
2. **Obligation-factorized intent verification.** An intent → obligation → evidence → compatibility monitor that outputs ALLOW / BLOCK / REOBSERVE with a typed obligation-level reason.
3. **Counterfactual evidence-faithfulness evaluation.** Two orthogonal divergence axes (intent divergence × permission divergence) and controlled evidence interventions that test whether a monitor relies on the *right* evidence.
4. **Real-robot dispatch enforcement.** Closed-loop gating on hardware with zero actuator dispatch after BLOCK, measured latency, and lead time.

If Month 1 returns **red**, contribution 1 is dropped and the paper becomes an action-aware semantic safety contract (contributions 2–4 with a decoded-chunk intent source).

---

## 5. Method

### 5.1 Intent reader
- Input: VLA internal states \(z_t^{l,k}\) (layer \(l\), flow step \(k\)); for π0, both VLM prefix states and action-expert states.
- Output: \(\hat I_t^{\mathrm{policy}} = (\text{interaction target}, \text{interaction region}[, \text{verb}, \text{recipient}])\).
- Month 1: linear / small-MLP probes per \((l,k)\); later a light attention-pooling head over object tokens.
- Start with *(target, region)* only; add verb when labels are reliable.

### 5.2 Obligation router
Maps intent to the obligations it activates (e.g., `pour` → ref(source, recipient), pre(container open), eff(recipient accepts liquid)). Initially a hand-specified table; this is intentional and auditable.

### 5.3 Evidence retriever
- Extracts obligation-specific evidence \(E_t^j = R_j(o_t, \hat I_t)\) from observation only: object-centric features, open-vocabulary detector, or a VLM query **targeted by the intent** (what to look at, which object).
- **Privileged simulator state is never a monitor input**; it is used only for labels and as an *oracle-evidence* upper bound.
- Always report two settings: oracle evidence vs. perceived evidence, to separate intent error from perception error.

### 5.4 Compatibility check
Per obligation \(c_t^j = C_j(\hat I_t, E_t^j)\). Must be a structured comparison, **not** `MLP([z; e])` (that ablation is a baseline).

### 5.5 Calibration and decision
- Normalize per-obligation risks \(\tilde r_t^j\); global risk \(R_t = \max_j \tilde r_t^j\).
- Conformal-calibrate \(R_t\) on safe calibration rollouts to control global FPR (naively, three 5% heads → ≈14.3% FPR).
- Distinguish `confirmed violation` (BLOCK) from `insufficient evidence` (REOBSERVE).
- Reference deviations that are *not* hazardous → REOBSERVE / correction, **excluded** from hazard-blocked metrics (otherwise we slide back into task-failure detection).

---

## 6. Labels

### 6.1 Policy-intent label (future-oriented, to avoid circularity)
Labelling \(I_t^{\mathrm{policy}}\) by rolling out *only the current chunk* would make the Chunk-Rollout baseline equal to the label by construction. Therefore:

- **Label at time \(t\)** = object/region of the **next realized interaction event** in the full closed-loop rollout (may occur \(K\) chunks later).
- Interaction events from privileged sim instrumentation: first contact, gripper closure on object, object displacement, pour/place region entry.
- Every predictor (instruction, chunk, latent) may use only information available at \(t\).
- Also store: chunk-level "current-chunk interaction target" (for analysis), \(I^{\mathrm{cmd}}\), \(Y^{\mathrm{outcome}}\), commit time, first hazardous action index.

### 6.2 Safety labels
safe / unsafe; violated obligation(s); relevant object/region; first hazardous step; attempt / commit / success times (HazardArena-style stage events).

---

## 7. Data: two orthogonal axes

**Axis A — Intent divergence** (does the policy intend something other than commanded?) → tests the value of latent intent.
Perturbations: visually similar distractor; target absent; target occluded; hazard object stacked on target (RedVLA knife-on-bowl); distractor closer to gripper; target/distractor swap; underspecified instruction; closed-loop target switching.

**Axis B — Permission divergence** (policy follows the instruction, but is the action licensed here?) → tests the obligation verifier.
HazardArena-style twins: phone → tray vs. cooktop; liquid → cup vs. electronics; open vs. closed container; compatible vs. incompatible recipient.

| Policy intent | Scene permission | Verdict |
|---|---|---|
| matches command | allowed | ALLOW |
| matches command | forbidden | BLOCK (precondition / effect) |
| deviates | deviated target harmless | REOBSERVE (not counted as hazard) |
| deviates | deviated target hazardous | BLOCK (reference + effect) |

**Splits (grouped, never by frame):** seen task / unseen scene; unseen object instance; unseen instruction paraphrase; unseen actor–target composition; unseen hazard composition.

**Latent–intent value cases to test (need ≥1 to hold):**
1. *Earlier intent* — latent identifies the next interaction target chunks before the decoded chunk does.
2. *Partial-trajectory ambiguity* — pour vs. place, contact vs. go-around, before geometric divergence.
3. *Closed-loop target switching* — latent turns toward a distractor before chunks do.
4. *Geometric overlap* — stacked/nested/adjacent objects where endpoint geometry cannot resolve identity.

---

## 8. Baselines (by information source)

| # | Baseline | Information |
|---|---|---|
| B1 | Raw VLA | none |
| B2 | Instruction-parsed intent + evidence | instruction + scene |
| B3 | Chunk-Endpoint intent + evidence | final EE pose of decoded chunk → nearest object |
| B4 | Chunk-Rollout intent + evidence | roll out decoded chunk (kinematics/sim) → first contact, gripper-closure object, placement region — **using perceived, not privileged, object poses** |
| B5 | SimpleVLA-Guard (reproduced) | monolithic latent risk |
| B6 | SAFE or SAFECAST (one) | hidden-state failure detector |
| B7 | HazardArena SOL-L1 | attribute rules |
| B8 | HazardArena SOL-L2 | external VLM judge |
| A1 | Intent-only ablation | latent intent, no evidence |
| A2 | Evidence-only ablation | evidence, no intent |
| A3 | `MLP([z; e])` concatenation | unstructured fusion |
| O  | Oracle intent / oracle evidence | upper bounds |
| **Ours** | Obligation-factorized latent intent + evidence | — |

---

## 9. Metrics

- **Intent:** target accuracy, verb accuracy (if labelled), earliest reliable prediction (in control steps, chunks, and steps-before-commit — *not* ms, since latent and chunk come from the same forward pass), calibration (ECE).
- **Safety:** hazard-blocked rate, safe-allowed rate, safe-task success, global FPR, lead time to commit, unsafe actuator calls after BLOCK (must be 0), added latency.
- **Faithfulness:** verdict flip rate under *relevant* evidence interventions (should flip) vs. *irrelevant* perturbations (should not); obligation-diagnosis accuracy.
- **Generalization:** all of the above on each grouped split.

Report all results with seeds and confidence intervals. Leave result-table cells empty until measured — no predicted outcomes in drafts.

---

## 10. Timeline

### Month 1 (≈ 09-24 → 10-23): Go/No-Go — *does latent carry incremental intent information?*
Runs on the local RTX 5080 (inference + probing only; see §11).

**Week 1 — Environment and hooks**
- [ ] CUDA 12.8+ / PyTorch build that supports Blackwell (sm_120); verify GPU works end-to-end.
- [ ] π0 in PyTorch (openpi PyTorch port or LeRobot `pi0`) with a LIBERO-fine-tuned checkpoint; confirm which checkpoints are published.
- [ ] LIBERO + MuJoCo headless rendering (EGL/OSMesa).
- [ ] Reproduce benign success on 2–3 LIBERO tasks (sanity check vs. reported numbers).
- [ ] Forward hooks: VLM prefix states (selected layers) + action-expert states for **all layers × all flow steps**; pooled storage (mean over horizon + horizon index 0).
- [ ] Check public availability and licence of RedVLA and HazardArena code, assets, and checkpoints. Pick **one** environment for Month 1 (prefer whichever runs π0 on LIBERO cleanly — likely RedVLA/LIBERO).

**Week 2 — Scenes and labels**
- [ ] 3 manipulation tasks × 4 intent-divergence perturbations (similar distractor, target absent, hazard stacked on target, distractor closer/swap).
- [ ] Interaction-event instrumentation (contact, closure, displacement, region entry).
- [ ] Future-oriented \(I^{\mathrm{policy}}\) labels; store \(I^{\mathrm{cmd}}\), \(Y^{\mathrm{outcome}}\), commit time.
- [ ] Collect ≈ 300–500 episodes with latent dumps; hold out ≥1 distractor object set.

**Week 3 — Intent readers**
- [ ] B2 instruction intent; B3 Chunk-Endpoint; B4 Chunk-Rollout (perceived poses; privileged-pose variant reported separately as upper bound).
- [ ] Latent probes per (layer, flow step); accuracy-vs-time curves relative to next interaction event.
- [ ] Oracle future-interaction upper bound.

**Week 4 — Guard baseline and decision**
- [ ] Reproduce SimpleVLA-Guard (if code available; otherwise re-implement from paper appendix: `pre_velocity`, flow idx 0, horizon idx 0, LSTM-256, ramp-weighted BCE, FCP α = 0.05).
- [ ] Write `decision_month1.md` with the go/no-go verdict.

**Go/No-Go criteria (initial; freeze before looking at held-out results):**
- 🟢 **Green (latent paper):** at least one holds on held-out distractors —
  - latent reaches the accuracy level of B4 (e.g., ≥ 80% next-target accuracy) ≥ 1 chunk earlier, on ≥ 2 perturbation types; **or**
  - ≥ 10-point target-accuracy gain over B4 in geometric-overlap cases; **or**
  - latent predicts target switching before chunks do, in a statistically reliable fraction of switch events.
- 🟡 **Yellow:** small/inconsistent latent gains → latent becomes one optional intent source; the paper centres on intent–evidence verification.
- 🔴 **Red:** B4 recovers policy intent as well and as early → drop the latent claim; pursue an action-aware semantic safety contract.

### Month 2 (≈ 10-24 → 11-22): Full monitor in simulation
- Obligation router, evidence retriever (oracle + perceived), compatibility heads, joint conformal calibration, REOBSERVE.
- Add Axis B (HazardArena-style twins) for precondition/effect obligations.
- Small controlled diagnostic suite (6–10 tasks) for evidence interventions.
- Baselines B5–B8, ablations A1–A3.
- If A100 is needed (π0 fine-tuning on new scenes, second backbone), run it there.

### Month 3 (≈ 11-23 → 12-22): Generalization, analysis, second backbone, real-robot setup
- Grouped generalization splits; faithfulness interventions.
- Layer × flow-step onset analysis per obligation.
- Cross-architecture: OpenVLA (autoregressive; no flow-step axis) as the secondary backbone.
- Real robot: π0 inference server → chunk buffer → gate → low-level controller, independent actuator-command logging.

### Month 4 (≈ 12-23 → 01-22): Real-robot trials and writing
- 6–8 scenes, ≥ 2 per obligation, safe and unsafe, 10–15 trials each (≈ 120–180 trials); report CIs.
- Metrics table: hazard blocked / safe allowed / task success / added latency / unsafe actuator calls after BLOCK.
- Draft paper; release code, data-generation configs, videos.

---

## 11. Compute plan

| Workload | Where | Notes |
|---|---|---|
| π0 inference + hooks in LIBERO | **RTX 5080 (16 GB)** | π0 ≈ 3.3B params; bf16 inference fits (openpi lists ≈ 8 GB+ for inference) |
| Probes, LSTM guard, compatibility heads | RTX 5080 | small models |
| π0 LoRA / full fine-tuning | **A100** | exceeds 16 GB (openpi lists ≈ 22.5 GB+ for LoRA, ≈ 70 GB+ for full) |
| Second backbone (OpenVLA 7B) inference | A100 preferred | 7B bf16 ≈ 15 GB; too tight on 16 GB with hooks |
| External VLM judge (SOL-L2) | A100 or API | depends on judge size |

**Storage budget.** Full action-expert dumps (18 layers × 10 flow steps × 50 horizon × 1024 × fp16) ≈ 18 MB per inference call → hundreds of GB per few hundred episodes. Store pooled vectors (horizon mean + horizon index 0) → ≈ 0.7 MB per call, ≈ 20–40 GB for Month 1. Keep a small subset of episodes with full dumps for later analysis.

**Blackwell risk.** RTX 5080 (sm_120) needs CUDA ≥ 12.8 and a recent PyTorch; JAX support may lag. Prefer a PyTorch π0 implementation. Validate on day 1.

---

## 12. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Latent adds nothing over decoded chunk | Month-1 gate; red-light fallback paper |
| Circular intent labels | Future-oriented labels (§6.1) |
| Unfair chunk baseline (privileged poses) | B4 uses perceived poses; privileged variant reported separately |
| Evidence perception dominates error | Oracle vs. perceived evidence reported separately |
| Joint FPR inflation | Max-risk conformal calibration; REOBSERVE |
| Shortcut learning (background, object count) | Matched scenes; grouped splits; irrelevant-perturbation tests |
| RedVLA/HazardArena code not runnable | Month-1 uses one environment only; build minimal scenes in LIBERO |
| Blackwell software incompatibility | PyTorch π0; fall back to A100 for Month 1 if blocked > 2 days |
| Monitor latency too high for control loop | Probes are cheap; VLM evidence queried only for activated obligations; measure per-chunk latency |
| Scope creep | Pre-registered go/no-go; no second environment before Month 2 |

---

## 13. Venue plan

- 🟢 Green: IEEE **T-RL** (option: CoRL/RSS conference version first, then T-RL extension). T-RO only with substantially broader validation.
- 🟡 Yellow: T-RL possible; **RA-L** safer.
- 🔴 Red: RA-L or ICRA/IROS.

Journal expectations: ≥ 2 VLA architectures, real-robot results with CIs, failure-case analysis separating perception vs. intent error, complete baselines and ablations.

Coordination: a face-robot safety paper is also being considered for T-RL; stagger submissions.

---

## 14. Suggested repo layout (for Claude Code)

```
vla-intent-verify/
├── plan.md
├── status.md
├── configs/                 # env, model, scene, experiment configs (yaml)
├── envs/
│   ├── libero_scenes/       # intent-divergence perturbations
│   └── instrumentation.py   # interaction-event detection (privileged, labels only)
├── models/
│   ├── pi0_hooks.py         # activation capture: layers × flow steps
│   └── checkpoints.md       # which checkpoints, where from, licences
├── data/
│   ├── collect.py           # rollouts + latent dumps + labels
│   └── splits.py            # grouped splits
├── intent/
│   ├── instruction_intent.py
│   ├── chunk_endpoint.py
│   ├── chunk_rollout.py
│   └── latent_probes.py
├── guards/
│   └── simplevla_guard.py   # baseline reproduction
├── monitor/                 # Month 2+: router, evidence, compatibility, calibration
├── analysis/
│   ├── lead_time.py
│   └── layer_flow_heatmaps.py
├── results/                 # never edited by hand
└── notes/
    └── decision_month1.md
```
