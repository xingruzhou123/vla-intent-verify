# Safe probing — Stage 1 design note (v0, 2026-11-02)

Source of truth for definitions: `world-model/docs/Safe Probing and Evidence Sufficiency Study Plan.md` (study plan, §1–§7).
Environment and labels: `envs/feasibility_carry/` (imported, never edited), labels frozen at version 3, feasible sets as in
`feasible.context_tables` (τ_c = 0.9, τ_s = 0.05, τ_d = 0.05), sweep `data/sweep_v1` (48 objects, R = 5 stored branch states).
Scope of this note: design + smoke test. Nothing here was run at Stage-1 scale and no model was trained.
Everything marked **Q** is a decision the owner has to take; the smoke test's measured numbers are in
`probing/results/smoke/report.md`.

A sibling effort exists in the same repo: `analysis/active_probe_gates.py` (owner request 2026-11-02, running while this
note was written) checks the value of probing with a *simulated* sensor model and no probe rollouts. This note is the
rollout-based design; its observation tiers (§4) are chosen so that the two can be compared (**Q4**).

---

## 1. Probe set

### 1.1 Trial lift (implemented: `probing/probes.py: ProbeSpec("lift", …)`)

| Parameter | v0 value | Reason |
|---|---|---|
| grasp | the CFD grasp primitive: ramp at 2 cm/s until both pads > 0.5 N, clamp at the force bound F_p, hold 1.5 s | identical mechanics to the carry's grasp, so a lift at F_p tests the same contact the carry would use |
| F_p | {7.29, 10.63, 15.49} N = CFD grid levels F_idx 1, 2, 3 | a lift at a grid level tests the static part of the slip boundary exactly at that level; 15.49 N is the highest level below every fragile tolerance (band ≥ 16.0 N) once the measured ≈ 0.3 N overshoot of the sustained 50 ms peak over the commanded bound is subtracted |
| h_p | 1 cm (variant 2 cm) | a slip during the probe drops the object ≤ h_p back onto the table; 1 cm is within the 6 cm face, so lowering with a slipped object is harmless |
| lift / top hold / lower / settle | 0.75 s / 0.5 s / 0.75 s / 0.25 s, min-jerk | peak acceleration 5.77 h/T² ≈ 0.1 m/s² (1 % g): a quasi-static test; the 0.5 s top hold is the slow-creep window (CFD measured creep < 0.8 mm per carry) |
| release | gripper opens at the grasp height, 0.5 s (env `release_settle`) | the post-probe state is "pre-contact-like": gripper open at the original pose, object wherever the probe left it |
| sim time / wall | 5.4 s sim; 1.2–1.4 s wall per branch when the machine is lightly loaded, 6.3 s mean at load ≈ 25 | measured in the smoke test (81 lift branches) |

What a lift at F_p reveals, physically: one bit — whether the static slip threshold F_stat(θ) ≈ m g / (2 μ) (plus 3–10 N for an
off-centre CoM, from the CFD boundaries) lies below F_p. From the CFD sweep (50 % slip crossing at s = 3.0, per object):
light-normal 4.6–9.8 N, light-slippery 8–20 N, medium-normal 10–22 N, heavy-normal 17–30 N, medium-slippery 24–45 N,
heavy-slippery 30–67 N. So 7.29 N separates light-normal from the rest, 10.63 N splits the light-slippery and
medium-normal bins, and 15.49 N is the decision-relevant bit for fragile contexts (a fragile context is feasible in CFD
only when F_min < f_tol ≤ 24 N; see §7). With the tier-2 load channel (§4) a held lift also measures m g (smoke: 5.1 N measured
vs 5.12 N true) and a slipping lift measures ≈ 2 μ_k F_p.

### 1.2 Light squeeze (implemented: `ProbeSpec("squeeze", …)`)

F_p ∈ {10.63, 15.49} N, ramp + 1.5 s hold, release, no motion; sim 2.6 s.

**Finding (by construction of the env, not a measurement):** fragility is a label parameter and does not enter the simulation
(`env.py` docstring; plan §3.2). The object is a rigid box: a squeeze produces the same record for a fragile and a rigid object
with the same (m, μ, d), and with the object on the table it carries no load, so the record is also independent of m and μ
(finger positions = object width, identical for all objects). A squeeze is therefore informative **only if damage is declared
observable** (e.g. a visible crush when the sustained pad force exceeds f_tol). Then it is a threshold test on f_tol that is
destructive exactly when it is informative: under the worst-case standard every admissible squeeze (F_p ≤ 15.5 N) is
uninformative; under a probabilistic standard with a uniform prior over the two fragility classes, a squeeze at 20 N damages
0.5 × P(f_tol < 20 | fragile) = 25 % of objects, far above any sensible probe-damage budget.

**Q3** — keep the squeeze as the *uninformative control probe* (recommended: it is exactly the "no deformation under a light squeeze
proves nothing about a hard one" case that H1a in Stage 2 should expose), make damage observable, or replace it by a lateral-pulse
friction probe (pads closed at ≤ 15.5 N, slow lateral drag on the table, friction from the slip onset — the sensor model that
`active_probe_gates.py` assumes). Not implemented here.

### 1.3 Multi-step combinations

Feasible in two forms, both implemented by the primitives: (a) *restore between probes* (C1 style): the outcome table of a
sequence is the product of the single-probe tables given θ and the state, so no extra simulation is needed; (b) *chained*
(C3 style, `execute_probe` without restore): the second probe starts from the post-first-probe state; K = 2 over 6 single probes
gives 36 chains per state. Under the idealised deterministic lift model (§7 preview) K = 2 resolves **no** additional context
because the three admissible lift levels are nested threshold tests and 15.49 N already carries the decisive bit; multi-step
matters only through noise at boundary cells and through the tier-2 channel (two forces bracket m/μ). **Q7:** K = 1 or 2.

## 2. Branching from the exact same state

The CFD env already stores the full MuJoCo integration state (`mjSTATE_INTEGRATION`: time, qpos, qvel, act, ctrl, warm-start,
user fields) at the pre-contact point and restores it with `mj_setState` (`env.save_state` / `env.restore_state`). The sweep saved
one such state per repeat (`data/sweep_v1/<object>/branch_state_r{0..4}.npz`). Stage 1 reuses exactly those states, so that:

| Condition | Decision input | Execution | New simulation needed |
|---|---|---|---|
| C0 | Θ0 | carry (F, s) from the stored state | none — the outcome is the stored CFD grid branch (F_idx, s_idx, r) |
| C1 | Θ0 + probe feedback o from a branch of the same state, state restored | restore → probe → record o → restore → carry (F, s) from the stored state | the probe branch only; the follow-up outcome is again the stored CFD branch (verified bit-identical, smoke check 5) |
| C2 | Θ0 + the deployment-visible post-probe state (object pose from the cameras, gripper open at the original pose), probe record masked | restore → probe → **no restore** → `carry_after(F, s)` | probe + carry branch, labelled with the frozen labeler on the carry segment |
| C3 | Θ0 + probe feedback o + post-probe state | as C2 | as C2 |

Implementation: `probes.run_probe` (restore + execute), `probes.execute_probe` (no restore; chaining and C2/C3),
`probes.carry_after` (follow-up grasp-and-carry, labels via `labels.all_labels` on the carry rows only, so that the probe's own
motion never counts as carry slip), `sweep.run_branch` (the CFD branch, used only to verify C1).

Two details that matter for exactness, both verified in the smoke test:
- The per-repeat friction perturbation (`mu_scale`, ±3 %) is a *model* field, not part of the saved state. Before loading state r
  the env must be reset with `set_perturbation(**meta["perturbations"][r])`; then the replay of a CFD grid branch is bit-identical
  (max |diff| = 0 in float32). Without it the replay of r > 0 differs (smoke check 5b). Note for CFD, not changed here:
  `refine.py` loads the stored states without re-applying the perturbation, so its bisection probes for r > 0 ran at μ_scale = 1.
- State change by a probe: a 15.49 N / 1 cm lift moved the object by 0.3–0.8 mm (held or slipped alike); the C3 carry from the
  post-probe state reproduced the C0 labels on 9 of 9 (state, object) pairs and the slip / final-position numbers to within 0.2 mm /
  1 mm. For balanced boxes the state-change arm of C0–C3 is therefore expected to be small; off-centre and chained probes remain to be measured.
- Env-level state outside MuJoCo (grip mode, force bound, phase, controller goal): `restore_state` restores the first two and
  resets the phase; the OSC goal is overwritten by the first control step of any branch. Exactness is checked with a fingerprint
  (state buffer + env fields + model fields: force range, friction, CoM offset, contact parameters) before and after
  probe + restore, and with bit-identical re-execution.

In C1 the plan requires that the decision maker be *told* the feedback is counterfactual. For the Stage-1 ground-truth decision
makers (§8) the update is set-based and the flag changes nothing; it becomes an input only for the learned models of Stage 2.

## 3. Repeats and noise sources

The simulator is deterministic (single-threaded MuJoCo, fixed seed): re-executing a probe from the same restored state gives a
bit-identical record (smoke check 2). Repeats therefore differ only through what is deliberately randomised:

| Source | Where | Magnitude | Role |
|---|---|---|---|
| pre-contact state perturbation (CFD repeats) | one stored state per repeat r | object pose sd 2 mm, yaw sd 2°, friction scale sd 3 % | the *physical* outcome distribution of a probe; the 3 % friction scale is a parameter nuisance, not sensor noise (**Q5b**: keep it as nuisance, as CFD does, or fold it into θ) |
| sensor noise (§4) | post hoc on the record, seeded | pose 2 mm per frame, force 0.5 N, load 0.5 N + 0.5 N bias | the *observation* distribution; free to draw many times |
| nothing else | — | — | no controller or contact randomness |

**Repeat count.** v0 recommendation: R = 5, reusing the CFD states (then C0 and C1 outcomes need no new carry simulation).
Away from the slip boundary a lift's outcome is the same in every repeat (smoke: 3/3 states agree on both clear objects, slip
0.0–0.7 mm held vs 9.3–9.7 mm slipped). The boundary object of the smoke test (medium_normal_balanced_1, static crossing 9.8 N,
lift at 10.63 N) held in all three states with creep growing 0.0 → 0.5 → 0.9 mm across the perturbed states: at 0.8 N above the
crossing the perturbations did not flip the outcome, so flips are confined to a band narrower than that; its width is a Stage-1 measurement. The price of
R = 5 is statistical: with binary outcomes Fisher's exact test can call two conditions distinguishable only when their repeat sets
are perfectly separated (5/0 vs 0/5, p = 0.0079; 5/0 vs 1/4 gives p = 0.048), so two conditions with true rates 0.9 and 0.1 are
called distinguishable with probability ≈ 0.35. **Q5a:** R = 5 (free) or R = 10 (5 new perturbed states per object: 240 approaches
plus, for C0/C1 at the new states, only the released actions, ≈ 1 h wall).

## 4. Observations recorded per probe

Stored per branch (ground truth, as CFD does): the full 500 Hz record (`SAMPLE_COLS`, float32 npz), the outcome summary
(`probes.probe_outcome`) and one or more seeded noisy observations (`probes.observe`). What the decision maker may see is tiered:

| Tier | Channel | Sensor (plan §1) | Noise model (assumption, `probes.SENSOR`) | Derived outcome |
|---|---|---|---|---|
| 1 | commanded F_p, h_p, durations | — | exact | — |
| 1 | finger normal force per pad (steady window) | gripper force | sd 0.5 N | confirms contact at F_p (uninformative beyond that: P1 of CFD shows force tracking R² ≥ 0.9999) |
| 1 | finger joint positions | proprioception | sd 0.2 mm | object width (identical for all objects) |
| 1 | end-effector pose and height trajectory | proprioception | exact | the commanded lift happened |
| 1 | object pose relative to the gripper at 20 Hz | RGB (two cameras) | sd 2 mm per frame, averaged over the 0.5 s top hold (10 frames) | slip estimate, height gain at the top, on-table flag; **outcome class** held (z gain > h/2 and slip < 5 mm) vs slipped |
| 2 | supported load during the top hold | arm external-force estimate (wrist F/T or joint torques) | sd 0.5 N + per-episode bias sd 0.5 N | m g when held; ≈ 2 μ_k F_p when slipping |
| 3 (TBD in plan) | per-pad tangential force, contact count, torsion | tactile | — | not used in v0 |
| never | f_tol, damage proxies, true m / μ / d, the simulator's slip label | — | — | labels and oracle only (as CFD's `labels.json`) |

**Q4:** tier 1 only (the plan's v0 sensor list read strictly) or tier 1 + 2 (what `active_probe_gates.py` assumes). **Q8:** whether a
damage event during a probe is observable (§1.2) and whether a slip-drop from h_p counts as damage (the frozen labeler attributes
damage only to pad forces, so a 1–2 cm drop is harmless by definition).

## 5. Θ(o): conditions consistent with the observations

Setting: a finite condition set Θ0 ⊂ {(object, fragility class)} (the prior set, **Q1**, §7), probes p ∈ P, outcome alphabet O_p
(tier 1: {held, slipped}; tier 2 adds the load quantised at 1 N; continuous channels are handled below). Stage 1 produces, for
every θ ∈ Θ0 and p, the R repeat outcomes {o_{θ,p,r}} and the Jeffreys-smoothed rates P̂(o | θ, p) = (n_o + ½)/(R + ½|O_p|).

**Indistinguishability test (distribution level, never a single observation).** θ ~_p θ' iff the two-sample exact test on the two
repeat sets does not reject at level α: Fisher's exact test for a binary alphabet, Fisher–Freeman–Halton for a larger one, and
for a continuous channel a permutation test (all C(2R, R) relabellings) on the difference of means. α = 0.01 (**recommended**: the
harmful error is to call two conditions distinguishable when they are not; with R = 5 this means "distinguishable only if the
five repeats of one condition all fall in an outcome class none of the other's five produced").

**Consistent set after one probe:** Θ(o) = ⋃ { [θ]_p : θ ∈ Θ0, o ∈ supp_R(θ, p) }, where supp_R(θ, p) is the set of outcome classes
seen in the R repeats of θ and [θ]_p = {θ' : θ' ~_p θ} is θ's indistinguishability class. In words: a condition stays in Θ(o) if it
produced o in some repeat, or if it is statistically indistinguishable from some condition that did. A condition is excluded only
when it never produced o in R repeats *and* its repeat distribution is distinguishable from that of every condition that did.
For a continuous channel "produced o" means o lies within the condition's repeat range widened by ±2 σ_sensor.
Several probes (C1 style, restore between them): Θ(o_1, …, o_k) = ⋂_i Θ(o_i).

For the probabilistic standard (§6) the same tables give a posterior π(θ | o) ∝ π0(θ) P̂(o | θ, p) with π0 uniform on Θ0, and the
credible set Θ_δ(o) = the smallest set of posterior mass ≥ 1 − δ.

Used by: the Stage-1 decision makers (§8, set update), the false-sufficiency metric (release of a with some θ ∈ Θ(o) for which
a ∉ 𝓕(θ)), and in Stage 2 the definition of "contexts that remain indistinguishable after the probe" for H1a (Θ(o) still contains
conditions with conflicting feasible sets).

**Q9:** "resolvable" under stochastic outcomes — worst case over the repeat support (recommended, matches the worst-case standard)
or with probability ≥ 1 − δ over the outcome distribution.

## 6. Risk standard — two options

Both reuse the CFD feasible set 𝓕(θ, u) (urgency u is known from the instruction) and the per-cell rates P̂_success, P̂_slip,
P̂_damage over the CFD repeats. "Safe" for the carry = slip ≤ τ_s and damage ≤ τ_d; "advances the task" = deadline success ≥ τ_c;
together, a ∈ 𝓕(θ, u). For a probe, "safe" = its own damage rate ≤ τ_d (and, pending **Q8**, no drop).

| | W: worst-case | P: probabilistic |
|---|---|---|
| action a admissible under Θ | a ∈ 𝓕(θ, u) for **every** θ ∈ Θ | E_π(θ|o)[P̂_damage(θ, a)] ≤ δ_d and E[P̂_slip(θ, a)] ≤ δ_s (δ = 0.05, the τ values), with π0 uniform on Θ0 |
| probe p admissible under Θ | P̂_damage(p | θ) ≤ τ_d for every θ ∈ Θ (with R = 5: no damage in any repeat) | E_π[P̂_damage(p | θ)] ≤ δ_p (plan: cap on the probe damage rate; proposed δ_p = 0.05) |
| value of the released action | min_θ∈Θ P̂_success(θ, u, a); refusal = 0 | E_π[P̂_success(θ, u, a)]; refusal = 0 |
| false sufficiency | release of a ∉ ⋂_θ∈Θ(o) 𝓕(θ, u) | release of a with E_π[unsafe] > δ (Θ(o) read as the (1 − δ)-credible set) |
| needs a prior | no | yes (uniform over the sampled objects of Θ0 = the CFD sampling distribution) |
| behaviour with physically infeasible conditions in Θ0 (69 of 192 contexts) | nothing is admissible until they are excluded; probing value = exclusion power; high refusal | graded; a probe is admissible as soon as the prior mass of damaging conditions is ≤ δ_p (e.g. a 16.5 N squeeze is admissible, 17 N is not, and neither tells much) |
| matches the plan's definitions | §3 definition and the H1b metric are stated set-wise | the plan's second clause ("risk over Θ(o) within budget") |

**Recommendation:** W as the primary standard for Stage 1 (taxonomy, Gate 1, false-sufficiency rate), P with δ = δ_p = 0.05
reported alongside as the sensitivity; freeze both before Stage 1 (plan §8, definition drift). **Q2.**

## 7. Assigning contexts to the four categories

Unit: a context c = (object, fragility class, urgency) as in CFD (192 contexts; the five stored states are its repeats). Inputs:
the CFD feasible sets 𝓕(θ, u) (reused unchanged, `feasible.context_tables`, label v3), the prior set Θ0 (**Q1**), the Stage-1 probe
tables, the risk standard, K.

| Category | Rule (standard W) | From existing CFD data? |
|---|---|---|
| physically infeasible | 𝓕(θ*, u) = ∅ | yes: 69 of 192 contexts (60 fragile, 9 rigid), identical to CFD's "infeasible by design" |
| no probe needed | ⋂_{θ ∈ Θ0} 𝓕(θ, u) ≠ ∅ | yes: with Θ0 = the own coarse-bin cell with fragility as stated it is CFD's "common feasible cell" (24 of 48 groups = 96 contexts); for any other Θ0 it is the same intersection over a different set |
| resolvable by safe probing | a strategy tree of admissible probes, depth ≤ K, exists such that for every outcome sequence θ* produces with P̂ > 0 the leaf's Θ(o) has a common feasible cell | needs the Stage-1 probe tables; exhaustive search over trees (K ≤ 2, ≤ 6 single probes) |
| unresolvable under stated limits | 𝓕(θ*, u) ≠ ∅ and neither of the above | same |

**Preview with an idealised probe model** (`probing/taxonomy_preview.py`, no simulation: a lift at F_p holds iff the CFD static slip
crossing F50(s = 3.0) < F_p; squeeze uninformative; lifts admissible iff F_p + 0.5 N ≤ f_tol for all θ ∈ Θ0; standard W):

| Θ0 option | K | no probe | resolvable | unresolvable | infeasible | fragile (no / res / unres / inf) | rigid |
|---|---|---|---|---|---|---|---|
| A: population, 48 objects × 2 fragility classes | 1, 2 | 0 | 40 | 83 | 69 | 0 / 20 / 16 / 60 | 0 / 20 / 67 / 9 |
| B: ε-neighbourhood of the coarse text incl. the fragility bin | 1, 2 | 0 | 50 | 73 | 69 | 0 / 25 / 11 / 60 | 0 / 25 / 62 / 9 |
| C: own bin cell, fragility as stated | 1, 2 | 96 | 19 | 8 | 69 | 20 / 13 / 3 / 60 | 76 / 6 / 5 / 9 |
| D: all 48 objects, fragility class as stated | 1, 2 | 0 | 85 | 38 | 69 | 0 / 20 / 16 / 60 | 0 / 65 / 22 / 9 |
| E: ε-neighbourhood over mass / friction / balance, fragility as stated | 1, 2 | 16 | 79 | 28 | 69 | 0 / 25 / 11 / 60 | 16 / 54 / 17 / 9 |

Reading. With the fragility class inside the prior (A, B) the unresolvable class dominates: every context that needs F > 15.5 N
and might be fragile cannot be resolved by any admissible probe, which is the fragility-is-unprobeable fact of §1.2, not a
limitation of the lift. With the own bin only (C) almost nothing needs a probe (CFD's finding that coarse text decides most
cases). Gate 1 asks for both middle categories to be sizeable: D or E give that (resolvable 79–85, unresolvable 28–38 of 192).
K = 2 changes nothing under the idealised model (§1.3). **Q1 (recommendation: E as the primary setting — ε-noisy coarse text as in
CFD on mass / friction / balance, fragility bin trusted — with A as the stress setting).**

## 8. Stage-1 decision makers (ground truth, no learning)

Common inputs: Θ0, u, the probe tables T[θ][p] (R repeat outcomes), the CFD tables 𝓕(θ, u) and P̂_success(θ, u, a), the standard
(W or P), the budget K and the probe-damage budget. All five end with the same release rule so that only the probe choice differs:

```
ReleaseRule(Θ):  A ← admissible actions under Θ (§6);  if A = ∅: refuse
                 W: a ← argmax_{a∈A} margin of a on ⋂_{θ∈Θ} 𝓕(θ,u)   (CFD teacher rule, feasible.margin_grid; ties toward s = 1)
                 P: a ← argmax_{a∈A} E_π[P̂_success(θ,u,a)]          (ties by margin)
Admissible(Θ):   probes p with P̂_damage(p|θ) ≤ τ_d ∀θ∈Θ (W)  /  E_π[P̂_damage(p|θ)] ≤ δ_p (P)
Update(Θ, p, o): §5 (set) / posterior (P)
```

1. **Conservative (no probe):** ReleaseRule(Θ0). Contacts 0. Its C0 outcome is the stored CFD branch.
2. **Fixed probe:** a single predeclared probe p_fix executed in every context, then ReleaseRule(Update(Θ0, p_fix, o)).
   p_fix must be admissible under Θ0 in every context: recommended lift 15.49 N / 1 cm (the decisive bit, §1.1). Contacts 1.
3. **Information-max:** for i = 1..K: among Admissible(Θ) pick argmax_p I(θ; o | Θ, p), the mutual information under the uniform
   distribution on Θ and the smoothed P̂(o | θ, p) (for the set update this is the expected log-reduction of |Θ|); stop if the best
   I is 0; execute, update. Then ReleaseRule(Θ).
4. **Decision-value:** for i = 1..K: among Admissible(Θ) pick argmax_p EV(p) = Σ_o P̂(o | Θ, p) V(Update(Θ, p, o)) − V(Θ), with
   P̂(o | Θ, p) the uniform mixture over Θ and V(Θ) = value of ReleaseRule(Θ) (§6 "value of the released action"; refusal 0);
   for K = 2 solve the two-step tree exactly (cheap); probe only if EV(p) > 0; else ReleaseRule(Θ).
5. **Extra-contact control** (plan §5 item 5): (a) *contact-matched fixed probe*: p_fix executed n_c times, n_c = the number of
   probes decision maker 4 used in that context (same contacts, information only from the repeated fixed probe); the gain of 4
   over 5a at equal contacts is the Gate-1 quantity; (b) *C2 of decision maker 4*: its probe executed, feedback masked, state not
   restored — isolates the state-change part of C3.

Per context and decision maker the conditions C0–C3 of §2 are evaluated and the plan's §6 metrics computed from the true θ*:
probe damage rate (over executed probes), false sufficiency rate (§6), dangerous release rate (P̂_damage(θ*, a) > τ_d for C1,
the realised branch for C3), safe completion (P̂_success(θ*, u, a) from the CFD cell for C0/C1, the new branch for C2/C3),
unnecessary refusal (refused with 𝓕(θ*, u) ≠ ∅), interaction cost. Gate 1: 4 beats 1, 2, 3 and 5a at equal contacts and equal
risk budget, with the resolvable and unresolvable categories both sizeable. **Q11:** "task return" = P̂_success of the released
action with refusal 0 (as above), or with an explicit penalty for a dangerous release.

## 9. Compute estimate for full Stage 1

Per-branch wall time: the quick check at load ≈ 20 gave lift probe 1.4 s (5.4 s sim), CFD carry replay 1.6 s, probe + carry 3.0 s;
the full smoke run at load ≈ 25 (14 unrelated simulations plus the active-probe analysis) gave 6.3 s / 5.7 s / 13.9 s means, i.e.
≈ 4× the table below; the CFD sweep's own mean was 0.96 s per carry on an idle machine (≈ 0.7× the table). The table uses the
load-20 numbers; multiply by 4 while the machine stays this busy. MuJoCo is single-threaded; the Ryzen
7700X (8 cores / 16 threads) ran the CFD sweep at 12 processes. **No GPU is used in Stage 1**; the RTX 5080 and the A100 only
matter for Stage 2 training.

| Item | Branches | Wall per branch | Core-hours | Wall at 12 processes |
|---|---|---|---|---|
| probe tables, 6 single probes, 48 objects × R = 5 states | 1 440 | 1.4 s | 0.6 | 3 min |
| chained K = 2 probes (C3 style), 36 chains | 8 640 | 2.8 s | 6.7 | 35 min |
| C2/C3 follow-up carries for the released actions (192 contexts × 5 states × 4 decision makers × 2 conditions) | 7 680 | 3.0 s | 6.4 | 32 min |
| C0/C1 follow-ups | 0 | — | 0 | stored CFD branches |
| optional: full post-probe grid (exact post-probe feasible sets), 6 probes × 48 cells | 69 120 | 3.0 s | 57.6 | 4.8 h |
| optional: R = 10 (5 new states per object: approaches + released actions) | ≈ 240 + 7 700 | 5 s / 1.6 s | 3.8 | 20 min |
| **minimal Stage 1 (rows 1–3)** | 17 760 | | **≈ 14** | **≈ 1.2 h** |
| **with the optional rows** | ≈ 95 000 | | **≈ 75** | **≈ 6 h** |

Storage: ≈ 0.3 MB per branch compressed float32 → 5 GB minimal, ≈ 30 GB with the full post-probe grid (store labels only there).
Full Stage 1 fits comfortably on the local machine in one day of wall time even at the current load; it does not need the A100.

## 10. What is reused from CFD, unchanged

`envs/feasibility_carry` (env, grasp/carry primitives, save/restore), `labels.py` v3 (slip, damage proxies, completed; applied to
probe records and to post-probe carries), `feasible.context_tables` / `margin_grid` (feasible sets, teacher rule), `sweep.run_branch`
(C1 verification), `data/sweep_v1` (stored branch states = the contexts; grid branches = the C0/C1 outcomes; probes.json = the
static slip crossings used in the preview), `data/objects_v1.json` (properties, f_tol per class, splits). Nothing in these was
modified; `probing/` only imports them.

## 11. Open questions for the owner (collected)

| # | Question | Options (recommendation first) |
|---|---|---|
| Q1 | prior set Θ0 | E (ε-noisy coarse text, fragility trusted); A (population) as stress; B, C, D as in §7 |
| Q2 | risk standard | W primary + P (δ = δ_p = 0.05) as sensitivity; or P primary |
| Q3 | squeeze | keep as uninformative control; make damage observable; replace by a lateral-pulse friction probe |
| Q4 | observation tier | 1 + 2 (load channel, aligned with `active_probe_gates.py`); or tier 1 strictly |
| Q5 | repeats | R = 5 reusing the CFD states; R = 10; is the 3 % friction scale nuisance (yes, as CFD) or part of θ |
| Q6 | probe parameters | grid-aligned forces 7.29 / 10.63 / 15.49 N, h = 1 cm, 0.5 N margin; or round 8 / 12 / 16 N (16 N exceeds f_tol = 16.1 N of light_slippery_balanced_0 with the sustained-peak overshoot) |
| Q7 | interaction budget K | 1 (the preview shows no gain from 2) or 2 |
| Q8 | damage observability and drops | damage events invisible, drops from h_p harmless (frozen labeler); or add a visible-damage channel / drop-damage rule |
| Q9 | "resolvable" under stochastic outcomes | worst case over the repeat support; or probability ≥ 1 − δ |
| Q10 | context unit | (object, fragility, urgency) × 5 states as in CFD |
| Q11 | task return for Gate 1 | P̂_success of the released action, refusal 0; or with a dangerous-release penalty |
| Q12 | C1 "told explicitly" | input flag for Stage 2 only (no effect on the Stage-1 decision makers) |
| Q13 | CFD note | `refine.py` re-uses stored states without the per-repeat friction scale (§2); owner's call whether that matters for CFD |
