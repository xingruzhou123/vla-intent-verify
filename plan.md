# Plan — Phase 0: novelty and necessity check for the semantic-obligations direction

**Branch:** `semantic-obligations` · **Owner:** Grace (UT Austin) · **Plan date:** 2026-11-03
**Brief (source of truth for this phase):** `notes/new_direction_brief.md` (owner's text, verbatim).
**Previous project:** closed 2026-11-03 — `notes/phase1_postmortem.md`, `notes/decision_active_probe.md`;
archived plans in `docs/archive/` (`plan_feasibility_v1.md`, `plan_intent_evidence_v0.1.md`). Nothing in them is active.

## Phase 0 — novelty and necessity check

**No implementation, no data collection, no fine-tuning.** Phase 0 is a paper study and a gate design only.

Candidate question (from the brief): can a VLA turn open-ended task semantics into explicit safety obligations, judge
whether the scene evidence suffices to verify them, ask a targeted question when it does not, and then generate
continuous actions that satisfy the obligations?

Deliverables, in the order the brief lists them (all in `notes/`, measured or cited facts only):

1. `notes/novelty_matrix.md` — structured novelty matrix over the nearest work: DeliGrasp-style semantics → physical
   parameters; ASIMOV / HazardArena semantic safety benchmarks; semantic constraint extraction and language-conditioned
   motion planning; KnowNo / CoFineLLM and later targeted clarification; VLA safety monitors, latent guards, action
   verifiers; CLEAR itself.
2. Which claims of the candidate are already covered (same file, explicit per-claim column).
3. 2–3 concrete tasks with real action–semantic coupling (obligation bears on the action the policy is about to
   produce; satisfying it trades off several continuous action dimensions; not fully compilable to fixed symbolic rules).
4. For each task, a strong modular baseline: VLM obligation extraction + rule/physics verifier, and VLM + motion optimizer.
5. A low-cost Gate 0 that compares at least: (1) VLM + rule/physics verifier, (2) VLM + strong motion optimizer,
   (3) VLA without explicit obligations, (4) joint obligation-conditioned VLA, (5) oracle obligation + optimizer.
   Gate 0 does not ask for a proof that only a VLA can do the task; it asks for a measured gain, under fair baselines,
   from accessing the action representation or jointly generating the continuous action that the modular pipeline
   does not have. Pre-register the pass criterion before anything is run.
6. If no task can pass Gate 0, recommend closing the direction. Do not construct a problem in order to use a VLA.

Exit of Phase 0: owner decision on (a) whether a Gate-0 task exists, (b) whether to run Gate 0. Only then is a
four-month plan written.

## Boundaries carried over from the closed project

- Bounded lesson, not a general one: for safety problems governed by low-dimensional rigid-body dynamics, check
  observability, analytic solvability and the simple-model ceiling before adding model capacity.
- Reusable assets: MuJoCo env (`envs/feasibility_carry/`), same-state counterfactual branching, two-implementation label
  cross-check, analytic baselines, information-conditioned performance estimates, nested evaluation, gate-before-training
  workflow. See `status.md` §2.
