# Project Summary

## Core idea
Existing latent guards (SAFE, SimpleVLA-Guard) predict whether a rollout resembles previously seen failures and output one risk score. We ask whether the VLA's **actual policy intent**, read from its internal action representation, is **licensed by the scene evidence** required to execute it safely. If not, the action chunk is blocked before dispatch, with a typed reason.

We separate three quantities that prior guards conflate:
- **Commanded intent** \(I^{\mathrm{cmd}}\): what the instruction asks for.
- **Policy intent** \(I^{\mathrm{policy}}\): the interaction the policy is currently forming (what we read from latents).
- **Realized outcome** \(Y^{\mathrm{outcome}}\): what physically happens.

## Central research question
When does a VLA's internal representation reveal its actual interaction intent beyond what can be inferred from the instruction or the decoded action chunk, and can that incremental information support evidence-grounded safety decisions before execution?

## Contributions (conditional on Month-1 gate)
1. **Policy-intent extraction.** Read the next interaction target (later: verb, recipient) from π0 internals; characterize when it becomes decodable across layer × flow step × chunk. Kept only if it adds information beyond instruction and decoded chunk.
2. **Obligation-factorized verification.** Intent → obligation → evidence → compatibility, over three obligations:
   - Reference fidelity — does the policy act on what the instruction refers to?
   - Execution precondition — does the scene state permit the operation?
   - Effect compatibility — is the predicted effect permissible here?
   Taxonomy is by *required evidence type*, orthogonal to RedVLA (temporal structure) and HazardArena (harm type). Joint conformal calibration; REOBSERVE for insufficient evidence.
3. **Counterfactual evidence-faithfulness evaluation.** Intent-divergence axis × permission-divergence axis, plus controlled evidence interventions testing whether the monitor relies on the right evidence.
4. **Real-robot dispatch enforcement.** Zero actuator commands after BLOCK; measured latency and lead time.

## Where latent intent could add value (need ≥ 1)
1. Earlier intent — decodable chunks before the decoded chunk reveals it.
2. Partial-trajectory ambiguity — pour vs. place before geometric divergence.
3. Closed-loop target switching — latent turns before chunks do.
4. Geometric overlap — stacked/nested/adjacent objects.

## Experiment plan
- **Month 1 (go/no-go, RTX 5080):** intent readers — instruction, chunk-endpoint, chunk-rollout (perceived poses), latent probes, oracle — on intent-divergence scenes with future-oriented labels; reproduce SimpleVLA-Guard. Verdict: green / yellow / red.
- **Month 2:** full monitor in simulation, permission-divergence twins, baselines and ablations.
- **Month 3:** generalization splits, onset analysis, OpenVLA as second backbone, real-robot setup.
- **Month 4:** 120–180 real-robot trials, writing.

## Venue
Green → IEEE T-RL (option: CoRL/RSS first). Yellow → T-RL or RA-L. Red → RA-L / ICRA / IROS.
