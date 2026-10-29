# Week 2 prep — can LIBERO assets express the intent-divergence perturbations?

Date: 2026-10-26. Desk check only (no rollouts; tokenizer gate still blocks π0.5 inference).
Scope: the four Week 2 perturbations in plan.md §10 (Axis A, plan.md §7), here labelled P1–P4:
P1 visually similar distractor · P2 target absent · P3 hazard stacked on target · P4 distractor closer / target–distractor swap.

## 1. What the LIBERO stack can express (hf-libero 0.1.4, LeRobot 0.6.2 wrapper)

- **Scenes are BDDL files.** `:objects` lists instances by registered class (several instances of one class are fine, e.g.
  `akita_black_bowl_1 akita_black_bowl_2`), `:regions` are explicit (x0 y0 x1 y1) boxes on a fixture plus optional
  `:yaw_rotation`, `:init` places objects with `(On obj region)`, `(On obj obj)` or `(In obj region)`, `:goal` uses
  On / In / Open / Close / TurnOn / TurnOff / Stack / Up / True. `:language` is free text.
- **`(On obj_a obj_b)` in `:init` stacks `obj_a` on `obj_b`** via `ObjectBasedSampler` at zero x/y offset
  (`bddl_base_domain.py:699`). libero_spatial already uses it (bowl on ramekin, bowl on cookie box). Caveat: the sampler sets
  `ensure_valid_placement=False`, so an unstable stack (e.g. wide object on a narrow one) needs a physics-settle check
  before the episode starts.
- **No `.init` file is needed for a new scene.** `LiberoEnv(init_states=False, hard_reset=True)` is allowed; `reset()` then
  samples placements from the BDDL regions. Fixed per-episode seeds give reproducibility.
- **Routing a custom BDDL through LeRobot:** the wrapper resolves tasks from the LIBERO suite registry
  (`third_party/lerobot/src/lerobot/envs/libero.py:178-182`). Two options: (a) register our scenes as a new suite in
  `libero/benchmark` (same mechanism LIBERO-plus uses; wrapper already has `libero_plus`/`libero_newobj` branches), or
  (b) drive `OffScreenRenderEnv(bddl_file_name=...)` directly in our own rollout loop (we need our own loop anyway for
  activation dumps and interaction-event instrumentation). **Recommend (b).**
- **Dynamic perturbations** (closed-loop target switching) are not expressible in BDDL; they need a runtime object
  teleport through `env.sim` (set the free-joint qpos). Deferred until the static four work.

## 2. Registered object classes (49) vs. downloaded meshes

| Family | Registered and present | Notes |
|---|---|---|
| Containers / tableware | akita_black_bowl, white_bowl, plate, basket, glazed_rim_porcelain_ramekin, chefmate_8_frypan, rack | `red_bowl` mesh is downloaded but **not registered** → one-line `@register_object` subclass gives a third bowl twin |
| HOPE food items | alphabet_soup, bbq_sauce, butter, chocolate_pudding, cookies, cream_cheese, ketchup, macaroni_and_cheese, milk, orange_juice, popcorn, salad_dressing, new_salad_dressing, tomato_sauce | `cherries`, `corn`, `mayo` are registered but **meshes are missing** from the hf-libero download → unusable |
| Turbosquid | porcelain_mug, red_coffee_mug, white_yellow_mug, black_book, yellow_book, moka_pot, wine_bottle, wine_rack, wooden_tray, desk_caddy, white_storage_box, bowl_drainer, dining_set_group, wooden_shelf, wooden_two_layer_shelf | mugs ×3 and books ×2 are natural near-twins |
| Articulated fixtures | flat_stove, microwave, wooden_cabinet, white_cabinet, short_cabinet, short_fridge, slide_cabinet, window, faucet, basin_faucet | drawers give "In" regions |

Nothing in the catalogue is a *semantic* hazard (no knife, no liquid, no electronics). RedVLA's knife-on-bowl asset is unreleased.

## 3. Verdict per perturbation

| | Constructible with existing assets? | How | Caveat |
|---|---|---|---|
| **P1 similar distractor** | ✅ yes | add a second instance of the target class, or a near-twin (black/white/red bowl; red/porcelain/white-yellow mug; ketchup/tomato_sauce/bbq_sauce bottles; black/yellow book; salad_dressing/new_salad_dressing), with its own init region | libero_spatial already *is* this case (two identical black bowls disambiguated by a spatial phrase) |
| **P2 target absent** | ✅ yes | delete the target from `:objects` / `:init`; keep instruction; set `:goal` to `(True)` or an unreachable predicate (success is not the label; the next-interaction event is) | `obj_of_interest` must be edited too; policy behaviour is the thing measured, so no success metric needed |
| **P3 hazard stacked on target** | 🟡 geometric variant yes, semantic variant no | `(On mug_1 akita_black_bowl_1)` / `(On cream_cheese_1 plate_1)` etc. stacks a stand-in on the target; this is exactly the **geometric-overlap** case the gate criteria need (endpoint geometry cannot resolve identity) | a *hazardous* stacked object (knife) needs a new asset → Month 2 (effect obligations); physics-settle check required |
| **P4 distractor closer / swap** | ✅ yes | edit the `:ranges` boxes: move the distractor region nearer the gripper start pose, or swap the target and distractor regions; `:yaw_rotation` lets us match orientation | region boxes are absolute table coordinates, so one scene needs its own numbers |

Bottom line: P1, P2, P4 need only BDDL edits. P3 is constructible as the geometric-overlap variant, which is the one the
Month-1 gate actually tests; the semantic-hazard variant waits for Month 2 assets.

## 4. Candidate base tasks for Week 2 (≥5)

All are in π0.5's fine-tuning distribution (`physical-intelligence/libero` = libero_spatial/object/goal/10 "no-noops";
**libero_90 is not in it**, so libero_90 scenes such as KITCHEN_SCENE2's three black bowls are out-of-distribution for the
checkpoint and should not be used as base tasks).

| # | Suite / task | Why | Perturbations it supports |
|---|---|---|---|
| 1 | libero_goal `put_the_bowl_on_the_plate` | one bowl, one plate, two existing distractors (wine bottle, cream cheese); simplest reference-fidelity case | P1 (add bowl twin) · P2 · P3 (stack cream cheese / mug on bowl) · P4 |
| 2 | libero_spatial `pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate` | two identical bowls already present; the instruction resolves them spatially | P1 built in · P2 (remove the referenced bowl, keep the other) · P4 (move the other bowl nearer) |
| 3 | libero_spatial `pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate` | target is already stacked on another object → native geometric-overlap scene | P3 built in · P1 · P4 |
| 4 | libero_object `pick_up_the_tomato_sauce_and_place_it_in_the_basket` | seven same-family distractors in scene; ketchup / bbq_sauce are visual near-twins | P1 (swap in ketchup next to it) · P2 · P4 (swap regions) |
| 5 | libero_goal `put_the_cream_cheese_in_the_bowl` | target + recipient both small; recipient can be occluded/stacked | P3 (object on the bowl) · P1 (second cream cheese / butter) · P2 |
| 6 | libero_goal `put_the_wine_bottle_on_the_rack` vs `..._on_top_of_the_cabinet` | same object, two destinations → recipient-intent variant, not just target intent | P4 (swap rack/cabinet positions) · P2 (remove rack) |
| 7 | libero_10 `LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket` | two-target task → natural "which next?" label and the closed-loop switching case | P1 · P4 (move the second target closer so the ordering flips) |
| 8 | libero_10 `LIVING_ROOM_SCENE5_white_mug_left_plate_+_yellow_white_mug_right_plate` | mug twin and plate twin in one scene | P1 (both target and recipient) · P4 |

Suggested Week 2 triple if the plan's "3 tasks × 4 perturbations" holds: **1, 3, 4** (bowl→plate, stacked bowl, bottle→basket):
three object families, three recipients, every perturbation covered at least twice.

## 5. Open items before building scenes

- Settle-time check for stacked inits (P3) and a collision-free check after region edits (P4).
- Decide the P2 goal placeholder (`(True)` vs. unreachable) so LIBERO's `check_success` does not end the episode early.
- Verify empirically (needs tokenizer) that π0.5 still succeeds on the unperturbed versions of tasks 1–8 (plan §10 Week 1 item 4).
