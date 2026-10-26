# Checkpoints

Large files live in the HuggingFace cache (`~/.cache/huggingface/hub`), never in the repo. Revisions are pinned in
`configs/pi05_libero.yaml`.

| Id | Revision | Model | Origin | Licence | Size | Downloaded |
|---|---|---|---|---|---|---|
| `lerobot/pi05_libero_base` | `a217bfd3` | π0.5, LIBERO | LeRobot conversion of Physical Intelligence `gs://openpi-assets/checkpoints/pi05_libero` (30k steps from `pi05_base` on `physical-intelligence/libero`) | Gemma Terms of Use (code Apache-2.0) | 14.5 GB fp32 | 2026-09-24 |
| `lerobot/pi05_libero_finetuned_v044` | `8e174154` | π0.5, LIBERO | `pi05_libero_base` + 6k steps, bf16, bs 256, on `HuggingFaceVLA/libero`; LeRobot's documented 97.5 % avg | Gemma Terms of Use | 7.5 GB bf16 | 2026-09-24 |

Not downloaded (Month 1 is π0.5 only): `lerobot/pi0_libero_base`, `lerobot/pi0_libero_finetuned_v044` (π0, SimpleVLA-Guard backbone; fetch when reproducing the guard).

Tokenizer: `google/paligemma-3b-pt-224` (gated; accepted on the project HF account).

## Month-1 selection

**Status 2026-09-24:** both checkpoints load and run on the RTX 5080 (bf16, `compile_model=false`):
4.14 B params, 8.71 GiB weights, 9.53 GiB peak, ~165 ms per 50-step chunk (10 flow steps) at batch 1;
forward hooks on `gemma_expert.model.layers[i]` and `action_out_proj` fire once per flow step (10 per chunk);
pre-velocity tensor is (B, 50, 1024) as in RedVLA appendix E.

LIBERO smoke test (libero_spatial tasks 0–2, 1 episode each, seed 1000, n_action_steps=10) is **blocked** on the
gated `google/paligemma-3b-pt-224` tokenizer (HF account `misaka1006` has not accepted it). Selection deferred until
the smoke test runs. Prior: `pi05_libero_finetuned_v044` (documented 97.5 % avg on LeRobot's eval stack); `base` is
the PI-original conversion and the closer match to RedVLA's checkpoint lineage.
