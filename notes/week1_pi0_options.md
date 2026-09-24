# Week 1 items 2–3: PyTorch π0 implementation and LIBERO checkpoints (2026-09-24)

Inspection only; nothing installed. Repos shallow-cloned into `third_party/openpi` (HEAD 215abfb) and
`third_party/lerobot` (HEAD e624f3f, 2026-09-24). Line numbers refer to those checkouts.

## 1. openpi PyTorch port vs. LeRobot pi0/pi05

| Criterion | openpi (PyTorch port) | LeRobot `pi0` / `pi05` |
|---|---|---|
| Python | `>=3.11` | `>=3.12` (pyproject L32) — **our venv is 3.11; needs recreating with uv's 3.12.12** |
| torch pin | `torch==2.7.1` hard pin (pyproject L29) → **would downgrade our 2.11.0+cu128**; 2.7.1 cu126 wheels have no sm_120 | `torch>=2.7,<2.12.0`, `torchvision>=0.22,<0.27` (L62-63), Linux source = cu128 index; uv.lock already resolves 2.11.0+cu128 / 0.26.0+cu128 → **no conflict** |
| JAX | hard deps: `jax[cuda12]==0.5.3`, `flax==0.10.2`, `orbax-checkpoint==0.11.13`; `models/model.py` imports jax/flax/orbax even on the PyTorch path; no torch-only extra | none (no jax/flax/orbax anywhere) |
| transformers | `==4.53.2`, plus copying `models_pytorch/transformers_replace/*` over site-packages (README L200-212); `PI0Pytorch.__init__` raises if the patch is missing | `>=5.4,<5.6` (transformers v5), subclasses `GemmaModel`/`PaliGemmaForConditionalGeneration` in `policies/pi_gemma.py`; no monkey-patching |
| Install | `uv sync` of the whole JAX+torch stack, git-pinned lerobot, tf-cpu optional; separate py3.8 client env for LIBERO (`examples/libero`: torch 1.11+cu113, robosuite 1.4.1, mujoco 3.2.3, LIBERO git submodule) via websocket policy server | `pip install -e ".[pi,libero]"`: pulls `hf-libero>=0.1.4` (PyPI package of LIBERO; lock resolves mujoco 3.8.1, robosuite 1.4.0), in-process `LiberoEnv` (`envs/libero.py`), `lerobot-eval` CLI, `MUJOCO_GL=egl` |
| LIBERO-fine-tuned π0 loads directly? | Only π0.5: `gs://openpi-assets/checkpoints/pi05_libero` (JAX/orbax) → convert with `examples/convert_jax_model_to_pytorch.py` (needs jax+orbax; assets-dir path bug at L536). **No fine-tuned π0 LIBERO checkpoint is hosted** (`pi0_libero` at config.py L653 is a training config from `pi0_base`). No pre-converted PyTorch checkpoints hosted by PI. | Yes: `PI0Policy.from_pretrained("lerobot/pi0_libero_finetuned_v044")`, `PI05Policy.from_pretrained("lerobot/pi05_libero_finetuned_v044")`; safetensors + config.json; `from_pretrained` also accepts openpi-PyTorch key layout via `_fix_pytorch_state_dict_keys` (modeling_pi0.py L865) |
| Flow-step loop | `PI0Pytorch.sample_actions(num_steps=10)`, Euler, t: 1→0, `denoise_step` per step (pi0_pytorch.py L376-462); `sample_actions` is wrapped in `torch.compile(mode="max-autotune")` by default (pi0_config.py L35) — must set `pytorch_compile_mode=None` to hook | `PI0Pytorch.sample_actions` → `common/flow_matching.py::euler_integrate` (L61), `config.num_inference_steps=10`, `denoise_step` per step (modeling_pi0.py L720-758; pi05 L882-919); no compile by default |
| Per-layer expert hidden states | `paligemma_with_expert.gemma_expert.model.layers[i]` (18 × `GemmaDecoderLayer`, width 1024); forward hooks fire at inference (suffix-only pass through HF `GemmaModel.forward`, KV-cached prefix); `output_hidden_states` exists in patched GemmaModel but the wrapper drops it | `policy.model.paligemma_with_expert.gemma_expert.model.layers[i]` (18 × PiGemma decoder layer, width 1024); forward hooks fire at inference (`PiGemmaModel.forward` loops the layers, pi_gemma.py L306-321); `output_hidden_states` supported by `PiGemmaModel` but not passed by the wrapper → use hooks |
| `pre_velocity` (SimpleVLA-Guard feature) | input to `model.action_out_proj` (Linear 1024→action_dim, pi0_pytorch.py L101/L462): (B, horizon, 1024), once per flow step | input to `model.action_out_proj` (Linear 1024→32, modeling_pi0.py L437/L758; pi05 L555/L919): (B, chunk=50, 1024), once per flow step. Same tensor as openpi. |
| Caveat (both) | Training-mode joint prefix+suffix forward uses `compute_layer_complete`, which calls submodules directly → layer-level hooks do not fire there; hook `layers[i].mlp` / `o_proj` if that path is ever needed. Not relevant for inference-time monitoring. | same |
| Reported LIBERO SR (π0.5) | 98.8 / 98.2 / 98.0 / 92.4, avg 96.85 (spatial/object/goal/10) | 97.0 / 99.0 / 98.0 / 96.0, avg 97.5 (pi05_libero_finetuned_v044, `n_action_steps=10`); no official π0 numbers; community: pi0_libero_finetuned ≈ 90% avg with `n_action_steps=10`, ≈ 65–75% with the stored default 50 (lerobot issues #2114, #4614) |
| Licence | code Apache-2.0; Gemma Terms of Use for weights (`LICENSE_GEMMA.txt`) | code Apache-2.0; HF cards tag weights "gemma" licence; gated `google/paligemma-3b-pt-224` tokenizer must be accepted (`hf auth login`) |

**Verdict:** LeRobot. openpi's PyTorch path cannot be installed without JAX and without replacing torch 2.11 with 2.7.1
(which has no sm_120 wheel), and it hosts no fine-tuned π0 LIBERO checkpoint. LeRobot is torch-native, already locks
2.11.0+cu128, ships both π0 and π0.5 LIBERO checkpoints in its own format, runs LIBERO in-process, and exposes the
same `action_out_proj` pre-velocity tensor and per-layer expert hooks. Cost: recreate `.venv` on Python 3.12
(uv-managed 3.12.12 already present), and transformers v5.

## 2. Public LIBERO-fine-tuned π0 / π0.5 checkpoints

| Checkpoint | Source / format | Notes | Licence |
|---|---|---|---|
| `gs://openpi-assets/checkpoints/pi05_libero` | Physical Intelligence, JAX/orbax | official; 30k steps from `pi05_base` on `physical-intelligence/libero`; the only PI-hosted LIBERO checkpoint (no π0 or π0-FAST LIBERO) | Apache-2.0 repo / Gemma ToU |
| `lerobot/pi05_libero_base` | HF, LeRobot safetensors (2025-09-09) | conversion of PI's `pi05_libero`; stores `n_action_steps=10`, `empty_cameras=1` | gemma |
| `lerobot/pi05_libero_finetuned_v044` (+ `_quantiles_v044`) | HF, LeRobot (2025-10-01) | `pi05_libero_base` + 6k steps bf16, bs 256 on `HuggingFaceVLA/libero`; **documented 97.5% avg** | gemma |
| `lerobot/pi05-libero` | HF, LeRobot (2026-02-20) | newer upload, no model card; provenance unknown — treat as unverified | gemma |
| `lerobot/pi0_libero_base` | HF, LeRobot (2025-09-09) | π0 LIBERO base; no card details | gemma |
| `lerobot/pi0_libero_finetuned_v044` | HF, LeRobot (2025-10-02) | base = `pi0_libero_base`, fine-tuned on `HuggingFaceVLA/libero`; no official SR; community ≈ 90% avg at `n_action_steps=10` | gemma |
| `lerobot/pi0fast-libero`, `lerobot/pi0fast-libero-v044` | HF | π0-FAST (autoregressive; not flow) — not useful for flow-step probes | gemma |
| `pepijn223/pi05_libero_6000`, `sunshk/pi05_libero_pytorch` | HF, third-party | openpi-PyTorch-format conversions of `pi05_libero`; unofficial | — |

**Which one RedVLA used.** Paper Table 11 (appendix E.1): π0 and π0.5 each use "1 unified (All suites)" checkpoint,
footnoted to `https://huggingface.co/physical-intelligence/`; chunk C=50, replan every 10 steps, 2 views, 8-d proprio,
20 Hz, H=520. The paper never names openpi/LeRobot or a checkpoint id. PI's HF org hosts datasets, not weights, so the
most consistent reading is PI's official `pi05_libero` (and a PI-provided π0 LIBERO checkpoint that is not publicly
listed) — i.e. the same weights `lerobot/pi05_libero_base` is converted from. SimpleVLA-Guard feature (appendix):
"pre velocity feature extracted from the π0 diffusion transformer", tensor (N_diff, N_horizon, 1024), diff idx 0,
horizon idx 0; LSTM-256, 1 layer, dropout 0.2, 1000 epochs Adam 5e-4 StepLR(300, 0.5), clip 1.0; 8 seen LIBERO-Long
tasks (640 rollouts, 70/30), 2 unseen tasks (80 safe + 80 unsafe).

**RedVLA code/assets status (2026-09-24):** project page https://redvla.github.io is a static page whose "Code" button is
`javascript:void(0)`; only link is the arXiv abstract. No GitHub repo, dataset, or SimpleVLA-Guard weights found.
Paper says release under CC BY-NC 4.0. → assume re-implementation from the appendix (plan.md Week 2 fallback).

## 3. Proposed install (pending go-ahead)

1. Recreate `.venv` with `uv venv --python 3.12`; reinstall `torch==2.11.0+cu128 torchvision==0.26.0+cu128` from the
   cu128 index first, then `uv pip install -e "third_party/lerobot[pi,libero]"` with a constraints file pinning torch
   and torchvision so resolution cannot replace them. Re-run `scripts/check_gpu.py`.
2. `hf auth login` after accepting the `google/paligemma-3b-pt-224` gate.
3. Download `lerobot/pi05_libero_finetuned_v044` (primary, matches RedVLA's π0.5 lineage and has documented SR) and
   `lerobot/pi0_libero_finetuned_v044` (π0, the SimpleVLA-Guard backbone); record in `models/checkpoints.md`.
4. Smoke test: `MUJOCO_GL=egl lerobot-eval --env.type=libero --env.task=libero_spatial --eval.n_episodes=2
   --policy.path=lerobot/pi05_libero_finetuned_v044 --policy.n_action_steps=10`.
