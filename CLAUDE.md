# CLAUDE.md
Research project: evidence-grounded verification of VLA policy intent. Read plan.md for the full plan; status.md for current state.
Background: docs/summary.md, docs/related_work.md. Paper PDFs in docs/papers/.

Rules:
- Current phase: Month 1 go/no-go (plan.md §10). Do not build Month 2+ components.
- Privileged simulator state is for labels and oracle baselines only, never a monitor input.
- Intent labels are future-oriented (next realized interaction event), see plan.md §6.1.
- Large artifacts (latent dumps, rollouts, checkpoints) go in data/ or results/, never committed.
- External repos (openpi, LIBERO, RedVLA, HazardArena) are cloned into third_party/; do not edit them in place — wrap or patch.
- Use a dedicated venv for this project (not physhall-safety's).
- After finishing a task, update status.md (snapshot table, weekly log, blockers).
- GPU: RTX 5080 (16 GB, Blackwell sm_120). Prefer PyTorch implementations; ask before starting anything that needs fine-tuning.
