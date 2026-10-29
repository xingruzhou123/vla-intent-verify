"""Pick-and-carry on the robosuite/MuJoCo Panda with a force-limited grasp (F) and a time-scaled carry (s).

Phase 1 of plan.md (§6). Simulator ground truth is used only for labels and oracle settings.
"""
from .env import FeasibilityCarry, CarryConfig  # noqa: F401
