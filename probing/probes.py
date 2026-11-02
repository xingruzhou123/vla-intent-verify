"""Probe actions and branch mechanics for the safe-probing study (probing/design.md).

Everything here drives the CFD environment through its public primitives (`grasp`, `move_minjerk`, `hold`, `release`,
`save_state`, `restore_state`) and tags phases with the env's own PH codes so that the frozen CFD labeler
(`labels.relative_motion`, `labels.damage_scores`, `labels.all_labels`) can be applied to probe records unchanged.
The CFD env, labeler and data are imported, never modified.

Probe set (design.md §1):
  trial lift   grasp with force bound F_p, lift h_p at a quasi-static speed, hold, lower back, settle, release
  light squeeze grasp with force bound F_p (ramp + hold), release — no motion
Branching (design.md §2): a probe branch always starts with `env.restore_state(saved)` on a stored CFD branch state.
C1 = probe branch, restore, follow-up action from the original state (= the stored CFD grid branch).
C3 = probe, no restore, follow-up action from the post-probe state (`carry_after`).
"""
import hashlib, time
from dataclasses import dataclass, asdict
import numpy as np
from envs.feasibility_carry.env import FeasibilityCarry, CarryConfig, GRASP_MAT, PH
from envs.feasibility_carry import labels

# ------------------------------------------------------------------------------------------------------ probe specs
@dataclass
class ProbeSpec:
    kind: str                 # "lift" | "squeeze"
    F: float                  # N, actuator force bound during the probe (same mechanism as the CFD grasp)
    h: float = 0.01           # m, lift height (lift only)
    t_lift: float = 0.75      # s, min-jerk lift duration (peak accel 5.77 h / t² ≈ 0.1 m/s² for 1 cm / 0.75 s)
    t_top: float = 0.5        # s, hold at the top (creep / slow slip detection window)
    t_lower: float = 0.75     # s, lower back to the grasp height
    t_settle: float = 0.25    # s, settle before release
    name: str = None

    def label(self):
        if self.name:
            return self.name
        return f"lift_F{self.F:g}_h{self.h * 100:g}cm" if self.kind == "lift" else f"squeeze_F{self.F:g}"


# the v0 probe set used by the smoke test (design.md §1; the owner decides the final set)
# forces = CFD grid levels F_idx 1, 2, 3 (7.29, 10.63, 15.49 N): a lift at grid level i tests the static part of the slip
# boundary exactly at that level; 15.49 N is the highest level that stays below every fragile tolerance (≥ 16.0 N) with the
# ≈ 0.3 N overshoot of the sustained 50 ms peak over the commanded bound (measured in the smoke test)
SMOKE_PROBES = [ProbeSpec("lift", 7.29), ProbeSpec("lift", 10.628), ProbeSpec("lift", 15.494), ProbeSpec("lift", 15.494, h=0.02),
                ProbeSpec("squeeze", 10.628), ProbeSpec("squeeze", 15.494)]

# sensor model for the observation channels a decision maker can see (design.md §4); all values are assumptions
SENSOR = {
    "rgb_pose_sd_m": 0.002,          # per-frame object-pose error from the cameras (relative to the gripper)
    "rgb_rate_hz": 20.0,             # frames averaged over the top-hold window
    "grip_force_sd_N": 0.5,          # finger (normal) force sensor
    "finger_pos_sd_m": 0.0002,       # finger joint encoder
    "wrist_load_sd_N": 0.5,          # tier 2: supported load from the arm's external-force estimate, noise after averaging
    "wrist_load_bias_sd_N": 0.5,     # tier 2: per-episode bias of that estimate
}


# -------------------------------------------------------------------------------------------- state fingerprints
def model_fingerprint(env):
    """Hash of the model-level fields that env code mutates (force bound, friction scale, CoM offset, contact)."""
    m = env.sim.model._model
    h = hashlib.sha256()
    for a in (m.actuator_forcerange, m.actuator_ctrlrange, m.geom_friction, m.geom_solref, m.geom_priority,
              m.body_ipos, m.body_mass, m.body_sameframe, m.body_simple):
        h.update(np.ascontiguousarray(a).tobytes())
    h.update(np.float64(m.opt.timestep).tobytes())
    return h.hexdigest()


def state_fingerprint(env):
    """Hash of everything the branch restores: the mjSTATE_INTEGRATION buffer (time, qpos, qvel, act, ctrl, warm-start,
    user fields), the env-level grasp state and phase, and the model fingerprint."""
    s = env.save_state()
    h = hashlib.sha256(np.ascontiguousarray(s["state"], dtype=np.float64).tobytes())
    h.update(f"{s['timestep']}|{s['cur_time']:.15g}|{s['grip_mode']}|{s['grip_F']:.15g}|{env._phase}".encode())
    h.update(model_fingerprint(env).encode())
    return h.hexdigest()


def record_hash(arr):
    return hashlib.sha256(np.ascontiguousarray(arr, dtype=np.float64).tobytes()).hexdigest()


def load_state(path):
    z = np.load(path, allow_pickle=True)
    return {k: z[k] if k == "state" else z[k].item() for k in z.files}


def make_env(obj, seed=0):
    """Env for one object of data/objects_<v>.json, built exactly as the CFD sweep driver builds it."""
    cfg = CarryConfig(obj_mass=float(obj["mass"]), obj_mu=float(obj["mu"]), com_offset_frac=float(obj["com_frac"]))
    env = FeasibilityCarry(cfg, seed=seed)
    env.reset()
    return env


# ---------------------------------------------------------------------------------------------------- execution
def run_probe(env, saved, spec: ProbeSpec, rec=None):
    """Restore the branch state and execute one probe. Returns (info, record rows of this probe, segments)."""
    env.restore_state(saved)
    return execute_probe(env, spec, rec)


def execute_probe(env, spec: ProbeSpec, rec=None):
    """Execute one probe from the *current* state (used for chaining probes and for C3). Phases are tagged with the
    env's PH codes: ramp/hold (grasp), lift, settle (top hold), lower, settle, release."""
    rec = [] if rec is None else rec
    n0 = len(rec)
    seg = {}
    t0 = time.perf_counter()
    gs = env.grasp(spec.F, rec)
    seg["grasp"] = (n0, len(rec))
    pos, _ = env.eef_pose()
    mat = GRASP_MAT
    if spec.kind == "lift":
        top = pos + np.array([0.0, 0.0, spec.h])
        a = len(rec); env._phase = PH["lift"];   env.move_minjerk(pos, top, mat, spec.t_lift, rec);   seg["lift"] = (a, len(rec))
        a = len(rec); env._phase = PH["settle"]; env.hold(top, mat, spec.t_top, rec);                 seg["top"] = (a, len(rec))
        a = len(rec); env._phase = PH["lower"];  env.move_minjerk(top, pos, mat, spec.t_lower, rec);  seg["lower"] = (a, len(rec))
        a = len(rec); env._phase = PH["settle"]; env.hold(pos, mat, spec.t_settle, rec);              seg["settle"] = (a, len(rec))
    elif spec.kind != "squeeze":
        raise ValueError(spec.kind)
    a = len(rec); env.release(pos, mat, rec); seg["release"] = (a, len(rec))
    wall = time.perf_counter() - t0
    arr = np.array(rec[n0:])
    seg = {k: (v[0] - n0, v[1] - n0) for k, v in seg.items()}
    info = {"probe": spec.label(), "spec": asdict(spec), "wall_s": wall, "sim_s": float(arr[-1, 0] - arr[0, 0]),
            "n_rows": int(len(arr)), "grasp": gs}
    return info, arr, seg


def carry_after(env, F, s, T_max_levels, fragility, rec=None):
    """Follow-up grasp-and-carry from the current (post-probe) state, labelled with the frozen CFD labeler on the carry
    segment alone (C2/C3). Returns (info, record rows of the carry)."""
    rec = [] if rec is None else rec
    n0 = len(rec)
    t0 = time.perf_counter()
    gs = env.grasp(F, rec)
    pos, mat = env.carry(s, rec)
    env.release(pos, mat, rec)
    wall = time.perf_counter() - t0
    arr = np.array(rec[n0:])
    lab = labels.all_labels(arr, env.SAMPLE_COLS, env.goal_pos(), env.cfg.pad_half, env.cfg.T_carry(s), T_max_levels,
                            fragility=fragility)
    return {"F": float(F), "s": float(s), "wall_s": wall, "sim_s": float(arr[-1, 0] - arr[0, 0]), **gs, **lab}, arr


# ----------------------------------------------------------------------------------------------------- outcomes
def probe_outcome(arr, cols, seg, spec: ProbeSpec, pad_half, fragility, delta_slip=labels.DELTA_SLIP):
    """Ground-truth outcome summary of one probe branch (what the simulator knows; the observation model in
    `observe` decides what a decision maker sees)."""
    R = labels.Rec(arr, cols)
    out = {"probe": spec.label()}
    sc = labels.damage_scores(R, fragility)                       # over ramp/hold/lift/lower/settle (never the release)
    out["score_N50"] = float(sc["N"]["50ms"]); out["peak_N"] = float(sc["peak_N"])
    out["damage"] = {name: bool(labels.damage_labels(sc, f)["damage_primary"]) for name, f in fragility.items()}
    g0, g1 = seg["grasp"]
    w = slice(max(g0, g1 - int(round(0.5 / R.dt))), g1)             # steady-state window of the grasp hold
    out["grip_f_left"] = float(R["f_left"][w].mean()); out["grip_f_right"] = float(R["f_right"][w].mean())
    out["finger_q1"] = float(R["q1"][w].mean()); out["finger_q2"] = float(R["q2"][w].mean())
    out["on_table_end"] = bool(R["on_table"][-1] > 0.5)
    if spec.kind == "lift":
        motion = labels.relative_motion(R)                        # reference: last 0.25 s of the grasp hold
        l0, l1 = seg["lift"]; t0_, t1_ = seg["top"]
        trans = motion["trans"][l0:t1_]
        z_ref = R["obj_z"][motion["ref_idx"][0]:motion["ref_idx"][1] + 1].mean()
        z_top = R["obj_z"][t0_:t1_].mean()
        ftl = R.vec("ft_left_", "yz")[t0_:t1_]; ftr = R.vec("ft_right_", "yz")[t0_:t1_]
        load = np.linalg.norm(ftl + ftr, axis=1).mean()               # tangential load carried by the two pads at the top
        out.update({"slip_trans_max_m": float(trans.max()), "z_gain_top_m": float(z_top - z_ref),
                    "on_table_top": float(R["on_table"][t0_:t1_].mean()),
                    "held": bool(trans.max() < delta_slip), "pad_load_top_N": float(load),
                    "torsion_max_deg": float(np.degrees(np.abs(motion["torsion"][l0:t1_]).max())),
                    "eef_z_gain_m": float(R["eef_z"][t0_:t1_].mean() - R["eef_z"][motion["ref_idx"][0]:motion["ref_idx"][1] + 1].mean()),
                    "obj_shift_after_m": float(np.linalg.norm(R.vec("obj_", "xy")[-1] - R.vec("obj_", "xy")[0]))})
    else:
        out.update({"obj_shift_after_m": float(np.linalg.norm(R.vec("obj_", "xy")[-1] - R.vec("obj_", "xy")[0]))})
    return out


def observe(outcome, spec: ProbeSpec, rng, sensor=SENSOR, tier=2):
    """Noisy observation of one probe outcome (design.md §4). Tier 1: cameras + proprioception + finger force.
    Tier 2 adds the supported-load estimate from the arm (wrist F/T or joint-torque based)."""
    n_frames = max(1, int(round(spec.t_top * sensor["rgb_rate_hz"])))
    o = {"probe": spec.label(), "F_cmd": spec.F,
         "grip_f_hat": float(0.5 * (outcome["grip_f_left"] + outcome["grip_f_right"]) + rng.normal(0, sensor["grip_force_sd_N"])),
         "finger_q_hat": float(outcome["finger_q1"] + rng.normal(0, sensor["finger_pos_sd_m"]))}
    if spec.kind == "lift":
        sd = sensor["rgb_pose_sd_m"] / np.sqrt(n_frames)
        o["slip_hat_m"] = float(abs(outcome["slip_trans_max_m"] + rng.normal(0, sd)))
        o["z_gain_hat_m"] = float(outcome["z_gain_top_m"] + rng.normal(0, sd))
        o["held_hat"] = bool(o["z_gain_hat_m"] > 0.5 * spec.h and o["slip_hat_m"] < labels.DELTA_SLIP)
        if tier >= 2:
            o["load_hat_N"] = float(outcome["pad_load_top_N"] + rng.normal(0, sensor["wrist_load_bias_sd_N"])
                                    + rng.normal(0, sensor["wrist_load_sd_N"]))
    return o
