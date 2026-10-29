"""Outcome labels, damage proxies and realized kinematics (plan.md §3.2, §3.5, §3.7). Implementation A.

A second, independent implementation (gates G7/G8 in docs/data_spec.md) is written separately and must not import this file.

Inputs are the per-substep record `rec` (rows × SAMPLE_COLS of env.py). All quantities are computed from constraint
forces and poses, never from penetrations. Fragility (f_tol), the window t_win, κ and the deadline are label parameters:
they are applied post hoc and do not enter the simulation, so one physical branch yields labels for every setting.
"""
import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.spatial.transform import Rotation as Rot
from .env import PH

# frozen label settings (plan §3.2, §3.5, §6 P5)
DELTA_SLIP = 0.005            # m, translation in the pad plane or pad-edge displacement from torsion
DROP_DIST = 0.08              # m, object centre farther than this from the pad midpoint = left the gripper
T_WIN = {"50ms": 0.050, "100ms": 0.100}
T_WIN_PRIMARY = "50ms"
KAPPAS = {"k0.5": 0.5, "k1": 1.0, "k2": 2.0}
KAPPA_PRIMARY = "k1"
F_TOL_MULT = {"x0.75": 0.75, "x1": 1.0, "x1.25": 1.25}
FRAGILITY = {"fragile": 20.0, "rigid": 50.0}   # N
GOAL_XY_TOL, GOAL_Z_TOL = 0.03, 0.01          # m
CARRY_PHASES = (PH["lift"], PH["lateral"], PH["lower"], PH["settle"])
LOAD_PHASES = (PH["ramp"], PH["hold"]) + CARRY_PHASES   # damage is evaluated over grasp + carry (never the release)


class Rec:
    """Column access helper."""
    def __init__(self, rec, cols):
        self.a = np.asarray(rec)
        self.c = {n: i for i, n in enumerate(cols)}

    def __getitem__(self, name):
        return self.a[:, self.c[name]]

    def vec(self, prefix, comps):
        return self.a[:, [self.c[f"{prefix}{k}"] for k in comps]]

    @property
    def dt(self):
        t = self["t"]
        return float(np.median(np.diff(t)))


# ----------------------------------------------------------------------------------------------------------- slip
def relative_motion(R):
    """Object motion relative to the pads, referenced to the end of the grasp hold.
    Returns per-row translation in the pad plane (eef y, z) [m], torsion angle about the closing axis (eef x) [rad],
    the full relative displacement and the reference index range."""
    phase = R["phase"]
    hold = np.where(phase == PH["hold"])[0]
    if len(hold) == 0:
        raise ValueError("no hold phase in record")
    t = R["t"]
    ref_idx = hold[t[hold] >= t[hold[-1]] - 0.25]            # last 0.25 s of the hold
    relpad = R.vec("relpad_", "xyz")
    ref_relpad = relpad[ref_idx].mean(axis=0)
    eq = R.vec("eef_q", "wxyz")[:, [1, 2, 3, 0]]              # scipy wants xyzw
    oq = R.vec("obj_q", "wxyz")[:, [1, 2, 3, 0]]
    rel = Rot.from_quat(eq).inv() * Rot.from_quat(oq)        # object orientation in the eef frame
    ref = rel[ref_idx].mean()
    rv = (ref.inv() * rel).as_rotvec()
    ang = rv[:, 0]                                            # component about eef x = closing axis (pad normal)
    d = relpad - ref_relpad
    trans = np.linalg.norm(d[:, 1:3], axis=1)                 # pad plane = eef (y, z)
    return {"trans": trans, "torsion": ang, "disp": d, "ref_relpad": ref_relpad,
            "ref_idx": (int(ref_idx[0]), int(ref_idx[-1]))}


def slip(R, motion, pad_half, delta_slip=DELTA_SLIP, drop_dist=DROP_DIST, lost_contact_t=0.05):
    """Slip during the carry: pad-plane translation > δ_slip, or torsion × pad_half > δ_slip (pad-edge displacement),
    or the object left the gripper (centre farther than drop_dist from the pad midpoint, or both pads out of contact
    for ≥ lost_contact_t during lift/lateral/lower)."""
    phase = R["phase"]
    carry = np.isin(phase, CARRY_PHASES)
    trans = motion["trans"][carry]
    tors = np.abs(motion["torsion"][carry])
    relpad = R.vec("relpad_", "xyz")[carry]
    far = np.linalg.norm(relpad, axis=1) > drop_dist
    moving = np.isin(phase, CARRY_PHASES[:3])
    no_contact = ((R["nc_left"] + R["nc_right"]) == 0)[moving]
    n = max(1, int(round(lost_contact_t / R.dt)))
    lost = len(no_contact) >= n and bool(np.lib.stride_tricks.sliding_window_view(no_contact, n).all(axis=1).any())
    dropped = bool(far.any() or lost)
    edge = tors * pad_half
    return {"slip_trans_max": float(trans.max()), "slip_torsion_max_deg": float(np.degrees(tors.max())),
            "slip_edge_max": float(edge.max()), "dropped": dropped,
            "slip": bool(trans.max() > delta_slip or edge.max() > delta_slip or dropped)}


# --------------------------------------------------------------------------------------------------------- damage
def sustained_peak(x, dt, t_win):
    """max over t of min over [t, t + t_win] of x — the largest level held continuously for t_win."""
    n = max(1, int(round(t_win / dt)))
    if len(x) < n:
        return float(x.min())
    return float(np.lib.stride_tricks.sliding_window_view(x, n).min(axis=1).max())


def damage_scores(R, fragility=FRAGILITY):
    """Per-pad load series over grasp + carry and the sustained-peak scores for every (t_win, κ).
    Returns {"N": {t_win: score}, "L": {t_win: {κ: score}}, "impulse": fn(f_tol)}."""
    mask = np.isin(R["phase"], LOAD_PHASES)
    dt = R.dt
    N = np.stack([R["f_left"][mask], R["f_right"][mask]], axis=1)
    Tt = np.stack([np.linalg.norm(R.vec("ft_left_", "yz")[mask], axis=1),
                   np.linalg.norm(R.vec("ft_right_", "yz")[mask], axis=1)], axis=1)   # |net tangential| per pad
    out = {"N": {}, "L": {}, "peak_N": float(N.max())}
    for wn, tw in T_WIN.items():
        out["N"][wn] = max(sustained_peak(N[:, k], dt, tw) for k in (0, 1))
        out["L"][wn] = {}
        for kn, kappa in KAPPAS.items():
            L = N + kappa * np.abs(Tt)
            out["L"][wn][kn] = max(sustained_peak(L[:, k], dt, tw) for k in (0, 1))
    Nmax = N.max(axis=1)
    out["impulse"] = {name: float(np.sum(np.maximum(0.0, Nmax - f) * dt)) for name, f in fragility.items()}
    return out


def damage_labels(scores, f_tol):
    """All damage labels for one tolerance value: primary (N, 50 ms), shear (L, κ = 1, 50 ms) and the sensitivity grid."""
    sens = {}
    for mn, mult in F_TOL_MULT.items():
        for wn in T_WIN:
            sens[f"N|{wn}|{mn}"] = bool(scores["N"][wn] > f_tol * mult)
            for kn in KAPPAS:
                sens[f"L|{wn}|{kn}|{mn}"] = bool(scores["L"][wn][kn] > f_tol * mult)
    return {"damage_primary": sens[f"N|{T_WIN_PRIMARY}|x1"],
            "damage_shear": sens[f"L|{T_WIN_PRIMARY}|{KAPPA_PRIMARY}|x1"],
            "damage_sensitivity": sens}


# ----------------------------------------------------------------------------------------------- completion, time
def completed(R, goal_pos, xy_tol=GOAL_XY_TOL, z_tol=GOAL_Z_TOL):
    """Object rests in the goal region after release: xy within xy_tol of the goal, z within z_tol of the resting
    height, in contact with the table. Geometric only; damage and slip enter through physical_safe."""
    last = R.a[-1]
    op = last[[R.c["obj_x"], R.c["obj_y"], R.c["obj_z"]]]
    xy = float(np.linalg.norm(op[:2] - goal_pos[:2]))
    dz = float(op[2] - goal_pos[2])
    on_table = bool(last[R.c["on_table"]] > 0.5)
    return {"final_xy_err": xy, "final_z_err": dz,
            "completed": bool(xy <= xy_tol and abs(dz) <= z_tol and on_table)}


def realized_accel(R, smooth=0.05):
    """Peak realized eef acceleration from the recorded eef linear velocity (finite difference, moving average over
    `smooth` seconds; 50 ms matches the damage window and removes the 20 Hz controller ripple): lift (+z during lift),
    lateral (|y| during lateral), lower (+z during lower = braking), resultant over the whole carry."""
    dt = R.dt
    v = R.vec("eef_v", "xyz")
    a = np.gradient(v, dt, axis=0)
    k = max(1, int(round(smooth / dt)))
    a = uniform_filter1d(a, size=k, axis=0, mode="nearest")
    ph = R["phase"]
    def pk(mask, fn):
        return float(fn(a[mask])) if mask.any() else float("nan")
    carry = np.isin(ph, CARRY_PHASES[:3])
    return {"a_lift": pk(ph == PH["lift"], lambda x: x[:, 2].max()),
            "a_lat": pk(ph == PH["lateral"], lambda x: np.abs(x[:, 1]).max()),
            "a_lower": pk(ph == PH["lower"], lambda x: x[:, 2].max()),
            "a_res": pk(carry, lambda x: np.linalg.norm(x, axis=1).max())}


# ------------------------------------------------------------------------------------------------------- all labels
def all_labels(rec, cols, goal_pos, pad_half, T_carry, T_max_levels, fragility=FRAGILITY):
    """Every label for one branch. `T_max_levels` = {urgency name: T_max}. Returns a flat dict plus per-fragility labels."""
    R = Rec(rec, cols)
    motion = relative_motion(R)
    sl = slip(R, motion, pad_half)
    cp = completed(R, goal_pos)
    acc = realized_accel(R)
    acc.update({k + "_20ms": v for k, v in realized_accel(R, smooth=0.02).items()})
    sc = damage_scores(R, fragility)
    out = {**sl, **cp, **acc, "T_carry": float(T_carry),
           "score_N": sc["N"], "score_L": sc["L"], "peak_N": sc["peak_N"], "impulse": sc["impulse"],
           "fragility": {}}
    for name, f_tol in fragility.items():
        dl = damage_labels(sc, f_tol)
        phys = bool(not sl["slip"] and not dl["damage_primary"])
        lab = {**dl, "f_tol": f_tol, "physical_safe": phys, "phys_success": bool(phys and cp["completed"]),
               "damage_impulse_D": sc["impulse"][name]}
        for lvl, T_max in T_max_levels.items():
            lab[f"deadline_success_{lvl}"] = bool(phys and cp["completed"] and T_carry <= T_max)
        out["fragility"][name] = lab
    return out
