"""Implementation B — independent second implementation of the outcome labels (gate G7) and of the feasible-set /
margin / teacher computation (gate G8), docs/data_spec.md §2 and CLAUDE.md. Written from the specification
(plan.md §3.2–3.5, docs/data_spec.md §1.2–1.3, label version 2 of 2026-11-02) without importing labels.py or
feasible.py; it shares only the raw record format (env.SAMPLE_COLS, phase codes) and uses different numerics:
hand-written quaternion algebra instead of scipy Rotation, a min-filter instead of sliding windows for the sustained
peak, run-length encoding for the lost-contact test, numpy group-by for the empirical rates and a distance transform on
the padded grid for the margin.

Both implementations are written by the same author; the independence is in the code path and the numerics, not in
the authorship. Disagreements are listed by analysis/crosscheck_g7g8.py, never silently resolved.
"""
import numpy as np
from scipy.ndimage import minimum_filter1d

# ---- frozen label settings, restated from the spec (plan §3.2, §3.5, §6 P5; docs/data_spec.md §1.2)
DELTA_SLIP = 0.005
DROP_DIST = 0.08
LOST_CONTACT_T = 0.05
REF_WINDOW = 0.25
T_WIN = {"50ms": 0.050, "100ms": 0.100}
KAPPAS = {"k0.5": 0.5, "k1": 1.0, "k2": 2.0}
F_TOL_MULT = {"x0.75": 0.75, "x1": 1.0, "x1.25": 1.25}
GOAL_XY_TOL, GOAL_Z_TOL = 0.03, 0.01
PHASE = {"approach": 0, "ramp": 1, "hold": 2, "lift": 3, "lateral": 4, "lower": 5, "settle": 6, "release": 7}
CARRY = (PHASE["lift"], PHASE["lateral"], PHASE["lower"], PHASE["settle"])
MOVING = (PHASE["lift"], PHASE["lateral"], PHASE["lower"])
LOAD = (PHASE["ramp"], PHASE["hold"]) + CARRY
# ---- frozen feasible-set settings (plan §3.3, §3.4)
TAU_C, TAU_S, TAU_D = 0.9, 0.05, 0.05
F_LO, F_HI, S_LO, S_HI, S_NOM = 5.0, 70.0, 0.5, 3.0, 1.0


def _check_phase_codes():
    """The phase codes are part of the record format; verify them against env.PHASES at import."""
    from .env import PHASES
    assert list(PHASES) == list(PHASE) and all(PHASES.index(k) == v for k, v in PHASE.items()), "phase codes differ from env.PHASES"


try:
    _check_phase_codes()
except ImportError:
    pass


# ------------------------------------------------------------------------------------------- quaternion algebra
def q_mul(a, b):
    """Hamilton product of (w, x, y, z) quaternions (row-wise)."""
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([aw * bw - ax * bx - ay * by - az * bz,
                     aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw], axis=-1)


def q_conj(q):
    return q * np.array([1.0, -1.0, -1.0, -1.0])


def q_norm(q):
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def q_to_rotvec(q):
    """Rotation vector of unit quaternions (w, x, y, z), angle in (-π, π]."""
    q = q_norm(q)
    q = np.where(q[..., :1] < 0, -q, q)                      # shortest arc
    sin_half = np.linalg.norm(q[..., 1:], axis=-1)
    ang = 2.0 * np.arctan2(sin_half, q[..., 0])
    scale = np.where(sin_half > 1e-12, ang / np.maximum(sin_half, 1e-12), 2.0)
    return q[..., 1:] * scale[..., None]


def q_mean(q):
    """Chordal mean of nearly identical unit quaternions (sign-aligned to the first)."""
    q = q_norm(q)
    q = np.where((q @ q[0]) [:, None] < 0, -q, q)
    return q_norm(q.mean(axis=0))


# ------------------------------------------------------------------------------------------------------ labeler B
class Record:
    def __init__(self, rec, cols):
        self.a = np.asarray(rec, float); self.c = {n: i for i, n in enumerate(cols)}
        t = self.a[:, self.c["t"]]
        self.dt = float(np.median(t[1:] - t[:-1]))

    def col(self, n):
        return self.a[:, self.c[n]]

    def cols(self, names):
        return self.a[:, [self.c[n] for n in names]]


def slip_labels_b(R, pad_half):
    ph = R.col("phase").astype(int)
    hold = np.flatnonzero(ph == PHASE["hold"])
    t = R.col("t")
    ref = hold[t[hold] >= t[hold[-1]] - REF_WINDOW]
    relpad = R.cols(["relpad_x", "relpad_y", "relpad_z"])
    d = relpad - relpad[ref].mean(axis=0)
    # relative orientation object-in-eef: q_rel = conj(q_eef) * q_obj; torsion = x-component of rotvec(conj(q_ref) * q_rel)
    q_e = R.cols(["eef_qw", "eef_qx", "eef_qy", "eef_qz"]); q_o = R.cols(["obj_qw", "obj_qx", "obj_qy", "obj_qz"])
    q_rel = q_mul(q_conj(q_norm(q_e)), q_norm(q_o))
    q_ref = q_mean(q_rel[ref])
    tors = q_to_rotvec(q_mul(q_conj(q_ref)[None, :], q_rel))[:, 0]
    carry = np.isin(ph, CARRY)
    trans = np.sqrt(d[:, 1] ** 2 + d[:, 2] ** 2)
    far = (np.linalg.norm(relpad, axis=1) > DROP_DIST) & carry
    moving = np.isin(ph, MOVING)
    nc = (R.col("nc_left") + R.col("nc_right")) == 0
    n = max(1, int(round(LOST_CONTACT_T / R.dt)))
    # run lengths of consecutive no-contact samples within the moving phases
    idx_moving = np.flatnonzero(moving)
    seq = nc[idx_moving]
    run = np.zeros(len(seq), int); cur = 0
    for k, v in enumerate(seq):
        cur = cur + 1 if v else 0; run[k] = cur
    lost_hits = np.flatnonzero(run >= n)
    cands = []
    if far.any():
        cands.append(int(np.flatnonzero(far)[0]))
    if len(lost_hits):
        cands.append(int(idx_moving[lost_hits[0] - n + 1]))
    dropped = bool(cands)
    drop_idx = min(cands) if dropped else None
    trans_max = float(trans[carry].max()); tors_max = float(np.abs(tors[carry]).max())
    return {"slip": bool(trans_max > DELTA_SLIP or tors_max * pad_half > DELTA_SLIP or dropped),
            "dropped": dropped, "drop_idx": drop_idx, "slip_trans_max": trans_max, "slip_torsion_max_deg": float(np.degrees(tors_max))}


def sustained_peak_b(x, n):
    """max over t of the minimum over [t, t + n) — min-filter with the window starting at t."""
    if len(x) < n:
        return float(x.min())
    m = minimum_filter1d(x, size=n, mode="nearest", origin=-(n // 2))     # window [t, t + n)
    return float(m[: len(x) - n + 1].max())


def damage_labels_b(R, fragility, drop_idx):
    ph = R.col("phase").astype(int)
    mask = np.isin(ph, LOAD)
    if drop_idx is not None:
        mask[drop_idx:] = False
    N = np.stack([R.col("f_left")[mask], R.col("f_right")[mask]], axis=1)
    T = np.stack([np.hypot(R.col("ft_left_y")[mask], R.col("ft_left_z")[mask]),
                  np.hypot(R.col("ft_right_y")[mask], R.col("ft_right_z")[mask])], axis=1)
    scores = {}
    for wn, tw in T_WIN.items():
        n = max(1, int(round(tw / R.dt)))
        scores[("N", wn)] = max(sustained_peak_b(N[:, k], n) for k in (0, 1))
        for kn, kap in KAPPAS.items():
            scores[("L", wn, kn)] = max(sustained_peak_b(N[:, k] + kap * T[:, k], n) for k in (0, 1))
    out = {}
    for name, f_tol in fragility.items():
        sens = {}
        for mn, mult in F_TOL_MULT.items():
            for wn in T_WIN:
                sens[f"N|{wn}|{mn}"] = bool(scores[("N", wn)] > f_tol * mult)
                for kn in KAPPAS:
                    sens[f"L|{wn}|{kn}|{mn}"] = bool(scores[("L", wn, kn)] > f_tol * mult)
        out[name] = {"damage_primary": sens["N|50ms|x1"], "damage_shear": sens["L|50ms|k1|x1"], "damage_sensitivity": sens}
    return out


def completed_b(R, goal_pos, dropped):
    last = R.a[-1]
    xy = float(np.hypot(last[R.c["obj_x"]] - goal_pos[0], last[R.c["obj_y"]] - goal_pos[1]))
    dz = abs(float(last[R.c["obj_z"]] - goal_pos[2]))
    geo = bool(xy <= GOAL_XY_TOL and dz <= GOAL_Z_TOL and last[R.c["on_table"]] > 0.5)
    return bool(geo and not dropped)


def labels_b(rec, cols, goal_pos, pad_half, T_carry, T_max_levels, fragility):
    """All labels of one branch, implementation B. Same output keys as labels.all_labels for the gated labels."""
    R = Record(rec, cols)
    sl = slip_labels_b(R, pad_half)
    comp = completed_b(R, goal_pos, sl["dropped"])
    dmg = damage_labels_b(R, fragility, sl["drop_idx"])
    out = {"slip": sl["slip"], "dropped": sl["dropped"], "drop_idx": sl["drop_idx"], "completed": comp, "fragility": {},
           "slip_trans_max": sl["slip_trans_max"], "slip_torsion_max_deg": sl["slip_torsion_max_deg"]}
    for name in fragility:
        phys = (not sl["slip"]) and (not dmg[name]["damage_primary"])
        lab = {**dmg[name], "physical_safe": phys, "phys_success": phys and comp}
        for lvl, T_max in T_max_levels.items():
            lab[f"deadline_success_{lvl}"] = bool(phys and comp and T_carry <= T_max)
        out["fragility"][name] = lab
    return out


# ------------------------------------------------------------------------------------------- feasible sets B
def rates_b(rows, nF, nS, frag, urg):
    """Empirical rates per grid cell by numpy group-by (rows must carry F_idx, s_idx and the labels)."""
    cell = np.array([r["F_idx"] * nS + r["s_idx"] for r in rows])
    cnt = np.bincount(cell, minlength=nF * nS).astype(float)
    def rate(vals):
        return (np.bincount(cell, weights=np.asarray(vals, float), minlength=nF * nS) / np.maximum(cnt, 1)).reshape(nF, nS)
    P = {"deadline_success": rate([r["fragility"][frag][f"deadline_success_{urg}"] for r in rows]),
         "slip": rate([r["slip"] for r in rows]),
         "damage": rate([r["fragility"][frag]["damage_primary"] for r in rows]),
         "completed": rate([r["completed"] for r in rows]),
         "phys_success": rate([r["fragility"][frag]["phys_success"] for r in rows])}
    return P, cnt.reshape(nF, nS)


def feasible_b(P):
    return (P["deadline_success"] >= TAU_C) & (P["slip"] <= TAU_S) & (P["damage"] <= TAU_D)


def margin_b(feas, Fs, ss):
    """Signed distance in normalised (F̃, s̃) to the nearest cell of the other class, virtual infeasible border one
    step beyond every edge. F̃ is linear in F, so the geomspace F levels are not uniform in F̃ and a lattice distance
    transform would be wrong; the distances are taken on the actual coordinates with a vectorised pairwise matrix."""
    fN = (np.asarray(Fs, float) - F_LO) / (F_HI - F_LO); sN = np.log(np.asarray(ss, float) / S_LO) / np.log(S_HI / S_LO)
    dF = float(np.mean(np.diff(fN))); dS = float(np.mean(np.diff(sN)))
    fP = np.r_[fN[0] - dF, fN, fN[-1] + dF]; sP = np.r_[sN[0] - dS, sN, sN[-1] + dS]
    nF, nS = feas.shape
    pad = np.zeros((nF + 2, nS + 2), bool); pad[1:-1, 1:-1] = feas
    FI, SI = np.meshgrid(fN, sN, indexing="ij"); FP, SP = np.meshgrid(fP, sP, indexing="ij")
    D = np.sqrt((FI.ravel()[:, None] - FP.ravel()[None, :]) ** 2 + (SI.ravel()[:, None] - SP.ravel()[None, :]) ** 2)   # [inner, padded]
    other = feas.ravel()[:, None] != pad.ravel()[None, :]
    Dm = np.where(other, D, np.inf).min(axis=1).reshape(nF, nS)
    return np.where(feas, Dm, -Dm)


def teacher_b(feas, margin, Fs, ss):
    if not feas.any():
        return None
    sN = np.log(np.asarray(ss) / S_LO) / np.log(S_HI / S_LO); s0 = np.log(S_NOM / S_LO) / np.log(S_HI / S_LO)
    best = None
    for i in range(len(Fs)):
        for j in range(len(ss)):
            if feas[i, j]:
                key = (round(float(margin[i, j]), 9), -abs(sN[j] - s0))
                if best is None or key > best[0]:
                    best = (key, (i, j))
    i, j = best[1]
    return {"F_idx": i, "s_idx": j, "F": Fs[i], "s": ss[j], "margin": float(margin[i, j])}


def context_b(rows, Fs, ss, frag, urg):
    P, n = rates_b(rows, len(Fs), len(ss), frag, urg)
    feas = feasible_b(P)
    m = margin_b(feas, Fs, ss)
    return {"P": P, "n": n, "feasible": feas, "margin": m, "teacher": teacher_b(feas, m, Fs, ss)}
