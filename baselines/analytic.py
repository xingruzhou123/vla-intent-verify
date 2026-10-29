"""Analytic baselines B5a (translational Coulomb) and B5b (wrench-aware) — docs/analytic_baselines.md.

Both output, for every s level, a feasible F interval [F_min(s), F_max(s)] from the realized accelerations logged in the
sweep, and select (F, s) with the same margin-maximising rule as the teacher (feasible.select_refined). They never see
simulator outcomes.

Geometry (plan §3.1, §3.5): the lateral carry runs along the closing axis, so the lateral inertial load m·a_lat is carried
by a normal-force difference between the pads (trailing pad F + m·a_lat/2, leading pad F − m·a_lat/2); gravity and the
vertical inertial loads are tangential and shared by the two pads. The slip condition is therefore checked per phase:
  lift / lower:  μ F ≥ m (g + a_vert) / 2                 (tangential load, N = F)
  lateral:       μ (F − m a_lat / 2) ≥ m g / 2             (leading pad unloaded by the inertial term)
and the primary-damage cap is the trailing pad: F + m a_lat / 2 ≤ f_tol.
(The first draft of docs/analytic_baselines.md wrote the lateral slip bound as m·sqrt(g² + a_lat²)/(2μ), which is the
rule for a carry orthogonal to the closing axis; corrected here and in the doc, 2026-10-29.)

B5b adds the CoM torque about the pad normal, M = m (g + a_vert) d / 2 per pad, resisted by the friction patch with
effective radius r_eff = 0.765 h (uniformly loaded square patch of half-size h = 8 mm), through the ellipsoidal limit
surface (f_t / μN)² + (M / (μ N r_eff))² ≤ 1.
"""
import numpy as np

G = 9.81
PAD_HALF = 0.008
R_EFF = 0.765 * PAD_HALF


def coulomb_interval(m, mu, f_tol, a_lift, a_lat, a_lower):
    """B5a. Returns (F_min, F_max); F_max may be below F_min (empty)."""
    F_lift = m * (G + a_lift) / (2 * mu)
    F_lower = m * (G + a_lower) / (2 * mu)
    F_lat = m * G / (2 * mu) + m * a_lat / 2
    F_min = max(F_lift, F_lower, F_lat)
    F_max = f_tol - m * a_lat / 2
    return float(F_min), float(F_max), {"lift": F_lift, "lateral": F_lat, "lower": F_lower}


def wrench_interval(m, mu, f_tol, d, a_lift, a_lat, a_lower, r_eff=R_EFF, kappa=None):
    """B5b. CoM offset d (m) orthogonal to closing. With kappa=None the damage cap is the primary proxy (trailing-pad
    normal force); with a kappa the shear-augmented cap N + κ|T| ≤ f_tol is used instead."""
    def need(f_t, M, N_offset):
        return np.sqrt(f_t ** 2 + (M / r_eff) ** 2) / mu + N_offset
    F_lift = need(m * (G + a_lift) / 2, m * (G + a_lift) * d / 2, 0.0)
    F_lower = need(m * (G + a_lower) / 2, m * (G + a_lower) * d / 2, 0.0)
    F_lat = need(m * G / 2, m * G * d / 2, m * a_lat / 2)
    F_min = max(F_lift, F_lower, F_lat)
    T_lat = np.sqrt((m * G / 2) ** 2 + (m * G * d / 2 / r_eff) ** 2)
    F_max = f_tol - m * a_lat / 2 - (0.0 if kappa is None else kappa * T_lat)
    return float(F_min), float(F_max), {"lift": F_lift, "lateral": F_lat, "lower": F_lower}


COARSE_BINS = {"mass": {"light": 0.5, "medium": 1.0, "heavy": 1.5}, "friction": {"slippery": 0.2, "normal": 0.4},
               "fragility": {"fragile": 20.0, "rigid": 50.0}, "balance": {"balanced": 0.0, "off-centre": 0.15 * 0.04}}


def coarse_bin(m, mu, f_tol, d):
    """Nearest bin for each attribute (what the coarse text would say without noise)."""
    pick = lambda table, v: min(table, key=lambda k: abs(table[k] - v))
    return {"mass": pick(COARSE_BINS["mass"], m), "friction": pick(COARSE_BINS["friction"], mu),
            "fragility": pick(COARSE_BINS["fragility"], f_tol), "balance": pick(COARSE_BINS["balance"], d)}


def coarse_values(bins):
    return (COARSE_BINS["mass"][bins["mass"]], COARSE_BINS["friction"][bins["friction"]],
            COARSE_BINS["fragility"][bins["fragility"]], COARSE_BINS["balance"][bins["balance"]])


def neighbour_bins(bins):
    """Every single-attribute neighbouring-bin replacement (the ε-noise model of plan §3.6, one attribute at a time)."""
    out = []
    for attr, table in COARSE_BINS.items():
        keys = list(table); k = keys.index(bins[attr])
        for kk in (k - 1, k + 1):
            if 0 <= kk < len(keys):
                b = dict(bins); b[attr] = keys[kk]; out.append((attr, keys[kk], b))
    return out


def intervals_for_sweep(accel_by_s, ss, m, mu, f_tol, d, rule, T_nom, T_max):
    """accel_by_s: list over s levels of dicts {a_lift, a_lat, a_lower} (realized, from the sweep)."""
    out = []
    for j, s in enumerate(ss):
        a = accel_by_s[j]
        if rule == "B5a":
            F_min, F_max, parts = coulomb_interval(m, mu, f_tol, a["a_lift"], a["a_lat"], a["a_lower"])
        else:
            F_min, F_max, parts = wrench_interval(m, mu, f_tol, d, a["a_lift"], a["a_lat"], a["a_lower"])
        ok = T_nom * s <= T_max
        out.append({"s_idx": j, "s": s, "deadline_ok": bool(ok), "F_min": F_min, "F_max": F_max, "parts": parts,
                    "feasible": bool(ok and F_max > F_min)})
    return out
