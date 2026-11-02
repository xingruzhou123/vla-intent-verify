"""Feasible set, margin, teacher selection and refined boundaries (plan.md §3.3, §3.4, §6). Implementation A.

A second, independent implementation (gate G8 in docs/data_spec.md) is written separately and must not import this file.

Conventions
- A *context* is (property combination, fragility level, urgency level). Fragility and urgency are label parameters,
  so every context of one physical sweep shares the same branches.
- Normalised coordinates: F̃ = (F − F_lo)/(F_hi − F_lo) (linear), s̃ = log(s/s_lo)/log(s_hi/s_lo) (log-linear).
- Margin on the grid: distance in (F̃, s̃) from a cell to the nearest cell outside 𝓕. Virtual cells one step beyond
  each grid edge count as outside 𝓕 (𝓕 ⊂ grid), so margins are bounded and the maximiser sits inside the feasible band.
  Negative outside 𝓕 (minus the distance to the nearest feasible cell).
- Refined feasible set: per s level, the F interval [F_min(s), F_max(s)] between the 50 % crossings of logistic fits
  to the slip and primary-damage outcomes over grid + probe branches at that s; s levels failing the deadline are outside.
"""
import numpy as np
from scipy.optimize import minimize

TAU = {"c": 0.9, "s": 0.05, "d": 0.05}      # plan §3.3 initial values (frozen before the main experiments)
ALPHA = BETA = 1.0
GAMMA = 0.0                                  # nominal-distance penalty; plan §3.4 leaves γ open — Phase 1 reports γ = 0
F_RANGE = (5.0, 70.0)
S_RANGE = (0.5, 3.0)
NOMINAL = (None, 1.0)                        # F0 is not used in Phase 1 (γ = 0); s0 = 1


def f_norm(F):
    return (np.asarray(F, float) - F_RANGE[0]) / (F_RANGE[1] - F_RANGE[0])


def s_norm(s):
    return np.log(np.asarray(s, float) / S_RANGE[0]) / np.log(S_RANGE[1] / S_RANGE[0])


# ----------------------------------------------------------------------------------------------------- grid tables
def context_tables(rows, Fs, ss, frag, urg, T_nom, T_max):
    """Empirical outcome rates per cell for one context, the feasible sets, margins, q and the grid teacher target."""
    nF, nS = len(Fs), len(ss)
    keys = ["deadline_success", "phys_success", "completed", "slip", "damage", "damage_shear"]
    P = {k: np.full((nF, nS), np.nan) for k in keys}
    n = np.zeros((nF, nS), int)
    for i in range(nF):
        for j in range(nS):
            cell = [r for r in rows if r["F_idx"] == i and r["s_idx"] == j]
            if not cell:
                continue
            n[i, j] = len(cell)
            fr = [r["fragility"][frag] for r in cell]
            P["deadline_success"][i, j] = np.mean([x[f"deadline_success_{urg}"] for x in fr])
            P["phys_success"][i, j] = np.mean([x["phys_success"] for x in fr])
            P["completed"][i, j] = np.mean([r["completed"] for r in cell])
            P["slip"][i, j] = np.mean([r["slip"] for r in cell])
            P["damage"][i, j] = np.mean([x["damage_primary"] for x in fr])
            P["damage_shear"][i, j] = np.mean([x["damage_shear"] for x in fr])
    deadline_ok = np.array([T_nom * s <= T_max for s in ss])
    feasible = (P["deadline_success"] >= TAU["c"]) & (P["slip"] <= TAU["s"]) & (P["damage"] <= TAU["d"])
    feasible_phys = (P["phys_success"] >= TAU["c"]) & (P["slip"] <= TAU["s"]) & (P["damage"] <= TAU["d"])
    margin = margin_grid(feasible, Fs, ss)
    q = P["deadline_success"] - ALPHA * P["slip"] - BETA * P["damage"]
    return {"P": P, "n": n, "deadline_ok": deadline_ok, "feasible": feasible, "feasible_phys": feasible_phys,
            "margin": margin, "margin_phys": margin_grid(feasible_phys, Fs, ss), "q": q,
            "teacher": teacher_grid(margin, feasible, Fs, ss), "Fs": Fs, "ss": ss, "frag": frag, "urg": urg}


def margin_grid(feasible, Fs, ss):
    """Signed distance in normalised (F̃, s̃) space to the nearest cell of the other class; virtual cells one grid step
    beyond every edge are infeasible."""
    nF, nS = feasible.shape
    fN, sN = f_norm(Fs), s_norm(ss)
    dF = np.diff(fN).mean() if nF > 1 else 1.0
    dS = np.diff(sN).mean() if nS > 1 else 1.0
    # padded grid with infeasible border
    pad = np.zeros((nF + 2, nS + 2), bool); pad[1:-1, 1:-1] = feasible
    fP = np.concatenate([[fN[0] - dF], fN, [fN[-1] + dF]]); sP = np.concatenate([[sN[0] - dS], sN, [sN[-1] + dS]])
    FF, SS = np.meshgrid(fP, sP, indexing="ij")
    out = np.zeros((nF, nS))
    for i in range(nF):
        for j in range(nS):
            same = feasible[i, j]
            other = pad != same
            if not other.any():
                out[i, j] = np.inf if same else -np.inf
                continue
            d = np.sqrt((FF - fN[i]) ** 2 + (SS - sN[j]) ** 2)[other].min()
            out[i, j] = d if same else -d
    return out


def teacher_grid(margin, feasible, Fs, ss, gamma=GAMMA):
    """(F*, s*) = argmax over 𝓕 of margin − γ‖(F̃, s̃) − (F̃0, s̃0)‖²; None if 𝓕 is empty (infeasible by design)."""
    if not feasible.any():
        return None
    best, arg = None, None
    for i in range(len(Fs)):
        for j in range(len(ss)):
            if not feasible[i, j]:
                continue
            ds = abs(s_norm(ss[j]) - s_norm(NOMINAL[1]))
            key = (round(float(margin[i, j]) - gamma * ds ** 2, 9), -ds)   # ties broken toward the nominal s
            if best is None or key > best:
                best, arg = key, (i, j)
    return {"F_idx": arg[0], "s_idx": arg[1], "F": Fs[arg[0]], "s": ss[arg[1]], "margin": float(margin[arg])}


def best_F_at_s(ctx, j):
    """Margin-maximising feasible F at grid level j (centre of the longest feasible run)."""
    col = ctx["feasible"][:, j]
    if not col.any():
        return None
    i = int(np.argmax(np.where(col, ctx["margin"][:, j], -np.inf)))
    return ctx["Fs"][i]


# ------------------------------------------------------------------------------------------- refined boundaries
def logistic_boundary(F, y, rep, increasing, n_boot=300, seed=0, w_min=0.3):
    """50 % crossing F50 of P(y = 1 | F) = σ(±(F − F50)/w) fitted by maximum likelihood (w ≥ w_min N) over all
    branches at one s level; 95 % CI by bootstrap over repeats. `increasing`: y rises with F (damage) or falls (slip)."""
    F, y, rep = np.asarray(F, float), np.asarray(y, float), np.asarray(rep)
    if len(F) == 0 or y.min() == y.max():
        return None
    sgn = 1.0 if increasing else -1.0

    def fit(Fv, yv):
        def nll(p):
            z = sgn * (Fv - p[0]) / max(p[1], w_min)
            return np.sum(np.logaddexp(0, -z) * yv + np.logaddexp(0, z) * (1 - yv))
        F0 = 0.5 * (Fv[yv == 1].mean() + Fv[yv == 0].mean())
        r = minimize(nll, [F0, 2.0], method="Nelder-Mead", options={"xatol": 1e-3, "fatol": 1e-6, "maxiter": 2000})
        return float(r.x[0]), float(max(r.x[1], w_min))

    F50, w = fit(F, y)
    rng = np.random.default_rng(seed)
    reps = np.unique(rep)
    boots = []
    for _ in range(n_boot):
        pick = rng.choice(reps, size=len(reps), replace=True)
        idx = np.concatenate([np.where(rep == p)[0] for p in pick])
        if y[idx].min() == y[idx].max():
            continue
        boots.append(fit(F[idx], y[idx])[0])
    ci = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))] if len(boots) >= 20 else [None, None]
    return {"F50": F50, "w": w, "ci95": ci, "n": int(len(F)), "n_boot": len(boots)}


def refined_intervals(rows, probes, Fs, ss, frag, urg, T_nom, T_max, n_boot=300):
    """Per s level: F_min from the slip boundary, F_max from the primary-damage boundary, deadline flag, and the
    refined feasible interval. Uses grid rows plus probe rows at that s level."""
    out = []
    for j, s in enumerate(ss):
        at = [r for r in rows if r["s_idx"] == j] + [p for p in probes if p.get("s_idx") == j]
        F = [r["F"] for r in at]; rep = [r["repeat"] for r in at]
        slip = logistic_boundary(F, [r["slip"] for r in at], rep, increasing=False, n_boot=n_boot)
        # the damage boundary is fitted on branches that did not slip: a dropped object can be caught and crushed
        # between the closing fingers, which is a different mechanism and already excluded by the slip label
        held = [r for r in at if not r["slip"]]
        Fh = [r["F"] for r in held]; reph = [r["repeat"] for r in held]
        dmg = logistic_boundary(Fh, [r["fragility"][frag]["damage_primary"] for r in held], reph, increasing=True, n_boot=n_boot)
        dmg_sh = logistic_boundary(Fh, [r["fragility"][frag]["damage_shear"] for r in held], reph, increasing=True, n_boot=n_boot)
        all_slip = all(r["slip"] for r in at); no_slip = not any(r["slip"] for r in at)
        all_dmg = all(r["fragility"][frag]["damage_primary"] for r in held) if held else True
        no_dmg = not any(r["fragility"][frag]["damage_primary"] for r in held) if held else False
        F_min = None if all_slip else (F_RANGE[0] if no_slip else slip["F50"])
        F_max = None if all_dmg else (F_RANGE[1] if no_dmg else dmg["F50"])
        ok = T_nom * s <= T_max
        feas = ok and F_min is not None and F_max is not None and F_max > F_min
        out.append({"s_idx": j, "s": s, "deadline_ok": bool(ok), "F_min": F_min, "F_max": F_max,
                    "slip_fit": slip, "damage_fit": dmg, "damage_shear_fit": dmg_sh, "feasible": bool(feas),
                    "slip_status": "all" if all_slip else ("none" if no_slip else "crossing"),
                    "damage_status": "all" if all_dmg else ("none" if no_dmg else "crossing")})
    return out


def select_refined(intervals, gamma=GAMMA):
    """Margin-maximising (F, s) on the refined feasible set. For each feasible s level the candidate is the interval
    centre; its margin is min(half-width in F̃, distance in s̃ to the nearest infeasible or off-grid s level)."""
    ss = [iv["s"] for iv in intervals]
    sN = s_norm(ss); dS = np.diff(sN).mean() if len(ss) > 1 else 1.0
    feas = [iv["feasible"] for iv in intervals]
    best, arg = None, None
    cands = []
    for j, iv in enumerate(intervals):
        if not iv["feasible"]:
            continue
        Fc = 0.5 * (iv["F_min"] + iv["F_max"])
        mF = 0.5 * (iv["F_max"] - iv["F_min"]) / (F_RANGE[1] - F_RANGE[0])
        # nearest infeasible s level (virtual levels beyond the grid are infeasible)
        dist = [abs(sN[k] - sN[j]) for k in range(len(ss)) if not feas[k]] + [sN[j] - (sN[0] - dS), (sN[-1] + dS) - sN[j]]
        mS = min(dist)
        m = min(mF, mS)
        ds = abs(sN[j] - s_norm(NOMINAL[1]))
        v = m - gamma * ds ** 2
        cands.append({"s_idx": j, "s": iv["s"], "F": Fc, "margin": m, "margin_F": mF, "margin_s": mS, "value": v})
        key = (round(v, 9), -ds)                                  # ties broken toward the nominal s
        if best is None or key > best:
            best, arg = key, cands[-1]
    return arg, cands


# -------------------------------------------------------------------------------------- P7 and fixed reference
def p7_context_dependence(selections, min_frac=0.25):
    """selections: list of (context name, s selected). Pass if ≥ min_frac of the feasible contexts select an s
    different from the modal s (threshold declared before seeing results, plan §6 P7)."""
    svals = [s for _, s in selections if s is not None]
    if not svals:
        return {"n_contexts": 0, "pass": None}
    vals, counts = np.unique(np.round(svals, 4), return_counts=True)
    modal = float(vals[np.argmax(counts)])
    frac = float(np.mean([abs(s - modal) > 1e-6 for s in svals]))
    return {"n_contexts": len(svals), "modal_s": modal, "frac_non_modal": frac, "threshold": min_frac, "pass": bool(frac >= min_frac),
            "selections": [(c, None if s is None else float(s)) for c, s in selections]}


def best_single_fixed(ctxs):
    """The single grid (F, s) maximising mean deadline-constrained safe success over all contexts, versus the
    per-context oracle (mean over contexts of the best cell)."""
    P = np.stack([c["P"]["deadline_success"] for c in ctxs])            # [ctx, F, s]
    mean = np.nanmean(P, axis=0)
    i, j = np.unravel_index(np.nanargmax(mean), mean.shape)
    Fs, ss = ctxs[0]["Fs"], ctxs[0]["ss"]
    oracle = float(np.nanmean([np.nanmax(c["P"]["deadline_success"]) for c in ctxs]))
    return {"F": Fs[i], "s": ss[j], "F_idx": int(i), "s_idx": int(j), "success": float(mean[i, j]),
            "per_context": [float(P[k, i, j]) for k in range(len(ctxs))], "oracle_per_context_success": oracle,
            "n_contexts": len(ctxs)}
