"""Three-gate check, second version (owner changes of 2026-11-03), on the extended object set (sweep v1 + extension).
No new phase, no π0.5, no probe rollouts: probe observations are simulated from the true object properties with the
sensor model of analysis/active_probe_gates.py (unchanged).

Changes against the first version (analysis/active_probe_gates.py, results/phase1/active_probe/):
  1. Oracle: out-of-sample — the analytic rules B5a / B5b evaluated on the TRUE properties (m, μ, d, f_tol) of the held-out
     object (multipliers c_min, c_max, r_eff calibrated on the fold's training units with true properties, and the
     uncalibrated c = 1 version), selections executed in the simulator with R paired repeats. The in-sample best-grid-cell
     oracle is reported only as a reference.
  2. Gate 2: unchanged sensor model; ρ_analytic with bootstrap CIs over units, and separately the absolute gap
     (J_oracle − J_probe+formula) in success points with a paired bootstrap CI.
  3. Gate 3: grouped splits (fold = object index mod n_folds, every object in exactly one held-out fold) and a learning
     curve — ρ_small at 25 / 50 / 75 / 100 % of the fold's training objects (stratified by bin cell, several subsample
     draws) — so that data scarcity can be separated from model capacity.
  4. The fraction of all contexts that are ambiguous (where probing can matter at all), with CIs.
  Also: bins now hold a variable number of objects (4 + 3–4 from the extension); "common-safe" means one grid (F, s) is
  feasible for every object of the bin; no dependency on coarse_bound2.json (its baselines are recomputed here with the
  same code paths); the calibration search is vectorised (select_batch replicates feasible.select_refined, checked).
  The decision-matrix row is NOT assigned mechanically (owner, 2026-11-03: Gate 3's failure in v1 was a data-size effect).

Usage: MUJOCO_GL=egl .venv/bin/python -m analysis.active_probe_gates2 --sweeps data/sweep_v1,data/sweep_v1_ext \
    --objects data/objects_v1_ext.json --out results/phase1/active_probe_v2 [--no_exec] [--seeds 10] [--skip_gate3]
"""
import argparse, json, pathlib, itertools, time
import numpy as np
from envs.feasibility_carry import feasible as fs
from envs.feasibility_carry.objects import CELLS
from baselines import analytic as B
from analysis.phase1_demo import load, accel_by_s, props, fmt
from analysis.phase1_full import in_refined, execute_selections, with_deadline, FRAGS, URGS, CAL_GRID, noisy_bins, NOISE_SEED
from analysis.coarse_bound2 import per_bin_means, est_from, argmax_cell
from analysis.active_probe_gates import SENSOR, probe_draw, boot_ci, G


def boot_diff_ci(a, b_, n=1000, seed=0):
    """Paired bootstrap over units of mean(a) − mean(b); units unscored in either (NaN) are dropped."""
    a, b_ = np.asarray(a, float), np.asarray(b_, float); keep = ~(np.isnan(a) | np.isnan(b_)); a, b_ = a[keep], b_[keep]
    if len(a) == 0:
        return (None, None)
    rng = np.random.default_rng(seed); out = []
    for _ in range(n):
        idx = rng.integers(len(a), size=len(a)); out.append(a[idx].mean() - b_[idx].mean())
    return (float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)))


def boot_rho(J_probe, J_coarse, J_oracle, n=1000, seed=0):
    """Bootstrap over units of ρ = (mean J_probe − mean J_coarse) / (mean J_oracle − mean J_coarse); NaN units dropped."""
    P, C, O = (np.asarray(x, float) for x in (J_probe, J_coarse, J_oracle)); keep = ~(np.isnan(P) | np.isnan(C) | np.isnan(O)); P, C, O = P[keep], C[keep], O[keep]
    if len(P) == 0:
        return (None, None)
    rng = np.random.default_rng(seed); out = []
    for _ in range(n):
        idx = rng.integers(len(P), size=len(P)); den = O[idx].mean() - C[idx].mean()
        if abs(den) > 1e-9:
            out.append((P[idx].mean() - C[idx].mean()) / den)
    return (float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))) if out else (None, None)

GATE1_D_SUCC, GATE1_INF_REL, GATE1_INF_ABS, RHO_STOP = 0.10, 0.40, 0.10, 0.8


# ------------------------------------------------------------------------------------------ vectorised rule + selection
def raw_bounds(rule, m, mu, f_tol, d, acc, r_eff):
    """Uncalibrated (F_min, F_max) per s level of one rule from property values (arrays broadcast over units)."""
    m, mu, f_tol, d = (np.asarray(x, float) for x in (m, mu, f_tol, d))
    Fmin, Fmax = [], []
    for a in acc:
        if rule == "B5a":
            lo = np.maximum.reduce([m * (G + a["a_lift"]) / (2 * mu), m * (G + a["a_lower"]) / (2 * mu), m * G / (2 * mu) + m * a["a_lat"] / 2])
        else:
            need = lambda f_t, M, N_off: np.sqrt(f_t ** 2 + (M / r_eff) ** 2) / mu + N_off
            lo = np.maximum.reduce([need(m * (G + a["a_lift"]) / 2, m * (G + a["a_lift"]) * d / 2, 0.0),
                                    need(m * (G + a["a_lower"]) / 2, m * (G + a["a_lower"]) * d / 2, 0.0),
                                    need(m * G / 2, m * G * d / 2, m * a["a_lat"] / 2)])
        Fmin.append(lo); Fmax.append(f_tol - m * a["a_lat"] / 2)
    return np.stack(Fmin, -1), np.stack(Fmax, -1)          # (..., S)


def select_batch(Fmin, Fmax, ok, ss):
    """feasible.select_refined (γ = 0) on arrays (..., S): returns (s_idx, F_centre, feasible-any) with the same tie rule
    (highest margin, ties toward the nominal s)."""
    sN = fs.s_norm(ss); dS = np.diff(sN).mean(); s0 = fs.s_norm(fs.NOMINAL[1]); S = len(ss)
    feas = ok & (Fmax > Fmin)
    mF = 0.5 * (Fmax - Fmin) / (fs.F_RANGE[1] - fs.F_RANGE[0])
    dist = np.abs(sN[None, :] - sN[:, None])                                       # (j, k)
    edge = np.minimum(sN - (sN[0] - dS), (sN[-1] + dS) - sN)                       # (j,)
    big = 1e9
    mS = np.zeros(feas.shape)
    for j in range(S):
        dk = np.where(~feas, dist[j], big).min(-1)                                  # nearest infeasible level
        mS[..., j] = np.minimum(dk, edge[j])
    m = np.where(feas, np.minimum(mF, mS), -np.inf)
    v = np.round(m, 9)                                                                # select_refined key = (round(v, 9), -ds)
    best_v = v.max(-1, keepdims=True)
    cand = (v == best_v) & feas
    j = np.where(cand, np.abs(sN - s0), np.inf).argmin(-1)                            # first j with the smallest |ds| among ties
    anyf = feas.any(-1)
    Fc = 0.5 * (np.take_along_axis(Fmin, j[..., None], -1) + np.take_along_axis(Fmax, j[..., None], -1))[..., 0]
    return j, Fc, anyf


def sel_dict(j, Fc, anyf, ss):
    return None if not anyf else {"s_idx": int(j), "s": float(ss[j]), "F": float(Fc)}


def fit_mult_batch(rule, est_units, acc_units, ok_units, ref_units, ss, c_grid=None):
    """Calibration of (c_min, c_max, r_eff) on training units: est_units = list of (m, μ, f_tol, d), acc_units = per unit
    accelerations (list over s), ok_units = deadline flags (U, S), ref_units = refined simulated intervals (feasible, F_min,
    F_max per s). Objective: the number of training units whose selection lies inside the refined interval (as
    coarse_bound2.fit_multipliers), ties → multipliers closest to 1."""
    c_min_g, c_max_g = (CAL_GRID["c_min"], CAL_GRID["c_max"]) if c_grid is None else c_grid
    r_grid = CAL_GRID["r_eff"] if rule == "B5b" else [B.R_EFF]
    U = len(est_units); S = len(ss)
    rf = np.array([[v["feasible"] for v in r] for r in ref_units])
    rlo = np.array([[v["F_min"] if v["F_min"] is not None else np.nan for v in r] for r in ref_units], float)
    rhi = np.array([[v["F_max"] if v["F_max"] is not None else np.nan for v in r] for r in ref_units], float)
    best = None
    for r_eff in r_grid:
        Fmin = np.zeros((U, S)); Fmax = np.zeros((U, S))
        for u, ((m, mu, f_tol, d), acc) in enumerate(zip(est_units, acc_units)):
            Fmin[u], Fmax[u] = raw_bounds(rule, m, mu, f_tol, d, acc, r_eff)
        cm = np.asarray(c_min_g)[:, None, None, None]; cx = np.asarray(c_max_g)[None, :, None, None]
        lo = cm * Fmin[None, None]; hi = cx * Fmax[None, None]                      # (Cmin, Cmax, U, S)
        okb = np.broadcast_to(ok_units[None, None], lo.shape)
        j, Fc, anyf = select_batch(lo, hi, okb, ss)
        inref = anyf & np.take_along_axis(np.broadcast_to(rf[None, None], lo.shape), j[..., None], -1)[..., 0]
        lo_r = np.take_along_axis(np.broadcast_to(rlo[None, None], lo.shape), j[..., None], -1)[..., 0]
        hi_r = np.take_along_axis(np.broadcast_to(rhi[None, None], lo.shape), j[..., None], -1)[..., 0]
        with np.errstate(invalid="ignore"):
            inref &= (lo_r <= Fc) & (Fc <= hi_r)
        hits = inref.sum(-1)                                                          # (Cmin, Cmax)
        for a, c_min in enumerate(c_min_g):
            for b, c_max in enumerate(c_max_g):
                key = (int(hits[a, b]), -abs(np.log(c_min)) - abs(np.log(c_max)))
                if best is None or key > best[0]:
                    best = (key, {"c_min": float(c_min), "c_max": float(c_max), "r_eff": float(r_eff), "train_success": float(hits[a, b]) / U})
    return best[1]


def rule_select(rule, est, acc, ss, ok, mm):
    Fmin, Fmax = raw_bounds(rule, *est, acc, mm["r_eff"])
    j, Fc, anyf = select_batch(mm["c_min"] * Fmin, mm["c_max"] * Fmax, ok, ss)
    return sel_dict(j, Fc, anyf, ss)


def probe_est(mode, obs, f_tol, k=2.0):
    if mode == "point":
        return (obs["m"], obs["mu"], f_tol, abs(obs["d"]))
    return (obs["m"] + k * obs["sm"], max(obs["mu"] - k * obs["smu"], 0.02), f_tol, abs(obs["d"]) + k * obs["sd"])


def selfcheck(ss, acc):
    """select_batch must reproduce feasible.select_refined on random intervals (with the rule_intervals layout)."""
    rng = np.random.default_rng(0); n_bad = 0
    for _ in range(300):
        lo = rng.uniform(5, 60, len(ss)); hi = lo + rng.uniform(-10, 30, len(ss)); ok = rng.random(len(ss)) > 0.3
        iv = [{"s_idx": j, "s": s, "deadline_ok": bool(ok[j]), "F_min": float(lo[j]), "F_max": float(hi[j]), "feasible": bool(ok[j] and hi[j] > lo[j])} for j, s in enumerate(ss)]
        a, _ = fs.select_refined(iv); j, Fc, anyf = select_batch(lo, hi, ok, ss); b = sel_dict(j, Fc, anyf, ss)
        if (a is None) != (b is None) or (a is not None and (a["s_idx"] != b["s_idx"] or abs(a["F"] - b["F"]) > 1e-9)):
            n_bad += 1
    assert n_bad == 0, f"select_batch disagrees with select_refined on {n_bad}/300 random cases"
    # raw_bounds vs baselines.analytic
    for rule in ("B5a", "B5b"):
        m, mu, f_tol, d = 1.2, 0.21, 22.0, 0.009
        Fmin, Fmax = raw_bounds(rule, m, mu, f_tol, d, acc, 0.013)
        for j, a in enumerate(acc):
            lo_, hi_, _ = (B.coulomb_interval(m, mu, f_tol, a["a_lift"], a["a_lat"], a["a_lower"]) if rule == "B5a"
                           else B.wrench_interval(m, mu, f_tol, d, a["a_lift"], a["a_lat"], a["a_lower"], r_eff=0.013))
            assert abs(lo_ - Fmin[j]) < 1e-9 and abs(hi_ - Fmax[j]) < 1e-9
    return True


def ci_s(ci, nd=3):
    return "" if ci is None or ci[0] is None else f"[{ci[0]:.{nd}f}, {ci[1]:.{nd}f}]"


# ------------------------------------------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweeps", required=True, help="comma-separated sweep roots"); ap.add_argument("--objects", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--no_exec", action="store_true"); ap.add_argument("--workers", type=int, default=12); ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--skip_gate3", action="store_true"); ap.add_argument("--n_folds", type=int, default=4)
    ap.add_argument("--curve_fracs", default="0.25,0.5,0.75,1.0"); ap.add_argument("--curve_reps", type=int, default=3)
    ap.add_argument("--ensemble", type=int, default=5); ap.add_argument("--tag", default="sweep v1 + extension")
    args = ap.parse_args(argv)
    t_start = time.perf_counter()
    out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)
    roots = [pathlib.Path(r) for r in args.sweeps.split(",")]
    objs = {o["object_id"]: o for o in json.loads(pathlib.Path(args.objects).read_text())["objects"]}
    def find(oid):
        for r in roots:
            if (r / oid / "probes.json").exists():
                return r / oid
        raise FileNotFoundError(oid)
    sweeps = {oid: load(find(oid)) for oid in objs}
    sw0 = next(iter(sweeps.values())); Fs, ss = sw0["meta"]["F"], sw0["meta"]["s"]; T_nom, T_max = sw0["meta"]["T_nominal"], sw0["meta"]["T_max"]
    lv = {sw["meta"].get("label_version") for sw in sweeps.values()}
    ctx = {(oid, fr, urg): fs.context_tables(sw["rows"], Fs, ss, fr, urg, T_nom, T_max[urg]) for oid, sw in sweeps.items() for fr in FRAGS for urg in URGS}
    ref = {}
    for oid, sw in sweeps.items():
        for fr in FRAGS:
            iv = fs.refined_intervals(sw["rows"], sw["probes"], Fs, ss, fr, "relaxed", T_nom, T_max["relaxed"], n_boot=0)
            for urg in URGS:
                ref[(oid, fr, urg)] = with_deadline(iv, T_nom, T_max[urg])
    acc = {oid: accel_by_s(sw) for oid, sw in sweeps.items()}
    ok_urg = {urg: np.array([T_nom * s <= T_max[urg] for s in ss]) for urg in URGS}
    selfcheck(ss, acc[next(iter(acc))])
    cell_of = {oid: f"{o['bins']['mass']}_{o['bins']['friction']}_{o['bins']['balance']}" for oid, o in objs.items()}
    idx_of = {oid: int(oid.rsplit("_", 1)[1]) for oid in objs}
    fold_of = {oid: idx_of[oid] % args.n_folds for oid in objs}
    units = [(oid, fr, urg) for oid in objs for fr in FRAGS for urg in URGS]
    oid_sorted = sorted(objs)
    truth = {u: (props(sweeps[u[0]])[0], props(sweeps[u[0]])[1], objs[u[0]]["f_tol"][u[1]], props(sweeps[u[0]])[2]) for u in units}
    R = {"sensor_model": SENSOR, "objects": len(objs), "label_versions": sorted(str(v) for v in lv), "n_folds": args.n_folds,
         "fold_sizes": {f: sum(1 for o in objs if fold_of[o] == f) for f in range(args.n_folds)}}
    lines = [f"# Active-probing direction — three-gate check v2 ({args.tag}; {len(objs)} objects, label version(s) {', '.join(R['label_versions'])})\n",
             "Owner changes of 2026-11-03: out-of-sample oracle (B5a / B5b on true properties, executed), bootstrap CIs and the absolute oracle − probe gap, grouped splits with a learning curve, and the ambiguous fraction. "
             "No new phase, no π0.5, no probe rollouts. Only measured numbers; CIs are 95 % bootstrap over units (1000 resamples); the decision-matrix row is not assigned by this script.\n",
             f"Folds: fold = per-cell object index mod {args.n_folds}; sizes {R['fold_sizes']} objects. Bins hold {min(sum(1 for o in objs if cell_of[o] == c) for c in {cell_of[o] for o in objs})}–{max(sum(1 for o in objs if cell_of[o] == c) for c in {cell_of[o] for o in objs})} objects.\n"]

    # ------------------------------------------------------------------ A. contexts and the ambiguous fraction
    group_of = {u: (cell_of[u[0]], u[1], u[2]) for u in units}
    members = {}
    for u in units:
        members.setdefault(group_of[u], []).append(u)
    def common_cell(us):
        return bool(np.logical_and.reduce([ctx[u]["feasible"] for u in us]).any()) if us else False
    cls = {}
    for u in units:
        g = group_of[u]; feas = ctx[u]["feasible"].any()
        if not feas: cls[u] = "infeasible"
        elif common_cell(members[g]): cls[u] = "common-safe"
        elif all(ctx[v]["feasible"].any() for v in members[g]): cls[u] = "ambiguous"
        else: cls[u] = "mixed"
    loo_amb = {u: bool(ctx[u]["feasible"].any() and not common_cell([v for v in members[group_of[u]] if v[0] != u[0]])) for u in units}
    counts = {k: sum(1 for u in units if cls[u] == k) for k in ("common-safe", "ambiguous", "mixed", "infeasible")}
    amb_broad = [u for u in units if cls[u] in ("ambiguous", "mixed")]; amb_strict = [u for u in units if cls[u] == "ambiguous"]
    feas_units = [u for u in units if cls[u] != "infeasible"]
    ind = lambda pred: [float(pred(u)) for u in units]
    amb_frac = {"broad_of_all": (len(amb_broad) / len(units), boot_ci(ind(lambda u: cls[u] in ("ambiguous", "mixed")))),
            "strict_of_all": (len(amb_strict) / len(units), boot_ci(ind(lambda u: cls[u] == "ambiguous"))),
            "broad_of_feasible": (len(amb_broad) / len(feas_units), boot_ci([float(cls[u] in ("ambiguous", "mixed")) for u in feas_units])),
            "strict_of_feasible": (len(amb_strict) / len(feas_units), boot_ci([float(cls[u] == "ambiguous") for u in feas_units])),
            "loo_ambiguous_of_all": (sum(loo_amb.values()) / len(units), boot_ci([float(loo_amb[u]) for u in units]))}
    # same fractions restricted to the original 48 objects (bins of 4), for comparison with v1
    orig = [u for u in units if idx_of[u[0]] < 4]
    R["counts"] = {"units": len(units), **counts, "ambiguous_broad": len(amb_broad), "feasible_units": len(feas_units), "loo_ambiguous_total": int(sum(loo_amb.values())),
                   "loo_survive_strict": int(sum(loo_amb[u] for u in amb_strict)), "loo_survive_broad": int(sum(loo_amb[u] for u in amb_broad)),
                   "by_fragility": {fr: {k: sum(1 for u in units if u[1] == fr and cls[u] == k) for k in counts} for fr in FRAGS},
                   "by_urgency": {urg: {k: sum(1 for u in units if u[2] == urg and cls[u] == k) for k in counts} for urg in URGS},
                   "by_object_set": {"original_48": {k: sum(1 for u in orig if cls[u] == k) for k in counts}, "extension": {k: sum(1 for u in units if idx_of[u[0]] >= 4 and cls[u] == k) for k in counts}},
                   "fractions": {k: {"value": v[0], "ci": v[1]} for k, v in amb_frac.items()}}
    lines += [f"## A. Independent contexts (unit = object × fragility × urgency; {len(objs)} × 2 × 2 = {len(units)})\n",
              "| class | definition | units | fragile / rigid | relaxed / urgent | original 48 objects / extension |", "|---|---|---|---|---|---|"]
    defs = {"common-safe": "one grid (F, s) is in 𝓕 of every object of the coarse bin (same fragility, urgency)",
            "ambiguous": "every object of the bin has a non-empty 𝓕 but no common cell", "mixed": "unit feasible, bin has no common cell, at least one sibling infeasible",
            "infeasible": "the unit's own 𝓕 is empty"}
    for k in counts:
        c = R["counts"]
        lines.append(f"| {k} | {defs[k]} | {counts[k]} | {c['by_fragility']['fragile'][k]} / {c['by_fragility']['rigid'][k]} | {c['by_urgency']['relaxed'][k]} / {c['by_urgency']['urgent'][k]} | {c['by_object_set']['original_48'][k]} / {c['by_object_set']['extension'][k]} |")
    lines.append(f"\n- Ambiguous, broad (ambiguous + mixed): **{len(amb_broad)}** units; strict: **{len(amb_strict)}**; feasible units {len(feas_units)}.")
    lines.append(f"- Leave-one-object-out (held-out unit feasible, all other objects of its bin without a common cell): {R['counts']['loo_ambiguous_total']} units; {R['counts']['loo_survive_strict']} of {len(amb_strict)} strict and {R['counts']['loo_survive_broad']} of {len(amb_broad)} broad units stay ambiguous when held out.")
    lines.append("\n### Fraction of contexts that are ambiguous (item 4: where probing can matter at all)\n")
    lines.append("| fraction | value [95 % CI] |"); lines.append("|---|---|")
    names = {"broad_of_all": "ambiguous (broad) / all units", "strict_of_all": "ambiguous (strict) / all units", "broad_of_feasible": "ambiguous (broad) / feasible units",
             "strict_of_feasible": "ambiguous (strict) / feasible units", "loo_ambiguous_of_all": "ambiguous under leave-one-object-out / all units"}
    for k, (v, ci) in amb_frac.items():
        lines.append(f"| {names[k]} | {v:.3f} {ci_s(ci)} |")
    nb_lo, nb_hi = min(len(set(u[0] for u in m)) for m in members.values()), max(len(set(u[0] for u in m)) for m in members.values())
    lines.append(f"\nThe classification depends on the bin size ({nb_lo}–{nb_hi} objects per bin here; v1 had 4 per bin and found 27 broad / 4 strict of 192): the more objects a bin holds, the harder a common cell is to find. "
                 f"Units are correlated within an object (4 per object); the CIs resample units.")

    # ------------------------------------------------------------------ calibrations per fold, selections, execution
    def draw_noise(seed):
        return {(oid, fr): noisy_bins(objs[oid]["bins"], fr, np.random.default_rng([seed, k, fi]))[0] for k, oid in enumerate(oid_sorted) for fi, fr in enumerate(FRAGS)}
    noise = draw_noise(NOISE_SEED)
    def draw_obs(seed):
        d = {}
        for k, oid in enumerate(oid_sorted):
            m, mu, dd = props(sweeps[oid]); d[oid] = probe_draw(m, mu, dd, np.random.default_rng([seed, k, 7]))
        return d
    obs_by_seed = [draw_obs(s) for s in range(args.seeds)]
    PROBE_VARIANTS = [("B5a", "point"), ("B5b", "point"), ("B5a", "interval"), ("B5b", "interval")]
    METHODS_EXEC = ["B5a-true", "B5b-true", "B5a-true-cal", "B5b-true-cal", "B5a-c", "B5b-c"] + [f"{r}-{m}-probe" for r, m in PROBE_VARIANTS]
    sel = {}                     # (method, unit, seed) -> selection dict or None
    jobs = {}; mults = {}
    def add_job(u, s_):
        if s_ is not None:
            key = f"{s_['F']:.3f}|{s_['s']:.4f}"; s_["exec_key"] = key
            jobs.setdefault(str(sweeps[u[0]]["dir"]), {})[key] = (float(s_["F"]), float(s_["s"]))
    for fold in range(args.n_folds):
        test = [u for u in units if fold_of[u[0]] == fold]; train = [u for u in units if fold_of[u[0]] != fold]
        train_obj = sorted({u[0] for u in train}); cal = per_bin_means(objs, train_obj)
        train_ne = [u for u in train if any(v["feasible"] for v in ref[u])]
        acc_tr = [acc[u[0]] for u in train_ne]; ok_tr = np.array([ok_urg[u[2]] for u in train_ne]); ref_tr = [ref[u] for u in train_ne]
        for rule in ("B5a", "B5b"):
            t0 = time.perf_counter()
            mults[(fold, f"{rule}-c")] = fit_mult_batch(rule, [est_from(cal, noise[(u[0], u[1])]) for u in train_ne], acc_tr, ok_tr, ref_tr, ss)
            mults[(fold, f"{rule}-true-cal")] = fit_mult_batch(rule, [truth[u] for u in train_ne], acc_tr, ok_tr, ref_tr, ss)
            for mode in ("point", "interval"):
                mults[(fold, f"{rule}-{mode}-probe")] = fit_mult_batch(rule, [probe_est(mode, obs_by_seed[0][u[0]], cal["fragility"][u[1]]) for u in train_ne], acc_tr, ok_tr, ref_tr, ss)
            print(f"fold {fold} {rule}: 4 calibrations in {time.perf_counter() - t0:.1f} s", flush=True)
        one = {"c_min": 1.0, "c_max": 1.0, "r_eff": B.R_EFF}
        for u in test:
            oid, fr, urg = u; ok = ok_urg[urg]
            for rule in ("B5a", "B5b"):
                sel[(f"{rule}-true", u, 0)] = rule_select(rule, truth[u], acc[oid], ss, ok, one)
                sel[(f"{rule}-true-cal", u, 0)] = rule_select(rule, truth[u], acc[oid], ss, ok, mults[(fold, f"{rule}-true-cal")])
                sel[(f"{rule}-c", u, 0)] = rule_select(rule, est_from(cal, noise[(oid, fr)]), acc[oid], ss, ok, mults[(fold, f"{rule}-c")])
                for mode in ("point", "interval"):
                    for seed, obs in enumerate(obs_by_seed):
                        sel[(f"{rule}-{mode}-probe", u, seed)] = rule_select(rule, probe_est(mode, obs[oid], cal["fragility"][fr]), acc[oid], ss, ok, mults[(fold, f"{rule}-{mode}-probe")])
            for m in METHODS_EXEC:
                add_job(u, sel[(m, u, 0)])
    n_jobs = sum(len(j) for j in jobs.values())
    print(f"executing {n_jobs} distinct selections over {len(jobs)} objects (cached ones are skipped)", flush=True)
    t0 = time.perf_counter()
    exec_res = execute_selections(jobs, args.workers) if (jobs and not args.no_exec) else {}
    print(f"execution done in {(time.perf_counter() - t0) / 60:.1f} min", flush=True)
    # per-unit outcome of every executed method (seed 0) and of the grid methods
    J = {}; INF = {}
    def outcome(u, s_):
        """(success, infeasible-or-None) of an executed selection on unit u."""
        oid, fr, urg = u; ne = bool(ctx[u]["feasible"].any())
        if s_ is None:
            return 0.0, (True if ne else None)
        e = exec_res.get(str(sweeps[oid]["dir"]), {}).get(s_["exec_key"])
        if e is None:
            return None, None
        P_d, P_s, P_dm = e[f"deadline_success|{fr}|{urg}"], e["slip"], e[f"damage|{fr}"]
        inf = not (P_d >= fs.TAU["c"] and P_s <= fs.TAU["s"] and P_dm <= fs.TAU["d"])
        return float(P_d), (bool(inf) if ne else None)
    for m in METHODS_EXEC:
        for u in units:
            J[(m, u)], INF[(m, u)] = outcome(u, sel[(m, u, 0)])
    for u in units:
        oid, fr, urg = u; P = ctx[u]["P"]["deadline_success"]; feas = ctx[u]["feasible"]; ne = bool(feas.any())
        ij = np.unravel_index(int(np.nanargmax(P)), P.shape) if ne else None
        J[("grid-oracle", u)] = float(P[ij]) if ne else 0.0; INF[("grid-oracle", u)] = (bool(not feas[ij]) if ne else None)
        sib = [v for v in members[group_of[u]] if fold_of[v[0]] != fold_of[oid]]
        if sib:
            ij = argmax_cell(np.mean([ctx[v]["P"]["deadline_success"] for v in sib], axis=0), Fs, ss)
            J[("lookup", u)] = float(P[ij]); INF[("lookup", u)] = (bool(not feas[ij]) if ne else None)
        else:
            J[("lookup", u)] = 0.0; INF[("lookup", u)] = (True if ne else None)
    missing = [k for k, v in J.items() if v is None]
    if missing:
        lines.append(f"\n**{len(missing)} selections were not executed (--no_exec); their success is left empty and excluded from the means.**")
    def stats(m, subset):
        s_ = [J[(m, u)] for u in subset if J[(m, u)] is not None]; i_ = [INF[(m, u)] for u in subset if INF[(m, u)] is not None]
        return {"n": len(subset), "n_scored": len(s_), "success": float(np.mean(s_)) if s_ else None, "success_ci": boot_ci(s_) if s_ else (None, None),
                "infeasible": float(np.mean(i_)) if i_ else None, "infeasible_ci": boot_ci([float(x) for x in i_]) if i_ else (None, None)}
    def vec(m, subset):
        return [J[(m, u)] if J[(m, u)] is not None else np.nan for u in subset]
    SUBSETS = [("ambiguous_broad", "Ambiguous subset, broad (ambiguous + mixed)", amb_broad), ("ambiguous_strict", "Ambiguous subset, strict", amb_strict),
               ("all", f"All {len(units)} units", units), ("feasible", "All feasible units", feas_units)]

    # ------------------------------------------------------------------ B. Gate 1
    lines.append("\n## B. Gate 1 — value of exact information (grouped folds; every rule calibrated on the fold's training units and executed with R paired repeats)\n")
    lines.append("Out-of-sample oracles: B5a / B5b on the held-out object's TRUE (m, μ, d, f_tol) with multipliers calibrated on the training folds' true properties (`-true-cal`) or uncalibrated c = 1 (`-true`); "
                 "the realized accelerations are the sweep medians as for every rule. The in-sample grid oracle (best cell of the held-out unit's own outcomes) is a reference only. "
                 "Coarse baselines: B5a-c / B5b-c (ε = 0.1 noisy bins, calibrated per fold) and the noise-free lookup from the training siblings of the bin.\n")
    # headline oracle: the true-property rule with the higher success on all units (stated before the subsets are read)
    ORACLE = max(("B5a-true-cal", "B5b-true-cal"), key=lambda m: (stats(m, units)["success"] or -1))
    R["oracle_oos"] = ORACLE
    lines.append(f"Headline out-of-sample oracle = **{ORACLE}** (the higher success of the two calibrated true-property rules on all {len(units)} units; B5a-true-cal {stats('B5a-true-cal', units)['success']:.3f}, B5b-true-cal {stats('B5b-true-cal', units)['success']:.3f}).\n")
    G1_METHODS = ["grid-oracle", "B5a-true", "B5b-true", "B5a-true-cal", "B5b-true-cal", "B5a-c", "B5b-c", "lookup"]
    g1 = {}
    for key, name, subset in SUBSETS:
        if not subset: continue
        lines.append(f"\n### {name} ({len(subset)} units)\n")
        lines.append("| method | safe success [95 % CI] | infeasible-selection rate (truth non-empty) [95 % CI] |"); lines.append("|---|---|---|")
        rows_ = {m: stats(m, subset) for m in G1_METHODS}
        for m in G1_METHODS:
            r = rows_[m]; lines.append(f"| {m}{' (in-sample reference)' if m == 'grid-oracle' else ''} | {fmt(r['success'], 3)} {ci_s(r['success_ci'])} | {fmt(r['infeasible'], 3)} {ci_s(r['infeasible_ci'])} |")
        best = max(("B5a-c", "B5b-c", "lookup"), key=lambda m: (rows_[m]["success"] or -1))
        if rows_[ORACLE]["success"] is None or rows_[best]["success"] is None:
            lines.append("\n(not executed: no Gate 1 verdict on this subset)"); g1[key] = {"rows": rows_, "pass": None}; continue
        d_succ = rows_[ORACLE]["success"] - rows_[best]["success"]; d_ci = boot_diff_ci(vec(ORACLE, subset), vec(best, subset))
        inf_b, inf_o = rows_[best]["infeasible"], rows_[ORACLE]["infeasible"]
        rel = (inf_b - inf_o) / inf_b if inf_b else None; absd = (inf_b - inf_o) if inf_b is not None and inf_o is not None else None
        passed = bool(d_succ >= GATE1_D_SUCC or (rel is not None and rel >= GATE1_INF_REL and absd >= GATE1_INF_ABS))
        lines.append(f"\nStronger coarse baseline: {best}. {ORACLE} − {best}: success {d_succ:+.3f} {ci_s(d_ci)} (criterion ≥ {GATE1_D_SUCC}); infeasible rate {fmt(inf_b, 3)} → {fmt(inf_o, 3)} "
                     f"(relative drop {fmt(rel, 2)}, absolute {fmt(absd, 3)}; criterion ≥ {GATE1_INF_REL} and ≥ {GATE1_INF_ABS}). **{'PASS' if passed else 'FAIL'}**")
        g1[key] = {"rows": rows_, "best_coarse": best, "d_success": d_succ, "d_success_ci": d_ci, "inf_rel_drop": rel, "inf_abs_drop": absd, "pass": passed}
    gate1_pass = g1["ambiguous_broad"]["pass"]
    lines.append(f"\n**Gate 1 verdict (ambiguous subset, broad, out-of-sample oracle {ORACLE}): {'PASS' if gate1_pass else 'FAIL'}.**")
    R["gate1"] = g1; R["gate1_pass"] = gate1_pass
    R["multipliers"] = {f"fold{f}|{m}": v for (f, m), v in mults.items()}

    # ------------------------------------------------------------------ C. Gate 2
    lines.append("\n## C. Gate 2 — probe + formula (sensor model unchanged from v1)\n")
    for k, v in SENSOR.items():
        lines.append(f"- **{k}:** {v if isinstance(v, str) else v.get('note', '') + ' → ' + ', '.join(f'{kk} = {vv}' for kk, vv in v.items() if kk != 'note')}")
    lines.append(f"\nUpdaters: calibrated per fold on the training units' seed-0 probe observations (full calibration grid); executed with seed-0 observations; interval membership over seeds 0–{args.seeds - 1}. "
                 f"ρ_analytic = (J_probe − J_coarse) / (J_oracle − J_coarse) with J_coarse = B5a-c and J_oracle = {ORACLE} (out-of-sample); gap = J_oracle − J_probe in success points (paired bootstrap over units).\n")
    lines.append("Calibrated multipliers: " + "; ".join(f"fold {f} {m}: c_min {v['c_min']:.2f}, c_max {v['c_max']:.2f}, r_eff {v['r_eff']*1e3:.0f} mm, train {v['train_success']:.2f}" for (f, m), v in mults.items()) + ".\n")
    g2 = {}
    for key, name, subset in SUBSETS:
        if not subset: continue
        lines.append(f"\n### {name} ({len(subset)} units)\n")
        lines.append("| updater | safe success [95 % CI] | infeasible rate | in refined interval, all seeds (rate outside) | gap = oracle − probe (success points) [95 % CI] | gap vs grid oracle | ρ_analytic [95 % CI] | ρ_analytic (infeasible) |")
        lines.append("|---|---|---|---|---|---|---|---|")
        Jo, Jc, Jg = vec(ORACLE, subset), vec("B5a-c", subset), vec("grid-oracle", subset)
        Ic = [INF[("B5a-c", u)] for u in subset if INF[("B5a-c", u)] is not None]
        res = {}
        for rule, mode in PROBE_VARIANTS:
            m = f"{rule}-{mode}-probe"; st = stats(m, subset); Jp = vec(m, subset)
            outside = [not in_refined(ref[u], sel[(m, u, seed)]) for u in subset if ctx[u]["feasible"].any() for seed in range(args.seeds)]
            with np.errstate(invalid="ignore"), __import__("warnings").catch_warnings():
                __import__("warnings").simplefilter("ignore")
                gap = float(np.nanmean(Jo) - np.nanmean(Jp)); gap_ci = boot_diff_ci(Jo, Jp); gap_g = float(np.nanmean(Jg) - np.nanmean(Jp))
                den = np.nanmean(Jo) - np.nanmean(Jc)
                rho = (np.nanmean(Jp) - np.nanmean(Jc)) / den if abs(den) > 1e-9 else None; rho_ci = boot_rho(Jp, Jc, Jo) if rho is not None else (None, None)
            Ip = [INF[(m, u)] for u in subset if INF[(m, u)] is not None]
            rho_inf = (np.mean(Ic) - np.mean(Ip)) / np.mean(Ic) if Ic and Ip and np.mean(Ic) > 0 else None
            res[m] = {**st, "outside_refined_allseeds": float(np.mean(outside)) if outside else None, "gap": gap, "gap_ci": gap_ci, "gap_vs_grid_oracle": gap_g, "rho": rho, "rho_ci": rho_ci, "rho_inf": rho_inf}
            lines.append(f"| {rule} {mode} | {fmt(st['success'], 3)} {ci_s(st['success_ci'])} | {fmt(st['infeasible'], 3)} | {fmt(res[m]['outside_refined_allseeds'], 3)} | {gap:+.3f} {ci_s(gap_ci)} | {gap_g:+.3f} | {fmt(rho, 3)} {ci_s(rho_ci, 2)} | {fmt(rho_inf, 3)} |")
        strongest = max(res, key=lambda k: (res[k]["success"] if res[k]["success"] is not None else -1))
        lines.append(f"\nReference: J_oracle ({ORACLE}) = {np.nanmean(Jo):.3f}, grid oracle = {np.nanmean(Jg):.3f}, J_coarse (B5a-c) = {np.nanmean(Jc):.3f}, B5a-c infeasible rate {fmt(float(np.mean(Ic)) if Ic else None, 3)}. "
                     f"Strongest updater: **{strongest}**, ρ_analytic = {fmt(res[strongest]['rho'], 3)} {ci_s(res[strongest]['rho_ci'], 2)}, gap {res[strongest]['gap']:+.3f} {ci_s(res[strongest]['gap_ci'])}.")
        g2[key] = {"res": res, "strongest": strongest, "J_oracle": float(np.nanmean(Jo)), "J_grid_oracle": float(np.nanmean(Jg)), "J_coarse": float(np.nanmean(Jc))}
    rho_an = g2["ambiguous_broad"]["res"][g2["ambiguous_broad"]["strongest"]]["rho"]; rho_an_ci = g2["ambiguous_broad"]["res"][g2["ambiguous_broad"]["strongest"]]["rho_ci"]
    gap_an = g2["ambiguous_broad"]["res"][g2["ambiguous_broad"]["strongest"]]["gap"]; gap_an_ci = g2["ambiguous_broad"]["res"][g2["ambiguous_broad"]["strongest"]]["gap_ci"]
    lines.append(f"\n**Gate 2 (ambiguous, broad): ρ_analytic = {fmt(rho_an, 3)} {ci_s(rho_an_ci, 2)}; gap oracle − probe = {gap_an:+.3f} {ci_s(gap_an_ci)} success points. "
                 f"{'ρ ≥ 0.8' if rho_an is not None and rho_an >= RHO_STOP else 'ρ < 0.8'} as a point estimate; the CI states what the data can exclude.**")
    R["gate2"] = g2; R["rho_analytic"] = rho_an; R["rho_analytic_ci"] = rho_an_ci; R["gap_analytic"] = gap_an; R["gap_analytic_ci"] = gap_an_ci

    # ------------------------------------------------------------------ D. Gate 3 with learning curve
    g3 = None
    if not args.skip_gate3:
        import torch, torch.nn as nn
        torch.set_num_threads(max(1, args.workers // 2))
        def feats(oid, fr, urg, obs):
            b = objs[oid]["bins"]
            x = [obs["m"], obs["mu"], obs["d"] * 10, obs["sm"], obs["smu"], obs["sd"] * 10]
            x += [float(b["mass"] == k) for k in ("light", "medium", "heavy")] + [float(b["friction"] == "normal"), float(b["balance"] == "off-centre"), float(fr == "rigid"), float(urg == "urgent")]
            return x
        class MLP(nn.Module):
            def __init__(self, d_in, d_out, h=64):
                super().__init__(); self.net = nn.Sequential(nn.Linear(d_in, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, d_out))
            def forward(self, x): return self.net(x)
        def train_mlp(X, Y, seed, epochs=300):
            torch.manual_seed(seed); m = MLP(X.shape[1], Y.shape[1]); opt = torch.optim.Adam(m.parameters(), lr=2e-3, weight_decay=1e-4)
            Xt, Yt = torch.tensor(X), torch.tensor(Y); mu_, sd_ = Xt.mean(0), Xt.std(0) + 1e-6; Xn = (Xt - mu_) / sd_
            for ep in range(epochs):
                perm = torch.randperm(len(Xt))
                for i in range(0, len(Xt), 128):
                    idx = perm[i:i + 128]; opt.zero_grad()
                    loss = nn.functional.binary_cross_entropy_with_logits(m(Xn[idx]), Yt[idx]); loss.backward(); opt.step()
            return m, mu_, sd_
        def predict(models, x):
            xt = torch.tensor(np.asarray(x, np.float32))[None]
            with torch.no_grad():
                ps = [torch.sigmoid(m((xt - mu_) / sd_))[0].numpy() for m, mu_, sd_ in models]
            return np.mean(ps, 0), np.std(ps, 0)
        sN = fs.s_norm(ss); s0 = fs.s_norm(1.0)
        def choose(score):
            best, arg = None, None
            for i in range(len(Fs)):
                for j in range(len(ss)):
                    key = (round(float(score[i * len(ss) + j]), 6), -abs(sN[j] - s0), -abs(i - (len(Fs) - 1) / 2))
                    if best is None or key > best: best, arg = key, (i, j)
            return arg
        n_aug = 20
        fracs = [float(x) for x in args.curve_fracs.split(",")]
        g3_rows = []          # (frac, rep, fold, unit, model, seed) -> success / infeasible
        n_train_obj = {}
        for fold in range(args.n_folds):
            test = [u for u in units if fold_of[u[0]] == fold]; train_obj_all = sorted({o for o in objs if fold_of[o] != fold})
            for frac in fracs:
                reps = 1 if frac >= 1.0 else args.curve_reps
                for rep in range(reps):
                    if frac >= 1.0:
                        train_obj = train_obj_all
                    else:   # stratified by bin cell: round(frac · n_cell) objects of every cell, seeded draw
                        rng = np.random.default_rng([500, fold, rep, int(frac * 100)]); train_obj = []
                        for cell in sorted({cell_of[o] for o in train_obj_all}):
                            pool = [o for o in train_obj_all if cell_of[o] == cell]; n_take = max(1, int(round(frac * len(pool))))
                            train_obj += list(rng.choice(pool, n_take, replace=False))
                        train_obj = sorted(train_obj)
                    n_train_obj[(frac, rep, fold)] = len(train_obj)
                    X, Y = [], []
                    for oid in train_obj:
                        m, mu, dd = props(sweeps[oid])
                        for fr in FRAGS:
                            for urg in URGS:
                                for a in range(n_aug):
                                    rng = np.random.default_rng([100 + fold, a, oid_sorted.index(oid), FRAGS.index(fr), URGS.index(urg)])
                                    X.append(feats(oid, fr, urg, probe_draw(m, mu, dd, rng))); Y.append(ctx[(oid, fr, urg)]["feasible"].astype(np.float32).ravel())
                    X, Y = np.asarray(X, np.float32), np.asarray(Y, np.float32)
                    t0 = time.perf_counter()
                    models = [train_mlp(X, Y, seed) for seed in range(args.ensemble)]
                    for u in test:
                        oid, fr, urg = u; P = ctx[u]["P"]["deadline_success"]; feas = ctx[u]["feasible"]
                        for seed, obs in enumerate(obs_by_seed):
                            pm, psd = predict(models, feats(oid, fr, urg, obs[oid]))
                            for name, score in (("mlp-mean", pm), ("ensemble-pessimistic", pm - psd)):
                                i, j = choose(score)
                                g3_rows.append({"frac": frac, "rep": rep, "fold": fold, "unit": u, "model": name, "seed": seed, "success": float(P[i, j]), "infeasible": bool(not feas[i, j]), "truth_nonempty": bool(feas.any())})
                    print(f"gate3 fold {fold} frac {frac} rep {rep}: {len(train_obj)} training objects, {len(X)} samples, {args.ensemble} MLPs in {time.perf_counter() - t0:.0f} s", flush=True)
        lines.append(f"\n## D. Gate 3 — small learned model with a learning curve (MLP 2 × 64 from the simulated probe observations + coarse bins + urgency → 48-cell feasibility mask; "
                     f"grouped {args.n_folds}-fold splits by object; {n_aug} noise draws per training unit; {args.ensemble}-member ensemble; subsets of the training objects stratified by bin cell, {args.curve_reps} subsample draw(s) per fraction below 100 %)\n")
        lines.append(f"Success = outcome rate of the chosen grid cell on the held-out unit (the stored grid outcomes are the executed outcomes of that cell), mean over {args.seeds} probe-noise seeds; ρ_small against J_coarse = B5a-c and J_oracle = {ORACLE}.\n")
        g3 = {"curve": {}, "n_train_obj": {f"{k[0]}|{k[1]}|{k[2]}": v for k, v in n_train_obj.items()}}
        for key, name, subset in SUBSETS:
            if not subset or key == "feasible": continue
            lines.append(f"\n### {name} ({len(subset)} units)\n")
            lines.append("| training objects (fraction; mean count per fold) | model | safe success [95 % CI] | infeasible rate | gap = oracle − model [95 % CI] | ρ_small [95 % CI] | ρ_small (infeasible) |")
            lines.append("|---|---|---|---|---|---|---|")
            Jo, Jc = vec(ORACLE, subset), vec("B5a-c", subset); Ic = [INF[("B5a-c", u)] for u in subset if INF[("B5a-c", u)] is not None]
            g3["curve"][key] = {}
            for frac in fracs:
                n_mean = np.mean([v for k, v in n_train_obj.items() if k[0] == frac])
                for model in (("mlp-mean", "ensemble-pessimistic") if frac >= 1.0 else ("mlp-mean",)):
                    per_unit = [np.mean([r["success"] for r in g3_rows if r["frac"] == frac and r["model"] == model and r["unit"] == u]) for u in subset]
                    inf_unit = [np.mean([r["infeasible"] for r in g3_rows if r["frac"] == frac and r["model"] == model and r["unit"] == u]) for u in subset if ctx[u]["feasible"].any()]
                    Jp = float(np.mean(per_unit)); den = np.nanmean(Jo) - np.nanmean(Jc)
                    rho = (Jp - np.nanmean(Jc)) / den if abs(den) > 1e-9 else None; rho_ci = boot_rho(per_unit, Jc, Jo) if rho is not None else (None, None)
                    gap = float(np.nanmean(Jo) - Jp); gap_ci = boot_diff_ci(Jo, per_unit)
                    rho_inf = (np.mean(Ic) - np.mean(inf_unit)) / np.mean(Ic) if Ic and inf_unit and np.mean(Ic) > 0 else None
                    g3["curve"][key][f"{frac}|{model}"] = {"n_train_obj_mean": float(n_mean), "J": Jp, "J_ci": boot_ci(per_unit), "infeasible": float(np.mean(inf_unit)) if inf_unit else None, "gap": gap, "gap_ci": gap_ci, "rho": rho, "rho_ci": rho_ci, "rho_inf": rho_inf}
                    r_ = g3["curve"][key][f"{frac}|{model}"]
                    lines.append(f"| {frac:.0%} ({n_mean:.1f}) | {model} | {Jp:.3f} {ci_s(r_['J_ci'])} | {fmt(r_['infeasible'], 3)} | {gap:+.3f} {ci_s(gap_ci)} | {fmt(rho, 3)} {ci_s(rho_ci, 2)} | {fmt(rho_inf, 3)} |")
        cb = g3["curve"]["ambiguous_broad"]
        R["gate3"] = g3; R["rho_small"] = cb[f"{fracs[-1]}|mlp-mean"]["rho"]; R["rho_small_ci"] = cb[f"{fracs[-1]}|mlp-mean"]["rho_ci"]
        R["rho_small_curve"] = {str(f): cb[f"{f}|mlp-mean"]["rho"] for f in fracs}
        lines.append(f"\n**Gate 3 (ambiguous, broad, mlp-mean): ρ_small at " + ", ".join(f"{f:.0%}: {fmt(cb[f'{f}|mlp-mean']['rho'], 3)}" for f in fracs) + f"; at 100 % {ci_s(R['rho_small_ci'], 2)}.**")

    # ------------------------------------------------------------------ summary (no mechanical matrix row)
    lines.append("\n## Summary for the decision note (matrix row not assigned here)\n")
    lines.append(f"- Ambiguous fraction: broad {amb_frac['broad_of_all'][0]:.3f} {ci_s(amb_frac['broad_of_all'][1])} of all units, strict {amb_frac['strict_of_all'][0]:.3f} {ci_s(amb_frac['strict_of_all'][1])}; {len(amb_broad)} broad / {len(amb_strict)} strict units.")
    lines.append(f"- Gate 1 (out-of-sample oracle {ORACLE}, ambiguous broad): {'PASS' if gate1_pass else 'FAIL'}; Δ success {g1['ambiguous_broad']['d_success']:+.3f} {ci_s(g1['ambiguous_broad']['d_success_ci'])}.")
    lines.append(f"- Gate 2: ρ_analytic = {fmt(rho_an, 3)} {ci_s(rho_an_ci, 2)}; gap {gap_an:+.3f} {ci_s(gap_an_ci)} success points.")
    if g3 is not None:
        lines.append(f"- Gate 3: ρ_small learning curve (mlp-mean) " + ", ".join(f"{f:.0%} → {fmt(R['rho_small_curve'][str(f)], 3)}" for f in fracs) + f"; 100 % CI {ci_s(R['rho_small_ci'], 2)}.")
    lines.append("- Owner rule (2026-11-03): a small model failing from data scarcity does not imply a larger model would succeed; the learning-curve trend, not the 100 % point alone, is the evidence on data scarcity vs capacity. The matrix row is the owner's reading.")
    lines.append(f"\nWall time {(time.perf_counter() - t_start) / 60:.1f} min.")
    (out / "report.md").write_text("\n".join(lines))
    (out / "gates.json").write_text(json.dumps(R, indent=1, default=lambda o: o.item() if hasattr(o, "item") else (list(o) if isinstance(o, (set, tuple)) else str(o))))
    (out / "selections.json").write_text(json.dumps([{"method": m, "unit": list(u), "seed": s, "sel": v, "success": J.get((m, u)) if s == 0 else None, "infeasible": INF.get((m, u)) if s == 0 else None, "cls": cls[u]}
                                                     for (m, u, s), v in sel.items()], indent=0, default=float))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
