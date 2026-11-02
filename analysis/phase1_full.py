"""Full-sweep Phase-1 evaluation (plan.md §6): P1–P7 with pass/fail on the 48 sampled objects, the formal P4 on the
held-out test-combos objects (B5a / B5b under oracle, coarse-uncalibrated and coarse-calibrated information), the
best-single-fixed reference, and post-hoc diagnostics (slip-boundary shift distribution, P3 recovery-window widths).
Every criterion is applied exactly as frozen in plan.md §6 / notes/decision_phase1.md; nothing is tuned here.

Usage:
  MUJOCO_GL=egl .venv/bin/python -m analysis.phase1_full --sweeps data/sweep_v1 --objects data/objects_v1.json \
      --out results/phase1/full_v1 [--no_exec] [--workers 12]

--no_exec skips executing the baselines' selected (F, s) in the simulator (docs/analytic_baselines.md §4); the
executed selections are cached per object in <sweep>/<object>/p4_exec.json (analysis data, never training data).
"""
import argparse, json, pathlib, os, itertools
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from scipy.stats import kendalltau, spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from envs.feasibility_carry import feasible as fs
from envs.feasibility_carry import labels as L
from envs.feasibility_carry.objects import BANDS, CENTRES
from baselines import analytic as B
from analysis.phase1_demo import load, accel_by_s, props, fmt, ci_str, COL

FRAGS = ["fragile", "rigid"]
URGS = ["relaxed", "urgent"]
P2_SHIFT_N = 9.75                     # 15 % of (70 − 5) N, absolute (frozen 2026-10-29)
P7_MIN_FRAC = 0.25
P4_INFEASIBLE_RATE = 0.10
NOISE_EPS = 0.1                       # plan §3.6: each bin independently replaced by a neighbouring bin with probability ε
NOISE_SEED = 0
P1_R2 = 0.9
P6_AGREE = 0.9
P5_TAU = 0.8
CAL_GRID = {"c_min": np.round(np.arange(0.5, 2.001, 0.05), 3), "c_max": np.round(np.arange(0.5, 1.301, 0.05), 3),
            "r_eff": [B.R_EFF, 0.008, 0.010, 0.013, 0.017, 0.022, 0.030]}
ATTR_KEYS = {"mass": list(BANDS["mass"]), "friction": list(BANDS["friction"]), "fragility": list(BANDS["fragility"]),
             "balance": list(BANDS["balance"])}


# ------------------------------------------------------------------------------------------------ helpers
def uncertainty(fit, bracket):
    """95 % CI ∪ bisection bracket for one refined boundary (plan §6: bracket when repeats agree)."""
    lo, hi = [], []
    if fit and fit["ci95"][0] is not None:
        lo.append(fit["ci95"][0]); hi.append(fit["ci95"][1])
    if bracket and bracket.get("status") == "bracketed":
        lo.append(bracket["lo"]); hi.append(bracket["hi"])
    if fit:
        lo.append(fit["F50"]); hi.append(fit["F50"])
    return (min(lo), max(hi)) if lo else None


def noisy_bins(bins, frag, rng, eps=NOISE_EPS):
    """ε-noise of plan §3.6 on the four coarse attributes (one draw per context)."""
    b = dict(bins); b["fragility"] = frag
    flips = {}
    for attr, keys in ATTR_KEYS.items():
        if rng.random() < eps:
            k = keys.index(b[attr])
            nb = [kk for kk in (k - 1, k + 1) if 0 <= kk < len(keys)]
            new = keys[rng.choice(nb)]
            flips[attr] = (b[attr], new); b[attr] = new
    return b, flips


def rule_intervals(rule, est, acc, ss, T_nom, T_max, c_min=1.0, c_max=1.0, r_eff=B.R_EFF):
    """Intervals of one rule from point estimates est = (m, μ, f_tol, d), with the calibration multipliers."""
    m, mu, f_tol, d = est
    out = []
    for j, s in enumerate(ss):
        a = acc[j]
        if rule == "B5a":
            F_min, F_max, _ = B.coulomb_interval(m, mu, f_tol, a["a_lift"], a["a_lat"], a["a_lower"])
        else:
            F_min, F_max, _ = B.wrench_interval(m, mu, f_tol, d, a["a_lift"], a["a_lat"], a["a_lower"], r_eff=r_eff)
        F_min, F_max = c_min * F_min, c_max * F_max
        ok = T_nom * s <= T_max
        out.append({"s_idx": j, "s": s, "deadline_ok": bool(ok), "F_min": F_min, "F_max": F_max, "feasible": bool(ok and F_max > F_min)})
    return out


def in_refined(iv_sim, sel):
    """Is the selected (F, s) inside the refined simulated feasible interval at that s level?"""
    if sel is None:
        return None
    v = iv_sim[sel["s_idx"]]
    return bool(v["feasible"] and v["F_min"] <= sel["F"] <= v["F_max"])


def alt_rows(rows, frag, key, T_max_levels):
    """Rows relabelled with the damage definition `key` of the P5 sensitivity grid (deadline success recomputed)."""
    out = []
    for r in rows:
        fr = dict(r["fragility"][frag]); dmg = fr["damage_sensitivity"][key]
        fr["damage_primary"] = dmg
        phys = (not r["slip"]) and (not dmg)
        fr["physical_safe"] = phys; fr["phys_success"] = phys and r["completed"]
        for lvl, T_max in T_max_levels.items():
            fr[f"deadline_success_{lvl}"] = bool(phys and r["completed"] and r["T_carry"] <= T_max)
        rr = dict(r); rr["fragility"] = {frag: fr}; out.append(rr)
    return out


def with_deadline(iv, T_nom, T_max):
    """Same refined boundaries under another deadline (boundaries do not depend on the urgency; only the flags do)."""
    out = []
    for v in iv:
        ok = T_nom * v["s"] <= T_max
        w = dict(v); w["deadline_ok"] = bool(ok); w["feasible"] = bool(ok and v["F_min"] is not None and v["F_max"] is not None and v["F_max"] > v["F_min"])
        out.append(w)
    return out


def _refine_worker(args):
    d, frag, Fs, ss, T_nom, T_max = args
    sw = load(d)
    return str(d), frag, fs.refined_intervals(sw["rows"], sw["probes"], Fs, ss, frag, "relaxed", T_nom, T_max)


# ----------------------------------------------------------------------------------- P4 execution (simulator)
def _exec_worker(args):
    obj_dir, jobs = args
    os.environ.setdefault("MUJOCO_GL", "egl")
    from envs.feasibility_carry.env import FeasibilityCarry, CarryConfig
    from envs.feasibility_carry.sweep import run_branch
    from envs.feasibility_carry.refine import load_states
    out = pathlib.Path(obj_dir)
    S = json.loads((out / "summary.json").read_text()); meta = S["meta"]
    cfg = CarryConfig(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in meta["config"].items()})
    env = FeasibilityCarry(cfg, seed=meta["seed"]); env.reset()
    states = load_states(out, meta["repeats"])
    frag = {k: float(v) for k, v in meta["label_settings"]["fragility"].items()}
    cache_path = out / "p4_exec.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    for key, (F, s) in jobs.items():
        if key in cache:
            continue
        infos = [run_branch(env, saved, F, s, meta["T_max"], frag)[0] for saved in states]
        rec = {"F": F, "s": s, "n": len(infos), "slip": float(np.mean([i["slip"] for i in infos])),
               "completed": float(np.mean([i["completed"] for i in infos]))}
        for fr in frag:
            rec[f"damage|{fr}"] = float(np.mean([i["fragility"][fr]["damage_primary"] for i in infos]))
            for urg in meta["T_max"]:
                rec[f"deadline_success|{fr}|{urg}"] = float(np.mean([i["fragility"][fr][f"deadline_success_{urg}"] for i in infos]))
        cache[key] = rec
        cache_path.write_text(json.dumps(cache, indent=1))
    return obj_dir, cache


def execute_selections(jobs_by_obj, workers):
    """jobs_by_obj: {obj_dir: {key: (F, s)}} → {obj_dir: cache}. Runs the pending ones with R paired repeats."""
    results = {}
    todo = [(d, j) for d, j in jobs_by_obj.items() if j]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for d, cache in ex.map(_exec_worker, todo):
            results[d] = cache
    return results


# ------------------------------------------------------------------------------------------------ figures
def fig_boundaries(name, sw, refined, base, f_tols, out):
    Fs, ss = sw["meta"]["F"], sw["meta"]["s"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for ax, frag in zip(axes, FRAGS):
        iv = refined[(frag, "relaxed")]; x = np.arange(len(ss))
        for j in range(len(ss)):
            if iv[j]["feasible"]:
                ax.add_patch(plt.Rectangle((j - 0.35, iv[j]["F_min"]), 0.7, iv[j]["F_max"] - iv[j]["F_min"], color=COL["sel"], alpha=0.15, lw=0))
        for key, col, lab in (("slip_fit", COL["slip"], "sim slip boundary F_min (50 % crossing, 95 % CI)"),
                              ("damage_fit", COL["damage"], "sim damage boundary F_max (primary proxy)")):
            y = np.array([iv[j][key]["F50"] if iv[j][key] else np.nan for j in range(len(ss))], float)
            lo = np.array([iv[j][key]["ci95"][0] if iv[j][key] and iv[j][key]["ci95"][0] is not None else np.nan for j in range(len(ss))], float)
            hi = np.array([iv[j][key]["ci95"][1] if iv[j][key] and iv[j][key]["ci95"][1] is not None else np.nan for j in range(len(ss))], float)
            ax.errorbar(x, y, yerr=[np.clip(y - lo, 0, None), np.clip(hi - y, 0, None)], fmt="o-", color=col, lw=2, ms=6, capsize=3, label=lab)
        for rule, ls in (("B5a", "--"), ("B5b", ":")):
            for setting, lw in (("oracle", 2), ("coarse-cal", 1)):
                b = base.get((frag, rule, setting))
                if b is None:
                    continue
                ax.plot(x, [v["F_min"] for v in b], ls=ls, lw=lw, color=COL[rule], label=f"{rule} {setting}: F_min")
                ax.plot(x, [v["F_max"] for v in b], ls=ls, lw=lw, color=COL[rule], alpha=0.6, label=f"{rule} {setting}: F_max")
        for j, s in enumerate(ss):
            if sw["meta"]["T_nominal"] * s > sw["meta"]["T_max"]["relaxed"]:
                ax.axvspan(j - 0.5, j + 0.5, color="k", alpha=0.08, lw=0)
        ax.set_xticks(x); ax.set_xticklabels([f"{s:g}" for s in ss]); ax.set_xlabel("s (carry time scaling)")
        ax.set_ylabel("F [N]"); ax.set_ylim(0, 75); ax.grid(alpha=0.25)
        ax.set_title(f"{name}, {frag} (f_tol = {f_tols[frag]:.1f} N)", fontsize=10); ax.legend(fontsize=6, loc="upper right")
    m, mu, d = props(sw)
    fig.suptitle(f"{name}: m = {m:.3f} kg, μ = {mu:.3f}, d = {d*1e3:.1f} mm — refined boundaries vs analytic rules (grey = fails relaxed deadline; green = refined feasible interval)", fontsize=9)
    fig.savefig(out / f"boundaries_{name}.png", dpi=120, bbox_inches="tight"); plt.close(fig)


def fig_heatmaps(name, sw, ctxs, f_tols, out):
    Fs, ss = sw["meta"]["F"], sw["meta"]["s"]
    fig, axes = plt.subplots(2, 4, figsize=(15, 8)); fig.subplots_adjust(hspace=0.45, wspace=0.35)
    for r, frag in enumerate(FRAGS):
        panels = [("deadline_success", "P(deadline success), relaxed", ctxs[(frag, "relaxed")]),
                  ("deadline_success", "P(deadline success), urgent", ctxs[(frag, "urgent")]),
                  ("slip", "P(slip)", ctxs[(frag, "relaxed")]), ("damage", f"P(damage, primary, f_tol = {f_tols[frag]:.1f} N)", ctxs[(frag, "relaxed")])]
        for cidx, (key, title, ctx) in enumerate(panels):
            ax = axes[r, cidx]
            im = ax.imshow(ctx["P"][key], origin="lower", cmap="Blues", vmin=0, vmax=1, aspect="auto")
            ax.set_xticks(range(len(ss))); ax.set_xticklabels([f"{s:g}" for s in ss], fontsize=8)
            ax.set_yticks(range(len(Fs))); ax.set_yticklabels([f"{F:.1f}" for F in Fs], fontsize=8)
            ax.set_xlabel("s"); ax.set_ylabel("F [N]"); ax.set_title(f"{frag}: {title}", fontsize=9)
            for i in range(len(Fs)):
                for j in range(len(ss)):
                    if ctx["feasible"][i, j]:
                        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec=COL["sel"], lw=1.5))
            for j, ok in enumerate(ctx["deadline_ok"]):
                if not ok:
                    ax.axvspan(j - 0.5, j + 0.5, color="k", alpha=0.08, lw=0)
            t = ctx["teacher"]
            if t is not None and key == "deadline_success":
                ax.plot(t["s_idx"], t["F_idx"], marker="*", ms=12, color=COL["sel"], mec="k")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, label="empirical rate over repeats")
    m, mu, d = props(sw)
    fig.suptitle(f"{name}: m = {m:.3f} kg, μ = {mu:.3f}, d = {d*1e3:.1f} mm — green = feasible cell, grey = fails deadline, ★ = grid teacher", fontsize=9)
    fig.savefig(out / f"heatmaps_{name}.png", dpi=120, bbox_inches="tight"); plt.close(fig)


def fig_diagnostics(diag, p4rows, out):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    ax = axes[0]
    for urg, col in (("relaxed", COL["slip"]), ("urgent", COL["damage"])):
        v = [d["shift"] for d in diag["slip_shift"][urg] if d["shift"] is not None]
        if v:
            ax.hist(v, bins=np.arange(-4, 20, 1), alpha=0.6, color=col, label=f"{urg}: n = {len(v)}")
    ax.axvline(P2_SHIFT_N, color="k", ls="--", lw=1, label="P2 threshold 9.75 N"); ax.set_xlabel("slip-boundary shift F_min(0.5) − F_min(slowest deadline-ok s) [N]"); ax.set_ylabel("objects"); ax.legend(fontsize=8); ax.grid(alpha=0.25)
    ax = axes[1]
    for urg, col in (("relaxed", COL["slip"]), ("urgent", COL["damage"])):
        v = [d["rel"] for d in diag["slip_shift"][urg] if d["rel"] is not None]
        if v:
            ax.hist(v, bins=np.arange(-0.1, 0.5, 0.025), alpha=0.6, color=col, label=urg)
    ax.set_xlabel("relative shift (shift / F_min at s = 0.5)"); ax.legend(fontsize=8); ax.grid(alpha=0.25)
    ax = axes[2]
    w = [d["widest_window"] for d in diag["p3_windows"] if d["widest_window"] is not None]
    if w:
        ax.hist(w, bins=np.arange(0, 40, 2), color=COL["sel"], alpha=0.7)
    ax.set_xlabel("P3 recovery window width at the best slower deadline-ok s [N]"); ax.set_ylabel("contexts"); ax.grid(alpha=0.25)
    fig.suptitle("Post-hoc diagnostics (not gates): slip-boundary shifts across the 48 objects and P3 recovery-window widths", fontsize=9)
    fig.savefig(out / "diagnostics_shift_window.png", dpi=120, bbox_inches="tight"); plt.close(fig)
    # P4 boundary errors
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, bt in zip(axes, ("F_min", "F_max")):
        labels_, data = [], []
        for rule in ("B5a", "B5b"):
            for setting in ("oracle", "coarse-uncal", "coarse-cal"):
                e = [r[f"err_{bt}"] for r in p4rows if r["rule"] == rule and r["setting"] == setting and r.get(f"err_{bt}") is not None]
                labels_.append(f"{rule}\n{setting}"); data.append(e if e else [np.nan])
        ax.boxplot(data, tick_labels=labels_); ax.set_ylabel(f"|{bt} rule − {bt} sim| [N]"); ax.grid(alpha=0.25)
        ax.set_title(f"held-out boundary error, {bt}", fontsize=10)
    fig.savefig(out / "p4_boundary_errors.png", dpi=120, bbox_inches="tight"); plt.close(fig)


# ------------------------------------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweeps", required=True); ap.add_argument("--objects", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--no_exec", action="store_true"); ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--figs", choices=["all", "some", "none"], default="all")
    args = ap.parse_args(argv)
    out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)
    OBJ = json.loads(pathlib.Path(args.objects).read_text())
    objs = {o["object_id"]: o for o in OBJ["objects"]}
    sweeps = {}
    for oid in objs:
        d = pathlib.Path(args.sweeps) / oid
        if (d / "summary.json").exists() and (d / "probes.json").exists():
            sweeps[oid] = load(d)
    missing = [oid for oid in objs if oid not in sweeps]
    lines = ["# Phase 1 — full sweep evaluation (plan §6)\n",
             f"Generated by analysis/phase1_full.py from {args.sweeps} ({len(sweeps)} of {len(objs)} objects present"
             + (f"; missing: {', '.join(missing)}" if missing else "") + "). Only measured numbers; empty cells = not measured.\n"]
    summary = {"n_objects": len(sweeps), "missing": missing, "checks": {}}
    sw0 = next(iter(sweeps.values())); Fs, ss = sw0["meta"]["F"], sw0["meta"]["s"]
    T_nom, T_max = sw0["meta"]["T_nominal"], sw0["meta"]["T_max"]
    j_slow = {urg: max(j for j, s in enumerate(ss) if T_nom * s <= T_max[urg]) for urg in URGS}
    lines.append(f"Grid: F = {Fs} N; s = {ss}; R = {sw0['meta']['repeats']}; T_carry(s₀) = {T_nom} s; T_max = {T_max} "
                 f"(slowest deadline-ok s: relaxed {ss[j_slow['relaxed']]:g}, urgent {ss[j_slow['urgent']]:g}). τ = {fs.TAU}, γ = {fs.GAMMA}; "
                 f"δ_slip = {L.DELTA_SLIP*1e3:g} mm; primary damage proxy = sustained {L.T_WIN_PRIMARY} per-pad normal force > f_tol. "
                 f"Objects: {OBJ['per_cell']} per cell, seed {OBJ['seed']}, bands {BANDS}.\n")

    # ---------------- per-object tables: contexts, refined intervals, accelerations
    ctx_all, ref_all, acc_all, ftol_all = {}, {}, {}, {}
    for oid, sw in sweeps.items():
        meta, rows, probes = sw["meta"], sw["rows"], sw["probes"]
        ftol_all[oid] = {k: float(v) for k, v in meta["label_settings"]["fragility"].items()}
        acc_all[oid] = accel_by_s(sw)
        for frag in FRAGS:
            for urg in URGS:
                ctx_all[(oid, frag, urg)] = fs.context_tables(rows, Fs, ss, frag, urg, T_nom, T_max[urg])
    with ProcessPoolExecutor(max_workers=args.workers) as ex:          # logistic fits with 300 bootstraps: ≈ 15 s per (object, fragility)
        for d, frag, iv in ex.map(_refine_worker, [(sw["dir"], frag, Fs, ss, T_nom, T_max["relaxed"]) for sw in sweeps.values() for frag in FRAGS]):
            oid = pathlib.Path(d).name
            for urg in URGS:
                ref_all[(oid, frag, urg)] = with_deadline(iv, T_nom, T_max[urg])
    total_branches = sum(len(sw["rows"]) for sw in sweeps.values()); total_probes = sum(len(sw["probes"]) for sw in sweeps.values())
    wall = np.mean([r["wall_s"] for sw in sweeps.values() for r in sw["rows"]])
    lines.append(f"Branches: {total_branches} grid + {total_probes} probes; mean wall per branch {wall:.2f} s.\n")

    # ---------------- object table
    lines.append("## Objects\n")
    lines.append("| object | split | m [kg] | μ | d [mm] | f_tol fragile / rigid [N] | a_lift / a_lat at s = 0.5 [m/s²] | P1 slope | P1 R² | finger pos [mm] |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    p1 = {}
    for oid, sw in sweeps.items():
        rows = sw["rows"]; m, mu, d = props(sw)
        cmd = np.array([r["F"] for r in rows for _ in (0, 1)]); meas = np.array([v for r in rows for v in (r["ss_f_left"], r["ss_f_right"])])
        A = np.vstack([cmd, np.ones_like(cmd)]).T; (a, b), *_ = np.linalg.lstsq(A, meas, rcond=None)
        r2 = 1 - ((meas - (a * cmd + b)) ** 2).sum() / ((meas - meas.mean()) ** 2).sum()
        per_level = [np.mean([v for r in rows if r["F_idx"] == i for v in (r["ss_f_left"], r["ss_f_right"])]) for i in range(len(Fs))]
        mono = bool(np.all(np.diff(per_level) > 0))
        q = [r["ss_q1"] * 1e3 for r in rows]
        p1[oid] = {"slope": float(a), "intercept": float(b), "r2": float(r2), "monotone": mono, "q_min_mm": float(min(q)), "q_max_mm": float(max(q))}
        lines.append(f"| {oid} | {objs[oid]['split']} | {m:.3f} | {mu:.3f} | {d*1e3:.1f} | {ftol_all[oid]['fragile']:.1f} / {ftol_all[oid]['rigid']:.1f} | "
                     f"{acc_all[oid][0]['a_lift']:.2f} / {acc_all[oid][0]['a_lat']:.2f} | {a:.4f} | {r2:.4f} | {min(q):.1f}–{max(q):.1f} |")

    # ---------------- P1
    r2s = [v["r2"] for v in p1.values()]
    p1_pass = all(v["r2"] >= P1_R2 and v["monotone"] and v["q_min_mm"] > 0.5 for v in p1.values())
    lines.append(f"\n## P1 — force control is real: **{'PASS' if p1_pass else 'FAIL'}**\n")
    lines.append(f"- R² of measured steady-state pad force vs commanded F over 8 levels × 30 branches × 2 pads, per object: min {min(r2s):.4f}, max {max(r2s):.4f} (criterion ≥ {P1_R2}); "
                 f"slope range {min(v['slope'] for v in p1.values()):.3f}–{max(v['slope'] for v in p1.values()):.3f}; monotone per-level means on {sum(v['monotone'] for v in p1.values())}/{len(p1)} objects; "
                 f"finger position ≥ {min(v['q_min_mm'] for v in p1.values()):.1f} mm from the 0 mm joint limit on every branch (criterion: not at the limit).")
    summary["checks"]["P1"] = {"pass": p1_pass, "r2_min": min(r2s), "r2_max": max(r2s), "per_object": p1}

    # ---------------- P2
    lines.append("\n## P2 — s moves a boundary (≥ 9.75 N between s = 0.5 and the slowest deadline-ok s, non-overlapping 95 % CI ∪ bisection bracket)\n")
    p2 = {urg: [] for urg in URGS}
    for urg in URGS:
        lines.append(f"\n### {urg} deadline (s = 0.5 vs s = {ss[j_slow[urg]]:g})\n")
        lines.append("| object | slip F_min(0.5) [unc.] | slip F_min(slow) [unc.] | slip shift | fragile F_max(0.5) | fragile F_max(slow) | shift | rigid F_max(0.5) | rigid F_max(slow) | shift | P2 on this object |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for oid, sw in sweeps.items():
            js = j_slow[urg]; br = sw["brackets"]; cells = []; passed = []
            iv = ref_all[(oid, "rigid", urg)]
            def entry(fit0, fit1, u0, u1):
                if fit0 is None or fit1 is None:
                    return None, None, None
                shift = fit0["F50"] - fit1["F50"]
                sep = (u0[1] < u1[0]) or (u1[1] < u0[0])
                return shift, sep, (abs(shift) >= P2_SHIFT_N and sep)
            # slip
            f0, f1 = iv[0]["slip_fit"] if iv[0]["slip_status"] == "crossing" else None, iv[js]["slip_fit"] if iv[js]["slip_status"] == "crossing" else None
            u0, u1 = uncertainty(f0, br.get("slip", {}).get("0")), uncertainty(f1, br.get("slip", {}).get(str(js)))
            sh, sep, ok = entry(f0, f1, u0, u1)
            rec = {"object": oid, "slip": {"F0": None if f0 is None else f0["F50"], "F1": None if f1 is None else f1["F50"], "u0": u0, "u1": u1, "shift": sh, "separated": sep, "pass": ok}}
            fmt_u = lambda f, u: "" if f is None else f"{f['F50']:.1f} [{u[0]:.1f}, {u[1]:.1f}]"
            cells += [fmt_u(f0, u0), fmt_u(f1, u1), "" if sh is None else f"{sh:+.1f}{' ✓' if ok else ''}"]
            passed.append(bool(ok))
            for frag in FRAGS:
                ivf = ref_all[(oid, frag, urg)]
                d0 = ivf[0]["damage_fit"] if ivf[0]["damage_status"] == "crossing" else None
                d1 = ivf[js]["damage_fit"] if ivf[js]["damage_status"] == "crossing" else None
                v0, v1 = uncertainty(d0, br.get("damage", {}).get(f"{frag}|0")), uncertainty(d1, br.get("damage", {}).get(f"{frag}|{js}"))
                shd, sepd, okd = entry(d0, d1, v0, v1)
                rec[f"damage_{frag}"] = {"F0": None if d0 is None else d0["F50"], "F1": None if d1 is None else d1["F50"], "u0": v0, "u1": v1, "shift": shd, "separated": sepd, "pass": okd}
                cells += [fmt_u(d0, v0), fmt_u(d1, v1), "" if shd is None else f"{shd:+.1f}{' ✓' if okd else ''}"]
                passed.append(bool(okd))
            rec["pass"] = any(passed); p2[urg].append(rec)
            lines.append(f"| {oid} | " + " | ".join(cells) + f" | {'✓' if rec['pass'] else '✗'} |")
        n_pass = sum(r["pass"] for r in p2[urg])
        lines.append(f"\n{urg}: **{n_pass} of {len(p2[urg])} objects** show a boundary shift ≥ {P2_SHIFT_N} N with separated uncertainty intervals (criterion: ≥ 2 property combinations).")
    p2_pass = {urg: sum(r["pass"] for r in p2[urg]) >= 2 for urg in URGS}
    lines.append(f"\n**P2: {'PASS' if p2_pass['relaxed'] else 'FAIL'}** under the relaxed deadline (primary; the demo reports used it), "
                 f"{'PASS' if p2_pass['urgent'] else 'FAIL'} under the urgent deadline. Objects passing: relaxed {sum(r['pass'] for r in p2['relaxed'])}, urgent {sum(r['pass'] for r in p2['urgent'])}.")
    summary["checks"]["P2"] = {"pass": p2_pass["relaxed"], "pass_by_urgency": p2_pass, "n_pass": {u: int(sum(r["pass"] for r in p2[u])) for u in URGS}, "threshold_N": P2_SHIFT_N, "rows": p2}

    # ---------------- P3
    lines.append("\n## P3 — slow-down recovery region (reference speed s = 0.5; grid feasibility under the primary proxy at f_tol × 1.0)\n")
    lines.append("| object | fragility | urgency | feasible cells | at s = 0.5 | recovery s levels (deadline-ok) | s_rec bracket | refined window at s = 0.5 [N] | widest refined window at a slower deadline-ok s [N] (s) | grid level inside | status |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    p3 = []
    for oid, sw in sweeps.items():
        for frag in FRAGS:
            for urg in URGS:
                ctx, iv = ctx_all[(oid, frag, urg)], ref_all[(oid, frag, urg)]
                feas = ctx["feasible"]; dl = ctx["deadline_ok"]
                any_feas = bool((feas & dl[None, :]).any())
                at_ref = bool(feas[:, 0].any())
                rec_levels = [j for j in range(1, len(ss)) if dl[j] and feas[:, j].any()]
                status = "infeasible by design" if not any_feas else ("feasible at reference" if at_ref else ("recovery" if rec_levels else "feasible only past the deadline"))
                win = lambda v: (v["F_max"] - v["F_min"]) if (v["F_min"] is not None and v["F_max"] is not None) else None
                w0 = win(iv[0]); slower = [v for v in iv[1:] if v["deadline_ok"] and win(v) is not None]
                best = max(slower, key=win) if slower else None
                inside = ""
                if best is not None and win(best) > 0:
                    inside = ", ".join(f"{F:.1f}" for F in Fs if best["F_min"] <= F <= best["F_max"]) or "none"
                srec = sw["brackets"].get("s_rec", {}).get(f"{frag}|{urg}", {})
                srec_s = f"[{srec['lo']:.2f}, {srec['hi']:.2f}] at F = {srec['F']:.1f}" if srec.get("status") == "bracketed" else srec.get("status", "")
                p3.append({"object": oid, "frag": frag, "urg": urg, "status": status, "n_feasible": int(feas.sum()), "recovery_levels": [ss[j] for j in rec_levels],
                           "window_ref": w0, "widest_window": None if best is None else win(best), "widest_s": None if best is None else best["s"], "grid_inside": inside, "s_rec": srec})
                lines.append(f"| {oid} | {frag} | {urg} | {int(feas.sum())} | {'✓' if at_ref else '✗'} | {', '.join(f'{ss[j]:g}' for j in rec_levels) or '—'} | {srec_s} | "
                             f"{'' if w0 is None else f'{w0:.1f}'} | {'' if best is None else f'{win(best):.1f} ({best['s']:g})'} | {inside} | {status} |")
    rec_objs = sorted({r["object"] for r in p3 if r["status"] == "recovery"})
    n_design = len([r for r in p3 if r["status"] == "infeasible by design"])
    p3_pass = len(rec_objs) >= 2
    lines.append(f"\n**P3: {'PASS' if p3_pass else 'FAIL'}** — {len(rec_objs)} property combinations (objects) show a recovery in at least one context "
                 f"(criterion ≥ 2): {', '.join(rec_objs) or '—'}. Contexts: {len(p3)} total, {n_design} infeasible by design (excluded), "
                 f"{len([r for r in p3 if r['status']=='recovery'])} recovery, {len([r for r in p3 if r['status']=='feasible at reference'])} feasible at the reference speed, "
                 f"{len([r for r in p3 if r['status']=='feasible only past the deadline'])} feasible only past the deadline.")
    summary["checks"]["P3"] = {"pass": p3_pass, "recovery_objects": rec_objs, "n_contexts": len(p3), "n_infeasible_by_design": n_design, "rows": p3}

    # ---------------- P4 (formal; held-out test-combos objects)
    lines.append("\n## P4 — analytic rules under realistic information (held-out test-combos objects)\n")
    train = [oid for oid in sweeps if objs[oid]["split"] == "train"]; test = [oid for oid in sweeps if objs[oid]["split"] != "train"]
    # calibrated per-bin point estimates from the training split (mean of true values)
    cal = {attr: {} for attr in ATTR_KEYS}
    for key in ATTR_KEYS["mass"]:
        cal["mass"][key] = float(np.mean([objs[o]["mass"] for o in train if objs[o]["bins"]["mass"] == key]))
    for key in ATTR_KEYS["friction"]:
        cal["friction"][key] = float(np.mean([objs[o]["mu"] for o in train if objs[o]["bins"]["friction"] == key]))
    for key in ATTR_KEYS["balance"]:
        cal["balance"][key] = float(np.mean([objs[o]["com_frac"] * 0.04 for o in train if objs[o]["bins"]["balance"] == key]))
    for key in ATTR_KEYS["fragility"]:
        cal["fragility"][key] = float(np.mean([objs[o]["f_tol"][key] for o in train]))
    uncal = {attr: dict(B.COARSE_BINS[attr]) for attr in ATTR_KEYS}
    est_from = lambda table, b: (table["mass"][b["mass"]], table["friction"][b["friction"]], table["fragility"][b["fragility"]], table["balance"][b["balance"]])
    # noise realisation (one draw per (object, fragility); logged)
    noise = {}
    for k, oid in enumerate(sorted(sweeps)):
        for fi, frag in enumerate(FRAGS):
            rng = np.random.default_rng([NOISE_SEED, k, fi])
            noise[(oid, frag)] = noisy_bins(objs[oid]["bins"], frag, rng)
    n_flips = sum(len(v[1]) for v in noise.values())
    lines.append(f"Training split: {len(train)} objects; held-out test-combos: {len(test)} objects ({len(test)*4} contexts). Coarse bins carry ε = {NOISE_EPS} noise "
                 f"(seed {NOISE_SEED}; {n_flips} attribute flips over {len(noise)*4} attribute draws). Calibrated per-bin estimates (training-split means): "
                 + "; ".join(f"{attr}: " + ", ".join(f"{k} = {v:.3f}" for k, v in cal[attr].items()) for attr in cal) + ".\n")
    # calibration of the multipliers on the training contexts with a non-empty refined feasible set
    train_ctx = [(oid, frag, urg) for oid in train for frag in FRAGS for urg in URGS if any(v["feasible"] for v in ref_all[(oid, frag, urg)])]
    def fit_multipliers(rule):
        best = None
        r_grid = CAL_GRID["r_eff"] if rule == "B5b" else [B.R_EFF]
        for c_min, c_max, r_eff in itertools.product(CAL_GRID["c_min"], CAL_GRID["c_max"], r_grid):
            hits = 0
            for (oid, frag, urg) in train_ctx:
                est = est_from(cal, noise[(oid, frag)][0])
                ivr = rule_intervals(rule, est, acc_all[oid], ss, T_nom, T_max[urg], c_min, c_max, r_eff)
                sel, _ = fs.select_refined(ivr)
                hits += bool(in_refined(ref_all[(oid, frag, urg)], sel))
            key = (hits, -abs(np.log(c_min)) - abs(np.log(c_max)))
            if best is None or key > best[0]:
                best = (key, {"c_min": float(c_min), "c_max": float(c_max), "r_eff": float(r_eff), "train_success": hits / len(train_ctx)})
        return best[1]
    mult = {rule: fit_multipliers(rule) for rule in ("B5a", "B5b")}
    lines.append("Calibrated multipliers (fitted on training contexts to maximise the fraction of selections inside the refined feasible interval; ties toward 1): "
                 + "; ".join(f"{rule}: c_min = {m['c_min']:.2f}, c_max = {m['c_max']:.2f}, r_eff = {m['r_eff']*1e3:.1f} mm, training success {m['train_success']:.3f}" for rule, m in mult.items()) + ".\n")
    summary["checks"]["P4"] = {"calibration": {"per_bin": cal, "multipliers": mult, "n_train_contexts": len(train_ctx)},
                               "noise": {f"{k[0]}|{k[1]}": {"bins": v[0], "flips": {a: list(f) for a, f in v[1].items()}} for k, v in noise.items()}}
    # held-out evaluation
    W_all = [v["F_max"] - v["F_min"] for oid in test for frag in FRAGS for v in ref_all[(oid, frag, "relaxed")] if v["feasible"]]
    W_med = float(np.median(W_all)) if W_all else None
    p4rows, sel_rows, jobs = [], [], {}
    settings = ("oracle", "coarse-uncal", "coarse-cal")
    base_fig = {}
    for oid in test:
        m, mu, d = props(sweeps[oid])
        for frag in FRAGS:
            for setting in settings:
                for rule in ("B5a", "B5b"):
                    if setting == "oracle":
                        est, mm = (m, mu, ftol_all[oid][frag], d), {"c_min": 1.0, "c_max": 1.0, "r_eff": B.R_EFF}
                    elif setting == "coarse-uncal":
                        est, mm = est_from(uncal, noise[(oid, frag)][0]), {"c_min": 1.0, "c_max": 1.0, "r_eff": B.R_EFF}
                    else:
                        est, mm = est_from(cal, noise[(oid, frag)][0]), mult[rule]
                    ivr_rel = rule_intervals(rule, est, acc_all[oid], ss, T_nom, T_max["relaxed"], mm["c_min"], mm["c_max"], mm["r_eff"])
                    base_fig[(oid, frag, rule, setting)] = ivr_rel
                    iv_sim = ref_all[(oid, frag, "relaxed")]
                    for j in range(len(ss)):
                        if not iv_sim[j]["deadline_ok"]:
                            continue
                        e_min = abs(ivr_rel[j]["F_min"] - iv_sim[j]["slip_fit"]["F50"]) if iv_sim[j]["slip_status"] == "crossing" else None
                        e_max = abs(ivr_rel[j]["F_max"] - iv_sim[j]["damage_fit"]["F50"]) if iv_sim[j]["damage_status"] == "crossing" else None
                        p4rows.append({"object": oid, "frag": frag, "s": ss[j], "rule": rule, "setting": setting, "err_F_min": e_min, "err_F_max": e_max})
                    for urg in URGS:
                        ivr = rule_intervals(rule, est, acc_all[oid], ss, T_nom, T_max[urg], mm["c_min"], mm["c_max"], mm["r_eff"])
                        sel, _ = fs.select_refined(ivr)
                        truth_nonempty = any(v["feasible"] for v in ref_all[(oid, frag, urg)])
                        row = {"object": oid, "frag": frag, "urg": urg, "rule": rule, "setting": setting, "truth_nonempty": truth_nonempty,
                               "sel": None if sel is None else {"F": sel["F"], "s": sel["s"], "s_idx": sel["s_idx"]},
                               "in_refined": in_refined(ref_all[(oid, frag, urg)], sel)}
                        if sel is not None:
                            key = f"{sel['F']:.3f}|{sel['s']:.4f}"
                            jobs.setdefault(str(sweeps[oid]["dir"]), {})[key] = (float(sel["F"]), float(sel["s"]))
                            row["exec_key"] = key
                        sel_rows.append(row)
    # execute selections in the simulator (R repeats, same branch states)
    exec_res = {}
    if not args.no_exec and jobs:
        exec_res = execute_selections(jobs, args.workers)
    for row in sel_rows:
        row["executed"] = None
        if row["sel"] is None:
            row["exec_infeasible"] = True if row["truth_nonempty"] else None    # no selection: fails if something was feasible
            continue
        cache = exec_res.get(str(sweeps[row["object"]]["dir"]), {})
        e = cache.get(row["exec_key"])
        if e is None:
            row["exec_infeasible"] = None; continue
        P_d, P_s, P_dm = e[f"deadline_success|{row['frag']}|{row['urg']}"], e["slip"], e[f"damage|{row['frag']}"]
        row["executed"] = {"deadline_success": P_d, "slip": P_s, "damage": P_dm, "n": e["n"]}
        row["exec_infeasible"] = not (P_d >= fs.TAU["c"] and P_s <= fs.TAU["s"] and P_dm <= fs.TAU["d"])
    # table
    lines.append(f"Median refined feasible-set width on held-out contexts (relaxed deadline, non-empty levels): W = {fmt(W_med)} N → boundary-error criterion W/4 = {fmt(None if W_med is None else W_med/4, 2)} N. "
                 "Boundary errors are measured per (object, fragility, deadline-ok s) against the 50 % logistic crossings where the simulated boundary is a crossing.\n")
    lines.append("| rule | setting | n boundary pairs | median \\|F_min err\\| [N] | median \\|F_max err\\| [N] | median of both | frac > W/4 | selections (truth non-empty) | inside refined interval | executed infeasible (𝓕 test on R repeats) | no selection | executed infeasible, all held-out contexts |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    p4_summary = {}
    for rule in ("B5a", "B5b"):
        for setting in settings:
            e_min = [r["err_F_min"] for r in p4rows if r["rule"] == rule and r["setting"] == setting and r["err_F_min"] is not None]
            e_max = [r["err_F_max"] for r in p4rows if r["rule"] == rule and r["setting"] == setting and r["err_F_max"] is not None]
            both = e_min + e_max
            sel_ne = [r for r in sel_rows if r["rule"] == rule and r["setting"] == setting and r["truth_nonempty"]]
            sel_all = [r for r in sel_rows if r["rule"] == rule and r["setting"] == setting]
            n_in = sum(bool(r["in_refined"]) for r in sel_ne); n_none = sum(r["sel"] is None for r in sel_ne)
            ex_ne = [r["exec_infeasible"] for r in sel_ne if r["exec_infeasible"] is not None]
            ex_all = [r["exec_infeasible"] for r in sel_all if r["exec_infeasible"] is not None]
            rate_ne = float(np.mean(ex_ne)) if ex_ne else None; rate_all = float(np.mean(ex_all)) if ex_all else None
            frac_w = float(np.mean([e > W_med / 4 for e in both])) if both and W_med else None
            p4_summary[f"{rule}|{setting}"] = {"n_pairs": len(both), "med_err_min": float(np.median(e_min)) if e_min else None, "med_err_max": float(np.median(e_max)) if e_max else None,
                                              "med_err": float(np.median(both)) if both else None, "frac_gt_W4": frac_w, "n_sel_nonempty": len(sel_ne), "n_in_refined": n_in,
                                              "n_no_selection": n_none, "exec_infeasible_rate_nonempty": rate_ne, "n_exec_nonempty": len(ex_ne), "exec_infeasible_rate_all": rate_all, "n_exec_all": len(ex_all)}
            lines.append(f"| {rule} | {setting} | {len(both)} | {fmt(p4_summary[f'{rule}|{setting}']['med_err_min'])} | {fmt(p4_summary[f'{rule}|{setting}']['med_err_max'])} | {fmt(p4_summary[f'{rule}|{setting}']['med_err'])} | "
                         f"{fmt(frac_w, 2)} | {len(sel_ne)} | {n_in}/{len(sel_ne)} | {'' if rate_ne is None else f'{int(sum(ex_ne))}/{len(ex_ne)} = {rate_ne:.2f}'} | {n_none} | {'' if rate_all is None else f'{int(sum(ex_all))}/{len(ex_all)} = {rate_all:.2f}'} |")
    # the stronger calibrated rule: lower executed infeasible-selection rate on held-out contexts with a non-empty truth; tie → lower median error
    def strength(rule):
        s = p4_summary[f"{rule}|coarse-cal"]
        return (s["exec_infeasible_rate_nonempty"] if s["exec_infeasible_rate_nonempty"] is not None else (1 - s["n_in_refined"] / max(1, s["n_sel_nonempty"])), s["med_err"] if s["med_err"] is not None else np.inf)
    stronger = min(("B5a", "B5b"), key=strength)
    st = p4_summary[f"{stronger}|coarse-cal"]
    rate_used = st["exec_infeasible_rate_nonempty"] if st["exec_infeasible_rate_nonempty"] is not None else None
    p4_pass = None
    if rate_used is not None and st["med_err"] is not None and W_med:
        p4_pass = bool(rate_used >= P4_INFEASIBLE_RATE or st["med_err"] > W_med / 4)
    lines.append(f"\nStronger calibrated rule (lower executed infeasible-selection rate; tie → lower median boundary error): **{stronger}-c** — "
                 f"infeasible-selection rate {fmt(rate_used, 2)} (criterion ≥ {P4_INFEASIBLE_RATE}), median boundary error {fmt(st['med_err'])} N (criterion > W/4 = {fmt(None if not W_med else W_med/4, 2)} N).")
    lines.append(f"\n**P4: {'PASS' if p4_pass else ('FAIL' if p4_pass is False else 'not evaluated (selections not executed)')}**")
    summary["checks"]["P4"].update({"pass": p4_pass, "stronger": stronger, "W_median": W_med, "table": p4_summary, "selections": sel_rows, "boundary_rows": p4rows})
    # per-selection listing
    lines.append("\n### Held-out selections (coarse-calibrated rules) and their executed outcomes\n")
    lines.append("| object | frag | urg | truth non-empty | B5a-c (F, s) | inside refined | executed P(dead. succ.) / P(slip) / P(dmg) | B5b-c (F, s) | inside refined | executed |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for oid in test:
        for frag in FRAGS:
            for urg in URGS:
                cells = []
                for rule in ("B5a", "B5b"):
                    r = next(x for x in sel_rows if x["object"] == oid and x["frag"] == frag and x["urg"] == urg and x["rule"] == rule and x["setting"] == "coarse-cal")
                    cells.append("∅" if r["sel"] is None else f"({r['sel']['F']:.1f}, {r['sel']['s']:g})")
                    cells.append("" if r["in_refined"] is None else ("✓" if r["in_refined"] else "✗"))
                    e = r["executed"]; cells.append("" if e is None else f"{e['deadline_success']:.1f} / {e['slip']:.1f} / {e['damage']:.1f}{' ✗' if r['exec_infeasible'] else ''}")
                r0 = next(x for x in sel_rows if x["object"] == oid and x["frag"] == frag and x["urg"] == urg and x["rule"] == "B5a" and x["setting"] == "coarse-cal")
                lines.append(f"| {oid} | {frag} | {urg} | {'✓' if r0['truth_nonempty'] else '✗'} | " + " | ".join(cells) + " |")

    # ---------------- P5
    lines.append("\n## P5 — damage-definition and threshold sensitivity\n")
    N_keys = [f"N|{w}|{m_}" for w in L.T_WIN for m_ in L.F_TOL_MULT]
    L_keys = [f"L|{w}|{k}|x1" for w in L.T_WIN for k in L.KAPPAS]
    mono_viol = {"f_tol": 0, "t_win": 0, "n_f_tol": 0, "n_t_win": 0}
    taus = []
    p5_verdicts = {}
    for key in N_keys + L_keys:
        # P2 (damage part + slip) and P3 under this damage definition, relaxed deadline for P2
        n_p2 = 0; rec_objs_alt = set(); became_infeasible = []
        for oid, sw in sweeps.items():
            rows, probes = sw["rows"], sw["probes"]
            passed = next(r for r in p2["relaxed"] if r["object"] == oid)["slip"]["pass"]
            for frag in FRAGS:
                ra = alt_rows(rows, frag, key, T_max); pa = alt_rows(probes, frag, key, T_max)
                iva = fs.refined_intervals(ra, pa, Fs, ss, frag, "relaxed", T_nom, T_max["relaxed"], n_boot=0)
                js = j_slow["relaxed"]
                d0 = iva[0]["damage_fit"] if iva[0]["damage_status"] == "crossing" else None
                d1 = iva[js]["damage_fit"] if iva[js]["damage_status"] == "crossing" else None
                if d0 and d1 and abs(d0["F50"] - d1["F50"]) >= P2_SHIFT_N:
                    # candidate: refit both boundaries with the bootstrap to assess separation (95 % CI; no bisection
                    # brackets exist for the alternative definitions)
                    fits = []
                    for jj in (0, js):
                        at = [r for r in ra if r["s_idx"] == jj] + [q for q in pa if q.get("s_idx") == jj]
                        held = [r for r in at if not r["slip"]]
                        fits.append(fs.logistic_boundary([r["F"] for r in held], [r["fragility"][frag]["damage_primary"] for r in held], [r["repeat"] for r in held], True))
                    u0, u1 = uncertainty(fits[0], None), uncertainty(fits[1], None)
                    if (u0[1] < u1[0]) or (u1[1] < u0[0]):
                        passed = True
                for urg in URGS:
                    ctxa = fs.context_tables(ra, Fs, ss, frag, urg, T_nom, T_max[urg])
                    feas, dl = ctxa["feasible"], ctxa["deadline_ok"]
                    base_status = next(r for r in p3 if r["object"] == oid and r["frag"] == frag and r["urg"] == urg)["status"]
                    if not (feas & dl[None, :]).any():
                        if base_status != "infeasible by design":
                            became_infeasible.append(f"{oid}|{frag}|{urg}")
                        continue
                    if not feas[:, 0].any() and any(dl[j] and feas[:, j].any() for j in range(1, len(ss))):
                        rec_objs_alt.add(oid)
            n_p2 += bool(passed)
        p5_verdicts[key] = {"P2_n_pass": n_p2, "P2": n_p2 >= 2, "P3_n_objects": len(rec_objs_alt), "P3": len(rec_objs_alt) >= 2, "became_infeasible": became_infeasible}
    # monotonicity of the damage boundary per (object, frag, s) and Kendall τ per context
    for oid, sw in sweeps.items():
        rows, probes = sw["rows"], sw["probes"]
        for frag in FRAGS:
            for j in range(len(ss)):
                at = [r for r in rows if r["s_idx"] == j] + [p for p in probes if p.get("s_idx") == j]
                held = [r for r in at if not r["slip"]]
                F = [r["F"] for r in held]; rep = [r["repeat"] for r in held]
                b = {}
                for w in L.T_WIN:
                    for m_ in L.F_TOL_MULT:
                        y = [r["fragility"][frag]["damage_sensitivity"][f"N|{w}|{m_}"] for r in held]
                        fit = fs.logistic_boundary(F, y, rep, True, n_boot=0) if held else None
                        b[(w, m_)] = None if (fit is None or all(y) or not any(y)) else fit["F50"]
                for w in L.T_WIN:
                    seq = [b[(w, m_)] for m_ in L.F_TOL_MULT]
                    if all(v is not None for v in seq):
                        mono_viol["n_f_tol"] += 1; mono_viol["f_tol"] += int(not (seq[0] <= seq[1] + 0.05 and seq[1] <= seq[2] + 0.05))
                for m_ in L.F_TOL_MULT:
                    seq = [b[(w, m_)] for w in L.T_WIN]
                    if all(v is not None for v in seq):
                        mono_viol["n_t_win"] += 1; mono_viol["t_win"] += int(not (seq[0] <= seq[1] + 0.05))
            for urg in URGS:
                ctx = ctx_all[(oid, frag, urg)]
                if not ctx["feasible"].any():
                    continue
                qN = (ctx["P"]["deadline_success"] - ctx["P"]["slip"] - ctx["P"]["damage"]).ravel()
                qL = (ctx["P"]["deadline_success"] - ctx["P"]["slip"] - ctx["P"]["damage_shear"]).ravel()
                if np.std(qN) > 0 and np.std(qL) > 0:
                    taus.append({"ctx": f"{oid}|{frag}|{urg}", "tau": float(kendalltau(qN, qL).statistic)})
    # pooled τ over all cells of non-empty contexts
    allN, allL = [], []
    for (oid, frag, urg), ctx in ctx_all.items():
        if ctx["feasible"].any():
            allN += list((ctx["P"]["deadline_success"] - ctx["P"]["slip"] - ctx["P"]["damage"]).ravel())
            allL += list((ctx["P"]["deadline_success"] - ctx["P"]["slip"] - ctx["P"]["damage_shear"]).ravel())
    tau_pooled = float(kendalltau(allN, allL).statistic) if allN else None
    tau_vals = [t["tau"] for t in taus]
    verdicts_stable = all(v["P2"] == p2_pass["relaxed"] and v["P3"] == p3_pass for v in p5_verdicts.values())
    p5_pass = bool(mono_viol["f_tol"] == 0 and mono_viol["t_win"] == 0 and verdicts_stable and tau_pooled is not None and tau_pooled >= P5_TAU)
    lines.append(f"- Damage boundary monotone in f_tol (×0.75 ≤ ×1 ≤ ×1.25) on {mono_viol['n_f_tol'] - mono_viol['f_tol']}/{mono_viol['n_f_tol']} (object, fragility, s, t_win) triples with three crossings; "
                 f"monotone in t_win (50 ms ≤ 100 ms) on {mono_viol['n_t_win'] - mono_viol['t_win']}/{mono_viol['n_t_win']} triples (tolerance 0.05 N).")
    lines.append(f"- Kendall τ between q under the primary (N) and shear (L, κ = 1) proxies: pooled over all grid cells of the {len([1 for c in ctx_all.values() if c['feasible'].any()])} non-empty contexts τ = {fmt(tau_pooled, 3)} (criterion ≥ {P5_TAU}); "
                 f"per context: median {fmt(float(np.median(tau_vals)) if tau_vals else None, 3)}, min {fmt(min(tau_vals) if tau_vals else None, 3)}, {sum(t >= P5_TAU for t in tau_vals)}/{len(tau_vals)} contexts ≥ {P5_TAU}.")
    lines.append("- P2 / P3 verdicts under every damage setting (relaxed deadline; objects passing P2 / objects with a recovery; contexts that become infeasible under the setting are listed, not counted):\n")
    lines.append("| setting | P2 objects | P2 | P3 objects | P3 | contexts that become infeasible |")
    lines.append("|---|---|---|---|---|---|")
    for key, v in p5_verdicts.items():
        lines.append(f"| {key} | {v['P2_n_pass']} | {'pass' if v['P2'] else 'fail'} | {v['P3_n_objects']} | {'pass' if v['P3'] else 'fail'} | {len(v['became_infeasible'])} |")
    lines.append(f"\n**P5: {'PASS' if p5_pass else 'FAIL'}** (monotone: {'yes' if mono_viol['f_tol'] == 0 and mono_viol['t_win'] == 0 else 'no'}; P2/P3 verdicts unchanged across settings: {'yes' if verdicts_stable else 'no'}; pooled τ ≥ 0.8: {'yes' if tau_pooled is not None and tau_pooled >= P5_TAU else 'no'}).")
    summary["checks"]["P5"] = {"pass": p5_pass, "monotonicity": mono_viol, "tau_pooled": tau_pooled, "tau_per_context": taus, "verdicts_by_setting": p5_verdicts, "verdicts_stable": verdicts_stable}

    # ---------------- P6
    lines.append("\n## P6 — repeatability (agreement ≥ 0.9 per cell outside the refined boundaries' uncertainty; inside cells reported separately)\n")
    label_fns = {"slip": lambda r, f, u: r["slip"], "completed": lambda r, f, u: r["completed"],
                 "damage_primary": lambda r, f, u: r["fragility"][f]["damage_primary"], "physical_safe": lambda r, f, u: r["fragility"][f]["physical_safe"],
                 "deadline_success": lambda r, f, u: r["fragility"][f][f"deadline_success_{u}"]}
    agree = {k: {"in": [], "out": []} for k in label_fns}
    fail_cells = []
    for oid, sw in sweeps.items():
        rows = sw["rows"]
        bcells = set()
        for frag in FRAGS:
            iv = ref_all[(oid, frag, "relaxed")]
            for j in range(len(ss)):
                spans = []
                u = uncertainty(iv[j]["slip_fit"] if iv[j]["slip_status"] == "crossing" else None, sw["brackets"].get("slip", {}).get(str(j)))
                if u: spans.append(u)
                u = uncertainty(iv[j]["damage_fit"] if iv[j]["damage_status"] == "crossing" else None, sw["brackets"].get("damage", {}).get(f"{frag}|{j}"))
                if u: spans.append(u)
                for i, F in enumerate(Fs):
                    if any(lo <= F <= hi for lo, hi in spans):
                        bcells.add((i, j))
        for i in range(len(Fs)):
            for j in range(len(ss)):
                cell = [r for r in rows if r["F_idx"] == i and r["s_idx"] == j]
                grp = "in" if (i, j) in bcells else "out"
                for k, fn in label_fns.items():
                    for frag in (FRAGS if k not in ("slip", "completed") else [None]):
                        for urg in (URGS if k == "deadline_success" else [None]):
                            vals = np.array([fn(r, frag, urg) for r in cell], float); a = max(vals.mean(), 1 - vals.mean())
                            agree[k][grp].append(a)
                            if grp == "out" and a < P6_AGREE:
                                fail_cells.append({"object": oid, "label": k, "frag": frag, "urg": urg, "F": Fs[i], "s": ss[j], "agreement": float(a),
                                                   "P_slip": float(np.mean([r["slip"] for r in cell])), "P_dropped": float(np.mean([r["dropped"] for r in cell]))})
    lines.append("| label | outside: cells | mean | min | cells < 0.9 | inside: cells | mean | min | cells < 0.9 |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for k, v in agree.items():
        o, b = np.array(v["out"]), np.array(v["in"])
        lines.append(f"| {k} | {len(o)} | {o.mean():.4f} | {o.min():.2f} | {int((o < P6_AGREE).sum())} | {len(b)} | {'' if not len(b) else f'{b.mean():.4f}'} | {'' if not len(b) else f'{b.min():.2f}'} | {'' if not len(b) else int((b < P6_AGREE).sum())} |")
    p6_pass = len(fail_cells) == 0
    lines.append(f"\n**P6: {'PASS' if p6_pass else 'FAIL'}** — {len(fail_cells)} (object, label, cell) entries below 0.9 outside boundary uncertainty"
                 + ("" if not fail_cells else ": " + "; ".join(f"{c['object']} {c['label']}{'' if c['frag'] is None else '/' + c['frag']}{'' if c['urg'] is None else '/' + c['urg']} F = {c['F']:.1f}, s = {c['s']:g}: {c['agreement']:.1f} (P_slip {c['P_slip']:.1f}, P_drop {c['P_dropped']:.1f})" for c in fail_cells[:40]) + (" …" if len(fail_cells) > 40 else "")) + ".")
    if fail_cells:
        n_drop = sum(c["P_slip"] >= 1.0 for c in fail_cells)
        lines.append(f"Diagnostic (not part of the criterion): {n_drop} of the {len(fail_cells)} failing entries are cells where every repeat slipped (P_slip = 1.0), i.e. the disagreement is in where a dropped object lands (`completed`) or whether it is crushed between the closing fingers (`damage_primary`).")
    lines.append(f"Solver settings (logged, unchanged): {sw0['meta']['solver']}; gripper {sw0['meta']['gripper']}; contact {sw0['meta']['contact']}.")
    summary["checks"]["P6"] = {"pass": p6_pass, "n_fail_cells": len(fail_cells), "fail_cells": fail_cells, "agreement_means": {k: {g: float(np.mean(v)) if v else None for g, v in vv.items()} for k, vv in agree.items()}}

    # ---------------- P7 and best single fixed
    sel_grid, sel_ref, sel_detail = [], [], []
    for (oid, frag, urg), ctx in ctx_all.items():
        t = ctx["teacher"]; sel, _ = fs.select_refined(ref_all[(oid, frag, urg)])
        sel_grid.append((f"{oid}|{frag}|{urg}", None if t is None else t["s"])); sel_ref.append((f"{oid}|{frag}|{urg}", None if sel is None else sel["s"]))
        sel_detail.append({"ctx": f"{oid}|{frag}|{urg}", "grid": None if t is None else (t["F"], t["s"]), "refined": None if sel is None else (sel["F"], sel["s"], sel["margin"])})
    r_grid, r_ref = fs.p7_context_dependence(sel_grid, P7_MIN_FRAC), fs.p7_context_dependence(sel_ref, P7_MIN_FRAC)
    lines.append(f"\n## P7 — context dependence of s (≥ {P7_MIN_FRAC:.0%} of feasible contexts select a non-modal s on the refined feasible set)\n")
    for lab, r in (("refined feasible set (criterion)", r_ref), ("grid teacher (reported)", r_grid)):
        hist = {}
        for _, s in r.get("selections", []):
            if s is not None: hist[s] = hist.get(s, 0) + 1
        lines.append(f"- **{lab}:** feasible contexts {r['n_contexts']} of {len(ctx_all)}; modal s = {r.get('modal_s')}; fraction non-modal = {fmt(r.get('frac_non_modal'), 3)}; "
                     f"selected-s histogram: " + ", ".join(f"s = {s:g}: {n}" for s, n in sorted(hist.items())) + f" → {'pass' if r['pass'] else ('fail' if r['pass'] is False else 'n/a')}.")
    p7_pass = bool(r_ref["pass"])
    lines.append(f"\n**P7: {'PASS' if p7_pass else 'FAIL'}**")
    summary["checks"]["P7"] = {"pass": p7_pass, "refined": {k: v for k, v in r_ref.items() if k != "selections"}, "grid": {k: v for k, v in r_grid.items() if k != "selections"}, "selections": sel_detail}
    train_ctxs = [ctx_all[(oid, frag, urg)] for oid in train for frag in FRAGS for urg in URGS]
    bsf = fs.best_single_fixed(train_ctxs)
    test_ctxs = [ctx_all[(oid, frag, urg)] for oid in test for frag in FRAGS for urg in URGS]
    bsf_test = float(np.nanmean([c["P"]["deadline_success"][bsf["F_idx"], bsf["s_idx"]] for c in test_ctxs])) if test_ctxs else None
    oracle_test = float(np.nanmean([np.nanmax(c["P"]["deadline_success"]) for c in test_ctxs])) if test_ctxs else None
    lines.append(f"\n## Best-single-fixed reference (post hoc; plan §6 / §8 R2)\n")
    lines.append(f"- Best single grid cell over the {bsf['n_contexts']} training contexts (both urgency levels): F = {bsf['F']:.1f} N, s = {bsf['s']:g}; mean deadline-constrained safe success {bsf['success']:.3f} vs per-context oracle {bsf['oracle_per_context_success']:.3f}. "
                 f"On the {len(test_ctxs)} held-out contexts the same cell scores {fmt(bsf_test, 3)} vs oracle {fmt(oracle_test, 3)}.")
    summary["best_single_fixed"] = {"train": {k: v for k, v in bsf.items() if k != "per_context"}, "test_success": bsf_test, "test_oracle": oracle_test}

    # ---------------- diagnostics (post hoc, not gates)
    diag = {"slip_shift": {urg: [] for urg in URGS}, "p3_windows": []}
    for urg in URGS:
        for r in p2[urg]:
            s_ = r["slip"]; sh = s_["shift"]
            diag["slip_shift"][urg].append({"object": r["object"], "shift": sh, "rel": None if (sh is None or not s_["F0"]) else sh / s_["F0"], "F0": s_["F0"]})
    for r in p3:
        if r["status"] == "recovery":
            diag["p3_windows"].append({"ctx": f"{r['object']}|{r['frag']}|{r['urg']}", "widest_window": r["widest_window"], "widest_s": r["widest_s"], "grid_inside": r["grid_inside"], "window_ref": r["window_ref"]})
    lines.append("\n## Diagnostics (post hoc; not part of any gate)\n")
    lines.append("Figure: diagnostics_shift_window.png.\n")
    for urg in URGS:
        v = [d["shift"] for d in diag["slip_shift"][urg] if d["shift"] is not None]; rv = [d["rel"] for d in diag["slip_shift"][urg] if d["rel"] is not None]
        if v:
            lines.append(f"- Slip-boundary shift F_min(0.5) − F_min({ss[j_slow[urg]]:g}), {urg} deadline, over {len(v)} objects with both crossings: "
                         f"min {min(v):.1f}, median {np.median(v):.1f}, mean {np.mean(v):.1f}, max {max(v):.1f} N; relative to F_min(0.5): min {min(rv):.3f}, median {np.median(rv):.3f}, max {max(rv):.3f}; "
                         f"{sum(x >= P2_SHIFT_N for x in v)} objects ≥ 9.75 N; {sum(x >= 0.15 * d['F0'] for x, d in zip(v, [d for d in diag['slip_shift'][urg] if d['shift'] is not None]))} objects ≥ 15 % of their own F_min(0.5).")
            top = sorted([d for d in diag["slip_shift"][urg] if d["shift"] is not None], key=lambda d: -d["shift"])[:5]
            lines.append("  largest: " + ", ".join(f"{d['object']} {d['shift']:+.1f} N ({d['rel']:.1%})" for d in top))
    w = [d["widest_window"] for d in diag["p3_windows"] if d["widest_window"] is not None]
    if w:
        lines.append(f"- P3 recovery-window widths (widest refined window at a slower deadline-ok s) over {len(w)} recovery contexts: min {min(w):.1f}, median {np.median(w):.1f}, max {max(w):.1f} N; "
                     f"{sum(d['grid_inside'] not in ('', 'none') for d in diag['p3_windows'])} of them contain a grid F level.")
        lines.append("  contexts: " + "; ".join(f"{d['ctx']}: {d['widest_window']:.1f} N at s = {d['widest_s']:g} (grid: {d['grid_inside'] or '—'})" for d in diag["p3_windows"]))
    else:
        lines.append("- No recovery contexts; no window widths.")
    summary["diagnostics"] = diag

    # ---------------- figures
    if args.figs != "none":
        for oid, sw in sweeps.items():
            if args.figs == "all" or oid in test or oid in rec_objs:
                base = {}
                if oid in test:
                    for frag in FRAGS:
                        for rule in ("B5a", "B5b"):
                            base[(frag, rule, "oracle")] = base_fig[(oid, frag, rule, "oracle")]; base[(frag, rule, "coarse-cal")] = base_fig[(oid, frag, rule, "coarse-cal")]
                else:
                    m, mu, d = props(sw)
                    for frag in FRAGS:
                        for rule in ("B5a", "B5b"):
                            base[(frag, rule, "oracle")] = rule_intervals(rule, (m, mu, ftol_all[oid][frag], d), acc_all[oid], ss, T_nom, T_max["relaxed"])
                refined = {(frag, urg): ref_all[(oid, frag, urg)] for frag in FRAGS for urg in URGS}
                fig_boundaries(oid, sw, refined, base, ftol_all[oid], out)
                if oid in rec_objs or oid in test:
                    fig_heatmaps(oid, sw, {(frag, urg): ctx_all[(oid, frag, urg)] for frag in FRAGS for urg in URGS}, ftol_all[oid], out)
        fig_diagnostics(diag, p4rows, out)

    # ---------------- verdict table
    lines.insert(2, "## Verdicts\n\n| check | result | key number |\n|---|---|---|\n"
                 f"| P1 | {'PASS' if p1_pass else 'FAIL'} | R² min {min(r2s):.4f} |\n"
                 f"| P2 | {'PASS' if p2_pass['relaxed'] else 'FAIL'} | objects with a ≥ 9.75 N separated shift: relaxed {sum(r['pass'] for r in p2['relaxed'])}, urgent {sum(r['pass'] for r in p2['urgent'])} (need ≥ 2) |\n"
                 f"| P3 | {'PASS' if p3_pass else 'FAIL'} | objects with a recovery context: {len(rec_objs)} (need ≥ 2) |\n"
                 f"| P4 | {'PASS' if p4_pass else ('FAIL' if p4_pass is False else 'n/a')} | {stronger}-c: infeasible-selection rate {fmt(rate_used, 2)} (≥ 0.10), median boundary error {fmt(st['med_err'])} N (> {fmt(None if not W_med else W_med/4, 2)}) |\n"
                 f"| P5 | {'PASS' if p5_pass else 'FAIL'} | pooled τ {fmt(tau_pooled, 3)}; monotonicity violations {mono_viol['f_tol']} + {mono_viol['t_win']}; verdicts stable {verdicts_stable} |\n"
                 f"| P6 | {'PASS' if p6_pass else 'FAIL'} | cells < 0.9 outside boundaries: {len(fail_cells)} |\n"
                 f"| P7 | {'PASS' if p7_pass else 'FAIL'} | non-modal fraction {fmt(r_ref.get('frac_non_modal'), 3)} of {r_ref['n_contexts']} feasible contexts (≥ 0.25) |\n")
    summary["verdict"] = {"P1": p1_pass, "P2": p2_pass["relaxed"], "P3": p3_pass, "P4": p4_pass, "P5": p5_pass, "P6": p6_pass, "P7": p7_pass}
    (out / "report.md").write_text("\n".join(lines))
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=lambda o: o.item() if hasattr(o, "item") else (list(o) if isinstance(o, (set, tuple)) else str(o))))
    print("\n".join(lines[:40]))
    print(f"... written {out/'report.md'} and summary.json")


if __name__ == "__main__":
    main()
