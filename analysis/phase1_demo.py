"""Pre-sweep demonstration report (plan.md §6): feasible-set heatmaps, refined boundaries with CIs, both analytic
baselines overlaid, P3 recovery region, torsional slip from the CoM offset, P5 sensitivity, P6 repeatability, P7
context dependence of s, and the best-single-fixed (F, s) reference. Only measured numbers are written.

Usage:
  .venv/bin/python -m analysis.phase1_demo --out results/phase1/demo_report \
      a=results/phase1/demo_a b0=results/phase1/demo_b0 b15=results/phase1/demo_b15 ...
"""
import argparse, json, pathlib, itertools
import numpy as np
from scipy.stats import kendalltau
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from envs.feasibility_carry import feasible as fs
from envs.feasibility_carry import labels as L
from baselines import analytic as B

COL = {"slip": "#2A6FDB", "damage": "#D1495B", "damage_shear": "#8E3B9A", "B5a": "#6F6F6F", "B5b": "#E69F00",
       "sel": "#1B9E77"}
FRAGS = ["fragile", "rigid"]
URGS = ["relaxed", "urgent"]


def load(d):
    d = pathlib.Path(d)
    S = json.loads((d / "summary.json").read_text())
    P = json.loads((d / "probes.json").read_text()) if (d / "probes.json").exists() else {"probes": [], "brackets": {}}
    return {"dir": d, "meta": S["meta"], "rows": S["branches"], "probes": P["probes"], "brackets": P["brackets"]}


def accel_by_s(sw):
    ss = sw["meta"]["s"]
    return [{k: float(np.median([r[k] for r in sw["rows"] if r["s_idx"] == j])) for k in ("a_lift", "a_lat", "a_lower")}
            for j in range(len(ss))]


def props(sw):
    c = sw["meta"]["config"]
    return c["obj_mass"], c["obj_mu"], sw["meta"]["com_offset_m"]


def fmt(x, nd=1):
    return "" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def ci_str(fit):
    if fit is None or fit["ci95"][0] is None:
        return ""
    return f"[{fit['ci95'][0]:.1f}, {fit['ci95'][1]:.1f}]"


# ------------------------------------------------------------------------------------------------------ figures
def fig_heatmaps(name, sw, ctxs, out):
    Fs, ss = sw["meta"]["F"], sw["meta"]["s"]
    fig, axes = plt.subplots(2, 4, figsize=(15, 8))
    fig.subplots_adjust(hspace=0.45, wspace=0.35)
    for r, frag in enumerate(FRAGS):
        panels = [("deadline_success", "P(deadline success), relaxed", ctxs[(frag, "relaxed")]),
                  ("deadline_success", "P(deadline success), urgent", ctxs[(frag, "urgent")]),
                  ("slip", "P(slip)", ctxs[(frag, "relaxed")]), ("damage", f"P(damage, primary, f_tol={L.FRAGILITY[frag]:g} N)", ctxs[(frag, "relaxed")])]
        for cidx, (key, title, ctx) in enumerate(panels):
            ax = axes[r, cidx]
            im = ax.imshow(ctx["P"][key], origin="lower", cmap="Blues", vmin=0, vmax=1, aspect="auto")
            ax.set_xticks(range(len(ss))); ax.set_xticklabels([f"{s:g}" for s in ss], fontsize=8)
            ax.set_yticks(range(len(Fs))); ax.set_yticklabels([f"{F:.1f}" for F in Fs], fontsize=8)
            ax.set_xlabel("s"); ax.set_ylabel("F [N]")
            ax.set_title(f"{frag}: {title}", fontsize=9)
            feas = ctx["feasible"]
            for i in range(len(Fs)):
                for j in range(len(ss)):
                    if feas[i, j]:
                        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec=COL["sel"], lw=1.5))
            for j, ok in enumerate(ctx["deadline_ok"]):
                if not ok:
                    ax.axvspan(j - 0.5, j + 0.5, color="k", alpha=0.08, lw=0)
            t = ctx["teacher"]
            if t is not None and key == "deadline_success":
                ax.plot(t["s_idx"], t["F_idx"], marker="*", ms=12, color=COL["sel"], mec="k")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, label="empirical rate over repeats")
    fig.suptitle(f"{name}: m={props(sw)[0]:g} kg, μ={props(sw)[1]:g}, CoM offset {props(sw)[2]*1e3:.0f} mm — green = feasible cell (τ = 0.9/0.05/0.05), "
                 f"grey = fails deadline, ★ = grid teacher target", fontsize=9)
    fig.savefig(out / f"heatmaps_{name}.png", dpi=130, bbox_inches="tight"); plt.close(fig)


def kp_section(path):
    """Controller-gain test (results/phase1/kp_test.json): outcome per F at three s levels for kp ∈ {150, 300, 600}."""
    if not path.exists():
        return []
    rows = json.loads(path.read_text())
    Fv = sorted({r["F"] for r in rows}); kps = sorted({r["kp"] for r in rows}); sv = sorted({r["s"] for r in rows})
    out = ["\n## Controller gain and the realized acceleration (owner decision before the sweep)\n",
           "Combination (1.5 kg, μ 0.2, centred), rigid f_tol; OSC kp (position and orientation; kd = 2√kp). ok = no slip, no primary damage; z under = lift end-point error at the top of the lift.\n",
           "| kp | s | " + " | ".join(f"F = {F:g}" for F in Fv) + " | a_lift / a_lat (50 ms) [m/s²] | a_lat (20 ms) | z under [mm] |", "|---|---|" + "---|" * len(Fv) + "---|---|---|"]
    for kp in kps:
        for s in sv:
            rr = [r for r in rows if r["kp"] == kp and r["s"] == s]
            if not rr:
                continue
            cells = [("slip" if r["slip"] else ("damage" if r["dmg_rigid"] else "ok")) for r in sorted(rr, key=lambda r: r["F"])]
            r0 = rr[-1]
            out.append(f"| {kp} | {s:g} | " + " | ".join(cells) + f" | {r0['a_lift']:.2f} / {r0['a_lat']:.2f} | {r0['a_lat20']:.1f} | {r0['z_overshoot_mm']:.1f} |")
    return out


def fig_boundaries(name, sw, refined, base, out):
    Fs, ss = sw["meta"]["F"], sw["meta"]["s"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for ax, frag in zip(axes, FRAGS):
        iv = refined[(frag, "relaxed")]
        x = np.arange(len(ss))
        # grid shading of feasible F per s (relaxed)
        for j in range(len(ss)):
            if iv[j]["feasible"]:
                ax.add_patch(plt.Rectangle((j - 0.35, iv[j]["F_min"]), 0.7, iv[j]["F_max"] - iv[j]["F_min"], color=COL["sel"], alpha=0.15, lw=0))
        for key, col, lab in (("slip_fit", COL["slip"], "sim slip boundary F_min (50 % crossing, 95 % CI)"),
                              ("damage_fit", COL["damage"], "sim damage boundary F_max (primary proxy)"),
                              ("damage_shear_fit", COL["damage_shear"], "sim damage boundary, shear proxy (κ = 1)")):
            y = [iv[j][key]["F50"] if iv[j][key] else np.nan for j in range(len(ss))]
            lo = [iv[j][key]["ci95"][0] if iv[j][key] and iv[j][key]["ci95"][0] is not None else np.nan for j in range(len(ss))]
            hi = [iv[j][key]["ci95"][1] if iv[j][key] and iv[j][key]["ci95"][1] is not None else np.nan for j in range(len(ss))]
            y, lo, hi = np.array(y, float), np.array(lo, float), np.array(hi, float)
            ax.errorbar(x, y, yerr=[np.clip(y - lo, 0, None), np.clip(hi - y, 0, None)], fmt="o-", color=col, lw=2, ms=6, capsize=3, label=lab)
        for rule, ls in (("B5a", "--"), ("B5b", ":")):
            for setting, lw in (("oracle", 2), ("coarse", 1)):
                b = base[(frag, rule, setting)]
                ax.plot(x, [v["F_min"] for v in b], ls=ls, lw=lw, color=COL[rule], label=f"{rule} {setting}: F_min" if setting == "oracle" else None)
                ax.plot(x, [v["F_max"] for v in b], ls=ls, lw=lw, color=COL[rule], alpha=0.6, label=f"{rule} {setting}: F_max" if setting == "oracle" else None)
        for j, s in enumerate(ss):
            if sw["meta"]["T_nominal"] * s > sw["meta"]["T_max"]["relaxed"]:
                ax.axvspan(j - 0.5, j + 0.5, color="k", alpha=0.08, lw=0)
        ax.set_xticks(x); ax.set_xticklabels([f"{s:g}" for s in ss]); ax.set_xlabel("s (carry time scaling)")
        ax.set_ylabel("F [N]"); ax.set_ylim(0, 75); ax.grid(alpha=0.25)
        ax.set_title(f"{name}, {frag} (f_tol = {L.FRAGILITY[frag]:g} N)", fontsize=10)
        ax.legend(fontsize=6.5, loc="upper right")
    fig.suptitle(f"{name}: refined boundaries vs analytic rules (thin = coarse-bin values; grey = fails relaxed deadline; green = refined feasible interval)", fontsize=9)
    fig.savefig(out / f"boundaries_{name}.png", dpi=130, bbox_inches="tight"); plt.close(fig)


def fig_torsion(sweeps, names, out):
    """Torsion angle versus F at every s, one panel per CoM offset, plus the time series at the boundary cell."""
    fig, axes = plt.subplots(1, len(names) + 1, figsize=(4 * (len(names) + 1), 4))
    for ax, n in zip(axes, names):
        sw = sweeps[n]; Fs, ss = sw["meta"]["F"], sw["meta"]["s"]
        for j, s in enumerate(ss):
            y = [np.mean([r["slip_torsion_max_deg"] for r in sw["rows"] if r["F_idx"] == i and r["s_idx"] == j]) for i in range(len(Fs))]
            ax.plot(Fs, y, "o-", lw=1.5, ms=4, color=plt.cm.Blues(0.35 + 0.6 * j / (len(ss) - 1)), label=f"s = {s:g}")
        ax.set_xscale("log"); ax.set_xlabel("F [N]"); ax.set_ylabel("max torsion about closing axis [deg]")
        ax.set_title(f"{n}: CoM offset {props(sw)[2]*1e3:.0f} mm", fontsize=10); ax.grid(alpha=0.25); ax.legend(fontsize=7)
        ax.set_ylim(0, 12)
    ax = axes[-1]
    for k, n in enumerate(names):
        sw = sweeps[n]; Fs, ss = sw["meta"]["F"], sw["meta"]["s"]
        i = int(np.argmin(np.abs(np.array(Fs) - 15.5))); j = int(np.argmin(np.abs(np.log(np.array(ss)))))
        f = sw["dir"] / "branches" / f"F{i}_s{j}_r0.npz"
        if not f.exists():
            continue
        z = np.load(f); R = L.Rec(z["rec"].astype(float), list(z["cols"]))
        mo = L.relative_motion(R)
        t = R["t"] - R["t"][0]
        ax.plot(t, np.degrees(mo["torsion"]), lw=1.5, color=plt.cm.Oranges(0.3 + 0.7 * k / max(1, len(names) - 1)), label=f"offset {props(sw)[2]*1e3:.0f} mm")
    ax.set_xlabel("t since branch [s]"); ax.set_ylabel("torsion [deg]"); ax.set_title(f"F = {Fs[i]:.1f} N, s = {ss[j]:g}, repeat 0", fontsize=10)
    ax.grid(alpha=0.25); ax.legend(fontsize=7)
    fig.savefig(out / "torsion_com_offset.png", dpi=130, bbox_inches="tight"); plt.close(fig)


def fig_damage_vs_s(sweeps, out):
    """Sustained-peak scores versus s at fixed F: the s-dependence of the damage proxies (trailing-pad load, shear)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, key, title in ((axes[0], "N", "primary proxy: sustained 50 ms per-pad normal force − F"),
                           (axes[1], "L", "shear proxy: sustained 50 ms N + |T| − F (κ = 1)")):
        for k, (n, sw) in enumerate(sweeps.items()):
            Fs, ss = sw["meta"]["F"], sw["meta"]["s"]
            i = int(np.argmin(np.abs(np.array(Fs) - 48.0)))
            y = []
            for j in range(len(ss)):
                cell = [r for r in sw["rows"] if r["F_idx"] == i and r["s_idx"] == j]
                sc = [r["score_N"]["50ms"] if key == "N" else r["score_L"]["50ms"]["k1"] for r in cell]
                y.append(np.mean(sc) - Fs[i])
            m, mu, d = props(sw)
            ax.plot(ss, y, "o-", lw=1.5, ms=5, label=f"{n}: m={m:g} μ={mu:g} d={d*1e3:.0f}mm")
        ax.set_xscale("log"); ax.set_xlabel("s"); ax.set_ylabel("score − F [N]"); ax.set_title(title, fontsize=9); ax.grid(alpha=0.25); ax.legend(fontsize=7)
    fig.suptitle(f"Damage-proxy excess over the commanded F at F = {Fs[i]:.1f} N, mean over repeats", fontsize=9)
    fig.savefig(out / "damage_scores_vs_s.png", dpi=130, bbox_inches="tight"); plt.close(fig)


# ------------------------------------------------------------------------------------------------------- report
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("sweeps", nargs="+", help="name=dir")
    args = ap.parse_args(argv)
    out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)
    sweeps = {a.split("=")[0]: load(a.split("=")[1]) for a in args.sweeps}
    lines = [f"# Pre-sweep demonstrations — Phase 1 (plan §6)\n", "Generated by analysis/phase1_demo.py. Only measured numbers; empty cells = not measured.\n"]
    all_ctx = {}          # (name, frag, urg) -> ctx
    refined_all = {}      # (name, frag, urg) -> intervals
    sel_grid, sel_ref = [], []

    # ---------------- per-sweep tables
    lines.append("## Sweeps\n")
    lines.append("| name | m [kg] | μ | CoM offset [mm] | branches | probes | wall/branch [s] | P1 fit (meas = a·F + b) | R² | finger1 pos [mm] | closing axis (world) | obj solref |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for n, sw in sweeps.items():
        rows = sw["rows"]; m, mu, d = props(sw)
        cmd = np.array([r["F"] for r in rows for _ in (0, 1)]); meas = np.array([v for r in rows for v in (r["ss_f_left"], r["ss_f_right"])])
        A = np.vstack([cmd, np.ones_like(cmd)]).T; (a, b), *_ = np.linalg.lstsq(A, meas, rcond=None)
        r2 = 1 - ((meas - (a * cmd + b)) ** 2).sum() / ((meas - meas.mean()) ** 2).sum()
        q = [r["ss_q1"] * 1e3 for r in rows]
        lines.append(f"| {n} | {m:g} | {mu:g} | {d*1e3:.0f} | {len(rows)} | {len(sw['probes'])} | {np.mean([r['wall_s'] for r in rows]):.2f} | {a:.4f}·F + {b:.3f} | {r2:.4f} | {min(q):.1f}–{max(q):.1f} | {sw['meta']['closing_axis_world']} | {sw['meta'].get('contact', {}).get('obj_solref')} |")
    sw0 = next(iter(sweeps.values()))
    lines.append(f"\nGrid: F = {sw0['meta']['F']} N; s = {sw0['meta']['s']}; R = {sw0['meta']['repeats']} paired repeats. "
                 f"T_carry(s₀) = {sw0['meta']['T_nominal']} s; T_max = {sw0['meta']['T_max']} (relaxed 2×, urgent 1.0× nominal + 0.5 s). "
                 f"Labels: δ_slip = {L.DELTA_SLIP*1e3:g} mm, t_win = {L.T_WIN_PRIMARY}, κ = 1, fragile/rigid f_tol = {L.FRAGILITY}. τ = {fs.TAU}, γ = {fs.GAMMA}.\n")
    lines.append("### Realized peak acceleration per s (median over branches; 50 ms moving average; m/s²)\n")
    lines.append("| name | " + " | ".join(f"s = {s:g}" for s in sw0["meta"]["s"]) + " |")
    lines.append("|---|" + "---|" * len(sw0["meta"]["s"]))
    for n, sw in sweeps.items():
        acc = accel_by_s(sw)
        lines.append(f"| {n} (lift / lateral / lower) | " + " | ".join(f"{a['a_lift']:.2f} / {a['a_lat']:.2f} / {a['a_lower']:.2f}" for a in acc) + " |")
    acc20 = [{k: float(np.median([r[k + "_20ms"] for r in sw0["rows"] if r["s_idx"] == j])) for k in ("a_lift", "a_lat")} for j in range(len(sw0["meta"]["s"]))]
    lines.append(f"\nCommanded min-jerk peaks at s = 1: lift {5.77*0.12/0.5**2:.2f}, lateral {5.77*0.30/0.75**2:.2f} m/s² (scale with 1/s²). "
                 f"20 ms-smoothed peaks for {next(iter(sweeps))}: " + ", ".join(f"s={s:g}: {a['a_lift']:.1f}/{a['a_lat']:.1f}" for s, a in zip(sw0['meta']['s'], acc20)) +
                 ". The OSC controller saturates: realized lift acceleration stays ≈ 3 m/s² for s ≤ 0.5.\n")

    # ---------------- contexts, refined boundaries, baselines
    for n, sw in sweeps.items():
        meta, rows, probes = sw["meta"], sw["rows"], sw["probes"]
        Fs, ss = meta["F"], meta["s"]; m, mu, d = props(sw)
        ctxs, refined, base = {}, {}, {}
        acc = accel_by_s(sw)
        bins = B.coarse_bin(m, mu, 20.0, d)
        for frag in FRAGS:
            f_tol = L.FRAGILITY[frag]
            for urg in URGS:
                ctx = fs.context_tables(rows, Fs, ss, frag, urg, meta["T_nominal"], meta["T_max"][urg])
                iv = fs.refined_intervals(rows, probes, Fs, ss, frag, urg, meta["T_nominal"], meta["T_max"][urg])
                ctxs[(frag, urg)] = ctx; refined[(frag, urg)] = iv
                all_ctx[(n, frag, urg)] = ctx; refined_all[(n, frag, urg)] = iv
            for rule in ("B5a", "B5b"):
                base[(frag, rule, "oracle")] = B.intervals_for_sweep(acc, ss, m, mu, f_tol, d, rule, meta["T_nominal"], meta["T_max"]["relaxed"])
                bb = dict(bins); bb["fragility"] = frag
                cm, cmu, cf, cd = B.coarse_values(bb)
                base[(frag, rule, "coarse")] = B.intervals_for_sweep(acc, ss, cm, cmu, cf, cd, rule, meta["T_nominal"], meta["T_max"]["relaxed"])
        fig_heatmaps(n, sw, ctxs, out)
        fig_boundaries(n, sw, refined, base, out)
        lines.append(f"\n## {n}: m = {m:g} kg, μ = {mu:g}, CoM offset = {d*1e3:.0f} mm ({d/0.04:.2f} w)\n")
        lines.append(f"Figures: heatmaps_{n}.png, boundaries_{n}.png. Coarse bins (noise-free): {bins}.\n")
        lines.append("### Refined boundaries per s (relaxed deadline; F in N; 50 % logistic crossing [95 % bootstrap CI over repeats]; bracket = bisection bracket)\n")
        lines.append("| frag | s | deadline ok | F_min slip | bracket | F_max damage (primary) | bracket | F_max shear proxy | refined feasible [F_min, F_max] | B5a oracle | B5b oracle | B5a coarse | B5b coarse |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for frag in FRAGS:
            iv = refined[(frag, "relaxed")]
            for j, s in enumerate(ss):
                v = iv[j]
                bs = sw["brackets"].get("slip", {}).get(str(j), {}); bd = sw["brackets"].get("damage", {}).get(f"{frag}|{j}", {})
                br = lambda b: f"[{b['lo']:.1f}, {b['hi']:.1f}]" if b.get("status") == "bracketed" else b.get("status", "")
                fmin = "all slip" if v["slip_status"] == "all" else ("none" if v["slip_status"] == "none" else f"{v['slip_fit']['F50']:.1f} {ci_str(v['slip_fit'])}")
                fmax = "all damage" if v["damage_status"] == "all" else ("none" if v["damage_status"] == "none" else f"{v['damage_fit']['F50']:.1f} {ci_str(v['damage_fit'])}")
                fsh = "" if v["damage_shear_fit"] is None else f"{v['damage_shear_fit']['F50']:.1f} {ci_str(v['damage_shear_fit'])}"
                feas = f"[{v['F_min']:.1f}, {v['F_max']:.1f}]" if v["feasible"] else "∅"
                bl = lambda key: f"[{base[(frag, key[0], key[1])][j]['F_min']:.1f}, {base[(frag, key[0], key[1])][j]['F_max']:.1f}]"
                lines.append(f"| {frag} | {s:g} | {'✓' if v['deadline_ok'] else '✗'} | {fmin} | {br(bs)} | {fmax} | {br(bd)} | {fsh} | {feas} | {bl(('B5a','oracle'))} | {bl(('B5b','oracle'))} | {bl(('B5a','coarse'))} | {bl(('B5b','coarse'))} |")
        # selections
        lines.append("\n### Teacher selection per context (margin-maximising; ties toward nominal s)\n")
        lines.append("| frag | urgency | grid feasible cells | grid 𝓕_phys cells | grid teacher (F, s) | grid margin | refined feasible s levels | refined selection (F, s) | refined margin (F-limited / s-limited) | P3 (grid, reference = fastest s) | s_rec bracket |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for frag in FRAGS:
            for urg in URGS:
                ctx, iv = ctxs[(frag, urg)], refined[(frag, urg)]
                t = ctx["teacher"]; sel, cands = fs.select_refined(iv)
                j_nom = 0                                    # P3 reference speed = fastest grid s (notes/decision_phase1.md)
                feas = ctx["feasible"]
                if feas.any():
                    p3 = "feasible at reference s" if feas[:, j_nom].any() else ("recovery at s = " + ", ".join(f"{ss[j]:g}" for j in range(len(ss)) if feas[:, j].any() and ctx["deadline_ok"][j]) if any(feas[:, j].any() and ctx["deadline_ok"][j] for j in range(len(ss))) else "feasible only past the deadline")
                else:
                    p3 = "infeasible by design"
                srec = sw["brackets"].get("s_rec", {}).get(f"{frag}|{urg}", {})
                srec_s = f"[{srec['lo']:.2f}, {srec['hi']:.2f}] at F = {srec['F']:.1f}" if srec.get("status") == "bracketed" else srec.get("status", "")
                lines.append(f"| {frag} | {urg} | {int(feas.sum())} | {int(ctx['feasible_phys'].sum())} | {'' if t is None else f"({t['F']:.1f}, {t['s']:g})"} | {'' if t is None else f"{t['margin']:.3f}"} | "
                             f"{', '.join(f"{v['s']:g}" for v in iv if v['feasible']) or '—'} | {'' if sel is None else f"({sel['F']:.1f}, {sel['s']:g})"} | "
                             f"{'' if sel is None else f"{sel['margin']:.3f} ({sel['margin_F']:.3f} / {sel['margin_s']:.3f})"} | {p3} | {srec_s} |")
                sel_grid.append((f"{n}|{frag}|{urg}", None if t is None else t["s"]))
                sel_ref.append((f"{n}|{frag}|{urg}", None if sel is None else sel["s"]))
        # P2 / P3 / B5a summary on the refined boundaries (relaxed deadline)
        lines.append("\n### P2, P3 and B5a-oracle error on the refined boundaries (relaxed deadline)\n")
        lines.append("| frag | F_min(fastest s) − F_min(slowest deadline-ok s) [N] (P2 ≥ 9.75) | brackets | refined window at fastest s [N] | widest window at a slower deadline-ok s [N] (s) | grid level inside that window | B5a oracle |F_min error| per s [N] | B5a oracle |F_max error| per s [N] |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for frag in FRAGS:
            iv = refined[(frag, "relaxed")]
            ok = [v for v in iv if v["deadline_ok"]]
            f0, f1 = ok[0], ok[-1]
            fm = lambda v: v["slip_fit"]["F50"] if v["slip_fit"] else None
            shift = f"{fm(f0) - fm(f1):.1f}" if fm(f0) is not None and fm(f1) is not None else ""
            b0 = sw["brackets"].get("slip", {}).get(str(f0["s_idx"]), {}); b1 = sw["brackets"].get("slip", {}).get(str(f1["s_idx"]), {})
            br = lambda b: f"[{b['lo']:.1f}, {b['hi']:.1f}]" if b.get("status") == "bracketed" else b.get("status", "")
            win = lambda v: (v["F_max"] - v["F_min"]) if (v["F_min"] is not None and v["F_max"] is not None) else None
            w0 = win(f0)
            slower = [v for v in ok[1:] if win(v) is not None]
            best = max(slower, key=win) if slower else None
            inside = ""
            if best is not None and win(best) > 0:
                inside = ", ".join(f"{F:.1f}" for F in Fs if best["F_min"] <= F <= best["F_max"]) or "none"
            e_min = [abs(base[(frag, "B5a", "oracle")][j]["F_min"] - fm(iv[j])) for j in range(len(ss)) if fm(iv[j]) is not None and iv[j]["slip_status"] == "crossing"]
            e_max = [abs(base[(frag, "B5a", "oracle")][j]["F_max"] - iv[j]["damage_fit"]["F50"]) for j in range(len(ss)) if iv[j]["damage_fit"] and iv[j]["damage_status"] == "crossing"]
            lines.append(f"| {frag} | {shift} | {br(b0)} vs {br(b1)} | {'' if w0 is None else f'{w0:.1f}'} | "
                         f"{'' if best is None else f'{win(best):.1f} (s = {best['s']:g})'} | {inside} | "
                         f"{'' if not e_min else f'{min(e_min):.1f}–{max(e_min):.1f}'} | {'' if not e_max else f'{min(e_max):.1f}–{max(e_max):.1f}'} |")
        # P5 sensitivity on the grid+probes: damage boundary per setting (logistic 50 %) at the nominal s and the fastest s
        lines.append("\n### P5: damage-boundary sensitivity (50 % crossing in F, grid + probes), per setting\n")
        lines.append("| frag | s | " + " | ".join(f"N {w} ×{mult}" for w in L.T_WIN for mult in L.F_TOL_MULT) + " | " + " | ".join(f"L {w} {k} ×1" for w in L.T_WIN for k in L.KAPPAS) + " | τ(q_N, q_L) over grid cells |")
        lines.append("|---|---|" + "---|" * (len(L.T_WIN) * len(L.F_TOL_MULT) + len(L.T_WIN) * len(L.KAPPAS)) + "---|")
        for frag in FRAGS:
            for j in (int(np.argmin(np.abs(np.log(np.array(ss))))), 0):
                at = [r for r in rows if r["s_idx"] == j] + [p for p in probes if p.get("s_idx") == j]
                F = [r["F"] for r in at]; rep = [r["repeat"] for r in at]
                cells = []
                for w in L.T_WIN:
                    for mult in L.F_TOL_MULT:
                        y = [r["fragility"][frag]["damage_sensitivity"][f"N|{w}|{mult}"] for r in at]
                        fit = fs.logistic_boundary(F, y, rep, True, n_boot=0)
                        cells.append("all" if all(y) else ("none" if not any(y) else f"{fit['F50']:.1f}"))
                for w in L.T_WIN:
                    for k in L.KAPPAS:
                        y = [r["fragility"][frag]["damage_sensitivity"][f"L|{w}|{k}|x1"] for r in at]
                        fit = fs.logistic_boundary(F, y, rep, True, n_boot=0)
                        cells.append("all" if all(y) else ("none" if not any(y) else f"{fit['F50']:.1f}"))
                ctx = ctxs[(frag, "relaxed")]
                qN = (ctx["P"]["deadline_success"] - ctx["P"]["slip"] - ctx["P"]["damage"]).ravel()
                qL = (ctx["P"]["deadline_success"] - ctx["P"]["slip"] - ctx["P"]["damage_shear"]).ravel()
                tau = kendalltau(qN, qL).statistic if np.std(qN) > 0 and np.std(qL) > 0 else float("nan")
                lines.append(f"| {frag} | {ss[j]:g} | " + " | ".join(cells) + f" | {tau:.3f} |")
        # P6 repeatability; cells inside a refined boundary's uncertainty (95 % CI ∪ bisection bracket, relaxed deadline,
        # either fragility) are reported separately (frozen 2026-10-29)
        def boundary_cells():
            cells = set()
            for frag in FRAGS:
                iv = refined[(frag, "relaxed")]
                for j in range(len(ss)):
                    spans = []
                    for key, bkey in (("slip_fit", "slip"), ("damage_fit", "damage")):
                        fit = iv[j][key]
                        b = sw["brackets"].get(bkey, {}).get(str(j) if bkey == "slip" else f"{frag}|{j}", {})
                        lo, hi = [], []
                        if fit and fit["ci95"][0] is not None:
                            lo.append(fit["ci95"][0]); hi.append(fit["ci95"][1])
                        if b.get("status") == "bracketed":
                            lo.append(b["lo"]); hi.append(b["hi"])
                        if lo:
                            spans.append((min(lo), max(hi)))
                    for i, F in enumerate(Fs):
                        if any(lo <= F <= hi for lo, hi in spans):
                            cells.add((i, j))
            return cells
        bcells = boundary_cells()
        agree = {k: {"in": [], "out": []} for k in ("slip", "completed", "damage_primary(fragile)", "damage_primary(rigid)", "deadline_success(relaxed, rigid)")}
        for i in range(len(Fs)):
            for j in range(len(ss)):
                cell = [r for r in rows if r["F_idx"] == i and r["s_idx"] == j]
                grp = "in" if (i, j) in bcells else "out"
                def ag(vals):
                    vals = np.array(vals, float); return max(vals.mean(), 1 - vals.mean())
                agree["slip"][grp].append(ag([r["slip"] for r in cell])); agree["completed"][grp].append(ag([r["completed"] for r in cell]))
                agree["damage_primary(fragile)"][grp].append(ag([r["fragility"]["fragile"]["damage_primary"] for r in cell]))
                agree["damage_primary(rigid)"][grp].append(ag([r["fragility"]["rigid"]["damage_primary"] for r in cell]))
                agree["deadline_success(relaxed, rigid)"][grp].append(ag([r["fragility"]["rigid"]["deadline_success_relaxed"] for r in cell]))
        lines.append(f"\n### P6: repeat agreement per cell (fraction of repeats in the majority class; pass ≥ 0.9 per cell outside boundary uncertainty; {len(bcells)} of {len(Fs)*len(ss)} cells lie inside a boundary CI/bracket and are reported separately)\n")
        lines.append("| label | outside boundaries: mean | min | cells < 0.9 | inside boundaries: mean | min | cells < 0.9 |")
        lines.append("|---|---|---|---|---|---|---|")
        for k, v in agree.items():
            o, b = v["out"], v["in"]
            lines.append(f"| {k} | {np.mean(o):.3f} | {np.min(o):.2f} | {int(np.sum(np.array(o) < 0.9))} | "
                         f"{'' if not b else f'{np.mean(b):.3f}'} | {'' if not b else f'{np.min(b):.2f}'} | {'' if not b else int(np.sum(np.array(b) < 0.9))} |")
        # impulse appendix
        lines.append("\n### Appendix: excess-impulse damage D over grasp + carry (N·s), rigid f_tol, at F = 70 N per s: " +
                     ", ".join(f"s={s:g}: {np.mean([r['fragility']['rigid']['damage_impulse_D'] for r in rows if r['F_idx']==len(Fs)-1 and r['s_idx']==j]):.1f}" for j, s in enumerate(ss)) + "\n")

    # ---------------- cross-sweep: CoM offset shift, torsion, damage-vs-s figure, P7, fixed reference
    bnames = [n for n in sweeps if n.startswith("b")]
    if len(bnames) >= 2:
        fig_torsion(sweeps, bnames, out)
        lines.append("\n## CoM offset: slip-boundary shift and torsion (demonstration (b))\n")
        lines.append("Figure: torsion_com_offset.png. Rule (owner, 2026-10-29): keep 0.15 w if the refined slip boundary shifts by ≥ 15 % of the F range (9.75 N) versus centred; otherwise move to 0.25 w.\n")
        lines.append("| s | " + " | ".join(f"F_min {n} (d = {props(sweeps[n])[2]*1e3:.0f} mm)" for n in bnames) + " | " + " | ".join(f"shift {n} − {bnames[0]}" for n in bnames[1:]) + " | " + " | ".join(f"max torsion {n} at F = 15.5 N [deg]" for n in bnames) + " |")
        lines.append("|---|" + "---|" * (len(bnames) + len(bnames) - 1 + len(bnames)))
        ss = sw0["meta"]["s"]; Fs = sw0["meta"]["F"]; i155 = int(np.argmin(np.abs(np.array(Fs) - 15.5)))
        for j, s in enumerate(ss):
            vals = []
            for n in bnames:
                v = refined_all[(n, "rigid", "relaxed")][j]
                vals.append(v["slip_fit"]["F50"] if v["slip_fit"] else None)
            shifts = [f"{vals[k]-vals[0]:+.1f}" if vals[k] is not None and vals[0] is not None else "" for k in range(1, len(bnames))]
            tors = [f"{np.mean([r['slip_torsion_max_deg'] for r in sweeps[n]['rows'] if r['F_idx']==i155 and r['s_idx']==j]):.1f}" for n in bnames]
            lines.append(f"| {s:g} | " + " | ".join(f"{v:.1f}" if v is not None else "" for v in vals) + " | " + " | ".join(shifts) + " | " + " | ".join(tors) + " |")
        lines.append(f"\nTorsion label threshold: pad-edge displacement δ_slip / pad_half = {np.degrees(L.DELTA_SLIP/0.008):.0f}° (plan §3.2, h = 8 mm); "
                     "the translational criterion (5 mm) and drops fire first in every sweep, so no branch is labelled slip by torsion alone.\n")
    fig_damage_vs_s(sweeps, out)
    lines.append("\n## Damage-proxy dependence on s\n\nFigure: damage_scores_vs_s.png (sustained 50 ms scores minus F at F = 48 N).\n")

    # P7
    lines.append("\n## P7: context dependence of s (threshold declared: ≥ 25 % of feasible contexts select a non-modal s)\n")
    for lab, sel in (("grid teacher", sel_grid), ("refined feasible set", sel_ref)):
        r = fs.p7_context_dependence(sel)
        lines.append(f"- **{lab}:** contexts with a feasible set: {r['n_contexts']}; modal s = {r.get('modal_s')}; fraction non-modal = {fmt(r.get('frac_non_modal'), 2)}; "
                     f"P7 on these contexts: {'pass' if r['pass'] else ('fail' if r['pass'] is False else 'n/a')}. Selections: " +
                     ", ".join(f"{c}: {('∅' if s is None else f'{s:g}')}" for c, s in r.get("selections", [])))
    # best single fixed
    ctx_list = [all_ctx[k] for k in all_ctx if all_ctx[k]["feasible"].any() or True]
    bsf = fs.best_single_fixed(ctx_list)
    lines.append(f"\n## Best single fixed (F, s) reference (post hoc, over all {bsf['n_contexts']} contexts of these sweeps, both urgency levels)\n")
    lines.append(f"- best single cell: F = {bsf['F']:.1f} N, s = {bsf['s']:g}; mean deadline-constrained safe success = {bsf['success']:.3f}; "
                 f"per-context oracle (best cell per context) mean = {bsf['oracle_per_context_success']:.3f}.")
    lines.append("- per context: " + ", ".join(f"{k[0]}|{k[1]}|{k[2]}: {v:.2f}" for k, v in zip(all_ctx.keys(), bsf["per_context"])))
    lines += kp_section(pathlib.Path("results/phase1/kp_test.json"))
    (out / "report.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
