"""Pilot report for Phase 1: P1 force tracking, finger position, outcome table, wall-clock.

Usage: .venv/bin/python -m analysis.phase1_pilot results/phase1/pilot
"""
import json, sys, pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main(path):
    out = pathlib.Path(path)
    S = json.loads((out / "summary.json").read_text())
    meta, rows = S["meta"], S["branches"]
    Fs, ss = meta["F"], meta["s"]
    # ---- P1: commanded vs measured steady-state force (both sides, all branches)
    cmd = np.array([r["F"] for r in rows for _ in (0, 1)])
    meas = np.array([v for r in rows for v in (r["ss_f_left"], r["ss_f_right"])])
    A = np.vstack([cmd, np.ones_like(cmd)]).T
    (slope, icpt), res, *_ = np.linalg.lstsq(A, meas, rcond=None)
    pred = slope * cmd + icpt
    r2 = 1 - ((meas - pred) ** 2).sum() / ((meas - meas.mean()) ** 2).sum()
    # ---- figure 1: tracking + finger position
    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    ax[0].plot(cmd, meas, "o", alpha=0.6, label="measured (per pad)")
    lim = [0, max(Fs) * 1.1]
    ax[0].plot(lim, lim, "k--", lw=1, label="y = x")
    ax[0].plot(lim, slope * np.array(lim) + icpt, "r-", lw=1, label=f"fit: {slope:.3f}F+{icpt:.2f}, R²={r2:.4f}")
    ax[0].set_xlabel("commanded forcerange F [N]"); ax[0].set_ylabel("steady-state pad normal force [N]"); ax[0].legend(fontsize=8)
    ax[0].set_title("P1: force tracking")
    q1 = np.array([r["ss_q1"] for r in rows]) * 1000; Fr = np.array([r["F"] for r in rows])
    ax[1].plot(Fr, q1, "o"); ax[1].axhline(0, color="k", ls="--", lw=1, label="closed limit (0 mm)")
    ax[1].axhline(meta["config"]["obj_half_size"][0] * 1000, color="g", ls=":", lw=1, label="object half-width")
    ax[1].set_xlabel("F [N]"); ax[1].set_ylabel("finger1 position at steady state [mm]"); ax[1].legend(fontsize=8)
    ax[1].set_title("finger position (not at limit?)")
    # force traces at s = s[len//2], repeat 0
    cols = meta["sample_cols"]; it, ifl, ifr = cols.index("t"), cols.index("f_left"), cols.index("f_right")
    j = len(ss) // 2
    for i, F in enumerate(Fs):
        f = out / "branches" / f"F{i}_s{j}_r0.npz"
        if f.exists():
            rec = np.load(f)["rec"]
            ax[2].plot(rec[:, it] - rec[0, it], np.maximum(rec[:, ifl], rec[:, ifr]), lw=0.8, label=f"F={F:g}")
    ax[2].set_xlabel("t since branch [s]"); ax[2].set_ylabel("max pad normal force [N]"); ax[2].legend(fontsize=8)
    ax[2].set_title(f"force traces, s={ss[j]:g}, repeat 0")
    fig.tight_layout(); fig.savefig(out / "p1_force_tracking.png", dpi=130)
    # ---- outcome table
    lines = []
    lines.append(f"# Phase-1 pilot report — {out}\n")
    c = meta["config"]; t = meta["tolerances"]
    lines.append(f"Object: half-size {c['obj_half_size']} m, mass {c['obj_mass']} kg, μ {c['obj_mu']}. "
                 f"Provisional tolerances: f_tol {t['f_tol']} N, D_max {t['D_max']} N·s, δ_slip {t['delta_slip']*1000:g} mm. "
                 f"Solver: {meta['solver']}. Gripper kp {meta['gripper']['kp']}.\n")
    lines.append(f"## P1 force tracking\n\nmeasured = {slope:.4f}·F + {icpt:.3f} N, R² = {r2:.4f} "
                 f"(n = {len(meas)} pad measurements over {len(rows)} branches). Figure: p1_force_tracking.png\n")
    lines.append("| F [N] | mean pad force [N] (L / R) | sd within hold [N] | finger1 pos [mm] | actuator force [N] | contact time [s] |")
    lines.append("|---|---|---|---|---|---|")
    for i, F in enumerate(Fs):
        rr = [r for r in rows if r["F_idx"] == i]
        lines.append(f"| {F:g} | {np.mean([r['ss_f_left'] for r in rr]):.2f} / {np.mean([r['ss_f_right'] for r in rr]):.2f} "
                     f"| {np.mean([r['ss_f_left_std'] for r in rr]):.2f} | {np.mean([r['ss_q1'] for r in rr])*1000:.2f} "
                     f"| {np.mean([r['ss_act_f1'] for r in rr]):.2f} | {np.mean([r['contact_time'] or np.nan for r in rr]):.2f} |")
    lines.append("\n## Outcomes per (F, s) — repeats listed as r0 / r1 …\n")
    lines.append("| F [N] | s | slip disp max [mm] | peak force [N] | D [N·s] | slip | damage (primary / secondary) | complete | final xy err [mm] |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for i, F in enumerate(Fs):
        for j, s in enumerate(ss):
            rr = sorted([r for r in rows if r["F_idx"] == i and r["s_idx"] == j], key=lambda r: r["repeat"])
            j_ = lambda k, f="{:.1f}": " / ".join(f.format(r[k]) for r in rr)
            b_ = lambda k: " / ".join(str(int(r[k])) for r in rr)
            lines.append(f"| {F:g} | {s:g} | {' / '.join(f'{r['slip_disp_max']*1000:.1f}' for r in rr)} | {j_('peak_force')} | {j_('D','{:.2f}')} "
                         f"| {b_('slip')} | {b_('damage_primary')} / {b_('damage_secondary')} | {b_('complete')} | {' / '.join(f'{r['final_xy_err']*1000:.0f}' for r in rr)} |")
    w = np.array([r["wall_s"] for r in rows]); sim = np.array([r["sim_s"] for r in rows])
    lines.append(f"\n## Wall-clock\n\nper branch: mean {w.mean():.2f} s, min {w.min():.2f} s, max {w.max():.2f} s "
                 f"(simulated {sim.mean():.1f} s per branch on average; {sim.sum()/w.sum():.1f}× real time). "
                 f"Full plan sweep (8 × 6 × 5 × 12 = 2880 branches) ≈ {2880*w.mean()/60:.0f} min at this rate.\n")
    (out / "report.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1])
