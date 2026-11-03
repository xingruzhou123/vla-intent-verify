"""Three-gate check for an "active probing before commit" direction (owner request, 2026-11-02). No new phase, no
π0.5, no probe rollouts: probe observations are simulated from the true object properties with a stated sensor model.

Stages (stop at the first failing gate, but every stage's numbers are computed so the report is complete):
  A. independent contexts: a unit is (object, fragility, urgency); classes common-safe / ambiguous-but-feasible / mixed /
     intrinsically infeasible, and how many ambiguous units survive leave-one-object-out.
  B. Gate 1 — value of exact information on the ambiguous subset: per-object oracle vs calibrated B5a vs noise-free
     coarse lookup (leave-one-object-out rows of analysis/coarse_bound2.py, label v3, executed rules).
  C. Gate 2 — probe + formula: simulated micro-lift (mass, CoM torque) and lateral-pulse (friction) readings with
     noise and bias; strongest analytic updater = calibrated Coulomb / wrench rule on interval (2σ worst-case) estimates
     with realized acceleration; ρ_analytic = (J_probe − J_coarse) / (J_oracle − J_coarse).
  D. Gate 3 — small learned model: MLP (and a 5-member ensemble with a pessimistic selection) from the same simulated
     observations to the 48-cell feasibility mask, grouped 4-fold splits by object index; ρ_small.

Usage: MUJOCO_GL=egl .venv/bin/python -m analysis.active_probe_gates --sweeps data/sweep_v1 --objects data/objects_v1.json \
    --cb2 results/phase1/coarse_bound_v2/coarse_bound2.json --out results/phase1/active_probe [--no_exec] [--seeds 10]
"""
import argparse, json, pathlib, itertools
import numpy as np
from envs.feasibility_carry import feasible as fs
from envs.feasibility_carry.objects import CELLS
from baselines import analytic as B
from analysis.phase1_demo import load, accel_by_s, props, fmt
from analysis.phase1_full import rule_intervals, in_refined, execute_selections, with_deadline, FRAGS, URGS, CAL_GRID, noisy_bins, NOISE_SEED
from analysis.coarse_bound2 import per_bin_means, est_from, argmax_cell

# ---- sensor model (assumptions, stated in the report)
SENSOR = {
    "mass": {"bias_sd_kg": 0.05, "noise_sd_kg": 0.02,
             "note": "micro-lift of ≈ 2 cm, mass from the Franka external-force estimate (O_F_ext_hat_K) averaged over 0.5 s; "
                     "per-episode bias ≈ 0.5 N (0.05 kg) from the payload/calibration model, noise ≈ 0.2 N after averaging"},
    "friction": {"rel_bias_sd": 0.05, "rel_noise_sd": 0.10,
                 "note": "lateral pulse at a probe force below every fragility tolerance (12 N) ramped until the tangential/normal force ratio "
                         "at the measured slip onset gives μ̂; onset detection and F/T noise give ≈ 10 % relative noise, ≈ 5 % relative bias"},
    "com": {"torque_sd_Nm": 0.02,
            "note": "torque about the pad normal during the micro-lift from the wrist F/T estimate, noise ≈ 0.02 N·m; d̂ = 2 M̂ / (m̂ g), sign known from the torque sign"},
    "fragility": "not probeable without risking damage: the calibrated coarse-bin estimate (training-split mean) is used, as in B5a-c",
    "acceleration": "realized peak accelerations per s level are the sweep medians (same as the rules)",
    "ci": "interval updater uses the ±2σ box of the total (bias ⊕ noise) standard deviation",
}
G = 9.81
F_GRID_N = 8; S_GRID_N = 6


def probe_draw(m, mu, d, rng):
    """One probe episode: noisy (m̂, μ̂, d̂) and their total standard deviations."""
    sm = np.hypot(SENSOR["mass"]["bias_sd_kg"], SENSOR["mass"]["noise_sd_kg"])
    m_hat = m + rng.normal(0, SENSOR["mass"]["bias_sd_kg"]) + rng.normal(0, SENSOR["mass"]["noise_sd_kg"])
    smu = mu * np.hypot(SENSOR["friction"]["rel_bias_sd"], SENSOR["friction"]["rel_noise_sd"])
    mu_hat = mu * (1 + rng.normal(0, SENSOR["friction"]["rel_bias_sd"]) + rng.normal(0, SENSOR["friction"]["rel_noise_sd"]))
    M = m * G * d / 2
    M_hat = M + rng.normal(0, SENSOR["com"]["torque_sd_Nm"])
    d_hat = 2 * M_hat / (max(m_hat, 0.1) * G)
    sd = 2 * SENSOR["com"]["torque_sd_Nm"] / (max(m_hat, 0.1) * G)
    return {"m": float(max(m_hat, 0.05)), "mu": float(max(mu_hat, 0.02)), "d": float(d_hat), "sm": float(sm), "smu": float(smu), "sd": float(sd)}


def probe_rule_intervals(rule, mode, obs, f_tol, acc, ss, T_nom, T_max, c_min=1.0, c_max=1.0, r_eff=B.R_EFF, k=2.0):
    """Analytic updater on probe observations. mode = 'point' (point estimates) or 'interval' (worst case over the ±kσ box:
    F_min at (m̂ + kσ, μ̂ − kσ, |d̂| + kσ), F_max at m̂ + kσ)."""
    if mode == "point":
        m_lo = m_hi = obs["m"]; mu_lo = obs["mu"]; d_hi = abs(obs["d"])
    else:
        m_hi = obs["m"] + k * obs["sm"]; mu_lo = max(obs["mu"] - k * obs["smu"], 0.02); d_hi = abs(obs["d"]) + k * obs["sd"]
    out = []
    for j, s in enumerate(ss):
        a = acc[j]
        if rule == "B5a":
            F_min, _, _ = B.coulomb_interval(m_hi, mu_lo, f_tol, a["a_lift"], a["a_lat"], a["a_lower"])
            _, F_max, _ = B.coulomb_interval(m_hi, mu_lo, f_tol, a["a_lift"], a["a_lat"], a["a_lower"])
        else:
            F_min, _, _ = B.wrench_interval(m_hi, mu_lo, f_tol, d_hi, a["a_lift"], a["a_lat"], a["a_lower"], r_eff=r_eff)
            _, F_max, _ = B.wrench_interval(m_hi, mu_lo, f_tol, d_hi, a["a_lift"], a["a_lat"], a["a_lower"], r_eff=r_eff)
        F_min, F_max = c_min * F_min, c_max * F_max
        ok = T_nom * s <= T_max
        out.append({"s_idx": j, "s": s, "deadline_ok": bool(ok), "F_min": F_min, "F_max": F_max, "feasible": bool(ok and F_max > F_min)})
    return out


def boot_ci(vals, n=1000, seed=0):
    v = np.asarray([x for x in vals if x is not None], float)
    if len(v) == 0:
        return (None, None)
    rng = np.random.default_rng(seed)
    b = [v[rng.integers(len(v), size=len(v))].mean() for _ in range(n)]
    return (float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5)))


def boot_diff_ci(a, b_, n=1000, seed=0):
    a, b_ = np.asarray(a, float), np.asarray(b_, float)
    rng = np.random.default_rng(seed); out = []
    for _ in range(n):
        idx = rng.integers(len(a), size=len(a)); out.append(a[idx].mean() - b_[idx].mean())
    return (float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)))


def boot_rho(J_probe, J_coarse, J_oracle, n=1000, seed=0):
    """Bootstrap over units of ρ = (mean J_probe − mean J_coarse) / (mean J_oracle − mean J_coarse)."""
    P, C, O = (np.asarray(x, float) for x in (J_probe, J_coarse, J_oracle))
    rng = np.random.default_rng(seed); out = []
    for _ in range(n):
        idx = rng.integers(len(P), size=len(P)); den = O[idx].mean() - C[idx].mean()
        if abs(den) > 1e-9:
            out.append((P[idx].mean() - C[idx].mean()) / den)
    return (float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))) if out else (None, None)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweeps", required=True); ap.add_argument("--objects", required=True); ap.add_argument("--cb2", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--no_exec", action="store_true"); ap.add_argument("--workers", type=int, default=12); ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--skip_gate3", action="store_true")
    args = ap.parse_args(argv)
    out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)
    objs = {o["object_id"]: o for o in json.loads(pathlib.Path(args.objects).read_text())["objects"]}
    sweeps = {oid: load(pathlib.Path(args.sweeps) / oid) for oid in objs}
    sw0 = next(iter(sweeps.values())); Fs, ss = sw0["meta"]["F"], sw0["meta"]["s"]; T_nom, T_max = sw0["meta"]["T_nominal"], sw0["meta"]["T_max"]
    ctx = {(oid, fr, urg): fs.context_tables(sw["rows"], Fs, ss, fr, urg, T_nom, T_max[urg]) for oid, sw in sweeps.items() for fr in FRAGS for urg in URGS}
    ref = {}
    for oid, sw in sweeps.items():
        for fr in FRAGS:
            iv = fs.refined_intervals(sw["rows"], sw["probes"], Fs, ss, fr, "relaxed", T_nom, T_max["relaxed"], n_boot=0)
            for urg in URGS:
                ref[(oid, fr, urg)] = with_deadline(iv, T_nom, T_max[urg])
    acc = {oid: accel_by_s(sw) for oid, sw in sweeps.items()}
    cell_of = {oid: f"{o['bins']['mass']}_{o['bins']['friction']}_{o['bins']['balance']}" for oid, o in objs.items()}
    idx_of = {oid: int(oid.rsplit("_", 1)[1]) for oid in objs}
    units = [(oid, fr, urg) for oid in objs for fr in FRAGS for urg in URGS]
    lines = ["# Active-probing direction — three-gate check (label version 3, sweep v1)\n",
             "Owner request 2026-11-02. No new phase, no π0.5, no probe rollouts: probe observations are simulated from the true properties with the sensor model of §C. Only measured numbers; CIs are 95 % bootstrap over units.\n"]
    R = {"sensor_model": SENSOR}

    # ------------------------------------------------------------------ A. independent contexts
    group_of = {u: (cell_of[u[0]], u[1], u[2]) for u in units}
    members = {}
    for u in units:
        members.setdefault(group_of[u], []).append(u)
    def common_cell(us):
        return bool(np.logical_and.reduce([ctx[u]["feasible"] for u in us]).any())
    cls = {}
    for u in units:
        g = group_of[u]; feas = ctx[u]["feasible"].any()
        if not feas:
            cls[u] = "infeasible"
        elif common_cell(members[g]):
            cls[u] = "common-safe"
        elif all(ctx[v]["feasible"].any() for v in members[g]):
            cls[u] = "ambiguous"
        else:
            cls[u] = "mixed"
    # LOO survival: the unit held out, its 3 training siblings define the group's ambiguity
    loo_amb = {}
    for u in units:
        sib = [v for v in members[group_of[u]] if v[0] != u[0]]
        loo_amb[u] = bool(ctx[u]["feasible"].any() and not common_cell(sib))
    counts = {k: sum(1 for u in units if cls[u] == k) for k in ("common-safe", "ambiguous", "mixed", "infeasible")}
    amb_broad = [u for u in units if cls[u] in ("ambiguous", "mixed")]
    amb_strict = [u for u in units if cls[u] == "ambiguous"]
    surv_strict = sum(loo_amb[u] for u in amb_strict); surv_broad = sum(loo_amb[u] for u in amb_broad)
    loo_total = sum(loo_amb.values())
    R["counts"] = {"units": len(units), **counts, "ambiguous_broad": len(amb_broad), "loo_ambiguous_total": loo_total, "loo_survive_strict": surv_strict, "loo_survive_broad": surv_broad,
                   "by_fragility": {fr: {k: sum(1 for u in units if u[1] == fr and cls[u] == k) for k in counts} for fr in FRAGS},
                   "by_urgency": {urg: {k: sum(1 for u in units if u[2] == urg and cls[u] == k) for k in counts} for urg in URGS}}
    lines += ["## A. Independent contexts (unit = object × fragility × urgency; 48 objects × 2 × 2 = 192)\n",
              "| class | definition | units | fragile / rigid | relaxed / urgent |", "|---|---|---|---|---|"]
    defs = {"common-safe": "some single grid (F, s) is in 𝓕 of all four objects of the coarse bin (same fragility, urgency)",
            "ambiguous": "every object of the bin has a non-empty 𝓕 but no common cell", "mixed": "unit feasible, bin has no common cell, and at least one sibling object is infeasible",
            "infeasible": "the unit's own 𝓕 is empty (intrinsically infeasible)"}
    for k in counts:
        lines.append(f"| {k} | {defs[k]} | {counts[k]} | {R['counts']['by_fragility']['fragile'][k]} / {R['counts']['by_fragility']['rigid'][k]} | {R['counts']['by_urgency']['relaxed'][k]} / {R['counts']['by_urgency']['urgent'][k]} |")
    lines.append(f"\n- Ambiguous in the broad sense (feasible unit in a bin without a common cell = ambiguous + mixed): **{len(amb_broad)}** units; strict (all four siblings feasible): **{len(amb_strict)}**.")
    lines.append(f"- Leave-one-object-out: a held-out unit is ambiguous when it is feasible and its three training siblings have no common cell. {loo_total} held-out units are ambiguous under LOO; "
                 f"of the {len(amb_strict)} strict / {len(amb_broad)} broad ambiguous units, {surv_strict} / {surv_broad} stay ambiguous when held out.")
    # proposal if < 50
    rate_strict = len(amb_strict) / len(units); rate_broad = len(amb_broad) / len(units)
    if len(amb_strict) < 50:
        need_units = 50 - len(amb_strict); need_obj_strict = int(np.ceil(need_units / max(rate_strict * 4, 1e-9)))
        need_obj_broad = int(np.ceil(max(0, 50 - len(amb_broad)) / max(rate_broad * 4, 1e-9)))
        lines.append(f"- **Proposal (not run):** at the measured rates ({rate_strict:.3f} strict, {rate_broad:.3f} broad ambiguous units per unit), reaching 50 strict ambiguous units needs ≈ {need_obj_strict} more continuously-sampled objects "
                     f"(≈ {need_obj_strict*240} grid + ≈ {need_obj_strict*250} probe branches ≈ {need_obj_strict*54/48:.0f} min wall at 12 processes, scaling the 54 min of the 48-object sweep); "
                     f"for 50 broad ambiguous units ≈ {need_obj_broad} objects. Sampling only the bins that produce ambiguity (the slippery and the heavy bins; see the per-bin table in coarse_bound_v2) would need fewer, at the cost of a non-uniform bin prior.")
        R["counts"]["proposal"] = {"objects_for_50_strict": need_obj_strict, "objects_for_50_broad": need_obj_broad}

    # ------------------------------------------------------------------ B. Gate 1
    cb2 = json.loads(pathlib.Path(args.cb2).read_text())["rows"]
    def cb2_val(u, method, key):
        oid, fr, urg = u; f = idx_of[oid]
        r = next(x for x in cb2 if x["fold"] == f and x["test"] == oid and x["frag"] == fr and x["urg"] == urg and x["method"] == method)
        v = r[key]
        return (None if v is None else float(v)) if key != "infeasible" else (None if v is None else bool(v))
    def gate1_table(subset, name):
        rows_ = {}
        for method in ("oracle", "B5a-c", "lookup"):
            succ = [cb2_val(u, method, "success") for u in subset]
            inf = [cb2_val(u, method, "infeasible") for u in subset]
            inf = [x for x in inf if x is not None]
            rows_[method] = {"n": len(subset), "success": float(np.mean(succ)), "success_ci": boot_ci(succ), "infeasible": float(np.mean(inf)) if inf else None, "infeasible_ci": boot_ci(inf) if inf else (None, None), "_succ": succ}
        best = max(("B5a-c", "lookup"), key=lambda m: rows_[m]["success"])
        d_succ = rows_["oracle"]["success"] - rows_[best]["success"]
        d_ci = boot_diff_ci(rows_["oracle"]["_succ"], rows_[best]["_succ"])
        inf_b = rows_[best]["infeasible"]; inf_o = rows_["oracle"]["infeasible"] or 0.0
        rel = (inf_b - inf_o) / inf_b if inf_b else None; absd = (inf_b - inf_o) if inf_b is not None else None
        passed = bool(d_succ >= 0.10 or (rel is not None and rel >= 0.40 and absd >= 0.10))
        lines.append(f"\n### {name} ({len(subset)} units)\n")
        lines.append("| method | safe success [95 % CI] | infeasible-selection rate [95 % CI] |"); lines.append("|---|---|---|")
        for m in ("oracle", "B5a-c", "lookup"):
            r = rows_[m]
            lines.append(f"| {m} | {r['success']:.3f} [{r['success_ci'][0]:.3f}, {r['success_ci'][1]:.3f}] | {fmt(r['infeasible'], 3)} {'' if r['infeasible'] is None else f'[{r['infeasible_ci'][0]:.3f}, {r['infeasible_ci'][1]:.3f}]'} |")
        lines.append(f"\nStronger coarse baseline: {best}. Oracle − {best}: success +{d_succ:.3f} [{d_ci[0]:.3f}, {d_ci[1]:.3f}] (criterion ≥ 0.10); infeasible rate {fmt(inf_b, 3)} → {inf_o:.3f} "
                     f"(relative drop {fmt(rel, 2)}, absolute {fmt(absd, 3)}; criterion ≥ 0.40 and ≥ 0.10). **{'PASS' if passed else 'FAIL'}**")
        for r in rows_.values(): r.pop("_succ")
        return {"rows": rows_, "best_coarse": best, "d_success": d_succ, "d_success_ci": d_ci, "inf_rel_drop": rel, "inf_abs_drop": absd, "pass": passed}
    lines.append("\n## B. Gate 1 — value of exact information (leave-one-object-out, label v3; rules calibrated per fold and executed)\n")
    g1 = {"ambiguous_strict": gate1_table(amb_strict, "Ambiguous subset, strict") if amb_strict else None,
          "ambiguous_broad": gate1_table(amb_broad, "Ambiguous subset, broad (ambiguous + mixed)"),
          "all": gate1_table(units, "All 192 units (full distribution)"),
          "feasible_only": gate1_table([u for u in units if cls[u] != "infeasible"], "All feasible units")}
    gate1_pass = bool(g1["ambiguous_broad"]["pass"])
    lines.append(f"\n**Gate 1 verdict (ambiguous subset, broad): {'PASS' if gate1_pass else 'FAIL'}**" + ("" if gate1_pass else " — stop the direction (the later gates are still computed below for completeness)."))
    R["gate1"] = g1; R["gate1_pass"] = gate1_pass

    # ------------------------------------------------------------------ C. Gate 2 — probe + formula
    lines.append("\n## C. Gate 2 — probe + formula (simulated probe observations)\n")
    lines.append("Sensor model (assumptions):\n")
    for k, v in SENSOR.items():
        lines.append(f"- **{k}:** {v if isinstance(v, str) else v.get('note', '') + ' → ' + ', '.join(f'{kk} = {vv}' for kk, vv in v.items() if kk != 'note')}")
    # probe draws per seed for every (object, fragility) — identical draws for both fragility labels of one object (same physical probe)
    def draws(seed):
        d = {}
        for k, oid in enumerate(sorted(objs)):
            m, mu, dd = props(sweeps[oid]); rng = np.random.default_rng([seed, k, 7])
            d[oid] = probe_draw(m, mu, dd, rng)
        return d
    obs_by_seed = [draws(s) for s in range(args.seeds)]
    # calibration of the probe updaters per fold on the fold's training units (seed-0 observations for training objects)
    VARIANTS = [("B5a", "point"), ("B5b", "point"), ("B5a", "interval"), ("B5b", "interval")]
    def fit_probe_mult(rule, mode, train_units, cal, obs):
        best = None
        r_grid = CAL_GRID["r_eff"] if rule == "B5b" else [B.R_EFF]
        for c_min, c_max, r_eff in itertools.product(CAL_GRID["c_min"][::2], CAL_GRID["c_max"][::2], r_grid):
            hits = 0
            for (oid, fr, urg) in train_units:
                ivr = probe_rule_intervals(rule, mode, obs[oid], cal["fragility"][fr], acc[oid], ss, T_nom, T_max[urg], c_min, c_max, r_eff)
                sel, _ = fs.select_refined(ivr)
                hits += bool(in_refined(ref[(oid, fr, urg)], sel))
            key = (hits, -abs(np.log(c_min)) - abs(np.log(c_max)))
            if best is None or key > best[0]:
                best = (key, {"c_min": float(c_min), "c_max": float(c_max), "r_eff": float(r_eff), "train_success": hits / len(train_units)})
        return best[1]
    probe_rows = []; jobs = {}
    fold_mult = {}
    for fold in range(4):
        test = [u for u in units if idx_of[u[0]] == fold]; train = [u for u in units if idx_of[u[0]] != fold]
        train_obj = sorted({u[0] for u in train}); cal = per_bin_means(objs, train_obj)
        train_ne = [u for u in train if any(v["feasible"] for v in ref[u])]
        for rule, mode in VARIANTS:
            mm = fit_probe_mult(rule, mode, train_ne, cal, obs_by_seed[0]); fold_mult[(fold, rule, mode)] = mm
            for u in test:
                oid, fr, urg = u
                for seed, obs in enumerate(obs_by_seed):
                    ivr = probe_rule_intervals(rule, mode, obs[oid], cal["fragility"][fr], acc[oid], ss, T_nom, T_max[urg], mm["c_min"], mm["c_max"], mm["r_eff"])
                    sel, _ = fs.select_refined(ivr)
                    row = {"fold": fold, "unit": u, "rule": rule, "mode": mode, "seed": seed, "F": None if sel is None else sel["F"], "s": None if sel is None else sel["s"],
                           "in_refined": in_refined(ref[u], sel), "truth_nonempty": bool(ctx[u]["feasible"].any()), "cls": cls[u]}
                    if sel is not None and seed == 0:
                        key = f"{sel['F']:.3f}|{sel['s']:.4f}"; row["exec_key"] = key
                        jobs.setdefault(str(sweeps[oid]["dir"]), {})[key] = (float(sel["F"]), float(sel["s"]))
                    probe_rows.append(row)
    exec_res = execute_selections(jobs, args.workers) if (jobs and not args.no_exec) else {}
    for row in probe_rows:
        row["success"] = None; row["infeasible"] = None
        if row["seed"] != 0:
            continue
        if row["F"] is None:
            row["success"] = 0.0; row["infeasible"] = True if row["truth_nonempty"] else None; continue
        e = exec_res.get(str(sweeps[row["unit"][0]]["dir"]), {}).get(row.get("exec_key"))
        if e is None:
            continue
        fr, urg = row["unit"][1], row["unit"][2]
        P_d, P_s, P_dm = e[f"deadline_success|{fr}|{urg}"], e["slip"], e[f"damage|{fr}"]
        row["success"] = P_d; row["infeasible"] = not (P_d >= fs.TAU["c"] and P_s <= fs.TAU["s"] and P_dm <= fs.TAU["d"])
    lines.append("\nCalibrated multipliers per fold (fit on the fold's training units with seed-0 probe observations, coarser grid than B5a-c: every second c value): " +
                 "; ".join(f"fold {f} {r}-{m}: c_min {v['c_min']:.2f}, c_max {v['c_max']:.2f}, r_eff {v['r_eff']*1e3:.0f} mm, train {v['train_success']:.2f}" for (f, r, m), v in fold_mult.items()) + ".\n")
    def gate2_table(subset, name):
        lines.append(f"\n### {name} ({len(subset)} units)\n")
        lines.append(f"| updater | safe success (executed, seed 0) [95 % CI] | infeasible rate (executed, truth non-empty) | in refined interval, seeds 0–{args.seeds - 1} (rate of infeasible by interval) | ρ_analytic (success) [95 % CI] | ρ_analytic (infeasible) |")
        lines.append("|---|---|---|---|---|---|")
        J_o = [cb2_val(u, "oracle", "success") for u in subset]; J_c = [cb2_val(u, "B5a-c", "success") for u in subset]
        I_c = [cb2_val(u, "B5a-c", "infeasible") for u in subset]; I_c = [x for x in I_c if x is not None]
        res = {}
        for rule, mode in VARIANTS:
            rs = [r for r in probe_rows if r["rule"] == rule and r["mode"] == mode and r["unit"] in subset and r["seed"] == 0]
            succ = [r["success"] for r in rs]; inf = [r["infeasible"] for r in rs if r["infeasible"] is not None]
            allseed = [r for r in probe_rows if r["rule"] == rule and r["mode"] == mode and r["unit"] in subset and r["truth_nonempty"]]
            inf_iv = float(np.mean([not bool(r["in_refined"]) for r in allseed])) if allseed else None
            Jp = float(np.mean(succ)) if succ and None not in succ else None
            rho = None; rho_ci = (None, None); rho_inf = None
            if Jp is not None and abs(np.mean(J_o) - np.mean(J_c)) > 1e-9:
                rho = (Jp - np.mean(J_c)) / (np.mean(J_o) - np.mean(J_c)); rho_ci = boot_rho(succ, J_c, J_o)
            if inf and I_c:
                rho_inf = (np.mean(I_c) - np.mean(inf)) / np.mean(I_c) if np.mean(I_c) > 0 else None
            res[f"{rule}-{mode}"] = {"J": Jp, "J_ci": boot_ci(succ) if Jp is not None else (None, None), "infeasible": float(np.mean(inf)) if inf else None, "inf_by_interval_allseeds": inf_iv, "rho": rho, "rho_ci": rho_ci, "rho_inf": rho_inf}
            r_ = res[f"{rule}-{mode}"]
            lines.append(f"| {rule} {mode} | {fmt(r_['J'], 3)} {'' if r_['J'] is None else f'[{r_['J_ci'][0]:.3f}, {r_['J_ci'][1]:.3f}]'} | {fmt(r_['infeasible'], 3)} | {fmt(inf_iv, 3)} | {fmt(rho, 3)} {'' if rho_ci[0] is None else f'[{rho_ci[0]:.2f}, {rho_ci[1]:.2f}]'} | {fmt(rho_inf, 3)} |")
        lines.append(f"\nReference: J_oracle = {np.mean(J_o):.3f}, J_coarse (B5a-c, same rule family without probes) = {np.mean(J_c):.3f}, infeasible rate of B5a-c {fmt(float(np.mean(I_c)) if I_c else None, 3)}.")
        strongest = max(res, key=lambda k: (res[k]["J"] if res[k]["J"] is not None else -1))
        lines.append(f"Strongest analytic updater on this subset: **{strongest}**, ρ_analytic = {fmt(res[strongest]['rho'], 3)}.")
        return {"res": res, "strongest": strongest, "J_oracle": float(np.mean(J_o)), "J_coarse": float(np.mean(J_c))}
    g2 = {"ambiguous_broad": gate2_table(amb_broad, "Ambiguous subset, broad"), "all": gate2_table(units, "All 192 units"),
          "ambiguous_strict": gate2_table(amb_strict, "Ambiguous subset, strict") if amb_strict else None}
    rho_an = g2["ambiguous_broad"]["res"][g2["ambiguous_broad"]["strongest"]]["rho"]
    gate2_stop = bool(rho_an is not None and rho_an >= 0.8)
    lines.append(f"\n**Gate 2 verdict (ambiguous subset, broad): ρ_analytic = {fmt(rho_an, 3)} → {'≥ 0.8: active identification + analytic control, not a learning paper' if gate2_stop else '< 0.8: analytic updater does not recover the oracle gap'}.**")
    R["gate2"] = g2; R["rho_analytic"] = rho_an; R["gate2_stop"] = gate2_stop

    # ------------------------------------------------------------------ D. Gate 3 — small learned model
    g3 = None
    if not args.skip_gate3:
        import torch, torch.nn as nn
        torch.manual_seed(0)
        def feats(oid, fr, urg, obs):
            b = objs[oid]["bins"]
            x = [obs["m"], obs["mu"], obs["d"] * 10, obs["sm"], obs["smu"], obs["sd"] * 10]
            x += [float(b["mass"] == k) for k in ("light", "medium", "heavy")] + [float(b["friction"] == "normal"), float(b["balance"] == "off-centre"), float(fr == "rigid"), float(urg == "urgent")]
            return x
        def target(u):
            return ctx[u]["feasible"].astype(np.float32).ravel()
        class MLP(nn.Module):
            def __init__(self, d_in, d_out, h=64):
                super().__init__(); self.net = nn.Sequential(nn.Linear(d_in, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, d_out))
            def forward(self, x): return self.net(x)
        def train_mlp(X, Y, seed, epochs=300):
            torch.manual_seed(seed); m = MLP(X.shape[1], Y.shape[1]); opt = torch.optim.Adam(m.parameters(), lr=2e-3, weight_decay=1e-4)
            Xt, Yt = torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32); mu_, sd_ = Xt.mean(0), Xt.std(0) + 1e-6
            for ep in range(epochs):
                perm = torch.randperm(len(Xt))
                for i in range(0, len(Xt), 128):
                    idx = perm[i:i + 128]; opt.zero_grad()
                    loss = nn.functional.binary_cross_entropy_with_logits(m((Xt[idx] - mu_) / sd_), Yt[idx]); loss.backward(); opt.step()
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
        g3_rows = []
        for fold in range(4):
            test = [u for u in units if idx_of[u[0]] == fold]; train = [u for u in units if idx_of[u[0]] != fold]
            X, Y = [], []
            for u in train:
                oid, fr, urg = u; m, mu, dd = props(sweeps[oid])
                for a in range(n_aug):
                    rng = np.random.default_rng([100 + fold, a, sorted(objs).index(oid), FRAGS.index(fr), URGS.index(urg)])
                    X.append(feats(oid, fr, urg, probe_draw(m, mu, dd, rng))); Y.append(target(u))
            X, Y = np.asarray(X, np.float32), np.asarray(Y, np.float32)
            models = [train_mlp(X, Y, seed) for seed in range(5)]
            for u in test:
                oid, fr, urg = u; P = ctx[u]["P"]["deadline_success"]; feas = ctx[u]["feasible"]
                for seed, obs in enumerate(obs_by_seed):
                    pm, psd = predict(models, feats(oid, fr, urg, obs[oid]))
                    for name, score in (("mlp-mean", pm), ("ensemble-pessimistic", pm - psd)):
                        i, j = choose(score)
                        g3_rows.append({"fold": fold, "unit": u, "model": name, "seed": seed, "cls": cls[u], "truth_nonempty": bool(feas.any()),
                                        "F": Fs[i], "s": ss[j], "p": float(score[i * len(ss) + j]), "success": float(P[i, j]), "infeasible": bool(not feas[i, j])})
            print(f"gate3 fold {fold}: trained 5 MLPs on {len(X)} samples", flush=True)
        def gate3_table(subset, name):
            lines.append(f"\n### {name} ({len(subset)} units)\n")
            lines.append(f"| model | safe success (grid cell on the held-out unit; mean over {args.seeds} probe-noise seeds) [95 % CI over units] | infeasible rate (truth non-empty) | ρ_small (success) [95 % CI] | ρ_small (infeasible) |")
            lines.append("|---|---|---|---|---|")
            J_o = [cb2_val(u, "oracle", "success") for u in subset]; J_c = [cb2_val(u, "B5a-c", "success") for u in subset]
            I_c = [cb2_val(u, "B5a-c", "infeasible") for u in subset]; I_c = [x for x in I_c if x is not None]
            res = {}
            for name in ("mlp-mean", "ensemble-pessimistic"):
                per_unit = [np.mean([r["success"] for r in g3_rows if r["model"] == name and r["unit"] == u]) for u in subset]
                inf_unit = [np.mean([r["infeasible"] for r in g3_rows if r["model"] == name and r["unit"] == u]) for u in subset if ctx[u]["feasible"].any()]
                Jp = float(np.mean(per_unit)); rho = (Jp - np.mean(J_c)) / (np.mean(J_o) - np.mean(J_c)) if abs(np.mean(J_o) - np.mean(J_c)) > 1e-9 else None
                rho_inf = (np.mean(I_c) - np.mean(inf_unit)) / np.mean(I_c) if I_c and np.mean(I_c) > 0 else None
                res[name] = {"J": Jp, "J_ci": boot_ci(per_unit), "infeasible": float(np.mean(inf_unit)) if inf_unit else None, "rho": rho, "rho_ci": boot_rho(per_unit, J_c, J_o) if rho is not None else (None, None), "rho_inf": rho_inf}
                r_ = res[name]
                lines.append(f"| {name} | {r_['J']:.3f} [{r_['J_ci'][0]:.3f}, {r_['J_ci'][1]:.3f}] | {fmt(r_['infeasible'], 3)} | {fmt(rho, 3)} {'' if r_['rho_ci'][0] is None else f'[{r_['rho_ci'][0]:.2f}, {r_['rho_ci'][1]:.2f}]'} | {fmt(rho_inf, 3)} |")
            return res
        lines.append("\n## D. Gate 3 — small learned model (MLP 2 × 64 from the same simulated probe observations + coarse bins → 48-cell feasibility mask; 4 grouped folds by object index; 20 noise draws per training unit; 5-member ensemble)\n")
        g3 = {"ambiguous_broad": gate3_table(amb_broad, "Ambiguous subset, broad"), "all": gate3_table(units, "All 192 units"),
              "ambiguous_strict": gate3_table(amb_strict, "Ambiguous subset, strict") if amb_strict else None}
        best3 = max(g3["ambiguous_broad"], key=lambda k: g3["ambiguous_broad"][k]["J"])
        R["gate3"] = g3; R["rho_small"] = g3["ambiguous_broad"][best3]["rho"]; R["gate3_best"] = best3
        lines.append(f"\n**Gate 3: best small model on the ambiguous subset = {best3}, ρ_small = {fmt(R['rho_small'], 3)}.**")

    # ------------------------------------------------------------------ matrix row
    lines.append("\n## Decision matrix\n")
    if not gate1_pass:
        row = "Gate 1 fails → stop the direction."
    elif gate2_stop:
        row = "ρ_analytic ≥ 0.8 → active identification + analytic control, no VLA."
    elif g3 is not None and R.get("rho_small") is not None and R["rho_small"] >= 0.8:
        row = "ρ_small close to 1 → learning paper, not VLA."
    else:
        row = "both insufficient → π0.5 worth testing (requires multiple language-conditioned tasks)."
    lines.append(f"- Gate 1 (ambiguous, broad): {'pass' if gate1_pass else 'fail'}; ρ_analytic = {fmt(rho_an, 3)}; ρ_small = {fmt(R.get('rho_small'), 3)}.")
    lines.append(f"- **Applicable row: {row}**")
    R["matrix_row"] = row
    (out / "report.md").write_text("\n".join(lines))
    (out / "gates.json").write_text(json.dumps(R, indent=1, default=lambda o: o.item() if hasattr(o, "item") else (list(o) if isinstance(o, (set, tuple)) else str(o))))
    (out / "probe_rows.json").write_text(json.dumps(probe_rows, indent=0, default=lambda o: list(o) if isinstance(o, tuple) else str(o)))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
