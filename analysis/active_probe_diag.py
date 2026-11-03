"""Diagnostic pass on the three-gate check v2 (owner request of 2026-11-03, before final closure). Existing 89-object
data only: no new objects, no learned model, no π0.5. Executes only analytic selections (inner-fold calibrations) in
the simulator; everything else is read from the cached v2 selections and executions.

  1. Updater-selection bias: nested selection of the analytic updater (for every outer fold the updater is chosen on
     inner folds of the training objects, each inner candidate calibrated on the inner training folds and EXECUTED on the
     inner held-out fold), plus the a-priori frozen updater (B5a on point estimates = the plan's primary rule B5a).
     ρ_analytic and the oracle − analytic gap with bootstrap CIs for both.
  2. Decomposition of the broad ambiguous units (all members feasible / mixed; action vs feasibility disagreement;
     ambiguity removable by a finer coarse vocabulary or by a continuous F), and where the Gate-1 oracle gain comes from.
  3. Case-by-case audit of the residual oracle − analytic gap with a fixed attribution procedure (stochastic boundary
     labels; boundary resolution; probe noise by property; CoM/torque physics; calibration; other), and a grouped
     out-of-fold test of whether the residual is predictable from observables.

Usage: MUJOCO_GL=egl .venv/bin/python -m analysis.active_probe_diag --sweeps data/sweep_v1,data/sweep_v1_ext \
    --objects data/objects_v1_ext.json --v2 results/phase1/active_probe_v2 --out results/phase1/active_probe_diag
"""
import argparse, json, pathlib, time
import numpy as np
from envs.feasibility_carry import feasible as fs
from envs.feasibility_carry.objects import BANDS
from baselines import analytic as B
from analysis.phase1_demo import load, accel_by_s, props, fmt
from analysis.phase1_full import in_refined, execute_selections, with_deadline, FRAGS, URGS, noisy_bins, NOISE_SEED
from analysis.coarse_bound2 import per_bin_means, est_from, argmax_cell
from analysis.active_probe_gates import probe_draw, boot_ci
from analysis.active_probe_gates2 import raw_bounds, select_batch, sel_dict, fit_mult_batch, rule_select, probe_est, boot_diff_ci, boot_rho, ci_s

VARIANTS = ["B5a-point", "B5b-point", "B5a-interval", "B5b-interval"]     # a-priori order (ties → earlier)
FROZEN = "B5a-point"                                                       # the plan's primary analytic rule, point estimates
ORACLE = "B5b-true-cal"; COARSE = "B5a-c"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweeps", required=True); ap.add_argument("--objects", required=True); ap.add_argument("--v2", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=12); ap.add_argument("--no_exec", action="store_true"); ap.add_argument("--n_folds", type=int, default=4)
    args = ap.parse_args(argv)
    t_start = time.perf_counter()
    out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True); v2 = pathlib.Path(args.v2)
    roots = [pathlib.Path(r) for r in args.sweeps.split(",")]
    objs = {o["object_id"]: o for o in json.loads(pathlib.Path(args.objects).read_text())["objects"]}
    def find(oid):
        for r in roots:
            if (r / oid / "probes.json").exists():
                return r / oid
        raise FileNotFoundError(oid)
    sweeps = {oid: load(find(oid)) for oid in objs}
    sw0 = next(iter(sweeps.values())); Fs, ss = sw0["meta"]["F"], sw0["meta"]["s"]; T_nom, T_max = sw0["meta"]["T_nominal"], sw0["meta"]["T_max"]
    ctx = {(oid, fr, urg): fs.context_tables(sw["rows"], Fs, ss, fr, urg, T_nom, T_max[urg]) for oid, sw in sweeps.items() for fr in FRAGS for urg in URGS}
    ref = {}
    for oid, sw in sweeps.items():
        for fr in FRAGS:
            iv = fs.refined_intervals(sw["rows"], sw["probes"], Fs, ss, fr, "relaxed", T_nom, T_max["relaxed"], n_boot=0)
            for urg in URGS:
                ref[(oid, fr, urg)] = with_deadline(iv, T_nom, T_max[urg])
    acc = {oid: accel_by_s(sw) for oid, sw in sweeps.items()}
    ok_urg = {urg: np.array([T_nom * s <= T_max[urg] for s in ss]) for urg in URGS}
    cell_of = {oid: f"{o['bins']['mass']}_{o['bins']['friction']}_{o['bins']['balance']}" for oid, o in objs.items()}
    idx_of = {oid: int(oid.rsplit("_", 1)[1]) for oid in objs}
    fold_of = {oid: idx_of[oid] % args.n_folds for oid in objs}
    units = [(oid, fr, urg) for oid in objs for fr in FRAGS for urg in URGS]
    oid_sorted = sorted(objs)
    truth = {u: (props(sweeps[u[0]])[0], props(sweeps[u[0]])[1], objs[u[0]]["f_tol"][u[1]], props(sweeps[u[0]])[2]) for u in units}
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
    amb = [u for u in units if cls[u] in ("ambiguous", "mixed")]
    feas_units = [u for u in units if cls[u] != "infeasible"]; inf_units = [u for u in units if cls[u] == "infeasible"]
    # cached v2 selections and outcomes (seed 0 executed; seeds 0–9 selections)
    S2 = json.loads((v2 / "selections.json").read_text()); G2 = json.loads((v2 / "gates.json").read_text())
    sel = {}; J = {}; INF = {}
    for r in S2:
        u = tuple(r["unit"]); sel[(r["method"], u, r["seed"])] = r["sel"]
        if r["seed"] == 0:
            J[(r["method"], u)] = r["success"]; INF[(r["method"], u)] = r["infeasible"]
    assert all(J[(ORACLE, u)] is not None for u in units), "v2 oracle not executed"
    for u in units:                      # noise-free coarse lookup, recomputed as in v2 (grid method, not in selections.json)
        oid, fr, urg = u; P = ctx[u]["P"]["deadline_success"]; feas = ctx[u]["feasible"]; ne = bool(feas.any())
        sib = [v for v in members[group_of[u]] if fold_of[v[0]] != fold_of[oid]]
        if sib:
            ij = argmax_cell(np.mean([ctx[v]["P"]["deadline_success"] for v in sib], axis=0), Fs, ss)
            J[("lookup", u)] = float(P[ij]); INF[("lookup", u)] = (bool(not feas[ij]) if ne else None)
        else:
            J[("lookup", u)] = 0.0; INF[("lookup", u)] = (True if ne else None)
    assert abs(np.mean([J[("lookup", u)] for u in amb]) - G2["gate1"]["ambiguous_broad"]["rows"]["lookup"]["success"]) < 1e-9
    J_coarse = {u: J[(COARSE, u)] for u in units}
    # probe observations and noisy bins as in v2 (same seeds)
    noise = {(oid, fr): noisy_bins(objs[oid]["bins"], fr, np.random.default_rng([NOISE_SEED, k, fi]))[0] for k, oid in enumerate(oid_sorted) for fi, fr in enumerate(FRAGS)}
    def draw_obs(seed):
        return {oid: probe_draw(*props(sweeps[oid]), np.random.default_rng([seed, k, 7])) for k, oid in enumerate(oid_sorted)}
    obs_by_seed = [draw_obs(s) for s in range(10)]
    # refit the outer multipliers (deterministic) and check them against v2
    mults = {}
    def calibrate(train_units, tag):
        train_obj = sorted({u[0] for u in train_units}); cal = per_bin_means(objs, train_obj)
        tr = [u for u in train_units if any(v["feasible"] for v in ref[u])]
        acc_tr = [acc[u[0]] for u in tr]; ok_tr = np.array([ok_urg[u[2]] for u in tr]); ref_tr = [ref[u] for u in tr]
        for rule in ("B5a", "B5b"):
            for mode in ("point", "interval"):
                mults[(tag, f"{rule}-{mode}")] = fit_mult_batch(rule, [probe_est(mode, obs_by_seed[0][u[0]], cal["fragility"][u[1]]) for u in tr], acc_tr, ok_tr, ref_tr, ss)
        return cal
    cal_outer = {}
    for k in range(args.n_folds):
        cal_outer[k] = calibrate([u for u in units if fold_of[u[0]] != k], ("outer", k))
        for v in VARIANTS:
            m2 = G2["multipliers"][f"fold{k}|{v}-probe"]; m1 = mults[(("outer", k), v)]
            assert all(abs(m1[q] - m2[q]) < 1e-9 for q in ("c_min", "c_max", "r_eff")), (k, v, m1, m2)
    R = {"objects": len(objs), "units": len(units), "ambiguous_broad": len(amb)}
    L = [f"# Active-probing check — diagnostic pass (owner request of 2026-11-03; {len(objs)} objects, {len(units)} units, {len(amb)} broad ambiguous)\n",
         "Existing data only (sweep v1 + extension, label v3). No new objects, no learned model, no π0.5. Only measured numbers; CIs are 95 % bootstrap over units (1000 resamples). "
         f"Oracle = {ORACLE} (out-of-sample, executed), J_coarse = {COARSE}, as in v2.\n"]

    # ============================================================== 1. nested updater selection
    L.append("## 1. Updater-selection bias: nested selection and the a-priori frozen updater\n")
    L.append("Nested scheme: for outer fold k, every inner fold j ≠ k is held out once; each of the four updaters is calibrated on the two remaining training folds "
             "(seed-0 probe observations, full calibration grid) and its selections on fold j are **executed** in the simulator (5 paired repeats). The updater with the highest "
             "mean safe success on the inner held-out *broad ambiguous* units (the population of ρ) is chosen for fold k (ties → a-priori order B5a-point, B5b-point, "
             f"B5a-interval, B5b-interval); fold k is then scored with that updater's outer selection from v2. Frozen updater = {FROZEN} (the plan's primary rule B5a with point estimates), fixed before this pass.\n")
    jobs = {}; inner_sel = {}
    def add_job(u, s_):
        if s_ is not None:
            key = f"{s_['F']:.3f}|{s_['s']:.4f}"; s_["exec_key"] = key
            jobs.setdefault(str(sweeps[u[0]]["dir"]), {})[key] = (float(s_["F"]), float(s_["s"]))
    for k in range(args.n_folds):
        for j in range(args.n_folds):
            if j == k: continue
            tr = [u for u in units if fold_of[u[0]] not in (k, j)]; te = [u for u in units if fold_of[u[0]] == j]
            cal = calibrate(tr, ("inner", k, j))
            for u in te:
                oid, fr, urg = u
                for v in VARIANTS:
                    rule, mode = v.split("-")
                    s_ = rule_select(rule, probe_est(mode, obs_by_seed[0][oid], cal["fragility"][fr]), acc[oid], ss, ok_urg[urg], mults[(("inner", k, j), v)])
                    inner_sel[(k, j, v, u)] = s_; add_job(u, s_)
    n_jobs = sum(len(x) for x in jobs.values()); print(f"inner-fold selections: {n_jobs} distinct (cached ones skipped)", flush=True)
    t0 = time.perf_counter()
    exec_res = execute_selections(jobs, args.workers) if (jobs and not args.no_exec) else {}
    print(f"execution {(time.perf_counter() - t0) / 60:.1f} min", flush=True)
    def outcome(u, s_):
        if s_ is None: return 0.0
        e = exec_res.get(str(sweeps[u[0]]["dir"]), {}).get(s_["exec_key"])
        return None if e is None else float(e[f"deadline_success|{u[1]}|{u[2]}"])
    inner_score = {}; chosen = {}
    for k in range(args.n_folds):
        for v in VARIANTS:
            vals = [outcome(u, inner_sel[(k, j, v, u)]) for j in range(args.n_folds) if j != k for u in amb if fold_of[u[0]] == j]
            vals_all = [outcome(u, inner_sel[(k, j, v, u)]) for j in range(args.n_folds) if j != k for u in units if fold_of[u[0]] == j]
            inner_score[(k, v)] = (float(np.mean([x for x in vals if x is not None])) if vals else None, float(np.mean([x for x in vals_all if x is not None])) if vals_all else None)
        chosen[k] = max(VARIANTS, key=lambda v: (inner_score[(k, v)][0] if inner_score[(k, v)][0] is not None else -1, -VARIANTS.index(v)))
    chosen_all = {k: max(VARIANTS, key=lambda v: (inner_score[(k, v)][1] if inner_score[(k, v)][1] is not None else -1, -VARIANTS.index(v))) for k in range(args.n_folds)}
    L.append("| outer fold | " + " | ".join(f"inner success {v} (ambiguous / all)" for v in VARIANTS) + " | chosen (ambiguous) | chosen (all units) |"); L.append("|---|" + "---|" * (len(VARIANTS) + 2))
    for k in range(args.n_folds):
        L.append(f"| {k} | " + " | ".join(f"{fmt(inner_score[(k, v)][0], 3)} / {fmt(inner_score[(k, v)][1], 3)}" for v in VARIANTS) + f" | {chosen[k]} | {chosen_all[k]} |")
    def J_of(method_of_unit):
        return {u: J[(f"{method_of_unit(u)}-probe", u)] for u in units}
    J_nested = J_of(lambda u: chosen[fold_of[u[0]]]); J_nested_all = J_of(lambda u: chosen_all[fold_of[u[0]]]); J_frozen = J_of(lambda u: FROZEN)
    J_or = {u: J[(ORACLE, u)] for u in units}
    def rho_row(name, Jm, subset):
        jp = [Jm[u] for u in subset]; jc = [J_coarse[u] for u in subset]; jo = [J_or[u] for u in subset]
        rho = (np.mean(jp) - np.mean(jc)) / (np.mean(jo) - np.mean(jc)); gap = np.mean(jo) - np.mean(jp)
        return {"name": name, "success": float(np.mean(jp)), "success_ci": boot_ci(jp), "gap": float(gap), "gap_ci": boot_diff_ci(jo, jp), "rho": float(rho), "rho_ci": boot_rho(jp, jc, jo)}
    sec1 = {}
    for key, name, subset in (("ambiguous_broad", f"Broad ambiguous units ({len(amb)})", amb), ("all", f"All {len(units)} units", units), ("feasible", f"Feasible units ({len(feas_units)})", feas_units)):
        rows = [rho_row("nested (chosen per fold on inner ambiguous units)", J_nested, subset), rho_row("nested (chosen per fold on all inner units)", J_nested_all, subset),
                rho_row(f"frozen a priori: {FROZEN}", J_frozen, subset)] + [rho_row(f"{v} (v2, in-sample choice among four)", J_of(lambda u, v=v: v), subset) for v in VARIANTS]
        sec1[key] = rows
        L.append(f"\n### {name}\n"); L.append("| updater | safe success [CI] | gap = oracle − analytic [CI] | ρ_analytic [CI] |"); L.append("|---|---|---|---|")
        for r in rows:
            L.append(f"| {r['name']} | {r['success']:.3f} {ci_s(r['success_ci'])} | {r['gap']:+.3f} {ci_s(r['gap_ci'])} | {r['rho']:.3f} {ci_s(r['rho_ci'], 2)} |")
    R["nested"] = {"chosen_per_fold_ambiguous": chosen, "chosen_per_fold_all": chosen_all, "inner_scores": {f"fold{k}|{v}": inner_score[(k, v)] for k in range(args.n_folds) for v in VARIANTS}, "rows": sec1,
                   "n_inner_selections": n_jobs}
    nb = sec1["ambiguous_broad"][0]; fz = sec1["ambiguous_broad"][2]
    L.append(f"\n**Result.** Nested ρ_analytic = **{nb['rho']:.3f} {ci_s(nb['rho_ci'], 2)}**, gap **{nb['gap']:+.3f} {ci_s(nb['gap_ci'])}**; frozen {FROZEN}: ρ = {fz['rho']:.3f} {ci_s(fz['rho_ci'], 2)}, gap {fz['gap']:+.3f} {ci_s(fz['gap_ci'])}. "
             f"The v2 in-sample choice (B5b-point on the same 65 units) gave 0.833 [0.67, 0.96]; the difference to the nested value is the selection optimism.")

    # ============================================================== 2. decomposition of the broad ambiguous units
    L.append("\n## 2. Decomposition of the broad ambiguous units\n")
    sub_bins = {}
    def sub_bin(oid):
        o = objs[oid]; b = o["bins"]; parts = []
        for attr, val in (("mass", o["mass"]), ("friction", o["mu"]), ("balance", o["com_frac"])):
            lo, hi = BANDS[attr][b[attr]]; parts.append(f"{b[attr]}{'-lo' if val < 0.5 * (lo + hi) else '-hi'}")
        return "_".join(parts)
    for u in units:
        sub_bins.setdefault((sub_bin(u[0]), u[1], u[2]), []).append(u)
    def common_cont(us):
        """Common continuous F at some s level: the refined intervals of all units intersect at that s."""
        for j in range(len(ss)):
            lo = max((ref[u][j]["F_min"] if ref[u][j]["feasible"] else np.inf) for u in us); hi = min((ref[u][j]["F_max"] if ref[u][j]["feasible"] else -np.inf) for u in us)
            if all(ref[u][j]["feasible"] for u in us) and lo <= hi: return True
        return False
    dec = {}
    for u in amb:
        sib = members[group_of[u]]; feas_sib = [v for v in sib if ctx[v]["feasible"].any()]; inf_sib = [v for v in sib if not ctx[v]["feasible"].any()]
        d = {"n_bin": len(sib), "n_feasible": len(feas_sib), "n_infeasible": len(inf_sib), "all_feasible": cls[u] == "ambiguous",
             "feasible_common_cell": common_cell(feas_sib),                   # True → feasibility disagreement only
             "feasible_common_continuous": common_cont(feas_sib),             # True → a common continuous F exists among the feasible siblings (grid artefact if no common cell)
             "subbin_common_cell": common_cell([v for v in sub_bins[(sub_bin(u[0]), u[1], u[2])] if ctx[v]["feasible"].any()]) and all(ctx[v]["feasible"].any() for v in sub_bins[(sub_bin(u[0]), u[1], u[2])]),
             "subbin_n": len(sub_bins[(sub_bin(u[0]), u[1], u[2])])}
        d["action_disagreement"] = not d["feasible_common_cell"]; d["feasibility_disagreement"] = len(inf_sib) > 0
        dec[u] = d
    n = lambda p: sum(1 for u in amb if p(dec[u]))
    rows2 = [("all members of the bin individually feasible (strict)", n(lambda d: d["all_feasible"])),
             ("mixed: at least one sibling intrinsically infeasible", n(lambda d: d["feasibility_disagreement"])),
             ("feasibility disagreement only (the feasible siblings share a grid cell)", n(lambda d: d["feasibility_disagreement"] and d["feasible_common_cell"])),
             ("action disagreement (the feasible siblings have no common grid cell)", n(lambda d: d["action_disagreement"])),
             ("  of which a common *continuous* F exists at some s (grid-discretisation artefact)", n(lambda d: d["action_disagreement"] and d["feasible_common_continuous"])),
             ("ambiguity removable by a finer coarse vocabulary (half-width mass/friction/balance sub-bins: all sub-bin siblings feasible with a common cell)", n(lambda d: d["subbin_common_cell"])),
             ("  of which also action disagreement at the full bin", n(lambda d: d["subbin_common_cell"] and d["action_disagreement"]))]
    L.append("| class (flags overlap) | units of 65 |"); L.append("|---|---|")
    for name, c in rows2: L.append(f"| {name} | {c} |")
    part = {}
    for u in amb:
        d = dec[u]; key = ("feasibility-only" if d["feasible_common_cell"] else "action+feasibility") + (" / sub-bin-removable" if d["subbin_common_cell"] else " / persists in sub-bins")
        part[key] = part.get(key, 0) + 1
    L.append("\nMutually exclusive partition (mixed units; no strict unit exists):\n"); L.append("| partition | units |"); L.append("|---|---|")
    for k_, c in sorted(part.items()): L.append(f"| {k_} | {c} |")
    L.append(f"\nSub-bins: each band split at its midpoint (mass 3 → 6, friction 2 → 4, off-centre 1 → 2; balanced stays one), {min(dec[u]['subbin_n'] for u in amb)}–{max(dec[u]['subbin_n'] for u in amb)} units per sub-bin group on the ambiguous units, "
             "so a common cell is mechanically easier to find there; the row is an upper bound on what a finer vocabulary removes.")
    # where the Gate-1 gain comes from
    J_lk = {u: J[("lookup", u)] for u in units}; I_lk = {u: INF[("lookup", u)] for u in units}
    gain = {u: J_or[u] - J_lk[u] for u in units}
    def gsum(subset, pred=lambda u: True):
        g = [gain[u] for u in subset if pred(u)]; return float(np.sum(g)), len(g)
    tot, _ = gsum(amb)
    g_wrong = gsum(amb, lambda u: I_lk[u] is True); g_feas = gsum(amb, lambda u: I_lk[u] is False)
    g_feas_sto = gsum(amb, lambda u: I_lk[u] is False and 0 < J_lk[u] < 1); g_neg = gsum(amb, lambda u: gain[u] < 0)
    L.append(f"\n### Where the Gate-1 gain on the 65 units comes from (oracle − lookup = {tot / len(amb):+.3f} success points; sum over units {tot:+.1f})\n")
    L.append("| lookup outcome on the unit | units | summed gain | share of the gain |"); L.append("|---|---|---|---|")
    for name, (g, c) in (("lookup cell infeasible for the unit (a different action was needed)", g_wrong), ("lookup cell feasible for the unit (gain = stochastic execution difference)", g_feas),
                         ("  of which lookup success strictly between 0 and 1", g_feas_sto), ("units with negative gain (lookup above oracle)", g_neg)):
        L.append(f"| {name} | {c} | {g:+.1f} | {g / tot:.2f} |")
    L.append("\nEvery unit of this population is feasible by definition, so **no part of the +0.265 comes from identifying infeasible instances**; it comes from choosing a different action "
             f"({g_wrong[0] / tot:.0%} of the gain, {g_wrong[1]} units where the lookup cell lies outside the unit's feasible set) and from execution noise on feasible cells ({g_feas[0] / tot:.0%}).")
    # the all-units gain and the infeasible instances
    tot_all, _ = gsum(units); parts_all = {k_: gsum(units, lambda u, k_=k_: cls[u] == k_) for k_ in ("common-safe", "mixed", "ambiguous", "infeasible")}
    L.append(f"\nOn all {len(units)} units (oracle − lookup = {tot_all / len(units):+.3f}): " + "; ".join(f"{k_} {c} units {g:+.1f} ({g / tot_all:.0%})" for k_, (g, c) in parts_all.items() if c) + ".")
    EXEC_METHODS = (ORACLE, "B5b-point-probe", "B5a-point-probe", "B5a-c")
    abst = {m: sum(1 for u in inf_units if sel[(m, u, 0)] is None) for m in EXEC_METHODS}
    exec_cache = {oid: json.loads((pathlib.Path(sweeps[oid]["dir"]) / "p4_exec.json").read_text()) for oid in objs}
    dmg = {}
    for m in EXEC_METHODS:
        vals = []
        for u in inf_units:
            s_ = sel[(m, u, 0)]
            if s_ is None: vals.append(0.0); continue
            e = exec_cache[u[0]].get(s_["exec_key"])
            vals.append(float(e[f"damage|{u[1]}"] > fs.TAU["d"]) if e else np.nan)
        dmg[m] = float(np.nanmean(vals))
    L.append(f"\nIdentifying infeasible instances ({len(inf_units)} units with an empty feasible set) is not credited by the success metric; it shows in abstention and damage: " +
             "; ".join(f"{m} abstains on {abst.get(m, 'n/a')} / {len(inf_units)}" + (f", attempts with damage on {dmg[m]:.2f}" if m in dmg else "") for m in list(abst)) +
             ". The lookup never abstains (it always plays the siblings' best cell).")
    R["decomposition"] = {"flags": {name: c for name, c in rows2}, "partition": part, "gain_ambiguous": {"total": tot, "wrong_action": g_wrong, "feasible_cell": g_feas, "feasible_stochastic": g_feas_sto, "negative": g_neg},
                          "gain_all": {k_: v for k_, v in parts_all.items()}, "abstain_infeasible": abst, "damage_on_infeasible": dmg, "per_unit": {"|".join(u): dec[u] for u in amb}}

    # ============================================================== 3. residual audit
    L.append("\n## 3. Case-by-case audit of the residual oracle − analytic gap\n")
    L.append("Cases: broad ambiguous units where the oracle's executed success exceeds the analytic updater's (seed-0 observation). Attribution procedure, applied in this order to each case "
             "(label-based counterfactuals use the refined simulated interval of the unit; only the actual selections were executed):\n"
             "1. **stochastic boundary labels** — the analytic selection lies inside the refined interval (feasible by the labels) but its execution fell short;\n"
             "2. **boundary resolution** — the selection lies outside the refined interval by no more than the logistic width w of the boundary fit at that s (F/s grid and bisection resolution);\n"
             "3. **probe-estimation noise (m / μ / CoM)** — substituting the true value of one property into the estimate (same rule, same multipliers) moves the selection inside the interval; "
             "the property with the fewest substitutions needed, ties by the largest standardised error;\n"
             "4. **omitted CoM/torque physics** — substituting all true properties does not fix a B5a updater while the B5b rule on true properties (same calibration family) does;\n"
             "5. **calibration** — the uncalibrated rule (c = 1, nominal r_eff) on the same noisy estimate selects inside the interval, or all true properties fix it only through the multipliers;\n"
             "6. **other** — none of the above.\n")
    def audit(variant):
        rule, mode = variant.split("-"); cases = []
        for u in amb:
            jo, jp = J_or[u], J[(f"{variant}-probe", u)]
            if jp is None or jo - jp <= 0: continue
            oid, fr, urg = u; k = fold_of[oid]; mm = mults[(("outer", k), variant)]; ob = obs_by_seed[0][oid]; ok = ok_urg[urg]; cal = cal_outer[k]
            est = probe_est(mode, ob, cal["fragility"][fr]); s_ = sel[(f"{variant}-probe", u, 0)]; iv = ref[u]
            c = {"unit": "|".join(u), "bin": cell_of[oid], "oracle": jo, "analytic": jp, "diff": jo - jp, "sel": s_, "oracle_sel": sel[(ORACLE, u, 0)],
                 "est": {"m": ob["m"], "mu": ob["mu"], "d": ob["d"]}, "true": {"m": truth[u][0], "mu": truth[u][1], "d": truth[u][3]},
                 "z": {"m": (ob["m"] - truth[u][0]) / ob["sm"], "mu": (ob["mu"] - truth[u][1]) / ob["smu"], "d": (abs(ob["d"]) - truth[u][3]) / ob["sd"] if ob["sd"] > 0 else 0.0}}
            inref = in_refined(iv, s_)
            if s_ is None:
                c["miss"] = "abstained"; dist = None
            else:
                v = iv[s_["s_idx"]]
                if not v["feasible"]: c["miss"] = "s level infeasible"; dist = None
                elif s_["F"] < v["F_min"]: c["miss"] = "F below slip boundary"; dist = v["F_min"] - s_["F"]
                elif s_["F"] > v["F_max"]: c["miss"] = "F above damage boundary"; dist = s_["F"] - v["F_max"]
                else: c["miss"] = "inside interval"; dist = 0.0
                c["w"] = (v["slip_fit"]["w"] if c["miss"] == "F below slip boundary" and v["slip_fit"] else v["damage_fit"]["w"] if c["miss"] == "F above damage boundary" and v["damage_fit"] else None)
            c["dist"] = dist
            fix = {}
            for prop_, i in (("m", 0), ("mu", 1), ("d", 3)):
                e2 = list(est); e2[i] = truth[u][i]; fix[prop_] = bool(in_refined(iv, rule_select(rule, tuple(e2), acc[oid], ss, ok, mm)))
            fix["all"] = bool(in_refined(iv, rule_select(rule, truth[u], acc[oid], ss, ok, mm)))
            fix["uncal"] = bool(in_refined(iv, rule_select(rule, est, acc[oid], ss, ok, {"c_min": 1.0, "c_max": 1.0, "r_eff": B.R_EFF})))
            fix["B5b_true_same_cal"] = bool(in_refined(iv, rule_select("B5b", truth[u], acc[oid], ss, ok, mults[(("outer", k), f"B5b-{mode}")])))
            c["fix"] = fix
            if inref: c["cause"] = "stochastic boundary labels"
            elif dist is not None and c.get("w") is not None and dist <= c["w"]: c["cause"] = "boundary resolution"
            elif any(fix[p] for p in ("m", "mu", "d")):
                cands = [p for p in ("m", "mu", "d") if fix[p]]; p = max(cands, key=lambda p: abs(c["z"][p])); c["cause"] = f"probe noise ({p})"
            elif not fix["all"] and rule == "B5a" and fix["B5b_true_same_cal"]: c["cause"] = "omitted CoM/torque physics"
            elif fix["uncal"] or fix["all"]: c["cause"] = "calibration"
            else: c["cause"] = "other"
            cases.append(c)
        return cases
    audits = {}
    for variant in sorted({chosen[k] for k in chosen} | {"B5b-point", FROZEN}, key=VARIANTS.index):
        cases = audit(variant); audits[variant] = cases
        tag = " (nested choice)" if variant in chosen.values() else ""
        L.append(f"\n### Updater {variant}{tag}: {len(cases)} cases, summed deficit {sum(c['diff'] for c in cases):.1f} of {len(amb)} units ({sum(c['diff'] for c in cases) / len(amb):.3f} success points)\n")
        causes = {}
        for c in cases: causes.setdefault(c["cause"], []).append(c)
        L.append("| cause | cases | summed deficit | share |"); L.append("|---|---|---|---|")
        for cz, cs in sorted(causes.items(), key=lambda kv: -sum(c["diff"] for c in kv[1])):
            L.append(f"| {cz} | {len(cs)} | {sum(c['diff'] for c in cs):.1f} | {sum(c['diff'] for c in cs) / max(sum(c['diff'] for c in cases), 1e-9):.2f} |")
        L.append("\n| unit | bin | oracle | analytic | miss | dist (N) | w (N) | z(m̂) | z(μ̂) | z(d̂) | fixes: m / μ / d / all / uncal | cause |"); L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for c in cases:
            f_ = c["fix"]; L.append(f"| {c['unit']} | {c['bin']} | {c['oracle']:.1f} | {c['analytic']:.1f} | {c['miss']} | {fmt(c['dist'], 2)} | {fmt(c.get('w'), 2)} | {c['z']['m']:+.1f} | {c['z']['mu']:+.1f} | {c['z']['d']:+.1f} | "
                      f"{int(f_['m'])} / {int(f_['mu'])} / {int(f_['d'])} / {int(f_['all'])} / {int(f_['uncal'])} | {c['cause']} |")
    R["audit"] = audits

    # ---- systematic pattern test: label-based residual over 65 units × 10 seeds, grouped out-of-fold
    L.append("\n### Is the residual systematic and observable? (grouped out-of-fold test)\n")
    L.append("Population: the 65 broad ambiguous units × 10 probe-noise seeds, analytic selections from v2 (label-based outcome: selection inside the refined interval). "
             "Features available to a residual corrector at run time: coarse bin (one-hot), fragility, urgency, the probe estimates (m̂, μ̂, d̂), the selected F and s. "
             "Model: ridge regression (λ = 1) on standardised features, trained on three folds, scored on the held-out fold (objects grouped). A predictable residual shows as out-of-fold R² > 0 for the signed "
             "margin to the nearest violated boundary and as out-of-fold AUC > 0.5 for the failure indicator.\n")
    cells = sorted(set(cell_of.values()))
    def features(u, s_, ob):
        x = [1.0 * (cell_of[u[0]] == c) for c in cells] + [1.0 * (u[1] == "fragile"), 1.0 * (u[2] == "urgent"), ob["m"], ob["mu"], abs(ob["d"]) * 1e2, s_["F"] / 10, fs.s_norm(s_["s"])]
        return x
    pat = {}
    for variant in audits:
        X, y_fail, y_marg, g = [], [], [], []
        for u in amb:
            for seed in range(10):
                s_ = sel[(f"{variant}-probe", u, seed)]
                if s_ is None: continue
                v = ref[u][s_["s_idx"]]; ob = obs_by_seed[seed][u[0]]
                if not v["feasible"]: marg = -10.0
                else: marg = min(s_["F"] - v["F_min"], v["F_max"] - s_["F"])
                X.append(features(u, s_, ob)); y_fail.append(float(marg < 0)); y_marg.append(marg); g.append(fold_of[u[0]])
        X = np.array(X); y_fail = np.array(y_fail); y_marg = np.array(y_marg); g = np.array(g)
        pred_m = np.zeros(len(y_marg)); pred_f = np.zeros(len(y_fail))
        for k in range(args.n_folds):
            tr, te = g != k, g == k
            if te.sum() == 0: continue
            mu_, sd_ = X[tr].mean(0), X[tr].std(0) + 1e-9; Xt = (X - mu_) / sd_; Xa = np.hstack([Xt, np.ones((len(Xt), 1))])
            for y, pred in ((y_marg, pred_m), (y_fail, pred_f)):
                w = np.linalg.solve(Xa[tr].T @ Xa[tr] + 1.0 * np.eye(Xa.shape[1]), Xa[tr].T @ y[tr]); pred[te] = Xa[te] @ w
        r2 = 1 - np.sum((y_marg - pred_m) ** 2) / np.sum((y_marg - y_marg.mean()) ** 2)
        pos, neg = pred_f[y_fail == 1], pred_f[y_fail == 0]
        auc = float(np.mean([(p > q) + 0.5 * (p == q) for p in pos for q in neg])) if len(pos) and len(neg) else None
        fail_by_cell = {c: (float(y_fail[X[:, cells.index(c)] == 1].mean()), int((X[:, cells.index(c)] == 1).sum())) for c in cells if (X[:, cells.index(c)] == 1).any()}
        n_fail = int(y_fail.sum())
        tally = {}; by_obj = {}
        for u in amb:
            for seed in range(10):
                s_ = sel[(f"{variant}-probe", u, seed)]
                if s_ is None: k_ = "abstained"
                else:
                    v = ref[u][s_["s_idx"]]; k_ = "s level infeasible" if not v["feasible"] else "F below slip boundary" if s_["F"] < v["F_min"] else "F above damage boundary" if s_["F"] > v["F_max"] else "inside"
                tally[k_] = tally.get(k_, 0) + 1
                if k_ != "inside": by_obj[u[0]] = by_obj.get(u[0], 0) + 1
        top_obj = sorted(by_obj.items(), key=lambda kv: -kv[1])[:6]
        pat[variant] = {"n": int(len(y_fail)), "n_fail": n_fail, "fail_rate": float(y_fail.mean()), "oof_r2_margin": float(r2), "oof_auc_failure": auc, "fail_by_cell": fail_by_cell, "miss_direction": tally, "misses_by_object": by_obj}
        L.append(f"- **{variant}:** {len(y_fail)} selections, {n_fail} outside the interval ({y_fail.mean():.3f}); out-of-fold R² of the signed margin **{r2:.3f}**, out-of-fold AUC of the failure indicator **{fmt(auc, 3)}**. "
                 "Failure rate by bin: " + ", ".join(f"{c} {r:.2f} (n = {n_})" for c, (r, n_) in sorted(fail_by_cell.items(), key=lambda kv: -kv[1][0])) + ". Miss direction over all 650 unit × seed selections: " + ", ".join(f"{k_} {v_}" for k_, v_ in sorted(tally.items(), key=lambda kv: -kv[1])) +
                 f". Misses concentrate in objects (misses of 10 seeds; {len(by_obj)} objects with any miss): " + ", ".join(f"{o} {c}" for o, c in top_obj) + ".")
    R["pattern"] = pat

    # ============================================================== summary
    R["wall_min"] = (time.perf_counter() - t_start) / 60
    L.append(f"\n## Summary\n")
    L.append(f"- Nested ρ_analytic {nb['rho']:.3f} {ci_s(nb['rho_ci'], 2)} (gap {nb['gap']:+.3f} {ci_s(nb['gap_ci'])}); frozen {FROZEN} ρ {fz['rho']:.3f} {ci_s(fz['rho_ci'], 2)} (gap {fz['gap']:+.3f} {ci_s(fz['gap_ci'])}); chosen updaters per fold {[chosen[k] for k in range(args.n_folds)]}.")
    L.append(f"- Broad ambiguous units: strict {rows2[0][1]}, mixed {rows2[1][1]}; feasibility-disagreement-only {rows2[2][1]}, action disagreement {rows2[3][1]} (continuous common F {rows2[4][1]}); sub-bin-removable {rows2[5][1]}. "
             f"Gain +{tot / len(amb):.3f}: {g_wrong[0] / tot:.0%} from a different action on {g_wrong[1]} units, {g_feas[0] / tot:.0%} execution noise, 0 % from identifying infeasible instances.")
    for variant, cases in audits.items():
        causes = {}
        for c in cases: causes[c["cause"]] = causes.get(c["cause"], 0) + c["diff"]
        L.append(f"- Residual audit {variant}: {len(cases)} cases; " + ", ".join(f"{cz} {v_:.1f}" for cz, v_ in sorted(causes.items(), key=lambda kv: -kv[1])) + f"; out-of-fold R² {pat[variant]['oof_r2_margin']:.2f}, AUC {fmt(pat[variant]['oof_auc_failure'], 2)}.")
    L.append(f"\nWall {R['wall_min']:.1f} min.")
    (out / "report.md").write_text("\n".join(L) + "\n")
    (out / "diag.json").write_text(json.dumps(R, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    print("\n".join(L[-8:]))


if __name__ == "__main__":
    main()
