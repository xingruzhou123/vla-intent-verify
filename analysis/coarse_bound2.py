"""Coarse-information bound, second version (owner items 3–5 of 2026-11-02):
- noise-free coarse-cell lookup (item 3 baseline), a noisy coarse lookup that sees the ε = 0.1 bin noise of plan §3.6 on
  both the training objects (table keyed by their noisy bins) and the held-out object (query with its noisy bins),
  the per-object oracle, the best single fixed (F, s), and B5a / B5b calibrated on the fold's training split and executed
  in the simulator (docs/analytic_baselines.md §4);
- leave-one-object-out over the 4 objects of every bin cell (item 4): fold k holds out object k of each cell (fold 3 is
  the canonical train / test-combos split of data/objects_v1.json); mean and spread across folds;
- every table split by difficulty (item 5): the (cell, fragility, urgency) groups with a grid (F, s) feasible for all
  four objects of the cell versus the groups without one.

Success = empirical P(deadline success) of the chosen grid cell on the held-out object (0 when nothing is chosen);
for the analytic rules the continuous selection is executed with R paired repeats from the stored branch states.
Infeasible selection = the chosen cell is outside the held-out context's feasible set 𝓕 (plan §3.3), the executed
𝓕 test for the rules.

Usage: MUJOCO_GL=egl .venv/bin/python -m analysis.coarse_bound2 --sweeps data/sweep_v1 --objects data/objects_v1.json --out results/phase1/coarse_bound_v2 [--no_exec] [--noise_seeds 20]
"""
import argparse, json, pathlib, itertools
import numpy as np
from envs.feasibility_carry import feasible as fs
from envs.feasibility_carry.objects import CELLS
from baselines import analytic as B
from analysis.phase1_demo import load, accel_by_s, props, fmt
from analysis.phase1_full import (noisy_bins, rule_intervals, in_refined, execute_selections, with_deadline, FRAGS, URGS,
                                  ATTR_KEYS, CAL_GRID, NOISE_SEED, NOISE_EPS)

METHODS = ["oracle", "lookup", "lookup_noisy", "best_fixed", "B5a-c", "B5b-c"]


def per_bin_means(objs, train):
    cal = {a: {} for a in ATTR_KEYS}
    for key in ATTR_KEYS["mass"]:
        cal["mass"][key] = float(np.mean([objs[o]["mass"] for o in train if objs[o]["bins"]["mass"] == key]))
    for key in ATTR_KEYS["friction"]:
        cal["friction"][key] = float(np.mean([objs[o]["mu"] for o in train if objs[o]["bins"]["friction"] == key]))
    for key in ATTR_KEYS["balance"]:
        cal["balance"][key] = float(np.mean([objs[o]["com_frac"] * 0.04 for o in train if objs[o]["bins"]["balance"] == key]))
    for key in ATTR_KEYS["fragility"]:
        cal["fragility"][key] = float(np.mean([objs[o]["f_tol"][key] for o in train]))
    return cal


def est_from(table, b):
    return (table["mass"][b["mass"]], table["friction"][b["friction"]], table["fragility"][b["fragility"]], table["balance"][b["balance"]])


def fit_multipliers(rule, train_ctx, cal, noise, acc, ref, ss, T_nom, T_max):
    best = None
    r_grid = CAL_GRID["r_eff"] if rule == "B5b" else [B.R_EFF]
    for c_min, c_max, r_eff in itertools.product(CAL_GRID["c_min"], CAL_GRID["c_max"], r_grid):
        hits = 0
        for (oid, frag, urg) in train_ctx:
            ivr = rule_intervals(rule, est_from(cal, noise[(oid, frag)][0]), acc[oid], ss, T_nom, T_max[urg], c_min, c_max, r_eff)
            sel, _ = fs.select_refined(ivr)
            hits += bool(in_refined(ref[(oid, frag, urg)], sel))
        key = (hits, -abs(np.log(c_min)) - abs(np.log(c_max)))
        if best is None or key > best[0]:
            best = (key, {"c_min": float(c_min), "c_max": float(c_max), "r_eff": float(r_eff), "train_success": hits / len(train_ctx)})
    return best[1]


def argmax_cell(Pm, Fs, ss):
    sN = fs.s_norm(ss); s0 = fs.s_norm(1.0); best, arg = None, None
    for i in range(len(Fs)):
        for j in range(len(ss)):
            key = (round(float(Pm[i, j]), 9), -abs(sN[j] - s0), -abs(i - (len(Fs) - 1) / 2))
            if best is None or key > best:
                best, arg = key, (i, j)
    return arg


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweeps", required=True); ap.add_argument("--objects", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--no_exec", action="store_true"); ap.add_argument("--workers", type=int, default=12); ap.add_argument("--noise_seeds", type=int, default=20)
    args = ap.parse_args(argv)
    out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)
    objs = {o["object_id"]: o for o in json.loads(pathlib.Path(args.objects).read_text())["objects"]}
    sweeps = {oid: load(pathlib.Path(args.sweeps) / oid) for oid in objs}
    sw0 = next(iter(sweeps.values())); Fs, ss = sw0["meta"]["F"], sw0["meta"]["s"]; T_nom, T_max = sw0["meta"]["T_nominal"], sw0["meta"]["T_max"]
    label_version = sw0["meta"].get("label_version", 1)
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
    cells = [f"{mb}_{fb}_{bb}" for (mb, fb, bb) in CELLS]
    # difficulty split on all four objects (fold-independent)
    common = {}
    for cell in cells:
        members = [o for o in objs if cell_of[o] == cell]
        for fr in FRAGS:
            for urg in URGS:
                allf = np.logical_and.reduce([ctx[(o, fr, urg)]["feasible"] for o in members])
                anyf = any(ctx[(o, fr, urg)]["feasible"].any() for o in members)
                common[(cell, fr, urg)] = {"common": bool(allf.any()), "n_common": int(allf.sum()), "any_feasible": anyf}
    # noise realisations: the spec draw (seed 0, same as phase1_full) plus extra seeds for the spread of the noisy lookup
    def draw(seed):
        d = {}
        for k, oid in enumerate(sorted(sweeps)):
            for fi, frag in enumerate(FRAGS):
                d[(oid, frag)] = noisy_bins(objs[oid]["bins"], frag, np.random.default_rng([seed, k, fi]))
        return d
    noise = draw(NOISE_SEED)
    extra = [draw(s) for s in range(1, args.noise_seeds)]

    rows = []            # one row per (fold, group, method)
    jobs = {}
    sel_rows = []
    for fold in range(4):
        test = [o for o in objs if idx_of[o] == fold]; train = [o for o in objs if idx_of[o] != fold]
        cal = per_bin_means(objs, train)
        train_ctx = [(o, fr, urg) for o in train for fr in FRAGS for urg in URGS if any(v["feasible"] for v in ref[(o, fr, urg)])]
        mult = {rule: fit_multipliers(rule, train_ctx, cal, noise, acc, ref, ss, T_nom, T_max) for rule in ("B5a", "B5b")}
        # best single fixed over the fold's training contexts
        Pall = np.mean([ctx[(o, fr, urg)]["P"]["deadline_success"] for o in train for fr in FRAGS for urg in URGS], axis=0)
        bsf = argmax_cell(Pall, Fs, ss)
        # noisy lookup tables (spec draw and extra seeds): key = noisy (mass, friction, balance, fragility) bins
        def table(nz):
            T = {}
            for o in train:
                for fr in FRAGS:
                    nb = nz[(o, fr)][0]; key = (nb["mass"], nb["friction"], nb["balance"], nb["fragility"])
                    for urg in URGS:
                        T.setdefault((key, urg), []).append(ctx[(o, fr, urg)]["P"]["deadline_success"])
            return {k: argmax_cell(np.mean(v, axis=0), Fs, ss) for k, v in T.items()}
        tab0 = table(noise); tabs = [table(nz) for nz in extra]
        for t in test:
            cell = cell_of[t]; members = [o for o in train if cell_of[o] == cell]
            for fr in FRAGS:
                for urg in URGS:
                    c = ctx[(t, fr, urg)]; P = c["P"]["deadline_success"]; feas = c["feasible"]; nonempty = bool(feas.any())
                    def rec(method, cellij, extra_fields=None):
                        r = {"fold": fold, "cell": cell, "frag": fr, "urg": urg, "test": t, "method": method, "truth_nonempty": nonempty,
                             "common": common[(cell, fr, urg)]["common"], "any_feasible": common[(cell, fr, urg)]["any_feasible"]}
                        if cellij is None:
                            r.update({"F": None, "s": None, "success": 0.0, "infeasible": True if nonempty else None})
                        else:
                            i, j = cellij; r.update({"F": Fs[i], "s": ss[j], "success": float(P[i, j]), "infeasible": bool(not feas[i, j])})
                        if extra_fields: r.update(extra_fields)
                        rows.append(r); return r
                    # oracle
                    rec("oracle", np.unravel_index(int(np.nanargmax(P)), P.shape) if nonempty else None)
                    # noise-free lookup on the true cell
                    Pm = np.mean([ctx[(o, fr, urg)]["P"]["deadline_success"] for o in members], axis=0)
                    rec("lookup", argmax_cell(Pm, Fs, ss))
                    # noisy lookup: query with the test object's noisy bins; fall back to the fold's best single fixed
                    nb = noise[(t, fr)][0]; key = ((nb["mass"], nb["friction"], nb["balance"], nb["fragility"]), urg)
                    r = rec("lookup_noisy", tab0.get(key, bsf), {"fallback": key not in tab0, "noisy_bins": nb})
                    sx = []
                    for nz, tb in zip(extra, tabs):
                        nb2 = nz[(t, fr)][0]; k2 = ((nb2["mass"], nb2["friction"], nb2["balance"], nb2["fragility"]), urg)
                        i, j = tb.get(k2, bsf); sx.append((float(P[i, j]), bool(not feas[i, j])))
                    r["extra_success"] = [a for a, _ in sx]; r["extra_infeasible"] = [b for _, b in sx]
                    rec("best_fixed", bsf)
                    # analytic rules (calibrated on the fold's training split, ε-noisy bins, executed)
                    m_, mu_, d_ = props(sweeps[t])
                    for rule in ("B5a", "B5b"):
                        mm = mult[rule]
                        ivr = rule_intervals(rule, est_from(cal, noise[(t, fr)][0]), acc[t], ss, T_nom, T_max[urg], mm["c_min"], mm["c_max"], mm["r_eff"])
                        sel, _ = fs.select_refined(ivr)
                        row = {"fold": fold, "cell": cell, "frag": fr, "urg": urg, "test": t, "method": f"{rule}-c", "truth_nonempty": nonempty,
                               "common": common[(cell, fr, urg)]["common"], "any_feasible": common[(cell, fr, urg)]["any_feasible"],
                               "F": None if sel is None else sel["F"], "s": None if sel is None else sel["s"], "in_refined": in_refined(ref[(t, fr, urg)], sel),
                               "success": None, "infeasible": None}
                        if sel is None:
                            row["success"] = 0.0; row["infeasible"] = True if nonempty else None
                        else:
                            key = f"{sel['F']:.3f}|{sel['s']:.4f}"; row["exec_key"] = key
                            jobs.setdefault(str(sweeps[t]["dir"]), {})[key] = (float(sel["F"]), float(sel["s"]))
                        sel_rows.append(row); rows.append(row)
        print(f"fold {fold}: train {len(train)}, test {len(test)}, multipliers {mult}, best fixed ({Fs[bsf[0]]:.1f}, {ss[bsf[1]]:g})", flush=True)
    # execute the rule selections
    exec_res = execute_selections(jobs, args.workers) if (jobs and not args.no_exec) else {}
    for row in sel_rows:
        if "exec_key" not in row:
            continue
        e = exec_res.get(str(sweeps[row["test"]]["dir"]), {}).get(row["exec_key"])
        if e is None:
            continue
        P_d, P_s, P_dm = e[f"deadline_success|{row['frag']}|{row['urg']}"], e["slip"], e[f"damage|{row['frag']}"]
        row["success"] = P_d; row["infeasible"] = not (P_d >= fs.TAU["c"] and P_s <= fs.TAU["s"] and P_dm <= fs.TAU["d"])
        row["executed"] = {"deadline_success": P_d, "slip": P_s, "damage": P_dm}

    # ---------------- aggregation
    def agg(sub):
        ne = [r for r in sub if r["truth_nonempty"] and r["infeasible"] is not None]
        inf_ne = float(np.mean([r["infeasible"] for r in ne])) if ne else None
        ev = [r for r in sub if r["infeasible"] is not None or not r["truth_nonempty"]]
        inf_all = float(np.mean([bool(r["infeasible"]) if r["infeasible"] is not None else (r["F"] is not None) for r in sub])) if sub else None
        succ = float(np.mean([r["success"] for r in sub if r["success"] is not None])) if sub else None
        succ_ne = float(np.mean([r["success"] for r in sub if r["truth_nonempty"] and r["success"] is not None])) if ne else None
        return {"n": len(sub), "n_nonempty": len(ne), "inf_ne": inf_ne, "inf_all": inf_all, "succ": succ, "succ_ne": succ_ne}
    SPLITS = {"all 48 groups": lambda r: True, "24 with a common feasible cell": lambda r: r["common"], "24 without": lambda r: not r["common"],
              "without, excluding groups with no feasible object": lambda r: (not r["common"]) and r["any_feasible"]}
    lines = [f"# Coarse-information bound v2 — noisy lookup, leave-one-object-out, difficulty split (label version {label_version})\n",
             f"Sweep {args.sweeps}; ε = {NOISE_EPS} bin noise, spec seed {NOISE_SEED} (+{len(extra)} extra seeds for the noisy-lookup spread); rules calibrated per fold on its 36 training objects and executed with R repeats. "
             "Success = P(deadline success) of the chosen cell on the held-out object (0 if nothing chosen); infeasible = chosen cell ∉ 𝓕 of the held-out context (executed 𝓕 test for the rules). Only measured numbers.\n"]
    def table(title, sub_rows, with_spread):
        lines.append(f"\n### {title}\n")
        lines.append("| method | split | n contexts | n truth non-empty | infeasible-selection rate (truth non-empty) | infeasible rate (all; no selection on an empty truth counts 0) | mean success (all) | mean success (truth non-empty) |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for method in METHODS:
            for sname, pred in SPLITS.items():
                if with_spread:
                    per_fold = [agg([r for r in sub_rows if r["method"] == method and r["fold"] == f and pred(r)]) for f in range(4)]
                    def ms(key, nd=3):
                        v = [p[key] for p in per_fold if p[key] is not None]
                        return "" if not v else f"{np.mean(v):.{nd}f} ± {np.std(v):.{nd}f}"
                    a0 = per_fold[0]
                    lines.append(f"| {method} | {sname} | {a0['n']} per fold | {sum(p['n_nonempty'] for p in per_fold)} over folds | {ms('inf_ne')} | {ms('inf_all')} | {ms('succ')} | {ms('succ_ne')} |")
                else:
                    a = agg([r for r in sub_rows if r["method"] == method and pred(r)])
                    lines.append(f"| {method} | {sname} | {a['n']} | {a['n_nonempty']} | {fmt(a['inf_ne'], 3)} | {fmt(a['inf_all'], 3)} | {fmt(a['succ'], 3)} | {fmt(a['succ_ne'], 3)} |")
    lines.append("\n## Item 3 — canonical split (fold 3 = test-combos objects of data/objects_v1.json)\n")
    table("Methods on the 48 held-out contexts, split by difficulty (item 5)", [r for r in rows if r["fold"] == 3], False)
    # noisy lookup spread over extra seeds (canonical fold)
    nz = [r for r in rows if r["fold"] == 3 and r["method"] == "lookup_noisy"]
    if nz and extra:
        for sname, pred in SPLITS.items():
            sub = [r for r in nz if pred(r)]
            if not sub: continue
            succ_seeds = [np.mean([r["extra_success"][k] for r in sub]) for k in range(len(extra))]
            inf_seeds = [np.mean([r["extra_infeasible"][k] for r in sub if r["truth_nonempty"]]) for k in range(len(extra)) if any(r["truth_nonempty"] for r in sub)]
            lines.append(f"- noisy lookup over {len(extra)} extra noise seeds, {sname}: success {np.mean(succ_seeds):.3f} ± {np.std(succ_seeds):.3f}; infeasible (truth non-empty) {np.mean(inf_seeds):.3f} ± {np.std(inf_seeds):.3f}" if inf_seeds else f"- noisy lookup, {sname}: success {np.mean(succ_seeds):.3f} ± {np.std(succ_seeds):.3f}")
        lines.append(f"- noisy-lookup queries that fell back to the best single fixed cell (noisy key unseen in training): {sum(r.get('fallback', False) for r in nz)} of {len(nz)}.")
    lines.append("\n## Item 4 — leave-one-object-out (4 folds; mean ± sd across folds)\n")
    table("All methods, mean ± sd across the 4 folds, split by difficulty (item 5)", rows, True)
    lines.append("\n### Per-fold values (all 48 groups)\n")
    lines.append("| method | fold 0 | fold 1 | fold 2 | fold 3 (canonical) |"); lines.append("|---|---|---|---|---|")
    for method in METHODS:
        cells_ = []
        for f in range(4):
            a = agg([r for r in rows if r["method"] == method and r["fold"] == f])
            cells_.append(f"inf {fmt(a['inf_ne'], 2)} / succ {fmt(a['succ'], 3)}")
        lines.append(f"| {method} | " + " | ".join(cells_) + " |")
    n_common = sum(v["common"] for v in common.values()); n_any = sum(v["any_feasible"] for v in common.values())
    lines.append(f"\nDifficulty split: {n_common} of 48 groups have a grid (F, s) feasible for all four objects; {48 - n_common} do not, of which {48 - n_any} have no feasible object at all (every method scores 0 there).")
    (out / "report.md").write_text("\n".join(lines))
    (out / "coarse_bound2.json").write_text(json.dumps({"rows": rows, "common": {f"{k[0]}|{k[1]}|{k[2]}": v for k, v in common.items()}, "label_version": label_version}, indent=1, default=float))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
