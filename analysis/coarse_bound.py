"""Coarse-information upper bound (owner request, 2026-11-02; no new simulation). For every coarse context cell
(mass bin × friction bin × balance bin × fragility bin) × urgency level, the single grid (F, s) that maximises the mean
deadline-constrained safe success over the cell's training objects (sweep outcomes) is chosen and evaluated on the
cell's held-out test object: success = its empirical P(deadline success) at that cell; infeasible = the cell is outside the
test object's feasible set 𝓕 (plan §3.3). Also reports, per cell, whether any single grid (F, s) is feasible for all
objects of the cell (training and test). Compared with B5a-c / B5b-c (executed selections from the full-sweep
analysis summary) and with the per-object oracle (best cell per test context).

Usage: .venv/bin/python -m analysis.coarse_bound --sweeps data/sweep_v1 --objects data/objects_v1.json --full results/phase1/full_v2/summary.json --out results/phase1/coarse_bound_v1
"""
import argparse, json, pathlib
import numpy as np
from envs.feasibility_carry import feasible as fs
from envs.feasibility_carry.objects import CELLS
from analysis.phase1_demo import load

FRAGS = ["fragile", "rigid"]; URGS = ["relaxed", "urgent"]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweeps", required=True); ap.add_argument("--objects", required=True); ap.add_argument("--full", required=True); ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)
    objs = {o["object_id"]: o for o in json.loads(pathlib.Path(args.objects).read_text())["objects"]}
    full = json.loads(pathlib.Path(args.full).read_text())
    sweeps = {oid: load(pathlib.Path(args.sweeps) / oid) for oid in objs}
    sw0 = next(iter(sweeps.values())); Fs, ss = sw0["meta"]["F"], sw0["meta"]["s"]; T_nom, T_max = sw0["meta"]["T_nominal"], sw0["meta"]["T_max"]
    sN = fs.s_norm(ss); s0 = fs.s_norm(1.0)
    ctx = {(oid, fr, urg): fs.context_tables(sw["rows"], Fs, ss, fr, urg, T_nom, T_max[urg]) for oid, sw in sweeps.items() for fr in FRAGS for urg in URGS}
    lines = ["# Coarse-information upper bound (no new simulation)\n",
             "Per coarse cell (mass × friction × balance × fragility) × urgency: the single grid (F, s) maximising mean P(deadline success) over the 3 training objects, evaluated on the held-out test object. Ties broken toward the nominal s, then toward the middle of the F grid. Only measured numbers.\n",
             "| cell | fragility | urgency | chosen (F, s) | train mean success | test object | test success | test cell feasible (𝓕) | test oracle (best cell) | any (F, s) feasible for all 4 objects | cells feasible for all 4 |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    rows = []
    for (mb, fb, bb) in CELLS:
        cell_objs = [oid for oid, o in objs.items() if o["bins"] == {"mass": mb, "friction": fb, "balance": bb}]
        train = [o for o in cell_objs if objs[o]["split"] == "train"]; test = [o for o in cell_objs if objs[o]["split"] != "train"]
        for fr in FRAGS:
            for urg in URGS:
                Pm = np.mean([ctx[(o, fr, urg)]["P"]["deadline_success"] for o in train], axis=0)
                best, arg = None, None
                for i in range(len(Fs)):
                    for j in range(len(ss)):
                        key = (round(float(Pm[i, j]), 9), -abs(sN[j] - s0), -abs(i - (len(Fs) - 1) / 2))
                        if best is None or key > best:
                            best, arg = key, (i, j)
                i, j = arg
                all_feas = np.logical_and.reduce([ctx[(o, fr, urg)]["feasible"] for o in cell_objs])
                for t in test:
                    c = ctx[(t, fr, urg)]
                    rec = {"cell": f"{mb}_{fb}_{bb}", "frag": fr, "urg": urg, "F": Fs[i], "s": ss[j], "train_success": float(Pm[i, j]), "test": t,
                           "test_success": float(c["P"]["deadline_success"][i, j]), "test_feasible": bool(c["feasible"][i, j]),
                           "test_truth_nonempty": bool(c["feasible"].any()), "test_oracle": float(np.nanmax(c["P"]["deadline_success"])),
                           "any_all4": bool(all_feas.any()), "n_all4": int(all_feas.sum())}
                    rows.append(rec)
                    lines.append(f"| {rec['cell']} | {fr} | {urg} | ({Fs[i]:.1f}, {ss[j]:g}) | {Pm[i, j]:.2f} | {t} | {rec['test_success']:.2f} | {'✓' if rec['test_feasible'] else '✗'} | {rec['test_oracle']:.2f} | {'✓' if rec['any_all4'] else '✗'} | {rec['n_all4']} |")
    ne = [r for r in rows if r["test_truth_nonempty"]]
    inf_rate = float(np.mean([not r["test_feasible"] for r in ne])); inf_all = float(np.mean([not r["test_feasible"] for r in rows]))
    succ = float(np.mean([r["test_success"] for r in rows])); succ_ne = float(np.mean([r["test_success"] for r in ne]))
    oracle = float(np.mean([r["test_oracle"] for r in rows])); oracle_ne = float(np.mean([r["test_oracle"] for r in ne]))
    p4 = full["checks"]["P4"]["table"]; bsf = full["best_single_fixed"]
    lines += ["\n## Summary on the 48 held-out contexts (12 test objects × 2 fragility × 2 urgency)\n",
              "| method | information | infeasible-selection rate (truth non-empty) | infeasible rate (all) | mean success (all) | mean success (truth non-empty) |", "|---|---|---|---|---|---|",
              f"| per-object oracle (best cell per test context) | true outcomes | 0 | — | {oracle:.3f} | {oracle_ne:.3f} |",
              f"| coarse-cell best single (F, s) | noise-free coarse bins + training outcomes | {inf_rate:.3f} ({sum(not r['test_feasible'] for r in ne)}/{len(ne)}) | {inf_all:.3f} ({sum(not r['test_feasible'] for r in rows)}/{len(rows)}) | {succ:.3f} | {succ_ne:.3f} |",
              f"| B5a-c (executed) | ε-noisy coarse bins, training-split calibration | {p4['B5a|coarse-cal']['exec_infeasible_rate_nonempty']:.3f} | {p4['B5a|coarse-cal']['exec_infeasible_rate_all']:.3f} | | |",
              f"| B5b-c (executed) | ε-noisy coarse bins, training-split calibration | {p4['B5b|coarse-cal']['exec_infeasible_rate_nonempty']:.3f} | {p4['B5b|coarse-cal']['exec_infeasible_rate_all']:.3f} | | |",
              f"| best single fixed over all training contexts ({bsf['train']['F']:.1f} N, s = {bsf['train']['s']:g}) | none | | | {bsf['test_success']:.3f} | |",
              "\nSuccess for the analytic rules is not tabulated here: their selections are continuous (F, s) values executed with 5 repeats and the infeasible-selection rate is the comparable number (full-sweep report, P4). "
              "The coarse-cell bound uses noise-free bins (the best case for coarse information); the rules see ε = 0.1 noisy bins.\n",
              "## Per physical cell: does one (F, s) fit all four objects?\n", "| cell | fragile/relaxed | fragile/urgent | rigid/relaxed | rigid/urgent |", "|---|---|---|---|---|"]
    for (mb, fb, bb) in CELLS:
        cell = f"{mb}_{fb}_{bb}"
        vals = []
        for fr in FRAGS:
            for urg in URGS:
                r = next(x for x in rows if x["cell"] == cell and x["frag"] == fr and x["urg"] == urg)
                vals.append(f"{'✓' if r['any_all4'] else '✗'} ({r['n_all4']})")
        lines.append(f"| {cell} | " + " | ".join(vals) + " |")
    n_any = sum(r["any_all4"] for r in rows); n_ne = sum(1 for r in rows if any(ctx[(o, r['frag'], r['urg'])]["feasible"].any() for o in objs if objs[o]["bins"] == {"mass": r["cell"].split("_")[0], "friction": r["cell"].split("_")[1], "balance": r["cell"].split("_")[2]}))
    lines.append(f"\n{n_any} of {len(rows)} (cell, fragility, urgency) groups have at least one grid (F, s) feasible for all four objects; {n_ne} groups have at least one object with a non-empty feasible set.")
    (out / "report.md").write_text("\n".join(lines))
    (out / "coarse_bound.json").write_text(json.dumps({"rows": rows, "summary": {"infeasible_rate_nonempty": inf_rate, "infeasible_rate_all": inf_all, "success": succ, "success_nonempty": succ_ne, "oracle": oracle, "oracle_nonempty": oracle_ne, "n_any_all4": n_any}}, indent=1))
    print("\n".join(lines[-25:]))


if __name__ == "__main__":
    main()
