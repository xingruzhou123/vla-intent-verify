"""Gates G7 (independent second labeler) and G8 (independent feasible-set computation) on a finished sweep:
implementation B (envs/feasibility_carry/crosscheck_b.py) is run on every stored branch and probe time series and
compared with the stored implementation-A labels (summary.json / probes.json) and with A's feasible sets, margins and
teacher targets (feasible.context_tables). Thresholds (CLAUDE.md, docs/data_spec.md §2): label agreement ≥ 95 % per
class; feasible sets identical on ≥ 98 % of grid cells; margins differ by ≤ 1 grid step. Disagreements are listed.

Usage: .venv/bin/python -m analysis.crosscheck_g7g8 --sweeps data/sweep_v1 --objects data/objects_v1.json --out results/phase1/crosscheck_v1 --workers 12
"""
import argparse, json, pathlib
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from envs.feasibility_carry import crosscheck_b as Bimp
from envs.feasibility_carry import feasible as fs
from envs.feasibility_carry.env import CarryConfig

FRAGS = ["fragile", "rigid"]; URGS = ["relaxed", "urgent"]
AGREE_MIN, FEAS_MIN = 0.95, 0.98
LABELS = ["slip", "completed", "dropped", "lost_contact", "damage_primary", "damage_shear", "physical_safe", "deadline_success_relaxed", "deadline_success_urgent"]


def worker(d):
    d = pathlib.Path(d)
    S = json.loads((d / "summary.json").read_text()); meta = S["meta"]
    P = json.loads((d / "probes.json").read_text())
    cfg = CarryConfig(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in meta["config"].items()})
    goal = np.array([cfg.goal_xy[0], cfg.goal_xy[1], 0.8 + cfg.obj_half_size[2]])
    frag = {k: float(v) for k, v in meta["label_settings"]["fragility"].items()}
    cmp = {k: [0, 0] for k in LABELS}; dis = []
    rows_b = []
    def one(row, path, kind):
        z = np.load(path)
        lb = Bimp.labels_b(z["rec"].astype(float), list(z["cols"]), goal, cfg.pad_half, cfg.T_carry(row["s"]), meta["T_max"], frag)
        for k in ("slip", "completed", "dropped", "lost_contact"):
            cmp[k][1] += 1; ok = bool(row[k]) == bool(lb[k]); cmp[k][0] += ok
            if not ok: dis.append({"object": d.name, "kind": kind, "F": row["F"], "s": row["s"], "repeat": row["repeat"], "label": k, "A": bool(row[k]), "B": bool(lb[k])})
        for fr in frag:
            for k in ("damage_primary", "damage_shear", "physical_safe", "deadline_success_relaxed", "deadline_success_urgent"):
                cmp[k][1] += 1; ok = bool(row["fragility"][fr][k]) == bool(lb["fragility"][fr][k]); cmp[k][0] += ok
                if not ok: dis.append({"object": d.name, "kind": kind, "F": row["F"], "s": row["s"], "repeat": row["repeat"], "label": f"{k}/{fr}", "A": bool(row["fragility"][fr][k]), "B": bool(lb["fragility"][fr][k])})
        if kind == "grid":
            rows_b.append({"F_idx": row["F_idx"], "s_idx": row["s_idx"], **lb})
    for r in S["branches"]:
        one(r, d / "branches" / f"F{r['F_idx']}_s{r['s_idx']}_r{r['repeat']}.npz", "grid")
    for r in P["probes"]:
        one(r, d / "probes" / f"p{r['probe_idx']}_r{r['repeat']}.npz", "probe")
    # G8: feasible sets, margins, teacher from B's own labels vs A's pipeline on A's labels
    Fs, ss = meta["F"], meta["s"]
    fN = (np.asarray(Fs) - 5.0) / 65.0; sN = np.log(np.asarray(ss) / 0.5) / np.log(6.0)
    step = min(np.diff(fN).min(), np.diff(sN).min())           # one grid step (the smallest spacing, conservative)
    g8 = []
    for fr in frag:
        for urg in URGS:
            A = fs.context_tables(S["branches"], Fs, ss, fr, urg, meta["T_nominal"], meta["T_max"][urg])
            Bc = Bimp.context_b(rows_b, Fs, ss, fr, urg)
            same = A["feasible"] == Bc["feasible"]
            mA, mB = A["margin"], Bc["margin"]
            finite = np.isfinite(mA) & np.isfinite(mB)
            with np.errstate(invalid="ignore"):                 # ±∞ − ±∞ on empty contexts is masked by `finite`
                dm = np.abs(mA - mB)[finite]
            tA, tB = A["teacher"], Bc["teacher"]
            g8.append({"object": d.name, "frag": fr, "urg": urg, "cells": int(same.size), "identical": int(same.sum()),
                       "margin_max_diff": float(dm.max()) if dm.size else 0.0, "margin_cells_over_step": int((dm > step).sum()),
                       "inf_mismatch": int((np.isfinite(mA) != np.isfinite(mB)).sum()),
                       "teacher_same": (tA is None and tB is None) or (tA is not None and tB is not None and tA["F_idx"] == tB["F_idx"] and tA["s_idx"] == tB["s_idx"]),
                       "teacher_A": None if tA is None else (tA["F_idx"], tA["s_idx"]), "teacher_B": None if tB is None else (tB["F_idx"], tB["s_idx"])})
    return {"object": d.name, "cmp": cmp, "disagreements": dis, "g8": g8, "step": float(step)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweeps", required=True); ap.add_argument("--objects", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args(argv)
    out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)
    objs = json.loads(pathlib.Path(args.objects).read_text())["objects"]
    dirs = [str(pathlib.Path(args.sweeps) / o["object_id"]) for o in objs if (pathlib.Path(args.sweeps) / o["object_id"] / "probes.json").exists()]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        res = list(ex.map(worker, dirs))
    tot = {k: [0, 0] for k in LABELS}; dis = []; g8 = []
    for r in res:
        for k in LABELS:
            tot[k][0] += r["cmp"][k][0]; tot[k][1] += r["cmp"][k][1]
        dis += r["disagreements"]; g8 += r["g8"]
    lines = [f"# G7 / G8 cross-check — implementation A vs B on {args.sweeps} ({len(res)} objects)\n",
             "Implementation B: envs/feasibility_carry/crosscheck_b.py (independent code path and numerics; same author). Only measured numbers.\n",
             "## G7 — label agreement (pass ≥ 95 % per class)\n", "| label | compared | agree | agreement | pass |", "|---|---|---|---|---|"]
    g7_pass = True
    for k in LABELS:
        a, n = tot[k]; frac = a / n if n else float("nan"); ok = frac >= AGREE_MIN; g7_pass &= ok
        lines.append(f"| {k} | {n} | {a} | {frac:.4f} | {'✓' if ok else '✗'} |")
    lines.append(f"\n**G7: {'PASS' if g7_pass else 'FAIL'}** — {len(dis)} disagreements in total" + (":" if dis else "."))
    for x in dis[:200]:
        lines.append(f"- {x['object']} {x['kind']} F = {x['F']:.2f}, s = {x['s']:.3f}, repeat {x['repeat']}: {x['label']} A = {x['A']}, B = {x['B']}")
    if len(dis) > 200:
        lines.append(f"- … {len(dis) - 200} more in crosscheck.json")
    cells = sum(g["cells"] for g in g8); ident = sum(g["identical"] for g in g8)
    over = sum(g["margin_cells_over_step"] for g in g8); infm = sum(g["inf_mismatch"] for g in g8)
    maxd = max(g["margin_max_diff"] for g in g8); tsame = sum(g["teacher_same"] for g in g8)
    step = res[0]["step"]
    g8_pass = (ident / cells >= FEAS_MIN) and over == 0 and infm == 0
    lines += ["\n## G8 — feasible sets, margins and teacher targets (pass: identical on ≥ 98 % of cells; margins within 1 grid step)\n",
              f"- grid cells compared: {cells} over {len(g8)} contexts; identical feasibility: {ident} ({ident/cells:.4f}).",
              f"- margins: max |m_A − m_B| = {maxd:.2e} (one grid step = {step:.3f} in normalised units); cells differing by more than one step: {over}; ±∞ mismatches: {infm}.",
              f"- teacher targets identical on {tsame} of {len(g8)} contexts.",
              f"\n**G8: {'PASS' if g8_pass else 'FAIL'}**"]
    bad = [g for g in g8 if g["identical"] != g["cells"] or g["margin_cells_over_step"] or not g["teacher_same"]]
    for g in bad[:100]:
        lines.append(f"- {g['object']} {g['frag']} {g['urg']}: identical {g['identical']}/{g['cells']}, max margin diff {g['margin_max_diff']:.3f}, teacher A {g['teacher_A']} vs B {g['teacher_B']}")
    (out / "report.md").write_text("\n".join(lines))
    (out / "crosscheck.json").write_text(json.dumps({"G7": {"pass": g7_pass, "totals": tot, "disagreements": dis}, "G8": {"pass": g8_pass, "cells": cells, "identical": ident, "margin_max_diff": maxd, "step": step, "contexts": g8}}, indent=1, default=float))
    print("\n".join(lines[:60]))


if __name__ == "__main__":
    main()
