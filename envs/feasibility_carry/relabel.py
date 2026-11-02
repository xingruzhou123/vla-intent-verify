"""Recompute every branch and probe label of a finished sweep from the stored time series (no simulation), with the
current labeler (labels.py, LABEL_VERSION). Used for the label-definition fix of 2026-11-02 (docs/data_spec.md §1.2).

The previous summary.json / probes.json are kept as summary.labels_v<old>.json / probes.labels_v<old>.json. Grasp
statistics (ss_*), wall time, perturbations and indices are copied from the old rows; everything produced by
labels.all_labels is replaced. Bisection brackets in probes.json are the probe locations actually simulated and are kept
as recorded. A stale p4_exec.json (rates under the old labels) is renamed so that it is re-executed.

Usage: .venv/bin/python -m envs.feasibility_carry.relabel data/sweep_v1 --workers 12
"""
import argparse, json, pathlib
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from .env import CarryConfig
from . import labels

KEEP = ("F", "s", "F_idx", "s_idx", "repeat", "pert", "wall_s", "sim_s", "probe_idx", "purpose")
CELL_LABELS = ("slip", "completed", "lost_contact", "dropped")
CELL_FRAG_LABELS = ("damage_primary", "damage_shear", "physical_safe", "deadline_success_relaxed", "deadline_success_urgent")


def cell_table(rows, nF, nS):
    """Per grid cell: the empirical rate of every label over the repeats, stored alongside the binary branch labels, and a
    `stochastic` flag = some label's repeats disagree (owner rule of 2026-11-02: after label v3 any remaining disagreement
    is physical stochasticity, not a label error)."""
    cells = []
    for i in range(nF):
        for j in range(nS):
            cell = [r for r in rows if r["F_idx"] == i and r["s_idx"] == j]
            if not cell:
                continue
            rates = {k: float(np.mean([r[k] for r in cell])) for k in CELL_LABELS}
            for frag in cell[0]["fragility"]:
                for k in CELL_FRAG_LABELS:
                    rates[f"{k}|{frag}"] = float(np.mean([r["fragility"][frag][k] for r in cell]))
            disagreeing = sorted(k for k, v in rates.items() if 0.0 < v < 1.0)
            cells.append({"F_idx": i, "s_idx": j, "n": len(cell), "rates": rates, "stochastic": bool(disagreeing), "disagreeing_labels": disagreeing})
    return cells


def relabel_dir(d):
    d = pathlib.Path(d)
    S = json.loads((d / "summary.json").read_text()); meta = S["meta"]
    old_v = meta.get("label_version", 1)
    if old_v == labels.LABEL_VERSION:
        if "cells" not in S:
            S["cells"] = cell_table(S["branches"], len(meta["F"]), len(meta["s"]))
            (d / "summary.json").write_text(json.dumps(S, indent=1, default=float))
            return str(d), "already at version %d; cell table added" % old_v, 0
        return str(d), "already at version %d" % old_v, 0
    cfg = CarryConfig(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in meta["config"].items()})
    table_offset = np.array((0.0, 0.0, 0.8))                  # env.FeasibilityCarry.table_offset (fixed)
    goal_pos = np.array([table_offset[0] + cfg.goal_xy[0], table_offset[1] + cfg.goal_xy[1], table_offset[2] + cfg.obj_half_size[2]])
    frag = {k: float(v) for k, v in meta["label_settings"]["fragility"].items()}
    n = 0

    def redo(row, path):
        nonlocal n
        z = np.load(path)
        lab = labels.all_labels(z["rec"].astype(float), list(z["cols"]), goal_pos, cfg.pad_half, cfg.T_carry(row["s"]), meta["T_max"], fragility=frag)
        new = {k: row[k] for k in KEEP if k in row}
        new.update({k: v for k, v in row.items() if k.startswith("ss_")})
        new.update(lab); n += 1
        return new

    rows = [redo(r, d / "branches" / f"F{r['F_idx']}_s{r['s_idx']}_r{r['repeat']}.npz") for r in S["branches"]]
    (d / f"summary.labels_v{old_v}.json").write_text(json.dumps(S, indent=1, default=float))
    meta["label_version"] = labels.LABEL_VERSION
    meta["label_settings"]["label_version"] = labels.LABEL_VERSION
    (d / "summary.json").write_text(json.dumps({"meta": meta, "branches": rows, "cells": cell_table(rows, len(meta["F"]), len(meta["s"]))}, indent=1, default=float))
    pp = d / "probes.json"
    if pp.exists():
        P = json.loads(pp.read_text())
        prow = [redo(r, d / "probes" / f"p{r['probe_idx']}_r{r['repeat']}.npz") for r in P["probes"]]
        (d / f"probes.labels_v{old_v}.json").write_text(json.dumps(P, indent=1, default=float))
        pp.write_text(json.dumps({"probes": prow, "brackets": P["brackets"], "label_version": labels.LABEL_VERSION}, indent=1, default=float))
    pe = d / "p4_exec.json"
    if pe.exists():
        pe.rename(d / f"p4_exec.labels_v{old_v}.json")
    return str(d), f"relabelled v{old_v} -> v{labels.LABEL_VERSION}", n


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("root"); ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args(argv)
    dirs = sorted(p for p in pathlib.Path(args.root).iterdir() if (p / "summary.json").exists())
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for d, msg, n in ex.map(relabel_dir, dirs):
            print(f"{pathlib.Path(d).name}: {msg} ({n} branches)", flush=True)


if __name__ == "__main__":
    main()
