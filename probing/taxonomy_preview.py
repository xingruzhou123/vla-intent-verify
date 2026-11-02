"""Preview of the four-way context taxonomy (study plan §3) from existing CFD data only — no simulation.

Idealised probe model (to be replaced by the Stage-1 probe repeat tables):
  trial lift at F_p holds  ⇔  F50_slip(s = 3.0) < F_p   (static slip boundary: 50 % crossing of the CFD logistic fit over
  grid + bisection probes at the slowest s level; deterministic, no state noise)
  light squeeze: carries no information (fragility does not enter the simulation; design.md §1.2)
Risk standard: worst-case. An action (F, s) is admissible under Θ iff it lies in the CFD feasible set 𝓕(θ) of every θ ∈ Θ;
a lift at F_p is admissible iff F_p + 0.5 N ≤ f_tol(θ) for every θ ∈ Θ (0.5 N = sustained-peak overshoot margin).
Θ0 options (design.md §7): A population (48 objects × 2 fragility classes); B ε-neighbourhood of the coarse text
(own bin cell ∪ single-attribute neighbour cells, fragility bin included); C own bin cell, fragility as stated;
D population of objects with the fragility class as stated (mass, friction, balance unknown, fragility trusted);
E ε-neighbourhood over mass / friction / balance with the fragility class as stated.
Usage: nice -n 10 .venv/bin/python -m probing.taxonomy_preview --sweeps data/sweep_v1 --objects data/objects_v1.json --out probing/results/smoke/taxonomy_preview.json
"""
import argparse, json, pathlib, itertools
import numpy as np
from envs.feasibility_carry import feasible as fs

FRAGS, URGS = ("fragile", "rigid"), ("relaxed", "urgent")
MASS_ORDER = ["light", "medium", "heavy"]
LIFT_LEVELS = [7.29, 10.628, 15.494, 22.589, 32.933]      # CFD grid levels F_idx 1–5
MARGIN = 0.5


def load(sweeps, objs):
    feas, f50 = {}, {}
    for o in objs:
        d = pathlib.Path(sweeps) / o["object_id"]
        S = json.loads((d / "summary.json").read_text()); meta, rows = S["meta"], S["branches"]
        probes = json.loads((d / "probes.json").read_text())["probes"] if (d / "probes.json").exists() else []
        Fs, ss = meta["F"], meta["s"]
        j = len(ss) - 1
        at = [r for r in rows if r["s_idx"] == j] + [p for p in probes if p.get("s_idx") == j]
        fit = fs.logistic_boundary([r["F"] for r in at], [r["slip"] for r in at], [r["repeat"] for r in at], increasing=False, n_boot=0)
        f50[o["object_id"]] = float("inf") if fit is None and all(r["slip"] for r in at) else (0.0 if fit is None else fit["F50"])
        for fr in FRAGS:
            for urg in URGS:
                feas[(o["object_id"], fr, urg)] = fs.context_tables(rows, Fs, ss, fr, urg, meta["T_nominal"], meta["T_max"][urg])["feasible"]
    return feas, f50


def neighbours(bins):
    """Own bin cell and every single-attribute neighbour (the ε-noise model of plan §3.6)."""
    out = [dict(bins)]
    k = MASS_ORDER.index(bins["mass"])
    for kk in (k - 1, k + 1):
        if 0 <= kk < 3:
            b = dict(bins); b["mass"] = MASS_ORDER[kk]; out.append(b)
    for attr, vals in (("friction", ("slippery", "normal")), ("balance", ("balanced", "off-centre")), ("fragility", FRAGS)):
        b = dict(bins); b[attr] = [v for v in vals if v != bins[attr]][0]; out.append(b)
    return out


def theta0(option, obj, frag, objs):
    own = {**obj["bins"], "fragility": frag}
    if option == "A":
        return [(o["object_id"], f) for o in objs for f in FRAGS]
    if option == "D":
        return [(o["object_id"], frag) for o in objs]
    cells = neighbours(own) if option in ("B", "E") else [own]
    if option == "E":
        cells = [c for c in cells if c["fragility"] == frag]
    keyset = {(c["mass"], c["friction"], c["balance"], c["fragility"]) for c in cells}
    return [(o["object_id"], f) for o in objs for f in FRAGS if (o["bins"]["mass"], o["bins"]["friction"], o["bins"]["balance"], f) in keyset]


def classify(ctx, option, K, feas, f50, objs, ftol):
    oid, frag, urg = ctx
    F_true = feas[ctx]
    if not F_true.any():
        return "physically infeasible"
    Th0 = theta0(option, next(o for o in objs if o["object_id"] == oid), frag, objs)
    common = lambda Th: np.logical_and.reduce([feas[(t, f, urg)] for t, f in Th]).any()
    if common(Th0):
        return "no probe needed"
    admissible = lambda Th: [Fp for Fp in LIFT_LEVELS if all(Fp + MARGIN <= ftol[t][f] for t, f in Th)]
    outcome = lambda t, Fp: f50[t] < Fp
    update = lambda Th, Fp, o: [(t, f) for t, f in Th if outcome(t, Fp) == o]

    def resolvable(Th, depth):
        if common(Th):
            return True
        if depth == 0:
            return False
        for Fp in admissible(Th):
            o = outcome(oid, Fp)
            Th1 = update(Th, Fp, o)
            if len(Th1) < len(Th) and resolvable(Th1, depth - 1):
                return True
        return False
    return "resolvable by safe probing" if resolvable(Th0, K) else "unresolvable under stated limits"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweeps", default="data/sweep_v1"); ap.add_argument("--objects", default="data/objects_v1.json")
    ap.add_argument("--out", default="probing/results/smoke/taxonomy_preview.json")
    args = ap.parse_args(argv)
    objs = json.loads(pathlib.Path(args.objects).read_text())["objects"]
    ftol = {o["object_id"]: o["f_tol"] for o in objs}
    feas, f50 = load(args.sweeps, objs)
    ctxs = [(o["object_id"], fr, urg) for o in objs for fr in FRAGS for urg in URGS]
    cats = ["no probe needed", "resolvable by safe probing", "unresolvable under stated limits", "physically infeasible"]
    res, lines = {}, []
    lines.append("| Θ0 option | K | " + " | ".join(cats) + " | fragile: " + " / ".join(c.split()[0] for c in cats) + " | rigid: same |")
    lines.append("|---|---|" + "---|" * len(cats) + "---|---|")
    for option in ("A", "B", "C", "D", "E"):
        for K in (1, 2):
            cl = {c: classify(c, option, K, feas, f50, objs, ftol) for c in ctxs}
            res[f"{option}|K{K}"] = {"|".join(c): v for c, v in cl.items()}
            n = {k: sum(v == k for v in cl.values()) for k in cats}
            nf = {fr: [sum(1 for c, v in cl.items() if c[1] == fr and v == k) for k in cats] for fr in FRAGS}
            lines.append(f"| {option} | {K} | " + " | ".join(str(n[k]) for k in cats) + " | " + " / ".join(map(str, nf["fragile"])) + " | " + " / ".join(map(str, nf["rigid"])) + " |")
    txt = "\n".join(lines)
    print(txt)
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.out).write_text(json.dumps({"table_md": txt, "lift_levels": LIFT_LEVELS, "margin_N": MARGIN, "f50_static": f50,
                                                  "assignments": res}, indent=1, default=float))


if __name__ == "__main__":
    main()
