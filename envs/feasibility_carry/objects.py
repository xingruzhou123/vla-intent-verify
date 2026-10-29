"""Sampled physical objects for the full sweep (plan §3.6, §6; owner requirement of 2026-10-29).

The coarse text the policy sees names a bin; the true value is sampled continuously inside the bin's band, so the
coarse setting never equals the oracle. Bands are separated by gaps so that a bin is unambiguous and the ε-noise
replacement by a neighbouring bin is a real information loss.

Usage:
  .venv/bin/python -m envs.feasibility_carry.objects --out data/objects_v1.json --per_cell 4 --seed 0
  .venv/bin/python -m envs.feasibility_carry.run_objects data/objects_v1.json --out_root data/sweep_v1 --parallel 12
"""
import argparse, json, pathlib, itertools
import numpy as np

BANDS = {
    "mass":      {"light": (0.35, 0.65), "medium": (0.80, 1.20), "heavy": (1.30, 1.70)},        # kg (centres 0.5 / 1.0 / 1.5)
    "friction":  {"slippery": (0.15, 0.25), "normal": (0.35, 0.45)},                            # μ (centres 0.2 / 0.4)
    "fragility": {"fragile": (16.0, 24.0), "rigid": (40.0, 60.0)},                              # f_tol, N (centres 20 / 50)
    "balance":   {"balanced": (0.0, 0.0), "off-centre": (0.20, 0.30)},                           # CoM offset, fraction of w (centre 0.25)
}
CENTRES = {attr: {k: 0.5 * (lo + hi) for k, (lo, hi) in bands.items()} for attr, bands in BANDS.items()}
CELLS = list(itertools.product(BANDS["mass"], BANDS["friction"], BANDS["balance"]))   # 12 physical bin cells


def sample_objects(per_cell=4, seed=0, test_per_cell=1):
    """per_cell objects in each of the 12 physical bin cells; the last `test_per_cell` of each cell go to the
    test-combos split (continuous values never seen in training). Fragility is sampled per object for both classes
    (label parameters: both are applied to every branch)."""
    rng = np.random.default_rng(seed)
    objs = []
    for (mb, fb, bb) in CELLS:
        for k in range(per_cell):
            u = lambda attr, key: float(rng.uniform(*BANDS[attr][key]))
            objs.append({
                "object_id": f"{mb}_{fb}_{bb}_{k}",
                "bins": {"mass": mb, "friction": fb, "balance": bb},
                "mass": round(u("mass", mb), 4), "mu": round(u("friction", fb), 4),
                "com_frac": round(u("balance", bb), 4),
                "f_tol": {"fragile": round(u("fragility", "fragile"), 2), "rigid": round(u("fragility", "rigid"), 2)},
                "split": "test-combos" if k >= per_cell - test_per_cell else "train",
            })
    return objs


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--per_cell", type=int, default=4)
    ap.add_argument("--test_per_cell", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    objs = sample_objects(args.per_cell, args.seed, args.test_per_cell)
    out = pathlib.Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"bands": BANDS, "centres": CENTRES, "seed": args.seed, "per_cell": args.per_cell,
                               "test_per_cell": args.test_per_cell, "objects": objs}, indent=1))
    print(f"wrote {len(objs)} objects to {out} ({sum(o['split']=='train' for o in objs)} train, "
          f"{sum(o['split']!='train' for o in objs)} test-combos)")


if __name__ == "__main__":
    main()
