"""Boundary refinement by bisection (plan.md §6, "Boundary refinement"). Probes are analysis data, never training data
(docs/data_spec.md §2a): they are written under <sweep>/probes/ and listed in <sweep>/probes.json.

For every s level of a finished grid sweep:
- slip boundary in F: bracket = (highest grid F that slips, lowest grid F that does not), bisected with R paired repeats
  per probe until the bracket is ≤ `bracket_N` or `max_probes` have been spent;
- primary-damage boundary in F, per fragility level, likewise (bracket = (highest grid F without damage, lowest with));
- P3 recovery threshold s_rec, per (fragility, urgency): if no grid F is feasible at the nominal s but some slower,
  deadline-feasible level is, bisect in log s between the nominal level and the fastest feasible slower level at that
  level's margin-maximising F.

Boundary estimates (50 % crossing of a logistic fit over grid + probe outcomes, bootstrap CI over repeats) are computed
in analysis (feasible.py); this module only runs the simulations.

Usage: MUJOCO_GL=egl .venv/bin/python -m envs.feasibility_carry.refine results/phase1/demo_a
"""
import argparse, json, pathlib, time
import numpy as np
from .env import FeasibilityCarry, CarryConfig
from .sweep import run_branch
from . import feasible as fs


def load_states(out, R):
    states = []
    for r in range(R):
        z = np.load(out / f"branch_state_r{r}.npz", allow_pickle=True)
        states.append({k: z[k] if k == "state" else z[k].item() for k in z.files})
    return states


class Prober:
    def __init__(self, env, out, states, T_max, rows):
        self.env, self.out, self.states, self.T_max, self.rows = env, out, states, T_max, rows
        (out / "probes").mkdir(exist_ok=True)
        self.k = len(rows)

    def probe(self, F, s, s_idx, purpose):
        infos = []
        for r, saved in enumerate(self.states):
            info, arr = run_branch(self.env, saved, F, s, self.T_max)
            info.update({"repeat": r, "s_idx": s_idx, "probe_idx": self.k, "purpose": purpose, "F_idx": None})
            np.savez_compressed(self.out / "probes" / f"p{self.k}_r{r}.npz", rec=arr.astype(np.float32),
                                cols=np.array(self.env.SAMPLE_COLS))
            self.rows.append(info); infos.append(info)
        self.k += 1
        return infos


def bisect_F(prober, s, s_idx, lo, hi, bad, purpose, tau, max_probes, bracket_N, log):
    """Bisect between lo (bad outcome) and hi (good) in F; `bad(info)` → outcome indicator; good = rate ≤ tau."""
    n = 0
    while hi - lo > bracket_N and n < max_probes:
        mid = 0.5 * (lo + hi)
        infos = prober.probe(mid, s, s_idx, purpose)
        rate = float(np.mean([bad(i) for i in infos]))
        if rate <= tau:
            hi = mid
        else:
            lo = mid
        n += 1
        log(f"    {purpose} s={s:.3f} probe F={mid:5.2f}: rate={rate:.2f} -> bracket [{lo:.2f}, {hi:.2f}]")
    return lo, hi, n


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep")
    ap.add_argument("--max_probes", type=int, default=4)
    ap.add_argument("--bracket_N", type=float, default=2.0)
    ap.add_argument("--tau_s", type=float, default=fs.TAU["s"])
    ap.add_argument("--tau_d", type=float, default=fs.TAU["d"])
    ap.add_argument("--no_srec", action="store_true")
    args = ap.parse_args(argv)
    out = pathlib.Path(args.sweep)
    S = json.loads((out / "summary.json").read_text())
    meta, rows = S["meta"], S["branches"]
    Fs, ss, R = meta["F"], meta["s"], meta["repeats"]
    cfg = CarryConfig(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in meta["config"].items()})
    env = FeasibilityCarry(cfg, seed=meta["seed"])
    env.reset()                                   # builds the model; states are restored into it
    states = load_states(out, R)
    probes_path = out / "probes.json"
    prows = json.loads(probes_path.read_text())["probes"] if probes_path.exists() else []
    prober = Prober(env, out, states, meta["T_max"], prows)
    logf = open(out / "refine_log.txt", "a")
    def log(msg):
        print(msg, flush=True); logf.write(msg + "\n"); logf.flush()
    log(f"[refine] {out}  grid {len(Fs)}x{len(ss)}x{R}; existing probes: {len(prows)}")
    brackets = {"slip": {}, "damage": {}, "s_rec": {}}
    t0 = time.perf_counter()
    for j, s in enumerate(ss):
        grid = [r for r in rows if r["s_idx"] == j]
        # ---- slip boundary
        P_slip = [np.mean([r["slip"] for r in grid if r["F_idx"] == i]) for i in range(len(Fs))]
        ok = [i for i in range(len(Fs)) if P_slip[i] <= args.tau_s]
        if not ok:
            brackets["slip"][j] = {"status": "above_grid"}
        elif ok[0] == 0:
            brackets["slip"][j] = {"status": "below_grid"}
        else:
            i = ok[0]
            lo, hi, n = bisect_F(prober, s, j, Fs[i - 1], Fs[i], lambda info: info["slip"], "slip", args.tau_s,
                                 args.max_probes, args.bracket_N, log)
            brackets["slip"][j] = {"status": "bracketed", "lo": lo, "hi": hi, "n_probes": n, "grid_lo": Fs[i - 1], "grid_hi": Fs[i]}
        # ---- primary damage boundary per fragility
        for frag in meta["label_settings"]["fragility"]:
            P_d = [np.mean([r["fragility"][frag]["damage_primary"] for r in grid if r["F_idx"] == i]) for i in range(len(Fs))]
            ok = [i for i in range(len(Fs)) if P_d[i] <= args.tau_d]
            key = f"{frag}|{j}"
            if not ok:
                brackets["damage"][key] = {"status": "below_grid"}      # every grid F damages
            elif ok[-1] == len(Fs) - 1:
                brackets["damage"][key] = {"status": "above_grid"}      # no grid F damages
            else:
                i = ok[-1]
                # bisect with roles swapped: lo = safe, hi = damaging
                lo, hi = Fs[i], Fs[i + 1]; n = 0
                while hi - lo > args.bracket_N and n < args.max_probes:
                    mid = 0.5 * (lo + hi)
                    infos = prober.probe(mid, s, j, f"damage_{frag}")
                    rate = float(np.mean([x["fragility"][frag]["damage_primary"] for x in infos]))
                    if rate <= args.tau_d:
                        lo = mid
                    else:
                        hi = mid
                    n += 1
                    log(f"    damage[{frag}] s={s:.3f} probe F={mid:5.2f}: rate={rate:.2f} -> bracket [{lo:.2f}, {hi:.2f}]")
                brackets["damage"][key] = {"status": "bracketed", "lo": lo, "hi": hi, "n_probes": n, "grid_lo": Fs[i], "grid_hi": Fs[i + 1]}
        (out / "probes.json").write_text(json.dumps({"probes": prows, "brackets": brackets}, indent=1, default=float))
    # ---- P3 recovery threshold s_rec
    if not args.no_srec:
        j_nom = int(np.argmin(np.abs(np.log(np.array(ss)))))
        for frag in meta["label_settings"]["fragility"]:
            for urg, T_max in meta["T_max"].items():
                ctx = fs.context_tables(rows, Fs, ss, frag, urg, meta["T_nominal"], T_max)
                feas = ctx["feasible"]                           # [F_idx, s_idx]
                if feas[:, j_nom].any():
                    brackets["s_rec"][f"{frag}|{urg}"] = {"status": "feasible_at_nominal"}
                    continue
                slower = [jj for jj in range(j_nom + 1, len(ss)) if ctx["deadline_ok"][jj] and feas[:, jj].any()]
                if not slower:
                    brackets["s_rec"][f"{frag}|{urg}"] = {"status": "no_recovery_on_grid"}
                    continue
                jr = slower[0]
                F_best = fs.best_F_at_s(ctx, jr)
                lo, hi = ss[j_nom], ss[jr]; n = 0
                while np.log(hi / lo) > np.log(ss[1] / ss[0]) / 4 and n < args.max_probes:
                    mid = float(np.sqrt(lo * hi))
                    infos = prober.probe(F_best, mid, None, f"s_rec_{frag}_{urg}")
                    ok = all(x["fragility"][frag][f"deadline_success_{urg}"] for x in infos)
                    if ok:
                        hi = mid
                    else:
                        lo = mid
                    n += 1
                    log(f"    s_rec[{frag},{urg}] F={F_best:.2f} probe s={mid:.3f}: all-ok={ok} -> bracket [{lo:.3f}, {hi:.3f}]")
                brackets["s_rec"][f"{frag}|{urg}"] = {"status": "bracketed", "lo": lo, "hi": hi, "F": F_best, "n_probes": n,
                                                       "s_nominal": ss[j_nom], "s_grid_recovery": ss[jr]}
    (out / "probes.json").write_text(json.dumps({"probes": prows, "brackets": brackets}, indent=1, default=float))
    log(f"[refine] done: {len(prows)} probe branches, {time.perf_counter()-t0:.0f} s")


if __name__ == "__main__":
    main()
