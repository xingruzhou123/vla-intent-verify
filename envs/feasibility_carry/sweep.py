"""Same-state F × s branching sweep. One interaction group per (object combo, repeat).

Usage (pilot):
  MUJOCO_GL=egl .venv/bin/python -m envs.feasibility_carry.sweep --out results/phase1/pilot \
      --F 8 15 30 60 --s 0.5 1.0 2.0 --repeats 2
"""
import argparse, json, time, pathlib, sys
import numpy as np
from .env import FeasibilityCarry, CarryConfig
from . import labels


def run_branch(env, saved, F, s, tol, rec_out):
    """Restore the branch state, grasp with F, carry with s, release, label. Returns (info, rec array)."""
    env.restore_state(saved)
    rec = []
    t0 = time.perf_counter()
    gs = env.grasp(F, rec)
    i_carry = len(rec)
    ref_rel = np.array(gs["ss_rel"])
    pos, mat = env.carry(s, rec)
    i_carry_end = len(rec)
    env.release(pos, mat, rec)
    wall = time.perf_counter() - t0
    arr = np.array(rec)
    labels.COLS = env.SAMPLE_COLS
    dmg = labels.damage(arr, tol["f_tol"], tol["D_max"])
    sl = labels.slip(arr, i_carry, i_carry_end, ref_rel, tol["delta_slip"])
    cp = labels.complete(arr, env.goal_pos(), dmg["damage_primary"])
    info = {"F": F, "s": s, "wall_s": wall, "sim_s": float(arr[-1, 0] - arr[0, 0]),
            "i_carry": i_carry, "i_carry_end": i_carry_end, **gs, **dmg, **sl, **cp}
    return info, arr


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--F", type=float, nargs="+", required=True)
    ap.add_argument("--s", type=float, nargs="+", required=True)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--mass", type=float, default=0.3)
    ap.add_argument("--mu", type=float, default=0.4)
    ap.add_argument("--f_tol", type=float, default=40.0, help="N, provisional damage tolerance")
    ap.add_argument("--D_max", type=float, default=5.0, help="N·s, provisional excess-impulse limit")
    ap.add_argument("--delta_slip", type=float, default=0.01, help="m, provisional slip threshold")
    ap.add_argument("--pose_sd", type=float, default=0.002, help="m, per-repeat object xy jitter")
    ap.add_argument("--yaw_sd", type=float, default=np.deg2rad(2.0))
    ap.add_argument("--mu_sd", type=float, default=0.03, help="relative friction noise per repeat")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    out = pathlib.Path(args.out); (out / "branches").mkdir(parents=True, exist_ok=True)
    cfg = CarryConfig(obj_mass=args.mass, obj_mu=args.mu)
    env = FeasibilityCarry(cfg, seed=args.seed)
    rng = np.random.default_rng(args.seed)
    tol = {"f_tol": args.f_tol, "D_max": args.D_max, "delta_slip": args.delta_slip}
    rows = []
    for r in range(args.repeats):
        pert = {"dxy": rng.normal(0, args.pose_sd, 2).tolist(), "dyaw": float(rng.normal(0, args.yaw_sd)),
                "mu_scale": float(1 + rng.normal(0, args.mu_sd))} if r > 0 else {"dxy": [0, 0], "dyaw": 0.0, "mu_scale": 1.0}
        env.set_perturbation(**pert)
        env.reset()
        t0 = time.perf_counter()
        env.approach()
        saved = env.save_state()
        print(f"[repeat {r}] approach done in {time.perf_counter()-t0:.1f}s wall; pert={pert}", flush=True)
        for i, F in enumerate(args.F):
            for j, s in enumerate(args.s):
                info, arr = run_branch(env, saved, F, s, tol, None)
                info.update({"repeat": r, "F_idx": i, "s_idx": j, **{f"pert_{k}": v for k, v in pert.items()}})
                np.savez_compressed(out / "branches" / f"F{i}_s{j}_r{r}.npz", rec=arr, cols=np.array(env.SAMPLE_COLS))
                rows.append(info)
                print(f"  F={F:5.1f} s={s:4.2f} r={r}: ss_f=({info['ss_f_left']:5.1f},{info['ss_f_right']:5.1f})N "
                      f"q1={info['ss_q1']*1000:5.2f}mm slip={info['slip_disp_max']*1000:5.1f}mm peak={info['peak_force']:5.1f}N "
                      f"D={info['D']:6.2f} -> slip={int(info['slip'])} dmg={int(info['damage_primary'])} "
                      f"complete={int(info['complete'])}  [{info['wall_s']:.2f}s wall / {info['sim_s']:.1f}s sim]", flush=True)
    meta = {"config": env.config_dict(), "tolerances": tol, "F": args.F, "s": args.s, "repeats": args.repeats,
            "solver": {"timestep": float(env.sim.model.opt.timestep), "noslip_iterations": int(env.sim.model.opt.noslip_iterations),
                       "impratio": float(env.sim.model.opt.impratio), "cone": int(env.sim.model.opt.cone),
                       "integrator": int(env.sim.model.opt.integrator)},
            "gripper": {"kp": float(env.sim.model.actuator_gainprm[env.grip_act_idx[0], 0])},
            "seed": args.seed, "sample_cols": env.SAMPLE_COLS}
    (out / "summary.json").write_text(json.dumps({"meta": meta, "branches": rows}, indent=1, default=float))
    print("wrote", out / "summary.json")


if __name__ == "__main__":
    main()
