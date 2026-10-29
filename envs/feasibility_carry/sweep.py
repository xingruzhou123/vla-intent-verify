"""Same-state F × s branching sweep for one physical property combination (m, μ, CoM offset).
Fragility, t_win, κ and the deadline are label parameters and are applied post hoc (labels.py), so one physical sweep
serves both fragility levels and both urgency levels.

Usage (plan §6 grid, R = 5):
  MUJOCO_GL=egl .venv/bin/python -m envs.feasibility_carry.sweep --out results/phase1/demo_a --mass 1.5 --mu 0.2 --repeats 5
"""
import argparse, json, time, pathlib
import numpy as np
from .env import FeasibilityCarry, CarryConfig
from . import labels

F_GRID = [round(float(x), 3) for x in np.geomspace(5.0, 70.0, 8)]     # plan §6 (frozen)
S_GRID = [round(float(x), 3) for x in np.geomspace(0.5, 3.0, 6)]
URGENCY = {"relaxed": 2.0, "urgent": 1.0}      # T_max = mult × T_carry(s0) + settle margin (plan §3.7; urgent = nominal-speed execution plus settling, frozen 2026-10-29)
SETTLE_MARGIN = 0.5
PERT = {"pose_sd": 0.002, "yaw_sd": float(np.deg2rad(2.0)), "mu_sd": 0.03}


def deadline_levels(cfg, urgency=URGENCY, settle=SETTLE_MARGIN):
    return {k: round(v * cfg.T_carry(1.0) + settle, 4) for k, v in urgency.items()}


def perturbation(rng, r):
    if r == 0:
        return {"dxy": [0.0, 0.0], "dyaw": 0.0, "mu_scale": 1.0}
    return {"dxy": rng.normal(0, PERT["pose_sd"], 2).tolist(), "dyaw": float(rng.normal(0, PERT["yaw_sd"])),
            "mu_scale": float(1 + rng.normal(0, PERT["mu_sd"]))}


def run_branch(env, saved, F, s, T_max_levels, fragility=None):
    """Restore the branch state, grasp with F, carry with s, release, label. Returns (info, rec array)."""
    env.restore_state(saved)
    rec = []
    t0 = time.perf_counter()
    gs = env.grasp(F, rec)
    pos, mat = env.carry(s, rec)
    env.release(pos, mat, rec)
    wall = time.perf_counter() - t0
    arr = np.array(rec)
    lab = labels.all_labels(arr, env.SAMPLE_COLS, env.goal_pos(), env.cfg.pad_half, env.cfg.T_carry(s), T_max_levels,
                            fragility=fragility or labels.FRAGILITY)
    info = {"F": float(F), "s": float(s), "wall_s": wall, "sim_s": float(arr[-1, 0] - arr[0, 0]), **gs, **lab}
    return info, arr


def make_env(args):
    cfg = CarryConfig(obj_mass=args.mass, obj_mu=args.mu, com_offset_frac=args.com_frac)
    return FeasibilityCarry(cfg, seed=args.seed)


def meta_dict(env, args, T_max, perts, fragility):
    m = env.sim.model
    return {"config": env.config_dict(), "F": args.F, "s": args.s, "repeats": args.repeats,
            "T_nominal": env.cfg.T_carry(1.0), "urgency_mult": URGENCY, "settle_margin": SETTLE_MARGIN, "T_max": T_max,
            "label_settings": {"delta_slip": labels.DELTA_SLIP, "drop_dist": labels.DROP_DIST, "t_win": labels.T_WIN,
                               "t_win_primary": labels.T_WIN_PRIMARY, "kappas": labels.KAPPAS,
                               "kappa_primary": labels.KAPPA_PRIMARY, "f_tol_mult": labels.F_TOL_MULT,
                               "fragility": fragility, "goal_tol": [labels.GOAL_XY_TOL, labels.GOAL_Z_TOL]},
            "object_id": args.object_id,
            "perturbation_sd": PERT, "perturbations": perts,
            "solver": {"timestep": float(m.opt.timestep), "noslip_iterations": int(m.opt.noslip_iterations),
                       "impratio": float(m.opt.impratio), "cone": int(m.opt.cone), "integrator": int(m.opt.integrator)},
            "gripper": {"kp": float(m.actuator_gainprm[env.grip_act_idx[0], 0]),
                        "finger_damping": float(m.dof_damping[m.jnt_dofadr[m.actuator_trnid[env.grip_act_idx[0], 0]]]),
                        "finger_frictionloss": float(m.dof_frictionloss[m.jnt_dofadr[m.actuator_trnid[env.grip_act_idx[0], 0]]])},
            "closing_axis_world": env.closing_axis_world().round(4).tolist(),
            "contact": {"obj_solref": m.geom_solref[env.obj_gids[0]].tolist(), "obj_solimp": m.geom_solimp[env.obj_gids[0]].tolist(),
                        "obj_friction": m.geom_friction[env.obj_gids[0]].tolist(), "obj_priority": int(m.geom_priority[env.obj_gids[0]]),
                        "obj_condim": int(m.geom_condim[env.obj_gids[0]])},
            "com_offset_m": float(env.cfg.com_offset_frac * 2 * env.cfg.obj_half_size[0]),
            "seed": args.seed, "sample_cols": env.SAMPLE_COLS}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--F", type=float, nargs="+", default=F_GRID)
    ap.add_argument("--s", type=float, nargs="+", default=S_GRID)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--mass", type=float, required=True)
    ap.add_argument("--mu", type=float, required=True)
    ap.add_argument("--com_frac", type=float, default=0.0, help="CoM offset along world x as a fraction of the width")
    ap.add_argument("--fragility", nargs="+", default=None, help="per-object damage tolerances, e.g. fragile=18.3 rigid=52.0 (default: bin centres 20 / 50 N)")
    ap.add_argument("--object_id", default=None, help="object id from objects.json (recorded in meta)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    out = pathlib.Path(args.out); (out / "branches").mkdir(parents=True, exist_ok=True)
    env = make_env(args)
    T_max = deadline_levels(env.cfg)
    fragility = {k: float(v) for k, v in (x.split("=") for x in args.fragility)} if args.fragility else dict(labels.FRAGILITY)
    rng = np.random.default_rng(args.seed)
    rows, perts = [], []
    for r in range(args.repeats):
        pert = perturbation(rng, r); perts.append(pert)
        env.set_perturbation(**pert)
        env.reset()
        t0 = time.perf_counter()
        env.approach()
        saved = env.save_state()
        np.savez_compressed(out / f"branch_state_r{r}.npz", **saved)
        print(f"[repeat {r}] approach done in {time.perf_counter()-t0:.1f}s wall; pert={pert}; "
              f"closing axis (world) = {env.closing_axis_world().round(3)}", flush=True)
        for i, F in enumerate(args.F):
            for j, s in enumerate(args.s):
                info, arr = run_branch(env, saved, F, s, T_max, fragility)
                info.update({"repeat": r, "F_idx": i, "s_idx": j, "pert": pert})
                np.savez_compressed(out / "branches" / f"F{i}_s{j}_r{r}.npz", rec=arr.astype(np.float32),
                                    cols=np.array(env.SAMPLE_COLS))
                rows.append(info)
                fr = info["fragility"]
                print(f"  F={F:5.1f} s={s:4.2f} r={r}: ss_f=({info['ss_f_left']:5.1f},{info['ss_f_right']:5.1f})N "
                      f"a=(lift {info['a_lift']:4.1f}, lat {info['a_lat']:4.1f}) "
                      f"slip: tr={info['slip_trans_max']*1000:5.1f}mm tor={info['slip_torsion_max_deg']:5.1f}° drop={int(info['dropped'])} "
                      f"N50={info['score_N']['50ms']:5.1f} L50={info['score_L']['50ms']['k1']:5.1f} "
                      f"-> slip={int(info['slip'])} dmg(frag/rigid)={int(fr['fragile']['damage_primary'])}/{int(fr['rigid']['damage_primary'])} "
                      f"done={int(info['completed'])} xy={info['final_xy_err']*1000:4.0f}mm [{info['wall_s']:.2f}s]", flush=True)
        (out / "summary.json").write_text(json.dumps({"meta": meta_dict(env, args, T_max, perts, fragility), "branches": rows},
                                                     indent=1, default=float))
    print("wrote", out / "summary.json")


if __name__ == "__main__":
    main()
