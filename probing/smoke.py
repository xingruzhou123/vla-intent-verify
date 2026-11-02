"""Smoke test for the probe set and the branch/restore mechanism (probing/design.md §9).

Runs on a few objects × the stored CFD branch states (contexts) × repeats, and checks:
  1. restore is exact: state fingerprint before a probe == fingerprint after probe + restore (and == the stored state);
  2. repeats vary only through the intended noise: re-executions from the same state are bit-identical, observations
     differ only through the sensor-noise draw; different stored states (CFD perturbation repeats) differ physically;
  3. probe outcomes are recorded (ground truth + noisy observation per branch);
  4. timing per branch (probe, probe + carry, grid carry);
  5. C1 replay: after a probe and restore, the CFD grid branch (F_idx 4, s_idx 2) reproduces the stored sweep record;
  6. C3 demo: probe, no restore, carry from the post-probe state — labels compared with the stored C0 branch.

Usage (from the repo root, CFD venv):
  MUJOCO_GL=egl nice -n 10 .venv/bin/python -m probing.smoke --out probing/results/smoke \
      --objects light_normal_balanced_0 heavy_slippery_balanced_0 --states 0 1 2 --repeats 3 --boundary medium_normal_balanced_0
"""
import argparse, json, pathlib, time, sys
import numpy as np
from envs.feasibility_carry import sweep as sw
from envs.feasibility_carry import labels
from . import probes as P

C1_CELL = (4, 2)          # F_idx, s_idx of the CFD grid cell used for the C1 replay and the C3 demo (F = 22.6 N, s = 1.02)


def load_sweep(root, oid):
    d = pathlib.Path(root) / oid
    S = json.loads((d / "summary.json").read_text())
    return d, S


def run_object(obj, sweep_root, states, repeats, probe_specs, extra_probes, out_dir, log, seed=0):
    d, S = load_sweep(sweep_root, obj["object_id"])
    meta = S["meta"]; Fs, ss = meta["F"], meta["s"]
    fragility = {k: float(v) for k, v in meta["label_settings"]["fragility"].items()}
    T_max = meta["T_max"]
    t0 = time.perf_counter()
    env = P.make_env(obj, seed=meta["seed"])
    log(f"[{obj['object_id']}] env built in {time.perf_counter() - t0:.1f} s; m={obj['mass']} mu={obj['mu']} d={obj['com_frac']}")
    rows, checks = [], []
    for r in states:
        pert = meta["perturbations"][r]
        env.set_perturbation(**pert)          # the friction scale is model-level: it must be re-applied before the state is restored
        env.reset()
        saved = P.load_state(d / f"branch_state_r{r}.npz")
        env.restore_state(saved)
        fp0 = P.state_fingerprint(env)
        fp_saved = P.record_hash(saved["state"])
        ctx = {"object_id": obj["object_id"], "state": r, "pert": pert}
        # ---- probes × repeats
        for spec in probe_specs + extra_probes:
            hashes = []
            for rep in range(repeats):
                info, arr, seg = P.run_probe(env, saved, spec)
                outc = P.probe_outcome(arr, env.SAMPLE_COLS, seg, spec, env.cfg.pad_half, fragility)
                rng = np.random.default_rng([seed, hash(obj["object_id"]) % 10**6, r, hash(spec.label()) % 10**6, rep])
                obs = P.observe(outc, spec, rng)
                env.restore_state(saved)
                fp1 = P.state_fingerprint(env)
                h = P.record_hash(arr); hashes.append(h)
                rows.append({**ctx, "kind": "probe", "repeat": rep, **info, "outcome": outc, "obs": obs, "rec_hash": h,
                             "restore_exact": fp1 == fp0, "fp_before": fp0, "fp_after": fp1})
                if rep == 0:
                    np.savez_compressed(out_dir / f"{obj['object_id']}_r{r}_{spec.label()}.npz", rec=arr.astype(np.float32),
                                        cols=np.array(env.SAMPLE_COLS), seg=json.dumps(seg))
                o = outc
                log(f"  r={r} {spec.label():18s} rep={rep}: wall {info['wall_s']:.2f}s sim {info['sim_s']:.2f}s rows {info['n_rows']} | "
                    f"grip ({o['grip_f_left']:.1f},{o['grip_f_right']:.1f})N N50={o['score_N50']:.1f} dmg(frag/rig)={int(o['damage']['fragile'])}/{int(o['damage']['rigid'])}"
                    + (f" | held={int(o['held'])} slip={o['slip_trans_max_m']*1000:.1f}mm zgain={o['z_gain_top_m']*1000:.1f}mm load={o['pad_load_top_N']:.1f}N shift={o['obj_shift_after_m']*1000:.1f}mm"
                       f" | obs held_hat={int(obs['held_hat'])} slip_hat={obs['slip_hat_m']*1000:.1f}mm load_hat={obs['load_hat_N']:.1f}N" if spec.kind == "lift"
                       else f" | shift={o['obj_shift_after_m']*1000:.1f}mm") + f" | restore_exact={fp1 == fp0}")
            checks.append({**ctx, "probe": spec.label(), "n_repeats": repeats, "identical_repeats": len(set(hashes)) == 1,
                           "restore_exact_all": all(x["restore_exact"] for x in rows[-repeats:])})
        # ---- C1 replay: restore, run the CFD grid cell, compare with the stored record
        i, j = C1_CELL
        info, arr = sw.run_branch(env, saved, Fs[i], ss[j], T_max, fragility)
        z = np.load(d / "branches" / f"F{i}_s{j}_r{r}.npz", allow_pickle=True)
        stored = z["rec"]
        same_shape = stored.shape == arr.shape
        maxdiff = float(np.abs(stored.astype(np.float64) - arr.astype(np.float32).astype(np.float64)).max()) if same_shape else None
        stored_row = next(b for b in S["branches"] if b["F_idx"] == i and b["s_idx"] == j and b["repeat"] == r)
        lab_same = all(info["fragility"][f]["phys_success"] == stored_row["fragility"][f]["phys_success"] and
                       info["fragility"][f]["damage_primary"] == stored_row["fragility"][f]["damage_primary"] for f in fragility) \
            and info["slip"] == stored_row["slip"] and info["completed"] == stored_row["completed"]
        env.restore_state(saved); fp2 = P.state_fingerprint(env)
        rows.append({**ctx, "kind": "c1_replay", "F": Fs[i], "s": ss[j], "wall_s": info["wall_s"], "sim_s": info["sim_s"], "n_rows": int(len(arr)),
                     "same_shape": same_shape, "max_abs_diff_vs_stored": maxdiff, "labels_match_stored": lab_same,
                     "labels": {"slip": info["slip"], "completed": info["completed"], "damage": {f: info["fragility"][f]["damage_primary"] for f in fragility}},
                     "restore_exact": fp2 == fp0})
        log(f"  r={r} C1 replay F={Fs[i]:.1f} s={ss[j]:.2f}: wall {info['wall_s']:.2f}s rows {len(arr)} (stored {stored.shape[0]}) max|diff| vs stored = {maxdiff} labels_match={lab_same} restore_exact={fp2 == fp0}")
        # ---- the same replay WITHOUT re-applying the perturbation (what refine.py does): documents the friction-scale quirk
        if r != 0:
            env.set_perturbation(); env.reset(); env.restore_state(saved)
            info_np, arr_np = sw.run_branch(env, saved, Fs[i], ss[j], T_max, fragility)
            md_np = float(np.abs(stored.astype(np.float64) - arr_np.astype(np.float32).astype(np.float64)).max()) if stored.shape == arr_np.shape else None
            rows.append({**ctx, "kind": "c1_replay_no_pert", "max_abs_diff_vs_stored": md_np, "mu_scale_applied": 1.0, "mu_scale_stored": pert["mu_scale"],
                         "labels": {"slip": info_np["slip"], "completed": info_np["completed"]}})
            log(f"  r={r} C1 replay without re-applying mu_scale={pert['mu_scale']:.4f}: max|diff| vs stored = {md_np}")
            env.set_perturbation(**pert); env.reset(); env.restore_state(saved)
            assert P.state_fingerprint(env) == fp0
        # ---- C3 demo: lift probe at 16 N, no restore, then the same grid cell from the post-probe state
        spec = P.ProbeSpec("lift", 15.494)
        env.restore_state(saved)
        rec = []
        t1 = time.perf_counter()
        pinfo, parr, seg = P.execute_probe(env, spec, rec)
        cinfo, carr = P.carry_after(env, Fs[i], ss[j], T_max, fragility, rec)
        wall = time.perf_counter() - t1
        np.savez_compressed(out_dir / f"{obj['object_id']}_r{r}_C3_{spec.label()}_F{i}_s{j}.npz", rec=np.array(rec, dtype=np.float32), cols=np.array(env.SAMPLE_COLS))
        rows.append({**ctx, "kind": "c3_probe_then_carry", "probe": spec.label(), "F": Fs[i], "s": ss[j], "wall_s": wall,
                     "wall_probe_s": pinfo["wall_s"], "wall_carry_s": cinfo["wall_s"], "sim_s": pinfo["sim_s"] + cinfo["sim_s"], "n_rows": len(rec),
                     "probe_outcome": P.probe_outcome(parr, env.SAMPLE_COLS, seg, spec, env.cfg.pad_half, fragility),
                     "labels": {"slip": cinfo["slip"], "completed": cinfo["completed"], "damage": {f: cinfo["fragility"][f]["damage_primary"] for f in fragility},
                                "slip_trans_max": cinfo["slip_trans_max"], "final_xy_err": cinfo["final_xy_err"]},
                     "labels_c0": {"slip": stored_row["slip"], "completed": stored_row["completed"], "damage": {f: stored_row["fragility"][f]["damage_primary"] for f in fragility},
                                   "slip_trans_max": stored_row["slip_trans_max"], "final_xy_err": stored_row["final_xy_err"]}})
        log(f"  r={r} C3 {spec.label()} then F={Fs[i]:.1f} s={ss[j]:.2f}: wall {wall:.2f}s (probe {pinfo['wall_s']:.2f} + carry {cinfo['wall_s']:.2f}) | "
            f"carry labels slip={int(cinfo['slip'])} done={int(cinfo['completed'])} vs C0 slip={int(stored_row['slip'])} done={int(stored_row['completed'])}; "
            f"slip {cinfo['slip_trans_max']*1000:.1f} vs {stored_row['slip_trans_max']*1000:.1f} mm; xy err {cinfo['final_xy_err']*1000:.0f} vs {stored_row['final_xy_err']*1000:.0f} mm")
    return rows, checks, {"fragility": fragility, "T_max": T_max, "F": Fs, "s": ss, "label_version": labels.LABEL_VERSION}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="probing/results/smoke")
    ap.add_argument("--sweep", default="data/sweep_v1")
    ap.add_argument("--objects_json", default="data/objects_v1.json")
    ap.add_argument("--objects", nargs="+", default=["light_normal_balanced_0", "heavy_slippery_balanced_0"])
    ap.add_argument("--states", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--boundary", default=None, help="extra object run with the 10.6 N lift only (static slip boundary ≈ 10–12 N)")
    ap.add_argument("--quick", action="store_true", help="one object, one state, one probe, one repeat")
    args = ap.parse_args(argv)
    out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "log.txt", "w")
    def log(msg):
        print(msg, flush=True); logf.write(msg + "\n"); logf.flush()
    objs = {o["object_id"]: o for o in json.loads(pathlib.Path(args.objects_json).read_text())["objects"]}
    specs = P.SMOKE_PROBES
    if args.quick:
        args.objects, args.states, args.repeats, specs = args.objects[:1], args.states[:1], 1, P.SMOKE_PROBES[2:3]
    t0 = time.perf_counter()
    all_rows, all_checks, metas = [], [], {}
    for oid in args.objects:
        rows, checks, m = run_object(objs[oid], args.sweep, args.states, args.repeats, specs, [], out, log)
        all_rows += rows; all_checks += checks; metas[oid] = m
    if args.boundary and not args.quick:
        rows, checks, m = run_object(objs[args.boundary], args.sweep, args.states, args.repeats, [], [P.ProbeSpec("lift", 10.628, name="lift_F10.628_h1cm_boundary")], out, log)
        all_rows += rows; all_checks += checks; metas[args.boundary] = m
    total = time.perf_counter() - t0
    (out / "summary.json").write_text(json.dumps({"args": vars(args), "probe_specs": [P.__dict__["asdict"](s) for s in specs], "sensor": P.SENSOR,
                                                  "meta": metas, "rows": all_rows, "checks": all_checks, "total_wall_s": total,
                                                  "load_avg_at_end": list(__import__("os").getloadavg())}, indent=1, default=float))
    log(f"done: {len(all_rows)} rows in {total/60:.1f} min wall; summary -> {out/'summary.json'}")


if __name__ == "__main__":
    main()
