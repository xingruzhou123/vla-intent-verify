"""Human audit sample (owner item 6, 2026-11-02; data gate G6 in docs/data_spec.md): 50 branches drawn at random,
stratified over the 8 label classes (positives and negatives of each), replayed from their stored branch state with
offscreen rendering, written as a short video and a frame strip with the recorded force / slip traces, plus an audit
sheet (branch id, class, predicted labels, blank verdict column) for the owner's hand review.

The replay re-simulates the branch (same state, same F, s and perturbation; the simulation is near-deterministic) and
the replayed labels are compared with the stored ones (replay-fidelity note on the sheet).

Usage: MUJOCO_GL=egl .venv/bin/python -m analysis.audit_sample --sweeps data/sweep_v1 --objects data/objects_v1.json --out results/phase1/audit --n 50 --seed 0
"""
import argparse, json, pathlib, csv, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CLASSES = ["slip", "completed", "lost_contact", "damage_primary", "damage_shear", "physical_safe", "deadline_success_relaxed", "deadline_success_urgent"]
PHASES = ["approach", "ramp", "hold", "lift", "lateral", "lower", "settle", "release"]


def label_value(row, cls, frag):
    if cls in ("slip", "completed", "lost_contact"):
        return bool(row[cls])
    return bool(row["fragility"][frag][cls])


def pick(rows_by_obj, n, seed):
    """Stratified draw: for each class, alternate positives and negatives; branch = (object, F_idx, s_idx, repeat, frag)."""
    rng = np.random.default_rng(seed)
    pool = [(oid, r) for oid, rows in rows_by_obj.items() for r in rows]
    picked, seen = [], set()
    per_class = n // len(CLASSES); extra = n - per_class * len(CLASSES)
    for ci, cls in enumerate(CLASSES):
        want = per_class + (1 if ci < extra else 0)
        for k in range(want):
            target = (k % 2 == 0)                                    # alternate positive / negative
            frag = ["fragile", "rigid"][rng.integers(2)]
            cands = [(oid, r) for oid, r in pool if label_value(r, cls, frag) == target and (oid, r["F_idx"], r["s_idx"], r["repeat"]) not in seen]
            if not cands:
                cands = [(oid, r) for oid, r in pool if (oid, r["F_idx"], r["s_idx"], r["repeat"]) not in seen]
            oid, r = cands[rng.integers(len(cands))]
            seen.add((oid, r["F_idx"], r["s_idx"], r["repeat"]))
            picked.append({"object": oid, "F_idx": r["F_idx"], "s_idx": r["s_idx"], "repeat": r["repeat"], "frag": frag, "class": cls, "target": target})
    return picked


def render_branch(d, meta, item, out_dir, fps=10, size=(480, 368)):
    os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco, imageio
    from envs.feasibility_carry.env import FeasibilityCarry, CarryConfig
    from envs.feasibility_carry.sweep import run_branch
    from envs.feasibility_carry.refine import load_states
    cfg = CarryConfig(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in meta["config"].items()})
    env = FeasibilityCarry(cfg, seed=meta["seed"]); env.reset()
    states = load_states(d, meta["repeats"])
    m, dd = env.sim.model._model, env.sim.data._data
    renderer = mujoco.Renderer(m, size[1], size[0])
    cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0.0, 0.03, 0.85]; cam.distance = 0.5; cam.azimuth = 150; cam.elevation = -18
    frames, times = [], []
    every = max(1, int(round(1.0 / fps / m.opt.timestep)))
    counter = {"k": 0}
    def hook(e):
        counter["k"] += 1
        if counter["k"] % every == 0:
            renderer.update_scene(dd, camera=cam); frames.append(renderer.render().copy()); times.append(float(dd.time))
    env.step_hook = hook
    frag = {k: float(v) for k, v in meta["label_settings"]["fragility"].items()}
    F, s = meta["F"][item["F_idx"]], meta["s"][item["s_idx"]]
    info, arr = run_branch(env, states[item["repeat"]], F, s, meta["T_max"], frag)
    env.step_hook = None
    stem = f"{item['object']}__F{item['F_idx']}_s{item['s_idx']}_r{item['repeat']}"
    imageio.mimwrite(out_dir / f"{stem}.mp4", frames, fps=fps, codec="libx264", quality=7)
    # frame strip: 8 frames at the phase transitions + end, with the recorded traces below
    cols = list(env.SAMPLE_COLS); ph = arr[:, cols.index("phase")]; t = arr[:, cols.index("t")]
    keys = []
    for p in range(1, 8):
        idx = np.flatnonzero(ph == p)
        if len(idx): keys.append(t[idx[0]])
    keys.append(t[-1]); keys = keys[:8]
    sel = [int(np.argmin(np.abs(np.array(times) - tk))) for tk in keys]
    fig = plt.figure(figsize=(16, 7)); gs = fig.add_gridspec(2, len(sel), height_ratios=[1.2, 1])
    for k, si in enumerate(sel):
        ax = fig.add_subplot(gs[0, k]); ax.imshow(frames[si]); ax.set_axis_off()
        pk = int(ph[np.argmin(np.abs(t - times[si]))]); ax.set_title(f"t = {times[si]-t[0]:.2f} s  {PHASES[pk]}", fontsize=8)
    ax = fig.add_subplot(gs[1, :])
    tt = t - t[0]
    ax.plot(tt, arr[:, cols.index("f_left")], lw=1, label="pad normal force left [N]"); ax.plot(tt, arr[:, cols.index("f_right")], lw=1, label="right [N]")
    ax.axhline(F, color="k", ls="--", lw=0.8, label=f"commanded F = {F:.1f} N")
    rp = arr[:, [cols.index("relpad_y"), cols.index("relpad_z")]]
    hold = np.flatnonzero(ph == 2); ref = rp[hold[t[hold] >= t[hold[-1]] - 0.25]].mean(axis=0) if len(hold) else rp[0]
    ax2 = ax.twinx(); ax2.plot(tt, np.linalg.norm(rp - ref, axis=1) * 1e3, color="C3", lw=1, label="pad-plane slip [mm]"); ax2.axhline(5, color="C3", ls=":", lw=0.8)
    ax2.plot(tt, (arr[:, cols.index("nc_left")] + arr[:, cols.index("nc_right")]) * 2, color="C2", lw=0.8, alpha=0.6, label="contacts × 2")
    ax2.set_ylabel("slip [mm] / contacts"); ax2.set_ylim(0, 60)
    for p in range(1, 8):
        idx = np.flatnonzero(ph == p)
        if len(idx): ax.axvspan(tt[idx[0]], tt[idx[-1]], color=plt.cm.Pastel1(p / 8), alpha=0.35, lw=0)
    ax.set_xlabel("t since branch [s]"); ax.set_ylabel("force [N]"); ax.set_ylim(0, max(75, F * 1.3)); ax.legend(fontsize=7, loc="upper left"); ax2.legend(fontsize=7, loc="upper right")
    lab = info["fragility"][item["frag"]]
    fig.suptitle(f"{stem}  F = {F:.1f} N, s = {s:g}, fragility {item['frag']} (f_tol {frag[item['frag']]:.1f} N) — stored class under audit: {item['class']} = {item['target']} | "
                 f"replay: slip {info['slip']}, lost_contact {info['lost_contact']}, completed {info['completed']}, damage {lab['damage_primary']}, shear {lab['damage_shear']}, "
                 f"deadline relaxed/urgent {lab['deadline_success_relaxed']}/{lab['deadline_success_urgent']}", fontsize=8)
    fig.savefig(out_dir / f"{stem}.png", dpi=110, bbox_inches="tight"); plt.close(fig)
    return stem, info


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweeps", required=True); ap.add_argument("--objects", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=50); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)
    out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)
    objs = [o["object_id"] for o in json.loads(pathlib.Path(args.objects).read_text())["objects"]]
    S = {oid: json.loads((pathlib.Path(args.sweeps) / oid / "summary.json").read_text()) for oid in objs}
    picked = pick({oid: S[oid]["branches"] for oid in objs}, args.n, args.seed)
    if args.limit: picked = picked[: args.limit]
    sheet = []
    for n, item in enumerate(picked):
        d = pathlib.Path(args.sweeps) / item["object"]; meta = S[item["object"]]["meta"]
        stored = next(r for r in S[item["object"]]["branches"] if r["F_idx"] == item["F_idx"] and r["s_idx"] == item["s_idx"] and r["repeat"] == item["repeat"])
        stem, info = render_branch(d, meta, item, out)
        sl = stored["fragility"][item["frag"]]; rl = info["fragility"][item["frag"]]
        stored_lab = {"slip": stored["slip"], "lost_contact": stored["lost_contact"], "completed": stored["completed"], "damage_primary": sl["damage_primary"], "damage_shear": sl["damage_shear"],
                      "physical_safe": sl["physical_safe"], "deadline_success_relaxed": sl["deadline_success_relaxed"], "deadline_success_urgent": sl["deadline_success_urgent"]}
        replay_lab = {"slip": info["slip"], "lost_contact": info["lost_contact"], "completed": info["completed"], "damage_primary": rl["damage_primary"], "damage_shear": rl["damage_shear"],
                      "physical_safe": rl["physical_safe"], "deadline_success_relaxed": rl["deadline_success_relaxed"], "deadline_success_urgent": rl["deadline_success_urgent"]}
        mism = [k for k in stored_lab if bool(stored_lab[k]) != bool(replay_lab[k])]
        sheet.append({"id": n, "branch": stem, "object": item["object"], "F_N": meta["F"][item["F_idx"]], "s": meta["s"][item["s_idx"]], "repeat": item["repeat"], "fragility": item["frag"],
                      "f_tol_N": meta["label_settings"]["fragility"][item["frag"]], "audited_class": item["class"], "stored_value": stored_lab[item["class"]],
                      **{f"pred_{k}": int(v) for k, v in stored_lab.items()}, "replay_mismatch": ";".join(mism), "slip_trans_max_mm": round(stored["slip_trans_max"] * 1e3, 2),
                      "score_N_50ms": round(stored["score_N"]["50ms"], 2), "final_xy_err_mm": round(stored["final_xy_err"] * 1e3, 1), "owner_verdict": "", "owner_note": ""})
        print(f"[{n+1}/{len(picked)}] {stem} class {item['class']} = {item['target']} replay mismatch: {mism or 'none'}", flush=True)
    with open(out / "audit_sheet.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sheet[0].keys())); w.writeheader(); w.writerows(sheet)
    lines = ["# Human audit sheet — Phase 1 sweep labels (G6)\n", f"{len(sheet)} branches, stratified over {len(CLASSES)} label classes (alternating positive / negative), seed {args.seed}. "
             "Each branch has <branch>.mp4 (replay, 10 fps) and <branch>.png (8 phase frames + recorded pad forces, pad-plane slip, contact count). "
             "Fill `owner_verdict` (ok / wrong / unsure) and `owner_note` in audit_sheet.csv; the same rows are below.\n",
             "Label definitions (version 3): slip = pad-plane translation > 5 mm, or pad-edge torsion > 5 mm, or the object left the gripper; lost_contact = both pads out of contact ≥ 50 ms before the scheduled release; "
             "completed = at rest within 3 cm / 1 cm of the goal, on the table, and no lost_contact; damage_primary = sustained (50 ms) per-pad normal force > f_tol while held; damage_shear = same with N + |T|; "
             "physical_safe = no slip ∧ no damage_primary; deadline_success_<urgency> = physical_safe ∧ completed ∧ T_carry ≤ T_max (relaxed 4.0 s, urgent 2.25 s).\n",
             "| id | branch | class | stored | slip | lost | completed | dmg | shear | safe | dl relaxed | dl urgent | replay mismatch | slip max [mm] | score_N | xy err [mm] | verdict | note |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in sheet:
        lines.append(f"| {r['id']} | {r['branch']} | {r['audited_class']} | {int(r['stored_value'])} | {r['pred_slip']} | {r['pred_lost_contact']} | {r['pred_completed']} | {r['pred_damage_primary']} | {r['pred_damage_shear']} | {r['pred_physical_safe']} | "
                     f"{r['pred_deadline_success_relaxed']} | {r['pred_deadline_success_urgent']} | {r['replay_mismatch'] or '—'} | {r['slip_trans_max_mm']} | {r['score_N_50ms']} | {r['final_xy_err_mm']} |  |  |")
    n_mis = sum(bool(r["replay_mismatch"]) for r in sheet)
    lines.append(f"\nReplay fidelity on this sample: {len(sheet) - n_mis} of {len(sheet)} branches reproduce every stored label on re-simulation from the stored state.")
    (out / "audit_sheet.md").write_text("\n".join(lines))
    print(f"written {out/'audit_sheet.csv'} and audit_sheet.md; replay mismatches: {n_mis}")


if __name__ == "__main__":
    main()
