"""Write probing/results/smoke/report.md from summary.json (only measured numbers)."""
import argparse, json, pathlib
import numpy as np
from scipy.stats import fisher_exact


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--dir", default="probing/results/smoke")
    args = ap.parse_args(argv)
    d = pathlib.Path(args.dir); S = json.loads((d / "summary.json").read_text())
    rows, checks = S["rows"], S["checks"]
    probes = [r for r in rows if r["kind"] == "probe"]; c1 = [r for r in rows if r["kind"] == "c1_replay"]
    c1np = [r for r in rows if r["kind"] == "c1_replay_no_pert"]; c3 = [r for r in rows if r["kind"] == "c3_probe_then_carry"]
    L = ["# Smoke test — probes and branch/restore mechanism\n",
         f"Objects {S['args']['objects']} (+ boundary object {S['args']['boundary']}), stored CFD states {S['args']['states']}, {S['args']['repeats']} repeats per (state, probe); "
         f"total wall {S['total_wall_s']/60:.1f} min; machine load average at the end {S['load_avg_at_end'][0]:.1f}. Label version {list(S['meta'].values())[0]['label_version']}.\n",
         "## Checks\n", "| check | result |", "|---|---|"]
    n_ok = sum(r["restore_exact"] for r in probes + c1); n = len(probes) + len(c1)
    L.append(f"| 1. restore exact (state fingerprint after probe + restore == before) | {n_ok} / {n} branches |")
    ident = sum(c["identical_repeats"] for c in checks)
    L.append(f"| 2. re-executions from the same state bit-identical (record hash) | {ident} / {len(checks)} (state, probe) groups |")
    L.append(f"| 3. outcomes recorded (ground truth + noisy observation) | {len(probes)} probe branches |")
    md = [r["max_abs_diff_vs_stored"] for r in c1]; lm = sum(r["labels_match_stored"] for r in c1)
    L.append(f"| 5. C1 replay of the CFD grid cell reproduces the stored sweep record | max abs diff over {len(c1)} replays = {max(md)}; labels match {lm} / {len(c1)} |")
    if c1np:
        L.append(f"| 5b. the same replay without re-applying the repeat's friction scale | max abs diff = {max(r['max_abs_diff_vs_stored'] for r in c1np):.3g} over {len(c1np)} replays (r > 0) |")
    L.append(f"| 6. C3 probe-then-carry branches run and labelled | {len(c3)} |")
    L += ["\n## Timing per branch (wall seconds; machine shared with other simulations)\n", "| branch type | n | mean | min | max | sim s |", "|---|---|---|---|---|---|"]
    def trow(name, rs):
        if rs: L.append(f"| {name} | {len(rs)} | {np.mean([r['wall_s'] for r in rs]):.2f} | {min(r['wall_s'] for r in rs):.2f} | {max(r['wall_s'] for r in rs):.2f} | {np.mean([r['sim_s'] for r in rs]):.2f} |")
    trow("lift probe", [r for r in probes if r["spec"]["kind"] == "lift"]); trow("squeeze probe", [r for r in probes if r["spec"]["kind"] == "squeeze"])
    trow("CFD carry replay (C1 follow-up)", c1); trow("probe + carry (C3)", c3)
    L += ["\n## Probe outcomes (ground truth, repeat 0 of each state; identical across repeats)\n",
          "| object | state | probe | held | slip max [mm] | z gain [mm] | pad load [N] | grip L/R [N] | N50 [N] | damage fragile / rigid (object's f_tol) | obj shift after [mm] | obs: held_hat, slip_hat [mm], load_hat [N] |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in probes:
        if r["repeat"] != 0: continue
        o, ob = r["outcome"], r["obs"]; ft = S["meta"][r["object_id"]]["fragility"]
        lift = r["spec"]["kind"] == "lift"
        L.append(f"| {r['object_id']} | {r['state']} | {r['probe']} | {int(o['held']) if lift else '–'} | {o['slip_trans_max_m']*1000:.1f} | {o['z_gain_top_m']*1000:.1f} | {o['pad_load_top_N']:.1f} |"
                 if lift else f"| {r['object_id']} | {r['state']} | {r['probe']} | – | – | – | – |")
        L[-1] += (f" {o['grip_f_left']:.1f}/{o['grip_f_right']:.1f} | {o['score_N50']:.1f} | {int(o['damage']['fragile'])} / {int(o['damage']['rigid'])} ({ft['fragile']:.1f} / {ft['rigid']:.1f}) | {o['obj_shift_after_m']*1000:.1f} | "
                  + (f"{int(ob['held_hat'])}, {ob['slip_hat_m']*1000:.1f}, {ob['load_hat_N']:.1f} |" if lift else "– |"))
    L += ["\n## Observation noise across repeats (same physics, different sensor draw)\n", "| object | state | probe | load_hat per repeat [N] | slip_hat per repeat [mm] |", "|---|---|---|---|---|"]
    for c in checks:
        rs = [r for r in probes if r["object_id"] == c["object_id"] and r["state"] == c["state"] and r["probe"] == c["probe"]]
        if rs and rs[0]["spec"]["kind"] == "lift":
            L.append(f"| {c['object_id']} | {c['state']} | {c['probe']} | {', '.join(f'{r['obs']['load_hat_N']:.2f}' for r in rs)} | {', '.join(f'{r['obs']['slip_hat_m']*1000:.2f}' for r in rs)} |")
    L += ["\n## C3 (probe, no restore, carry) vs C0 (stored CFD branch), same state and cell\n",
          "| object | state | probe | C0: slip / done / dmg(frag) | C3: slip / done / dmg(frag) | slip max C0 → C3 [mm] | final xy err C0 → C3 [mm] | obj shift by probe [mm] |", "|---|---|---|---|---|---|---|---|"]
    for r in c3:
        a, b = r["labels_c0"], r["labels"]
        L.append(f"| {r['object_id']} | {r['state']} | {r['probe']} | {int(a['slip'])} / {int(a['completed'])} / {int(a['damage']['fragile'])} | {int(b['slip'])} / {int(b['completed'])} / {int(b['damage']['fragile'])} | "
                 f"{a['slip_trans_max']*1000:.1f} → {b['slip_trans_max']*1000:.1f} | {a['final_xy_err']*1000:.0f} → {b['final_xy_err']*1000:.0f} | {r['probe_outcome']['obj_shift_after_m']*1000:.1f} |")
    # indistinguishability illustration: held counts over the states for the 15.49 N lift
    L += ["\n## Distribution-level test on the smoke data (illustration of §5 of design.md)\n"]
    by = {}
    for r in probes:
        if r["repeat"] == 0 and r["spec"]["kind"] == "lift":
            by.setdefault((r["object_id"], r["probe"]), []).append(int(r["outcome"]["held"]))
    objs = S["args"]["objects"]
    for p in sorted({k[1] for k in by}):
        if all((o, p) in by for o in objs):
            a, b = by[(objs[0], p)], by[(objs[1], p)]
            tab = [[sum(a), len(a) - sum(a)], [sum(b), len(b) - sum(b)]]
            L.append(f"- {p}: held counts over the stored states {objs[0]} {sum(a)}/{len(a)} vs {objs[1]} {sum(b)}/{len(b)} → Fisher exact p = {fisher_exact(tab)[1]:.3f} "
                     f"(with 3 states the two objects are **not** distinguishable at α = 0.05 even with perfectly separated outcomes; R = 5 gives p = 0.0079).")
    (d / "report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
