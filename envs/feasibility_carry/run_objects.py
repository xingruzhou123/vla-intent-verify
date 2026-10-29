"""Run the grid sweep and the boundary refinement for every object of an objects file, several processes in parallel
(MuJoCo is single-threaded). Not to be started before the owner confirms the full sweep (CLAUDE.md).

Usage: MUJOCO_GL=egl .venv/bin/python -m envs.feasibility_carry.run_objects data/objects_v1.json --out_root data/sweep_v1 --parallel 12
"""
import argparse, json, pathlib, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor


def run_one(obj, out_root, repeats, refine):
    out = pathlib.Path(out_root) / obj["object_id"]
    out.mkdir(parents=True, exist_ok=True)
    frag = [f"{k}={v}" for k, v in obj["f_tol"].items()]
    cmd = [sys.executable, "-m", "envs.feasibility_carry.sweep", "--out", str(out), "--mass", str(obj["mass"]),
           "--mu", str(obj["mu"]), "--com_frac", str(obj["com_frac"]), "--repeats", str(repeats),
           "--object_id", obj["object_id"], "--fragility", *frag]
    t0 = time.perf_counter()
    with open(out / "log.txt", "w") as f:
        r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    if r.returncode == 0 and refine:
        with open(out / "refine_stdout.txt", "w") as f:
            r = subprocess.run([sys.executable, "-m", "envs.feasibility_carry.refine", str(out)], stdout=f, stderr=subprocess.STDOUT)
    return obj["object_id"], r.returncode, time.perf_counter() - t0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("objects")
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--parallel", type=int, default=12)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--no_refine", action="store_true")
    ap.add_argument("--only", nargs="*", default=None, help="object ids to run (default all)")
    args = ap.parse_args(argv)
    objs = json.loads(pathlib.Path(args.objects).read_text())["objects"]
    if args.only:
        objs = [o for o in objs if o["object_id"] in args.only]
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.parallel) as ex:
        for oid, rc, dt in ex.map(lambda o: run_one(o, args.out_root, args.repeats, not args.no_refine), objs):
            print(f"{oid}: rc={rc} {dt/60:.1f} min", flush=True)
    print(f"done {len(objs)} objects in {(time.perf_counter()-t0)/60:.1f} min wall")


if __name__ == "__main__":
    main()
