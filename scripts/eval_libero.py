"""Run lerobot-eval with the project config (configs/pi05_libero.yaml).

Usage:
  .venv/bin/python scripts/eval_libero.py --policy lerobot/pi05_libero_base --suite libero_spatial \
      --task-ids 0 1 2 --n-episodes 1 --out results/smoke/base
Any extra args after `--` are passed through to lerobot-eval.
"""
import argparse, os, subprocess, sys, pathlib, yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/pi05_libero.yaml"))
    ap.add_argument("--policy", default=None, help="Hub id or local path; default = config policy.path")
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--task-ids", type=int, nargs="*", default=None)
    ap.add_argument("--n-episodes", type=int, default=10)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=None)
    args, extra = ap.parse_known_args()
    extra = [a for a in extra if a != "--"]

    cfg = yaml.safe_load(open(args.config))
    pol, env, ev = cfg["policy"], cfg["env"], cfg["eval"]
    policy = args.policy or pol["path"]
    if not policy:
        sys.exit("no policy: pass --policy or set policy.path in the config")

    cmd = [
        str(ROOT / ".venv/bin/lerobot-eval"),
        f"--output_dir={args.out}",
        f"--policy.path={policy}",
        f"--policy.n_action_steps={pol['n_action_steps']}",
        f"--policy.num_inference_steps={pol['num_inference_steps']}",
        f"--policy.empty_cameras={pol['empty_cameras']}",
        f"--policy.dtype={pol['dtype']}",
        f"--policy.compile_model={str(pol['compile_model']).lower()}",
        f"--policy.device=cuda",
        f"--env.type={env['type']}",
        f"--env.task={args.suite}",
        f"--env.control_mode={env['control_mode']}",
        f"--env.init_states={str(env['init_states']).lower()}",
        f"--env.hard_reset={str(env['hard_reset']).lower()}",
        f"--env.max_parallel_tasks={env['max_parallel_tasks']}",
        f"--eval.batch_size={ev['batch_size']}",
        f"--eval.n_episodes={args.n_episodes}",
        f"--seed={args.seed if args.seed is not None else ev['seed']}",
    ] + extra
    if args.task_ids is not None:
        cmd.append("--env.task_ids=[" + ",".join(map(str, args.task_ids)) + "]")
    os.environ.setdefault("MUJOCO_GL", env.get("mujoco_gl", "egl"))
    print(" ".join(cmd), flush=True)
    sys.exit(subprocess.call(cmd, cwd=ROOT))

if __name__ == "__main__":
    main()
