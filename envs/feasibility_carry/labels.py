"""Outcome labels and damage functionals (plan.md §3.2, §3.5). Implementation A.

A second, independent implementation (gate G7/G8 in docs/data_spec.md) is written separately and must not import this file.
"""
import numpy as np

COLS = None  # filled by env.SAMPLE_COLS at call time


def _col(name):
    return COLS.index(name)


def damage(rec, f_tol, D_max):
    """Primary: accumulated excess impulse D = ∫ max(0, f_n − f_tol) dt with f_n = max per-side normal force.
    Secondary: peak f_n > f_tol. Returns dict with both."""
    t = rec[:, _col("t")]
    fn = np.maximum(rec[:, _col("f_left")], rec[:, _col("f_right")])
    dt = np.diff(t, prepend=t[0])
    D = float(np.sum(np.maximum(0.0, fn - f_tol) * dt))
    peak = float(fn.max())
    return {"D": D, "peak_force": peak, "damage_primary": bool(D > D_max), "damage_secondary": bool(peak > f_tol)}


def slip(rec, i_carry_start, i_carry_end, ref_rel, delta_slip, drop_dist=0.08):
    """Object displacement in the gripper frame during the carry relative to its post-grasp position.
    Slip if max displacement > delta_slip, or the object leaves the gripper (distance > drop_dist)."""
    rel = rec[i_carry_start:i_carry_end, [_col("rel_x"), _col("rel_y"), _col("rel_z")]]
    disp = np.linalg.norm(rel - ref_rel, axis=1)
    dropped = bool((np.linalg.norm(rel, axis=1) > drop_dist).any())
    return {"slip_disp_max": float(disp.max()), "dropped": dropped,
            "slip": bool(disp.max() > delta_slip or dropped)}


def complete(rec, goal_pos, damaged, xy_tol=0.03, z_tol=0.01):
    """Object rests in the goal region at the end (after release) and is not damaged."""
    last = rec[-1]
    op = last[[_col("obj_x"), _col("obj_y"), _col("obj_z")]]
    in_xy = float(np.linalg.norm(op[:2] - goal_pos[:2])) <= xy_tol
    at_z = abs(float(op[2] - goal_pos[2])) <= z_tol
    on_table = bool(last[_col("on_table")] > 0.5)
    return {"final_xy_err": float(np.linalg.norm(op[:2] - goal_pos[:2])), "final_z_err": float(op[2] - goal_pos[2]),
            "complete": bool(in_xy and at_z and on_table and not damaged)}
