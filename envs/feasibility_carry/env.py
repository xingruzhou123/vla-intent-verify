"""FeasibilityCarry environment.

Mechanics (plan.md §3.1, §3.5, §6):
- F: the gripper actuator's `forcerange` upper bound in Newtons. The finger position target is ramped past the joint
  range, so once the pads touch the object the position actuator saturates and the applied finger force is exactly F.
  F is set once at grasp start and held for the whole grasp-and-carry primitive.
- s: time scaling of the carry segment. The carry is a fixed min-jerk Cartesian path (lift, lateral, lower) through
  fixed waypoints; every segment duration is multiplied by s (velocity ∝ 1/s, acceleration ∝ 1/s²). Tracked by
  robosuite's OSC_POSE controller in absolute mode.
- Carry direction: the lateral segment runs along the gripper's closing axis (world +y) — a deliberate design choice
  (plan §3.5): the object's lateral inertial load is then carried by a normal-force difference between the pads.
- Grasp orientation: canonical top-down grasp with vertical pads (closing axis = world y, pad plane = world x–z).
  The Panda home orientation would tilt the pads by ≈ 7° about the closing axis and make the box rock on pad edges.
- Object: parametric box (mass, friction, CoM offset along world x = the horizontal axis orthogonal to closing, set
  through `body_ipos`). Fragility is a label parameter and does not enter the simulation.
- Branch point: full MuJoCo integration state (time, qpos, qvel, act, warm-start, ctrl) at the pre-contact state,
  saved with mj_getState and restored with mj_setState.
- The object's own friction governs the contact (geom priority 1); robosuite's finger pads have μ = 2 and MuJoCo
  would otherwise take the max of the two.
- Recording: every simulation substep writes one row (`SAMPLE_COLS`): per-pad normal force, net tangential force,
  summed tangential magnitude and contact count; finger joints and actuator forces; eef pose and linear velocity;
  object pose; object position relative to the eef and relative to the midpoint of the two pads (eef frame).
  The eef frame has x = closing axis, y = world x, z = down. Finger joints have damping (100 N·s/m), armature and
  1 N friction loss, so under lateral acceleration the finger–object assembly drifts in the hand frame without any pad
  slip; slip is therefore measured relative to the pads (`relpad_*`), not the hand (`rel_*`).
"""
from dataclasses import dataclass, asdict
import numpy as np
import mujoco
from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.placement_samplers import UniformRandomSampler
from robosuite.controllers import load_controller_config
import robosuite.utils.transform_utils as T

PHASES = ("approach", "ramp", "hold", "lift", "lateral", "lower", "settle", "release")
PH = {n: i for i, n in enumerate(PHASES)}

# canonical grasp orientation: eef x = world y (closing), eef y = world x, eef z = −world z (columns are eef axes in world)
GRASP_MAT = np.array([[0.0, 1.0, 0.0],
                      [1.0, 0.0, 0.0],
                      [0.0, 0.0, -1.0]])


@dataclass
class CarryConfig:
    # object (parametric; true values are labels only)
    obj_half_size: tuple = (0.02, 0.02, 0.03)   # m; 4 cm wide (closing axis), 4 cm deep, 6 cm tall
    obj_mass: float = 1.0                        # kg
    obj_mu: float = 0.4                          # sliding friction coefficient (object geom, priority 1)
    obj_solref: tuple = (0.005, 1.0)             # contact (timeconst, dampratio) of the object geom; governs pad–object and
                                                 # table–object contacts (priority 1). robosuite's 0.02 s default lets the held
                                                 # object creep ≈ 4 mm across the pads under the oscillating normal load of a
                                                 # fast carry (soft tangential constraint); 0.005 s reduces this to < 0.8 mm
                                                 # with unchanged force tracking and sustained-peak scores (checked 2026-10-29).
    com_offset_frac: float = 0.0                 # CoM offset along the body x axis (world x, orthogonal to closing), as a fraction of the width 2·half_size[0]
    obj_xy: tuple = (0.0, -0.12)                 # start, relative to table centre
    goal_xy: tuple = (0.0, 0.18)                 # goal, relative to table centre (0.30 m lateral carry along +y)
    # approach (before the branch point)
    approach_hover: float = 0.15                 # m above grasp height
    approach_durations: tuple = (1.5, 1.5)       # s: to hover, descend
    approach_settle: float = 0.5                 # s
    # grasp
    grip_open_force: float = 20.0                # N, forcerange used while opening/approaching
    grip_ramp_speed: float = 0.02                # m/s finger target ramp before contact
    grip_ramp_timeout: float = 3.0               # s
    grip_hold: float = 1.5                       # s with the force bound active before the carry starts
    grip_steady_window: float = 0.5              # s at the end of the hold used for the steady-state measurement
    contact_threshold: float = 0.5               # N per side to declare contact during the ramp
    # carry (nominal durations, multiplied by s) — plan §3.1: 0.12 m / 0.5 s, 0.30 m / 0.75 s, lower / 0.5 s
    lift_height: float = 0.12                    # m
    carry_durations: tuple = (0.5, 0.75, 0.5)    # s: lift, lateral, lower  (T_carry(s0 = 1) = 1.75 s)
    lower_clearance: float = 0.005               # m above the pickup height at the end of lowering
    lower_settle: float = 0.5                    # s hold at the final pose before release (not scaled by s)
    release_settle: float = 0.5                  # s after opening the gripper
    # control
    control_freq: int = 20
    osc_kp: float = 150.0
    pad_half: float = 0.008                      # m, finger-pad half-extent (torsional slip lever arm, plan §3.2)

    def T_carry(self, s=1.0):
        return float(s * sum(self.carry_durations))


def _minjerk(t, T):
    """Min-jerk progress in [0,1] for t in [0,T]."""
    x = np.clip(t / T, 0.0, 1.0)
    return 10 * x**3 - 15 * x**4 + 6 * x**5


class FeasibilityCarry(SingleArmEnv):
    """Single Panda, table, one parametric box, fixed carry path."""

    SAMPLE_COLS = (["t", "phase", "f_left", "f_right", "ft_left_y", "ft_left_z", "ft_right_y", "ft_right_z",
                    "fts_left", "fts_right", "nc_left", "nc_right", "on_table", "q1", "q2", "act_f1", "act_f2"]
                   + [f"eef_{c}" for c in "xyz"] + [f"eef_q{c}" for c in "wxyz"] + [f"eef_v{c}" for c in "xyz"]
                   + [f"obj_{c}" for c in "xyz"] + [f"obj_q{c}" for c in "wxyz"]
                   + [f"rel_{c}" for c in "xyz"] + [f"relpad_{c}" for c in "xyz"]
                   + [f"padL_{c}" for c in "xyz"] + [f"padR_{c}" for c in "xyz"])

    def __init__(self, cfg: CarryConfig = None, seed: int = 0):
        self.cfg = cfg or CarryConfig()
        self.table_full_size = (0.8, 0.8, 0.05)
        self.table_offset = np.array((0.0, 0.0, 0.8))
        self.rng = np.random.default_rng(seed)
        ctrl = load_controller_config(default_controller="OSC_POSE")
        ctrl["control_delta"] = False
        ctrl["kp"] = self.cfg.osc_kp
        self._grip_mode = "open"      # open | ramp | clamp
        self._grip_t0 = 0.0
        self._grip_F = self.cfg.grip_open_force
        self._phase = PH["approach"]
        self._perturb = {"dxy": np.zeros(2), "dyaw": 0.0, "mu_scale": 1.0}
        super().__init__(
            robots="Panda",
            controller_configs=ctrl,
            initialization_noise=None,
            use_camera_obs=False,
            has_renderer=False,
            has_offscreen_renderer=False,
            control_freq=self.cfg.control_freq,
            horizon=10**9,
            ignore_done=True,
            hard_reset=True,
        )
        self._n_substeps = int(self.control_timestep / self.model_timestep)
        self._f6 = np.zeros(6)
        self._v6 = np.zeros(6)

    # ------------------------------------------------------------------ model
    def _load_model(self):
        super()._load_model()
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)
        arena = TableArena(table_full_size=self.table_full_size, table_friction=(1, 0.005, 0.0001),
                           table_offset=self.table_offset)
        arena.set_origin([0, 0, 0])
        hs = np.array(self.cfg.obj_half_size)
        volume = float(np.prod(2 * hs))
        self.obj = BoxObject(name="obj", size=list(hs), density=self.cfg.obj_mass / volume,
                             friction=[self.cfg.obj_mu, 0.005, 0.0001], rgba=[0.2, 0.5, 0.9, 1])
        x, y = self.cfg.obj_xy
        dx, dy = self._perturb["dxy"]
        self.placement_initializer = UniformRandomSampler(
            name="ObjectSampler", mujoco_objects=self.obj,
            x_range=[x + dx, x + dx], y_range=[y + dy, y + dy],
            rotation=self._perturb["dyaw"], rotation_axis="z",
            ensure_object_boundary_in_range=False, ensure_valid_placement=False,
            reference_pos=self.table_offset, z_offset=0.001)
        self.model = ManipulationTask(mujoco_arena=arena,
                                      mujoco_robots=[r.robot_model for r in self.robots],
                                      mujoco_objects=self.obj)

    def _setup_references(self):
        super()._setup_references()
        m = self.sim.model
        self.obj_bid = m.body_name2id(self.obj.root_body)
        self.obj_gids = [m.geom_name2id(g) for g in self.obj.contact_geoms]
        r = self.robots[0]
        g = r.gripper.important_geoms
        self.left_gids = [m.geom_name2id(n) for n in g["left_finger"]]
        self.right_gids = [m.geom_name2id(n) for n in g["right_finger"]]
        self.pad_gids = [m.geom_name2id(r.gripper.naming_prefix + "finger1_pad_collision"),
                         m.geom_name2id(r.gripper.naming_prefix + "finger2_pad_collision")]
        self.table_gid = m.geom_name2id("table_collision")
        self.finger_qpos_idx = list(r._ref_gripper_joint_pos_indexes)
        self.grip_act_idx = list(r._ref_joint_gripper_actuator_indexes)
        self.eef_sid = m.site_name2id(r.gripper.important_sites["grip_site"])
        self._obj_gids_set = set(self.obj_gids)
        self._left_set, self._right_set = set(self.left_gids), set(self.right_gids)
        self._apply_model_params()

    def _apply_model_params(self):
        """Model-level settings that a hard reset would discard."""
        m = self.sim.model
        for gid in self.obj_gids:
            m.geom_priority[gid] = 1                      # object μ governs the finger–object contact
            m.geom_friction[gid, 0] = self.cfg.obj_mu * self._perturb["mu_scale"]
            m.geom_solref[gid] = list(self.cfg.obj_solref)
        d = self.cfg.com_offset_frac * 2.0 * self.cfg.obj_half_size[0]
        m.body_ipos[self.obj_bid] = [d, 0.0, 0.0]        # CoM offset along body x (≈ world x; orthogonal to closing)
        # the compiler precomputed these flags for ipos = 0; with them set MuJoCo skips the inertial-frame offset
        # (xipos = xpos) and uses the diagonal mass-matrix shortcut, so the offset would have no effect at all
        m.body_sameframe[self.obj_bid] = 0
        m.body_simple[self.obj_bid] = 0
        for a in self.grip_act_idx:
            m.actuator_ctrlrange[a] = [-0.3, 0.3]          # allow targets past the joint range (force saturation)
        self.set_grip_force(self._grip_F)

    def _reset_internal(self):
        super()._reset_internal()
        placements = self.placement_initializer.sample()
        for pos, quat, obj in placements.values():
            self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(pos), np.array(quat)]))
        self._grip_mode = "open"
        self._grip_F = self.cfg.grip_open_force
        self._phase = PH["approach"]
        self._apply_model_params()

    # ---------------------------------------------------------------- gripper
    def set_grip_force(self, F):
        self._grip_F = float(F)
        for a in self.grip_act_idx:
            self.sim.model.actuator_forcerange[a] = [-self._grip_F, self._grip_F]

    def _grip_targets(self):
        """Finger position targets (finger1 ∈ [0,0.04] open=0.04; finger2 mirrored)."""
        if self._grip_mode == "open":
            u = 0.04
        elif self._grip_mode == "ramp":
            u = 0.04 - self.cfg.grip_ramp_speed * (self.sim.data.time - self._grip_t0)
            u = max(u, -0.3)
        else:  # clamp: target far past closed → actuator saturates at ±F
            u = -0.3
        return np.array([u, -u])

    def _pre_action(self, action, policy_step=False):
        super()._pre_action(action, policy_step)
        self.sim.data.ctrl[self.grip_act_idx] = self._grip_targets()

    # ------------------------------------------------------------ measurement
    def pad_forces(self, em=None):
        """Per-pad contact load on the object: (normal, net tangential vector in the eef pad plane (y, z), Σ|tangential|,
        n_contacts) for left and right, plus the object–table contact flag. Forces from mj_contactForce, mapped to the
        world frame and split along the contact normal (constraint forces, not penetrations)."""
        d, m = self.sim.data._data, self.sim.model._model
        if em is None:
            em = np.array(d.site_xmat[self.eef_sid]).reshape(3, 3)
        acc = {"l": [0.0, np.zeros(3), 0.0, 0], "r": [0.0, np.zeros(3), 0.0, 0]}
        on_table = False
        f6 = self._f6
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = c.geom1, c.geom2
            if g1 in self._obj_gids_set:
                other = g2
            elif g2 in self._obj_gids_set:
                other = g1
            else:
                continue
            if other == self.table_gid:
                on_table = True
                continue
            side = "l" if other in self._left_set else ("r" if other in self._right_set else None)
            if side is None:
                continue
            mujoco.mj_contactForce(m, d, i, f6)
            frame = np.asarray(c.frame).reshape(3, 3)       # rows: normal, tangent1, tangent2 (world)
            fw = frame.T @ f6[:3]
            n = frame[0]
            fn = float(fw @ n)
            ft = fw - fn * n
            a = acc[side]
            a[0] += abs(fn); a[1] += ft; a[2] += float(np.linalg.norm(ft)); a[3] += 1
        out = []
        for side in ("l", "r"):
            a = acc[side]
            ft_eef = em.T @ a[1]
            out.append((a[0], ft_eef[1:3], a[2], a[3]))
        return out[0], out[1], on_table

    def eef_pose(self):
        d = self.sim.data
        pos = np.array(d.site_xpos[self.eef_sid])
        mat = np.array(d.site_xmat[self.eef_sid]).reshape(3, 3)
        return pos, mat

    def eef_lin_vel(self):
        mujoco.mj_objectVelocity(self.sim.model._model, self.sim.data._data, mujoco.mjtObj.mjOBJ_SITE,
                                 self.eef_sid, self._v6, 0)
        return np.array(self._v6[3:])

    def obj_pose(self):
        d = self.sim.data
        return np.array(d.body_xpos[self.obj_bid]), np.array(d.body_xquat[self.obj_bid])

    def pad_midpoint(self):
        d = self.sim.data
        return 0.5 * (np.array(d.geom_xpos[self.pad_gids[0]]) + np.array(d.geom_xpos[self.pad_gids[1]]))

    def sample(self):
        """One recorded row at the simulation rate."""
        ep, em = self.eef_pose()
        (fl, ftl, ftsl, ncl), (fr, ftr, ftsr, ncr), on_table = self.pad_forces(em)
        q = self.sim.data.qpos[self.finger_qpos_idx]
        af = self.sim.data.actuator_force[self.grip_act_idx]
        eq = T.convert_quat(T.mat2quat(em), to="wxyz")
        ev = self.eef_lin_vel()
        op, oq = self.obj_pose()
        d = self.sim.data
        pl, pr = np.array(d.geom_xpos[self.pad_gids[0]]), np.array(d.geom_xpos[self.pad_gids[1]])
        rel = em.T @ (op - ep)
        relpad = em.T @ (op - 0.5 * (pl + pr))
        return np.concatenate([[self.sim.data.time, self._phase, fl, fr], ftl, ftr, [ftsl, ftsr, ncl, ncr, float(on_table)],
                               q, af, ep, eq, ev, op, oq, rel, relpad, pl, pr])

    # --------------------------------------------------------------- stepping
    def run_step(self, pos, mat, rec=None):
        """One policy step toward absolute eef pose (pos, mat); records every substep into rec (list)."""
        aa = T.quat2axisangle(T.mat2quat(mat))
        action = np.concatenate([pos, aa, [0.0]])          # gripper slot ignored; ctrl set in _pre_action
        policy_step = True
        for _ in range(self._n_substeps):
            self.sim.forward()
            self._pre_action(action, policy_step)
            self.sim.step()
            policy_step = False
            if rec is not None:
                rec.append(self.sample())
            if self._grip_mode == "ramp":
                (fl, *_), (fr, *_), _ = self.pad_forces()
                if (fl > self.cfg.contact_threshold and fr > self.cfg.contact_threshold) or \
                        self.sim.data.time - self._grip_t0 > self.cfg.grip_ramp_timeout:
                    self._grip_mode = "clamp"
                    self._phase = PH["hold"]
                    self._grip_contact_time = self.sim.data.time
        self.timestep += 1
        self.cur_time += self.control_timestep

    def move_minjerk(self, p_from, p_to, mat, duration, rec=None):
        n = max(1, int(round(duration * self.cfg.control_freq)))
        for k in range(1, n + 1):
            p = p_from + (p_to - p_from) * _minjerk(k / self.cfg.control_freq, duration)
            self.run_step(p, mat, rec)

    def hold(self, pos, mat, duration, rec=None):
        for _ in range(int(round(duration * self.cfg.control_freq))):
            self.run_step(pos, mat, rec)

    # ------------------------------------------------------------- branching
    def save_state(self):
        m, d = self.sim.model._model, self.sim.data._data
        spec = mujoco.mjtState.mjSTATE_INTEGRATION
        buf = np.empty(mujoco.mj_stateSize(m, spec))
        mujoco.mj_getState(m, d, buf, spec)
        return {"state": buf, "timestep": self.timestep, "cur_time": self.cur_time,
                "grip_mode": self._grip_mode, "grip_F": self._grip_F}

    def restore_state(self, saved):
        m, d = self.sim.model._model, self.sim.data._data
        mujoco.mj_setState(m, d, np.asarray(saved["state"], dtype=float), mujoco.mjtState.mjSTATE_INTEGRATION)
        mujoco.mj_forward(m, d)
        self.timestep, self.cur_time = int(saved["timestep"]), float(saved["cur_time"])
        self._grip_mode = str(saved["grip_mode"])
        self.set_grip_force(float(saved["grip_F"]))
        self._phase = PH["approach"]
        ctl = self.robots[0].controller
        ctl.update(force=True)
        ctl.reset_goal()

    # --------------------------------------------------------------- phases
    def set_perturbation(self, dxy=(0.0, 0.0), dyaw=0.0, mu_scale=1.0):
        self._perturb = {"dxy": np.asarray(dxy, float), "dyaw": float(dyaw), "mu_scale": float(mu_scale)}

    def grasp_height(self):
        return self.table_offset[2] + self.cfg.obj_half_size[2]

    def approach(self, rec=None):
        """From the home pose: hover above the object, descend with the gripper open in the canonical grasp
        orientation, settle. Ends at the branch point."""
        self._grip_mode = "open"
        self._phase = PH["approach"]
        self.set_grip_force(self.cfg.grip_open_force)
        p0, _ = self.eef_pose()
        mat = GRASP_MAT
        op, _ = self.obj_pose()
        zg = self.grasp_height()
        hover = np.array([op[0], op[1], zg + self.cfg.approach_hover])
        grasp = np.array([op[0], op[1], zg])
        self.move_minjerk(p0, hover, mat, self.cfg.approach_durations[0], rec)
        self.move_minjerk(hover, grasp, mat, self.cfg.approach_durations[1], rec)
        self.hold(grasp, mat, self.cfg.approach_settle, rec)
        return grasp, mat

    def closing_axis_world(self):
        d = self.sim.data
        v = np.array(d.geom_xpos[self.pad_gids[1]]) - np.array(d.geom_xpos[self.pad_gids[0]])
        return v / np.linalg.norm(v)

    def grasp(self, F, rec):
        """Close with force bound F; ramp until contact, then clamp and hold. Returns steady-state stats."""
        pos, mat = self.eef_pose()
        self.set_grip_force(F)
        self._grip_mode = "ramp"
        self._phase = PH["ramp"]
        self._grip_t0 = self.sim.data.time
        self._grip_contact_time = None
        n0 = len(rec)
        while self._grip_mode == "ramp":
            self.run_step(pos, mat, rec)
        self.hold(pos, mat, self.cfg.grip_hold, rec)
        arr = np.array(rec[n0:])
        c = self.SAMPLE_COLS.index
        w = arr[arr[:, 0] >= arr[-1, 0] - self.cfg.grip_steady_window]
        return {
            "contact_time": None if self._grip_contact_time is None else self._grip_contact_time - self._grip_t0,
            "ss_f_left": float(w[:, c("f_left")].mean()), "ss_f_right": float(w[:, c("f_right")].mean()),
            "ss_f_left_std": float(w[:, c("f_left")].std()), "ss_f_right_std": float(w[:, c("f_right")].std()),
            "ss_ft_left": float(np.linalg.norm(w[:, [c("ft_left_y"), c("ft_left_z")]], axis=1).mean()),
            "ss_ft_right": float(np.linalg.norm(w[:, [c("ft_right_y"), c("ft_right_z")]], axis=1).mean()),
            "ss_nc_left": float(w[:, c("nc_left")].mean()), "ss_nc_right": float(w[:, c("nc_right")].mean()),
            "ss_q1": float(w[:, c("q1")].mean()), "ss_q2": float(w[:, c("q2")].mean()),
            "ss_act_f1": float(w[:, c("act_f1")].mean()), "ss_act_f2": float(w[:, c("act_f2")].mean()),
        }

    def carry(self, s, rec):
        """Fixed path (lift, lateral along +y = closing axis, lower) with all durations scaled by s. Gripper stays clamped."""
        p0, mat = self.eef_pose()
        mat = GRASP_MAT
        gx, gy = self.cfg.goal_xy
        top = p0 + np.array([0, 0, self.cfg.lift_height])
        over = np.array([self.table_offset[0] + gx, self.table_offset[1] + gy, top[2]])
        down = np.array([over[0], over[1], p0[2] + self.cfg.lower_clearance])
        d = [x * s for x in self.cfg.carry_durations]
        self._phase = PH["lift"];    self.move_minjerk(p0, top, mat, d[0], rec)
        self._phase = PH["lateral"]; self.move_minjerk(top, over, mat, d[1], rec)
        self._phase = PH["lower"];   self.move_minjerk(over, down, mat, d[2], rec)
        self._phase = PH["settle"];  self.hold(down, mat, self.cfg.lower_settle, rec)   # not scaled by s
        return down, mat

    def release(self, pos, mat, rec):
        self._grip_mode = "open"
        self._phase = PH["release"]
        self.set_grip_force(self.cfg.grip_open_force)
        self.hold(pos, mat, self.cfg.release_settle, rec)

    def goal_pos(self):
        return np.array([self.table_offset[0] + self.cfg.goal_xy[0], self.table_offset[1] + self.cfg.goal_xy[1],
                         self.table_offset[2] + self.cfg.obj_half_size[2]])

    def config_dict(self):
        return asdict(self.cfg)
