# Research note — MuJoCo flex (deformable) bodies for the carry task

Date: 2026-11-02. Owner request of 2026-11-02 (research only, no implementation in the repo). Measured numbers come from a
throw-away probe in the session scratchpad (a 4 × 4 × 6 cm `flexcomp` grid box between two force-driven pads on a
position-servo carrier executing the grasp → lift 0.12 m in 0.5 s → carry 0.30 m in 0.75 s sequence; no Panda arm). The probe
is not repo code and was not kept.

## 1. What MuJoCo offers (3.8.1 installed; latest release 3.17.0, 2026-10-22)

- A flex is "a collection of MuJoCo bodies connected with massless stretchable elements" (capsules / triangles / tetrahedra for
  dim 1 / 2 / 3), created from grids or meshes by `flexcomp`. Two deformation models: an edge-based soft equality constraint
  (edge `stiffness`/`damping`, "permits large time steps") and a continuum model ("each element is in a constant stress state",
  piecewise-linear finite elements, Saint Venant-Kirchhoff, set by `elasticity young="…" poisson="…"`). Contact is generalised to
  elements (two colliding tetrahedra can involve up to 8 bodies); `radius` makes the elements smooth and volumetric.
- `dof="trilinear"` / `"quadratic"` (3.3.0 / 3.4.0): the flex is parametrised by 8 or 27 cell nodes instead of one body per
  vertex; the docs say these "can result in significantly faster simulation times". In 3.8.1 they support multi-cell flexes
  and a `strain` equality type but not the stretch elasticity of full flexes.
- What newer releases change, all relevant to this task: 3.11 solves flex elasticity implicitly inside CG; 3.13 adds the
  `discrete` integrator (flex elasticity and passive flex contact handled implicitly; the docs call it "new (September 2026),
  under active development"); 3.15 adds experimental Stable Neo-Hookean 3D elasticity (requires `discrete`); 3.16 corrects the
  shear contribution of interpolated flexes ("existing models may change behavior"); 3.17 adds self-collision for interpolated
  flexes and per-model arena sizing. Unsupported combinations raise runtime errors: flex elasticity under PGS, flex elasticity
  with `noslip`, interpolated flexes attached to moving bodies under Newton (CG required).
- Deformation read-back: `mjData.flexedge_length` against `mjModel.flexedge_length0` gives per-edge strain directly for
  full flexes; per-element deformation gradients follow from `flexvert_xpos` and `flex_elem`. No stress or strain-energy
  field is exposed; for interpolated flexes the edge lengths are not updated (the probe read NaN strains).

## 2. Probe results (MuJoCo 3.8.1, `implicitfast` + Newton unless stated, dt = 2 ms, pyramidal cone, impratio 20)

Setup: flex box 1 kg, 96 vertices (292 dof) at 4 × 4 × 6 points, vertex radius 3 mm, Poisson 0.2, edge damping 1, contact
`solref` (0.005, 1) as in the rigid sweep; pads 16 × 16 mm, μ = 0.4, force ramped to F over 0.5 s.

| configuration | stable | wall per simulated second | pad gap at the end (box is 40 mm) | object displacement vs carrier after the carry [mm] | edge strain p95 / max during hold |
|---|---|---|---|---|---|
| rigid box (reference, same rig) | yes | 0.008 s | 40.0 | 0 / 0 / −7.5 (settling) | — |
| E = 5 × 10⁴ Pa, F = 15 N | yes | 0.27 s | — | dropped during the carry | 0.23 / 0.66 |
| E = 5 × 10⁴ Pa, F = 30 N | yes | 0.64–0.68 s | 9.3 | −0.5 / −1.0 / −3.6 (held) | 0.30 / 0.91 |
| E = 5 × 10⁴ Pa, F = 60 N | yes | 0.65 s | 7.8 | 0.2 / −0.1 / −6.8 (held) | 0.36 / 0.72 |
| E = 1 × 10⁵ Pa, F = 30 N | yes | 0.59 s | 3.1 | −1.6 / 0.5 / −32.7 (crushed flat, sagged) | 0.16 / 0.43 |
| E = 1 × 10⁵ Pa, F = 30 N, dt = 1 ms | yes | 1.04 s | 15.9 | −0.4 / 1.4 / −8.9 (held) | 0.15 / 0.42 |
| E = 2 × 10⁵ Pa, F = 30 N | **no** (blows up at t = 0.03 s) | — | — | — | — |
| E = 5 × 10⁵ Pa, any F, dt 2 or 1 ms, Newton or CG, solref 0.005 or 0.02 | **no** | — | — | — | — |
| E = 5 × 10⁶ Pa | **no** | — | — | — | — |
| E = 5 × 10⁴ Pa, F = 30 N, solref (0.02, 1) | yes | 0.74 s | 9.4 | −0.7 / 2.2 / −3.0 (held) | 0.25 / 0.91 |
| E = 5 × 10⁴ Pa, F = 30 N, dt = 1 ms | yes | 0.69 s | pads crossed | dropped | 0.31 / 0.74 |
| E = 5 × 10⁴ Pa, F = 30 N, CG solver | yes | 0.40 s | pads crossed | dropped | 0.29 / 0.79 |
| E = 5 × 10⁴ Pa, F = 30 N, 3 × 3 × 4 points, radius 5 mm | yes | 0.10 s | pads crossed | dropped | 0.17 / 0.31 |
| E = 5 × 10⁴ Pa, F = 60 N, dt = 1 ms | yes | 1.25 s | 7.5 | 0.9 / 0.1 / −3.5 (held) | 0.40 / 0.92 |
| trilinear dof, 2 × 2 × 2 or 4 × 4 × 6, any solver | runs | 0.02–0.06 s | pads crossed | not grasped (pads pass through the box) | n/a |

Other observations: any positive Rayleigh `elasticity damping` made the flex blow up within 10 ms, with or without contact;
`implicit` (as opposed to `implicitfast`) was unstable with pads; a 5 × 5 × 8 flex (604 dof) was unstable at E = 5 × 10⁵.

## 3. Answers to the three questions

**Can a gripper grasp and carry a flex object stably?** Only for very soft material in this build. The stable range is
E ≲ 10⁵ Pa (foam); E ≥ 2 × 10⁵ Pa diverges at any tested time step, solver and contact softness. A 1 kg box at 5 × 10⁴ Pa is
squeezed from 40 mm to 8–9 mm at 30 N (p95 edge strain 0.3) and still carried through the 7 m/s² lateral move with
≤ 4 mm displacement, but the same physical parameters drop the object when the time step is halved, when the solver is
changed to CG, or when the mesh is coarsened. A cardboard or plastic box (E ≈ 10⁸–10⁹ Pa) is three to four orders of
magnitude outside the stable range; the task would have to be redefined around sponges, bread or fruit-like objects.
Trilinear flexes, the fast option, did not hold against the pads at all in 3.8.1.

**Is a strain- or deformation-based damage measure available and not dominated by solver parameters?** Available: per-edge
strain from `flexedge_length`, cheap and already logged in the probe. Not solver-independent in this build: at fixed physics
(E = 5 × 10⁴, F = 30 N) the p95 strain moved from 0.30 to 0.25 when `solref` went from 0.005 to 0.02 s (−17 %), and the
grasp outcome itself (held vs dropped) flipped with the time step, the solver and the mesh resolution. The max strain
(0.72–0.92) was not monotone in F (0.91 at 30 N, 0.72 at 60 N). A damage label built on these numbers would carry the
solver settings as hidden parameters; the current sustained-normal-force proxy does not have this problem (P6 passes
outside boundary uncertainty on the rigid sweep).

**Simulation speed vs the current 0.72 s per branch.** The rigid branch costs ≈ 0.12 s of wall time per simulated second
(0.72 s for ≈ 6 s of simulation including the Panda and OSC). The flex box alone adds 0.6–0.7 s per simulated second at
dt = 2 ms (1.0–1.25 s at 1 ms) for 96 vertices, i.e. a branch of ≈ 4.5 s (dt 2 ms) to ≈ 7 s (dt 1 ms): 6–10× slower. The
48-object sweep (≈ 23 500 branches) would take ≈ 3–5 h wall at 12 processes instead of 30 min; the full planned dataset
(§7) scales the same way. Trilinear flexes would be ≈ 1.5× the rigid cost but do not grasp in 3.8.1.

## 4. Estimated prototyping time

1. Upgrade MuJoCo to ≥ 3.15 (`discrete` integrator, implicit flex elasticity, Stable Neo-Hookean) and re-verify robosuite
   1.4 / LIBERO / the existing rigid sweep (P1, P6 reruns on two objects): 1–2 days. robosuite only requires `mujoco ≥ 2.3`,
   but nothing in the venv has been tested past 3.8.1.
2. Flex box inside the robosuite MJCF (`flexcomp` as a child of the object body, pad contact, `activelayers`, arena memory),
   strain read-back in `SAMPLE_COLS`, a damage proxy on p95 strain: 2–3 days.
3. Stability and speed envelope on the new build (E range, dt, solver), repeat of the solver-sensitivity table above, and
   a P6-style repeatability check: 2–3 days.
4. Only then a two-object pilot sweep (≈ 1 h wall each at the measured speed).

Total ≈ 1.5–2 weeks before a pilot, with the outcome of step 3 unknown: the 3.8.1 results say the stiffness range that
grasps stably is foam, not boxes, and the newer integrator is documented as under active development. Recommendation
for the owner: keep rigid bodies with the force-based damage proxy for Phase 1 and the RSS data; treat flex objects as
a separate feasibility item after the MuJoCo upgrade, scoped to soft objects (sponge / fruit) where a strain-based damage
label is physically meaningful.

Sources: MuJoCo documentation (XML reference `flexcomp` / `deformable/flex`, Modeling chapter "Deformable objects",
Computation chapter, changelog 3.2.4–3.17.0, read 2026-11-02 from mujoco.readthedocs.io/en/stable); installed package
`mujoco 3.8.1` (`MjModel.flex_*`, `MjData.flexedge_length`, `flexvert_xpos`); the probe runs listed above.
