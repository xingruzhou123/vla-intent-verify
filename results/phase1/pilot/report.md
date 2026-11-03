# Phase-1 pilot report — results/phase1/pilot

Object: half-size [0.02, 0.02, 0.03] m, mass 0.3 kg, μ 0.4. Provisional tolerances: f_tol 40.0 N, D_max 5.0 N·s, δ_slip 10 mm. Solver: {'timestep': 0.002, 'noslip_iterations': 0, 'impratio': 20.0, 'cone': 1, 'integrator': 0}. Gripper kp 1000.0.

## P1 force tracking

measured = 0.9998·F + -0.007 N, R² = 1.0000 (n = 48 pad measurements over 24 branches). Figure: p1_force_tracking.png

| F [N] | mean pad force [N] (L / R) | sd within hold [N] | finger1 pos [mm] | actuator force [N] | contact time [s] |
|---|---|---|---|---|---|
| 8 | 7.95 / 8.04 | 0.00 | 19.64 | -8.00 | 1.15 |
| 15 | 14.94 / 15.04 | 0.01 | 19.50 | -15.00 | 1.15 |
| 30 | 29.93 / 30.03 | 0.05 | 19.30 | -30.00 | 1.15 |
| 60 | 59.94 / 60.04 | 0.01 | 18.96 | -60.00 | 1.15 |

## Outcomes per (F, s) — repeats listed as r0 / r1 …

| F [N] | s | slip disp max [mm] | peak force [N] | D [N·s] | slip | damage (primary / secondary) | complete | final xy err [mm] |
|---|---|---|---|---|---|---|---|---|
| 8 | 0.5 | 3.5 / 3.5 | 8.5 / 8.5 | 0.00 / 0.00 | 0 / 0 | 0 / 0 / 0 / 0 | 1 / 1 | 3 / 3 |
| 8 | 1 | 2.0 / 2.0 | 8.4 / 8.3 | 0.00 / 0.00 | 0 / 0 | 0 / 0 / 0 / 0 | 1 / 1 | 2 / 2 |
| 8 | 2 | 1.1 / 1.1 | 8.1 / 8.1 | 0.00 / 0.00 | 0 / 0 | 0 / 0 / 0 / 0 | 1 / 1 | 0 / 0 |
| 15 | 0.5 | 3.4 / 3.5 | 15.5 / 15.5 | 0.00 / 0.00 | 0 / 0 | 0 / 0 / 0 / 0 | 1 / 1 | 3 / 3 |
| 15 | 1 | 2.0 / 2.0 | 15.3 / 15.4 | 0.00 / 0.00 | 0 / 0 | 0 / 0 / 0 / 0 | 1 / 1 | 2 / 2 |
| 15 | 2 | 1.1 / 1.1 | 15.1 / 15.1 | 0.00 / 0.00 | 0 / 0 | 0 / 0 / 0 / 0 | 1 / 1 | 0 / 0 |
| 30 | 0.5 | 3.4 / 3.4 | 30.6 / 30.7 | 0.00 / 0.00 | 0 / 0 | 0 / 0 / 0 / 0 | 1 / 1 | 3 / 3 |
| 30 | 1 | 2.0 / 2.0 | 30.7 / 30.7 | 0.00 / 0.00 | 0 / 0 | 0 / 0 / 0 / 0 | 1 / 1 | 2 / 2 |
| 30 | 2 | 1.1 / 1.1 | 30.6 / 30.7 | 0.00 / 0.00 | 0 / 0 | 0 / 0 / 0 / 0 | 1 / 1 | 0 / 0 |
| 60 | 0.5 | 3.4 / 3.4 | 61.2 / 61.1 | 76.45 / 75.49 | 0 / 0 | 1 / 1 / 1 / 1 | 0 / 0 | 3 / 3 |
| 60 | 1 | 2.0 / 2.0 | 61.2 / 61.1 | 111.27 / 110.30 | 0 / 0 | 1 / 1 / 1 / 1 | 0 / 0 | 2 / 2 |
| 60 | 2 | 1.0 / 1.1 | 61.2 / 61.1 | 181.14 / 180.19 | 0 / 0 | 1 / 1 / 1 / 1 | 0 / 0 | 0 / 0 |

## Wall-clock

per branch: mean 0.82 s, min 0.56 s, max 1.20 s (simulated 7.8 s per branch on average; 9.5× real time). Full plan sweep (8 × 6 × 5 × 12 = 2880 branches) ≈ 39 min at this rate.
