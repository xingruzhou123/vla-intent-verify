# Smoke test — probes and branch/restore mechanism

Objects ['light_normal_balanced_0', 'heavy_slippery_balanced_0'] (+ boundary object medium_normal_balanced_1), stored CFD states [0, 1, 2], 3 repeats per (state, probe); total wall 14.7 min; machine load average at the end 24.6. Label version 3.

## Checks

| check | result |
|---|---|
| 1. restore exact (state fingerprint after probe + restore == before) | 126 / 126 branches |
| 2. re-executions from the same state bit-identical (record hash) | 39 / 39 (state, probe) groups |
| 3. outcomes recorded (ground truth + noisy observation) | 117 probe branches |
| 5. C1 replay of the CFD grid cell reproduces the stored sweep record | max abs diff over 9 replays = 0.0; labels match 9 / 9 |
| 5b. the same replay without re-applying the repeat's friction scale | max abs diff = 77.4 over 6 replays (r > 0) |
| 6. C3 probe-then-carry branches run and labelled | 9 |

## Timing per branch (wall seconds; machine shared with other simulations)

| branch type | n | mean | min | max | sim s |
|---|---|---|---|---|---|
| lift probe | 81 | 6.32 | 1.18 | 11.69 | 5.40 |
| squeeze probe | 36 | 3.60 | 0.66 | 7.39 | 3.15 |
| CFD carry replay (C1 follow-up) | 9 | 5.67 | 3.11 | 8.53 | 5.40 |
| probe + carry (C3) | 9 | 13.87 | 9.48 | 17.41 | 10.80 |

## Probe outcomes (ground truth, repeat 0 of each state; identical across repeats)

| object | state | probe | held | slip max [mm] | z gain [mm] | pad load [N] | grip L/R [N] | N50 [N] | damage fragile / rigid (object's f_tol) | obj shift after [mm] | obs: held_hat, slip_hat [mm], load_hat [N] |
|---|---|---|---|---|---|---|---|---|---|---|---|
| light_normal_balanced_0 | 0 | lift_F7.29_h1cm | 1 | 0.0 | 9.4 | 5.1 | 7.3/7.3 | 7.6 | 0 / 0 (18.7 / 47.8) | 0.4 | 1, 0.2, 5.4 |
| light_normal_balanced_0 | 0 | lift_F10.628_h1cm | 1 | 0.0 | 9.4 | 5.1 | 10.6/10.6 | 11.0 | 0 / 0 (18.7 / 47.8) | 0.4 | 1, 0.1, 5.2 |
| light_normal_balanced_0 | 0 | lift_F15.494_h1cm | 1 | 0.0 | 9.4 | 5.1 | 15.5/15.5 | 15.8 | 0 / 0 (18.7 / 47.8) | 0.4 | 1, 0.1, 4.7 |
| light_normal_balanced_0 | 0 | lift_F15.494_h2cm | 1 | 0.0 | 19.4 | 5.1 | 15.5/15.5 | 15.8 | 0 / 0 (18.7 / 47.8) | 0.4 | 1, 0.4, 4.2 |
| light_normal_balanced_0 | 0 | squeeze_F10.628 | – | – | – | – | 10.6/10.6 | 10.6 | 0 / 0 (18.7 / 47.8) | 0.1 | – |
| light_normal_balanced_0 | 0 | squeeze_F15.494 | – | – | – | – | 15.5/15.5 | 15.5 | 0 / 0 (18.7 / 47.8) | 0.1 | – |
| light_normal_balanced_0 | 1 | lift_F7.29_h1cm | 1 | 0.3 | 9.2 | 5.1 | 7.3/7.2 | 7.6 | 0 / 0 (18.7 / 47.8) | 0.4 | 1, 0.2, 5.3 |
| light_normal_balanced_0 | 1 | lift_F10.628_h1cm | 1 | 0.2 | 9.3 | 5.1 | 10.6/10.6 | 11.0 | 0 / 0 (18.7 / 47.8) | 0.3 | 1, 0.8, 4.6 |
| light_normal_balanced_0 | 1 | lift_F15.494_h1cm | 1 | 0.1 | 9.3 | 5.1 | 15.4/15.4 | 15.9 | 0 / 0 (18.7 / 47.8) | 0.3 | 1, 0.2, 5.7 |
| light_normal_balanced_0 | 1 | lift_F15.494_h2cm | 1 | 0.1 | 19.3 | 5.1 | 15.4/15.4 | 15.9 | 0 / 0 (18.7 / 47.8) | 0.3 | 1, 0.1, 6.2 |
| light_normal_balanced_0 | 1 | squeeze_F10.628 | – | – | – | – | 10.6/10.6 | 10.6 | 0 / 0 (18.7 / 47.8) | 0.1 | – |
| light_normal_balanced_0 | 1 | squeeze_F15.494 | – | – | – | – | 15.4/15.4 | 15.4 | 0 / 0 (18.7 / 47.8) | 0.1 | – |
| light_normal_balanced_0 | 2 | lift_F7.29_h1cm | 1 | 0.7 | 8.8 | 5.1 | 7.2/7.2 | 7.6 | 0 / 0 (18.7 / 47.8) | 0.4 | 1, 1.1, 4.5 |
| light_normal_balanced_0 | 2 | lift_F10.628_h1cm | 1 | 0.4 | 9.0 | 5.1 | 10.5/10.5 | 11.1 | 0 / 0 (18.7 / 47.8) | 0.3 | 1, 0.6, 5.6 |
| light_normal_balanced_0 | 2 | lift_F15.494_h1cm | 1 | 0.3 | 9.2 | 5.1 | 15.3/15.3 | 16.0 | 0 / 0 (18.7 / 47.8) | 0.3 | 1, 0.4, 7.1 |
| light_normal_balanced_0 | 2 | lift_F15.494_h2cm | 1 | 0.2 | 19.2 | 5.1 | 15.3/15.3 | 16.0 | 0 / 0 (18.7 / 47.8) | 0.3 | 1, 1.5, 6.2 |
| light_normal_balanced_0 | 2 | squeeze_F10.628 | – | – | – | – | 10.5/10.5 | 10.5 | 0 / 0 (18.7 / 47.8) | 0.1 | – |
| light_normal_balanced_0 | 2 | squeeze_F15.494 | – | – | – | – | 15.3/15.3 | 15.3 | 0 / 0 (18.7 / 47.8) | 0.1 | – |
| heavy_slippery_balanced_0 | 0 | lift_F7.29_h1cm | 0 | 9.7 | 0.0 | 2.8 | 7.3/7.3 | 7.5 | 0 / 0 (23.8 / 55.5) | 0.2 | 0, 10.3, 3.2 |
| heavy_slippery_balanced_0 | 0 | lift_F10.628_h1cm | 0 | 9.5 | 0.0 | 4.1 | 10.6/10.6 | 10.8 | 0 / 0 (23.8 / 55.5) | 0.0 | 0, 9.1, 3.9 |
| heavy_slippery_balanced_0 | 0 | lift_F15.494_h1cm | 0 | 9.4 | 0.0 | 6.0 | 15.5/15.5 | 15.9 | 0 / 0 (23.8 / 55.5) | 0.5 | 0, 9.0, 6.9 |
| heavy_slippery_balanced_0 | 0 | lift_F15.494_h2cm | 0 | 19.4 | 0.0 | 5.9 | 15.5/15.5 | 15.9 | 0 / 0 (23.8 / 55.5) | 0.1 | 0, 18.0, 6.0 |
| heavy_slippery_balanced_0 | 0 | squeeze_F10.628 | – | – | – | – | 10.6/10.6 | 10.7 | 0 / 0 (23.8 / 55.5) | 0.1 | – |
| heavy_slippery_balanced_0 | 0 | squeeze_F15.494 | – | – | – | – | 15.5/15.5 | 15.5 | 0 / 0 (23.8 / 55.5) | 0.1 | – |
| heavy_slippery_balanced_0 | 1 | lift_F7.29_h1cm | 0 | 9.7 | 0.0 | 2.8 | 7.3/7.3 | 7.5 | 0 / 0 (23.8 / 55.5) | 0.2 | 0, 9.4, 3.3 |
| heavy_slippery_balanced_0 | 1 | lift_F10.628_h1cm | 0 | 9.5 | 0.0 | 4.1 | 10.6/10.6 | 10.9 | 0 / 0 (23.8 / 55.5) | 0.2 | 0, 9.7, 4.8 |
| heavy_slippery_balanced_0 | 1 | lift_F15.494_h1cm | 0 | 9.4 | 0.0 | 6.0 | 15.5/15.5 | 15.9 | 0 / 0 (23.8 / 55.5) | 0.5 | 0, 10.0, 6.5 |
| heavy_slippery_balanced_0 | 1 | lift_F15.494_h2cm | 0 | 19.4 | 0.0 | 5.9 | 15.5/15.5 | 15.9 | 0 / 0 (23.8 / 55.5) | 0.4 | 0, 18.8, 5.5 |
| heavy_slippery_balanced_0 | 1 | squeeze_F10.628 | – | – | – | – | 10.6/10.6 | 10.7 | 0 / 0 (23.8 / 55.5) | 0.1 | – |
| heavy_slippery_balanced_0 | 1 | squeeze_F15.494 | – | – | – | – | 15.5/15.5 | 15.5 | 0 / 0 (23.8 / 55.5) | 0.1 | – |
| heavy_slippery_balanced_0 | 2 | lift_F7.29_h1cm | 0 | 9.7 | 0.0 | 2.9 | 7.2/7.2 | 7.5 | 0 / 0 (23.8 / 55.5) | 0.1 | 0, 10.3, 3.7 |
| heavy_slippery_balanced_0 | 2 | lift_F10.628_h1cm | 0 | 9.5 | 0.0 | 4.2 | 10.6/10.6 | 10.9 | 0 / 0 (23.8 / 55.5) | 0.2 | 0, 9.2, 4.8 |
| heavy_slippery_balanced_0 | 2 | lift_F15.494_h1cm | 0 | 9.3 | 0.0 | 6.1 | 15.5/15.5 | 15.9 | 0 / 0 (23.8 / 55.5) | 0.3 | 0, 9.5, 6.4 |
| heavy_slippery_balanced_0 | 2 | lift_F15.494_h2cm | 0 | 19.3 | 0.0 | 6.1 | 15.5/15.5 | 15.9 | 0 / 0 (23.8 / 55.5) | 0.4 | 0, 19.4, 6.0 |
| heavy_slippery_balanced_0 | 2 | squeeze_F10.628 | – | – | – | – | 10.6/10.6 | 10.7 | 0 / 0 (23.8 / 55.5) | 0.1 | – |
| heavy_slippery_balanced_0 | 2 | squeeze_F15.494 | – | – | – | – | 15.5/15.5 | 15.5 | 0 / 0 (23.8 / 55.5) | 0.1 | – |
| medium_normal_balanced_1 | 0 | lift_F10.628_h1cm_boundary | 1 | 0.0 | 9.0 | 8.8 | 10.6/10.6 | 11.2 | 0 / 0 (17.0 / 45.8) | 0.8 | 1, 0.5, 8.0 |
| medium_normal_balanced_1 | 1 | lift_F10.628_h1cm_boundary | 1 | 0.5 | 8.5 | 8.8 | 10.6/10.6 | 11.1 | 0 / 0 (17.0 / 45.8) | 0.8 | 1, 0.9, 9.2 |
| medium_normal_balanced_1 | 2 | lift_F10.628_h1cm_boundary | 1 | 0.9 | 8.2 | 8.8 | 10.5/10.5 | 11.0 | 0 / 0 (17.0 / 45.8) | 0.7 | 1, 1.4, 9.0 |

## Observation noise across repeats (same physics, different sensor draw)

| object | state | probe | load_hat per repeat [N] | slip_hat per repeat [mm] |
|---|---|---|---|---|
| light_normal_balanced_0 | 0 | lift_F7.29_h1cm | 5.40, 5.10, 5.16 | 0.20, 1.03, 0.52 |
| light_normal_balanced_0 | 0 | lift_F10.628_h1cm | 5.20, 4.25, 4.03 | 0.11, 0.43, 0.30 |
| light_normal_balanced_0 | 0 | lift_F15.494_h1cm | 4.67, 3.54, 4.52 | 0.09, 0.05, 1.05 |
| light_normal_balanced_0 | 0 | lift_F15.494_h2cm | 4.18, 4.57, 5.81 | 0.41, 0.33, 1.11 |
| light_normal_balanced_0 | 1 | lift_F7.29_h1cm | 5.35, 5.48, 6.07 | 0.21, 0.95, 1.01 |
| light_normal_balanced_0 | 1 | lift_F10.628_h1cm | 4.60, 5.27, 3.36 | 0.81, 0.03, 0.33 |
| light_normal_balanced_0 | 1 | lift_F15.494_h1cm | 5.69, 4.42, 5.30 | 0.24, 0.30, 0.60 |
| light_normal_balanced_0 | 1 | lift_F15.494_h2cm | 6.17, 5.22, 5.87 | 0.09, 0.10, 0.26 |
| light_normal_balanced_0 | 2 | lift_F7.29_h1cm | 4.47, 5.32, 5.42 | 1.05, 0.35, 0.64 |
| light_normal_balanced_0 | 2 | lift_F10.628_h1cm | 5.63, 5.74, 4.35 | 0.64, 0.43, 0.63 |
| light_normal_balanced_0 | 2 | lift_F15.494_h1cm | 7.14, 5.13, 6.56 | 0.43, 0.02, 0.12 |
| light_normal_balanced_0 | 2 | lift_F15.494_h2cm | 6.16, 4.75, 4.88 | 1.49, 0.09, 0.11 |
| heavy_slippery_balanced_0 | 0 | lift_F7.29_h1cm | 3.20, 3.54, 3.87 | 10.25, 11.10, 10.31 |
| heavy_slippery_balanced_0 | 0 | lift_F10.628_h1cm | 3.86, 3.36, 3.76 | 9.09, 9.25, 8.21 |
| heavy_slippery_balanced_0 | 0 | lift_F15.494_h1cm | 6.88, 5.89, 5.76 | 8.99, 8.57, 8.76 |
| heavy_slippery_balanced_0 | 0 | lift_F15.494_h2cm | 5.98, 6.70, 6.01 | 18.01, 19.47, 19.82 |
| heavy_slippery_balanced_0 | 1 | lift_F7.29_h1cm | 3.34, 3.21, 2.63 | 9.41, 10.11, 9.98 |
| heavy_slippery_balanced_0 | 1 | lift_F10.628_h1cm | 4.78, 4.06, 5.30 | 9.73, 9.28, 9.62 |
| heavy_slippery_balanced_0 | 1 | lift_F15.494_h1cm | 6.48, 4.59, 5.96 | 10.04, 9.86, 8.59 |
| heavy_slippery_balanced_0 | 1 | lift_F15.494_h2cm | 5.54, 6.28, 6.36 | 18.76, 19.56, 19.09 |
| heavy_slippery_balanced_0 | 2 | lift_F7.29_h1cm | 3.70, 3.58, 2.27 | 10.28, 8.67, 10.20 |
| heavy_slippery_balanced_0 | 2 | lift_F10.628_h1cm | 4.78, 4.17, 4.61 | 9.17, 9.59, 10.42 |
| heavy_slippery_balanced_0 | 2 | lift_F15.494_h1cm | 6.36, 5.60, 5.22 | 9.46, 9.25, 9.09 |
| heavy_slippery_balanced_0 | 2 | lift_F15.494_h2cm | 6.04, 7.18, 6.52 | 19.44, 19.92, 19.12 |
| medium_normal_balanced_1 | 0 | lift_F10.628_h1cm_boundary | 8.00, 8.25, 10.03 | 0.46, 0.28, 0.14 |
| medium_normal_balanced_1 | 1 | lift_F10.628_h1cm_boundary | 9.23, 9.00, 9.06 | 0.93, 0.47, 0.66 |
| medium_normal_balanced_1 | 2 | lift_F10.628_h1cm_boundary | 8.97, 8.60, 9.34 | 1.35, 1.03, 1.39 |

## C3 (probe, no restore, carry) vs C0 (stored CFD branch), same state and cell

| object | state | probe | C0: slip / done / dmg(frag) | C3: slip / done / dmg(frag) | slip max C0 → C3 [mm] | final xy err C0 → C3 [mm] | obj shift by probe [mm] |
|---|---|---|---|---|---|---|---|
| light_normal_balanced_0 | 0 | lift_F15.494_h1cm | 0 / 1 / 1 | 0 / 1 / 1 | 0.6 → 0.6 | 4 → 4 | 0.4 |
| light_normal_balanced_0 | 1 | lift_F15.494_h1cm | 0 / 1 / 1 | 0 / 1 / 1 | 0.6 → 0.6 | 4 → 4 | 0.3 |
| light_normal_balanced_0 | 2 | lift_F15.494_h1cm | 0 / 1 / 1 | 0 / 1 / 1 | 0.7 → 0.6 | 4 → 4 | 0.3 |
| heavy_slippery_balanced_0 | 0 | lift_F15.494_h1cm | 1 / 0 / 0 | 1 / 0 / 0 | 122.9 → 122.9 | 300 → 301 | 0.5 |
| heavy_slippery_balanced_0 | 1 | lift_F15.494_h1cm | 1 / 0 / 0 | 1 / 0 / 0 | 122.9 → 122.9 | 301 → 301 | 0.5 |
| heavy_slippery_balanced_0 | 2 | lift_F15.494_h1cm | 1 / 0 / 0 | 1 / 0 / 0 | 123.0 → 123.0 | 300 → 300 | 0.3 |
| medium_normal_balanced_1 | 0 | lift_F15.494_h1cm | 0 / 1 / 1 | 0 / 1 / 1 | 0.6 → 0.6 | 5 → 5 | 0.8 |
| medium_normal_balanced_1 | 1 | lift_F15.494_h1cm | 0 / 1 / 1 | 0 / 1 / 1 | 0.7 → 0.6 | 5 → 5 | 0.7 |
| medium_normal_balanced_1 | 2 | lift_F15.494_h1cm | 0 / 1 / 1 | 0 / 1 / 1 | 0.8 → 0.6 | 6 → 5 | 0.8 |

## Distribution-level test on the smoke data (illustration of §5 of design.md)

- lift_F10.628_h1cm: held counts over the stored states light_normal_balanced_0 3/3 vs heavy_slippery_balanced_0 0/3 → Fisher exact p = 0.100 (with 3 states the two objects are **not** distinguishable at α = 0.05 even with perfectly separated outcomes; R = 5 gives p = 0.0079).
- lift_F15.494_h1cm: held counts over the stored states light_normal_balanced_0 3/3 vs heavy_slippery_balanced_0 0/3 → Fisher exact p = 0.100 (with 3 states the two objects are **not** distinguishable at α = 0.05 even with perfectly separated outcomes; R = 5 gives p = 0.0079).
- lift_F15.494_h2cm: held counts over the stored states light_normal_balanced_0 3/3 vs heavy_slippery_balanced_0 0/3 → Fisher exact p = 0.100 (with 3 states the two objects are **not** distinguishable at α = 0.05 even with perfectly separated outcomes; R = 5 gives p = 0.0079).
- lift_F7.29_h1cm: held counts over the stored states light_normal_balanced_0 3/3 vs heavy_slippery_balanced_0 0/3 → Fisher exact p = 0.100 (with 3 states the two objects are **not** distinguishable at α = 0.05 even with perfectly separated outcomes; R = 5 gives p = 0.0079).
