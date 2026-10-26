# Headless Isaac Sim stair benchmark

Run from the `training` directory in the environment that has Isaac Lab 2.3.2
and the editable JEPLO packages installed. The default evaluates the CTS
**student** in `my_policy_r2/model_19999.pt` for 100 independent trials at
each of 0.5 and 0.7 m/s:

```bash
cd /root/autodl-tmp/JEPLO-main/training
PYTHONUNBUFFERED=1 /root/autodl-tmp/env_jeplo/bin/python -u scripts/benchmark_stairs.py \
  --headless --device cuda:0
```

Use `--repeats 1` for the first smoke test. `--output_dir` selects a new,
nonexistent output directory. `--checkpoint` selects another `.pt` file.
The command starts Isaac Sim headless and does not request camera rendering.
The existing lidar raycaster still requires an Isaac-compatible GPU and driver.
The CTS runner prints a teacher/student environment split during initialization.
With one environment it reserves that slot for teacher training bookkeeping,
but this benchmark calls the separate student inference policy for every trial.
`benchmark_stairs` in that printout is the name of the stair terrain course.

Each trial starts the robot 3.0 m before a straight ten-step course. Every
rise is 0.20 m and every run is 0.30 m. The top landing is 1.20 m long and
1.60 m wide. The robot and terrain receive fixed 0.4 static and dynamic
friction in Isaac. The external command is `(0.5 or 0.7, 0, 0)`. As in the
MuJoCo stair benchmark, an IMU-style yaw hold modifies the policy's `wz`
request using `clamp(2 × heading_error − 0.25 × gyro_z, −1, 1)`; use
`--no_yaw_hold` for an open-loop comparison. There is no simulator-pose
centering. No
startup mass/COM randomization, pushes, joint offset, or reset pose randomization
is applied. The policy's trained depth observation path remains in use.

Success requires all four foot body origins inside and near the top landing,
recent force-sensor contact for all feet, the base over the landing and at
least 0.16 m above it, tilt at most 30 degrees, and a 0.2 s continuous hold.
A fall (base below 0.16 m or tilt above 60 degrees for 0.2 s) and leaving the
0.8 m half-width corridor take precedence. The deadline is 16 simulated
seconds. This is labelled `stair_landing_isaac_contact_proxy_v1` because Isaac
reports foot contact forces without the landing-geom and contact-normal detail
that the MuJoCo `stair_landing_v1` scorer uses. The landing footprint, height,
posture, and hold thresholds match; the contact test is an approximation.

The output directory contains `metadata.json` (including checkpoint hash),
`trials.csv` (flushed after every trial), and `summary.json` (per-speed success
counts and rates). `metadata.json` is written at startup; `trials.csv` appears
only after the environment and policy have initialized. A stopped run retains
the completed trial rows. Results
from this in-process Isaac student and the MuJoCo deployment student can help
localize a gap, but they also differ in contact solver and depth/deployment
path, so a rate difference alone does not prove one unique cause.
