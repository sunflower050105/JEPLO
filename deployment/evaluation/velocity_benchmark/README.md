# Automated 0.8 versus 1.0 m/s diagnostic

This experiment runs **one Go2 at a time**, sends a constant forward command,
records its motion, and exports a comparison. Start with one trial at each speed
to inspect instability. Use `--repeats 10` for ten sequential pairs once the setup
is working. There is no need to spawn 500 robots or environments.

The controller remains closed loop: each action comes from the current
proprioception, depth, and recurrent state. Repeated trials restart the simulator,
depth publisher, and controller, so their histories are fresh. **Video replay**
only renders the recorded trajectory; it does not replay actions or rerun physics.

This folder owns the runner, scoring implementation, tests, and video renderer.
It builds separate `unitree_mujoco_velocity` and `lidar_depth_velocity` executables.
The default simulator executable and the earlier see-then-hidden experiment
remain available. No see-then-hidden or added occlusion-band flags are used here.

**Box scoring correction (`box_ascent_v2`):** the default box task now ends after
climbing onto the second step. The previous rule required crossing beyond the
far edge and could fail a completed climb during descent. Old results, including
`runs/boxes_100pairs`, still describe that old crossing task; they have not been
changed or re-scored. Collect a new run for ascent success rates. Rebuild the
benchmark simulator inside Ubuntu 22.04 before using the updated runner.

## Build inside the Ubuntu 22.04 container

Enter the existing `ubuntu-22-04` distrobox and change to this repository root.
On this workstation that root is `/run/host/home/lin/Jeplo/JEPLO-main` inside the
container. Do not build these binaries on the Ubuntu host and run them in the
container: their system library versions differ.

```bash
distrobox enter ubuntu-22-04
cd /run/host/home/lin/Jeplo/JEPLO-main

cmake -S deployment/unitree_mujoco/simulate \
  -B deployment/unitree_mujoco/simulate/build-velocity-ubuntu2204 \
  -DCMAKE_BUILD_TYPE=Release
cmake --build deployment/unitree_mujoco/simulate/build-velocity-ubuntu2204 \
  --target unitree_mujoco_velocity --parallel 2

cmake -S deployment/evaluation \
  -B deployment/evaluation/build-velocity-ubuntu2204 -DCMAKE_BUILD_TYPE=Release
cmake --build deployment/evaluation/build-velocity-ubuntu2204 --parallel 2
ctest --test-dir deployment/evaluation/build-velocity-ubuntu2204 --output-on-failure
```

Create an isolated Python environment for the runner and video renderer. If
`venv` is unavailable, install `python3.10-venv` in the container first.

```bash
/usr/bin/python3 -m venv --system-site-packages deployment/evaluation/velocity_benchmark/.venv
deployment/evaluation/velocity_benchmark/.venv/bin/python -m pip install \
  -r deployment/evaluation/velocity_benchmark/requirements.txt
```

MuJoCo Python is pinned to **3.3.6**, matching this repository's C++ library. The
video renderer loads a binary snapshot of the exact model used by the trial,
including the terrain positions. Revisit the pin if you change MuJoCo versions.

The existing policy launcher needs its ONNX Runtime and TensorRT libraries.
For this workstation, set these in the same terminal:

```bash
export LD_LIBRARY_PATH="/home/lin/ubuntu_22_04/onnxruntime-linux-x64-gpu-1.22.0/lib:/home/lin/ubuntu_22_04/TensorRT-10.9.0.34/targets/x86_64-linux-gnu/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

## First two videos

Close previous MuJoCo, depth-publisher, policy, and keyboard-controller processes.
The runner starts all three processes itself and replaces the keyboard controller.
It is restricted to loopback simulation, DDS domain 1; it does not target hardware.

```bash
deployment/evaluation/velocity_benchmark/.venv/bin/python \
  deployment/evaluation/velocity_benchmark/run.py --check

deployment/evaluation/velocity_benchmark/.venv/bin/python \
  deployment/evaluation/velocity_benchmark/run.py \
  --model-dir my_policy_r2 --terrain flat --repeats 1 --video
```

The first TensorRT initialization may take several minutes. The runner answers
the launcher's Enter prompt automatically, waits for policy and depth activity,
then runs a two-simulation-second zero-command warmup. It checks that the base
is near the starting position and heading before issuing the test command.
You do not need to press keys or scan an obstacle manually. Avoid resetting,
pausing, changing terrain, dragging the robot, or changing simulation speed.
Press Ctrl+C in the runner terminal to stop; it terminates only its own children.
Use `--startup-timeout 600` if building new TensorRT engines needs more than the
default 300 seconds.

Outputs go to a new timestamped folder under `velocity_benchmark/runs/`. Use
`--output /absolute/path/new-folder` to choose one. Existing folders are never
overwritten. Video rendering happens **after all trials have finished**, avoiding
competition with policy inference during the runs.

For the existing two-box course:

```bash
deployment/evaluation/velocity_benchmark/.venv/bin/python \
  deployment/evaluation/velocity_benchmark/run.py \
  --model-dir my_policy_r2 --terrain boxes --box-task ascent --repeats 1 --video
```

The terrain switcher selects the original `boxes` group automatically. This tests
both existing boxes, with their existing dimensions; it is not a newly invented
single-box course. `box1` has a 0.3 m top and `box2` a 0.7 m top in this scene.
Both speeds use the same scene. The default time limits are 8 simulation seconds
for flat ground and 12 for boxes; override with `--duration`.

## Separate ten-step stair ascent

The stair benchmark selects
[`scene_benchmark_stairs.xml`](../../unitree_mujoco/unitree_robots/go2/scene_benchmark_stairs.xml)
explicitly. The ordinary `scene.xml`, flat benchmark, and two-box benchmark
still use their existing scenes and scoring rules. Rebuild `unitree_mujoco_velocity`
and the evaluation tests after changing the C++ scorer or terrain switcher.

This course has ten consecutive 0.20 m rises and 0.30 m runs, reaching 2.00 m,
followed by a 1.20 m long landing. It is a straight-on ascent: it does not
reproduce the paper's randomized approach headings. The two default commands
are 0.8 and 1.0 m/s, with a 16 simulation-second limit. To inspect one pair:

```bash
deployment/evaluation/velocity_benchmark/.venv/bin/python \
  deployment/evaluation/velocity_benchmark/run.py \
  --model-dir my_policy_r2 --terrain benchmark_stairs --repeats 1 --video
```

Success (`stair_landing_v1`) means all four feet fit on the top landing, each
has made geometric contact with its top within 0.5 s, and the base is within
the landing footprint, at least 0.16 m above its surface, and tilted no more
than 30 degrees. Those conditions must hold continuously for 0.2 s before the
time limit. Fall and leaving the 0.8 m half-width course corridor take priority.
The runner keeps the forward command active until success or failure. Reaching
the tenth tread with only the front feet, or passing the landing in the air,
does not count. This rule is explicit for this experiment; the paper does not
publish an equivalent exact scoring rule, so these rates are not a direct
reproduction of its 59.2% stair figure.

After confirming the first videos, use `--repeats 500` for 500 trials **at each
speed** (1,000 sequential trials total). Inspect `timing_ok` and the real-time
factor before interpreting speed differences. The runner currently restarts the
simulator, depth publisher, and controller for every trial; it does not reuse
processes or run parallel environments.

To collect **100 trials per speed (200 total), without rendering videos**:

```bash
deployment/evaluation/velocity_benchmark/.venv/bin/python \
  deployment/evaluation/velocity_benchmark/run.py \
  --model-dir my_policy_r2 --terrain boxes --box-task ascent --repeats 100 \
  --output deployment/evaluation/velocity_benchmark/runs/boxes_ascent_100pairs_v2
```

Use a new output name if that directory already exists. The motion trace is
still saved, so individual videos can be rendered later. Read `summary.csv` for
the success rates and check that its `scoring_rule` is `box_ascent_v2`.

## What counts as success?

These are explicit **pilot diagnostic rules**, not a claim to reproduce the
paper's exact success criterion. Inspect the videos before choosing the final
paper protocol, and freeze those rules before collecting the larger dataset.

| Outcome | Rule |
| --- | --- |
| Flat success | Remain in the course for the full duration and advance at least `0.8 × commanded speed × duration`. Change the fraction with `--min-progress-ratio`. |
| Box ascent success (default) | All four feet have reached the top of `box2`, and the robot meets the footprint/contact/posture conditions below continuously for 0.2 s, by the time limit. The trial ends immediately; descent is excluded. |
| Legacy box crossing (`--box-task traverse`) | Base crosses the far edge of `box2` plus 0.4 m and is beyond it at a sample at least one second after the first crossing, without an earlier failure. This preserves the original rule; it is not an ascent-only score or a foot-contact landing test. |
| Fall | Base height below 0.16 m **or** total tilt above 60° continuously for at least 0.2 s. |
| Left course | Absolute world lateral position exceeds 1.0 m on flat ground or 0.5 m on the box course. |
| Timeout | Time limit reached before the progress/completion rule. |
| Invalid | Startup, process, stale-depth, paused-simulator, or other infrastructure failure prevents a complete trial. Reported separately from robot failures. |

The ascent detector is separate in `box_ascent.h`. It uses the following fixed
conditions at each recorded sample, identically for both speeds:

- All four foot spheres (`FL`, `FR`, `RL`, `RR`) fit inside the second box's XY
  footprint, including their radii. Their bottoms are at or above its top, with
  3 cm tolerance for contact penetration.
- Every foot has made geometric contact with that top within the last 0.5 s.
  Contacts can alternate during trotting: four simultaneous contacts are not
  required. Side/riser contacts and calf contacts do not qualify. A qualifying
  contact has separation at most 5 mm, position within 3 cm of the top plane,
  and absolute normal alignment with the top normal of at least 0.7.
- The base projects inside the top footprint, has at least 0.16 m clearance
  above it, and has total tilt at most 30 degrees.
- These conditions hold for 0.2 s continuously. Leaving them resets the hold;
  a sample gap over 0.1 s resets both the hold and contact history. Fall or
  left-course failure takes priority if detected on the same sample.

Thus, front feet reaching the second step, the body merely passing an X
coordinate, or flying over the platform without foot contact cannot qualify.
The detector recomputes positions and geometric contacts from the saved pose in
separate MuJoCo state; it neither alters physics nor provides policy inputs.
This is a short upright ascent criterion, not proof of a stationary stance or
of load-bearing force at every foot. It does not assess what happens during a
later descent. No stop command is needed to qualify: forward command remains
constant until the trial ends.

The general fall rule's height uses world Z: it is a ground-fall check, separate
from the ascent rule's platform-relative body clearance.
The controller's existing OOD/safety behavior remains active. If it switches to
damping, the physical result is still scored normally; inspect `deploy.log` to
separate a safety intervention from an unconstrained policy failure.

Success rate is `successes / valid completed trials`. Attempted and invalid
counts are exported too, so a failed setup cannot silently improve or worsen the
percentage. Flat success allows some tracking error; the reported mean speed
and tracking RMSE tell you how closely the robot actually followed its command.
Early failures have shorter measurement windows; do not compare their mean
speeds as though all trials lasted the same duration. Successful ascents now end
early too. `ascent_time_s` measures completion from the trial start, including
the approach. `goal_x_m` is null for ascent because it has no scalar finish line.

Scoring rules are versioned in the manifest, per-trial results, CSVs, and new
video labels. Summary aggregation keeps different rules separate. Do not pool
`box_ascent_v2` and `box_traverse_v1` rates or label an old crossing result as an
ascent result. Freeze the selected criterion before the new data collection.

## What is controlled, and what changes?

The selected checkpoint, controller settings, terrain, sensor settings, and
starting-pose tolerances are fixed. Only forward command changes within a pair;
lateral and yaw commands stay zero. Pair order alternates to reduce a systematic
0.8-first/1.0-second ordering effect.

| Setting | Default here |
| --- | --- |
| Policy control | Existing deployment default: 50 Hz wall clock |
| Estimator updates | Existing deployment default: every 5 control steps |
| LiDAR scan scheduling | 20 Hz simulation time, configurable with `--lidar-hz` |
| Depth publishing | Existing publisher: 10 Hz wall clock |
| Scan accumulation | 5 scans, the normal publisher default; `--stacked-frames` can override |
| Point downsampling | Every fourth point, following the main README |
| Depth image | 25×60 for `my_policy_r2`; use the shape required by your exported estimator |
| Cage mask | Normal cage mask enabled; same seed for both speeds in each pair |
| Extra object hiding | Disabled |

The cage seed increments between pairs. This preserves the ordinary mask
distribution while avoiding a different random cage pattern within a speed
comparison. `--no-cage-mask` explicitly disables it for a separate condition.
The old occlusion experiment used ten accumulated scans and disabled the cage
mask in its example; those choices are **not** inherited here. This benchmark
preserves the normal deployment defaults, not an assumed exact timing match to
training. Do not change these settings between the two speeds.

The simulator's normal scan FOV is used. `--legacy-lidar-fov` additionally applies
its legacy restriction if that is part of the baseline you intend to measure.
All selected values and exact launch commands are saved.

Different speeds naturally produce different proprioception and depth values.
For obstacle analysis, compare rows near the same world X/distance from the box,
and at first contact, rather than expecting identical sensor images at the same
elapsed time. State logs provide world X and elapsed time for this alignment.

## Reading the output

The run folder contains:

- `summary.csv` / `summary.json`: success counts and rates for each speed.
- `results.csv` / `results.json`: every attempt, including invalid attempts.
- `manifest.json`: settings, model/binary/config SHA-256 hashes, Git revision,
  and whether the checkout had local changes.

Each trial subfolder contains:

- `video.mp4`: rendered recorded motion, with speed/time/tilt labels.
- `state.csv`: simulation time, wall timestamp, world position, forward speed,
  roll/pitch/yaw/tilt, and every generalized position used to render the video.
  Ascent diagnostics follow the `qpos_*` columns: `ascent_on_top_mask`,
  `ascent_contact_mask`, `ascent_recent_contact_mask`, `ascent_candidate`, and
  `ascent_hold_s`. Mask bits 1/2/4/8 mean FL/FR/RL/RR; 15 means all four.
- `scene.mjb`: model snapshot with the actual terrain layout.
- `result.json`: physical outcome and metrics, including real-time factor,
  scoring rule, ascent time (null unless successful ascent), and ascent thresholds.
- `start.json`: measured starting pose and command-switch wall timestamp.
- `sent_commands.csv`: commands sent at 50 Hz wall clock, including warmup.
- `depth_timing.csv`: depth timestamps observed by an additional subscriber.
- `deploy.log`, `depth.log`, `simulator.log`: original process diagnostics.
- `commands.json`: exact argument arrays for the three child processes.

The new observer does **not** tell you precisely when the estimator consumed
each frame. `publisher_scan_receive_ns` is when the depth publisher received its
newest input scan, not the LiDAR's original acquisition time; `depth_publish_ns`
is when it published the aggregated image. The observer timestamp is when the
runner received it. Use the existing controller `[LATENCY]` and `[HISTORY]`
diagnostics for estimator consumption and depth-pair behavior.

This is why a 100 ms publication interval and a 64 ms age should not be treated as
the same quantity. Also inspect `real_time_factor`: the policy and publisher use
wall clock while physics and ray scans use simulation time. If it differs much
from 1, the intended simulation-time control/perception rates have changed.
The runner flags `timing_ok=false` when the whole-trial ratio is outside
`[0.95, 1.05]`. This flag does not erase the physical outcome or exclude it from
the pilot success count. Check short-window RTF messages as well: a whole-trial
average can hide a brief slowdown. Controller `[HISTORY]` messages reporting
identical depth channels are retained; this experiment does not fix the existing
depth-buffer update behavior.

The command sender and simulator communicate asynchronously. The GO marker and
first forward message occur in one runner iteration, but the controller consumes
the command on its next cycle. The boundary is therefore not an exact lockstep
control tick; raw timestamps are saved rather than claiming perfect alignment.

## Rendering again without rerunning the robot

```bash
deployment/evaluation/velocity_benchmark/.venv/bin/python \
  deployment/evaluation/velocity_benchmark/render_video.py \
  deployment/evaluation/velocity_benchmark/runs/RUN/pair001_v0.8 --backend egl
```

If EGL is unavailable locally, try `--backend glfw` from the same desktop
terminal. This is MuJoCo OpenGL rendering; it does not depend on Isaac Sim's RTX
renderer. The video uses nearest recorded poses at 30 fps, with no invented
physics between samples. The original 50 Hz target state trace remains available.
Camera images are a visualization of the trajectory, not the depth observations
fed into the policy. `first_frame.png` provides a quick rendering check.
Use `--azimuth`, `--elevation`, and `--distance` to inspect the same recorded
motion from another view. Re-rendering replaces that trial's `video.mp4`.

## Before a larger experiment

Use `--repeats 10` after inspecting the first pair. Repetitions give repeatability
evidence for this particular checkpoint, scene, and initialization protocol.
They do not establish performance over a broad terrain distribution. The mask
seeds vary between pairs, while the initial pose and terrain remain fixed.
Asynchronous process timing can still vary. With the cage disabled, repeating a
nearly deterministic fixed scene 500 times does not create 500 diverse scenarios.

For a paper, define the distribution of terrain/start conditions and use paired
seeds for both speeds. Parallel environments can make that faster, but a serial
runner is valid when it samples the same defined trial distribution. This first
implementation is for collecting interpretable videos and a transparent pilot
success comparison before committing to that larger protocol.

## Scoring regression tests

After the build above, `ctest` covers ascent before descent, front-feet-only and
airborne false positives, alternating diagonal contacts, interrupted holds,
stale contacts, missing samples, riser contacts, low body clearance, the deadline,
and the legacy flat/crossing rules. Runner tests also check task selection and
that results with different criteria are never pooled:

```bash
deployment/evaluation/velocity_benchmark/.venv/bin/python -m unittest discover \
  -s deployment/evaluation/velocity_benchmark -p test_runner.py
```
