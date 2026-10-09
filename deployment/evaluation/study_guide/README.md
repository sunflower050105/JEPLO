# Study guide: the “see, then hidden” box test

This folder explains the first diagnostic condition installed for the reproduced
`my_policy_r2` policy. The [short run guide](../README.md) is the quickest way to
start it; this document explains the reasoning and the implementation.
The box target is the first condition; the same machinery can now target the
`stairs` body or a single step such as `stair1`.

## 1. What question does this test ask?

The robot approaches the first box in MuJoCo. It initially receives normal
LiDAR depth images of that box. After the box has appeared in at least three
consecutive LiDAR scans, and the LiDAR is within 1.2 m of the box surface, the
test replaces the **projected box region** in subsequent depth images with the
missing/far-depth value. The box itself remains in the physics scene.

The practical question is: **after a valid visual preview, can the reproduced
controller negotiate the box when the current depth image no longer shows it?**
Compare this condition with an otherwise identical run in which the box remains
visible. The intervention changes the observation, not the policy weights,
training code, physical box, or command.

This first condition is a *behavioral probe*. A performance drop would show
sensitivity to losing current box depth after preview. Continued success would
be consistent with useful retained information, but would **not by itself prove**
that the learned recurrent state represents the box. Proprioception, earlier
depth frames, and other cues can also support the behavior.

## 2. What actually reaches the policy?

The relevant runtime path is:

```text
MuJoCo box + LiDAR rays
  → point cloud and optional target-mask metadata (ZMQ :5590)
  → depth publisher: 25×60 depth, 10-scan minimum, target masking (ZMQ :5560)
  → deployment launcher: current and previous depth images
  → recurrent sensor estimator: feature + updated hidden state
  → policy: proprioceptive history + estimator feature → actions
```

The exported `my_policy_r2` metadata records a `2 × 25 × 60` depth input and a
512-dimensional estimator hidden state. Its exported policy consumes
`proprio_hist` and `high_feat`; the depth stack enters the **sensor estimator**,
not that policy ONNX model directly. The launcher feeds the estimator's
`hidden_out` back as its next `hidden_in`. See the
[model metadata](../../../training/logs/rsl_rl/go2_loco/my_policy_r2/exported/test_cases/metadata.json)
and [deployment source](../../go2_deploy/deploy/src/deploy.cpp).

Do not confuse the two time windows:

| Window | Where it lives | What it contains |
| --- | --- | --- |
| `--stacked-frames 10` | Depth publisher | The pixelwise minimum of the latest 10 LiDAR scans. |
| Two depth channels | Deployment launcher | The previous and latest **published** depth images fed to the estimator. |

The LiDAR defaults to 20 scans/s and the depth publisher emits 10 images/s.
Therefore, after the publisher's first masked output, the launcher may still
present one older clean image in its two-channel input. When discussing
*recurrent* memory, examine behavior after both image channels are masked.

## 3. Trigger and mask, step by step

The optional experiment is enabled only by `--see-then-hidden` in the simulator.
Its implementation is isolated in
[`see_then_hidden.h`](../../unitree_mujoco/simulate/src/see_then_hidden.h).

1. MuJoCo identifies a named box geometry (`box1` by default) or the box geoms
   directly inside a named body (`stairs`).
2. A scan counts as a preview only if at least one ray hit any selected geom at
   a valid range between 0.1 and 2.0 m. A scan with no such hit resets the
   consecutive-preview counter.
3. The trigger also requires LiDAR-to-target **surface** distance ≤ 1.2 m. For
   a body, this is the nearest selected box geom, not the body's center. Both
   conditions must be true on a LiDAR scan.
4. Once triggered, `hidden=1` stays active until simulation time resets or the
   target is moved out of the active scene. The simulator prints `HIDE`.
5. The corners of each target box geom within the 2 m depth range are projected
   into the spherical grid. The union of their bounding rectangles, clipped to
   the `25 × 60` image with a one-pixel margin, becomes the mask. It moves with
   the robot and terrain.

Ground-truth target geometry is used **only to define the evaluation intervention**.
It is not sent as a feature to the estimator or policy. The rectangular mask can
cover nearby background inside its bounds; this is a limitation when
interpreting the result.

The depth range is 2 m. A real return at 0.8 m is approximately `0.8 / 2 = 0.4`
in the normalized image; a hidden pixel is set to `1.0`, the existing
missing/far-depth convention. It is not set to zero.

## 4. Why the publisher also changes

The simulator sends a legacy-compatible point cloud with optional suffix:

```text
[12 float pose][N × 3 float local points][25 × 60 byte mask][16 byte footer]
```

The footer identifies the extension, grid size, and whether hiding is active.
When `active=0`, the projected mask is metadata only: the clean preview still
reaches the estimator. The publisher stores that mask alongside each depth
scan. When `active` first becomes 1, it erases the target pixels in those earlier
stored scans at **their own old pixel positions**. New scans are masked before
entering the history. The current target region is also masked after the 10-scan
minimum is computed. This prevents a previously seen return from reappearing
outside the target's current pixels as the viewpoint changes, while retaining
other historical depth pixels.

The wire format is in
[`see_then_hidden_protocol.h`](../see_then_hidden_protocol.h); the receiving and
accumulation code is in
[`lidar_depth_pub.cpp`](../../lidar_depth_pub/lidar_depth_pub.cpp). Without the
experiment flag, simulator messages keep the old format. A `25×60`/`25×120`
mismatch is rejected instead of silently misaligning the mask.

## 5. Files and deployment pieces

| File | Purpose |
| --- | --- |
| [`param.h`](../../unitree_mujoco/simulate/src/param.h) | Parses test flags and defaults. |
| [`main.cc`](../../unitree_mujoco/simulate/src/main.cc) | Enables the test on the LiDAR publisher. |
| [`mid360_lidar.h`](../../unitree_mujoco/simulate/src/mid360_lidar.h) | Traces rays, gets hit geometry IDs, and appends mask metadata. |
| [`see_then_hidden.h`](../../unitree_mujoco/simulate/src/see_then_hidden.h) | Preview counter, trigger, moving projection, and CSV diagnostics. |
| [`lidar_depth_pub.cpp`](../../lidar_depth_pub/lidar_depth_pub.cpp) | Converts points to depth and applies the mask across accumulation. |
| [`CMakeLists.txt`](../CMakeLists.txt) | Builds the simulation-only depth publisher without ROS/Livox. |
| [`test_see_then_hidden.cpp`](../tests/test_see_then_hidden.cpp) | Checks preview, trigger, reset, and message parsing in a small MuJoCo scene. |

The evaluation build uses the *same* depth-publisher source as the regular ROS
build, with `JEPLO_SIM_ONLY` selecting its simulation path. The ordinary real
LiDAR path has no see-then-hidden trigger.

## 6. Run one masked trial

Run each command in a **separate Ubuntu 22.04 container terminal**, starting
from the repository root. The simulator requires the project's MuJoCo and
Unitree SDK2 dependencies. The host-built binary in `simulate/build/` cannot
run in this container because its `yaml-cpp`, Boost, glibc, and C++ runtime
requirements differ. Build in `build-ubuntu2204/` instead. Generated binaries
under `build*/` and `output/` are ignored by Git. The original deployment
launcher in `output/deploy` is the container-compatible one on this
workstation; see the [project README](../../../README.md) for its normal build.

Build or check the simulator and evaluation publisher **inside the container**:

```bash
cmake -S deployment/unitree_mujoco/simulate \
  -B deployment/unitree_mujoco/simulate/build-ubuntu2204 -DCMAKE_BUILD_TYPE=Release
cmake --build deployment/unitree_mujoco/simulate/build-ubuntu2204 \
  --target unitree_mujoco --parallel 2

cmake -S deployment/evaluation -B deployment/evaluation/build-ubuntu2204 \
  -DCMAKE_BUILD_TYPE=Release
cmake --build deployment/evaluation/build-ubuntu2204 --parallel 2
ctest --test-dir deployment/evaluation/build-ubuntu2204 --output-on-failure
```

Terminal 1 — simulator:

```bash
cd deployment/unitree_mujoco/simulate/build-ubuntu2204
./unitree_mujoco --lidar --lidar-legacy-fov --see-then-hidden \
  --see-then-hidden-distance 1.2 --see-then-hidden-preview-scans 3 \
  --see-then-hidden-target box1 \
  --see-then-hidden-log /tmp/jeplo-see-then-hidden-trial01.csv
```

For the staircase, restart this simulator with
`--see-then-hidden-target stairs` and a new log filename, then press **Ctrl+1**
to select stairs. The body target groups its active step geoms; a single geom
target such as `stair1` hides only that step. Changing terrain with Ctrl+1 or
Ctrl+2 does **not** change a running simulator's target flag. The other
terminals use the same commands.

Terminal 2 — simulation depth publisher:

```bash
./deployment/evaluation/build-ubuntu2204/lidar_depth_sim --sim --fov 25x60 \
  --downsample-rate 4 --stacked-frames 10 --no-cage-mask
```

`25x60` matches this estimator's exported input shape;
`--stacked-frames 10` matches the local training configuration. The project's
simulation example uses downsampling by four. `--no-cage-mask` makes the box
intervention easier to isolate; keep that choice identical in control trials.

Terminal 3 — depth viewer (recommended for checking the mask):

```bash
cd deployment/go2_deploy
python3 scripts/view_lidar_depth.py
```

Terminal 4 — reproduced policy:

```bash
cd deployment/go2_deploy/output
ort_lib=/home/lin/ubuntu_22_04/onnxruntime-linux-x64-gpu-1.22.0/lib
trt_lib=/home/lin/ubuntu_22_04/TensorRT-10.9.0.34/targets/x86_64-linux-gnu/lib
LD_LIBRARY_PATH="$ort_lib:$trt_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  ./deploy --net lo --model-dir my_policy_r2 \
  --ood-count-threshold 15 --raw-actions --trt-cache
```

The first launch may spend time building TensorRT engines. The launcher
validates them against its ONNX test cases, then prints a prompt to press
**Enter** before robot control begins. Both local library directories in the
command above are required for this GPU ONNX Runtime build.

Terminal 5 — keyboard controller:

```bash
cd deployment/go2_deploy
python3 scripts/pygame_wm_control.py
```

In the MuJoCo window, press **Ctrl+2** to place the boxes terrain in front of
the robot, then drive toward the first box. Do this before collecting the trial.
The box is initially hidden elsewhere in the scene, so a CSV with only its
header *before* Ctrl+2 is expected. A valid triggered trial should show
`[SeeThenHidden] HIDE` in the simulator, `target mask ON` in the publisher, and a
rectangular far-depth area over the box in the viewer. Give each trial a
distinct CSV path; restarting the simulator with the same path overwrites the
old log.

## 7. Control trial and observations

For the visible control, restart the same simulator command **without**
`--see-then-hidden`, `--see-then-hidden-distance`,
`--see-then-hidden-preview-scans`, `--see-then-hidden-target`, and
`--see-then-hidden-log`. Keep the publisher, policy, controller command, scene,
and initial pose as similar as possible. The policy and depth publisher remain
the same in both conditions.

The CSV columns are `sim_time_s`, target-surface `target_range_m`, number of
valid `target_hits`, `consecutive_preview_scans`, `hidden` (0 or 1),
`mask_pixels`, and the sensor and nearest target-geom center positions.
`mask_pixels` is zero during preview, even though the projected mask is
transmitted as metadata. If the target is more than
10 m away or inactive, no per-scan row is written. A trial that never reaches a
`hidden=1` row did **not** test this condition.

Before collecting many trials, define one outcome rule. For example, record
whether the robot passes the far side of the box and stays upright for two
seconds, plus unintended collision and fall events. Use the same rule in both
conditions. A simple record sheet is:

| Trial | Condition | Command | Preview scans at trigger | Trigger range | Crossed box? | Fell? | Notes |
| --- | --- | --- | ---: | ---: | --- | --- | --- |
| 01 | Visible control | fixed command | — | — |  |  |  |
| 02 | See, then hidden | same command |  |  |  |  |  |

First confirm that the reproduced policy handles the **visible** box
reliably. If both conditions fail, the reproduction or task setup needs work
before the occlusion gap can be assessed. If visible runs succeed and valid
hidden runs fail more often, that supports a behavioral gap worth following up.
If both succeed, this specific distance and mask have not exposed a gap.

## 8. Limits and next research steps

- The installed unit test checks trigger logic and wire parsing. Build and
  startup checks do **not** measure traversal performance; that requires driven
  trials.
- The mask is a projected rectangle, not perfect object segmentation. Nearby
  pixels may be removed. This needs to be considered when interpreting a drop.
- Keep `box1` and `stairs` as separate conditions in the results table: the
  grouped stair mask covers multiple steps and can have a different area.
- The two-image estimator input can briefly retain an older clean image after
  the first masked output. Inspect behavior after that image has been replaced.
- This condition alone cannot separate remembered vision from proprioceptive or
  reactive control. A later “never seen” control, plus repeated paired trials,
  would strengthen a paper claim. That condition is **not installed yet**.

The current result is therefore an instrumented experiment, not evidence that
the proposed gap already exists.
