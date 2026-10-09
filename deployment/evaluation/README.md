# See, then hidden: first diagnostic condition

For the rationale, data path, and interpretation of results, read the
[study guide](study_guide/README.md).

This evaluation-only MuJoCo test leaves the policy and training code unchanged.
The robot sees the first box (`box1`) for at least three distinct LiDAR scans.
When the LiDAR is within 1.2 m of that box's surface, its projected region is
masked to the normal missing-depth value (`1.0`) on every subsequent depth
image. The box stays physical in MuJoCo. The mask follows it as the robot moves.
The publisher stores each clean scan's projected box region alongside the depth.
At mask onset, it removes those earlier box pixels at their original positions,
then masks each new scan before accumulation and the current box region after
accumulation. Other depth pixels remain intact. On simulation reset, the test
returns to the clean-preview phase.

The target geometry and trigger use MuJoCo ground truth only to define the
evaluation intervention. They are never supplied to the policy. The trigger
requires actual simulated LiDAR hits on the target, so a trial with no visual
preview will not silently count as "seen, then hidden".
The mask is the box's projected rectangular extent with a one-pixel margin;
it can also cover nearby background pixels within that rectangle.

## Build

Build the customized MuJoCo simulator as in the project README. It needs the
Unitree SDK2 CMake package available to CMake. The evaluation depth publisher
requires only C++17 and libzmq; it compiles the simulation path of the same
source used by the normal ROS publisher:

```bash
cmake -S deployment/evaluation -B deployment/evaluation/build -DCMAKE_BUILD_TYPE=Release
cmake --build deployment/evaluation/build --parallel
ctest --test-dir deployment/evaluation/build --output-on-failure
```

On this workstation, `output/deploy_see_then_hidden` is a rebuilt copy of the
unchanged deployment launcher; the older `output/deploy` links to unavailable
libraries. On another machine, use its working deployment launcher instead.

## Run a pilot trial

Start these processes in separate terminals from the repository root. Choose
the boxes terrain with **Ctrl+2** in the MuJoCo window before driving. The
simulator prints `HIDE` when the target becomes masked. Use the depth viewer to
check that the box region becomes far-depth while the surroundings remain visible.

```bash
cd deployment/unitree_mujoco/simulate/build
./unitree_mujoco --lidar --lidar-legacy-fov --see-then-hidden \
  --see-then-hidden-distance 1.2 --see-then-hidden-preview-scans 3 \
  --see-then-hidden-target box1 --see-then-hidden-log /tmp/jeplo-see-then-hidden.csv
```

```bash
./deployment/evaluation/build/lidar_depth_sim --sim --fov 25x60 \
  --downsample-rate 4 --stacked-frames 10 --no-cage-mask
```

```bash
cd deployment/go2_deploy/output
./deploy_see_then_hidden --net lo --model-dir my_policy_r2 \
  --ood-count-threshold 15 --raw-actions
```

Drive using `deployment/go2_deploy/scripts/pygame_wm_control.py` as in the main
README. For a clean comparison, run the same scene and commands without
`--see-then-hidden`, using the same depth publisher settings. The command above
uses the main README's simulation downsampling and disables the unrelated cage
mask so the box preview is clear. The local training configuration for
`my_policy_r2` records `lidar_num_stacked: 10`; verify that
the chosen checkpoint really used that setting before comparing trials.

The CSV contains simulation time, target surface range, target LiDAR hit count,
consecutive preview scans, mask state, mask area, and sensor/target positions.
Reject trials with no `hidden=1` row or with too few preview scans. This first
installation logs the perception intervention; traversal success and contacts
must still be scored from the simulator run. During the first depth update after
mask onset, the policy's two-frame input can contain an earlier clean image;
inspect later updates when attributing behavior to recurrent memory.
For the gap check, compare repeated runs with the same initial pose, box
placement, and command in the unmasked and masked conditions. A hidden-condition
failure is informative only if the reproduced policy first traverses the
unmasked box reliably; record success, box contact, and falls for both.

The test is simulation-only. The ordinary LiDAR publisher and the real-robot
path have no trigger; unmodified simulator messages still use the legacy point
cloud format.
