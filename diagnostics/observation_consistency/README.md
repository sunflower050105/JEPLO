# Observation consistency checks for `my_policy_r2`

These tools answer three different questions, in this order:

1. Given identical raw state and depth input, do the two observation pipelines produce the same tensors?
2. Given identical model input tensors, do the exported models reproduce the saved reference outputs?
3. During a real diagnostic rollout, which tensors and which depth frames actually reach inference?

Run and review each check before interpreting the next. A mismatch is a useful result;
the scripts do not change the policy or automatically repair a mismatch.

## One command per machine: unattended sequence

Both machines can run independently at the same time. Within each machine, stages
run sequentially. AutoDL runs Check 1A, Check 1B, live Isaac capture, then trace
analysis. The local machine runs Check 1A, Check 1B, Check 2 replay, builds the
separate controller, runs live MuJoCo capture, then analyzes its traces. Check 2
uses the local ONNX environment; it is not duplicated in the Isaac environment.

In your AutoDL SSH terminal:

```bash
/root/autodl-tmp/env_jeplo/bin/python -u \
  /root/autodl-tmp/JEPLO-main/diagnostics/observation_consistency/run_all.py \
  --side isaac
```

From your local host terminal, this command enters the existing build container:

```bash
distrobox enter ubuntu-22-04 -- env \
  LD_LIBRARY_PATH=/home/lin/ubuntu_22_04/onnxruntime-linux-x64-gpu-1.22.0/lib:/home/lin/ubuntu_22_04/TensorRT-10.9.0.34/targets/x86_64-linux-gnu/lib \
  /run/host/home/lin/Jeplo/JEPLO-main/deployment/evaluation/velocity_benchmark/.venv/bin/python -u \
  /run/host/home/lin/Jeplo/JEPLO-main/diagnostics/observation_consistency/run_all.py \
  --side mujoco --setup-replay
```

`--setup-replay` creates a fresh venv inside the batch output and installs only
the CPU analysis requirements there. It does not modify the benchmark Python or
the remote environment. Alternatively, use `--replay-python PATH_TO_PYTHON` to
reuse an existing analysis environment. A fresh private controller is built by
default; `--audit-build PATH` reuses a prior diagnostic build.

Defaults: r2, speeds 0.8 and 1.0 m/s, one trial per speed, 30 seconds per trial,
friction 1.0, seed 0, standard/no occlusion. MuJoCo uses 50 Hz, ten accumulated
scans, and stride eight; Isaac retains its current training configuration.
Set `--repeats N`, `--duration SECONDS`, or `--speeds ...` on both commands if
needed. Leave other MuJoCo controllers stopped and keep the PC awake. These
commands run in the foreground: keep the terminals and SSH session open.

Each invocation prints a new `runs/SIDE-batch-TIMESTAMP/` directory containing
`contract/`, `depth/`, `live/`, `analysis/`, local `replay/` and `controller/`,
numbered stage logs, and `summary.json`. Long stages write their progress to the
printed log path. A numerical mismatch with a completed report is recorded and
the next stage runs. An execution/setup failure stops the batch and preserves
the logs. Final exit status: 0 completed; 1 completed with measured mismatches;
2 execution error/interruption. There is no overall sim-to-sim verdict.

Add `--dry-run` to save/print the command plan without building, installing, or
launching simulations. After both batches finish, sync down the Isaac batch and
use the comparison commands below, substituting the new `contract/`, `depth/`,
and `live/` paths. Cross-machine comparisons and fresh Isaac-trace model replay
require those downloaded results and are not performed by the independent runs.

## Isolation from your existing work

All additions are under `diagnostics/observation_consistency/`. No existing source,
benchmark, trained checkpoint, ONNX export, launch command, or baseline executable
is edited by these tools. Every output directory must be new; existing results are
never overwritten.

Check 1 reads baseline source and executes it in a small harness with supplied
data. Check 2 reads model/test files. Check 3 Isaac installs observers only in the
diagnostic Python process. Check 3 MuJoCo generates a **separate source tree and
controller binary**, with copied models and its own TensorRT cache. The generated
`instrumentation.diff` shows every insertion for review.

The live diagnostics use the existing simulator/depth binaries and launch one
simulation at a time. Close other MuJoCo controllers/benchmarks before running:
the existing DDS domain and ZMQ ports are shared. Recording introduces overhead;
use these runs to inspect observations, and keep them separate from success-rate
results. No live simulations were run while implementing this suite.

## What the policy input means

The student estimator receives `prop_hist [1,450]`, `depth_stack [1,2,25,60]`, and
`hidden_in [1,512]`. The r2 policy export receives `proprio_hist [1,450]` and
`high_feat [1,512]`, and returns `actions [1,12]`. The 450 values are ten 45-value
frames, newest first. Within each frame:

| Zero-based indices | Meaning | Scaling |
|---|---|---|
| 0–2 | Body angular velocity | ×0.25 |
| 3–5 | Gravity projected into body frame | No additional scale |
| 6–7 | Forward/lateral velocity command | ×2 |
| 8 | Yaw-rate command | ×0.25 |
| 9–20 | Joint position minus default | ×1 |
| 21–32 | Joint velocity | ×0.05 |
| 33–44 | Previous raw policy action | ×0.1 |

The canonical joint order is FL, FR, RL, RR hips; then thighs; then calves.
The Isaac live trace records the actual articulation names; the MuJoCo live trace
records the controller's names and SDK mapping. The injected-state check alone
does not prove the engine supplies that order or the assumed coordinate frame.

The two depth channels are completed images. Ten raw scans accumulated into one
image and two images supplied to the estimator are different history operations.

## Dependencies and paths

Examples below start at the repository root. Use your existing Isaac Python for
the Isaac commands. Check 1 Isaac requires Torch and NumPy but does not launch
Isaac Sim. Check 1 MuJoCo needs NumPy, a C++17 compiler, and Eigen headers.

For local CPU analysis/replay, a separate environment can be created:

```bash
cd /home/lin/Jeplo/JEPLO-main
python3 -m venv diagnostics/observation_consistency/.venv
diagnostics/observation_consistency/.venv/bin/python -m pip install \
  -r diagnostics/observation_consistency/requirements.txt
```

In the commands below, `python` means a Python with the dependencies for that
check. Do not install the CPU ONNX Runtime package over an existing GPU runtime;
the separate environment avoids that conflict. The MuJoCo launcher can use your
existing `deployment/evaluation/velocity_benchmark/.venv/bin/python` instead.

Identical diagnostic files are also placed in the local `JEPLO-autodl` tree. Sync
that **new directory** to the remote repository before remote use. The examples
assume the remote repository is `/root/autodl-tmp/JEPLO-main`. No remote upload is
performed by these scripts.

### Verified AutoDL environment (2026-11-05)

The existing `/root/autodl-tmp/env_jeplo` environment was checked after repair:
Python 3.11.16, NumPy 1.26.0, Torch 2.7.0+cu128, Isaac Sim 5.1.0.0, and
Isaac Lab 2.3.2.post1. Preserve Isaac's `packaging==23.0` requirement.

Only three package versions changed to resolve dependency conflicts:

| Package | Previous | Verified |
|---|---|---|
| click | 8.5.0 | 8.1.7 |
| wheel | 0.47.0 | 0.45.1 |
| wandb | 0.30.0 | 0.26.0 |

`pip check`, imports of `isaaclab.app.AppLauncher` and the r2 CTS runner, and a
small CUDA computation passed. All 885 checked Python standard-library source
files matched their recorded checksums. This does not certify a full simulator
launch or rollout. The original package files and before/after version records
are backed up remotely at `/root/autodl-tmp/env-repair-backups/20261105-030738/`.
Installing packages changes the remote environment; Git commits and code sync
do not reproduce those installations automatically. ONNX Runtime remains absent
from this environment: run Check 2 in the separate local analysis environment.

## Check 1A: order, scaling, gravity formula and history

On AutoDL, from the remote repository root:

```bash
cd /root/autodl-tmp/JEPLO-main
/root/autodl-tmp/env_jeplo/bin/python \
  diagnostics/observation_consistency/check1_contract.py \
  --side isaac --output diagnostics/observation_consistency/runs/isaac-contract
```

Locally, in the environment with `g++` and Eigen:

```bash
cd /home/lin/Jeplo/JEPLO-main
python diagnostics/observation_consistency/check1_contract.py \
  --side mujoco --output diagnostics/observation_consistency/runs/mujoco-contract
```

After downloading the remote `isaac-contract` folder, compare it with the local
folder. Replace the first path if you downloaded it into `JEPLO-autodl`:

```bash
python diagnostics/observation_consistency/compare_contract.py \
  --isaac /home/lin/Jeplo/JEPLO-autodl/diagnostics/observation_consistency/runs/isaac-contract \
  --mujoco diagnostics/observation_consistency/runs/mujoco-contract \
  --output diagnostics/observation_consistency/runs/contract-comparison
```

Read `report.json` and `depth_frame_ids.csv`. The CSV records which synthetic
completed image occupies each channel at each control tick. Frame ID zero is
initial padding. Current expected findings are:

- Normal-range proprioception and its ten-frame history agree to float precision.
- Depth histories differ. Isaac keeps two older completed images with a one-image
  delay. The controller uses the previous control tick's latest image and the
  current latest image. These are often the same completed image.

This deterministic delivery schedule publishes immediately before every fifth
tick. It exposes the delay discrepancy even when the estimator happens to consume
two distinct frames. The live check measures your actual asynchronous phase and
duplicate frequency. Do not interpret the fixture's frequency as a live rate.

Implementation details for review: the Isaac adapter compiles the original
`Go2LocoEnv._get_observations` AST. A local Torch proxy returns 0.5 for `rand_like`,
neutralizing its additive proprioception noise only in this harness. All packing
and history operations remain the original method. The C++ adapter copies the
original gravity function, joint lookup, proprioception and history code into
`extracted_controller_check.cpp`. Its configuration fields come from the original
controller. The name-based configuration initialization is reproduced explicitly
in the harness; no SDK, model inference or physics starts.

The fixture specifies default angles and body-frame quantities; live capture is
still needed to confirm actual engine state. Extreme clipping, sensor noise and
action application are not certified by this test.

## Check 1B: depth projection, preprocessing and accumulation

This test sends exactly the same sensor-local point clouds into the real Python
and C++ projection methods. The standard 25×60 grid, 2 m range, and selected
point stride are held fixed. It compares nearest-hit binning, normalization and
minimum accumulation over ten frames. Noise, dropout and masks are disabled in
this harness only. The world-to-sensor transform is identity because the supplied
points already live in the sensor frame.

On AutoDL:

```bash
/root/autodl-tmp/env_jeplo/bin/python \
  diagnostics/observation_consistency/check1_depth.py --side isaac \
  --downsample-rate 8 --output diagnostics/observation_consistency/runs/isaac-depth
```

Locally:

```bash
python diagnostics/observation_consistency/check1_depth.py --side mujoco \
  --downsample-rate 8 --output diagnostics/observation_consistency/runs/mujoco-depth

# After downloading the Isaac output:
python diagnostics/observation_consistency/check1_depth.py \
  --compare /home/lin/Jeplo/JEPLO-autodl/diagnostics/observation_consistency/runs/isaac-depth \
            diagnostics/observation_consistency/runs/mujoco-depth \
  --output diagnostics/observation_consistency/runs/depth-comparison
```

`clean`/`clean_min` exercise ordinary returns. `edge`/`edge_min` separately exercise
range limits. With the current sources, ordinary returns agree to about 5.4e-7
normalized depth, but the exactly-0.1 m fixture differs: Python adds 1e-6 to radius
before filtering; C++ rejects radius <=0.1 m. The edge report therefore exits 1.
That finding alone does not establish an important effect on stair performance.

This comparison does not certify identical LiDAR rays, point density, scan phase,
noise, FOV selection at runtime, or sensor mounting pose. Check those recorded
settings and live outputs separately. Equal scan counts alone do not prove them.

## Check 2: identical-input model replay

Start with the ten reference cases per model already beside `my_policy_r2`:

```bash
diagnostics/observation_consistency/.venv/bin/python \
  diagnostics/observation_consistency/check2_replay.py \
  --model-dir deployment/go2_deploy/model/my_policy_r2 \
  --output diagnostics/observation_consistency/runs/r2-replay-cpu
```

The tool records model/reference hashes and reports maximum/mean error for each
output tensor. Empty reference sets, wrong shapes and nonfinite values cannot
pass. The default elementwise tolerance is `atol=1e-4, rtol=1e-4`; exit 1 means
that criterion failed. It is deliberately tighter than the existing controller's
startup validation, which uses different maximum/mean thresholds.

During implementation, CPU replay of the existing saved cases exceeded this
strict criterion: worst raw action difference was about 0.00525 and worst
estimator feature difference about 0.00304. These are numerical measurements,
not proof of an export bug or of the cause of failed stairs. Saved references may
also reflect the precision settings of the original GPU run. Use fresh Isaac
capture below to establish current provenance and investigate before changing a
tolerance. The original NPZ metadata does not identify its checkpoint hash.

For a GPU runtime installed in another environment, you can explicitly select
`--provider CUDAExecutionProvider` or `--provider TensorrtExecutionProvider`.
Unavailable requested providers are rejected. TensorRT caches go inside the
new output folder. Provider fallback for unsupported operators can still occur;
the report records the configured providers. CPU agreement does not certify the
production TensorRT path.

Fresh references require no re-export: replay the exact inputs and outputs from
a Check 3 Isaac trace using `--trace PATH_TO_TRACE` in place of the saved cases:

```bash
python diagnostics/observation_consistency/check2_replay.py \
  --model-dir deployment/go2_deploy/model/my_policy_r2 \
  --trace /path/to/downloaded/isaac/captures/standard_v0.8_trial000 \
  --output diagnostics/observation_consistency/runs/r2-fresh-reference-replay
```

Each estimator replay uses the recorded `hidden_in`, so it tests one-step
equivalence at the same recurrent state. It does not test a newly unrolled GRU
sequence or show that different observation histories are equivalent.

## Check 3A: capture actual Isaac inference

Run from the AutoDL repository root using this new launcher:

```bash
cd /root/autodl-tmp/JEPLO-main
VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json PYTHONUNBUFFERED=1 \
  /root/autodl-tmp/env_jeplo/bin/python \
  diagnostics/observation_consistency/check3_isaac.py \
  --output diagnostics/observation_consistency/runs/isaac-live \
  -- --speeds 0.8 1.0 --repeats 1 --duration 30 --friction 1.0 --seed 0 \
     --occlusion_patterns standard \
     --checkpoint logs/rsl_rl/go2_loco/my_policy_r2/model_19999.pt
```

Arguments after `--` go to the existing Isaac stair benchmark. The wrapper chooses
a separate benchmark output directory and saves captures under
`isaac-live/captures/standard_v0.8_trial000/` and `standard_v1_trial000/`.
It uses the current training observation configuration; it does not override
scan timing/history to make it match MuJoCo. Inspect `metadata.json` for actual
rates, stack size, delay, sensor pose, FOV, noise/dropout settings and joint names.

The hook tracks every actual replacement of the depth history tensor. Thus frame
IDs represent completed images; `-1` identifies reset padding. The estimator
records contain the **exact two channels it consumed**, not a publisher preview.
The `standard` benchmark retains its existing LiDAR noise and dropout behavior.

## Check 3B: capture actual MuJoCo inference

Build the separate controller inside the same Ubuntu 22.04 distrobox where your
current controller is built. Do not run the baseline `run_build.sh`: that script
replaces the baseline output directory.

```bash
distrobox enter ubuntu-22-04
cd /run/host/home/lin/Jeplo/JEPLO-main

python3 diagnostics/observation_consistency/prepare_mujoco_capture.py \
  --output diagnostics/observation_consistency/build-capture

# Review this file before building:
less diagnostics/observation_consistency/build-capture/instrumentation.diff

cmake -S diagnostics/observation_consistency/build-capture/source \
  -B diagnostics/observation_consistency/build-capture/build -DCMAKE_BUILD_TYPE=Release
cmake --build diagnostics/observation_consistency/build-capture/build \
  --target deploy --parallel 2
```

The new executable is `build-capture/build/deploy_observation_audit`. The source
generator also supports `--build` to execute those two build steps. It copies
`config.json`, r2 ONNX files and reference cases into its own build folder. It
does not link its cache directory to the baseline model directory.

Use your existing runtime library setup in that same container terminal:

```bash
export LD_LIBRARY_PATH="/home/lin/ubuntu_22_04/onnxruntime-linux-x64-gpu-1.22.0/lib:/home/lin/ubuntu_22_04/TensorRT-10.9.0.34/targets/x86_64-linux-gnu/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

deployment/evaluation/velocity_benchmark/.venv/bin/python \
  diagnostics/observation_consistency/check3_mujoco.py \
  --audit-build diagnostics/observation_consistency/build-capture \
  --output diagnostics/observation_consistency/runs/mujoco-live \
  -- --speeds 0.8 1.0 --repeats 1 --duration 30 --seed 0 \
     --lidar-hz 50 --stacked-frames 10 --downsample-rate 8 \
     --stair-contact-friction 1.0 --occlusions standard
```

The default comparison uses `standard` in both engines. In the current MuJoCo
suite that also disables the cage mask. If your older 73%/52% runs used a cage
mask, this diagnostic condition is a separate comparison and must be labeled as
such. You can pass other benchmark arguments after `--`, but the wrapper always
selects `my_policy_r2`, its separate controller, a new output directory and
retained diagnostic logs.

Trace folders are beneath
`mujoco-live/benchmark/pair001_standard_v0.8/observations/` and
`pair001_standard_v1/observations/`. Each trial also retains its exact launch
commands, model hashes, depth settings, controller log and timing diagnostics.
`runtime.json` in the trace records joint mapping, observation scales and model
shapes. The existing baseline executable remains available at its original path.

## Read the live results

Analyze each trace independently, or create one compact comparison table:

```bash
python diagnostics/observation_consistency/compare_live.py \
  --isaac /home/lin/Jeplo/JEPLO-autodl/diagnostics/observation_consistency/runs/isaac-live/captures/standard_v0.8_trial000 \
  --mujoco diagnostics/observation_consistency/runs/mujoco-live/benchmark/pair001_standard_v0.8/observations \
  --output diagnostics/observation_consistency/runs/live-comparison-v0.8
```

Read `comparison.md` first, then each side's `report.json` and `depth_timing.csv`.
You can also run `analyze_trace.py --trace DIR --output NEW_DIR` on either trace.

| Field | What to inspect |
|---|---|
| Reused depth frame fraction | Same completed image ID in both estimator channels |
| Equal depth pixels fraction | Equal content, which can also occur in a static scene |
| Frame spacing | Expected Isaac spacing is about 100 ms after startup |
| Channel ages | Isaac delay versus deployment transport/history behavior |
| Proprio history shift error | Consecutive frames should shift by exactly 45 entries |
| Residual from clean formula | Includes deliberate Isaac observation noise |
| Zero/out-of-range depth fractions | Padding, invalid data or preprocessing problems |
| Wall and simulation intervals | Scheduling and real-time-factor effects |

The two age measurements have different semantics: Isaac uses simulation time
when an image is completed; MuJoCo timestamps the publisher's receipt of the
newest raw scan. The latter is not the original LiDAR acquisition time. Startup
is included and padding is identified rather than silently discarded. Inspect
the existing MuJoCo real-time-factor logs when comparing wall time to simulation
time.

Do not interpret whole-rollout depth statistics as pixelwise scene equivalence.
For a geometry investigation, use the shared point-cloud check first, then capture
a common physical scene and sensor pose in both engines. The live captures here
do not force trajectories or ray patterns to agree. Isaac records base pose
relative to its terrain origin; the controller's DDS inputs do not contain an
equivalent base-position measurement, so this tool does not fabricate one.

## Code map and validation

| File | Responsibility |
|---|---|
| `check1_contract.py` | Inject fixtures into original state-to-observation code |
| `compare_contract.py` | Component errors and depth-frame identity timeline |
| `check1_depth.py` | Identical-cloud projection/normalization/minimum checks |
| `check2_replay.py` | ONNX replay against saved or freshly captured references |
| `check3_isaac.py` | Process-local observation/estimator/policy capture hooks |
| `prepare_mujoco_capture.py` | Isolated source generation with exact anchors and reviewable diff |
| `audit_capture.h` | C++ tensor recording used only by the generated controller |
| `check3_mujoco.py` | Separate launch command using the private controller |
| `common.py` | Trace format, hashes, strict comparison and output handling |
| `analyze_trace.py`, `compare_live.py` | Human-readable numeric summaries |
| `test_checks.py` | Recorder interoperability and observer noninterference tests |

Run the simulator-free tests with a Torch-enabled Python:

```bash
python -m unittest discover -s diagnostics/observation_consistency -p test_checks.py -v
```

Tests cover nonfinite/shape rejection, refusal to overwrite, interrupted-trace
reading, C++/Python binary compatibility, source-anchor drift, and an Isaac
observer mock that verifies unchanged actions and delayed frame IDs. The full
generated controller was syntax-checked against the local SDK/ONNX headers;
full linking and real Isaac/MuJoCo rollouts remain for your target environments.

The implementation-time numeric reports are retained in `validation/`:
`contract_report.json`, `depth_report.json`, and `r2_cpu_replay_report.json`.
These are offline measurements from the current local sources and the existing
saved r2 references, not new simulator benchmark results.

Exit codes: 0 means the requested diagnostic completed and its explicit checks
passed; 1 with a completed report means a measured mismatch or tensor issue.
A traceback or build failure is a tool/setup error, even if it also returns 1.
`compare_live.py` produces a descriptive
table, not an overall sim-to-sim pass/fail verdict. Keep the generated source,
configuration, hashes and reports with each run so later changes remain auditable.
