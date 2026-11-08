# Next experiment: MuJoCo depth history

Recorded 2026-11-05; first controller change implemented locally 2026-11-06.
The policy weights are unchanged. Paired legacy/delayed trials completed on
2026-11-06; the latest-two trial is pending.

## Hypothesis

The MuJoCo controller's depth history does not match the history seen during
Isaac training. Isaac keeps three completed images with `depth_delay = 1` and
supplies the older two to the estimator. MuJoCo currently pairs the latest
published image with the image seen on the previous 50 Hz control tick. That
pair can contain the same completed image twice.

The [JEPLO paper](https://arxiv.org/html/2609.15770v1) lists a 0.1 s depth
delay in its domain-randomization table, but gives a single fixed value. The
training code sets `depth_delay = 1` with a 10 Hz depth update and does not
sample the delay. Thus this controller comparison tests the fixed training
delay against fresher received frames, not a sampled delay distribution.

The deterministic Check 1A cross-comparison found a depth-history mismatch of
0.02 normalized units at tick 41: Isaac frame IDs `[7, 8]` versus MuJoCo
`[9, 9]` (`runs/cross-20261105-contract/report.json` and `depth_frame_ids.csv`).
The live MuJoCo traces had identical IDs in 116/116 estimator calls at 0.8 m/s
and 1/126 at 1.0 m/s (`runs/mujoco-batch-20261105-032423-601523/live/benchmark/`).
The phase of the publisher and control loop therefore matters; a single live
trial cannot establish the typical duplicate rate.

## Proposed change and evaluation

1. **Implemented:** The controller defaults to `--depth-history delayed` in
   loopback simulation and retains `--depth-history legacy` for paired baseline
   trials. Hardware defaults to legacy until the change is evaluated there.
   On each received completed image, the receiver advances a three-image ring.
   Estimator channels
   0 and 1 use the older two images; the newest image is held back one frame.
   Control ticks do not advance the ring. Startup uses zero padding: after one
   received image the pair is `[0,0]`, after two it is `[0,first]`, and after
   three it is `[first,second]`. If the publisher stops, the last pair remains
   available; the estimator still runs on schedule, so it can reuse that pair
   across calls, but it does not put one image in both channels after warmup.
2. **Verified in synthetic input:** The simulator-free Check 1A comparison now
   passes all 60 ticks against the saved Isaac fixture. The `depth_history`
   tensor difference is exactly zero. The C++ history test and controller build
   also pass.
3. **Short live timing check completed:** An 8-simulation-second 0.8 m/s MuJoCo
   capture in delayed mode recorded 92 estimator calls. Reused frame IDs were
   `0/92`, exactly equal channel pairs were `0/92`, and median source-timestamp
   spacing was about 99.1 ms. The selected channels' median apparent ages were
   about 305 and 206 ms. This is older than the earlier legacy 0.8 m/s capture,
   where both duplicated channels were about 61 ms old, and numerically older
   than Isaac's 200/100 ms simulation-time ages; the clock semantics still
   differ. The trial timed out before completing stairs under the deliberately
   shortened 8 s limit, so it is not a success-rate comparison. Trace:
   `runs/local-depth-delay-live-20261106/analysis/report.json`.
   Distinct IDs alone do not prove that the pixel content is useful.

   **Why the live ages were about one 100 ms cycle older:** Matching the 91
   estimator timestamps with the publisher's `depth_timing.csv` shows that the
   estimator ran a median 6.1 ms *before* the next publication. The latest
   completed publication already available was a median 93.9 ms old. The
   delayed ring held that latest image back, so the newer image actually fed
   to the estimator had been published a median 193.9 ms earlier; its raw-scan
   receipt stamp was about 205.6 ms old. In Isaac, depth update and estimator
   invocation are synchronized on the same 100 ms simulation tick, so the
   just-completed image is held back and the fed images are about 100/200 ms
   old. This phase explains approximately 100 ms of the live age gap; source
   scan receipt versus image completion explains an additional smaller offset.
   It is a scheduling/phase observation from this one run, not a reason to
   change the selection rule without paired evaluation.
4. **Paired velocity trials completed:** With 100 valid trials per speed and
   mode, legacy succeeded in 82/100 at 0.8 m/s and 61/100 at 1.0 m/s; delayed
   succeeded in 69/100 and 65/100. Both runs used the same model, binaries,
   terrain, scan settings, and seeds. Across controller logs, legacy fed the
   same image ID in both channels in 83.2% of estimator calls at 0.8 m/s and
   85.2% at 1.0 m/s; delayed did so in 0%. Median selected newer-channel
   scan-to-estimator age across trials rose from about 59 to 166 ms at 0.8 m/s
   and 66 to 161 ms at 1.0 m/s. All trials passed the runner's timing check.
   Results: `deployment/evaluation/velocity_benchmark/runs/depth-history-legacy-20261106-151917/`
   and `deployment/evaluation/velocity_benchmark/runs/depth-history-delayed-20261106-165901/`.
   These runs show that eliminating duplicate IDs with an extra held-back frame
   did not improve success at both speeds. Distinct IDs do not establish
   distinct pixel content, and speed-specific success differences do not prove
   which policy input caused them.
5. **Next experiment implemented:** `--depth-history latest-two` advances only
   on received completed images and feeds the two most recent distinct image
   IDs, without holding the newest image back. Run it with the same benchmark
   settings and compare success and age against both completed modes. An
   8-simulation-second local smoke run recorded 0/101 duplicate-ID pairs,
   about 101 ms channel spacing, and about 49 ms selected newer-channel
   scan-to-estimator age. It timed out under that shortened limit, so it is
   not a success-rate result. Smoke output:
   `deployment/evaluation/velocity_benchmark/runs/latest-two-smoke-20261108-184238/`.

This tests policy effectiveness, not only computational efficiency. Latest-two
isolates image-ID duplication from the added one-frame delay, while differing
from the fixed delayed history used in Isaac training.

## Other diagnostic findings to discuss

1. **Check 1A, confirmed:** Normal-range proprioception components and history
   agree under the injected fixture. Depth history does not. The fixture tests
   observation construction, not live sensors or physics.
2. **Check 1B, confirmed narrow edge case:** Shared point-cloud projection and
   accumulation pass for ordinary returns (maximum normalized difference about
   `5.4e-7`). The exactly 0.1 m range-boundary case fails (maximum difference
   about `0.95`) because the filtering rules differ. Its frequency and effect
   in live scenes have not been measured. Revisit whether MuJoCo should accept
   exactly 0.1 m, or whether its lower cutoff should be removed. Treat these as
   separate candidate changes: removing the cutoff would also admit points
   below 0.1 m and needs a live near-range/false-hit check. No change yet.
3. **Check 2, older saved-case provenance remains unclear:** CPU ONNX replay of
   the ten saved cases per model fails the chosen `1e-4` elementwise tolerance
   for all 50 output comparisons. The largest policy action difference is about
   `0.00525`; the largest estimator output difference is about `0.00304`. Those
   saved references do not identify a checkpoint hash. Their file timestamps
   are close to the ONNX export files, so they may have been produced together.
   The training entry point enables CUDA TF32, which is a plausible precision
   explanation for the larger saved-case errors but has not been tested. On
   2026-11-05, replay
   using both fresh Isaac Check 3 traces and the same ONNX files passed all
   1,500 tensor comparisons (1,070 policy calls and 215 estimator calls, each
   with feature and hidden outputs). The largest fresh policy action difference
   was about `2.64e-5`; the largest estimator output difference was about
   `9.96e-6`. This supports current one-step CPU parity, while the cause of the
   older saved-case mismatch and TensorRT parity remain unverified. Fresh
   reports: `runs/cross-20261105-fresh-isaac-replay-v0.8/report.json` and
   `runs/cross-20261105-fresh-isaac-replay-v1.0/report.json`.
   **Working decision:** treat the current `policy_r2` PyTorch-to-ONNX CPU
   export as passing this one-step numerical check; do not block the next
   diagnostic discussion on the older saved cases. The older `metadata.json`
   does not prove that its references are wrong, and the failing samples do not
   by themselves prove a fault in the current policy. The replay compares the
   recorded PyTorch reference with ONNX Runtime CPU for the *same input*; it
   does not compare two ONNX runs, joint commands, or Isaac against MuJoCo.
4. **Check 3, live depth statistics differ:** Isaac estimator stacks had
   distinct IDs and 100 ms spacing in the captured trials. MuJoCo reused the
   same ID in 116/116 estimator calls at 0.8 m/s, but only 1/126 at 1.0 m/s.
   Isaac's two frames were 200 and 100 ms old in simulation time; at 1.0 m/s,
   MuJoCo's mean ages were about 120 and 21 ms using publisher-receipt wall
   timestamps. The mean fraction of pixels below the 2 m maximum was about
   95.4% in Isaac versus 71.5% and 66.0% in MuJoCo at 0.8 and 1.0 m/s.
   However, the engines did not follow identical poses or ray patterns, and
   their timestamp definitions differ. These are observations, not a
   controlled pixelwise equivalence test. Reports:
   `runs/cross-20261105-live-v0.8/comparison.md` and
   `runs/cross-20261105-live-v1.0/comparison.md`.

   **Metric interpretation:** `equal depth pixels fraction` is the fraction of
   estimator calls for which the *two whole depth channels* are exactly equal
   (`np.array_equal`), not the fraction of matching pixel positions. In the
   MuJoCo 0.8 m/s capture it was 1.0 because every pair reused one completed
   image; in the 1.0 m/s capture it was 1/126, matching the one reused-ID pair.
   This does not mean one image was held for the entire trial: successive
   estimator calls can use different completed image IDs, each duplicated
   within its own two-channel input. `depth channel spacing` is the difference
   between the two channels' source timestamps: median 0 ms for the duplicate
   0.8 m/s pairs and about 101 ms for the mostly distinct 1.0 m/s pairs.

   **Timestamp limitation:** The MuJoCo image publisher stamps an aggregate
   with the wall-clock *receipt time of its newest raw scan*, not the time the
   completed image was constructed or buffered. Pixels may include returns
   from older scans in that aggregate. Isaac reports age from image-completion
   simulation time. Thus MuJoCo's numerically smaller channel ages in these
   reports do not establish that its full images or individual pixels are
   fresher end to end. The observed 0.8 m/s pair had two copies of one image
   around 61 ms after its newest-scan receipt; the mostly distinct 1.0 m/s
   pairs had median apparent ages around 120 and 20 ms, versus Isaac's 200 and
   100 ms on its different clock.

The implemented depth-history change addresses item 1 and part of item 4. It does
not by itself resolve the range-boundary rule, old saved-case provenance,
production-provider parity, or other sensor/physics differences.
