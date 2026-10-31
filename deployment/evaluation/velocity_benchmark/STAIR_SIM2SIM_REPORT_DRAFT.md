# Draft report: stair benchmark and sim-to-sim transfer

**Date:** 31 October 2026
**Status:** Internal draft; results describe this checkpoint and test course, not the original paper's reported benchmark.

## Aim and test conditions

The current study evaluates the `my_policy_r2` student policy on a straight, ten-step staircase with 0.20 m rises, 0.30 m runs, and a top-landing success criterion. It asks two related questions: how sensitive is MuJoCo performance to stair contact friction, and how much performance changes between the Isaac Sim evaluation and the MuJoCo deployment pipeline.

The policy's **training** environment used terrain static/dynamic friction 1.0 and randomized robot-body static/dynamic friction over 0–2 at startup ([saved training configuration](/home/lin/Jeplo/JEPLO-autodl/training/logs/rsl_rl/go2_loco/my_policy_r2/params/env.yaml:279)). The separate **Isaac stair evaluation** overrides both robot and terrain friction to fixed 0.4 ([evaluation setup](/home/lin/Jeplo/JEPLO-autodl/training/scripts/benchmark_stairs.py:192)). MuJoCo's baseline has effective foot–stair sliding friction 0.4; the 1.0 diagnostic changes the stair/landing contact to effective 1.0 without changing the foot geom's stored 0.4 coefficient or the approach floor ([MuJoCo trial code](/home/lin/Jeplo/JEPLO-main/deployment/evaluation/velocity_benchmark/velocity_trial.h:225)). Equal numerical coefficients are therefore *nominally matched*, while the engines' contact models remain different.

## Results

**MuJoCo friction sensitivity, without the paper-style occlusion suite.** Each cell is successes out of 100 valid trials. Both runs used the regular benchmark depth path, commanded the same speeds, used seed 0, and allowed 30 simulated seconds. The saved runner script hash and video rendering backend differ between runs; their recorded physics, policy, and depth settings otherwise agree.

| Command | Effective stair friction 0.4 | Effective stair friction 1.0 | Difference |
|---|---:|---:|---:|
| 0.5 m/s | 40/100 | 78/100 | +38 percentage points |
| 0.7 m/s | 18/100 | 45/100 | +27 percentage points |

Sources: [0.4 summary](/home/lin/Jeplo/JEPLO-main/deployment/evaluation/velocity_benchmark/runs/stairs02_03_100pairs_no_mask/summary.csv) and [1.0 summary](/home/lin/Jeplo/JEPLO-main/deployment/evaluation/velocity_benchmark/runs/stairs02_03_100pairs_no_mask_v2/summary.csv).

**Isaac–MuJoCo comparison with four fixed depth-mask types.** Each entry is successes out of 100 valid trials at **0.8 / 1.0 m/s**. Both use the same mask generator and mask seeds, but Isaac fixed robot and terrain friction at 0.4 and allowed 16 s; MuJoCo used effective stair friction 1.0 and allowed 30 s. The final column rescales the *existing* MuJoCo outcomes to a 16 s success deadline by counting recorded successful ascents with `ascent_time_s <= 16`; it is a post hoc count, not a new simulation.

| Mask | Isaac, 16 s | MuJoCo, 30 s | MuJoCo successes by 16 s |
|---|---:|---:|---:|
| Blob | 80 / 38 | 37 / 14 | 23 / 12 |
| Cross | 85 / 41 | 40 / 12 | 32 / 9 |
| Cut | 0 / 0 | 0 / 0 | 0 / 0 |
| Stripe | 81 / 43 | 33 / 25 | 23 / 15 |

Sources: [Isaac metadata](/home/lin/Jeplo/JEPLO-autodl/training/outputs/stair_occlusion_benchmark/20261028-012732/metadata.json) and [summary](/home/lin/Jeplo/JEPLO-autodl/training/outputs/stair_occlusion_benchmark/20261028-012732/summary.json); [MuJoCo manifest](/home/lin/Jeplo/JEPLO-main/deployment/evaluation/velocity_benchmark/runs/stairs_occlusion_100pairs/manifest.json), [summary](/home/lin/Jeplo/JEPLO-main/deployment/evaluation/velocity_benchmark/runs/stairs_occlusion_100pairs/summary.csv), and [trial-level ascent times](/home/lin/Jeplo/JEPLO-main/deployment/evaluation/velocity_benchmark/runs/stairs_occlusion_100pairs/results.csv).

A separate clean-depth Isaac run reported 85/100 at 0.8 m/s and 41/100 at 1.0 m/s, but used a 25 s deadline; it is not pooled with the 16 s mask suite ([clean-run metadata](/home/lin/Jeplo/JEPLO-autodl/training/outputs/stair_benchmark/20261027-040559/metadata.json), [summary](/home/lin/Jeplo/JEPLO-autodl/training/outputs/stair_benchmark/20261027-040559/summary.json)).

## Interpretation and limits

Increasing effective stair friction substantially improved MuJoCo success at the two speeds where a paired friction comparison exists (0.5 and 0.7 m/s). At the same 1.0 m/s command and mask, Isaac's success rates exceed MuJoCo's for blob, cross, and stripe, even though the recorded MuJoCo test used a longer deadline. This is evidence of a **sim-to-sim plus deployment-pipeline performance gap** for this policy and course. The available runs do not isolate the cause: friction semantics, contact solving, depth and inference paths, and the landing contact tests differ. Isaac uses a foot-force-based contact proxy; MuJoCo inspects geom contacts and normals.

The existing MuJoCo 0.4 runs are at 0.5/0.7 m/s, whereas the 100-trial Isaac and MuJoCo mask suites are at 0.8/1.0 m/s. Thus there is not yet a same-speed, same-mask, nominal-0.4 cross-engine dataset in the saved results. The apparent similarity between Isaac at 1.0 m/s and MuJoCo at 0.8 m/s also weakens when MuJoCo successes are counted by 16 s. The fixed 0.4 Isaac evaluation is distinct from the randomized-friction training distribution. Finally, these masks approximate the paper's four illustrations; their exact parameters were not published.

## Next comparison for a paper

Run MuJoCo at **0.4** effective stair friction with 0.8 and 1.0 m/s commands, the same masks and seeds, and a 16 s deadline; include an unmasked `standard` control in both engines. Retain the 1.0 MuJoCo condition as a clearly labeled friction sensitivity analysis. Before attributing any remaining gap to the physics engine, verify that the deployed ONNX policy matches the Isaac checkpoint's inference on identical observations and document the residual depth, actuator, and success-rule differences.
