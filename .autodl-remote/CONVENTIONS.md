# AutoDL Remote Conventions

This file is created locally by `autodl-remote` to record project-specific remote workflow conventions.

## Runtime

- Activate command: `conda activate /root/autodl-tmp/env_jeplo`
- Python command: `/root/autodl-tmp/env_jeplo/bin/python`
- Install packages: ask before installing

## Remote Paths

- Remote root: `/root/autodl-tmp/JEPLO-main`
- Model dir: `/root/autodl-tmp/models`
- Data dir: `<fill me>`
- Output dirs: `/root/autodl-tmp/JEPLO-main/training/outputs/stair_benchmark` and `/root/autodl-tmp/JEPLO-main/training/outputs/stair_occlusion_benchmark`

## Sync Rules

- Edit code locally first, then upload selected files.
- Do not download models, datasets, checkpoints, or large outputs unless explicitly asked.
- Prefer remote `cat`, `tail`, `ls`, and `tree` before downloading files.

## Remote Execution

- Long task backend: normal detach by default; use `tmux` when live pane monitoring or multiple parallel runs are useful.

## Notes

- Run the headless r2 student stair benchmark from `training/`; see `training/scripts/BENCHMARK_STAIRS.md`.
- Current instance (2026-11-10): `autodl-clone-17738` at `connect.weste.seetacloud.com:17738`, project root `/root/autodl-tmp/JEPLO-main`. The local project binding uses this account. `autodl-clone-10292` (port 10292, current from 2026-11-05) refused SSH on 2026-11-10. The earlier `autodl-clone-11649` account is retained separately.
- 2026-11-10: Isaac stair scorer is `stair_landing_isaac_contact_proxy_v2` (7 cm foot-height cap removed to match MuJoCo) and occlusion masks are `isaac_stair_occlusion_masks_v2` (cut mirrored vertically; other masks unchanged). Originals are in `.autodl-remote/backups/20261110-scorer-cut-fix/`. Do not compare v1 and v2 success rates directly.
- Earlier clone accounts `autodl-clone-16265` and `autodl-clone` use ports `16265` and `45112`; both reset SSH on 2026-11-04.
- Previous instance remains available as the separate `autodl-gpu` account.

- Observation diagnostics: sync only `diagnostics/observation_consistency/` to the same project-relative path. Run from the repository root; outputs go under `diagnostics/observation_consistency/runs/`. Keep baseline benchmark sources, model files, and executables unchanged. The user runs the checks; do not launch simulations as part of upload verification.
- Environment verification/repair on 2026-11-05 (explicitly authorized by the user): the previously missing `urllib` files were already present when rechecked; all 885 checked standard-library `.py` files matched the Conda Python checksums. Cause of the earlier absence is unknown. Conda's older `packaging` manifest differs from the installed pip version; preserve `packaging==23.0`, required by Isaac Sim, and do not blindly restore Conda's `packaging==26.3` files.
- Repaired only `click` 8.5.0 → 8.1.7, `wheel` 0.47.0 → 0.45.1, and `wandb` 0.30.0 → 0.26.0 to satisfy Isaac Sim's Click/Packaging pins. Original package files, before/after versions, and install report: `/root/autodl-tmp/env-repair-backups/20261105-030738/`. `pip check` passes; Isaac AppLauncher, r2 CTS runner, and CUDA computation pass. No full simulator launch or benchmark was performed. Torch, NumPy, Isaac packages, policy checkpoints, and baseline sources were unchanged.
- ONNX Runtime is absent from the remote Isaac environment; Check 2 CPU replay is intended for the separate local analysis environment described in the diagnostics README. It is not required for Isaac Check 1 or live capture.
