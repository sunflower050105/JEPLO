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
- Output dir: `/root/autodl-tmp/JEPLO-main/training/outputs/stair_benchmark`

## Sync Rules

- Edit code locally first, then upload selected files.
- Do not download models, datasets, checkpoints, or large outputs unless explicitly asked.
- Prefer remote `cat`, `tail`, `ls`, and `tree` before downloading files.

## Remote Execution

- Long task backend: normal detach by default; use `tmux` when live pane monitoring or multiple parallel runs are useful.

## Notes

- Run the headless r2 student stair benchmark from `training/`; see `training/scripts/BENCHMARK_STAIRS.md`.
- The currently bound instance reported `torch.cuda.is_available() == False` on 2026-10-25, so use the cloned GPU instance for a physics smoke test and full run.
