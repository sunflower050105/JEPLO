#!/usr/bin/env python3
"""Run this machine's diagnostic stages sequentially, with separate logs/results.

Isaac collects Check 1 and Check 3; ONNX Check 2 runs on the MuJoCo/local side.
No cross-simulator comparison is possible until both machines' outputs are local.
This launcher changes no baseline files and never installs into the Isaac env.
"""

import argparse
from datetime import datetime
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def classify_result(code, report=None, analysis=False):
    """Only an explicitly completed failed comparison may continue after exit 1."""
    if code == 0:
        return "completed"
    if code == 1 and report is not None and report.is_file():
        try:
            data = json.loads(report.read_text())
            if data.get("passed") is False:
                return "mismatch"
            if analysis and isinstance(data.get("issues"), list) and data["issues"]:
                return "mismatch"
        except (OSError, ValueError, AttributeError):
            pass
    return "error"


class Batch:
    def __init__(self, out, side, dry_run):
        self.out = out
        self.dry_run = dry_run
        self.env = dict(os.environ, PYTHONUNBUFFERED="1")
        if side == "isaac":
            self.env.setdefault("VK_ICD_FILENAMES", "/etc/vulkan/icd.d/nvidia_icd.json")
        self.summary = {"side": side, "status": "running", "stages": []}
        self.save()

    def save(self):
        destination = self.out / "summary.json"
        temporary = self.out / "summary.json.tmp"
        temporary.write_text(json.dumps(self.summary, indent=2) + "\n")
        temporary.replace(destination)

    def run(self, name, command, report=None, analysis=False):
        command = [str(x) for x in command]
        log = self.out / f"{len(self.summary['stages']) + 1:02d}-{name}.log"
        stage = dict(name=name, command=command, log=str(log), status="planned")
        self.summary["stages"].append(stage)
        self.save()
        print(f"[START] {name}: {shlex.join(command)}", flush=True)
        if self.dry_run:
            return
        stage["status"] = "running"
        self.save()
        print(f"[LOG] {log}", flush=True)
        with log.open("w") as stream:
            result = subprocess.run(command, cwd=ROOT, env=self.env,
                                    stdout=stream, stderr=subprocess.STDOUT)
        stage["returncode"] = result.returncode
        stage["status"] = classify_result(result.returncode, report, analysis)
        self.save()
        print(f"[{stage['status'].upper()}] {name}", flush=True)
        if stage["status"] == "error":
            # Preserve the full log; print just enough context to diagnose a failure.
            with log.open(errors="replace") as stream:
                from collections import deque
                print("".join(deque(stream, maxlen=30)), flush=True)
            raise RuntimeError(f"{name} failed; see {log}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--side", choices=["isaac", "mujoco"], required=True)
    p.add_argument("--output", type=Path, help="New directory; default uses timestamp")
    p.add_argument("--speeds", nargs="+", type=float, default=[0.8, 1.0])
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--duration", type=float, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--friction", type=float, default=1.0)
    replay = p.add_mutually_exclusive_group()
    replay.add_argument("--setup-replay", action="store_true",
                        help="MuJoCo only: create/install a private replay venv inside this batch")
    replay.add_argument("--replay-python", type=Path,
                        help="MuJoCo only: existing CPU ONNX analysis Python")
    p.add_argument("--audit-build", type=Path,
                   help="MuJoCo only: reuse a diagnostic build; otherwise generate and build one")
    p.add_argument("--dry-run", action="store_true",
                   help="Save/print commands without running them or installing packages")
    a = p.parse_args()
    if (a.repeats < 1 or not math.isfinite(a.duration) or a.duration <= 0
            or not math.isfinite(a.friction) or a.friction < 0
            or any(not math.isfinite(v) or v <= 0 for v in a.speeds)):
        p.error("Use positive speeds, duration and repeats, and nonnegative friction")
    if a.side == "isaac" and (a.setup_replay or a.replay_python or a.audit_build):
        p.error("Replay setup and controller build options apply only to MuJoCo")
    out = (a.output or HERE / "runs" /
           f"{a.side}-batch-{datetime.now():%Y%m%d-%H%M%S-%f}").resolve()
    out.mkdir(parents=True, exist_ok=False)
    batch = Batch(out, a.side, a.dry_run)
    batch.summary["check2_location"] = "local MuJoCo machine"
    python = sys.executable

    def tool(name, *args, interpreter=python):
        return [interpreter, HERE / name, *args]

    try:
        if a.side == "mujoco":
            replay_python = (a.replay_python or HERE / ".venv/bin/python").absolute()
            if a.setup_replay:
                replay_python = out / "replay-env/bin/python"
                batch.run("create-replay-env", [python, "-m", "venv", out / "replay-env"])
                batch.run("install-replay-deps", [replay_python, "-m", "pip", "install",
                                                 "-r", HERE / "requirements.txt"])
            batch.run("preflight-replay", [replay_python, "-c", "import numpy, onnxruntime"])
        batch.run("check1-contract", tool("check1_contract.py", "--side", a.side,
                  "--output", out / "contract"), out / "contract/report.json")
        batch.run("check1-depth", tool("check1_depth.py", "--side", a.side,
                  "--downsample-rate", "8", "--output", out / "depth"))
        shared = ["--speeds", *map(str, a.speeds), "--repeats", str(a.repeats),
                  "--duration", str(a.duration), "--seed", str(a.seed)]
        if a.side == "mujoco":
            batch.run("check2-replay", tool("check2_replay.py", "--model-dir",
                      ROOT / "deployment/go2_deploy/model/my_policy_r2", "--output",
                      out / "replay", interpreter=replay_python), out / "replay/report.json")
            build = a.audit_build.resolve() if a.audit_build else out / "controller"
            if not a.audit_build:
                batch.run("build-capture", tool("prepare_mujoco_capture.py", "--output", build, "--build"))
            batch.run("check3-live", tool("check3_mujoco.py", "--audit-build", build,
                      "--output", out / "live", "--", *shared, "--stair-contact-friction",
                      str(a.friction), "--lidar-hz", "50", "--stacked-frames", "10",
                      "--downsample-rate", "8", "--occlusions", "standard"))
            traces = sorted((out / "live/benchmark").glob("*/observations/metadata.json"))
        else:
            batch.run("check3-live", tool("check3_isaac.py", "--output", out / "live",
                      "--", *shared, "--friction", str(a.friction), "--headless",
                      "--device", "cuda:0", "--occlusion_patterns", "standard",
                      "--checkpoint", ROOT / "training/logs/rsl_rl/go2_loco/my_policy_r2/model_19999.pt"))
            traces = sorted((out / "live/captures").glob("*/metadata.json"))
        if not a.dry_run and not traces:
            raise RuntimeError("Live stage produced no trace metadata; inspect the live log")
        for index, metadata in enumerate(traces):
            analysis = out / "analysis" / f"trace-{index:03d}"
            batch.run(f"analyze-{index:03d}", tool("analyze_trace.py", "--trace",
                      metadata.parent, "--output", analysis), analysis / "report.json", analysis=True)
        mismatch = any(s["status"] == "mismatch" for s in batch.summary["stages"])
        batch.summary["status"] = "planned" if a.dry_run else (
            "completed_with_mismatches" if mismatch else "completed")
        batch.save()
        print(f"[DONE] {batch.summary['status']}; results: {out}", flush=True)
        return 1 if mismatch else 0
    except (Exception, KeyboardInterrupt) as exc:
        batch.summary["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "error"
        batch.summary["error"] = str(exc)
        batch.save()
        print(f"[ERROR] {exc}; results: {out}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
