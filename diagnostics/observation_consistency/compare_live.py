#!/usr/bin/env python3
"""Create a compact comparison table from two live traces; no pose alignment assumed."""

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
from common import HERE, new_output


def lookup(obj, path):
    for key in path.split("."):
        if not isinstance(obj, dict):
            return None
        obj = obj.get(key)
    return obj


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--isaac", type=Path, required=True)
    p.add_argument("--mujoco", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    out = new_output(a.output)
    reports = []
    for side, path in (("isaac", a.isaac), ("mujoco", a.mujoco)):
        result = subprocess.run(
            [
                sys.executable,
                str(HERE / "analyze_trace.py"),
                "--trace",
                str(path),
                "--output",
                str(out / side),
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode not in (0, 1) or not (out / side / "report.json").exists():
            raise RuntimeError(result.stderr or result.stdout)
        report = json.loads((out / side / "report.json").read_text())
        if report["side"] != side:
            raise ValueError(f"Wrong trace side supplied for {side}")
        reports.append(report)
    metrics = {
        "Policy samples": "policy_records",
        "Estimator samples": "estimator_records",
        "Reused depth frame fraction": "depth.reused_frame.mean",
        "Equal depth pixels fraction": "depth.equal_pixels.mean",
        "Depth channel spacing median (ms)": "depth.frame_spacing_ms.p50",
        "Older channel age median (ms)": "depth.frame0_age_ms.p50",
        "Newer channel age median (ms)": "depth.frame1_age_ms.p50",
        "Estimator wall interval median (ms)": "estimator_wall_interval_ms.p50",
        "Estimator simulation interval median (ms)": "estimator_sim_interval_ms.p50",
        "Proprio history shift max error": "proprio_history_shift_max_abs.max",
        "Depth zero fraction": "depth.zero_fraction.mean",
        "Depth out-of-range fraction": "depth.outside_0_1_fraction.mean",
        "Depth pixels below max range": "depth.fraction_below_max_range.mean",
    }
    rows = [
        dict(
            metric=label,
            isaac=lookup(reports[0], path),
            mujoco=lookup(reports[1], path),
        )
        for label, path in metrics.items()
    ]
    with (out / "comparison.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "isaac", "mujoco"])
        writer.writeheader()
        writer.writerows(rows)

    def fmt(v):
        return (
            "unavailable"
            if v is None
            else f"{v:.6g}" if isinstance(v, float) else str(v)
        )

    text = [
        "# Live observation comparison",
        "",
        "| Metric | Isaac | MuJoCo |",
        "|---|---:|---:|",
    ]
    text += [
        f"| {r['metric']} | {fmt(r['isaac'])} | {fmt(r['mujoco'])} |" for r in rows
    ]
    text += [
        "",
        "These are trajectory-level summaries, not an equivalence verdict. Scene, speed, pose and startup can affect depth statistics.",
        "Isaac ages use simulation image time; MuJoCo ages use publisher scan-receive wall time. They are different clock semantics.",
        "Inspect each report's source metadata, noise residuals and per-frame CSV before interpreting a difference.",
        "Use Check 1 for identical-state comparisons and Check 2 for identical-input model comparisons.",
    ]
    (out / "comparison.md").write_text("\n".join(text) + "\n")
    print(f"Comparison: {out/'comparison.md'}")


if __name__ == "__main__":
    main()
