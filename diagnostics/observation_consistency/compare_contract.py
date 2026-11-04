#!/usr/bin/env python3
"""Compare Check 1 outputs. A depth-history mismatch is a measured result, exit 1."""

import argparse
import csv
import json
from pathlib import Path
import numpy as np
from common import COMPONENTS, error_metrics, new_output, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--isaac", type=Path, required=True)
    p.add_argument("--mujoco", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    reports = [json.loads((d / "report.json").read_text()) for d in (a.isaac, a.mujoco)]
    if reports[0]["fixture_sha256"] != reports[1]["fixture_sha256"]:
        raise ValueError("Fixtures differ; direct comparison is invalid")
    x, y = [
        np.load(d / "observations.npz", allow_pickle=False) for d in (a.isaac, a.mujoco)
    ]
    out = new_output(a.output)
    checks = {
        name: error_metrics(y["proprio_hist"][:, s:e], x["proprio_hist"][:, s:e])
        for name, (s, e) in COMPONENTS.items()
    }
    checks["proprio_history"] = error_metrics(y["proprio_hist"], x["proprio_hist"])
    checks["depth_history"] = error_metrics(y["depth_stack"], x["depth_stack"])
    with (out / "depth_frame_ids.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "tick",
                "estimator_tick",
                "isaac_old",
                "isaac_new",
                "mujoco_old",
                "mujoco_new",
            ]
        )
        for i, (dx, dy) in enumerate(zip(x["depth_stack"], y["depth_stack"])):
            writer.writerow(
                [
                    i,
                    i % 5 == 0,
                    *np.rint(dx[:, 0, 0] * 100).astype(int),
                    *np.rint(dy[:, 0, 0] * 100).astype(int),
                ]
            )
    result = dict(
        checks=checks,
        passed=all(v["passed"] for v in checks.values()),
        input_reports=reports,
        note="Zero frame ID is initial padding. Synthetic completed frames arrive every five control ticks.",
    )
    write_json(out / "report.json", result)
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
