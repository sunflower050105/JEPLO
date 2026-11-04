#!/usr/bin/env python3
"""Summarize actual consumed tensors, timing, history and noise without raw dumps.

Reports distributions for a single run. Comparing two reports is useful, but
pixelwise comparisons between unrelated live trajectories are intentionally not
performed. Frame IDs detect reuse even when a static scene has equal pixels.
"""

import argparse
import csv
import json
from pathlib import Path
import numpy as np
from common import COMPONENTS, new_output, read_trace, write_json


def stats(values):
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    if not a.size:
        return None
    return dict(
        n=int(a.size),
        mean=float(a.mean()),
        min=float(a.min()),
        p50=float(np.median(a)),
        p95=float(np.quantile(a, 0.95)),
        max=float(a.max()),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trace", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    out = new_output(a.output)
    metadata = json.loads((a.trace / "metadata.json").read_text())
    if (a.trace / "runtime.json").is_file():
        metadata["runtime"] = json.loads((a.trace / "runtime.json").read_text())
    policy = []
    estimator = []
    current = []
    noise = {key: [] for key in COMPONENTS}
    shift_errors = []
    previous = None
    issues = []
    depth_rows = []
    for meta, arrays in read_trace(a.trace):
        if any(not np.isfinite(v).all() for v in arrays.values()):
            issues.append(
                dict(step=meta["step"], kind=meta["kind"], issue="nonfinite tensor")
            )
        if meta["kind"] == "policy":
            policy.append(meta)
            prop = arrays["proprio_hist"][0]
            current.append(prop[:45])
            if previous is not None and meta["step"] == previous[0] + 1:
                shift_errors.append(
                    float(np.max(np.abs(prop[45:] - previous[1][:-45])))
                )
            previous = (meta["step"], prop.copy())
            if all(
                k in arrays
                for k in (
                    "gyro",
                    "command",
                    "joint_pos",
                    "joint_vel",
                    "default_joint_pos",
                    "previous_action",
                    "quaternion_wxyz",
                )
            ):
                w, x, y, z = arrays["quaternion_wxyz"][0].astype(float)
                gravity = np.array(
                    [2 * (w * y - x * z), -2 * (z * y + w * x), 1 - 2 * (w * w + z * z)]
                )
                expected = np.r_[
                    arrays["gyro"][0] * 0.25,
                    gravity,
                    arrays["command"][0] * [2, 2, 0.25],
                    arrays["joint_pos"][0] - arrays["default_joint_pos"][0],
                    arrays["joint_vel"][0] * 0.05,
                    arrays["previous_action"][0] * 0.1,
                ]
                for name, (start, end) in COMPONENTS.items():
                    noise[name].extend((prop[start:end] - expected[start:end]).tolist())
        elif meta["kind"] == "estimator":
            estimator.append(meta)
            depth = arrays["depth_stack"]
            if depth.shape != (1, 2, 25, 60):
                issues.append(
                    dict(
                        step=meta["step"],
                        issue="unexpected r2 depth shape",
                        shape=list(depth.shape),
                    )
                )
                continue
            ids = meta.get("depth_frame_ids", [-1, -1])
            id_known = (
                all(i >= 0 for i in ids)
                if metadata["side"] == "isaac"
                else all(i > 0 for i in ids)
            )
            times = meta.get("depth_frame_times_s")
            if times is not None:
                clock = meta["sim_time_s"]
            else:
                clock = meta["wall_time_ns"] * 1e-9
                times = [
                    t * 1e-9 if t else None
                    for t in meta.get("depth_frame_times_ns", [0, 0])
                ]
            ages = [(clock - t) * 1000 if t is not None else None for t in times]
            spacing = (
                (times[1] - times[0]) * 1000
                if all(t is not None for t in times)
                else None
            )
            d = depth[0]
            depth_rows.append(
                dict(
                    step=meta["step"],
                    frame0=ids[0],
                    frame1=ids[1],
                    reused_frame=int(ids[0] == ids[1]) if id_known else None,
                    equal_pixels=int(np.array_equal(d[0], d[1])),
                    frame0_age_ms=ages[0],
                    frame1_age_ms=ages[1],
                    frame_spacing_ms=spacing,
                    channel_mean_abs=float(np.abs(d[0] - d[1]).mean()),
                    fraction_below_max_range=float((d < 1).mean()),
                    zero_fraction=float((d == 0).mean()),
                    outside_0_1_fraction=float(((d < 0) | (d > 1)).mean()),
                    p10_m=float(np.quantile(d, 0.1) * 2),
                    median_m=float(np.median(d) * 2),
                    p90_m=float(np.quantile(d, 0.9) * 2),
                )
            )
    if not policy or not estimator:
        raise ValueError(
            "Trace must contain policy AND estimator records; capture may have failed before inference"
        )
    with (out / "depth_timing.csv").open("w", newline="") as f:
        if depth_rows:
            writer = csv.DictWriter(f, fieldnames=list(depth_rows[0]))
            writer.writeheader()
            writer.writerows(depth_rows)

    def intervals(rows, field, scale):
        values = [r[field] * scale for r in rows if field in r]
        return stats(np.diff(values))

    tensor_stats = {}
    current = np.stack(current)
    for name, (s, e) in COMPONENTS.items():
        tensor_stats[name] = dict(
            values=stats(current[:, s:e].ravel()),
            residual_from_clean_formula=stats(noise[name]),
        )
    result = dict(
        side=metadata["side"],
        policy_records=len(policy),
        estimator_records=len(estimator),
        proprio_components=tensor_stats,
        proprio_history_shift_max_abs=stats(shift_errors),
        policy_wall_interval_ms=intervals(policy, "wall_time_ns", 1e-6),
        estimator_wall_interval_ms=intervals(estimator, "wall_time_ns", 1e-6),
        policy_sim_interval_ms=intervals(policy, "sim_time_s", 1000),
        estimator_sim_interval_ms=intervals(estimator, "sim_time_s", 1000),
        depth={
            key: stats([r[key] for r in depth_rows if r[key] is not None])
            for key in (
                "reused_frame",
                "equal_pixels",
                "frame0_age_ms",
                "frame1_age_ms",
                "frame_spacing_ms",
                "channel_mean_abs",
                "fraction_below_max_range",
                "zero_fraction",
                "outside_0_1_fraction",
                "p10_m",
                "median_m",
                "p90_m",
            )
        },
        issues=issues,
        source_metadata=metadata,
        interpretation=[
            "Reused-frame mean is the fraction of estimator calls reusing the same frame in both channels.",
            "Equal pixels alone do not establish frame reuse; a static scene can produce equal frames.",
            "Isaac depth ages use simulation image time. MuJoCo ages use publisher scan-receive wall time.",
            "fraction_below_max_range counts values <1; noise may turn a max-range return into <1.",
            "Clean-formula residual includes intended Isaac noise; it is not automatically a packing error.",
            "This is not an overall sim-to-sim PASS/FAIL or a success-rate evaluation.",
        ],
    )
    write_json(out / "report.json", result)
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in ("source_metadata", "proprio_components")
            },
            indent=2,
        )
    )
    print(f"Full report: {out/'report.json'}")
    return 1 if issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
