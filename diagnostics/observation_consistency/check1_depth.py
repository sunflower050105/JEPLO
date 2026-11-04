#!/usr/bin/env python3
"""Feed identical sensor-local point clouds through both real projection methods.

Run once per side; compare resulting NPZ files with --compare. The clean fixture
checks spherical bins, row/column orientation, nearest range, divide-by-2 scaling,
and temporal minimum. Edge fixtures separately expose range/bin boundary rules.
Noise, dropout, masks, sensor pose, and the ray sampling pattern are deliberately
outside this deterministic test. Their live settings remain in Check 3 metadata.
"""

import argparse
import math
from pathlib import Path
import subprocess
from types import SimpleNamespace as NS
import numpy as np
from common import ROOT, between, error_metrics, new_output, sha256, write_json
from check1_contract import class_method

GENERATOR = "training/go2_parkour/go2_parkour/sensor/spherical_depth_generator.py"
PUBLISHER = "deployment/lidar_depth_pub/lidar_depth_pub.cpp"


def clouds():
    rows, cols = np.meshgrid(np.arange(25), np.arange(60), indexing="ij")
    theta = np.deg2rad(-4.5 + (rows + 0.15) * 2)
    phi = np.deg2rad(-60 + (cols + 0.15) * 2)
    direction = np.stack(
        [np.cos(theta) * np.cos(phi), np.cos(theta) * np.sin(phi), np.sin(theta)], -1
    )
    radius = 0.3 + rows * 0.02 + cols * 0.01
    clean = []
    for frame in range(10):
        # Both closer and farther returns hit each bin. Different frames win in
        # different pixels, so reducing over the wrong axis cannot pass.
        r = radius + ((rows + cols + frame) % 10) * 0.01
        points = (direction * r[..., None]).reshape(-1, 3)
        clean.append(np.concatenate([points, points * 1.25]).astype(np.float32))
    edge = np.array(
        [
            [0, 0, 0],
            [0.1, 0, 0],
            [0.09999, 0, 0],
            [0.10001, 0.02, 0],
            [2.0, 0, 0],
            [2.0001, 0, 0],
            [1, 0, 0.1],
            [1, 0, 0.4],
        ],
        dtype=np.float32,
    )
    # Put each edge point into its own scan so a closer return cannot hide it.
    return {"clean": np.stack(clean), "edge": edge[:, None, :]}


def isaac(repo, sets, stride):
    import torch

    project = class_method(
        repo / GENERATOR,
        "SphericalDepthGenerator",
        "_project_to_spherical_grid",
        {"torch": torch, "math_utils": NS(quat_apply_inverse=lambda q, p: p)},
    )
    process = class_method(
        repo / GENERATOR,
        "SphericalDepthGenerator",
        "_process_scan",
        {"torch": torch, "math": math},
    )
    # Identity transform because fixture points already live in the sensor frame.
    # The raycaster/world-to-sensor transform is not claimed to pass by this test.
    obj = NS(
        _get_sensor_pose_w=lambda: (torch.zeros(1, 3), torch.tensor([[1.0, 0, 0, 0]])),
        scanner=NS(data=NS()),
        num_envs=1,
        device="cpu",
        vertical_num_rays=25,
        horizontal_num_rays=60,
        _theta_low=math.radians(-4.5),
        _phi_low=math.radians(-60),
        _theta_res=math.radians(2),
        _phi_res=math.radians(2),
        max_distance=2.0,
        noise_std=0.0,
        bottom_dropout_fraction=0.0,
    )
    result = {}
    for name, scans in sets.items():
        frames = []
        for points in scans:
            obj.scanner.data.ray_hits_w = torch.tensor(points[::stride][None])
            frames.append(process(obj, project(obj))[0].numpy())
        result[name] = np.stack(frames)
        result[name + "_min"] = np.min(result[name], axis=0)
    return result


def mujoco(repo, out, sets, stride, compiler):
    source = (repo / PUBLISHER).read_text()
    project = "static void projectRawPointsToGrid(" + between(
        source, "static void projectRawPointsToGrid(", "static void runZmqSource("
    )
    normalize = "static void normalizeGrid(" + between(
        source,
        "static void normalizeGrid(",
        "// Aggregate ring buffer by taking element-wise minimum across all frames.",
    )
    accumulate = "static void accumulateMin(" + between(
        source, "static void accumulateMin(", "// ── Latency stamps"
    )
    constants = between(
        source,
        "// Vertical FOV: -4.5° to 45.5° (total 50°)",
        "// Temporal accumulation",
    )
    program = r"""
#include <algorithm>
#include <cmath>
#include <fstream>
#include <vector>
static constexpr int kGridRows=25;
static int gGridCols=60, gGridSize=1500;
@CONSTANTS@
@PROJECT@
@NORMALIZE@
@ACCUMULATE@
int main(int argc,char**argv) {
  if(argc!=6)return 2;
  int frames=std::stoi(argv[3]),points=std::stoi(argv[4]),stride=std::stoi(argv[5]);
  gPhiMinDeg=-60;gPhiMaxDeg=60;gPhiTotalDeg=120;
  gPhiMinRad=gPhiMinDeg*kDeg2Rad;gPhiResRad=gPhiTotalDeg*kDeg2Rad/gGridCols;
  std::ifstream in(argv[1],std::ios::binary);std::ofstream out(argv[2],std::ios::binary);
  std::vector<float> cloud(points*3),history(frames*gGridSize),aggregated(gGridSize);
  for(int f=0;f<frames;++f){
    in.read(reinterpret_cast<char*>(cloud.data()),cloud.size()*sizeof(float));
    float*grid=history.data()+f*gGridSize;
    projectRawPointsToGrid(cloud.data(),points,grid,stride);normalizeGrid(grid);
  }
  accumulateMin(history.data(),frames,aggregated.data());
  out.write(reinterpret_cast<const char*>(history.data()),history.size()*sizeof(float));
  out.write(reinterpret_cast<const char*>(aggregated.data()),aggregated.size()*sizeof(float));
  return in.good()&&out.good()?0:3;
}
"""
    for name, value in dict(
        CONSTANTS=constants, PROJECT=project, NORMALIZE=normalize, ACCUMULATE=accumulate
    ).items():
        program = program.replace(f"@{name}@", value)
    cpp = out / "extracted_depth_check.cpp"
    cpp.write_text(program)
    subprocess.run(
        [compiler, "-std=c++17", "-O2", str(cpp), "-o", str(out / "depth_check")],
        check=True,
    )
    result = {}
    for name, scans in sets.items():
        scans.astype("<f4").tofile(out / f"{name}_input.bin")
        subprocess.run(
            [
                str(out / "depth_check"),
                str(out / f"{name}_input.bin"),
                str(out / f"{name}_output.bin"),
                str(len(scans)),
                str(scans.shape[1]),
                str(stride),
            ],
            check=True,
        )
        values = np.fromfile(out / f"{name}_output.bin", dtype="<f4").reshape(
            -1, 25, 60
        )
        result[name], result[name + "_min"] = values[:-1], values[-1]
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--side", choices=["isaac", "mujoco"])
    mode.add_argument(
        "--compare", nargs=2, type=Path, metavar=("ISAAC_DIR", "MUJOCO_DIR")
    )
    p.add_argument("--repo", type=Path, default=ROOT)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--downsample-rate", type=int, default=8)
    p.add_argument("--compiler", default="g++")
    a = p.parse_args()
    if a.downsample_rate < 1:
        p.error("downsample-rate must be positive")
    out = new_output(a.output)
    if a.compare:
        import json

        manifests = [json.loads((d / "metadata.json").read_text()) for d in a.compare]
        if (
            manifests[0]["fixture_hash"] != manifests[1]["fixture_hash"]
            or manifests[0]["downsample_rate"] != manifests[1]["downsample_rate"]
        ):
            raise ValueError(
                "Different fixtures or downsampling: direct comparison is invalid"
            )
        x, y = [np.load(d / "depth.npz", allow_pickle=False) for d in a.compare]
        checks = {
            name: error_metrics(y[name], x[name], atol=1e-6, rtol=1e-6)
            for name in x.files
        }
        result = dict(
            checks=checks,
            passed=all(v["passed"] for v in checks.values()),
            input_metadata=manifests,
            note="Edge differences expose range/bin rules; they do not quantify the effect on policy success.",
        )
        write_json(out / "report.json", result)
        print(json.dumps(result, indent=2))
        return 0 if result["passed"] else 1
    sets = clouds()
    # Hash fixture bytes independently of NPZ ZIP timestamps.
    import hashlib

    fixture_hash = hashlib.sha256(
        b"".join(x.tobytes() for x in sets.values())
    ).hexdigest()
    result = (
        isaac(a.repo, sets, a.downsample_rate)
        if a.side == "isaac"
        else mujoco(a.repo, out, sets, a.downsample_rate, a.compiler)
    )
    np.savez_compressed(out / "depth.npz", **result)
    write_json(
        out / "metadata.json",
        dict(
            side=a.side,
            downsample_rate=a.downsample_rate,
            fixture_hash=fixture_hash,
            source_sha256=sha256(
                a.repo / (GENERATOR if a.side == "isaac" else PUBLISHER)
            ),
            grid=[25, 60],
            range_m=2,
            noise=False,
            dropout=False,
            masks=False,
            sensor_frame="identity",
            scope="Shared input cloud: projection, filtering, normalization and temporal minimum only",
        ),
    )
    print(f"Saved {out/'depth.npz'}; compare after running the other side.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
