#!/usr/bin/env python3
"""Execute baseline observation code on deterministic fixtures, without simulation.

Isaac: compile the actual _get_observations method from its AST and supply mock
sensor/robot data. Only random proprioception noise is neutralized (rand_like=.5).
MuJoCo: compile the actual C++ packing/history snippets in a stand-alone harness.
Generated source is retained for review. Neither adapter edits baseline files.

This checks the state-to-tensor transformation. It does NOT prove that either
engine's IMU, articulation joint order, or raycaster supplies the assumed state;
the live capture records those actual inputs for the next check.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
from pathlib import Path
import subprocess
from types import SimpleNamespace as NS
import numpy as np

from common import (
    ROOT,
    JOINTS,
    COMPONENTS,
    between,
    error_metrics,
    new_output,
    sha256,
    write_json,
)

ENV = "training/go2_parkour/go2_parkour/tasks/go2_loco_env.py"
CFG = "training/go2_parkour/go2_parkour/tasks/go2_loco_env_cfg.py"
DEPLOY = "deployment/go2_deploy/deploy/src/deploy.cpp"


def class_method(path, class_name, method_name, globals_dict):
    tree = ast.parse(path.read_text())
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name
    )
    method = next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method_name
    )
    module = ast.Module(body=[method], type_ignores=[])
    exec(compile(module, str(path), "exec"), globals_dict)
    return globals_dict[method_name]


def config_literals(path):
    tree = ast.parse(path.read_text())
    names = {
        "num_obs_hist",
        "num_depth_stack",
        "depth_delay",
        "depth_update_freq",
        "obs_proprio_dim",
        "lidar_num_stacked",
    }
    values = {}
    for n in ast.walk(tree):
        if isinstance(n, (ast.Assign, ast.AnnAssign)):
            target = n.targets[0] if isinstance(n, ast.Assign) else n.target
            if isinstance(target, ast.Name) and target.id in names:
                values[target.id] = ast.literal_eval(n.value)
    if values.keys() != names:
        raise ValueError(
            "Training config changed: cannot resolve observation constants"
        )
    return values


def fixture():
    """Nonzero, asymmetric values expose swaps that a neutral standing pose hides."""
    defaults = np.array(
        [0.1, -0.1, 0.1, -0.1, 0.8, 0.8, 1, 1, -1.5, -1.5, -1.5, -1.5], np.float32
    )
    rows = []
    for step in range(60):
        axis = np.eye(3)[(step // 10) % 3]
        angle = (0.15 + 0.01 * step) * (-1 if step % 2 else 1)
        quat = np.r_[math.cos(angle / 2), axis * math.sin(angle / 2)].astype(np.float32)
        w, x, y, z = quat.astype(np.float64)
        gravity = np.array(
            [2 * (w * y - x * z), -2 * (y * z + w * x), 1 - 2 * (w * w + z * z)],
            np.float32,
        )
        gyro = np.array([0.13, -0.27, 0.39], np.float32) + step * 0.001
        command = np.array([0.8, -0.12, 0.21], np.float32)
        q = defaults + np.arange(1, 13, dtype=np.float32) * 0.007 + step * 0.0001
        dq = np.arange(1, 13, dtype=np.float32) * -0.11 + step * 0.002
        action = np.arange(1, 13, dtype=np.float32) * 0.03 - 0.2 + step * 0.001
        # Unique completed image every five 50 Hz ticks; encode its ID as depth.
        rows.append(
            dict(
                step=step,
                quat=quat.tolist(),
                gravity=gravity.tolist(),
                gyro=gyro.tolist(),
                command=command.tolist(),
                q=q.tolist(),
                dq=dq.tolist(),
                action=action.tolist(),
                depth_id=step // 5 + 1,
            )
        )
    return dict(
        version=1,
        joints=JOINTS,
        defaults=defaults.tolist(),
        rows=rows,
        note="Synthetic fixtures, radians/rad/s; input depth is already aggregated.",
    )


def expected_proprio(f):
    return np.array(
        [
            np.r_[
                np.array(r["gyro"]) * 0.25,
                r["gravity"],
                np.array(r["command"]) * [2, 2, 0.25],
                np.array(r["q"]) - f["defaults"],
                np.array(r["dq"]) * 0.05,
                np.array(r["action"]) * 0.1,
            ]
            for r in f["rows"]
        ],
        dtype=np.float32,
    )


def run_isaac(repo, out, f):
    import torch

    cfg = config_literals(repo / CFG)
    if (
        cfg["num_obs_hist"] != 10
        or cfg["num_depth_stack"] != 2
        or cfg["obs_proprio_dim"] != 45
    ):
        raise ValueError(
            "This audit is scoped to my_policy_r2: 10x45 proprio and 2x25x60 depth"
        )

    class QuietTorch:
        # The production method adds (rand_like * 2 - 1) * noise_buf.
        def rand_like(self, tensor):
            return torch.full_like(tensor, 0.5)

        def __getattr__(self, name):
            return getattr(torch, name)

    method = class_method(
        repo / ENV, "Go2LocoEnv", "_get_observations", {"torch": QuietTorch()}
    )
    zeros = lambda *shape: torch.zeros(*shape, dtype=torch.float32)
    sensor = NS(update=lambda step: None, get_depth_image=lambda: zeros(1, 25, 60))
    data = NS(
        root_pos_w=zeros(1, 3),
        root_com_lin_vel_b=zeros(1, 3),
        body_pos_w=zeros(1, 4, 3),
        applied_torque=zeros(1, 12),
        default_joint_pos=torch.tensor([f["defaults"]]),
    )
    env = NS(
        cfg=NS(**cfg, is_play_env=False, observation_space=845),
        num_envs=1,
        device="cpu",
        _num_actions=12,
        _feet_ids_cs=[0, 1, 2, 3],
        _feet_ids_bd=[0, 1, 2, 3],
        _contact_sensor=NS(data=NS(net_forces_w=zeros(1, 19, 3))),
        _height_scanner=NS(data=NS(ray_hits_w=zeros(1, 300, 3))),
        _robot=NS(
            data=data,
            actuators={"GO2HV": NS(stiffness=zeros(1, 12), damping=zeros(1, 12))},
        ),
        _spherical_depth_gen=sensor,
        _flat_terrain_mask=torch.tensor([True]),
        _obs_buffer=zeros(1, 10, 45),
        _depth_buffer=zeros(1, 2 + cfg["depth_delay"], 25, 60),
        _physics_parameters=zeros(1, 3),
        _delta_masses=zeros(1, 1),
    )
    histories, depths = [], []
    for row in f["rows"]:
        data.root_com_ang_vel_b = torch.tensor([row["gyro"]])
        data.projected_gravity_b = torch.tensor([row["gravity"]])
        data.joint_pos, data.joint_vel = torch.tensor([row["q"]]), torch.tensor(
            [row["dq"]]
        )
        env._cmd_lin_vel, env._cmd_ang_vel = torch.tensor(
            [row["command"][:2]]
        ), torch.tensor([[row["command"][2]]])
        env._actions = torch.tensor([row["action"]])
        env.common_step_counter = row["step"]
        sensor.get_depth_image = lambda r=row: torch.full(
            (1, 25, 60), r["depth_id"] / 100
        )
        observations = method(env)
        histories.append(observations["policy"][0, :450].numpy().copy())
        depths.append(observations["depth"][0].numpy().copy())
    write_json(out / "effective_config.json", cfg)
    return np.stack(histories), np.stack(depths)


def run_mujoco(repo, out, f, compiler, eigen_include):
    source = (repo / DEPLOY).read_text()
    config = (
        "struct LocomotionConfig {"
        + between(source, "struct LocomotionConfig {", "    void init() {")
        + "};"
    )
    gravity = "Vector3f get_gravity_orientation_eigen" + between(
        source, "Vector3f get_gravity_orientation_eigen", "uint32_t crc32_core"
    )
    pack = between(
        source,
        "        // Build proprioception observation (45 dims)",
        "        // --- Refresh depth buffer every step (policy consumes depth at 50 Hz) ---",
    )
    depth = between(
        source,
        "        // --- Refresh depth buffer every step (policy consumes depth at 50 Hz) ---",
        "        // --- Sensor estimator step (every wm_update_interval steps) ---",
    )
    mapping = between(
        source,
        "        // Update joint positions and velocities",
        "        // Get IMU state",
    )
    harness = r"""
#include <algorithm>
#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include "json.hpp"
using namespace Eigen;
using json = nlohmann::json;
@CONFIG@
@GRAVITY@
struct Motor {float position, velocity; float q() {return position;} float dq() {return velocity;}};
struct LowState {std::array<Motor,12> motors; auto &motor_state() {return motors;}};
int main(int argc, char **argv) {
  if (argc != 3) return 2;
  std::ifstream input(argv[1]); json fixture; input >> fixture;
  std::ofstream output(argv[2], std::ios::binary);
  LocomotionConfig config_;
  config_.default_angles.resize(12);
  // Same name-based setup as LocomotionConfig::init, without model/SDK setup.
  for (int i=0;i<12;++i) {
    config_.joint2motor_idx[i] = std::find(config_.motor_joint_names.begin(), config_.motor_joint_names.end(),
        config_.sim_joint_names[i])-config_.motor_joint_names.begin();
    config_.default_angles[i] = config_.default_joint_angles.at(config_.sim_joint_names[i]);
  }
  VectorXf qj_(12), dqj_(12); Vector3f cmd_;
  LowState low_state_;
  std::vector<float> proprio_obs_(45), prop_hist_(450), raw_action_(12), depth_buffer_(3000);
  std::vector<float> prev_depth_frame_(1500), latest_depth_frame_(1500);
  bool depth_prev_initialized_=false;
  uint64_t prev_depth_scan_ns_=0, latest_depth_scan_ns_=0, prev_depth_seq_=0, latest_depth_seq_=0;
  uint64_t estimator_prev_scan_ns_=0, estimator_prev_seq_=0, estimator_latest_seq_=0;
  uint64_t estimator_depth_scan_ns_=0, estimator_depth_publish_ns_=0, estimator_depth_recv_ns_=0;
  uint64_t latest_depth_publish_ns_=0, latest_depth_recv_ns_=0;
  std::mutex depth_mutex_;
  std::vector<std::string> sdk={"FR_hip_joint","FR_thigh_joint","FR_calf_joint","FL_hip_joint",
      "FL_thigh_joint","FL_calf_joint","RR_hip_joint","RR_thigh_joint","RR_calf_joint",
      "RL_hip_joint","RL_thigh_joint","RL_calf_joint"};
  auto names=fixture["joints"].get<std::vector<std::string>>();
  for (const auto &r:fixture["rows"]) {
    for (int i=0;i<12;++i) {
      auto canonical=std::find(names.begin(),names.end(),sdk[i])-names.begin();
      low_state_.motors[i]={r["q"][canonical].get<float>(),r["dq"][canonical].get<float>()};
      raw_action_[i]=r["action"][i].get<float>();
    }
    @MAPPING@
    auto q=r["quat"].get<std::vector<float>>(); Quaternionf quat(q[0],q[1],q[2],q[3]);
    auto gravity_orientation=get_gravity_orientation_eigen(quat);
    auto ang_vel=r["gyro"].get<std::array<float,3>>();
    for(int i=0;i<3;++i) cmd_[i]=r["command"][i].get<float>();
    latest_depth_seq_=r["depth_id"].get<uint64_t>();
    std::fill(latest_depth_frame_.begin(),latest_depth_frame_.end(),latest_depth_seq_/100.f);
    @PACK@
    @DEPTH@
    output.write(reinterpret_cast<const char*>(prop_hist_.data()),450*sizeof(float));
    output.write(reinterpret_cast<const char*>(depth_buffer_.data()),3000*sizeof(float));
  }
  return output.good()?0:3;
}
"""
    for name, snippet in dict(
        CONFIG=config, GRAVITY=gravity, MAPPING=mapping, PACK=pack, DEPTH=depth
    ).items():
        harness = harness.replace(f"@{name}@", snippet)
    cpp = out / "extracted_controller_check.cpp"
    cpp.write_text(
        "// Generated from "
        + str(repo / DEPLOY)
        + "\n// SHA256 "
        + sha256(repo / DEPLOY)
        + "\n"
        + harness
    )
    command = [
        compiler,
        "-std=c++17",
        "-O2",
        "-I" + str(eigen_include),
        "-I" + str(repo / "deployment/go2_deploy/deploy/include"),
        str(cpp),
        "-o",
        str(out / "contract_check"),
    ]
    write_json(out / "compile_command.json", command)
    subprocess.run(command, check=True)
    subprocess.run(
        [
            str(out / "contract_check"),
            str(out / "fixtures.json"),
            str(out / "actual.bin"),
        ],
        check=True,
    )
    actual = np.fromfile(out / "actual.bin", dtype=np.float32).reshape(
        len(f["rows"]), 3450
    )
    return actual[:, :450], actual[:, 450:].reshape(-1, 2, 25, 60)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--side", choices=["isaac", "mujoco"], required=True)
    p.add_argument("--repo", type=Path, default=ROOT)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--compiler", default="g++")
    p.add_argument("--eigen-include", type=Path, default=Path("/usr/include/eigen3"))
    a = p.parse_args()
    repo = a.repo.resolve()
    out = new_output(a.output)
    f = fixture()
    write_json(out / "fixtures.json", f)
    hist, depth = (
        run_isaac(repo, out, f)
        if a.side == "isaac"
        else run_mujoco(repo, out, f, a.compiler, a.eigen_include)
    )
    expected = expected_proprio(f)
    reference_hist = np.zeros_like(hist)
    for i in range(len(hist)):
        for lag in range(min(i + 1, 10)):
            reference_hist[i, lag * 45 : (lag + 1) * 45] = expected[i - lag]
    checks = {
        name: error_metrics(hist[:, start:end], expected[:, start:end])
        for name, (start, end) in COMPONENTS.items()
    }
    checks["proprio_history_newest_first"] = error_metrics(hist, reference_hist)
    np.savez_compressed(out / "observations.npz", proprio_hist=hist, depth_stack=depth)
    paths = [ENV, CFG] if a.side == "isaac" else [DEPLOY]
    report = dict(
        side=a.side,
        fixture_sha256=sha256(out / "fixtures.json"),
        sources={s: sha256(repo / s) for s in paths},
        checks=checks,
        passed=all(v["passed"] for v in checks.values()),
        scope="Packing on injected state; engine sensor frames/order still require live verification.",
    )
    write_json(out / "report.json", report)
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
