#!/usr/bin/env python3
"""Generate a separate instrumented controller tree. Baseline sources stay read-only.

Each insertion requires an exact unique source anchor. The generated diff and
hashes are saved for review. Model files/test cases are COPIED: TensorRT caches
therefore belong to this diagnostic build, never the baseline model directory.
"""

import argparse
import difflib
from pathlib import Path
import shutil
import subprocess
from common import ROOT, HERE, new_output, sha256, unique_replace, write_json


def instrument(source, source_hash):
    source = unique_replace(
        source,
        '#include "imu_yaw_hold.h"',
        '#include "imu_yaw_hold.h"\n#define JEPLO_AUDIT_SOURCE_SHA "'
        + source_hash
        + '"\n#include "audit_capture.h"',
    )
    source = unique_replace(
        source,
        "    LocomotionConfig config_;",
        "    LocomotionConfig config_;\n    ObservationAudit audit_;",
    )
    source = unique_replace(
        source,
        "        // Initialize ZMQ after ONNX shapes have sized the depth buffers.",
        r"""
        audit_.configuration({{"joint_names",config_.sim_joint_names},
            {"motor_joint_names",config_.motor_joint_names}, {"joint2motor_idx",config_.joint2motor_idx},
            {"sensor_size",config_.sensor_size}, {"control_dt",config_.control_dt},
            {"wm_update_interval",config_.wm_update_interval}, {"obs_clip",config_.obs_clip},
            {"linear_scale",config_.lin_vel_scale}, {"angular_scale",config_.ang_vel_scale},
            {"joint_position_scale",config_.dof_pos_scale}, {"joint_velocity_scale",config_.dof_vel_scale},
            {"previous_action_scale",config_.action_obs_scale}, {"raw_actions",config_.use_raw_actions},
            {"policy_path",config_.policy_path}, {"sensor_estimator_path",config_.sensor_estimator_path}});
        // Initialize ZMQ after ONNX shapes have sized the depth buffers.""",
    )
    source = unique_replace(
        source,
        '    int net_idx = net_interface == "lo" ? 1 : 0;',
        '    if (net_interface != "lo") throw std::runtime_error("Observation audit is simulation-only");\n'
        '    int net_idx = net_interface == "lo" ? 1 : 0;',
    )
    source = unique_replace(
        source,
        "            auto actions = run_policy();",
        r"""
            auto actions = run_policy();
            // Diagnostic snapshot: raw_action_ is still the previous model action.
            const std::vector<float> audit_quat{quat.w(),quat.x(),quat.y(),quat.z()};
            audit_.record("policy", wm_step_counter_,
                {{"wall_time_ns",wall_clock_ns()}, {"estimator_scheduled",wm_step_counter_%config_.wm_update_interval==0}}, {
                audit_tensor("proprio_hist",prop_hist_,{1,prop_hist_.size()}),
                audit_tensor("high_feat",high_feat_,{1,high_feat_.size()}),
                audit_tensor("actions",actions,{1,actions.size()}),
                audit_tensor("current_proprio",proprio_obs_,{1,45}),
                audit_tensor("joint_pos",qj_,{1,12}), audit_tensor("joint_vel",dqj_,{1,12}),
                audit_tensor("default_joint_pos",config_.default_angles,{1,12}),
                audit_tensor("previous_action",raw_action_,{1,12}),
                audit_tensor("quaternion_wxyz",audit_quat,{1,4}),
                audit_tensor("gyro",ang_vel,{1,3}), audit_tensor("command",cmd_,{1,3})});
""",
    )
    source = unique_replace(
        source,
        "            auto output_tensors = sensor_estimator_session_->Run(",
        "            const auto audit_hidden_in = wm_hidden_;\n"
        "            auto output_tensors = sensor_estimator_session_->Run(",
    )
    source = unique_replace(
        source,
        "            std::copy_n(hidden_ptr, config_.gru_hidden_dim, wm_hidden_.begin());",
        r"""
            std::copy_n(hidden_ptr, config_.gru_hidden_dim, wm_hidden_.begin());
            audit_.record("estimator",wm_step_counter_, {
                {"wall_time_ns",consume_ns}, {"depth_frame_ids",{stamps.prev_seq,stamps.latest_seq}},
                {"depth_frame_times_ns",{stamps.prev_scan_ns,stamps.scan_ns}},
                {"latest_publish_ns",stamps.publish_ns}, {"latest_receive_ns",stamps.recv_ns}}, {
                audit_tensor("prop_hist",prop_hist_,{1,prop_hist_.size()}),
                audit_tensor("depth_stack",depth_buffer_,{1,static_cast<size_t>(config_.sensor_size[0]),
                    static_cast<size_t>(config_.sensor_size[1]),static_cast<size_t>(config_.sensor_size[2])}),
                audit_tensor("hidden_in",audit_hidden_in,{1,audit_hidden_in.size()}),
                audit_tensor("feature",high_feat_,{1,high_feat_.size()}),
                audit_tensor("hidden_out",wm_hidden_,{1,wm_hidden_.size()})});
""",
    )
    return source


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, default=ROOT)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--build",
        action="store_true",
        help="Also configure/build here using installed deployment dependencies",
    )
    p.add_argument("--cmake-arg", action="append", default=[])
    a = p.parse_args()
    repo = a.repo.resolve()
    out = new_output(a.output)
    baseline = repo / "deployment/go2_deploy/deploy"
    target = out / "source"
    target.mkdir()
    (target / "src").mkdir()
    for name in ("include",):
        shutil.copytree(baseline / name, target / name)
    for path in (baseline / "src").iterdir():
        if path.is_file():
            shutil.copy2(path, target / "src" / path.name)
    original = (baseline / "src/deploy.cpp").read_text()
    modified = instrument(original, sha256(baseline / "src/deploy.cpp"))
    (target / "src/deploy.cpp").write_text(modified)
    shutil.copy2(HERE / "audit_capture.h", target / "src/audit_capture.h")
    (out / "instrumentation.diff").write_text(
        "".join(
            difflib.unified_diff(
                original.splitlines(True),
                modified.splitlines(True),
                fromfile="baseline/deploy.cpp",
                tofile="diagnostic/deploy.cpp",
            )
        )
    )
    (target / "CMakeLists.txt").write_text(
        (baseline / "CMakeLists.txt").read_text()
        + '\nset_target_properties(deploy PROPERTIES OUTPUT_NAME "deploy_observation_audit")\n'
    )
    build = out / "build"
    build.mkdir()
    shutil.copy2(baseline / "config.json", build / "config.json")
    model = repo / "deployment/go2_deploy/model/my_policy_r2"
    private_model = build / "model/my_policy_r2"
    private_model.mkdir(parents=True)
    for name in ("policy.onnx", "sensor_estimator.onnx"):
        shutil.copy2(model / name, private_model / name)
    if (model / "test_cases").is_dir():
        shutil.copytree(model / "test_cases", private_model / "test_cases")
    commands = [
        [
            "cmake",
            "-S",
            str(target),
            "-B",
            str(build),
            "-DCMAKE_BUILD_TYPE=Release",
            *a.cmake_arg,
        ],
        ["cmake", "--build", str(build), "--target", "deploy", "--parallel", "2"],
    ]
    write_json(
        out / "build_manifest.json",
        dict(
            source_sha256=sha256(baseline / "src/deploy.cpp"),
            instrumented_sha256=sha256(target / "src/deploy.cpp"),
            commands=commands,
            model_hashes={
                n: sha256(private_model / n)
                for n in ("policy.onnx", "sensor_estimator.onnx")
            },
            baseline_config_sha256=sha256(baseline / "config.json"),
        ),
    )
    if a.build:
        for command in commands:
            subprocess.run(command, check=True)
    print(
        f"Generated diagnostic controller: {target}\nReview: {out/'instrumentation.diff'}"
    )
    print(
        "Build commands are saved in build_manifest.json. Baseline files and binaries were not modified."
    )


if __name__ == "__main__":
    main()
