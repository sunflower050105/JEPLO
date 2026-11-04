#!/usr/bin/env python3
"""Separate Isaac launcher: observe the existing stair benchmark in this process.

Hooks are installed only after the diagnostic launcher imports the benchmark.
They copy tensors at inference boundaries and track actual depth-buffer shifts.
They never alter observations, noise, masks, model state, actions or update rates.
GPU-to-CPU copies add overhead; these runs are for diagnostics, not new benchmark
success-rate claims. Run this using the Isaac Lab Python environment.
"""

import argparse
import inspect
import os
from pathlib import Path
import runpy
import sys
import time
from common import ROOT, TraceWriter, new_output, sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--max-records",
        type=int,
        default=5000,
        help="Per trial; includes policy and estimator rows",
    )
    parser.add_argument("benchmark_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.max_records < 1:
        parser.error("max-records must be positive")
    repo = args.repo.resolve()
    script = repo / "training/scripts/benchmark_stairs.py"
    if not script.is_file():
        parser.error(
            f"Isaac benchmark missing: {script}. Run from JEPLO-autodl or pass --repo."
        )
    forwarded = (
        args.benchmark_args[1:]
        if args.benchmark_args[:1] == ["--"]
        else args.benchmark_args
    )
    out = new_output(args.output)
    sys.path.insert(0, str(script.parent))
    os.chdir(repo / "training")
    sys.argv = [
        str(script),
        "--headless",
        "--device",
        "cuda:0",
        "--friction",
        "1.0",
        "--speeds",
        "0.8",
        "--repeats",
        "1",
        "--duration",
        "30",
        "--seed",
        "0",
        "--occlusion_patterns",
        "standard",
        *forwarded,
        "--output_dir",
        str(out / "benchmark"),
    ]
    write_json(
        out / "launch.json",
        dict(argv=sys.argv, benchmark_sha256=sha256(script), diagnostic=True),
    )
    loaded = runpy.run_path(str(script), run_name="observation_audit_benchmark")
    # Function globals are the live namespace used by main/run_trial.
    ns = loaded["main"].__globals__
    Env = ns["BenchmarkEnv"]
    Runner = ns["OnPolicyRunnerTeacherCTS"]
    checkpoint = ns["args"].checkpoint.resolve()
    identity = dict(
        checkpoint=str(checkpoint),
        checkpoint_sha256=sha256(checkpoint),
        sources={
            inspect.getfile(obj): sha256(inspect.getfile(obj))
            for obj in (Env._get_observations, Env.__mro__[1], Runner)
        },
    )
    install_hooks(ns, out, args.max_records, identity)
    ns["main"]()


def install_hooks(ns, out, max_records, identity):
    """Install process-local observers; exposed separately for simulator-free tests."""
    import torch

    Env = ns["BenchmarkEnv"]
    Runner = ns["OnPolicyRunnerTeacherCTS"]
    active = {"writer": None}

    original_get = Env._get_observations

    def observed_get(self, *pos, **kw):
        length = self.cfg.num_depth_stack + self.cfg.depth_delay
        if not hasattr(self, "_audit_depth_ids"):
            self._audit_depth_ids = [-1] * length
            self._audit_depth_times = [None] * length
            self._audit_depth_seq = 0
        before = self._depth_buffer
        result = original_get(self, *pos, **kw)
        # Track an actual buffer replacement, including repeated observation calls.
        if self._depth_buffer is not before:
            self._audit_depth_seq += 1
            self._audit_depth_ids = self._audit_depth_ids[1:] + [self._audit_depth_seq]
            self._audit_depth_times = self._audit_depth_times[1:] + [
                float(self.common_step_counter * self.step_dt)
            ]
        return result

    Env._get_observations = observed_get
    original_reset = Env._reset_idx

    def observed_reset(self, *pos, **kw):
        result = original_reset(self, *pos, **kw)
        length = self.cfg.num_depth_stack + self.cfg.depth_delay
        self._audit_depth_ids = [-1] * length
        self._audit_depth_times = [None] * length
        self._audit_depth_seq = 0
        return result

    Env._reset_idx = observed_reset

    class ObservedPolicy:
        def __init__(self, inner, raw):
            self.inner = inner
            self.raw = raw

        def __getattr__(self, name):
            return getattr(self.inner, name)

        @property
        def step_counter(self):
            return self.inner.step_counter

        @step_counter.setter
        def step_counter(self, value):
            self.inner.step_counter = value

        def __call__(self, obs, extras=None):
            writer = active["writer"]
            if writer is None or writer.count >= writer.max_records:
                return self.inner(obs, extras)
            raw = self.raw
            i = self.inner
            step = i.step_counter
            prop = i.policy.extract_student_proprio_hist(obs).detach().clone()
            depth = extras["observations"]["depth"].detach().clone()
            update = step % i.update_interval == 0
            hidden = (
                (
                    torch.zeros(
                        (prop.shape[0], i.estimator_high.get_hidden_dim()),
                        device=prop.device,
                    )
                    if i.hidden is None
                    else i.hidden.detach().clone()
                )
                if update
                else None
            )
            now = time.time_ns()
            result = i(obs, extras)
            meta = dict(
                step=step,
                wall_time_ns=now,
                sim_time_s=float(raw.common_step_counter * raw.step_dt),
                depth_frame_ids=raw._audit_depth_ids[: raw.cfg.num_depth_stack],
                depth_frame_times_s=raw._audit_depth_times[: raw.cfg.num_depth_stack],
            )
            if update:
                writer.record(
                    "estimator",
                    meta,
                    dict(
                        prop_hist=prop,
                        depth_stack=depth,
                        hidden_in=hidden,
                        feature=i.sensor_latent,
                        hidden_out=i.hidden,
                    ),
                )
            data = raw._robot.data
            writer.record(
                "policy",
                dict(meta, estimator_scheduled=update),
                dict(
                    proprio_hist=prop,
                    high_feat=i.sensor_latent,
                    actions=result,
                    current_proprio=prop[:, :45],
                    joint_pos=data.joint_pos,
                    joint_vel=data.joint_vel,
                    default_joint_pos=data.default_joint_pos,
                    previous_action=raw._actions,
                    quaternion_wxyz=data.root_quat_w,
                    gyro=data.root_com_ang_vel_b,
                    projected_gravity=data.projected_gravity_b,
                    command=torch.cat([raw._cmd_lin_vel, raw._cmd_ang_vel], dim=-1),
                    base_position_relative_origin=data.root_pos_w
                    - raw._terrain.env_origins,
                ),
            )
            return result

    original_policy = Runner.get_inference_policy

    def observed_policy(self, *pos, **kw):
        return ObservedPolicy(original_policy(self, *pos, **kw), self.env.unwrapped)

    Runner.get_inference_policy = observed_policy
    original_trial = ns["run_trial"]

    def observed_trial(env, policy, speed, trial_index, origin, pattern=None):
        raw = env.unwrapped
        directory = (
            out
            / "captures"
            / f"{pattern or 'training_masks'}_v{speed:g}_trial{trial_index:03d}"
        )
        cfg = raw.cfg
        metadata = dict(
            identity,
            side="isaac",
            clock="simulation_time_s",
            speed=speed,
            trial=trial_index,
            pattern=pattern,
            joint_names=list(raw._robot.joint_names),
            scan_time_semantics="completed depth image time in simulation; no transport delay",
            config=dict(
                step_dt=raw.step_dt,
                raw_scan_update_period=cfg.mid360.update_period,
                lidar_num_stacked=cfg.lidar_num_stacked,
                num_depth_stack=cfg.num_depth_stack,
                depth_delay=cfg.depth_delay,
                depth_update_freq=cfg.depth_update_freq,
                lidar_noise_std=cfg.lidar_noise_std,
                bottom_dropout_fraction=cfg.lidar_bottom_dropout_fraction,
                sensor_offset_position=list(cfg.mid360.offset.pos),
                sensor_offset_wxyz=list(cfg.mid360.offset.rot),
                horizontal_fov=list(cfg.mid360.pattern_cfg.horizontal_fov_deg),
                vertical_fov=list(cfg.mid360.pattern_cfg.vertical_fov_deg),
            ),
        )
        writer = TraceWriter(directory, metadata, max_records)
        active["writer"] = writer
        try:
            return original_trial(env, policy, speed, trial_index, origin, pattern)
        finally:
            writer.close()
            active["writer"] = None

    ns["run_trial"] = observed_trial


if __name__ == "__main__":
    main()
