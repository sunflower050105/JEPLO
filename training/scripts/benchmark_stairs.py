#!/usr/bin/env python3
"""Headless Isaac Sim ascent benchmark for the r2 CTS student policy.

Run from training/ using the Isaac Lab Python environment. One environment is
reused for sequential, independently reset trials; no camera or RTX rendering.
"""

import argparse
import csv
import hashlib
import json
import math
import random
import traceback
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=Path, default=Path("logs/rsl_rl/go2_loco/my_policy_r2/model_19999.pt"))
parser.add_argument("--speeds", type=float, nargs="+", default=[0.5, 0.7])
parser.add_argument("--repeats", type=int, default=100, help="Trials per speed")
parser.add_argument("--duration", type=float, default=16.0, help="Maximum simulated seconds per trial")
parser.add_argument("--no_yaw_hold", action="store_true", help="Disable IMU-style stair heading correction")
parser.add_argument("--seed", type=int, default=17)
parser.add_argument("--output_dir", type=Path, default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.repeats < 1 or args.duration <= 0 or not args.speeds or any(speed <= 0 for speed in args.speeds):
    parser.error("repeats, duration, and every speed must be positive")
args.headless = True
args.enable_cameras = False
app = AppLauncher(args).app

import go2_parkour.tasks  # noqa: E402,F401
import numpy as np  # noqa: E402
import torch  # noqa: E402
from go2_parkour.rsl_rl_wrapper import RslRlVecEnvWrapper  # noqa: E402
from go2_parkour.tasks.go2_loco_env import Go2LocoEnv  # noqa: E402
from go2_parkour.terrain.benchmark_stairs import BenchmarkStairsCfg  # noqa: E402
from go2_parkour.terrain.parkour_terrain_generator_cfg import ParkourTerrainGeneratorCfg  # noqa: E402
from isaaclab.sim import RigidBodyMaterialCfg  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from rsl_rl.runners.on_policy_runner_teacher_cts import OnPolicyRunnerTeacherCTS  # noqa: E402

import cli_args  # noqa: E402
from stair_benchmark_scoring import LandingScorer  # noqa: E402


class BenchmarkEnv(Go2LocoEnv):
    """Use the trained observation/action path with fixed benchmark commands."""

    fixed_speed = 0.5
    yaw_hold_enabled = True

    def _setup_rewards(self):
        super()._setup_rewards()
        # This unique terrain name must be treated as a direct-command course,
        # while its non-flat observation bit remains truthful to the policy.
        self._non_flat_direct_command_terrain_mask[:] = True
        self._direct_command_terrain_mask[:] = True
        self._waypoint_command_terrain_mask[:] = False

    def _reset_goals(self, masks=None):
        # The training task squeezes its mask to a scalar for a single env.
        # Keep the benchmark's one-element mask one-dimensional instead.
        if masks is None:
            masks = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        masks = masks.reshape(-1)
        self._env_next_goals_idx[masks] = 0
        self._cmd_goal_w[masks, :] = self._env_goals[masks, 0, :]

    def _reset_speed(self, masks=None):
        if masks is None:
            masks = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        masks = masks.reshape(-1)
        if not bool(torch.any(masks)):
            return
        self._cmd_lin_vel[masks, 0] = self.fixed_speed
        self._cmd_lin_vel[masks, 1] = 0.0
        self._cmd_ang_vel[masks, 0] = 0.0
        self._cmd_speed[masks, 0] = self.fixed_speed
        self._cmd_in_place_rotation[masks] = False
        self._cmd_in_place_ang_vel[masks] = 0.0
        self._cmd_heading_offset[masks] = 0.0
        self._cmd_resample_intervals[masks] = float("inf")
        self._cmd_resample_accums[masks] = 0.0

    def _pre_physics_step(self, actions):
        if self.yaw_hold_enabled:
            quat = self._robot.data.root_quat_w
            yaw = torch.atan2(
                2 * (quat[:, 0] * quat[:, 3] + quat[:, 1] * quat[:, 2]),
                1 - 2 * (quat[:, 2] ** 2 + quat[:, 3] ** 2),
            )
            error = torch.atan2(torch.sin(-yaw), torch.cos(-yaw))
            gyro_z = self._robot.data.root_ang_vel_b[:, 2]
            self._cmd_ang_vel[:, 0] = (2.0 * error - 0.25 * gyro_z).clamp(-1.0, 1.0)
        super()._pre_physics_step(actions)

    def _reset_idx(self, env_ids):
        # Bypass terrain curriculum and restore the same physical starting pose.
        self._first_reset = True
        self.common_step_counter = 0
        self._terrain.terrain_levels[:] = 0
        super()._reset_idx(env_ids)
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        self._depth_buffer[env_ids] = 0.0
        self._spherical_depth_gen.reset(env_ids)
        for actuator in self._robot.actuators.values():
            actuator.stiffness[env_ids, :] = 30.0
            actuator.damping[env_ids, :] = 1.0

    def _get_dones(self):
        # Score falls ourselves at 0.2 s, as in the MuJoCo benchmark. The base
        # task's earlier tilt termination would erase state before scoring.
        # The runner also owns the 16 s time limit.
        terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        timed_out = torch.zeros_like(terminated)
        return terminated, timed_out


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def configure_env():
    cfg = parse_env_cfg("Unitree-Go2-Locomotion", device=args.device, num_envs=1, use_fabric=True)
    cfg.seed = args.seed
    cfg.episode_length_s = args.duration
    cfg.scene.num_envs = 1
    cfg.scene.env_spacing = 9.0
    cfg.is_play_env = False  # avoids viewport and debug draw in headless mode
    cfg.terrain.terrain_generator = ParkourTerrainGeneratorCfg(
        size=(10.4, 10.4),
        border_width=0.5,
        sub_terrain_border_width=0.0,
        num_rows=2,  # TerrainGenerator's curriculum formula requires >1.
        num_cols=1,
        horizontal_scale=0.05,
        vertical_scale=0.005,
        slope_threshold=0.5,
        difficulty_range=(0.0, 0.0),
        use_cache=False,
        curriculum=True,
        sub_terrains={"benchmark_stairs": BenchmarkStairsCfg(proportion=1.0)},
    )
    material = RigidBodyMaterialCfg(
        friction_combine_mode="average",
        restitution_combine_mode="average",
        static_friction=0.4,
        dynamic_friction=0.4,
        restitution=0.0,
    )
    cfg.terrain.physics_material = material
    cfg.sim.physics_material = material
    cfg.events.randomize_rigid_body_mass_base = None
    cfg.events.randomize_rigid_body_mass_others = None
    cfg.events.randomize_com_positions = None
    cfg.events.randomize_push_robot = None
    cfg.events.randomize_rigid_body_material.params.update(
        static_friction_range=(0.4, 0.4),
        dynamic_friction_range=(0.4, 0.4),
        restitution_range=(0.0, 0.0),
        num_buckets=1,
    )
    cfg.events.reset_robot_joints.params.update(position_range=(1.0, 1.0), velocity_range=(0.0, 0.0))
    cfg.events.randomize_motor_zero_offset.params["offset_range"] = (0.0, 0.0)
    cfg.events.reset_base.params["pose_range"] = {key: (0.0, 0.0) for key in ("x", "y", "z", "roll", "pitch", "yaw")}
    cfg.events.reset_base.params["velocity_range"] = {
        key: (0.0, 0.0) for key in ("x", "y", "z", "roll", "pitch", "yaw")
    }
    return cfg


def state(env):
    raw = env.unwrapped
    base = raw._robot.data.root_pos_w[0].detach().cpu().tolist()
    quat = raw._robot.data.root_quat_w[0].detach().cpu().tolist()
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, 1.0 - 2.0 * (quat[1] ** 2 + quat[2] ** 2)))))
    feet = raw._robot.data.body_pos_w[0, raw._feet_ids_bd, :].detach().cpu().tolist()
    forces = raw._contact_sensor.data.net_forces_w[0, raw._feet_ids_cs, :]
    contacts = (torch.linalg.norm(forces, dim=-1) > 0.1).detach().cpu().tolist()
    return base, feet, contacts, tilt


def run_trial(env, policy, speed, trial_index, origin):
    print(f"[TRIAL_START] speed={speed:g} trial={trial_index + 1}/{args.repeats}", flush=True)
    trial_seed = args.seed + trial_index
    random.seed(trial_seed)
    np.random.seed(trial_seed)
    torch.manual_seed(trial_seed)
    BenchmarkEnv.fixed_speed = speed
    obs, extras = env.reset()
    # The runner's inference policy creates recurrent state under
    # @torch.inference_mode(). Clear it in that same mode between trials.
    with torch.inference_mode():
        policy.reset(torch.ones(1, dtype=torch.bool, device=env.unwrapped.device))
    policy.step_counter = 0
    start_x = float(origin[0])
    start_y = float(origin[1])
    scorer = LandingScorer(start_x + 6.0, start_x + 7.2, start_y, float(origin[2]) + 2.0)
    max_steps = math.ceil(args.duration / env.unwrapped.step_dt)
    outcome = "timeout"
    ascent_time = None
    max_progress = 0.0
    elapsed = 0.0
    for step in range(1, max_steps + 1):
        with torch.no_grad():
            # Keep inference tensors out of Isaac Lab's mutable simulation state.
            actions = policy(obs, extras).clone()
            obs, _, _, extras = env.step(actions)
        elapsed = step * env.unwrapped.step_dt
        base, feet, contacts, tilt = state(env)
        if not all(math.isfinite(value) for value in [*base, tilt, *(axis for foot in feet for axis in foot)]):
            raise RuntimeError(f"Non-finite robot state in trial {trial_index} at t={elapsed:.3f}s")
        max_progress = max(max_progress, base[0] - start_x)
        outcome_now, _ = scorer.update(elapsed, base, feet, contacts, tilt)
        if outcome_now is not None:
            outcome = outcome_now
            ascent_time = scorer.achieved_at
            break
    return {
        "speed_mps": speed,
        "trial": trial_index,
        "seed": trial_seed,
        "valid": True,
        "success": outcome == "success",
        "outcome": outcome,
        "duration_s": round(elapsed, 4),
        "ascent_time_s": ascent_time,
        "max_progress_m": round(max_progress, 4),
        "scoring_rule": LandingScorer.rule,
    }


def main():
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint missing: {checkpoint}")
    output_dir = args.output_dir or Path("outputs/stair_benchmark") / datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=False)
    metadata = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "policy": "student CTS, in-process Isaac inference",
        "speeds_mps": args.speeds,
        "repeats_per_speed": args.repeats,
        "seed": args.seed,
        "time_limit_s": args.duration,
        "steps": 10,
        "rise_m": 0.2,
        "run_m": 0.3,
        "landing_length_m": 1.2,
        "stair_half_width_m": 0.8,
        "approach_length_m": 3.0,
        "imu_yaw_hold": not args.no_yaw_hold,
        "terrain_and_robot_friction": 0.4,
        "scoring_rule": LandingScorer.rule,
        "contact_note": "Isaac foot force plus landing footprint/height proxy; engine contacts are not identical to MuJoCo geom contacts.",
        "headless": True,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"[START] output={output_dir} checkpoint={checkpoint} branch=student", flush=True)
    env = None
    try:
        cfg = configure_env()
        BenchmarkEnv.yaw_hold_enabled = not args.no_yaw_hold
        agent_cfg = cli_args.parse_rsl_rl_cfg("Unitree-Go2-Locomotion", argparse.Namespace(
            resume=False, load_run=None, checkpoint=None, run_name=None, logger=None, log_project_name=None
        ))
        env = RslRlVecEnvWrapper(BenchmarkEnv(cfg=cfg), clip_actions=agent_cfg.clip_actions)
        origin = env.unwrapped._terrain.env_origins[0].detach().cpu().tolist()
        train_cfg = agent_cfg.to_dict()
        runner = OnPolicyRunnerTeacherCTS(env, train_cfg, log_dir=None, device=agent_cfg.device)
        runner.load(str(checkpoint), load_optimizer=False)
        policy = runner.get_inference_policy(device=env.unwrapped.device)
        print("[POLICY] student inference for every trial; CTS environment split above is training bookkeeping", flush=True)
        rows = []
        with (output_dir / "trials.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=[
                "speed_mps", "trial", "seed", "valid", "success", "outcome", "duration_s",
                "ascent_time_s", "max_progress_m", "scoring_rule",
            ])
            writer.writeheader()
            for speed in args.speeds:
                for trial in range(args.repeats):
                    result = run_trial(env, policy, speed, trial, origin)
                    writer.writerow(result)
                    stream.flush()
                    rows.append(result)
                    print(f"[PROGRESS] speed={speed:g} trial={trial + 1}/{args.repeats} outcome={result['outcome']}", flush=True)
        summary = []
        for speed in args.speeds:
            subset = [row for row in rows if row["speed_mps"] == speed and row["valid"]]
            successes = sum(row["success"] for row in subset)
            item = {"speed_mps": speed, "valid_trials": len(subset), "successes": successes,
                    "success_rate": successes / len(subset) if subset else None}
            summary.append(item)
            print(f"[METRIC] speed={speed:g} success={successes}/{len(subset)} rate={item['success_rate']:.3f}", flush=True)
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(f"[DONE] output={output_dir}", flush=True)
    except BaseException:
        # Isaac Kit shutdown can mask Python exceptions and return exit code 0.
        # Let the failed process exit with its original exception instead.
        traceback.print_exc()
        raise
    else:
        if env is not None:
            env.close()
        app.close()


if __name__ == "__main__":
    main()
