"""Simulator-free regression tests for recording integrity and observer behavior."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
import numpy as np
from common import HERE, ROOT, TraceWriter, error_metrics, read_trace, unique_replace
from prepare_mujoco_capture import instrument


class UtilityTests(unittest.TestCase):
    def test_comparison_rejects_nonfinite_shape_and_empty(self):
        self.assertFalse(error_metrics([float("nan")], [0])["passed"])
        self.assertFalse(error_metrics(np.zeros((2, 2)), np.zeros(4))["passed"])
        self.assertFalse(error_metrics([], [])["passed"])
        self.assertTrue(error_metrics([1.0000001], [1])["passed"])
        self.assertFalse(error_metrics([1.1], [1])["passed"])

    def test_anchors_fail_closed(self):
        with self.assertRaises(ValueError):
            unique_replace("a a", "a", "b")
        with self.assertRaises(ValueError):
            unique_replace("a", "c", "b")
        source = (ROOT / "deployment/go2_deploy/deploy/src/deploy.cpp").read_text()
        result = instrument(source, "test_hash")
        self.assertIn('audit_.record("estimator"', result)
        self.assertIn('audit_.record("policy"', result)
        self.assertEqual(
            source, (ROOT / "deployment/go2_deploy/deploy/src/deploy.cpp").read_text()
        )

    def test_trace_roundtrip_and_interruption(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace"
            w = TraceWriter(path, {"side": "test"}, max_records=1)
            expected = np.arange(12, dtype=np.float64).reshape(1, 3, 4)
            w.record("estimator", {"step": 2}, {"depth": expected})
            w.record("policy", {"step": 3}, {"ignored": expected})
            w.close()
            with (path / "records.jsonl").open("a") as f:
                f.write('{"partial":')
            rows = list(read_trace(path))
            self.assertEqual(len(rows), 1)
            np.testing.assert_array_equal(rows[0][1]["depth"], expected)
            with self.assertRaises(FileExistsError):
                TraceWriter(path, {})

    def test_cpp_python_trace_compatibility(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "writer.cpp").write_text("""
#define JEPLO_AUDIT_SOURCE_SHA "fixture"
#include "audit_capture.h"
int main(){ObservationAudit audit;std::vector<float>x{.25f,.5f,.75f};
audit.configuration({{"control_dt",.02}});
audit.record("policy",7,{{"wall_time_ns",123}}, {audit_tensor("gyro",x,{1,3})});}
""")
            subprocess.run(
                [
                    "g++",
                    "-std=c++17",
                    "-I" + str(HERE),
                    "-I" + str(ROOT / "deployment/go2_deploy/deploy/include"),
                    str(root / "writer.cpp"),
                    "-o",
                    str(root / "writer"),
                ],
                check=True,
            )
            env = dict(os.environ, JEPLO_OBSERVATION_AUDIT_DIR=str(root / "trace"))
            subprocess.run([str(root / "writer")], env=env, check=True)
            row, arrays = next(read_trace(root / "trace"))
            self.assertEqual(row["step"], 7)
            np.testing.assert_array_equal(arrays["gyro"], [[0.25, 0.5, 0.75]])
            self.assertEqual(
                json.loads((root / "trace/runtime.json").read_text())["control_dt"],
                0.02,
            )


@unittest.skipUnless(
    importlib.util.find_spec("torch"), "Torch required for Isaac observer mock"
)
class IsaacObserverTests(unittest.TestCase):
    def test_observer_preserves_actions_and_tracks_delayed_frames(self):
        import torch
        from check3_isaac import install_hooks

        class Env:
            def __init__(self):
                self.unwrapped = self
                self.common_step_counter = 0
                self.step_dt = 0.02
                self._depth_buffer = torch.zeros(1, 3, 25, 60)
                self.cfg = NS(
                    num_depth_stack=2,
                    depth_delay=1,
                    depth_update_freq=5,
                    lidar_num_stacked=10,
                    lidar_noise_std=0.005,
                    lidar_bottom_dropout_fraction=0.05,
                    mid360=NS(
                        update_period=0.02,
                        offset=NS(pos=(0.275, 0, 0.148), rot=(0, 1, 0, 0)),
                        pattern_cfg=NS(
                            horizontal_fov_deg=(-60, 60), vertical_fov_deg=(-4.5, 45.5)
                        ),
                    ),
                )
                data = NS(
                    joint_pos=torch.zeros(1, 12),
                    joint_vel=torch.zeros(1, 12),
                    default_joint_pos=torch.zeros(1, 12),
                    root_quat_w=torch.tensor([[1.0, 0, 0, 0]]),
                    root_com_ang_vel_b=torch.zeros(1, 3),
                    projected_gravity_b=torch.tensor([[0.0, 0, -1]]),
                    root_pos_w=torch.zeros(1, 3),
                )
                self._robot = NS(data=data, joint_names=[str(i) for i in range(12)])
                self._terrain = NS(env_origins=torch.zeros(1, 3))
                self._actions = torch.zeros(1, 12)
                self._cmd_lin_vel = torch.zeros(1, 2)
                self._cmd_ang_vel = torch.zeros(1, 1)

            def _get_observations(self):
                if self.common_step_counter % 5 == 0:
                    self._depth_buffer = torch.cat(
                        [
                            self._depth_buffer[:, 1:],
                            torch.full(
                                (1, 1, 25, 60), self.common_step_counter / 100 + 0.01
                            ),
                        ],
                        1,
                    )
                return {
                    "policy": torch.ones(1, 450) * self.common_step_counter,
                    "depth": self._depth_buffer[:, :2],
                }

            def _reset_idx(self):
                self._depth_buffer.zero_()
                self.common_step_counter = 0

        class Policy:
            def __init__(self):
                self.policy = NS(extract_student_proprio_hist=lambda obs: obs)
                self.estimator_high = NS(get_hidden_dim=lambda: 512)
                self.step_counter = 0
                self.update_interval = 5
                self.hidden = None
                self.sensor_latent = None

            def __call__(self, obs, extras):
                if self.hidden is None:
                    self.hidden = torch.zeros(1, 512)
                    self.sensor_latent = self.hidden.clone()
                if self.step_counter % 5 == 0:
                    self.hidden = self.hidden + 1
                    self.sensor_latent = self.hidden.clone()
                self.step_counter += 1
                return obs[:, :12] + self.hidden[:, :12]

            def reset(self, *args):
                self.hidden = None

        class Runner:
            def __init__(self, env):
                self.env = env

            def get_inference_policy(self):
                return Policy()

        def trial(env, policy, speed, index, origin, pattern=None):
            env._reset_idx()
            policy.reset()
            policy.step_counter = 0
            outputs = []
            for step in range(16):
                env.common_step_counter = step
                obs = env._get_observations()
                outputs.append(policy(obs["policy"], {"observations": obs}).clone())
            return torch.stack(outputs)

        env = Env()
        reference = trial(env, Policy(), 0.8, 0, None)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ns = {
                "BenchmarkEnv": Env,
                "OnPolicyRunnerTeacherCTS": Runner,
                "run_trial": trial,
            }
            install_hooks(ns, root, 100, {"test": True})
            actual = ns["run_trial"](
                env, Runner(env).get_inference_policy(), 0.8, 0, None, "standard"
            )
            torch.testing.assert_close(actual, reference, atol=0, rtol=0)
            rows = list(read_trace(root / "captures/standard_v0.8_trial000"))
            estimators = [m for m, a in rows if m["kind"] == "estimator"]
            self.assertEqual([m["step"] for m in estimators], [0, 5, 10, 15])
            self.assertEqual(estimators[0]["depth_frame_ids"], [-1, -1])
            self.assertEqual(estimators[-1]["depth_frame_ids"], [2, 3])
            self.assertEqual(estimators[-1]["depth_frame_times_s"], [0.1, 0.2])
            subprocess.run(
                [
                    sys.executable,
                    str(HERE / "analyze_trace.py"),
                    "--trace",
                    str(root / "captures/standard_v0.8_trial000"),
                    "--output",
                    str(root / "analysis"),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            report = json.loads((root / "analysis/report.json").read_text())
            self.assertEqual(report["policy_records"], 16)
            self.assertAlmostEqual(report["depth"]["frame_spacing_ms"]["p50"], 100.0)


if __name__ == "__main__":
    unittest.main()
