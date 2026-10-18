import struct
import unittest
from pathlib import Path

import run


class ProtocolTests(unittest.TestCase):
    def test_depth_trailer_and_malformed_frame(self):
        frame = struct.pack("=II", 60, 25) + bytes(60 * 25 * 4) + struct.pack("=QQ", 100, 200)
        self.assertEqual(run.decode_depth(frame), (60, 25, 100, 200))
        self.assertIsNone(run.decode_depth(frame[:-1]))
        self.assertIsNone(run.decode_depth(b"bad"))

    def test_invalid_runs_do_not_inflate_failure_rate(self):
        summary = run.summarize([
            {"command_mps": 0.8, "valid": True, "success": True},
            {"command_mps": 0.8, "valid": True, "success": False},
            {"command_mps": 0.8, "valid": False, "success": False},
            {"command_mps": 1.0, "valid": False, "success": False},
        ])
        self.assertEqual(summary[0]["success_rate"], 0.5)
        self.assertEqual(summary[0]["valid_trials"], 2)
        self.assertEqual(summary[0]["invalid_trials"], 1)
        self.assertIsNone(summary[1]["success_rate"])

    def test_commands_are_simulation_only_and_use_paired_cage_seed(self):
        args = run.parser().parse_args([])
        args.duration = 8.0
        first = run.commands(args, Path("/tmp/trial1"), 0.8, 17)
        second = run.commands(args, Path("/tmp/trial2"), 1.0, 17)
        self.assertEqual(first["depth"], second["depth"])
        self.assertIn("lo", first["deploy"])
        self.assertIn("--raw-actions", first["deploy"])
        self.assertNotIn("--see-then-hidden", first["simulator"])
        self.assertNotIn("--no-cage-mask", first["depth"])

    def test_box_task_default_and_legacy_selection(self):
        for task, rule in (("ascent", "box_ascent_v2"), ("traverse", "box_traverse_v1")):
            options = ["--terrain", "boxes"]
            if task == "traverse":
                options += ["--box-task", task]
            args = run.parser().parse_args(options)
            self.assertEqual(run.scoring_rule(args), rule)
            command = run.commands(args, Path("/tmp/trial"), 0.8, 17)["simulator"]
            self.assertEqual(command[command.index("--benchmark-box-task") + 1], task)

    def test_different_scoring_rules_are_never_pooled(self):
        rows = run.summarize([
            {"command_mps": 0.8, "valid": True, "success": True, "scoring_rule": "box_ascent_v2"},
            {"command_mps": 0.8, "valid": True, "success": False, "scoring_rule": "box_traverse_v1"},
        ])
        self.assertEqual(len(rows), 2)
        self.assertEqual([r["success_rate"] for r in rows], [1.0, 0.0])


if __name__ == "__main__":
    unittest.main()
