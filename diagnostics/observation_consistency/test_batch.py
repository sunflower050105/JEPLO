"""Test unattended continuation versus execution failure, without simulators."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

from run_all import Batch, classify_result


class BatchTests(unittest.TestCase):
    def test_only_completed_mismatch_can_continue(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "report.json"
            self.assertEqual(classify_result(1, report), "error")
            report.write_text("broken json")
            self.assertEqual(classify_result(1, report), "error")
            report.write_text(json.dumps({"passed": False}))
            self.assertEqual(classify_result(1, report), "mismatch")
            self.assertEqual(classify_result(2, report), "error")
            report.write_text(json.dumps({"issues": ["nonfinite input"]}))
            self.assertEqual(classify_result(1, report), "error")
            self.assertEqual(classify_result(1, report, analysis=True), "mismatch")

    def test_failed_comparison_continues_but_crash_raises(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            batch = Batch(out, "mujoco", False)
            report = out / "report.json"
            batch.run("comparison", [sys.executable, "-c",
                      "import pathlib,sys; pathlib.Path(sys.argv[1]).write_text('{\"passed\":false}'); sys.exit(1)",
                      str(report)], report)
            batch.run("next-stage", [sys.executable, "-c", "print('continued')"])
            with self.assertRaises(RuntimeError):
                batch.run("crash", [sys.executable, "-c", "raise RuntimeError('setup failure')"])
            stages = json.loads((out / "summary.json").read_text())["stages"]
            self.assertEqual([s["status"] for s in stages], ["mismatch", "completed", "error"])


if __name__ == "__main__":
    unittest.main()
