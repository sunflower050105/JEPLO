#!/usr/bin/env python3
"""Replay identical tensors through ONNX; compare with saved PyTorch/live outputs.

CPU replay checks export numerics. Choose the actual deployment provider to test
that runtime too. This uses recorded hidden_in per sample (one-step parity), not
a new closed-loop rollout or a claim about the quality of the learned policy.
"""

import argparse
import json
from pathlib import Path
import numpy as np
from common import ROOT, error_metrics, new_output, read_trace, sha256, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model-dir",
        type=Path,
        default=ROOT / "deployment/go2_deploy/model/my_policy_r2",
    )
    group = p.add_mutually_exclusive_group()
    group.add_argument("--cases", type=Path, help="Defaults to model-dir/test_cases")
    group.add_argument(
        "--trace", type=Path, help="Check 3 trace with exact runtime inputs/outputs"
    )
    p.add_argument(
        "--provider",
        choices=[
            "CPUExecutionProvider",
            "CUDAExecutionProvider",
            "TensorrtExecutionProvider",
        ],
        default="CPUExecutionProvider",
    )
    p.add_argument("--atol", type=float, default=1e-4)
    p.add_argument("--rtol", type=float, default=1e-4)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    import onnxruntime as ort

    if a.provider not in ort.get_available_providers():
        p.error(f"Requested provider is unavailable: {ort.get_available_providers()}")
    if not np.isfinite([a.atol, a.rtol]).all() or a.atol < 0 or a.rtol < 0:
        p.error("Tolerances must be finite and nonnegative")
    out = new_output(a.output)
    providers = [a.provider]
    if a.provider == "TensorrtExecutionProvider":
        providers = [
            (
                a.provider,
                {
                    "trt_fp16_enable": False,
                    "trt_engine_cache_enable": True,
                    "trt_engine_cache_path": str(out / "trt_cache"),
                },
            ),
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]
    sessions = {
        kind: ort.InferenceSession(
            str(a.model_dir / f"{kind}.onnx"), providers=providers
        )
        for kind in ("policy", "sensor_estimator")
    }
    if a.trace:
        cases = (
            (
                "sensor_estimator" if meta["kind"] == "estimator" else "policy",
                f"step_{meta['step']}",
                arrays,
            )
            for meta, arrays in read_trace(a.trace)
            if meta["kind"] in ("estimator", "policy")
        )
    else:
        case_dir = a.cases or a.model_dir / "test_cases"

        def saved_cases():
            for kind in sessions:
                for path in sorted((case_dir / kind).glob("sample_*.npz")):
                    with np.load(path, allow_pickle=False) as z:
                        yield kind, str(path), dict(z)

        cases = saved_cases()
    rows = []
    counts = {kind: 0 for kind in sessions}
    reference_hashes = {}
    for kind, label, arrays in cases:
        if not a.trace:
            reference_hashes[label] = sha256(Path(label))
        session = sessions[kind]
        # Aliases are explicit because the two exports use different input names.
        if kind == "policy" and "proprio_hist" not in arrays and "prop_hist" in arrays:
            arrays = dict(arrays, proprio_hist=arrays["prop_hist"])
        feed = {
            i.name: np.asarray(arrays[i.name], dtype=np.float32)
            for i in session.get_inputs()
        }
        if not all(np.isfinite(v).all() for v in feed.values()):
            raise ValueError(f"Nonfinite replay input: {label}")
        result = session.run(None, feed)
        compared = 0
        for node, actual in zip(session.get_outputs(), result):
            if node.name not in arrays:
                continue
            metrics = error_metrics(actual, arrays[node.name], a.atol, a.rtol)
            rows.append(dict(model=kind, sample=label, tensor=node.name, **metrics))
            compared += 1
        required = {"actions"} if kind == "policy" else {"feature", "hidden_out"}
        if not required.issubset(arrays) or not compared:
            raise ValueError(f"Missing reference outputs for {kind}: {label}")
        counts[kind] += 1
    passed = all(counts.values()) and all(r["passed"] for r in rows)
    if a.trace:
        reference_hashes = {
            name: sha256(a.trace / name)
            for name in ("metadata.json", "records.jsonl", "tensors.bin")
        }
    report = dict(
        passed=bool(passed),
        counts=counts,
        comparisons=rows,
        onnxruntime_version=ort.__version__,
        requested_provider=a.provider,
        session_providers={k: s.get_providers() for k, s in sessions.items()},
        model_hashes={f"{k}.onnx": sha256(a.model_dir / f"{k}.onnx") for k in sessions},
        input_source=str(a.trace or (a.cases or a.model_dir / "test_cases")),
        reference_hashes=reference_hashes,
        scope="Identical-input one-step parity; provider fallback may execute unsupported operators.",
    )
    write_json(out / "report.json", report)
    print(
        json.dumps(
            {
                k: v
                for k, v in report.items()
                if k not in ("comparisons", "reference_hashes")
            },
            indent=2,
        )
    )
    print(f"Detailed tensor errors: {out/'report.json'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
