"""File formats and small utilities shared by the isolated observation checks.

No module in this directory is imported by a baseline launcher. A trace consists
of JSONL metadata and little-endian float32 arrays in tensors.bin. Each JSON row
is committed only after its binary arrays have been flushed, so an interrupted
diagnostic run can still be read up to its last complete row.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
JOINTS = [
    f"{leg}_{part}_joint"
    for part in ("hip", "thigh", "calf")
    for leg in ("FL", "FR", "RL", "RR")
]
COMPONENTS = {
    "angular_velocity": (0, 3),
    "projected_gravity": (3, 6),
    "command": (6, 9),
    "joint_position": (9, 21),
    "joint_velocity": (21, 33),
    "previous_raw_action": (33, 45),
}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def new_output(path):
    path = Path(path).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=False)
    return path


def unique_replace(text, old, new):
    """Fail closed if a baseline edit makes an instrumentation anchor ambiguous."""
    if text.count(old) != 1:
        raise ValueError(
            f"Expected one source anchor, found {text.count(old)}: {old[:100]!r}"
        )
    return text.replace(old, new, 1)


def between(text, start, end):
    if text.count(start) != 1 or text.count(end) != 1:
        raise ValueError(
            f"Source anchors changed: {start!r} / {end!r}; review the adapter."
        )
    return text.split(start, 1)[1].split(end, 1)[0]


def error_metrics(actual, expected, atol=1e-6, rtol=1e-6):
    a, b = np.asarray(actual), np.asarray(expected)
    result = {"actual_shape": list(a.shape), "expected_shape": list(b.shape)}
    if a.shape != b.shape or not a.size:
        return dict(result, passed=False, reason="shape mismatch or empty tensor")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        return dict(result, passed=False, reason="NaN or infinity")
    delta = np.abs(a.astype(np.float64) - b.astype(np.float64))
    index = np.unravel_index(int(delta.argmax()), delta.shape)
    return dict(
        result,
        passed=bool(np.all(delta <= atol + rtol * np.abs(b))),
        max_abs=float(delta.max()),
        mean_abs=float(delta.mean()),
        worst_index=[int(i) for i in index],
        actual_at_worst=float(a[index]),
        expected_at_worst=float(b[index]),
        atol=atol,
        rtol=rtol,
    )


class TraceWriter:
    def __init__(self, directory, metadata, max_records=5000):
        self.directory = new_output(directory)
        self.count = 0
        self.max_records = max_records
        self.data = (self.directory / "tensors.bin").open("wb")
        self.index = (self.directory / "records.jsonl").open("w")
        write_json(
            self.directory / "metadata.json",
            dict(metadata, format="jeplo-observation-trace-v1"),
        )

    def record(self, kind, metadata, arrays):
        if self.count >= self.max_records:
            return
        descriptors = {}
        for name, value in arrays.items():
            if hasattr(value, "detach"):
                value = value.detach().cpu().numpy()
            array = np.ascontiguousarray(value, dtype="<f4")
            descriptors[name] = {"offset": self.data.tell(), "shape": list(array.shape)}
            self.data.write(array.tobytes())
        self.data.flush()
        self.index.write(
            json.dumps(dict(metadata, kind=kind, arrays=descriptors), allow_nan=False)
            + "\n"
        )
        self.index.flush()
        self.count += 1

    def close(self):
        self.data.close()
        self.index.close()


def read_trace(directory):
    directory = Path(directory)
    with (
        (directory / "tensors.bin").open("rb") as data,
        (directory / "records.jsonl").open() as index,
    ):
        for line in index:
            # Ignore only a final partial line, never a corrupt complete record.
            if not line.endswith("\n"):
                break
            record = json.loads(line)
            arrays = {}
            for name, descriptor in record.pop("arrays").items():
                count = int(np.prod(descriptor["shape"]))
                data.seek(descriptor["offset"])
                blob = data.read(count * 4)
                if len(blob) != count * 4:
                    raise ValueError("Truncated tensors.bin for a committed trace row")
                arrays[name] = np.frombuffer(blob, dtype="<f4").reshape(
                    descriptor["shape"]
                )
            yield record, arrays
