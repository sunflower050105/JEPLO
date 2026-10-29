"""Fixed-per-trial depth-image masks for the stair occlusion benchmark.

These are normalized-coordinate approximations of the four visual patterns in
Fig. 5(a-d) of the JEPLO paper. The paper does not publish mask parameters.
``True`` denotes a pixel replaced by the no-return value (1.0).
Copied from JEPLO-autodl/training/scripts/stair_occlusion_masks.py so the
MuJoCo and Isaac Sim diagnostics use the same generator and NumPy RNG.
"""

from __future__ import annotations

import numpy as np

PATTERNS = ("standard", "blob", "cross", "cut", "stripe")
ALGORITHM = "isaac_stair_occlusion_masks_v1"


def make_mask(pattern: str, height: int, width: int, seed: int) -> np.ndarray:
    """Return a reproducible mask held fixed throughout one trial."""
    if pattern not in PATTERNS:
        raise ValueError(f"Unknown occlusion pattern {pattern!r}; expected one of {PATTERNS}")
    if height < 8 or width < 8:
        raise ValueError("Depth image must be at least 8 by 8 pixels")
    if pattern == "standard":
        return np.zeros((height, width), dtype=bool)

    rng = np.random.default_rng(seed)
    y, x = np.meshgrid(
        np.linspace(-1.0, 1.0, height, dtype=np.float32),
        np.linspace(-1.0, 1.0, width, dtype=np.float32),
        indexing="ij",
    )

    if pattern == "blob":
        mask = np.zeros((height, width), dtype=bool)
        # Three overlapping rounded patches form one irregular horizontal blob.
        for center_x in (-0.55, 0.0, 0.55):
            cx = center_x + rng.uniform(-0.08, 0.08)
            cy = rng.uniform(-0.13, 0.13)
            radius_x = rng.uniform(0.28, 0.38)
            radius_y = rng.uniform(0.34, 0.48)
            mask |= ((x - cx) / radius_x) ** 2 + ((y - cy) / radius_y) ** 2 <= 1.0
        return mask

    if pattern == "cross":
        center_x = rng.uniform(-0.08, 0.08)
        center_y = rng.uniform(-0.08, 0.08)
        thickness = rng.uniform(0.13, 0.20)
        return (np.abs((y - center_y) - (x - center_x)) <= thickness) | (
            np.abs((y - center_y) + (x - center_x)) <= thickness
        )

    if pattern == "cut":
        # Large diagonal right-hand loss, matching the paper's cut example.
        boundary = rng.uniform(0.03, 0.17) - rng.uniform(0.25, 0.38) * y
        return x >= boundary

    # Two vertical bands similar to the physical sensor-cage pillars.
    mask = np.zeros((height, width), dtype=bool)
    for center_x in (-0.57, 0.57):
        center = center_x + rng.uniform(-0.06, 0.06)
        slope = rng.uniform(-0.07, 0.07)
        half_width = rng.uniform(0.14, 0.20)
        mask |= np.abs(x - (center + slope * y)) <= half_width
    return mask


def write_pgm(path, mask: np.ndarray) -> None:
    """Save a binary PGM that the isolated MuJoCo depth publisher can verify."""
    if mask.ndim != 2 or mask.dtype != np.bool_:
        raise ValueError("Expected a two-dimensional boolean occlusion mask")
    height, width = mask.shape
    header = f"P5\n{width} {height}\n255\n".encode("ascii")
    path.write_bytes(header + (mask.astype(np.uint8) * 255).tobytes())
