"""Fixed, straight ten-step ascent for the headless policy benchmark."""

from __future__ import annotations

import numpy as np
import trimesh
from go2_parkour.terrain.parkour_terrain_generator_cfg import ParkourSubTerrainBaseCfg
from isaaclab.utils import configclass


def benchmark_stairs_terrain(difficulty: float, cfg: BenchmarkStairsCfg, num_goals: int):
    """Return a box mesh and the bookkeeping expected by ParkourTerrainGenerator.

    The mesh has exact box faces; no height-field resolution rounds the tread run.
    The origin is the robot's start, 3.0 m before the first riser by default.
    """
    del difficulty
    end_x = cfg.first_step_x + cfg.steps * cfg.run + cfg.landing_length
    if end_x > cfg.size[0] or cfg.first_step_x <= cfg.approach_length:
        raise ValueError("Benchmark stairs do not fit inside the terrain tile")
    if 2 * cfg.half_width >= cfg.size[1]:
        raise ValueError("Stair width does not fit inside the terrain tile")

    center_y = cfg.size[1] / 2
    meshes = []

    def box(extents, center):
        mesh = trimesh.creation.box(extents=extents)
        mesh.apply_translation(center)
        meshes.append(mesh)

    box((cfg.size[0], cfg.size[1], 0.10), (cfg.size[0] / 2, center_y, -0.05))
    for step in range(1, cfg.steps + 1):
        height = step * cfg.rise
        x = cfg.first_step_x + (step - 0.5) * cfg.run
        box((cfg.run, 2 * cfg.half_width, height), (x, center_y, height / 2))
    top = cfg.steps * cfg.rise
    landing_start = cfg.first_step_x + cfg.steps * cfg.run
    box(
        (cfg.landing_length, 2 * cfg.half_width, top),
        (landing_start + cfg.landing_length / 2, center_y, top / 2),
    )

    origin = np.array([cfg.first_step_x - cfg.approach_length, center_y, 0.0])
    goals = np.zeros((num_goals, 2), dtype=np.float64)
    goal_heights = np.zeros(num_goals, dtype=np.int16)
    goal_y_shift_ranges = np.zeros(num_goals, dtype=np.float64)
    shape = (int(cfg.size[0] / cfg.horizontal_scale) + 1, int(cfg.size[1] / cfg.horizontal_scale) + 1)
    empty_mask = np.zeros(shape, dtype=bool)
    return meshes, origin, goals, goal_heights, empty_mask, empty_mask.copy(), goal_y_shift_ranges


@configclass
class BenchmarkStairsCfg(ParkourSubTerrainBaseCfg):
    function = benchmark_stairs_terrain
    rise: float = 0.20
    run: float = 0.30
    steps: int = 10
    landing_length: float = 1.20
    half_width: float = 0.80
    approach_length: float = 3.0
    first_step_x: float = 4.0
