#!/usr/bin/env python3
"""Render the exact recorded MuJoCo poses; does not rerun the policy or physics."""

import argparse
import csv
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trial", type=Path)
    parser.add_argument("--backend", choices=["egl", "glfw", "osmesa"], default="egl")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--azimuth", type=float, default=115)
    parser.add_argument("--elevation", type=float, default=-30)
    parser.add_argument("--distance", type=float, default=3.0)
    args = parser.parse_args()
    if args.fps < 1 or args.width < 1 or args.height < 1 or args.distance <= 0:
        parser.error("FPS and image dimensions must be positive.")
    os.environ["MUJOCO_GL"] = args.backend
    import imageio.v2 as imageio
    import mujoco
    import numpy as np
    from PIL import Image, ImageDraw

    with (args.trial / "state.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise RuntimeError("No recorded states to render.")
    result_path = args.trial / "result.json"
    result = json.loads(result_path.read_text()) if result_path.exists() else {}
    if result.get("mujoco_version", mujoco.__version__) != mujoco.__version__:
        raise RuntimeError(f"Install mujoco=={result['mujoco_version']} to load this MJB snapshot.")
    shared_model = args.trial.parent / "render_model.mjb"
    # Older runs stored one snapshot per trial.
    model_path = shared_model if shared_model.exists() else args.trial / "scene.mjb"
    model = mujoco.MjModel.from_binary_path(str(model_path))
    data = mujoco.MjData(model)
    times = np.array([float(row["trial_time_s"]) for row in rows])
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, args.width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, args.height)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance = args.distance
    camera.azimuth = args.azimuth
    camera.elevation = args.elevation
    output = args.trial / "video.mp4"
    with mujoco.Renderer(model, height=args.height, width=args.width) as renderer, \
            imageio.get_writer(output, fps=args.fps, codec="libx264", quality=8,
                               macro_block_size=2) as writer:
        for t in np.arange(times[0], times[-1] + 0.5 / args.fps, 1 / args.fps):
            # Nearest recorded pose avoids inventing unobserved joint motion.
            after = int(np.searchsorted(times, t))
            candidates = {max(0, min(after, len(rows) - 1)), max(0, after - 1)}
            index = min(candidates, key=lambda i: abs(times[i] - t))
            row = rows[index]
            data.qpos[:] = [float(row[f"qpos_{i}"]) for i in range(model.nq)]
            data.time = float(row["sim_time_s"])
            mujoco.mj_forward(model, data)
            camera.lookat[:] = [float(row["x_m"]) + 0.45, float(row["y_m"]), 0.4]
            renderer.update_scene(data, camera=camera)
            frame = Image.fromarray(renderer.render())
            draw = ImageDraw.Draw(frame)
            draw.rectangle((0, 0, args.width, 48), fill=(15, 20, 25))
            label = (f"{result.get('scoring_rule', result.get('terrain', 'trial'))}  "
                     f"command={result.get('command_mps', '?')} m/s  "
                     f"t={times[index]:.2f} s  measured vx={float(row['forward_speed_mps']):.2f} m/s")
            draw.text((12, 8), label, fill="white")
            draw.text((12, 27), f"Recorded-state rendering | tilt={float(row['tilt_deg']):.1f} deg | "
                      f"outcome={result.get('outcome', 'incomplete')}", fill="white")
            writer.append_data(np.asarray(frame))
            if index == 0:
                frame.save(args.trial / "first_frame.png")
    print(f"Video: {output}")


if __name__ == "__main__":
    main()
