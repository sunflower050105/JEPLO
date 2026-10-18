#!/usr/bin/env python3
"""Sequential, simulation-only velocity trials using the existing JEPLO controller."""

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import socket
import struct
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_status(directory):
    try:
        return json.loads((directory / "status.json").read_text())
    except FileNotFoundError:
        return None


def decode_depth(message):
    """Return publisher receive/publication times, not estimator consumption time."""
    if len(message) < 8:
        return None
    width, height = struct.unpack_from("=II", message)
    payload_end = 8 + width * height * 4
    if len(message) != payload_end + 16:
        return None
    scan_receive_ns, publish_ns = struct.unpack_from("=QQ", message, payload_end)
    return width, height, scan_receive_ns, publish_ns


def commands(args, trial, speed, seed):
    sim = [str(args.sim_bin), "--lidar", "--lidar-hz", str(args.lidar_hz),
           "--network", "lo", "--domain_id", "1", "--benchmark-dir", str(trial),
           "--benchmark-terrain", args.terrain, "--benchmark-speed", str(speed),
           "--benchmark-box-task", args.box_task,
           "--benchmark-duration", str(args.duration), "--benchmark-min-progress-ratio",
           str(args.min_progress_ratio)]
    if args.legacy_lidar_fov:
        sim.append("--lidar-legacy-fov")
    depth = [str(args.depth_bin), "--sim", "--fov", args.fov,
             "--downsample-rate", str(args.downsample_rate),
             "--stacked-frames", str(args.stacked_frames), "--cage-seed", str(seed)]
    if args.no_cage_mask:
        depth.append("--no-cage-mask")
    policy = [str(args.deploy_bin), "--net", "lo", "--model-dir", args.model_dir,
              "--ood-count-threshold", "15", "--raw-actions", "--trt-cache"]
    return {"simulator": sim, "depth": depth, "deploy": policy}


def scoring_rule(args):
    if args.terrain == "flat":
        return "flat_progress_v1"
    return "box_ascent_v2" if args.box_task == "ascent" else "box_traverse_v1"


def summarize(results):
    summary = []
    groups = {(r["command_mps"], r.get("scoring_rule", "unspecified")) for r in results}
    for speed, rule in sorted(groups):
        attempted = [r for r in results if r["command_mps"] == speed and
                     r.get("scoring_rule", "unspecified") == rule]
        valid = [r for r in attempted if r["valid"]]
        successes = sum(r["success"] for r in valid)
        summary.append({"command_mps": speed, "scoring_rule": rule, "attempted": len(attempted),
                        "valid_trials": len(valid), "invalid_trials": len(attempted) - len(valid),
                        "successes": successes,
                        "success_rate": successes / len(valid) if valid else None})
    return summary


def export_results(output, results):
    write_json(output / "results.json", results)
    write_json(output / "summary.json", summarize(results))
    fields = ["pair_id", "command_mps", "terrain", "box_task", "scoring_rule",
              "valid", "success", "outcome", "duration_s", "ascent_time_s",
              "progress_m", "mean_speed_mps", "speed_rmse_mps", "max_tilt_deg",
              "real_time_factor", "timing_ok", "directory", "error"]
    with (output / "results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    with (output / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, ["command_mps", "scoring_rule", "attempted", "valid_trials",
                                        "invalid_trials", "successes", "success_rate"])
        writer.writeheader()
        writer.writerows(summarize(results))


def preflight(args):
    if importlib.util.find_spec("zmq") is None:
        raise RuntimeError("pyzmq is missing. Use the benchmark venv from README.md.")
    for binary in (args.sim_bin, args.depth_bin, args.deploy_bin):
        if not binary.is_file() or not os.access(binary, os.X_OK):
            raise RuntimeError(f"Executable missing: {binary}. Build inside Ubuntu 22.04 first.")
        linked = subprocess.run(["ldd", str(binary)], capture_output=True, text=True)
        missing = [line.strip() for line in linked.stdout.splitlines() if "not found" in line]
        if missing:
            raise RuntimeError(f"{binary.name}: missing libraries: {missing}. Set LD_LIBRARY_PATH as documented.")
    cfg = json.loads((args.deploy_bin.parent / "config.json").read_text())
    if max(args.speeds) > cfg["linear_velocity_clip"]:
        raise RuntimeError("Requested speed exceeds deployment linear_velocity_clip.")
    model = args.deploy_bin.parent / "model" / args.model_dir
    for name in ("policy.onnx", "sensor_estimator.onnx"):
        if not (model / name).is_file():
            raise RuntimeError(f"Model file missing: {model / name}")
    if not os.environ.get("DISPLAY"):
        raise RuntimeError("DISPLAY is required for this local MuJoCo simulator. Run from a desktop container terminal.")
    for port in (5560, 5562, 5590):
        with socket.socket() as sock:
            try:
                sock.bind(("0.0.0.0", port))
            except OSError as exc:
                raise RuntimeError(f"Port {port} is occupied. Close the old simulator/depth/controller processes.") from exc
    # DDS has no exclusive port binding: reject a second controller/simulator.
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            executable = (entry / "exe").resolve(strict=True).name
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if executable in {"deploy", "deploy_see_then_hidden", "unitree_mujoco", "unitree_mujoco_velocity"}:
            raise RuntimeError(f"Existing {executable} process {entry.name}; close it before running the benchmark.")
    if args.video:
        for module in ("mujoco", "imageio", "imageio_ffmpeg", "PIL"):
            if importlib.util.find_spec(module) is None:
                raise RuntimeError(f"Video dependency {module} missing. Install requirements.txt in the benchmark venv.")
    return model


def stop_children(children):
    for process in reversed(children):
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + 3
    for process in children:
        try:
            process.wait(timeout=max(0.01, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()


def run_trial(args, directory, speed, seed):
    import zmq

    directory.mkdir()
    argv = commands(args, directory, speed, seed)
    write_json(directory / "commands.json", argv)
    children, logs = [], []
    context = zmq.Context()
    publisher = context.socket(zmq.PUB)
    publisher.setsockopt(zmq.LINGER, 0)
    publisher.bind("tcp://127.0.0.1:5562")
    depth_sub = context.socket(zmq.SUB)
    depth_sub.setsockopt(zmq.SUBSCRIBE, b"")
    depth_sub.setsockopt(zmq.LINGER, 0)
    depth_sub.connect("tcp://127.0.0.1:5560")
    started = time.monotonic()
    armed_at = None
    ready_sim = None
    last_depth = 0.0
    last_sim = None
    sim_changed_at = started
    policy_ready = policy_running = False
    next_tick = started
    try:
        for name in ("simulator", "depth", "deploy"):
            log = (directory / f"{name}.log").open("w")
            logs.append(log)
            process = subprocess.Popen(["stdbuf", "-oL", "-eL", *argv[name]],
                                       cwd=args.deploy_bin.parent if name == "deploy" else ROOT,
                                       stdin=subprocess.PIPE if name == "deploy" else subprocess.DEVNULL,
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            children.append(process)
            if name == "deploy":
                # The runner is restricted to simulation/loopback; answer its ONNX-ready prompt.
                process.stdin.write(b"\n")
                process.stdin.flush()
                process.stdin.close()

        with (directory / "deploy.log").open() as policy_log, \
                (directory / "sent_commands.csv").open("w", newline="") as cmd_file, \
                (directory / "depth_timing.csv").open("w", newline="") as depth_file:
            cmd_csv = csv.writer(cmd_file)
            cmd_csv.writerow(["wall_ns", "command_vx", "command_vy", "command_wz", "phase"])
            depth_csv = csv.writer(depth_file)
            depth_csv.writerow(["observer_wall_ns", "publisher_scan_receive_ns", "depth_publish_ns", "phase"])
            while True:
                now = time.monotonic()
                if (directory / "result.json").exists():
                    result = json.loads((directory / "result.json").read_text())
                    if result.get("scoring_rule") != scoring_rule(args):
                        raise RuntimeError("Simulator scoring rule differs from requested rule; rebuild the benchmark.")
                    result["valid"] = True
                    result["timing_ok"] = 0.95 <= result["real_time_factor"] <= 1.05
                    return result
                for name, child in zip(("simulator", "depth", "deploy"), children):
                    if child.poll() is not None:
                        raise RuntimeError(f"{name} exited with {child.returncode}; see {name}.log")
                text = policy_log.read()
                policy_ready |= "robot ready" in text
                policy_running |= "Current cmd:" in text
                status = read_status(directory)
                if status and status["sim_time_s"] != last_sim:
                    last_sim = status["sim_time_s"]
                    sim_changed_at = now
                if now - sim_changed_at > 10:
                    raise RuntimeError("Simulation stopped advancing for 10 s (paused UI or startup failure).")
                while True:
                    try:
                        message = depth_sub.recv(zmq.NOBLOCK)
                    except zmq.Again:
                        break
                    decoded = decode_depth(message)
                    if decoded:
                        width, height, scan_ns, publish_ns = decoded
                        if f"{height}x{width}" != args.fov:
                            raise RuntimeError("Depth grid differs from requested FOV.")
                        if scan_ns > 0 and 0 <= time.time_ns() - scan_ns < 1_000_000_000:
                            last_depth = now
                        depth_csv.writerow([time.time_ns(), scan_ns, publish_ns,
                                            "trial" if armed_at else "startup"])
                if armed_at is None:
                    if now - started > args.startup_timeout:
                        raise RuntimeError("Controller startup timed out; inspect deploy.log.")
                    if policy_ready and policy_running and status and now - last_depth < 0.5:
                        if ready_sim is None:
                            ready_sim = status["sim_time_s"]
                            print("  Controller active; warming up at zero command.", flush=True)
                        if status["sim_time_s"] - ready_sim >= args.warmup:
                            if (status["z_m"] < 0.16 or status["tilt_deg"] > 20 or
                                    abs(status["x_m"]) > 0.15 or abs(status["y_m"]) > 0.15 or
                                    abs(status["yaw_deg"]) > 5):
                                raise RuntimeError(f"Robot not at the allowed starting pose after warmup: {status}")
                            write_json(directory / "start.json", {**status, "command_switch_wall_ns": time.time_ns()})
                            (directory / "GO").touch()
                            armed_at = now
                            print(f"  Driving at {speed:.1f} m/s for up to {args.duration:g} simulation seconds.", flush=True)
                else:
                    if now - last_depth > 1:
                        raise RuntimeError("No fresh LiDAR-derived depth for 1 wall second during trial.")
                    if now - armed_at > max(30, args.duration * 4):
                        raise RuntimeError("Trial wall timeout; inspect simulation real-time factor.")
                velocity = speed if armed_at is not None else 0.0
                publisher.send(struct.pack("=fff", velocity, 0.0, 0.0))
                cmd_csv.writerow([time.time_ns(), velocity, 0, 0, "trial" if armed_at else "startup"])
                next_tick += 0.02
                delay = next_tick - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                else:
                    next_tick = time.monotonic()
    finally:
        for _ in range(3):
            publisher.send(struct.pack("=fff", 0.0, 0.0, 0.0))
            time.sleep(0.02)
        stop_children(children)
        for log in logs:
            log.close()
        publisher.close()
        depth_sub.close()
        context.term()


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--speeds", type=float, nargs="+", default=[0.8, 1.0])
    p.add_argument("--repeats", type=int, default=1, help="Paired trials per speed, run sequentially")
    p.add_argument("--terrain", choices=["flat", "boxes"], default="flat")
    p.add_argument("--box-task", choices=["ascent", "traverse"], default="ascent",
                   help="Boxes: climb onto box2 (default), or legacy far-edge crossing")
    p.add_argument("--duration", type=float, help="Simulation seconds; flat=8, boxes=12")
    p.add_argument("--warmup", type=float, default=2.0)
    p.add_argument("--min-progress-ratio", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=0, help="Cage mask seed; incremented per pair")
    p.add_argument("--model-dir", default="my_policy_r2")
    p.add_argument("--fov", choices=["25x60", "25x90", "25x120"], default="25x60")
    p.add_argument("--lidar-hz", type=float, default=20)
    p.add_argument("--stacked-frames", type=int, default=5)
    p.add_argument("--downsample-rate", type=int, default=4)
    p.add_argument("--no-cage-mask", action="store_true")
    p.add_argument("--legacy-lidar-fov", action="store_true")
    p.add_argument("--sim-bin", type=Path, default=ROOT / "deployment/unitree_mujoco/simulate/build-velocity-ubuntu2204/unitree_mujoco_velocity")
    p.add_argument("--depth-bin", type=Path, default=ROOT / "deployment/evaluation/build-velocity-ubuntu2204/lidar_depth_velocity")
    p.add_argument("--deploy-bin", type=Path, default=ROOT / "deployment/go2_deploy/output/deploy")
    p.add_argument("--startup-timeout", type=float, default=300)
    p.add_argument("--output", type=Path)
    p.add_argument("--video", action="store_true", help="Render all saved trajectories after the trials finish")
    p.add_argument("--render-backend", choices=["egl", "glfw", "osmesa"], default="egl")
    p.add_argument("--check", action="store_true", help="Check dependencies without starting trials")
    p.add_argument("--dry-run", action="store_true", help="Print exact process commands without launching")
    return p


def main():
    def terminate(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, terminate)
    p = parser()
    args = p.parse_args()
    args.duration = args.duration if args.duration is not None else (8.0 if args.terrain == "flat" else 12.0)
    values = args.speeds + [args.duration, args.warmup, args.lidar_hz, args.startup_timeout]
    if not all(math.isfinite(x) and x > 0 for x in values):
        p.error("Speeds, duration, warmup, LiDAR rate, and timeout must be finite and positive.")
    if args.repeats < 1 or args.stacked_frames < 1 or args.downsample_rate < 1:
        p.error("Repeats, stacked frames, and downsample rate must be positive integers.")
    if not 0 < args.min_progress_ratio <= 1 or not 0 <= args.seed <= 2**32 - args.repeats:
        p.error("Progress ratio must be in (0,1]; seed range must fit uint32 for all pairs.")
    if len(set(args.speeds)) != len(args.speeds):
        p.error("Speeds must be distinct.")
    for attr in ("sim_bin", "depth_bin", "deploy_bin"):
        setattr(args, attr, getattr(args, attr).expanduser().resolve())
    output = (args.output or HERE / "runs" / time.strftime("%Y%m%d-%H%M%S")).expanduser().resolve()
    if args.dry_run:
        print(json.dumps(commands(args, output / "pair001_v0.8", args.speeds[0], args.seed), indent=2))
        return 0
    model = preflight(args)
    if args.check:
        print("Preflight passed. Simulator, depth publisher, policy models, libraries, and ports are ready.")
        return 0
    output.mkdir(parents=True, exist_ok=False)
    manifest = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    manifest["scoring_rule"] = scoring_rule(args)
    manifest["hashes"] = {str(path): sha256(path) for path in
                          [model / "policy.onnx", model / "sensor_estimator.onnx",
                           args.deploy_bin.parent / "config.json", args.deploy_bin,
                           args.sim_bin, args.depth_bin, HERE / "run.py",
                           HERE / "velocity_trial.h", HERE / "box_ascent.h"]}
    manifest["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    manifest["git_dirty"] = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())
    write_json(output / "manifest.json", manifest)
    results = []
    interrupted = False
    print(f"Results: {output}", flush=True)
    print(f"Scoring rule: {scoring_rule(args)}", flush=True)
    for pair in range(args.repeats):
        speeds = args.speeds if pair % 2 == 0 else list(reversed(args.speeds))
        for speed in speeds:
            directory = output / f"pair{pair + 1:03d}_v{speed:g}"
            print(f"Pair {pair + 1}/{args.repeats}, {speed:g} m/s, {args.terrain}", flush=True)
            try:
                result = run_trial(args, directory, speed, args.seed + pair)
            except KeyboardInterrupt:
                result = {"valid": False, "success": False, "outcome": "interrupted", "error": "User interrupted"}
                interrupted = True
            except Exception as exc:
                result = {"valid": False, "success": False, "outcome": "invalid", "error": str(exc)}
                # Infrastructure failures need attention; avoid repeating a broken setup.
                interrupted = True
            result.update(pair_id=pair + 1, command_mps=speed, directory=str(directory))
            result.setdefault("scoring_rule", scoring_rule(args))
            result.setdefault("terrain", args.terrain)
            result.setdefault("box_task", args.box_task if args.terrain == "boxes" else None)
            results.append(result)
            export_results(output, results)
            print(f"  {result['outcome']}: {result.get('error', '')}", flush=True)
            if result["valid"]:
                print(f"  Mean vx={result['mean_speed_mps']:.3f} m/s; "
                      f"real-time factor={result['real_time_factor']:.3f}", flush=True)
                if not result["timing_ok"]:
                    print("  TIMING FLAG: simulation/wall-clock ratio is outside [0.95, 1.05]; "
                          "inspect timing before comparing this run.", flush=True)
            if interrupted:
                break
        if interrupted:
            break
    video_failed = False
    if args.video:
        # Render only after all trials, so rendering never competes with a subsequent trial.
        for result in results:
            directory = Path(result["directory"])
            if (directory / "state.csv").exists():
                completed = subprocess.run([sys.executable, str(HERE / "render_video.py"),
                                            str(directory), "--backend", args.render_backend])
                if completed.returncode:
                    video_failed = True
                    print(f"Video failed for {directory}; state.csv is preserved for a later retry.", file=sys.stderr)
    for row in summarize(results):
        rate = "n/a" if row["success_rate"] is None else f"{100 * row['success_rate']:.1f}%"
        print(f"{row['command_mps']:g} m/s [{row['scoring_rule']}]: "
              f"{row['successes']}/{row['valid_trials']} success ({rate}); "
              f"{row['invalid_trials']} invalid")
    print(f"Full results: {output / 'results.csv'}")
    return 1 if interrupted or video_failed else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (RuntimeError, OSError) as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)
