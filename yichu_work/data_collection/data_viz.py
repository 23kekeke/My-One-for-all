"""
Visualize recorded dataset with Rerun (rerun.io)

Usage:
  python data_viz.py episode_0000/
  python data_viz.py episode_0000/ --speed 2.0
  python data_viz.py ../collected_data/episode_0000/ --fps 30

Requires: rerun-sdk (pip install rerun-sdk)
"""

import json
import argparse
import time
from pathlib import Path
import numpy as np
import rerun as rr
import cv2
from PIL import Image


def load_video_frames(episode_dir, episode_data):
    """Pre-load all video frames into memory as numpy arrays."""
    storage_format = episode_data.get("storage_format", "images")
    video_files = episode_data.get("video_files", {})
    frames = episode_data.get("frames", [])
    cameras = list(frames[0].get("images", {}).keys()) if frames else []

    if storage_format == "video":
        print("  Loading video frames...")
        video_frames = {}
        for cam_name in cameras:
            video_file = video_files.get(cam_name)
            if not video_file:
                continue
            video_path = episode_dir / video_file
            if not video_path.exists():
                print(f"  ⚠️  Video not found: {video_path}")
                continue
            cap = cv2.VideoCapture(str(video_path))
            cam_frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                cam_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cap.release()
            video_frames[cam_name] = cam_frames
            print(f"    {cam_name}: {len(cam_frames)} frames")
        return video_frames, cameras, storage_format
    else:
        print("  Loading image frames...")
        image_frames = {}
        for cam_name in cameras:
            cam_frames = []
            for frame in frames:
                img_filename = frame.get("images", {}).get(cam_name)
                if img_filename:
                    img_path = episode_dir / img_filename
                    if img_path.exists():
                        img = Image.open(img_path).convert("RGB")
                        cam_frames.append(np.array(img))
            image_frames[cam_name] = cam_frames
            print(f"    {cam_name}: {len(cam_frames)} frames")
        return image_frames, cameras, storage_format


def build_scalars_from_joints(frames, joint_names_dict):
    """Build time-indexed scalar lists for each joint."""
    import collections
    scalars = collections.defaultdict(list)
    timestamps = []
    for frame in frames:
        ts = frame.get("timestamp", 0)
        timestamps.append(ts)
        obs = frame.get("observation", {})
        for part_key, joint_list in joint_names_dict.items():
            joint_state_key = f"{part_key}_joint_states"
            if joint_state_key not in obs:
                continue
            positions = obs[joint_state_key].get("positions", [])
            for i, jname in enumerate(joint_list):
                if i < len(positions):
                    scalars[(part_key, i, jname)].append((ts, positions[i]))
    return scalars, timestamps


def main():
    parser = argparse.ArgumentParser(description="Visualize recorded dataset with Rerun")
    parser.add_argument("episode_dir", type=str, help="Episode directory (e.g. episode_0000/)")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier")
    parser.add_argument("--fps", type=float, default=30.0, help="Target display FPS")
    parser.add_argument("--no-loop", action="store_true", help="Disable loop playback")
    parser.add_argument("--connect", type=str, default=None,
                        help="Connect to existing Rerun viewer (e.g. 127.0.0.1:9876)")
    args = parser.parse_args()

    episode_path = Path(args.episode_dir)
    episode_json = episode_path / "episode.json"
    if not episode_json.exists():
        print(f"Error: {episode_json} not found")
        return

    print(f"Loading {episode_json}...")
    with open(episode_json) as f:
        episode_data = json.load(f)

    frames = episode_data.get("frames", [])
    if not frames:
        print("Error: No frames found")
        return

    joint_names_dict = episode_data.get("joint_names", {})
    robot_model = episode_data.get("model", "unknown")
    num_frames = len(frames)
    duration = episode_data.get("duration", 0)
    task = episode_data.get("task", "unknown")

    print(f"  Model: {robot_model}")
    print(f"  Task: {task}")
    print(f"  Frames: {num_frames}, Duration: {duration:.2f}s")

    # Pre-load frames into memory for smooth playback
    video_frames, cameras, storage_format = load_video_frames(episode_path, episode_data)

    # Get timestamps
    timestamps = [f.get("timestamp", i / 30.0) for i, f in enumerate(frames)]
    t0 = timestamps[0] if timestamps else 0

    # Build scalar data for joint positions
    scalars, _ = build_scalars_from_joints(frames, joint_names_dict)

    # ---- Timestamp analysis ----
    timestamps_np = np.array(timestamps, dtype=np.float64)
    intervals = np.diff(timestamps_np)  # frame-to-frame intervals
    if len(intervals) > 1:
        target_period = np.median(intervals)  # estimated ideal frame period
        ideal_times = timestamps_np[0] + np.arange(num_frames) * target_period
        drift = timestamps_np - ideal_times  # cumulative drift from ideal clock
    else:
        target_period = 0
        drift = np.zeros(num_frames)
    expected_fps = 1.0 / target_period if target_period > 0 else 0

    # Stats for display
    if len(intervals) > 0:
        interval_stats = {
            "mean": float(np.mean(intervals)),
            "median": float(np.median(intervals)),
            "min": float(np.min(intervals)),
            "max": float(np.max(intervals)),
            "std": float(np.std(intervals)),
        }
    else:
        interval_stats = {}

    # Setup Rerun
    rr.init("data_viz", spawn=True)

    # Set up the view
    rr.log("description", rr.TextDocument(
        f"# {robot_model}\n"
        f"Task: {task}  |  "
        f"Frames: {num_frames}  |  "
        f"Duration: {duration:.2f}s\n"
        f"Cameras: {', '.join(cameras)}",
        media_type=rr.MediaType.MARKDOWN
    ), static=True)

    # ---- Timing summary ----
    if interval_stats:
        summary_lines = [
            f"## Frame Timing ({expected_fps:.1f} Hz target)\n",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Frames | {num_frames} |",
            f"| Duration | {duration:.2f}s |",
            f"| Mean interval | {interval_stats['mean']*1000:.1f} ms |",
            f"| Median interval | {interval_stats['median']*1000:.1f} ms |",
            f"| Min interval | {interval_stats['min']*1000:.1f} ms |",
            f"| Max interval | {interval_stats['max']*1000:.1f} ms |",
            f"| Std interval | {interval_stats['std']*1000:.2f} ms |",
            f"| Total drift | {drift[-1]*1000:.1f} ms |",
        ]
        summary = "\n".join(summary_lines)
        rr.log("timing/summary", rr.TextDocument(summary, media_type=rr.MediaType.MARKDOWN), static=True)

    print("\nStarting visualization...")
    print("  Press Ctrl+C to exit\n")

    play_loop = not args.no_loop
    frame_idx = 0

    try:
        while True:
            if frame_idx >= num_frames:
                if play_loop:
                    frame_idx = 0
                    rr.set_time("timestamp", timestamp=0)
                else:
                    break

            frame = frames[frame_idx]
            ts = frame.get("timestamp", t0 + frame_idx / args.fps)
            obs = frame.get("observation", {})
            act = frame.get("action", {})

            rr.set_time("timestamp", timestamp=ts)
            rr.set_time("frame", sequence=frame.get("frame_id", frame_idx))

            # ---- SeriesLines config on first frame ----
            if frame_idx == 0:
                for (part_key, joint_idx, jname) in scalars:
                    rr.log(f"{part_key}/obs/{jname}", rr.SeriesLines())
                    rr.log(f"{part_key}/act/{jname}", rr.SeriesLines())
                rr.log("timing/frame_interval", rr.SeriesLines())
                rr.log("timing/drift", rr.SeriesLines())

            # ---- Camera images ----
            img_data = frame.get("images", {})
            for cam_name in cameras:
                if storage_format == "video" and cam_name in video_frames:
                    cam_frames = video_frames.get(cam_name, [])
                    if frame_idx < len(cam_frames):
                        rr.log(f"camera/{cam_name}", rr.Image(cam_frames[frame_idx]))
                elif storage_format == "images":
                    img_filename = img_data.get(cam_name)
                    if img_filename:
                        img_path = episode_path / img_filename
                        if img_path.exists():
                            img = Image.open(img_path).convert("RGB")
                            rr.log(f"camera/{cam_name}", rr.Image(np.array(img)))

            # ---- Joint positions (observation) ----
            for (part_key, joint_idx, jname) in scalars:
                obs_joint_key = f"{part_key}_joint_states"
                joint_obs = obs.get(obs_joint_key, {})
                positions = joint_obs.get("positions", [])
                if joint_idx < len(positions):
                    rr.log(f"{part_key}/obs/{jname}", rr.Scalars(positions[joint_idx]))

            # ---- Joint actions ----
            for (part_key, joint_idx, jname) in scalars:
                action_key = f"{part_key}_actions"
                part_act = act.get(action_key, {})
                act_positions = part_act.get("positions", [])
                if joint_idx < len(act_positions):
                    rr.log(f"{part_key}/act/{jname}", rr.Scalars(act_positions[joint_idx]))

            # ---- Frame timing visualization ----
            if frame_idx == 0:
                rr.log("timing/frame_interval", rr.Scalars(0.0))
                rr.log("timing/drift", rr.Scalars(0.0))
            elif frame_idx < num_frames:
                dt_ms = (timestamps[frame_idx] - timestamps[frame_idx - 1]) * 1000
                rr.log("timing/frame_interval", rr.Scalars(dt_ms))
                rr.log("timing/drift", rr.Scalars(drift[frame_idx] * 1000))

            # Frame rate control
            if frame_idx < num_frames - 1:
                dt = (timestamps[frame_idx + 1] - timestamps[frame_idx]) / args.speed
                sleep_time = max(0, dt - 1.0 / args.fps + 0.001)
                time.sleep(sleep_time)

            frame_idx += 1

    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        rr.disconnect()


if __name__ == "__main__":
    main()
