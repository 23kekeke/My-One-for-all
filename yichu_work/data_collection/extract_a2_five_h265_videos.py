#!/usr/bin/env python3
"""Extract five A2 HEVC streams from rosbag2 and remux them to MP4."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import rosbag2_py
from foxglove_msgs.msg import CompressedVideo
from rclpy.serialization import deserialize_message


CAMERA_TOPICS = {
    "/aima/hal/rgbd_camera/head_front/color/h265": "head_camera",
    "/aima/hal/rgbd_camera/hand_left/color/h265": "left_hand_camera",
    "/aima/hal/rgbd_camera/hand_right/color/h265": "right_hand_camera",
    "/aima/hal/fish_eye_camera/chest_left/color/h265": "left_chest_camera",
    "/aima/hal/fish_eye_camera/chest_right/color/h265": "right_chest_camera",
}

EXPECTED_TYPE = "foxglove_msgs/msg/CompressedVideo"


def timestamp_ns(message: CompressedVideo, bag_timestamp_ns: int) -> tuple[int, str]:
    value = (
        int(message.timestamp.sec) * 1_000_000_000
        + int(message.timestamp.nanosec)
    )
    if value > 0:
        return value, "message"
    return int(bag_timestamp_ns), "bag"


def h265_nal_types(data: bytes) -> list[int]:
    result: list[int] = []
    index = 0
    while index + 3 <= len(data):
        prefix_length = 0
        if data[index : index + 4] == b"\x00\x00\x00\x01":
            prefix_length = 4
        elif data[index : index + 3] == b"\x00\x00\x01":
            prefix_length = 3
        if prefix_length:
            header_index = index + prefix_length
            if header_index < len(data):
                result.append((data[header_index] >> 1) & 0x3F)
            index = header_index + 1
        else:
            index += 1
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bag_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    return parser.parse_args()


def remux_to_mp4(raw_path: Path, mp4_path: Path, fps: float) -> None:
    command = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "warning",
        "-fflags",
        "+genpts",
        "-r",
        f"{fps:.9f}",
        "-i",
        str(raw_path),
        "-map",
        "0:v:0",
        "-c:v",
        "copy",
        "-tag:v",
        "hvc1",
        "-movflags",
        "+faststart",
        str(mp4_path),
    ]
    subprocess.run(command, check=True)


def main() -> int:
    args = parse_args()
    bag_dir = args.bag_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not (bag_dir / "metadata.yaml").is_file():
        raise SystemExit(f"metadata.yaml not found: {bag_dir}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"Output directory is not empty: {output_dir}")
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg was not found in PATH")

    output_dir.mkdir(parents=True, exist_ok=True)

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag_dir), storage_id="sqlite3"),
        rosbag2_py.ConverterOptions("cdr", "cdr"),
    )
    recorded_types = {
        item.name: item.type for item in reader.get_all_topics_and_types()
    }
    for topic in CAMERA_TOPICS:
        if recorded_types.get(topic) != EXPECTED_TYPE:
            raise SystemExit(
                f"{topic}: expected {EXPECTED_TYPE}, "
                f"found {recorded_types.get(topic)!r}"
            )

    stats: dict[str, dict[str, Any]] = {
        camera: {
            "topic": topic,
            "source_frames": 0,
            "dropped_before_keyframe": 0,
            "kept_frames": 0,
            "byte_count": 0,
            "stream_started": False,
            "timestamps": [],
        }
        for topic, camera in CAMERA_TOPICS.items()
    }
    temporary_paths: list[tuple[Path, Path]] = []

    try:
        with ExitStack() as stack:
            video_files = {}
            csv_writers = {}

            for camera in CAMERA_TOPICS.values():
                raw_path = output_dir / f"{camera}.h265"
                raw_tmp = output_dir / f"{camera}.h265.tmp"
                csv_path = output_dir / f"{camera}_timestamps.csv"
                csv_tmp = output_dir / f"{camera}_timestamps.csv.tmp"

                temporary_paths.extend(
                    [(raw_tmp, raw_path), (csv_tmp, csv_path)]
                )
                video_files[camera] = stack.enter_context(raw_tmp.open("wb"))
                csv_file = stack.enter_context(
                    csv_tmp.open("w", newline="", encoding="utf-8")
                )
                writer = csv.writer(csv_file)
                writer.writerow(
                    [
                        "frame_index",
                        "source_frame_index",
                        "timestamp_ns",
                        "timestamp_s",
                        "timestamp_source",
                        "bag_timestamp_ns",
                        "data_size",
                        "nal_types",
                    ]
                )
                csv_writers[camera] = writer

            while reader.has_next():
                topic, serialized_data, bag_timestamp_ns = reader.read_next()
                camera = CAMERA_TOPICS.get(topic)
                if camera is None:
                    continue

                message = deserialize_message(serialized_data, CompressedVideo)
                if message.format.lower() not in {"h265", "hevc"}:
                    raise RuntimeError(
                        f"Unexpected format on {topic}: {message.format!r}"
                    )
                encoded_data = bytes(message.data)
                entry = stats[camera]
                source_index = int(entry["source_frames"])
                entry["source_frames"] = source_index + 1
                if not encoded_data:
                    raise RuntimeError(
                        f"Empty H.265 access unit on {topic}, "
                        f"source index {source_index}"
                    )

                nal_types = h265_nal_types(encoded_data)
                if not bool(entry["stream_started"]):
                    has_parameter_sets = {32, 33, 34}.issubset(nal_types)
                    has_random_access = any(
                        16 <= nal_type <= 23 for nal_type in nal_types
                    )
                    if not (has_parameter_sets and has_random_access):
                        entry["dropped_before_keyframe"] = (
                            int(entry["dropped_before_keyframe"]) + 1
                        )
                        continue
                    entry["stream_started"] = True

                source_timestamp_ns, source_kind = timestamp_ns(
                    message, bag_timestamp_ns
                )
                timestamps = entry["timestamps"]
                if not isinstance(timestamps, list):
                    raise RuntimeError("Internal timestamp state is invalid")
                if timestamps and source_timestamp_ns <= timestamps[-1]:
                    raise RuntimeError(
                        f"Non-monotonic timestamp on {topic}: "
                        f"{source_timestamp_ns} after {timestamps[-1]}"
                    )

                video_files[camera].write(encoded_data)
                csv_writers[camera].writerow(
                    [
                        len(timestamps),
                        source_index,
                        source_timestamp_ns,
                        f"{source_timestamp_ns / 1_000_000_000:.9f}",
                        source_kind,
                        int(bag_timestamp_ns),
                        len(encoded_data),
                        " ".join(str(value) for value in nal_types),
                    ]
                )
                timestamps.append(source_timestamp_ns)
                entry["kept_frames"] = int(entry["kept_frames"]) + 1
                entry["byte_count"] = int(entry["byte_count"]) + len(
                    encoded_data
                )

        for temporary_path, final_path in temporary_paths:
            os.replace(temporary_path, final_path)

        metadata = {
            "source_bag": str(bag_dir),
            "codec": "h265",
            "cameras": {},
        }
        for camera, entry in stats.items():
            timestamps = entry["timestamps"]
            if not isinstance(timestamps, list) or not timestamps:
                raise RuntimeError(
                    f"No decodable H.265 keyframe found for {camera}"
                )
            duration_s = (
                (timestamps[-1] - timestamps[0]) / 1_000_000_000
                if len(timestamps) > 1
                else 0.0
            )
            measured_fps = (
                (len(timestamps) - 1) / duration_s
                if duration_s > 0
                else 30.0
            )
            raw_path = output_dir / f"{camera}.h265"
            mp4_path = output_dir / f"{camera}.mp4"
            remux_to_mp4(raw_path, mp4_path, measured_fps)

            metadata["cameras"][camera] = {
                "topic": entry["topic"],
                "source_frames": int(entry["source_frames"]),
                "dropped_before_keyframe": int(
                    entry["dropped_before_keyframe"]
                ),
                "kept_frames": int(entry["kept_frames"]),
                "first_timestamp_ns": timestamps[0],
                "last_timestamp_ns": timestamps[-1],
                "duration_s": duration_s,
                "measured_fps": measured_fps,
                "byte_count": int(entry["byte_count"]),
                "h265_file": raw_path.name,
                "mp4_file": mp4_path.name,
                "timestamp_file": f"{camera}_timestamps.csv",
            }

        metadata_path = output_dir / "video_metadata.json"
        with metadata_path.open("w", encoding="utf-8") as file:
            json.dump(metadata, file, indent=2, ensure_ascii=False)
    except Exception:
        for temporary_path, _ in temporary_paths:
            temporary_path.unlink(missing_ok=True)
        raise

    for camera, entry in metadata["cameras"].items():
        print(
            f"{camera}: source_frames={entry['source_frames']}, "
            f"dropped_before_keyframe={entry['dropped_before_keyframe']}, "
            f"kept_frames={entry['kept_frames']}, "
            f"fps={entry['measured_fps']:.6f}, "
            f"duration={entry['duration_s']:.6f}s, "
            f"mp4={entry['mp4_file']}"
        )
    print(f"output_dir={output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
