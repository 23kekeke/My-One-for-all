#!/usr/bin/env python3
"""Extract A2 head HEVC video and joint-state observations from a ROS 2 bag."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import rosbag2_py
from foxglove_msgs.msg import CompressedVideo
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import JointState


CAMERA_TOPIC = "/aima/hal/rgbd_camera/head_front/color/h265"
JOINT_TOPICS = {
    "/motion/control/arm_joint_state": "arm_joint_state",
    "/motion/control/hand_joint_state": "hand_joint_state",
    "/motion/control/neck_joint_state": "neck_joint_state",
}


def stamp_ns(stamp: object, bag_timestamp_ns: int) -> tuple[int, str]:
    value = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
    if value > 0:
        return value, "message"
    return int(bag_timestamp_ns), "bag"


def h265_nal_types(data: bytes) -> list[int]:
    """Return HEVC NAL-unit types from Annex-B encoded data."""
    result: list[int] = []
    index = 0
    while index + 4 <= len(data):
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


def float_list(values: object) -> list[float]:
    return [float(value) for value in values]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bag_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    bag_dir = args.bag_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not (bag_dir / "metadata.yaml").is_file():
        raise SystemExit(f"metadata.yaml not found: {bag_dir}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag_dir), storage_id="sqlite3"),
        rosbag2_py.ConverterOptions("cdr", "cdr"),
    )

    recorded_types = {
        item.name: item.type for item in reader.get_all_topics_and_types()
    }
    expected = {
        CAMERA_TOPIC: "foxglove_msgs/msg/CompressedVideo",
        **{topic: "sensor_msgs/msg/JointState" for topic in JOINT_TOPICS},
    }
    for topic, expected_type in expected.items():
        if recorded_types.get(topic) != expected_type:
            raise SystemExit(
                f"{topic}: expected {expected_type}, "
                f"found {recorded_types.get(topic)!r}"
            )

    video_path = output_dir / "head_camera.h265"
    video_tmp = output_dir / "head_camera.h265.tmp"
    csv_path = output_dir / "head_camera_timestamps.csv"
    csv_tmp = output_dir / "head_camera_timestamps.csv.tmp"
    json_path = output_dir / "episode.json"
    json_tmp = output_dir / "episode.json.tmp"

    camera_timestamps: list[int] = []
    joint_streams: dict[str, list[dict[str, object]]] = {
        name: [] for name in JOINT_TOPICS.values()
    }
    source_frames = 0
    dropped_before_keyframe = 0
    byte_count = 0
    stream_started = False

    try:
        with video_tmp.open("wb") as video_file, csv_tmp.open(
            "w", newline="", encoding="utf-8"
        ) as csv_file:
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

            while reader.has_next():
                topic, serialized_data, bag_timestamp_ns = reader.read_next()

                if topic == CAMERA_TOPIC:
                    message = deserialize_message(serialized_data, CompressedVideo)
                    if message.format.lower() not in {"h265", "hevc"}:
                        raise RuntimeError(
                            f"Unexpected camera format: {message.format!r}"
                        )
                    encoded_data = bytes(message.data)
                    if not encoded_data:
                        raise RuntimeError(
                            f"Empty H.265 frame at source index {source_frames}"
                        )

                    nal_types = h265_nal_types(encoded_data)
                    source_index = source_frames
                    source_frames += 1

                    if not stream_started:
                        has_parameter_sets = {32, 33, 34}.issubset(nal_types)
                        has_random_access = any(
                            16 <= nal_type <= 23 for nal_type in nal_types
                        )
                        if not (has_parameter_sets and has_random_access):
                            dropped_before_keyframe += 1
                            continue
                        stream_started = True

                    timestamp_ns, timestamp_source = stamp_ns(
                        message.timestamp, bag_timestamp_ns
                    )
                    if camera_timestamps and timestamp_ns <= camera_timestamps[-1]:
                        raise RuntimeError(
                            f"Non-monotonic camera timestamp: {timestamp_ns}"
                        )

                    video_file.write(encoded_data)
                    writer.writerow(
                        [
                            len(camera_timestamps),
                            source_index,
                            timestamp_ns,
                            f"{timestamp_ns / 1_000_000_000:.9f}",
                            timestamp_source,
                            int(bag_timestamp_ns),
                            len(encoded_data),
                            " ".join(str(value) for value in nal_types),
                        ]
                    )
                    camera_timestamps.append(timestamp_ns)
                    byte_count += len(encoded_data)
                    continue

                stream_name = JOINT_TOPICS.get(topic)
                if stream_name is None:
                    continue

                message = deserialize_message(serialized_data, JointState)
                timestamp_ns, timestamp_source = stamp_ns(
                    message.header.stamp, bag_timestamp_ns
                )
                joint_streams[stream_name].append(
                    {
                        "timestamp_ns": timestamp_ns,
                        "timestamp_source": timestamp_source,
                        "bag_timestamp_ns": int(bag_timestamp_ns),
                        "name": list(message.name),
                        "position": float_list(message.position),
                        "velocity": float_list(message.velocity),
                        "effort": float_list(message.effort),
                    }
                )

        if not camera_timestamps:
            raise RuntimeError("No decodable H.265 keyframe was found")

        duration = (
            (camera_timestamps[-1] - camera_timestamps[0]) / 1_000_000_000
            if len(camera_timestamps) > 1
            else 0.0
        )
        measured_fps = (
            (len(camera_timestamps) - 1) / duration if duration > 0 else 0.0
        )

        episode = {
            "format_version": 1,
            "episode_id": bag_dir.name,
            "source_bag": str(bag_dir),
            "action_available": False,
            "camera": {
                "head_camera": {
                    "topic": CAMERA_TOPIC,
                    "codec": "h265",
                    "elementary_stream_file": video_path.name,
                    "timestamp_file": csv_path.name,
                    "source_frame_count": source_frames,
                    "dropped_before_keyframe": dropped_before_keyframe,
                    "frame_count": len(camera_timestamps),
                    "first_timestamp_ns": camera_timestamps[0],
                    "last_timestamp_ns": camera_timestamps[-1],
                    "duration_s": duration,
                    "measured_fps": measured_fps,
                    "byte_count": byte_count,
                }
            },
            "joint_states": {
                name: {
                    "topic": topic,
                    "sample_count": len(joint_streams[name]),
                    "samples": joint_streams[name],
                }
                for topic, name in JOINT_TOPICS.items()
            },
        }

        with json_tmp.open("w", encoding="utf-8") as json_file:
            json.dump(
                episode,
                json_file,
                ensure_ascii=False,
                separators=(",", ":"),
            )

        os.replace(video_tmp, video_path)
        os.replace(csv_tmp, csv_path)
        os.replace(json_tmp, json_path)
    except Exception:
        video_tmp.unlink(missing_ok=True)
        csv_tmp.unlink(missing_ok=True)
        json_tmp.unlink(missing_ok=True)
        raise

    print(
        f"head_camera: source_frames={source_frames}, "
        f"dropped_before_keyframe={dropped_before_keyframe}, "
        f"kept_frames={len(camera_timestamps)}, "
        f"duration={duration:.6f}s, measured_fps={measured_fps:.6f}, "
        f"bytes={byte_count}"
    )
    for name, samples in joint_streams.items():
        print(f"{name}: samples={len(samples)}")
    print(f"output_dir={output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
