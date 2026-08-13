#!/usr/bin/env python3
"""Extract A2 head/hand HEVC streams and joint-state observations."""

from __future__ import annotations

import argparse
import csv
import json
import os
from contextlib import ExitStack
from pathlib import Path

import rosbag2_py
from foxglove_msgs.msg import CompressedVideo
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import JointState


CAMERA_TOPICS = {
    "/aima/hal/rgbd_camera/head_front/color/h265": "head_camera",
    "/aima/hal/rgbd_camera/hand_left/color/h265": "left_hand_camera",
    "/aima/hal/rgbd_camera/hand_right/color/h265": "right_hand_camera",
}
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
        **{
            topic: "foxglove_msgs/msg/CompressedVideo"
            for topic in CAMERA_TOPICS
        },
        **{topic: "sensor_msgs/msg/JointState" for topic in JOINT_TOPICS},
    }
    for topic, expected_type in expected.items():
        if recorded_types.get(topic) != expected_type:
            raise SystemExit(
                f"{topic}: expected {expected_type}, "
                f"found {recorded_types.get(topic)!r}"
            )

    camera_stats = {
        name: {
            "source_frames": 0,
            "dropped_before_keyframe": 0,
            "byte_count": 0,
            "stream_started": False,
            "timestamps": [],
        }
        for name in CAMERA_TOPICS.values()
    }
    joint_streams: dict[str, list[dict[str, object]]] = {
        name: [] for name in JOINT_TOPICS.values()
    }
    temporary_paths: list[tuple[Path, Path]] = []

    try:
        with ExitStack() as stack:
            video_outputs = {}
            csv_writers = {}

            for camera_name in CAMERA_TOPICS.values():
                video_path = output_dir / f"{camera_name}.h265"
                video_tmp = output_dir / f"{camera_name}.h265.tmp"
                csv_path = output_dir / f"{camera_name}_timestamps.csv"
                csv_tmp = output_dir / f"{camera_name}_timestamps.csv.tmp"
                temporary_paths.extend(
                    [(video_tmp, video_path), (csv_tmp, csv_path)]
                )

                video_outputs[camera_name] = stack.enter_context(
                    video_tmp.open("wb")
                )
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
                csv_writers[camera_name] = writer

            while reader.has_next():
                topic, serialized_data, bag_timestamp_ns = reader.read_next()
                camera_name = CAMERA_TOPICS.get(topic)

                if camera_name is not None:
                    message = deserialize_message(serialized_data, CompressedVideo)
                    if message.format.lower() not in {"h265", "hevc"}:
                        raise RuntimeError(
                            f"Unexpected format on {topic}: {message.format!r}"
                        )
                    encoded_data = bytes(message.data)
                    stats = camera_stats[camera_name]
                    source_index = int(stats["source_frames"])
                    stats["source_frames"] = source_index + 1
                    if not encoded_data:
                        raise RuntimeError(
                            f"Empty H.265 frame on {topic}, index {source_index}"
                        )

                    nal_types = h265_nal_types(encoded_data)
                    if not bool(stats["stream_started"]):
                        has_parameter_sets = {32, 33, 34}.issubset(nal_types)
                        has_random_access = any(
                            16 <= nal_type <= 23 for nal_type in nal_types
                        )
                        if not (has_parameter_sets and has_random_access):
                            stats["dropped_before_keyframe"] = (
                                int(stats["dropped_before_keyframe"]) + 1
                            )
                            continue
                        stats["stream_started"] = True

                    timestamp_ns, timestamp_source = stamp_ns(
                        message.timestamp, bag_timestamp_ns
                    )
                    timestamps = stats["timestamps"]
                    if not isinstance(timestamps, list):
                        raise RuntimeError("Internal timestamp state is invalid")
                    if timestamps and timestamp_ns <= timestamps[-1]:
                        raise RuntimeError(
                            f"Non-monotonic timestamp on {topic}: {timestamp_ns}"
                        )

                    video_outputs[camera_name].write(encoded_data)
                    csv_writers[camera_name].writerow(
                        [
                            len(timestamps),
                            source_index,
                            timestamp_ns,
                            f"{timestamp_ns / 1_000_000_000:.9f}",
                            timestamp_source,
                            int(bag_timestamp_ns),
                            len(encoded_data),
                            " ".join(str(value) for value in nal_types),
                        ]
                    )
                    timestamps.append(timestamp_ns)
                    stats["byte_count"] = (
                        int(stats["byte_count"]) + len(encoded_data)
                    )
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

        cameras_json = {}
        for topic, camera_name in CAMERA_TOPICS.items():
            stats = camera_stats[camera_name]
            timestamps = stats["timestamps"]
            if not isinstance(timestamps, list) or not timestamps:
                raise RuntimeError(
                    f"No decodable H.265 keyframe found on {topic}"
                )
            duration = (
                (timestamps[-1] - timestamps[0]) / 1_000_000_000
                if len(timestamps) > 1
                else 0.0
            )
            measured_fps = (
                (len(timestamps) - 1) / duration if duration > 0 else 0.0
            )
            cameras_json[camera_name] = {
                "topic": topic,
                "codec": "h265",
                "elementary_stream_file": f"{camera_name}.h265",
                "timestamp_file": f"{camera_name}_timestamps.csv",
                "source_frame_count": int(stats["source_frames"]),
                "dropped_before_keyframe": int(
                    stats["dropped_before_keyframe"]
                ),
                "frame_count": len(timestamps),
                "first_timestamp_ns": timestamps[0],
                "last_timestamp_ns": timestamps[-1],
                "duration_s": duration,
                "measured_fps": measured_fps,
                "byte_count": int(stats["byte_count"]),
            }

        episode = {
            "format_version": 1,
            "episode_id": bag_dir.name,
            "source_bag": str(bag_dir),
            "action_available": False,
            "cameras": cameras_json,
            "joint_states": {
                name: {
                    "topic": topic,
                    "sample_count": len(joint_streams[name]),
                    "samples": joint_streams[name],
                }
                for topic, name in JOINT_TOPICS.items()
            },
        }

        json_path = output_dir / "episode.json"
        json_tmp = output_dir / "episode.json.tmp"
        temporary_paths.append((json_tmp, json_path))
        with json_tmp.open("w", encoding="utf-8") as json_file:
            json.dump(
                episode,
                json_file,
                ensure_ascii=False,
                separators=(",", ":"),
            )

        for temporary_path, final_path in temporary_paths:
            os.replace(temporary_path, final_path)
    except Exception:
        for temporary_path, _ in temporary_paths:
            temporary_path.unlink(missing_ok=True)
        raise

    for camera_name, camera in cameras_json.items():
        print(
            f"{camera_name}: "
            f"source_frames={camera['source_frame_count']}, "
            f"dropped_before_keyframe={camera['dropped_before_keyframe']}, "
            f"kept_frames={camera['frame_count']}, "
            f"duration={camera['duration_s']:.6f}s, "
            f"measured_fps={camera['measured_fps']:.6f}, "
            f"bytes={camera['byte_count']}"
        )
    for name, samples in joint_streams.items():
        print(f"{name}: samples={len(samples)}")
    print(f"output_dir={output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
