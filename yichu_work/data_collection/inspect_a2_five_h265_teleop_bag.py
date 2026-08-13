#!/usr/bin/env python3
"""Inspect A2 five-camera H.265 teleoperation rosbag quality without replay."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

import rosbag2_py
from foxglove_msgs.msg import CompressedVideo
from geometry_msgs.msg import PoseArray
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Imu, JointState


CAMERA_TOPICS = {
    "/aima/hal/rgbd_camera/head_front/color/h265": "head_camera",
    "/aima/hal/rgbd_camera/hand_left/color/h265": "left_hand_camera",
    "/aima/hal/rgbd_camera/hand_right/color/h265": "right_hand_camera",
    "/aima/hal/fish_eye_camera/chest_left/color/h265": "left_chest_camera",
    "/aima/hal/fish_eye_camera/chest_right/color/h265": "right_chest_camera",
}

JOINT_TOPICS = {
    "/motion/control/arm_joint_state": "arm_joint_state",
    "/motion/control/hand_joint_state": "hand_joint_state",
    "/motion/control/neck_joint_state": "neck_joint_state",
    "/motion/control/arm_joint_command": "arm_joint_command",
    "/motion/control/hand_joint_command": "hand_joint_command",
    "/motion/control/neck_joint_command": "neck_joint_command",
}

HAND_POSE_TOPIC = "/motion_control/hand_pose_state"
IMU_TOPIC = "/body_drive/imu/data"


def timestamp_ns(stamp: Any, bag_timestamp_ns: int) -> tuple[int, str]:
    value = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
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


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def timing_stats(
    timestamps: list[int],
    receive_offsets_ns: list[int],
    expected_hz: float | None,
) -> dict[str, Any]:
    deltas_ns = [
        current - previous
        for previous, current in zip(timestamps, timestamps[1:])
    ]
    positive_s = [value / 1_000_000_000 for value in deltas_ns if value > 0]
    duplicate_or_reverse = sum(value <= 0 for value in deltas_ns)
    duration_s = (
        (timestamps[-1] - timestamps[0]) / 1_000_000_000
        if len(timestamps) > 1
        else 0.0
    )
    measured_hz = (
        (len(timestamps) - 1) / duration_s
        if len(timestamps) > 1 and duration_s > 0
        else 0.0
    )
    expected_period_s = 1.0 / expected_hz if expected_hz else None
    long_gap_threshold_s = (
        max(0.1, expected_period_s * 3.0)
        if expected_period_s is not None
        else 0.1
    )
    long_gap_count = sum(value > long_gap_threshold_s for value in positive_s)

    offsets_ms = [value / 1_000_000 for value in receive_offsets_ns]
    return {
        "count": len(timestamps),
        "first_timestamp_ns": timestamps[0] if timestamps else None,
        "last_timestamp_ns": timestamps[-1] if timestamps else None,
        "duration_s": duration_s,
        "measured_hz": measured_hz,
        "duplicate_or_reverse_count": duplicate_or_reverse,
        "interval_ms": {
            "median": (
                statistics.median(positive_s) * 1000 if positive_s else None
            ),
            "p95": (
                percentile(positive_s, 0.95) * 1000 if positive_s else None
            ),
            "p99": (
                percentile(positive_s, 0.99) * 1000 if positive_s else None
            ),
            "max": max(positive_s) * 1000 if positive_s else None,
        },
        "long_gap_threshold_ms": long_gap_threshold_s * 1000,
        "long_gap_count": long_gap_count,
        "receive_minus_source_ms": {
            "median": statistics.median(offsets_ms) if offsets_ms else None,
            "p95": percentile(offsets_ms, 0.95),
            "p99": percentile(offsets_ms, 0.99),
            "min": min(offsets_ms) if offsets_ms else None,
            "max": max(offsets_ms) if offsets_ms else None,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bag_dir", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        help="Output JSON report path (default: BAG_DIR/quality_report.json)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    bag_dir = args.bag_dir.resolve()
    output_path = (
        args.output.resolve()
        if args.output
        else bag_dir / "quality_report.json"
    )
    if not (bag_dir / "metadata.yaml").is_file():
        raise SystemExit(f"metadata.yaml not found: {bag_dir}")

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag_dir), storage_id="sqlite3"),
        rosbag2_py.ConverterOptions("cdr", "cdr"),
    )
    recorded_types = {
        item.name: item.type for item in reader.get_all_topics_and_types()
    }

    camera_data: dict[str, dict[str, Any]] = {
        name: {
            "topic": topic,
            "timestamps": [],
            "receive_offsets_ns": [],
            "timestamp_sources": Counter(),
            "byte_count": 0,
            "format_values": Counter(),
            "first_decodable_index": None,
            "first_decodable_timestamp_ns": None,
            "nal_type_counts": Counter(),
        }
        for topic, name in CAMERA_TOPICS.items()
    }
    joint_data: dict[str, dict[str, Any]] = {
        name: {
            "topic": topic,
            "timestamps": [],
            "receive_offsets_ns": [],
            "timestamp_sources": Counter(),
            "schemas": Counter(),
            "first_sample": None,
        }
        for topic, name in JOINT_TOPICS.items()
    }
    other_data = {
        "hand_pose_state": {
            "topic": HAND_POSE_TOPIC,
            "timestamps": [],
            "receive_offsets_ns": [],
            "timestamp_sources": Counter(),
            "pose_counts": Counter(),
        },
        "imu": {
            "topic": IMU_TOPIC,
            "timestamps": [],
            "receive_offsets_ns": [],
            "timestamp_sources": Counter(),
        },
    }

    while reader.has_next():
        topic, serialized_data, bag_timestamp_ns = reader.read_next()

        camera_name = CAMERA_TOPICS.get(topic)
        if camera_name is not None:
            message = deserialize_message(serialized_data, CompressedVideo)
            source_ns, source_kind = timestamp_ns(
                message.timestamp, bag_timestamp_ns
            )
            entry = camera_data[camera_name]
            frame_index = len(entry["timestamps"])
            encoded_data = bytes(message.data)
            nal_types = h265_nal_types(encoded_data)

            entry["timestamps"].append(source_ns)
            entry["receive_offsets_ns"].append(
                int(bag_timestamp_ns) - source_ns
            )
            entry["timestamp_sources"][source_kind] += 1
            entry["byte_count"] += len(encoded_data)
            entry["format_values"][message.format] += 1
            entry["nal_type_counts"].update(nal_types)

            has_parameter_sets = {32, 33, 34}.issubset(nal_types)
            has_random_access = any(16 <= value <= 23 for value in nal_types)
            if (
                entry["first_decodable_index"] is None
                and has_parameter_sets
                and has_random_access
            ):
                entry["first_decodable_index"] = frame_index
                entry["first_decodable_timestamp_ns"] = source_ns
            continue

        joint_name = JOINT_TOPICS.get(topic)
        if joint_name is not None:
            message = deserialize_message(serialized_data, JointState)
            source_ns, source_kind = timestamp_ns(
                message.header.stamp, bag_timestamp_ns
            )
            entry = joint_data[joint_name]
            entry["timestamps"].append(source_ns)
            entry["receive_offsets_ns"].append(
                int(bag_timestamp_ns) - source_ns
            )
            entry["timestamp_sources"][source_kind] += 1
            schema = (
                len(message.name),
                len(message.position),
                len(message.velocity),
                len(message.effort),
            )
            entry["schemas"][schema] += 1
            if entry["first_sample"] is None:
                entry["first_sample"] = {
                    "frame_id": message.header.frame_id,
                    "names": list(message.name),
                    "position": [float(value) for value in message.position],
                    "velocity": [float(value) for value in message.velocity],
                    "effort_length": len(message.effort),
                    "effort_first_40": [
                        float(value) for value in message.effort[:40]
                    ],
                }
            continue

        if topic == HAND_POSE_TOPIC:
            message = deserialize_message(serialized_data, PoseArray)
            source_ns, source_kind = timestamp_ns(
                message.header.stamp, bag_timestamp_ns
            )
            entry = other_data["hand_pose_state"]
            entry["timestamps"].append(source_ns)
            entry["receive_offsets_ns"].append(
                int(bag_timestamp_ns) - source_ns
            )
            entry["timestamp_sources"][source_kind] += 1
            entry["pose_counts"][len(message.poses)] += 1
            continue

        if topic == IMU_TOPIC:
            message = deserialize_message(serialized_data, Imu)
            source_ns, source_kind = timestamp_ns(
                message.header.stamp, bag_timestamp_ns
            )
            entry = other_data["imu"]
            entry["timestamps"].append(source_ns)
            entry["receive_offsets_ns"].append(
                int(bag_timestamp_ns) - source_ns
            )
            entry["timestamp_sources"][source_kind] += 1

    report: dict[str, Any] = {
        "bag_dir": str(bag_dir),
        "recorded_types": recorded_types,
        "cameras": {},
        "joint_streams": {},
        "other_streams": {},
    }

    decodable_starts = []
    camera_ends = []
    for name, entry in camera_data.items():
        stats = timing_stats(
            entry["timestamps"], entry["receive_offsets_ns"], 30.0
        )
        stats.update(
            {
                "topic": entry["topic"],
                "byte_count": entry["byte_count"],
                "format_values": dict(entry["format_values"]),
                "timestamp_sources": dict(entry["timestamp_sources"]),
                "first_decodable_index": entry["first_decodable_index"],
                "first_decodable_timestamp_ns": entry[
                    "first_decodable_timestamp_ns"
                ],
                "nal_type_counts": {
                    str(key): value
                    for key, value in sorted(entry["nal_type_counts"].items())
                },
            }
        )
        interval_max = stats["interval_ms"]["max"]
        stats["quality_pass"] = bool(
            stats["count"] > 0
            and stats["measured_hz"] >= 27.0
            and stats["duplicate_or_reverse_count"] == 0
            and interval_max is not None
            and interval_max <= 200.0
            and stats["first_decodable_timestamp_ns"] is not None
        )
        report["cameras"][name] = stats
        if stats["first_decodable_timestamp_ns"] is not None:
            decodable_starts.append(stats["first_decodable_timestamp_ns"])
        if stats["last_timestamp_ns"] is not None:
            camera_ends.append(stats["last_timestamp_ns"])

    if len(decodable_starts) == len(CAMERA_TOPICS) and len(camera_ends) == len(
        CAMERA_TOPICS
    ):
        common_start = max(decodable_starts)
        common_end = min(camera_ends)
        report["common_decodable_window"] = {
            "start_timestamp_ns": common_start,
            "end_timestamp_ns": common_end,
            "duration_s": max(0.0, (common_end - common_start) / 1e9),
        }
    else:
        report["common_decodable_window"] = None

    for name, entry in joint_data.items():
        expected_hz = 500.0 if name.endswith("_command") else 100.0
        stats = timing_stats(
            entry["timestamps"], entry["receive_offsets_ns"], expected_hz
        )
        stats.update(
            {
                "topic": entry["topic"],
                "timestamp_sources": dict(entry["timestamp_sources"]),
                "schemas": [
                    {
                        "name_length": schema[0],
                        "position_length": schema[1],
                        "velocity_length": schema[2],
                        "effort_length": schema[3],
                        "count": count,
                    }
                    for schema, count in entry["schemas"].most_common()
                ],
                "first_sample": entry["first_sample"],
            }
        )
        report["joint_streams"][name] = stats

    for name, entry in other_data.items():
        expected_hz = 30.0 if name == "hand_pose_state" else 1000.0
        stats = timing_stats(
            entry["timestamps"], entry["receive_offsets_ns"], expected_hz
        )
        stats.update(
            {
                "topic": entry["topic"],
                "timestamp_sources": dict(entry["timestamp_sources"]),
            }
        )
        if "pose_counts" in entry:
            stats["pose_counts"] = {
                str(key): value
                for key, value in entry["pose_counts"].items()
            }
        report["other_streams"][name] = stats

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    print("===== camera quality =====")
    for name, stats in report["cameras"].items():
        print(
            f"{name}: count={stats['count']}, "
            f"hz={stats['measured_hz']:.3f}, "
            f"p99_ms={stats['interval_ms']['p99']:.3f}, "
            f"max_ms={stats['interval_ms']['max']:.3f}, "
            f"first_idr_index={stats['first_decodable_index']}, "
            f"pass={stats['quality_pass']}"
        )

    print("===== joint/action schemas =====")
    for name, stats in report["joint_streams"].items():
        print(
            f"{name}: count={stats['count']}, "
            f"hz={stats['measured_hz']:.3f}, "
            f"schemas={stats['schemas']}"
        )

    print("===== common decodable window =====")
    print(report["common_decodable_window"])
    print(f"report={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
