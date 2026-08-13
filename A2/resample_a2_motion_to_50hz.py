#!/usr/bin/env python3
"""Create a 50 Hz JSON motion representation from one raw A2 rosbag.

The raw rosbag remains the source of truth. Joint states are linearly
interpolated, commands use zero-order hold, /tf is rate-limited without
splitting TFMessage bundles, and /tf_static is copied once.
"""

from __future__ import annotations

import argparse
import bisect
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import rosbag2_py
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import JointState
from tf2_msgs.msg import TFMessage


JOINT_TOPICS = {
    "/motion/control/arm_joint_state": ("arm_joint_state", "linear"),
    "/motion/control/hand_joint_state": ("hand_joint_state", "linear"),
    "/motion/control/neck_joint_state": ("neck_joint_state", "linear"),
    "/motion/control/arm_joint_command": ("arm_joint_command", "zoh"),
    "/motion/control/hand_joint_command": ("hand_joint_command", "zoh"),
    "/motion/control/neck_joint_command": ("neck_joint_command", "zoh"),
}
TF_TOPIC = "/tf"
TF_STATIC_TOPIC = "/tf_static"
NANOSECONDS = 1_000_000_000


@dataclass
class JointSample:
    timestamp_ns: int
    names: list[str]
    position: list[float]
    velocity: list[float]
    effort: list[float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bag_dir", type=Path)
    parser.add_argument("output_json", type=Path)
    parser.add_argument("--hz", type=float, default=50.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def message_stamp_ns(message: JointState, bag_timestamp_ns: int) -> int:
    stamp = message.header.stamp
    value = int(stamp.sec) * NANOSECONDS + int(stamp.nanosec)
    return value if value > 0 else int(bag_timestamp_ns)


def float_list(values: Any) -> list[float]:
    return [float(value) for value in values]


def joint_sample(message: JointState, bag_timestamp_ns: int) -> JointSample:
    return JointSample(
        timestamp_ns=message_stamp_ns(message, bag_timestamp_ns),
        names=[str(value) for value in message.name],
        position=float_list(message.position),
        velocity=float_list(message.velocity),
        effort=float_list(message.effort),
    )


def vector_at(
    left: list[float],
    right: list[float],
    ratio: float,
) -> list[float]:
    if len(left) != len(right):
        return list(left)
    return [
        first + (second - first) * ratio
        for first, second in zip(left, right)
    ]


def sample_joint(
    samples: list[JointSample],
    timestamps: list[int],
    target_ns: int,
    method: str,
) -> dict[str, Any]:
    right_index = bisect.bisect_right(timestamps, target_ns)
    left_index = max(0, right_index - 1)
    left = samples[left_index]

    if method == "zoh" or right_index >= len(samples):
        selected = left
        return {
            "source_timestamp_ns": selected.timestamp_ns,
            "name": selected.names,
            "position": selected.position,
            "velocity": selected.velocity,
            "effort": selected.effort,
        }

    right = samples[right_index]
    span = right.timestamp_ns - left.timestamp_ns
    ratio = 0.0 if span <= 0 else (target_ns - left.timestamp_ns) / span
    return {
        "source_timestamp_ns": [left.timestamp_ns, right.timestamp_ns],
        "interpolation_ratio": ratio,
        "name": left.names if left.names == right.names else [],
        "position": vector_at(left.position, right.position, ratio),
        "velocity": vector_at(left.velocity, right.velocity, ratio),
        "effort": vector_at(left.effort, right.effort, ratio),
    }


def transform_to_dict(transform: Any) -> dict[str, Any]:
    stamp = transform.header.stamp
    return {
        "timestamp_ns": int(stamp.sec) * NANOSECONDS + int(stamp.nanosec),
        "frame_id": str(transform.header.frame_id),
        "child_frame_id": str(transform.child_frame_id),
        "translation": [
            float(transform.transform.translation.x),
            float(transform.transform.translation.y),
            float(transform.transform.translation.z),
        ],
        "rotation_xyzw": [
            float(transform.transform.rotation.x),
            float(transform.transform.rotation.y),
            float(transform.transform.rotation.z),
            float(transform.transform.rotation.w),
        ],
    }


def tf_message_to_dict(
    message: TFMessage,
    bag_timestamp_ns: int,
) -> dict[str, Any]:
    return {
        "receive_timestamp_ns": int(bag_timestamp_ns),
        "transforms": [
            transform_to_dict(transform) for transform in message.transforms
        ],
    }


def main() -> int:
    args = parse_args()
    if args.hz <= 0:
        raise SystemExit("--hz must be positive")

    bag_dir = args.bag_dir.resolve()
    output_json = args.output_json.resolve()
    if not (bag_dir / "metadata.yaml").is_file():
        raise SystemExit(f"metadata.yaml not found: {bag_dir}")
    if output_json.exists() and not args.overwrite:
        raise SystemExit(f"output exists (use --overwrite): {output_json}")

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag_dir), storage_id="sqlite3"),
        rosbag2_py.ConverterOptions("cdr", "cdr"),
    )
    recorded_types = {
        item.name: item.type for item in reader.get_all_topics_and_types()
    }
    required_types = {
        **{topic: "sensor_msgs/msg/JointState" for topic in JOINT_TOPICS},
        TF_TOPIC: "tf2_msgs/msg/TFMessage",
        TF_STATIC_TOPIC: "tf2_msgs/msg/TFMessage",
    }
    errors = [
        f"{topic}: expected {expected}, found {recorded_types.get(topic)!r}"
        for topic, expected in required_types.items()
        if recorded_types.get(topic) != expected
    ]
    if errors:
        raise SystemExit("Required topic/type check failed:\n" + "\n".join(errors))

    joints: dict[str, list[JointSample]] = {
        topic: [] for topic in JOINT_TOPICS
    }
    tf_messages: list[dict[str, Any]] = []
    tf_static_messages: list[dict[str, Any]] = []

    while reader.has_next():
        topic, serialized, bag_timestamp_ns = reader.read_next()
        if topic in JOINT_TOPICS:
            message = deserialize_message(serialized, JointState)
            joints[topic].append(joint_sample(message, bag_timestamp_ns))
        elif topic == TF_TOPIC:
            message = deserialize_message(serialized, TFMessage)
            tf_messages.append(tf_message_to_dict(message, bag_timestamp_ns))
        elif topic == TF_STATIC_TOPIC:
            message = deserialize_message(serialized, TFMessage)
            tf_static_messages.append(
                tf_message_to_dict(message, bag_timestamp_ns)
            )

    missing = [topic for topic, samples in joints.items() if not samples]
    if missing:
        raise SystemExit("Required motion streams contain no data: " + ", ".join(missing))
    if not tf_messages:
        raise SystemExit(f"{TF_TOPIC} contains no data")
    if not tf_static_messages:
        raise SystemExit(f"{TF_STATIC_TOPIC} contains no data")

    for samples in joints.values():
        samples.sort(key=lambda item: item.timestamp_ns)

    start_ns = max(samples[0].timestamp_ns for samples in joints.values())
    end_ns = min(samples[-1].timestamp_ns for samples in joints.values())
    if end_ns <= start_ns:
        raise SystemExit("Required motion streams have no common time window")

    step_ns = round(NANOSECONDS / args.hz)
    grid = range(start_ns, end_ns + 1, step_ns)
    joint_timestamps = {
        topic: [sample.timestamp_ns for sample in samples]
        for topic, samples in joints.items()
    }
    frames: list[dict[str, Any]] = []
    for index, target_ns in enumerate(grid):
        signals = {}
        for topic, (name, method) in JOINT_TOPICS.items():
            signals[name] = sample_joint(
                joints[topic],
                joint_timestamps[topic],
                target_ns,
                method,
            )
        frames.append(
            {
                "index": index,
                "timestamp_ns": target_ns,
                "signals": signals,
            }
        )

    tf_step_ns = step_ns
    kept_tf: list[dict[str, Any]] = []
    last_tf_ns: int | None = None
    for item in tf_messages:
        timestamp_ns = int(item["receive_timestamp_ns"])
        if last_tf_ns is None or timestamp_ns - last_tf_ns >= tf_step_ns:
            kept_tf.append(item)
            last_tf_ns = timestamp_ns

    document = {
        "schema_version": 1,
        "source_bag": str(bag_dir),
        "target_hz": args.hz,
        "time_window": {
            "start_timestamp_ns": start_ns,
            "end_timestamp_ns": end_ns,
            "duration_s": (end_ns - start_ns) / NANOSECONDS,
        },
        "sampling": {
            "state": "linear_interpolation",
            "command": "zero_order_hold",
            "tf": "message_rate_limited_without_splitting",
            "tf_static": "copied_once",
        },
        "source_counts": {
            **{name: len(joints[topic]) for topic, (name, _) in JOINT_TOPICS.items()},
            "tf": len(tf_messages),
            "tf_static": len(tf_static_messages),
        },
        "frames": frames,
        "tf": kept_tf,
        "tf_static": tf_static_messages[-1],
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_json.with_name(f".{output_json.name}.tmp")
    temporary.write_text(
        json.dumps(document, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    os.replace(temporary, output_json)
    print(
        f"output={output_json} frames={len(frames)} "
        f"duration_s={(end_ns - start_ns) / NANOSECONDS:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
