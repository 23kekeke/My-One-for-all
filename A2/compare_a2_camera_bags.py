#!/usr/bin/env python3
"""Compare H.265 camera packets in an Orin rosbag and a Spark rosbag."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import rosbag2_py
from foxglove_msgs.msg import CompressedVideo
from rclpy.serialization import deserialize_message


CAMERA_TOPICS = (
    "/aima/hal/rgbd_camera/head_front/color/h265",
    "/aima/hal/rgbd_camera/hand_left/color/h265",
    "/aima/hal/rgbd_camera/hand_right/color/h265",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("orin_bag", type=Path)
    parser.add_argument("spark_bag", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def message_timestamp_ns(message: CompressedVideo) -> int:
    return (
        int(message.timestamp.sec) * 1_000_000_000
        + int(message.timestamp.nanosec)
    )


def load_camera_packets(bag_dir: Path) -> dict[str, dict[str, Any]]:
    if not (bag_dir / "metadata.yaml").is_file():
        raise RuntimeError(f"metadata.yaml not found: {bag_dir}")

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag_dir), storage_id="sqlite3"),
        rosbag2_py.ConverterOptions("cdr", "cdr"),
    )
    recorded_types = {
        item.name: item.type for item in reader.get_all_topics_and_types()
    }
    for topic in CAMERA_TOPICS:
        actual = recorded_types.get(topic)
        if actual != "foxglove_msgs/msg/CompressedVideo":
            raise RuntimeError(
                f"{bag_dir}: {topic} has type {actual!r}, "
                "expected foxglove_msgs/msg/CompressedVideo"
            )

    result: dict[str, dict[str, Any]] = {
        topic: {
            "packets": {},
            "duplicate_timestamp_count": 0,
            "zero_timestamp_count": 0,
        }
        for topic in CAMERA_TOPICS
    }
    while reader.has_next():
        topic, serialized_data, bag_timestamp_ns = reader.read_next()
        if topic not in result:
            continue
        message = deserialize_message(serialized_data, CompressedVideo)
        timestamp_ns = message_timestamp_ns(message)
        if timestamp_ns <= 0:
            timestamp_ns = int(bag_timestamp_ns)
            result[topic]["zero_timestamp_count"] += 1
        packets: dict[int, str] = result[topic]["packets"]
        if timestamp_ns in packets:
            result[topic]["duplicate_timestamp_count"] += 1
        packets[timestamp_ns] = hashlib.sha256(bytes(message.data)).hexdigest()
    return result


def compare_topic(
    orin: dict[str, Any],
    spark: dict[str, Any],
) -> dict[str, Any]:
    orin_packets: dict[int, str] = orin["packets"]
    spark_packets: dict[int, str] = spark["packets"]
    orin_all = set(orin_packets)
    spark_all = set(spark_packets)

    if not orin_all or not spark_all:
        return {
            "orin_count": len(orin_all),
            "spark_count": len(spark_all),
            "overlap_duration_s": 0.0,
            "orin_overlap_count": 0,
            "spark_overlap_count": 0,
            "common_timestamp_count": 0,
            "missing_in_spark_count": 0,
            "extra_in_spark_count": 0,
            "payload_mismatch_count": 0,
            "missing_in_spark_ratio": 1.0,
            "transport_pass": False,
        }

    overlap_start = max(min(orin_all), min(spark_all))
    overlap_end = min(max(orin_all), max(spark_all))
    if overlap_end < overlap_start:
        orin_overlap: set[int] = set()
        spark_overlap: set[int] = set()
    else:
        orin_overlap = {
            value for value in orin_all
            if overlap_start <= value <= overlap_end
        }
        spark_overlap = {
            value for value in spark_all
            if overlap_start <= value <= overlap_end
        }

    common = orin_overlap & spark_overlap
    missing = orin_overlap - spark_overlap
    extra = spark_overlap - orin_overlap
    mismatched = {
        value for value in common
        if orin_packets[value] != spark_packets[value]
    }
    missing_ratio = (
        len(missing) / len(orin_overlap) if orin_overlap else 1.0
    )

    return {
        "orin_count": len(orin_all),
        "spark_count": len(spark_all),
        "overlap_start_timestamp_ns": overlap_start,
        "overlap_end_timestamp_ns": overlap_end,
        "overlap_duration_s": max(0, overlap_end - overlap_start) / 1e9,
        "orin_overlap_count": len(orin_overlap),
        "spark_overlap_count": len(spark_overlap),
        "common_timestamp_count": len(common),
        "missing_in_spark_count": len(missing),
        "extra_in_spark_count": len(extra),
        "payload_mismatch_count": len(mismatched),
        "missing_in_spark_ratio": missing_ratio,
        "orin_duplicate_timestamp_count": orin[
            "duplicate_timestamp_count"
        ],
        "spark_duplicate_timestamp_count": spark[
            "duplicate_timestamp_count"
        ],
        "transport_pass": bool(
            orin_overlap
            and not mismatched
            and missing_ratio <= 0.01
        ),
    }


def main() -> int:
    args = parse_args()
    orin_bag = args.orin_bag.resolve()
    spark_bag = args.spark_bag.resolve()
    output = (
        args.output.resolve()
        if args.output
        else spark_bag / "camera_transport_comparison.json"
    )

    print(f"Loading Orin bag:  {orin_bag}")
    orin_data = load_camera_packets(orin_bag)
    print(f"Loading Spark bag: {spark_bag}")
    spark_data = load_camera_packets(spark_bag)

    comparisons = {
        topic: compare_topic(orin_data[topic], spark_data[topic])
        for topic in CAMERA_TOPICS
    }
    overall_pass = all(
        item["transport_pass"] for item in comparisons.values()
    )
    report = {
        "format_version": 1,
        "orin_bag": str(orin_bag),
        "spark_bag": str(spark_bag),
        "topics": comparisons,
        "overall_transport_pass": overall_pass,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    for topic, item in comparisons.items():
        print(f"===== {topic} =====")
        print(
            f"overlap={item['overlap_duration_s']:.3f}s "
            f"orin={item['orin_overlap_count']} "
            f"spark={item['spark_overlap_count']} "
            f"missing_in_spark={item['missing_in_spark_count']} "
            f"payload_mismatch={item['payload_mismatch_count']} "
            f"pass={item['transport_pass']}"
        )
    print(f"overall_transport_pass={overall_pass}")
    print(f"report={output}")
    return 0 if overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
