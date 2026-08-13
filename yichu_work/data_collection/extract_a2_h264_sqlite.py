#!/usr/bin/env python3
"""Extract A2 Annex-B H.264 camera frames from a rosbag2 SQLite dataset.

Each foxglove_msgs/msg/CompressedVideo message contains exactly one encoded
video frame. The encoded bytes are concatenated without decoding or
re-encoding, while the message and bag timestamps are written to CSV.
"""

from __future__ import annotations

import argparse
import csv
import os
from contextlib import ExitStack
from pathlib import Path

import rosbag2_py
from foxglove_msgs.msg import CompressedVideo
from rclpy.serialization import deserialize_message


CAMERA_TOPICS = {
    "/aima/hal/rgbd_camera/head_front/color/h264": "head_camera",
    "/aima/hal/fish_eye_camera/chest_left/color/h264": "left_camera",
    "/aima/hal/fish_eye_camera/chest_right/color/h264": "right_camera",
}

EXPECTED_TYPE = "foxglove_msgs/msg/CompressedVideo"


def message_timestamp_ns(
    message: CompressedVideo, bag_timestamp_ns: int
) -> tuple[int, str]:
    timestamp = message.timestamp
    value = int(timestamp.sec) * 1_000_000_000 + int(timestamp.nanosec)
    if value > 0:
        return value, "message"
    return int(bag_timestamp_ns), "bag"


def annex_b_nal_types(data: bytes) -> list[int]:
    """Return H.264 NAL unit types found in one Annex-B access unit."""
    starts: list[tuple[int, int]] = []
    index = 0
    while index + 3 <= len(data):
        if data[index : index + 4] == b"\x00\x00\x00\x01":
            starts.append((index, 4))
            index += 4
        elif data[index : index + 3] == b"\x00\x00\x01":
            starts.append((index, 3))
            index += 3
        else:
            index += 1

    result: list[int] = []
    for start, prefix_length in starts:
        header_index = start + prefix_length
        if header_index < len(data):
            result.append(data[header_index] & 0x1F)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "bag_dir",
        type=Path,
        help="Input rosbag2 directory containing metadata.yaml and a .db3 file",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="New or empty directory for .h264 files and timestamp CSV files",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    bag_dir = args.bag_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not (bag_dir / "metadata.yaml").is_file():
        raise SystemExit(f"metadata.yaml not found in bag directory: {bag_dir}")
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
    for topic in CAMERA_TOPICS:
        recorded_type = recorded_types.get(topic)
        if recorded_type != EXPECTED_TYPE:
            raise SystemExit(
                f"Expected {topic} to have type {EXPECTED_TYPE}, "
                f"found {recorded_type!r}"
            )

    packet_counts = {camera: 0 for camera in CAMERA_TOPICS.values()}
    source_packet_counts = {camera: 0 for camera in CAMERA_TOPICS.values()}
    dropped_before_keyframe = {camera: 0 for camera in CAMERA_TOPICS.values()}
    byte_counts = {camera: 0 for camera in CAMERA_TOPICS.values()}
    first_timestamps = {camera: None for camera in CAMERA_TOPICS.values()}
    last_timestamps = {camera: None for camera in CAMERA_TOPICS.values()}
    previous_timestamps = {camera: None for camera in CAMERA_TOPICS.values()}
    stream_started = {camera: False for camera in CAMERA_TOPICS.values()}

    temporary_paths: list[tuple[Path, Path]] = []

    try:
        with ExitStack() as stack:
            video_outputs = {}
            csv_writers = {}

            for camera in CAMERA_TOPICS.values():
                video_path = output_dir / f"{camera}.h264"
                video_tmp = video_path.with_suffix(".h264.tmp")
                csv_path = output_dir / f"{camera}_timestamps.csv"
                csv_tmp = csv_path.with_suffix(".csv.tmp")

                temporary_paths.extend(
                    [(video_tmp, video_path), (csv_tmp, csv_path)]
                )

                video_outputs[camera] = stack.enter_context(video_tmp.open("wb"))
                csv_file = stack.enter_context(
                    csv_tmp.open("w", newline="", encoding="utf-8")
                )
                writer = csv.writer(csv_file)
                writer.writerow(
                    [
                        "frame_index",
                        "source_packet_index",
                        "timestamp_ns",
                        "timestamp_s",
                        "timestamp_source",
                        "bag_timestamp_ns",
                        "data_size",
                        "format",
                        "frame_id",
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
                if message.format.lower() != "h264":
                    raise RuntimeError(
                        f"Unexpected format on {topic}: {message.format!r}"
                    )

                encoded_data = bytes(message.data)
                if not encoded_data:
                    raise RuntimeError(
                        f"Empty H.264 frame on {topic}, "
                        f"packet {packet_counts[camera]}"
                    )

                nal_types = annex_b_nal_types(encoded_data)
                source_packet_index = source_packet_counts[camera]
                source_packet_counts[camera] += 1

                if not stream_started[camera]:
                    # A rosbag can begin in the middle of a GOP. Starting from
                    # a delta frame produces "non-existing PPS" errors. A2's
                    # encoder repeats SPS/PPS with an IDR frame, so wait for
                    # an access unit containing all three NAL types.
                    decodable_start = {5, 7, 8}.issubset(nal_types)
                    if not decodable_start:
                        dropped_before_keyframe[camera] += 1
                        continue
                    stream_started[camera] = True

                timestamp_ns, timestamp_source = message_timestamp_ns(
                    message, bag_timestamp_ns
                )
                previous = previous_timestamps[camera]
                if previous is not None and timestamp_ns <= previous:
                    raise RuntimeError(
                        f"Non-monotonic timestamp on {topic}: "
                        f"{timestamp_ns} after {previous}"
                    )

                video_outputs[camera].write(encoded_data)
                csv_writers[camera].writerow(
                    [
                        packet_counts[camera],
                        source_packet_index,
                        timestamp_ns,
                        f"{timestamp_ns / 1_000_000_000:.9f}",
                        timestamp_source,
                        int(bag_timestamp_ns),
                        len(encoded_data),
                        message.format,
                        message.frame_id,
                        " ".join(str(value) for value in nal_types),
                    ]
                )

                if first_timestamps[camera] is None:
                    first_timestamps[camera] = timestamp_ns
                last_timestamps[camera] = timestamp_ns
                previous_timestamps[camera] = timestamp_ns
                packet_counts[camera] += 1
                byte_counts[camera] += len(encoded_data)

        for temporary_path, final_path in temporary_paths:
            os.replace(temporary_path, final_path)
    except Exception:
        for temporary_path, _ in temporary_paths:
            temporary_path.unlink(missing_ok=True)
        raise

    for camera in CAMERA_TOPICS.values():
        count = packet_counts[camera]
        first = first_timestamps[camera]
        last = last_timestamps[camera]
        duration = (
            (last - first) / 1_000_000_000
            if first is not None and last is not None and last > first
            else 0.0
        )
        fps = (count - 1) / duration if count > 1 and duration > 0 else 0.0
        print(
            f"{camera}: source_frames={source_packet_counts[camera]}, "
            f"dropped_before_keyframe={dropped_before_keyframe[camera]}, "
            f"kept_frames={count}, bytes={byte_counts[camera]}, "
            f"duration={duration:.3f}s, measured_fps={fps:.3f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
