#!/usr/bin/env python3
"""Convert paired A2 AimRT MCAP files into a reviewable episode.

The script keeps raw source timestamps. It decodes the head H.264 stream to
an MP4 and samples high-frequency x86 joint/IMU state at each decoded video
frame timestamp. Source MCAP files are opened read-only.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import OrderedDict
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable

import av
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory


HEAD_VIDEO_TOPIC = "/aima/hal/rgbd_camera/head_front/color/h264"
JOINT_TOPICS = (
    "/body_drive/leg_joint_state",
    "/body_drive/arm_joint_state",
    "/body_drive/neck_joint_state",
)
IMU_TOPIC = "/body_drive/imu/data"


@dataclass
class Sample:
    timestamp_ns: int
    value: Any


def decoded_messages(path: Path, topic: str) -> Iterable[tuple[int, Any]]:
    """Yield `(mcap_log_time_ns, decoded_message)` for one ROS 2 topic."""
    with path.open("rb") as source:
        reader = make_reader(source, decoder_factories=[DecoderFactory()])
        for _, _, message, decoded in reader.iter_decoded_messages(topics=[topic]):
            yield message.log_time, decoded


def mcap_time_range(path: Path) -> tuple[int, int]:
    with path.open("rb") as source:
        stats = make_reader(source).get_summary().statistics
    return stats.message_start_time, stats.message_end_time


def decode_video(
    video_mcap: Path,
    output_mp4: Path,
    requested_fps: float | None,
    start_ns: int,
    end_ns: int,
) -> list[int]:
    """Decode H.264 packets, then encode a portable H.264 MP4.

    MCAP packet timestamps are written separately because the source stream is
    sparse/variable-rate and an MP4 player normally uses a constant frame rate.
    """
    decoder = av.CodecContext.create("h264", "r")
    decoded: list[tuple[int, Any]] = []

    for timestamp_ns, message in decoded_messages(video_mcap, HEAD_VIDEO_TOPIC):
        packet = av.Packet(message.data)
        try:
            frames = decoder.decode(packet)
        except av.error.InvalidDataError:
            # A bag chunk can begin with a P-frame that refers to an earlier
            # keyframe. Skip it and wait for the next self-contained keyframe.
            continue
        for frame in frames:
            decoded.append((timestamp_ns, frame))

    # Flush delayed codec frames, retaining the last packet timestamp.
    if decoded:
        last_timestamp_ns = decoded[-1][0]
        try:
            delayed_frames = decoder.decode(None)
        except av.error.InvalidDataError:
            delayed_frames = []
        for frame in delayed_frames:
            decoded.append((last_timestamp_ns, frame))

    # Continue decoding from the beginning to obtain H.264 keyframe context,
    # then retain only frames that also have x86 state coverage.
    decoded = [(timestamp, frame) for timestamp, frame in decoded if start_ns <= timestamp <= end_ns]

    if len(decoded) < 2:
        raise RuntimeError(
            "No decodable head video frames were found. The selected MCAP may start "
            "after an H.264 keyframe or contain an unsupported stream."
        )

    timestamps_ns = [item[0] for item in decoded]
    duration_s = (timestamps_ns[-1] - timestamps_ns[0]) / 1_000_000_000
    inferred_fps = max(1.0, (len(decoded) - 1) / max(duration_s, 1e-9))
    # H.264 packets can emit several delayed frames with the same bag time.
    # Use the requested rate for a portable MP4; exact times remain in CSV.
    fps = requested_fps or 2.0

    output_mp4.parent.mkdir(parents=True, exist_ok=True)
    container = av.open(str(output_mp4), "w")
    stream = container.add_stream("libx264", rate=Fraction(int(round(fps)), 1))
    stream.width = decoded[0][1].width
    stream.height = decoded[0][1].height
    stream.pix_fmt = "yuv420p"

    for index, (_, frame) in enumerate(decoded):
        frame = frame.reformat(format="yuv420p")
        frame.pts = index
        frame.time_base = Fraction(1, 1) / Fraction(str(fps))
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()

    return timestamps_ns


def nearest_samples(state_mcap: Path, topic: str, targets_ns: list[int]) -> list[Sample | None]:
    """Find the nearest message from one high-rate topic for every target time."""
    result: list[Sample | None] = [None] * len(targets_ns)
    target_index = 0
    previous: Sample | None = None

    for timestamp_ns, message in decoded_messages(state_mcap, topic):
        current = Sample(timestamp_ns, message)
        while target_index < len(targets_ns) and targets_ns[target_index] <= timestamp_ns:
            if previous is None or timestamp_ns - targets_ns[target_index] < targets_ns[target_index] - previous.timestamp_ns:
                result[target_index] = current
            else:
                result[target_index] = previous
            target_index += 1
        previous = current

    while target_index < len(targets_ns):
        result[target_index] = previous
        target_index += 1
    return result


def add_joint_values(row: dict[str, Any], prefix: str, message: Any) -> None:
    for joint in message.joints:
        base = f"{prefix}.{joint.name}"
        row[f"{base}.position"] = joint.position
        row[f"{base}.velocity"] = joint.velocity
        row[f"{base}.effort"] = joint.effort


def add_imu_values(row: dict[str, Any], message: Any) -> None:
    row.update(
        {
            "imu.orientation.x": message.orientation.x,
            "imu.orientation.y": message.orientation.y,
            "imu.orientation.z": message.orientation.z,
            "imu.orientation.w": message.orientation.w,
            "imu.angular_velocity.x": message.angular_velocity.x,
            "imu.angular_velocity.y": message.angular_velocity.y,
            "imu.angular_velocity.z": message.angular_velocity.z,
            "imu.linear_acceleration.x": message.linear_acceleration.x,
            "imu.linear_acceleration.y": message.linear_acceleration.y,
            "imu.linear_acceleration.z": message.linear_acceleration.z,
        }
    )


def write_aligned_states(state_mcap: Path, timestamps_ns: list[int], output_csv: Path) -> dict[str, int]:
    samples = {topic: nearest_samples(state_mcap, topic, timestamps_ns) for topic in (*JOINT_TOPICS, IMU_TOPIC)}
    rows: list[dict[str, Any]] = []

    for index, timestamp_ns in enumerate(timestamps_ns):
        row: dict[str, Any] = OrderedDict(
            frame_index=index,
            video_timestamp_ns=timestamp_ns,
            video_timestamp_s=timestamp_ns / 1_000_000_000,
        )
        for topic in JOINT_TOPICS:
            sample = samples[topic][index]
            if sample is not None:
                row[f"{topic}.source_timestamp_ns"] = sample.timestamp_ns
                add_joint_values(row, topic.rsplit("/", 1)[-1], sample.value)
        imu = samples[IMU_TOPIC][index]
        if imu is not None:
            row["imu.source_timestamp_ns"] = imu.timestamp_ns
            add_imu_values(row, imu.value)
        rows.append(row)

    fieldnames = list(OrderedDict.fromkeys(key for row in rows for key in row))
    with output_csv.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return {topic: sum(item is not None for item in items) for topic, items in samples.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-mcap", type=Path, required=True)
    parser.add_argument("--state-mcap", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="Must not already exist")
    parser.add_argument("--fps", type=float, default=2.0, help="MP4 FPS; source timestamps remain exact in CSV")
    args = parser.parse_args()

    if args.out.exists():
        raise SystemExit(f"Refusing to overwrite existing output: {args.out}")
    if not args.video_mcap.is_file() or not args.state_mcap.is_file():
        raise SystemExit("Both --video-mcap and --state-mcap must be existing files")

    video_start, video_end = mcap_time_range(args.video_mcap)
    state_start, state_end = mcap_time_range(args.state_mcap)
    overlap_start, overlap_end = max(video_start, state_start), min(video_end, state_end)
    if overlap_start >= overlap_end:
        raise SystemExit("The two MCAP files have no overlapping log-time range")

    args.out.mkdir(parents=True)
    frame_timestamps = decode_video(
        args.video_mcap,
        args.out / "head.mp4",
        args.fps,
        overlap_start,
        overlap_end,
    )

    sample_counts = write_aligned_states(args.state_mcap, frame_timestamps, args.out / "states.csv")
    with (args.out / "frame_timestamps.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(["frame_index", "video_timestamp_ns", "video_timestamp_s"])
        for index, timestamp_ns in enumerate(frame_timestamps):
            writer.writerow([index, timestamp_ns, timestamp_ns / 1_000_000_000])

    metadata = {
        "format": "a2_episode_v1",
        "video_topic": HEAD_VIDEO_TOPIC,
        "joint_topics": list(JOINT_TOPICS),
        "imu_topic": IMU_TOPIC,
        "source": {"orin_mcap": str(args.video_mcap), "x86_mcap": str(args.state_mcap)},
        "mcap_log_time_overlap_ns": [overlap_start, overlap_end],
        "aligned_frame_count": len(frame_timestamps),
        "state_samples_available": sample_counts,
        "note": "states.csv uses nearest x86 state sample for each source video timestamp.",
    }
    (args.out / "episode.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
