#!/usr/bin/env python3
"""Convert raw ROS 2 Image messages in one or more A2 MCAP files to MP4.

Input MCAP files are opened read-only.  The CSV retains each source MCAP log
timestamp; the MP4 uses the requested constant playback rate for easy review.
"""

from __future__ import annotations

import argparse
import csv
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory


DEFAULT_TOPIC = "/aima/hal/rgbd_camera/head_front/color"


def image_to_frame(message: object) -> av.VideoFrame:
    """Convert the common ROS sensor_msgs/Image encodings to a PyAV frame."""
    encoding = str(message.encoding).lower()
    height, width, step = int(message.height), int(message.width), int(message.step)
    data = np.frombuffer(message.data, dtype=np.uint8)
    if height <= 0 or width <= 0 or data.size < height * step:
        raise ValueError(f"invalid Image dimensions: {width}x{height}, step={step}")

    rows = data[: height * step].reshape(height, step)
    if encoding in {"rgb8", "bgr8"}:
        pixels = rows[:, : width * 3].reshape(height, width, 3)
        # ROS calls these rgb8/bgr8; FFmpeg/PyAV calls the same layouts
        # rgb24/bgr24.
        return av.VideoFrame.from_ndarray(
            pixels, format={"rgb8": "rgb24", "bgr8": "bgr24"}[encoding]
        )
    if encoding in {"rgba8", "bgra8"}:
        pixels = rows[:, : width * 4].reshape(height, width, 4)
        return av.VideoFrame.from_ndarray(
            pixels, format={"rgba8": "rgba", "bgra8": "bgra"}[encoding]
        )
    if encoding in {"mono8", "8uc1"}:
        pixels = rows[:, :width]
        return av.VideoFrame.from_ndarray(pixels, format="gray")
    raise ValueError(f"unsupported Image encoding: {message.encoding!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True, help="MCAP files in time order")
    parser.add_argument("--out", type=Path, required=True, help="Output MP4 (must not exist)")
    parser.add_argument("--topic", default=DEFAULT_TOPIC)
    parser.add_argument("--fps", type=float, required=True, help="Constant MP4 playback rate")
    args = parser.parse_args()

    if args.out.exists():
        raise SystemExit(f"Refusing to overwrite existing output: {args.out}")
    if args.fps <= 0:
        raise SystemExit("--fps must be positive")
    missing = [str(path) for path in args.inputs if not path.is_file()]
    if missing:
        raise SystemExit("Missing input MCAP: " + ", ".join(missing))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    timestamps: list[int] = []
    container: av.container.OutputContainer | None = None
    stream: av.video.stream.VideoStream | None = None
    frame_index = 0

    try:
        for path in args.inputs:
            with path.open("rb") as source:
                reader = make_reader(source, decoder_factories=[DecoderFactory()])
                for _, _, message, decoded in reader.iter_decoded_messages(topics=[args.topic]):
                    frame = image_to_frame(decoded).reformat(format="yuv420p")
                    if container is None:
                        container = av.open(str(args.out), "w")
                        stream = container.add_stream("libx264", rate=Fraction(str(args.fps)))
                        stream.width, stream.height = frame.width, frame.height
                        stream.pix_fmt = "yuv420p"
                    if frame.width != stream.width or frame.height != stream.height:
                        raise ValueError("image size changed within the input MCAP files")
                    frame.pts = frame_index
                    frame.time_base = Fraction(1, 1) / Fraction(str(args.fps))
                    for packet in stream.encode(frame):
                        container.mux(packet)
                    timestamps.append(message.log_time)
                    frame_index += 1
        if container is None or stream is None:
            raise RuntimeError(f"no messages found for topic {args.topic}")
        for packet in stream.encode():
            container.mux(packet)
    finally:
        if container is not None:
            container.close()

    timestamps_csv = args.out.with_suffix(".timestamps.csv")
    with timestamps_csv.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(["frame_index", "mcap_log_time_ns", "mcap_log_time_s"])
        for index, timestamp_ns in enumerate(timestamps):
            writer.writerow([index, timestamp_ns, timestamp_ns / 1_000_000_000])

    duration_s = (timestamps[-1] - timestamps[0]) / 1e9 if len(timestamps) > 1 else 0.0
    measured_fps = (len(timestamps) - 1) / duration_s if duration_s else 0.0
    print(f"frames: {len(timestamps)}")
    print(f"source_duration_s: {duration_s:.6f}")
    print(f"measured_fps: {measured_fps:.6f}")
    print(f"mp4: {args.out}")
    print(f"timestamps: {timestamps_csv}")


if __name__ == "__main__":
    main()
