#!/usr/bin/env python3
"""Create validated 15 Hz H.265 MP4 files from an extracted A2 episode.

The input directory must contain episode.json plus the elementary H.265 files
written by extract_a2_head_hands_h265_episode.py. A source already near 15 Hz
is remuxed without re-encoding. A faster source is fully decoded and encoded
again at 15 Hz; compressed H.265 packets are never dropped directly.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any


CAMERAS = ("head_camera", "left_hand_camera", "right_hand_camera")


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode_dir", type=Path)
    parser.add_argument("--output-fps", type=float, default=15.0)
    parser.add_argument(
        "--copy-tolerance-hz",
        type=float,
        default=0.75,
        help="Remux without re-encoding when source FPS is this close to output FPS",
    )
    parser.add_argument("--crf", type=int, default=23)
    parser.add_argument("--preset", default="medium")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def ffprobe(path: Path) -> dict[str, Any]:
    result = run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,width,height,avg_frame_rate,nb_read_frames",
            "-show_entries",
            "format=duration,size",
            "-of",
            "json",
            str(path),
        ]
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {path}: {result.stderr}")
    return json.loads(result.stdout)


def fraction_to_float(value: str) -> float:
    numerator, denominator = value.split("/", 1)
    denominator_value = float(denominator)
    if denominator_value == 0:
        return 0.0
    return float(numerator) / denominator_value


def validate_probe(
    path: Path,
    probe: dict[str, Any],
    expected_fps: float,
    fps_tolerance_hz: float,
) -> None:
    streams = probe.get("streams", [])
    if len(streams) != 1:
        raise RuntimeError(f"expected one video stream in {path}, found {len(streams)}")
    stream = streams[0]
    if stream.get("codec_name") != "hevc":
        raise RuntimeError(
            f"expected HEVC/H.265 in {path}, found {stream.get('codec_name')!r}"
        )
    measured_fps = fraction_to_float(str(stream.get("avg_frame_rate", "0/1")))
    if abs(measured_fps - expected_fps) > fps_tolerance_hz:
        raise RuntimeError(
            f"unexpected FPS in {path}: {measured_fps:.6f}, "
            f"expected {expected_fps:.6f} "
            f"(tolerance {fps_tolerance_hz:.3f} Hz)"
        )
    frame_count = int(stream.get("nb_read_frames", 0))
    if frame_count <= 0:
        raise RuntimeError(f"no decodable frames found in {path}")


def validate_decode(path: Path) -> None:
    result = run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(path),
            "-f",
            "null",
            "-",
        ]
    )
    if result.returncode != 0:
        raise RuntimeError(f"decode validation failed for {path}: {result.stderr}")


def convert(
    raw_path: Path,
    output_path: Path,
    source_fps: float,
    output_fps: float,
    copy_tolerance_hz: float,
    crf: int,
    preset: str,
) -> tuple[str, dict[str, Any]]:
    temporary_path = output_path.with_name(f".{output_path.name}.tmp.mp4")
    temporary_path.unlink(missing_ok=True)

    common = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-fflags",
        "+genpts",
        "-r",
        f"{source_fps:.9f}",
        "-f",
        "hevc",
        "-i",
        str(raw_path),
        "-an",
    ]
    if abs(source_fps - output_fps) <= copy_tolerance_hz:
        mode = "remux_copy"
        command = common + [
            "-c:v",
            "copy",
            "-tag:v",
            "hvc1",
            "-movflags",
            "+faststart",
            str(temporary_path),
        ]
    else:
        mode = "decode_reencode"
        command = common + [
            "-vf",
            f"fps={output_fps:.9f}",
            "-c:v",
            "libx265",
            "-preset",
            preset,
            "-crf",
            str(crf),
            "-pix_fmt",
            "yuv420p",
            "-tag:v",
            "hvc1",
            "-movflags",
            "+faststart",
            str(temporary_path),
        ]

    result = run(command)
    if result.returncode != 0:
        temporary_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"ffmpeg {mode} failed for {raw_path}:\n{result.stderr}"
        )
    validate_decode(temporary_path)
    probe = ffprobe(temporary_path)
    # A stream-copy remux preserves the camera's measured native cadence; it
    # does not synthesize an exact CFR timeline. Re-encoded output, however,
    # must match the requested target cadence.
    expected_fps = source_fps if mode == "remux_copy" else output_fps
    validate_probe(
        temporary_path,
        probe,
        expected_fps=expected_fps,
        fps_tolerance_hz=0.15,
    )
    os.replace(temporary_path, output_path)
    return mode, probe


def main() -> int:
    args = parse_args()
    if args.output_fps <= 0:
        raise SystemExit("--output-fps must be positive")
    if not 0 <= args.crf <= 51:
        raise SystemExit("--crf must be in [0, 51]")
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise SystemExit("ffmpeg and ffprobe must be available in PATH")

    episode_dir = args.episode_dir.resolve()
    json_path = episode_dir / "episode.json"
    if not json_path.is_file():
        raise SystemExit(f"episode.json not found: {json_path}")

    episode = json.loads(json_path.read_text(encoding="utf-8"))
    camera_metadata = episode.get("cameras", {})
    results: dict[str, Any] = {}

    for name in CAMERAS:
        metadata = camera_metadata.get(name)
        if not isinstance(metadata, dict):
            raise RuntimeError(f"camera metadata missing: {name}")
        source_fps = float(metadata.get("measured_fps", 0.0))
        if source_fps <= 0:
            raise RuntimeError(f"invalid measured_fps for {name}: {source_fps}")
        raw_name = str(metadata["elementary_stream_file"])
        raw_path = episode_dir / raw_name
        output_path = episode_dir / f"{name}.mp4"
        if not raw_path.is_file():
            raise RuntimeError(f"H.265 file missing: {raw_path}")
        if output_path.exists() and not args.overwrite:
            raise RuntimeError(
                f"output already exists (use --overwrite): {output_path}"
            )

        mode, probe = convert(
            raw_path=raw_path,
            output_path=output_path,
            source_fps=source_fps,
            output_fps=args.output_fps,
            copy_tolerance_hz=args.copy_tolerance_hz,
            crf=args.crf,
            preset=args.preset,
        )
        results[name] = {
            "source_fps": source_fps,
            "requested_output_fps": args.output_fps,
            "measured_output_fps": fraction_to_float(
                str(probe["streams"][0].get("avg_frame_rate", "0/1"))
            ),
            "mode": mode,
            "mp4_file": output_path.name,
            "ffprobe": probe,
        }
        print(
            f"{name}: source_fps={source_fps:.6f}, "
            f"output_fps={args.output_fps:.6f}, mode={mode}, "
            f"file={output_path}"
        )

    episode["video_conversion"] = {
        "target_fps": args.output_fps,
        "codec": "h265",
        "packet_drop_used": False,
        "cameras": results,
    }
    temporary_json = json_path.with_name(".episode.json.tmp")
    temporary_json.write_text(
        json.dumps(episode, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    os.replace(temporary_json, json_path)
    print(f"updated={json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
