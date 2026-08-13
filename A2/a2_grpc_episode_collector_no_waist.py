#!/usr/bin/env python3
"""Spark-side A2 gRPC episode collector.

This program is intentionally read-only with respect to the robot. It only
connects to an A2 edge bridge, receives streams, buffers them, and writes
episodes on Spark.
"""

from __future__ import annotations

import argparse
import base64
import bisect
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import threading
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Deque, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import grpc
    from generated import a2_data_pb2
    from generated import a2_data_pb2_grpc
except ImportError as exc:
    raise SystemExit(
        "Missing Spark gRPC dependencies or generated protobuf files.\n"
        "Run the commands in README_SPARK_GRPC.md first.\n"
        f"Original error: {exc}"
    )


VIDEO_KINDS = {"video"}
STATE_KINDS = {"state", "pending_hand_state"}
COMMAND_KINDS = {"command"}
TF_KINDS = {"tf", "tf_static"}
OPAQUE_KINDS = {"opaque"}


@dataclass(frozen=True)
class StreamSpec:
    name: str
    topic: str
    kind: str
    target_hz: float
    required: bool


@dataclass
class Record:
    stream_name: str
    ros_type: str
    sequence: int
    source_timestamp_ns: int
    sample_timestamp_ns: int
    edge_receive_timestamp_ns: int
    local_receive_timestamp_ns: int
    kind: str
    payload: Any
    has_vps: bool = False
    has_sps: bool = False
    has_pps: bool = False
    is_irap: bool = False

    @property
    def timestamp_ns(self) -> int:
        """Timestamp on the emitted 15/50 Hz sampling grid."""
        return self.sample_timestamp_ns or self.source_timestamp_ns

    def signal_dict(self) -> Dict[str, Any]:
        return {
            "sequence": self.sequence,
            "source_timestamp_ns": self.source_timestamp_ns,
            "sample_timestamp_ns": self.timestamp_ns,
            "edge_receive_timestamp_ns": self.edge_receive_timestamp_ns,
            "spark_receive_timestamp_ns": self.local_receive_timestamp_ns,
            "ros_type": self.ros_type,
            "data": self.payload,
        }


def load_config(path: Path) -> Tuple[Dict[str, Any], Dict[str, StreamSpec]]:
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    specs = {}
    for item in raw["streams"]:
        spec = StreamSpec(
            name=str(item["name"]),
            topic=str(item["topic"]),
            kind=str(item["kind"]),
            target_hz=float(item["target_hz"]),
            required=bool(item["required"]),
        )
        if spec.name in specs:
            raise ValueError(f"duplicate stream name: {spec.name}")
        specs[spec.name] = spec
    return raw, specs


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(q * len(ordered)) - 1))
    return float(ordered[index])


def rate_hz(records: Sequence[Record]) -> float:
    if len(records) < 2:
        return 0.0
    duration = (records[-1].timestamp_ns - records[0].timestamp_ns) / 1e9
    return (len(records) - 1) / duration if duration > 0 else 0.0


def annexb_nal_units(data: bytes) -> List[Tuple[int, bytes]]:
    """Return (HEVC nal_unit_type, complete Annex-B NAL bytes)."""
    starts: List[Tuple[int, int]] = []
    i = 0
    size = len(data)
    while i + 3 <= size:
        if data[i : i + 4] == b"\x00\x00\x00\x01":
            starts.append((i, 4))
            i += 4
        elif data[i : i + 3] == b"\x00\x00\x01":
            starts.append((i, 3))
            i += 3
        else:
            i += 1
    units: List[Tuple[int, bytes]] = []
    for index, (start, prefix_len) in enumerate(starts):
        end = starts[index + 1][0] if index + 1 < len(starts) else size
        header = start + prefix_len
        if header < end:
            nal_type = (data[header] >> 1) & 0x3F
            units.append((nal_type, data[start:end]))
    return units


def h265_flags(data: bytes) -> Tuple[bool, bool, bool, bool]:
    types = {nal_type for nal_type, _ in annexb_nal_units(data)}
    return 32 in types, 33 in types, 34 in types, any(16 <= item <= 23 for item in types)


def json_safe_opaque(content_type: str, data: bytes) -> Any:
    if content_type in {"application/json", "text/json"}:
        try:
            return json.loads(data.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            pass
    return {
        "content_type": content_type or "application/octet-stream",
        "base64": base64.b64encode(data).decode("ascii"),
    }


def record_from_envelope(envelope: Any, spec: StreamSpec) -> Record:
    sample_type = envelope.WhichOneof("sample")
    flags = (False, False, False, False)
    if sample_type == "video":
        data = bytes(envelope.video.annexb)
        detected = h265_flags(data)
        flags = (
            bool(envelope.video.has_vps or detected[0]),
            bool(envelope.video.has_sps or detected[1]),
            bool(envelope.video.has_pps or detected[2]),
            bool(envelope.video.is_irap or detected[3]),
        )
        payload: Any = data
    elif sample_type == "joint":
        payload = {
            "name": list(envelope.joint.name),
            "position": list(envelope.joint.position),
            "velocity": list(envelope.joint.velocity),
            "effort": list(envelope.joint.effort),
        }
    elif sample_type == "tf":
        payload = {
            "transforms": [
                {
                    "parent_frame": transform.parent_frame,
                    "child_frame": transform.child_frame,
                    "timestamp_ns": int(transform.timestamp_ns),
                    "translation_xyz": list(transform.translation_xyz),
                    "rotation_xyzw": list(transform.rotation_xyzw),
                }
                for transform in envelope.tf.transforms
            ]
        }
    elif sample_type == "opaque":
        payload = {
            "type_name": envelope.opaque.type_name,
            "value": json_safe_opaque(
                envelope.opaque.content_type, bytes(envelope.opaque.data)
            ),
        }
    else:
        raise ValueError(
            f"{envelope.stream_name}: envelope has no supported sample payload"
        )
    return Record(
        stream_name=envelope.stream_name,
        ros_type=envelope.ros_type,
        sequence=int(envelope.sequence),
        source_timestamp_ns=int(envelope.source_timestamp_ns),
        sample_timestamp_ns=int(envelope.sample_timestamp_ns),
        edge_receive_timestamp_ns=int(envelope.edge_receive_timestamp_ns),
        local_receive_timestamp_ns=time.time_ns(),
        kind=spec.kind,
        payload=payload,
        has_vps=flags[0],
        has_sps=flags[1],
        has_pps=flags[2],
        is_irap=flags[3],
    )


class StreamBuffer:
    def __init__(self, seconds: float):
        self.seconds_ns = int(seconds * 1e9)
        self.records: Deque[Record] = deque()
        self.total_received = 0
        self.sequence_gaps = 0
        self.last_sequence: Optional[int] = None

    def append(self, record: Record) -> None:
        if self.last_sequence is not None and record.sequence > self.last_sequence + 1:
            self.sequence_gaps += record.sequence - self.last_sequence - 1
        if self.last_sequence is None or record.sequence > self.last_sequence:
            self.last_sequence = record.sequence
        self.total_received += 1
        self.records.append(record)
        cutoff = record.timestamp_ns - self.seconds_ns
        while self.records and self.records[0].timestamp_ns < cutoff:
            self.records.popleft()

    def snapshot(self) -> List[Record]:
        return list(self.records)


def latest_parameter_sets(records: Sequence[Record], end_index: int) -> Optional[bytes]:
    found: Dict[int, bytes] = {}
    for record in records[: end_index + 1]:
        for nal_type, unit in annexb_nal_units(record.payload):
            if nal_type in {32, 33, 34}:
                found[nal_type] = unit
    if not all(item in found for item in (32, 33, 34)):
        return None
    return found[32] + found[33] + found[34]


def decodable_video_preroll(
    records: Sequence[Record], logical_start_ns: int
) -> Tuple[bytes, List[Record]]:
    irap_indices = [
        index
        for index, record in enumerate(records)
        if record.is_irap and record.timestamp_ns <= logical_start_ns
    ]
    if not irap_indices:
        raise RuntimeError("no IRAP frame exists before the episode trigger")
    start_index = irap_indices[-1]
    parameter_blob = latest_parameter_sets(records, start_index)
    if parameter_blob is None:
        raise RuntimeError("VPS/SPS/PPS are not available before the selected IRAP")
    return parameter_blob, list(records[start_index:])


class EpisodeWriter:
    def __init__(
        self,
        output_root: Path,
        specs: Dict[str, StreamSpec],
        config: Dict[str, Any],
        logical_start_ns: int,
        trigger_wall_ns: int,
        snapshots: Dict[str, List[Record]],
    ):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.name = f"episode_{timestamp}"
        self.output_root = output_root
        self.partial_dir = output_root / f".{self.name}.partial"
        self.final_dir = output_root / self.name
        suffix = 1
        while self.partial_dir.exists() or self.final_dir.exists():
            self.name = f"episode_{timestamp}_{suffix:02d}"
            self.partial_dir = output_root / f".{self.name}.partial"
            self.final_dir = output_root / self.name
            suffix += 1
        self.partial_dir.mkdir(parents=True)
        self.specs = specs
        self.config = config
        self.logical_start_ns = logical_start_ns
        self.trigger_start_wall_ns = trigger_wall_ns
        self.trigger_stop_wall_ns: Optional[int] = None
        self.logical_stop_ns: Optional[int] = None
        self.video_handles: Dict[str, Any] = {}
        self.video_records: Dict[str, List[Record]] = defaultdict(list)
        self.signal_records: Dict[str, List[Record]] = defaultdict(list)
        self.preroll: Dict[str, Dict[str, Any]] = {}
        self.ffmpeg_results: Dict[str, Dict[str, Any]] = {}

        for spec in specs.values():
            if spec.kind in VIDEO_KINDS:
                raw_path = self.partial_dir / f"{spec.name}.h265"
                self.video_handles[spec.name] = raw_path.open("wb")
                parameter_blob, video_records = decodable_video_preroll(
                    snapshots[spec.name], logical_start_ns
                )
                self.video_handles[spec.name].write(parameter_blob)
                for record in video_records:
                    self._write_video(record)
                self.preroll[spec.name] = {
                    "first_frame_timestamp_ns": video_records[0].timestamp_ns,
                    "logical_start_timestamp_ns": logical_start_ns,
                    "preroll_ms": (
                        logical_start_ns - video_records[0].timestamp_ns
                    )
                    / 1e6,
                    "preroll_frames": sum(
                        item.timestamp_ns < logical_start_ns
                        for item in video_records
                    ),
                }

        earliest_video_ns = min(
            records[0].timestamp_ns
            for records in self.video_records.values()
            if records
        )
        for spec in specs.values():
            if spec.kind not in VIDEO_KINDS:
                for record in snapshots[spec.name]:
                    if (
                        record.timestamp_ns >= earliest_video_ns
                        or spec.kind == "tf_static"
                    ):
                        self.signal_records[spec.name].append(record)

    def _write_video(self, record: Record) -> None:
        self.video_handles[record.stream_name].write(record.payload)
        self.video_records[record.stream_name].append(record)

    def write(self, record: Record) -> None:
        if record.kind in VIDEO_KINDS:
            self._write_video(record)
        else:
            self.signal_records[record.stream_name].append(record)

    def close_inputs(self, logical_stop_ns: int, trigger_stop_wall_ns: int) -> None:
        self.logical_stop_ns = logical_stop_ns
        self.trigger_stop_wall_ns = trigger_stop_wall_ns
        for handle in self.video_handles.values():
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()

    def _remux_videos(self) -> None:
        ffmpeg = str(self.config["ffmpeg"])
        fps = float(self.config["camera_target_hz"])
        for stream_name in self.video_handles:
            raw_path = self.partial_dir / f"{stream_name}.h265"
            mp4_path = self.partial_dir / f"{stream_name}.mp4"
            command = [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-r",
                str(fps),
                "-i",
                str(raw_path),
                "-c:v",
                "copy",
                "-tag:v",
                "hvc1",
                str(mp4_path),
            ]
            result = subprocess.run(command, capture_output=True, text=True)
            success = result.returncode == 0 and mp4_path.exists() and mp4_path.stat().st_size > 0
            decode_result = None
            if success:
                decode_command = [
                    ffmpeg,
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-i",
                    str(mp4_path),
                    "-f",
                    "null",
                    "-",
                ]
                decode_result = subprocess.run(
                    decode_command, capture_output=True, text=True
                )
                success = decode_result.returncode == 0
            self.ffmpeg_results[stream_name] = {
                "success": success,
                "returncode": result.returncode,
                "stderr": result.stderr[-2000:],
                "decode_returncode": (
                    decode_result.returncode if decode_result is not None else None
                ),
                "decode_stderr": (
                    decode_result.stderr[-2000:] if decode_result is not None else ""
                ),
                "raw_file": raw_path.name,
                "mp4_file": mp4_path.name if success else None,
            }
            if success:
                raw_path.unlink()

    @staticmethod
    def _nearest_index(records: Sequence[Record], timestamp_ns: int) -> Tuple[int, int]:
        timestamps = [record.timestamp_ns for record in records]
        index = bisect.bisect_left(timestamps, timestamp_ns)
        candidates = []
        if index < len(records):
            candidates.append(index)
        if index > 0:
            candidates.append(index - 1)
        best = min(
            candidates,
            key=lambda item: abs(records[item].timestamp_ns - timestamp_ns),
        )
        return best, abs(records[best].timestamp_ns - timestamp_ns)

    @staticmethod
    def _previous_record(
        records: Sequence[Record], timestamp_ns: int
    ) -> Optional[Record]:
        if not records:
            return None
        timestamps = [record.timestamp_ns for record in records]
        index = bisect.bisect_right(timestamps, timestamp_ns) - 1
        return records[max(0, index)]

    @staticmethod
    def _interpolate_joint(
        records: Sequence[Record], timestamp_ns: int
    ) -> Optional[Dict[str, Any]]:
        if not records:
            return None
        if not isinstance(records[0].payload, dict) or "position" not in records[0].payload:
            source = EpisodeWriter._previous_record(records, timestamp_ns)
            return (
                {
                    "source_timestamp_ns": source.source_timestamp_ns,
                    "sample_timestamp_ns": source.timestamp_ns,
                    **source.payload,
                }
                if source
                else None
            )
        timestamps = [record.timestamp_ns for record in records]
        right = bisect.bisect_left(timestamps, timestamp_ns)
        if right <= 0:
            source = records[0]
            return {
                "source_timestamp_ns": source.source_timestamp_ns,
                "sample_timestamp_ns": source.timestamp_ns,
                **source.payload,
            }
        if right >= len(records):
            source = records[-1]
            return {
                "source_timestamp_ns": source.source_timestamp_ns,
                "sample_timestamp_ns": source.timestamp_ns,
                **source.payload,
            }
        left_record = records[right - 1]
        right_record = records[right]
        left_payload = left_record.payload
        right_payload = right_record.payload
        left_pos = left_payload.get("position", [])
        right_pos = right_payload.get("position", [])
        if len(left_pos) != len(right_pos):
            source = left_record
            return {
                "source_timestamp_ns": source.source_timestamp_ns,
                "sample_timestamp_ns": source.timestamp_ns,
                **source.payload,
            }
        span = right_record.timestamp_ns - left_record.timestamp_ns
        ratio = (
            (timestamp_ns - left_record.timestamp_ns) / span if span > 0 else 0.0
        )

        def interpolate_field(name: str) -> List[float]:
            first = left_payload.get(name, [])
            second = right_payload.get(name, [])
            if len(first) != len(second):
                return list(first)
            return [a + ratio * (b - a) for a, b in zip(first, second)]

        return {
            "source_timestamps_ns": [
                left_record.source_timestamp_ns,
                right_record.source_timestamp_ns,
            ],
            "sample_timestamps_ns": [
                left_record.timestamp_ns,
                right_record.timestamp_ns,
            ],
            "name": list(left_payload.get("name", [])),
            "position": interpolate_field("position"),
            "velocity": interpolate_field("velocity"),
            "effort": interpolate_field("effort"),
        }

    def _build_episode_json(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        if self.logical_stop_ns is None:
            raise RuntimeError("episode has not been stopped")
        head_records = self.video_records["head_camera"]
        frame_candidates = [
            (index, record)
            for index, record in enumerate(head_records)
            if self.logical_start_ns
            <= record.timestamp_ns
            <= self.logical_stop_ns
        ]
        hard_limit_ns = int(
            float(self.config["camera_alignment_hard_limit_ms"]) * 1e6
        )
        frames = []
        rejected_frames = 0
        for frame_id, (head_index, head_record) in enumerate(frame_candidates):
            image_indices = {"head_camera": head_index}
            image_offsets_ms = {"head_camera": 0.0}
            frame_ok = True
            for camera in ("left_hand_camera", "right_hand_camera"):
                index, delta_ns = self._nearest_index(
                    self.video_records[camera], head_record.timestamp_ns
                )
                if delta_ns > hard_limit_ns:
                    frame_ok = False
                    break
                image_indices[camera] = index
                image_offsets_ms[camera] = delta_ns / 1e6
            if not frame_ok:
                rejected_frames += 1
                continue

            observation = {}
            action = {}
            for name in (
                "arm_joint_state",
                "hand_joint_state",
                "neck_joint_state",
            ):
                observation[name] = self._interpolate_joint(
                    self.signal_records[name], head_record.timestamp_ns
                )
            for name in (
                "arm_joint_command",
                "hand_joint_command",
                "neck_joint_command",
            ):
                source = self._previous_record(
                    self.signal_records[name], head_record.timestamp_ns
                )
                action[name] = (
                    {
                        "source_timestamp_ns": source.source_timestamp_ns,
                        "sample_timestamp_ns": source.timestamp_ns,
                        **source.payload,
                    }
                    if source
                    else None
                )
            tf_record = self._previous_record(
                self.signal_records["tf"], head_record.timestamp_ns
            )
            if tf_record:
                observation["tf"] = {
                    "source_timestamp_ns": tf_record.source_timestamp_ns,
                    "sample_timestamp_ns": tf_record.timestamp_ns,
                    **tf_record.payload,
                }
            frames.append(
                {
                    "frame_id": len(frames),
                    "timestamp_ns": head_record.timestamp_ns,
                    "timestamp": head_record.timestamp_ns / 1e9,
                    "images": image_indices,
                    "camera_offset_ms": image_offsets_ms,
                    "observation": observation,
                    "action": action,
                }
            )

        tf_static_records = self.signal_records.get("tf_static", [])
        tf_static = tf_static_records[-1].signal_dict() if tf_static_records else None
        episode_json = {
            "episode_name": self.name,
            "camera_fps": float(self.config["camera_target_hz"]),
            "signal_hz": float(self.config["signal_target_hz"]),
            "logical_start_timestamp_ns": self.logical_start_ns,
            "logical_stop_timestamp_ns": self.logical_stop_ns,
            "duration_s": (self.logical_stop_ns - self.logical_start_ns) / 1e9,
            "num_frames": len(frames),
            "video_files": {
                name: f"{name}.mp4" for name in self.video_handles
            },
            "frames": frames,
        }
        alignment = {
            "candidate_head_frames": len(frame_candidates),
            "accepted_frames": len(frames),
            "rejected_frames": rejected_frames,
            "hard_limit_ms": float(
                self.config["camera_alignment_hard_limit_ms"]
            ),
            "tf_static_present": tf_static is not None,
        }
        with (self.partial_dir / "tf_static.json").open("w", encoding="utf-8") as handle:
            json.dump(tf_static, handle, ensure_ascii=False, indent=2)
        return episode_json, alignment

    def _write_signals(self) -> None:
        if self.logical_stop_ns is None:
            raise RuntimeError("episode has not been stopped")
        output = {
            "target_hz": float(self.config["signal_target_hz"]),
            "logical_start_timestamp_ns": self.logical_start_ns,
            "logical_stop_timestamp_ns": self.logical_stop_ns,
            "streams": {},
        }
        for name, records in self.signal_records.items():
            if name == "tf_static":
                continue
            output["streams"][name] = [
                record.signal_dict()
                for record in records
                if self.logical_start_ns
                <= record.timestamp_ns
                <= self.logical_stop_ns
            ]
        with (self.partial_dir / "signals_50hz.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(output, handle, ensure_ascii=False, separators=(",", ":"))

    def _stream_quality(self, spec: StreamSpec, records: Sequence[Record]) -> Dict[str, Any]:
        if self.logical_stop_ns is None:
            raise RuntimeError("episode has not been stopped")
        selected = [
            record
            for record in records
            if self.logical_start_ns
            <= record.timestamp_ns
            <= self.logical_stop_ns
        ]
        gaps_ms = [
            (right.timestamp_ns - left.timestamp_ns) / 1e6
            for left, right in zip(selected, selected[1:])
        ]
        rate = rate_hz(selected)
        sequence_gaps = sum(
            max(0, right.sequence - left.sequence - 1)
            for left, right in zip(selected, selected[1:])
        )
        if spec.kind == "tf_static":
            passed = len(records) > 0
        else:
            tolerance = (
                float(self.config["camera_rate_tolerance"])
                if spec.kind == "video"
                else float(self.config["signal_rate_tolerance"])
            )
            lower = spec.target_hz * (1.0 - tolerance)
            upper = spec.target_hz * (1.0 + tolerance)
            passed = bool(selected) and lower <= rate <= upper
        result = {
            "required": spec.required,
            "kind": spec.kind,
            "target_hz": spec.target_hz,
            "count": len(selected),
            "rate_hz": rate,
            "p99_gap_ms": percentile(gaps_ms, 0.99),
            "max_gap_ms": max(gaps_ms, default=0.0),
            "sequence_gaps": sequence_gaps,
            "pass": passed,
        }
        if spec.kind == "video":
            result.update(
                {
                    "has_vps": any(record.has_vps for record in records),
                    "has_sps": any(record.has_sps for record in records),
                    "has_pps": any(record.has_pps for record in records),
                    "has_irap": any(record.is_irap for record in records),
                    "mp4_success": self.ffmpeg_results.get(spec.name, {}).get(
                        "success", False
                    ),
                }
            )
            result["pass"] = bool(
                result["pass"]
                and result["has_vps"]
                and result["has_sps"]
                and result["has_pps"]
                and result["has_irap"]
                and result["mp4_success"]
            )
        return result

    def _write_manifest(
        self, quality: Dict[str, Any], alignment: Dict[str, Any], valid: bool
    ) -> None:
        manifest = {
            "schema_version": 1,
            "episode_name": self.name,
            "valid": valid,
            "created_at": datetime.now().isoformat(),
            "trigger_start_wall_timestamp_ns": self.trigger_start_wall_ns,
            "trigger_stop_wall_timestamp_ns": self.trigger_stop_wall_ns,
            "logical_start_source_timestamp_ns": self.logical_start_ns,
            "logical_stop_source_timestamp_ns": self.logical_stop_ns,
            "preroll": self.preroll,
            "streams": [asdict(spec) for spec in self.specs.values()],
            "hand_semantics": {
                "stream": "hand_joint_state",
                "position_expected_length": 20,
                "effort_expected_length": 260,
                "semantic_status": "pending",
            },
            "ffmpeg": self.ffmpeg_results,
            "alignment": alignment,
            "quality_summary": {
                "valid": valid,
                "failed_required_streams": [
                    name
                    for name, item in quality["streams"].items()
                    if item["required"] and not item["pass"]
                ],
            },
        }
        with (self.partial_dir / "manifest.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2)

    def _write_hashes(self) -> None:
        lines = []
        for path in sorted(self.partial_dir.iterdir(), key=lambda item: item.name):
            if not path.is_file() or path.name == "SHA256SUMS":
                continue
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            lines.append(f"{digest.hexdigest()}  {path.name}")
        (self.partial_dir / "SHA256SUMS").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )

    def finalize(self) -> Tuple[Path, bool]:
        self._remux_videos()
        episode_json, alignment = self._build_episode_json()
        with (self.partial_dir / "episode.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(episode_json, handle, ensure_ascii=False, separators=(",", ":"))
        self._write_signals()

        quality_streams = {}
        for name, spec in self.specs.items():
            records = (
                self.video_records.get(name, [])
                if spec.kind == "video"
                else self.signal_records.get(name, [])
            )
            quality_streams[name] = self._stream_quality(spec, records)
        valid = (
            all(
                not item["required"] or item["pass"]
                for item in quality_streams.values()
            )
            and alignment["accepted_frames"] > 0
            and alignment["rejected_frames"] == 0
        )
        quality = {
            "valid": valid,
            "streams": quality_streams,
            "alignment": alignment,
        }
        with (self.partial_dir / "quality_report.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(quality, handle, ensure_ascii=False, indent=2)
        self._write_manifest(quality, alignment, valid)
        self._write_hashes()

        destination = self.final_dir if valid else Path(f"{self.final_dir}_INVALID")
        self.partial_dir.rename(destination)
        return destination, valid


class SparkCollector:
    def __init__(
        self,
        config: Dict[str, Any],
        specs: Dict[str, StreamSpec],
        server: str,
    ):
        self.config = config
        self.specs = specs
        self.server = server
        self.output_root = Path(config["output_root"])
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.buffers = {
            name: StreamBuffer(float(config["buffer_seconds"])) for name in specs
        }
        self.last_error: Dict[str, str] = {}
        self.connected: Dict[str, bool] = defaultdict(bool)
        self.receiver_threads: List[threading.Thread] = []
        self.active_episode: Optional[EpisodeWriter] = None
        self.ready_since: Optional[float] = None
        self.health_thread: Optional[threading.Thread] = None

    def ingest(self, envelope: Any) -> None:
        name = envelope.stream_name
        spec = self.specs.get(name)
        if spec is None:
            return
        record = record_from_envelope(envelope, spec)
        if record.source_timestamp_ns <= 0 or record.timestamp_ns <= 0:
            raise ValueError(
                f"{name}: source_timestamp_ns and sample_timestamp_ns must be positive"
            )
        with self.lock:
            self.buffers[name].append(record)
            if self.active_episode is not None:
                self.active_episode.write(record)

    def _receiver(self, stream_name: str) -> None:
        options = [
            ("grpc.max_receive_message_length", 64 * 1024 * 1024),
            ("grpc.keepalive_time_ms", 10_000),
            ("grpc.keepalive_timeout_ms", 5_000),
            ("grpc.keepalive_permit_without_calls", 1),
        ]
        while not self.stop_event.is_set():
            channel = None
            try:
                channel = grpc.insecure_channel(self.server, options=options)
                # Creating a Channel object does not mean that a TCP/gRPC
                # connection has been established. Wait for the channel to
                # become READY before reporting connected=True.
                grpc.channel_ready_future(channel).result(timeout=3.0)
                stub = a2_data_pb2_grpc.A2DataServiceStub(channel)
                request = a2_data_pb2.SubscribeRequest(stream_names=[stream_name])
                with self.lock:
                    self.connected[stream_name] = True
                    self.last_error.pop(stream_name, None)
                for envelope in stub.Subscribe(request):
                    if self.stop_event.is_set():
                        break
                    self.ingest(envelope)
            except Exception as exc:
                with self.lock:
                    self.connected[stream_name] = False
                    self.last_error[stream_name] = f"{type(exc).__name__}: {exc}"
                if not self.stop_event.wait(1.0):
                    continue
            finally:
                if channel is not None:
                    channel.close()

    def start(self) -> None:
        for stream_name in self.specs:
            thread = threading.Thread(
                target=self._receiver,
                args=(stream_name,),
                daemon=True,
                name=f"grpc-{stream_name}",
            )
            thread.start()
            self.receiver_threads.append(thread)
        self.health_thread = threading.Thread(
            target=self._health_loop, daemon=True, name="health-printer"
        )
        self.health_thread.start()

    def _recent_records(self, name: str, seconds: float = 3.0) -> List[Record]:
        records = self.buffers[name].snapshot()
        if not records:
            return []
        cutoff = records[-1].timestamp_ns - int(seconds * 1e9)
        return [record for record in records if record.timestamp_ns >= cutoff]

    def readiness(self) -> Tuple[bool, Dict[str, Dict[str, Any]]]:
        report = {}
        all_ready = True
        with self.lock:
            for name, spec in self.specs.items():
                records = self._recent_records(name)
                rate = rate_hz(records)
                connected = self.connected.get(name, False)
                if spec.kind == "tf_static":
                    ready = connected and bool(records)
                else:
                    tolerance = (
                        float(self.config["camera_rate_tolerance"])
                        if spec.kind == "video"
                        else float(self.config["signal_rate_tolerance"])
                    )
                    ready = (
                        connected
                        and len(records) >= 2
                        and spec.target_hz * (1 - tolerance)
                        <= rate
                        <= spec.target_hz * (1 + tolerance)
                    )
                details: Dict[str, Any] = {
                    "connected": connected,
                    "count_3s": len(records),
                    "rate_hz": rate,
                    "ready": ready,
                }
                if spec.kind == "video":
                    full_video_buffer = self.buffers[name].snapshot()
                    details.update(
                        {
                            "vps": any(record.has_vps for record in full_video_buffer),
                            "sps": any(record.has_sps for record in full_video_buffer),
                            "pps": any(record.has_pps for record in full_video_buffer),
                            "irap": any(record.is_irap for record in full_video_buffer),
                        }
                    )
                    ready = ready and all(
                        details[item] for item in ("vps", "sps", "pps", "irap")
                    )
                    details["ready"] = ready
                if name in self.last_error:
                    details["error"] = self.last_error[name]
                report[name] = details
                if spec.required and not ready:
                    all_ready = False

            now = time.monotonic()
            if all_ready:
                if self.ready_since is None:
                    self.ready_since = now
                warmed = now - self.ready_since >= float(
                    self.config["warmup_seconds"]
                )
            else:
                self.ready_since = None
                warmed = False
            return all_ready and warmed, report

    def _health_loop(self) -> None:
        while not self.stop_event.wait(1.0):
            ready, report = self.readiness()
            summary = " ".join(
                f"{name}={item['rate_hz']:.1f}"
                if self.specs[name].kind != "tf_static"
                else f"{name}={'yes' if item['count_3s'] else 'no'}"
                for name, item in report.items()
            )
            state = "RECORDING" if self.active_episode else ("READY" if ready else "WAIT")
            print(f"\n[{state}] {summary}", flush=True)

    def start_episode(self) -> Optional[Path]:
        with self.lock:
            ready, report = self.readiness()
            if not ready:
                missing = [
                    name
                    for name, item in report.items()
                    if self.specs[name].required and not item["ready"]
                ]
                print("Cannot start: required streams are not ready:")
                for name in missing:
                    print(f"  - {name}: {report[name]}")
                return None
            if self.active_episode is not None:
                print("An episode is already recording")
                return None
            head_records = self.buffers["head_camera"].snapshot()
            logical_start_ns = head_records[-1].timestamp_ns
            snapshots = {
                name: buffer.snapshot() for name, buffer in self.buffers.items()
            }
            writer = EpisodeWriter(
                output_root=self.output_root,
                specs=self.specs,
                config=self.config,
                logical_start_ns=logical_start_ns,
                trigger_wall_ns=time.time_ns(),
                snapshots=snapshots,
            )
            self.active_episode = writer
            print(f"Recording {writer.name}")
            return writer.partial_dir

    def stop_episode(self) -> Optional[Tuple[Path, bool]]:
        with self.lock:
            writer = self.active_episode
        if writer is None:
            print("No active episode")
            return None
        post_roll = float(self.config["post_roll_seconds"])
        print(f"Stopping after {post_roll:.1f}s post-roll...")
        time.sleep(post_roll)
        with self.lock:
            head_records = self.buffers["head_camera"].snapshot()
            logical_stop_ns = head_records[-1].timestamp_ns
            trigger_stop_wall_ns = time.time_ns()
            self.active_episode = None
            writer.close_inputs(logical_stop_ns, trigger_stop_wall_ns)
        try:
            path, valid = writer.finalize()
            print(f"Saved: {path}")
            print(f"Episode valid: {valid}")
            return path, valid
        except Exception:
            print(f"Finalization failed; partial data retained at {writer.partial_dir}")
            raise

    def monitor(self, seconds: float) -> int:
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline and not self.stop_event.wait(0.5):
            pass
        ready, report = self.readiness()
        print(json.dumps({"overall_ready": ready, "streams": report}, indent=2))
        return 0 if ready else 2

    def shutdown(self) -> None:
        if self.active_episode is not None:
            try:
                self.stop_episode()
            except Exception as exc:
                print(f"Failed to finalize active episode during shutdown: {exc}")
        self.stop_event.set()
        for thread in self.receiver_threads:
            thread.join(timeout=2.0)


def parse_args() -> argparse.Namespace:
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=base / "config" / "spark_collector_no_waist.json",
    )
    parser.add_argument("--server", help="Override gRPC server host:port")
    parser.add_argument(
        "--output-root", type=Path, help="Override episode output directory"
    )
    parser.add_argument(
        "--monitor-seconds",
        type=float,
        default=0.0,
        help="Monitor only; do not create episodes",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config, specs = load_config(args.config)
    if args.server:
        config["server"] = args.server
    if args.output_root:
        config["output_root"] = str(args.output_root)
    collector = SparkCollector(config, specs, str(config["server"]))
    collector.start()
    try:
        if args.monitor_seconds > 0:
            return collector.monitor(args.monitor_seconds)
        print("Spark collector started. It never publishes robot commands.")
        print("Wait for READY, then press Enter to start an episode.")
        print("Press Enter again to stop; enter q while idle to quit.")
        while True:
            command = input("\n> ").strip().lower()
            if command == "q" and collector.active_episode is None:
                break
            if collector.active_episode is None:
                collector.start_episode()
            else:
                collector.stop_episode()
    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        collector.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
