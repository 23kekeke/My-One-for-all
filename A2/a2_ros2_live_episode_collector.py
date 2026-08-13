#!/usr/bin/env python3
"""Persistent A2 ROS 2 subscriber with Enter-controlled raw rosbag episodes.

The subscriptions stay alive for the whole collection session.  Pressing
Enter starts a new episode by flushing an in-memory pre-roll buffer beginning
at the latest decodable H.265 keyframe for each camera.  Pressing Enter again
stops and closes the episode without tearing down the subscriptions.

The robot publishes the three H.265 streams at approximately 30 Hz.  This
collector deliberately stores every compressed packet.  Conversion to the
15 Hz training representation happens on Spark after the raw episode closes.
Motion topics are also stored at their source rate and are resampled to 50 Hz
only when producing the training representation.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import signal
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import rclpy
import rosbag2_py
from foxglove_msgs.msg import CompressedVideo
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from rclpy.serialization import serialize_message
from sensor_msgs.msg import JointState
from tf2_msgs.msg import TFMessage


@dataclass(frozen=True)
class TopicSpec:
    name: str
    type_name: str
    message_type: type
    category: str
    minimum_hz: float
    required: bool = True
    transient_local: bool = False
    source_name: str | None = None

    @property
    def subscription_name(self) -> str:
        return self.source_name or self.name


TOPICS = [
    TopicSpec(
        "/aima/hal/rgbd_camera/head_front/color/h265",
        "foxglove_msgs/msg/CompressedVideo",
        CompressedVideo,
        "camera",
        28.0,
    ),
    TopicSpec(
        "/aima/hal/rgbd_camera/hand_left/color/h265",
        "foxglove_msgs/msg/CompressedVideo",
        CompressedVideo,
        "camera",
        28.0,
    ),
    TopicSpec(
        "/aima/hal/rgbd_camera/hand_right/color/h265",
        "foxglove_msgs/msg/CompressedVideo",
        CompressedVideo,
        "camera",
        28.0,
    ),
    TopicSpec(
        "/motion/control/arm_joint_state",
        "sensor_msgs/msg/JointState",
        JointState,
        "state",
        90.0,
        source_name="/a2/relay/motion/control/arm_joint_state",
    ),
    TopicSpec(
        "/motion/control/hand_joint_state",
        "sensor_msgs/msg/JointState",
        JointState,
        "state",
        90.0,
        source_name="/a2/relay/motion/control/hand_joint_state",
    ),
    TopicSpec(
        "/motion/control/neck_joint_state",
        "sensor_msgs/msg/JointState",
        JointState,
        "state",
        90.0,
        source_name="/a2/relay/motion/control/neck_joint_state",
    ),
    TopicSpec(
        "/motion/control/arm_joint_command",
        "sensor_msgs/msg/JointState",
        JointState,
        "action",
        47.5,
        source_name="/a2/relay/motion/control/arm_joint_command",
    ),
    TopicSpec(
        "/motion/control/hand_joint_command",
        "sensor_msgs/msg/JointState",
        JointState,
        "action",
        47.5,
        source_name="/a2/relay/motion/control/hand_joint_command",
    ),
    TopicSpec(
        "/motion/control/neck_joint_command",
        "sensor_msgs/msg/JointState",
        JointState,
        "action",
        1.0,
        required=False,
        source_name="/a2/relay/motion/control/neck_joint_command",
    ),
    TopicSpec(
        "/tf",
        "tf2_msgs/msg/TFMessage",
        TFMessage,
        "transform",
        50.0,
        source_name="/a2/relay/tf",
    ),
    TopicSpec(
        "/tf_static",
        "tf2_msgs/msg/TFMessage",
        TFMessage,
        "static_transform",
        0.0,
        transient_local=True,
        source_name="/a2/relay/tf_static",
    ),
]


@dataclass
class BufferedMessage:
    monotonic_ns: int
    receive_timestamp_ns: int
    serialized_data: bytes
    decodable_keyframe: bool = False


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


def is_decodable_h265_keyframe(message: CompressedVideo) -> bool:
    nal_types = h265_nal_types(bytes(message.data))
    return {32, 33, 34}.issubset(nal_types) and any(
        16 <= value <= 23 for value in nal_types
    )


def topic_metadata(spec: TopicSpec) -> Any:
    values = {
        "name": spec.name,
        "type": spec.type_name,
        "serialization_format": "cdr",
    }
    try:
        return rosbag2_py.TopicMetadata(
            **values,
            offered_qos_profiles="",
        )
    except TypeError:
        return rosbag2_py.TopicMetadata(**values)


class A2LiveCollector(Node):
    def __init__(
        self,
        output_root: Path,
        buffer_seconds: float,
        rate_window_seconds: float,
    ) -> None:
        super().__init__("a2_spark_live_episode_collector")
        self.output_root = output_root
        self.buffer_seconds = buffer_seconds
        self.rate_window_seconds = rate_window_seconds
        self.lock = threading.RLock()
        self.buffers: dict[str, deque[BufferedMessage]] = {
            spec.name: deque() for spec in TOPICS
        }
        self.writer: Any | None = None
        self.recording = False
        self.episode_dir: Path | None = None
        self.episode_control: dict[str, Any] | None = None

        normal_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=100,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        static_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        # rclpy.node.Node exposes ``subscriptions`` as a read-only property
        # in Jazzy. Keep explicit references under a private, non-conflicting
        # name so subscriptions remain alive for the full collector session.
        self._subscription_handles = []
        for spec in TOPICS:
            callback = self._make_callback(spec)
            subscription = self.create_subscription(
                spec.message_type,
                spec.subscription_name,
                callback,
                static_qos if spec.transient_local else normal_qos,
            )
            self._subscription_handles.append(subscription)

    def _make_callback(self, spec: TopicSpec) -> Callable[[Any], None]:
        def callback(message: Any) -> None:
            monotonic_ns = time.monotonic_ns()
            receive_timestamp_ns = self.get_clock().now().nanoseconds
            serialized_data = bytes(serialize_message(message))
            keyframe = (
                is_decodable_h265_keyframe(message)
                if spec.category == "camera"
                else False
            )
            buffered = BufferedMessage(
                monotonic_ns=monotonic_ns,
                receive_timestamp_ns=receive_timestamp_ns,
                serialized_data=serialized_data,
                decodable_keyframe=keyframe,
            )

            with self.lock:
                topic_buffer = self.buffers[spec.name]
                topic_buffer.append(buffered)
                if spec.transient_local:
                    while len(topic_buffer) > 1:
                        topic_buffer.popleft()
                else:
                    cutoff = monotonic_ns - int(self.buffer_seconds * 1e9)
                    while (
                        topic_buffer
                        and topic_buffer[0].monotonic_ns < cutoff
                    ):
                        topic_buffer.popleft()

                if self.recording and self.writer is not None:
                    self.writer.write(
                        spec.name,
                        serialized_data,
                        receive_timestamp_ns,
                    )

        return callback

    def health_snapshot(self) -> tuple[bool, list[dict[str, Any]]]:
        now = time.monotonic_ns()
        rate_cutoff = now - int(self.rate_window_seconds * 1e9)
        rows: list[dict[str, Any]] = []
        ready = True

        with self.lock:
            for spec in TOPICS:
                values = self.buffers[spec.name]
                recent = [
                    item for item in values if item.monotonic_ns >= rate_cutoff
                ]
                rate = len(recent) / self.rate_window_seconds
                age_s = (
                    (now - values[-1].monotonic_ns) / 1e9
                    if values
                    else None
                )
                has_keyframe = (
                    any(item.decodable_keyframe for item in values)
                    if spec.category == "camera"
                    else True
                )

                if spec.transient_local:
                    topic_ready = bool(values)
                else:
                    topic_ready = bool(
                        values
                        and age_s is not None
                        and age_s <= 1.0
                        and rate >= spec.minimum_hz
                        and has_keyframe
                    )

                if spec.required and not topic_ready:
                    ready = False
                rows.append(
                    {
                        "topic": spec.name,
                        "source_topic": spec.subscription_name,
                        "category": spec.category,
                        "rate_hz": rate,
                        "age_s": age_s,
                        "has_keyframe": has_keyframe,
                        "ready": topic_ready,
                    }
                )

        return ready, rows

    def print_health(self) -> bool:
        ready, rows = self.health_snapshot()
        print()
        print("===== A2 live stream health =====")
        for row in rows:
            age = (
                "none"
                if row["age_s"] is None
                else f"{row['age_s']:.3f}s"
            )
            keyframe = (
                f", keyframe={row['has_keyframe']}"
                if row["category"] == "camera"
                else ""
            )
            status = "READY" if row["ready"] else "WAIT"
            print(
                f"[{status:5}] {row['rate_hz']:8.1f} Hz "
                f"age={age}{keyframe}  {row['source_topic']}"
            )
        print(f"overall_ready={ready}")
        return ready

    def _new_writer(self, episode_dir: Path) -> Any:
        writer = rosbag2_py.SequentialWriter()
        writer.open(
            rosbag2_py.StorageOptions(
                uri=str(episode_dir),
                storage_id="sqlite3",
            ),
            rosbag2_py.ConverterOptions("", ""),
        )
        for spec in TOPICS:
            writer.create_topic(topic_metadata(spec))
        return writer

    def start_episode(self) -> Path:
        trigger_monotonic_ns = time.monotonic_ns()
        trigger_wall_ns = self.get_clock().now().nanoseconds

        with self.lock:
            if self.recording:
                raise RuntimeError("An episode is already recording")

            ready, rows = self.health_snapshot()
            if not ready:
                missing = [row["topic"] for row in rows if not row["ready"]]
                raise RuntimeError(
                    "Required streams are not ready: " + ", ".join(missing)
                )

            episode_name = datetime.now().strftime("episode_%Y%m%d_%H%M%S")
            episode_dir = self.output_root / episode_name
            suffix = 1
            while episode_dir.exists():
                episode_dir = self.output_root / f"{episode_name}_{suffix:02d}"
                suffix += 1

            camera_cutoffs: dict[str, int] = {}
            camera_keyframes: dict[str, dict[str, int]] = {}
            for spec in TOPICS:
                if spec.category != "camera":
                    continue
                keyframes = [
                    item
                    for item in self.buffers[spec.name]
                    if item.decodable_keyframe
                    and item.monotonic_ns <= trigger_monotonic_ns
                ]
                if not keyframes:
                    raise RuntimeError(
                        f"No buffered decodable keyframe for {spec.name}"
                    )
                keyframe = keyframes[-1]
                camera_cutoffs[spec.name] = keyframe.monotonic_ns
                camera_keyframes[spec.name] = {
                    "receive_timestamp_ns": keyframe.receive_timestamp_ns,
                    "milliseconds_before_trigger": int(
                        (trigger_monotonic_ns - keyframe.monotonic_ns) / 1e6
                    ),
                }

            non_camera_cutoff = min(camera_cutoffs.values())
            episode_start_receive_ns = min(
                value["receive_timestamp_ns"]
                for value in camera_keyframes.values()
            )
            writer = self._new_writer(episode_dir)

            pre_roll: list[tuple[int, str, bytes]] = []
            for spec in TOPICS:
                values = self.buffers[spec.name]
                if spec.transient_local:
                    selected = list(values)[-1:]
                else:
                    cutoff = (
                        camera_cutoffs[spec.name]
                        if spec.category == "camera"
                        else non_camera_cutoff
                    )
                    selected = [
                        item for item in values if item.monotonic_ns >= cutoff
                    ]
                for item in selected:
                    # A transient-local /tf_static sample may have been
                    # received minutes before this episode. Its transform
                    # header stamps must remain unchanged, but using that old
                    # receive time as the rosbag timestamp would incorrectly
                    # extend the episode duration. Place the latched sample at
                    # the episode start on the bag timeline.
                    write_timestamp_ns = (
                        episode_start_receive_ns
                        if spec.transient_local
                        else item.receive_timestamp_ns
                    )
                    pre_roll.append(
                        (
                            write_timestamp_ns,
                            spec.name,
                            item.serialized_data,
                        )
                    )

            pre_roll.sort(key=lambda item: item[0])
            for receive_timestamp_ns, topic, serialized_data in pre_roll:
                writer.write(topic, serialized_data, receive_timestamp_ns)

            self.writer = writer
            self.recording = True
            self.episode_dir = episode_dir
            self.episode_control = {
                "format_version": 2,
                "episode_name": episode_dir.name,
                "bag_start_timestamp_ns": episode_start_receive_ns,
                "trigger_start_timestamp_ns": trigger_wall_ns,
                "trigger_stop_timestamp_ns": None,
                "buffer_seconds": self.buffer_seconds,
                "camera_keyframes": camera_keyframes,
                "pre_roll_message_count": len(pre_roll),
                "topics": [
                    {
                        "name": spec.name,
                        "source_name": spec.subscription_name,
                        "type": spec.type_name,
                        "category": spec.category,
                    }
                    for spec in TOPICS
                ],
            }

        print(
            f"Episode started: {episode_dir} "
            f"(pre-roll messages={len(pre_roll)})"
        )
        return episode_dir

    def stop_episode(self) -> Path:
        stop_timestamp_ns = self.get_clock().now().nanoseconds
        with self.lock:
            if not self.recording or self.writer is None:
                raise RuntimeError("No episode is currently recording")

            self.recording = False
            writer = self.writer
            episode_dir = self.episode_dir
            episode_control = self.episode_control
            self.writer = None
            self.episode_dir = None
            self.episode_control = None

        if hasattr(writer, "close"):
            writer.close()
        del writer
        gc.collect()

        if episode_dir is None or episode_control is None:
            raise RuntimeError("Internal episode state is incomplete")
        episode_control["trigger_stop_timestamp_ns"] = stop_timestamp_ns
        control_path = episode_dir / "episode_control.json"
        control_tmp = episode_dir / "episode_control.json.tmp"
        with control_tmp.open("w", encoding="utf-8") as file:
            json.dump(episode_control, file, indent=2, ensure_ascii=False)
        os.replace(control_tmp, control_path)

        print(f"Episode stopped and closed: {episode_dir}")
        return episode_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/home/yichu/A2/data"),
    )
    parser.add_argument("--buffer-seconds", type=float, default=5.0)
    parser.add_argument("--rate-window-seconds", type=float, default=2.0)
    parser.add_argument("--post-roll-seconds", type=float, default=1.0)
    parser.add_argument(
        "--monitor-seconds",
        type=float,
        default=0.0,
        help="Only monitor stream health for N seconds, then exit",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    rclpy.init()
    collector = A2LiveCollector(
        output_root=output_root,
        buffer_seconds=args.buffer_seconds,
        rate_window_seconds=args.rate_window_seconds,
    )
    executor_thread = threading.Thread(
        target=rclpy.spin,
        args=(collector,),
        name="rclpy-spin",
        daemon=True,
    )
    executor_thread.start()

    shutting_down = False

    def request_shutdown(_signum: int, _frame: Any) -> None:
        nonlocal shutting_down
        shutting_down = True

    signal.signal(signal.SIGTERM, request_shutdown)

    try:
        if args.monitor_seconds > 0:
            deadline = time.monotonic() + args.monitor_seconds
            while time.monotonic() < deadline and not shutting_down:
                time.sleep(min(2.0, max(0.0, deadline - time.monotonic())))
                collector.print_health()
            return 0

        print(
            "Persistent subscriptions started. Waiting for every required "
            "stream and one decodable keyframe from each camera."
        )
        while not shutting_down:
            while not collector.print_health():
                if shutting_down:
                    break
                time.sleep(2.0)
            if shutting_down:
                break

            command = input(
                "\nAll streams READY. Press Enter to START an episode "
                "(or type q then Enter to quit): "
            ).strip().lower()
            if command == "q":
                break

            try:
                episode_dir = collector.start_episode()
            except RuntimeError as error:
                print(f"Start rejected: {error}")
                continue

            input(
                f"RECORDING {episode_dir.name}. "
                "Press Enter to STOP the episode: "
            )
            if args.post_roll_seconds > 0:
                print(
                    f"Capturing {args.post_roll_seconds:.1f}s post-roll; "
                    "keep the robot still."
                )
                time.sleep(args.post_roll_seconds)
            collector.stop_episode()

    except (KeyboardInterrupt, EOFError):
        print("\nShutdown requested.")
    finally:
        if collector.recording:
            try:
                collector.stop_episode()
            except Exception as error:
                print(f"Failed to close active episode cleanly: {error}")
        collector.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        executor_thread.join(timeout=5.0)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
