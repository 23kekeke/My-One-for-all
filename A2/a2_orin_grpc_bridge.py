#!/usr/bin/env python3
"""Read-only ROS 2 to gRPC bridge for the AgiBot A2 Orin.

The bridge subscribes continuously before an episode is triggered on Spark.
Video packets are forwarded unchanged because dropping inter-predicted H.265
packets would corrupt the bitstream. Joint and dynamic TF streams are sampled
onto a 50 Hz grid using latest-value hold while preserving each ROS source
timestamp separately from the grid timestamp.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import queue
import signal
import threading
import time
from collections import defaultdict, deque
from concurrent import futures
from dataclasses import dataclass
from typing import Any, Deque, Dict, Iterable, List, Optional, Tuple

import grpc
import rclpy
from foxglove_msgs.msg import CompressedVideo
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from sensor_msgs.msg import JointState
from tf2_msgs.msg import TFMessage

from generated import a2_data_pb2
from generated import a2_data_pb2_grpc


VIDEO_KIND = "video"
JOINT_KINDS = {"state", "pending_hand_state", "command"}
TF_KIND = "tf"
TF_STATIC_KIND = "tf_static"


@dataclass(frozen=True)
class StreamSpec:
    name: str
    topic: str
    ros_type: str
    kind: str
    target_hz: float
    required: bool
    width: int = 0
    height: int = 0


@dataclass
class LatestSample:
    source_timestamp_ns: int
    edge_receive_timestamp_ns: int
    payload: Any


class RuntimeState:
    def __init__(
        self,
        specs: Dict[str, StreamSpec],
        health_window_seconds: float,
        subscriber_queue_size: int,
    ):
        self.specs = specs
        self.health_window_ns = int(health_window_seconds * 1e9)
        self.subscriber_queue_size = subscriber_queue_size
        self.lock = threading.RLock()
        self.condition = threading.Condition(self.lock)
        self.latest: Dict[str, LatestSample] = {}
        self.next_sample_ns: Dict[str, int] = {}
        self.sequence: Dict[str, int] = defaultdict(int)
        self.received: Dict[str, int] = defaultdict(int)
        self.dropped: Dict[str, int] = defaultdict(int)
        self.emitted_times: Dict[str, Deque[int]] = {
            name: deque() for name in specs
        }
        self.last_envelope: Dict[str, Any] = {}
        self.last_source_timestamp_ns: Dict[str, int] = defaultdict(int)
        self.subscribers: Dict[int, Tuple[set[str], queue.Queue[Any]]] = {}
        self.next_subscriber_id = 1
        self.shutdown_event = threading.Event()

    def update_latest(self, name: str, sample: LatestSample) -> None:
        spec = self.specs[name]
        with self.lock:
            self.received[name] += 1
            self.latest[name] = sample
            self.last_source_timestamp_ns[name] = sample.source_timestamp_ns
            if spec.target_hz > 0 and name not in self.next_sample_ns:
                period_ns = round(1e9 / spec.target_hz)
                self.next_sample_ns[name] = (
                    (sample.source_timestamp_ns + period_ns - 1) // period_ns
                ) * period_ns

    def update_tf_latest(self, name: str, sample: LatestSample) -> None:
        """Merge dynamic TF by frame pair before sampling the 50 Hz snapshot."""
        with self.lock:
            existing = self.latest.get(name)
            merged = {
                (item["parent_frame"], item["child_frame"]): item
                for item in (existing.payload if existing is not None else [])
            }
            for item in sample.payload:
                merged[(item["parent_frame"], item["child_frame"])] = item
            self.received[name] += 1
            self.latest[name] = LatestSample(
                source_timestamp_ns=sample.source_timestamp_ns,
                edge_receive_timestamp_ns=sample.edge_receive_timestamp_ns,
                payload=list(merged.values()),
            )
            self.last_source_timestamp_ns[name] = sample.source_timestamp_ns
            spec = self.specs[name]
            if name not in self.next_sample_ns:
                period_ns = round(1e9 / spec.target_hz)
                self.next_sample_ns[name] = (
                    (sample.source_timestamp_ns + period_ns - 1) // period_ns
                ) * period_ns

    def next_sequence(self, name: str) -> int:
        self.sequence[name] += 1
        return self.sequence[name]

    def publish(self, envelope: Any) -> None:
        name = envelope.stream_name
        emitted_ns = int(envelope.sample_timestamp_ns)
        with self.lock:
            self.last_envelope[name] = envelope
            history = self.emitted_times[name]
            history.append(emitted_ns)
            cutoff = emitted_ns - self.health_window_ns
            while history and history[0] < cutoff:
                history.popleft()

            for stream_names, output_queue in self.subscribers.values():
                if name not in stream_names:
                    continue
                try:
                    output_queue.put_nowait(envelope)
                except queue.Full:
                    try:
                        output_queue.get_nowait()
                    except queue.Empty:
                        pass
                    try:
                        output_queue.put_nowait(envelope)
                    except queue.Full:
                        pass
                    self.dropped[name] += 1
            self.condition.notify_all()

    def register_subscriber(
        self, stream_names: Iterable[str]
    ) -> Tuple[int, queue.Queue[Any]]:
        names = set(stream_names)
        unknown = names.difference(self.specs)
        if unknown:
            raise KeyError(", ".join(sorted(unknown)))
        output_queue: queue.Queue[Any] = queue.Queue(
            maxsize=self.subscriber_queue_size
        )
        with self.lock:
            subscriber_id = self.next_subscriber_id
            self.next_subscriber_id += 1
            self.subscribers[subscriber_id] = (names, output_queue)
            for name in names:
                if self.specs[name].kind != TF_STATIC_KIND:
                    continue
                envelope = self.last_envelope.get(name)
                if envelope is not None:
                    output_queue.put_nowait(envelope)
        return subscriber_id, output_queue

    def unregister_subscriber(self, subscriber_id: int) -> None:
        with self.lock:
            self.subscribers.pop(subscriber_id, None)

    def health(self, name: str) -> Tuple[bool, float, int, int, int, str]:
        spec = self.specs[name]
        with self.lock:
            now_ns = time.time_ns()
            cutoff = now_ns - self.health_window_ns
            history = [
                value for value in self.emitted_times[name] if value >= cutoff
            ]
            static_present = name in self.last_envelope
            received = self.received[name]
            dropped = self.dropped[name]
            last_source = self.last_source_timestamp_ns[name]
        if spec.kind == TF_STATIC_KIND:
            ready = static_present
            rate = 0.0
        elif len(history) >= 2:
            duration = (history[-1] - history[0]) / 1e9
            rate = (len(history) - 1) / duration if duration > 0 else 0.0
            ready = rate >= spec.target_hz * 0.95
        else:
            rate = 0.0
            ready = False
        detail = "ready" if ready else "waiting for emitted samples"
        return ready, rate, received, dropped, last_source, detail


def stamp_to_ns(stamp: Any) -> int:
    value = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
    return value


def source_or_receive(stamp: Any, receive_timestamp_ns: int) -> int:
    value = stamp_to_ns(stamp)
    return value if value > 0 else receive_timestamp_ns


def annexb_nal_types(data: bytes) -> set[int]:
    types: set[int] = set()
    index = 0
    size = len(data)
    while index + 3 <= size:
        prefix = 0
        if data[index : index + 4] == b"\x00\x00\x00\x01":
            prefix = 4
        elif data[index : index + 3] == b"\x00\x00\x01":
            prefix = 3
        if prefix:
            header = index + prefix
            if header < size:
                types.add((data[header] >> 1) & 0x3F)
            index = header + 1
        else:
            index += 1
    return types


def copy_joint_payload(message: JointState) -> Dict[str, List[Any]]:
    return {
        "name": list(message.name),
        "position": [float(value) for value in message.position],
        "velocity": [float(value) for value in message.velocity],
        "effort": [float(value) for value in message.effort],
    }


def copy_tf_payload(message: TFMessage) -> List[Dict[str, Any]]:
    output = []
    for transform in message.transforms:
        output.append(
            {
                "parent_frame": transform.header.frame_id,
                "child_frame": transform.child_frame_id,
                "timestamp_ns": stamp_to_ns(transform.header.stamp),
                "translation_xyz": [
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
        )
    return output


class A2BridgeNode(Node):
    def __init__(self, specs: Dict[str, StreamSpec], runtime: RuntimeState):
        super().__init__("a2_grpc_bridge")
        self.specs = specs
        self.runtime = runtime
        # Node already exposes a read-only ``subscriptions`` property.
        # Keep explicit references under a private, non-conflicting name so
        # rclpy subscriptions are not garbage-collected.
        self._subscription_handles = []
        self._create_subscriptions()
        self.scheduler = self.create_timer(0.002, self._emit_sample_grids)

    @staticmethod
    def _sensor_qos() -> QoSProfile:
        return QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=20,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
        )

    @staticmethod
    def _tf_static_qos() -> QoSProfile:
        return QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )

    def _create_subscriptions(self) -> None:
        for spec in self.specs.values():
            if spec.kind == VIDEO_KIND:
                callback = lambda msg, name=spec.name: self._video_callback(name, msg)
                message_type = CompressedVideo
                qos = self._sensor_qos()
            elif spec.kind in JOINT_KINDS:
                callback = lambda msg, name=spec.name: self._joint_callback(name, msg)
                message_type = JointState
                qos = self._sensor_qos()
            elif spec.kind in {TF_KIND, TF_STATIC_KIND}:
                callback = lambda msg, name=spec.name: self._tf_callback(name, msg)
                message_type = TFMessage
                qos = (
                    self._tf_static_qos()
                    if spec.kind == TF_STATIC_KIND
                    else self._sensor_qos()
                )
            else:
                raise ValueError(f"unsupported stream kind: {spec.kind}")
            self._subscription_handles.append(
                self.create_subscription(message_type, spec.topic, callback, qos)
            )
            self.get_logger().info(f"subscribed {spec.name}: {spec.topic}")

    def _video_callback(self, name: str, message: CompressedVideo) -> None:
        spec = self.specs[name]
        receive_ns = time.time_ns()
        source_ns = source_or_receive(message.timestamp, receive_ns)
        data = bytes(message.data)
        nal_types = annexb_nal_types(data)
        envelope = a2_data_pb2.StreamEnvelope(
            stream_name=name,
            ros_type=spec.ros_type,
            sequence=self.runtime.next_sequence(name),
            source_timestamp_ns=source_ns,
            edge_receive_timestamp_ns=receive_ns,
            sample_timestamp_ns=source_ns,
            video=a2_data_pb2.VideoPacket(
                annexb=data,
                format=message.format or "h265",
                frame_id=message.frame_id,
                width=spec.width,
                height=spec.height,
                has_vps=32 in nal_types,
                has_sps=33 in nal_types,
                has_pps=34 in nal_types,
                is_irap=any(16 <= item <= 23 for item in nal_types),
            ),
        )
        with self.runtime.lock:
            self.runtime.received[name] += 1
            self.runtime.last_source_timestamp_ns[name] = source_ns
        self.runtime.publish(envelope)

    def _joint_callback(self, name: str, message: JointState) -> None:
        receive_ns = time.time_ns()
        source_ns = source_or_receive(message.header.stamp, receive_ns)
        self.runtime.update_latest(
            name,
            LatestSample(
                source_timestamp_ns=source_ns,
                edge_receive_timestamp_ns=receive_ns,
                payload=copy_joint_payload(message),
            ),
        )

    def _tf_callback(self, name: str, message: TFMessage) -> None:
        receive_ns = time.time_ns()
        transforms = copy_tf_payload(message)
        stamps = [item["timestamp_ns"] for item in transforms if item["timestamp_ns"] > 0]
        source_ns = max(stamps, default=receive_ns)
        spec = self.specs[name]
        sample = LatestSample(source_ns, receive_ns, transforms)
        if spec.kind == TF_STATIC_KIND:
            with self.runtime.lock:
                self.runtime.received[name] += 1
                self.runtime.last_source_timestamp_ns[name] = source_ns
            self.runtime.publish(self._tf_envelope(name, sample, source_ns))
        else:
            self.runtime.update_tf_latest(name, sample)

    def _joint_envelope(
        self, name: str, sample: LatestSample, sample_timestamp_ns: int
    ) -> Any:
        spec = self.specs[name]
        payload = sample.payload
        return a2_data_pb2.StreamEnvelope(
            stream_name=name,
            ros_type=spec.ros_type,
            sequence=self.runtime.next_sequence(name),
            source_timestamp_ns=sample.source_timestamp_ns,
            edge_receive_timestamp_ns=sample.edge_receive_timestamp_ns,
            sample_timestamp_ns=sample_timestamp_ns,
            joint=a2_data_pb2.JointSample(
                name=payload["name"],
                position=payload["position"],
                velocity=payload["velocity"],
                effort=payload["effort"],
            ),
        )

    def _tf_envelope(
        self, name: str, sample: LatestSample, sample_timestamp_ns: int
    ) -> Any:
        spec = self.specs[name]
        transforms = [
            a2_data_pb2.Transform(
                parent_frame=item["parent_frame"],
                child_frame=item["child_frame"],
                timestamp_ns=item["timestamp_ns"],
                translation_xyz=item["translation_xyz"],
                rotation_xyzw=item["rotation_xyzw"],
            )
            for item in sample.payload
        ]
        return a2_data_pb2.StreamEnvelope(
            stream_name=name,
            ros_type=spec.ros_type,
            sequence=self.runtime.next_sequence(name),
            source_timestamp_ns=sample.source_timestamp_ns,
            edge_receive_timestamp_ns=sample.edge_receive_timestamp_ns,
            sample_timestamp_ns=sample_timestamp_ns,
            tf=a2_data_pb2.TransformBatch(transforms=transforms),
        )

    def _emit_sample_grids(self) -> None:
        now_ns = self.get_clock().now().nanoseconds
        pending: List[Any] = []
        with self.runtime.lock:
            for name, spec in self.specs.items():
                if spec.kind in {VIDEO_KIND, TF_STATIC_KIND}:
                    continue
                sample = self.runtime.latest.get(name)
                next_ns = self.runtime.next_sample_ns.get(name)
                if sample is None or next_ns is None:
                    continue
                period_ns = round(1e9 / spec.target_hz)
                emitted = 0
                while next_ns <= now_ns and emitted < 5:
                    if spec.kind in JOINT_KINDS:
                        pending.append(self._joint_envelope(name, sample, next_ns))
                    elif spec.kind == TF_KIND:
                        pending.append(self._tf_envelope(name, sample, next_ns))
                    next_ns += period_ns
                    emitted += 1
                if next_ns <= now_ns:
                    skipped = (now_ns - next_ns) // period_ns + 1
                    next_ns += skipped * period_ns
                    self.runtime.dropped[name] += int(skipped)
                self.runtime.next_sample_ns[name] = next_ns
        for envelope in pending:
            self.runtime.publish(envelope)


class A2DataService(a2_data_pb2_grpc.A2DataServiceServicer):
    def __init__(
        self,
        runtime: RuntimeState,
        bridge_name: str,
        bridge_version: str,
        robot_id: str,
    ):
        self.runtime = runtime
        self.bridge_name = bridge_name
        self.bridge_version = bridge_version
        self.robot_id = robot_id

    def GetManifest(self, request: Any, context: grpc.ServicerContext) -> Any:
        descriptors = [
            a2_data_pb2.StreamDescriptor(
                stream_name=spec.name,
                ros_topic=spec.topic,
                ros_type=spec.ros_type,
                kind=spec.kind,
                target_hz=spec.target_hz,
                required=spec.required,
            )
            for spec in self.runtime.specs.values()
        ]
        return a2_data_pb2.StreamManifest(
            bridge_name=self.bridge_name,
            bridge_version=self.bridge_version,
            robot_id=self.robot_id,
            streams=descriptors,
        )

    def Subscribe(self, request: Any, context: grpc.ServicerContext) -> Iterable[Any]:
        names = list(request.stream_names) or list(self.runtime.specs)
        try:
            subscriber_id, output_queue = self.runtime.register_subscriber(names)
        except KeyError as exc:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, f"unknown streams: {exc}")
            return
        try:
            while context.is_active() and not self.runtime.shutdown_event.is_set():
                try:
                    yield output_queue.get(timeout=0.5)
                except queue.Empty:
                    continue
        finally:
            self.runtime.unregister_subscriber(subscriber_id)

    def WatchHealth(
        self, request: Any, context: grpc.ServicerContext
    ) -> Iterable[Any]:
        names = list(request.stream_names) or list(self.runtime.specs)
        unknown = set(names).difference(self.runtime.specs)
        if unknown:
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                f"unknown streams: {', '.join(sorted(unknown))}",
            )
            return
        while context.is_active() and not self.runtime.shutdown_event.wait(1.0):
            streams = []
            for name in names:
                ready, rate, received, dropped, last_source, detail = (
                    self.runtime.health(name)
                )
                streams.append(
                    a2_data_pb2.StreamHealth(
                        stream_name=name,
                        ready=ready,
                        rate_hz=rate,
                        received=received,
                        dropped=dropped,
                        last_source_timestamp_ns=last_source,
                        detail=detail,
                    )
                )
            yield a2_data_pb2.HealthSnapshot(
                timestamp_ns=time.time_ns(), streams=streams
            )


def load_config(path: Path) -> Tuple[Dict[str, Any], Dict[str, StreamSpec]]:
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    specs: Dict[str, StreamSpec] = {}
    for item in config["streams"]:
        spec = StreamSpec(
            name=str(item["name"]),
            topic=str(item["topic"]),
            ros_type=str(item["ros_type"]),
            kind=str(item["kind"]),
            target_hz=float(item["target_hz"]),
            required=bool(item["required"]),
            width=int(item.get("width", 0)),
            height=int(item.get("height", 0)),
        )
        if spec.name in specs:
            raise ValueError(f"duplicate stream name: {spec.name}")
        specs[spec.name] = spec
    return config, specs


def parse_args() -> argparse.Namespace:
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=base / "config" / "orin_bridge.json"
    )
    parser.add_argument("--listen", help="Override listen address, e.g. 0.0.0.0:50061")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config, specs = load_config(args.config)
    listen = str(args.listen or config["listen"])
    runtime = RuntimeState(
        specs=specs,
        health_window_seconds=float(config["health_window_seconds"]),
        subscriber_queue_size=int(config["subscriber_queue_size"]),
    )

    rclpy.init()
    node = A2BridgeNode(specs, runtime)
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)

    grpc_server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=max(16, len(specs) * 2)),
        options=[
            ("grpc.max_send_message_length", 64 * 1024 * 1024),
            ("grpc.keepalive_time_ms", 10_000),
            ("grpc.keepalive_timeout_ms", 5_000),
            ("grpc.keepalive_permit_without_calls", 1),
        ],
    )
    a2_data_pb2_grpc.add_A2DataServiceServicer_to_server(
        A2DataService(
            runtime=runtime,
            bridge_name=str(config["bridge_name"]),
            bridge_version=str(config["bridge_version"]),
            robot_id=str(config["robot_id"]),
        ),
        grpc_server,
    )
    if grpc_server.add_insecure_port(listen) == 0:
        raise RuntimeError(f"failed to bind gRPC listen address: {listen}")
    grpc_server.start()
    node.get_logger().info(f"A2 gRPC bridge listening on {listen}")

    stopping = threading.Event()

    def request_stop(signum: int, frame: Any) -> None:
        del signum, frame
        stopping.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    spin_thread = threading.Thread(target=executor.spin, name="ros2-executor")
    spin_thread.start()
    try:
        while not stopping.wait(0.5):
            pass
    finally:
        runtime.shutdown_event.set()
        grpc_server.stop(grace=2.0).wait(timeout=3.0)
        executor.shutdown(timeout_sec=3.0)
        spin_thread.join(timeout=3.0)
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
