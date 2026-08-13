#!/usr/bin/env python3
"""Relay A2 motion and TF topics from the robot x86 network to Spark.

This node subscribes to the original robot topics but publishes only under
the /a2/relay namespace. It never publishes to /motion/control, so it cannot
enter the robot control path.
"""

from __future__ import annotations

import signal
import threading
import time
from dataclasses import dataclass
from typing import Any

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from sensor_msgs.msg import JointState
from tf2_msgs.msg import TFMessage


@dataclass(frozen=True)
class RelaySpec:
    source: str
    destination: str
    message_type: type
    transient_local: bool = False


RELAY_SPECS = [
    RelaySpec(
        "/motion/control/arm_joint_state",
        "/a2/relay/motion/control/arm_joint_state",
        JointState,
    ),
    RelaySpec(
        "/motion/control/hand_joint_state",
        "/a2/relay/motion/control/hand_joint_state",
        JointState,
    ),
    RelaySpec(
        "/motion/control/neck_joint_state",
        "/a2/relay/motion/control/neck_joint_state",
        JointState,
    ),
    RelaySpec(
        "/motion/control/arm_joint_command",
        "/a2/relay/motion/control/arm_joint_command",
        JointState,
    ),
    RelaySpec(
        "/motion/control/hand_joint_command",
        "/a2/relay/motion/control/hand_joint_command",
        JointState,
    ),
    RelaySpec(
        "/motion/control/neck_joint_command",
        "/a2/relay/motion/control/neck_joint_command",
        JointState,
    ),
    RelaySpec("/tf", "/a2/relay/tf", TFMessage),
    RelaySpec(
        "/tf_static",
        "/a2/relay/tf_static",
        TFMessage,
        transient_local=True,
    ),
]


class A2MotionRelay(Node):
    def __init__(self) -> None:
        super().__init__("a2_orin_ros2_motion_relay")
        self._lock = threading.Lock()
        self._counts = {spec.destination: 0 for spec in RELAY_SPECS}
        self._last_report_ns = time.monotonic_ns()
        self._publisher_handles = []
        self._subscription_handles = []

        normal_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=200,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        static_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        for spec in RELAY_SPECS:
            qos = static_qos if spec.transient_local else normal_qos
            publisher = self.create_publisher(
                spec.message_type,
                spec.destination,
                qos,
            )
            subscription = self.create_subscription(
                spec.message_type,
                spec.source,
                self._make_callback(spec, publisher),
                qos,
            )
            self._publisher_handles.append(publisher)
            self._subscription_handles.append(subscription)
            self.get_logger().info(
                f"relay {spec.source} -> {spec.destination}"
            )

        self._report_timer = self.create_timer(2.0, self._report_rates)

    def _make_callback(self, spec: RelaySpec, publisher: Any):
        def callback(message: Any) -> None:
            publisher.publish(message)
            with self._lock:
                self._counts[spec.destination] += 1

        return callback

    def _report_rates(self) -> None:
        now_ns = time.monotonic_ns()
        with self._lock:
            elapsed_s = max((now_ns - self._last_report_ns) / 1e9, 1e-6)
            counts = dict(self._counts)
            for topic in self._counts:
                self._counts[topic] = 0
            self._last_report_ns = now_ns

        summary = " ".join(
            f"{topic.rsplit('/', 1)[-1]}={count / elapsed_s:.1f}Hz"
            for topic, count in counts.items()
        )
        self.get_logger().info(summary)


def main() -> int:
    rclpy.init()
    node = A2MotionRelay()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    stopping = threading.Event()

    def request_stop(_signum: int, _frame: Any) -> None:
        stopping.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    try:
        while rclpy.ok() and not stopping.is_set():
            executor.spin_once(timeout_sec=0.2)
    finally:
        executor.remove_node(node)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
