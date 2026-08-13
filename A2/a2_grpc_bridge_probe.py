#!/usr/bin/env python3
"""Connectivity and stream-health probe for the A2 Orin gRPC bridge."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import grpc

from generated import a2_data_pb2
from generated import a2_data_pb2_grpc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default="192.168.2.50:50061")
    parser.add_argument("--seconds", type=float, default=10.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    options = [
        ("grpc.max_receive_message_length", 64 * 1024 * 1024),
        ("grpc.keepalive_time_ms", 10_000),
        ("grpc.keepalive_timeout_ms", 5_000),
        ("grpc.keepalive_permit_without_calls", 1),
    ]
    channel = grpc.insecure_channel(args.server, options=options)
    try:
        grpc.channel_ready_future(channel).result(timeout=5.0)
    except grpc.FutureTimeoutError:
        print(f"ERROR: gRPC server is not ready: {args.server}", file=sys.stderr)
        return 2

    stub = a2_data_pb2_grpc.A2DataServiceStub(channel)
    manifest = stub.GetManifest(a2_data_pb2.Empty(), timeout=5.0)
    print(
        json.dumps(
            {
                "bridge_name": manifest.bridge_name,
                "bridge_version": manifest.bridge_version,
                "robot_id": manifest.robot_id,
                "streams": [
                    {
                        "name": item.stream_name,
                        "topic": item.ros_topic,
                        "ros_type": item.ros_type,
                        "kind": item.kind,
                        "target_hz": item.target_hz,
                        "required": item.required,
                    }
                    for item in manifest.streams
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    expected = {item.stream_name for item in manifest.streams}
    seen = set()
    deadline = time.monotonic() + args.seconds
    call = stub.WatchHealth(a2_data_pb2.HealthRequest())
    try:
        for snapshot in call:
            report = {
                item.stream_name: {
                    "ready": item.ready,
                    "rate_hz": round(item.rate_hz, 3),
                    "received": int(item.received),
                    "dropped": int(item.dropped),
                    "last_source_timestamp_ns": int(
                        item.last_source_timestamp_ns
                    ),
                    "detail": item.detail,
                }
                for item in snapshot.streams
            }
            seen.update(
                name
                for name, item in report.items()
                if item["received"] > 0 or item["ready"]
            )
            print(json.dumps(report, ensure_ascii=False, sort_keys=True))
            if time.monotonic() >= deadline:
                break
    finally:
        call.cancel()
        channel.close()

    missing = sorted(expected.difference(seen))
    if missing:
        print(
            "ERROR: streams without received data: " + ", ".join(missing),
            file=sys.stderr,
        )
        return 3
    print("PASS: all manifest streams have received data")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
