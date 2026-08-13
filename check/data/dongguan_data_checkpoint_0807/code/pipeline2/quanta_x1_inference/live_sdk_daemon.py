"""Persistent x2robot SDK daemon for Quanta X1 live_runner (no per-step subprocess).

Runs in xr_lerobot venv. Keeps one ``x2robot.connect`` session and serves
capture / execute over length-prefixed JSON-RPC (default 127.0.0.1:15100).

Topology:
  dev PC:     tunnel 127.0.0.1:15051 -> 177 -> robot 246:50051
              daemon --server 127.0.0.1:15051
  DGX 177:    daemon --server 192.168.36.246:50051  (direct, no tunnel)
  live_runner --sdk-backend daemon --sdk-daemon-url 127.0.0.1:15100
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path
from typing import Any

_PIPELINE2 = Path(__file__).resolve().parents[1]
if str(_PIPELINE2) not in sys.path:
    sys.path.insert(0, str(_PIPELINE2))

from quanta_x1_inference.live_capture import (  # noqa: E402
    LIVE_ACK_TOKEN,
    connect_robot,
    run_capture_with_robot,
    run_execute_with_robot,
)
from quanta_x1_inference.live_sdk_rpc import JsonRpcClient, serve_forever  # noqa: E402

DAEMON_VERSION = "quanta_x1_live_sdk_daemon_v1"
DEFAULT_DAEMON_PORT = 15100


class LiveSdkDaemonState:
    def __init__(self, *, server: str) -> None:
        self.server = server
        self.lock = threading.Lock()
        self.configure_mode_done = False
        self.connect_ms: float | None = None
        print(f"Connecting to x2://{server} ...", flush=True)
        t0 = time.perf_counter()
        self.robot = connect_robot(server)
        self.connect_ms = (time.perf_counter() - t0) * 1000.0
        print(f"SDK connected in {self.connect_ms:.1f} ms", flush=True)

    def handle(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        if method == "ping":
            return {
                "daemon_version": DAEMON_VERSION,
                "server": self.server,
                "configure_mode_done": self.configure_mode_done,
                "connect_ms": self.connect_ms,
            }
        if method == "capture":
            output_dir = Path(params["output_dir"])
            with self.lock:
                report = run_capture_with_robot(
                    self.robot,
                    output_dir=output_dir,
                    server=self.server,
                )
            return {"capture_report": report}

        if method == "execute":
            if params.get("acknowledge") != LIVE_ACK_TOKEN:
                raise ValueError(f"execute requires acknowledge={LIVE_ACK_TOKEN!r}")
            joint_targets = params["joint_targets"]
            configure = bool(params.get("configure_mode", False))
            with self.lock:
                # Honor live_runner's configure_mode on each new run (robot may leave SDK mode).
                result = run_execute_with_robot(
                    self.robot,
                    joint_targets=joint_targets,
                    gripper_target=float(params["gripper_target"]),
                    settle_sec=float(params["settle_sec"]),
                    max_joint_delta_rad=float(params["max_joint_delta_rad"]),
                    configure_mode=configure,
                    server=self.server,
                )
                if configure:
                    self.configure_mode_done = True
            return {"execute_result": result}

        if method == "shutdown":
            return {"shutting_down": True}

        raise ValueError(f"unknown method: {method!r}")


class LiveSdkDaemonClient:
    """Client used by live_runner when --sdk-backend daemon."""

    def __init__(self, url: str) -> None:
        self.url = url
        self._client = JsonRpcClient(url, default_port=DEFAULT_DAEMON_PORT)

    def close(self) -> None:
        self._client.close()

    def ping(self) -> dict[str, Any]:
        return self._client.call("ping")

    def capture(self, *, output_dir: Path) -> tuple[dict[str, Any], str]:
        t0 = time.perf_counter()
        result = self._client.call("capture", {"output_dir": str(output_dir.resolve())})
        wall_ms = (time.perf_counter() - t0) * 1000.0
        report = result["capture_report"]
        daemon_ms = float(result.get("daemon_wall_ms", wall_ms))
        log = (
            f"live_sdk_daemon capture ok output_dir={output_dir.resolve()} "
            f"rpc_ms={wall_ms:.1f} daemon_wall_ms={daemon_ms:.1f}\n"
        )
        return report, log

    def execute(
        self,
        *,
        joint_targets: list[float],
        gripper_target: float,
        settle_sec: float,
        max_joint_delta_rad: float,
        configure_mode: bool,
        output_json: Path,
    ) -> tuple[dict[str, Any], str]:
        t0 = time.perf_counter()
        result = self._client.call(
            "execute",
            {
                "joint_targets": [float(x) for x in joint_targets],
                "gripper_target": float(gripper_target),
                "settle_sec": float(settle_sec),
                "max_joint_delta_rad": float(max_joint_delta_rad),
                "configure_mode": bool(configure_mode),
                "acknowledge": LIVE_ACK_TOKEN,
            },
        )
        wall_ms = (time.perf_counter() - t0) * 1000.0
        execute_result = result["execute_result"]
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(
            json.dumps(execute_result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        daemon_ms = float(result.get("daemon_wall_ms", wall_ms))
        log = (
            f"live_sdk_daemon execute ok output_json={output_json.resolve()} "
            f"rpc_ms={wall_ms:.1f} daemon_wall_ms={daemon_ms:.1f}\n"
        )
        return execute_result, log


def run_serve(args: argparse.Namespace) -> int:
    state = LiveSdkDaemonState(server=args.server)

    def handler(method: str, params: dict[str, Any]) -> dict[str, Any]:
        return state.handle(method, params)

    serve_forever(bind_host=args.bind, bind_port=args.port, handler=handler)
    return 0


def run_ping_cli(args: argparse.Namespace) -> int:
    client = LiveSdkDaemonClient(args.url)
    try:
        info = client.ping()
        print(json.dumps(info, indent=2))
    finally:
        client.close()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Quanta X1 persistent live SDK daemon.")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="Start daemon (xr_lerobot venv).")
    serve.add_argument(
        "--server",
        default="127.0.0.1:15051",
        help="Robot gRPC address. Dev PC: 127.0.0.1:15051 (tunnel). DGX 177: 192.168.36.246:50051",
    )
    serve.add_argument("--bind", default="127.0.0.1", help="Daemon listen address.")
    serve.add_argument("--port", type=int, default=DEFAULT_DAEMON_PORT)
    serve.set_defaults(func=run_serve)

    ping = sub.add_parser("ping", help="Health-check daemon.")
    ping.add_argument("--url", default=f"127.0.0.1:{DEFAULT_DAEMON_PORT}")
    ping.set_defaults(func=run_ping_cli)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
