"""Persistent x2robot SDK daemon for biman live_runner (32D / 3 cameras).

Default listen: 127.0.0.1:15101 (pipeline2 x1 daemon uses 15100).
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
PIPELINE2 = ROOT / "pipeline2"
PIPELINE3 = ROOT / "pipeline3_biman"
if str(PIPELINE2) not in sys.path:
    sys.path.insert(0, str(PIPELINE2))
if str(PIPELINE3) not in sys.path:
    sys.path.insert(0, str(PIPELINE3))

from quanta_biman_inference.constants import LIVE_ACK_TOKEN  # noqa: E402
from quanta_biman_inference.live_capture import (  # noqa: E402
    END_POSE_MAX_ATTEMPTS,
    END_POSE_PRE_DELAY_SEC,
    connect_robot,
    is_retryable_sdk_read_error,
    run_capture_with_robot,
    run_configure_joint_mode_with_robot,
    run_execute_end_pose_trajectory_with_robot,
    run_execute_trajectory_with_robot,
    run_execute_with_robot,
    run_set_dual_end_pose_with_robot,
    run_set_end_pose_with_robot,
    run_set_lift_with_robot,
)
from quanta_x1_inference.live_sdk_rpc import JsonRpcClient, serve_forever  # noqa: E402

DAEMON_VERSION = "quanta_biman_live_sdk_daemon_v7"
DEFAULT_DAEMON_PORT = 15101


class LiveSdkDaemonState:
    def __init__(
        self,
        *,
        server: str,
        end_pose_pre_delay_sec: float,
        end_pose_max_attempts: int,
    ) -> None:
        self.server = server
        self.end_pose_pre_delay_sec = end_pose_pre_delay_sec
        self.end_pose_max_attempts = end_pose_max_attempts
        self.lock = threading.Lock()
        self.configure_mode_done = False
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
                try:
                    report = run_capture_with_robot(
                        self.robot,
                        output_dir=output_dir,
                        server=self.server,
                        end_pose_pre_delay_sec=self.end_pose_pre_delay_sec,
                        end_pose_max_attempts=self.end_pose_max_attempts,
                    )
                except RuntimeError as exc:
                    # Long home / end_pose streams can reset the gRPC channel; reconnect once.
                    if not is_retryable_sdk_read_error(exc):
                        raise
                    print(
                        f"[live_sdk_daemon] capture failed ({exc!r}); reconnecting to "
                        f"x2://{self.server} and retrying once...",
                        flush=True,
                    )
                    try:
                        close = getattr(self.robot, "close", None)
                        if callable(close):
                            close()
                    except Exception:
                        pass
                    t0 = time.perf_counter()
                    self.robot = connect_robot(self.server)
                    self.connect_ms = (time.perf_counter() - t0) * 1000.0
                    print(
                        f"[live_sdk_daemon] reconnected in {self.connect_ms:.1f} ms",
                        flush=True,
                    )
                    report = run_capture_with_robot(
                        self.robot,
                        output_dir=output_dir,
                        server=self.server,
                        end_pose_pre_delay_sec=self.end_pose_pre_delay_sec,
                        end_pose_max_attempts=self.end_pose_max_attempts,
                    )
                    report["daemon_reconnect"] = True
            return {"capture_report": report}

        if method == "execute":
            if params.get("acknowledge") != LIVE_ACK_TOKEN:
                raise ValueError(f"execute requires acknowledge={LIVE_ACK_TOKEN!r}")
            arm = str(params["arm"]).lower()
            if arm not in {"left", "right"}:
                raise ValueError("--arm must be left or right")
            configure = bool(params.get("configure_mode", False))
            with self.lock:
                # Honor live_runner's configure_mode on each new run. Do not skip just
                # because a prior session already configured — the robot may have left SDK mode.
                result = run_execute_with_robot(
                    self.robot,
                    arm=arm,  # type: ignore[arg-type]
                    joint_targets=params["joint_targets"],
                    gripper_target=float(params["gripper_target"]),
                    settle_sec=float(params["settle_sec"]),
                    max_joint_delta_rad=float(params["max_joint_delta_rad"]),
                    configure_mode=configure,
                    server=self.server,
                )
                if configure:
                    self.configure_mode_done = True
            return {"execute_result": result}

        if method == "execute_trajectory":
            if params.get("acknowledge") != LIVE_ACK_TOKEN:
                raise ValueError(f"execute_trajectory requires acknowledge={LIVE_ACK_TOKEN!r}")
            arm = str(params["arm"]).lower()
            if arm not in {"left", "right"}:
                raise ValueError("--arm must be left or right")
            configure = bool(params.get("configure_mode", False))
            waypoints = params["waypoints"]
            policy_joints = [wp["joint_targets"] for wp in waypoints]
            policy_grippers = [float(wp["gripper_target"]) for wp in waypoints]
            with self.lock:
                result = run_execute_trajectory_with_robot(
                    self.robot,
                    arm=arm,  # type: ignore[arg-type]
                    policy_joints=policy_joints,
                    policy_grippers=policy_grippers,
                    control_hz=float(params["control_hz"]),
                    train_fps=float(params["train_fps"]),
                    max_joint_delta_rad=float(params["max_joint_delta_rad"]),
                    trajectory_settle_sec=float(params["trajectory_settle_sec"]),
                    configure_mode=configure,
                    server=self.server,
                )
                if configure:
                    self.configure_mode_done = True
            return {"execute_result": result}

        if method == "execute_end_pose_trajectory":
            if params.get("acknowledge") != LIVE_ACK_TOKEN:
                raise ValueError(
                    f"execute_end_pose_trajectory requires acknowledge={LIVE_ACK_TOKEN!r}"
                )
            arm = str(params["arm"]).lower()
            if arm not in {"left", "right"}:
                raise ValueError("--arm must be left or right")
            waypoints = params["waypoints"]
            if not isinstance(waypoints, list) or not waypoints:
                raise ValueError("execute_end_pose_trajectory requires non-empty waypoints")
            policy_end_poses = []
            policy_grippers = []
            for index, wp in enumerate(waypoints):
                if not isinstance(wp, dict) or not isinstance(wp.get("end_pose"), dict):
                    raise ValueError(f"waypoints[{index}] needs end_pose object")
                policy_end_poses.append(wp["end_pose"])
                policy_grippers.append(float(wp["gripper_target"]))
            configure = bool(params.get("configure_mode", False))
            with self.lock:
                result = run_execute_end_pose_trajectory_with_robot(
                    self.robot,
                    arm=arm,  # type: ignore[arg-type]
                    policy_end_poses=policy_end_poses,
                    policy_grippers=policy_grippers,
                    configure_mode=configure,
                    server=self.server,
                    interpolate_hz=float(params.get("interpolate_hz", 50.0)),
                    max_linear_speed_m_s=float(params.get("max_linear_speed_m_s", 0.08)),
                    max_step_m=float(params.get("max_step_m", 0.02)),
                    min_duration_sec=float(params.get("min_duration_sec", 0.0)),
                    train_fps=float(params.get("train_fps", 15.0)),
                    trajectory_settle_sec=float(params.get("trajectory_settle_sec", 0.0)),
                )
                if configure:
                    self.configure_mode_done = True
            return {"execute_end_pose_trajectory_result": result}

        if method == "set_lift":
            if params.get("acknowledge") != LIVE_ACK_TOKEN:
                raise ValueError(f"set_lift requires acknowledge={LIVE_ACK_TOKEN!r}")
            with self.lock:
                result = run_set_lift_with_robot(
                    self.robot,
                    position_m=float(params["position_m"]),
                    settle_sec=float(params.get("settle_sec", 1.0)),
                    configure_sdk_mode=bool(params.get("configure_sdk_mode", True)),
                    server=self.server,
                )
            return {"set_lift_result": result}

        if method == "set_end_pose":
            if params.get("acknowledge") != LIVE_ACK_TOKEN:
                raise ValueError(f"set_end_pose requires acknowledge={LIVE_ACK_TOKEN!r}")
            arm = str(params["arm"]).lower()
            if arm not in {"left", "right"}:
                raise ValueError("--arm must be left or right")
            end_pose = params.get("end_pose")
            if not isinstance(end_pose, dict):
                raise ValueError("set_end_pose requires end_pose object")
            configure = bool(params.get("configure_mode", False))
            with self.lock:
                result = run_set_end_pose_with_robot(
                    self.robot,
                    arm=arm,  # type: ignore[arg-type]
                    end_pose=end_pose,
                    gripper_target=float(params["gripper_target"]),
                    settle_sec=float(params.get("settle_sec", 1.0)),
                    configure_mode=configure,
                    server=self.server,
                    interpolate_hz=float(params.get("interpolate_hz", 10.0)),
                    max_linear_speed_m_s=float(params.get("max_linear_speed_m_s", 0.015)),
                    min_duration_sec=float(params.get("min_duration_sec", 5.0)),
                    max_step_m=float(params.get("max_step_m", 0.008)),
                )
                if configure:
                    self.configure_mode_done = True
            return {"set_end_pose_result": result}

        if method == "set_dual_end_pose":
            if params.get("acknowledge") != LIVE_ACK_TOKEN:
                raise ValueError(f"set_dual_end_pose requires acknowledge={LIVE_ACK_TOKEN!r}")
            left_end_pose = params.get("left_end_pose")
            right_end_pose = params.get("right_end_pose")
            if not isinstance(left_end_pose, dict) or not isinstance(right_end_pose, dict):
                raise ValueError("set_dual_end_pose requires left_end_pose and right_end_pose objects")
            configure = bool(params.get("configure_mode", False))
            with self.lock:
                result = run_set_dual_end_pose_with_robot(
                    self.robot,
                    left_end_pose=left_end_pose,
                    left_gripper_target=float(params["left_gripper_target"]),
                    right_end_pose=right_end_pose,
                    right_gripper_target=float(params["right_gripper_target"]),
                    settle_sec=float(params.get("settle_sec", 1.0)),
                    configure_mode=configure,
                    server=self.server,
                    interpolate_hz=float(params.get("interpolate_hz", 10.0)),
                    max_linear_speed_m_s=float(params.get("max_linear_speed_m_s", 0.015)),
                    min_duration_sec=float(params.get("min_duration_sec", 5.0)),
                    max_step_m=float(params.get("max_step_m", 0.008)),
                )
                if configure:
                    self.configure_mode_done = True
            return {"set_dual_end_pose_result": result}

        if method == "configure_joint_mode":
            if params.get("acknowledge") != LIVE_ACK_TOKEN:
                raise ValueError(f"configure_joint_mode requires acknowledge={LIVE_ACK_TOKEN!r}")
            with self.lock:
                result = run_configure_joint_mode_with_robot(
                    self.robot,
                    server=self.server,
                )
                self.configure_mode_done = True
            return {"configure_joint_mode_result": result}

        if method == "shutdown":
            return {"shutting_down": True}

        raise ValueError(f"unknown method: {method!r}")


class LiveSdkDaemonClient:
    def __init__(self, url: str) -> None:
        self.url = url
        # Home end_pose interpolation can take tens of seconds per arm.
        self._client = JsonRpcClient(url, default_port=DEFAULT_DAEMON_PORT, timeout_sec=300.0)

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
            f"biman live_sdk_daemon capture ok output_dir={output_dir.resolve()} "
            f"rpc_ms={wall_ms:.1f} daemon_wall_ms={daemon_ms:.1f}\n"
        )
        return report, log

    def execute(
        self,
        *,
        arm: str,
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
                "arm": arm,
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
            f"biman live_sdk_daemon execute ok arm={arm} output_json={output_json.resolve()} "
            f"rpc_ms={wall_ms:.1f} daemon_wall_ms={daemon_ms:.1f}\n"
        )
        return execute_result, log

    def execute_trajectory(
        self,
        *,
        arm: str,
        waypoints: list[dict[str, Any]],
        control_hz: float,
        train_fps: float,
        trajectory_settle_sec: float,
        max_joint_delta_rad: float,
        configure_mode: bool,
        output_json: Path,
    ) -> tuple[dict[str, Any], str]:
        t0 = time.perf_counter()
        result = self._client.call(
            "execute_trajectory",
            {
                "arm": arm,
                "waypoints": waypoints,
                "control_hz": float(control_hz),
                "train_fps": float(train_fps),
                "trajectory_settle_sec": float(trajectory_settle_sec),
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
            f"biman live_sdk_daemon execute_trajectory ok arm={arm} "
            f"output_json={output_json.resolve()} rpc_ms={wall_ms:.1f} "
            f"daemon_wall_ms={daemon_ms:.1f} substeps={execute_result.get('dense_substeps')}\n"
        )
        return execute_result, log

    def execute_end_pose_trajectory(
        self,
        *,
        arm: str,
        waypoints: list[dict[str, Any]],
        interpolate_hz: float,
        max_linear_speed_m_s: float,
        max_step_m: float,
        min_duration_sec: float,
        train_fps: float,
        trajectory_settle_sec: float,
        configure_mode: bool,
        output_json: Path,
    ) -> tuple[dict[str, Any], str]:
        t0 = time.perf_counter()
        result = self._client.call(
            "execute_end_pose_trajectory",
            {
                "arm": arm,
                "waypoints": waypoints,
                "interpolate_hz": float(interpolate_hz),
                "max_linear_speed_m_s": float(max_linear_speed_m_s),
                "max_step_m": float(max_step_m),
                "min_duration_sec": float(min_duration_sec),
                "train_fps": float(train_fps),
                "trajectory_settle_sec": float(trajectory_settle_sec),
                "configure_mode": bool(configure_mode),
                "acknowledge": LIVE_ACK_TOKEN,
            },
        )
        wall_ms = (time.perf_counter() - t0) * 1000.0
        traj_result = result["execute_end_pose_trajectory_result"]
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(
            json.dumps(traj_result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        daemon_ms = float(result.get("daemon_wall_ms", wall_ms))
        log = (
            f"biman live_sdk_daemon execute_end_pose_trajectory ok arm={arm} "
            f"policy_wp={traj_result.get('policy_waypoints')} "
            f"dense={traj_result.get('dense_waypoints')} "
            f"wall_sec={traj_result.get('wall_sec')} "
            f"pos_err_m={traj_result.get('position_error_m')} "
            f"output_json={output_json.resolve()} rpc_ms={wall_ms:.1f} "
            f"daemon_wall_ms={daemon_ms:.1f}\n"
        )
        return traj_result, log

    def set_lift(
        self,
        *,
        position_m: float,
        settle_sec: float,
        configure_sdk_mode: bool,
        output_json: Path,
    ) -> tuple[dict[str, Any], str]:
        t0 = time.perf_counter()
        result = self._client.call(
            "set_lift",
            {
                "position_m": float(position_m),
                "settle_sec": float(settle_sec),
                "configure_sdk_mode": bool(configure_sdk_mode),
                "acknowledge": LIVE_ACK_TOKEN,
            },
        )
        wall_ms = (time.perf_counter() - t0) * 1000.0
        set_lift_result = result["set_lift_result"]
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(
            json.dumps(set_lift_result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        daemon_ms = float(result.get("daemon_wall_ms", wall_ms))
        log = (
            f"biman live_sdk_daemon set_lift ok position_m={position_m:.4f} "
            f"output_json={output_json.resolve()} rpc_ms={wall_ms:.1f} "
            f"daemon_wall_ms={daemon_ms:.1f}\n"
        )
        return set_lift_result, log

    def set_end_pose(
        self,
        *,
        arm: str,
        end_pose: dict[str, Any],
        gripper_target: float,
        settle_sec: float,
        configure_mode: bool,
        output_json: Path,
        interpolate_hz: float = 10.0,
        max_linear_speed_m_s: float = 0.015,
        min_duration_sec: float = 5.0,
        max_step_m: float = 0.008,
    ) -> tuple[dict[str, Any], str]:
        t0 = time.perf_counter()
        result = self._client.call(
            "set_end_pose",
            {
                "arm": arm,
                "end_pose": end_pose,
                "gripper_target": float(gripper_target),
                "settle_sec": float(settle_sec),
                "configure_mode": bool(configure_mode),
                "interpolate_hz": float(interpolate_hz),
                "max_linear_speed_m_s": float(max_linear_speed_m_s),
                "min_duration_sec": float(min_duration_sec),
                "max_step_m": float(max_step_m),
                "acknowledge": LIVE_ACK_TOKEN,
            },
        )
        wall_ms = (time.perf_counter() - t0) * 1000.0
        set_end_pose_result = result["set_end_pose_result"]
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(
            json.dumps(set_end_pose_result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        daemon_ms = float(result.get("daemon_wall_ms", wall_ms))
        log = (
            f"biman live_sdk_daemon set_end_pose ok arm={arm} "
            f"waypoints={set_end_pose_result.get('dense_waypoints')} "
            f"wall_sec={set_end_pose_result.get('wall_sec')} "
            f"pos_err_m={set_end_pose_result.get('position_error_m')} "
            f"output_json={output_json.resolve()} rpc_ms={wall_ms:.1f} "
            f"daemon_wall_ms={daemon_ms:.1f}\n"
        )
        return set_end_pose_result, log

    def set_dual_end_pose(
        self,
        *,
        left_end_pose: dict[str, Any],
        left_gripper_target: float,
        right_end_pose: dict[str, Any],
        right_gripper_target: float,
        settle_sec: float,
        configure_mode: bool,
        output_json: Path,
        interpolate_hz: float = 10.0,
        max_linear_speed_m_s: float = 0.015,
        min_duration_sec: float = 5.0,
        max_step_m: float = 0.008,
    ) -> tuple[dict[str, Any], str]:
        t0 = time.perf_counter()
        result = self._client.call(
            "set_dual_end_pose",
            {
                "left_end_pose": left_end_pose,
                "left_gripper_target": float(left_gripper_target),
                "right_end_pose": right_end_pose,
                "right_gripper_target": float(right_gripper_target),
                "settle_sec": float(settle_sec),
                "configure_mode": bool(configure_mode),
                "interpolate_hz": float(interpolate_hz),
                "max_linear_speed_m_s": float(max_linear_speed_m_s),
                "min_duration_sec": float(min_duration_sec),
                "max_step_m": float(max_step_m),
                "acknowledge": LIVE_ACK_TOKEN,
            },
        )
        wall_ms = (time.perf_counter() - t0) * 1000.0
        dual_result = result["set_dual_end_pose_result"]
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(
            json.dumps(dual_result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        daemon_ms = float(result.get("daemon_wall_ms", wall_ms))
        log = (
            f"biman live_sdk_daemon set_dual_end_pose ok "
            f"waypoints={dual_result.get('dense_waypoints')} "
            f"wall_sec={dual_result.get('wall_sec')} "
            f"left_err={dual_result.get('left_position_error_m')} "
            f"right_err={dual_result.get('right_position_error_m')} "
            f"output_json={output_json.resolve()} rpc_ms={wall_ms:.1f} "
            f"daemon_wall_ms={daemon_ms:.1f}\n"
        )
        return dual_result, log

    def configure_joint_mode(
        self,
        *,
        output_json: Path,
    ) -> tuple[dict[str, Any], str]:
        t0 = time.perf_counter()
        result = self._client.call(
            "configure_joint_mode",
            {"acknowledge": LIVE_ACK_TOKEN},
        )
        wall_ms = (time.perf_counter() - t0) * 1000.0
        cfg_result = result["configure_joint_mode_result"]
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(
            json.dumps(cfg_result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        daemon_ms = float(result.get("daemon_wall_ms", wall_ms))
        log = (
            f"biman live_sdk_daemon configure_joint_mode ok "
            f"output_json={output_json.resolve()} rpc_ms={wall_ms:.1f} "
            f"daemon_wall_ms={daemon_ms:.1f}\n"
        )
        return cfg_result, log


def run_serve(args: argparse.Namespace) -> int:
    state = LiveSdkDaemonState(
        server=args.server,
        end_pose_pre_delay_sec=float(args.end_pose_pre_delay_sec),
        end_pose_max_attempts=int(args.end_pose_max_attempts),
    )

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
    parser = argparse.ArgumentParser(description="Quanta biman persistent live SDK daemon.")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="Start daemon (xr_lerobot venv).")
    serve.add_argument("--server", default="127.0.0.1:15051")
    serve.add_argument("--bind", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=DEFAULT_DAEMON_PORT)
    serve.add_argument("--end-pose-pre-delay-sec", type=float, default=END_POSE_PRE_DELAY_SEC)
    serve.add_argument("--end-pose-max-attempts", type=int, default=END_POSE_MAX_ATTEMPTS)
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
