"""Read-only live capture and optional right-arm execution via x2robot SDK.

Runs in the ``xr_lerobot`` Python (``x2robot`` is not installed in the GR00T venv).
``live_runner.py`` invokes this module as a subprocess for capture / actuation.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

_PIPELINE2 = Path(__file__).resolve().parents[1]
if str(_PIPELINE2) not in sys.path:
    sys.path.insert(0, str(_PIPELINE2))

from lerobot_export_utils import end_pose_to_eef_9d  # noqa: E402
from quat_rot6d_utils import add_rot6d_to_end_pose  # noqa: E402
from quanta_x1_inference.action_decode import (  # noqa: E402
    SDK_RIGHT_ARM_LOWER,
    SDK_RIGHT_ARM_UPPER,
    clip_joints_to_sdk,
)

CAPTURE_VERSION = "quanta_x1_live_capture_v1"
EXECUTE_VERSION = "quanta_x1_live_execute_v1"
LIVE_ACK_TOKEN = "QUANTA_X1_16D_LIVE"


@dataclass(frozen=True)
class LiveComponents:
    head_camera: np.ndarray
    right_arm_camera: np.ndarray
    eef_9d: np.ndarray
    gripper_position: float
    joint_position: np.ndarray

    def to_dict(self) -> dict[str, Any]:
        return {
            "head_camera_shape": list(self.head_camera.shape),
            "right_arm_camera_shape": list(self.right_arm_camera.shape),
            "eef_9d": self.eef_9d.astype(float).tolist(),
            "gripper_position": float(self.gripper_position),
            "joint_position": self.joint_position.astype(float).tolist(),
        }


def clean_env() -> dict[str, str]:
    env = os.environ.copy()
    for key in (
        "http_proxy",
        "https_proxy",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "all_proxy",
        "ALL_PROXY",
    ):
        env.pop(key, None)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(_PIPELINE2), env.get("PYTHONPATH", "")]
    ).strip(os.pathsep)
    return env


def resolve_robot_python(path: Path | None = None) -> Path:
    if path is not None:
        candidate = Path(path)
        if not candidate.is_file():
            raise FileNotFoundError(f"--robot-python not found: {candidate}")
        return candidate

    default = Path("/home/ubuntu/anaconda3/envs/xr_lerobot/bin/python")
    if default.is_file():
        return default
    raise FileNotFoundError(
        "Could not find xr_lerobot python. Pass --robot-python explicitly."
    )


def stamp_to_dict(msg: Any) -> dict[str, Any]:
    header = getattr(msg, "header", None)
    stamp = getattr(header, "stamp", None)
    if stamp is None:
        return {
            "sec": None,
            "nanosec": None,
            "timestamp_sec": None,
            "frame_id": "",
        }
    sec = int(getattr(stamp, "sec", 0))
    nanosec = int(getattr(stamp, "nanosec", 0))
    return {
        "sec": sec,
        "nanosec": nanosec,
        "timestamp_sec": sec + nanosec * 1e-9,
        "frame_id": str(getattr(header, "frame_id", "")),
    }


def timed_call(name: str, fn: Any) -> tuple[Any, dict[str, Any]]:
    start_monotonic = time.monotonic()
    start_epoch = time.time()
    try:
        value = fn()
    except Exception as exc:
        elapsed_ms = (time.monotonic() - start_monotonic) * 1000.0
        raise RuntimeError(f"{name} failed after {elapsed_ms:.3f} ms: {exc!r}") from exc
    end_monotonic = time.monotonic()
    end_epoch = time.time()
    return value, {
        "request_start_monotonic_sec": float(start_monotonic),
        "request_end_monotonic_sec": float(end_monotonic),
        "request_start_epoch_sec": float(start_epoch),
        "request_end_epoch_sec": float(end_epoch),
        "rpc_elapsed_ms": float((end_monotonic - start_monotonic) * 1000.0),
    }


def decode_jpeg_to_rgb(msg: Any, *, name: str) -> np.ndarray:
    jpeg_bytes = bytes(getattr(msg, "data", b""))
    if not jpeg_bytes:
        raise RuntimeError(f"{name}: empty JPEG payload")
    encoded = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"{name}: OpenCV JPEG decode failed")
    return np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def sdk_end_pose_to_dict(msg: Any) -> dict[str, Any]:
    pose = getattr(msg, "pose", msg)
    position = getattr(pose, "position", pose)
    orientation = getattr(pose, "orientation", None)
    if orientation is None:
        raise RuntimeError("SDK end_pose missing orientation")
    return {
        "position": {
            "x": float(getattr(position, "x", 0.0)),
            "y": float(getattr(position, "y", 0.0)),
            "z": float(getattr(position, "z", 0.0)),
        },
        "orientation": {
            "x": float(getattr(orientation, "x", 0.0)),
            "y": float(getattr(orientation, "y", 0.0)),
            "z": float(getattr(orientation, "z", 0.0)),
            "w": float(getattr(orientation, "w", 0.0)),
        },
    }


def joint_state_to_dict(msg: Any) -> dict[str, Any]:
    return {
        "name": list(getattr(msg, "name", [])),
        "position": [float(x) for x in getattr(msg, "position", [])],
        "velocity": [float(x) for x in getattr(msg, "velocity", [])],
        "effort": [float(x) for x in getattr(msg, "effort", [])],
    }


def gripper_position_value(msg: Any) -> float:
    for attr in ("position", "value", "opening", "current_position"):
        if hasattr(msg, attr):
            return float(getattr(msg, attr))
    raise RuntimeError(f"Could not read gripper position from {type(msg).__name__}")


def end_pose_to_eef_9d_from_sdk(msg: Any) -> np.ndarray:
    end_pose = add_rot6d_to_end_pose(sdk_end_pose_to_dict(msg))
    return end_pose_to_eef_9d(end_pose)


def connect_robot(server: str) -> Any:
    from x2robot import connect

    return connect(f"x2://{server}")


def capture_live_components(robot: Any) -> tuple[LiveComponents, dict[str, Any]]:
    readonly_calls = {
        "head_rgb": robot.head_camera.get_rgb_image,
        "right_arm_rgb": robot.right_arm_camera.get_raw_image,
        "right_arm_state": robot.right_arm.get_joint_states,
        "right_gripper_position": robot.right_gripper.get_position,
        "right_end_pose": robot.right_arm.get_end_pose,
    }

    whole_start = time.monotonic()
    values: dict[str, Any] = {}
    timing: dict[str, dict[str, Any]] = {}

    with ThreadPoolExecutor(max_workers=len(readonly_calls)) as executor:
        futures = {
            executor.submit(timed_call, name, fn): name
            for name, fn in readonly_calls.items()
        }
        for future in as_completed(futures):
            name = futures[future]
            value, item_timing = future.result()
            values[name] = value
            timing[name] = item_timing

    whole_elapsed_ms = float((time.monotonic() - whole_start) * 1000.0)

    head_rgb = decode_jpeg_to_rgb(values["head_rgb"], name="head_rgb")
    right_rgb = decode_jpeg_to_rgb(values["right_arm_rgb"], name="right_arm_rgb")
    right_arm = joint_state_to_dict(values["right_arm_state"])
    joint_position = np.asarray(right_arm["position"], dtype=np.float32)
    if joint_position.shape != (6,):
        raise ValueError(f"right_arm joint count {joint_position.shape}, expected (6,)")

    gripper_position = gripper_position_value(values["right_gripper_position"])
    eef_9d = end_pose_to_eef_9d_from_sdk(values["right_end_pose"])

    components = LiveComponents(
        head_camera=head_rgb,
        right_arm_camera=right_rgb,
        eef_9d=eef_9d,
        gripper_position=float(gripper_position),
        joint_position=joint_position,
    )

    meta = {
        "capture_version": CAPTURE_VERSION,
        "capture_mode": "five_readonly_rpcs_concurrent",
        "whole_request_wall_elapsed_ms": whole_elapsed_ms,
        "right_arm": right_arm,
        "right_gripper_raw_sdk_position": float(gripper_position),
        "right_arm_end_pose": add_rot6d_to_end_pose(
            sdk_end_pose_to_dict(values["right_end_pose"])
        ),
        "camera_timestamps": {
            "head_rgb": stamp_to_dict(values["head_rgb"]),
            "right_arm_rgb": stamp_to_dict(values["right_arm_rgb"]),
        },
        "per_rpc": timing,
        "safety_statement": (
            "Read-only capture. No mode switch, no arm command, "
            "no gripper command, no reset/homing/stop/emergency."
        ),
    }
    return components, meta


def save_capture_artifacts(
    components: LiveComponents,
    meta: dict[str, Any],
    *,
    output_dir: Path,
    server: str,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    head_path = output_dir / "head_camera.jpg"
    wrist_path = output_dir / "right_arm_camera.jpg"
    cv2.imwrite(str(head_path), cv2.cvtColor(components.head_camera, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(wrist_path), cv2.cvtColor(components.right_arm_camera, cv2.COLOR_RGB2BGR))

    report = {
        **meta,
        "captured_at": datetime.now().isoformat(),
        "server": server,
        "image_paths": {
            "head_camera": str(head_path.resolve()),
            "right_arm_camera": str(wrist_path.resolve()),
        },
        "components": components.to_dict(),
    }
    report_path = output_dir / "capture.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report["capture_json"] = str(report_path.resolve())
    return report


def read_right_arm(robot: Any) -> np.ndarray:
    state = robot.right_arm.get_joint_states()
    joints = np.asarray(list(state.position), dtype=np.float32)
    if joints.shape != (6,):
        raise ValueError(f"right_arm shape {joints.shape}, expected (6,)")
    return joints


def read_right_gripper(robot: Any) -> float:
    return gripper_position_value(robot.right_gripper.get_position())


def configure_joint_position_mode(robot: Any) -> None:
    from x2robot.sdk import (
        ManipulatorControlMode,
        ManipulatorControlModeParam,
        RobotModeParam,
        RobotWorkMode,
    )

    robot.system.set_work_mode(RobotModeParam(mode=RobotWorkMode.SDK))
    robot.robot_control.set_manipulator_control_mode(
        ManipulatorControlModeParam(mode=ManipulatorControlMode.MANIPULATOR_JOINT_POSITIONS)
    )


def bound_absolute_target(
    current: np.ndarray,
    target: np.ndarray,
    *,
    max_joint_delta_rad: float,
) -> tuple[np.ndarray, np.ndarray]:
    current = np.asarray(current, dtype=np.float32).reshape(6)
    target = np.asarray(target, dtype=np.float32).reshape(6)
    clipped = clip_joints_to_sdk(target)
    delta = clipped - current
    if max_joint_delta_rad > 0:
        delta = np.clip(delta, -max_joint_delta_rad, max_joint_delta_rad)
    command = clip_joints_to_sdk(current + delta)
    actual_delta = command - current
    return command, actual_delta


def execute_absolute_step(
    robot: Any,
    *,
    joint_targets: Sequence[float] | np.ndarray,
    gripper_target: float,
    settle_sec: float,
    max_joint_delta_rad: float,
    configure_mode: bool,
) -> dict[str, Any]:
    from x2robot.sdk import GripperPosition, JointPositions

    before_joints = read_right_arm(robot)
    before_gripper = read_right_gripper(robot)

    sdk_target, actual_delta = bound_absolute_target(
        before_joints,
        np.asarray(joint_targets, dtype=np.float32),
        max_joint_delta_rad=max_joint_delta_rad,
    )

    if configure_mode:
        configure_joint_position_mode(robot)

    robot.right_arm.set_joint_positions(
        JointPositions(positions=sdk_target.astype(float).tolist())
    )
    robot.right_gripper.set_position(GripperPosition(position=float(gripper_target)))

    if settle_sec > 0:
        time.sleep(settle_sec)

    after_joints = read_right_arm(robot)
    after_gripper = read_right_gripper(robot)
    observed_joint_delta = after_joints - before_joints
    tracking_error = observed_joint_delta - actual_delta

    return {
        "execute_version": EXECUTE_VERSION,
        "actuation_scope": "right_arm_and_gripper_only",
        "before_joints_rad": before_joints.astype(float).tolist(),
        "before_gripper": float(before_gripper),
        "policy_joint_targets_rad": np.asarray(joint_targets, dtype=np.float32).astype(float).tolist(),
        "sdk_joint_targets_rad": sdk_target.astype(float).tolist(),
        "command_joint_delta_rad": actual_delta.astype(float).tolist(),
        "gripper_target": float(gripper_target),
        "after_joints_rad": after_joints.astype(float).tolist(),
        "after_gripper": float(after_gripper),
        "observed_joint_delta_rad": observed_joint_delta.astype(float).tolist(),
        "tracking_error_rad": tracking_error.astype(float).tolist(),
        "max_abs_command_delta_rad": float(np.max(np.abs(actual_delta))),
        "max_abs_observed_delta_rad": float(np.max(np.abs(observed_joint_delta))),
        "max_abs_tracking_error_rad": float(np.max(np.abs(tracking_error))),
        "sdk_joint_limits_lower_rad": SDK_RIGHT_ARM_LOWER.astype(float).tolist(),
        "sdk_joint_limits_upper_rad": SDK_RIGHT_ARM_UPPER.astype(float).tolist(),
        "settle_sec": float(settle_sec),
        "max_joint_delta_rad": float(max_joint_delta_rad),
    }


def run_capture_with_robot(
    robot: Any,
    *,
    output_dir: Path,
    server: str,
) -> dict[str, Any]:
    """Capture with an existing SDK connection (daemon / in-process reuse)."""
    components, meta = capture_live_components(robot)
    return save_capture_artifacts(
        components,
        meta,
        output_dir=output_dir,
        server=server,
    )


def run_execute_with_robot(
    robot: Any,
    *,
    joint_targets: Sequence[float] | np.ndarray,
    gripper_target: float,
    settle_sec: float,
    max_joint_delta_rad: float,
    configure_mode: bool,
    server: str,
) -> dict[str, Any]:
    """Execute one step with an existing SDK connection."""
    result = execute_absolute_step(
        robot,
        joint_targets=joint_targets,
        gripper_target=gripper_target,
        settle_sec=settle_sec,
        max_joint_delta_rad=max_joint_delta_rad,
        configure_mode=configure_mode,
    )
    result["executed_at"] = datetime.now().isoformat()
    result["server"] = server
    return result


def run_capture_cli(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    robot = connect_robot(args.server)
    report = run_capture_with_robot(
        robot,
        output_dir=output_dir,
        server=args.server,
    )
    print(f"capture_dir={output_dir.resolve()}")
    print(f"capture_json={report['capture_json']}")
    return 0


def run_execute_cli(args: argparse.Namespace) -> int:
    if not args.allow_motion:
        raise SystemExit("--allow-motion is required for execute mode")
    if args.acknowledge != LIVE_ACK_TOKEN:
        raise SystemExit(f"--acknowledge must be exactly {LIVE_ACK_TOKEN!r}")

    joint_targets = json.loads(args.joint_targets_json)
    if not isinstance(joint_targets, list) or len(joint_targets) != 6:
        raise SystemExit("--joint-targets-json must be a JSON list of 6 floats")

    robot = connect_robot(args.server)
    result = run_execute_with_robot(
        robot,
        joint_targets=joint_targets,
        gripper_target=float(args.gripper_target),
        settle_sec=float(args.settle_sec),
        max_joint_delta_rad=float(args.max_joint_delta_rad),
        configure_mode=bool(args.configure_mode),
        server=args.server,
    )

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"execute_json={output_json.resolve()}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Quanta X1 live SDK capture / execute helper.")
    sub = parser.add_subparsers(dest="command", required=True)

    capture = sub.add_parser("capture", help="Read-only concurrent capture.")
    capture.add_argument("--server", default="127.0.0.1:15051")
    capture.add_argument("--output-dir", type=Path, required=True)
    capture.set_defaults(func=run_capture_cli)

    execute = sub.add_parser("execute", help="Send one absolute right-arm + gripper command.")
    execute.add_argument("--server", default="127.0.0.1:15051")
    execute.add_argument("--joint-targets-json", required=True)
    execute.add_argument("--gripper-target", type=float, required=True)
    execute.add_argument("--output-json", type=Path, required=True)
    execute.add_argument("--settle-sec", type=float, default=1.5)
    execute.add_argument("--max-joint-delta-rad", type=float, default=0.05)
    execute.add_argument("--configure-mode", action="store_true")
    execute.add_argument("--allow-motion", action="store_true")
    execute.add_argument("--acknowledge", default="")
    execute.set_defaults(func=run_execute_cli)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
