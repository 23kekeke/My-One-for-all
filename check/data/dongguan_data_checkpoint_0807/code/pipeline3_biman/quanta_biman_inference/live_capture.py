"""Read-only biman capture and optional left/right arm execution via x2robot SDK.

Runs in the ``xr_lerobot`` Python. ``live_runner.py`` invokes this module as a subprocess.
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
from typing import Any, Literal, Sequence

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PIPELINE2 = ROOT / "pipeline2"
PIPELINE3 = ROOT / "pipeline3_biman"
if str(PIPELINE2) not in sys.path:
    sys.path.insert(0, str(PIPELINE2))
if str(PIPELINE3) not in sys.path:
    sys.path.insert(0, str(PIPELINE3))

from lerobot_export_utils import end_pose_to_eef_9d  # noqa: E402
from quat_rot6d_utils import add_rot6d_to_end_pose  # noqa: E402
from quanta_biman_inference.sdk_joint_limits import (  # noqa: E402
    SDK_LEFT_ARM_LOWER,
    SDK_LEFT_ARM_UPPER,
    SDK_RIGHT_ARM_LOWER,
    SDK_RIGHT_ARM_UPPER,
    clip_joints_to_sdk,
)
from quanta_biman_inference.constants import LIVE_ACK_TOKEN, TRAIN_FPS  # noqa: E402

ArmName = Literal["left", "right"]

CAPTURE_VERSION = "quanta_biman_live_capture_v2"
EXECUTE_VERSION = "quanta_biman_live_execute_v1"
EXECUTE_TRAJECTORY_VERSION = "quanta_biman_live_execute_trajectory_v1"
EXECUTE_END_POSE_TRAJECTORY_VERSION = "quanta_biman_live_execute_end_pose_trajectory_v1"

# SDK read RPCs (joint_states / end_pose / gripper / cameras) can briefly return
# UNAVAILABLE / connection-reset after long motion (e.g. dual home ~15s).
SDK_READ_MAX_ATTEMPTS = 6
SDK_READ_RETRY_BACKOFF_SEC = 0.12
END_POSE_PRE_DELAY_SEC = 0.10
POST_MOTION_READ_DELAY_SEC = 0.08
CAPTURE_CAMERA_PRE_DELAY_SEC = 0.05

# Backward-compatible aliases used by capture end_pose path.
END_POSE_MAX_ATTEMPTS = SDK_READ_MAX_ATTEMPTS
END_POSE_RETRY_BACKOFF_SEC = SDK_READ_RETRY_BACKOFF_SEC

_END_POSE_EEF_CACHE: dict[str, np.ndarray] = {}
_JOINT_STATE_CACHE: dict[str, np.ndarray] = {}
_GRIPPER_CACHE: dict[str, float] = {}


@dataclass(frozen=True)
class LiveComponents:
    head_camera: np.ndarray
    left_arm_camera: np.ndarray
    right_arm_camera: np.ndarray
    left_eef_9d: np.ndarray
    left_gripper_position: float
    left_joint_position: np.ndarray
    right_eef_9d: np.ndarray
    right_gripper_position: float
    right_joint_position: np.ndarray

    def to_dict(self) -> dict[str, Any]:
        return {
            "head_camera_shape": list(self.head_camera.shape),
            "left_arm_camera_shape": list(self.left_arm_camera.shape),
            "right_arm_camera_shape": list(self.right_arm_camera.shape),
            "left_eef_9d": self.left_eef_9d.astype(float).tolist(),
            "left_gripper_position": float(self.left_gripper_position),
            "left_joint_position": self.left_joint_position.astype(float).tolist(),
            "right_eef_9d": self.right_eef_9d.astype(float).tolist(),
            "right_gripper_position": float(self.right_gripper_position),
            "right_joint_position": self.right_joint_position.astype(float).tolist(),
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
        [str(PIPELINE3), str(PIPELINE2), env.get("PYTHONPATH", "")]
    ).strip(os.pathsep)
    return env


def resolve_robot_python(path: Path | None = None) -> Path:
    if path is not None:
        candidate = Path(path)
        if not candidate.is_file():
            raise FileNotFoundError(f"--robot-python not found: {candidate}")
        return candidate

    env_candidates = (
        os.environ.get("ROBOT_PYTHON"),
        os.environ.get("PY_SDK"),
    )
    hardcoded = (
        Path.home() / "miniconda3/envs/xr_lerobot/bin/python",
        Path.home() / "anaconda3/envs/xr_lerobot/bin/python",
        Path("/home/yichu/miniconda3/envs/xr_lerobot/bin/python"),
        Path("/home/ubuntu/anaconda3/envs/xr_lerobot/bin/python"),
    )
    for raw in (*env_candidates, *hardcoded):
        if not raw:
            continue
        candidate = Path(raw).expanduser()
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "Could not find xr_lerobot python. Pass --robot-python explicitly "
        "(or export PY_SDK / ROBOT_PYTHON)."
    )


def stamp_to_dict(msg: Any) -> dict[str, Any]:
    header = getattr(msg, "header", None)
    stamp = getattr(header, "stamp", None)
    if stamp is None:
        return {"sec": None, "nanosec": None, "timestamp_sec": None, "frame_id": ""}
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


def is_retryable_sdk_read_error(exc: BaseException) -> bool:
    message = repr(exc).lower()
    return (
        "no recent" in message
        or "statuscode.unavailable" in message
        or "status = statuscode.unavailable" in message
        or "connection reset" in message
        or "recvmsg" in message
        or "broken pipe" in message
        or "transport is closing" in message
        or "socket closed" in message
    )


def retry_sdk_call(
    name: str,
    fn: Any,
    *,
    pre_delay_sec: float = 0.0,
    max_attempts: int = SDK_READ_MAX_ATTEMPTS,
    retry_backoff_sec: float = SDK_READ_RETRY_BACKOFF_SEC,
    cache_dict: dict[str, Any] | None = None,
    cache_key: str | None = None,
    fallback: Any | None = None,
) -> tuple[Any, dict[str, Any]]:
    if pre_delay_sec > 0:
        time.sleep(pre_delay_sec)

    meta: dict[str, Any] = {
        "attempts": 0,
        "retry_backoff_sec": retry_backoff_sec,
        "used_cache_fallback": False,
        "used_command_fallback": False,
    }
    last_error: RuntimeError | None = None
    for attempt in range(1, max_attempts + 1):
        meta["attempts"] = attempt
        try:
            value, item_timing = timed_call(name, fn)
            meta.update(item_timing)
            if cache_dict is not None and cache_key is not None:
                if isinstance(value, np.ndarray):
                    cache_dict[cache_key] = value.copy()
                else:
                    cache_dict[cache_key] = value
            return value, meta
        except RuntimeError as exc:
            last_error = exc
            if attempt < max_attempts and is_retryable_sdk_read_error(exc):
                time.sleep(retry_backoff_sec * attempt)
                continue
            break

    if cache_dict is not None and cache_key is not None and cache_key in cache_dict:
        meta["used_cache_fallback"] = True
        meta["fallback_reason"] = repr(last_error)
        cached = cache_dict[cache_key]
        return (cached.copy() if isinstance(cached, np.ndarray) else cached), meta

    if fallback is not None:
        meta["used_command_fallback"] = True
        meta["fallback_reason"] = repr(last_error)
        fb = np.asarray(fallback, dtype=np.float32) if not isinstance(fallback, (int, float)) else fallback
        return fb, meta

    if last_error is not None:
        raise last_error
    raise RuntimeError(f"{name} failed with no fallback")


def fetch_end_pose_eef(
    robot: Any,
    arm: ArmName,
    *,
    pre_delay_sec: float = END_POSE_PRE_DELAY_SEC,
    max_attempts: int = END_POSE_MAX_ATTEMPTS,
    retry_backoff_sec: float = END_POSE_RETRY_BACKOFF_SEC,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Read arm end_pose → 9D eef with retry; fall back to last good sample if needed."""
    arm_obj = robot.left_arm if arm == "left" else robot.right_arm
    rpc_name = f"{arm}_end_pose"

    def fetch() -> np.ndarray:
        msg = arm_obj.get_end_pose()
        return end_pose_to_eef_9d_from_sdk(msg)

    eef, timing_meta = retry_sdk_call(
        rpc_name,
        fetch,
        pre_delay_sec=pre_delay_sec,
        max_attempts=max_attempts,
        retry_backoff_sec=retry_backoff_sec,
        cache_dict=_END_POSE_EEF_CACHE,
        cache_key=arm,
    )
    return np.asarray(eef, dtype=np.float32), timing_meta


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


def capture_live_components(
    robot: Any,
    *,
    end_pose_pre_delay_sec: float = END_POSE_PRE_DELAY_SEC,
    end_pose_max_attempts: int = END_POSE_MAX_ATTEMPTS,
) -> tuple[LiveComponents, dict[str, Any]]:
    def fetch_arm_state(arm: ArmName) -> Any:
        getter = robot.left_arm if arm == "left" else robot.right_arm

        def read_state() -> Any:
            return getter.get_joint_states()

        state, _meta = retry_sdk_call(
            f"{arm}_arm_state",
            read_state,
            max_attempts=end_pose_max_attempts,
        )
        return state

    def fetch_gripper(arm: ArmName) -> float:
        value, _meta = read_gripper(robot, arm, max_attempts=end_pose_max_attempts)
        return value

    timing_holder: dict[str, dict[str, Any]] = {}

    def fetch_camera(name: str, fn: Any) -> Any:
        value, meta = retry_sdk_call(
            name,
            fn,
            pre_delay_sec=CAPTURE_CAMERA_PRE_DELAY_SEC,
            max_attempts=end_pose_max_attempts,
            retry_backoff_sec=SDK_READ_RETRY_BACKOFF_SEC,
        )
        timing_holder[name] = meta
        return value

    # Cameras + joints/grippers concurrent; all use retry (streams often drop after long home).
    readonly_calls = {
        "head_rgb": lambda: fetch_camera("head_rgb", robot.head_camera.get_rgb_image),
        "left_arm_rgb": lambda: fetch_camera(
            "left_arm_rgb", robot.left_arm_camera.get_raw_image
        ),
        "right_arm_rgb": lambda: fetch_camera(
            "right_arm_rgb", robot.right_arm_camera.get_raw_image
        ),
        "left_arm_state": lambda: fetch_arm_state("left"),
        "right_arm_state": lambda: fetch_arm_state("right"),
        "left_gripper_position": lambda: fetch_gripper("left"),
        "right_gripper_position": lambda: fetch_gripper("right"),
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
            # Prefer inner retry meta for cameras; keep outer wall for others.
            if name in timing_holder:
                merged = dict(timing_holder[name])
                merged["outer_rpc_elapsed_ms"] = item_timing.get("rpc_elapsed_ms")
                timing[name] = merged
            else:
                timing[name] = item_timing

    left_eef_9d, left_end_pose_timing = fetch_end_pose_eef(
        robot,
        "left",
        pre_delay_sec=end_pose_pre_delay_sec,
        max_attempts=end_pose_max_attempts,
    )
    timing["left_end_pose"] = left_end_pose_timing
    right_eef_9d, right_end_pose_timing = fetch_end_pose_eef(
        robot,
        "right",
        pre_delay_sec=0.0,
        max_attempts=end_pose_max_attempts,
    )
    timing["right_end_pose"] = right_end_pose_timing

    whole_elapsed_ms = float((time.monotonic() - whole_start) * 1000.0)

    head_rgb = decode_jpeg_to_rgb(values["head_rgb"], name="head_rgb")
    left_rgb = decode_jpeg_to_rgb(values["left_arm_rgb"], name="left_arm_rgb")
    right_rgb = decode_jpeg_to_rgb(values["right_arm_rgb"], name="right_arm_rgb")

    left_arm = joint_state_to_dict(values["left_arm_state"])
    right_arm = joint_state_to_dict(values["right_arm_state"])
    left_joints = np.asarray(left_arm["position"], dtype=np.float32)
    right_joints = np.asarray(right_arm["position"], dtype=np.float32)
    if left_joints.shape != (6,) or right_joints.shape != (6,):
        raise ValueError(
            f"joint count left={left_joints.shape} right={right_joints.shape}, expected (6,)"
        )

    components = LiveComponents(
        head_camera=head_rgb,
        left_arm_camera=left_rgb,
        right_arm_camera=right_rgb,
        left_eef_9d=left_eef_9d,
        left_gripper_position=float(values["left_gripper_position"]),
        left_joint_position=left_joints,
        right_eef_9d=right_eef_9d,
        right_gripper_position=float(values["right_gripper_position"]),
        right_joint_position=right_joints,
    )

    end_pose_fallback = {
        arm: timing[f"{arm}_end_pose"].get("used_cache_fallback", False)
        for arm in ("left", "right")
    }
    meta = {
        "capture_version": CAPTURE_VERSION,
        "capture_mode": "seven_readonly_rpcs_concurrent_with_camera_retry_plus_end_pose_retry",
        "end_pose_retry": {
            "pre_delay_sec": end_pose_pre_delay_sec,
            "max_attempts": end_pose_max_attempts,
            "used_cache_fallback": end_pose_fallback,
        },
        "whole_request_wall_elapsed_ms": whole_elapsed_ms,
        "left_arm": left_arm,
        "right_arm": right_arm,
        "left_gripper_raw_sdk_position": float(components.left_gripper_position),
        "right_gripper_raw_sdk_position": float(components.right_gripper_position),
        "camera_timestamps": {
            "head_rgb": stamp_to_dict(values["head_rgb"]),
            "left_arm_rgb": stamp_to_dict(values["left_arm_rgb"]),
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
    left_path = output_dir / "left_arm_camera.jpg"
    right_path = output_dir / "right_arm_camera.jpg"
    cv2.imwrite(str(head_path), cv2.cvtColor(components.head_camera, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(left_path), cv2.cvtColor(components.left_arm_camera, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(right_path), cv2.cvtColor(components.right_arm_camera, cv2.COLOR_RGB2BGR))

    report = {
        **meta,
        "captured_at": datetime.now().isoformat(),
        "server": server,
        "image_paths": {
            "head_camera": str(head_path.resolve()),
            "left_arm_camera": str(left_path.resolve()),
            "right_arm_camera": str(right_path.resolve()),
        },
        "components": components.to_dict(),
    }
    report_path = output_dir / "capture.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report["capture_json"] = str(report_path.resolve())
    return report


def read_arm_joints(
    robot: Any,
    arm: ArmName,
    *,
    pre_delay_sec: float = 0.0,
    max_attempts: int = SDK_READ_MAX_ATTEMPTS,
    fallback_joints: np.ndarray | Sequence[float] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    getter = robot.left_arm if arm == "left" else robot.right_arm

    def fetch() -> np.ndarray:
        state = getter.get_joint_states()
        joints = np.asarray(list(state.position), dtype=np.float32)
        if joints.shape != (6,):
            raise ValueError(f"{arm}_arm shape {joints.shape}, expected (6,)")
        return joints

    joints, meta = retry_sdk_call(
        f"{arm}_joint_states",
        fetch,
        pre_delay_sec=pre_delay_sec,
        max_attempts=max_attempts,
        cache_dict=_JOINT_STATE_CACHE,
        cache_key=arm,
        fallback=fallback_joints,
    )
    return np.asarray(joints, dtype=np.float32).reshape(6), meta


def read_gripper(
    robot: Any,
    arm: ArmName,
    *,
    pre_delay_sec: float = 0.0,
    max_attempts: int = SDK_READ_MAX_ATTEMPTS,
    fallback_position: float | None = None,
) -> tuple[float, dict[str, Any]]:
    getter = robot.left_gripper if arm == "left" else robot.right_gripper

    def fetch() -> float:
        return gripper_position_value(getter.get_position())

    value, meta = retry_sdk_call(
        f"{arm}_gripper_position",
        fetch,
        pre_delay_sec=pre_delay_sec,
        max_attempts=max_attempts,
        cache_dict=_GRIPPER_CACHE,
        cache_key=arm,
        fallback=fallback_position,
    )
    return float(value), meta


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


def configure_end_pose_mode(robot: Any) -> None:
    from x2robot.sdk import (
        ManipulatorControlMode,
        ManipulatorControlModeParam,
        RobotModeParam,
        RobotWorkMode,
    )

    robot.system.set_work_mode(RobotModeParam(mode=RobotWorkMode.SDK))
    robot.robot_control.set_manipulator_control_mode(
        ManipulatorControlModeParam(mode=ManipulatorControlMode.MANIPULATOR_END_POSE)
    )


def build_sdk_pose_from_end_pose(end_pose: dict[str, Any]) -> Any:
    """Build x2robot Pose from per_task_home end_pose block (position_m + orientation_xyzw)."""
    from x2robot.geometry_msgs import Point, Pose, Quaternion

    if "position_m" in end_pose:
        pos = end_pose["position_m"]
    elif "position" in end_pose:
        pos = end_pose["position"]
    else:
        raise KeyError("end_pose missing position_m/position")

    if "orientation_xyzw" in end_pose:
        ori = end_pose["orientation_xyzw"]
    elif "orientation" in end_pose:
        ori = end_pose["orientation"]
    else:
        raise KeyError("end_pose missing orientation_xyzw/orientation")

    pose = Pose()
    pose.position = Point(x=float(pos["x"]), y=float(pos["y"]), z=float(pos["z"]))
    pose.orientation = Quaternion(
        x=float(ori["x"]),
        y=float(ori["y"]),
        z=float(ori["z"]),
        w=float(ori["w"]),
    )
    return pose


def end_pose_target_xyz_quat(end_pose: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    if "position_m" in end_pose:
        pos = end_pose["position_m"]
    else:
        pos = end_pose["position"]
    if "orientation_xyzw" in end_pose:
        ori = end_pose["orientation_xyzw"]
    else:
        ori = end_pose["orientation"]
    xyz = np.asarray([pos["x"], pos["y"], pos["z"]], dtype=np.float32)
    quat = np.asarray([ori["x"], ori["y"], ori["z"], ori["w"]], dtype=np.float32)
    return xyz, quat


def _read_end_pose_raw(
    robot: Any,
    arm: ArmName,
    *,
    pre_delay_sec: float = 0.0,
    max_attempts: int = END_POSE_MAX_ATTEMPTS,
) -> tuple[dict[str, Any], np.ndarray, dict[str, Any]]:
    arm_obj = robot.left_arm if arm == "left" else robot.right_arm
    rpc_name = f"{arm}_end_pose_raw"

    def fetch() -> dict[str, Any]:
        return sdk_end_pose_to_dict(arm_obj.get_end_pose())

    raw, meta = retry_sdk_call(
        rpc_name,
        fetch,
        pre_delay_sec=pre_delay_sec,
        max_attempts=max_attempts,
    )
    eef_9d = end_pose_to_eef_9d(add_rot6d_to_end_pose(raw))
    _END_POSE_EEF_CACHE[arm] = np.asarray(eef_9d, dtype=np.float32).copy()
    return raw, np.asarray(eef_9d, dtype=np.float32), meta


def _normalize_quat(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(4)
    n = float(np.linalg.norm(q))
    if n < 1e-12:
        return np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return q / n


def _slerp_quat(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear interpolation; q as xyzw."""
    q0 = _normalize_quat(q0)
    q1 = _normalize_quat(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        out = q0 + t * (q1 - q0)
        return _normalize_quat(out)
    theta_0 = float(np.arccos(np.clip(dot, -1.0, 1.0)))
    sin_theta_0 = float(np.sin(theta_0))
    theta = theta_0 * float(t)
    s0 = float(np.sin(theta_0 - theta) / sin_theta_0)
    s1 = float(np.sin(theta) / sin_theta_0)
    return _normalize_quat(s0 * q0 + s1 * q1)


def _pose_dict_from_xyz_quat(xyz: np.ndarray, quat: np.ndarray) -> dict[str, Any]:
    return {
        "position_m": {"x": float(xyz[0]), "y": float(xyz[1]), "z": float(xyz[2])},
        "orientation_xyzw": {
            "x": float(quat[0]),
            "y": float(quat[1]),
            "z": float(quat[2]),
            "w": float(quat[3]),
        },
    }


def densify_end_pose_waypoints(
    start_xyz: np.ndarray,
    start_quat: np.ndarray,
    start_gripper: float,
    target_xyz: np.ndarray,
    target_quat: np.ndarray,
    target_gripper: float,
    *,
    control_hz: float,
    max_linear_speed_m_s: float,
    min_duration_sec: float = 1.0,
    max_step_m: float = 0.008,
    skip_start_waypoint: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Return (xyz[N,3], quat[N,4], gripper[N], duration_sec).

    Path is strictly start→target (no overshoot). Optionally skip the start
    waypoint so we never re-command a possibly-stale "current" pose.
    Gripper is NOT interpolated: every waypoint uses ``target_gripper``.
    """
    if control_hz <= 0:
        raise ValueError("control_hz must be > 0 for end_pose interpolation")
    if max_linear_speed_m_s <= 0:
        raise ValueError("max_linear_speed_m_s must be > 0")

    start_xyz = np.asarray(start_xyz, dtype=np.float64).reshape(3)
    target_xyz = np.asarray(target_xyz, dtype=np.float64).reshape(3)
    start_quat = _normalize_quat(start_quat)
    target_quat = _normalize_quat(target_quat)

    dist = float(np.linalg.norm(target_xyz - start_xyz))
    qdot = float(np.clip(abs(np.dot(start_quat, target_quat)), 0.0, 1.0))
    ang = float(2.0 * np.arccos(qdot))
    duration = max(float(min_duration_sec), dist / float(max_linear_speed_m_s), ang / 0.4)

    # Also enforce max Cartesian step between consecutive commands.
    n_by_step = max(1, int(np.ceil(dist / max(float(max_step_m), 1e-6))))
    n_by_hz = max(1, int(np.ceil(duration * control_hz)))
    n_segments = max(n_by_step, n_by_hz)
    n = n_segments + 1
    alphas = np.linspace(0.0, 1.0, n, dtype=np.float64)

    xyzs = np.stack([(1.0 - a) * start_xyz + a * target_xyz for a in alphas], axis=0)
    quats = np.stack([_slerp_quat(start_quat, target_quat, float(a)) for a in alphas], axis=0)
    # Gripper: no interpolation — command the target immediately on every waypoint.
    del start_gripper  # retained in signature for call-site compat
    grips = np.full(n, float(target_gripper), dtype=np.float64)
    if skip_start_waypoint and len(xyzs) > 1:
        xyzs, quats, grips = xyzs[1:], quats[1:], grips[1:]
    return xyzs, quats, grips, float(duration)


def densify_end_pose_trajectory(
    start_xyz: np.ndarray,
    start_quat: np.ndarray,
    policy_xyzs: Sequence[np.ndarray],
    policy_quats: Sequence[np.ndarray],
    policy_grippers: Sequence[float],
    *,
    control_hz: float,
    max_linear_speed_m_s: float,
    max_step_m: float,
    min_duration_sec: float = 0.0,
    train_fps: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, list[int]]:
    """Densify adjacent policy end-poses into one stream.

    Gripper is NOT interpolated: each dense substep uses that segment's target gripper.
    Returns (xyz, quat, grip, planned_duration_sec, dense_counts_per_policy_step).
    """
    if len(policy_xyzs) != len(policy_quats) or len(policy_xyzs) != len(policy_grippers):
        raise ValueError("policy xyz/quat/gripper length mismatch")
    if len(policy_xyzs) < 1:
        raise ValueError("need at least one policy end_pose waypoint")

    segment_floor = float(min_duration_sec)
    if train_fps is not None and float(train_fps) > 0:
        segment_floor = max(segment_floor, 1.0 / float(train_fps))

    cur_xyz = np.asarray(start_xyz, dtype=np.float64).reshape(3)
    cur_quat = _normalize_quat(start_quat)
    all_xyz: list[np.ndarray] = []
    all_quat: list[np.ndarray] = []
    all_grip: list[np.ndarray] = []
    counts: list[int] = []
    total_dur = 0.0

    for xyz_t, quat_t, grip_t in zip(policy_xyzs, policy_quats, policy_grippers, strict=True):
        xyzs, quats, grips, dur = densify_end_pose_waypoints(
            cur_xyz,
            cur_quat,
            0.0,
            np.asarray(xyz_t, dtype=np.float64).reshape(3),
            _normalize_quat(quat_t),
            float(grip_t),
            control_hz=float(control_hz),
            max_linear_speed_m_s=float(max_linear_speed_m_s),
            min_duration_sec=float(segment_floor),
            max_step_m=float(max_step_m),
            skip_start_waypoint=True,
        )
        all_xyz.append(xyzs)
        all_quat.append(quats)
        all_grip.append(grips)
        counts.append(int(len(xyzs)))
        total_dur += float(dur)
        cur_xyz = np.asarray(xyz_t, dtype=np.float64).reshape(3)
        cur_quat = _normalize_quat(quat_t)

    return (
        np.concatenate(all_xyz, axis=0),
        np.concatenate(all_quat, axis=0),
        np.concatenate(all_grip, axis=0),
        float(total_dur),
        counts,
    )


def execute_end_pose_trajectory(
    robot: Any,
    *,
    arm: ArmName,
    policy_end_poses: Sequence[dict[str, Any]],
    policy_grippers: Sequence[float],
    configure_mode: bool,
    interpolate_hz: float,
    max_linear_speed_m_s: float,
    max_step_m: float,
    min_duration_sec: float = 0.0,
    train_fps: float = TRAIN_FPS,
    trajectory_settle_sec: float = 0.0,
) -> dict[str, Any]:
    """One-shot stream of densified absolute eef waypoints (no mid-horizon re-read/settle)."""
    from x2robot.sdk import GripperPosition

    if len(policy_end_poses) != len(policy_grippers):
        raise ValueError("policy_end_poses and policy_grippers length mismatch")
    if len(policy_end_poses) < 1:
        raise ValueError("need at least one policy end_pose waypoint")
    if float(interpolate_hz) <= 0 or float(max_linear_speed_m_s) <= 0:
        raise ValueError("require interpolate_hz>0 and max_linear_speed_m_s>0")

    arm_obj = robot.left_arm if arm == "left" else robot.right_arm
    gripper_obj = robot.left_gripper if arm == "left" else robot.right_gripper

    if configure_mode:
        configure_end_pose_mode(robot)
        time.sleep(0.5)

    before_raw, before_eef_9d, before_pose_meta = _read_end_pose_raw(
        robot, arm, pre_delay_sec=0.05, max_attempts=END_POSE_MAX_ATTEMPTS
    )
    before_gripper, before_gripper_meta = read_gripper(robot, arm)

    start_xyz = np.asarray(
        [before_raw["position"]["x"], before_raw["position"]["y"], before_raw["position"]["z"]],
        dtype=np.float64,
    )
    start_quat = np.asarray(
        [
            before_raw["orientation"]["x"],
            before_raw["orientation"]["y"],
            before_raw["orientation"]["z"],
            before_raw["orientation"]["w"],
        ],
        dtype=np.float64,
    )

    policy_xyzs: list[np.ndarray] = []
    policy_quats: list[np.ndarray] = []
    for pose in policy_end_poses:
        xyz, quat = end_pose_target_xyz_quat(pose)
        policy_xyzs.append(np.asarray(xyz, dtype=np.float64))
        policy_quats.append(np.asarray(quat, dtype=np.float64))

    xyzs, quats, grips, duration_plan_sec, counts = densify_end_pose_trajectory(
        start_xyz,
        start_quat,
        policy_xyzs,
        policy_quats,
        policy_grippers,
        control_hz=float(interpolate_hz),
        max_linear_speed_m_s=float(max_linear_speed_m_s),
        max_step_m=float(max_step_m),
        min_duration_sec=float(min_duration_sec),
        train_fps=float(train_fps),
    )
    n_waypoints = int(len(xyzs))
    period_sec = 1.0 / float(interpolate_hz)
    final_xyz = policy_xyzs[-1]
    final_quat = policy_quats[-1]
    final_grip = float(policy_grippers[-1])
    print(
        f"[set_end_pose] arm={arm} TRAJECTORY policy_wp={len(policy_end_poses)} "
        f"dense={n_waypoints} ~{duration_plan_sec:.2f}s "
        f"(hz={interpolate_hz}, vmax={max_linear_speed_m_s}m/s, step<={max_step_m}m, "
        f"train_fps={train_fps})",
        flush=True,
    )

    loop_start = time.perf_counter()
    schedule_at = loop_start
    prev_xyz = start_xyz.copy()
    for xyz, quat, grip in zip(xyzs, quats, grips, strict=True):
        step = float(np.linalg.norm(xyz - prev_xyz))
        if step > float(max_step_m) * 1.5 + 1e-6:
            raise RuntimeError(
                f"abort end_pose trajectory: planned step {step:.4f}m exceeds cap {max_step_m}m"
            )
        pose = build_sdk_pose_from_end_pose(_pose_dict_from_xyz_quat(xyz, quat))
        arm_obj.set_end_pose(pose)
        gripper_obj.set_position(GripperPosition(position=float(grip)))
        prev_xyz = xyz
        schedule_at += period_sec
        sleep_sec = schedule_at - time.perf_counter()
        if sleep_sec > 0:
            time.sleep(sleep_sec)

    final_pose = build_sdk_pose_from_end_pose(_pose_dict_from_xyz_quat(final_xyz, final_quat))
    for _ in range(3):
        arm_obj.set_end_pose(final_pose)
        gripper_obj.set_position(GripperPosition(position=float(final_grip)))
        time.sleep(period_sec)

    if float(trajectory_settle_sec) > 0:
        time.sleep(float(trajectory_settle_sec))

    wall_sec = time.perf_counter() - loop_start
    after_raw, after_eef_9d, after_pose_meta = _read_end_pose_raw(
        robot,
        arm,
        pre_delay_sec=POST_MOTION_READ_DELAY_SEC,
        max_attempts=END_POSE_MAX_ATTEMPTS,
    )
    after_gripper, after_gripper_meta = read_gripper(
        robot,
        arm,
        pre_delay_sec=0.0,
        fallback_position=float(final_grip),
    )
    after_xyz = np.asarray(
        [after_raw["position"]["x"], after_raw["position"]["y"], after_raw["position"]["z"]],
        dtype=np.float64,
    )
    after_quat = np.asarray(
        [
            after_raw["orientation"]["x"],
            after_raw["orientation"]["y"],
            after_raw["orientation"]["z"],
            after_raw["orientation"]["w"],
        ],
        dtype=np.float64,
    )
    position_error_m = float(np.linalg.norm(after_xyz - final_xyz))
    dot = float(
        np.clip(abs(np.dot(_normalize_quat(after_quat), _normalize_quat(final_quat))), 0.0, 1.0)
    )
    orientation_error_rad = float(2.0 * np.arccos(dot))

    return {
        "execute_version": EXECUTE_END_POSE_TRAJECTORY_VERSION,
        "arm": arm,
        "actuation_scope": f"{arm}_arm_end_pose_and_gripper",
        "control_mode": "MANIPULATOR_END_POSE",
        "execution_mode": "end_pose_trajectory",
        "interpolate_hz": float(interpolate_hz),
        "max_linear_speed_m_s": float(max_linear_speed_m_s),
        "min_duration_sec": float(min_duration_sec),
        "max_step_m": float(max_step_m),
        "train_fps": float(train_fps),
        "trajectory_settle_sec": float(trajectory_settle_sec),
        "policy_waypoints": int(len(policy_end_poses)),
        "dense_waypoints": n_waypoints,
        "dense_counts_per_policy_step": counts,
        "planned_duration_sec": float(duration_plan_sec),
        "wall_sec": float(wall_sec),
        "sdk_read_retry": {
            "before_end_pose": before_pose_meta,
            "before_gripper": before_gripper_meta,
            "after_end_pose": after_pose_meta,
            "after_gripper": after_gripper_meta,
        },
        "before_end_pose": before_raw,
        "before_eef_9d": before_eef_9d.astype(float).tolist(),
        "before_gripper": float(before_gripper),
        "final_policy_end_pose": _pose_dict_from_xyz_quat(final_xyz, final_quat),
        "final_policy_gripper": float(final_grip),
        "after_end_pose": after_raw,
        "after_eef_9d": after_eef_9d.astype(float).tolist(),
        "after_gripper": float(after_gripper),
        "position_error_m": position_error_m,
        "orientation_error_rad": orientation_error_rad,
    }


def execute_end_pose_step(
    robot: Any,
    *,
    arm: ArmName,
    end_pose: dict[str, Any],
    gripper_target: float,
    settle_sec: float,
    configure_mode: bool,
    interpolate_hz: float = 10.0,
    max_linear_speed_m_s: float = 0.015,
    min_duration_sec: float = 5.0,
    max_step_m: float = 0.008,
) -> dict[str, Any]:
    """Single slow interpolated set_end_pose stream (no one-shot, no overshoot)."""
    from x2robot.sdk import GripperPosition

    arm_obj = robot.left_arm if arm == "left" else robot.right_arm
    gripper_obj = robot.left_gripper if arm == "left" else robot.right_gripper

    if configure_mode:
        configure_end_pose_mode(robot)
        # Mode switch can make get_end_pose stale; settle then read for planning.
        time.sleep(0.5)

    before_raw, before_eef_9d, before_pose_meta = _read_end_pose_raw(
        robot, arm, pre_delay_sec=0.05, max_attempts=END_POSE_MAX_ATTEMPTS
    )
    # Second read to reject one-frame glitches as trajectory start.
    before_raw2, _, _ = _read_end_pose_raw(
        robot, arm, pre_delay_sec=0.05, max_attempts=END_POSE_MAX_ATTEMPTS
    )
    before_gripper, before_gripper_meta = read_gripper(robot, arm)

    target_xyz, target_quat = end_pose_target_xyz_quat(end_pose)
    start_xyz = np.asarray(
        [before_raw2["position"]["x"], before_raw2["position"]["y"], before_raw2["position"]["z"]],
        dtype=np.float64,
    )
    start_xyz_alt = np.asarray(
        [before_raw["position"]["x"], before_raw["position"]["y"], before_raw["position"]["z"]],
        dtype=np.float64,
    )
    if float(np.linalg.norm(start_xyz - start_xyz_alt)) > 0.03:
        # Prefer the later read; log the discrepancy.
        print(
            f"[set_end_pose] arm={arm} start-pose mismatch "
            f"{float(np.linalg.norm(start_xyz - start_xyz_alt)):.3f}m — using latest read",
            flush=True,
        )
    start_quat = np.asarray(
        [
            before_raw2["orientation"]["x"],
            before_raw2["orientation"]["y"],
            before_raw2["orientation"]["z"],
            before_raw2["orientation"]["w"],
        ],
        dtype=np.float64,
    )

    if float(interpolate_hz) <= 0 or float(max_linear_speed_m_s) <= 0:
        raise ValueError("end_pose one-shot disabled; require interpolate_hz>0 and speed>0")

    xyzs, quats, grips, duration_plan_sec = densify_end_pose_waypoints(
        start_xyz,
        start_quat,
        float(before_gripper),
        target_xyz,
        target_quat,
        float(gripper_target),
        control_hz=float(interpolate_hz),
        max_linear_speed_m_s=float(max_linear_speed_m_s),
        min_duration_sec=float(min_duration_sec),
        max_step_m=float(max_step_m),
        skip_start_waypoint=True,
    )
    n_waypoints = int(len(xyzs))
    period_sec = 1.0 / float(interpolate_hz)
    print(
        f"[set_end_pose] arm={arm} ONE path start="
        f"({start_xyz[0]:.3f},{start_xyz[1]:.3f},{start_xyz[2]:.3f}) -> "
        f"target=({target_xyz[0]:.3f},{target_xyz[1]:.3f},{target_xyz[2]:.3f}) "
        f"dist={float(np.linalg.norm(target_xyz - start_xyz)):.3f}m  "
        f"{n_waypoints} wp / ~{duration_plan_sec:.1f}s "
        f"(hz={interpolate_hz}, vmax={max_linear_speed_m_s}m/s, step<={max_step_m}m)",
        flush=True,
    )
    loop_start = time.perf_counter()
    schedule_at = loop_start
    prev_xyz = start_xyz.copy()
    for xyz, quat, grip in zip(xyzs, quats, grips, strict=True):
        step = float(np.linalg.norm(xyz - prev_xyz))
        if step > float(max_step_m) * 1.5 + 1e-6:
            raise RuntimeError(
                f"abort set_end_pose: planned step {step:.4f}m exceeds cap {max_step_m}m "
                f"(start read may be wrong)"
            )
        pose = build_sdk_pose_from_end_pose(_pose_dict_from_xyz_quat(xyz, quat))
        arm_obj.set_end_pose(pose)
        gripper_obj.set_position(GripperPosition(position=float(grip)))
        prev_xyz = xyz
        schedule_at += period_sec
        sleep_sec = schedule_at - time.perf_counter()
        if sleep_sec > 0:
            time.sleep(sleep_sec)

    # Hold final target a few cycles so SDK doesn't coast past it.
    final_pose = build_sdk_pose_from_end_pose(_pose_dict_from_xyz_quat(target_xyz, target_quat))
    for _ in range(3):
        arm_obj.set_end_pose(final_pose)
        gripper_obj.set_position(GripperPosition(position=float(gripper_target)))
        time.sleep(period_sec)

    wall_sec = time.perf_counter() - loop_start
    settle = max(float(settle_sec), 1.0)
    time.sleep(settle)

    after_raw, after_eef_9d, after_pose_meta = _read_end_pose_raw(
        robot,
        arm,
        pre_delay_sec=POST_MOTION_READ_DELAY_SEC,
        max_attempts=END_POSE_MAX_ATTEMPTS,
    )
    after_gripper, after_gripper_meta = read_gripper(
        robot,
        arm,
        pre_delay_sec=0.0,
        fallback_position=float(gripper_target),
    )

    after_xyz = np.asarray(
        [after_raw["position"]["x"], after_raw["position"]["y"], after_raw["position"]["z"]],
        dtype=np.float32,
    )
    after_quat = np.asarray(
        [
            after_raw["orientation"]["x"],
            after_raw["orientation"]["y"],
            after_raw["orientation"]["z"],
            after_raw["orientation"]["w"],
        ],
        dtype=np.float32,
    )
    position_error_m = float(np.linalg.norm(after_xyz - target_xyz.astype(np.float32)))
    dot = float(np.clip(abs(np.dot(_normalize_quat(after_quat), _normalize_quat(target_quat))), 0.0, 1.0))
    orientation_error_rad = float(2.0 * np.arccos(dot))

    return {
        "execute_version": "quanta_biman_set_end_pose_v3_single_path",
        "arm": arm,
        "actuation_scope": f"{arm}_arm_end_pose_and_gripper",
        "control_mode": "MANIPULATOR_END_POSE",
        "execution_mode": "interpolated_end_pose_single_path",
        "interpolate_hz": float(interpolate_hz),
        "max_linear_speed_m_s": float(max_linear_speed_m_s),
        "min_duration_sec": float(min_duration_sec),
        "max_step_m": float(max_step_m),
        "planned_duration_sec": float(duration_plan_sec),
        "dense_waypoints": int(n_waypoints),
        "wall_sec": float(wall_sec),
        "sdk_read_retry": {
            "before_end_pose": before_pose_meta,
            "before_gripper": before_gripper_meta,
            "after_end_pose": after_pose_meta,
            "after_gripper": after_gripper_meta,
        },
        "before_end_pose": before_raw2,
        "before_eef_9d": before_eef_9d.astype(float).tolist(),
        "before_gripper": float(before_gripper),
        "target_end_pose": _pose_dict_from_xyz_quat(target_xyz, target_quat),
        "gripper_target": float(gripper_target),
        "after_end_pose": after_raw,
        "after_eef_9d": after_eef_9d.astype(float).tolist(),
        "after_gripper": float(after_gripper),
        "position_error_m": position_error_m,
        "orientation_error_rad": orientation_error_rad,
        "settle_sec": float(settle),
    }


def _pad_end_pose_waypoints(
    xyzs: np.ndarray,
    quats: np.ndarray,
    grips: np.ndarray,
    n: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Hold the final pose so both arms share the same tick count."""
    cur = int(len(xyzs))
    if cur >= n:
        return xyzs, quats, grips
    pad = int(n - cur)
    xyzs = np.concatenate([xyzs, np.repeat(xyzs[-1:], pad, axis=0)], axis=0)
    quats = np.concatenate([quats, np.repeat(quats[-1:], pad, axis=0)], axis=0)
    grips = np.concatenate([grips, np.repeat(grips[-1:], pad, axis=0)], axis=0)
    return xyzs, quats, grips


def _plan_one_arm_start(
    robot: Any,
    arm: ArmName,
    end_pose: dict[str, Any],
    gripper_target: float,
) -> dict[str, Any]:
    before_raw, before_eef_9d, before_pose_meta = _read_end_pose_raw(
        robot, arm, pre_delay_sec=0.05, max_attempts=END_POSE_MAX_ATTEMPTS
    )
    before_raw2, _, _ = _read_end_pose_raw(
        robot, arm, pre_delay_sec=0.05, max_attempts=END_POSE_MAX_ATTEMPTS
    )
    before_gripper, before_gripper_meta = read_gripper(robot, arm)
    target_xyz, target_quat = end_pose_target_xyz_quat(end_pose)
    start_xyz = np.asarray(
        [before_raw2["position"]["x"], before_raw2["position"]["y"], before_raw2["position"]["z"]],
        dtype=np.float64,
    )
    start_xyz_alt = np.asarray(
        [before_raw["position"]["x"], before_raw["position"]["y"], before_raw["position"]["z"]],
        dtype=np.float64,
    )
    if float(np.linalg.norm(start_xyz - start_xyz_alt)) > 0.03:
        print(
            f"[set_end_pose] arm={arm} start-pose mismatch "
            f"{float(np.linalg.norm(start_xyz - start_xyz_alt)):.3f}m — using latest read",
            flush=True,
        )
    start_quat = np.asarray(
        [
            before_raw2["orientation"]["x"],
            before_raw2["orientation"]["y"],
            before_raw2["orientation"]["z"],
            before_raw2["orientation"]["w"],
        ],
        dtype=np.float64,
    )
    return {
        "before_raw2": before_raw2,
        "before_eef_9d": before_eef_9d,
        "before_pose_meta": before_pose_meta,
        "before_gripper": float(before_gripper),
        "before_gripper_meta": before_gripper_meta,
        "start_xyz": start_xyz,
        "start_quat": start_quat,
        "target_xyz": target_xyz,
        "target_quat": target_quat,
        "gripper_target": float(gripper_target),
    }


def execute_dual_end_pose_step(
    robot: Any,
    *,
    left_end_pose: dict[str, Any],
    left_gripper_target: float,
    right_end_pose: dict[str, Any],
    right_gripper_target: float,
    settle_sec: float,
    configure_mode: bool,
    interpolate_hz: float = 10.0,
    max_linear_speed_m_s: float = 0.015,
    min_duration_sec: float = 5.0,
    max_step_m: float = 0.008,
) -> dict[str, Any]:
    """Slow interpolate BOTH arms on the same clock (same vmax / hz / step).

    Previously home moved left fully, then right — left looked much slower.
    """
    from x2robot.sdk import GripperPosition

    if float(interpolate_hz) <= 0 or float(max_linear_speed_m_s) <= 0:
        raise ValueError("dual end_pose requires interpolate_hz>0 and speed>0")

    if configure_mode:
        configure_end_pose_mode(robot)
        time.sleep(0.5)

    left_plan = _plan_one_arm_start(robot, "left", left_end_pose, left_gripper_target)
    right_plan = _plan_one_arm_start(robot, "right", right_end_pose, right_gripper_target)

    left_xyzs, left_quats, left_grips, left_dur = densify_end_pose_waypoints(
        left_plan["start_xyz"],
        left_plan["start_quat"],
        left_plan["before_gripper"],
        left_plan["target_xyz"],
        left_plan["target_quat"],
        left_plan["gripper_target"],
        control_hz=float(interpolate_hz),
        max_linear_speed_m_s=float(max_linear_speed_m_s),
        # Per-arm min_duration would stretch the shorter arm and make it look slower.
        min_duration_sec=0.0,
        max_step_m=float(max_step_m),
        skip_start_waypoint=True,
    )
    right_xyzs, right_quats, right_grips, right_dur = densify_end_pose_waypoints(
        right_plan["start_xyz"],
        right_plan["start_quat"],
        right_plan["before_gripper"],
        right_plan["target_xyz"],
        right_plan["target_quat"],
        right_plan["gripper_target"],
        control_hz=float(interpolate_hz),
        max_linear_speed_m_s=float(max_linear_speed_m_s),
        min_duration_sec=0.0,
        max_step_m=float(max_step_m),
        skip_start_waypoint=True,
    )

    # Same strategy: identical vmax; shorter arm holds final pose while longer finishes.
    n_min = max(1, int(np.ceil(float(min_duration_sec) * float(interpolate_hz))))
    n_waypoints = max(int(len(left_xyzs)), int(len(right_xyzs)), n_min)
    left_xyzs, left_quats, left_grips = _pad_end_pose_waypoints(
        left_xyzs, left_quats, left_grips, n_waypoints
    )
    right_xyzs, right_quats, right_grips = _pad_end_pose_waypoints(
        right_xyzs, right_quats, right_grips, n_waypoints
    )
    duration_plan_sec = float(n_waypoints) / float(interpolate_hz)
    period_sec = 1.0 / float(interpolate_hz)

    ls = left_plan["start_xyz"]
    lt = left_plan["target_xyz"]
    rs = right_plan["start_xyz"]
    rt = right_plan["target_xyz"]
    print(
        f"[set_end_pose] DUAL sync path "
        f"left ({ls[0]:.3f},{ls[1]:.3f},{ls[2]:.3f})->({lt[0]:.3f},{lt[1]:.3f},{lt[2]:.3f}) "
        f"dist={float(np.linalg.norm(lt - ls)):.3f}m | "
        f"right ({rs[0]:.3f},{rs[1]:.3f},{rs[2]:.3f})->({rt[0]:.3f},{rt[1]:.3f},{rt[2]:.3f}) "
        f"dist={float(np.linalg.norm(rt - rs)):.3f}m | "
        f"{n_waypoints} wp / ~{duration_plan_sec:.1f}s "
        f"(hz={interpolate_hz}, vmax={max_linear_speed_m_s}m/s, step<={max_step_m}m)",
        flush=True,
    )

    left_arm = robot.left_arm
    right_arm = robot.right_arm
    left_gripper = robot.left_gripper
    right_gripper = robot.right_gripper

    loop_start = time.perf_counter()
    schedule_at = loop_start
    prev_left = left_plan["start_xyz"].copy()
    prev_right = right_plan["start_xyz"].copy()
    for i in range(n_waypoints):
        lxyz, lquat, lgrip = left_xyzs[i], left_quats[i], left_grips[i]
        rxyz, rquat, rgrip = right_xyzs[i], right_quats[i], right_grips[i]
        lstep = float(np.linalg.norm(lxyz - prev_left))
        rstep = float(np.linalg.norm(rxyz - prev_right))
        # Hold ticks may be ~0; moving ticks must respect cap.
        if lstep > float(max_step_m) * 1.5 + 1e-6:
            raise RuntimeError(
                f"abort dual set_end_pose left: planned step {lstep:.4f}m exceeds cap {max_step_m}m"
            )
        if rstep > float(max_step_m) * 1.5 + 1e-6:
            raise RuntimeError(
                f"abort dual set_end_pose right: planned step {rstep:.4f}m exceeds cap {max_step_m}m"
            )
        left_arm.set_end_pose(build_sdk_pose_from_end_pose(_pose_dict_from_xyz_quat(lxyz, lquat)))
        right_arm.set_end_pose(build_sdk_pose_from_end_pose(_pose_dict_from_xyz_quat(rxyz, rquat)))
        left_gripper.set_position(GripperPosition(position=float(lgrip)))
        right_gripper.set_position(GripperPosition(position=float(rgrip)))
        prev_left = lxyz
        prev_right = rxyz
        schedule_at += period_sec
        sleep_sec = schedule_at - time.perf_counter()
        if sleep_sec > 0:
            time.sleep(sleep_sec)

    left_final = build_sdk_pose_from_end_pose(
        _pose_dict_from_xyz_quat(left_plan["target_xyz"], left_plan["target_quat"])
    )
    right_final = build_sdk_pose_from_end_pose(
        _pose_dict_from_xyz_quat(right_plan["target_xyz"], right_plan["target_quat"])
    )
    for _ in range(3):
        left_arm.set_end_pose(left_final)
        right_arm.set_end_pose(right_final)
        left_gripper.set_position(GripperPosition(position=float(left_gripper_target)))
        right_gripper.set_position(GripperPosition(position=float(right_gripper_target)))
        time.sleep(period_sec)

    wall_sec = time.perf_counter() - loop_start
    settle = max(float(settle_sec), 1.0)
    time.sleep(settle)

    def _arm_after(arm: ArmName, plan: dict[str, Any], grip_target: float) -> dict[str, Any]:
        after_raw, after_eef_9d, after_pose_meta = _read_end_pose_raw(
            robot,
            arm,
            pre_delay_sec=POST_MOTION_READ_DELAY_SEC,
            max_attempts=END_POSE_MAX_ATTEMPTS,
        )
        after_gripper, after_gripper_meta = read_gripper(
            robot,
            arm,
            pre_delay_sec=0.0,
            fallback_position=float(grip_target),
        )
        after_xyz = np.asarray(
            [after_raw["position"]["x"], after_raw["position"]["y"], after_raw["position"]["z"]],
            dtype=np.float32,
        )
        after_quat = np.asarray(
            [
                after_raw["orientation"]["x"],
                after_raw["orientation"]["y"],
                after_raw["orientation"]["z"],
                after_raw["orientation"]["w"],
            ],
            dtype=np.float32,
        )
        target_xyz = plan["target_xyz"].astype(np.float32)
        target_quat = plan["target_quat"].astype(np.float32)
        position_error_m = float(np.linalg.norm(after_xyz - target_xyz))
        dot = float(
            np.clip(abs(np.dot(_normalize_quat(after_quat), _normalize_quat(target_quat))), 0.0, 1.0)
        )
        orientation_error_rad = float(2.0 * np.arccos(dot))
        return {
            "before_end_pose": plan["before_raw2"],
            "before_eef_9d": plan["before_eef_9d"].astype(float).tolist(),
            "before_gripper": float(plan["before_gripper"]),
            "target_end_pose": _pose_dict_from_xyz_quat(plan["target_xyz"], plan["target_quat"]),
            "gripper_target": float(grip_target),
            "after_end_pose": after_raw,
            "after_eef_9d": after_eef_9d.astype(float).tolist(),
            "after_gripper": float(after_gripper),
            "position_error_m": position_error_m,
            "orientation_error_rad": orientation_error_rad,
            "sdk_read_retry": {
                "before_end_pose": plan["before_pose_meta"],
                "before_gripper": plan["before_gripper_meta"],
                "after_end_pose": after_pose_meta,
                "after_gripper": after_gripper_meta,
            },
        }

    left_result = _arm_after("left", left_plan, left_gripper_target)
    right_result = _arm_after("right", right_plan, right_gripper_target)

    return {
        "execute_version": "quanta_biman_set_dual_end_pose_v1",
        "actuation_scope": "left_and_right_arm_end_pose_and_gripper",
        "control_mode": "MANIPULATOR_END_POSE",
        "execution_mode": "interpolated_dual_end_pose_sync",
        "interpolate_hz": float(interpolate_hz),
        "max_linear_speed_m_s": float(max_linear_speed_m_s),
        "min_duration_sec": float(min_duration_sec),
        "max_step_m": float(max_step_m),
        "planned_duration_sec": float(duration_plan_sec),
        "left_planned_duration_sec": float(left_dur),
        "right_planned_duration_sec": float(right_dur),
        "dense_waypoints": int(n_waypoints),
        "wall_sec": float(wall_sec),
        "settle_sec": float(settle),
        "left": left_result,
        "right": right_result,
        # Flat aliases so preposition can reuse single-arm field names.
        "left_position_error_m": left_result["position_error_m"],
        "right_position_error_m": right_result["position_error_m"],
        "left_orientation_error_rad": left_result["orientation_error_rad"],
        "right_orientation_error_rad": right_result["orientation_error_rad"],
    }


def run_configure_joint_position_mode_with_robot(
    robot: Any,
    *,
    server: str,
) -> dict[str, Any]:
    """Switch manipulator control mode back to JOINT_POSITIONS for policy execute."""
    configure_joint_position_mode(robot)
    return {
        "configure_version": "quanta_biman_configure_joint_mode_v1",
        "control_mode": "MANIPULATOR_JOINT_POSITIONS",
        "executed_at": datetime.now().isoformat(),
        "server": server,
    }


def read_lift_position_m(robot: Any) -> float:
    msg = robot.lift.get_lift_position()
    return float(msg.position)


def set_lift_position_m(
    robot: Any,
    *,
    position_m: float,
    settle_sec: float = 1.0,
    configure_sdk_mode: bool = True,
    tolerance_m: float = 0.005,
    max_attempts: int = 30,
    poll_interval_sec: float = 0.5,
) -> dict[str, Any]:
    """Move lift to absolute height (meters) via LiftController.

    Matches robot_control_yichu.py: JOINT mode + poll until within tolerance.
    A single short sleep is NOT enough — logs showed target~0.30 but after~0.35.
    """
    from x2robot.sdk import (
        LiftPosition,
        ManipulatorControlMode,
        ManipulatorControlModeParam,
        RobotModeParam,
        RobotWorkMode,
    )

    target = float(position_m)
    before = read_lift_position_m(robot)
    if configure_sdk_mode:
        robot.system.set_work_mode(RobotModeParam(mode=RobotWorkMode.SDK))
    # Lift moves reliably in JOINT mode (same as examples/robot_control_yichu.py).
    robot.robot_control.set_manipulator_control_mode(
        ManipulatorControlModeParam(mode=ManipulatorControlMode.MANIPULATOR_JOINT_POSITIONS)
    )

    reached = False
    last_error_message = ""
    attempts_used = 0
    after = before
    for attempt in range(1, int(max_attempts) + 1):
        attempts_used = attempt
        after = read_lift_position_m(robot)
        err = abs(after - target)
        print(
            f"[set_lift] attempt {attempt}/{max_attempts}: "
            f"lift={after:.4f}m target={target:.4f}m err={err:.4f}m",
            flush=True,
        )
        if err <= float(tolerance_m):
            reached = True
            break
        result = robot.lift.set_lift_position(LiftPosition(position=target))
        ok = bool(getattr(result, "is_success", True))
        if not ok:
            last_error_message = str(getattr(result, "error_message", "") or "set_lift failed")
            print(f"[set_lift] command failed: {last_error_message}", flush=True)
        time.sleep(float(poll_interval_sec))
    else:
        after = read_lift_position_m(robot)

    # Optional extra settle after reaching (or after timeout).
    if float(settle_sec) > 0:
        time.sleep(float(settle_sec))
        after = read_lift_position_m(robot)

    return {
        "lift_version": "quanta_biman_set_lift_v2_poll",
        "before_lift_position_m": float(before),
        "target_lift_position_m": target,
        "after_lift_position_m": float(after),
        "settle_sec": float(settle_sec),
        "tolerance_m": float(tolerance_m),
        "max_attempts": int(max_attempts),
        "attempts_used": int(attempts_used),
        "poll_interval_sec": float(poll_interval_sec),
        "reached": bool(reached),
        "abs_error_m": abs(float(after) - target),
        "last_error_message": last_error_message,
    }


def bound_absolute_target(
    current: np.ndarray,
    target: np.ndarray,
    *,
    arm: ArmName,
    max_joint_delta_rad: float,
) -> tuple[np.ndarray, np.ndarray]:
    current = np.asarray(current, dtype=np.float32).reshape(6)
    target = np.asarray(target, dtype=np.float32).reshape(6)
    clipped = clip_joints_to_sdk(arm, target)
    delta = clipped - current
    if max_joint_delta_rad > 0:
        delta = np.clip(delta, -max_joint_delta_rad, max_joint_delta_rad)
    command = clip_joints_to_sdk(arm, current + delta)
    actual_delta = command - current
    return command, actual_delta


def execute_absolute_step(
    robot: Any,
    *,
    arm: ArmName,
    joint_targets: Sequence[float] | np.ndarray,
    gripper_target: float,
    settle_sec: float,
    max_joint_delta_rad: float,
    configure_mode: bool,
) -> dict[str, Any]:
    from x2robot.sdk import GripperPosition, JointPositions

    arm_obj = robot.left_arm if arm == "left" else robot.right_arm
    gripper_obj = robot.left_gripper if arm == "left" else robot.right_gripper
    lower = SDK_LEFT_ARM_LOWER if arm == "left" else SDK_RIGHT_ARM_LOWER
    upper = SDK_LEFT_ARM_UPPER if arm == "left" else SDK_RIGHT_ARM_UPPER

    before_joints, before_joints_meta = read_arm_joints(robot, arm)
    before_gripper, before_gripper_meta = read_gripper(robot, arm)

    sdk_target, actual_delta = bound_absolute_target(
        before_joints,
        np.asarray(joint_targets, dtype=np.float32),
        arm=arm,
        max_joint_delta_rad=max_joint_delta_rad,
    )

    if configure_mode:
        configure_joint_position_mode(robot)

    arm_obj.set_joint_positions(JointPositions(positions=sdk_target.astype(float).tolist()))
    gripper_obj.set_position(GripperPosition(position=float(gripper_target)))

    if settle_sec > 0:
        time.sleep(settle_sec)

    after_joints, after_joints_meta = read_arm_joints(
        robot,
        arm,
        pre_delay_sec=POST_MOTION_READ_DELAY_SEC,
        fallback_joints=sdk_target,
    )
    after_gripper, after_gripper_meta = read_gripper(
        robot,
        arm,
        pre_delay_sec=0.0,
        fallback_position=float(gripper_target),
    )
    observed_joint_delta = after_joints - before_joints
    tracking_error = observed_joint_delta - actual_delta

    return {
        "execute_version": EXECUTE_VERSION,
        "arm": arm,
        "actuation_scope": f"{arm}_arm_and_gripper",
        "sdk_read_retry": {
            "before_joints": before_joints_meta,
            "before_gripper": before_gripper_meta,
            "after_joints": after_joints_meta,
            "after_gripper": after_gripper_meta,
        },
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
        "sdk_joint_limits_lower_rad": lower.astype(float).tolist(),
        "sdk_joint_limits_upper_rad": upper.astype(float).tolist(),
        "settle_sec": float(settle_sec),
        "max_joint_delta_rad": float(max_joint_delta_rad),
    }


def densify_arm_waypoints(
    start_joints: np.ndarray,
    start_gripper: float,
    policy_joints: Sequence[np.ndarray | Sequence[float]],
    policy_grippers: Sequence[float],
    *,
    control_hz: float,
    train_fps: float = TRAIN_FPS,
) -> tuple[np.ndarray, np.ndarray]:
    """Linear joint interpolation on the dev machine (one segment per policy step).

    Gripper is NOT interpolated: each dense substep uses the segment target gripper.
    """
    if control_hz <= 0:
        raise ValueError("control_hz must be > 0")
    if train_fps <= 0:
        raise ValueError("train_fps must be > 0")
    if len(policy_joints) != len(policy_grippers):
        raise ValueError("policy_joints and policy_grippers length mismatch")

    joints_path = [np.asarray(start_joints, dtype=np.float32).reshape(6)]
    grip_path = [float(start_gripper)]
    for joints, grip in zip(policy_joints, policy_grippers, strict=True):
        joints_path.append(np.asarray(joints, dtype=np.float32).reshape(6))
        grip_path.append(float(grip))

    sub_steps_per_segment = max(1, int(round(control_hz / train_fps)))
    dense_joints: list[np.ndarray] = []
    dense_grippers: list[float] = []
    for seg_index in range(len(joints_path) - 1):
        j0, j1 = joints_path[seg_index], joints_path[seg_index + 1]
        g1 = grip_path[seg_index + 1]
        for step_index in range(sub_steps_per_segment):
            alpha = step_index / sub_steps_per_segment
            dense_joints.append(j0 + alpha * (j1 - j0))
            dense_grippers.append(float(g1))
    dense_joints.append(joints_path[-1])
    dense_grippers.append(grip_path[-1])
    return np.stack(dense_joints, axis=0), np.asarray(dense_grippers, dtype=np.float64)


def execute_interpolated_trajectory(
    robot: Any,
    *,
    arm: ArmName,
    start_joints: np.ndarray,
    start_gripper: float,
    policy_joints: Sequence[np.ndarray | Sequence[float]],
    policy_grippers: Sequence[float],
    control_hz: float,
    train_fps: float,
    max_joint_delta_rad: float,
    trajectory_settle_sec: float,
    configure_mode: bool,
) -> dict[str, Any]:
    from x2robot.sdk import GripperPosition, JointPositions

    control_hz = min(float(control_hz), 200.0)
    arm_obj = robot.left_arm if arm == "left" else robot.right_arm
    gripper_obj = robot.left_gripper if arm == "left" else robot.right_gripper

    dense_joints, dense_grippers = densify_arm_waypoints(
        start_joints,
        start_gripper,
        policy_joints,
        policy_grippers,
        control_hz=control_hz,
        train_fps=train_fps,
    )

    if configure_mode:
        configure_joint_position_mode(robot)

    period_sec = 1.0 / control_hz
    commanded_joints = np.asarray(start_joints, dtype=np.float32).reshape(6)
    max_abs_delta = 0.0
    loop_start = time.perf_counter()
    schedule_at = loop_start

    for target_joints, target_gripper in zip(dense_joints, dense_grippers, strict=True):
        sdk_target, actual_delta = bound_absolute_target(
            commanded_joints,
            target_joints,
            arm=arm,
            max_joint_delta_rad=max_joint_delta_rad,
        )
        arm_obj.set_joint_positions(JointPositions(positions=sdk_target.astype(float).tolist()))
        gripper_obj.set_position(GripperPosition(position=float(target_gripper)))
        commanded_joints = sdk_target
        max_abs_delta = max(max_abs_delta, float(np.max(np.abs(actual_delta))))

        schedule_at += period_sec
        sleep_sec = schedule_at - time.perf_counter()
        if sleep_sec > 0:
            time.sleep(sleep_sec)

    if trajectory_settle_sec > 0:
        time.sleep(trajectory_settle_sec)

    after_joints, after_joints_meta = read_arm_joints(
        robot,
        arm,
        pre_delay_sec=POST_MOTION_READ_DELAY_SEC,
        fallback_joints=commanded_joints,
    )
    after_gripper, after_gripper_meta = read_gripper(
        robot,
        arm,
        pre_delay_sec=0.0,
        fallback_position=float(dense_grippers[-1]),
    )

    wall_sec = time.perf_counter() - loop_start
    return {
        "execute_version": EXECUTE_TRAJECTORY_VERSION,
        "execution_mode": "interpolated_trajectory",
        "interpolation_host": "dev_machine_daemon",
        "arm": arm,
        "control_hz": control_hz,
        "train_fps": float(train_fps),
        "policy_waypoints": int(len(policy_joints)),
        "dense_substeps": int(len(dense_joints)),
        "substeps_per_policy_step": max(1, int(round(control_hz / train_fps))),
        "trajectory_settle_sec": float(trajectory_settle_sec),
        "max_joint_delta_rad": float(max_joint_delta_rad),
        "before_joints_rad": np.asarray(start_joints, dtype=np.float32).astype(float).tolist(),
        "before_gripper": float(start_gripper),
        "final_policy_joints_rad": np.asarray(policy_joints[-1], dtype=np.float32).astype(float).tolist(),
        "final_policy_gripper": float(policy_grippers[-1]),
        "after_joints_rad": after_joints.astype(float).tolist(),
        "after_gripper": float(after_gripper),
        "max_abs_command_delta_rad": max_abs_delta,
        "wall_sec": float(wall_sec),
        "sdk_read_retry": {
            "after_joints": after_joints_meta,
            "after_gripper": after_gripper_meta,
        },
    }


def run_execute_trajectory_with_robot(
    robot: Any,
    *,
    arm: ArmName,
    policy_joints: Sequence[Sequence[float] | np.ndarray],
    policy_grippers: Sequence[float],
    control_hz: float,
    train_fps: float,
    max_joint_delta_rad: float,
    trajectory_settle_sec: float,
    configure_mode: bool,
    server: str,
) -> dict[str, Any]:
    start_joints, _ = read_arm_joints(robot, arm)
    start_gripper, _ = read_gripper(robot, arm)
    result = execute_interpolated_trajectory(
        robot,
        arm=arm,
        start_joints=start_joints,
        start_gripper=start_gripper,
        policy_joints=policy_joints,
        policy_grippers=policy_grippers,
        control_hz=control_hz,
        train_fps=train_fps,
        max_joint_delta_rad=max_joint_delta_rad,
        trajectory_settle_sec=trajectory_settle_sec,
        configure_mode=configure_mode,
    )
    result["executed_at"] = datetime.now().isoformat()
    result["server"] = server
    return result


def run_capture_with_robot(
    robot: Any,
    *,
    output_dir: Path,
    server: str,
    end_pose_pre_delay_sec: float = END_POSE_PRE_DELAY_SEC,
    end_pose_max_attempts: int = END_POSE_MAX_ATTEMPTS,
) -> dict[str, Any]:
    components, meta = capture_live_components(
        robot,
        end_pose_pre_delay_sec=end_pose_pre_delay_sec,
        end_pose_max_attempts=end_pose_max_attempts,
    )
    return save_capture_artifacts(
        components,
        meta,
        output_dir=output_dir,
        server=server,
    )


def run_execute_with_robot(
    robot: Any,
    *,
    arm: ArmName,
    joint_targets: Sequence[float] | np.ndarray,
    gripper_target: float,
    settle_sec: float,
    max_joint_delta_rad: float,
    configure_mode: bool,
    server: str,
) -> dict[str, Any]:
    result = execute_absolute_step(
        robot,
        arm=arm,
        joint_targets=joint_targets,
        gripper_target=gripper_target,
        settle_sec=settle_sec,
        max_joint_delta_rad=max_joint_delta_rad,
        configure_mode=configure_mode,
    )
    result["executed_at"] = datetime.now().isoformat()
    result["server"] = server
    return result


def run_set_lift_with_robot(
    robot: Any,
    *,
    position_m: float,
    settle_sec: float,
    configure_sdk_mode: bool,
    server: str,
) -> dict[str, Any]:
    result = set_lift_position_m(
        robot,
        position_m=position_m,
        settle_sec=settle_sec,
        configure_sdk_mode=configure_sdk_mode,
    )
    result["executed_at"] = datetime.now().isoformat()
    result["server"] = server
    return result


# LiveSdkDaemonClient defaults (also used by set_end_pose CLI)
DEFAULT_END_POSE_INTERPOLATE_HZ = 10.0
DEFAULT_END_POSE_MAX_LINEAR_SPEED_M_S = 0.015
DEFAULT_END_POSE_MIN_DURATION_SEC = 5.0
DEFAULT_END_POSE_MAX_STEP_M = 0.008


def run_set_end_pose_with_robot(
    robot: Any,
    *,
    arm: ArmName,
    end_pose: dict[str, Any],
    gripper_target: float,
    settle_sec: float,
    configure_mode: bool,
    server: str,
    interpolate_hz: float = DEFAULT_END_POSE_INTERPOLATE_HZ,
    max_linear_speed_m_s: float = DEFAULT_END_POSE_MAX_LINEAR_SPEED_M_S,
    min_duration_sec: float = DEFAULT_END_POSE_MIN_DURATION_SEC,
    max_step_m: float = DEFAULT_END_POSE_MAX_STEP_M,
) -> dict[str, Any]:
    result = execute_end_pose_step(
        robot,
        arm=arm,
        end_pose=end_pose,
        gripper_target=gripper_target,
        settle_sec=settle_sec,
        configure_mode=configure_mode,
        interpolate_hz=float(interpolate_hz),
        max_linear_speed_m_s=float(max_linear_speed_m_s),
        min_duration_sec=float(min_duration_sec),
        max_step_m=float(max_step_m),
    )
    result["executed_at"] = datetime.now().isoformat()
    result["server"] = server
    return result


def run_execute_end_pose_trajectory_with_robot(
    robot: Any,
    *,
    arm: ArmName,
    policy_end_poses: Sequence[dict[str, Any]],
    policy_grippers: Sequence[float],
    configure_mode: bool,
    server: str,
    interpolate_hz: float,
    max_linear_speed_m_s: float,
    max_step_m: float,
    min_duration_sec: float = 0.0,
    train_fps: float = TRAIN_FPS,
    trajectory_settle_sec: float = 0.0,
) -> dict[str, Any]:
    result = execute_end_pose_trajectory(
        robot,
        arm=arm,
        policy_end_poses=policy_end_poses,
        policy_grippers=policy_grippers,
        configure_mode=configure_mode,
        interpolate_hz=float(interpolate_hz),
        max_linear_speed_m_s=float(max_linear_speed_m_s),
        max_step_m=float(max_step_m),
        min_duration_sec=float(min_duration_sec),
        train_fps=float(train_fps),
        trajectory_settle_sec=float(trajectory_settle_sec),
    )
    result["executed_at"] = datetime.now().isoformat()
    result["server"] = server
    return result


def run_set_dual_end_pose_with_robot(
    robot: Any,
    *,
    left_end_pose: dict[str, Any],
    left_gripper_target: float,
    right_end_pose: dict[str, Any],
    right_gripper_target: float,
    settle_sec: float,
    configure_mode: bool,
    server: str,
    interpolate_hz: float = DEFAULT_END_POSE_INTERPOLATE_HZ,
    max_linear_speed_m_s: float = DEFAULT_END_POSE_MAX_LINEAR_SPEED_M_S,
    min_duration_sec: float = DEFAULT_END_POSE_MIN_DURATION_SEC,
    max_step_m: float = DEFAULT_END_POSE_MAX_STEP_M,
) -> dict[str, Any]:
    result = execute_dual_end_pose_step(
        robot,
        left_end_pose=left_end_pose,
        left_gripper_target=float(left_gripper_target),
        right_end_pose=right_end_pose,
        right_gripper_target=float(right_gripper_target),
        settle_sec=settle_sec,
        configure_mode=configure_mode,
        interpolate_hz=float(interpolate_hz),
        max_linear_speed_m_s=float(max_linear_speed_m_s),
        min_duration_sec=float(min_duration_sec),
        max_step_m=float(max_step_m),
    )
    result["executed_at"] = datetime.now().isoformat()
    result["server"] = server
    return result


def run_configure_joint_mode_with_robot(
    robot: Any,
    *,
    server: str,
) -> dict[str, Any]:
    return run_configure_joint_position_mode_with_robot(robot, server=server)


def run_capture_cli(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    robot = connect_robot(args.server)
    report = run_capture_with_robot(
        robot,
        output_dir=output_dir,
        server=args.server,
        end_pose_pre_delay_sec=float(args.end_pose_pre_delay_sec),
        end_pose_max_attempts=int(args.end_pose_max_attempts),
    )
    print(f"capture_dir={output_dir.resolve()}")
    print(f"capture_json={report['capture_json']}")
    return 0


def run_execute_cli(args: argparse.Namespace) -> int:
    if not args.allow_motion:
        raise SystemExit("--allow-motion is required for execute mode")
    if args.acknowledge != LIVE_ACK_TOKEN:
        raise SystemExit(f"--acknowledge must be exactly {LIVE_ACK_TOKEN!r}")

    arm = args.arm.lower()
    if arm not in {"left", "right"}:
        raise SystemExit("--arm must be left or right")

    joint_targets = json.loads(args.joint_targets_json)
    if not isinstance(joint_targets, list) or len(joint_targets) != 6:
        raise SystemExit("--joint-targets-json must be a JSON list of 6 floats")

    robot = connect_robot(args.server)
    result = run_execute_with_robot(
        robot,
        arm=arm,  # type: ignore[arg-type]
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


def run_execute_trajectory_cli(args: argparse.Namespace) -> int:
    if not args.allow_motion:
        raise SystemExit("--allow-motion is required for execute_trajectory mode")
    if args.acknowledge != LIVE_ACK_TOKEN:
        raise SystemExit(f"--acknowledge must be exactly {LIVE_ACK_TOKEN!r}")

    arm = args.arm.lower()
    if arm not in {"left", "right"}:
        raise SystemExit("--arm must be left or right")

    waypoints = json.loads(args.waypoints_json)
    if not isinstance(waypoints, list) or not waypoints:
        raise SystemExit("--waypoints-json must be a non-empty JSON list")
    policy_joints: list[list[float]] = []
    policy_grippers: list[float] = []
    for index, waypoint in enumerate(waypoints):
        if not isinstance(waypoint, dict):
            raise SystemExit(f"waypoint[{index}] must be an object")
        joints = waypoint.get("joint_targets")
        if not isinstance(joints, list) or len(joints) != 6:
            raise SystemExit(f"waypoint[{index}].joint_targets must be a list of 6 floats")
        policy_joints.append([float(x) for x in joints])
        policy_grippers.append(float(waypoint["gripper_target"]))

    robot = connect_robot(args.server)
    result = run_execute_trajectory_with_robot(
        robot,
        arm=arm,  # type: ignore[arg-type]
        policy_joints=policy_joints,
        policy_grippers=policy_grippers,
        control_hz=float(args.control_hz),
        train_fps=float(args.train_fps),
        max_joint_delta_rad=float(args.max_joint_delta_rad),
        trajectory_settle_sec=float(args.trajectory_settle_sec),
        configure_mode=bool(args.configure_mode),
        server=args.server,
    )

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"execute_trajectory_json={output_json.resolve()}")
    return 0


def run_execute_end_pose_trajectory_cli(args: argparse.Namespace) -> int:
    if not args.allow_motion:
        raise SystemExit("--allow-motion is required for execute-end-pose-trajectory mode")
    if args.acknowledge != LIVE_ACK_TOKEN:
        raise SystemExit(f"--acknowledge must be exactly {LIVE_ACK_TOKEN!r}")

    arm = args.arm.lower()
    if arm not in {"left", "right"}:
        raise SystemExit("--arm must be left or right")

    waypoints = json.loads(args.waypoints_json)
    if not isinstance(waypoints, list) or not waypoints:
        raise SystemExit("--waypoints-json must be a non-empty JSON list")
    policy_end_poses: list[dict[str, Any]] = []
    policy_grippers: list[float] = []
    for index, waypoint in enumerate(waypoints):
        if not isinstance(waypoint, dict):
            raise SystemExit(f"waypoint[{index}] must be an object")
        end_pose = waypoint.get("end_pose")
        if not isinstance(end_pose, dict):
            raise SystemExit(f"waypoint[{index}].end_pose must be an object")
        policy_end_poses.append(end_pose)
        policy_grippers.append(float(waypoint["gripper_target"]))

    robot = connect_robot(args.server)
    result = run_execute_end_pose_trajectory_with_robot(
        robot,
        arm=arm,  # type: ignore[arg-type]
        policy_end_poses=policy_end_poses,
        policy_grippers=policy_grippers,
        configure_mode=bool(args.configure_mode),
        server=args.server,
        interpolate_hz=float(args.interpolate_hz),
        max_linear_speed_m_s=float(args.max_linear_speed_m_s),
        max_step_m=float(args.max_step_m),
        min_duration_sec=float(args.min_duration_sec),
        train_fps=float(args.train_fps),
        trajectory_settle_sec=float(args.trajectory_settle_sec),
    )

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"execute_end_pose_trajectory_json={output_json.resolve()}")
    return 0


def run_set_lift_cli(args: argparse.Namespace) -> int:
    if not args.allow_motion:
        raise SystemExit("--allow-motion is required for set-lift mode")
    if args.acknowledge != LIVE_ACK_TOKEN:
        raise SystemExit(f"--acknowledge must be exactly {LIVE_ACK_TOKEN!r}")

    robot = connect_robot(args.server)
    result = run_set_lift_with_robot(
        robot,
        position_m=float(args.position_m),
        settle_sec=float(args.settle_sec),
        configure_sdk_mode=bool(args.configure_sdk_mode),
        server=args.server,
    )
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"set_lift_json={output_json.resolve()}")
    return 0


def run_set_end_pose_cli(args: argparse.Namespace) -> int:
    if not args.allow_motion:
        raise SystemExit("--allow-motion is required for set-end-pose mode")
    if args.acknowledge != LIVE_ACK_TOKEN:
        raise SystemExit(f"--acknowledge must be exactly {LIVE_ACK_TOKEN!r}")

    arm = args.arm.lower()
    if arm not in {"left", "right"}:
        raise SystemExit("--arm must be left or right")

    end_pose = json.loads(args.end_pose_json)
    if not isinstance(end_pose, dict):
        raise SystemExit("--end-pose-json must be a JSON object")

    robot = connect_robot(args.server)
    result = run_set_end_pose_with_robot(
        robot,
        arm=arm,  # type: ignore[arg-type]
        end_pose=end_pose,
        gripper_target=float(args.gripper_target),
        settle_sec=float(args.settle_sec),
        configure_mode=bool(args.configure_mode),
        server=args.server,
        interpolate_hz=float(args.interpolate_hz),
        max_linear_speed_m_s=float(args.max_linear_speed_m_s),
        min_duration_sec=float(args.min_duration_sec),
        max_step_m=float(args.max_step_m),
    )
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"set_end_pose_json={output_json.resolve()}")
    return 0


def run_set_dual_end_pose_cli(args: argparse.Namespace) -> int:
    if not args.allow_motion:
        raise SystemExit("--allow-motion is required for set-dual-end-pose mode")
    if args.acknowledge != LIVE_ACK_TOKEN:
        raise SystemExit(f"--acknowledge must be exactly {LIVE_ACK_TOKEN!r}")

    left_end_pose = json.loads(args.left_end_pose_json)
    right_end_pose = json.loads(args.right_end_pose_json)
    if not isinstance(left_end_pose, dict) or not isinstance(right_end_pose, dict):
        raise SystemExit("--left/right-end-pose-json must be JSON objects")

    robot = connect_robot(args.server)
    result = run_set_dual_end_pose_with_robot(
        robot,
        left_end_pose=left_end_pose,
        left_gripper_target=float(args.left_gripper_target),
        right_end_pose=right_end_pose,
        right_gripper_target=float(args.right_gripper_target),
        settle_sec=float(args.settle_sec),
        configure_mode=bool(args.configure_mode),
        server=args.server,
        interpolate_hz=float(args.interpolate_hz),
        max_linear_speed_m_s=float(args.max_linear_speed_m_s),
        min_duration_sec=float(args.min_duration_sec),
        max_step_m=float(args.max_step_m),
    )
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"set_dual_end_pose_json={output_json.resolve()}")
    return 0


def run_configure_joint_mode_cli(args: argparse.Namespace) -> int:
    if not args.allow_motion:
        raise SystemExit("--allow-motion is required for configure-joint-mode")
    if args.acknowledge != LIVE_ACK_TOKEN:
        raise SystemExit(f"--acknowledge must be exactly {LIVE_ACK_TOKEN!r}")
    robot = connect_robot(args.server)
    result = run_configure_joint_mode_with_robot(robot, server=args.server)
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"configure_joint_mode_json={output_json.resolve()}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Quanta biman live SDK capture / execute helper.")
    sub = parser.add_subparsers(dest="command", required=True)

    capture = sub.add_parser("capture", help="Read-only concurrent capture (3 cameras + biman state).")
    capture.add_argument("--server", default="127.0.0.1:15051")
    capture.add_argument("--output-dir", type=Path, required=True)
    capture.add_argument(
        "--end-pose-pre-delay-sec",
        type=float,
        default=END_POSE_PRE_DELAY_SEC,
        help="Sleep before first end_pose read (helps after arm motion).",
    )
    capture.add_argument(
        "--end-pose-max-attempts",
        type=int,
        default=END_POSE_MAX_ATTEMPTS,
        help="Retry count for get_end_pose on transient UNAVAILABLE.",
    )
    capture.set_defaults(func=run_capture_cli)

    execute = sub.add_parser("execute", help="Send one absolute arm + gripper command.")
    execute.add_argument("--server", default="127.0.0.1:15051")
    execute.add_argument("--arm", choices=("left", "right"), required=True)
    execute.add_argument("--joint-targets-json", required=True)
    execute.add_argument("--gripper-target", type=float, required=True)
    execute.add_argument("--output-json", type=Path, required=True)
    execute.add_argument("--settle-sec", type=float, default=1.5)
    execute.add_argument("--max-joint-delta-rad", type=float, default=0.05)
    execute.add_argument("--configure-mode", action="store_true")
    execute.add_argument("--allow-motion", action="store_true")
    execute.add_argument("--acknowledge", default="")
    execute.set_defaults(func=run_execute_cli)

    execute_traj = sub.add_parser(
        "execute_trajectory",
        help="Interpolate policy waypoints on dev machine and stream joint commands to robot.",
    )
    execute_traj.add_argument("--server", default="127.0.0.1:15051")
    execute_traj.add_argument("--arm", choices=("left", "right"), required=True)
    execute_traj.add_argument(
        "--waypoints-json",
        required=True,
        help='JSON list of {"joint_targets":[6 floats],"gripper_target":float}',
    )
    execute_traj.add_argument("--output-json", type=Path, required=True)
    execute_traj.add_argument("--control-hz", type=float, default=100.0)
    execute_traj.add_argument("--train-fps", type=float, default=TRAIN_FPS)
    execute_traj.add_argument("--trajectory-settle-sec", type=float, default=0.05)
    execute_traj.add_argument("--max-joint-delta-rad", type=float, default=0.05)
    execute_traj.add_argument("--configure-mode", action="store_true")
    execute_traj.add_argument("--allow-motion", action="store_true")
    execute_traj.add_argument("--acknowledge", default="")
    execute_traj.set_defaults(func=run_execute_trajectory_cli)

    end_pose_traj = sub.add_parser(
        "execute-end-pose-trajectory",
        help=(
            "Densify adjacent absolute end_pose waypoints once and stream set_end_pose "
            "(no mid-horizon re-read/settle; gripper jumps per policy step)."
        ),
    )
    end_pose_traj.add_argument("--server", default="127.0.0.1:15051")
    end_pose_traj.add_argument("--arm", choices=("left", "right"), required=True)
    end_pose_traj.add_argument(
        "--waypoints-json",
        required=True,
        help='JSON list of {"end_pose":{position_m,orientation_xyzw},"gripper_target":float}',
    )
    end_pose_traj.add_argument("--output-json", type=Path, required=True)
    end_pose_traj.add_argument("--interpolate-hz", type=float, default=50.0)
    end_pose_traj.add_argument("--max-linear-speed-m-s", type=float, default=0.08)
    end_pose_traj.add_argument("--max-step-m", type=float, default=0.02)
    end_pose_traj.add_argument("--min-duration-sec", type=float, default=0.0)
    end_pose_traj.add_argument("--train-fps", type=float, default=TRAIN_FPS)
    end_pose_traj.add_argument("--trajectory-settle-sec", type=float, default=0.0)
    end_pose_traj.add_argument("--configure-mode", action="store_true")
    end_pose_traj.add_argument("--allow-motion", action="store_true")
    end_pose_traj.add_argument("--acknowledge", default="")
    end_pose_traj.set_defaults(func=run_execute_end_pose_trajectory_cli)

    set_lift = sub.add_parser("set-lift", help="Move lift to absolute height (meters).")
    set_lift.add_argument("--server", default="127.0.0.1:15051")
    set_lift.add_argument("--position-m", type=float, required=True)
    set_lift.add_argument("--output-json", type=Path, required=True)
    set_lift.add_argument("--settle-sec", type=float, default=1.0)
    set_lift.add_argument("--configure-sdk-mode", action="store_true")
    set_lift.add_argument("--allow-motion", action="store_true")
    set_lift.add_argument("--acknowledge", default="")
    set_lift.set_defaults(func=run_set_lift_cli)

    set_end = sub.add_parser(
        "set-end-pose",
        help="Move one arm via interpolated set_end_pose (position_m + orientation_xyzw) + gripper.",
    )
    set_end.add_argument("--server", default="127.0.0.1:15051")
    set_end.add_argument("--arm", choices=("left", "right"), required=True)
    set_end.add_argument(
        "--end-pose-json",
        required=True,
        help='JSON: {"position_m":{x,y,z},"orientation_xyzw":{x,y,z,w}}',
    )
    set_end.add_argument("--gripper-target", type=float, required=True)
    set_end.add_argument("--output-json", type=Path, required=True)
    set_end.add_argument("--settle-sec", type=float, default=1.0)
    set_end.add_argument("--configure-mode", action="store_true")
    set_end.add_argument("--interpolate-hz", type=float, default=10.0)
    set_end.add_argument("--max-linear-speed-m-s", type=float, default=0.015)
    set_end.add_argument("--min-duration-sec", type=float, default=5.0)
    set_end.add_argument("--max-step-m", type=float, default=0.008)
    set_end.add_argument("--allow-motion", action="store_true")
    set_end.add_argument("--acknowledge", default="")
    set_end.set_defaults(func=run_set_end_pose_cli)

    set_dual = sub.add_parser(
        "set-dual-end-pose",
        help="Move BOTH arms together via synced interpolated set_end_pose (same vmax).",
    )
    set_dual.add_argument("--server", default="127.0.0.1:15051")
    set_dual.add_argument("--left-end-pose-json", required=True)
    set_dual.add_argument("--left-gripper-target", type=float, required=True)
    set_dual.add_argument("--right-end-pose-json", required=True)
    set_dual.add_argument("--right-gripper-target", type=float, required=True)
    set_dual.add_argument("--output-json", type=Path, required=True)
    set_dual.add_argument("--settle-sec", type=float, default=1.0)
    set_dual.add_argument("--configure-mode", action="store_true")
    set_dual.add_argument("--interpolate-hz", type=float, default=10.0)
    set_dual.add_argument("--max-linear-speed-m-s", type=float, default=0.015)
    set_dual.add_argument("--min-duration-sec", type=float, default=5.0)
    set_dual.add_argument("--max-step-m", type=float, default=0.008)
    set_dual.add_argument("--allow-motion", action="store_true")
    set_dual.add_argument("--acknowledge", default="")
    set_dual.set_defaults(func=run_set_dual_end_pose_cli)

    cfg_joint = sub.add_parser(
        "configure-joint-mode",
        help="Switch manipulator control mode to MANIPULATOR_JOINT_POSITIONS.",
    )
    cfg_joint.add_argument("--server", default="127.0.0.1:15051")
    cfg_joint.add_argument("--output-json", type=Path, required=True)
    cfg_joint.add_argument("--allow-motion", action="store_true")
    cfg_joint.add_argument("--acknowledge", default="")
    cfg_joint.set_defaults(func=run_configure_joint_mode_cli)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
