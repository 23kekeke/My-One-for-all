"""Biman shadow/live runner: capture -> GR00T infer -> optional SDK execution."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from quanta_biman_inference.action_decode import (
    decode_action_at_step,
    decoded_step_to_dict,
    resolve_execute_arms,
    validate_execution_horizon,
)
from quanta_biman_inference.constants import (
    CANONICAL_TASKS,
    CAPTURE_SUBPROCESS_MAX_ATTEMPTS,
    DEFAULT_CHECKPOINT,
    DEFAULT_EXECUTE_VIA,
    END_POSE_MAX_ATTEMPTS,
    END_POSE_PRE_DELAY_SEC,
    EXECUTE_SUBPROCESS_MAX_ATTEMPTS,
    LIVE_RUNS_TMP,
    MAX_SDK_CONTROL_HZ,
    OBSERVATION_DELTA_INDICES,
    POLICY_END_POSE_INTERPOLATE_HZ,
    POLICY_END_POSE_MAX_LINEAR_SPEED_M_S,
    POLICY_END_POSE_MAX_STEP_M,
    POLICY_END_POSE_MIN_DURATION_SEC,
    POLICY_END_POSE_SETTLE_SEC,
    TRAIN_FPS,
    DEFAULT_EXECUTE_INTERPOLATE_HZ,
    DEFAULT_TRAJECTORY_SETTLE_SEC,
    TASK2_NEAR_HANDLE_DEFAULT_MAX_JOINT_DELTA_RAD,
    TASK2_NEAR_HANDLE_DEFAULT_MAX_STEPS,
    TASK2_NEAR_HANDLE_DEFAULT_TOLERANCE_RAD,
    TASK2_NEAR_HANDLE_GRIPPER,
    TASK2_NEAR_HANDLE_JOINTS,
    TASK2_NEAR_HANDLE_MIN_EEF_X_M,
    TASK2_PREPOSITION_DEFAULT_MAX_JOINT_DELTA_RAD,
    TASK2_PREPOSITION_DEFAULT_MAX_STEPS,
    TASK2_PREPOSITION_DEFAULT_TOLERANCE_RAD,
    TASK2_RIGHT_ARM_START_GRIPPER,
    TASK2_RIGHT_ARM_START_JOINTS,
)
from quanta_biman_inference.live_capture import LIVE_ACK_TOKEN, clean_env, resolve_robot_python
from quanta_biman_inference.live_sdk_daemon import LiveSdkDaemonClient
from quanta_x1_inference.live_sdk_rpc import ping_daemon
from quanta_biman_inference.observation import (
    SparseTemporalBuffer,
    build_observation_from_components,
    task_text_for_index,
    validate_state_keys,
    validate_temporal_config,
)
from quanta_biman_inference.policy import load_policy, resolve_checkpoint

RUNNER_VERSION = "quanta_biman_live_runner_v1"
DEFAULT_SDK_DAEMON_URL = "127.0.0.1:15101"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


def json_float_list(values: np.ndarray | list[float]) -> str:
    return json.dumps(np.asarray(values, dtype=np.float64).astype(float).tolist())


def format_subprocess_float(value: float) -> str:
    """Format float for argparse CLI without scientific notation (e.g. -1e-05 breaks --flag)."""
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"non-finite subprocess float: {value}")
    text = f"{value:.12f}".rstrip("0").rstrip(".")
    if text in ("", "-"):
        return "0"
    return text


def run_capture_subprocess(
    *,
    robot_python: Path,
    server: str,
    output_dir: Path,
    end_pose_pre_delay_sec: float = 0.10,
    end_pose_max_attempts: int = 6,
    max_subprocess_attempts: int = 3,
) -> tuple[dict[str, Any], str]:
    command = [
        str(robot_python),
        "-m",
        "quanta_biman_inference.live_capture",
        "capture",
        "--server",
        server,
        "--output-dir",
        str(output_dir),
        "--end-pose-pre-delay-sec",
        str(end_pose_pre_delay_sec),
        "--end-pose-max-attempts",
        str(end_pose_max_attempts),
    ]

    last_error: RuntimeError | None = None
    for attempt in range(1, max_subprocess_attempts + 1):
        finished = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=clean_env(),
            check=False,
        )
        log = finished.stdout
        if finished.returncode == 0:
            capture_json = output_dir / "capture.json"
            if not capture_json.is_file():
                last_error = RuntimeError(
                    f"Capture output missing: {capture_json}\n{log[-2000:]}"
                )
            else:
                report = json.loads(capture_json.read_text(encoding="utf-8"))
                if attempt > 1:
                    report.setdefault("capture_subprocess_retry", {})["attempt"] = attempt
                return report, log

        last_error = RuntimeError(
            f"Capture subprocess failed with code {finished.returncode} "
            f"(attempt {attempt}/{max_subprocess_attempts}):\n{log[-4000:]}"
        )
        if attempt < max_subprocess_attempts and "end_pose" in log.lower():
            time.sleep(0.25 * attempt)
            continue
        raise last_error

    raise last_error or RuntimeError("Capture subprocess failed")


def run_execute_subprocess(
    *,
    robot_python: Path,
    server: str,
    arm: str,
    joint_targets: np.ndarray,
    gripper_target: float,
    output_json: Path,
    settle_sec: float,
    max_joint_delta_rad: float,
    configure_mode: bool,
    max_subprocess_attempts: int = 3,
) -> tuple[dict[str, Any], str]:
    command = [
        str(robot_python),
        "-m",
        "quanta_biman_inference.live_capture",
        "execute",
        "--server",
        server,
        "--arm",
        arm,
        "--joint-targets-json",
        json_float_list(joint_targets),
        f"--gripper-target={format_subprocess_float(gripper_target)}",
        "--output-json",
        str(output_json),
        f"--settle-sec={format_subprocess_float(settle_sec)}",
        f"--max-joint-delta-rad={format_subprocess_float(max_joint_delta_rad)}",
        "--allow-motion",
        "--acknowledge",
        LIVE_ACK_TOKEN,
    ]
    if configure_mode:
        command.append("--configure-mode")

    last_error: RuntimeError | None = None
    for attempt in range(1, max_subprocess_attempts + 1):
        finished = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=clean_env(),
            check=False,
        )
        log = finished.stdout
        if finished.returncode == 0:
            if not output_json.is_file():
                last_error = RuntimeError(f"Execute output missing: {output_json}\n{log[-2000:]}")
            else:
                result = json.loads(output_json.read_text(encoding="utf-8"))
                if attempt > 1:
                    result.setdefault("execute_subprocess_retry", {})["attempt"] = attempt
                return result, log

        last_error = RuntimeError(
            f"Execute subprocess failed with code {finished.returncode} "
            f"(attempt {attempt}/{max_subprocess_attempts}):\n{log[-4000:]}"
        )
        retryable = any(
            token in log.lower()
            for token in ("joint states", "end_pose", "unavailable", "no recent")
        )
        if attempt < max_subprocess_attempts and retryable:
            time.sleep(0.25 * attempt)
            continue
        raise last_error

    raise last_error or RuntimeError("Execute subprocess failed")


def policy_waypoints_for_arm(planned_steps: list[dict[str, Any]], arm: str) -> list[dict[str, Any]]:
    return [
        {
            "joint_targets": step[arm]["sdk_joint_targets_rad"],
            "gripper_target": float(step[arm]["gripper_position"]),
        }
        for step in planned_steps
    ]


def policy_end_pose_waypoints_for_arm(
    planned_steps: list[dict[str, Any]], arm: str
) -> list[dict[str, Any]]:
    return [
        {
            "end_pose": step[arm]["end_pose"],
            "gripper_target": float(step[arm]["gripper_position"]),
        }
        for step in planned_steps
    ]


def run_execute_trajectory_subprocess(
    *,
    robot_python: Path,
    server: str,
    arm: str,
    waypoints: list[dict[str, Any]],
    output_json: Path,
    control_hz: float,
    train_fps: float,
    trajectory_settle_sec: float,
    max_joint_delta_rad: float,
    configure_mode: bool,
    max_subprocess_attempts: int = 3,
) -> tuple[dict[str, Any], str]:
    command = [
        str(robot_python),
        "-m",
        "quanta_biman_inference.live_capture",
        "execute_trajectory",
        "--server",
        server,
        "--arm",
        arm,
        "--waypoints-json",
        json.dumps(waypoints, ensure_ascii=False),
        "--output-json",
        str(output_json),
        f"--control-hz={format_subprocess_float(control_hz)}",
        f"--train-fps={format_subprocess_float(train_fps)}",
        f"--trajectory-settle-sec={format_subprocess_float(trajectory_settle_sec)}",
        f"--max-joint-delta-rad={format_subprocess_float(max_joint_delta_rad)}",
        "--allow-motion",
        "--acknowledge",
        LIVE_ACK_TOKEN,
    ]
    if configure_mode:
        command.append("--configure-mode")

    last_error: RuntimeError | None = None
    for attempt in range(1, max_subprocess_attempts + 1):
        finished = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=clean_env(),
            check=False,
        )
        log = finished.stdout
        if finished.returncode == 0:
            if not output_json.is_file():
                last_error = RuntimeError(
                    f"Execute trajectory output missing: {output_json}\n{log[-2000:]}"
                )
            else:
                result = json.loads(output_json.read_text(encoding="utf-8"))
                if attempt > 1:
                    result.setdefault("execute_subprocess_retry", {})["attempt"] = attempt
                return result, log

        last_error = RuntimeError(
            f"Execute trajectory subprocess failed with code {finished.returncode} "
            f"(attempt {attempt}/{max_subprocess_attempts}):\n{log[-4000:]}"
        )
        if attempt < max_subprocess_attempts:
            time.sleep(0.25 * attempt)
            continue
        raise last_error

    raise last_error or RuntimeError("Execute trajectory subprocess failed")


def run_execute_trajectory_backend(
    *,
    sdk_backend: str,
    daemon_client: LiveSdkDaemonClient | None,
    robot_python: Path,
    server: str,
    arm: str,
    waypoints: list[dict[str, Any]],
    output_json: Path,
    control_hz: float,
    train_fps: float,
    trajectory_settle_sec: float,
    max_joint_delta_rad: float,
    configure_mode: bool,
    max_subprocess_attempts: int,
) -> tuple[dict[str, Any], str]:
    if sdk_backend == "daemon" and daemon_client is not None:
        return daemon_client.execute_trajectory(
            arm=arm,
            waypoints=waypoints,
            control_hz=float(control_hz),
            train_fps=float(train_fps),
            trajectory_settle_sec=float(trajectory_settle_sec),
            max_joint_delta_rad=float(max_joint_delta_rad),
            configure_mode=bool(configure_mode),
            output_json=output_json,
        )
    return run_execute_trajectory_subprocess(
        robot_python=robot_python,
        server=server,
        arm=arm,
        waypoints=waypoints,
        output_json=output_json,
        control_hz=control_hz,
        train_fps=train_fps,
        trajectory_settle_sec=trajectory_settle_sec,
        max_joint_delta_rad=max_joint_delta_rad,
        configure_mode=configure_mode,
        max_subprocess_attempts=max_subprocess_attempts,
    )


def run_execute_end_pose_trajectory_subprocess(
    *,
    robot_python: Path,
    server: str,
    arm: str,
    waypoints: list[dict[str, Any]],
    output_json: Path,
    interpolate_hz: float,
    max_linear_speed_m_s: float,
    max_step_m: float,
    min_duration_sec: float,
    train_fps: float,
    trajectory_settle_sec: float,
    configure_mode: bool,
    max_subprocess_attempts: int = 3,
) -> tuple[dict[str, Any], str]:
    command = [
        str(robot_python),
        "-m",
        "quanta_biman_inference.live_capture",
        "execute-end-pose-trajectory",
        "--server",
        server,
        "--arm",
        arm,
        "--waypoints-json",
        json.dumps(waypoints, ensure_ascii=False),
        "--output-json",
        str(output_json),
        "--interpolate-hz",
        format_subprocess_float(interpolate_hz),
        "--max-linear-speed-m-s",
        format_subprocess_float(max_linear_speed_m_s),
        "--max-step-m",
        format_subprocess_float(max_step_m),
        "--min-duration-sec",
        format_subprocess_float(min_duration_sec),
        "--train-fps",
        format_subprocess_float(train_fps),
        "--trajectory-settle-sec",
        format_subprocess_float(trajectory_settle_sec),
        "--allow-motion",
        "--acknowledge",
        LIVE_ACK_TOKEN,
    ]
    if configure_mode:
        command.append("--configure-mode")

    last_error: RuntimeError | None = None
    for attempt in range(1, max_subprocess_attempts + 1):
        finished = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=clean_env(),
            check=False,
        )
        log = finished.stdout
        if finished.returncode == 0 and output_json.is_file():
            return json.loads(output_json.read_text(encoding="utf-8")), log
        last_error = RuntimeError(
            f"execute-end-pose-trajectory failed "
            f"(attempt {attempt}/{max_subprocess_attempts}):\n{log}"
        )
        if attempt < max_subprocess_attempts:
            time.sleep(0.25 * attempt)
            continue
        raise last_error
    assert last_error is not None
    raise last_error


def run_execute_end_pose_trajectory_backend(
    *,
    sdk_backend: str,
    daemon_client: LiveSdkDaemonClient | None,
    robot_python: Path,
    server: str,
    arm: str,
    waypoints: list[dict[str, Any]],
    output_json: Path,
    interpolate_hz: float,
    max_linear_speed_m_s: float,
    max_step_m: float,
    min_duration_sec: float,
    train_fps: float,
    trajectory_settle_sec: float,
    configure_mode: bool,
    max_subprocess_attempts: int,
) -> tuple[dict[str, Any], str]:
    if sdk_backend == "daemon" and daemon_client is not None:
        return daemon_client.execute_end_pose_trajectory(
            arm=arm,
            waypoints=waypoints,
            interpolate_hz=float(interpolate_hz),
            max_linear_speed_m_s=float(max_linear_speed_m_s),
            max_step_m=float(max_step_m),
            min_duration_sec=float(min_duration_sec),
            train_fps=float(train_fps),
            trajectory_settle_sec=float(trajectory_settle_sec),
            configure_mode=bool(configure_mode),
            output_json=output_json,
        )
    return run_execute_end_pose_trajectory_subprocess(
        robot_python=robot_python,
        server=server,
        arm=arm,
        waypoints=waypoints,
        output_json=output_json,
        interpolate_hz=interpolate_hz,
        max_linear_speed_m_s=max_linear_speed_m_s,
        max_step_m=max_step_m,
        min_duration_sec=min_duration_sec,
        train_fps=train_fps,
        trajectory_settle_sec=trajectory_settle_sec,
        configure_mode=configure_mode,
        max_subprocess_attempts=max_subprocess_attempts,
    )


def resolve_sdk_backend(args: argparse.Namespace) -> tuple[str, LiveSdkDaemonClient | None]:
    backend = str(args.sdk_backend)
    if backend == "subprocess":
        return backend, None
    if backend != "daemon":
        raise ValueError("--sdk-backend must be daemon or subprocess")

    url = str(args.sdk_daemon_url)
    print(f"Checking SDK daemon at {url} ...", flush=True)
    if not ping_daemon(url, default_port=15101, timeout_sec=2.0):
        if args.sdk_backend_required:
            raise RuntimeError(
                f"SDK daemon not reachable at {url}. "
                "Start: xr_lerobot python -m quanta_biman_inference.live_sdk_daemon serve ..."
            )
        print(
            f"WARNING: SDK daemon not reachable at {url}; falling back to subprocess.",
            flush=True,
        )
        return "subprocess", None
    return "daemon", LiveSdkDaemonClient(url)


def run_capture_backend(
    *,
    sdk_backend: str,
    daemon_client: LiveSdkDaemonClient | None,
    robot_python: Path,
    server: str,
    output_dir: Path,
    end_pose_pre_delay_sec: float,
    end_pose_max_attempts: int,
    max_subprocess_attempts: int,
) -> tuple[dict[str, Any], str]:
    if sdk_backend == "daemon" and daemon_client is not None:
        last_error: Exception | None = None
        for attempt in range(1, max(1, int(max_subprocess_attempts)) + 1):
            try:
                report, log = daemon_client.capture(output_dir=output_dir)
                if attempt > 1:
                    report.setdefault("capture_daemon_retry", {})["attempt"] = attempt
                    log = f"{log}capture_daemon_retry attempt={attempt}\n"
                return report, log
            except Exception as exc:
                last_error = exc
                msg = repr(exc).lower()
                retryable = (
                    "unavailable" in msg
                    or "connection reset" in msg
                    or "recvmsg" in msg
                    or "broken pipe" in msg
                    or "rpc failed" in msg
                )
                if attempt < max_subprocess_attempts and retryable:
                    print(
                        f"[capture] daemon retry {attempt}/{max_subprocess_attempts}: {exc}",
                        flush=True,
                    )
                    time.sleep(0.35 * attempt)
                    continue
                raise
        raise last_error or RuntimeError("daemon capture failed")
    return run_capture_subprocess(
        robot_python=robot_python,
        server=server,
        output_dir=output_dir,
        end_pose_pre_delay_sec=end_pose_pre_delay_sec,
        end_pose_max_attempts=end_pose_max_attempts,
        max_subprocess_attempts=max_subprocess_attempts,
    )


def run_execute_backend(
    *,
    sdk_backend: str,
    daemon_client: LiveSdkDaemonClient | None,
    robot_python: Path,
    server: str,
    arm: str,
    joint_targets: np.ndarray,
    gripper_target: float,
    output_json: Path,
    settle_sec: float,
    max_joint_delta_rad: float,
    configure_mode: bool,
    max_subprocess_attempts: int,
) -> tuple[dict[str, Any], str]:
    if sdk_backend == "daemon" and daemon_client is not None:
        return daemon_client.execute(
            arm=arm,
            joint_targets=joint_targets.astype(float).tolist(),
            gripper_target=float(gripper_target),
            settle_sec=float(settle_sec),
            max_joint_delta_rad=float(max_joint_delta_rad),
            configure_mode=bool(configure_mode),
            output_json=output_json,
        )
    return run_execute_subprocess(
        robot_python=robot_python,
        server=server,
        arm=arm,
        joint_targets=joint_targets,
        gripper_target=gripper_target,
        output_json=output_json,
        settle_sec=settle_sec,
        max_joint_delta_rad=max_joint_delta_rad,
        configure_mode=configure_mode,
        max_subprocess_attempts=max_subprocess_attempts,
    )


def run_set_end_pose_subprocess(
    *,
    robot_python: Path,
    server: str,
    arm: str,
    end_pose: dict[str, Any],
    gripper_target: float,
    output_json: Path,
    settle_sec: float,
    configure_mode: bool,
    interpolate_hz: float,
    max_linear_speed_m_s: float,
    min_duration_sec: float,
    max_step_m: float,
    max_subprocess_attempts: int = EXECUTE_SUBPROCESS_MAX_ATTEMPTS,
) -> tuple[dict[str, Any], str]:
    command = [
        str(robot_python),
        "-m",
        "quanta_biman_inference.live_capture",
        "set-end-pose",
        "--server",
        server,
        "--arm",
        arm,
        "--end-pose-json",
        json.dumps(end_pose, ensure_ascii=False),
        "--gripper-target",
        format_subprocess_float(gripper_target),
        "--output-json",
        str(output_json),
        "--settle-sec",
        format_subprocess_float(settle_sec),
        "--interpolate-hz",
        format_subprocess_float(interpolate_hz),
        "--max-linear-speed-m-s",
        format_subprocess_float(max_linear_speed_m_s),
        "--min-duration-sec",
        format_subprocess_float(min_duration_sec),
        "--max-step-m",
        format_subprocess_float(max_step_m),
        "--allow-motion",
        "--acknowledge",
        LIVE_ACK_TOKEN,
    ]
    if configure_mode:
        command.append("--configure-mode")

    last_error: RuntimeError | None = None
    for attempt in range(1, max_subprocess_attempts + 1):
        finished = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=clean_env(),
            check=False,
        )
        log = finished.stdout
        if finished.returncode == 0 and output_json.is_file():
            return json.loads(output_json.read_text(encoding="utf-8")), log
        last_error = RuntimeError(
            f"set-end-pose subprocess failed (attempt {attempt}/{max_subprocess_attempts}):\n{log}"
        )
    assert last_error is not None
    raise last_error


def run_set_end_pose_backend(
    *,
    sdk_backend: str,
    daemon_client: LiveSdkDaemonClient | None,
    robot_python: Path,
    server: str,
    arm: str,
    end_pose: dict[str, Any],
    gripper_target: float,
    output_json: Path,
    settle_sec: float,
    configure_mode: bool,
    interpolate_hz: float,
    max_linear_speed_m_s: float,
    min_duration_sec: float,
    max_step_m: float,
    max_subprocess_attempts: int,
) -> tuple[dict[str, Any], str]:
    if sdk_backend == "daemon" and daemon_client is not None:
        return daemon_client.set_end_pose(
            arm=arm,
            end_pose=end_pose,
            gripper_target=float(gripper_target),
            settle_sec=float(settle_sec),
            configure_mode=bool(configure_mode),
            output_json=output_json,
            interpolate_hz=float(interpolate_hz),
            max_linear_speed_m_s=float(max_linear_speed_m_s),
            min_duration_sec=float(min_duration_sec),
            max_step_m=float(max_step_m),
        )
    return run_set_end_pose_subprocess(
        robot_python=robot_python,
        server=server,
        arm=arm,
        end_pose=end_pose,
        gripper_target=gripper_target,
        output_json=output_json,
        settle_sec=settle_sec,
        configure_mode=configure_mode,
        interpolate_hz=interpolate_hz,
        max_linear_speed_m_s=max_linear_speed_m_s,
        min_duration_sec=min_duration_sec,
        max_step_m=max_step_m,
        max_subprocess_attempts=max_subprocess_attempts,
    )


def run_set_dual_end_pose_subprocess(
    *,
    robot_python: Path,
    server: str,
    left_end_pose: dict[str, Any],
    left_gripper_target: float,
    right_end_pose: dict[str, Any],
    right_gripper_target: float,
    output_json: Path,
    settle_sec: float,
    configure_mode: bool,
    interpolate_hz: float,
    max_linear_speed_m_s: float,
    min_duration_sec: float,
    max_step_m: float,
    max_subprocess_attempts: int = EXECUTE_SUBPROCESS_MAX_ATTEMPTS,
) -> tuple[dict[str, Any], str]:
    command = [
        str(robot_python),
        "-m",
        "quanta_biman_inference.live_capture",
        "set-dual-end-pose",
        "--server",
        server,
        "--left-end-pose-json",
        json.dumps(left_end_pose, ensure_ascii=False),
        "--left-gripper-target",
        format_subprocess_float(left_gripper_target),
        "--right-end-pose-json",
        json.dumps(right_end_pose, ensure_ascii=False),
        "--right-gripper-target",
        format_subprocess_float(right_gripper_target),
        "--output-json",
        str(output_json),
        "--settle-sec",
        format_subprocess_float(settle_sec),
        "--interpolate-hz",
        format_subprocess_float(interpolate_hz),
        "--max-linear-speed-m-s",
        format_subprocess_float(max_linear_speed_m_s),
        "--min-duration-sec",
        format_subprocess_float(min_duration_sec),
        "--max-step-m",
        format_subprocess_float(max_step_m),
        "--allow-motion",
        "--acknowledge",
        LIVE_ACK_TOKEN,
    ]
    if configure_mode:
        command.append("--configure-mode")

    last_error: RuntimeError | None = None
    for attempt in range(1, max_subprocess_attempts + 1):
        finished = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=clean_env(),
            check=False,
        )
        log = finished.stdout
        if finished.returncode == 0 and output_json.is_file():
            return json.loads(output_json.read_text(encoding="utf-8")), log
        last_error = RuntimeError(
            f"set-dual-end-pose subprocess failed "
            f"(attempt {attempt}/{max_subprocess_attempts}):\n{log}"
        )
    assert last_error is not None
    raise last_error


def run_set_dual_end_pose_backend(
    *,
    sdk_backend: str,
    daemon_client: LiveSdkDaemonClient | None,
    robot_python: Path,
    server: str,
    left_end_pose: dict[str, Any],
    left_gripper_target: float,
    right_end_pose: dict[str, Any],
    right_gripper_target: float,
    output_json: Path,
    settle_sec: float,
    configure_mode: bool,
    interpolate_hz: float,
    max_linear_speed_m_s: float,
    min_duration_sec: float,
    max_step_m: float,
    max_subprocess_attempts: int,
) -> tuple[dict[str, Any], str]:
    if sdk_backend == "daemon" and daemon_client is not None:
        return daemon_client.set_dual_end_pose(
            left_end_pose=left_end_pose,
            left_gripper_target=float(left_gripper_target),
            right_end_pose=right_end_pose,
            right_gripper_target=float(right_gripper_target),
            settle_sec=float(settle_sec),
            configure_mode=bool(configure_mode),
            output_json=output_json,
            interpolate_hz=float(interpolate_hz),
            max_linear_speed_m_s=float(max_linear_speed_m_s),
            min_duration_sec=float(min_duration_sec),
            max_step_m=float(max_step_m),
        )
    return run_set_dual_end_pose_subprocess(
        robot_python=robot_python,
        server=server,
        left_end_pose=left_end_pose,
        left_gripper_target=left_gripper_target,
        right_end_pose=right_end_pose,
        right_gripper_target=right_gripper_target,
        output_json=output_json,
        settle_sec=settle_sec,
        configure_mode=configure_mode,
        interpolate_hz=interpolate_hz,
        max_linear_speed_m_s=max_linear_speed_m_s,
        min_duration_sec=min_duration_sec,
        max_step_m=max_step_m,
        max_subprocess_attempts=max_subprocess_attempts,
    )


def planned_steps_from_action(
    action_dict: dict[str, np.ndarray],
    *,
    execution_horizon: int,
) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    for step_index in range(execution_horizon):
        decoded = decode_action_at_step(action_dict, step_index)
        steps.append({"step_index": step_index, **decoded_step_to_dict(decoded)})
    return steps


def _right_joint_error_rad(current_joints: np.ndarray, target_joints: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(current_joints, dtype=np.float32) - target_joints))


def _preposition_reached(
    *,
    joint_error: float,
    tolerance_rad: float,
    right_eef_x: float | None,
    min_eef_x_m: float | None,
) -> bool:
    if min_eef_x_m is not None and right_eef_x is not None and right_eef_x >= min_eef_x_m:
        return True
    return joint_error <= tolerance_rad


def resolve_task2_preposition_limits(
    args: argparse.Namespace,
    *,
    mode: str,
) -> tuple[int, float, float, float | None]:
    max_steps = int(args.preposition_max_steps)
    max_joint_delta_rad = float(args.preposition_max_joint_delta_rad)
    tolerance_rad = float(args.preposition_tolerance_rad)
    min_eef_x_m: float | None = None

    if mode == "near_handle":
        if max_steps == TASK2_PREPOSITION_DEFAULT_MAX_STEPS:
            max_steps = TASK2_NEAR_HANDLE_DEFAULT_MAX_STEPS
        if max_joint_delta_rad == TASK2_PREPOSITION_DEFAULT_MAX_JOINT_DELTA_RAD:
            max_joint_delta_rad = TASK2_NEAR_HANDLE_DEFAULT_MAX_JOINT_DELTA_RAD
        if tolerance_rad == TASK2_PREPOSITION_DEFAULT_TOLERANCE_RAD:
            tolerance_rad = TASK2_NEAR_HANDLE_DEFAULT_TOLERANCE_RAD
        min_eef_x_m = float(args.preposition_min_eef_x_m)
    return max_steps, max_joint_delta_rad, tolerance_rad, min_eef_x_m


def run_task2_preposition(
    *,
    mode: str,
    robot_python: Path,
    server: str,
    sdk_backend: str,
    daemon_client: LiveSdkDaemonClient | None,
    run_root: Path,
    settle_sec: float,
    max_joint_delta_rad: float,
    tolerance_rad: float,
    max_steps: int,
    min_eef_x_m: float | None,
    configure_mode_once: bool,
) -> tuple[dict[str, Any], bool]:
    """Move right arm to a task2 SDK pre-position target before cycle 1 (left arm untouched)."""
    if mode == "near_handle":
        target_joints = np.asarray(TASK2_NEAR_HANDLE_JOINTS, dtype=np.float32)
        target_gripper = float(TASK2_NEAR_HANDLE_GRIPPER)
        preposition_dir = run_root / "preposition_task2_near_handle"
        label = "near handle (eef x>=0.38 zone)"
    elif mode == "home":
        target_joints = np.asarray(TASK2_RIGHT_ARM_START_JOINTS, dtype=np.float32)
        target_gripper = float(TASK2_RIGHT_ARM_START_GRIPPER)
        preposition_dir = run_root / "preposition_task2_home"
        label = "training home"
    else:
        raise ValueError(f"unknown task2 preposition mode: {mode}")

    preposition_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "preposition_mode": mode,
        "target_joints_rad": target_joints.astype(float).tolist(),
        "target_gripper": target_gripper,
        "tolerance_rad": float(tolerance_rad),
        "max_joint_delta_rad": float(max_joint_delta_rad),
        "max_steps": int(max_steps),
        "min_eef_x_m": min_eef_x_m,
        "steps": [],
    }

    eef_hint = f", min_eef_x={min_eef_x_m:.2f}m" if min_eef_x_m is not None else ""
    print(
        f"[task2 preposition:{mode}] right arm -> {label} (max {max_steps} steps, "
        f"tol={tolerance_rad:.3f} rad, delta={max_joint_delta_rad:.3f}{eef_hint})",
        flush=True,
    )

    for step_index in range(max_steps):
        capture_dir = preposition_dir / f"step_{step_index:02d}" / "capture"
        capture_report, capture_log = run_capture_backend(
            sdk_backend=sdk_backend,
            daemon_client=daemon_client,
            robot_python=robot_python,
            server=server,
            output_dir=capture_dir,
            end_pose_pre_delay_sec=END_POSE_PRE_DELAY_SEC,
            end_pose_max_attempts=END_POSE_MAX_ATTEMPTS,
            max_subprocess_attempts=CAPTURE_SUBPROCESS_MAX_ATTEMPTS,
        )
        (preposition_dir / f"step_{step_index:02d}_capture_stdout.log").write_text(
            capture_log,
            encoding="utf-8",
        )
        components = capture_report["components"]
        current_joints = np.asarray(components["right_joint_position"], dtype=np.float32)
        right_eef_x = float(np.asarray(components["right_eef_9d"], dtype=np.float32)[0])
        joint_error = _right_joint_error_rad(current_joints, target_joints)
        step_record: dict[str, Any] = {
            "step_index": step_index,
            "before_joints_rad": current_joints.astype(float).tolist(),
            "before_eef_x_m": right_eef_x,
            "joint_error_l2_rad": joint_error,
        }

        if _preposition_reached(
            joint_error=joint_error,
            tolerance_rad=tolerance_rad,
            right_eef_x=right_eef_x,
            min_eef_x_m=min_eef_x_m,
        ):
            step_record["status"] = "reached"
            report["steps"].append(step_record)
            report["ok"] = True
            report["final_joint_error_l2_rad"] = joint_error
            report["final_eef_x_m"] = right_eef_x
            report["steps_used"] = step_index
            write_json(preposition_dir / "preposition_report.json", report)
            print(
                f"[task2 preposition:{mode}] reached in {step_index} step(s) "
                f"(L2={joint_error:.4f}, eef_x={right_eef_x:.3f})"
            )
            return report, configure_mode_once

        execute_json = preposition_dir / f"step_{step_index:02d}_execute.json"
        execute_result, execute_log = run_execute_backend(
            sdk_backend=sdk_backend,
            daemon_client=daemon_client,
            robot_python=robot_python,
            server=server,
            arm="right",
            joint_targets=target_joints,
            gripper_target=target_gripper,
            output_json=execute_json,
            settle_sec=settle_sec,
            max_joint_delta_rad=max_joint_delta_rad,
            configure_mode=configure_mode_once,
            max_subprocess_attempts=EXECUTE_SUBPROCESS_MAX_ATTEMPTS,
        )
        (preposition_dir / f"step_{step_index:02d}_execute_stdout.log").write_text(
            execute_log,
            encoding="utf-8",
        )
        configure_mode_once = False
        step_record["status"] = "executed"
        step_record["execute_json"] = str(execute_json.resolve())
        step_record["command_max_abs_delta_rad"] = execute_result.get("max_abs_command_delta_rad")
        step_record["after_joints_rad"] = execute_result.get("after_joints_rad")
        report["steps"].append(step_record)

    last_step = report["steps"][-1]
    report["ok"] = False
    report["final_joint_error_l2_rad"] = last_step["joint_error_l2_rad"]
    report["final_eef_x_m"] = last_step.get("before_eef_x_m")
    report["steps_used"] = max_steps
    write_json(preposition_dir / "preposition_report.json", report)
    print(
        f"[task2 preposition:{mode}] WARNING: did not reach target after {max_steps} steps "
        f"(L2={report['final_joint_error_l2_rad']:.4f}, "
        f"eef_x={report.get('final_eef_x_m')})",
        flush=True,
    )
    return report, configure_mode_once


def _load_saved_camera(path: str) -> np.ndarray:
    import cv2

    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Failed to read camera image: {path}")
    return np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def run_live_cycle(
    *,
    policy: Any,
    robot_python: Path,
    server: str,
    sdk_backend: str,
    daemon_client: LiveSdkDaemonClient | None,
    cycle_dir: Path,
    task_text: str,
    task_index: int | None,
    mode: str,
    execute: bool,
    execute_arms: str,
    execution_horizon: int,
    settle_sec: float,
    max_joint_delta_rad: float,
    execute_interpolate_hz: float,
    trajectory_settle_sec: float,
    train_fps: float,
    configure_mode_once: bool,
    temporal_buffer: SparseTemporalBuffer,
    execute_via: str = DEFAULT_EXECUTE_VIA,
    end_pose_interpolate_hz: float = POLICY_END_POSE_INTERPOLATE_HZ,
    end_pose_max_linear_speed_m_s: float = POLICY_END_POSE_MAX_LINEAR_SPEED_M_S,
    end_pose_min_duration_sec: float = POLICY_END_POSE_MIN_DURATION_SEC,
    end_pose_max_step_m: float = POLICY_END_POSE_MAX_STEP_M,
    end_pose_settle_sec: float = POLICY_END_POSE_SETTLE_SEC,
) -> tuple[dict[str, Any], bool]:
    capture_dir = cycle_dir / "capture"
    capture_report, capture_log = run_capture_backend(
        sdk_backend=sdk_backend,
        daemon_client=daemon_client,
        robot_python=robot_python,
        server=server,
        output_dir=capture_dir,
        end_pose_pre_delay_sec=END_POSE_PRE_DELAY_SEC,
        end_pose_max_attempts=END_POSE_MAX_ATTEMPTS,
        max_subprocess_attempts=CAPTURE_SUBPROCESS_MAX_ATTEMPTS,
    )
    (cycle_dir / "capture_stdout.log").write_text(capture_log, encoding="utf-8")

    components = capture_report["components"]
    image_paths = capture_report["image_paths"]
    flat, parsed = build_observation_from_components(
        head_camera=_load_saved_camera(image_paths["head_camera"]),
        left_arm_camera=_load_saved_camera(image_paths["left_arm_camera"]),
        right_arm_camera=_load_saved_camera(image_paths["right_arm_camera"]),
        left_eef_9d=np.asarray(components["left_eef_9d"], dtype=np.float32),
        left_gripper_position=float(components["left_gripper_position"]),
        left_joint_position=np.asarray(components["left_joint_position"], dtype=np.float32),
        right_eef_9d=np.asarray(components["right_eef_9d"], dtype=np.float32),
        right_gripper_position=float(components["right_gripper_position"]),
        right_joint_position=np.asarray(components["right_joint_position"], dtype=np.float32),
        task_text=task_text,
        modality_configs=policy.modality_configs,
        temporal_buffer=temporal_buffer,
    )

    infer_start = time.perf_counter()
    action_dict, _info = policy.get_action(parsed)
    inference_sec = time.perf_counter() - infer_start

    planned_steps = planned_steps_from_action(
        action_dict,
        execution_horizon=execution_horizon,
    )
    arms_to_execute = resolve_execute_arms(execute_arms, task_index=task_index)
    execute_via_norm = str(execute_via).lower().strip()

    report: dict[str, Any] = {
        "runner_version": RUNNER_VERSION,
        "cycle_dir": str(cycle_dir.resolve()),
        "mode": mode,
        "execute_requested": bool(execute),
        "execute_via": execute_via_norm,
        "execute_arms": execute_arms,
        "arms_to_execute": list(arms_to_execute),
        "execution_horizon": execution_horizon,
        "execute_interpolate_hz": float(execute_interpolate_hz),
        "trajectory_settle_sec": float(trajectory_settle_sec),
        "train_fps": float(train_fps),
        "task_index": task_index,
        "task_text": task_text,
        "capture_report_path": str((capture_dir / "capture.json").resolve()),
        "capture_timing_ms": capture_report.get("whole_request_wall_elapsed_ms"),
        "sdk_backend": sdk_backend,
        "observation_state32": components,
        "temporal_buffer_len": len(temporal_buffer),
        "temporal_delta_indices": list(temporal_buffer.delta_indices),
        "inference_sec": float(inference_sec),
        "planned_steps": planned_steps,
        "decision": "SHADOW_LOG_ONLY",
        "executed_steps": [],
    }

    motion_enabled = mode == "live" and execute and bool(arms_to_execute)
    if not motion_enabled:
        report["reason"] = "shadow_or_dry_run_or_no_execute_arms"
        write_json(cycle_dir / "cycle_report.json", report)
        return report, configure_mode_once

    executed_steps: list[dict[str, Any]] = []

    if execute_via_norm == "end_pose":
        report["end_pose_interpolate_hz"] = float(end_pose_interpolate_hz)
        report["end_pose_max_linear_speed_m_s"] = float(end_pose_max_linear_speed_m_s)
        report["end_pose_min_duration_sec"] = float(end_pose_min_duration_sec)
        report["end_pose_max_step_m"] = float(end_pose_max_step_m)
        report["end_pose_execution_mode"] = "trajectory"
        # One RPC per arm for the whole horizon: densify adjacent policy eefs,
        # no mid-horizon re-read/settle; gripper jumps per policy step.
        step_executions: list[dict[str, Any]] = []
        for arm in arms_to_execute:
            waypoints = policy_end_pose_waypoints_for_arm(planned_steps, arm)
            execute_json = cycle_dir / f"execute_{arm}_end_pose_trajectory.json"
            execute_result, execute_log = run_execute_end_pose_trajectory_backend(
                sdk_backend=sdk_backend,
                daemon_client=daemon_client,
                robot_python=robot_python,
                server=server,
                arm=arm,
                waypoints=waypoints,
                output_json=execute_json,
                interpolate_hz=float(end_pose_interpolate_hz),
                max_linear_speed_m_s=float(end_pose_max_linear_speed_m_s),
                max_step_m=float(end_pose_max_step_m),
                min_duration_sec=float(end_pose_min_duration_sec),
                train_fps=float(train_fps),
                trajectory_settle_sec=float(end_pose_settle_sec),
                configure_mode=configure_mode_once,
                max_subprocess_attempts=EXECUTE_SUBPROCESS_MAX_ATTEMPTS,
            )
            (cycle_dir / f"execute_{arm}_end_pose_trajectory_stdout.log").write_text(
                execute_log,
                encoding="utf-8",
            )
            step_executions.append(
                {
                    "arm": arm,
                    "execute_json": str(execute_json.resolve()),
                    "readback": execute_result,
                }
            )
            configure_mode_once = False
        executed_steps.append(
            {
                "step_index": "trajectory",
                "execution_mode": "end_pose_trajectory",
                "policy_waypoints": int(execution_horizon),
                "executions": step_executions,
            }
        )
    else:
        use_trajectory = execute_interpolate_hz > 0
        if use_trajectory:
            for arm in arms_to_execute:
                waypoints = policy_waypoints_for_arm(planned_steps, arm)
                execute_json = cycle_dir / f"execute_{arm}_trajectory.json"
                execute_result, execute_log = run_execute_trajectory_backend(
                    sdk_backend=sdk_backend,
                    daemon_client=daemon_client,
                    robot_python=robot_python,
                    server=server,
                    arm=arm,
                    waypoints=waypoints,
                    output_json=execute_json,
                    control_hz=execute_interpolate_hz,
                    train_fps=train_fps,
                    trajectory_settle_sec=trajectory_settle_sec,
                    max_joint_delta_rad=max_joint_delta_rad,
                    configure_mode=configure_mode_once,
                    max_subprocess_attempts=EXECUTE_SUBPROCESS_MAX_ATTEMPTS,
                )
                (cycle_dir / f"execute_{arm}_trajectory_stdout.log").write_text(
                    execute_log,
                    encoding="utf-8",
                )
                configure_mode_once = False
                executed_steps.append(
                    {
                        "step_index": "trajectory",
                        "execution_mode": "interpolated_trajectory",
                        "executions": [
                            {
                                "arm": arm,
                                "execute_json": str(execute_json.resolve()),
                                "readback": execute_result,
                            }
                        ],
                    }
                )
        else:
            for step in planned_steps:
                step_executions: list[dict[str, Any]] = []
                for arm in arms_to_execute:
                    arm_payload = step[arm]
                    execute_json = cycle_dir / f"execute_{arm}_step_{step['step_index']:02d}.json"
                    execute_result, execute_log = run_execute_backend(
                        sdk_backend=sdk_backend,
                        daemon_client=daemon_client,
                        robot_python=robot_python,
                        server=server,
                        arm=arm,
                        joint_targets=np.asarray(
                            arm_payload["sdk_joint_targets_rad"], dtype=np.float32
                        ),
                        gripper_target=float(arm_payload["gripper_position"]),
                        output_json=execute_json,
                        settle_sec=settle_sec,
                        max_joint_delta_rad=max_joint_delta_rad,
                        configure_mode=configure_mode_once,
                        max_subprocess_attempts=EXECUTE_SUBPROCESS_MAX_ATTEMPTS,
                    )
                    (cycle_dir / f"execute_{arm}_step_{step['step_index']:02d}_stdout.log").write_text(
                        execute_log,
                        encoding="utf-8",
                    )
                    step_executions.append(
                        {
                            "arm": arm,
                            "execute_json": str(execute_json.resolve()),
                            "readback": execute_result,
                        }
                    )
                    configure_mode_once = False

                executed_steps.append(
                    {
                        "step_index": step["step_index"],
                        "execution_mode": "joint",
                        "executions": step_executions,
                    }
                )

    report["decision"] = "LIVE_EXECUTED"
    report["executed_steps"] = executed_steps
    write_json(cycle_dir / "cycle_report.json", report)
    return report, configure_mode_once


def validate_live_args(args: argparse.Namespace) -> None:
    if args.mode not in {"shadow", "live"}:
        raise ValueError("--mode must be shadow or live")
    if args.cycles < 1:
        raise ValueError("--cycles must be >= 1")
    if args.execution_horizon < 1:
        raise ValueError("--execution-horizon must be >= 1")

    if args.task_text and args.task_index is not None:
        expected = task_text_for_index(args.task_index)
        if args.task_text != expected:
            raise ValueError(
                f"--task-text {args.task_text!r} does not match task_index {args.task_index}: {expected!r}"
            )

    if args.task_text is None and args.task_index is None:
        raise ValueError("Provide --task-index (0/1/2) or --task-text")

    if args.mode == "live" and args.execute:
        if args.acknowledge != LIVE_ACK_TOKEN:
            raise ValueError(f"Live execution requires --acknowledge {LIVE_ACK_TOKEN!r}")
    elif args.execute:
        raise ValueError("--execute is only valid with --mode live")

    if args.execute_interpolate_hz < 0:
        raise ValueError("--execute-interpolate-hz must be >= 0")
    if args.execute_interpolate_hz > MAX_SDK_CONTROL_HZ:
        raise ValueError(
            f"--execute-interpolate-hz must be <= {MAX_SDK_CONTROL_HZ} (SDK control limit)"
        )
    if args.trajectory_settle_sec < 0:
        raise ValueError("--trajectory-settle-sec must be >= 0")
    if args.train_fps <= 0:
        raise ValueError("--train-fps must be > 0")

    execute_via = str(getattr(args, "execute_via", DEFAULT_EXECUTE_VIA)).lower().strip()
    if execute_via not in {"joint", "end_pose"}:
        raise ValueError("--execute-via must be 'joint' or 'end_pose'")
    args.execute_via = execute_via
    if float(getattr(args, "end_pose_interpolate_hz", POLICY_END_POSE_INTERPOLATE_HZ)) <= 0:
        raise ValueError("--end-pose-interpolate-hz must be > 0")
    if float(getattr(args, "end_pose_max_linear_speed_m_s", 0)) <= 0:
        raise ValueError("--end-pose-max-linear-speed-m-s must be > 0")
    if float(getattr(args, "end_pose_min_duration_sec", 0)) < 0:
        raise ValueError("--end-pose-min-duration-sec must be >= 0")
    if float(getattr(args, "end_pose_max_step_m", 0)) <= 0:
        raise ValueError("--end-pose-max-step-m must be > 0")
    if float(getattr(args, "end_pose_settle_sec", 0)) < 0:
        raise ValueError("--end-pose-settle-sec must be >= 0")

    if args.preposition_task2_home and args.preposition_task2_near_handle:
        raise ValueError(
            "Use at most one of --preposition-task2-home or --preposition-task2-near-handle"
        )

    preposition_mode: str | None = None
    if args.preposition_task2_near_handle:
        preposition_mode = "near_handle"
    elif args.preposition_task2_home:
        preposition_mode = "home"

    if preposition_mode is not None:
        if args.task_index != 1:
            raise ValueError(
                f"--preposition-task2-{preposition_mode.replace('_', '-')} requires --task-index 1 (task2)"
            )
        if args.mode != "live" or not args.execute:
            raise ValueError(
                f"--preposition-task2-{preposition_mode.replace('_', '-')} requires --mode live --execute "
                "(moves right arm via SDK before inference)"
            )
        if args.preposition_max_steps < 1:
            raise ValueError("--preposition-max-steps must be >= 1")
        if args.preposition_max_joint_delta_rad <= 0:
            raise ValueError("--preposition-max-joint-delta-rad must be > 0")
        if args.preposition_tolerance_rad <= 0:
            raise ValueError("--preposition-tolerance-rad must be > 0")
        if args.preposition_min_eef_x_m <= 0:
            raise ValueError("--preposition-min-eef-x-m must be > 0")


def resolve_task_text(args: argparse.Namespace) -> tuple[str, int | None]:
    if args.task_index is not None:
        return task_text_for_index(args.task_index), int(args.task_index)
    if args.task_text is not None:
        for idx, text in CANONICAL_TASKS.items():
            if text == args.task_text:
                return args.task_text, idx
        return args.task_text, None
    raise ValueError("task not specified")


def build_run_manifest(
    *,
    args: argparse.Namespace,
    checkpoint: Path,
    robot_python: Path,
    run_root: Path,
    task_text: str,
    task_index: int | None,
) -> dict[str, Any]:
    return {
        "runner_version": RUNNER_VERSION,
        "started_at": datetime.now().isoformat(),
        "mode": args.mode,
        "execute": bool(args.execute),
        "execute_arms": args.execute_arms,
        "server": args.server,
        "checkpoint": str(checkpoint.resolve()),
        "robot_python": str(robot_python.resolve()),
        "cycles": args.cycles,
        "execution_horizon": args.execution_horizon,
        "task_index": task_index,
        "task_text": task_text,
        "settle_sec": args.settle_sec,
        "max_joint_delta_rad": args.max_joint_delta_rad,
        "interval_sec": args.interval_sec,
        "sdk_backend": args.sdk_backend,
        "sdk_daemon_url": str(args.sdk_daemon_url),
        "action_representation": "absolute_32d_biman",
        "observation_delta_indices": OBSERVATION_DELTA_INDICES,
        "run_root": str(run_root.resolve()),
        "preposition_task2_home": bool(args.preposition_task2_home),
        "preposition_task2_near_handle": bool(args.preposition_task2_near_handle),
        "preposition_min_eef_x_m": args.preposition_min_eef_x_m,
        "preposition_max_joint_delta_rad": args.preposition_max_joint_delta_rad,
        "preposition_tolerance_rad": args.preposition_tolerance_rad,
        "preposition_max_steps": args.preposition_max_steps,
        "execute_via": str(getattr(args, "execute_via", DEFAULT_EXECUTE_VIA)),
        "execute_interpolate_hz": float(args.execute_interpolate_hz),
        "trajectory_settle_sec": float(args.trajectory_settle_sec),
        "train_fps": float(args.train_fps),
        "end_pose_interpolate_hz": float(
            getattr(args, "end_pose_interpolate_hz", POLICY_END_POSE_INTERPOLATE_HZ)
        ),
        "end_pose_max_linear_speed_m_s": float(
            getattr(args, "end_pose_max_linear_speed_m_s", POLICY_END_POSE_MAX_LINEAR_SPEED_M_S)
        ),
        "end_pose_min_duration_sec": float(
            getattr(args, "end_pose_min_duration_sec", POLICY_END_POSE_MIN_DURATION_SEC)
        ),
        "end_pose_max_step_m": float(
            getattr(args, "end_pose_max_step_m", POLICY_END_POSE_MAX_STEP_M)
        ),
        "end_pose_settle_sec": float(
            getattr(args, "end_pose_settle_sec", POLICY_END_POSE_SETTLE_SEC)
        ),
    }


def run_live_runner(args: argparse.Namespace) -> dict[str, Any]:
    print("quanta_biman live_runner starting...", flush=True)
    validate_live_args(args)

    checkpoint = resolve_checkpoint(args.checkpoint)
    if checkpoint is None or not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    robot_python = resolve_robot_python(args.robot_python)
    task_text, task_index = resolve_task_text(args)

    run_root = args.run_root
    if run_root is None:
        run_root = LIVE_RUNS_TMP / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = Path(run_root)
    run_root.mkdir(parents=True, exist_ok=True)

    manifest = build_run_manifest(
        args=args,
        checkpoint=checkpoint,
        robot_python=robot_python,
        run_root=run_root,
        task_text=task_text,
        task_index=task_index,
    )
    write_json(run_root / "run_manifest.json", manifest)

    sdk_backend, daemon_client = resolve_sdk_backend(args)
    print(f"SDK backend: {sdk_backend}" + (f" ({args.sdk_daemon_url})" if sdk_backend == "daemon" else ""))

    print(f"Loading policy from {checkpoint} ...", flush=True)
    policy = load_policy(checkpoint)
    validate_state_keys(policy.modality_configs)
    validate_temporal_config(policy.modality_configs)
    validate_execution_horizon(policy.modality_configs, args.execution_horizon)
    print("Policy loaded. Starting cycles...", flush=True)

    temporal_buffer = SparseTemporalBuffer()
    configure_mode_once = True
    cycle_reports: list[dict[str, Any]] = []
    preposition_report: dict[str, Any] | None = None

    try:
        if args.preposition_task2_near_handle:
            preposition_mode = "near_handle"
        elif args.preposition_task2_home:
            preposition_mode = "home"
        else:
            preposition_mode = None

        if preposition_mode is not None:
            max_steps, max_delta, tolerance, min_eef_x = resolve_task2_preposition_limits(
                args,
                mode=preposition_mode,
            )
            preposition_report, configure_mode_once = run_task2_preposition(
                mode=preposition_mode,
                robot_python=robot_python,
                server=args.server,
                sdk_backend=sdk_backend,
                daemon_client=daemon_client,
                run_root=run_root,
                settle_sec=args.settle_sec,
                max_joint_delta_rad=max_delta,
                tolerance_rad=tolerance,
                max_steps=max_steps,
                min_eef_x_m=min_eef_x,
                configure_mode_once=configure_mode_once,
            )
            temporal_buffer = SparseTemporalBuffer()
            write_json(run_root / "preposition_task2_summary.json", preposition_report)

        for cycle_index in range(1, args.cycles + 1):
            cycle_dir = run_root / f"cycle_{cycle_index:03d}"
            cycle_dir.mkdir(parents=True, exist_ok=True)
            print(
                f"[cycle {cycle_index}/{args.cycles}] mode={args.mode} "
                f"execute={bool(args.execute)} task={task_text!r} horizon={args.execution_horizon}"
            )

            cycle_report, configure_mode_once = run_live_cycle(
                policy=policy,
                robot_python=robot_python,
                server=args.server,
                sdk_backend=sdk_backend,
                daemon_client=daemon_client,
                cycle_dir=cycle_dir,
                task_text=task_text,
                task_index=task_index,
                mode=args.mode,
                execute=bool(args.execute),
                execute_arms=args.execute_arms,
                execution_horizon=args.execution_horizon,
                settle_sec=args.settle_sec,
                max_joint_delta_rad=args.max_joint_delta_rad,
                execute_interpolate_hz=args.execute_interpolate_hz,
                trajectory_settle_sec=args.trajectory_settle_sec,
                train_fps=args.train_fps,
                configure_mode_once=configure_mode_once,
                temporal_buffer=temporal_buffer,
                execute_via=str(getattr(args, "execute_via", DEFAULT_EXECUTE_VIA)),
                end_pose_interpolate_hz=float(
                    getattr(args, "end_pose_interpolate_hz", POLICY_END_POSE_INTERPOLATE_HZ)
                ),
                end_pose_max_linear_speed_m_s=float(
                    getattr(
                        args,
                        "end_pose_max_linear_speed_m_s",
                        POLICY_END_POSE_MAX_LINEAR_SPEED_M_S,
                    )
                ),
                end_pose_min_duration_sec=float(
                    getattr(args, "end_pose_min_duration_sec", POLICY_END_POSE_MIN_DURATION_SEC)
                ),
                end_pose_max_step_m=float(
                    getattr(args, "end_pose_max_step_m", POLICY_END_POSE_MAX_STEP_M)
                ),
                end_pose_settle_sec=float(
                    getattr(args, "end_pose_settle_sec", POLICY_END_POSE_SETTLE_SEC)
                ),
            )
            step0 = cycle_report["planned_steps"][0]
            cycle_reports.append(
                {
                    "cycle_index": cycle_index,
                    "decision": cycle_report["decision"],
                    "inference_sec": cycle_report["inference_sec"],
                    "right_step0_joints": step0["right"]["sdk_joint_targets_rad"],
                    "left_step0_joints": step0["left"]["sdk_joint_targets_rad"],
                }
            )
            print(
                f"  decision={cycle_report['decision']} infer={cycle_report['inference_sec']:.3f}s "
                f"right0={step0['right']['sdk_joint_targets_rad']} "
                f"left0={step0['left']['sdk_joint_targets_rad']}"
            )

            if cycle_index < args.cycles and args.interval_sec > 0:
                time.sleep(args.interval_sec)
    finally:
        if daemon_client is not None:
            daemon_client.close()

    summary = {
        "ok": True,
        "run_root": str(run_root.resolve()),
        "mode": args.mode,
        "execute": bool(args.execute),
        "task_index": task_index,
        "task_text": task_text,
        "cycles_completed": len(cycle_reports),
        "preposition_task2": preposition_report,
        "cycle_reports": cycle_reports,
    }
    write_json(run_root / "run_summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Quanta biman live runner. Default is shadow "
            "(capture + inference + log, no motion)."
        )
    )
    parser.add_argument("--mode", choices=("shadow", "live"), default="shadow")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--acknowledge", default="")
    parser.add_argument("--server", default="127.0.0.1:15051")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--robot-python", type=Path, default=None)
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--execution-horizon", type=int, default=1)
    parser.add_argument(
        "--task-index",
        type=int,
        choices=sorted(CANONICAL_TASKS),
        default=None,
        help="0=switch, 1=rotate/open, 2=raise left + push door",
    )
    parser.add_argument("--task-text", default=None)
    parser.add_argument(
        "--execute-arms",
        choices=("auto", "none", "left", "right", "both"),
        default="auto",
        help="Which arms to send to SDK when --execute (auto follows task_index).",
    )
    parser.add_argument(
        "--execute-via",
        choices=("joint", "end_pose"),
        default=DEFAULT_EXECUTE_VIA,
        help=(
            "Policy motion path: joint=set_joint_positions trajectory; "
            "end_pose=decoded absolute eef_9d via set_end_pose "
            f"(default {DEFAULT_EXECUTE_VIA})."
        ),
    )
    parser.add_argument("--settle-sec", type=float, default=0.6)
    parser.add_argument("--max-joint-delta-rad", type=float, default=0.05)
    parser.add_argument("--interval-sec", type=float, default=0.0)
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument(
        "--sdk-backend",
        choices=("daemon", "subprocess"),
        default="daemon",
        help="daemon=reuse persistent live_sdk_daemon (fast); subprocess=legacy per-call spawn",
    )
    parser.add_argument(
        "--sdk-daemon-url",
        default=DEFAULT_SDK_DAEMON_URL,
        help="JSON-RPC address of biman live_sdk_daemon (default 127.0.0.1:15101)",
    )
    parser.add_argument(
        "--sdk-backend-required",
        action="store_true",
        help="Fail if daemon is unreachable instead of falling back to subprocess",
    )
    parser.add_argument(
        "--preposition-task2-home",
        action="store_true",
        help=(
            "task2 only (--task-index 1): SDK move to training ready pose (eef x~0.07) "
            "before cycle 1; requires --mode live --execute"
        ),
    )
    parser.add_argument(
        "--preposition-task2-near-handle",
        action="store_true",
        help=(
            "task2 only (--task-index 1): SDK move to handle zone (training x>=0.38) "
            "before cycle 1, then run policy; requires --mode live --execute"
        ),
    )
    parser.add_argument(
        "--preposition-min-eef-x-m",
        type=float,
        default=TASK2_NEAR_HANDLE_MIN_EEF_X_M,
        help="Near-handle pre-position success when right eef x >= this (default 0.36)",
    )
    parser.add_argument(
        "--preposition-max-joint-delta-rad",
        type=float,
        default=TASK2_PREPOSITION_DEFAULT_MAX_JOINT_DELTA_RAD,
        help="Max joint step during task2 pre-position (default 0.10)",
    )
    parser.add_argument(
        "--preposition-tolerance-rad",
        type=float,
        default=TASK2_PREPOSITION_DEFAULT_TOLERANCE_RAD,
        help="Stop pre-position when L2 joint error <= this (default 0.05)",
    )
    parser.add_argument(
        "--preposition-max-steps",
        type=int,
        default=TASK2_PREPOSITION_DEFAULT_MAX_STEPS,
        help="Max capture/execute iterations for task2 pre-position (default 25)",
    )
    parser.add_argument(
        "--execute-interpolate-hz",
        type=float,
        default=DEFAULT_EXECUTE_INTERPOLATE_HZ,
        help=(
            "If >0, interpolate policy horizon on dev machine and stream joint commands "
            "at this Hz (max 200, SDK limit). 0=legacy step-by-step execute."
        ),
    )
    parser.add_argument(
        "--trajectory-settle-sec",
        type=float,
        default=DEFAULT_TRAJECTORY_SETTLE_SEC,
        help="Sleep once after each interpolated trajectory (default 0.05; mid-trajectory settle=0)",
    )
    parser.add_argument(
        "--train-fps",
        type=float,
        default=TRAIN_FPS,
        help="Policy waypoint spacing for interpolation (default 15, matches training)",
    )
    parser.add_argument(
        "--end-pose-interpolate-hz",
        type=float,
        default=POLICY_END_POSE_INTERPOLATE_HZ,
        help=f"set_end_pose control Hz when --execute-via end_pose (default {POLICY_END_POSE_INTERPOLATE_HZ})",
    )
    parser.add_argument(
        "--end-pose-max-linear-speed-m-s",
        type=float,
        default=POLICY_END_POSE_MAX_LINEAR_SPEED_M_S,
        help=(
            "Max Cartesian speed for policy end_pose execute "
            f"(default {POLICY_END_POSE_MAX_LINEAR_SPEED_M_S} m/s)"
        ),
    )
    parser.add_argument(
        "--end-pose-min-duration-sec",
        type=float,
        default=POLICY_END_POSE_MIN_DURATION_SEC,
        help=(
            "Extra per-segment duration floor for end_pose trajectory "
            f"(default {POLICY_END_POSE_MIN_DURATION_SEC}; also floored by 1/train-fps)"
        ),
    )
    parser.add_argument(
        "--end-pose-max-step-m",
        type=float,
        default=POLICY_END_POSE_MAX_STEP_M,
        help=f"Max waypoint jump for policy end_pose (default {POLICY_END_POSE_MAX_STEP_M} m)",
    )
    parser.add_argument(
        "--end-pose-settle-sec",
        type=float,
        default=POLICY_END_POSE_SETTLE_SEC,
        help=(
            "Settle once after the whole end_pose trajectory "
            f"(default {POLICY_END_POSE_SETTLE_SEC}; not per policy step)"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    _PIPELINE3 = Path(__file__).resolve().parents[1]
    if str(_PIPELINE3) not in sys.path:
        sys.path.insert(0, str(_PIPELINE3))

    parser = build_parser()
    args = parser.parse_args(argv)
    summary = run_live_runner(args)
    print("\nBiman live runner complete.")
    print(f"  run_root: {summary['run_root']}")
    print(f"  mode: {summary['mode']} execute={summary['execute']}")
    print(f"  task: {summary['task_text']}")
    print(f"  cycles: {summary['cycles_completed']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
