"""Dongguan per-task home preposition (lift + dual-arm set_end_pose) before inference."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from dongguan_inference.constants import (
    HOME_PREPOSITION_DEFAULT_ARM_SETTLE_SEC,
    HOME_PREPOSITION_DEFAULT_INTERPOLATE_HZ,
    HOME_PREPOSITION_DEFAULT_LIFT_SETTLE_SEC,
    HOME_PREPOSITION_DEFAULT_MAX_LINEAR_SPEED_M_S,
    HOME_PREPOSITION_DEFAULT_MAX_STEP_M,
    HOME_PREPOSITION_DEFAULT_MAX_STEPS,
    HOME_PREPOSITION_DEFAULT_MIN_DURATION_SEC,
    HOME_PREPOSITION_DEFAULT_ORIENT_TOLERANCE_RAD,
    HOME_PREPOSITION_DEFAULT_TOLERANCE_M,
)
from dongguan_inference.per_task_home import arm_end_pose_targets, get_task_home
from quanta_biman_inference.constants import (
    EXECUTE_SUBPROCESS_MAX_ATTEMPTS,
    LIVE_ACK_TOKEN,
)
from quanta_biman_inference.live_capture import clean_env
from quanta_biman_inference.live_runner import (
    format_subprocess_float,
    run_set_dual_end_pose_backend,
    run_set_end_pose_backend,
    write_json,
)
from quanta_biman_inference.live_sdk_daemon import LiveSdkDaemonClient


def _position_error_m(current: dict[str, Any], target: dict[str, Any]) -> float:
    if "position_m" in target:
        t = target["position_m"]
    else:
        t = target["position"]
    c = current["position"] if "position" in current else current["position_m"]
    cur = np.asarray([c["x"], c["y"], c["z"]], dtype=np.float32)
    tgt = np.asarray([t["x"], t["y"], t["z"]], dtype=np.float32)
    return float(np.linalg.norm(cur - tgt))


def _orientation_error_rad(current: dict[str, Any], target: dict[str, Any]) -> float:
    if "orientation_xyzw" in target:
        t = target["orientation_xyzw"]
    else:
        t = target["orientation"]
    c = current["orientation"] if "orientation" in current else current["orientation_xyzw"]
    cq = np.asarray([c["x"], c["y"], c["z"], c["w"]], dtype=np.float32)
    tq = np.asarray([t["x"], t["y"], t["z"], t["w"]], dtype=np.float32)
    # Normalize defensively.
    cq = cq / max(float(np.linalg.norm(cq)), 1e-8)
    tq = tq / max(float(np.linalg.norm(tq)), 1e-8)
    dot = float(np.clip(abs(np.dot(cq, tq)), 0.0, 1.0))
    return float(2.0 * np.arccos(dot))


def run_set_lift_subprocess(
    *,
    robot_python: Path,
    server: str,
    position_m: float,
    output_json: Path,
    settle_sec: float,
    configure_sdk_mode: bool,
    max_subprocess_attempts: int = EXECUTE_SUBPROCESS_MAX_ATTEMPTS,
) -> tuple[dict[str, Any], str]:
    command = [
        str(robot_python),
        "-m",
        "quanta_biman_inference.live_capture",
        "set-lift",
        "--server",
        server,
        "--position-m",
        format_subprocess_float(position_m),
        "--output-json",
        str(output_json),
        "--settle-sec",
        format_subprocess_float(settle_sec),
        "--allow-motion",
        "--acknowledge",
        LIVE_ACK_TOKEN,
    ]
    if configure_sdk_mode:
        command.append("--configure-sdk-mode")

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
            f"set-lift subprocess failed (attempt {attempt}/{max_subprocess_attempts}):\n{log}"
        )
    assert last_error is not None
    raise last_error


def run_set_lift_backend(
    *,
    sdk_backend: str,
    daemon_client: LiveSdkDaemonClient | None,
    robot_python: Path,
    server: str,
    position_m: float,
    output_json: Path,
    settle_sec: float,
    configure_sdk_mode: bool,
    max_subprocess_attempts: int,
) -> tuple[dict[str, Any], str]:
    if sdk_backend == "daemon" and daemon_client is not None:
        return daemon_client.set_lift(
            position_m=float(position_m),
            settle_sec=float(settle_sec),
            configure_sdk_mode=bool(configure_sdk_mode),
            output_json=output_json,
        )
    return run_set_lift_subprocess(
        robot_python=robot_python,
        server=server,
        position_m=position_m,
        output_json=output_json,
        settle_sec=settle_sec,
        configure_sdk_mode=configure_sdk_mode,
        max_subprocess_attempts=max_subprocess_attempts,
    )



def run_configure_joint_mode_backend(
    *,
    sdk_backend: str,
    daemon_client: LiveSdkDaemonClient | None,
    robot_python: Path,
    server: str,
    output_json: Path,
    max_subprocess_attempts: int = EXECUTE_SUBPROCESS_MAX_ATTEMPTS,
) -> tuple[dict[str, Any], str]:
    if sdk_backend == "daemon" and daemon_client is not None:
        return daemon_client.configure_joint_mode(output_json=output_json)

    command = [
        str(robot_python),
        "-m",
        "quanta_biman_inference.live_capture",
        "configure-joint-mode",
        "--server",
        server,
        "--output-json",
        str(output_json),
        "--allow-motion",
        "--acknowledge",
        LIVE_ACK_TOKEN,
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
        if finished.returncode == 0 and output_json.is_file():
            return json.loads(output_json.read_text(encoding="utf-8")), log
        last_error = RuntimeError(
            f"configure-joint-mode failed (attempt {attempt}/{max_subprocess_attempts}):\n{log}"
        )
    assert last_error is not None
    raise last_error


def run_dongguan_home_preposition(
    *,
    task_index: int,
    home_json: Path | None,
    robot_python: Path,
    server: str,
    sdk_backend: str,
    daemon_client: LiveSdkDaemonClient | None,
    run_root: Path,
    settle_sec: float | None = None,
    max_joint_delta_rad: float | None = None,  # unused; kept for call-site compat
    tolerance_rad: float | None = None,  # unused; kept for call-site compat
    max_steps: int | None = None,
    lift_settle_sec: float | None = None,
    configure_mode_once: bool = True,
    skip_lift: bool = False,
    tolerance_m: float | None = None,
    orient_tolerance_rad: float | None = None,
    interpolate_hz: float | None = None,
    max_linear_speed_m_s: float | None = None,
    min_duration_sec: float | None = None,
    leave_in_end_pose_mode: bool = False,
) -> tuple[dict[str, Any], bool]:
    """Interpolated set_end_pose home.

    By default switches back to JOINT mode for joint policy execute.
    Pass leave_in_end_pose_mode=True when policy execute uses set_end_pose.
    """
    del max_joint_delta_rad, tolerance_rad  # joint-path leftovers

    home = get_task_home(task_index, path=home_json)
    left_pose, left_grip = arm_end_pose_targets(home, "left")
    right_pose, right_grip = arm_end_pose_targets(home, "right")
    # Task0 only (optional): aged gripper tip compensation before SDK. Task1/2 untouched.
    from dongguan_inference.end_pose_bias import maybe_pitch_up_end_pose

    right_pose = maybe_pitch_up_end_pose(
        right_pose, task_index=int(task_index), arm="right"
    )
    lift_m = float(home["lift_position_m"])

    # Preposition is a single slow extend to the home end_pose (farthest point).
    # Never retry/retract — a second try after a far miss looks like "冲出去再缩回".
    requested_max_steps = int(HOME_PREPOSITION_DEFAULT_MAX_STEPS if max_steps is None else max_steps)
    max_steps = 1
    if requested_max_steps != 1:
        print(
            f"[dongguan home] ignoring preposition_max_steps={requested_max_steps}; "
            "forcing single path to home (no retract)",
            flush=True,
        )
    lift_settle_sec = float(
        HOME_PREPOSITION_DEFAULT_LIFT_SETTLE_SEC if lift_settle_sec is None else lift_settle_sec
    )
    arm_settle_sec = float(
        HOME_PREPOSITION_DEFAULT_ARM_SETTLE_SEC if settle_sec is None else settle_sec
    )
    tolerance_m = float(
        HOME_PREPOSITION_DEFAULT_TOLERANCE_M if tolerance_m is None else tolerance_m
    )
    orient_tolerance_rad = float(
        HOME_PREPOSITION_DEFAULT_ORIENT_TOLERANCE_RAD
        if orient_tolerance_rad is None
        else orient_tolerance_rad
    )
    interpolate_hz = float(
        HOME_PREPOSITION_DEFAULT_INTERPOLATE_HZ if interpolate_hz is None else interpolate_hz
    )
    max_linear_speed_m_s = float(
        HOME_PREPOSITION_DEFAULT_MAX_LINEAR_SPEED_M_S
        if max_linear_speed_m_s is None
        else max_linear_speed_m_s
    )
    min_duration_sec = float(
        HOME_PREPOSITION_DEFAULT_MIN_DURATION_SEC if min_duration_sec is None else min_duration_sec
    )
    max_step_m = float(HOME_PREPOSITION_DEFAULT_MAX_STEP_M)

    preposition_dir = run_root / f"preposition_task{task_index}_home"
    preposition_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "preposition_mode": "dongguan_per_task_home_dual_end_pose_sync",
        "control_mode": "MANIPULATOR_END_POSE",
        "task_index": int(task_index),
        "task_id": home.get("task_id"),
        "language": home.get("language"),
        "lift_position_m": lift_m,
        "left_target_end_pose": left_pose,
        "left_target_gripper": left_grip,
        "right_target_end_pose": right_pose,
        "right_target_gripper": right_grip,
        "tolerance_m": tolerance_m,
        "orient_tolerance_rad": orient_tolerance_rad,
        "interpolate_hz": interpolate_hz,
        "max_linear_speed_m_s": max_linear_speed_m_s,
        "min_duration_sec": min_duration_sec,
        "max_step_m": max_step_m,
        "max_steps": max_steps,
        "skip_lift": bool(skip_lift),
        "steps": [],
    }

    rx = float(right_pose["position_m"]["x"])
    ry = float(right_pose["position_m"]["y"])
    rz = float(right_pose["position_m"]["z"])
    print(
        f"[dongguan home] task_index={task_index} task_id={home.get('task_id')} "
        f"lift={lift_m:.4f}m -> DUAL sync slow extend to preposition "
        f"(same vmax, both arms together, no retract) "
        f"right_target=({rx:.3f},{ry:.3f},{rz:.3f}) "
        f"(hz={interpolate_hz:.0f}, vmax={max_linear_speed_m_s:.3f}m/s, "
        f"step<={max_step_m:.3f}m)",
        flush=True,
    )

    if not skip_lift:
        lift_json = preposition_dir / "set_lift.json"
        lift_result, lift_log = run_set_lift_backend(
            sdk_backend=sdk_backend,
            daemon_client=daemon_client,
            robot_python=robot_python,
            server=server,
            position_m=lift_m,
            output_json=lift_json,
            settle_sec=lift_settle_sec,
            configure_sdk_mode=True,
            max_subprocess_attempts=EXECUTE_SUBPROCESS_MAX_ATTEMPTS,
        )
        (preposition_dir / "set_lift_stdout.log").write_text(lift_log, encoding="utf-8")
        report["lift"] = lift_result
        print(
            f"[dongguan home] lift {lift_result.get('before_lift_position_m')} -> "
            f"{lift_result.get('after_lift_position_m')} "
            f"(target={lift_m:.4f}, err={lift_result.get('abs_error_m')}, "
            f"reached={lift_result.get('reached')}, attempts={lift_result.get('attempts_used')})",
            flush=True,
        )
        if not bool(lift_result.get("reached", False)):
            print(
                "[dongguan home] WARNING: lift did not reach tolerance — "
                "check set_lift.json / daemon logs before trusting torso height",
                flush=True,
            )

    # One dual path: both arms current → home end_pose on the same clock.
    step_index = 0
    step_record: dict[str, Any] = {"step_index": step_index, "status": "dual_sync_path"}

    dual_json = preposition_dir / f"step_{step_index:02d}_set_dual_end_pose.json"
    dual_result, dual_log = run_set_dual_end_pose_backend(
        sdk_backend=sdk_backend,
        daemon_client=daemon_client,
        robot_python=robot_python,
        server=server,
        left_end_pose=left_pose,
        left_gripper_target=left_grip,
        right_end_pose=right_pose,
        right_gripper_target=right_grip,
        output_json=dual_json,
        settle_sec=arm_settle_sec,
        configure_mode=configure_mode_once,
        interpolate_hz=interpolate_hz,
        max_linear_speed_m_s=max_linear_speed_m_s,
        min_duration_sec=min_duration_sec,
        max_step_m=max_step_m,
        max_subprocess_attempts=EXECUTE_SUBPROCESS_MAX_ATTEMPTS,
    )
    (preposition_dir / f"step_{step_index:02d}_set_dual_end_pose_stdout.log").write_text(
        dual_log,
        encoding="utf-8",
    )

    left_block = dual_result.get("left", {})
    right_block = dual_result.get("right", {})
    left_pos_err = float(
        dual_result.get(
            "left_position_error_m",
            left_block.get("position_error_m", _position_error_m(left_block["after_end_pose"], left_pose)),
        )
    )
    right_pos_err = float(
        dual_result.get(
            "right_position_error_m",
            right_block.get(
                "position_error_m", _position_error_m(right_block["after_end_pose"], right_pose)
            ),
        )
    )
    left_ori_err = float(
        dual_result.get(
            "left_orientation_error_rad",
            left_block.get(
                "orientation_error_rad",
                _orientation_error_rad(left_block["after_end_pose"], left_pose),
            ),
        )
    )
    right_ori_err = float(
        dual_result.get(
            "right_orientation_error_rad",
            right_block.get(
                "orientation_error_rad",
                _orientation_error_rad(right_block["after_end_pose"], right_pose),
            ),
        )
    )

    step_record["dual_execute"] = {
        "dense_waypoints": dual_result.get("dense_waypoints"),
        "wall_sec": dual_result.get("wall_sec"),
        "left_planned_duration_sec": dual_result.get("left_planned_duration_sec"),
        "right_planned_duration_sec": dual_result.get("right_planned_duration_sec"),
    }
    step_record["left_execute"] = {
        "position_error_m": left_pos_err,
        "orientation_error_rad": left_ori_err,
        "target_end_pose": left_block.get("target_end_pose"),
        "after_end_pose": left_block.get("after_end_pose"),
    }
    step_record["right_execute"] = {
        "position_error_m": right_pos_err,
        "orientation_error_rad": right_ori_err,
        "target_end_pose": right_block.get("target_end_pose"),
        "after_end_pose": right_block.get("after_end_pose"),
    }

    left_ok = left_pos_err <= tolerance_m and left_ori_err <= orient_tolerance_rad
    right_ok = right_pos_err <= tolerance_m and right_ori_err <= orient_tolerance_rad
    reached = bool(left_ok and right_ok)
    step_record["status"] = "reached" if reached else "missed_but_no_retract"
    report["steps"].append(step_record)
    report["ok"] = reached
    report["final_left_position_error_m"] = left_pos_err
    report["final_right_position_error_m"] = right_pos_err
    report["final_left_orientation_error_rad"] = left_ori_err
    report["final_right_orientation_error_rad"] = right_ori_err
    report["steps_used"] = 1
    print(
        f"[dongguan home] done (DUAL sync, stop at preposition) "
        f"wp={dual_result.get('dense_waypoints')} "
        f"{dual_result.get('wall_sec')}s "
        f"left_err={left_pos_err:.4f}m right_err={right_pos_err:.4f}m ok={reached}",
        flush=True,
    )
    if not reached:
        print(
            f"[dongguan home] WARNING: residual error after dual sync path "
            f"(left={left_pos_err:.4f}m, right={right_pos_err:.4f}m) — "
            "NOT retrying / NOT retracting",
            flush=True,
        )

    # Critical: joint policy execute needs JOINT mode; end_pose execute stays in END_POSE.
    if leave_in_end_pose_mode:
        report["control_mode_after_home"] = "MANIPULATOR_END_POSE"
        write_json(preposition_dir / "preposition_report.json", report)
        print(
            "[dongguan home] leaving MANIPULATOR_END_POSE for policy end_pose execute",
            flush=True,
        )
        # Already configured during dual path; do not force joint reconfigure.
        return report, False

    joint_json = preposition_dir / "configure_joint_mode.json"
    joint_result, joint_log = run_configure_joint_mode_backend(
        sdk_backend=sdk_backend,
        daemon_client=daemon_client,
        robot_python=robot_python,
        server=server,
        output_json=joint_json,
    )
    (preposition_dir / "configure_joint_mode_stdout.log").write_text(joint_log, encoding="utf-8")
    report["configure_joint_mode"] = joint_result
    report["control_mode_after_home"] = "MANIPULATOR_JOINT_POSITIONS"
    write_json(preposition_dir / "preposition_report.json", report)
    print(
        "[dongguan home] switched to MANIPULATOR_JOINT_POSITIONS for policy execute",
        flush=True,
    )
    # Return True so first live cycle also re-asserts joint mode if needed.
    return report, True
