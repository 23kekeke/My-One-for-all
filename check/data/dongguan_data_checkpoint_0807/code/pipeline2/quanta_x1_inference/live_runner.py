"""Phase 5 live runner: shadow inference logging, then optional H=1 execution."""

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

from quanta_x1_inference.action_decode import (
    decode_action_at_step,
    validate_execution_horizon,
)
from quanta_x1_inference.constants import (
    DEFAULT_LIVE_CHECKPOINT,
    LIVE_RUNS_TMP,
    TASK_TEXT,
)
from quanta_x1_inference.live_capture import LIVE_ACK_TOKEN, clean_env, resolve_robot_python
from quanta_x1_inference.live_sdk_daemon import LiveSdkDaemonClient
from quanta_x1_inference.live_sdk_rpc import ping_daemon
from quanta_x1_inference.observation import build_observation_from_components
from quanta_x1_inference.open_loop import write_json
from quanta_x1_inference.policy import load_policy, resolve_checkpoint

RUNNER_VERSION = "quanta_x1_live_runner_v1"
DEFAULT_SDK_DAEMON_URL = "127.0.0.1:15100"

def json_float_list(values: np.ndarray | list[float]) -> str:
    return json.dumps(np.asarray(values, dtype=np.float64).astype(float).tolist())


def run_capture_subprocess(
    *,
    robot_python: Path,
    server: str,
    output_dir: Path,
) -> tuple[dict[str, Any], str]:
    command = [
        str(robot_python),
        "-m",
        "quanta_x1_inference.live_capture",
        "capture",
        "--server",
        server,
        "--output-dir",
        str(output_dir),
    ]
    finished = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=clean_env(),
        check=False,
    )
    log = finished.stdout
    if finished.returncode != 0:
        raise RuntimeError(
            f"Capture subprocess failed with code {finished.returncode}:\n{log[-4000:]}"
        )

    capture_json = output_dir / "capture.json"
    if not capture_json.is_file():
        raise RuntimeError(f"Capture output missing: {capture_json}\n{log[-2000:]}")

    report = json.loads(capture_json.read_text(encoding="utf-8"))
    return report, log


def run_execute_subprocess(
    *,
    robot_python: Path,
    server: str,
    joint_targets: np.ndarray,
    gripper_target: float,
    output_json: Path,
    settle_sec: float,
    max_joint_delta_rad: float,
    configure_mode: bool,
) -> tuple[dict[str, Any], str]:
    command = [
        str(robot_python),
        "-m",
        "quanta_x1_inference.live_capture",
        "execute",
        "--server",
        server,
        "--joint-targets-json",
        json_float_list(joint_targets),
        "--gripper-target",
        str(float(gripper_target)),
        "--output-json",
        str(output_json),
        "--settle-sec",
        str(settle_sec),
        "--max-joint-delta-rad",
        str(max_joint_delta_rad),
        "--allow-motion",
        "--acknowledge",
        LIVE_ACK_TOKEN,
    ]
    if configure_mode:
        command.append("--configure-mode")

    finished = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=clean_env(),
        check=False,
    )
    log = finished.stdout
    if finished.returncode != 0:
        raise RuntimeError(
            f"Execute subprocess failed with code {finished.returncode}:\n{log[-4000:]}"
        )
    if not output_json.is_file():
        raise RuntimeError(f"Execute output missing: {output_json}\n{log[-2000:]}")
    return json.loads(output_json.read_text(encoding="utf-8")), log


def resolve_sdk_backend(args: argparse.Namespace) -> tuple[str, LiveSdkDaemonClient | None]:
    backend = str(args.sdk_backend)
    if backend == "subprocess":
        return backend, None
    if backend != "daemon":
        raise ValueError("--sdk-backend must be daemon or subprocess")

    url = str(args.sdk_daemon_url)
    if not ping_daemon(url, default_port=15100, timeout_sec=2.0):
        if args.sdk_backend_required:
            raise RuntimeError(
                f"SDK daemon not reachable at {url}. "
                "Start: xr_lerobot python -m quanta_x1_inference.live_sdk_daemon serve ..."
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
) -> tuple[dict[str, Any], str]:
    if sdk_backend == "daemon" and daemon_client is not None:
        return daemon_client.capture(output_dir=output_dir)
    return run_capture_subprocess(
        robot_python=robot_python,
        server=server,
        output_dir=output_dir,
    )


def run_execute_backend(
    *,
    sdk_backend: str,
    daemon_client: LiveSdkDaemonClient | None,
    robot_python: Path,
    server: str,
    joint_targets: np.ndarray,
    gripper_target: float,
    output_json: Path,
    settle_sec: float,
    max_joint_delta_rad: float,
    configure_mode: bool,
) -> tuple[dict[str, Any], str]:
    if sdk_backend == "daemon" and daemon_client is not None:
        return daemon_client.execute(
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
        joint_targets=joint_targets,
        gripper_target=gripper_target,
        output_json=output_json,
        settle_sec=settle_sec,
        max_joint_delta_rad=max_joint_delta_rad,
        configure_mode=configure_mode,
    )


def planned_steps_from_action(
    action_dict: dict[str, np.ndarray],
    *,
    execution_horizon: int,
) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    for step_index in range(execution_horizon):
        decoded = decode_action_at_step(action_dict, step_index)
        steps.append(
            {
                "step_index": step_index,
                "vector16": decoded.vector16.astype(float).tolist(),
                "eef_9d": decoded.eef_9d.astype(float).tolist(),
                "gripper_position": float(decoded.gripper_position),
                "joint_position_rad": decoded.joint_position.astype(float).tolist(),
                "sdk_joint_targets_rad": decoded.sdk_joint_targets.astype(float).tolist(),
            }
        )
    return steps


def run_live_cycle(
    *,
    policy: Any,
    robot_python: Path,
    server: str,
    sdk_backend: str,
    daemon_client: LiveSdkDaemonClient | None,
    cycle_dir: Path,
    task_text: str,
    mode: str,
    execute: bool,
    execution_horizon: int,
    settle_sec: float,
    max_joint_delta_rad: float,
    configure_mode_once: bool,
) -> tuple[dict[str, Any], bool]:
    capture_dir = cycle_dir / "capture"
    capture_report, capture_log = run_capture_backend(
        sdk_backend=sdk_backend,
        daemon_client=daemon_client,
        robot_python=robot_python,
        server=server,
        output_dir=capture_dir,
    )
    (cycle_dir / "capture_stdout.log").write_text(capture_log, encoding="utf-8")

    components = capture_report["components"]
    flat, parsed = build_observation_from_components(
        head_camera=_load_saved_camera(capture_report["image_paths"]["head_camera"]),
        right_arm_camera=_load_saved_camera(capture_report["image_paths"]["right_arm_camera"]),
        eef_9d=np.asarray(components["eef_9d"], dtype=np.float32),
        gripper_position=float(components["gripper_position"]),
        joint_position=np.asarray(components["joint_position"], dtype=np.float32),
        task_text=task_text,
        modality_configs=policy.modality_configs,
    )

    infer_start = time.perf_counter()
    action_dict, _info = policy.get_action(parsed)
    inference_sec = time.perf_counter() - infer_start

    planned_steps = planned_steps_from_action(
        action_dict,
        execution_horizon=execution_horizon,
    )

    report: dict[str, Any] = {
        "runner_version": RUNNER_VERSION,
        "cycle_dir": str(cycle_dir.resolve()),
        "mode": mode,
        "execute_requested": bool(execute),
        "execution_horizon": execution_horizon,
        "task_text": task_text,
        "capture_report_path": str((capture_dir / "capture.json").resolve()),
        "capture_timing_ms": capture_report.get("whole_request_wall_elapsed_ms"),
        "sdk_backend": sdk_backend,
        "observation_state16": {
            "eef_9d": components["eef_9d"],
            "gripper_position": float(components["gripper_position"]),
            "joint_position": components["joint_position"],
        },
        "inference_sec": float(inference_sec),
        "planned_steps": planned_steps,
        "decision": "SHADOW_LOG_ONLY",
        "executed_steps": [],
    }

    motion_enabled = mode == "live" and execute
    if not motion_enabled:
        report["reason"] = "shadow_or_dry_run"
        write_json(cycle_dir / "cycle_report.json", report)
        return report, configure_mode_once

    executed_steps: list[dict[str, Any]] = []
    for step in planned_steps:
        execute_json = cycle_dir / f"execute_step_{step['step_index']:02d}.json"
        execute_result, execute_log = run_execute_backend(
            sdk_backend=sdk_backend,
            daemon_client=daemon_client,
            robot_python=robot_python,
            server=server,
            joint_targets=np.asarray(step["sdk_joint_targets_rad"], dtype=np.float32),
            gripper_target=float(step["gripper_position"]),
            output_json=execute_json,
            settle_sec=settle_sec,
            max_joint_delta_rad=max_joint_delta_rad,
            configure_mode=configure_mode_once,
        )
        (cycle_dir / f"execute_step_{step['step_index']:02d}_stdout.log").write_text(
            execute_log,
            encoding="utf-8",
        )
        executed_steps.append(
            {
                "step_index": step["step_index"],
                "execute_json": str(execute_json.resolve()),
                "readback": execute_result,
            }
        )
        configure_mode_once = False

    report["decision"] = "LIVE_EXECUTED"
    report["executed_steps"] = executed_steps
    write_json(cycle_dir / "cycle_report.json", report)
    return report, configure_mode_once


def _load_saved_camera(path: str) -> np.ndarray:
    import cv2

    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Failed to read camera image: {path}")
    return np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def validate_live_args(args: argparse.Namespace) -> None:
    if args.mode not in {"shadow", "live"}:
        raise ValueError("--mode must be shadow or live")
    if args.cycles < 1:
        raise ValueError("--cycles must be >= 1")
    if args.execution_horizon < 1:
        raise ValueError("--execution-horizon must be >= 1")

    if args.mode == "live" and args.execute:
        if args.acknowledge != LIVE_ACK_TOKEN:
            raise ValueError(
                f"Live execution requires --acknowledge {LIVE_ACK_TOKEN!r}"
            )
    elif args.execute:
        raise ValueError("--execute is only valid with --mode live")


def build_run_manifest(
    *,
    args: argparse.Namespace,
    checkpoint: Path,
    robot_python: Path,
    run_root: Path,
) -> dict[str, Any]:
    return {
        "runner_version": RUNNER_VERSION,
        "started_at": datetime.now().isoformat(),
        "mode": args.mode,
        "execute": bool(args.execute),
        "server": args.server,
        "checkpoint": str(checkpoint.resolve()),
        "robot_python": str(robot_python.resolve()),
        "cycles": args.cycles,
        "execution_horizon": args.execution_horizon,
        "task_text": args.task_text,
        "settle_sec": args.settle_sec,
        "max_joint_delta_rad": args.max_joint_delta_rad,
        "interval_sec": args.interval_sec,
        "sdk_backend": args.sdk_backend,
        "sdk_daemon_url": str(args.sdk_daemon_url),
        "control_scope": "right_arm_and_gripper_only",
        "action_representation": "absolute_16d",
        "run_root": str(run_root.resolve()),
    }


def run_live_runner(args: argparse.Namespace) -> dict[str, Any]:
    validate_live_args(args)

    checkpoint = resolve_checkpoint(args.checkpoint)
    if checkpoint is None or not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    robot_python = resolve_robot_python(args.robot_python)

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
    )
    write_json(run_root / "run_manifest.json", manifest)

    sdk_backend, daemon_client = resolve_sdk_backend(args)
    print(f"SDK backend: {sdk_backend}" + (f" ({args.sdk_daemon_url})" if sdk_backend == "daemon" else ""))

    print(f"Loading policy from {checkpoint} ...")
    policy = load_policy(checkpoint)
    validate_execution_horizon(policy.modality_configs, args.execution_horizon)

    configure_mode_once = True
    cycle_reports: list[dict[str, Any]] = []

    try:
        for cycle_index in range(1, args.cycles + 1):
            cycle_dir = run_root / f"cycle_{cycle_index:03d}"
            cycle_dir.mkdir(parents=True, exist_ok=True)
            print(
                f"[cycle {cycle_index}/{args.cycles}] mode={args.mode} "
                f"execute={bool(args.execute)} horizon={args.execution_horizon}"
            )

            cycle_report, configure_mode_once = run_live_cycle(
                policy=policy,
                robot_python=robot_python,
                server=args.server,
                sdk_backend=sdk_backend,
                daemon_client=daemon_client,
                cycle_dir=cycle_dir,
                task_text=args.task_text,
                mode=args.mode,
                execute=bool(args.execute),
                execution_horizon=args.execution_horizon,
                settle_sec=args.settle_sec,
                max_joint_delta_rad=args.max_joint_delta_rad,
                configure_mode_once=configure_mode_once,
            )
            cycle_reports.append(
                {
                    "cycle_index": cycle_index,
                    "decision": cycle_report["decision"],
                    "inference_sec": cycle_report["inference_sec"],
                    "planned_step0_joints": cycle_report["planned_steps"][0]["sdk_joint_targets_rad"],
                }
            )
            print(
                f"  decision={cycle_report['decision']} "
                f"infer={cycle_report['inference_sec']:.3f}s "
                f"step0_joints={cycle_report['planned_steps'][0]['sdk_joint_targets_rad']}"
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
        "cycles_completed": len(cycle_reports),
        "cycle_reports": cycle_reports,
    }
    write_json(run_root / "run_summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Quanta X1 Phase 5 live runner. Default is shadow "
            "(capture + inference + log, no motion)."
        )
    )
    parser.add_argument(
        "--mode",
        choices=("shadow", "live"),
        default="shadow",
        help="shadow=log only; live=allows execution when --execute is set",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Send SDK commands (requires --mode live and --acknowledge)",
    )
    parser.add_argument(
        "--acknowledge",
        default="",
        help=f"Required safety token for --execute: {LIVE_ACK_TOKEN}",
    )
    parser.add_argument("--server", default="127.0.0.1:15051")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_LIVE_CHECKPOINT)
    parser.add_argument(
        "--robot-python",
        type=Path,
        default=None,
        help="xr_lerobot python with x2robot installed",
    )
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--execution-horizon", type=int, default=1)
    parser.add_argument("--task-text", default=TASK_TEXT)
    parser.add_argument("--settle-sec", type=float, default=1.5)
    parser.add_argument(
        "--max-joint-delta-rad",
        type=float,
        default=0.05,
        help="Per-cycle joint delta cap applied before SDK clip (small-step safety)",
    )
    parser.add_argument("--interval-sec", type=float, default=0.5)
    parser.add_argument(
        "--run-root",
        type=Path,
        default=None,
        help=f"Output directory (default: {LIVE_RUNS_TMP}/<timestamp>)",
    )
    parser.add_argument(
        "--sdk-backend",
        choices=("daemon", "subprocess"),
        default="daemon",
        help="daemon=reuse persistent live_sdk_daemon (fast); subprocess=legacy per-call spawn",
    )
    parser.add_argument(
        "--sdk-daemon-url",
        default=DEFAULT_SDK_DAEMON_URL,
        help="JSON-RPC address of live_sdk_daemon (default 127.0.0.1:15100)",
    )
    parser.add_argument(
        "--sdk-backend-required",
        action="store_true",
        help="Fail if daemon is unreachable instead of falling back to subprocess",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    _PIPELINE2 = Path(__file__).resolve().parents[1]
    if str(_PIPELINE2) not in sys.path:
        sys.path.insert(0, str(_PIPELINE2))

    parser = build_parser()
    args = parser.parse_args(argv)
    summary = run_live_runner(args)
    print(f"\nPhase 5 live runner complete.")
    print(f"  run_root: {summary['run_root']}")
    print(f"  mode: {summary['mode']} execute={summary['execute']}")
    print(f"  cycles: {summary['cycles_completed']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
