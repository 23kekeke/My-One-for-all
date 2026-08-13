"""Dongguan shadow/live runner entry (wraps quanta_biman_inference.live_runner).

Adds per-task home preposition (lift + set_end_pose) before live execute cycles.
Default policy execute path is end_pose (decoded absolute eef_9d → set_end_pose).
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

_PIPELINE5 = Path(__file__).resolve().parents[1]
if str(_PIPELINE5) not in sys.path:
    sys.path.insert(0, str(_PIPELINE5))

from dongguan_inference import bootstrap_infer


def _inject_dongguan_home_args(parser: argparse.ArgumentParser) -> None:
    from dongguan_inference.constants import (
        DEFAULT_PER_TASK_HOME,
        HOME_PREPOSITION_DEFAULT_LIFT_SETTLE_SEC,
        HOME_PREPOSITION_DEFAULT_MAX_STEPS,
        HOME_PREPOSITION_DEFAULT_ORIENT_TOLERANCE_RAD,
        HOME_PREPOSITION_DEFAULT_TOLERANCE_M,
    )

    # Dongguan robot default (direct LAN). Override with --server if needed.
    for action in parser._actions:
        if action.dest == "server" and action.default in ("127.0.0.1:15051", None):
            action.default = "192.168.1.103:50051"
            action.help = (
                "Robot gRPC address (Dongguan default 192.168.1.103:50051; "
                "use 127.0.0.1:15051 only with SSH tunnel)"
            )
        if action.dest == "task_index":
            action.help = "0=grasp(task3), 1=rotate(task4), 2=pull(task5)"
        if action.dest == "execute_via":
            action.default = "end_pose"
            action.help = (
                "Policy motion path (Dongguan default: end_pose = absolute eef_9d via "
                "set_end_pose; use joint for set_joint_positions)"
            )

    parser.add_argument(
        "--preposition-home",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Before live cycles: set lift + move both arms to per_task_home[task_index] "
            "via set_end_pose (position_m + orientation_xyzw). "
            "Default ON for --mode live --execute; use --no-preposition-home to skip."
        ),
    )
    parser.add_argument(
        "--per-task-home-json",
        type=Path,
        default=DEFAULT_PER_TASK_HOME,
        help=f"Path to per_task_home.json (default: {DEFAULT_PER_TASK_HOME})",
    )
    parser.add_argument(
        "--preposition-skip-lift",
        action="store_true",
        help="Skip LiftController.set_lift_position during home preposition",
    )
    parser.add_argument(
        "--preposition-lift-settle-sec",
        type=float,
        default=HOME_PREPOSITION_DEFAULT_LIFT_SETTLE_SEC,
        help=f"Settle time after lift move (default {HOME_PREPOSITION_DEFAULT_LIFT_SETTLE_SEC})",
    )
    parser.add_argument(
        "--preposition-tolerance-m",
        type=float,
        default=HOME_PREPOSITION_DEFAULT_TOLERANCE_M,
        help=(
            "Stop home preposition when end-pose xyz L2 error <= this (meters) "
            f"(default {HOME_PREPOSITION_DEFAULT_TOLERANCE_M})"
        ),
    )
    parser.add_argument(
        "--preposition-orient-tolerance-rad",
        type=float,
        default=HOME_PREPOSITION_DEFAULT_ORIENT_TOLERANCE_RAD,
        help=(
            "Stop home preposition when quaternion angle error <= this (radians) "
            f"(default {HOME_PREPOSITION_DEFAULT_ORIENT_TOLERANCE_RAD})"
        ),
    )
    parser.add_argument(
        "--preposition-strict",
        action="store_true",
        help=(
            "Abort live run if home end_pose residual exceeds tolerance "
            "(default: warn and continue; home is single-path / no retract)"
        ),
    )
    parser.add_argument(
        "--task0-end-pose-pitch-up-deg",
        type=float,
        default=None,
        help=(
            "Optional. Task0 only: before SDK, pitch right end_pose up by this many "
            "degrees (positive=up on this robot). Omit or 0 = no compensation "
            "(original behavior). Task1/2 never apply."
        ),
    )
    for action in parser._actions:
        if action.dest == "preposition_max_steps":
            action.default = HOME_PREPOSITION_DEFAULT_MAX_STEPS
            action.help = (
                "Unused (home is always a single slow path to preposition; "
                f"default {HOME_PREPOSITION_DEFAULT_MAX_STEPS}, no retract)"
            )
        if action.dest == "preposition_max_joint_delta_rad":
            action.help = (
                "(unused for Dongguan end_pose home) kept for pipeline3 compat"
            )
        if action.dest == "preposition_tolerance_rad":
            action.help = (
                "(unused for Dongguan end_pose home; use --preposition-tolerance-m)"
            )


def should_preposition_home(args: argparse.Namespace) -> bool:
    if getattr(args, "preposition_home", None) is False:
        return False
    if getattr(args, "preposition_home", None) is True:
        return True
    return bool(args.mode == "live" and args.execute)


def validate_dongguan_home_args(args: argparse.Namespace) -> None:
    if not should_preposition_home(args):
        return
    if args.task_index is None:
        raise ValueError("--preposition-home requires --task-index 0/1/2")
    if args.mode != "live" or not args.execute:
        raise ValueError(
            "--preposition-home requires --mode live --execute "
            "(moves lift + arms via SDK before inference)"
        )
    if getattr(args, "preposition_task2_home", False) or getattr(
        args, "preposition_task2_near_handle", False
    ):
        raise ValueError("Do not combine Dongguan --preposition-home with task2 preposition flags")
    home_json = Path(args.per_task_home_json)
    if not home_json.is_file():
        raise FileNotFoundError(f"--per-task-home-json not found: {home_json}")


def run_dongguan_live_runner(args: argparse.Namespace) -> dict[str, Any]:
    from dongguan_inference.preposition import run_dongguan_home_preposition
    from quanta_biman_inference.action_decode import validate_execution_horizon
    from quanta_biman_inference.constants import (
        LIVE_RUNS_TMP,
        POLICY_END_POSE_INTERPOLATE_HZ,
        POLICY_END_POSE_MAX_LINEAR_SPEED_M_S,
        POLICY_END_POSE_MAX_STEP_M,
        POLICY_END_POSE_MIN_DURATION_SEC,
        POLICY_END_POSE_SETTLE_SEC,
    )
    from quanta_biman_inference.live_runner import (
        build_run_manifest,
        resolve_sdk_backend,
        resolve_task_text,
        run_live_cycle,
        run_task2_preposition,
        resolve_task2_preposition_limits,
        validate_live_args,
        write_json,
    )
    from quanta_biman_inference.live_capture import resolve_robot_python
    from quanta_biman_inference.observation import (
        SparseTemporalBuffer,
        validate_state_keys,
        validate_temporal_config,
    )
    from quanta_biman_inference.policy import load_policy, resolve_checkpoint

    print("dongguan live_runner starting...", flush=True)
    validate_dongguan_home_args(args)
    validate_live_args(args)

    checkpoint = resolve_checkpoint(args.checkpoint)
    if checkpoint is None or not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    robot_python = resolve_robot_python(args.robot_python)
    task_text, task_index = resolve_task_text(args)
    execute_via = str(getattr(args, "execute_via", "end_pose")).lower().strip()

    from dongguan_inference.end_pose_bias import set_task0_pitch_context

    pitch_raw = getattr(args, "task0_end_pose_pitch_up_deg", None)
    pitch_deg = 0.0 if pitch_raw is None else float(pitch_raw)
    set_task0_pitch_context(task_index=task_index, pitch_up_deg=pitch_deg)
    if task_index == 0 and abs(pitch_deg) > 1e-9:
        print(
            f"[pipeline5] task0 right end_pose pitch-up={pitch_deg:.2f}deg "
            "(applied before SDK home + live trajectory; task1/2 unchanged)",
            flush=True,
        )
    elif task_index == 0:
        print(
            "[pipeline5] task0: no end_pose pitch compensation "
            "(pass --task0-end-pose-pitch-up-deg DEG to enable)",
            flush=True,
        )

    run_root = args.run_root
    if run_root is None:
        run_root = LIVE_RUNS_TMP / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = Path(run_root)
    run_root.mkdir(parents=True, exist_ok=True)

    do_home = should_preposition_home(args)
    manifest = build_run_manifest(
        args=args,
        checkpoint=checkpoint,
        robot_python=robot_python,
        run_root=run_root,
        task_text=task_text,
        task_index=task_index,
    )
    manifest["preposition_home"] = do_home
    manifest["per_task_home_json"] = str(Path(args.per_task_home_json).resolve())
    manifest["preposition_skip_lift"] = bool(getattr(args, "preposition_skip_lift", False))
    manifest["execute_via"] = execute_via
    manifest["task0_end_pose_pitch_up_deg"] = pitch_deg if task_index == 0 else 0.0
    write_json(run_root / "run_manifest.json", manifest)

    sdk_backend, daemon_client = resolve_sdk_backend(args)
    print(
        f"SDK backend: {sdk_backend}"
        + (f" ({args.sdk_daemon_url})" if sdk_backend == "daemon" else "")
    )
    print(f"execute_via={execute_via}", flush=True)

    print(f"Loading policy from {checkpoint} ...", flush=True)
    policy = load_policy(checkpoint)
    validate_state_keys(policy.modality_configs)
    validate_temporal_config(policy.modality_configs)
    validate_execution_horizon(policy.modality_configs, args.execution_horizon)
    print("Policy loaded.", flush=True)

    temporal_buffer = SparseTemporalBuffer()
    configure_mode_once = True
    cycle_reports: list[dict[str, Any]] = []
    preposition_report: dict[str, Any] | None = None
    task2_preposition_report: dict[str, Any] | None = None

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
            task2_preposition_report, configure_mode_once = run_task2_preposition(
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
            write_json(run_root / "preposition_task2_summary.json", task2_preposition_report)

        if do_home:
            assert task_index is not None
            preposition_report, configure_mode_once = run_dongguan_home_preposition(
                task_index=int(task_index),
                home_json=Path(args.per_task_home_json),
                robot_python=robot_python,
                server=args.server,
                sdk_backend=sdk_backend,
                daemon_client=daemon_client,
                run_root=run_root,
                settle_sec=args.settle_sec,
                max_steps=int(args.preposition_max_steps),
                lift_settle_sec=float(args.preposition_lift_settle_sec),
                configure_mode_once=configure_mode_once,
                skip_lift=bool(args.preposition_skip_lift),
                tolerance_m=float(args.preposition_tolerance_m),
                orient_tolerance_rad=float(args.preposition_orient_tolerance_rad),
                leave_in_end_pose_mode=(execute_via == "end_pose"),
            )
            temporal_buffer = SparseTemporalBuffer()
            write_json(run_root / "preposition_home_summary.json", preposition_report)
            if not preposition_report.get("ok", False):
                # Single-path home never retracts/retries; residual ~cm is common.
                # Soft-continue so inference can still start (use --preposition-strict to abort).
                if bool(getattr(args, "preposition_strict", False)):
                    raise RuntimeError(
                        "Dongguan home preposition did not reach tolerance; "
                        "refusing to start inference (--preposition-strict). "
                        "See preposition_home_summary.json"
                    )
                print(
                    "[dongguan home] WARNING: residual above tolerance — "
                    "continuing to inference (pass --preposition-strict to abort instead). "
                    f"left={preposition_report.get('final_left_position_error_m')}m "
                    f"right={preposition_report.get('final_right_position_error_m')}m",
                    flush=True,
                )

        print("Starting cycles...", flush=True)
        for cycle_index in range(1, args.cycles + 1):
            cycle_dir = run_root / f"cycle_{cycle_index:03d}"
            cycle_dir.mkdir(parents=True, exist_ok=True)
            print(
                f"[cycle {cycle_index}/{args.cycles}] mode={args.mode} "
                f"execute={bool(args.execute)} via={execute_via} task={task_text!r} "
                f"horizon={args.execution_horizon}"
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
                execute_via=execute_via,
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
            if execute_via == "end_pose":
                right_xyz = step0["right"]["end_pose"]["position_m"]
                left_xyz = step0["left"]["end_pose"]["position_m"]
                cycle_reports.append(
                    {
                        "cycle_index": cycle_index,
                        "decision": cycle_report["decision"],
                        "inference_sec": cycle_report["inference_sec"],
                        "execute_via": execute_via,
                        "right_step0_xyz": [
                            right_xyz["x"],
                            right_xyz["y"],
                            right_xyz["z"],
                        ],
                        "left_step0_xyz": [
                            left_xyz["x"],
                            left_xyz["y"],
                            left_xyz["z"],
                        ],
                    }
                )
                print(
                    f"  decision={cycle_report['decision']} infer={cycle_report['inference_sec']:.3f}s "
                    f"right0_xyz=({right_xyz['x']:.3f},{right_xyz['y']:.3f},{right_xyz['z']:.3f}) "
                    f"left0_xyz=({left_xyz['x']:.3f},{left_xyz['y']:.3f},{left_xyz['z']:.3f})"
                )
            else:
                cycle_reports.append(
                    {
                        "cycle_index": cycle_index,
                        "decision": cycle_report["decision"],
                        "inference_sec": cycle_report["inference_sec"],
                        "execute_via": execute_via,
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
        "execute_via": execute_via,
        "task_index": task_index,
        "task_text": task_text,
        "cycles_completed": len(cycle_reports),
        "preposition_home": preposition_report,
        "preposition_task2": task2_preposition_report,
        "cycle_reports": cycle_reports,
    }
    write_json(run_root / "run_summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    bootstrap_infer()
    from quanta_biman_inference import live_runner as biman_lr

    parser = biman_lr.build_parser()
    _inject_dongguan_home_args(parser)
    args = parser.parse_args(argv)
    summary = run_dongguan_live_runner(args)
    print("\nDongguan live runner complete.")
    print(f"  run_root: {summary['run_root']}")
    print(f"  mode: {summary['mode']} execute={summary['execute']} via={summary['execute_via']}")
    print(f"  task: {summary['task_text']}")
    print(f"  cycles: {summary['cycles_completed']}")
    home = summary.get("preposition_home")
    if home is not None:
        print(
            f"  home: ok={home.get('ok')} steps={home.get('steps_used')} "
            f"lift={home.get('lift_position_m')} "
            f"left_pos_err={home.get('final_left_position_error_m')} "
            f"right_pos_err={home.get('final_right_position_error_m')} "
            f"mode_after={home.get('control_mode_after_home')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
