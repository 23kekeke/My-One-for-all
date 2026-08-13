import time
import json
import signal
import sys
from pathlib import Path
from typing import Annotated, Optional

import typer
import numpy as np
from x2robot import connect
from x2robot.sdk import (
    RobotModeParam, RobotWorkMode,
    ManipulatorControlModeParam, ManipulatorControlMode,
    JointPositions, GripperPosition,
)


app = typer.Typer(add_completion=False)


VALID_MODES = {"faithful", "realtime", "fixed"}
VALID_SAMPLE_MODES = {"nearest", "linear"}
VALID_INVALID_GRIPPER = {"skip", "clamp", "error"}


def _as_1d_float(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float64).reshape(-1)


def load_joint_data(npz_path: str):
    data_obj = np.load(npz_path, allow_pickle=True)
    files = set(data_obj.files)

    timestamps = np.asarray(data_obj["timestamps"], dtype=np.float64)
    left_pos = np.asarray(data_obj["left_arm_position"], dtype=np.float64)
    right_pos = np.asarray(data_obj["right_arm_position"], dtype=np.float64)

    left_gripper_pos = np.asarray(data_obj["left_gripper_position"], dtype=np.float64) if "left_gripper_position" in files else None
    right_gripper_pos = np.asarray(data_obj["right_gripper_position"], dtype=np.float64) if "right_gripper_position" in files else None

    try:
        joint_names = data_obj["joint_names"]
    except KeyError:
        joint_names = None

    meta_path = Path(npz_path).parent / "episode.json"
    meta = {}
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)

    duration = float(timestamps[-1] - timestamps[0]) if len(timestamps) > 1 else 0.0
    inferred_hz = (len(timestamps) - 1) / duration if duration > 0 else 0.0
    meta_hz = meta.get("avg_hz", None)

    print(f"Loaded {len(timestamps)} frames from {npz_path}")
    print(f"  duration: {meta.get('duration', round(duration, 3))}s, avg_hz: {meta_hz if meta_hz is not None else round(inferred_hz, 2)}")
    print(f"  task: {meta.get('task', '?')}, robot: {meta.get('robot_model', '?')}")
    print(f"  left_arm_position: {left_pos.shape}, right_arm_position: {right_pos.shape}")
    if left_gripper_pos is not None:
        print(f"  left_gripper_position: {left_gripper_pos.shape}, min={left_gripper_pos.min():.6f}, max={left_gripper_pos.max():.6f}")
    else:
        print("  left_gripper_position: not found")
    if right_gripper_pos is not None:
        print(f"  right_gripper_position: {right_gripper_pos.shape}, min={right_gripper_pos.min():.6f}, max={right_gripper_pos.max():.6f}")
    else:
        print("  right_gripper_position: not found")

    return {
        "timestamps": timestamps,
        "relative_ts": timestamps - timestamps[0],
        "left_pos": left_pos,
        "right_pos": right_pos,
        "left_gripper_pos": left_gripper_pos,
        "right_gripper_pos": right_gripper_pos,
        "meta": meta,
        "joint_names": joint_names,
        "duration": duration,
        "inferred_hz": inferred_hz,
    }


def sample_nearest(relative_ts: np.ndarray, values: np.ndarray, elapsed: float) -> tuple[int, np.ndarray]:
    """Return nearest previous sample for time elapsed."""
    idx = int(np.searchsorted(relative_ts, elapsed, side="right") - 1)
    idx = max(0, min(idx, len(relative_ts) - 1))
    return idx, values[idx]


def sample_linear(relative_ts: np.ndarray, values: np.ndarray, elapsed: float) -> tuple[int, np.ndarray]:
    """Return linearly interpolated sample for time elapsed."""
    if elapsed <= relative_ts[0]:
        return 0, values[0]
    if elapsed >= relative_ts[-1]:
        return len(relative_ts) - 1, values[-1]

    hi = int(np.searchsorted(relative_ts, elapsed, side="right"))
    lo = hi - 1
    t0 = relative_ts[lo]
    t1 = relative_ts[hi]
    if t1 <= t0:
        return lo, values[lo]
    alpha = (elapsed - t0) / (t1 - t0)
    return lo, values[lo] * (1.0 - alpha) + values[hi] * alpha


def sample_at(relative_ts: np.ndarray, values: Optional[np.ndarray], elapsed: float, sample_mode: str) -> tuple[int, Optional[np.ndarray]]:
    if values is None:
        return -1, None
    if sample_mode == "linear":
        return sample_linear(relative_ts, values, elapsed)
    return sample_nearest(relative_ts, values, elapsed)


def infer_replay_hz(meta: dict, inferred_hz: float, hz: Optional[float]) -> float:
    if hz is not None and hz > 0:
        return float(hz)
    meta_hz = meta.get("avg_hz", None)
    try:
        meta_hz = float(meta_hz)
    except Exception:
        meta_hz = 0.0
    if meta_hz > 0:
        return meta_hz
    if inferred_hz > 0:
        return float(inferred_hz)
    return 30.0


def maybe_gripper_value(arr: Optional[np.ndarray], i_or_value, is_value: bool) -> Optional[float]:
    if arr is None:
        return None
    if is_value:
        return float(_as_1d_float(i_or_value)[0])
    return float(_as_1d_float(arr[int(i_or_value)])[0])


class GripperSender:
    def __init__(
        self,
        robot,
        replay_gripper: bool,
        gripper_eps: float,
        gripper_hz: float,
        invalid_policy: str,
        left_min: Optional[float],
        left_max: Optional[float],
        right_min: Optional[float],
        right_max: Optional[float],
        verbose_invalid: bool,
    ):
        self.robot = robot
        self.replay_gripper = replay_gripper
        self.gripper_eps = float(gripper_eps)
        self.gripper_hz = float(gripper_hz)
        self.invalid_policy = invalid_policy
        self.left_min = left_min
        self.left_max = left_max
        self.right_min = right_min
        self.right_max = right_max
        self.verbose_invalid = verbose_invalid
        self.last_left = None
        self.last_right = None
        self.last_send_t = 0.0
        self.cmd_count = 0
        self.skip_count = 0
        self.invalid_count = 0
        self._reported_invalid = set()

    def _within_or_fix(self, side: str, value: float) -> Optional[float]:
        mn = self.left_min if side == "left" else self.right_min
        mx = self.left_max if side == "left" else self.right_max

        if mn is None and mx is None:
            return value

        invalid = False
        if mn is not None and value < mn:
            invalid = True
        if mx is not None and value > mx:
            invalid = True

        if not invalid:
            return value

        if self.invalid_policy == "clamp":
            fixed = value
            if mn is not None:
                fixed = max(mn, fixed)
            if mx is not None:
                fixed = min(mx, fixed)
            return float(fixed)
        if self.invalid_policy == "error":
            raise ValueError(f"{side} gripper value out of range: {value}")
        return None

    def _should_send(self, side: str, value: float) -> bool:
        last = self.last_left if side == "left" else self.last_right
        if last is None:
            return True
        return abs(value - last) > self.gripper_eps

    def send(self, frame_i: int, left_value: Optional[float], right_value: Optional[float]):
        if not self.replay_gripper:
            return
        now = time.monotonic()
        if self.gripper_hz > 0 and self.last_send_t > 0:
            if now - self.last_send_t < 1.0 / self.gripper_hz:
                return

        sent_any = False
        for side, value in (("left", left_value), ("right", right_value)):
            if value is None:
                continue
            try:
                value = self._within_or_fix(side, float(value))
                if value is None:
                    self.skip_count += 1
                    continue
                if not self._should_send(side, value):
                    self.skip_count += 1
                    continue

                if side == "left":
                    self.robot.left_gripper.set_position(GripperPosition(position=value))
                    self.last_left = value
                else:
                    self.robot.right_gripper.set_position(GripperPosition(position=value))
                    self.last_right = value
                self.cmd_count += 1
                sent_any = True
            except Exception as e:
                self.invalid_count += 1
                key = (side, round(float(value), 6) if value is not None else None)
                if self.verbose_invalid or key not in self._reported_invalid:
                    print(f"\n  frame {frame_i} {side} gripper send failed/skipped value={value}: {e}")
                    self._reported_invalid.add(key)
                if self.invalid_policy == "error":
                    raise
        if sent_any:
            self.last_send_t = now


def setup_robot(server: str):
    robot = connect(f"x2://{server}")
    model = robot.get_robot_model()
    print(f"Connected: {model}")
    robot.system.set_work_mode(RobotModeParam(mode=RobotWorkMode.SDK), timeout=10)
    for attempt in range(3):
        r = robot.robot_control.set_manipulator_control_mode(
            ManipulatorControlModeParam(mode=ManipulatorControlMode.MANIPULATOR_JOINT_POSITIONS),
            timeout=10,
        )
        if r.is_success:
            break
        print(f"set_manipulator_control_mode attempt {attempt + 1}/3 failed")
        time.sleep(1)
    return robot


def send_arm(robot, left_q: np.ndarray, right_q: np.ndarray):
    robot.left_arm.set_joint_positions(JointPositions(positions=np.asarray(left_q, dtype=float).tolist()))
    robot.right_arm.set_joint_positions(JointPositions(positions=np.asarray(right_q, dtype=float).tolist()))


def run_faithful(robot, traj, speed: float, loop: bool, gripper_sender: GripperSender):
    timestamps = traj["timestamps"]
    relative_ts = traj["relative_ts"]
    left_pos = traj["left_pos"]
    right_pos = traj["right_pos"]
    left_g = traj["left_gripper_pos"]
    right_g = traj["right_gripper_pos"]
    n = len(timestamps)

    loop_count = 0
    while True:
        loop_count += 1
        if loop:
            print(f"\n=== Replay loop {loop_count} | mode=faithful ===")
        replay_start = time.monotonic()
        cmd_count = 0
        max_lag = 0.0

        for i in range(n):
            target_real = replay_start + relative_ts[i] / speed
            while time.monotonic() < target_real:
                # Avoid 100% CPU busy spin while keeping reasonable timing.
                remain = target_real - time.monotonic()
                if remain > 0.002:
                    time.sleep(min(0.001, remain))

            lag = max(0.0, time.monotonic() - target_real)
            max_lag = max(max_lag, lag)

            try:
                send_arm(robot, left_pos[i], right_pos[i])
                gripper_sender.send(
                    i,
                    maybe_gripper_value(left_g, i, is_value=False),
                    maybe_gripper_value(right_g, i, is_value=False),
                )
                cmd_count += 1
            except Exception as e:
                print(f"\n  frame {i} send failed: {e}")

            if i % max(1, n // 100) == 0 or i == n - 1:
                elapsed = time.monotonic() - replay_start
                pct = (i + 1) / n * 100
                rate = cmd_count / elapsed if elapsed > 0 else 0
                print(
                    f"\r  {pct:.0f}% ({i+1}/{n}) | {elapsed:.1f}s | {rate:.0f} Hz | max_lag={max_lag*1000:.1f}ms | gripper_cmds={gripper_sender.cmd_count}",
                    end="", flush=True,
                )
        elapsed = time.monotonic() - replay_start
        rate = cmd_count / elapsed if elapsed > 0 else 0
        print(f"\nReplay completed: {cmd_count} frames in {elapsed:.2f}s ({rate:.0f} Hz), max_lag={max_lag*1000:.1f}ms")
        if not loop:
            break
        print("Looping... (Ctrl+C to stop)")


def run_realtime(robot, traj, speed: float, loop: bool, sample_mode: str, gripper_sender: GripperSender, sleep_s: float):
    relative_ts = traj["relative_ts"]
    left_pos = traj["left_pos"]
    right_pos = traj["right_pos"]
    left_g_arr = traj["left_gripper_pos"]
    right_g_arr = traj["right_gripper_pos"]
    duration = float(relative_ts[-1]) if len(relative_ts) else 0.0
    n = len(relative_ts)

    loop_count = 0
    while True:
        loop_count += 1
        if loop:
            print(f"\n=== Replay loop {loop_count} | mode=realtime, sample={sample_mode} ===")
        start = time.monotonic()
        last_i = -1
        sent = 0
        skipped_est = 0

        while True:
            elapsed = (time.monotonic() - start) * speed
            if elapsed >= duration:
                i = n - 1
            else:
                i, left_q = sample_at(relative_ts, left_pos, elapsed, sample_mode)
                _, right_q = sample_at(relative_ts, right_pos, elapsed, sample_mode)
                _, lg_vec = sample_at(relative_ts, left_g_arr, elapsed, sample_mode)
                _, rg_vec = sample_at(relative_ts, right_g_arr, elapsed, sample_mode)

            if elapsed >= duration:
                left_q = left_pos[-1]
                right_q = right_pos[-1]
                lg_vec = left_g_arr[-1] if left_g_arr is not None else None
                rg_vec = right_g_arr[-1] if right_g_arr is not None else None

            if i != last_i:
                if last_i >= 0 and i > last_i + 1:
                    skipped_est += i - last_i - 1
                try:
                    send_arm(robot, left_q, right_q)
                    gripper_sender.send(
                        i,
                        maybe_gripper_value(left_g_arr, lg_vec, is_value=True) if lg_vec is not None else None,
                        maybe_gripper_value(right_g_arr, rg_vec, is_value=True) if rg_vec is not None else None,
                    )
                    sent += 1
                except Exception as e:
                    print(f"\n  frame {i} send failed: {e}")
                last_i = i

                if sent % max(1, n // 100) == 0 or elapsed >= duration:
                    wall_elapsed = time.monotonic() - start
                    pct = min(100.0, elapsed / duration * 100) if duration > 0 else 100.0
                    rate = sent / wall_elapsed if wall_elapsed > 0 else 0
                    print(
                        f"\r  {pct:.0f}% | frame={i+1}/{n} | {wall_elapsed:.1f}s | {rate:.0f} Hz | skipped~={skipped_est} | gripper_cmds={gripper_sender.cmd_count}",
                        end="", flush=True,
                    )
            if elapsed >= duration:
                break
            if sleep_s > 0:
                time.sleep(sleep_s)
        wall_elapsed = time.monotonic() - start
        print(f"\nRealtime replay completed: sent={sent}, skipped~={skipped_est}, wall={wall_elapsed:.2f}s, recorded={duration/speed:.2f}s at speed={speed}")
        if not loop:
            break
        print("Looping... (Ctrl+C to stop)")


def run_fixed(robot, traj, speed: float, loop: bool, sample_mode: str, replay_hz: float, gripper_sender: GripperSender):
    relative_ts = traj["relative_ts"]
    left_pos = traj["left_pos"]
    right_pos = traj["right_pos"]
    left_g_arr = traj["left_gripper_pos"]
    right_g_arr = traj["right_gripper_pos"]
    duration = float(relative_ts[-1]) if len(relative_ts) else 0.0
    n = len(relative_ts)
    period = 1.0 / replay_hz

    loop_count = 0
    while True:
        loop_count += 1
        if loop:
            print(f"\n=== Replay loop {loop_count} | mode=fixed, sample={sample_mode}, hz={replay_hz:.2f} ===")
        start = time.monotonic()
        tick = 0
        sent = 0
        max_lag = 0.0

        while True:
            target_time = start + tick * period
            now = time.monotonic()
            if target_time > now:
                time.sleep(max(0.0, target_time - now))
            lag = max(0.0, time.monotonic() - target_time)
            max_lag = max(max_lag, lag)

            elapsed = (tick * period) * speed
            if elapsed >= duration:
                break

            i, left_q = sample_at(relative_ts, left_pos, elapsed, sample_mode)
            _, right_q = sample_at(relative_ts, right_pos, elapsed, sample_mode)
            _, lg_vec = sample_at(relative_ts, left_g_arr, elapsed, sample_mode)
            _, rg_vec = sample_at(relative_ts, right_g_arr, elapsed, sample_mode)

            try:
                send_arm(robot, left_q, right_q)
                gripper_sender.send(
                    i,
                    maybe_gripper_value(left_g_arr, lg_vec, is_value=True) if lg_vec is not None else None,
                    maybe_gripper_value(right_g_arr, rg_vec, is_value=True) if rg_vec is not None else None,
                )
                sent += 1
            except Exception as e:
                print(f"\n  fixed tick {tick}, frame {i} send failed: {e}")

            if sent % max(1, int(replay_hz)) == 0:
                wall_elapsed = time.monotonic() - start
                pct = min(100.0, elapsed / duration * 100) if duration > 0 else 100.0
                rate = sent / wall_elapsed if wall_elapsed > 0 else 0
                print(
                    f"\r  {pct:.0f}% | tick={tick} frame={i+1}/{n} | {wall_elapsed:.1f}s | {rate:.0f} Hz | max_lag={max_lag*1000:.1f}ms | gripper_cmds={gripper_sender.cmd_count}",
                    end="", flush=True,
                )
            tick += 1
        wall_elapsed = time.monotonic() - start
        print(f"\nFixed-rate replay completed: sent={sent}, wall={wall_elapsed:.2f}s, target_hz={replay_hz:.2f}, actual_hz={sent/wall_elapsed if wall_elapsed>0 else 0:.1f}, max_lag={max_lag*1000:.1f}ms")
        if not loop:
            break
        print("Looping... (Ctrl+C to stop)")


@app.command()
def main(
    server: Annotated[str, typer.Option(help="Robot server address, without x2://")] = "localhost:50051",
    data: Annotated[str, typer.Option(help="Path to joint_data.npz file")] = ...,
    speed: Annotated[float, typer.Option(help="Playback speed multiplier. 1.0 means original duration.")] = 1.0,
    loop: Annotated[bool, typer.Option(help="Loop replay indefinitely")] = False,
    mode: Annotated[str, typer.Option(help="Replay mode: faithful, realtime, fixed")] = "faithful",
    sample: Annotated[str, typer.Option(help="Sampling mode for realtime/fixed: nearest or linear")] = "nearest",
    hz: Annotated[Optional[float], typer.Option(help="Target Hz for fixed mode. If omitted, use episode avg_hz or inferred Hz.")] = None,
    no_replay_gripper: Annotated[bool, typer.Option("--no-replay-gripper", help="Disable gripper replay")] = False,
    gripper_eps: Annotated[float, typer.Option(help="Only send gripper if target changes by more than this. 0 means every eligible frame.")] = 0.0,
    gripper_hz: Annotated[float, typer.Option(help="Max gripper send Hz. 0 means no separate limit.")] = 0.0,
    invalid_gripper: Annotated[str, typer.Option(help="Invalid gripper policy when explicit min/max is provided or RPC fails: skip, clamp, error")] = "skip",
    left_gripper_min: Annotated[Optional[float], typer.Option(help="Optional left gripper min target. Required for clamp to be meaningful.")] = None,
    left_gripper_max: Annotated[Optional[float], typer.Option(help="Optional left gripper max target. Required for clamp to be meaningful.")] = None,
    right_gripper_min: Annotated[Optional[float], typer.Option(help="Optional right gripper min target. Required for clamp to be meaningful.")] = None,
    right_gripper_max: Annotated[Optional[float], typer.Option(help="Optional right gripper max target. Required for clamp to be meaningful.")] = None,
    verbose_invalid_gripper: Annotated[bool, typer.Option(help="Print every invalid gripper error instead of only the first unique value.")] = False,
    realtime_sleep: Annotated[float, typer.Option(help="Small sleep in realtime mode loop to reduce CPU. 0.001 is usually enough.")] = 0.001,
):
    signal.signal(signal.SIGINT, lambda s, f: sys.exit(0))
    mode = mode.lower().strip()
    sample = sample.lower().strip()
    invalid_gripper = invalid_gripper.lower().strip()
    if mode not in VALID_MODES:
        raise typer.BadParameter(f"mode must be one of {sorted(VALID_MODES)}")
    if sample not in VALID_SAMPLE_MODES:
        raise typer.BadParameter(f"sample must be one of {sorted(VALID_SAMPLE_MODES)}")
    if invalid_gripper not in VALID_INVALID_GRIPPER:
        raise typer.BadParameter(f"invalid_gripper must be one of {sorted(VALID_INVALID_GRIPPER)}")
    if speed <= 0:
        raise typer.BadParameter("speed must be > 0")

    traj = load_joint_data(data)
    n = len(traj["timestamps"])
    if n == 0:
        print("No data to replay")
        return

    replay_hz = infer_replay_hz(traj["meta"], traj["inferred_hz"], hz)
    print(f"Replay config: mode={mode}, sample={sample}, speed={speed}, fixed_hz={replay_hz:.2f}")
    print(f"Gripper config: replay={not no_replay_gripper}, eps={gripper_eps}, max_hz={gripper_hz}, invalid={invalid_gripper}")
    if invalid_gripper == "clamp" and any(v is None for v in [left_gripper_min, left_gripper_max, right_gripper_min, right_gripper_max]):
        print("Warning: invalid_gripper=clamp but one or more min/max limits are missing. Missing side/limit will not be clamped.")

    robot = setup_robot(server)
    gripper_sender = GripperSender(
        robot=robot,
        replay_gripper=not no_replay_gripper,
        gripper_eps=gripper_eps,
        gripper_hz=gripper_hz,
        invalid_policy=invalid_gripper,
        left_min=left_gripper_min,
        left_max=left_gripper_max,
        right_min=right_gripper_min,
        right_max=right_gripper_max,
        verbose_invalid=verbose_invalid_gripper,
    )

    if mode == "faithful":
        run_faithful(robot, traj, speed, loop, gripper_sender)
    elif mode == "realtime":
        run_realtime(robot, traj, speed, loop, sample, gripper_sender, realtime_sleep)
    else:
        run_fixed(robot, traj, speed, loop, sample, replay_hz, gripper_sender)

    print(
        f"Final gripper stats: cmds={gripper_sender.cmd_count}, skipped={gripper_sender.skip_count}, invalid_errors={gripper_sender.invalid_count}"
    )


if __name__ == "__main__":
    app()
