import time
import signal
import sys
import json
import threading
import pickle
import tempfile
from pathlib import Path
from typing import Annotated, Optional
import typer
import numpy as np
from x2robot import connect
from x2robot.sdk import RobotModeParam, RobotWorkMode

ARM_JOINT_PREFIXES = ("left_arm_joint", "right_arm_joint")
LEFT_GRIPPER_JOINT_NAME = "left_arm_gripper"
RIGHT_GRIPPER_JOINT_NAME = "right_arm_gripper"


class JointRecorder:
    def __init__(self, robot, output_dir: Path):
        self.robot = robot
        self.output_dir = output_dir
        .output_dir.mkdir(parents=True, exist_ok=True)
        self.is_recording = False
        self.episode_count = 0self
        self._temp_file = None
        self._thread = None
        self._joint_indices = None
        self._joint_names = None
        self._gripper_names = None

    def start_recording(self):
        self.is_recording = True
        self._temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pkl")
        self._thread = threading.Thread(target=self._record_worker, daemon=True)
        self._thread.start()

    def _record_worker(self):
        try:
            stream = self.robot.state.get_all_joint_states_stream(timeout=None)
            for msg in stream:
                if not self.is_recording:
                    break
                names = list(msg.name) if msg.name else []
                if self._joint_indices is None:
                    left_idx = [i for i, n in enumerate(names) if n.startswith("left_arm_joint")]
                    right_idx = [i for i, n in enumerate(names) if n.startswith("right_arm_joint")]
                    left_gripper_idx = [i for i, n in enumerate(names) if n == LEFT_GRIPPER_JOINT_NAME]
                    right_gripper_idx = [i for i, n in enumerate(names) if n == RIGHT_GRIPPER_JOINT_NAME]
                    if not left_idx and not right_idx:
                        print(f"  warning: no arm joints found in stream, names={names}")
                        self.is_recording = False
                        break
                    self._joint_indices = {
                        "left": left_idx,
                        "right": right_idx,
                        "left_gripper": left_gripper_idx,
                        "right_gripper": right_gripper_idx,
                    }
                    # Keep original joint_names behavior unchanged: arm joint names only.
                    self._joint_names = [names[i] for i in (left_idx + right_idx)]
                    self._gripper_names = [names[i] for i in (left_gripper_idx + right_gripper_idx)]
                    if not left_gripper_idx:
                        print(f"  warning: {LEFT_GRIPPER_JOINT_NAME} not found in stream")
                    if not right_gripper_idx:
                        print(f"  warning: {RIGHT_GRIPPER_JOINT_NAME} not found in stream")
                ts = time.time()
                pos = np.array(msg.position, dtype=np.float32) if msg.position else np.array([], dtype=np.float32)
                vel = np.array(msg.velocity, dtype=np.float32) if msg.velocity else None
                eff = np.array(msg.effort, dtype=np.float32) if msg.effort else None
                data = {
                    "ts": ts,
                    "left_pos": pos[self._joint_indices["left"]],
                    "left_vel": vel[self._joint_indices["left"]] if vel is not None else None,
                    "left_eff": eff[self._joint_indices["left"]] if eff is not None else None,
                    "right_pos": pos[self._joint_indices["right"]],
                    "right_vel": vel[self._joint_indices["right"]] if vel is not None else None,
                    "right_eff": eff[self._joint_indices["right"]] if eff is not None else None,
                    "left_gripper_pos": pos[self._joint_indices["left_gripper"]],
                    "left_gripper_vel": vel[self._joint_indices["left_gripper"]] if vel is not None else None,
                    "left_gripper_eff": eff[self._joint_indices["left_gripper"]] if eff is not None else None,
                    "right_gripper_pos": pos[self._joint_indices["right_gripper"]],
                    "right_gripper_vel": vel[self._joint_indices["right_gripper"]] if vel is not None else None,
                    "right_gripper_eff": eff[self._joint_indices["right_gripper"]] if eff is not None else None,
                }
                pickle.dump(data, self._temp_file)
                self._temp_file.flush()
        except Exception as e:
            print(f"  recording stream error: {e}")
            self.is_recording = False

    def stop_recording(self, task: str = "", robot_model: str = "") -> Optional[dict]:
        self.is_recording = False
        if self._thread:
            self._thread.join(timeout=10)
        if self._temp_file is None:
            return None
        self._temp_file.close()
        temp_path = Path(self._temp_file.name)
        data_list = []
        try:
            with open(temp_path, "rb") as f:
                while True:
                    try:
                        data_list.append(pickle.load(f))
                    except EOFError:
                        break
        except Exception:
            pass
        temp_path.unlink(missing_ok=True)
        if not data_list:
            return None
        timestamps = np.array([d["ts"] for d in data_list])
        left_pos = np.array([d["left_pos"] for d in data_list])
        right_pos = np.array([d["right_pos"] for d in data_list])
        left_vel = None
        if data_list[0]["left_vel"] is not None:
            left_vel = np.array([d["left_vel"] for d in data_list])
        left_eff = None
        if data_list[0]["left_eff"] is not None:
            left_eff = np.array([d["left_eff"] for d in data_list])
        right_vel = None
        if data_list[0]["right_vel"] is not None:
            right_vel = np.array([d["right_vel"] for d in data_list])
        right_eff = None
        if data_list[0]["right_eff"] is not None:
            right_eff = np.array([d["right_eff"] for d in data_list])
        left_gripper_pos = np.array([d["left_gripper_pos"] for d in data_list])
        right_gripper_pos = np.array([d["right_gripper_pos"] for d in data_list])
        left_gripper_vel = None
        if data_list[0]["left_gripper_vel"] is not None:
            left_gripper_vel = np.array([d["left_gripper_vel"] for d in data_list])
        left_gripper_eff = None
        if data_list[0]["left_gripper_eff"] is not None:
            left_gripper_eff = np.array([d["left_gripper_eff"] for d in data_list])
        right_gripper_vel = None
        if data_list[0]["right_gripper_vel"] is not None:
            right_gripper_vel = np.array([d["right_gripper_vel"] for d in data_list])
        right_gripper_eff = None
        if data_list[0]["right_gripper_eff"] is not None:
            right_gripper_eff = np.array([d["right_gripper_eff"] for d in data_list])
        duration = timestamps[-1] - timestamps[0]
        avg_hz = (len(timestamps) - 1) / duration if duration > 0 else 0
        ep_dir = self.output_dir / f"episode_{self.episode_count:04d}"
        ep_dir.mkdir(parents=True, exist_ok=True)
        npz_kwargs = {
            "timestamps": timestamps,
            "left_arm_position": left_pos,
            "right_arm_position": right_pos,
            "joint_names": self._joint_names if self._joint_names else [],
        }
        if left_vel is not None:
            npz_kwargs["left_arm_velocity"] = left_vel
        if left_eff is not None:
            npz_kwargs["left_arm_effort"] = left_eff
        if right_vel is not None:
            npz_kwargs["right_arm_velocity"] = right_vel
        if right_eff is not None:
            npz_kwargs["right_arm_effort"] = right_eff
        if left_gripper_pos.size > 0:
            npz_kwargs["left_gripper_position"] = left_gripper_pos
        if right_gripper_pos.size > 0:
            npz_kwargs["right_gripper_position"] = right_gripper_pos
        if self._gripper_names:
            npz_kwargs["gripper_joint_names"] = self._gripper_names
        if left_gripper_vel is not None and left_gripper_vel.size > 0:
            npz_kwargs["left_gripper_velocity"] = left_gripper_vel
        if left_gripper_eff is not None and left_gripper_eff.size > 0:
            npz_kwargs["left_gripper_effort"] = left_gripper_eff
        if right_gripper_vel is not None and right_gripper_vel.size > 0:
            npz_kwargs["right_gripper_velocity"] = right_gripper_vel
        if right_gripper_eff is not None and right_gripper_eff.size > 0:
            npz_kwargs["right_gripper_effort"] = right_gripper_eff
        np.savez_compressed(ep_dir / "joint_data.npz", **npz_kwargs)
        meta = {
            "episode_id": self.episode_count,
            "task": task,
            "robot_model": robot_model,
            "num_frames": len(timestamps),
            "duration": round(duration, 3),
            "avg_hz": round(avg_hz, 1),
        }
        with open(ep_dir / "episode.json", "w") as f:
            json.dump(meta, f, indent=2)
        self.episode_count += 1
        self._joint_indices = None
        self._joint_names = None
        self._gripper_names = None
        return meta


def main(
    server: Annotated[str, typer.Option(help="Robot server address, e.g. localhost:50051")] = "localhost:50051",
    out: Annotated[str, typer.Option(help="Output directory")] = "./recorded_joint_data",
    task: Annotated[str, typer.Option(help="Task name")] = "default",
):
    signal.signal(signal.SIGINT, lambda s, f: sys.exit(0))
    robot = connect(f"x2://{server}")
    model = robot.get_robot_model()
    print(f"Connected: {model}")
    #robot.system.set_work_mode(RobotModeParam(mode=RobotWorkMode.SDK), timeout=10)
    recorder = JointRecorder(robot, Path(out))
    print(f"Output: {recorder.output_dir}")
    print("Press Enter to START recording, then press Enter again to STOP\n")
    while True:
        print(f"\n=== Episode {recorder.episode_count} ===")
        input("Press Enter to START recording...")
        recorder.start_recording()
        print("Recording... press Enter to STOP\n")
        try:
            import select
            while True:
                if select.select([sys.stdin], [], [], 0.5)[0]:
                    input()
                    break
                if not recorder.is_recording:
                    print("  recording stopped unexpectedly")
                    break
        except KeyboardInterrupt:
            break
        info = recorder.stop_recording(task=task, robot_model=model)
        if info:
            print(f"Saved episode {info['episode_id']}: {info['num_frames']} frames, {info['duration']:.2f}s, {info['avg_hz']:.0f} Hz")
        act = input("\nContinue? (y/n, default n): ").strip().lower()
        if act != "y":
            break
    print(f"\nDone. {recorder.episode_count} episodes in {recorder.output_dir}")


if __name__ == "__main__":
    typer.run(main)
