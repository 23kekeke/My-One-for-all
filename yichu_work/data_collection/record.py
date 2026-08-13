import time
import signal
import sys
from pathlib import Path
from typing import Annotated
import typer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_collection.data_collector import DataCollector
from data_collection.collection_config import CollectionConfig
from x2robot import connect

def main(
    server: Annotated[str, typer.Option(help="Server address, e.g. localhost:50051")] = "localhost:50051",
    out: Annotated[str, typer.Option(help="Output directory")] = "./collected_data",
    hz: Annotated[float, typer.Option(help="Target recording frequency")] = 30.0,
    task: Annotated[str, typer.Option(help="Task name for recording")] = "pull the door",
):
    signal.signal(signal.SIGINT, lambda s, f: sys.exit(0))

    robot = connect(f"x2://{server}")
    print(f"Connected: {robot.get_robot_model()}")

    config = CollectionConfig()
    config.slave_joint_names = [
        'left_arm_joint_states',
        'right_arm_joint_states',
        'lift_joint_states',
        'left_gripper_joint_states',
        'right_gripper_joint_states',
        'head_joint_states',
    ]
    config.enable_head_rgb_stream = True
    config.enable_left_arm_rgb_stream = True
    config.enable_right_arm_rgb_stream = True
    config.enable_left_arm_end_pose = True
    config.enable_right_arm_end_pose = True
    config.enable_odometry = True
    config.enable_chassis_imu = True





    collector = DataCollector(
        robot=robot,
        output_dir=out,
        target_hz=hz,
        collection_config=config,
        image_quality=95,
        downsample_joint_states=True,
        use_video_storage=True,
    )

    print(f"Output: {collector.output_dir}")
    print(f"Target FPS: {collector.target_hz} Hz\n")

    episode_idx = 0
    while True:
        current_ep = collector.episode_count
        print(f"\n=== Episode {current_ep} ===")
        input("Press Enter to START recording...")

        collector.start_recording(task=task)
        print("Recording... press Enter to STOP\n")

        import select
        while True:
            try:
                if select.select([sys.stdin], [], [], 1)[0]:
                    input()
                    break
                collector.print_stats()
            except KeyboardInterrupt:
                break

        if collector.is_recording:
            info = collector.stop_recording()
            if info:
                print(f"\nSaved Episode {info['episode_id']}: {info['num_frames']} frames, {info['duration']:.2f}s")

        act = input("\nContinue? (y/n/r, default n): ").strip().lower()
        if act == "r":
            if not collector.reject_last_episode():
                print("Unable to record the rejection safely; stopping without overwrite.")
                break
            print("\n=== Re-recording the same episode ===\n")
            continue
        elif act != "y":
            break
        episode_idx += 1

    print(f"\nDone. {collector.episode_count} episodes in {collector.output_dir}")


if __name__ == "__main__":
    typer.run(main)
