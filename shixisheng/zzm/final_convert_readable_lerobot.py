#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROS2 MCAP -> LeRobot 数据集转换脚本（可播放视频版）

核心改动：
1. 直接读取 .mcap 文件；也兼容 .mcap.zstd。
2. 不再手动拼 LeRobot 的 parquet / mp4 / meta，而是调用 LeRobotDataset.create/add_frame/save_episode/finalize。
   这样 LeRobot 会自动生成更规范的 data、videos、meta 结构。
3. 图像按 LeRobot 常用的 RGB + CHW(uint8) 格式写入。
4. 转换结束后，可选用 ffmpeg 把所有 mp4 转成 H.264/avc1/yuv420p，方便 VS Code / 浏览器预览。

示例：
python final_convert_readable_lerobot.py \
  --mcap /home/yichu/shixisheng/zzm/pulldoor_0001@MASTER_SLAVE_MODE@2026_05_29_10_41_39/xxx.mcap \
  --output-dir /home/yichu/shixisheng/zzm/lerobot_dataset_readable \
  --repo-id local/pulldoor_readable \
  --task "pull door" \
  --overwrite

如果你的文件仍然是 .mcap.zstd，也可以直接：
python final_convert_readable_lerobot.py --mcap xxx.mcap.zstd --output-dir ... --repo-id ... --overwrite
"""

import argparse
import inspect
import shutil
import subprocess
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import cv2
import numpy as np
from tqdm import tqdm

import zstandard as zstd
import mcap.reader as mcap_reader

from rclpy.serialization import deserialize_message
from sensor_msgs.msg import JointState, CompressedImage
from std_msgs.msg import Float64MultiArray


# ========== 1. 根据你的机器人实际 topic 配置这里 ==========
# 图像 topic：默认使用压缩图像 sensor_msgs/msg/CompressedImage
CAMERA_TOPICS = {
    "head_front": "/camera_head_front/color/image_raw/compressed",
    "chest_front": "/camera_chest_front/color/image_raw/compressed",
    "camera1": "/camera1/usb_cam1/image_raw/image_compressed",
    "camera3": "/camera3/usb_cam3/image_raw/image_compressed",
}

# 状态 topic：JointState。最终拼接为 14 维 observation.state
STATE_TOPICS = {
    "left_arm": "/left_arm/joint_states",             # 6维
    "right_arm": "/right_arm/joint_states",           # 6维
    "left_gripper": "/left_gripper/joint_states",     # 1维
    "right_gripper": "/right_gripper/joint_states",   # 1维
}

# 动作 topic：Float64MultiArray。最终拼接为 14 维 action
ACTION_TOPICS = {
    "left_arm": "/left_arm_joint_controller/commands",          # 6维
    "right_arm": "/right_arm_joint_controller/commands",        # 6维
    "left_gripper": "/left_gripper_controller/commands",        # 1维
    "right_gripper": "/right_gripper_controller/commands",      # 1维
}

STATE_NAMES = [
    "left_arm_joint1", "left_arm_joint2", "left_arm_joint3",
    "left_arm_joint4", "left_arm_joint5", "left_arm_joint6",
    "right_arm_joint1", "right_arm_joint2", "right_arm_joint3",
    "right_arm_joint4", "right_arm_joint5", "right_arm_joint6",
    "left_arm_gripper", "right_arm_gripper",
]

ACTION_NAMES = [
    "left_arm_joint1_cmd", "left_arm_joint2_cmd", "left_arm_joint3_cmd",
    "left_arm_joint4_cmd", "left_arm_joint5_cmd", "left_arm_joint6_cmd",
    "right_arm_joint1_cmd", "right_arm_joint2_cmd", "right_arm_joint3_cmd",
    "right_arm_joint4_cmd", "right_arm_joint5_cmd", "right_arm_joint6_cmd",
    "left_arm_gripper_cmd", "right_arm_gripper_cmd",
]


# ========== 2. LeRobot 兼容导入 ==========
def import_lerobot_dataset():
    """不同版本 LeRobot 的导入路径可能不同，这里做兼容。"""
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        return LeRobotDataset
    except Exception:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
        return LeRobotDataset


def create_dataset_compatible(
    LeRobotDataset,
    repo_id: str,
    root: Path,
    fps: int,
    robot_type: str,
    features: Dict[str, Any],
    use_videos: bool = True,
    video_backend: Optional[str] = "pyav",
):
    """
    不同 LeRobot 版本 create() 支持的参数不完全一致。
    这里用 inspect 过滤掉当前版本不支持的参数，减少版本报错。
    """
    sig = inspect.signature(LeRobotDataset.create)
    kwargs = {
        "repo_id": repo_id,
        "root": root,
        "fps": fps,
        "robot_type": robot_type,
        "features": features,
        "use_videos": use_videos,
        "video_backend": video_backend,
    }
    filtered = {k: v for k, v in kwargs.items() if k in sig.parameters and v is not None}
    return LeRobotDataset.create(**filtered)


# ========== 3. MCAP 读取与 ROS 消息解析 ==========
@contextmanager
def open_mcap_reader(mcap_path: Path):
    """
    打开 .mcap 或 .mcap.zstd。
    - .mcap：直接读取
    - .mcap.zstd：边解压边读取
    """
    f = open(mcap_path, "rb")
    stream = None
    try:
        if mcap_path.name.endswith(".zstd"):
            dctx = zstd.ZstdDecompressor()
            stream = dctx.stream_reader(f)
            reader = mcap_reader.make_reader(stream)
        else:
            reader = mcap_reader.make_reader(f)
        yield reader
    finally:
        if stream is not None:
            stream.close()
        f.close()


def decode_compressed_image(msg_data: bytes) -> Optional[np.ndarray]:
    """把 sensor_msgs/msg/CompressedImage 解码为 RGB HWC uint8 图像。"""
    try:
        msg = deserialize_message(msg_data, CompressedImage)
        bgr = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            return None
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return rgb.astype(np.uint8)
    except Exception:
        return None


def parse_joint_state(msg_data: bytes) -> Optional[np.ndarray]:
    """解析 sensor_msgs/msg/JointState，取 position。"""
    try:
        msg = deserialize_message(msg_data, JointState)
        return np.asarray(msg.position, dtype=np.float32)
    except Exception:
        return None


def parse_action(msg_data: bytes) -> Optional[np.ndarray]:
    """解析 std_msgs/msg/Float64MultiArray，作为关节控制命令。"""
    try:
        msg = deserialize_message(msg_data, Float64MultiArray)
        return np.asarray(msg.data, dtype=np.float32)
    except Exception:
        return None


def read_mcap(mcap_path: Path) -> Dict[str, Dict[str, Any]]:
    """读取 MCAP 中需要的图像、状态、动作 topic。"""
    print(f"Reading MCAP: {mcap_path}")

    data: Dict[str, Dict[str, Any]] = {}
    for name in CAMERA_TOPICS:
        data[f"image:{name}"] = {"ts": [], "data": []}
    for name in STATE_TOPICS:
        data[f"state:{name}"] = {"ts": [], "data": []}
    for name in ACTION_TOPICS:
        data[f"action:{name}"] = {"ts": [], "data": []}

    topic_to_key = {}
    for name, topic in CAMERA_TOPICS.items():
        topic_to_key[topic] = ("image", name)
    for name, topic in STATE_TOPICS.items():
        topic_to_key[topic] = ("state", name)
    for name, topic in ACTION_TOPICS.items():
        topic_to_key[topic] = ("action", name)

    with open_mcap_reader(mcap_path) as reader:
        for _schema, channel, message in tqdm(reader.iter_messages(), desc="Reading messages"):
            topic = channel.topic
            if topic not in topic_to_key:
                continue

            kind, name = topic_to_key[topic]
            t = message.log_time * 1e-9

            if kind == "image":
                item = decode_compressed_image(message.data)
            elif kind == "state":
                item = parse_joint_state(message.data)
            else:
                item = parse_action(message.data)

            if item is None:
                continue

            key = f"{kind}:{name}"
            data[key]["ts"].append(t)
            data[key]["data"].append(item)

    # 时间戳转 numpy，方便后续 searchsorted
    for key in data:
        data[key]["ts"] = np.asarray(data[key]["ts"], dtype=np.float64)

    print("\nLoaded message counts:")
    for key, value in data.items():
        print(f"  {key}: {len(value['data'])}")

    return data


# ========== 4. 时间同步与向量拼接 ==========
def nearest_index(times: np.ndarray, target_t: float) -> int:
    """在时间数组中找最接近 target_t 的索引。"""
    if len(times) == 0:
        return 0
    idx = np.searchsorted(times, target_t)
    if idx <= 0:
        return 0
    if idx >= len(times):
        return len(times) - 1
    before = idx - 1
    after = idx
    if abs(times[before] - target_t) <= abs(times[after] - target_t):
        return before
    return after


def pad_or_trim(vec: np.ndarray, dim: int) -> np.ndarray:
    """把向量补齐或截断到指定维度。"""
    vec = np.asarray(vec, dtype=np.float32).reshape(-1)
    if len(vec) < dim:
        return np.pad(vec, (0, dim - len(vec))).astype(np.float32)
    if len(vec) > dim:
        return vec[:dim].astype(np.float32)
    return vec.astype(np.float32)


def get_nearest_item(
    data: Dict[str, Dict[str, Any]],
    key: str,
    target_t: float,
    max_dt: float,
):
    """按时间同步取最近的数据；超过容差就返回 None。"""
    times = data[key]["ts"]
    if len(times) == 0:
        return None
    idx = nearest_index(times, target_t)
    if abs(times[idx] - target_t) > max_dt:
        return None
    return data[key]["data"][idx]


def build_synced_frames(
    data: Dict[str, Dict[str, Any]],
    fps: int,
    task: str,
    reference_camera: str,
    max_image_dt: float,
    max_state_dt: float,
    max_action_dt: float,
    dummy_action: bool = False,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    将不同频率的图像、状态、动作 topic 按时间同步，生成 LeRobot 的 frame 列表。
    每个 frame 包含：observation.state、action、observation.images.*、task。
    """

    # 只使用实际存在消息的相机，避免某个相机没录到导致整个 episode 失败
    active_cameras = [name for name in CAMERA_TOPICS if len(data[f"image:{name}"]["data"]) > 0]
    if not active_cameras:
        raise RuntimeError("No camera messages found. Please check CAMERA_TOPICS.")

    if reference_camera not in active_cameras:
        print(f"Warning: reference camera '{reference_camera}' is empty. Use '{active_cameras[0]}' instead.")
        reference_camera = active_cameras[0]

    # 状态 topic 必须存在，否则没有 observation.state
    required_state_keys = [f"state:{name}" for name in ["left_arm", "right_arm", "left_gripper", "right_gripper"]]
    for key in required_state_keys:
        if len(data[key]["data"]) == 0:
            raise RuntimeError(f"State topic is empty: {key}. Please check STATE_TOPICS.")

    # 默认使用真实 action；如果 action 缺失，可以用 --dummy-action 生成零动作跑通格式
    required_action_keys = [f"action:{name}" for name in ["left_arm", "right_arm", "left_gripper", "right_gripper"]]
    if not dummy_action:
        for key in required_action_keys:
            if len(data[key]["data"]) == 0:
                raise RuntimeError(f"Action topic is empty: {key}. Use --dummy-action only for format testing.")

    # 用各 topic 的重叠时间段作为有效范围
    all_keys_for_range = [f"image:{c}" for c in active_cameras] + required_state_keys
    if not dummy_action:
        all_keys_for_range += required_action_keys

    start_t = max(data[k]["ts"][0] for k in all_keys_for_range)
    end_t = min(data[k]["ts"][-1] for k in all_keys_for_range)
    if end_t <= start_t:
        raise RuntimeError(f"Invalid common time range: {start_t} -> {end_t}")

    target_times = np.arange(start_t, end_t, 1.0 / fps, dtype=np.float64)

    print("\nSynchronization configuration:")
    print(f"  Active cameras: {active_cameras}")
    print(f"  Reference camera: {reference_camera}")
    print(f"  Common time range: {start_t:.6f} -> {end_t:.6f}")
    print(f"  Target FPS: {fps}")
    print(f"  Target frames: {len(target_times)}")

    frames: List[Dict[str, Any]] = []
    dropped = 0

    for target_t in tqdm(target_times, desc="Synchronizing frames"):
        frame: Dict[str, Any] = {"task": task}
        ok = True

        # 图像：LeRobot 成功脚本使用 CHW，这里也转为 CHW uint8
        for cam in active_cameras:
            img = get_nearest_item(data, f"image:{cam}", target_t, max_image_dt)
            if img is None:
                ok = False
                break
            img_chw = np.ascontiguousarray(img.transpose(2, 0, 1))  # RGB HWC -> RGB CHW
            frame[f"observation.images.{cam}"] = img_chw

        if not ok:
            dropped += 1
            continue

        # 状态：left_arm(6) + right_arm(6) + left_gripper(1) + right_gripper(1) = 14维
        left_arm_state = get_nearest_item(data, "state:left_arm", target_t, max_state_dt)
        right_arm_state = get_nearest_item(data, "state:right_arm", target_t, max_state_dt)
        left_gripper_state = get_nearest_item(data, "state:left_gripper", target_t, max_state_dt)
        right_gripper_state = get_nearest_item(data, "state:right_gripper", target_t, max_state_dt)

        if any(x is None for x in [left_arm_state, right_arm_state, left_gripper_state, right_gripper_state]):
            dropped += 1
            continue

        state = np.concatenate([
            pad_or_trim(left_arm_state, 6),
            pad_or_trim(right_arm_state, 6),
            pad_or_trim(left_gripper_state, 1),
            pad_or_trim(right_gripper_state, 1),
        ]).astype(np.float32)
        frame["observation.state"] = state

        # 动作：真实控制命令，维度同样是 14维
        if dummy_action:
            action = np.zeros((14,), dtype=np.float32)
        else:
            left_arm_action = get_nearest_item(data, "action:left_arm", target_t, max_action_dt)
            right_arm_action = get_nearest_item(data, "action:right_arm", target_t, max_action_dt)
            left_gripper_action = get_nearest_item(data, "action:left_gripper", target_t, max_action_dt)
            right_gripper_action = get_nearest_item(data, "action:right_gripper", target_t, max_action_dt)

            if any(x is None for x in [left_arm_action, right_arm_action, left_gripper_action, right_gripper_action]):
                dropped += 1
                continue

            action = np.concatenate([
                pad_or_trim(left_arm_action, 6),
                pad_or_trim(right_arm_action, 6),
                pad_or_trim(left_gripper_action, 1),
                pad_or_trim(right_gripper_action, 1),
            ]).astype(np.float32)

        frame["action"] = action
        frames.append(frame)

    print(f"\nSynchronized frames: {len(frames)}")
    print(f"Dropped frames: {dropped}")

    if not frames:
        raise RuntimeError("No synchronized frames found. Try increasing --max-image-dt / --max-state-dt / --max-action-dt.")

    return frames, active_cameras


# ========== 5. 写入 LeRobot 数据集 ==========
def build_features(first_frame: Dict[str, Any], active_cameras: List[str], use_videos: bool) -> Dict[str, Any]:
    """根据第一帧自动生成 LeRobot features。"""
    features: Dict[str, Any] = {
        "observation.state": {
            "dtype": "float32",
            "shape": (14,),
            "names": STATE_NAMES,
        },
        "action": {
            "dtype": "float32",
            "shape": (14,),
            "names": ACTION_NAMES,
        },
    }

    for cam in active_cameras:
        key = f"observation.images.{cam}"
        img = first_frame[key]
        c, h, w = img.shape
        features[key] = {
            "dtype": "video" if use_videos else "image",
            "shape": (c, h, w),
            "names": ["channels", "height", "width"],
        }

    return features


def write_lerobot_dataset(
    frames: List[Dict[str, Any]],
    active_cameras: List[str],
    output_dir: Path,
    repo_id: str,
    robot_type: str,
    fps: int,
    use_videos: bool,
    video_backend: Optional[str],
    overwrite: bool,
):
    """用 LeRobot 官方 API 写入数据集，而不是手写 meta/parquet/mp4。"""
    if output_dir.exists():
        if overwrite:
            print(f"Removing existing output dir: {output_dir}")
            shutil.rmtree(output_dir)
        else:
            raise FileExistsError(f"{output_dir} already exists. Use --overwrite to replace it.")

    features = build_features(frames[0], active_cameras, use_videos)

    print("\nLeRobot features:")
    for k, v in features.items():
        print(f"  {k}: {v}")

    LeRobotDataset = import_lerobot_dataset()
    dataset = create_dataset_compatible(
        LeRobotDataset=LeRobotDataset,
        repo_id=repo_id,
        root=output_dir,
        fps=fps,
        robot_type=robot_type,
        features=features,
        use_videos=use_videos,
        video_backend=video_backend,
    )

    for i, frame in enumerate(tqdm(frames, desc="Writing LeRobot frames")):
        dataset.add_frame(frame)
        if i > 0 and i % 100 == 0:
            print(f"  added {i}/{len(frames)} frames")

    dataset.save_episode()

    # finalize/consolidate 会生成统计信息、索引和最终 meta。
    if hasattr(dataset, "finalize"):
        dataset.finalize()
    elif hasattr(dataset, "consolidate"):
        dataset.consolidate()

    print("\nLeRobot dataset saved:", output_dir)


# ========== 6. 可选：把 mp4 转成 VS Code 更容易播放的 H.264 ==========
def transcode_videos_to_h264(root: Path):
    """
    将 LeRobot 生成的 mp4 转为 H.264/avc1/yuv420p。
    这一步主要是为了 VS Code / 浏览器预览更稳定，不影响训练字段本身。
    """
    videos = sorted(root.glob("videos/observation.images.*/chunk-*/file-*.mp4"))
    if not videos:
        print("No mp4 videos found for transcoding.")
        return

    print("\nTranscoding videos to H.264 for VS Code preview...")
    for video in videos:
        tmp = video.with_name(video.stem + "_h264_tmp.mp4")
        cmd = [
            "ffmpeg", "-y",
            "-i", str(video),
            "-c:v", "libx264",
            "-profile:v", "baseline",
            "-level", "4.0",
            "-pix_fmt", "yuv420p",
            "-tag:v", "avc1",
            "-movflags", "+faststart",
            "-an",
            str(tmp),
        ]
        print(f"  {video}")
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            tmp.replace(video)
        except FileNotFoundError:
            print("  ffmpeg not found. Skip H.264 transcoding.")
            return
        except subprocess.CalledProcessError as e:
            print(f"  Warning: ffmpeg failed for {video}: {e}")
            if tmp.exists():
                tmp.unlink()

    print("H.264 transcoding finished.")


# ========== 7. 主函数 ==========
def parse_args():
    parser = argparse.ArgumentParser(description="Convert ROS2 MCAP to readable LeRobot dataset.")
    parser.add_argument("--mcap", required=True, help="Input .mcap or .mcap.zstd file")
    parser.add_argument("--output-dir", required=True, help="LeRobot output root")
    parser.add_argument("--repo-id", default="local/pulldoor_readable", help="LeRobot repo id")
    parser.add_argument("--robot-type", default="quanta_x1_raw_joints")
    parser.add_argument("--task", default="pull door")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--reference-camera", default="head_front")
    parser.add_argument("--max-image-dt", type=float, default=0.12, help="image sync tolerance, seconds")
    parser.add_argument("--max-state-dt", type=float, default=0.05, help="state sync tolerance, seconds")
    parser.add_argument("--max-action-dt", type=float, default=0.05, help="action sync tolerance, seconds")
    parser.add_argument("--use-videos", action="store_true", default=True, help="store images as videos")
    parser.add_argument("--no-videos", dest="use_videos", action="store_false", help="store images individually instead of videos")
    parser.add_argument("--video-backend", default="pyav", choices=["pyav", "opencv"])
    parser.add_argument("--dummy-action", action="store_true", help="use zero action only for format testing")
    parser.add_argument("--no-transcode", action="store_true", help="do not transcode mp4 to H.264 after conversion")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    mcap_path = Path(args.mcap).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not mcap_path.exists():
        raise FileNotFoundError(f"MCAP file not found: {mcap_path}")

    print("=" * 70)
    print("ROS2 MCAP -> LeRobot readable dataset")
    print("=" * 70)
    print("Input MCAP:", mcap_path)
    print("Output dir:", output_dir)
    print("Repo id:", args.repo_id)
    print("Task:", args.task)

    data = read_mcap(mcap_path)
    frames, active_cameras = build_synced_frames(
        data=data,
        fps=args.fps,
        task=args.task,
        reference_camera=args.reference_camera,
        max_image_dt=args.max_image_dt,
        max_state_dt=args.max_state_dt,
        max_action_dt=args.max_action_dt,
        dummy_action=args.dummy_action,
    )

    write_lerobot_dataset(
        frames=frames,
        active_cameras=active_cameras,
        output_dir=output_dir,
        repo_id=args.repo_id,
        robot_type=args.robot_type,
        fps=args.fps,
        use_videos=args.use_videos,
        video_backend=args.video_backend,
        overwrite=args.overwrite,
    )

    if args.use_videos and not args.no_transcode:
        transcode_videos_to_h264(output_dir)

    print("\nDone.")
    print("You can check videos with:")
    print(f"  ffprobe -hide_banner {output_dir}/videos/observation.images.camera3/chunk-000/file-000.mp4")
    print("You should see: Video: h264 ... avc1 ... yuv420p")


if __name__ == "__main__":
    main()
