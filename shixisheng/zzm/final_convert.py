#!/usr/bin/env python3
"""
完整的ROS2 bag到LeRobot标准格式转换脚本
包含所有必要的metadata文件
"""
import json
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import zstandard as zstd
import mcap.reader as mcap_reader

from rclpy.serialization import deserialize_message
from sensor_msgs.msg import JointState, CompressedImage
from std_msgs.msg import Float64MultiArray

def decode_compressed_image(msg_data):
    """解码压缩图像"""
    try:
        msg = deserialize_message(msg_data, CompressedImage)
        img = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
        if img is not None:
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except Exception as e:
        pass
    return None

def parse_joint_state(msg_data):
    """解析关节状态"""
    try:
        msg = deserialize_message(msg_data, JointState)
        return np.array(msg.position, dtype=np.float32)
    except Exception as e:
        return None

def parse_action(msg_data):
    """解析动作命令"""
    try:
        msg = deserialize_message(msg_data, Float64MultiArray)
        return np.array(msg.data, dtype=np.float32)
    except Exception as e:
        return None

def nearest_by_time(target_t, ts_list, data_list, max_diff=0.05):
    """按时间查找最近的数据"""
    if len(ts_list) == 0:
        return None
    
    idx = np.searchsorted(ts_list, target_t)
    
    candidates = []
    if idx > 0:
        candidates.append(idx - 1)
    if idx < len(ts_list):
        candidates.append(idx)
    
    best_idx = min(candidates, key=lambda i: abs(ts_list[i] - target_t))
    diff = abs(ts_list[best_idx] - target_t)
    
    if diff > max_diff:
        return None
    
    return data_list[best_idx]

def read_mcap_zstd(mcap_file):
    """读取mcap.zstd文件"""
    print(f"Reading {mcap_file.name}...")
    
    data = {
        # 图像
        'head_front': {'ts': [], 'data': []},
        'chest_front': {'ts': [], 'data': []},
        'camera1': {'ts': [], 'data': []},
        'camera3': {'ts': [], 'data': []},
        
        # 状态
        'left_arm_state': {'ts': [], 'data': []},
        'right_arm_state': {'ts': [], 'data': []},
        'left_gripper_state': {'ts': [], 'data': []},
        'right_gripper_state': {'ts': [], 'data': []},
        
        # 动作
        'left_arm_action': {'ts': [], 'data': []},
        'right_arm_action': {'ts': [], 'data': []},
        'left_gripper_action': {'ts': [], 'data': []},
        'right_gripper_action': {'ts': [], 'data': []},
    }
    
    # topic映射
    topic_mapping = {
        '/camera_head_front/color/image_raw/compressed': 'head_front',
        '/camera_chest_front/color/image_raw/compressed': 'chest_front',
        '/camera1/usb_cam1/image_raw/image_compressed': 'camera1',
        '/camera3/usb_cam3/image_raw/image_compressed': 'camera3',
        
        '/left_arm/joint_states': 'left_arm_state',
        '/right_arm/joint_states': 'right_arm_state',
        '/left_gripper/joint_states': 'left_gripper_state',
        '/right_gripper/joint_states': 'right_gripper_state',
        
        '/left_arm_joint_controller/commands': 'left_arm_action',
        '/right_arm_joint_controller/commands': 'right_arm_action',
        '/left_gripper_controller/commands': 'left_gripper_action',
        '/right_gripper_controller/commands': 'right_gripper_action',
    }
    
    with open(mcap_file, 'rb') as f:
        # 解压zstd
        dctx = zstd.ZstdDecompressor()
        stream = dctx.stream_reader(f)
        reader = mcap_reader.make_reader(stream)
        
        for schema, channel, message in tqdm(reader.iter_messages(), desc="Reading messages"):
            topic = channel.topic
            t = message.log_time / 1e9
            
            if topic not in topic_mapping:
                continue
            
            key = topic_mapping[topic]
            
            if 'img' in key or 'camera' in key or 'head_front' in key or 'chest_front' in key:
                # 图像
                img = decode_compressed_image(message.data)
                if img is not None:
                    data[key]['ts'].append(t)
                    data[key]['data'].append(img)
            elif 'state' in key:
                # 状态
                state = parse_joint_state(message.data)
                if state is not None:
                    data[key]['ts'].append(t)
                    data[key]['data'].append(state)
            elif 'action' in key:
                # 动作
                action = parse_action(message.data)
                if action is not None:
                    data[key]['ts'].append(t)
                    data[key]['data'].append(action)
    
    # 转换为numpy数组
    for key in data:
        data[key]['ts'] = np.array(data[key]['ts'], dtype=np.float64)
    
    return data

def nearest_index(times, target_t):
    """
    找到时间数组中与目标时间最接近的索引
    
    Args:
        times: numpy数组，时间戳列表
        target_t: 目标时间
    
    Returns:
        最接近的索引
    """
    if len(times) == 0:
        return 0
    
    idx = np.searchsorted(times, target_t)
    
    # 边界处理
    if idx <= 0:
        return 0
    if idx >= len(times):
        return len(times) - 1
    
    # 比较前后哪个更接近
    before = idx - 1
    after = idx
    
    if abs(times[before] - target_t) <= abs(times[after] - target_t):
        return before
    return after

def build_episode_dataframe(data, ep_idx=0, fps=30):
    """
    构建episode数据框 - 使用原始时间戳同步（参考您同事的方法）
    
    Args:
        data: 从mcap读取的原始数据字典
        ep_idx: episode索引
        fps: 目标帧率（用于metadata，不影响实际时间戳）
    
    Returns:
        df: pandas DataFrame包含所有帧数据
        video_frames: 字典，包含各相机的图像序列
        stats: 统计信息字典
    """
    # 选择head_front相机作为时间基准（频率最高且稳定）
    base_ts = data['head_front']['ts']
    base_imgs = data['head_front']['data']
    
    if len(base_ts) == 0:
        print("  ❌ No head camera images, skip this episode")
        return None, None, None
    
    print(f"  Reference camera (head_front): {len(base_ts)} frames")
    
    rows = []
    video_frames = {
        'head_front': [],
        'chest_front': [],
        'camera1': [],
        'camera3': []
    }
    
    timestamps = []
    states_list = []
    actions_list = []
    
    # 同步容差：50ms
    max_sync_diff = 0.05
    
    # 统计同步失败的数量
    sync_failures = {
        'chest_front': 0,
        'camera1': 0,
        'camera3': 0,
        'left_arm_state': 0,
        'right_arm_state': 0,
        'left_gripper_state': 0,
        'right_gripper_state': 0,
        'left_arm_action': 0,
        'right_arm_action': 0,
        'left_gripper_action': 0,
        'right_gripper_action': 0
    }
    
    # 记录起始时间（用于归一化）
    start_time = base_ts[0]
    
    print(f"  Processing frames with time synchronization...")
    
    for frame_idx, target_t in enumerate(base_ts):
        # ========== 1. 获取所有相机图像 ==========
        # head_front（基准）
        head_img = base_imgs[frame_idx]
        
        # chest_front
        chest_idx = nearest_index(data['chest_front']['ts'], target_t)
        chest_t, chest_img = data['chest_front']['ts'][chest_idx], data['chest_front']['data'][chest_idx]
        if abs(chest_t - target_t) > max_sync_diff:
            sync_failures['chest_front'] += 1
            continue
        
        # camera1 (left wrist)
        camera1_idx = nearest_index(data['camera1']['ts'], target_t)
        camera1_t, camera1_img = data['camera1']['ts'][camera1_idx], data['camera1']['data'][camera1_idx]
        if abs(camera1_t - target_t) > max_sync_diff:
            sync_failures['camera1'] += 1
            continue
        
        # camera3 (right wrist)
        camera3_idx = nearest_index(data['camera3']['ts'], target_t)
        camera3_t, camera3_img = data['camera3']['ts'][camera3_idx], data['camera3']['data'][camera3_idx]
        if abs(camera3_t - target_t) > max_sync_diff:
            sync_failures['camera3'] += 1
            continue
        
        # ========== 2. 获取关节状态 ==========
        # left arm state (6 joints)
        left_arm_idx = nearest_index(data['left_arm_state']['ts'], target_t)
        left_arm_t, left_arm_state = data['left_arm_state']['ts'][left_arm_idx], data['left_arm_state']['data'][left_arm_idx]
        if abs(left_arm_t - target_t) > max_sync_diff:
            sync_failures['left_arm_state'] += 1
            continue
        # 确保有6个关节
        if len(left_arm_state) < 6:
            left_arm_state = np.pad(left_arm_state, (0, 6 - len(left_arm_state)))
        elif len(left_arm_state) > 6:
            left_arm_state = left_arm_state[:6]
        
        # right arm state (6 joints)
        right_arm_idx = nearest_index(data['right_arm_state']['ts'], target_t)
        right_arm_t, right_arm_state = data['right_arm_state']['ts'][right_arm_idx], data['right_arm_state']['data'][right_arm_idx]
        if abs(right_arm_t - target_t) > max_sync_diff:
            sync_failures['right_arm_state'] += 1
            continue
        if len(right_arm_state) < 6:
            right_arm_state = np.pad(right_arm_state, (0, 6 - len(right_arm_state)))
        elif len(right_arm_state) > 6:
            right_arm_state = right_arm_state[:6]
        
        # left gripper state (1 joint)
        left_gripper_idx = nearest_index(data['left_gripper_state']['ts'], target_t)
        left_gripper_t, left_gripper_state = data['left_gripper_state']['ts'][left_gripper_idx], data['left_gripper_state']['data'][left_gripper_idx]
        if abs(left_gripper_t - target_t) > max_sync_diff:
            sync_failures['left_gripper_state'] += 1
            continue
        if len(left_gripper_state) < 1:
            left_gripper_state = np.zeros(1)
        else:
            left_gripper_state = left_gripper_state[:1]
        
        # right gripper state (1 joint)
        right_gripper_idx = nearest_index(data['right_gripper_state']['ts'], target_t)
        right_gripper_t, right_gripper_state = data['right_gripper_state']['ts'][right_gripper_idx], data['right_gripper_state']['data'][right_gripper_idx]
        if abs(right_gripper_t - target_t) > max_sync_diff:
            sync_failures['right_gripper_state'] += 1
            continue
        if len(right_gripper_state) < 1:
            right_gripper_state = np.zeros(1)
        else:
            right_gripper_state = right_gripper_state[:1]
        
        # ========== 3. 获取动作命令 ==========
        # left arm action (6 joints)
        left_arm_action_idx = nearest_index(data['left_arm_action']['ts'], target_t)
        left_arm_action_t, left_arm_action = data['left_arm_action']['ts'][left_arm_action_idx], data['left_arm_action']['data'][left_arm_action_idx]
        if abs(left_arm_action_t - target_t) > max_sync_diff:
            sync_failures['left_arm_action'] += 1
            continue
        if len(left_arm_action) < 6:
            left_arm_action = np.pad(left_arm_action, (0, 6 - len(left_arm_action)))
        elif len(left_arm_action) > 6:
            left_arm_action = left_arm_action[:6]
        
        # right arm action (6 joints)
        right_arm_action_idx = nearest_index(data['right_arm_action']['ts'], target_t)
        right_arm_action_t, right_arm_action = data['right_arm_action']['ts'][right_arm_action_idx], data['right_arm_action']['data'][right_arm_action_idx]
        if abs(right_arm_action_t - target_t) > max_sync_diff:
            sync_failures['right_arm_action'] += 1
            continue
        if len(right_arm_action) < 6:
            right_arm_action = np.pad(right_arm_action, (0, 6 - len(right_arm_action)))
        elif len(right_arm_action) > 6:
            right_arm_action = right_arm_action[:6]
        
        # left gripper action (1 joint)
        left_gripper_action_idx = nearest_index(data['left_gripper_action']['ts'], target_t)
        left_gripper_action_t, left_gripper_action = data['left_gripper_action']['ts'][left_gripper_action_idx], data['left_gripper_action']['data'][left_gripper_action_idx]
        if abs(left_gripper_action_t - target_t) > max_sync_diff:
            sync_failures['left_gripper_action'] += 1
            continue
        if len(left_gripper_action) < 1:
            left_gripper_action = np.zeros(1)
        else:
            left_gripper_action = left_gripper_action[:1]
        
        # right gripper action (1 joint)
        right_gripper_action_idx = nearest_index(data['right_gripper_action']['ts'], target_t)
        right_gripper_action_t, right_gripper_action = data['right_gripper_action']['ts'][right_gripper_action_idx], data['right_gripper_action']['data'][right_gripper_action_idx]
        if abs(right_gripper_action_t - target_t) > max_sync_diff:
            sync_failures['right_gripper_action'] += 1
            continue
        if len(right_gripper_action) < 1:
            right_gripper_action = np.zeros(1)
        else:
            right_gripper_action = right_gripper_action[:1]
        
        # ========== 4. 构建状态向量 (14维) ==========
        # 顺序: left_arm(6) + right_arm(6) + left_gripper(1) + right_gripper(1) = 14
        observation_state = np.concatenate([
            left_arm_state,   # 6
            right_arm_state,  # 6
            left_gripper_state,  # 1
            right_gripper_state  # 1
        ]).astype(np.float32)
        
        # ========== 5. 构建动作向量 (14维) ==========
        action = np.concatenate([
            left_arm_action,   # 6
            right_arm_action,  # 6
            left_gripper_action,  # 1
            right_gripper_action  # 1
        ]).astype(np.float32)
        
        # ========== 6. 验证维度 ==========
        if len(observation_state) != 14:
            print(f"    Warning: state dimension is {len(observation_state)}, expected 14")
            if len(observation_state) < 14:
                observation_state = np.pad(observation_state, (0, 14 - len(observation_state)))
            else:
                observation_state = observation_state[:14]
        
        if len(action) != 14:
            print(f"    Warning: action dimension is {len(action)}, expected 14")
            if len(action) < 14:
                action = np.pad(action, (0, 14 - len(action)))
            else:
                action = action[:14]
        
        # ========== 7. 保存这一帧 ==========
        # 计算归一化时间戳（相对于第一帧）
        timestamp = target_t - start_time
        
        rows.append({
            "episode_index": ep_idx,
            "frame_index": frame_idx,
            "timestamp": timestamp,
            "observation.state": observation_state.tolist(),
            "action": action.tolist(),
        })
        
        timestamps.append(target_t)
        states_list.append(observation_state)
        actions_list.append(action)
        
        # 保存图像（用于后续视频生成）
        video_frames['head_front'].append(head_img)
        video_frames['chest_front'].append(chest_img)
        video_frames['camera1'].append(camera1_img)
        video_frames['camera3'].append(camera3_img)
        
        # 进度显示（每100帧显示一次）
        if (frame_idx + 1) % 100 == 0:
            print(f"    Processed {frame_idx + 1}/{len(base_ts)} frames...")
    
    # ========== 8. 打印同步统计 ==========
    print(f"\n  📊 Synchronization Statistics:")
    total_processed = len(rows)
    total_original = len(base_ts)
    print(f"    Original frames: {total_original}")
    print(f"    Kept frames: {total_processed}")
    print(f"    Dropped frames: {total_original - total_processed}")
    
    if total_original - total_processed > 0:
        print(f"    Sync failures by topic:")
        for topic, count in sync_failures.items():
            if count > 0:
                print(f"      - {topic}: {count}")
    
    # ========== 9. 检查结果 ==========
    if len(rows) == 0:
        print("  ❌ No synchronized frames found!")
        return None, None, None
    
    # ========== 10. 创建DataFrame ==========
    df = pd.DataFrame(rows)
    
    # ========== 11. 计算统计数据 ==========
    states_array = np.array(states_list)
    actions_array = np.array(actions_list)
    
    stats = {
        "observation.state": {
            "mean": states_array.mean(axis=0).tolist(),
            "std": states_array.std(axis=0).tolist(),
            "min": states_array.min(axis=0).tolist(),
            "max": states_array.max(axis=0).tolist()
        },
        "action": {
            "mean": actions_array.mean(axis=0).tolist(),
            "std": actions_array.std(axis=0).tolist(),
            "min": actions_array.min(axis=0).tolist(),
            "max": actions_array.max(axis=0).tolist()
        },
        "timestamp": {
            "mean": float(np.mean(timestamps)),
            "std": float(np.std(timestamps)),
            "min": float(np.min(timestamps)),
            "max": float(np.max(timestamps))
        }
    }
    
    print(f"\n  ✅ Built {len(df)} synchronized frames")
    print(f"  📈 State range: [{states_array.min():.3f}, {states_array.max():.3f}]")
    print(f"  📈 Action range: [{actions_array.min():.3f}, {actions_array.max():.3f}]")
    
    return df, video_frames, stats

def save_video(images, video_path, fps=30):
    """保存视频"""
    if len(images) == 0:
        return False
    
    video_path = Path(video_path)
    video_path.parent.mkdir(parents=True, exist_ok=True)
    
    h, w, _ = images[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(video_path), fourcc, fps, (w, h))
    
    for img in images:
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        out.write(img_bgr)
    
    out.release()
    print(f"  Saved video: {video_path} ({len(images)} frames)")
    return True

def main():
    bag_dir = Path("/home/yichu/shixisheng/zzm/pulldoor_0001@MASTER_SLAVE_MODE@2026_05_29_10_41_39")
    output_dir = Path("/home/yichu/shixisheng/zzm/lerobot_dataset_complete")
    
    # 查找mcap.zstd文件
    mcap_files = list(bag_dir.glob("*.mcap.zstd"))
    if not mcap_files:
        print(f"No .mcap.zstd file found in {bag_dir}")
        return
    
    mcap_file = mcap_files[0]
    
    # 创建输出目录
    data_dir = output_dir / "data" / "chunk-000"
    videos_dir = output_dir / "videos"
    meta_dir = output_dir / "meta"
    
    data_dir.mkdir(parents=True, exist_ok=True)
    videos_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("ROS2 Bag to LeRobot Dataset Converter")
    print("="*60)
    print(f"Input: {mcap_file.name}")
    print(f"Output: {output_dir}")
    
    # 读取数据
    data = read_mcap_zstd(mcap_file)
    
    print("\n📊 Topic message counts:")
    for key, value in data.items():
        print(f"  {key}: {len(value['data'])} messages")
    
    # 构建episode
    print("\n🔄 Building episode dataframe...")
    df, video_frames, stats = build_episode_dataframe(data, ep_idx=0, fps=30)
    
    if df is None:
        print("No valid frames found!")
        return
    
    # 保存parquet
    parquet_path = data_dir / "file-000.parquet"
    df.to_parquet(parquet_path, index=False)
    print(f"\n💾 Saved parquet: {parquet_path}")
    print(f"   Frames: {len(df)}")
    print(f"   Size: {parquet_path.stat().st_size / (1024 * 1024):.2f} MB")
    
    # 保存视频
    print("\n🎬 Saving videos...")
    for cam_name, frames in video_frames.items():
        if frames:
            video_path = videos_dir / f"observation.images.{cam_name}" / "chunk-000" / "file-000.mp4"
            save_video(frames, video_path, fps=30)
    
    # 保存stats.json
    print("\n📊 Saving stats.json...")
    with open(meta_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    
    # 保存info.json
    print("📋 Saving info.json...")
    
    # 相机分辨率配置
    camera_shapes = {
        'head_front': [3, 720, 1280],
        'chest_front': [3, 480, 640],
        'camera1': [3, 480, 640],
        'camera3': [3, 480, 640]
    }
    
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": [14],
            "names": [
                "left_arm_joint1", "left_arm_joint2", "left_arm_joint3",
                "left_arm_joint4", "left_arm_joint5", "left_arm_joint6",
                "right_arm_joint1", "right_arm_joint2", "right_arm_joint3",
                "right_arm_joint4", "right_arm_joint5", "right_arm_joint6",
                "left_arm_gripper", "right_arm_gripper"
            ]
        },
        "action": {
            "dtype": "float32",
            "shape": [14],
            "names": [
                "left_arm_joint1_cmd", "left_arm_joint2_cmd", "left_arm_joint3_cmd",
                "left_arm_joint4_cmd", "left_arm_joint5_cmd", "left_arm_joint6_cmd",
                "right_arm_joint1_cmd", "right_arm_joint2_cmd", "right_arm_joint3_cmd",
                "right_arm_joint4_cmd", "right_arm_joint5_cmd", "right_arm_joint6_cmd",
                "left_arm_gripper_cmd", "right_arm_gripper_cmd"
            ]
        }
    }
    
    # 添加图像特征
    for cam_name in video_frames.keys():
        features[f"observation.images.{cam_name}"] = {
            "dtype": "video",
            "shape": camera_shapes.get(cam_name, [3, 480, 640]),
            "names": ["channels", "height", "width"]
        }
    
    # 添加时间戳和索引特征
    for feat in ['timestamp', 'frame_index', 'episode_index', 'index', 'task_index']:
        features[feat] = {
            "dtype": "float32" if feat == 'timestamp' else "int64",
            "shape": [1],
            "names": None
        }
    
    info = {
        "codebase_version": "v3.0",
        "fps": 30,
        "features": features,
        "total_episodes": 1,
        "total_frames": len(df),
        "total_tasks": 1,
        "chunks_size": 1000,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "robot_type": "quanta_x1_raw_joints",
        "splits": {"train": "0:1"}
    }
    
    with open(meta_dir / "info.json", "w") as f:
        json.dump(info, f, indent=2)
    
    # 保存tasks.parquet
    print("📋 Saving tasks.parquet...")
    tasks_df = pd.DataFrame([{
        "task_index": 0,
        "task": "pull door"
    }])
    tasks_df.to_parquet(meta_dir / "tasks.parquet", index=False)
    
    # 保存episodes.parquet
    print("📋 Saving episodes.parquet...")
    episodes_df = pd.DataFrame([{
        "episode_index": 0,
        "task_index": 0,
        "length": len(df)
    }])
    episodes_df.to_parquet(meta_dir / "episodes.parquet", index=False)
    
    # 计算并显示统计信息
    print("\n" + "="*60)
    print("✅ CONVERSION COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"📁 Output directory: {output_dir}")
    print(f"📊 Total frames: {len(df)}")
    print(f"🎬 Videos saved: {len(video_frames)} cameras")
    for cam_name, frames in video_frames.items():
        print(f"   - {cam_name}: {len(frames)} frames")
    print(f"\n📈 State statistics:")
    print(f"   Min: {np.min(stats['observation.state']['min']):.4f}")
    print(f"   Max: {np.max(stats['observation.state']['max']):.4f}")
    print(f"   Mean: {np.mean(stats['observation.state']['mean']):.4f}")
    print(f"\n📁 Output structure:")
    print(f"   {output_dir}/")
    print(f"   ├── data/chunk-000/file-000.parquet")
    print(f"   ├── videos/observation.images.*/chunk-000/file-000.mp4")
    print(f"   └── meta/")
    print(f"       ├── info.json")
    print(f"       ├── stats.json")
    print(f"       ├── episodes.parquet")
    print(f"       └── tasks.parquet")

if __name__ == "__main__":
    import numpy as np
    import pandas as pd
    main()