# 远征 A2 正式数采流程（Spark）

更新日期：2026-07-31

本目录只负责 Spark 端。Orin 相机配置保持厂家稳定配置，不再修改：

- 头部 H.265：约 30 Hz；
- 左手 H.265：约 30 Hz；
- 右手 H.265：约 30 Hz；
- 运动数据按 ROS 源频率采集。

正式策略是“原始数据完整保存，训练数据离线生成”：

```text
Orin 原生 ROS 2 数据
        ↓ 有线网络，长期订阅
Spark 原始 rosbag（视频约30 Hz，运动数据原频）
        ↓ episode 结束后离线处理
三路 H.265 MP4 15 fps + signals_50hz.json
```

禁止在 ROS H.265 压缩包层面每隔一包丢一包。HEVC 帧存在参考依赖，直接丢包可能
造成花屏、不可解码或时间轴错误。30→15 fps 必须完整解码后重新编码。

## 1. 正式 topic

三路相机：

```text
/aima/hal/rgbd_camera/head_front/color/h265
/aima/hal/rgbd_camera/hand_left/color/h265
/aima/hal/rgbd_camera/hand_right/color/h265
```

运动数据：

```text
/motion/control/arm_joint_state
/motion/control/hand_joint_state
/motion/control/neck_joint_state
/motion/control/arm_joint_command
/motion/control/hand_joint_command
/motion/control/neck_joint_command
/tf
/tf_static
```

腰部数据本阶段暂不加入。

## 2. 文件说明

- `a2_ros2_live_episode_collector.py`
  - Spark 上长期订阅；
  - 三路相机按不低于 28 Hz 验收；
  - 必需运动 topic 按不低于 47.5 Hz 验收；
  - 回车开始/结束 episode，不重建 subscriber；
  - 从三路相机最近的可解码 H.265 关键帧开始写入；
  - 原始 H.265 包及运动消息全部写入 rosbag。
- `convert_h265_episode_to_15fps.py`
  - 将解出的约 30 Hz H.265 完整解码并重新编码为 15 fps MP4；
  - 如果输入本身已约 15 Hz，才允许无损 remux；
  - 使用 ffmpeg 完整解码验证和 ffprobe 验收。
- `resample_a2_motion_to_50hz.py`
  - state 线性插值到 50 Hz；
  - command 使用零阶保持到 50 Hz；
  - `/tf` 限频到最高 50 Hz，但不拆分单条 TFMessage；
  - `/tf_static` 保存一份；
  - 任一必需运动流为空时拒绝生成正式结果。
- `process_a2_episode.sh`
  - 一键完成视频提取、15 fps 转码、运动数据 50 Hz 重采样和校验和生成。

`a2_grpc_*` 和 `a2_orin_grpc_bridge.py` 是早期实验文件，不是正式主链路。
A2 官方高频数据接口使用 ROS 2/FastDDS，正式采集不依赖自建 gRPC Bridge。

## 3. Spark运行环境

当前脚本需要 ROS 2 Python 包：

```bash
source /opt/ros/jazzy/setup.bash
```

不要在 Conda `base` 中安装 ROS 依赖。已有 `/home/yichu/A2/.venv` 主要用于早期
gRPC 实验，不负责提供 `rclpy`、`rosbag2_py` 和 ROS message 包。

正式采集使用 ROS 2 Humble/FastDDS 独立容器，避免修改 Spark 已有 Jazzy、Conda
和其他机器人数采环境。宿主机 Jazzy 直连测试只能短暂收到部分相机数据，已经停止
作为采集入口；宿主机仍可用于离线处理。

## 3.1 构建隔离Humble采集环境

以下操作只新增一个Docker镜像，不安装或覆盖宿主机ROS包：

```bash
cd /home/yichu/A2

chmod +x \
  build_spark_humble_container.sh \
  run_spark_humble_monitor.sh \
  run_spark_humble_collector.sh

./build_spark_humble_container.sh
```

构建后先进行30秒监测：

```bash
./run_spark_humble_monitor.sh 30
```

只有三路相机、必需state/command、TF全部持续收到且 `overall_ready=True`，才能执行：

```bash
./run_spark_humble_collector.sh
```

容器使用 `--network host` 和Spark专用FastDDS XML；数据通过目录挂载直接写入宿主机
`/home/yichu/A2/data/raw`。容器退出不会删除已经写入的episode。

## 4. 开始采集前检查

Spark 与 Orin 有线地址：

```text
Spark：192.168.2.94
Orin： 192.168.2.50
```

```bash
ip route get 192.168.2.50
ping -c 5 192.168.2.50
```

应走 Spark 有线接口，延迟应约为亚毫秒且无丢包。

ROS 环境：

```bash
conda deactivate 2>/dev/null || true
source /opt/ros/jazzy/setup.bash

export ROS_DOMAIN_ID=232
unset ROS_LOCALHOST_ONLY
export ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export FASTRTPS_DEFAULT_PROFILES_FILE=/home/yichu/A2/config/spark_ros_dds_configuration.xml
```

Spark必须使用针对自身有线地址生成的FastDDS配置，不能原样使用Orin配置。先从
Orin复制当前生效的XML，再只替换本地接口白名单：

```bash
mkdir -p /home/yichu/A2/config

scp \
  agi@192.168.2.50:/agibot/software/v0/entry/bin/cfg/ros_dds_configuration.xml \
  /home/yichu/A2/config/orin_ros_dds_configuration.xml

/usr/bin/python3 \
  /home/yichu/A2/make_spark_fastdds_profile.py \
  /home/yichu/A2/config/orin_ros_dds_configuration.xml \
  /home/yichu/A2/config/spark_ros_dds_configuration.xml \
  --local-address 192.168.2.94 \
  --overwrite
```

生成后，Spark配置的 `interfaceWhiteList` 必须只包含Spark有线地址
`192.168.2.94`。这个操作不会修改Orin。

首次运行或脚本更新后执行：

```bash
chmod +x \
  /home/yichu/A2/verify_spark_pipeline.sh \
  /home/yichu/A2/process_a2_episode.sh

/home/yichu/A2/verify_spark_pipeline.sh
```

仅发现 topic 名称不代表能收到数据。正式监测使用Humble容器：

```bash
/home/yichu/A2/run_spark_humble_monitor.sh 30
```

三路相机应接近 30 Hz。state、command 和 `/tf` 必须在实际遥操工作状态下检查；
未遥操时 command 可能没有数据，因此不能用静止状态判定正式采集链路失败。

## 5. 正式采集

```bash
/home/yichu/A2/run_spark_humble_collector.sh
```

操作方式：

1. 程序启动后持续订阅，但尚未写 episode；
2. 等全部流显示 `READY`；
3. 遥操准备完成后按回车，开始一个 episode；
4. 完成动作后再次回车；
5. 程序多录1秒 post-roll，然后关闭该 episode；
6. subscriber 保持运行，可继续录下一条；
7. 输入 `q` 退出。

每个原始 episode 包含：

```text
/home/yichu/A2/data/raw/episode_YYYYMMDD_HHMMSS/
├── episode_YYYYMMDD_HHMMSS_0.db3
├── metadata.yaml
└── episode_control.json
```

原始 rosbag 是唯一的原始真值，处理完成后也不要删除。

## 6. 生成15 fps视频和50 Hz运动数据

结束采集程序后处理一条 episode：

```bash
chmod +x /home/yichu/A2/process_a2_episode.sh

/home/yichu/A2/process_a2_episode.sh \
  /home/yichu/A2/data/raw/episode_YYYYMMDD_HHMMSS
```

默认输出到：

```text
/home/yichu/A2/processed/episode_YYYYMMDD_HHMMSS/
├── head_camera.h265
├── left_hand_camera.h265
├── right_hand_camera.h265
├── head_camera.mp4
├── left_hand_camera.mp4
├── right_hand_camera.mp4
├── *_timestamps.csv
├── episode.json
├── signals_50hz.json
└── SHA256SUMS
```

其中 `.h265` 是从原始 rosbag 提取的30 Hz码流，MP4是训练使用的15 fps结果。

## 7. 验收

校验文件：

```bash
cd /home/yichu/A2/processed/episode_YYYYMMDD_HHMMSS
sha256sum -c SHA256SUMS
```

检查视频：

```bash
for video in head_camera.mp4 left_hand_camera.mp4 right_hand_camera.mp4
do
  echo "===== $video ====="
  ffprobe -v error -count_frames -select_streams v:0 \
    -show_entries stream=codec_name,width,height,avg_frame_rate,nb_read_frames \
    -of default=noprint_wrappers=1 "$video"

  ffmpeg -v error -i "$video" -f null -
done
```

正式结果必须满足：

- 三路 MP4 均为 HEVC/H.265；
- 三路 MP4 均可完整解码；
- `avg_frame_rate` 为15 fps附近；
- `signals_50hz.json` 成功生成；
- 六路必需 state/command 均有数据；
- 原始 rosbag、处理结果和校验和均保留。

如果 `neck_joint_command` 等必需 command 在 episode 中为0条，处理程序会拒绝生成
正式50 Hz结果。这类 episode 可用于观察数据调试，但不能作为包含action的正式训练数据。
