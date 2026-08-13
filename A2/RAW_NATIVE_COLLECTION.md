# A2 原频采集：三路 H.265 + 原频运动数据

本流程不在采集阶段修改频率：

- 三路相机保存机器人发布的全部 H.265 包，通常约 30 Hz；
- arm/hand/neck state 保存全部消息，通常约 100 Hz；
- command/action 按发布端实际频率保存，遥操时可能高于 100 Hz；
- `/tf` 按实际频率保存；
- `/tf_static` 使用 transient-local QoS 并保存一份；
- 不在线解码、不转 JPEG、不抽帧、不插值、不重采样。

Spark 最终生成的原始 episode 是 rosbag2/sqlite3。运动 Relay 的输入 topic
虽然位于 `/a2/relay/...`，写入 rosbag 时会恢复为标准原 topic 名，因此现有
检查和离线处理脚本不需要认识 Relay topic。

## 1. 将 Relay 文件从 Spark 复制到 Orin

在 Spark 执行：

```bash
ssh agi@192.168.2.50 'mkdir -p /agibot/data/a2_ros_relay'

rsync -av \
  /home/yichu/A2/a2_orin_ros2_motion_relay.py \
  /home/yichu/A2/run_orin_ros2_motion_relay.sh \
  agi@192.168.2.50:/agibot/data/a2_ros_relay/
```

## 2. 首次在 Orin 前台验证 Relay

在 Orin 执行：

```bash
cd /agibot/data/a2_ros_relay
chmod +x run_orin_ros2_motion_relay.sh

/usr/bin/python3 -m py_compile a2_orin_ros2_motion_relay.py
./run_orin_ros2_motion_relay.sh
```

保持该终端运行。每两秒应输出各路转发频率。正常情况下：

- 三路 state 约 100 Hz；
- arm/hand command 在遥操工作时有数据；
- neck command 是否有数据取决于遥操系统是否发布头颈目标；
- `tf_static` 通常只在启动或订阅建立时收到一次，之后显示 0 Hz 是正常的。

Relay 只发布 `/a2/relay/...`，不会向 `/motion/control/...` 写入数据。

## 3. Spark 先做 30 秒监测

在 Spark 执行：

```bash
cd /home/yichu/A2

chmod +x \
  run_spark_humble_monitor.sh \
  run_spark_humble_collector.sh

./run_spark_humble_monitor.sh 30
```

验收重点：

- 三路 H.265 接近 30 Hz，且 `keyframe=True`；
- 三路 relay state 不低于 90 Hz；
- arm/hand command 在遥操工作时不低于 47.5 Hz；
- relay `/tf` 持续有数据；
- relay `/tf_static` 已收到；
- 最终显示 `overall_ready=True`。

`neck_joint_command` 当前只监测、不作为启动硬门槛，因为机器人可能不发布
该动作流。它仍会被创建并在有消息时完整写入 rosbag。

## 4. Spark 开始采集

```bash
cd /home/yichu/A2
./run_spark_humble_collector.sh
```

采集器会一直保持 subscriber，不会在每个 episode 开始时重新订阅：

1. 等待所有必需流 READY；
2. 第一次按回车开始 episode；
3. 第二次按回车结束 episode；
4. 继续等待下一次回车，可连续采集多个 episode；
5. 输入 `q` 并回车退出。

数据保存到：

```text
/home/yichu/A2/data/raw/episode_YYYYMMDD_HHMMSS/
├── metadata.yaml
├── *.db3
└── episode_control.json
```

相机从最近一个可独立解码的 H.265 关键帧开始写入。运动消息使用同一时间
窗口的预录数据，因此 episode 开始处不会因为相机等待 IDR 而缺少运动数据。

## 5. 原始数据验收

```bash
episode="/home/yichu/A2/data/raw/episode_YYYYMMDD_HHMMSS"

du -sh "$episode"
ros2 bag info "$episode"
```

按 bag 的实际 Duration 粗略验收：

```text
每路相机 count ≈ Duration × 30
每路 state count ≈ Duration × 100
command count = 发布端实际频率 × Duration
/tf count = 发布端实际频率 × Duration
/tf_static count >= 1
```

由于 episode 包含关键帧预录和停止后的短 post-roll，bag Duration 可能比两次
回车之间的人工计时略长，这是预期行为。

## 6. 需要 30 Hz MP4 时

原始 rosbag 始终保留。先用现有提取器导出三路 `.h265` 和时间戳，再调用：

```bash
/usr/bin/python3 \
  /home/yichu/A2/convert_h265_episode_to_15fps.py \
  OUTPUT_DIR \
  --output-fps 30
```

输入接近 30 Hz 时该工具采用无损 remux，不抽帧、不重新编码。文件名虽然沿用
早期的 `to_15fps`，实际输出频率由 `--output-fps` 参数决定。

