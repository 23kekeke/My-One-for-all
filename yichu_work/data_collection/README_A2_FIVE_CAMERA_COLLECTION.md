# A2 五路 H.265 遥操数据采集

本文档用于远征 A2 的五路相机、机器人状态和真实遥操 command 采集。

当前五路视频均已验证可持续发布约 30 Hz：

```text
/aima/hal/rgbd_camera/head_front/color/h265
/aima/hal/rgbd_camera/hand_left/color/h265
/aima/hal/rgbd_camera/hand_right/color/h265
/aima/hal/fish_eye_camera/chest_left/color/h265
/aima/hal/fish_eye_camera/chest_right/color/h265
```

## 重要原则

1. 五路相机和核心状态是必需数据，禁止通过单次 `ros2 topic list` 动态过滤。
2. 必须显式将所有必需 topic 传给 `ros2 bag record`。
3. 遥操动作只能在所有 topic 完成订阅且编码器预热后开始。
4. 每个 episode 开始和结束时，机器人都应保留静止段。
5. 原始 rosbag 永久保留；MP4 和 JSON 在 Spark 离线生成。
6. 包含 command 的 rosbag 禁止在真实机器人上执行 `ros2 bag play`。

## 1. Orin 环境

以下命令均在 `agi@ubuntu-orin` 上执行：

```bash
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=232
export ROS_LOCALHOST_ONLY=0
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export FASTRTPS_DEFAULT_PROFILES_FILE=/agibot/software/v0/entry/bin/cfg/ros_dds_configuration.xml
```

## 2. 采集前进程检查

```bash
pgrep -af '[h]al_d415|[h]al_d405|[t]zcamera'
```

应存在：

```text
hal_d415    头部相机
hal_d405    左右手相机
tzcamera    左右胸相机
```

如果 `hal_d405` 尚未接入官方常驻服务，需要保持其前台启动终端运行：

```bash
/agibot/software/v0/scripts/hal_d405/start_hal_d405.sh
```

不要重复启动第二个 `hal_d405` 实例。

## 3. 五路视频预检

```bash
for topic in \
  /aima/hal/rgbd_camera/head_front/color/h265 \
  /aima/hal/rgbd_camera/hand_left/color/h265 \
  /aima/hal/rgbd_camera/hand_right/color/h265 \
  /aima/hal/fish_eye_camera/chest_left/color/h265 \
  /aima/hal/fish_eye_camera/chest_right/color/h265
do
  echo
  echo "===== $topic ====="
  timeout --foreground -s INT 6s ros2 topic hz "$topic" || true
done
```

要求：

- 五路均持续收到消息；
- 平均帧率最好为 28～31 Hz；
- 任何一路无消息时禁止开始正式 episode。

`ros2 topic info` 偶尔可能因 DDS discovery 显示 `Unknown topic`；以实际 `topic hz` 收到连续消息为准。

## 4. 录制内容

正式 Orin rosbag 当前包含：

### 五路视频

```text
/aima/hal/rgbd_camera/head_front/color/h265
/aima/hal/rgbd_camera/hand_left/color/h265
/aima/hal/rgbd_camera/hand_right/color/h265
/aima/hal/fish_eye_camera/chest_left/color/h265
/aima/hal/fish_eye_camera/chest_right/color/h265
```

### Observation

```text
/motion/control/arm_joint_state
/motion/control/hand_joint_state
/motion/control/neck_joint_state
/motion_control/hand_pose_state
/body_drive/imu/data
/tf
/tf_static
```

### Action 候选

```text
/motion/control/arm_joint_command
/motion/control/hand_joint_command
```

遥操测试中已测得：

```text
arm_joint_command   约 500 Hz
hand_joint_command  约 500 Hz
arm/hand/neck state 约 100 Hz
hand_pose_state     约 30 Hz
IMU                 约 1000 Hz
```

`neck_joint_command` 目前没有消息，暂不加入必需 topic。X86 上的腰、腿、底盘、里程计等数据待完成接口盘点后追加。

## 5. 创建 episode 路径

```bash
record_root="/agibot/data/a2_five_h265_teleop"
episode_name="episode_$(date +%Y%m%d_%H%M%S)"
bag_path="$record_root/$episode_name"

mkdir -p "$record_root"

echo "本次 episode：$bag_path"
```

必须确认 `bag_path` 非空且位于指定目录：

```bash
case "$bag_path" in
  /agibot/data/a2_five_h265_teleop/episode_*)
    ;;
  *)
    echo "错误：bag_path 不安全：$bag_path"
    exit 1
    ;;
esac
```

## 6. 推荐录制方式：启动时暂停

先检查当前 Humble 是否支持：

```bash
ros2 bag record --help | grep start-paused
```

如果支持，执行：

```bash
ros2 bag record \
  --start-paused \
  -s sqlite3 \
  --max-cache-size 1073741824 \
  -o "$bag_path" \
  /aima/hal/rgbd_camera/head_front/color/h265 \
  /aima/hal/rgbd_camera/hand_left/color/h265 \
  /aima/hal/rgbd_camera/hand_right/color/h265 \
  /aima/hal/fish_eye_camera/chest_left/color/h265 \
  /aima/hal/fish_eye_camera/chest_right/color/h265 \
  /motion/control/arm_joint_state \
  /motion/control/hand_joint_state \
  /motion/control/neck_joint_state \
  /motion/control/arm_joint_command \
  /motion/control/hand_joint_command \
  /motion_control/hand_pose_state \
  /body_drive/imu/data \
  /tf \
  /tf_static
```

严格按照以下顺序操作：

1. 机器人保持静止。
2. 检查日志中十四个 topic 均出现 `Subscribed to topic`。
3. 等待 `All requested topics are subscribed`。
4. 再静止等待 3 秒。
5. 按一次空格恢复写入。
6. 继续静止 2～3 秒，等待五路 H.265 关键帧。
7. 通知同事开始遥操。
8. 动作结束后保持静止 2 秒。
9. 按一次空格暂停写入。
10. 按 `Ctrl+C`。
11. 等待 `Writing remaining messages from cache` 和 `Recording stopped`。

不要在出现 `All requested topics are subscribed` 前开始动作。

## 7. 如果不支持 `--start-paused`

去掉 `--start-paused`，其余 topic 保持完全相同。

录制器启动后：

1. 机器人保持静止；
2. 等待全部十四个 topic 完成订阅；
3. 再等待 5 秒；
4. 开始遥操；
5. 动作结束后静止 2 秒；
6. 按 `Ctrl+C`；
7. Spark 离线裁掉开头预热段。

## 8. Orin 端验收

```bash
test -f "$bag_path/metadata.yaml" || {
  echo "错误：rosbag 不完整"
  exit 1
}

du -sh "$bag_path"
ros2 bag info "$bag_path"
```

必须在 `Topic information` 中看到全部五路 H.265。任何一路完全缺失或 Count 为 0，该 episode 判为无效。

建议验收指标：

```text
五路 H.265             平均 28～31 Hz
arm/hand/neck state    不低于 90 Hz
arm/hand command       接近 500 Hz
hand pose              接近 30 Hz
IMU                    接近 1000 Hz
```

生成校验文件：

```bash
cd "$bag_path"
sha256sum ./*.db3 metadata.yaml | tee SHA256SUMS
```

## 9. 通过专用网线传到 Spark

Spark 有线地址：

```text
192.168.2.94
```

在 Orin 先验证：

```bash
ip route get 192.168.2.94
ping -c 5 192.168.2.94
```

创建 Spark 目录：

```bash
ssh yichu@192.168.2.94 \
  "mkdir -p /home/yichu/yichu_work/datasets/a2/raw_five_h265/$episode_name"
```

传输：

```bash
rsync -av \
  --partial \
  --append-verify \
  --info=progress2 \
  "$bag_path/" \
  "yichu@192.168.2.94:/home/yichu/yichu_work/datasets/a2/raw_five_h265/$episode_name/"
```

传输前再次检查，防止空变量将 `/` 作为源目录：

```bash
test -n "$bag_path" \
  && test -d "$bag_path" \
  && case "$bag_path" in
       /agibot/data/a2_five_h265_teleop/episode_*) true ;;
       *) false ;;
     esac \
  || {
    echo "错误：bag_path 为空或不安全，停止传输"
    exit 1
  }
```

Spark 校验：

```bash
cd "/home/yichu/yichu_work/datasets/a2/raw_five_h265/$episode_name"
sha256sum -c SHA256SUMS
ros2 bag info .
```

## 10. Spark 离线处理原则

五路相机不能按相同帧号直接对齐。必须使用每路消息时间戳：

```text
共同开始时间 = 五路首个可解码关键帧时间的最大值
共同结束时间 = 五路最后帧时间的最小值
```

处理流程：

1. 每路跳过首个可解码关键帧之前的数据；
2. 取五路共同有效时间区间；
3. 建立统一 30 Hz 时间轴；
4. 视频按时间戳选择最近帧；
5. state 按时间戳插值；
6. command 按时间戳选择最近目标或保持最近有效目标；
7. 生成五路 MP4、时间戳 CSV 和 `episode.json`；
8. 保留原始 H.265 与 rosbag，便于重新转换。

## 11. 安全警告

包含以下 topic 的 bag 可能包含真实控制指令：

```text
/motion/control/arm_joint_command
/motion/control/hand_joint_command
```

禁止在连接真实机器人的 ROS domain 中运行：

```bash
ros2 bag play <bag>
```

如需分析，只能通过 `rosbag2_py`/SQLite 离线读取，或在完全隔离且没有机器人控制节点的环境中处理。
