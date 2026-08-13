# A2 Spark gRPC Episode Collector

本目录中的代码只运行在 Spark：

```text
/home/yichu/A2
```

它不会 SSH 到 Orin、不会修改 Orin 文件，也不会发布任何机器人控制 topic。

## 1. 文件说明

```text
/home/yichu/A2/
├── a2_grpc_episode_collector_no_waist.py
├── proto/a2_data.proto
├── config/spark_collector_no_waist.json
├── compile_proto.sh
├── requirements-spark.txt
├── A2_GRPC_ACQUISITION_PLAN_15HZ_50HZ.md
└── README_SPARK_GRPC.md
```

- `proto/a2_data.proto`：Spark 与 Orin bridge 必须共同遵守的数据协议；
- `config/spark_collector_no_waist.json`：三路相机和运动 stream 配置（腰部暂不纳入）；
- `a2_grpc_episode_collector_no_waist.py`：Spark 长连接、缓存、episode 和落盘（腰部暂不纳入）；
- `compile_proto.sh`：在 Spark 独立虚拟环境中生成 Python gRPC 文件。

## 2. Spark独立环境

不要在 Conda `base` 中安装，也不要修改已有机器人项目环境。

```bash
cd /home/yichu/A2

conda deactivate 2>/dev/null || true

/usr/bin/python3 -m venv /home/yichu/A2/.venv

/home/yichu/A2/.venv/bin/python -m pip install --upgrade pip
/home/yichu/A2/.venv/bin/python -m pip install \
  -r /home/yichu/A2/requirements-spark.txt
```

安装范围只在 `/home/yichu/A2/.venv`。

## 3. 生成protobuf文件

```bash
cd /home/yichu/A2
chmod +x /home/yichu/A2/compile_proto.sh
/home/yichu/A2/compile_proto.sh
```

生成：

```text
/home/yichu/A2/generated/a2_data_pb2.py
/home/yichu/A2/generated/a2_data_pb2_grpc.py
```

语法检查：

```bash
/home/yichu/A2/.venv/bin/python -m py_compile \
  /home/yichu/A2/a2_grpc_episode_collector_no_waist.py \
  /home/yichu/A2/generated/a2_data_pb2.py \
  /home/yichu/A2/generated/a2_data_pb2_grpc.py
```

## 4. Orin bridge必须提供的stream_name

这里的 `stream_name` 是 gRPC 协议内部稳定名称，不是 ROS topic 全名。

```text
head_camera
left_hand_camera
right_hand_camera
arm_joint_state
hand_joint_state
neck_joint_state
waist_state
arm_joint_command
hand_joint_command
neck_joint_command
tf
tf_static
waist_move_value_raw
```

映射关系已经写在：

```text
/home/yichu/A2/config/spark_collector_no_waist.json
```

Orin bridge 必须使用完全相同的名称，否则 Spark 会忽略消息。

每条 `StreamEnvelope` 必须同时填写两个时间：

- `source_timestamp_ns`：原始 ROS 消息时间戳；
- `sample_timestamp_ns`：bridge 输出的15 Hz或50 Hz采样网格时间戳。

相机通常令两者相同。command 使用零阶保持时，同一条原始 command 可以对应多个
50 Hz采样点，因此 `source_timestamp_ns` 可以重复，但 `sample_timestamp_ns`
必须按20 ms严格递增。Spark 使用 sample timestamp 计算频率和对齐，同时保留
source timestamp 用于追溯。

## 5. 在Orin bridge完成前可做的检查

确认配置：

```bash
/home/yichu/A2/.venv/bin/python -m json.tool \
  /home/yichu/A2/config/spark_collector_no_waist.json >/dev/null
```

确认 FFmpeg：

```bash
/usr/bin/ffmpeg -version | head -n 1
```

查看 collector 帮助：

```bash
/home/yichu/A2/.venv/bin/python \
  /home/yichu/A2/a2_grpc_episode_collector_no_waist.py \
  --help
```

## 6. Orin bridge完成后的第一步：只监测

默认预期 Orin gRPC 地址：

```text
192.168.2.50:50061
```

先确认网络和端口：

```bash
ping -c 5 192.168.2.50
nc -vz 192.168.2.50 50061
```

只监测 20 秒，不创建 episode：

```bash
/home/yichu/A2/.venv/bin/python \
  /home/yichu/A2/a2_grpc_episode_collector_no_waist.py \
  --config /home/yichu/A2/config/spark_collector_no_waist.json \
  --monitor-seconds 20
```

只有输出：

```text
"overall_ready": true
```

才进入正式录制。

如果三路 H.265 实际仍为 30 Hz，监测会判定不满足 15 Hz，不允许开始 episode。
collector 不会通过丢弃 H.265 帧伪造 15 Hz。

## 7. 正式录制

```bash
/home/yichu/A2/.venv/bin/python \
  /home/yichu/A2/a2_grpc_episode_collector_no_waist.py \
  --config /home/yichu/A2/config/spark_collector_no_waist.json
```

交互：

```text
WAIT  → 等待全部流连续稳定
READY → 可以按回车
第一次回车 → 开始 episode
第二次回车 → 停止，追加 post-roll 并落盘
空闲时输入 q → 退出
```

某个必需 stream 不满足频率、没有 H.265 参数集/IRAP、没有 command 或没有
`tf_static` 时，程序不会开始正式 episode。

## 8. 输出

有效 episode：

```text
/home/yichu/A2/data/episode_YYYYMMDD_HHMMSS/
```

无效 episode：

```text
/home/yichu/A2/data/episode_YYYYMMDD_HHMMSS_INVALID/
```

文件：

```text
head_camera.mp4
left_hand_camera.mp4
right_hand_camera.mp4
episode.json
signals_50hz.json
tf_static.json
manifest.json
quality_report.json
SHA256SUMS
```

如果 MP4 封装或完整解码失败，对应 `.h265` 原始文件会保留，episode 标记为
`INVALID`，便于排查。

## 9. 当前实现的时间逻辑

- 三路 H.265 分别保持最近 5 秒缓存；
- 开始时从各自最近的 VPS/SPS/PPS + IRAP 写入；
- 头部相机源时间戳是 15 Hz 训练主时间轴；
- 左右手相机按最近时间戳对齐；
- state 从 50 Hz 线性插值到 15 Hz；
- command 使用目标时刻之前最近值，零阶保持；
- 完整 50 Hz 数据另外保存在 `signals_50hz.json`；
- `tf_static` 每个 episode 保存一份。

## 10. 安全限制

- collector 只实现 gRPC client；
- proto 中没有机器人控制 RPC；
- collector 不依赖 ROS 2，也不会发布 ROS topic；
- 不要把采集目录里的 command 数据回放到真实机器人；
- `/motion/control/hand_joint_state` 的20维 position和260维 effort只原样保存，
  在厂商提供语义前不用于正式训练映射。
