# 【已废弃】A2 仿北大 gRPC 数据获取架构

> 本文件是早期设计记录，不再作为执行方案。A2 正式主链路已确定为 ROS 2/FastDDS：
> Orin 三路 H.265 保持约30 Hz，Spark完整保存原始 rosbag，episode结束后转15 fps，
> 运动数据离线重采样到50 Hz。请以 `/home/yichu/A2/README.md` 为唯一执行入口。

# 原设计：三路 H.265 / 15 Hz / 运动数据 50 Hz

更新日期：2026-07-30

## 1. 正式需求基线

### 1.1 三路 H.265 相机

只采集以下三路相机，不采集左右胸部相机，也不传输 raw `/color`：

1. `/aima/hal/rgbd_camera/head_front/color/h265`
2. `/aima/hal/rgbd_camera/hand_left/color/h265`
3. `/aima/hal/rgbd_camera/hand_right/color/h265`

目标：三路都稳定为 15 Hz，并保持 H.265 编码。

### 1.2 运动与坐标数据

目标：除静态 TF 外，以下运动数据稳定保存或统一重采样为 50 Hz：

- `/motion/control/arm_joint_state`
- `/motion/control/hand_joint_state`
- `/motion/control/neck_joint_state`
- `/motion/control/waist_state`
- `/motion/control/arm_joint_command`
- `/motion/control/hand_joint_command`
- `/motion/control/neck_joint_command`
- `/tf`
- `/tf_static`
- `/motion/control/waist_state/pb_3Aaimdk_2Eprotocol_2EWaistMoveValue`

`/motion/control/hand_joint_state` 在原始清单中重复了一次，实际只订阅一次。

`/tf_static` 是 transient-local 静态数据，不要求 50 Hz。采集进程启动时读取并
缓存，每个 episode 写入一份。

### 1.3 手部位置待定

已知 `/motion/control/hand_joint_state`：

- `position` 为 20 维；
- `effort` 为 260 维；
- `name` 为空。

在厂商提供字段定义前：

- 完整原样保存；
- 记录每条消息的数组长度；
- 不自行命名 20 个位置字段；
- 不把 260 维 effort 当成标准关节力矩；
- 暂不映射为最终 VLA state；
- 在 manifest 中标记 `semantic_status: pending`。

## 2. 仿北大但适配 A2 的总体结构

北大使用机器人 SDK gRPC Server，Spark 通过 gRPC stream 获取相机和运动数据，
再按 episode 保存 MP4 和 JSON。

A2 采用相同的系统边界：

```text
A2 Orin ROS 2
        ↓
a2_edge_bridge（机器人侧 gRPC Server）
        ↓ 有线网络，长期 gRPC streaming
Spark /home/yichu/A2
        ↓
a2_grpc_episode_collector
        ↓
回车开始 / 回车停止
        ↓
三路 H.265 MP4 + 15 Hz episode.json + 50 Hz signals_50hz.json
```

A2 与北大的关键差别：

- 北大传输可独立解码的压缩图片帧；
- A2 直接传输 H.265 Annex-B access unit；
- 北大在 Spark 解码并重编码 H.264；
- A2 优先不解码、不重编码，直接将 H.265 封装为 MP4；
- 北大 action 多数由 state 派生；
- A2 action 必须使用真实 command；
- A2 episode 之间保持 ROS 和 gRPC 长连接，不重新订阅。

## 3. 三路 H.265 如何达到 15 Hz

### 3.1 不能简单隔帧丢弃

H.265 包含 VPS、SPS、PPS、IDR 和依赖前序帧的 P/B 帧。如果输入是 30 Hz，
bridge 不能简单每隔一帧丢一帧，否则剩余帧可能依赖已经丢掉的参考帧，产生：

- 马赛克；
- 解码错误；
- MP4 无法从头播放；
- 帧数看似 15 Hz，但数据不可用于训练。

### 3.2 优先方案：机器人编码器直接输出 15 Hz

正式推荐：

```text
相机原始采集
→ A2 硬件编码器直接配置 H.265 15 fps
→ ROS H.265 topic 稳定发布 15 Hz
→ edge bridge 原样转发
→ Spark -c:v copy 封装 MP4
```

必须分别验证三路：

```text
head_front H.265 ≈ 15 Hz
hand_left H.265  ≈ 15 Hz
hand_right H.265 ≈ 15 Hz
```

### 3.3 备选方案：保持 30 Hz，Spark 转码为 15 Hz

如果厂商编码器无法配置为 15 Hz：

```text
机器人保持完整 30 Hz H.265
→ gRPC 原样传到 Spark
→ Spark 完整解码
→ 选择 15 Hz 帧
→ 重新编码 H.265 15 Hz MP4
```

这个方案可以正确生成 15 Hz，但会增加 Spark 的解码和重新编码开销，并产生一次
有损压缩。

### 3.4 不接受的方案

不允许：

```text
30 Hz H.265 → bridge 每隔一条 ROS 消息丢一条 → 假装 15 Hz
```

除非厂商明确保证每条消息都是包含 VPS/SPS/PPS 的独立 IDR 图像；当前数据已经
观察到 IDR 之前存在多条帧，因此不能做这个假设。

## 4. 机器人侧 a2_edge_bridge

### 4.1 职责

- 程序启动时创建全部 ROS subscriber；
- episode 之间保持 subscriber；
- 三路 H.265 原样读取和转发；
- state/command/waist/TF 规范化为 protobuf；
- 保存源时间戳、edge 接收时间戳和连续序号；
- 为每路数据维护独立有界队列；
- 输出实时健康状态和丢包计数；
- 只读，不发布任何机器人控制命令。

### 4.2 H.265 消息

每路相机使用独立 gRPC server-streaming RPC，避免某一路阻塞另外两路或运动数据。

每条消息至少携带：

```text
stream_name
source_timestamp_ns
edge_receive_timestamp_ns
sequence
format=h265
frame_id
Annex-B payload
是否包含 VPS/SPS/PPS
是否包含 IDR/IRAP
```

bridge 不解码 H.265，不修改 payload。

### 4.3 state 50 Hz

原 state 约为 100 Hz。使用消息源时间戳门控到 50 Hz：

- 每 20 ms 选择一条；
- position/velocity/effort/name 原样保存；
- 不使用 `sleep()` 在 callback 中限频；
- 不改变数组顺序；
- 维度变化立即报告错误。

### 4.4 command 50 Hz

原 command 可能超过 400 Hz。使用固定 50 Hz 时间网格：

- 每个采样点保存该时刻之前最近的 command；
- 使用零阶保持，不做线性插值；
- 保留实际 command 的 source timestamp；
- 没有新 command 时可重复上一目标，但必须记录复用来源；
- 整段没有 command 时判定为缺流，不能用 state 补造。

### 4.5 TF

- `/tf`：以 50 Hz 为保存上限，保留每条消息中的完整 transform 集合；
- `/tf_static`：使用 transient-local QoS 获取一次并缓存；
- 不丢失父子 frame 名称和原始时间戳。

### 4.6 waist

验证阶段同时保留：

1. `/motion/control/waist_state`；
2. `/motion/control/waist_state/pb_3Aaimdk_2Eprotocol_2EWaistMoveValue`。

第二路先作为原始协议审计数据保存。等厂商提供 schema 后，再决定它是否进入
正式 observation。

## 5. gRPC 协议建议

```protobuf
message StreamEnvelope {
  string stream_name = 1;
  string ros_type = 2;
  uint64 sequence = 3;
  int64 source_timestamp_ns = 4;
  int64 edge_receive_timestamp_ns = 5;
  int64 sample_timestamp_ns = 6;
  bytes payload = 7;
  string encoding = 8;
  uint32 flags = 9;
}

service A2DataService {
  rpc GetManifest(Empty) returns (StreamManifest);
  rpc WatchHealth(HealthRequest) returns (stream HealthSnapshot);
  rpc Subscribe(StreamRequest) returns (stream StreamEnvelope);
}
```

H.265 的 `flags` 至少区分：

- VPS；
- SPS；
- PPS；
- IDR/IRAP；
- 普通依赖帧。

`source_timestamp_ns` 是原始 ROS 时间；`sample_timestamp_ns` 是 edge bridge
生成的15/50 Hz采样网格时间。尤其对于零阶保持的 command，原始时间可以重复，
但采样时间必须按20 ms递增。

## 6. Spark 长连接与 episode

### 6.1 状态机

```text
CONNECTING
    ↓ 全部 gRPC stream 已建立
WARMING_UP
    ↓ 连续 3 秒满足频率、关键帧和运动数据要求
READY
    ↓ 第一次回车
RECORDING
    ↓ 第二次回车
POST_ROLL
    ↓ 0.5～1 秒
FINALIZING
    ├─ 验收通过 → VALID
    └─ 缺流/损坏 → INVALID
```

回车只改变本地 episode 状态，不能重新创建 ROS subscriber 或 gRPC stream。

### 6.2 环形缓存

- 三路 H.265 各维护 3～5 秒缓存；
- 缓存必须包含最近的 VPS/SPS/PPS 和 IDR；
- state、command、waist、TF 维护相同时长缓存；
- `tf_static` 长期缓存。

### 6.3 episode 开始

第一次回车后：

1. 记录用户触发时间；
2. 为三路相机分别找到触发前最近的可解码关键帧；
3. 从 VPS/SPS/PPS + IDR 开始写入；
4. state/command 从最早视频关键帧之前开始写入；
5. 继续实时接收，不重新订阅。

三路关键帧不一定同时出现，因此每路 MP4 可能包含不同时长的预录。训练使用的
共同有效窗口由三路首个可解码时间中的最晚者确定，并写入 manifest。

### 6.4 episode 停止

第二次回车后：

1. 记录停止触发时间；
2. 保留短 post-roll；
3. 原子关闭三路码流和运动数据；
4. 封装 MP4；
5. 生成 JSON、manifest、质量报告和哈希；
6. gRPC 连接保持运行，等待下一条 episode。

## 7. 时间对齐

### 7.1 15 Hz训练时间轴

以头部 H.265 的 15 Hz 源时间戳作为主时间轴：

- 左手、右手视频：在共同有效窗口内按最近源时间戳对应；
- state：50 Hz 数据线性插值到头部时间戳；
- command：取目标时间之前最近值，零阶保持；
- TF：取目标时间之前最近的完整 TF 状态；
- `tf_static`：episode 级静态数据。

如果源相机仍为 30 Hz、最后转码为 15 Hz，则必须以转码后实际保留帧的源时间戳
建立主时间轴，不能只修改 MP4 的 fps metadata。

### 7.2 50 Hz原始运动数据

领导要求运动数据稳定 50 Hz，因此除了 15 Hz 训练帧，还必须保留完整 50 Hz
数据，不能只保存对齐到视频后的 15 Hz 结果。

## 8. action定义

```text
observation.arm  ← arm_joint_state
action.arm       ← arm_joint_command

observation.hand ← hand_joint_state
action.hand      ← hand_joint_command

observation.neck ← neck_joint_state
action.neck      ← neck_joint_command

observation.waist ← waist_state
```

禁止把 state 复制为 action。

`neck_joint_command` 之前实测为 0 条。现在它是必需数据，正式采集前必须确认遥操
期间能够持续发布并得到 50 Hz 输出；否则 episode 判为 INVALID。

## 9. episode输出

如果机器人编码器已经直接输出 15 Hz H.265：

```text
episode_XXXX/
├── head_camera.mp4          # HEVC/H.265, 15 Hz, stream copy
├── left_hand_camera.mp4     # HEVC/H.265, 15 Hz, stream copy
├── right_hand_camera.mp4    # HEVC/H.265, 15 Hz, stream copy
├── episode.json             # 15 Hz训练帧
├── signals_50hz.json        # 50 Hz原始运动数据
├── tf_static.json
├── manifest.json
├── quality_report.json
└── SHA256SUMS
```

视频优先使用：

```text
ffmpeg ... -c:v copy ...
```

不进行 RGB/JPEG 中转，不重新编码。

## 10. 验收标准

### 10.1 三路 H.265

- 三路全部存在；
- 每路平均频率 15 Hz，允许误差 ±5%；
- P99 帧间隔不超过 100 ms；
- 最大连续空洞不超过 200 ms；
- 源时间戳严格递增；
- 每路都能找到 VPS/SPS/PPS 和 IDR；
- MP4 从第一可解码帧到最后一帧完整解码；
- 三路存在共同有效时间窗口；
- MP4 帧数和 15 Hz JSON 对应关系可验证。

### 10.2 运动数据

- 必需 state/command 平均频率 50 Hz，允许误差 ±5%；
- P99 间隔不超过 30 ms；
- 最大连续空洞不超过 100 ms；
- 数组维度在 episode 内保持一致；
- source timestamp 严格可追溯；
- 不允许用 state 补造缺失 command。

### 10.3 连续采集

- 启动后连续 3 秒健康才进入 READY；
- episode 之间不重新建立 ROS/gRPC 流；
- 连续至少 20 个 episode 无缺流；
- 任一路相机、必需 command 或 waist 缺失时自动标记 INVALID；
- 网络中断、磁盘不足和进程异常均生成明确错误报告；
- edge bridge 永远不发布 control topic。

## 11. 实施顺序

### 阶段 A：确认源端 15 Hz 能力

先确认三个 H.265 编码器是否能直接配置为 15 Hz。这个结果决定能否无重编码保存。

### 阶段 B：单路 H.265 gRPC

头部 H.265 → edge bridge → gRPC → Spark，验证 VPS/SPS/PPS、IDR、序号和时间戳。

### 阶段 C：三路 H.265

三路同时传输并连续运行 10 分钟，检查 Orin CPU、网络吞吐和 Spark 丢包。

### 阶段 D：50 Hz运动数据

加入 state、command、waist、TF，验证维度和源时间戳。

### 阶段 E：episode

实现长期连接、健康门禁、关键帧预缓存、回车开始/停止和 VALID/INVALID。

### 阶段 F：MP4/JSON

生成三路 H.265 MP4、15 Hz `episode.json`、50 Hz `signals_50hz.json`、manifest
和质量报告。

### 阶段 G：正式验收

完成连续 20 个 episode、掉线、重启、磁盘不足和 LeRobot/VLA 转换验证。

## 12. 当前首要确认项

在写正式 gRPC 采集代码之前，必须先确认：

1. 三路 H.265 是否都能从编码器源头调整到 15 Hz；
2. 三路 H.265 的分辨率、GOP 和 IDR 周期；
3. `/motion/control/waist_state` 的 ROS 类型和字段；
4. waist protobuf wrapper 的 payload 是否可解码；
5. 遥操期间 `neck_joint_command` 是否实际发布；
6. 手部 20 维 position 与 260 维 effort 的厂商定义。
