# 北大方案复盘与 A2 数据获取架构规划

## 1. 已确认结论

北大 Spark 侧通过 `x2robot` Python SDK 连接机器人端
`x2://<robot-ip>:50051`。结合 SDK 文档、流式 API 和错误处理代码，可以确认
底层使用 gRPC/TCP 流式 RPC 获取相机、关节、末端位姿、里程计和 IMU 数据。

这里的“gRPC 获取数据”不等于传输 H.264/H.265 原始码流：

- 相机消息的 `data` 是能够被 PIL 直接解码的压缩图片帧；
- Spark 解码为 RGB 后，再以 JPEG 重新编码到临时文件；
- episode 停止后，Spark 把 JPEG 帧送给 FFmpeg；
- FFmpeg 使用 `libx264 -crf 23` 生成 MP4。

北大现有采集器也不是长期保持所有数据流：

- 第一次回车后调用 `start_recording()`；
- `start_recording()` 为每个传感器创建线程并打开 gRPC stream；
- 第二次回车后调用 `stop_recording()`；
- 停止 stream、读取临时文件、对齐、编码并保存 episode；
- 下一个 episode 会再次建立各路 stream。

## 2. 北大数据链路

```text
机器人端 SDK/gRPC Server :50051
        │
        │ gRPC server-streaming
        ▼
Spark x2robot Python SDK
        │
        ├─ 头部相机压缩图片帧
        ├─ 左臂相机压缩图片帧
        ├─ 右臂相机压缩图片帧
        ├─ 各部件 JointState
        ├─ 左右末端位姿
        ├─ 里程计
        └─ IMU
        │
        ▼
每路一个 Python 采集线程
        │
        ├─ 相机：解码 RGB → JPEG → pickle 临时文件
        └─ 状态：protobuf 字段 → NumPy/字典 → pickle 临时文件
        │
        ▼
停止 episode
        │
        ├─ 以头部相机时间戳作为 30 Hz 主时间轴
        ├─ 关节状态线性插值
        ├─ 其他传感器最近邻
        └─ 其他相机最近邻选帧
        │
        ▼
episode_XXXX/
        ├─ head_camera.mp4
        ├─ left_arm_camera.mp4
        ├─ right_arm_camera.mp4
        └─ episode.json
```

数据集根目录另外维护 `dataset_metadata.json`。

## 3. 北大 episode 逻辑

### 3.1 开始

1. 创建各相机和传感器的临时 pickle 文件。
2. 设置 `is_recording=True`。
3. 启动各路 gRPC stream 线程。
4. 每条消息使用 `header.stamp` 作为采集时间戳。
5. 实时将数据追加到本地临时文件，避免长 episode 占满内存。

### 3.2 停止

1. 设置 `is_recording=False`。
2. 等待线程退出。
3. 关闭并读取临时文件。
4. 验证所有启用的数据源非空。
5. 以头部相机时间戳为统一帧时间轴。
6. 把所有 observation 对齐到该时间轴。
7. 三路相机编码为固定帧率 H.264 MP4。
8. observation/action/时间戳写入 `episode.json`。
9. 原子替换 JSON，并更新数据集总 metadata。

## 4. 北大 action 的真实语义

北大原始采集脚本没有从独立的遥操 command stream 保存关节 action。

- 关节 `action` 在保存时复制当前帧关节 state；
- 部分末端位姿和里程计 action 使用下一帧 observation；
- LeRobot 转换脚本还可以用未来若干帧 state 构造 action chunk。

因此北大数据里的 `action` 是训练目标的后处理定义，不是严格意义上的真实控制
命令。A2 已经发现独立的 `arm_joint_command` 和 `hand_joint_command`，应优先
原样保存这些 command，并把 future-state action 作为可选派生字段，不能混写。

## 5. 北大方案的空间特点

北大数据空间较小的主要原因是：

1. 不保存 RGB 原始数组；
2. 采集中间态使用 JPEG；
3. 最终使用 H.264 MP4；
4. JSON 只记录帧索引和数值；
5. 不把每帧 JPEG 作为最终数据长期保留。

但图像经历了“输入压缩图 → RGB 解码 → JPEG 重编码 → H.264 重编码”，存在重复
有损压缩和额外 CPU/内存开销。

## 6. A2 推荐总体架构

A2 推荐采用“机器人侧 ROS 适配器 + gRPC 数据桥 + Spark episode 采集器”的
混合方案，复用北大易管理的 gRPC 边界，同时保留 A2 的 H.265 原码流。

```text
A2 Orin ROS 2                         Spark /home/yichu/A2
────────────────────                 ─────────────────────────
五路 H.265       ┐
state/command    ├─ persistent ROS    a2_edge_bridge
IMU/TF/pose      ┘  subscribers            │
                                             │ gRPC streaming
                                             ▼
                                      a2_episode_collector
                                             │
                                   常连接 + 5 秒环形缓存
                                             │
                            回车开始 ─────────┤──────── 回车结束
                                             │
                                             ▼
                                      episode_XXXX/
                                      ├─ 5 × camera.mp4
                                      ├─ episode.json
                                      ├─ manifest.json
                                      └─ quality_report.json
```

如果腰/腿/底盘/手指数据只存在于 A2 x86，则在 x86 部署第二个只读 edge bridge，
由 Spark 同时连接 Orin 和 x86。两块板的数据必须保留消息原始时间戳，并在上线前
验证时钟同步。

## 7. A2 组件职责

### 7.1 Orin/x86 edge bridge

- 进程启动后一次性建立 ROS 订阅，episode 之间不重建；
- 只读订阅，不发布机器人控制命令；
- H.265 保留 Annex-B access unit，不解码、不转 JPEG；
- 状态和 command 序列化为稳定 protobuf schema；
- 每条 envelope 携带 topic、类型、源时间戳、接收时间戳和序号；
- 暴露 Health、ListStreams 和 SubscribeStreams gRPC 接口；
- 网络短暂中断时保留有限环形缓冲并报告丢包。

### 7.2 Spark episode collector

- 程序启动即连接所有 edge stream；
- 五路相机分别维护 VPS/SPS/PPS + IDR 可解码环形缓存；
- 状态、action 和 IMU 维护相同时长缓存；
- 全部必需数据健康后才允许开始 episode；
- 第一次回车只切换 episode 状态，不新建远端订阅；
- 从五路相机共同可解码的起点开始写入；
- 第二次回车后追加短 post-roll，再原子关闭 episode；
- 某一路缺失、时间戳回退或序号跳变时标记 episode 无效。

### 7.3 视频落盘

- 采集阶段保存 H.265 原始 access unit 或直接写入可恢复的分段容器；
- episode 结束后使用 FFmpeg `-c:v copy` 封装 MP4；
- 禁止先解码为 RGB/JPEG再重新编码；
- MP4 必须从 VPS/SPS/PPS + IDR 开始并通过完整解码验收。

### 7.4 数值数据落盘

原始层必须保留：

- 双臂、双手、头颈、腰腿/底盘 state；
- 左右末端位姿；
- 里程计、IMU、TF、静态 TF；
- 遥操 command/action；
- 原始源时间戳、接收时间戳和序号。

训练层再以 30 Hz 相机公共时间轴生成 `episode.json`：

- observation：使用 state；
- action：优先使用真实 command；
- `derived_future_state_action`：如需要，作为单独字段生成；
- 高频 state/action 原始数据不能因生成 30 Hz JSON 而被删除。

## 8. 实施阶段

### 阶段 0：接口冻结

列全 Orin 与 x86 上的 topic、类型、字段长度、发布频率、QoS、发布节点和时钟源。
明确手指 20 维 position、260 维 effort 的协议含义。

### 阶段 1：最小 gRPC 链路

只传头部 H.265、双臂 state 和真实 arm command，连续运行 10 分钟，验证序号、
延迟、吞吐和重连。

### 阶段 2：五路相机并发

加入左右手和左右胸相机，确认所有相机持续约 30 Hz，Spark 可同时接收而不丢帧。

### 阶段 3：全部运动和传感器

加入手指、头颈、腰腿/底盘、末端位姿、里程计、IMU、TF 和真实 command。

### 阶段 4：episode 状态机

实现 Ready → Armed → Recording → Finalizing → Valid/Invalid；按回车只改变状态，
不重新订阅。

### 阶段 5：MP4/JSON

五路 H.265 无重编码封装 MP4；生成原始结构化数据、30 Hz 对齐 JSON、manifest
和哈希。

### 阶段 6：长时间验收

完成多 episode 连续采集、断网恢复、进程重启、磁盘不足和相机掉线测试，再进入
正式 VLA 数采。

## 9. 上线验收底线

- 五路相机各自平均频率接近 30 Hz，数量偏差不超过 3%；
- 相机源时间戳严格递增，P99 帧间隔小于 50 ms；
- 不允许任一路相机 Count 为 0；
- MP4 可从第 0 帧解码到最后一帧；
- state 约 100 Hz，真实 command 在遥操期间持续存在；
- 五路视频具有共同的可解码时间窗口；
- episode 开始不依赖新建 DDS/gRPC 订阅；
- JSON 帧数与五路 MP4 帧数一致；
- 原始 action 与派生 action 使用不同字段；
- 每个 episode 有哈希、质量报告和完整/无效状态。

## 10. 当前决策

现有 `a2_live_episode_collector.py` 可作为“长期订阅 + 环形缓存 + episode
状态机”的 ROS 直连原型。正式架构是否采用 gRPC bridge，应先完成阶段 1 的
吞吐与稳定性对比：

1. Spark 直接 DDS 长期订阅；
2. Orin edge bridge → gRPC → Spark。

选择标准不是“是否和北大名字相同”，而是五路 30 Hz H.265 加全部运动数据在
连续多 episode 下的零缺流、低抖动和可恢复性。结合目前 Spark 能发现 topic
但曾无法收到实际 payload 的情况，gRPC bridge 是优先验证路线。
