# Dongguan GR00T — DGX 离线部署包

独立 infer 环境，适用于 DGX 搬迁至其他网络（无需访问原训练机或 HuggingFace）。

## 目录结构

```text
deploy_dgx/
  env.sh                    # 环境变量（必 source）
  verify_deploy.sh          # 部署验收
  pack_for_transfer.sh      # 打成 tar 便于搬运
  manifest.json
  models/
    GR00T-N1.7-3B/          # QLoRA rebuild base (~13G)
    Cosmos-Reason2-2B/      # VLM 本地路径 (~9G)
  checkpoints/
    checkpoint-14000/       # finetuned (~12G each)
    checkpoint-15000/       # open-loop 最优，推荐默认
  code/
    isaacGr00t/             # GR00T + .venv (~16G)
    pipeline2/              # hooks + quanta_x1_inference
    pipeline3_biman/        # live_runner / SDK daemon
    pipeline5_dongguan_relative/  # dongguan infer
  pack/                     # tar 输出（运行 pack_for_transfer.sh 后）
  tmp/dongguan_inference/   # parity / live 日志
```

**总体积约 ~58GB**（2×ckpt + 2×base model + venv + 代码）

## 在本机构建（源机器）

```bash
cd ~/projects/zibianliang_env/pipeline5_dongguan_relative/deploy
chmod +x setup_deploy_dgx.sh pack_for_transfer.sh verify_deploy.sh

# 填充 deploy_dgx（rsync，约 30–60 分钟）
RSYNC_PROGRESS="--info=progress2" ./setup_deploy_dgx.sh

# 验收
source /data/dongguan_data_checkpoint_0807/deploy_dgx/env.sh
/data/dongguan_data_checkpoint_0807/deploy_dgx/verify_deploy.sh

# 打包搬运（可选，生成 pack/*.tar.zst）
/data/dongguan_data_checkpoint_0807/deploy_dgx/pack_for_transfer.sh
```

## 远程 DGX 恢复

```bash
# 若用 tar 搬运，在目标机解压到同一路径：
mkdir -p /data/dongguan_data_checkpoint_0807/deploy_dgx
cd /data/dongguan_data_checkpoint_0807/deploy_dgx
tar -xf pack/code.tar.zst      # 或 .tar
tar -xf pack/models.tar.zst
tar -xf pack/checkpoints.tar.zst

source env.sh
./verify_deploy.sh
```

若 `.venv` 无法启动（路径变化），在目标机重建：

```bash
cd $DONGGUAN_DEPLOY_ROOT/code/isaacGr00t
uv venv .venv --python 3.10
source .venv/bin/activate
uv pip install -e .
# 再装训练时相同的 QLoRA 依赖（bitsandbytes, peft, flash-attn 等）
```

## 常用 infer 命令

```bash
source "$DONGGUAN_DEPLOY_ROOT/env.sh"   # 或本包根目录 env.sh
export CHECKPOINT="$DEPLOY/checkpoints/multi345_v1/checkpoint-15000"
# 机器人 gRPC（东莞直连）；也可直接用 env 里的 $ROBOT_SERVER
ROBOT="${ROBOT_SERVER:-192.168.1.103:50051}"

# --- 终端 B：SDK daemon（xr_lerobot，不要用 GR00T venv）---
# 需重启以加载 set_lift（daemon v2）
export PYTHONPATH="$CODE/pipeline2:$CODE/pipeline3_biman:${PYTHONPATH:-}"
$PY_SDK -m quanta_biman_inference.live_sdk_daemon serve --server "$ROBOT"
# daemon JSON-RPC 默认 127.0.0.1:15101（可用 --sdk-daemon-url 覆盖）

# --- 终端 C：parity / shadow / live ---
# parity
$PY parity_check_relative.py --checkpoint "$CHECKPOINT" --load-policy

# shadow 冒烟
$PY -m dongguan_inference.live_runner \
  --mode shadow \
  --checkpoint "$CHECKPOINT" \
  --task-index 1 \
  --cycles 1 \
  --execution-horizon 1 \
  --interval-sec 0 \
  --server "$ROBOT" \
  --sdk-backend daemon \
  --sdk-daemon-url 127.0.0.1:15101 \
  --sdk-backend-required

# shadow 热身（前 7 cycle 为 temporal warmup）
$PY -m dongguan_inference.live_runner \
  --mode shadow \
  --checkpoint "$CHECKPOINT" \
  --task-index 1 \
  --cycles 10 \
  --execution-horizon 1 \
  --interval-sec 0 \
  --server "$ROBOT" \
  --sdk-backend-required

# live（危险；默认先 per-task home 再推理）
$PY -m dongguan_inference.live_runner \
  --mode live \
  --execute \
  --acknowledge QUANTA_BIMAN_32D_LIVE \
  --checkpoint "$CHECKPOINT" \
  --task-index 1 \
  --execute-arms auto \
  --cycles 40 \
  --execution-horizon 1 \
  --execute-interpolate-hz 100 \
  --trajectory-settle-sec 0.05 \
  --max-joint-delta-rad 0.05 \
  --settle-sec 0.6 \
  --interval-sec 0 \
  --server "$ROBOT" \
  --sdk-backend-required \
  --per-task-home-json "$DEPLOY/per_task_home.json" \
  --preposition-tolerance-m 0.02 \
  --preposition-orient-tolerance-rad 0.20 \
  --preposition-max-steps 5 \
  --preposition-lift-settle-sec 1.0
```

完整逐步说明见 `code/pipeline5_dongguan_relative/infer_dongguan.txt`。

## live_runner 可调参数

所有原 pipeline3 参数仍可用；东莞额外加了 home 相关项。`python -m dongguan_inference.live_runner --help` 可看全量。

| 参数 | 默认 / 建议 | 说明 |
|------|-------------|------|
| `--mode` | `shadow` / `live` | shadow 只推不发；live+`--execute` 才动臂 |
| `--execute` | off | 真机动臂开关 |
| `--acknowledge` | — | live 必须：`QUANTA_BIMAN_32D_LIVE` |
| `--server` | **`192.168.1.103:50051`** | 机器人 gRPC |
| `--sdk-daemon-url` | `127.0.0.1:15101` | 本机 daemon JSON-RPC |
| `--sdk-backend` | `daemon` | `daemon` / `subprocess` |
| `--sdk-backend-required` | off | daemon 不可达则失败（建议开） |
| `--checkpoint` | ckpt-15000 | 策略权重 |
| `--task-index` | 0/1/2 | grasp / rotate / pull |
| `--execute-arms` | `auto` | `auto`/`right`/`left`/`both`/`none` |
| `--cycles` | 3 | 推理循环次数 |
| `--execution-horizon` | 1 | 每 cycle 执行几步；shadow/live 先从 1 |
| `--execute-interpolate-hz` | 0 | `>0` 时本机插值后流式下发（建议 100） |
| `--trajectory-settle-sec` | 0.05 | 插值轨迹结束后等待 |
| `--max-joint-delta-rad` | 0.05 | 单步关节限幅；稳后可试 0.08 |
| `--settle-sec` | 0.6 | 每步 execute 后等待 |
| `--interval-sec` | 0 | cycle 间隔 |
| `--train-fps` | 15 | 插值用训练帧率 |
| `--run-root` | 自动时间戳目录 | 日志输出根目录 |
| `--robot-python` | xr_lerobot python | subprocess 后端用 |
| **`--preposition-home` / `--no-preposition-home`** | live+execute **默认开** | 先到 per-task home 再推理 |
| `--per-task-home-json` | `$DEPLOY/per_task_home.json` | home 位姿文件 |
| `--preposition-skip-lift` | off | 只动臂、不调升降台 |
| `--preposition-lift-settle-sec` | 1.0 | lift 到位后等待 |
| `--preposition-tolerance-m` | 0.02 | home end_pose xyz L2 阈值（米） |
| `--preposition-orient-tolerance-rad` | 0.20 | home 姿态角阈值（弧度） |
| `--preposition-max-steps` | 5 | set_end_pose 最大重试次数 |

## Checkpoint 选择

| Checkpoint | Open-loop MSE (5 ep) |
|------------|----------------------|
| **15000** | **0.00120** ← 推荐 |
| 14000 | 0.00218 |

```bash
export CHECKPOINT="$DEPLOY/checkpoints/multi345_v1/checkpoint-15000"   # 或 14000
```

## task_index

| Index | 任务 |
|-------|------|
| 0 | grasp (task3) |
| 1 | rotate (task4) |
| 2 | pull (task5) |

## per-task home（live 必做）

见 [`per_task_home.json`](per_task_home.json) / [`README_per_task_home.md`](README_per_task_home.md)。

`dongguan_inference.live_runner --mode live --execute` **默认**先按 `--task-index`：

1. `set_lift(lift_position_m)`
2. `MANIPULATOR_END_POSE` 模式
3. 左臂 / 右臂 `set_end_pose`（position_m + orientation_xyzw）+ gripper
4. 到位后再开推理 cycle

跳过：`--no-preposition-home`（不推荐）。

## 注意事项

- **必须** `source env.sh`；`PYTHONPATH` 中 pipeline5 需在 pipeline2 之前（ViT QLoRA hooks）
- 不要用 `quanta_biman_inference.live_runner`，用 `dongguan_inference.live_runner`
- live 前 7 cycle 为 temporal warmup（`[-7,-1,0]`）
- SDK 侧仍用 `pipeline3_biman` 的 `live_sdk_daemon`（xr_lerobot 环境，非 GR00T venv）；需支持 `set_lift`（daemon v2）
- 机器人 gRPC：`--server 192.168.1.103:50051`（直连）。`127.0.0.1:15051` 仅用于旧 SSH 隧道拓扑
- 换 task 前确认 lift：task3/4 ≈ 0.30 m，task5 ≈ 0.26 m
