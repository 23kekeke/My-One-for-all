# pipeline7_dongguan_relative_eef_only — 训练 README

东莞 LeRobot 数据集上的 **eef-only RELATIVE action（20D）+ 三帧历史 + 双 QLoRA (r=32)** 训练流程。  
数据处理见 [`README.md`](README.md)。

---

## 1. 训练契约（相对 pipeline5）

| 项 | pipeline5 | pipeline7 |
|----|-----------|-----------|
| State | 32D × 6 组（eef+gripper+joint ×2） | **相同** |
| Action | 32D × 6 组 | **20D × 4 组**（eef+gripper only，**无 joint**） |
| eef | RELATIVE | **RELATIVE** |
| gripper | ABSOLUTE | **ABSOLUTE** |
| `delta_indices` | `[-7,-1,0]` @ 15fps | **相同** |
| Parquet BC | `action[t]=state[t+1]` | 同左（按 action key 切片） |
| `relative_stats.json` | 4 keys（eef+joint） | **2 keys**（`left_eef_9d`, `right_eef_9d`） |
| LeRobot | `lerobot/multi_345` | **`lerobot_p7/multi_345`** |
| Checkpoint | `output/` | **`output_p7/`** |
| Base 模型 | `GR00T-N1.7-3B` | **同左（不可复用 pipeline5 ckpt）** |

@ 15fps，`[-7,-1,0]` ≈ **0.47s / 0.07s / 当前** 三时刻观测。

### Action 分组（4 组）

| 组 | 表示 |
|----|------|
| left/right `eef_9d` | RELATIVE |
| left/right `gripper` | ABSOLUTE |

`meta/relative_stats.json`：仅上述 2 个 RELATIVE eef 组的 normalize 边界。

---

## 2. QLoRA：LLM 与 ViT

| 模块 | 底座 | 可训练 |
|------|------|--------|
| LLM | 4-bit NF4 冻结 | **PEFT LoRA r=32**（α=64） |
| ViT | 4-bit NF4 冻结 | **PEFT LoRA r=32** |
| DiT / projector / vlln | bf16 | 全量微调 |

**不要**用 `tune_visual=True` 代替 ViT QLoRA。  
实现文件：`gr00t_qlora_hooks.py`。

---

## 3. 路径约定

| 用途 | 路径 |
|------|------|
| LeRobot 数据集 | `/data/dongguan_data_checkpoint_0807/lerobot_p7/multi_345` |
| 训练 checkpoint | `/data/dongguan_data_checkpoint_0807/output_p7/` |
| 代码 | `~/projects/zibianliang_env/pipeline7_dongguan_relative_eef_only/` |
| Base 模型 | `~/projects/zibianliang_env/GR00T-N1.7-3B` |
| Cosmos VLM | `~/projects/zibianliang_env/Cosmos-Reason2-2B` |
| Smoke 报告 | `/data/dongguan_data_checkpoint_0807/tmp/step_smoke_p7/` |

---

## 4. Task 语言（来自 manifest）

| task_index | GR00T language（`tasks.jsonl`） |
|------------|----------------------------------|
| 0 | Use only your right hand to grasp the door handle. Keep your left arm still. |
| 1 | Use only your right hand to rotate the door handle to unlock. Keep your left arm still. |
| 2 | Use only your right hand to pull the cabinet door open. Keep your left arm still. |

---

## 5. 训练前检查清单

- [x] 数据处理 Step4–6 完成（multi_345）
- [x] `pre_train_confirm.json` → `ready_for_train=true`
- [x] **A2** `confirm_train_paths.json` → `ready_for_launch=true`
- [x] Step7 overfit smoke 通过（Phase B）
- [ ] `dongguan_eef_only_relative_config.py` 中 `OBSERVATION_DELTA_INDICES = [-7, -1, 0]`
- [ ] 视频软链目标（pipeline5 lerobot）仍可访问

```bash
cd ~/projects/zibianliang_env/pipeline7_dongguan_relative_eef_only

../isaacGr00t/.venv/bin/python pre_train_confirm.py
../isaacGr00t/.venv/bin/python confirm_train_paths.py
```

---

## 6. 训练日志与 run 对比

### 6.1 每个 run 目录（`output_p7/<run_name>/`）

| 文件 | 内容 |
|------|------|
| `train.log` | 完整 stdout/stderr（`launch_train.sh` tee） |
| `launch_command.sh` | 实际执行的命令行 |
| `run_manifest.json` | 超参、dataset 规模、eef-only 契约、QLoRA r=32+ViT |
| `training_summary.json` | loss 首尾/min/max、checkpoint 列表 |
| `loss_series.json` | 从 `checkpoint-*/trainer_state.json` 汇总 |
| `experiment_cfg/config.yaml` | GR00T 完整 config |

### 6.2 全局索引

| 文件 | 内容 |
|------|------|
| `output_p7/runs_index.jsonl` | 每次训练结束追加一行摘要 |

---

## 7. 环境变量

```bash
export GR00T_COSMOS_MODEL_PATH="/home/ubuntu/projects/zibianliang_env/Cosmos-Reason2-2B"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export NO_ALBUMENTATIONS_UPDATE=1
export PIPELINE2_QLORA=1
export GROOT_PATCH_MISTRAL=1
export GROOT_HF_LOCAL_FIRST=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="/home/ubuntu/projects/zibianliang_env/isaacGr00t:/home/ubuntu/projects/zibianliang_env/pipeline2:/home/ubuntu/projects/zibianliang_env/pipeline3_biman:/home/ubuntu/projects/zibianliang_env/pipeline7_dongguan_relative_eef_only:${PYTHONPATH:-}"
```

---

## 8. 训练步骤

可复制命令见 [`train_dongguan_eef_only.txt`](train_dongguan_eef_only.txt)。

### Step A — 路径确认（A2）

```bash
cd ~/projects/zibianliang_env/pipeline7_dongguan_relative_eef_only
../isaacGr00t/.venv/bin/python confirm_train_paths.py
```

报告：`tmp/step_smoke_p7/confirm_train_paths.json`。

### Step B — Overfit smoke（Phase B）

```bash
../isaacGr00t/.venv/bin/python step7_smoke_overfit_relative.py \
  --source-dataset-path /data/dongguan_data_checkpoint_0807/lerobot_p7/multi_345 \
  --episode-index 0 --max-steps 320 --save-steps 160
```

报告：`tmp/step_smoke_p7/overfit_report.json`  
Checkpoint：`output_p7/overfit_ep0_smoke/`（ep0 overfit，loss 1.24→0.63 @320 steps，gbs=2 ga=2）

### Step C — 正式训练（multi_345，推荐）

**一键启动（含 train.log）：**

```bash
cd ~/projects/zibianliang_env/pipeline7_dongguan_relative_eef_only
./launch_train.sh multi345_v1
```

**v1 默认超参：**

| 项 | 值 | 说明 |
|----|-----|------|
| 数据集 | `lerobot_p7/multi_345` | 772 ep / 88514 frames |
| `max_steps` | 15000 | |
| `global_batch_size` | 2 | |
| `gradient_accumulation_steps` | 4 | 有效 batch = **8** |
| `save_steps` | 1000 | 保留最近 7 个 ckpt |
| `state_dropout_prob` | 0.2 | overfit 用 0.0 |
| action 模式 | **RELATIVE eef + ABS gripper** | 20D，无 joint |

断点续训：`./launch_train.sh multi345_v1 --resume-from-checkpoint`（同一 output_dir）。

可选 W&B：`./launch_train.sh multi345_v1 --use-wandb`（project: `dongguan-pipeline7`）。

---

## 9. 与 pipeline5 checkpoint 的关系

| Checkpoint | 能否用于 pipeline7 |
|------------|-------------------|
| pipeline5 relative `[-7,-1,0]` 32D action | ❌ action head 维度不匹配（32 vs 20） |
| pipeline7 eef-only relative | ✅ 仅配 `lerobot_p7` + 本目录 modality |

**必须**从 `GR00T-N1.7-3B` 从头训；勿把 pipeline5 finetuned ckpt 作 `--base-model-path`。

---

## 10. 文件索引（训练相关）

```text
pipeline7_dongguan_relative_eef_only/
  README_train.md                         # 本文件
  train_dongguan_eef_only.txt             # 可复制命令
  confirm_train_paths.py                  # A2 路径确认
  launch_train.sh                         # 正式训练 launcher
  dongguan_eef_only_relative_config.py    # delta [-7,-1,0] + 20D action
  dongguan_eef_only_relative_modality.py
  dongguan_eef_only_finetune_entry.py
  gr00t_dongguan_hooks.py
  gr00t_qlora_hooks.py
  train_logging_utils.py
  train_smoke_utils.py
  pre_train_confirm.py
  step7_smoke_overfit_relative.py
  step8_open_loop_sweep_relative.py       # Phase D（待落地）
  parity_check_relative.py                # Phase D（待落地）
```

---

## 11. 故障排查

| 现象 | 可能原因 |
|------|----------|
| 训练报 missing mp4 | p5 lerobot 被删或软链断裂 → 检查 `confirm_train_paths` 的 `videos_resolvable` |
| 从 pipeline5 ckpt 续训失败 | action_dim 32→20 不兼容 → 改用 `GR00T-N1.7-3B` |
| OOM | 减 batch、增 grad accum |
| loss 与 p5 不可直接比 | action 空间不同（20D eef-only vs 32D full） |
