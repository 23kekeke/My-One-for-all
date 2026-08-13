# task_45 ACT 复现项目

本目录用于把 `task_45_0001` 的 SDK 原始数据转换为 14 维 LeRobot 数据集，并复现现有的 ACT 训练配置。

## 目录说明

- `config.sh`：所有输入、输出和代码路径。
- `tools/convert_to_lerobot_async.py`：从公共转换器复制的独立版本，已允许 `quanta_x1_raw_joints` 对应 `quanta_x1` 数据。
- `tools/audit_raw_dataset.py`：只读检查原始数据。
- `tools/validate_lerobot_dataset.py`：通过 LeRobot 实际加载转换结果并检查三路相机和 14 维状态/动作。
- `scripts/03_train_smoke.sh`：2,000 步冒烟训练。
- `scripts/04_train_baseline.sh`：复现参考 ACT 的 1,000,000 步训练。
- `datasets/`：转换后的数据。
- `outputs/`：模型和 checkpoint。

LeRobot 核心源码仍使用：

```text
/home/yichu/yichu_work/lerobot
```

这样不会修改公共 LeRobot，也不会维护一份容易过期的重复源码。训练脚本通过 `PYTHONPATH` 明确加载该仓库。

## 第 0 步：进入目录并赋予脚本执行权限

```bash
conda activate act_train
cd /home/yichu/shixisheng/zzm/act_task45
chmod +x scripts/*.sh
```

默认使用 Conda 环境 `xr_lerobot`。如果环境名或路径不同，先修改 `config.sh`。

## 第 1 步：动态检查原始数据

以下命令以实际存在的 `episode_*` 目录为准，不要求每组数据具有固定数量。元数据数量不一致只作为提示；只有文件缺失、JSON 损坏、时间戳异常、状态/动作不是 14 维或出现 NaN/Inf 才判定失败。

```bash
python tools/check_dataset_structure.py \
  --dataset /home/yichu/yichu_work/datasets/sdk_dongguan/task_45_0001 \
  --actual-only \
  --deep-json
```

检查其他数据时，只修改 `--dataset` 后面的路径。预期最后输出：

```text
Status: PASSED
```

## 第 2 步：按清单批量转换为 LeRobot 数据集

需要转换的原始数据组及其先后顺序写在txt文件里面：

```text
/home/yichu/shixisheng/zzm/act_task45/dataset_groups.txt
```

示例：

```text
task_45_0001
task_45_0002
task_45_0003
task_45_0004
task_45_0005
```

每行一个数据组。清单从上到下决定组间顺序；每组内部自动按照 `episode_0000、episode_0001、...` 的数字顺序转换。空行和 `#` 开头的注释会被忽略。缺少 `dataset_metadata.json` 的数据组不要加入清单。

确认清单后执行：

```bash
./scripts/01_convert.sh
```

脚本会逐组执行深度检查，然后分别输出到：

```text
datasets/converted/task_45_0001_raw14
datasets/converted/task_45_0002_raw14
datasets/converted/task_45_0003_raw14
...
```

各组不会在转换阶段混到同一个目录。实际转换顺序保存在：

```text
reports/conversion_order.tsv
```

转换脚本不会覆盖已有输出。如果转换中断并留下不完整目录，请先人工检查，然后将旧目录改名备份，再重新运行。

## 第 3 步：检查转换数据结构是否符合要求
```
cd /home/yichu/shixisheng/zzm/act_task45

python tools/validate_lerobot_dataset.py \
  --root /home/yichu/shixisheng/zzm/act_task45/datasets/converted/task_45_0001_raw14 \
  --repo-id dongguan/task_45_0001_raw14
```
## 第 4 步：合并数据
```
./scripts/03_merge_all.sh
```
## 第 3 步：验证转换结果

```bash
./scripts/02_validate_all.sh
```

它会通过 LeRobot 加载数据并解码三个样本的三路视频。预期最后输出：

```text
LEROBOT DATASET VALIDATION PASSED
```

只有所有组都验证通过后才能合并：

```bash
./scripts/03_merge_all.sh
```

合并后的数据集为：

```text
/home/yichu/shixisheng/zzm/act_task45/datasets/task_45_clean_merged
```

合并顺序与 `dataset_groups.txt` 完全一致；每组内部保持数字 episode 顺序。合并后 LeRobot 会重新生成连续的全局 episode 编号，原始组名和原始编号可通过 `reports/conversion_order.tsv` 追溯。

## 第 4 步：运行 2,000 步冒烟训练

```bash
./scripts/03_train_smoke.sh
```

输出位置：

```text
/home/yichu/shixisheng/zzm/act_task45/outputs/act_task45_smoke
```

检查项目：

1. GPU 能正常使用。
2. 三路视频能被 DataLoader 读取。
3. 输入和输出维度均为 14。
4. loss 下降且没有 NaN。
5. 第 1,000 和第 2,000 步能保存 checkpoint。

## 第 5 步：运行 ACT 基线训练

冒烟训练完全通过后执行：

```bash
./scripts/04_train_baseline.sh
```

输出位置：

```text
/home/yichu/shixisheng/zzm/act_task45/outputs/act_task45_baseline
```

基线参数为三路相机、14 维关节状态/动作、ResNet18、100 步 action chunk、VAE latent 32、KL 权重 10、batch size 16 和 1,000,000 次更新。

## 重要说明

原始数据中的同帧 `action` 与 `observation.state` 完全一致。ACT 仍可通过未来 100 帧学习后续轨迹，但这代表“未来实测关节位置”，不是真正独立记录的遥操作目标命令。先用它复现基线；后续改进应优先记录真实目标动作和命令时间戳。

首次部署到真实机器人时，不要直接连续执行 100 步。先打印动作并检查 14 维顺序、关节限位和夹爪范围，再以低速、短执行窗口和急停保护进行测试。
