# pipeline7_dongguan_relative_eef_only

东莞 cabinet 数据：**state 32D 不变**，**action 20D（仅 eef_9d + gripper，RELATIVE eef）**。

## 与 pipeline5 差异

| | pipeline5 | pipeline7 |
|--|-----------|-----------|
| State | 32D × 6 组 | **相同** |
| Action | 32D × 6 组 | **20D × 4 组**（无 joint） |
| Step0–3 | 跑 | **复用 p5 tmp** |
| LeRobot | `lerobot/multi_345` | **`lerobot_p7/multi_345`** |
| 视频 | copy | **软链自 p5** |
| Checkpoint | `output/` | **`output_p7/`**（训练阶段） |

## 数据流程（Step4–6）

```bash
cd ~/projects/zibianliang_env/pipeline7_dongguan_relative_eef_only

# Step4: 从 p5 step3_rot6d 导出；mp4 软链自 p5 lerobot
python step4_export_lerobot.py \
  --input-root /data/dongguan_data_checkpoint_0807/tmp/step3_rot6d \
  --output-root /data/dongguan_data_checkpoint_0807/lerobot_p7/multi_345 \
  --link-videos-from /data/dongguan_data_checkpoint_0807/lerobot/multi_345

../isaacGr00t/.venv/bin/python step5_generate_relative_stats.py

../isaacGr00t/.venv/bin/python step6_smoke_loader_relative.py

../isaacGr00t/.venv/bin/python pre_train_confirm.py
```

训练前确认：`tmp/step_smoke_p7/pre_train_confirm.json` 中 `ready_for_train=true`。

训练流程见 [`README_train.md`](README_train.md)。
