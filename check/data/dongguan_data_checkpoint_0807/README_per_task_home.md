# per_task_home — DGX 部署预位说明（end_pose）

配合 [`per_task_home.json`](per_task_home.json) 使用。  
从东莞 multi_345 训练数据统计：**每个 task 在 episode 起始帧（t=0）的 median home 位姿**。

**预位方式：`set_end_pose`（position + quaternion），不含 joint。**

---

## 1. 为什么用 quaternion 而不是 rot6d

| 表示 | 维数 | 用途 |
|------|------|------|
| **orientation_xyzw** | 4 | SDK `set_end_pose` **必须用这个** |
| rot6d | 6 | 仅 GR00T 训练/推理内部 eef_9d，**不要发给 SDK** |

JSON 里每只手臂只给 `end_pose.position_m` + `end_pose.orientation_xyzw`。

---

## 2. 文件

| 文件 | 用途 |
|------|------|
| `deploy/per_task_home.json` | **传给 DGX**（schema_version `"2"`） |
| `deploy/README_per_task_home.md` | 本说明 |
| `compute_per_task_home.py` | 重新生成 JSON |

```bash
scp ~/projects/zibianliang_env/pipeline7_dongguan_relative_eef_only/deploy/per_task_home.json \
    ~/projects/zibianliang_env/pipeline7_dongguan_relative_eef_only/deploy/README_per_task_home.md \
    dgx:/path/on/dgx/deploy/
```

---

## 3. task_index 对照

| `task_index` | `task_id` | 任务 | ep 数 |
|--------------|-----------|------|-------|
| **0** | 3 | grasp 门把手 | 302 |
| **1** | 4 | rotate 解锁 | 235 |
| **2** | 5 | pull 开门 | 235 |

---

## 4. JSON 结构（schema v2）

```json
{
  "schema_version": "2",
  "preposition_mode": "end_pose",
  "orientation_format": "quaternion_xyzw",
  "per_task_home": [
    {
      "task_index": 0,
      "task_id": "3",
      "language": "...",
      "lift_position_m": 0.3008,
      "execute_arms": ["right"],
      "left_arm": {
        "gripper_position": -0.0055,
        "end_pose": {
          "position_m": { "x": 0.16, "y": -0.018, "z": -0.051 },
          "orientation_xyzw": { "x": ..., "y": ..., "z": ..., "w": ... }
        }
      },
      "right_arm": { "gripper_position": ..., "end_pose": { ... } }
    }
  ]
}
```

| 字段 | 说明 |
|------|------|
| `lift_position_m` | 升降台（`LiftController.set_lift_position`） |
| `left_arm.end_pose` | 左臂 home；部署后保持不动 |
| `right_arm.end_pose` | 右臂 home；preposition 目标 |
| `gripper_position` | 夹爪目标（与训练一致） |

**不含：** `joint_position_rad`、`eef_9d`、`state_32d`。

---

## 5. 三个 task 摘要

| task_index | task_id | lift (m) | 右臂 xyz (m) |
|------------|---------|----------|--------------|
| 0 | 3 | 0.3008 | 0.210, 0.022, -0.053 |
| 1 | 4 | 0.3006 | 0.316, -0.029, -0.040 |
| 2 | 5 | **0.2595** | 0.316, 0.117, 0.028 |

task5 的 lift 明显低于 task3/4，换 task 前务必重设 lift。

---

## 6. DGX 预位流程

```text
1. home = per_task_home[task_index]

2. set_lift_position(home.lift_position_m)

3. MANIPULATOR_END_POSE 模式

4. 左臂：set_end_pose(home.left_arm.end_pose) + gripper
   → 之后保持不动

5. 右臂：set_end_pose(home.right_arm.end_pose) + gripper

6. live capture 读真实 32D state，填 GR00T history（前 7 帧 padding）

7. language = home.language → inference
```

---

## 7. SDK 调用示例

```python
from x2robot.geometry_msgs import Pose, Point, Quaternion

def load_pose(arm_block: dict) -> Pose:
    ep = arm_block["end_pose"]
    p = ep["position_m"]
    q = ep["orientation_xyzw"]
    pose = Pose()
    pose.position = Point(x=p["x"], y=p["y"], z=p["z"])
    pose.orientation = Quaternion(x=q["x"], y=q["y"], z=q["z"], w=q["w"])
    return pose

home = data["per_task_home"][task_index]
robot.left_arm.set_end_pose(load_pose(home["left_arm"]))
robot.right_arm.set_end_pose(load_pose(home["right_arm"]))
# + set gripper from home["*_arm"]["gripper_position"]
```

---

## 8. 重新生成

```bash
cd ~/projects/zibianliang_env/pipeline7_dongguan_relative_eef_only
../isaacGr00t/.venv/bin/python compute_per_task_home.py
```

---

## 9. 故障排查

| 现象 | 检查 |
|------|------|
| 姿态异常 | 确认用 `orientation_xyzw`，不要发 rot6d |
| 高度不对 | `lift_position_m` 是否匹配当前 task |
| GR00T state 不对 | preposition 后用 **live capture** 读 32D，不要手写 joint |
