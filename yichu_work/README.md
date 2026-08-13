# 数据录制
## 录制准备工作
### 准备conda环境
```sh
conda activate xr_lerobot


```
### 启动机器人
+ 机器人开机，等待零点复位
+ **每次开始录制前**，需要将机器人移动到指定位置，**误操作移动头部、误操作移动腰部**时也需要执行下面的代码，复位到指定位置，命令如下：
```sh 
python /home/yichu/yichu_work/sdk_robot-main/examples/robot_control_yichu.py \
  --server 192.168.1.103:50051 \
  --lift-to 0.5 \
  --head \
  --arms
```
```sh 
python /home/yichu/yichu_work/sdk_robot-main/examples/robot_control_yichu.py \
  --server 192.168.1.103:50051 \
  --lift-to 0.3 \
  --head \
  --arms
```
#### 参数说明：
+ server：机器人地址
+ --lift-to：腰部移动到指定位置
+ --lift-move：腰部移动相对位置
+ --lift-zero：腰部降到最低
+ --head：头部回中
+ --head-yaw-to：头部偏航角度
+ --head-pitch-to：头部俯仰角度
+ --arms：手臂归零
+ --chassis-x：底盘前后移动
+ --chassis-yaw：底盘旋转


### subtask列表
| 步骤| 动作 |位置|
|------|--------|---|   
| 1 | Right hand presses the switch |2|
| 2 | Right hand rotate the handle and open the door|2|
| 3 | Raise the left hand, and push the door to 90 degrees with the right hand.|2|
| 4 | Left hand taps the screen to wake it up,observes the screen |1|
| 5 | Left hand close the door |3|
| 6 | Right hand close the door firmly and rotate the handle|2|
| 7 | Right hand press down the handle |2|
| 8(123) | Open the door |2|
| 9(67) | close the door |2|
### subtask列表in 东莞
| 步骤| 动作 |位置|
|------|--------|---|   
| 1 | Left hand reaches toward the front of the cabinet door for observation. 
| 2 | Right hand reaches for the door handle.
| 3 | Right hand grasps the door handle.
| 4 | Right hand rotates the door handle to unlock.
| 5 | Right hand pulls the cabinet door open.
| 6 | Left hand lifts up.
| 7 | Right hand opens the cabinet door fully. 
| 8 | Left hand lowers and reaches toward the front of the cabinet door for observation. 
| 9 | Right hand reaches for the black rotary switch.
| 10 | Right hand rotates the black rotary switch.
| 11 | Right hand moves to the green button. 
| 12 | Right hand presses the green button.
| 13 | Right hand moves to the gray button and pauses for 3 seconds.
| 14 | Right hand moves back to the black rotary switch.
| 15 | Right hand rotates the black rotary switch.
| 16 | Right hand returns to the home position.
| 17 | Left hand pushes the cabinet door closed. 
| 18 | Right hand rotates the door handle to 90° and holds the cabinet door in place. 
| 19 | Right hand rotates the door handle to close and lock the cabinet door.
| 4 and 5 |Right hand rotates the door handle to unlock and right hand pulls the cabinet door open.
| all | Stage 2.
+ 每个subtask有指定的底盘位置，见地面标记，由靠近机柜到远离机柜分别为位置123，对应底盘后侧，**误操作了机器人底盘、腰部、头部**，则删掉该条ROS BAG数据，取消SDK数据保存，手动移动底盘回原位
#### 数据量要求：
| 步骤| 第1轮数据量 |  
|------|--------|   
| 1至7 | 各200组，共1400组 |     
| 1+2+3 | 200组|
| 6+7 | 200组|
| 合计 | 1800组|


### 录制流程
+ ROS BAG和SDK需要同步录制，尽可能同步点击开始录制
+ 遥操开机，浏览器登陆，切换到数采模式，注意需要**先进入数采模式**，再开启SDK录制，否则数采模式进不去


## ROS BAG录制
### 录制要求
+ UI界面每次开机需要创建任务，要求**填写任务名称**，名称格式：subtask_name，subtask为子任务列表，name为录制人员名称，便于溯源
+ 录制结果需要定期通过scp从机器人内部拷贝到本地，**每个subtask负责人员负责管理自己的数据**，拷贝命令如下,注意该拷贝命令是整个文件夹都拷贝过来，需要确认数据是否都是自己录制的数据，**设备密码是123**。
``` sh
scp -r xr@192.168.36.116:/home/xr/bagfiles/bags /home/yichu/yichu_work/datasets/rosbag
```
+ 由于机器人内部设备硬盘资源有限，需要及时拷贝到本地，并且删除机器人内部数据，命令如下，建议**每天清理+每次切换subtask清理**。
``` sh
rm -r xr@192.168.36.116:/home/xr/bagfiles/bags 
```


## SDK录制
### 录制命令
```bash
python /home/yichu/yichu_work/data_collection/record.py \
--server 192.168.1.103:50051  \
--out /home/yichu/yichu_work/datasets/sdk_dongguan/task_all_new_0001 \
--task "Stage 2."
```

+ 参数说明：
+ server:机器人地址
+ out:输出目录，**需要录制时修改**
+ task:对应subtask内容
### 交互流程
+ 回车space：每条轨迹录制开始需要回车,每次结束录制需要回车space
+ 遥操结束后需要选择y/n/r:
+ y表示成功录制，保存轨迹;
+ n表示录制到此结束，保存轨迹并结束;
+ r表示失败录制，将自动覆盖上一条的录制内容;
### 输出结构
```
./collected_data/
├── dataset_metadata.json
└── episode_0000/
    ├── episode.json         # 帧数据（对齐后的关节、位姿、时间戳等）
    ├── head_rgb_stream.mp4  # 头部 RGB 视频
    ├── left_arm_rgb_stream.mp4
    └── right_arm_rgb_stream.mp4
```
+ 现在记录的数据有：左臂关节状态、右臂关节状态、左臂末端位姿、右臂末端位姿、里程计、imu、头部关节、腰部关节
### SDK录制数据可视化
```sh
python /home/yichu/yichu_work/data_collection/data_viz.py \
/home/yichu/yichu_work/datasets/sdk/task_3_0001/episode_0006
```
+ 回放轨迹路径自行修改

### 转lerobot格式
```sh
python3 /home/yichu/yichu_work/data_collection/convert_to_lerobot.py \
--input-dir /home/yichu/yichu_work/datasets/sdk/pull_the_door \
--output-dir /home/yichu/yichu_work/datasets/sdk_to_lerobot/test \
--repo-id "my_robot/dataset" \
--robot-type "quanta_x1" \
--use-videos 
```


**async(异步执行 速度更快 推荐使用)**
```sh
python3 /home/yichu/yichu_work/data_collection/convert_to_lerobot_async.py \
--input-dir /home/yichu/yichu_work/datasets/sdk/task_3_0001 \
--output-dir /home/yichu/yichu_work/datasets/sdk_to_lerobot/task_3_0001 \
--repo-id "my_robot/dataset" \
--robot-type "quanta_x1" \
--use-videos
--processes 4
```
+ 参数说明:
+ input-dir:输入路径
+ output-dir:输出路径
+ 其他参数不用更改
+ **转换速度不快，可以边录边转**

### 合并lerobot数据集
+ 必要时需要进行数据合并
```sh
HF_LEROBOT_HOME=/home/yichu/yichu_work/datasets/sdk_to_lerobot lerobot-edit-dataset \
    --new_repo_id merged \
    --operation.type merge \
    --operation.repo_ids "['pull_the_door_1','pull_the_door']" \
    --new_root=/home/yichu/yichu_work/datasets/lerobot/merged_0527
```
+ 参数说明:
+ HF_LEROBOT_HOME:原数据集路径
+ operation.repo_ids:原数据集下需要合并的文件夹名称
+ new_root:合并后的数据保存路径


### 可视化lerobot数据集
```sh
python /home/yichu/yichu_work/lerobot/src/lerobot/scripts/lerobot_dataset_viz.py \
--repo-id my_robot/dataset \
--root /home/yichu/yichu_work/datasets/sdk_to_lerobot/pull_the_door \
--episode-index 0
```
+ root:可视化数据集路径
+ episode-index:指定轨迹



# 系统迁移
+ 用于配置新的数采设备
## miniconda安装
```sh
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

sh Miniconda3-latest-Linux-x86_64.sh
```
## conda环境安装
```sh
conda create -n xr_lerobot python=3.12
```
## 自变量的sdk
+ 在https://github.com/X-Square-Robot/sdk_robot/releases中下周sdk的whl
+ x2robot-1.0.7-py3-none-any.whl 
+ 安装:
```sh 
pip install x2robot-1.0.7-py3-none-any.whl
```
+ 遇到库版本不对：
```sh
pip install betterproto2==0.9.0
```

## 下载lerobot
```sh
https://github.com/huggingface/lerobot

cd lerobot

pip install -e .
```

+ 如果需要数据集：

```sh
pip install 'lerobot[datasets]'
pip install 'lerobot[viz]'
```
+ 使用说明文档：
+ https://huggingface.co/docs/lerobot/installation




## lerobot录制方式（备用） 
### 激活lerobot环境

```sh
conda activate xr_lerobot
```

### 录制命令
```sh
python -m lerobot.scripts.lerobot_record_x_robot \
    --robot.type=x_robot \
    --robot.server="192.168.36.116:50051" \
    --robot.head_camera.width=1280 --robot.head_camera.height=720 \
    --robot.enable_lift=true \
    --dataset.repo_id=my_name/xrobot_dataset \
    --dataset.root=/home/yichu/dataset/subtask_name_data \
    --dataset.num_episodes=100 \
    --dataset.single_task="Pick up" \
    --display_data=true \
    --resume=False \
    --robot.log_alignment_stats=true
```

### 可能需要更改的参数
+ --robot.server：换机器人的时候需要更改
+ --dataset.root：录制数据的目录
+ --dataset.num_episodes：最大录制数量，到数量自动停，可以提前停
+ --dataset.single_task：该条数据的任务名称
+ --display_data：rerun可视化的开关接口

### 录制过程的指令
+ 常规流程
+ 0 → 停止录制 (stop_recording)
+ 1 → 完成并保存当前 episode (exit_early)
+ 2 → 丢弃当前 episode 重新录制 (rerecord_episode)
+ 3 → 开始录制 (start_recording)

### rerun回放可视化
```
python /home/yichu/yichu_work/lerobot/src/lerobot/scripts/lerobot_dataset_viz.py \
    --repo-id my_name/xrobot_dataset \
    --episode-index 0000 \
    --root /home/yichu/dataset/sdk/pull_the_door_1/

```
### 可能需要更改的参数
+ --episode-index：指定回放轨迹
+ --root：指定回放数据集

### 回放轨迹运动
+ 使用时，一定要谨慎，尽量先打印，再send_action() 
(/home/yichu/yichu_work/lerobot/src/lerobot/robots/x_robot/x_robot.py 185-187行)

```sh
python /home/yichu/yichu_work/lerobot/src/lerobot/scripts/lerobot_replay_x_robot.py \
--robot.type=x_robot  \
--robot.server="192.168.36.116:50051"  \
--dataset.repo_id=my_name/xrobot_dataset  \
--dataset.root=/home/yichu/yichu_work/sdk_robot-main/examples/data_collection/lerobot_data \
--dataset.episode=0
```



#### 推理代码
```sh
# 朴素版本
python /home/yichu/yichu_work/lerobot/src/lerobot/scripts/inference_joint_xsquare.py \
    --robot.type=x_robot \
    --robot.server="192.168.36.116:50051" \
    --policy.device=cuda \
    --policy.path=/home/yichu/yichu_work/models/act_pull_the_door_0430000\
    --task="pull the door" \
    --robot.max_relative_target=0.05


# 平滑版本
 python /home/yichu/yichu_work/lerobot/src/lerobot/scripts/inference_joint_xaquare_smooth.py  \
    --robot.type=x_robot   \
    --robot.server="192.168.36.116:50051"  \
    --policy.device=cuda  \
    --policy.path=/home/yichu/yichu_work/models/act_pull_the_door_0430000  \
    --task="pull the door"   \
    --robot.max_relative_target=0.05
```

#### 归零操作


这个地方是有点小问题，你移动之后最好看下，是不是目标位置，然后再移动一次

查看腰部具体在那个位置：
```sh
python lift_control.py --action stream --server 192.168.36.116:50051
```

身体上下移动到固定位置：
```sh
python /home/yichu/yichu_work/sdk_robot-main/examples/quanta_x1/lift_control_move.py --action move --target 0.31 --server 192.168.36.116:50051
```

机械臂归零
```sh
python /home/yichu/yichu_work/sdk_robot-main/examples/arm_control_latest.py --server 192.168.36.116:50051
```




```sh
python3 /home/yichu/yichu_work/data_collection/convert_to_lerobot_async.py \
--input-dir /home/yichu/yichu_work/datasets/sdk/task_7_0001 \
--output-dir /home/yichu/yichu_work/datasets/sdk_to_lerobot/task_7_0001 \
--repo-id "my_robot/dataset" \
--robot-type "quanta_x1" \
--use-videos \
--processes 4
```





(xr_lerobot) yichu@spark-2df2:~$ python3 /home/yichu/yichu_work/data_collection/convert_to_lerobot_async.py \                                                                       --input-dir /home/yichu/yichu_work/datasets/sdk/task_3_0001 \
--output-dir /home/yichu/yichu_work/datasets/sdk_to_lerobot/task_3_0001 \
--repo-id "my_robot/dataset" \
--robot-type "quanta_x1" \
--use-videos