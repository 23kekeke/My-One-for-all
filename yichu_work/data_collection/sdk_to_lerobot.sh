for i in 2 3 4 5 6 7 8; do
    python3 /home/yichu/yichu_work/data_collection/convert_to_lerobot_async.py \
        --input-dir /home/yichu/yichu_work/datasets/sdk/pull_the_door_${i} \
        --output-dir /home/yichu/yichu_work/datasets/sdk_to_lerobot/pull_the_door_${i} \
        --repo-id "my_robot/dataset" \
        --robot-type "quanta_x1" \
        --use-videos \
        --processes 4
done
