#!/usr/bin/env bash

# Shared paths for the task_45 ACT reproduction project.
PROJECT_DIR="/home/yichu/shixisheng/zzm/act_task45"
LEROBOT_REPO="/home/yichu/yichu_work/lerobot"
RAW_DATASET="/home/yichu/yichu_work/datasets/sdk_dongguan/task_45_0001"
CONVERTED_DATASET="$PROJECT_DIR/datasets/task_45_raw_merged"
OUTPUT_ROOT="$PROJECT_DIR/outputs"
DATASET_REPO_ID="dongguan/task_45_raw_merged"
RAW_BASE="/home/yichu/yichu_work/datasets/sdk_dongguan"
GROUP_LIST="$PROJECT_DIR/dataset_groups.txt"
CONVERTED_BASE="$PROJECT_DIR/datasets/converted"
MERGED_DATASET="$PROJECT_DIR/datasets/task_45_raw_merged"
MERGED_REPO_ID="dongguan/task_45_raw_merged"
CONDA_ENV="act_train"
CONDA_SH="/home/yichu/miniconda3/etc/profile.d/conda.sh"

export PROJECT_DIR LEROBOT_REPO RAW_DATASET CONVERTED_DATASET OUTPUT_ROOT
export DATASET_REPO_ID CONDA_ENV CONDA_SH
export RAW_BASE GROUP_LIST CONVERTED_BASE MERGED_DATASET MERGED_REPO_ID
