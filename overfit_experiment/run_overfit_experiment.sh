#!/bin/bash

# OpenVLA-OFT overfit 实验
# 验证新的视觉 token 筛选逻辑能否顺利训练

set -e

# 实验配置
VLA_PATH="/root/workspace/LightVLA/checkpoints/openvla-libero-spatial"
DATA_ROOT_DIR="/root/workspace/LightVLA/datasets/libero_overfit"
DATASET_NAME="libero_spatial_no_noops_mini"
RUN_ROOT_DIR="/root/workspace/LightVLA/logs/overfit_experiment"
SCRIPT_DIR="/root/workspace/LightVLA/overfit_experiment"

# 创建日志目录
mkdir -p ${RUN_ROOT_DIR}

echo "🚀 开始 OpenVLA-OFT overfit 实验..."
echo "模型路径: ${VLA_PATH}"
echo "数据路径: ${DATA_ROOT_DIR}"
echo "数据集名称: ${DATASET_NAME}"
echo "日志目录: ${RUN_ROOT_DIR}"

# 检查模型是否存在
if [ ! -d "${VLA_PATH}" ]; then
    echo "❌ 错误: 模型路径不存在: ${VLA_PATH}"
    echo "请确保 openvla-libero-spatial checkpoint 存在"
    exit 1
fi

# 检查数据是否存在
if [ ! -d "${DATA_ROOT_DIR}" ]; then
    echo "❌ 错误: 数据路径不存在: ${DATA_ROOT_DIR}"
    echo "请先运行: python ${SCRIPT_DIR}/prepare_libero_overfit_data.py"
    exit 1
fi

echo "✅ 检查通过，开始训练..."

# 启动 overfit 训练
cd /root/workspace/LightVLA

# 设置多 GPU 环境变量
export CUDA_VISIBLE_DEVICES=0,1
export NCCL_SOCKET_IFNAME=lo
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=12355

torchrun \
  --standalone \
  --nnodes 1 \
  --nproc-per-node 2 \
  vla-scripts/finetune.py \
  --vla_path "${VLA_PATH}" \
  --data_root_dir "${DATA_ROOT_DIR}" \
  --dataset_name "${DATASET_NAME}" \
  --use_l1_regression True \
  --use_diffusion False \
  --use_film False \
  --num_images_in_input 2 \
  --use_proprio True \
  --batch_size 2 \
  --learning_rate 1e-3 \
  --num_steps_before_decay 100 \
  --max_steps 500 \
  --save_freq 50 \
  --save_latest_checkpoint_only False \
  --image_aug False \
  --lora_rank 16 \
  --run_root_dir "${RUN_ROOT_DIR}" \
  --shuffle_buffer_size 1000

echo "🎉 overfit 实验完成!"
echo "检查日志和 checkpoint: ${RUN_ROOT_DIR}"
