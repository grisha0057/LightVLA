#!/bin/bash

# OpenVLA-OFT overfit 实验
# 验证新的视觉 token 筛选逻辑能否顺利训练

set -e
set -o pipefail

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

# 修复容器主机名解析问题：若 hostname 无法在 /etc/hosts 中解析，会导致 c10d 反复失败
HN=$(hostname)
if ! grep -q "\b${HN}\b" /etc/hosts 2>/dev/null; then
  echo "🧩 /etc/hosts 未包含主机名 ${HN}，尝试写入 127.0.0.1 映射以避免分布式初始化出错。"
  {
    echo "127.0.0.1 ${HN}"
  } >> /etc/hosts 2>/dev/null || echo "⚠️ 无法写入 /etc/hosts（可能缺少权限），请手动添加: '127.0.0.1 ${HN}'"
fi

# 分布式/网络环境变量（IPv4-only，loopback）
if [ -z "${MASTER_ADDR:-}" ]; then
  if [ "${IFACE}" != "lo" ] && command -v ip >/dev/null 2>&1; then
    IP4=$(ip -4 addr show dev "${IFACE}" | awk '/inet /{print $2}' | cut -d/ -f1 | head -n1)
    export MASTER_ADDR=${IP4:-127.0.0.1}
  else
    export MASTER_ADDR=127.0.0.1
  fi
fi
export MASTER_PORT=${MASTER_PORT:-12355}
# 可通过 OVERFIT_IFACE=eth0 指定网卡，默认回环 lo
IFACE=${OVERFIT_IFACE:-lo}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-${IFACE}}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}
export NCCL_SOCKET_FAMILY=${NCCL_SOCKET_FAMILY:-AF_INET}
export NCCL_ASYNC_ERROR_HANDLING=${NCCL_ASYNC_ERROR_HANDLING:-1}
export NCCL_BLOCKING_WAIT=${NCCL_BLOCKING_WAIT:-1}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-lo}
export GLOO_DISABLE_IPV6=${GLOO_DISABLE_IPV6:-1}
export GLOO_DEVICE_TRANSPORT=${GLOO_DEVICE_TRANSPORT:-TCP}
export TORCH_DISTRIBUTED_DEBUG=${TORCH_DISTRIBUTED_DEBUG:-DETAIL}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}

# GPU 进程数控制：设置 OVERFIT_SINGLE_GPU=1 可单卡调试
if [ "${OVERFIT_SINGLE_GPU:-0}" = "1" ]; then
  export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
  NPROC=1
else
  export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
  NPROC=2
fi

LOG_FILE=${RUN_ROOT_DIR}/train_$(date +%Y%m%d_%H%M%S).log

LR=${OVERFIT_LR:-1e-4}
MAX_STEPS=${OVERFIT_MAX_STEPS:-1000}
SAVE_FREQ=${OVERFIT_SAVE_FREQ:-1000}
MILESTONES=${OVERFIT_DECAY_MILESTONES:-"[100000]"}
GAMMA=${OVERFIT_DECAY_GAMMA:-0.5}
WARMUP_STEPS=${OVERFIT_WARMUP_STEPS:-1000}
COVERAGE_WARMUP=${OVERFIT_COVERAGE_WARMUP:-1.0}
COVERAGE_TARGET=${OVERFIT_COVERAGE_TARGET:-1.0}

# 新增剪枝超参（可通过环境变量覆盖）
PRUNE_AGGREGATION=${PRUNE_AGGREGATION:-"logsumexp"}
PRUNE_LSE_TEMP=${PRUNE_LSE_TEMP:-1.0}
PRUNE_RESCALE=${PRUNE_RESCALE:-True}
PRUNE_CLIP=${PRUNE_CLIP:-10.0}

if [ "${NPROC}" = "1" ]; then
  echo "🔧 单卡模式：跳过 torchrun，直接运行 Python 以避免分布式初始化。"
  stdbuf -oL -eL python -u vla-scripts/finetune.py \
    --vla_path "${VLA_PATH}" \
    --data_root_dir "${DATA_ROOT_DIR}" \
    --dataset_name "${DATASET_NAME}" \
    --use_l1_regression True \
    --use_diffusion False \
    --use_film False \
    --num_images_in_input 2 \
    --use_proprio True \
    --batch_size 1 \
    --learning_rate ${LR} \
    --lr_warmup_steps ${WARMUP_STEPS} \
    --lr_decay_milestones ${MILESTONES} \
    --lr_decay_gamma ${GAMMA} \
    --prune_coverage_warmup ${COVERAGE_WARMUP} \
    --prune_coverage_target ${COVERAGE_TARGET} \
    --prune_disable True \
    --prune_prompt_aggregation ${PRUNE_AGGREGATION} \
    --prune_logsumexp_temperature ${PRUNE_LSE_TEMP} \
    --prune_soft_rescale_mean_preserve ${PRUNE_RESCALE} \
    --prune_soft_rescale_clip ${PRUNE_CLIP} \
    --grad_accumulation_steps 16 \
    --max_steps ${MAX_STEPS} \
    --save_freq ${SAVE_FREQ} \
    --save_latest_checkpoint_only False \
    --image_aug False \
    --lora_rank 8 \
    --run_root_dir "${RUN_ROOT_DIR}" \
    --shuffle_buffer_size 1 \
    --log_freq 20 2>&1 | tee -a ${LOG_FILE}
else
  # 多卡模式：恢复 --standalone，减少不必要的名字解析与外部依赖
  stdbuf -oL -eL torchrun \
    --standalone \
    --nnodes 1 \
    --nproc-per-node ${NPROC} \
    --max-restarts 0 \
    vla-scripts/finetune.py \
  --vla_path "${VLA_PATH}" \
  --data_root_dir "${DATA_ROOT_DIR}" \
  --dataset_name "${DATASET_NAME}" \
  --use_l1_regression True \
  --use_diffusion False \
  --use_film False \
  --num_images_in_input 2 \
  --use_proprio True \
  --batch_size 1 \
  --learning_rate ${LR} \
  --lr_warmup_steps ${WARMUP_STEPS} \
  --lr_decay_milestones ${MILESTONES} \
  --lr_decay_gamma ${GAMMA} \
  --prune_coverage_warmup ${COVERAGE_WARMUP} \
  --prune_coverage_target ${COVERAGE_TARGET} \
  --prune_disable True \
  --prune_prompt_aggregation ${PRUNE_AGGREGATION} \
  --prune_logsumexp_temperature ${PRUNE_LSE_TEMP} \
  --prune_soft_rescale_mean_preserve ${PRUNE_RESCALE} \
  --prune_soft_rescale_clip ${PRUNE_CLIP} \
  --grad_accumulation_steps 16 \
  --max_steps ${MAX_STEPS} \
  --save_freq ${SAVE_FREQ} \
  --save_latest_checkpoint_only False \
  --image_aug False \
  --lora_rank 8 \
  --run_root_dir "${RUN_ROOT_DIR}" \
  --shuffle_buffer_size 1 \
  --log_freq 20 2>&1 | tee -a ${LOG_FILE}
fi

echo "🎉 overfit 实验完成!"
echo "检查日志和 checkpoint: ${RUN_ROOT_DIR}"
