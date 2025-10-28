#!/bin/bash

# 评测 Step 50 Checkpoint (最新训练)

set -e

echo "============================================"
echo "🎮 评测 Step 50 Checkpoint"
echo "============================================"
echo ""

# 激活 conda 环境
source /usr/local/miniconda3/etc/profile.d/conda.sh
conda activate openvla-oft
echo "✅ 已激活 conda 环境: openvla-oft"

# 渲染配置
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
echo "✅ 使用 OSMesa 软件渲染"

# Checkpoint 路径（使用最新训练的 Step 50）
CHECKPOINT_PATH="/root/workspace/LightVLA/logs/libero_spatial_training/libero_spatial_20251027_093955/openvla-libero-spatial+libero_spatial_no_noops+b16+lr-0.0001+lora-r8+dropout-0.02025-10-27 09:40:23.659495--50_chkpt"
# 验证checkpoint存在
if [ ! -d "${CHECKPOINT_PATH}" ]; then
    echo "❌ 错误: Checkpoint 不存在: ${CHECKPOINT_PATH}"
    exit 1
fi

echo "📦 Checkpoint: ${CHECKPOINT_PATH}"
echo "📊 Checkpoint 大小: $(du -sh "${CHECKPOINT_PATH}" | cut -f1)"
echo ""

# 评估配置
EVAL_GPUS="0,1"          # 使用2个GPU评测（当前系统只有2个GPU）
NUM_TRIALS=4             # 每个任务4次试验
LORA_RANK=8              # LoRA rank
# 注意：Coverage将使用checkpoint中保存的config.json配置（prune_target_coverage=0.95）

# 输出目录
OUTPUT_DIR="/root/workspace/LightVLA/logs/libero_spatial_training/libero_spatial_20251027_085014/eval_logs"
mkdir -p "${OUTPUT_DIR}"

echo "⚙️  评测配置："
echo "  - GPU: ${EVAL_GPUS}"
echo "  - 每任务试验次数: ${NUM_TRIALS}"
echo "  - LoRA Rank: ${LORA_RANK}"
echo "  - Coverage: 使用checkpoint的config.json配置"
echo "  - 日志目录: ${OUTPUT_DIR}"
echo ""

# 设置GPU
export CUDA_VISIBLE_DEVICES=${EVAL_GPUS}

cd /root/workspace/LightVLA

echo "🚀 开始评测..."
echo "============================================"
echo ""

# 运行评测
python -u experiments/robot/libero/run_libero_eval.py \
    --pretrained_checkpoint "${CHECKPOINT_PATH}" \
    --task_suite_name "libero_spatial" \
    --use_l1_regression True \
    --use_diffusion False \
    --use_film False \
    --num_images_in_input 2 \
    --use_proprio True \
    --lora_rank ${LORA_RANK} \
    --center_crop False \
    --num_trials_per_task ${NUM_TRIALS} \
    --run_id_note "step_50_eval" \
    --local_log_dir "${OUTPUT_DIR}" \
    --save_rollout_video False \
    --seed 7 2>&1 | tee "${OUTPUT_DIR}/eval_step50_$(date +%Y%m%d_%H%M%S).log"

EVAL_EXIT_CODE=$?

echo ""
echo "============================================"
if [ ${EVAL_EXIT_CODE} -eq 0 ]; then
    echo "✅ 评测完成！"
else
    echo "❌ 评测失败 (exit code: ${EVAL_EXIT_CODE})"
fi
echo "============================================"
echo ""

# 显示结果摘要
echo "📊 结果摘要："
echo "============================================"
LATEST_LOG=$(ls -t "${OUTPUT_DIR}"/eval_step50_*.log 2>/dev/null | head -1)
if [ -f "${LATEST_LOG}" ]; then
    echo "最新日志: ${LATEST_LOG}"
    echo ""
    echo "成功率统计:"
    grep "Overall success rate" "${LATEST_LOG}" || echo "未找到成功率统计"
    echo ""
    echo "各任务详细结果:"
    grep "Task " "${LATEST_LOG}" | grep "success rate" || echo "未找到任务详情"
else
    echo "未找到日志文件"
fi
echo "============================================"

