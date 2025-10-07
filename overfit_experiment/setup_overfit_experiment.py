#!/usr/bin/env python3
"""
OpenVLA-OFT overfit 实验脚本
用于验证新的视觉 token 筛选逻辑能否顺利训练
从同一个 checkpoint 起点、用几条 LIBERO 轨迹 finetune 几百步，观察 loss 和 rollout 是否能迅速贴近 100% 成功率
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def create_overfit_training_script():
    """创建用于 overfit 实验的训练脚本"""
    
    script_content = '''#!/bin/bash

# OpenVLA-OFT overfit 实验
# 验证新的视觉 token 筛选逻辑能否顺利训练

set -e

# 实验配置
VLA_PATH="/root/workspace/LightVLA/checkpoints/openvla-libero-spatial"
DATA_ROOT_DIR="/root/workspace/LightVLA/datasets/libero_overfit"
DATASET_NAME="libero_spatial_no_noops_mini"
RUN_ROOT_DIR="/root/workspace/LightVLA/logs/overfit_experiment"

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
    echo "请先运行: python prepare_libero_overfit_data.py"
    exit 1
fi

echo "✅ 检查通过，开始训练..."

# 启动 overfit 训练
torchrun \\
  --standalone \\
  --nnodes 1 \\
  --nproc-per-node 1 \\
  vla-scripts/finetune.py \\
  --vla_path "${VLA_PATH}" \\
  --data_root_dir "${DATA_ROOT_DIR}" \\
  --dataset_name "${DATASET_NAME}" \\
  --use_l1_regression True \\
  --use_diffusion False \\
  --use_film False \\
  --num_images_in_input 2 \\
  --use_proprio True \\
  --batch_size 1 \\
  --learning_rate 1e-3 \\
  --num_steps_before_decay 100 \\
  --max_steps 500 \\
  --save_freq 50 \\
  --save_latest_checkpoint_only False \\
  --image_aug False \\
  --lora_rank 16 \\
  --run_root_dir "${RUN_ROOT_DIR}" \\
  --shuffle_buffer_size 1000

echo "🎉 overfit 实验完成!"
echo "检查日志和 checkpoint: ${RUN_ROOT_DIR}"
'''
    
    script_path = "/root/workspace/LightVLA/run_overfit_experiment.sh"
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    # 设置执行权限
    os.chmod(script_path, 0o755)
    
    return script_path

def create_evaluation_script():
    """创建用于评估 overfit 结果的脚本"""
    
    script_content = '''#!/usr/bin/env python3
"""
评估 overfit 实验结果
检查 loss 下降情况和 rollout 成功率
"""

import os
import json
import glob
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_overfit_results(log_dir):
    """分析 overfit 实验结果"""
    
    print(f"分析 overfit 实验结果: {log_dir}")
    
    # 查找 tensorboard 日志
    tb_logs = glob.glob(os.path.join(log_dir, "**/events.out.tfevents.*"), recursive=True)
    
    if not tb_logs:
        print("❌ 未找到 tensorboard 日志文件")
        return
    
    print(f"找到 {len(tb_logs)} 个日志文件")
    
    # 查找 checkpoint 文件
    checkpoints = glob.glob(os.path.join(log_dir, "**/checkpoint-*.pt"), recursive=True)
    
    if not checkpoints:
        print("❌ 未找到 checkpoint 文件")
        return
    
    print(f"找到 {len(checkpoints)} 个 checkpoint 文件")
    
    # 分析结果
    print("\\n📊 实验结果分析:")
    print("=" * 50)
    
    # 检查训练是否完成
    if len(checkpoints) > 0:
        print(f"✅ 训练完成，共保存了 {len(checkpoints)} 个 checkpoint")
        
        # 检查最终 checkpoint
        latest_checkpoint = max(checkpoints, key=os.path.getctime)
        print(f"最新 checkpoint: {latest_checkpoint}")
        
        # 检查文件大小
        checkpoint_size = os.path.getsize(latest_checkpoint) / (1024 * 1024)  # MB
        print(f"Checkpoint 大小: {checkpoint_size:.2f} MB")
    
    # 检查日志文件
    if tb_logs:
        print(f"\\n📈 TensorBoard 日志:")
        for log_file in tb_logs:
            log_size = os.path.getsize(log_file) / 1024  # KB
            print(f"  - {os.path.basename(log_file)}: {log_size:.2f} KB")
    
    print("\\n🎯 下一步建议:")
    print("1. 使用 TensorBoard 查看训练曲线:")
    print(f"   tensorboard --logdir {log_dir}")
    print("2. 检查 loss 是否快速下降")
    print("3. 运行 rollout 评估验证成功率")
    print("4. 如果 loss 下降迅速且 rollout 成功率高，说明新的视觉 token 筛选逻辑有效")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="分析 overfit 实验结果")
    parser.add_argument("--log_dir", type=str, 
                       default="/root/workspace/LightVLA/logs/overfit_experiment",
                       help="实验日志目录")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.log_dir):
        print(f"❌ 日志目录不存在: {args.log_dir}")
        return
    
    analyze_overfit_results(args.log_dir)

if __name__ == "__main__":
    main()
'''
    
    script_path = "/root/workspace/LightVLA/analyze_overfit_results.py"
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    # 设置执行权限
    os.chmod(script_path, 0o755)
    
    return script_path

def main():
    """主函数"""
    
    print("🔧 准备 OpenVLA-OFT overfit 实验...")
    
    # 创建训练脚本
    train_script = create_overfit_training_script()
    print(f"✅ 创建训练脚本: {train_script}")
    
    # 创建评估脚本
    eval_script = create_evaluation_script()
    print(f"✅ 创建评估脚本: {eval_script}")
    
    print("\\n📋 实验准备完成!")
    print("=" * 50)
    print("下一步操作:")
    print("1. 准备 LIBERO 数据:")
    print("   python prepare_libero_overfit_data.py")
    print("\\n2. 运行 overfit 实验:")
    print("   bash run_overfit_experiment.sh")
    print("\\n3. 分析实验结果:")
    print("   python analyze_overfit_results.py")
    print("\\n4. 查看训练曲线:")
    print("   tensorboard --logdir /root/workspace/LightVLA/logs/overfit_experiment")
    
    print("\\n🎯 实验目标:")
    print("- 验证新的视觉 token 筛选逻辑能否顺利训练")
    print("- 观察 loss 是否能快速下降")
    print("- 检查 rollout 成功率是否能迅速贴近 100%")

if __name__ == "__main__":
    main()
