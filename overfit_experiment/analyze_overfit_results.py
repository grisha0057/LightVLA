#!/usr/bin/env python3
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
    print("\n📊 实验结果分析:")
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
        print(f"\n📈 TensorBoard 日志:")
        for log_file in tb_logs:
            log_size = os.path.getsize(log_file) / 1024  # KB
            print(f"  - {os.path.basename(log_file)}: {log_size:.2f} KB")
    
    print("\n🎯 下一步建议:")
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
