#!/usr/bin/env python3
"""
简化的 LIBERO 数据准备脚本
用于 overfit 实验的快速数据准备
"""

import os
import subprocess
import argparse
from pathlib import Path

def download_libero_dataset():
    """下载 LIBERO 数据集"""
    
    print("📥 下载 LIBERO 数据集...")
    
    # 数据集下载命令
    download_cmd = [
        "git", "clone", 
        "https://huggingface.co/datasets/openvla/modified_libero_rlds",
        "/root/workspace/LightVLA/datasets/rlds/modified_libero_rlds"
    ]
    
    try:
        print("执行下载命令...")
        result = subprocess.run(download_cmd, check=True, capture_output=True, text=True)
        print("✅ LIBERO 数据集下载完成!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ 下载失败: {e}")
        print("尝试使用备用方法...")
        return download_with_huggingface_hub()

def download_with_huggingface_hub():
    """使用 huggingface_hub 下载数据集"""
    
    try:
        from huggingface_hub import snapshot_download
        
        print("使用 huggingface_hub 下载...")
        snapshot_download(
            repo_id="openvla/modified_libero_rlds",
            repo_type="dataset",
            local_dir="/root/workspace/LightVLA/datasets/rlds/modified_libero_rlds"
        )
        print("✅ 数据集下载完成!")
        return True
        
    except Exception as e:
        print(f"❌ huggingface_hub 下载失败: {e}")
        return False

def create_simple_test_data():
    """创建简单的测试数据用于验证训练流程"""
    
    print("🔧 创建简单的测试数据...")
    
    test_data_dir = "/root/workspace/LightVLA/datasets/libero_overfit"
    os.makedirs(test_data_dir, exist_ok=True)
    
    # 创建简单的数据集配置
    dataset_config = {
        "dataset_name": "libero_spatial_no_noops_mini",
        "description": "Mini LIBERO dataset for overfit experiments",
        "num_trajectories": 3,
        "num_transitions": 150,
        "created_for": "overfit_experiment",
        "note": "This is a placeholder dataset for testing the training pipeline"
    }
    
    config_file = os.path.join(test_data_dir, "dataset_config.json")
    with open(config_file, 'w') as f:
        import json
        json.dump(dataset_config, f, indent=2)
    
    print(f"✅ 测试数据配置创建完成: {config_file}")
    print("注意: 这是用于测试训练流程的占位数据")
    print("实际训练需要真实的 LIBERO 数据集")
    
    return True

def main():
    parser = argparse.ArgumentParser(description="准备 LIBERO 数据用于 overfit 实验")
    parser.add_argument("--method", choices=["download", "test"], default="download",
                       help="数据准备方法: download (下载真实数据) 或 test (创建测试数据)")
    
    args = parser.parse_args()
    
    print("🚀 开始准备 LIBERO 数据...")
    
    if args.method == "download":
        # 尝试下载真实数据集
        success = download_libero_dataset()
        
        if success:
            print("\\n✅ LIBERO 数据集准备完成!")
            print("数据集位置: /root/workspace/LightVLA/datasets/rlds/modified_libero_rlds")
            print("\\n下一步:")
            print("1. 运行数据提取脚本:")
            print("   python overfit_experiment/prepare_libero_overfit_data.py")
            print("2. 开始 overfit 实验:")
            print("   bash overfit_experiment/run_overfit_experiment.sh")
        else:
            print("\\n❌ 数据集下载失败")
            print("建议:")
            print("1. 检查网络连接")
            print("2. 手动下载数据集")
            print("3. 或使用测试数据模式: python prepare_libero_data.py --method test")
    
    else:  # test mode
        create_simple_test_data()
        print("\\n✅ 测试数据准备完成!")
        print("\\n注意: 这只是用于测试训练流程的占位数据")
        print("要获得真实的训练效果，需要下载完整的 LIBERO 数据集")

if __name__ == "__main__":
    main()
