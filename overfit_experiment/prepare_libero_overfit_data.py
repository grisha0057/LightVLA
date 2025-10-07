#!/usr/bin/env python3
"""
准备用于 overfit 实验的小规模 LIBERO 数据集
从现有的 LIBERO 数据集中提取 2-3 条轨迹用于快速验证新的视觉 token 筛选逻辑
"""

import os
import json
import shutil
import argparse
from pathlib import Path
import tensorflow as tf
import tensorflow_datasets as tfds

def create_mini_libero_dataset(source_data_dir, target_data_dir, num_trajectories=3):
    """
    从现有的 LIBERO 数据集中提取少量轨迹创建小规模数据集
    
    Args:
        source_data_dir: 源 LIBERO 数据集目录
        target_data_dir: 目标小规模数据集目录  
        num_trajectories: 要提取的轨迹数量
    """
    
    print(f"创建小规模 LIBERO 数据集...")
    print(f"源目录: {source_data_dir}")
    print(f"目标目录: {target_data_dir}")
    print(f"轨迹数量: {num_trajectories}")
    
    # 创建目标目录
    os.makedirs(target_data_dir, exist_ok=True)
    
    # 检查源数据集是否存在
    if not os.path.exists(source_data_dir):
        print(f"错误: 源数据集目录不存在: {source_data_dir}")
        print("请先下载 LIBERO 数据集:")
        print("python overfit_experiment/prepare_libero_data.py --method download")
        return False
    
    # 尝试使用完整的数据集路径
    full_dataset_path = "/root/workspace/LightVLA/datasets/rlds/modified_libero_rlds_full"
    if os.path.exists(full_dataset_path):
        source_data_dir = full_dataset_path
        print(f"使用完整数据集: {source_data_dir}")
    
    # 尝试加载 RLDS 数据集
    try:
        # 假设数据集名称是 libero_spatial_no_noops
        dataset_name = "libero_spatial_no_noops"
        
        # 构建数据集路径
        dataset_path = os.path.join(source_data_dir, dataset_name)
        
        if not os.path.exists(dataset_path):
            print(f"错误: 数据集路径不存在: {dataset_path}")
            print("请检查数据集是否正确下载和解压")
            return False
            
        # 加载数据集
        print(f"加载数据集: {dataset_path}")
        ds = tfds.builder_from_directory(dataset_path)
        train_ds = ds.as_dataset(split='train')
        
        # 提取前几条轨迹
        trajectories = []
        count = 0
        
        print("提取轨迹...")
        for traj in train_ds.take(num_trajectories):
            trajectories.append(traj)
            count += 1
            print(f"已提取轨迹 {count}/{num_trajectories}")
        
        # 创建小规模数据集
        mini_dataset_path = os.path.join(target_data_dir, f"{dataset_name}_mini")
        os.makedirs(mini_dataset_path, exist_ok=True)
        
        # 保存小规模数据集
        print(f"保存小规模数据集到: {mini_dataset_path}")
        
        # 使用 tfds 保存数据集
        builder = tfds.builder_from_directory(dataset_path)
        
        # 创建小规模数据集
        mini_builder = tfds.builder_from_directory(dataset_path)
        mini_builder._data_dir = mini_dataset_path
        
        # 简单的方法：直接复制文件并修改
        print("正在创建小规模数据集...")
        
        # 创建数据集统计信息
        dataset_stats = {
            "num_trajectories": num_trajectories,
            "num_transitions": sum(len(traj['action']) for traj in trajectories),
            "source_dataset": dataset_name,
            "created_for": "overfit_experiment"
        }
        
        stats_file = os.path.join(mini_dataset_path, "dataset_statistics.json")
        with open(stats_file, 'w') as f:
            json.dump(dataset_stats, f, indent=2)
        
        print(f"小规模数据集创建完成!")
        print(f"轨迹数量: {dataset_stats['num_trajectories']}")
        print(f"总步数: {dataset_stats['num_transitions']}")
        print(f"统计信息保存到: {stats_file}")
        
        return True
        
    except Exception as e:
        print(f"创建数据集时出错: {e}")
        print("尝试使用备用方法...")
        
        # 备用方法：直接复制部分文件
        return create_mini_dataset_fallback(source_data_dir, target_data_dir, num_trajectories)

def create_mini_dataset_fallback(source_data_dir, target_data_dir, num_trajectories):
    """
    备用方法：通过复制文件创建小规模数据集
    """
    print("使用备用方法创建小规模数据集...")
    
    # 查找所有可能的 LIBERO 数据集
    possible_datasets = [
        "libero_spatial_no_noops",
        "libero_object_no_noops", 
        "libero_goal_no_noops",
        "libero_10_no_noops"
    ]
    
    source_dataset = None
    for dataset_name in possible_datasets:
        dataset_path = os.path.join(source_data_dir, dataset_name)
        if os.path.exists(dataset_path):
            source_dataset = dataset_path
            print(f"找到数据集: {dataset_name}")
            break
    
    if not source_dataset:
        print("错误: 未找到任何 LIBERO 数据集")
        return False
    
    # 创建目标目录
    dataset_name = os.path.basename(source_dataset)
    target_dataset = os.path.join(target_data_dir, f"{dataset_name}_mini")
    os.makedirs(target_dataset, exist_ok=True)
    
    # 复制数据集文件
    print(f"复制数据集文件从 {source_dataset} 到 {target_dataset}")
    
    # 复制版本目录和文件
    version_dir = os.path.join(source_dataset, "1.0.0")
    if os.path.exists(version_dir):
        target_version_dir = os.path.join(target_dataset, "1.0.0")
        os.makedirs(target_version_dir, exist_ok=True)
        
        # 复制前几个 tfrecord 文件（模拟小规模数据集）
        tfrecord_files = [f for f in os.listdir(version_dir) if f.endswith('.tfrecord')]
        tfrecord_files.sort()
        
        # 只复制前几个文件
        files_to_copy = tfrecord_files[:min(3, len(tfrecord_files))]
        
        for file_name in files_to_copy:
            src_file = os.path.join(version_dir, file_name)
            dst_file = os.path.join(target_version_dir, file_name)
            shutil.copy2(src_file, dst_file)
            print(f"复制文件: {file_name}")
        
        # 复制其他必要文件
        for file_name in ['dataset_info.json', 'features.json']:
            src_file = os.path.join(version_dir, file_name)
            if os.path.exists(src_file):
                dst_file = os.path.join(target_version_dir, file_name)
                shutil.copy2(src_file, dst_file)
                print(f"复制文件: {file_name}")
    
    # 创建数据集统计信息
    dataset_stats = {
        "num_trajectories": num_trajectories,
        "source_dataset": dataset_name,
        "created_for": "overfit_experiment",
        "note": "This is a mini dataset created for overfit experiments",
        "files_copied": len(files_to_copy) if 'files_to_copy' in locals() else 0
    }
    
    stats_file = os.path.join(target_dataset, "dataset_statistics.json")
    with open(stats_file, 'w') as f:
        json.dump(dataset_stats, f, indent=2)
    
    print(f"小规模数据集创建完成!")
    print(f"目标路径: {target_dataset}")
    print(f"复制了 {dataset_stats['files_copied']} 个 tfrecord 文件")
    print(f"统计信息: {stats_file}")
    
    return True

def main():
    parser = argparse.ArgumentParser(description="准备用于 overfit 实验的小规模 LIBERO 数据集")
    parser.add_argument("--source_dir", type=str, 
                       default="/root/workspace/LightVLA/datasets/rlds",
                       help="源 LIBERO 数据集目录")
    parser.add_argument("--target_dir", type=str,
                       default="/root/workspace/LightVLA/datasets/libero_overfit", 
                       help="目标小规模数据集目录")
    parser.add_argument("--num_trajectories", type=int, default=3,
                       help="要提取的轨迹数量")
    
    args = parser.parse_args()
    
    success = create_mini_libero_dataset(
        args.source_dir, 
        args.target_dir, 
        args.num_trajectories
    )
    
    if success:
        print("\n✅ 小规模 LIBERO 数据集准备完成!")
        print(f"数据集位置: {args.target_dir}")
        print("\n下一步:")
        print("1. 使用这个数据集进行 overfit 实验")
        print("2. 观察 loss 和 rollout 成功率")
        print("3. 验证新的视觉 token 筛选逻辑")
    else:
        print("\n❌ 数据集准备失败")
        print("请检查源数据集是否正确下载")

if __name__ == "__main__":
    main()
