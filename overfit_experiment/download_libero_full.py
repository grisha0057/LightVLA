#!/usr/bin/env python3
"""
使用 huggingface_hub 下载完整的 LIBERO 数据集
"""

import os
from huggingface_hub import snapshot_download

def download_libero_dataset():
    """下载完整的 LIBERO 数据集"""
    
    print("📥 使用 huggingface_hub 下载完整的 LIBERO 数据集...")
    
    try:
        # 下载数据集
        local_dir = "/root/workspace/LightVLA/datasets/rlds/modified_libero_rlds_full"
        
        print(f"下载到: {local_dir}")
        
        snapshot_download(
            repo_id="openvla/modified_libero_rlds",
            repo_type="dataset",
            local_dir=local_dir,
            local_dir_use_symlinks=False
        )
        
        print("✅ LIBERO 数据集下载完成!")
        
        # 检查下载的数据大小
        import subprocess
        result = subprocess.run(['du', '-sh', local_dir], capture_output=True, text=True)
        print(f"数据集大小: {result.stdout.strip()}")
        
        return True
        
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        return False

if __name__ == "__main__":
    success = download_libero_dataset()
    
    if success:
        print("\n✅ 数据集下载完成!")
        print("下一步:")
        print("1. 提取小规模数据集:")
        print("   python overfit_experiment/prepare_libero_overfit_data.py")
        print("2. 开始 overfit 实验:")
        print("   bash overfit_experiment/run_overfit_experiment.sh")
    else:
        print("\n❌ 数据集下载失败")
