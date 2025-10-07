# OpenVLA-OFT Overfit 实验指南

## 实验目标
验证"新的视觉 token 筛选逻辑能否顺利训练"，通过在原 OpenVLA-OFT 微调脚本上跑一个极小规模的 overfit 实验：
- 从同一个 checkpoint 起点
- 用几条 LIBERO 轨迹 finetune 几百步
- 观察 loss 和 rollout 是否能迅速贴近 100% 成功率

## 快速开始

### 1. 准备数据
```bash
# 使用测试数据（快速验证）
python overfit_experiment/prepare_libero_data.py --method test

# 或下载真实 LIBERO 数据集
python overfit_experiment/prepare_libero_data.py --method download
```

### 2. 运行实验
```bash
bash overfit_experiment/run_overfit_experiment.sh
```

### 3. 分析结果
```bash
python overfit_experiment/analyze_overfit_results.py
```

## 实验准备

### 环境检查
确保以下环境已准备就绪：
- ✅ OpenVLA checkpoint: `/root/workspace/LightVLA/checkpoints/openvla-libero-spatial`
- ✅ 训练脚本: `vla-scripts/finetune.py`
- ✅ 实验脚本已创建

### 数据准备方式

#### 方式 A: 下载真实 LIBERO 数据集
```bash
# 下载完整的 LIBERO 数据集
python overfit_experiment/prepare_libero_data.py --method download

# 提取小规模数据集用于 overfit
python overfit_experiment/prepare_libero_overfit_data.py
```

#### 方式 B: 使用测试数据（快速验证）
```bash
# 创建测试数据用于验证训练流程
python overfit_experiment/prepare_libero_data.py --method test
```

## 实验配置

### 训练参数
- **模型**: openvla-libero-spatial checkpoint
- **数据**: 2-3 条 LIBERO 轨迹
- **训练步数**: 500 步
- **批次大小**: 1 (单 GPU)
- **学习率**: 1e-3 (较高，用于快速 overfit)
- **LoRA rank**: 16 (较小，快速适应)

### 预期结果
如果新的视觉 token 筛选逻辑有效，应该观察到：
1. **Loss 快速下降**: 在几百步内 loss 显著降低
2. **Rollout 成功率提升**: 在训练过程中成功率迅速接近 100%
3. **训练稳定性**: 没有出现 loss 震荡或发散

## 文件说明

### 核心脚本
- `prepare_libero_data.py`: 数据准备脚本
- `prepare_libero_overfit_data.py`: 小规模数据集提取脚本
- `run_overfit_experiment.sh`: overfit 训练脚本
- `analyze_overfit_results.py`: 结果分析脚本
- `setup_overfit_experiment.py`: 实验设置脚本

### 目录结构
```
/root/workspace/LightVLA/
├── overfit_experiment/          # overfit 实验脚本目录
│   ├── prepare_libero_data.py
│   ├── prepare_libero_overfit_data.py
│   ├── run_overfit_experiment.sh
│   ├── analyze_overfit_results.py
│   ├── setup_overfit_experiment.py
│   └── README.md               # 本说明文件
├── datasets/
│   ├── rlds/                    # 原始 LIBERO 数据集
│   └── libero_overfit/          # 小规模 overfit 数据集
├── logs/
│   └── overfit_experiment/      # 实验日志和 checkpoint
└── checkpoints/
    └── openvla-libero-spatial/  # 预训练模型
```

## 故障排除

### 常见问题
1. **数据集下载失败**
   - 检查网络连接
   - 尝试使用 `--method test` 创建测试数据

2. **训练 OOM 错误**
   - 减少 batch_size 到 1
   - 减少 LoRA rank
   - 使用更少的轨迹

3. **模型加载失败**
   - 检查 checkpoint 路径
   - 确认模型文件完整性

### 调试建议
- 使用 TensorBoard 监控训练过程
- 检查日志文件中的错误信息
- 验证数据加载是否正常

## 实验步骤详解

### 1. 数据准备
```bash
# 下载 LIBERO 数据集
python overfit_experiment/prepare_libero_data.py --method download

# 提取小规模数据集
python overfit_experiment/prepare_libero_overfit_data.py
```

### 2. 运行 Overfit 实验
```bash
# 启动 overfit 训练
bash overfit_experiment/run_overfit_experiment.sh
```

### 3. 分析实验结果
```bash
# 分析训练结果
python overfit_experiment/analyze_overfit_results.py

# 查看训练曲线
tensorboard --logdir /root/workspace/LightVLA/logs/overfit_experiment
```

## 下一步
实验完成后，根据结果决定：
- ✅ **成功**: 新的视觉 token 筛选逻辑有效，可以继续大规模训练
- ❌ **失败**: 需要调整筛选逻辑或训练参数
- ⚠️ **部分成功**: 需要进一步分析和优化