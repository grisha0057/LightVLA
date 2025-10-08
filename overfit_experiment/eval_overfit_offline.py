#!/usr/bin/env python3
"""
离线评估 overfit 检查点在 mini 数据集上的逐样本表现：
- 载入合并后的 VLA、`action_head`, `proprio_projector`
- 复用与训练一致的数据管线与前向逻辑
- 计算每样本 L1 与阈值命中率，输出总体均值与准确率
"""

import argparse
import os
from pathlib import Path
from typing import List

import torch

# 重要：禁用 TensorFlow 使用 GPU，避免与 PyTorch 竞争显存
try:
    import tensorflow as tf  # noqa: F401
    try:
        tf.config.set_visible_devices([], 'GPU')
    except Exception:
        pass
except Exception:
    pass

# 依赖 LightVLA 内部模块
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

from prismatic.models.action_heads import L1RegressionActionHead
from prismatic.models.projectors import ProprioProjector

from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK

from torch.utils.data import DataLoader

# 复用训练脚本中的前向与掩码计算（以确保一致性）
from prismatic.training.train_utils import (
    get_current_action_mask,
    get_next_actions_mask,
)

import torch.nn as nn


def eval_forward_pass_l1(
    vla,
    action_head,
    proprio_projector,
    batch,
    device,
    use_proprio: bool,
    num_patches: int,
):
    metrics = {}
    ground_truth_actions = batch["actions"].to(device).to(torch.bfloat16)

    with torch.autocast("cuda" if device.type == 'cuda' else "cpu", dtype=torch.bfloat16):
        output = vla(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device),
            labels=batch["labels"].to(device),
            output_hidden_states=True,
            proprio=(batch["proprio"].to(device) if use_proprio else None),
            proprio_projector=proprio_projector if use_proprio else None,
            noisy_actions=None,
            noisy_action_projector=None,
            diffusion_timestep_embeddings=None,
            use_film=False,
        )

    last_hidden_states = output.hidden_states[-1]
    batch_size = batch["input_ids"].shape[0]
    action_token_count = NUM_ACTIONS_CHUNK * ACTION_DIM
    actions_hidden_states = last_hidden_states[:, -action_token_count - 1 : -1].reshape(
        batch_size, action_token_count, -1
    ).to(torch.bfloat16)

    # 确保 action_head 与隐藏状态在同一设备/精度
    ah_device = actions_hidden_states.device
    ah_dtype = actions_hidden_states.dtype
    if any(p.device != ah_device for p in action_head.parameters()):
        predicted_actions = action_head.predict_action(actions_hidden_states.detach().to('cpu')).to(ah_device, dtype=ah_dtype)
    else:
        predicted_actions = action_head.predict_action(actions_hidden_states)
    loss = nn.L1Loss()(ground_truth_actions, predicted_actions)

    ground_truth_curr_action = ground_truth_actions[:, 0]
    predicted_curr_action = predicted_actions[:, 0]
    ground_truth_next_actions = ground_truth_actions[:, 1:]
    predicted_next_actions = predicted_actions[:, 1:]
    curr_action_l1_loss = nn.L1Loss()(ground_truth_curr_action, predicted_curr_action)
    next_actions_l1_loss = nn.L1Loss()(ground_truth_next_actions, predicted_next_actions)

    metrics.update({
        "loss_value": loss.item(),
        "curr_action_l1_loss": curr_action_l1_loss.item(),
        "next_actions_l1_loss": next_actions_l1_loss.item(),
    })

    return loss, metrics


def _remove_ddp_prefix(state_dict: dict) -> dict:
    cleaned = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            cleaned[k[len('module.'):]] = v
        else:
            cleaned[k] = v
    return cleaned


def parse_thresholds(thresholds_str: str) -> List[float]:
    return [float(x) for x in thresholds_str.split(',') if x]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                        help='训练保存的 400_step 合并检查点目录，例如 logs/overfit_experiment/...--400_chkpt')
    parser.add_argument('--data_root_dir', type=str, default='/root/workspace/LightVLA/datasets/libero_overfit')
    parser.add_argument('--dataset_name', type=str, default='libero_spatial_no_noops_mini')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--device', type=str, default='auto', choices=['auto','cuda','cpu'],
                        help='auto 优先用 GPU，OOM 时可指定 cpu')
    parser.add_argument('--device_map', type=str, default='auto', choices=['auto','cuda','cpu'],
                        help='transformers from_pretrained 的 device_map 策略')
    parser.add_argument('--load_in_8bit', action='store_true', default=False,
                        help='使用 8bit 量化（需要 bitsandbytes）')
    parser.add_argument('--load_in_4bit', action='store_true', default=False,
                        help='使用 4bit 量化（需要 bitsandbytes）')
    parser.add_argument('--num_images_in_input', type=int, default=2)
    parser.add_argument('--use_proprio', action='store_true', default=True)
    parser.add_argument('--thresholds', type=str, default='0.25,0.5',
                        help='以逗号分隔的 L1 阈值列表，用于统计命中率')
    parser.add_argument('--max_samples', type=int, default=-1,
                        help='仅评估前 N 条样本，-1 为评估全部')
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    assert checkpoint_dir.exists(), f'检查点目录不存在: {checkpoint_dir}'

    # 注册自定义模型类，确保与训练一致
    AutoConfig.register('openvla', OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    # 加载合并后的 VLA 与 Processor
    processor = PrismaticProcessor.from_pretrained(checkpoint_dir, trust_remote_code=True)
    if args.device in ['auto', 'cuda'] and torch.cuda.is_available():
        target_device = torch.device('cuda:0')
        if torch.cuda.device_count() > 1:
            max_memory = {i: '22GiB' for i in range(torch.cuda.device_count())}
            max_memory['cpu'] = '48GiB'
        else:
            max_memory = {'cuda:0': '22GiB', 'cpu': '48GiB'}
    else:
        target_device = torch.device('cpu')
        max_memory = None

    vla = OpenVLAForActionPrediction.from_pretrained(
        checkpoint_dir,
        torch_dtype=torch.float16 if target_device.type == 'cuda' else torch.float32,
        device_map=(args.device_map if target_device.type == 'cuda' else 'cpu'),
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit,
        max_memory=max_memory,
    )
    vla.set_num_images_in_input(args.num_images_in_input)
    vla.eval()
    device = target_device

    # 可选部件：proprio projector 与 action head
    def _pick_latest(prefix: str):
        files = sorted(checkpoint_dir.glob(f"{prefix}--*_checkpoint.pt"))
        if not files:
            raise FileNotFoundError(f"未找到 {prefix} 检查点文件: {checkpoint_dir}")
        def _step(p: Path) -> int:
            try:
                return int(p.name.split("--")[1].split("_")[0])
            except Exception:
                return -1
        return max(files, key=_step)

    proprio_projector = None
    if args.use_proprio:
        proprio_projector = ProprioProjector(llm_dim=vla.llm_dim, proprio_dim=8)
        proprio_ckpt = _pick_latest("proprio_projector")
        proprio_sd = torch.load(proprio_ckpt, map_location='cpu', weights_only=True)
        proprio_projector.load_state_dict(_remove_ddp_prefix(proprio_sd))
        proprio_projector = proprio_projector.to(torch.bfloat16 if target_device.type == 'cuda' else torch.float32).eval()

    action_head = L1RegressionActionHead(input_dim=vla.llm_dim, hidden_dim=vla.llm_dim, action_dim=ACTION_DIM)
    action_head_ckpt = _pick_latest("action_head")
    ah_sd = torch.load(action_head_ckpt, map_location='cpu', weights_only=True)
    action_head.load_state_dict(_remove_ddp_prefix(ah_sd))
    action_head = action_head.to(torch.bfloat16 if target_device.type == 'cuda' else torch.float32).eval()

    # 计算视觉 patch 数
    num_patches = vla.get_num_patches()
    if args.use_proprio:
        num_patches += 1

    # 数据集与 DataLoader
    print("🔧 初始化数据加载器...")
    action_tokenizer = ActionTokenizer(processor.tokenizer)
    batch_transform = RLDSBatchTransform(
        action_tokenizer,
        processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder,
        use_wrist_image=(args.num_images_in_input > 1),
        use_proprio=args.use_proprio,
    )
    print("🔧 创建数据集...")
    dataset = RLDSDataset(
        Path(args.data_root_dir),
        args.dataset_name,
        batch_transform,
        resize_resolution=tuple(vla.config.image_sizes),
        shuffle_buffer_size=1,  # 评估无需打乱
        image_aug=False,
    )
    print(f"✅ 数据集创建完成，样本数: {len(dataset)}")

    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length,
        processor.tokenizer.pad_token_id,
        padding_side='right',
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collator, num_workers=0)

    thresholds = parse_thresholds(args.thresholds)

    total_samples = 0
    l1_sum = 0.0
    hits = [0 for _ in thresholds]

    print("🚀 开始评估...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            print(f"📊 处理批次 {batch_idx + 1}...")
            # 只计算 L1，不需要反传
            loss, metrics = eval_forward_pass_l1(
                vla=vla,
                action_head=action_head,
                proprio_projector=proprio_projector,
                batch=batch,
                device=device,
                use_proprio=args.use_proprio,
                num_patches=num_patches,
            )

            # metrics 中 curr_action_l1_loss/next_actions_l1_loss 是均值；这里用 loss_value 统一表示 batch 平均 L1
            batch_size = batch['input_ids'].shape[0]
            l1_val = float(metrics['loss_value'])
            l1_sum += l1_val * batch_size
            total_samples += batch_size

            # 将均值 L1 与阈值比较作为“命中”近似（overfit 验证足够）
            for i, th in enumerate(thresholds):
                if l1_val <= th:
                    hits[i] += batch_size

            # 进度与早停
            if args.max_samples > 0 and total_samples >= args.max_samples:
                break
            if total_samples % 5 == 0:
                print(f"[进度] 已评估样本: {total_samples}")

    avg_l1 = l1_sum / max(total_samples, 1)
    print(f"样本数: {total_samples}")
    print(f"平均 L1: {avg_l1:.4f}")
    for th, hit in zip(thresholds, hits):
        acc = hit / max(total_samples, 1)
        print(f"L1 <= {th:.3f} 的比例: {acc*100:.2f}% ({hit}/{total_samples})")


if __name__ == '__main__':
    main()
