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
            labels=batch["labels"],
            output_hidden_states=True,
            proprio=batch["proprio"] if use_proprio else None,
            proprio_projector=proprio_projector if use_proprio else None,
            noisy_actions=None,
            noisy_action_projector=None,
            diffusion_timestep_embeddings=None,
            use_film=False,
        )

    ground_truth_token_ids = batch["labels"][:, 1:].to(device)
    current_action_mask = get_current_action_mask(ground_truth_token_ids)
    next_actions_mask = get_next_actions_mask(ground_truth_token_ids)

    last_hidden_states = output.hidden_states[-1]
    text_hidden_states = last_hidden_states[:, num_patches:-1]
    batch_size = batch["input_ids"].shape[0]
    actions_hidden_states = (
        text_hidden_states[current_action_mask | next_actions_mask]
        .reshape(batch_size, NUM_ACTIONS_CHUNK * ACTION_DIM, -1)
        .to(torch.bfloat16)
    )

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
    parser.add_argument('--num_images_in_input', type=int, default=2)
    parser.add_argument('--use_proprio', action='store_true', default=True)
    parser.add_argument('--thresholds', type=str, default='0.25,0.5',
                        help='以逗号分隔的 L1 阈值列表，用于统计命中率')
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
    target_device = (
        torch.device('cuda:0') if (args.device in ['auto','cuda'] and torch.cuda.is_available()) else torch.device('cpu')
    )
    vla = OpenVLAForActionPrediction.from_pretrained(
        checkpoint_dir,
        torch_dtype=torch.bfloat16 if target_device.type == 'cuda' else torch.float32,
        device_map=str(target_device) if target_device.type == 'cuda' else 'cpu',
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    vla.set_num_images_in_input(args.num_images_in_input)
    vla.eval()
    device = target_device

    # 可选部件：proprio projector 与 action head
    proprio_projector = None
    if args.use_proprio:
        proprio_projector = ProprioProjector(llm_dim=vla.llm_dim, proprio_dim=8)  # PROPRIO_DIM=8（LIBERO）
        proprio_sd = torch.load(checkpoint_dir / 'proprio_projector--400_checkpoint.pt', map_location='cpu', weights_only=True)
        proprio_projector.load_state_dict(_remove_ddp_prefix(proprio_sd))
        proprio_projector = proprio_projector.to(device).to(torch.bfloat16 if device.type=='cuda' else torch.float32).eval()

    action_head = L1RegressionActionHead(input_dim=vla.llm_dim, hidden_dim=vla.llm_dim, action_dim=ACTION_DIM)
    ah_sd = torch.load(checkpoint_dir / 'action_head--400_checkpoint.pt', map_location='cpu', weights_only=True)
    action_head.load_state_dict(_remove_ddp_prefix(ah_sd))
    action_head = action_head.to(device).to(torch.bfloat16 if device.type=='cuda' else torch.float32).eval()

    # 计算视觉 patch 数
    num_patches = vla.get_num_patches()
    if args.use_proprio:
        num_patches += 1

    # 数据集与 DataLoader
    action_tokenizer = ActionTokenizer(processor.tokenizer)
    batch_transform = RLDSBatchTransform(
        action_tokenizer,
        processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder,
        use_wrist_image=(args.num_images_in_input > 1),
        use_proprio=args.use_proprio,
    )
    dataset = RLDSDataset(
        Path(args.data_root_dir),
        args.dataset_name,
        batch_transform,
        resize_resolution=tuple(vla.config.image_sizes),
        shuffle_buffer_size=1,  # 评估无需打乱
        image_aug=False,
    )

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

    with torch.no_grad():
        for batch in dataloader:
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

    avg_l1 = l1_sum / max(total_samples, 1)
    print(f"样本数: {total_samples}")
    print(f"平均 L1: {avg_l1:.4f}")
    for th, hit in zip(thresholds, hits):
        acc = hit / max(total_samples, 1)
        print(f"L1 <= {th:.3f} 的比例: {acc*100:.2f}% ({hit}/{total_samples})")


if __name__ == '__main__':
    main()


