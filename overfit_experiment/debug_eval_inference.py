#!/usr/bin/env python3
"""
调用 `OpenVLA.predict_action` 对一个 checkpoint 做少量样本的硬剪推理，
并在与 ground truth 同一尺度上计算 L1。
"""

import argparse
from pathlib import Path
import json

import torch
import numpy as np

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticProcessor
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.models.backbones.llm.prompting import PurePromptBuilder


def load_checkpoint(checkpoint_dir: Path):
    processor = PrismaticProcessor.from_pretrained(checkpoint_dir, trust_remote_code=True)
    
    # 手动加载dataset_statistics.json
    with open(checkpoint_dir / "dataset_statistics.json", "r") as f:
        norm_stats = json.load(f)
    
    # 创建配置并设置norm_stats
    config = OpenVLAConfig.from_pretrained(checkpoint_dir, trust_remote_code=True)
    config.norm_stats = norm_stats
    
    model = OpenVLAForActionPrediction.from_pretrained(
        checkpoint_dir,
        config=config,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    return processor, model, device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--data_root_dir", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--num_images_in_input", type=int, default=2)
    parser.add_argument("--use_proprio", action="store_true", default=True)
    parser.add_argument("--max_samples", type=int, default=20)
    args = parser.parse_args()

    ckpt_dir = Path(args.checkpoint_dir)
    processor, model, device = load_checkpoint(ckpt_dir)
    model.set_num_images_in_input(args.num_images_in_input)

    tokenizer = ActionTokenizer(processor.tokenizer)
    batch_transform = RLDSBatchTransform(
        tokenizer,
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
        resize_resolution=tuple(model.config.image_sizes),
        shuffle_buffer_size=1,
        train=True,  # 使用train split，因为数据集只有train
        image_aug=False,
    )
    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length,
        processor.tokenizer.pad_token_id,
        padding_side="right",
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, collate_fn=collator, num_workers=0)

    total_l1 = 0.0
    total = 0

    for batch in loader:
        if total >= args.max_samples > 0:
            break

        sample = {k: v[0] for k, v in batch.items()}

        pixel_values = sample["pixel_values"]
        images = torch.tensor(pixel_values).unsqueeze(0).to(device)
        proprio = sample.get("proprio")
        if proprio is not None:
            proprio = torch.tensor(proprio).unsqueeze(0).to(device)

        predicted_actions, _ = model.predict_action(
            input_ids=sample["input_ids"].unsqueeze(0).to(device),
            attention_mask=sample["attention_mask"].unsqueeze(0).to(device),
            pixel_values=images.to(torch.bfloat16 if device.type == "cuda" else torch.float32),
            proprio=proprio.to(torch.bfloat16 if proprio is not None and device.type == "cuda" else torch.float32)
            if proprio is not None
            else None,
            use_film=False,
            unnorm_key="libero_spatial_no_noops_mini",  # 使用我们实际训练的数据集名称
        )

        gt = np.array(sample["actions"])
        if torch.is_tensor(predicted_actions):
            predicted_actions = predicted_actions.detach().cpu().numpy()

        l1 = np.abs(predicted_actions - gt).mean()
        total_l1 += l1
        total += 1

    avg_l1 = total_l1 / max(total, 1)
    print(f"样本数: {total}")
    print(f"平均 L1: {avg_l1:.4f}")


if __name__ == "__main__":
    main()
