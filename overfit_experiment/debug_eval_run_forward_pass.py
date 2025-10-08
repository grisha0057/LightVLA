#!/usr/bin/env python3
"""
复用训练时的 run_forward_pass 对已训练 checkpoint 做一次单 batch 验证。
"""

import argparse
from pathlib import Path

import torch

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticProcessor
from prismatic.models.action_heads import L1RegressionActionHead
from prismatic.models.projectors import ProprioProjector
from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK
from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.vla.action_tokenizer import ActionTokenizer

import sys
import importlib.util
spec = importlib.util.spec_from_file_location("finetune", "/root/workspace/LightVLA/vla-scripts/finetune.py")
finetune = importlib.util.module_from_spec(spec)
spec.loader.exec_module(finetune)
run_forward_pass = finetune.run_forward_pass


def _remove_ddp_prefix(state_dict: dict) -> dict:
    cleaned = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            cleaned[k[len('module.'):]] = v
        else:
            cleaned[k] = v
    return cleaned


def load_latest(prefix: str, ckpt_dir: Path) -> Path:
    files = sorted(ckpt_dir.glob(f"{prefix}--*_checkpoint.pt"))
    if not files:
        raise FileNotFoundError(f"未找到 {prefix} 检查点文件: {ckpt_dir}")
    def _step(p: Path) -> int:
        try:
            return int(p.name.split('--')[1].split('_')[0])
        except Exception:
            return -1
    return max(files, key=_step)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--data_root_dir", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--use_proprio", action="store_true", default=True)
    parser.add_argument("--num_images_in_input", type=int, default=2)
    args = parser.parse_args()

    ckpt_dir = Path(args.checkpoint_dir)

    processor = PrismaticProcessor.from_pretrained(ckpt_dir, trust_remote_code=True)

    device_id = 0 if torch.cuda.is_available() else -1
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    
    vla = OpenVLAForActionPrediction.from_pretrained(
        ckpt_dir,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
        device_map=device,
    )
    vla.set_num_images_in_input(args.num_images_in_input)
    # 使用eval模式来测试推理时的硬剪枝逻辑
    vla = vla.eval()

    proprio_projector = None
    if args.use_proprio:
        proprio_projector = ProprioProjector(llm_dim=vla.llm_dim, proprio_dim=8)
        proprio_sd = torch.load(load_latest("proprio_projector", ckpt_dir), map_location="cpu")
        proprio_projector.load_state_dict(_remove_ddp_prefix(proprio_sd))
        proprio_projector = proprio_projector.to(device_id).eval()

    action_head = L1RegressionActionHead(input_dim=vla.llm_dim, hidden_dim=vla.llm_dim, action_dim=ACTION_DIM)
    ah_sd = torch.load(load_latest("action_head", ckpt_dir), map_location="cpu")
    action_head.load_state_dict(_remove_ddp_prefix(ah_sd))
    action_head = action_head.to(device_id).to(torch.bfloat16).eval()
    
    # 创建一个简单的包装器来模拟DDP的.module属性
    class DDPWrapper:
        def __init__(self, module):
            self.module = module
    action_head = DDPWrapper(action_head)

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
        shuffle_buffer_size=1,
        image_aug=False,
    )
    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length,
        processor.tokenizer.pad_token_id,
        padding_side="right",
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, collate_fn=collator, num_workers=0)

    batch = next(iter(loader))
    
    # 确保batch中的所有数据都在正确的设备上
    batch = {k: v.to(device_id) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    # 在eval模式下，需要动态计算num_patches，因为会进行硬剪枝
    # 我们先运行一次forward pass来获取实际的hidden states长度
    with torch.no_grad():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            test_output = vla(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device_id),
                labels=batch["labels"],
                output_hidden_states=True,
                proprio=batch["proprio"] if args.use_proprio else None,
                proprio_projector=proprio_projector if args.use_proprio else None,
            )
    
    # 根据实际的hidden states长度计算num_patches
    actual_seq_len = test_output.hidden_states[-1].shape[1]
    text_seq_len = batch["labels"].shape[1] - 1  # 减去最后一个token
    num_patches = actual_seq_len - text_seq_len - 1  # 减去最后一个token
    
    print(f"🔍 调试信息:")
    print(f"  num_patches: {num_patches}")
    print(f"  batch['labels'].shape: {batch['labels'].shape}")
    print(f"  batch['input_ids'].shape: {batch['input_ids'].shape}")
    print(f"  vla.config.image_sizes: {vla.config.image_sizes}")
    print(f"  vla.vision_backbone.get_num_patches(): {vla.vision_backbone.get_num_patches()}")
    print(f"  vla.vision_backbone.get_num_images_in_input(): {vla.vision_backbone.get_num_images_in_input()}")
    print(f"  vla.get_num_patches(): {vla.get_num_patches()}")

    with torch.no_grad():
        # 先运行一次forward pass看看实际的hidden states形状
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = vla(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device_id),
                labels=batch["labels"],
                output_hidden_states=True,
                proprio=batch["proprio"] if args.use_proprio else None,
                proprio_projector=proprio_projector if args.use_proprio else None,
            )
        
        print(f"🔍 Forward pass结果:")
        print(f"  output.hidden_states[-1].shape: {output.hidden_states[-1].shape}")
        print(f"  last_hidden_states shape: {output.hidden_states[-1].shape}")
        
        # 计算实际的text_hidden_states长度
        actual_text_length = output.hidden_states[-1].shape[1] - num_patches - 1
        print(f"  actual_text_length: {actual_text_length}")
        print(f"  num_patches: {num_patches}")
        
        # 调整num_patches以确保text_hidden_states有正确的长度
        if actual_text_length != batch["labels"].shape[1] - 1:
            # 重新计算num_patches
            num_patches = output.hidden_states[-1].shape[1] - (batch["labels"].shape[1] - 1) - 1
            print(f"  调整后的num_patches: {num_patches}")
        
        # 现在运行完整的run_forward_pass
        loss, metrics = run_forward_pass(
            vla=vla,
            action_head=action_head,
            noisy_action_projector=None,
            proprio_projector=proprio_projector,
            batch=batch,
            action_tokenizer=action_tokenizer,
            device_id=device_id,
            use_l1_regression=True,
            use_diffusion=False,
            use_proprio=args.use_proprio,
            use_film=False,
            num_patches=num_patches,
            compute_diffusion_l1=False,
            num_diffusion_steps_train=None,
        )

    print("--- Single batch metrics ---")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()
