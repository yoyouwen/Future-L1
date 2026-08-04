#!/usr/bin/env python3
"""Run one Future-L1 SFT forward pass, optionally followed by backward.

This script never creates a Trainer and never saves a model. It expects an
existing Qwen2.5-VL/Qwen3-VL/Qwen3.5 checkpoint plus TwiFF-format JSON/video
data. Remote checkpoint access is disabled unless --allow-download is passed.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-latent-tokens", type=int, default=4)
    parser.add_argument("--latent-lambda", type=float, default=0.2)
    parser.add_argument("--latent-loss", choices=("mse", "sim"), default="mse")
    parser.add_argument("--backward", action="store_true")
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args()

    os.environ["FUTURE_L1_SKIP_FINAL_SAVE"] = "1"
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))

    import torch
    from transformers import AutoConfig, AutoProcessor

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("FAIL CUDA requested but torch.cuda.is_available() is false", file=sys.stderr)
        return 1
    if not args.data_path.exists():
        print(f"FAIL data path does not exist: {args.data_path}", file=sys.stderr)
        return 1

    from src.dataset.twiff_sft_dataset import TwiFFDataCollator, TwiFFSFTDataset
    from src.model.future_l1 import (
        FutureL1_Qwen2_5_VL,
        FutureL1_Qwen3VL,
        FutureL1_Qwen3_5_VL,
        QWEN3_5_BACKBONE_AVAILABLE,
    )
    from src.params import DataArguments
    from src.train.train_utils import get_vision_tower
    from src.train.monkey_patch_forward import (
        replace_qwen2_5_vl_generation_forward,
        replace_qwen2_5_with_mixed_modality_forward,
        replace_qwen3_5_generation_forward,
        replace_qwen3_5_with_mixed_modality_forward,
        replace_qwen3_vl_generation_forward,
        replace_qwen3_with_mixed_modality_forward,
    )

    local_only = not args.allow_download
    config = AutoConfig.from_pretrained(args.model_path, local_files_only=local_only)
    config.use_projection_head = False
    config.latent_loss = args.latent_loss
    config.latent_lambda = args.latent_lambda
    config.max_latent_token = args.max_latent_tokens
    config.train_fixed_latent_budget = None
    config.pool_after_proj = True
    config.use_cache = False

    if config.model_type == "qwen2_5_vl":
        replace_qwen2_5_with_mixed_modality_forward()
        replace_qwen2_5_vl_generation_forward()
        model_cls = FutureL1_Qwen2_5_VL
    elif config.model_type == "qwen3_vl":
        replace_qwen3_with_mixed_modality_forward()
        replace_qwen3_vl_generation_forward()
        model_cls = FutureL1_Qwen3VL
    elif config.model_type == "qwen3_5" and QWEN3_5_BACKBONE_AVAILABLE:
        replace_qwen3_5_with_mixed_modality_forward()
        replace_qwen3_5_generation_forward()
        model_cls = FutureL1_Qwen3_5_VL
    else:
        print(f"FAIL unsupported model_type={config.model_type!r}", file=sys.stderr)
        return 1

    device = torch.device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    processor = AutoProcessor.from_pretrained(args.model_path, local_files_only=local_only)
    chat_template_path = repo_root / "chat_template.json"
    with chat_template_path.open("r", encoding="utf-8") as handle:
        future_l1_chat_template = json.load(handle)["chat_template"]
    processor.chat_template = future_l1_chat_template
    processor.tokenizer.chat_template = future_l1_chat_template
    print(f"using Future-L1 chat template from {chat_template_path}")
    processor.tokenizer.add_tokens(
        ["<|latent|>", "<|latent_start|>", "<|latent_end|>"],
        special_tokens=False,
    )
    config.latent_id = processor.tokenizer.convert_tokens_to_ids("<|latent|>")
    config.latent_start_id = processor.tokenizer.convert_tokens_to_ids("<|latent_start|>")
    config.latent_end_id = processor.tokenizer.convert_tokens_to_ids("<|latent_end|>")

    print(f"loading model_type={config.model_type} dtype={dtype} device={device}")
    load_kwargs = {
        "config": config,
        "dtype": dtype,
        "attn_implementation": "sdpa",
        "local_files_only": local_only,
        "low_cpu_mem_usage": True,
    }
    if device.type == "cuda":
        # Load each safetensors shard directly onto the single target GPU.
        # Loading onto CPU first and then calling model.to(cuda) can trigger
        # many delayed mmap page faults when checkpoints live on RunPod's
        # network volume, making the process appear hung in folio_wait_bit.
        load_kwargs["device_map"] = {"": str(device)}

    model = model_cls.from_pretrained(args.model_path, **load_kwargs)
    if model.get_input_embeddings().num_embeddings < len(processor.tokenizer):
        model.resize_token_embeddings(len(processor.tokenizer))
    if device.type != "cuda":
        model.to(device)
    # Match scripts/train_twiff.sh: keep the shared vision teacher and merger
    # frozen while allowing the language model to receive gradients.
    vision_tower = get_vision_tower(model)
    for parameter in vision_tower.parameters():
        parameter.requires_grad = False
    model.train(mode=args.backward)

    data_args = DataArguments(
        data_path=[str(args.data_path)],
        use_twiff_dataset=True,
        image_min_pixels=2 * 32 * 32,
        image_max_pixels=128 * 32 * 32,
        video_max_pixels=128 * 32 * 32,
        max_latent_token=args.max_latent_tokens,
        nframes=4,
        random_seed=0,
    )
    dataset = TwiFFSFTDataset(str(args.data_path))
    collator = TwiFFDataCollator(processor=processor, args=data_args)
    batch = collator([dataset[0]])
    if not batch:
        print("FAIL collator returned an empty batch", file=sys.stderr)
        return 1
    batch = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
        if value is not None
    }

    latent_count = int((batch["input_ids"] == config.latent_id).sum().item())
    latent_mask_count = int(batch["image_out_mask"].sum().item())
    print(
        f"latent_id={config.latent_id} latent_tokens={latent_count} "
        f"image_out_mask={latent_mask_count}"
    )
    if latent_count <= 0 or latent_mask_count != latent_count:
        print("FAIL collated latent token/mask alignment", file=sys.stderr)
        return 1

    with torch.set_grad_enabled(args.backward):
        outputs = model(**batch)
        loss = outputs.loss
        ce_loss = getattr(outputs, "ce_loss", None)
        latent_loss = getattr(outputs, "latent_loss", None)

    values = {
        "loss": loss,
        "ce_loss": ce_loss,
        "latent_loss": latent_loss,
    }
    for name, value in values.items():
        scalar = None if value is None else float(value.detach().float().mean().cpu())
        print(f"{name}={scalar}")
        if scalar is None or not math.isfinite(scalar):
            print(f"FAIL {name} is missing or non-finite", file=sys.stderr)
            return 1

    if args.backward:
        loss.backward()
        finite_grads = 0
        nonfinite_grads = 0
        for parameter in model.parameters():
            if parameter.grad is None:
                continue
            if torch.isfinite(parameter.grad).all():
                finite_grads += 1
            else:
                nonfinite_grads += 1
        print(f"backward finite_grad_tensors={finite_grads} nonfinite_grad_tensors={nonfinite_grads}")
        if finite_grads == 0 or nonfinite_grads:
            print("FAIL backward gradients", file=sys.stderr)
            return 1

    print("OVERALL PASS (no Trainer, no checkpoint save)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
