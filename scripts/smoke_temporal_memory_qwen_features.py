#!/usr/bin/env python3
"""Bridge the temporal-memory prototype to frozen Qwen3-VL visual features.

Rendered clips contain a colored state in every frame and a brief marker in one
frame. The answer is the state in the clip immediately after the marker. Only
the small temporal-memory or uniform-token control is trained; Qwen stays
frozen and is deleted after extracting a tiny reusable visual prototype bank.

This is a controlled visual-feature test, not a real benchmark evaluation.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-samples", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--allow-download", action="store_true")
    return parser.parse_args()


def render_prototypes(size: int = 224):
    from PIL import Image, ImageDraw

    colors = (
        (220, 45, 45),
        (40, 170, 75),
        (45, 95, 220),
        (235, 190, 35),
    )
    marker_centers = (
        (34, 34),
        (size - 34, 34),
        (34, size - 34),
        (size - 34, size - 34),
    )
    images = []
    # Per state: one unmarked frame followed by four marker-location variants.
    for state, color in enumerate(colors):
        for marker_variant in range(5):
            image = Image.new("RGB", (size, size), (238, 238, 238))
            draw = ImageDraw.Draw(image)
            margin = 42
            draw.rounded_rectangle(
                (margin, margin, size - margin, size - margin),
                radius=18,
                fill=color,
                outline=(25, 25, 25),
                width=5,
            )
            draw.text((size // 2 - 10, size // 2 - 8), str(state), fill=(255, 255, 255))
            if marker_variant:
                cx, cy = marker_centers[marker_variant - 1]
                draw.ellipse((cx - 17, cy - 17, cx + 17, cy + 17), fill=(255, 255, 255), outline=(0, 0, 0), width=5)
                draw.ellipse((cx - 6, cy - 6, cx + 6, cy + 6), fill=(0, 0, 0))
            images.append(image)
    return images


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))

    import torch
    import torch.nn.functional as F
    from transformers import AutoConfig, AutoProcessor, Qwen3VLForConditionalGeneration

    from src.model.temporal_latent_memory import TemporalLatentMemory, UniformTokenBaseline

    if args.steps < 1 or args.batch_size < 1 or args.eval_samples < 1:
        raise ValueError("steps, batch-size, and eval-samples must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    local_only = not args.allow_download

    config = AutoConfig.from_pretrained(args.model_path, local_files_only=local_only)
    if config.model_type != "qwen3_vl":
        raise ValueError(f"expected qwen3_vl checkpoint, got {config.model_type!r}")
    processor = AutoProcessor.from_pretrained(
        args.model_path,
        min_pixels=224 * 224,
        max_pixels=224 * 224,
        local_files_only=local_only,
    )
    print(f"loading frozen Qwen3-VL visual encoder from {args.model_path}")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map={"": str(device)},
        low_cpu_mem_usage=True,
        local_files_only=local_only,
    ).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    images = render_prototypes()
    image_batch = processor.image_processor(images=images, return_tensors="pt")
    pixel_values = image_batch["pixel_values"].to(device=device, dtype=torch.bfloat16)
    image_grid_thw = image_batch["image_grid_thw"].to(device)
    prompt = "Which color state appears immediately after the briefly marked frame?"
    question_ids = processor.tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)

    with torch.inference_mode():
        image_features, _ = model.model.get_image_features(pixel_values, image_grid_thw)
        if len({tuple(features.shape) for features in image_features}) != 1:
            raise RuntimeError("prototype images produced inconsistent visual-token shapes")
        prototype_features = torch.stack(image_features).float()
        question_embedding = model.get_input_embeddings()(question_ids).mean(dim=1).float().squeeze(0)

    qwen_dim = prototype_features.shape[-1]
    reduced_dim = 128
    projection_generator = torch.Generator().manual_seed(98765)
    fixed_projection = torch.randn(qwen_dim, reduced_dim, generator=projection_generator).to(device)
    fixed_projection = fixed_projection / math.sqrt(reduced_dim)
    prototype_features = prototype_features @ fixed_projection
    question_embedding = question_embedding @ fixed_projection
    tokens_per_frame = prototype_features.shape[1]
    print(
        f"PASS frozen_features prototypes={prototype_features.shape[0]} "
        f"tokens_per_frame={tokens_per_frame} qwen_dim={qwen_dim} reduced_dim={reduced_dim}"
    )

    # Qwen is no longer needed: the controlled task reuses the 20 frozen visual
    # prototypes, keeping optimization fast and ensuring the backbone cannot fit.
    del model, pixel_values, image_batch, image_features, fixed_projection
    torch.cuda.empty_cache()

    num_classes = 4
    num_clips = 8
    frames_per_clip = 4
    slots_per_clip = 2

    def make_batch(batch_size: int, generator: torch.Generator):
        states = torch.randint(num_classes, (batch_size, num_clips), generator=generator, device="cpu").to(device)
        target_clip = torch.randint(num_clips - 1, (batch_size,), generator=generator, device="cpu").to(device)
        marker_frame = torch.randint(frames_per_clip, (batch_size,), generator=generator, device="cpu").to(device)
        marker_variant = torch.randint(1, 5, (batch_size,), generator=generator, device="cpu").to(device)

        prototype_indices = states * 5
        batch_indices = torch.arange(batch_size, device=device)
        prototype_indices = prototype_indices.unsqueeze(-1).expand(-1, -1, frames_per_clip).clone()
        prototype_indices[batch_indices, target_clip, marker_frame] += marker_variant
        tokens = prototype_features[prototype_indices]
        tokens = tokens.flatten(2, 3)
        labels = states[batch_indices, target_clip + 1]
        questions = question_embedding.unsqueeze(0).expand(batch_size, -1)
        return tokens, questions, labels

    def train_model(trainable_model, seed_offset: int):
        trainable_model.to(device).train()
        optimizer = torch.optim.AdamW(trainable_model.parameters(), lr=3e-4, weight_decay=1e-3)
        generator = torch.Generator().manual_seed(args.seed + seed_offset)
        final_loss = None
        for _ in range(args.steps):
            clip_tokens, questions, labels = make_batch(args.batch_size, generator)
            loss = F.cross_entropy(trainable_model(clip_tokens, questions)["logits"], labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_model.parameters(), 1.0)
            optimizer.step()
            final_loss = float(loss.detach())
        return final_loss

    memory_model = TemporalLatentMemory(
        input_dim=reduced_dim,
        memory_dim=64,
        slots_per_clip=slots_per_clip,
        num_heads=4,
        num_layers=2,
        num_classes=num_classes,
        max_clips=num_clips,
    )
    baseline_model = UniformTokenBaseline(
        input_dim=reduced_dim,
        memory_dim=64,
        global_tokens=num_clips * slots_per_clip,
        num_heads=4,
        num_layers=2,
        num_classes=num_classes,
    )
    memory_loss = train_model(memory_model, 100)
    baseline_loss = train_model(baseline_model, 200)

    eval_generator = torch.Generator().manual_seed(args.seed + 10_000)
    eval_tokens, eval_questions, eval_labels = make_batch(args.eval_samples, eval_generator)
    memory_model.eval()
    baseline_model.eval()
    with torch.inference_mode():
        memory_accuracy = (memory_model(eval_tokens, eval_questions)["logits"].argmax(-1) == eval_labels).float().mean()
        baseline_accuracy = (baseline_model(eval_tokens, eval_questions)["logits"].argmax(-1) == eval_labels).float().mean()
        shuffled_trials = []
        for shuffle_seed in range(5):
            torch.manual_seed(args.seed + 20_000 + shuffle_seed)
            prediction = memory_model(eval_tokens, eval_questions, shuffle_memory=True)["logits"].argmax(-1)
            shuffled_trials.append((prediction == eval_labels).float().mean())
        shuffled_accuracy = torch.stack(shuffled_trials).mean()

    result = {
        "feature_source": "frozen_qwen3_vl_rendered_frames",
        "steps_per_model": args.steps,
        "clips": num_clips,
        "frames_per_clip": frames_per_clip,
        "local_tokens_per_clip": frames_per_clip * tokens_per_frame,
        "global_tokens_each": num_clips * slots_per_clip,
        "memory_final_train_loss": round(memory_loss, 6),
        "baseline_final_train_loss": round(baseline_loss, 6),
        "memory_accuracy": round(float(memory_accuracy), 4),
        "uniform_token_accuracy": round(float(baseline_accuracy), 4),
        "shuffled_memory_accuracy": round(float(shuffled_accuracy), 4),
        "memory_gain": round(float(memory_accuracy - baseline_accuracy), 4),
    }
    print(json.dumps(result, indent=2))
    passed = (
        memory_accuracy >= 0.65
        and memory_accuracy >= baseline_accuracy + 0.05
        and memory_accuracy >= shuffled_accuracy + 0.05
    )
    if not passed:
        print("OVERALL FAIL: Qwen-feature temporal memory did not clear feasibility thresholds")
        return 1
    print("OVERALL PASS: temporal latent memory transfers to frozen Qwen3-VL visual features")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
