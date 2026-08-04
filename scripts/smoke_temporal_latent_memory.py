#!/usr/bin/env python3
"""Controlled feasibility test for hierarchical temporal latent memory.

The synthetic feature task hides one state token inside every clip and a marker
inside one clip. The label is the state in the immediately following clip.
Local latent compression must find sparse events and temporal memory must retain
their order. This is a mechanism smoke, not a real-video benchmark result.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-samples", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))

    import torch
    import torch.nn.functional as F

    from src.model.temporal_latent_memory import TemporalLatentMemory, UniformTokenBaseline

    if args.steps < 1 or args.batch_size < 1 or args.eval_samples < 1:
        raise ValueError("steps, batch-size, and eval-samples must be positive")
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    input_dim = 64
    num_clips = 8
    tokens_per_clip = 16
    slots_per_clip = 2
    num_classes = 4

    # Fixed semantic codebook shared by train and held-out synthetic samples.
    codebook_generator = torch.Generator().manual_seed(12345)
    state_codes = F.normalize(torch.randn(num_classes, input_dim, generator=codebook_generator), dim=-1)
    state_marker = F.normalize(torch.randn(input_dim, generator=codebook_generator), dim=-1)
    query_marker = F.normalize(torch.randn(input_dim, generator=codebook_generator), dim=-1)
    question_code = F.normalize(torch.randn(input_dim, generator=codebook_generator), dim=-1)
    state_codes = state_codes.to(device)
    state_marker = state_marker.to(device)
    query_marker = query_marker.to(device)
    question_code = question_code.to(device)

    def make_batch(batch_size: int, generator: torch.Generator):
        tokens = torch.randn(
            batch_size,
            num_clips,
            tokens_per_clip,
            input_dim,
            generator=generator,
        ).to(device) * 0.08
        states = torch.randint(num_classes, (batch_size, num_clips), generator=generator).to(device)
        target_clip = torch.randint(num_clips - 1, (batch_size,), generator=generator).to(device)
        state_positions = torch.randint(tokens_per_clip, (batch_size, num_clips), generator=generator).to(device)
        marker_positions = torch.randint(tokens_per_clip, (batch_size,), generator=generator).to(device)

        batch_indices = torch.arange(batch_size, device=device)
        for clip in range(num_clips):
            positions = state_positions[:, clip]
            tokens[batch_indices, clip, positions] += 3.0 * state_marker + 3.0 * state_codes[states[:, clip]]
        # Avoid overwriting the state token in the marked clip.
        collision = marker_positions == state_positions[batch_indices, target_clip]
        marker_positions = torch.where(collision, (marker_positions + 1) % tokens_per_clip, marker_positions)
        tokens[batch_indices, target_clip, marker_positions] += 4.0 * query_marker

        labels = states[batch_indices, target_clip + 1]
        questions = question_code.unsqueeze(0).expand(batch_size, -1)
        questions = questions + 0.02 * torch.randn(batch_size, input_dim, generator=generator).to(device)
        return tokens, questions, labels

    def train_model(model, seed_offset: int):
        model.to(device).train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
        generator = torch.Generator().manual_seed(args.seed + seed_offset)
        final_loss = None
        for _ in range(args.steps):
            tokens, questions, labels = make_batch(args.batch_size, generator)
            loss = F.cross_entropy(model(tokens, questions)["logits"], labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            final_loss = float(loss.detach())
        return final_loss

    memory_model = TemporalLatentMemory(
        input_dim=input_dim,
        memory_dim=64,
        slots_per_clip=slots_per_clip,
        num_heads=4,
        num_layers=2,
        num_classes=num_classes,
        max_clips=num_clips,
    )
    baseline_model = UniformTokenBaseline(
        input_dim=input_dim,
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
        "device": str(device),
        "steps_per_model": args.steps,
        "trainable_parameters_memory": sum(p.numel() for p in memory_model.parameters() if p.requires_grad),
        "trainable_parameters_baseline": sum(p.numel() for p in baseline_model.parameters() if p.requires_grad),
        "global_tokens_each": num_clips * slots_per_clip,
        "memory_final_train_loss": round(memory_loss, 6),
        "baseline_final_train_loss": round(baseline_loss, 6),
        "memory_accuracy": round(float(memory_accuracy), 4),
        "uniform_token_accuracy": round(float(baseline_accuracy), 4),
        "shuffled_memory_accuracy": round(float(shuffled_accuracy), 4),
        "memory_gain": round(float(memory_accuracy - baseline_accuracy), 4),
    }
    print(json.dumps(result, indent=2))

    # These thresholds are deliberately modest: this smoke only establishes
    # that learned local compression and temporal order carry useful signal.
    passed = (
        memory_accuracy >= 0.80
        and memory_accuracy >= baseline_accuracy + 0.10
        and memory_accuracy >= shuffled_accuracy + 0.10
    )
    if not passed:
        print("OVERALL FAIL: temporal-memory mechanism did not clear feasibility thresholds")
        return 1
    print("OVERALL PASS: controlled temporal latent-memory mechanism is learnable")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
