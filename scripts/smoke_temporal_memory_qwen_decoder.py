#!/usr/bin/env python3
"""Inject learned temporal-memory tokens into a frozen Qwen3-VL decoder.

This controlled test uses rendered frames encoded by the frozen Qwen vision
tower. A small temporal-memory module compresses eight clips to 16 continuous
tokens, and a trainable bridge maps them into Qwen's embedding space. The
frozen language decoder scores A/B/C/D directly through its LM head.

Only the temporal memory and bridge are optimized. No checkpoint is saved.
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
    parser.add_argument("--memory-warmup-steps", type=int, default=500)
    parser.add_argument("--memory-warmup-batch-size", type=int, default=32)
    parser.add_argument("--decoder-steps", type=int, default=300)
    parser.add_argument("--decoder-batch-size", type=int, default=8)
    parser.add_argument(
        "--injection-mode",
        choices=("cross_attention", "prefix"),
        default="cross_attention",
    )
    parser.add_argument("--eval-samples", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--allow-download", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))

    import torch
    import torch.nn.functional as F
    from torch import nn
    from transformers import AutoConfig, AutoProcessor, Qwen3VLForConditionalGeneration

    from scripts.smoke_temporal_memory_qwen_features import render_prototypes
    from src.model.temporal_latent_memory import TemporalLatentMemory

    numeric_args = (
        args.memory_warmup_steps,
        args.memory_warmup_batch_size,
        args.decoder_steps,
        args.decoder_batch_size,
        args.eval_samples,
        args.eval_batch_size,
    )
    if any(value < 1 for value in numeric_args):
        raise ValueError("all step, batch, and sample counts must be positive")
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
    print(f"loading frozen Qwen3-VL from {args.model_path}")
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
    semantic_question = "Which color state appears immediately after the briefly marked frame?"
    semantic_ids = processor.tokenizer(semantic_question, return_tensors="pt")["input_ids"].to(device)
    decoder_prompt = (
        "Video memory is provided before this question. Which color state appears "
        "immediately after the briefly marked frame? Answer with A, B, C, or D. Answer:"
    )
    decoder_prompt_ids = processor.tokenizer(decoder_prompt, return_tensors="pt")["input_ids"].to(device)

    def single_token_id(text: str) -> int:
        ids = processor.tokenizer.encode(text, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"expected one token for {text!r}, got ids={ids}")
        return ids[0]

    answer_texts = [" A", " B", " C", " D"]
    try:
        answer_token_ids = torch.tensor([single_token_id(text) for text in answer_texts], device=device)
    except ValueError:
        answer_texts = ["A", "B", "C", "D"]
        answer_token_ids = torch.tensor([single_token_id(text) for text in answer_texts], device=device)
    if answer_token_ids.unique().numel() != 4:
        raise RuntimeError(f"answer token ids are not unique: {answer_token_ids.tolist()}")

    # These frozen outputs later feed trainable components. no_grad avoids an
    # unnecessary graph without giving them inference-only tensor semantics.
    with torch.no_grad():
        image_features, _ = model.model.get_image_features(pixel_values, image_grid_thw)
        if len({tuple(features.shape) for features in image_features}) != 1:
            raise RuntimeError("prototype images produced inconsistent visual-token shapes")
        prototype_features = torch.stack(image_features).float()
        embedding_layer = model.get_input_embeddings()
        semantic_embedding = embedding_layer(semantic_ids).mean(dim=1).float().squeeze(0)
        decoder_prompt_embeddings = embedding_layer(decoder_prompt_ids).detach()

    qwen_dim = prototype_features.shape[-1]
    reduced_dim = 128
    projection_generator = torch.Generator().manual_seed(98765)
    fixed_projection = torch.randn(qwen_dim, reduced_dim, generator=projection_generator).to(device)
    fixed_projection = fixed_projection / math.sqrt(reduced_dim)
    prototype_features = prototype_features @ fixed_projection
    semantic_embedding = semantic_embedding @ fixed_projection
    tokens_per_frame = prototype_features.shape[1]
    print(
        f"PASS frozen_features prototypes={prototype_features.shape[0]} "
        f"tokens_per_frame={tokens_per_frame} qwen_dim={qwen_dim} "
        f"answer_token_ids={answer_token_ids.tolist()}"
    )
    del pixel_values, image_batch, image_features, fixed_projection
    torch.cuda.empty_cache()

    num_classes = 4
    num_clips = 8
    frames_per_clip = 4
    slots_per_clip = 2
    memory_dim = 64

    def make_batch(batch_size: int, generator: torch.Generator):
        states = torch.randint(num_classes, (batch_size, num_clips), generator=generator).to(device)
        target_clip = torch.randint(num_clips - 1, (batch_size,), generator=generator).to(device)
        marker_frame = torch.randint(frames_per_clip, (batch_size,), generator=generator).to(device)
        marker_variant = torch.randint(1, 5, (batch_size,), generator=generator).to(device)
        prototype_indices = states * 5
        batch_indices = torch.arange(batch_size, device=device)
        prototype_indices = prototype_indices.unsqueeze(-1).expand(-1, -1, frames_per_clip).clone()
        prototype_indices[batch_indices, target_clip, marker_frame] += marker_variant
        clip_tokens = prototype_features[prototype_indices].flatten(2, 3)
        labels = states[batch_indices, target_clip + 1]
        questions = semantic_embedding.unsqueeze(0).expand(batch_size, -1)
        return clip_tokens, questions, labels

    memory_model = TemporalLatentMemory(
        input_dim=reduced_dim,
        memory_dim=memory_dim,
        slots_per_clip=slots_per_clip,
        num_heads=4,
        num_layers=2,
        num_classes=num_classes,
        max_clips=num_clips,
    ).to(device)

    # Warm up the temporal mechanism with its small diagnostic classifier.
    warmup_optimizer = torch.optim.AdamW(memory_model.parameters(), lr=3e-4, weight_decay=1e-3)
    warmup_generator = torch.Generator().manual_seed(args.seed + 100)
    memory_model.train()
    warmup_loss = None
    for _ in range(args.memory_warmup_steps):
        clip_tokens, questions, labels = make_batch(args.memory_warmup_batch_size, warmup_generator)
        warmup_loss_tensor = F.cross_entropy(memory_model(clip_tokens, questions)["logits"], labels)
        warmup_optimizer.zero_grad(set_to_none=True)
        warmup_loss_tensor.backward()
        torch.nn.utils.clip_grad_norm_(memory_model.parameters(), 1.0)
        warmup_optimizer.step()
        warmup_loss = float(warmup_loss_tensor.detach())
    del warmup_optimizer

    # Do not spend time backpropagating through the frozen 8B decoder unless
    # the temporal mechanism has first learned the controlled visual task.
    warmup_eval_generator = torch.Generator().manual_seed(args.seed + 9_000)
    warmup_eval_tokens, warmup_eval_questions, warmup_eval_labels = make_batch(512, warmup_eval_generator)
    memory_model.eval()
    with torch.inference_mode():
        warmup_prediction = memory_model(warmup_eval_tokens, warmup_eval_questions)["logits"].argmax(-1)
        warmup_accuracy = float((warmup_prediction == warmup_eval_labels).float().mean())
    del warmup_eval_tokens, warmup_eval_questions, warmup_eval_labels, warmup_prediction
    print(f"PASS memory_warmup loss={warmup_loss:.6f} heldout_accuracy={warmup_accuracy:.4f}")
    if warmup_accuracy < 0.60:
        print("OVERALL FAIL: memory warmup did not learn; decoder stage skipped")
        return 1

    decoder = model.model.language_model
    candidate_lm_weights = model.lm_head.weight.index_select(0, answer_token_ids).detach()

    memory_bridge = nn.Sequential(nn.LayerNorm(memory_dim), nn.Linear(memory_dim, qwen_dim)).to(device)
    nn.init.normal_(memory_bridge[1].weight, mean=0.0, std=0.02)
    nn.init.zeros_(memory_bridge[1].bias)

    class DecoderMemoryCrossAttention(nn.Module):
        """Small explicit read interface between frozen Qwen and latent memory."""

        def __init__(self, decoder_dim: int, latent_dim: int, adapter_dim: int = 256):
            super().__init__()
            self.query_norm = nn.LayerNorm(decoder_dim)
            self.memory_norm = nn.LayerNorm(latent_dim)
            self.query_projection = nn.Linear(decoder_dim, adapter_dim)
            self.memory_projection = nn.Linear(latent_dim, adapter_dim)
            self.attention = nn.MultiheadAttention(adapter_dim, 4, batch_first=True)
            self.output_projection = nn.Linear(adapter_dim, decoder_dim)
            self.output_gate = nn.Parameter(torch.tensor(0.1))

        def forward(self, decoder_hidden, latent_memory):
            query = self.query_projection(self.query_norm(decoder_hidden)).unsqueeze(1)
            keys = self.memory_projection(self.memory_norm(latent_memory))
            retrieved, attention = self.attention(query, keys, keys, need_weights=True)
            delta = self.output_projection(retrieved.squeeze(1))
            fused = decoder_hidden.float() + self.output_gate.tanh() * delta
            return fused, attention.squeeze(1)

    cross_attention_adapter = DecoderMemoryCrossAttention(qwen_dim, memory_dim).to(device)

    # Cache the frozen decoder's question representation. Cross-attention mode
    # trains only an explicit memory-read adapter on top of this state.
    # no_grad is required because adapter layers later save this constant input
    # while computing their own weight gradients.
    with torch.no_grad():
        base_prompt_attention = torch.ones_like(decoder_prompt_ids)
        base_prompt_positions = torch.arange(decoder_prompt_ids.shape[1], device=device).unsqueeze(0)
        base_prompt_output = decoder(
            input_ids=None,
            inputs_embeds=decoder_prompt_embeddings,
            attention_mask=base_prompt_attention,
            position_ids=base_prompt_positions,
            use_cache=False,
            return_dict=True,
        )
        base_prompt_hidden = base_prompt_output.last_hidden_state[:, -1].detach()
    del base_prompt_output

    def build_decoder_memory(clip_tokens, questions, shuffle_memory: bool, zero_memory: bool):
        memory_outputs = memory_model(clip_tokens, questions, shuffle_memory=shuffle_memory)
        memory = memory_outputs["memory"]
        retrieved = torch.einsum("bl,bld->bd", memory_outputs["read_attention"], memory)
        decoder_memory = torch.cat((retrieved.unsqueeze(1), memory[:, 1:]), dim=1)
        if zero_memory:
            decoder_memory = torch.zeros_like(decoder_memory)
        return decoder_memory

    def prefix_logits(clip_tokens, questions, shuffle_memory: bool = False, zero_memory: bool = False):
        decoder_memory = build_decoder_memory(clip_tokens, questions, shuffle_memory, zero_memory)
        prefix = memory_bridge(decoder_memory)
        prefix = prefix.to(decoder_prompt_embeddings.dtype)
        prompt = decoder_prompt_embeddings.expand(prefix.shape[0], -1, -1)
        inputs_embeds = torch.cat((prefix, prompt), dim=1)
        sequence_length = inputs_embeds.shape[1]
        attention_mask = torch.ones(prefix.shape[0], sequence_length, dtype=torch.long, device=device)
        # This is a vision-free text-LM pass, matching the repository's
        # auxiliary latent-prefix decoder path.
        position_ids = torch.arange(sequence_length, device=device)
        position_ids = position_ids.unsqueeze(0).expand(prefix.shape[0], -1)
        outputs = decoder(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            return_dict=True,
        )
        final_hidden = outputs.last_hidden_state[:, -1]
        return F.linear(final_hidden, candidate_lm_weights).float()

    def cross_attention_logits(clip_tokens, questions, shuffle_memory: bool = False, zero_memory: bool = False):
        decoder_memory = build_decoder_memory(clip_tokens, questions, shuffle_memory, zero_memory)
        prompt_hidden = base_prompt_hidden.expand(decoder_memory.shape[0], -1)
        fused_hidden, _ = cross_attention_adapter(prompt_hidden, decoder_memory)
        return F.linear(fused_hidden.to(candidate_lm_weights.dtype), candidate_lm_weights).float()

    decoder_logits = prefix_logits if args.injection_mode == "prefix" else cross_attention_logits

    if args.injection_mode == "prefix":
        injection_parameters = list(memory_bridge.parameters())
    else:
        injection_parameters = list(cross_attention_adapter.parameters())

    # Prefix mode differentiates through the frozen decoder. Cross-attention
    # mode uses its cached question state and an explicit trainable memory read.
    decoder_optimizer = torch.optim.AdamW(
        [
            {"params": list(memory_model.parameters()), "lr": 5e-5},
            {"params": injection_parameters, "lr": 1e-3},
        ],
        weight_decay=1e-3,
    )
    decoder_generator = torch.Generator().manual_seed(args.seed + 200)
    decoder_loss = None
    memory_model.train()
    memory_bridge.train()
    cross_attention_adapter.train()
    for step in range(args.decoder_steps):
        clip_tokens, questions, labels = make_batch(args.decoder_batch_size, decoder_generator)
        loss_tensor = F.cross_entropy(decoder_logits(clip_tokens, questions), labels)
        decoder_optimizer.zero_grad(set_to_none=True)
        loss_tensor.backward()
        torch.nn.utils.clip_grad_norm_(
            list(memory_model.parameters()) + injection_parameters,
            1.0,
        )
        decoder_optimizer.step()
        decoder_loss = float(loss_tensor.detach())
        if step == 0 or (step + 1) % 25 == 0:
            print(f"decoder_step={step + 1} loss={decoder_loss:.6f}")

    memory_model.eval()
    memory_bridge.eval()
    cross_attention_adapter.eval()
    eval_generator = torch.Generator().manual_seed(args.seed + 10_000)
    eval_tokens, eval_questions, eval_labels = make_batch(args.eval_samples, eval_generator)

    def evaluate(mode: str) -> float:
        correct = 0
        total = 0
        for start in range(0, args.eval_samples, args.eval_batch_size):
            end = min(start + args.eval_batch_size, args.eval_samples)
            with torch.inference_mode():
                logits = decoder_logits(
                    eval_tokens[start:end],
                    eval_questions[start:end],
                    shuffle_memory=mode == "shuffled",
                    zero_memory=mode == "zero",
                )
            correct += int((logits.argmax(-1) == eval_labels[start:end]).sum())
            total += end - start
        return correct / total

    normal_accuracy = evaluate("normal")
    shuffled_trials = []
    for trial in range(3):
        torch.manual_seed(args.seed + 20_000 + trial)
        shuffled_trials.append(evaluate("shuffled"))
    shuffled_accuracy = sum(shuffled_trials) / len(shuffled_trials)
    zero_memory_accuracy = evaluate("zero")

    result = {
        "decoder": "frozen_qwen3_vl_lm_head_abcd",
        "injection_mode": args.injection_mode,
        "memory_warmup_steps": args.memory_warmup_steps,
        "memory_warmup_accuracy": round(warmup_accuracy, 4),
        "decoder_steps": args.decoder_steps,
        "global_memory_tokens": num_clips * slots_per_clip,
        "warmup_final_loss": round(warmup_loss, 6),
        "decoder_final_loss": round(decoder_loss, 6),
        "normal_memory_accuracy": round(normal_accuracy, 4),
        "shuffled_memory_accuracy": round(shuffled_accuracy, 4),
        "zero_memory_accuracy": round(zero_memory_accuracy, 4),
        "order_gain": round(normal_accuracy - shuffled_accuracy, 4),
        "memory_gain_over_zero": round(normal_accuracy - zero_memory_accuracy, 4),
    }
    print(json.dumps(result, indent=2))
    passed = (
        normal_accuracy >= 0.60
        and normal_accuracy >= shuffled_accuracy + 0.10
        and normal_accuracy >= zero_memory_accuracy + 0.10
    )
    if not passed:
        print("OVERALL FAIL: continuous memory did not reliably control frozen Qwen decoding")
        return 1
    print("OVERALL PASS: continuous temporal memory controls frozen Qwen A/B/C/D decoding")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
