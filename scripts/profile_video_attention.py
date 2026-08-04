#!/usr/bin/env python3
"""Profile the current Qwen3-VL dense video-attention baseline.

For one video, preprocess and prefill the same prompt with several requested
frame counts. The script reports the resulting multimodal sequence length,
video placeholder-token count, theoretical dense-attention pair count, GPU
prefill time, and peak allocated memory. It does not generate, train, or save.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import time
from pathlib import Path


def parse_frame_counts(value: str) -> list[int]:
    counts = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not counts or any(count < 2 or count % 2 for count in counts):
        raise argparse.ArgumentTypeError("frame counts must be positive even integers, e.g. 4,8,16,32")
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--video-path", type=Path, required=True)
    parser.add_argument("--frame-counts", type=parse_frame_counts, default=parse_frame_counts("4,8,16,32"))
    parser.add_argument("--pixels-per-frame", type=int, default=224 * 224)
    parser.add_argument("--prompt", default="Describe the motion in this video briefly.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--allow-download", action="store_true")
    return parser.parse_args()


def build_inputs(processor, process_vision_info, video_path: Path, frames: int, pixels: int, prompt: str):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": str(video_path),
                    "nframes": frames,
                    "min_pixels": pixels,
                    "max_pixels": pixels,
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages,
        return_video_kwargs=True,
        image_patch_size=16,
        return_video_metadata=True,
    )
    video_metadata = None
    if video_inputs is not None:
        video_inputs, video_metadata = map(list, zip(*video_inputs))
    return processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        video_metadata=video_metadata,
        **video_kwargs,
        do_resize=False,
        return_tensors="pt",
    )


def main() -> int:
    args = parse_args()
    if not args.video_path.is_file():
        raise FileNotFoundError(f"video not found: {args.video_path}")
    if args.pixels_per_frame <= 0:
        raise ValueError("--pixels-per-frame must be positive")

    import torch
    from qwen_vl_utils import process_vision_info
    from transformers import AutoConfig, AutoProcessor, Qwen3VLForConditionalGeneration

    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA was requested but is unavailable")

    local_only = not args.allow_download
    config = AutoConfig.from_pretrained(args.model_path, local_files_only=local_only)
    if config.model_type != "qwen3_vl":
        raise ValueError(f"this profiler currently expects qwen3_vl, got {config.model_type!r}")

    device = torch.device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    processor = AutoProcessor.from_pretrained(
        args.model_path,
        min_pixels=args.pixels_per_frame,
        max_pixels=args.pixels_per_frame,
        local_files_only=local_only,
    )
    print(f"loading model={args.model_path} device={device} dtype={dtype}")
    load_kwargs = {
        "dtype": dtype,
        "attn_implementation": "sdpa",
        "local_files_only": local_only,
        "low_cpu_mem_usage": True,
    }
    if device.type == "cuda":
        load_kwargs["device_map"] = {"": str(device)}
    model = Qwen3VLForConditionalGeneration.from_pretrained(args.model_path, **load_kwargs).eval()
    if device.type != "cuda":
        model.to(device)
    backbone = model.model

    video_token_id = getattr(model.config, "video_token_id", None)
    if video_token_id is None:
        video_token_id = processor.tokenizer.convert_tokens_to_ids("<|video_pad|>")

    # One untimed warm-up avoids charging first-use CUDA/kernel setup to the
    # smallest requested frame count.
    warmup_inputs = build_inputs(
        processor,
        process_vision_info,
        args.video_path,
        min(args.frame_counts),
        args.pixels_per_frame,
        args.prompt,
    )
    warmup_inputs = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in warmup_inputs.items()}
    with torch.inference_mode():
        _ = backbone(**warmup_inputs, use_cache=False, return_dict=True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    del warmup_inputs, _
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    results = []
    for requested_frames in args.frame_counts:
        preprocess_start = time.perf_counter()
        inputs = build_inputs(
            processor,
            process_vision_info,
            args.video_path,
            requested_frames,
            args.pixels_per_frame,
            args.prompt,
        )
        preprocess_seconds = time.perf_counter() - preprocess_start

        sequence_tokens = int(inputs["input_ids"].shape[1])
        video_tokens = int((inputs["input_ids"] == video_token_id).sum().item())
        grid = inputs.get("video_grid_thw")
        grid_thw = None if grid is None else [int(value) for value in grid[0].tolist()]
        attention_pairs = sequence_tokens * sequence_tokens
        inputs = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in inputs.items()}

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
            memory_before = torch.cuda.memory_allocated(device)
        else:
            memory_before = 0
        start = time.perf_counter()
        with torch.inference_mode():
            outputs = backbone(**inputs, use_cache=False, return_dict=True)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        prefill_seconds = time.perf_counter() - start
        peak_bytes = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
        peak_delta_bytes = max(0, peak_bytes - memory_before)

        row = {
            "requested_frames": requested_frames,
            "video_grid_thw": grid_thw,
            "sequence_tokens": sequence_tokens,
            "video_tokens": video_tokens,
            "attention_pairs": attention_pairs,
            "preprocess_seconds": round(preprocess_seconds, 4),
            "prefill_seconds": round(prefill_seconds, 4),
            "peak_allocated_gib": round(peak_bytes / 1024**3, 3),
            "peak_forward_delta_gib": round(peak_delta_bytes / 1024**3, 3),
        }
        if not math.isfinite(prefill_seconds):
            raise RuntimeError("non-finite timing result")
        results.append(row)
        print(json.dumps(row, ensure_ascii=False))

        del inputs, outputs
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    base_pairs = results[0]["attention_pairs"]
    print("\nframes  seq_tokens  video_tokens  pair_ratio  prefill_s  peak_GiB")
    for row in results:
        ratio = row["attention_pairs"] / base_pairs
        print(
            f"{row['requested_frames']:>6}  {row['sequence_tokens']:>10}  "
            f"{row['video_tokens']:>12}  {ratio:>10.2f}  "
            f"{row['prefill_seconds']:>9.4f}  {row['peak_allocated_gib']:>8.3f}"
        )
    print("OVERALL PASS (inference-only dense video prefill profile; no generation or save)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
