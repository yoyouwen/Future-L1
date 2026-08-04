#!/usr/bin/env python3
"""Load a tiny Future-L1/TwiFF SFT batch and print its fields and shapes.

With no arguments this performs a schema-only smoke. Supplying --data-path
loads up to two normalized records. Supplying both --data-path and --model-path
also runs the official processor/collator path. Model downloads are disabled by
default; pass --allow-download explicitly if that is desired.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


EXPECTED_TWIFF_SCHEMA = {
    "conversations": [
        {"from": "human", "value": "<image> What happens next?"},
        {"from": "gpt", "value": "THOUGHT 1: ... <image> ..."},
    ],
    "video": "relative/or/absolute/video.mp4",
    "image": [1],
    "reasoning_image": [2],
    "answer": "final answer",
    "cot": "optional text-only CoT",
}


def describe(name: str, value: Any) -> None:
    if value is None:
        print(f"{name}: None")
    elif hasattr(value, "shape"):
        print(
            f"{name}: shape={tuple(value.shape)} "
            f"dtype={getattr(value, 'dtype', 'n/a')} device={getattr(value, 'device', 'n/a')}"
        )
    elif isinstance(value, (list, tuple)):
        print(f"{name}: {type(value).__name__} len={len(value)}")
    else:
        print(f"{name}: {type(value).__name__} value={value!r}")


def peek_json(path: Path, limit: int) -> list[dict[str, Any]]:
    files = sorted(path.glob("*.json")) if path.is_dir() else [path]
    rows: list[dict[str, Any]] = []
    for file_path in files:
        with file_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        payload = payload if isinstance(payload, list) else [payload]
        rows.extend(row for row in payload if isinstance(row, dict))
        if len(rows) >= limit:
            break
    return rows[:limit]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=Path)
    parser.add_argument("--model-path")
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--max-latent-tokens", type=int, default=4)
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    print("repo_root", repo_root)
    print("expected_twiff_schema")
    print(json.dumps(EXPECTED_TWIFF_SCHEMA, indent=2, ensure_ascii=False))

    if args.data_path is None:
        print("SKIP dataset load: provide --data-path to inspect records")
        print("SKIP collator: provide both --data-path and --model-path")
        return 0
    if not args.data_path.exists():
        print(f"FAIL data path does not exist: {args.data_path}", file=sys.stderr)
        return 1

    raw_rows = peek_json(args.data_path, args.limit)
    if not raw_rows:
        print("FAIL no JSON records found", file=sys.stderr)
        return 1
    print(f"PASS raw_json records={len(raw_rows)}")
    for idx, row in enumerate(raw_rows):
        print(f"raw_record_{idx}_keys={sorted(row)}")

    from src.dataset.twiff_sft_dataset import TwiFFSFTDataset

    dataset = TwiFFSFTDataset(str(args.data_path))
    print(f"PASS normalized_dataset rows={len(dataset)}")
    for idx in range(min(args.limit, len(dataset))):
        row = dataset[idx]
        print(
            f"normalized_record_{idx} source_format={row.get('source_format')!r} "
            f"keys={sorted(row)}"
        )

    if args.model_path is None:
        print("SKIP collator: provide --model-path for the official processor path")
        return 0

    from transformers import AutoProcessor
    from src.dataset.twiff_sft_dataset import TwiFFDataCollator
    from src.params import DataArguments

    processor = AutoProcessor.from_pretrained(
        args.model_path,
        local_files_only=not args.allow_download,
    )
    chat_template_path = repo_root / "chat_template.json"
    with chat_template_path.open("r", encoding="utf-8") as handle:
        future_l1_chat_template = json.load(handle)["chat_template"]
    # The base Qwen3-VL template may omit assistant-side images. Future-L1's
    # template deliberately renders images for every role so those placeholders
    # can be rewritten into latent spans by the collator.
    processor.chat_template = future_l1_chat_template
    processor.tokenizer.chat_template = future_l1_chat_template
    print(f"PASS Future-L1 chat template loaded from {chat_template_path}")
    processor.tokenizer.add_tokens(
        ["<|latent|>", "<|latent_start|>", "<|latent_end|>"],
        special_tokens=False,
    )
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
    collator = TwiFFDataCollator(processor=processor, args=data_args)
    examples = [dataset[idx] for idx in range(min(args.limit, len(dataset)))]
    batch = collator(examples)
    if not batch:
        print("FAIL collator returned an empty batch", file=sys.stderr)
        return 1

    print("PASS collator")
    for key in sorted(batch):
        describe(key, batch[key])

    required = {"input_ids", "attention_mask", "labels"}
    missing = sorted(required.difference(batch))
    if missing:
        print(f"FAIL missing required fields: {missing}", file=sys.stderr)
        return 1
    if "pixel_values_latent" not in batch:
        print("FAIL missing pixel_values_latent; sample may not contain a future visual hint", file=sys.stderr)
        return 1
    latent_id = processor.tokenizer.convert_tokens_to_ids("<|latent|>")
    latent_count = int((batch["input_ids"] == latent_id).sum().item())
    image_out_count = int(batch.get("image_out_mask", batch["input_ids"].new_zeros(1)).sum().item())
    print(f"latent_token_id={latent_id} latent_token_count={latent_count}")
    print(f"image_out_mask_count={image_out_count}")
    if latent_count <= 0:
        print("FAIL no latent tokens in collated input_ids", file=sys.stderr)
        return 1
    if image_out_count != latent_count:
        print(
            f"FAIL image_out_mask count {image_out_count} != latent token count {latent_count}",
            file=sys.stderr,
        )
        return 1
    print("OVERALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
