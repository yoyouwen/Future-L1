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
    print("OVERALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
