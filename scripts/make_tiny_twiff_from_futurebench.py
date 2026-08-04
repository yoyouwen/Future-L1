#!/usr/bin/env python3
"""Create a deterministic one-record TwiFF SFT smoke dataset.

The source record and video come from a local FutureBench manifest. The SFT
annotation is intentionally synthetic: an early uniformly sampled frame is
used as observed evidence and a later frame as the future/latent target. This
is suitable for exercising Future-L1's dataset, collator, and forward/loss
paths; it is not training or benchmark data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        rows = next(
            (
                payload[key]
                for key in ("data", "test", "annotations", "samples")
                if isinstance(payload.get(key), list)
            ),
            [],
        )
    else:
        rows = []

    rows = [row for row in rows if isinstance(row, dict)]
    if not rows:
        raise ValueError(f"No FutureBench records found in {path}")
    return rows


def question_with_options(qa: dict[str, Any]) -> str:
    question = str(qa.get("Question", "")).strip()
    if "A)" in question and "D)" in question:
        return question

    options = qa.get("Options", {})
    missing = [key for key in "ABCD" if key not in options]
    if missing:
        raise ValueError(f"Missing answer options: {missing}")
    rendered = "\n".join(f"{key}) {options[key]}" for key in "ABCD")
    return f"{question}\n{rendered}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--futurebench-json", type=Path, required=True)
    parser.add_argument(
        "--video-root",
        type=Path,
        required=True,
        help="Directory against which FutureBench video_path is resolved.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--question-frame", type=int, default=2)
    parser.add_argument("--reasoning-frame", type=int, default=6)
    parser.add_argument("--allow-missing-video", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for name, value in (
        ("question-frame", args.question_frame),
        ("reasoning-frame", args.reasoning_frame),
    ):
        if not 1 <= value <= 8:
            raise ValueError(f"--{name} must be in the TwiFF frame pool range 1..8")
    if args.reasoning_frame <= args.question_frame:
        raise ValueError("--reasoning-frame must be later than --question-frame")

    rows = load_rows(args.futurebench_json)
    if not -len(rows) <= args.index < len(rows):
        raise IndexError(f"--index {args.index} is outside {len(rows)} records")
    row = rows[args.index]

    qa = row.get("qa")
    if not isinstance(qa, dict):
        raise ValueError("Selected record has no qa object")
    video_path = row.get("video_path")
    if not video_path:
        raise ValueError("Selected record has no video_path")

    video = Path(str(video_path))
    if not video.is_absolute():
        video = args.video_root / video
    video = video.resolve()
    if not video.is_file() and not args.allow_missing_video:
        raise FileNotFoundError(f"Video does not exist: {video}")

    answer = str(qa.get("Answer", "")).strip().upper()
    if answer not in set("ABCD"):
        raise ValueError(f"Expected answer A/B/C/D, got {answer!r}")

    sample = {
        "conversations": [
            {
                "from": "human",
                "value": "<image>\n" + question_with_options(qa),
            },
            {
                "from": "gpt",
                "value": (
                    "THOUGHT 1: I inspect the observed event and infer its likely "
                    "continuation. <image>\n"
                    f"The best answer is {answer}."
                ),
            },
        ],
        "video": str(video),
        "image": [args.question_frame],
        "reasoning_image": [args.reasoning_frame],
        "answer": answer,
        "cot": "Synthetic smoke annotation using a real FutureBench video.",
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump([sample], handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print(f"PASS wrote {args.output}")
    print(f"source_id={row.get('id')}")
    print(f"question_type={row.get('question_type')}")
    print(f"video={video}")
    print(f"video_exists={video.is_file()}")
    print(f"observed_frame={args.question_frame}")
    print(f"reasoning_frame={args.reasoning_frame}")
    print(f"answer={answer}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
