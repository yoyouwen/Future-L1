#!/usr/bin/env python3
"""Generate a tiny synthetic MP4 and matching TwiFF SFT smoke record.

The video shows a red square moving continuously from left to right. An early
uniformly sampled frame is used as observed evidence and a later frame is used
as the assistant-side reasoning image / latent target. The artifact is only
for exercising dataset, collator, forward, and backward code paths.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=64)
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--question-frame", type=int, default=2)
    parser.add_argument("--reasoning-frame", type=int, default=6)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.frames < 16:
        raise ValueError("--frames must be at least 16")
    if args.fps <= 0:
        raise ValueError("--fps must be positive")
    if args.size < 64:
        raise ValueError("--size must be at least 64")
    for name, value in (
        ("question-frame", args.question_frame),
        ("reasoning-frame", args.reasoning_frame),
    ):
        if not 1 <= value <= 8:
            raise ValueError(f"--{name} must be in the TwiFF frame pool range 1..8")
    if args.reasoning_frame <= args.question_frame:
        raise ValueError("--reasoning-frame must be later than --question-frame")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    video_path = args.output_dir / "moving_red_square.mp4"
    json_path = args.output_dir / "train.json"

    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        (args.size, args.size),
    )
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not initialize the MP4 writer")

    square = max(20, args.size // 6)
    margin = max(8, args.size // 16)
    y0 = (args.size - square) // 2
    usable_x = args.size - square - 2 * margin

    for index in range(args.frames):
        progress = index / (args.frames - 1)
        x0 = margin + int(round(usable_x * progress))
        frame = np.full((args.size, args.size, 3), 245, dtype=np.uint8)
        cv2.rectangle(
            frame,
            (x0, y0),
            (x0 + square, y0 + square),
            (0, 0, 230),
            thickness=-1,
        )
        cv2.putText(
            frame,
            f"t={index:02d}",
            (8, args.size - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (40, 40, 40),
            1,
            cv2.LINE_AA,
        )
        writer.write(frame)
    writer.release()

    capture = cv2.VideoCapture(str(video_path))
    decoded_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    decoded_fps = float(capture.get(cv2.CAP_PROP_FPS))
    capture.release()
    if decoded_frames <= 0 or decoded_fps <= 0:
        raise RuntimeError(f"Generated video is unreadable: {video_path}")

    sample = {
        "conversations": [
            {
                "from": "human",
                "value": (
                    "<image>\nWhat will the red square most likely do next?\n"
                    "A) Continue moving to the right\n"
                    "B) Move back to the left\n"
                    "C) Change into a blue circle\n"
                    "D) Disappear immediately"
                ),
            },
            {
                "from": "gpt",
                "value": (
                    "THOUGHT 1: The red square has progressed horizontally from left "
                    "to right, so the motion is likely to continue. <image>\n"
                    "The best answer is A."
                ),
            },
        ],
        "video": str(video_path.resolve()),
        "image": [args.question_frame],
        "reasoning_image": [args.reasoning_frame],
        "answer": "A",
        "cot": "The observed trajectory continues from left to right.",
    }
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump([sample], handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print(f"PASS video={video_path}")
    print(f"video_frames={decoded_frames}")
    print(f"video_fps={decoded_fps:.2f}")
    print(f"video_size_bytes={video_path.stat().st_size}")
    print(f"PASS twiff_json={json_path}")
    print(f"observed_frame={args.question_frame}")
    print(f"reasoning_frame={args.reasoning_frame}")
    print("answer=A")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
