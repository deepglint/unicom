# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import os
import traceback
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import tqdm
from PIL import Image, PngImagePlugin

from third_party.openeqa.utils.google_utils import call_google_api, set_google_key
from third_party.openeqa.utils.prompt_utils import load_prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default="data/open-eqa-v0.json",
        help="path to EQA dataset (default: data/open-eqa-v0.json)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-pro-vision",
        help="Google model (default: gemini-pro-vision)",
    )
    parser.add_argument(
        "--frames-directory",
        type=Path,
        default="data/frames",
        help="path episode histories (default: data/frames)",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=15,
        help="number of frames (default: 15)",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=512,
        help="image size (default: 512)",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default="data/results",
        help="output directory (default: data/results)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="continue running on API errors (default: false)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="only process the first 5 questions",
    )
    args = parser.parse_args()
    args.output_directory.mkdir(parents=True, exist_ok=True)
    args.output_path = args.output_directory / (
        args.dataset.stem + "-{}.json".format(args.model)
    )
    return args


def parse_gemini_output(input: str, output: str) -> str:
    start_idx = output.find("A:")
    if start_idx == -1:
        return output.replace("A:", "").strip()
    end_idx = output.find("\n", start_idx)
    if end_idx == -1:
        return output[start_idx:].replace("A:", "").strip()
    return output[start_idx:end_idx].replace("A:", "").strip()


def ask_question(
    frame_paths: List,
    question: str,
    image_size: int,
    google_model: str,
    google_key: Optional[str] = None,
    force: bool = False,
) -> Optional[str]:
    try:
        set_google_key(key=google_key)

        frames = [cv2.imread(p) for p in frame_paths]
        size = max(frames[0].shape)
        frames = [
            cv2.resize(img, dsize=None, fx=image_size / size, fy=image_size / size)
            for img in frames
        ]
        frames = [Image.fromarray(img) for img in frames]

        prompt = load_prompt("gemini-pro-vision")
        prefix, suffix = prompt.split("User Query:")
        suffix = "User Query:" + suffix.format(question=question)

        messages = []
        messages += [prefix]
        messages += frames
        messages += [suffix]

        output = call_google_api(
            message=messages,
            model=google_model,
        )
        return parse_gemini_output(input, output)
    except Exception as e:
        if not force:
            traceback.print_exc()
            raise e


def main(args: argparse.Namespace):
    # check for google api key
    assert "GOOGLE_API_KEY" in os.environ

    # load dataset
    dataset = json.load(args.dataset.open("r"))
    print("found {:,} questions".format(len(dataset)))

    # load results
    results = []
    if args.output_path.exists():
        results = json.load(args.output_path.open())
        print("found {:,} existing results".format(len(results)))
    completed = [item["question_id"] for item in results]

    # process data
    for idx, item in enumerate(tqdm.tqdm(dataset)):
        if args.dry_run and idx >= 5:
            break

        # skip completed questions
        question_id = item["question_id"]
        if question_id in completed:
            continue  # skip existing

        # extract scene paths
        folder = args.frames_directory / item["episode_history"]
        frames = sorted(folder.glob("*-rgb.png"))
        indices = np.round(np.linspace(0, len(frames) - 1, args.num_frames)).astype(int)
        paths = [str(frames[i]) for i in indices]

        # generate answer
        question = item["question"]
        answer = ask_question(
            frame_paths=paths,
            question=question,
            image_size=args.image_size,
            google_model=args.model,
            force=args.force,
        )

        # store results
        results.append({"question_id": question_id, "answer": answer})
        json.dump(results, args.output_path.open("w"), indent=2)

    # save at end (redundant)
    json.dump(results, args.output_path.open("w"), indent=2)
    print("saving {:,} answers".format(len(results)))


if __name__ == "__main__":
    main(parse_args())
