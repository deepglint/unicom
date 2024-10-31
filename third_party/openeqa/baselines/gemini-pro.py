# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import os
import traceback
from pathlib import Path
from typing import Optional

import tqdm

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
        default="gemini-pro",
        help="Google model (default: gemini-pro)",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default="data/results",
        help="output directory (default: data/results)",
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
    question: str,
    google_key: Optional[str] = None,
    google_model: str = "gemini-pro",
) -> Optional[str]:
    try:
        prompt = load_prompt("blind-llm")
        set_google_key(key=google_key)
        message = prompt.format(question=question)
        output = call_google_api(
            message=message,
            model=google_model,
        )
        return parse_gemini_output(message, output)
    except Exception as e:
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

        # generate answer
        question = item["question"]
        answer = ask_question(question=question, google_model=args.model)

        # store results
        results.append({"question_id": question_id, "answer": answer})
        json.dump(results, args.output_path.open("w"), indent=2)

    # save at end (redundant)
    json.dump(results, args.output_path.open("w"), indent=2)
    print("saving {:,} answers".format(len(results)))


if __name__ == "__main__":
    main(parse_args())
